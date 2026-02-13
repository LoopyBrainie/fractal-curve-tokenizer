# -*- coding: utf-8 -*-
"""
Hierarchical Soft-Hard Attention (HSHA)

数学形式化
============

三区域划分:
    LCA(i, j) = ℓ

    M_region(ℓ) = HARD_ZERO     if ℓ < ℓ_min
                = SOFT_POSITIVE if ℓ_min ≤ ℓ < ℓ_soft
                = HARD_ONE      if ℓ ≥ ℓ_soft

完整注意力公式:
    Ã[i,j] = QK^T/√d_k[i,j] + α(ℓ) · B_hilbert[i,j]

    α(ℓ) = 0                          if ℓ < ℓ_min
         = τ · g(ℓ - ℓ_min)           if ℓ_min ≤ ℓ < ℓ_soft
         = τ                            if ℓ ≥ ℓ_soft

    g(x) = sigmoid(x - ℓ_soft/2)       平滑过渡

    Attn[i,j] = exp(Ã[i,j]) / Σ_k exp(Ã[i,k])

与最佳实现对齐
==============
- Zhao et al. 2024: Hilbert Curve 扫描策略 + 局部性保证
- Li & Xu 2025: NAP 邻居感知 + 100% 梯度覆盖率
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_pytorch.core.constants import GRAD_CLAMP_BOUND
from vit_pytorch.core.config import AttentionEncoderConfig
from vit_pytorch.layers.embeddings.fractal_path import VectorizedPathEncoder


@dataclass
class HierarchicalAttentionConfig:
    """分层软硬注意力配置

    三区域划分:
        - HARD_ZERO (ℓ < ℓ_min): 强制排除远距离 Token
        - SOFT_POSITIVE (ℓ_min ≤ ℓ < ℓ_soft): 软偏置鼓励
        - HARD_ONE (ℓ ≥ ℓ_soft): 完全鼓励
    """
    # 阈值参数
    lca_min: float = 1.0       # ℓ_min: 硬排除阈值 (默认1=象限边界)
    lca_soft: float = 2.0      # ℓ_soft: 软偏置上限 (默认2=子象限)

    # 缩放参数
    use_temperature: bool = True
    temperature_init: float = 1.0  # τ 初始化

    # 可学习参数
    learn_thresholds: bool = True
    learn_temperature: bool = True

    # 偏置参数
    use_hilbert_bias: bool = True
    hilbert_bias_scale: float = 1.0

    # I163-1: 分层自适应边界选项
    use_compact_bound: bool = False  # 使用优化后的紧边界 (N/3, N/4)

    # 与现有配置集成
    attention_config: Optional[AttentionEncoderConfig] = None

    @classmethod
    def from_attention_config(cls, config: AttentionEncoderConfig) -> "HierarchicalAttentionConfig":
        """从 AttentionEncoderConfig 创建分层配置"""
        return cls(
            attention_config=config,
            use_hilbert_bias=True,
            hilbert_bias_scale=config.hilbert_bias_init,
        )


class HierarchicalMaskBuilder(nn.Module):
    """分层掩码构建器

    基于 LCA 深度构建三区域掩码:
        M(ℓ) = 0     if ℓ < ℓ_min (HARD_ZERO)
             = w(ℓ) if ℓ_min ≤ ℓ < ℓ_soft (SOFT_POSITIVE)
             = 1     if ℓ ≥ ℓ_soft (HARD_ONE)

    其中 w(ℓ) = sigmoid(ℓ - ℓ_soft/2) 提供平滑过渡
    """

    def __init__(
        self,
        max_level: int,
        lca_min: float = 1.0,
        lca_soft: float = 2.0,
        learn_thresholds: bool = True,
    ):
        """初始化分层掩码构建器

        Args:
            max_level: 最大四叉树深度
            lca_min: 硬排除阈值 (LCA < lca_min → mask = 0)
            lca_soft: 软偏置上限 (LCA ≥ lca_soft → mask = 1)
            learn_thresholds: 是否使用可学习阈值
        """
        super().__init__()
        self.max_level = max_level
        self.lca_min_init = lca_min
        self.lca_soft_init = lca_soft
        self.learn_thresholds = learn_thresholds

        # 可学习阈值 (使用 clamp 约束到 [0, max_level])
        if learn_thresholds:
            self.lca_min_param = nn.Parameter(
                torch.tensor(lca_min, dtype=torch.float32)
            )
            self.lca_soft_param = nn.Parameter(
                torch.tensor(lca_soft, dtype=torch.float32)
            )
        else:
            self.register_buffer("lca_min_const", torch.tensor(float(lca_min)))
            self.register_buffer("lca_soft_const", torch.tensor(float(lca_soft)))
            self.lca_min_param = None
            self.lca_soft_param = None

    def _get_lca_min(self) -> torch.Tensor:
        """获取 lca_min (可学习或固定)"""
        if self.lca_min_param is not None:
            # Clamp 约束到 [0, max_level]
            return self.lca_min_param.clamp(0, self.max_level)
        return self.lca_min_const  # 类型: ignore

    def _get_lca_soft(self) -> torch.Tensor:
        """获取 lca_soft (可学习或固定)"""
        if self.lca_soft_param is not None:
            # Clamp 约束到 [0, max_level]
            return self.lca_soft_param.clamp(0, self.max_level)
        return self.lca_soft_const  # 类型: ignore

    # I163-1: 分层自适应边界 - 批量计算
    def get_compact_bound_batch(
        self,
        lca_depths: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        """批量计算分层自适应空间距离边界

        数学原理 (I163-1):
            保守界: ‖pos_i - pos_j‖_∞ ≤ N / 2^l
            问题: ℓ=0,1 时过度保守

            优化界:
                ℓ=0: N/3 (Hilbert遍历特性)
                ℓ=1: N/4 (象限紧凑性)
                ℓ≥2: N/2^l (已接近理论极限)

        Args:
            lca_depths: [..., N, N] LCA 深度矩阵
            N: 网格边长

        Returns:
            [..., N, N] 空间距离边界矩阵
        """
        # 初始化为保守界
        bounds = torch.zeros_like(lca_depths, dtype=torch.float32)

        # ℓ=0: N/3
        bounds = torch.where(lca_depths == 0, N / 3.0, bounds)

        # ℓ=1: N/4
        bounds = torch.where(lca_depths == 1, N / 4.0, bounds)

        # ℓ≥2: N/2^l
        mask_ge_2 = lca_depths >= 2
        bounds = torch.where(
            mask_ge_2,
            N / (2 ** lca_depths.float()),
            bounds
        )

        return bounds

    def forward(self, lca_depths: torch.Tensor) -> torch.Tensor:
        """构建分层掩码

        Args:
            lca_depths: [B, N, N] 或 [N, N] LCA 深度矩阵

        Returns:
            mask: [B, 1, N, N] 或 [1, 1, N, N] 分层掩码
        """
        # 处理 2D 输入 [N, N] → 添加 batch 维度
        was_2d = False
        if lca_depths.dim() == 2:
            was_2d = True
            lca_depths = lca_depths.unsqueeze(0)  # [1, N, N]

        lca_min = self._get_lca_min()
        lca_soft = self._get_lca_soft()

        # 三区域划分
        # 区域 1: HARD_ZERO (ℓ < ℓ_min)
        mask = torch.zeros_like(lca_depths, dtype=torch.float32)

        # 区域 2: SOFT_POSITIVE (ℓ_min ≤ ℓ < ℓ_soft)
        # 使用 sigmoid 平滑过渡: g(ℓ) = sigmoid(ℓ - ℓ_soft/2)
        soft_mask = torch.sigmoid(lca_depths.float() - lca_soft / 2)

        # 区域 3: HARD_ONE (ℓ ≥ ℓ_soft)
        hard_one_mask = (lca_depths >= lca_soft).float()

        # 组合掩码
        # ℓ < ℓ_min: 0
        # ℓ_min ≤ ℓ < ℓ_soft: soft_mask
        # ℓ ≥ ℓ_soft: 1
        mask = torch.where(
            lca_depths < lca_min,
            torch.zeros_like(lca_depths, dtype=torch.float32),
            torch.where(
                lca_depths < lca_soft,
                soft_mask,
                hard_one_mask,
            )
        )

        # 扩展维度
        # [B, N, N] → [B, 1, N, N]
        mask = mask.unsqueeze(1)

        # 如果原始输入是 2D，移除 batch 维度
        # 但保持 4D 形状: [1, 1, N, N]
        if was_2d:
            # 输入 [N, N] → 输出 [1, 1, N, N]
            pass  # 保持 4D 形状不变

        return mask


class HierarchicalSoftHardAttention(nn.Module):
    """分层软硬注意力模块

    数学形式:
        Ã[i,j] = QK^T/√d_k[i,j] + α(ℓ) · B_hilbert[i,j]

        α(ℓ) = 0                          if ℓ < ℓ_min
             = τ · sigmoid(ℓ - ℓ_soft/2)   if ℓ_min ≤ ℓ < ℓ_soft
             = τ                            if ℓ ≥ ℓ_soft

        Attn[i,j] = softmax(Ã[i,j] · M(ℓ)) where M(ℓ) is the hierarchical mask

    特性:
        1. 硬约束: 排除远距离 Token (LCA < ℓ_min)
        2. 软偏置: 鼓励 Hilbert 邻域注意力
        3. 可学习参数: 阈值和温度自动优化
    """

    def __init__(
        self,
        dim: int,
        max_level: int,
        heads: int,
        config: Optional[HierarchicalAttentionConfig] = None,
    ):
        """初始化分层软硬注意力

        Args:
            dim: 嵌入维度
            max_level: 最大四叉树深度
            heads: 注意力头数
            config: 分层注意力配置 (默认使用标准配置)
        """
        super().__init__()
        if config is None:
            config = HierarchicalAttentionConfig()

        self.dim = dim
        self.heads = heads
        self.max_level = max_level
        self.config = config

        inner_dim = dim  # For output projection
        self.scale = dim ** -0.5

        # Q, K, V 投影
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        # 输出投影
        self.to_out = nn.Linear(dim, dim)

        # 分层掩码构建器
        self.mask_builder = HierarchicalMaskBuilder(
            max_level=max_level,
            lca_min=config.lca_min,
            lca_soft=config.lca_soft,
            learn_thresholds=config.learn_thresholds,
        )

        # Hilbert 偏置编码器
        if config.use_hilbert_bias:
            self.lca_embedding = nn.Embedding(max_level + 1, heads)

            # 初始化 (与 LCAHilbertBias 一致)
            with torch.no_grad():
                depths = torch.arange(max_level + 1, dtype=torch.float32)
                linear_depths = depths / max_level
                init_values = linear_depths.unsqueeze(1).expand(-1, heads)
                self.lca_embedding.weight.copy_(init_values)

        else:
            self.lca_embedding = None

        # 温度缩放 (可学习)
        if config.use_temperature and config.learn_temperature:
            self.temperature = nn.Parameter(torch.tensor(config.temperature_init))
        else:
            self.register_buffer("temperature_const", torch.tensor(config.temperature_init))
            self.temperature = None

        # Dropout
        self.dropout = nn.Dropout(0.0)

    def _get_temperature(self) -> torch.Tensor:
        """获取温度参数 (可学习或固定)"""
        if self.temperature is not None:
            return F.softplus(self.temperature).clamp(max=GRAD_CLAMP_BOUND)
        return self.temperature_const

    def _compute_lca(
        self,
        regions: torch.Tensor,
        image_size: int,
    ) -> torch.Tensor:
        """计算 LCA 深度矩阵

        Args:
            regions: [B, N, 4] 区域边界
            image_size: 图像边长

        Returns:
            lca_depths: [B, N, N] LCA 深度矩阵
        """
        # 转换为 long 类型 (compute_paths_from_regions 需要整数)
        regions_long = regions.long()

        # 从区域计算四叉树路径
        paths = VectorizedPathEncoder.compute_paths_from_regions(
            regions_long, image_size, self.max_level
        )

        # 计算 LCA 深度
        lca_depths = VectorizedPathEncoder.compute_common_ancestor_depth(paths)

        return lca_depths

    def forward(
        self,
        x: torch.Tensor,
        regions: Optional[torch.Tensor] = None,
        image_size: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播

        Args:
            x: [B, N, D] 输入张量
            regions: [B, N, 4] 区域边界 (可选)
            image_size: 图像边长 (可选)
            attention_mask: [B, N] 注意力掩码 (可选)

        Returns:
            output: [B, N, D] 输出张量
        """
        B, N, _ = x.shape

        # Q, K, V 投影
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # 计算 QK^T / √d_k
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 计算分层掩码和 Hilbert 偏置
        mask = None
        hilbert_bias = None

        if regions is not None and image_size is not None:
            # 1. 计算 LCA 深度矩阵
            lca_depths = self._compute_lca(regions, image_size)

            # 2. 构建分层掩码
            mask = self.mask_builder(lca_depths)  # [B, 1, N, N]

            # 3. 计算 Hilbert 偏置
            if self.lca_embedding is not None:
                hilbert_bias = self.lca_embedding(lca_depths)  # [B, N, N, H]
                hilbert_bias = hilbert_bias.permute(0, 3, 1, 2)  # [B, H, N, N]

        # 应用分层软硬约束
        if mask is not None and hilbert_bias is not None:
            temperature = self._get_temperature()

            # α(ℓ) · B_hilbert
            scaled_bias = hilbert_bias * mask * temperature

            # 添加偏置到注意力分数
            dots = dots + scaled_bias

            # 应用硬掩码 (ℓ < ℓ_min → -inf)
            dots = dots.masked_fill(mask == 0, -1e9)

        # 应用注意力掩码 (padding mask)
        if attention_mask is not None:
            # attention_mask: [B, N], True = padding
            dots = dots.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2), -1e9)

        # Softmax
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)

        # 加权聚合
        out = torch.matmul(attn, v)

        # 重组输出: [B, H, N, D] → [B, N, D]
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


# 辅助函数
def rearrange(t, pattern: str, **kwargs):
    """简化的 einops rearrange"""
    import einops
    return einops.rearrange(t, pattern, **kwargs)


# ==================== 配置预设 ====================

# 标准配置 (推荐)
STANDARD_CONFIG = HierarchicalAttentionConfig(
    lca_min=1.0,      # ℓ_min: 硬排除阈值 (象限边界)
    lca_soft=2.0,     # ℓ_soft: 软偏置上限 (子象限)
    learn_thresholds=True,
    learn_temperature=True,
    use_temperature=True,
    temperature_init=1.0,
)

# 严格配置 (强局部性)
STRICT_CONFIG = HierarchicalAttentionConfig(
    lca_min=2.0,      # ℓ_min: 更严格的硬排除
    lca_soft=3.0,     # ℓ_soft: 更严格的软偏置
    learn_thresholds=False,
    learn_temperature=False,
    use_temperature=False,
)

# 宽松配置 (弱局部性)
RELAXED_CONFIG = HierarchicalAttentionConfig(
    lca_min=0.0,      # ℓ_min: 无硬排除
    lca_soft=1.5,    # ℓ_soft: 较小的软偏置范围
    learn_thresholds=True,
    learn_temperature=True,
    use_temperature=True,
    temperature_init=0.5,
)
