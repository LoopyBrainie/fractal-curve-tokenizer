# -*- coding: utf-8 -*-
"""
Hilbert 感知多尺度注意力模块

数学形式化
============

标准多头注意力:
    Attention(Q, K, V) = softmax(QK^T / √d_k) · V

Hilbert 感知注意力:
    HilbertAttn(Q, K, V) = softmax(QK^T / √d_k · σ_scale + B_hilbert + B_level) · V

偏置项
------
1. LCA Hilbert Bias (最近公共祖先):
   B_hilbert[i,j] = LCAEmbed(LCA(i,j))
   利用四叉树 LCA 深度直接编码空间距离，参数极少 (~100)

2. Level Bias (相对层级偏置):
   B_level[i,j] = Embedding(clamp(d_i - d_j + L, 0, 2L))

3. Level Scaling (层级缩放):
   σ_scale(d) = LevelScaleEmb(d)
   深层 token 使用较小缩放

类对照表
----------
+-------------------------------+------------------------------------------+
| 类                             | 数学定义                                   |
+===============================+==========================================+
| LCAHilbertBias                | B[i,j] = LCAEmbed(LCA(i,j))              |
| HilbertAwareMultiScaleAttention| Attn + B_hilbert + B_level              |
+-------------------------------+------------------------------------------+

P11-8 简化: 移除未使用的 LowRankHilbertBias 和 HierarchicalHilbertBias
"""

from __future__ import annotations

import math
import weakref
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .constants import *
from .config import (
    ShapeScaleEncoderConfig,
    AreaEncoderConfig,
    LCAEncoderConfig,
    AttentionEncoderConfig,
)
from .levels_info import LevelsInfo  # I98-4
from .embed_fractal_path import VectorizedPathEncoder
from .depth_utils import (
    compute_region_shape_scale,
    compute_shape_scale_similarity,
    compute_normalized_area,
    compute_area_similarity,
)

# I106-1: Flash Attention 2 条件导入
try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class HilbertBiasBase(ABC, nn.Module):
    """Hilbert Bias 抽象基类。
    
    统一处理 2D/3D levels_info 维度转换，子类只需实现核心计算逻辑。
    
    数学形式化
    ----------
    输入规范化:
        L' = normalize(L)  where  L ∈ R^{(S, Info)} → L' ∈ R^{(1, S, Info)}
                                  L ∈ R^{(B, S, Info)} → L' = L
    
    输出形状约定:
        - 2D 输入 → (H, S, S) 输出
        - 3D 输入 → (B, H, S, S) 输出
    
    设计原则 (P5-8):
        内部统一使用 3D 格式 (B, S, Info) 处理，避免代码重复
    """
    
    def forward(self, levels_info: LevelsInfo) -> Optional[torch.Tensor]:
        """计算 Hilbert Bias。

        Args:
            levels_info: LevelsInfo 实例，形状为 [B, S, D+1]

        Returns:
            偏置矩阵: (B, H, S, S) 或 None
        """
        # I98-4: 兼容 raw tensor 和 LevelsInfo 对象
        was_2d = False
        if isinstance(levels_info, torch.Tensor):
            # 转换为 LevelsInfo，确保数据类型为 Long
            if levels_info.dtype != torch.long:
                levels_info = levels_info.long()

            # 处理 2D tensor (S, Info) -> 添加 batch 维度
            if levels_info.dim() == 2:
                was_2d = True
                levels_info = levels_info.unsqueeze(0)  # (1, S, Info)

            # 从数据形状推断 max_level: info_dim = max_level + 1
            info_dim = levels_info.shape[-1]

            # I98-4: 如果 info_dim == 1（只有深度列），返回 None
            # 这是测试期望的行为，表示信息不足无法计算 LCA
            if info_dim <= 1:
                return None

            inferred_max_level = info_dim - 1

            levels_info = LevelsInfo(data=levels_info, max_level=inferred_max_level)

        if levels_info.data.numel() == 0:
            return None

        # 调用子类实现的核心计算
        bias = self._compute_bias_3d(levels_info)

        if bias is None:
            return None

        # 如果原始输入是 2D，移除 batch 维度以保持向后兼容
        if was_2d:
            bias = bias.squeeze(0)

        return bias
    
    @abstractmethod
    def _compute_bias_3d(self, levels_info: LevelsInfo) -> Optional[torch.Tensor]:
        """核心计算逻辑（子类实现）。

        Args:
            levels_info: LevelsInfo 实例

        Returns:
            偏置矩阵 (B, H, S, S) 或 None
        """
        ...


class LCAHilbertBias(HilbertBiasBase):
    """基于最近公共祖先 (LCA) 的 Hilbert Bias 实现。

    I122-2 简化: 移除 τ_h 温度参数
    ===================================
    原设计使用 τ_h 缩放 LCA 偏置: B'[h,i,j] = τ_h · B[h,i,j]
    问题: λ × √d_k 可学习缩放已可吸收 τ_h 的效果 → 参数冗余

    简化后设计:
        B[i,j] = LCAEmbed(LCA(i,j))

    偏置强度由 hilbert_bias_scale × √d_k 统一控制 (见 attention 方法)

    数学原理
    ========
    利用四叉树编码的核心性质: LCA 深度直接编码空间距离。

    定理 (LCA-距离等价性):
        对于四叉树编码的两个 token i, j:
        LCA(i, j) = ℓ  ⟹  ‖pos_i - pos_j‖_∞ ≤ N / 2^ℓ

    其中 N 是网格边长，ℓ 是 LCA 深度。

    P11-3 修复: 从 regions 直接计算路径
    ===================================
    问题: 原始设计中 levels_info 的路径部分全为 0，导致 LCA 失效。

    根本原因:
        - TensorSplitResult 只存储 hilbert_indices 和 regions，不存储路径
        - 代码注释声称 "path 可从 hilbert_idx 恢复" 是数学错误
        - Hilbert index XOR ≠ Quadtree LCA (验证仅 56% 一致性)

    解决方案:
        添加 forward_from_regions() 方法，从 regions 向量化计算真实四叉树路径，
        然后计算正确的 LCA 深度矩阵。

    复杂度分析
    ==========
    - 参数量: O((D+1) × H) ≈ 128 (vs Low-Rank ~50K)
    - 计算量: O(N² × D) 用于 LCA 计算
    - 显存: O(N²) 用于偏置矩阵

    优势
    ====
    1. 显式几何意义: LCA 深度 ⟺ 空间距离
    2. 参数极少: ~100× 少于 Low-Rank
    3. 无需学习距离: 距离信息由编码结构直接提供
    4. 可解释性强: 偏置值可直接对应空间邻近程度
    """

    def __init__(
        self,
        max_level: int,
        heads: int,
        use_fp16: bool = False,  # I104-3: FP16 存储选项
    ) -> None:
        """初始化 LCA Hilbert Bias。

        Args:
            max_level: 最大四叉树深度 (决定 LCA 取值范围)
            heads: 注意力头数
            use_fp16: (I104-3) 是否使用 FP16 存储 LCA embedding
                - True: 内存节省 50%，精度损失可忽略
                - False: 使用 FP32 (默认)

        I122-2: 移除 τ_h 温度参数，由 hilbert_bias_scale × √d_k 统一缩放
        """
        super().__init__()
        self.max_level = max_level
        self.heads = heads
        self.use_fp16 = use_fp16

        # LCA 深度嵌入表: depth ∈ {0, 1, ..., max_level} → R^heads
        # 深度 0 表示完全不同的根节点，深度 max_level 表示相邻或相同
        self.lca_embedding = nn.Embedding(max_level + 1, heads)

        # I104-3: FP16 转换 + 权重裁剪保护
        # FP16 精度 (~10⁻³) 满足 LCA 偏置需求 ([-2, 2] 范围)
        # 添加权重裁剪防止梯度爆炸导致数值不稳定
        if use_fp16:
            self.lca_embedding.weight = nn.Parameter(
                self.lca_embedding.weight.to(torch.float16)
            )
            # I108-6: 注册钩子以裁剪梯度，防止 FP16 溢出
            # 使用 GRAD_CLAMP_BOUND (20.0) 替代原 10.0
            # 数学依据: P(|grad| > 20) ≈ 10^-6 << 4.5% (原边界裁剪率)
            self.lca_embedding.weight.register_hook(
                lambda grad: grad.clamp(min=-GRAD_CLAMP_BOUND, max=GRAD_CLAMP_BOUND)
            )

        # I122-2: 移除温度参数初始化，由 hilbert_bias_scale 统一缩放
        self._init_weights()

    def _init_weights(self) -> None:
        """初始化 LCA 嵌入权重。

        采用线性初始化 (I121-8 修复):
            embed[d] ∝ d / max_level

        数学依据:
            Hilbert 曲线性质: 相邻 index 的空间距离 ∝ |i - j| / 2^L (线性)
            原对数初始化 log(1+d) 与 Hilbert 局部性不匹配:
                - 深度 0→4: 线性期望 16x，log 仅 1.6x
            线性初始化使深度差异更显著，符合 Hilbert 空间局部性保证。

        这样深层 (邻近) token 获得更高的初始偏置。
        """
        with torch.no_grad():
            depths = torch.arange(self.max_level + 1, dtype=torch.float32)
            # I121-8 修复: 线性归一化深度: [0, 1]
            # Hilbert 局部性是线性的，而非对数的
            linear_depths = depths / self.max_level
            # 广播到所有 heads，加小随机扰动
            init_values = linear_depths.unsqueeze(1).expand(-1, self.heads)
            self.lca_embedding.weight.copy_(init_values)
            # I98-3: 添加小随机扰动以打破对称性
            # 使用常量 EMBEDDING_INIT_STD 而非硬编码 0.02
            self.lca_embedding.weight.add_(
                torch.randn_like(self.lca_embedding.weight) * EMBEDDING_INIT_STD
            )
    
    def _compute_bias_3d(self, levels_info: LevelsInfo) -> Optional[torch.Tensor]:
        """计算基于 LCA 的 Hilbert Bias（核心 3D 实现）。

        数学形式:
            LCA[b,i,j] = sum_d prod_{k<=d} 1[p_i[k] = p_j[k]]
            Bias[b,i,j] = Embedding(LCA[b,i,j])

        Args:
            levels_info: LevelsInfo 实例

        Returns:
            (B, H, S, S) 偏置矩阵，若无效则返回 None
        """
        data = levels_info.data
        batch_size, seq_len, info_dim = data.shape
        if info_dim <= 1:
            return None

        # I32-2: 提取深度列，识别 padding token
        depths = data[:, :, 0]  # [B, S]
        # Padding mask: True 表示 padding token (depth == -1)
        padding_mask = depths == -1

        # 提取四叉树路径: (B, S, Path)
        paths = data[:, :, 1:].long()

        # P0 修复: 对于 padding token，将路径设为特殊值，避免影响 LCA 计算
        # 有效 token 的路径是 0-3，padding token 设为 -1 可安全排除
        # 后续 clamp(0, 3) 会将 -1 变为 0，但 padding_mask 会排除这些位置
        paths = paths.clone()  # 避免原地修改
        paths[padding_mask] = -1  # 特殊值，不会与有效路径混淆

        # P-OPT: 移除冗余的范围检查，clamp 已经提供了安全保证
        # 之前的检查 + clamp 模式是冗余的：检查触发 GPU 同步但 clamp 仍会执行
        # 直接 clamp，避免 GPU-CPU 同步开销
        paths = paths.clamp(0, 3)

        # CRIT-3: 移除缓存，直接计算 LCA
        # 数学分析: 训练中每张图像不同，不存在等价输入复用
        # 缓存命中率趋近于 0，缓存是过早优化，应移除
        lca_depths = VectorizedPathEncoder.compute_common_ancestor_depth(paths)

        # P-OPT: 仅在调试模式检查 LCA 有效性，避免每次 forward 都触发 GPU 同步
        # I34-13: 静默钳位掩盖计算 bug，所以在调试时抛出异常
        if getattr(self, '_debug_mode', False):
            lca_invalid = (lca_depths < 0).any() or (lca_depths > self.max_level).any()
            if lca_invalid:
                min_depth = lca_depths.min().item()
                actual_max = lca_depths.max().item()
                raise ValueError(
                    f"LCA depth out of bounds [0, {self.max_level}]: "
                    f"min={min_depth}, max={actual_max}. "
                    "This indicates a bug in LCA computation."
                )

        # 批量嵌入: (B, S, S, H)
        bias = self.lca_embedding(lca_depths)

        # I32-2: 将涉及 padding token 的位置设为 0
        # Padding token 不应参与空间注意力偏置计算
        # 创建广播到 (B, S, S) 的 padding mask
        padding_2d = padding_mask.unsqueeze(2) | padding_mask.unsqueeze(1)  # [B, S, S]
        padding_2d = padding_2d.unsqueeze(-1)  # [B, S, S, 1] for broadcasting with H
        bias = bias.masked_fill(padding_2d, 0.0)

        # I122-2: 移除温度缩放，由 hilbert_bias_scale × √d_k 统一控制

        # 调整形状: (B, H, S, S)
        return bias.permute(0, 3, 1, 2)

    @torch._dynamo.disable  # I99-1: 排除 torch.compile 追踪，避免 Triton 编译错误
    def forward_from_regions(
        self,
        regions: torch.Tensor,
        image_size: int,
    ) -> Optional[torch.Tensor]:
        """从 regions 直接计算 LCA Hilbert Bias (P11-3 修复)。
        
        这是推荐的调用方式，绕过有问题的 levels_info 接口。
        
        数学原理:
            1. 从区域边界计算中心点: cx = (x1+x2)/2, cy = (y1+y2)/2
            2. 向量化计算四叉树路径: path[d] = bit(cx, D-d) + 2*bit(cy, D-d)
            3. 向量化计算 LCA: LCA[i,j] = len(common_prefix(path_i, path_j))
            4. 查表获取偏置: B[i,j] = LCAEmbed(LCA[i,j])
        
        Args:
            regions: 区域边界张量
                - 2D: (N, 4) 格式 [x1, y1, x2, y2]，单样本
                - 3D: (B, N, 4) 格式，批量
            image_size: 图像边长 (假设正方形)
            
        Returns:
            偏置矩阵:
                - 2D 输入 → (H, S, S)
                - 3D 输入 → (B, H, S, S)
            若输入无效则返回 None
        """
        if regions.numel() == 0:
            return None
        
        # 记录原始维度
        was_2d = regions.dim() == 2
        if was_2d:
            regions = regions.unsqueeze(0)  # [1, N, 4]
        
        B, N, _ = regions.shape

        # 转换为整数坐标 (确保 bit shift 操作正确)
        regions = regions.long()

        # 从 regions 计算四叉树路径
        paths = VectorizedPathEncoder.compute_paths_from_regions(
            regions, image_size, self.max_level
        )  # [B, N, max_level]
        
        # 计算 LCA 深度矩阵
        lca_depths = VectorizedPathEncoder.compute_common_ancestor_depth(paths)
        # P-OPT: 仅在调试模式检查 LCA，移除热路径中的 GPU 同步
        # I34-13: LCA 钳位警告 - 静默钳位可能隐藏计算 bug
        if getattr(self, '_debug_mode', False):
            lca_invalid = (lca_depths < 0).any() or (lca_depths > self.max_level).any()
            if lca_invalid:
                warnings.warn(
                    f"LCA depth clamped to [0, {self.max_level}]. "
                    f"Min: {lca_depths.min().item():.2f}, Max: {lca_depths.max().item():.2f}"
                )
        lca_depths = lca_depths.clamp(0, self.max_level)  # [B, N, N]
        
        # 批量嵌入: (B, N, N, H)
        bias = self.lca_embedding(lca_depths)

        # I122-2: 移除温度缩放，由 hilbert_bias_scale × √d_k 统一控制

        # 调整形状: (B, H, S, S)
        bias = bias.permute(0, 3, 1, 2)
        
        # 若原始输入为 2D，移除 batch 维度
        if was_2d:
            bias = bias.squeeze(0)  # (H, S, S)
        
        return bias


class   HilbertAwareMultiScaleAttention(nn.Module):
    """Hilbert 曲线感知的多尺度注意力机制 (I98-3: 协议驱动配置化).

    通过编码层级深度和 Hilbert 路径关系来调制注意力权重。

    P0 修复: max_level 参数统一为标准数学术语（分形四叉树递归深度），
    与 FractalConfig, LevelsInfo, StreamingFractalTokenizerV3 保持一致。

    P11-8 简化: 移除 bias_mode 和 low_rank_r 参数，仅保留 LCA 模式。

    I98-3: 新增 encoder_config 参数，支持协议驱动的编码器配置。

    Attributes:
        heads: 注意力头数
        dim_head: 每个头的维度
        max_level: 最大深度 (应与 tokenizer.max_level 匹配)
        use_hilbert_bias: 是否使用 Hilbert 偏置
        use_level_scaling: 是否使用层级缩放
        scale: 注意力缩放因子
        config: AttentionEncoderConfig (I98-3)
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        max_level: int = 8,  # P0 修复: 统一使用 max_level
        use_hilbert_bias: bool = True,
        use_level_scaling: bool = True,
        # I31-3: 仿射调制参数 (向后兼容)
        use_affine_modulation: bool = True,
        fourier_levels: int = 4,
        # I97-10: 新增层次化注意力模式
        use_hierarchical_attention: bool = False,
        # I98-3: 新协议驱动配置
        encoder_config: Optional[AttentionEncoderConfig] = None,
        # I104-3: FP16 存储选项
        use_fp16: bool = False,
    ) -> None:
        """初始化 HilbertAwareMultiScaleAttention (I98-3 协议驱动版本).

        P0 修复: max_level 是标准数学术语，直接描述四叉树递归深度：
            - 深度 0: 整幅图像（根节点）
            - 深度 d: 图像被划分为 4^d 个 patch
            - 深度 D: 最细粒度 (max_level)

        P11-8 简化: 移除 bias_mode 和 low_rank_r 参数，仅保留 LCA 模式。

        I31-3: 添加仿射调制支持，通过面积信息调制注意力偏置。

        I98-3: 新增 encoder_config 参数，支持协议驱动的编码器配置。
        当 encoder_config 存在时，忽略 use_affine_modulation 和 fourier_levels 参数。

        I97-10: 新增 use_hierarchical_attention 参数，实现深度内独立 Attention。
        数学形式: Attn(X) = ⊕_d softmax(Q_d K_d^T / √d_k + B_d) V_d

        I122-2: 移除 lca_temperature，由 hilbert_bias_scale × √d_k 统一缩放

        Args:
            dim: 输入维度
            heads: 注意力头数
            dim_head: 每个头的维度
            dropout: Dropout 比率
            max_level: 最大深度 (P0: 统一使用 max_level，与 tokenizer.max_level 对齐)
            use_hilbert_bias: 是否使用 Hilbert 路径偏置 (使用 LCA 模式)
            use_level_scaling: 是否使用层级缩放
            use_affine_modulation: (I31-3) 是否使用仿射调制偏置，默认 True (向后兼容)
            fourier_levels: (I31-3) 傅里叶频率级别数，默认 4 (向后兼容)
            use_hierarchical_attention: (I97-10) 是否使用深度内独立 Attention，默认 False
            encoder_config: (I98-3) AttentionEncoderConfig，协议驱动配置
            use_fp16: (I104-3) 是否使用 FP16 存储 LCA embedding，默认 False
        """
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.max_level = max_level  # P0 修复: 统一使用 max_level
        self.use_hilbert_bias = use_hilbert_bias
        self.use_level_scaling = use_level_scaling
        self.use_fp16 = use_fp16  # I104-3
        # I97-10: 新增层次化注意力模式
        self.use_hierarchical_attention = use_hierarchical_attention

        # I98-3: 处理配置
        if encoder_config is not None:
            self.config = encoder_config
            # 从配置中获取参数，忽略旧参数
            self.use_affine_modulation = encoder_config.area.fourier_levels > 0
        else:
            self.config = AttentionEncoderConfig(
                area=AreaEncoderConfig(fourier_levels=fourier_levels),
            )
            self.use_affine_modulation = use_affine_modulation

        # I24-11: 注意力权重存储开关 (默认关闭以节省内存)
        # 评估时设为 True 以支持 attention 可视化和分析
        self.store_attn_weights: bool = False
        self._last_attn_weights: Optional[torch.Tensor] = None

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # P11-8 简化: 仅使用 LCA 模式
        if use_hilbert_bias:
            self.hilbert_bias_impl: Optional[nn.Module] = LCAHilbertBias(
                max_level=max_level,
                heads=heads,
                use_fp16=use_fp16,  # I104-3
            )
        else:
            self.hilbert_bias_impl = None

        # I31-3: 仿射调制偏置 (可选) - I98-3: 支持协议驱动配置
        if self.use_affine_modulation:
            self.affine_modulated_bias: Optional[nn.Module] = AffineModulatedBias(
                dim=dim,
                max_level=max_level,
                config=self.config,
                use_fp16=use_fp16,  # I104-3
            )
        else:
            self.affine_modulated_bias = None

        if use_level_scaling:
            # P11-4 修复: 使用 Softplus 约束确保 level_scale > 0
            # 原设计使用 N(1.0, 0.1) 初始化，但无正性约束，有偏梯度可导致负值
            # 新设计: softplus(_level_scale_raw) ∈ (0, +∞)
            # I98-3: 从配置读取初始化值，默认 softplus(0.54) ≈ 1.0
            self._level_scale_raw: Optional[nn.Embedding] = nn.Embedding(max_level + 1, heads)
            nn.init.constant_(self._level_scale_raw.weight, self.config.level_scale_init)
        else:
            self._level_scale_raw = None

        # I97-10: 层级化注意力的深度缩放因子
        # 每个深度有独立的缩放因子，用于深度内 Attention
        if use_hierarchical_attention:
            self._hierarchical_depth_scale = nn.Parameter(torch.ones(max_level + 1, heads))
            # I98-3: 从配置读取初始化边界，默认 [0.5, 1.5]
            low, high = self.config.hierarchical_scale_bounds
            nn.init.uniform_(self._hierarchical_depth_scale, low, high)
        else:
            self._hierarchical_depth_scale = None

        # I97-7: 可学习偏置缩放因子 (Softplus 约束)
        # 使用 softplus 确保 λ > 0，梯度稳定
        # I98-3: 从配置读取初始化值
        # I113-11: 初始化 scale=1.0，配合 √d_k 量纲对齐后有效 scale ≈ √d_k
        self._hilbert_bias_scale_raw = nn.Parameter(
            torch.tensor(self.config.hilbert_bias_init)
        )
        self._level_bias_scale_raw = nn.Parameter(
            torch.tensor(self.config.level_bias_init)
        )

        # I108-1: Affine 偏置归一化参数
        # 问题: B_a 量级 O(dim)，与 B_h(O(1)), B_l(O(10)) 量纲不一致
        # 解决方案: B_a' = B_a / κ * γ
        #   - κ = exp(φ) 归一化因子，初始化 log(dim) 使 κ ≈ dim
        #   - γ = softplus(θ) 缩放因子，初始化 0 使 γ ≈ 1
        # 预期效果: B_a' ≈ O(1)
        self._affine_bias_norm_raw = nn.Parameter(
            torch.tensor(np.log(dim))  # κ ≈ dim
        )
        self._affine_bias_scale_raw = nn.Parameter(
            torch.tensor(0.0)  # γ ≈ 1
        )

        # A19: 移除可学习 scale_weights，保留标准 1/√d_k
        # 理由: LayerNorm 已将 Q,K 方差控制在 1，1/√d_k 已足够
        # 双重缩放导致 Var(dots) ≈ 0.69 而非理论最优的 1.0
        self._scale_weights_raw = None
        self.relative_pos_embedding = nn.Embedding(2 * max_level + 1, heads)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    # I98-3: 获取配置方法
    def get_config(self) -> AttentionEncoderConfig:
        """获取当前编码器配置 (I98-3)."""
        return self.config

    # I97-7: 可学习偏置缩放因子属性 (CRIT-2: 添加 Clamp 上界)
    @property
    def hilbert_bias_scale(self) -> torch.Tensor:
        """获取 Hilbert 偏置缩放因子 (可学习, Softplus + Clamp 约束).

        I113-11: 量纲对齐 - 实际有效 scale = λ * √d_k
        其中 λ = softplus(w)，w 初始化为 0 (λ ≈ 1.0)

        CRIT-2 修正: 添加 clamp(max=SCALE_CLAMP_BOUND) 防止数值溢出。
        I108-6 修正: 从 10.0 提升到 15.0 覆盖更多 softplus 输出范围。

        数学:
            λ = min(softplus(w), SCALE_CLAMP_BOUND) ∈ [0, 15.0]
            B_hilbert_eff = B_hilbert * λ * √d_k

        理由:
            - I113-11: 与 QK^T / √d_k 量纲对齐
            - 初始 λ=1.0，配合 √d_k ≈ 5.66，有效 scale ≈ 5.66
            - softplus 无上限，训练中可能增长到数千
            - 15.0 上界保证: max(bias * λ * √d_k) ≤ 15.0 * √d_k << FP32 安全边界 50
        """
        return F.softplus(self._hilbert_bias_scale_raw).clamp(max=SCALE_CLAMP_BOUND)

    @property
    def level_bias_scale(self) -> torch.Tensor:
        """获取层级偏置缩放因子 (可学习, Softplus + Clamp 约束).

        I113-11: 量纲对齐 - 实际有效 scale = λ * √d_k
        其中 λ = softplus(w)，w 初始化为 0 (λ ≈ 1.0)

        CRIT-2 修正: 添加 clamp(max=SCALE_CLAMP_BOUND) 保持与 hilbert_bias_scale 一致。
        I108-6 修正: 从 10.0 提升到 15.0。
        """
        return F.softplus(self._level_bias_scale_raw).clamp(max=SCALE_CLAMP_BOUND)

    @property
    def affine_bias_norm(self) -> torch.Tensor:
        """获取 Affine 偏置归一化因子 κ = exp(φ).

        I108-1 修复: B_a 量级 O(dim)，需要归一化到 O(1)。

        数学:
            κ = exp(_affine_bias_norm_raw) > 0

        初始化: φ = log(dim)，使得 κ ≈ dim
        预期效果: B_a / κ ≈ O(1)
        """
        return torch.exp(self._affine_bias_norm_raw)

    @property
    def affine_bias_scale(self) -> torch.Tensor:
        """获取 Affine 偏置缩放因子 γ = softplus(θ).

        I108-1 修复: 归一化后的 B_a 需要可学习的缩放控制。

        数学:
            γ = softplus(_affine_bias_scale_raw) ∈ (0, +∞)

        初始化: θ = 0，使得 γ ≈ 1 (保持原有量级)
        """
        return F.softplus(self._affine_bias_scale_raw)

    def _compute_hilbert_bias(
        self,
        levels_info: Optional[LevelsInfo] = None,
        regions: Optional[torch.Tensor] = None,
        image_size: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """计算基于 Hilbert 路径的注意力偏置。

        Args:
            levels_info: LevelsInfo 实例（可选）
            regions: (P11-3) 区域边界张量，形状为 (B, N, 4)
                     格式 [x1, y1, x2, y2]
            image_size: (P11-3) 图像边长，与 regions 配合使用

        Returns:
            Hilbert 偏置张量，形状为 (H, S, S) 或 (B, H, S, S)，若无效则返回 None
        """
        if not self.use_hilbert_bias:
            return None

        if self.hilbert_bias_impl is None:
            return None

        # P11-3: 优先使用 regions 直接计算 (LCA 模式)
        if regions is not None and image_size is not None:
            if isinstance(self.hilbert_bias_impl, LCAHilbertBias):
                return self.hilbert_bias_impl.forward_from_regions(regions, image_size)

        # 回退到 levels_info
        if levels_info is not None and levels_info.data.numel() > 0:
            return self.hilbert_bias_impl(levels_info)

        return None

    def _compute_level_bias(self, levels_info: LevelsInfo) -> Optional[torch.Tensor]:
        """计算基于层级差异的相对位置偏置。

        Args:
            levels_info: LevelsInfo 实例

        Returns:
            层级偏置张量，形状为 [B, H, S, S]，若无效则返回 None
        """
        if levels_info.data.numel() == 0:
            return None

        depths = levels_info.depths  # (B, S)
        level_diff = depths.unsqueeze(2) - depths.unsqueeze(1)  # (B, S, S)
        level_diff = level_diff.clamp(-self.max_level, self.max_level) + self.max_level
        rel_pos_bias = self.relative_pos_embedding(level_diff)  # (B, S, S, H)
        return rel_pos_bias.permute(0, 3, 1, 2)  # (B, H, S, S)

    # I106-1: Flash Attention 2 偏置格式转换
    def _prepare_flash_attn_bias(
        self,
        hilbert_bias: Optional[torch.Tensor],
        level_bias: Optional[torch.Tensor],
        affine_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """将多个偏置融合为 Flash Attention 2 兼容格式。

        数学形式:
            B_total = B_hilbert + B_level + B_affine

        Flash Attention 2 要求偏置形状: [B, 1, N, N] 或 [B, N, N]

        Args:
            hilbert_bias: [B, H, N, N] 或 [H, N, N]
            level_bias: [B, H, N, N] 或 [H, N, N]
            affine_bias: [B, dim, N, N] (dim = heads * head_dim)

        Returns:
            bias: [B, 1, N, N] 或 None (若无偏置)
        """
        biases = []

        # 处理 Hilbert 偏置
        if hilbert_bias is not None:
            if hilbert_bias.dim() == 3:
                # [H, N, N] -> [B, 1, N, N] (通过广播)
                biases.append(hilbert_bias.unsqueeze(0))
            else:
                # [B, H, N, N] -> [B, 1, N, N] (对 heads 求平均)
                biases.append(hilbert_bias.mean(dim=1, keepdim=True))

        # 处理 Level 偏置
        if level_bias is not None:
            if level_bias.dim() == 3:
                # [H, N, N] -> [B, 1, N, N]
                biases.append(level_bias.unsqueeze(0))
            else:
                # [B, H, N, N] -> [B, 1, N, N]
                biases.append(level_bias.mean(dim=1, keepdim=True))

        # 处理 Affine 偏置
        if affine_bias is not None:
            if affine_bias.dim() == 4:
                # I108-1: [B, dim, N, N] -> [B, 1, N, N] (对 heads 和 head_dim 求平均)
                # 然后应用归一化和缩放: B_a' = B_a / κ * γ
                # dim = heads * head_dim
                actual_head_dim = affine_bias.shape[1] // self.heads
                affine_avg = affine_bias.reshape(
                    affine_bias.shape[0], self.heads, actual_head_dim,
                    affine_bias.shape[2], affine_bias.shape[3]
                ).mean(dim=[1, 2]).unsqueeze(1)  # [B, 1, N, N]

                # I108-1: 归一化 + 缩放
                # B_a' = B_a / κ * γ (κ = exp(φ), γ = softplus(θ))
                affine_normalized = affine_avg / self.affine_bias_norm
                affine_scaled = affine_normalized * self.affine_bias_scale
                biases.append(affine_scaled)

        if not biases:
            return None

        # 融合所有偏置 (逐元素相加)
        # Flash Attention 2 会将 [B, 1, N, N] 广播到 [B, H, N, N]
        bias_total = biases[0]
        for bias in biases[1:]:
            bias_total = bias_total + bias

        # I109-1: 监控偏置量级（P-OPT: 仅在调试模式触发）
        # 数学依据: clamp(-50, 50) 是数值稳定设计，softmax(50) ≈ one-hot
        # P-OPT: 使用 getattr 保护避免 GPU-CPU 同步，仅调试时触发
        if (
            torch.is_grad_enabled()
            and bias_total.numel() > 0
            and getattr(self, '_debug_mode', False)
        ):
            bias_abs_max = bias_total.abs().max().item()
            # 当偏置量级接近 clamp 边界时发出警告
            if bias_abs_max > 40:  # 接近 50 的 80%
                import warnings
                warnings.warn(
                    f"I109-1: Bias magnitude {bias_abs_max:.1f} approaching clamp bound {LOGIT_CLAMP_BOUND}. "
                    f"This indicates strong attention bias signals."
                )

        # I108-1/I108-6: 安全 clamp 防止 softmax 饱和
        # 使用 LOGIT_CLAMP_BOUND (50.0) 常量
        # 数学依据: softmax(x > 50) ≈ one-hot，远小于 FP16 上界 65504
        return bias_total.clamp(min=-LOGIT_CLAMP_BOUND, max=LOGIT_CLAMP_BOUND)

    def _forward_hierarchical(
        self,
        x: torch.Tensor,
        levels_info: LevelsInfo,
        attention_mask: Optional[torch.Tensor],
        regions: Optional[torch.Tensor],
        image_size: Optional[int],
        batch: int,
        seq_len: int,
    ) -> torch.Tensor:
        """I103-1: 完全向量化的深度内独立Attention前向传播。

        数学形式:
            Attn(X) = ⊕_d softmax(Q K^T / √d_k ⊙ M_d M_d^T + B_d) V

        其中:
        - ⊕_d 表示按深度拼接
        - M_d = 1[depth = d] 是深度 d 的掩码
        - ⊙ 表示逐元素乘法 (掩码操作)
        - B_d 是深度 d 的 Hilbert 偏置 (批量计算)

        I103-1 优化:
        - 批量 Hilbert 计算: forward_from_regions(regions, image_size) 替代循环
        - 掩码注意力: 使用 depth_mask 限制深度内交互
        - 性能提升: 减少 87.5% kernel 启动 (B×D → D)

        Args:
            x: 输入张量，形状为 [B, N, D]
            levels_info: LevelsInfo 实例
            attention_mask: 注意力掩码
            regions: 区域边界张量
            image_size: 图像边长
            batch: batch size
            seq_len: 序列长度

        Returns:
            输出张量，形状为 [B, N, D]
        """
        # 提取深度信息
        depths = levels_info.depths  # [B, N]

        # LayerNorm + QKV投影
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # 初始化输出
        output = torch.zeros(batch, seq_len, self.heads * self.dim_head, device=x.device, dtype=x.dtype)

        # 深度缩放因子 (用于层级化注意力)
        if self._hierarchical_depth_scale is not None:
            depth_scales = self._hierarchical_depth_scale  # [max_level+1, heads]
        else:
            depth_scales = None

        # I103-1: 批量 Hilbert 偏置 (一次调用处理整个 batch)
        # hilbert_bias: [B, H, N, N] 或 None
        hilbert_bias_batch = None
        if self.use_hilbert_bias and self.hilbert_bias_impl is not None and regions is not None and image_size is not None:
            hilbert_bias_batch = self.hilbert_bias_impl.forward_from_regions(regions, image_size)

        # I108-3: 预分配循环内复用的索引张量，避免重复计算
        batch_indices = torch.arange(batch, device=x.device, dtype=torch.int64).view(-1, 1)  # [B, 1]
        pos_indices = torch.arange(seq_len, device=x.device, dtype=torch.int64).view(1, -1)  # [1, N]
        combined = (batch_indices * seq_len + pos_indices)  # [B, N]

        # P-OPT: 将 QK^T 和 Hilbert bias 移到循环外，只计算一次
        # 循环内只做 mask 和 gather，减少 80% 计算量
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # I103-1: 添加批量 Hilbert 偏置
        # I113-11: 量纲对齐 - 乘以 √d_k 确保与 QK^T / √d_k 量级相当
        if hilbert_bias_batch is not None:
            dots = dots + hilbert_bias_batch * self.hilbert_bias_scale * math.sqrt(self.dim_head)

        # 单次 Softmax（深度缩放在 gather 时应用）
        attn_full = self.attend(dots)

        # 预分配深度缩放缓冲区
        if depth_scales is not None:
            depth_scales_buffer = torch.zeros(batch, seq_len, self.heads * self.dim_head,
                                              device=x.device, dtype=x.dtype)

        # 深度循环：只做 mask 和 gather
        for d in range(self.max_level + 1):
            # 深度 d 的 token 掩码 [B, N]
            depth_mask = (depths == d)
            depth_count = depth_mask.sum(dim=1)  # [B]

            # 检查是否有深度 d 的 token（跨所有 batch）
            # P-OPT: 使用 torch.any() 代替 sum() == 0，避免 GPU-CPU 同步
            if not depth_count.any():
                continue

            # 掩码: 只保留深度 d 的 token 之间的注意力
            # 使用向量化操作避免临时张量创建
            mask_2d = depth_mask.unsqueeze(1) & depth_mask.unsqueeze(2)
            attn_d = attn_full.masked_fill(~mask_2d.unsqueeze(1), float('-inf'))

            # 将非深度 d 的 token 对应的行置零
            attn_d = attn_d.masked_fill(~depth_mask.unsqueeze(1).unsqueeze(3), 0)
            attn_d = self.dropout(attn_d)

            # 加权聚合
            out_d = torch.matmul(attn_d, v)  # [B, H, N, d_k]

            # 应用深度缩放（如果启用）
            if depth_scales is not None:
                scale_d = depth_scales[d].view(1, 1, self.heads * self.dim_head)
                out_d = out_d * scale_d

            # 使用 scatter_add 将结果放回对应位置
            out_flat = rearrange(out_d, "b h n d -> b n (h d)")
            valid_mask = depth_mask
            out_valid = out_flat[valid_mask]
            valid_indices = combined[valid_mask]

            batch_idx = valid_indices // seq_len
            pos_idx = valid_indices % seq_len
            output[batch_idx, pos_idx, :] = out_valid

        return self.to_out(output)

    @torch._dynamo.disable  # I99-1: 排除 torch.compile 追踪，避免 Triton 编译错误
    def forward(
        self,
        x: torch.Tensor,
        levels_info: Optional[LevelsInfo] = None,
        attention_mask: Optional[torch.Tensor] = None,
        regions: Optional[torch.Tensor] = None,
        image_size: Optional[int] = None,
    ) -> torch.Tensor:
        """前向传播。

        I98-4: levels_info 参数类型从 torch.Tensor 改为 LevelsInfo

        Args:
            x: 输入张量，形状为 [B, N, D]
            levels_info: LevelsInfo 实例（可选）
            attention_mask: 注意力掩码（可选）
            regions: 区域边界张量，形状为 [B, N, 4]，格式 [x1, y1, x2, y2]
            image_size: 图像边长，与 regions 配合使用

        Returns:
            输出张量，形状为 [B, N, D]
        """
        batch, seq_len, _ = x.shape

        # I98-4: 兼容 raw tensor 和 LevelsInfo 对象
        if isinstance(levels_info, torch.Tensor):
            # 转换为 LevelsInfo，确保数据类型为 Long
            if levels_info.dtype != torch.long:
                levels_info = levels_info.long()

            # 处理 2D tensor (S, Info) -> 添加 batch 维度
            if levels_info.dim() == 2:
                levels_info = levels_info.unsqueeze(0)  # (1, S, Info)

            # 从数据形状推断 max_level: info_dim = max_level + 1
            info_dim = levels_info.shape[-1]
            inferred_max_level = info_dim - 1

            levels_info = LevelsInfo(data=levels_info, max_level=inferred_max_level)

        # I97-10: 层级化注意力模式
        if self.use_hierarchical_attention and levels_info is not None and levels_info.data.numel() > 0:
            return self._forward_hierarchical(
                x, levels_info, attention_mask, regions, image_size, batch, seq_len
            )

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # P-OPT: 检查是否有偏置，无偏置时使用 Flash SDP
        has_level_scaling = self.use_level_scaling and levels_info is not None and levels_info.data.numel() > 0
        has_affine_bias = (self.use_affine_modulation and
                          regions is not None and image_size is not None and
                          self.affine_modulated_bias is not None)
        has_hilbert_bias = (self.use_hilbert_bias and
                           self.hilbert_bias_impl is not None and
                           (levels_info is not None or (regions is not None and image_size is not None)))

        # I106-1: 是否有任何偏置需要处理
        has_any_bias = has_level_scaling or has_affine_bias or has_hilbert_bias

        use_flash_sdp = not has_any_bias

        # I106-1: Flash Attention 2 集成
        use_flash_attn_2 = (
            FLASH_ATTN_AVAILABLE and
            has_any_bias and
            not self.use_hierarchical_attention  # 层级化注意力不支持
        )

        if use_flash_attn_2:
            # Flash Attention 2 路径：计算偏置并融合
            # Flash Attention 2 支持加性偏置，内存复杂度 O(N * B_tile)

            # 计算各偏置
            _affine_bias = None
            if has_affine_bias:
                _affine_bias = self.affine_modulated_bias(regions, image_size)

            _hilbert_bias = None
            if has_hilbert_bias:
                _hilbert_bias = self._compute_hilbert_bias(
                    levels_info=levels_info,
                    regions=regions,
                    image_size=image_size,
                )

            _level_bias = None
            if levels_info is not None and levels_info.data.numel() > 0:
                _level_bias = self._compute_level_bias(levels_info)

            # 融合偏置为 Flash Attention 2 格式
            combined_bias = self._prepare_flash_attn_bias(_hilbert_bias, _level_bias, _affine_bias)

            # 处理 level scaling (在 bias 融合后应用)
            level_scaling_tensor = None
            if has_level_scaling:
                assert self._level_scale_raw is not None
                depths = levels_info.depths
                depths_clamped = depths.clamp(min=0, max=self.max_level)
                level_scaling_tensor = F.softplus(self._level_scale_raw(depths_clamped))  # (B, S, H)

            # Flash Attention 2 需要变长序列参数
            cu_seqlens = torch.arange(
                0, seq_len + 1, device=q.device, dtype=torch.int32
            )  # [seq_len + 1]

            # 调用 Flash Attention 2
            attn = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens=cu_seqlens,
                max_seqlen=seq_len,
                bias=combined_bias,
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=self.scale,
            )

            # I24-11: 条件存储注意力权重
            if self.store_attn_weights:
                self._last_attn_weights = attn.detach()

            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

            # 应用 level scaling (需要在 softmax 之后、output 之前)
            if level_scaling_tensor is not None:
                # level_scaling: (B, S, H) -> (B, H, S, 1)
                level_scaling_tensor = level_scaling_tensor.permute(0, 2, 1).unsqueeze(-1)
                out = out * level_scaling_tensor

            out = rearrange(out, "b h n d -> b n (h d)")
            return self.to_out(out)

        if use_flash_sdp:
            # P-OPT: 使用 Flash SDP 优化 (30-50% 加速)
            # Flash SDP 不支持自定义偏置，所以只有无偏置时使用
            if attention_mask is not None:
                # 需要将 mask 转换为正确的格式
                attn_mask = attention_mask.float().squeeze(1).unsqueeze(-1)  # [B, 1, 1, N]
                attn_mask = (1.0 - attn_mask) * torch.finfo(q.dtype).min
            else:
                attn_mask = None

            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                scale=self.scale,
            )

            # I24-11: 条件存储注意力权重 (评估时启用)
            if self.store_attn_weights:
                self._last_attn_weights = attn.detach()

            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
            out = rearrange(out, "b h n d -> b n (h d)")
            return self.to_out(out)

        # 标准实现（有偏置时使用）
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # A19: 移除可学习 scale_weights，仅使用标准 1/√d_k
        # Var(dots) = 1.0 (理论最优)，softmax 输入在 O(1) 量级

        if has_level_scaling:
            # Type guard: guaranteed non-None when use_level_scaling is True
            assert self._level_scale_raw is not None

            depths = levels_info.depths  # [B, S]
            # I98-4: clamp depths to [0, max_level] to handle padding sentinel (-1)
            depths_clamped = depths.clamp(min=0, max=self.max_level)
            # I34-15: 移除 2D 分支死代码，levels_info 始终为 3D
            # P11-4: Softplus 约束确保 level_scales ∈ (0, +∞)
            level_scales = F.softplus(self._level_scale_raw(depths_clamped))  # (B, S, H)
            level_scales = level_scales.permute(0, 2, 1).unsqueeze(-1)  # (B, H, S, 1)

            dots = dots * level_scales

        if levels_info is not None:
            # I31-3: 仿射调制优先 (当启用且 regions 可用时)
            if self.use_affine_modulation and regions is not None and image_size is not None:
                affine_bias = self.affine_modulated_bias(regions, image_size)
                if affine_bias is not None:
                    # affine_bias: [B, dim, N, N]
                    # 需要广播到 [B, H, N, N]，通过对 dim 维度求均值
                    # 因为 dim = heads * head_dim，求均值后形状保持 [B, dim, N, N]
                    # 然后需要重塑为 [B, H, N, N]
                    if affine_bias.dim() == 4:
                        # I34-1 Fix: Reshape and average over head_dim to preserve per-head independence
                        # affine_bias: [B, dim, N, N] where dim = heads * head_dim
                        # Reshape to: [B, heads, head_dim, N, N]
                        # Mean over head_dim to get: [B, heads, N, N]
                        N = affine_bias.shape[-1]
                        # I35 Fix: Dynamically compute head_dim from affine_bias shape
                        # This handles cases where dim_head was initialized with default=64
                        actual_head_dim = affine_bias.shape[1] // self.heads
                        affine_bias_avg = affine_bias.reshape(
                            batch, self.heads, actual_head_dim, N, N
                        ).mean(dim=2)

                        # I108-1: 归一化 + 缩放
                        # B_a' = B_a / κ * γ (与 Flash Attention 路径一致)
                        affine_normalized = affine_bias_avg / self.affine_bias_norm
                        affine_scaled = affine_normalized * self.affine_bias_scale
                        dots = dots + affine_scaled
            else:
                hilbert_bias = self._compute_hilbert_bias(
                    levels_info=levels_info,
                    regions=regions,
                    image_size=image_size,
                )
                if hilbert_bias is not None:
                    # hilbert_bias: (H, S, S) or (B, H, S, S)
                    # I113-11: 量纲对齐 - 乘以 √d_k 确保与 QK^T / √d_k 量级相当
                    dim_scale = math.sqrt(self.dim_head)
                    if hilbert_bias.dim() == 3:
                        dots = dots + hilbert_bias.unsqueeze(0) * self.hilbert_bias_scale * dim_scale
                    else:
                        dots = dots + hilbert_bias * self.hilbert_bias_scale * dim_scale

            # I113-11: 量纲对齐常量 - 定义在 level_bias 分支外部
            dim_scale = math.sqrt(self.dim_head)
            level_bias = self._compute_level_bias(levels_info)
            if level_bias is not None:
                # level_bias: (H, S, S) or (B, H, S, S)
                # I113-11: 量纲对齐 - 乘以 √d_k 确保与 QK^T / √d_k 量级相当
                if level_bias.dim() == 3:
                    dots = dots + level_bias.unsqueeze(0) * self.level_bias_scale * dim_scale
                else:
                    dots = dots + level_bias * self.level_bias_scale * dim_scale

        if attention_mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            dots.masked_fill_(~attention_mask.bool(), mask_value)

        attn = self.attend(dots)

        # I24-11: 条件存储注意力权重 (评估时启用)
        if self.store_attn_weights:
            self._last_attn_weights = attn.detach()

        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# ==================== I31/I35: 形状-尺度编码器 ====================


class ShapeScaleEncoder(nn.Module):
    """形状-尺度编码器 (I31 原始，I35: 迭代改进，I98-3: 协议驱动配置化)

    数学形式化
    ==========

    问题背景 (I35-1):
        原始实现使用独立投影处理 aspect_ratio r 和 normalized_area s:
            r = log(w/h) ∈ (-∞, ∞)
            s = (w/W)(h/H) ∈ [0, 1]

        当 s 固定时，r 的信息被独立编码，导致特征相关灾难。

    I35 Phase 1: 特征解耦
        使用单一 MLP 编码组合特征 [r, s]，消除独立投影带来的相关性问题。

    I35 Phase 2: 效率优化
        新增直接偏置计算: B[i,j] = BiasEncoder([r_i, s_i, r_j, s_j])
        FLOPs 减少 85%: 1.87M → 0.28M

    I35-5: 特征归一化
        r_norm = tanh(r / (1 + |r|)) ∈ (-1, 1)
        s_log = log(s + ε) ∈ (-∞, 0]
        消除异构性导致的优化偏差。

    I35-2: 非零初始化
        τ = 0.1 确保训练初期有梯度回传。

    I98-3: 协议驱动配置
        使用 ShapeScaleEncoderConfig 替代硬编码参数。

    架构:
        encoder: [r, s] -> Linear(2, hidden) -> GELU -> Linear(hidden, dim/2) -> GELU -> Linear(dim/2, dim)
        bias_encoder: [r_i, s_i, r_j, s_j] -> Linear(4, hidden) -> GELU -> Linear(hidden, hidden/2) -> GELU -> Linear(hidden/2, 1)

    属性
    ----
    config : ShapeScaleEncoderConfig
        编码器配置 (I98-3)
    encoder : nn.Sequential
        组合特征编码器: 2 -> hidden -> dim/2 -> dim
    bias_encoder : nn.Sequential
        直接偏置编码器: 4 -> hidden -> hidden/2 -> 1 (I35 Phase 2)
    shape_scale_weight : nn.Parameter
        可学习权重 (初始化值来自 config)
    """

    def __init__(
        self,
        dim: int,
        config: Optional[ShapeScaleEncoderConfig] = None,
    ):
        """初始化形状-尺度编码器 (I98-3 协议驱动版本).

        参数
        ----
        dim : int
            输出嵌入维度
        config : ShapeScaleEncoderConfig, optional
            编码器配置，为 None 时使用默认配置
        """
        super().__init__()

        if config is None:
            config = ShapeScaleEncoderConfig()

        self.config = config
        self.dim = dim
        self.hidden_dim = config.hidden_dim

        # 组合特征编码器 (I35-1 核心改进)
        # 输入: [r, s] 形状 [B, N, 2]
        # 输出: [B, N, dim]
        self.encoder = nn.Sequential(
            nn.Linear(2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )

        # I35 Phase 2: 偏置编码器 (直接计算 B[i,j])
        # 输入: [r_i, s_i, r_j, s_j] 形状 [B, N, N, 4]
        # 输出: [B, N, N, 1] 标量偏置
        self.bias_encoder = nn.Sequential(
            nn.Linear(4, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )

        # 可学习权重 (非零初始化，渐进启用)
        # I35-2 修复: τ = 0 阻塞梯度，改用 config.weight_init 确保训练初期有梯度
        self.shape_scale_weight = nn.Parameter(
            torch.tensor(config.weight_init)
        )

        # 初始化权重
        self._init_weights()

    def get_config(self) -> ShapeScaleEncoderConfig:
        """获取当前配置 (I98-3).

        返回
        ----
        ShapeScaleEncoderConfig
            当前配置，包含实际运行参数
        """
        return ShapeScaleEncoderConfig(
            hidden_dim=self.hidden_dim,
            output_dim=self.dim,
            weight_init=self.shape_scale_weight.item(),
        )

    def _init_weights(self):
        """初始化权重。"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        regions: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """计算形状-尺度嵌入 (I35 Phase 1 特征解耦版本)。

        参数
        ----
        regions : torch.Tensor
            区域边界张量，形状 [B, N, 4]
            格式: [x1, y1, x2, y2]
        image_size : Tuple[int, int]
            (W, H) 图像尺寸

        返回
        ----
        torch.Tensor
            形状-尺度嵌入，形状 [B, N, dim]
        """
        B, N, _ = regions.shape
        W, H = image_size

        # 计算纵横比和面积 [B, N]
        # I99-1 OPT: 使用 EPS 常量替代硬编码 1e-8
        aspect_ratios, normalized_areas = compute_region_shape_scale(
            regions, (W, H), epsilon=EPS
        )

        # I35-5: 特征归一化 (异质性特征空间对齐)
        # 原始特征空间异构性:
        #   aspect_ratios = log(w/h) ∈ (-∞, ∞)  无界
        #   normalized_areas = (w/W)(h/H) ∈ [0, 1]  有界
        # 归一化后:
        #   r_norm = tanh(r / (1 + |r|)) ∈ (-1, 1)  有界、对称
        #   s_log = log(s + ε) ∈ (-∞, 0]           展开到对称空间
        # 数学效果: Var(r_norm) ≈ Var(s_log)，消除异构性导致的优化偏差
        aspect_ratios_norm = torch.tanh(aspect_ratios / (1 + aspect_ratios.abs()))
        normalized_areas_log = torch.log(normalized_areas + EPS)

        # 组合特征 [B, N, 2] - I35-1 核心改进
        combined = torch.stack([aspect_ratios_norm, normalized_areas_log], dim=-1)

        # 编码 [B, N, dim]
        output = self.encoder(combined)

        # 应用可学习权重
        output = output * self.shape_scale_weight

        return output

    def forward_with_bias(
        self,
        regions: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """直接计算形状-尺度偏置矩阵 [B, 1, N, N] (I35 Phase 2: 效率优化)。

        相比 embed + 外积方案，FLOPs 减少 85%:
            原方案: embed O(N×d) + 外积 O(N²×d)
            新方案: 直接计算 O(N²×d)

        数学形式化
        ==========

        直接偏置计算:
            B_SS[i,j] = Encoder([r_i, s_i, r_j, s_j])_mean

        其中:
            r_i = tanh(aspect_i / (1 + |aspect_i|)) ∈ (-1, 1)
            s_i = log(area_i + ε) ∈ (-∞, 0]

        复杂度对比 (B=8, N=85, d=256):
            原方案: 1,871,360 FLOPs
            新方案: 280,960 FLOPs
            减少: 85%

        参数
        ----
        regions : torch.Tensor
            区域边界张量，形状 [B, N, 4]
        image_size : Tuple[int, int]
            (W, H) 图像尺寸

        返回
        ----
        torch.Tensor
            形状-尺度偏置，形状 [B, 1, N, N]
        """
        B, N, _ = regions.shape
        W, H = image_size

        # 计算并归一化特征 [B, N]
        # I99-1 OPT: 使用 EPS 常量替代硬编码 1e-8
        aspect_ratios, normalized_areas = compute_region_shape_scale(
            regions, (W, H), epsilon=EPS
        )
        aspect_ratios_norm = torch.tanh(aspect_ratios / (1 + aspect_ratios.abs()))
        normalized_areas_log = torch.log(normalized_areas + EPS)

        # 构建所有组合的特征 [B, N, N, 4]
        # r_i, s_i: 查询对 (query pair) 的特征
        # r_j, s_j: 键对 (key pair) 的特征
        r_i = aspect_ratios_norm.unsqueeze(2).expand(-1, -1, N)  # [B, N, N]
        r_j = aspect_ratios_norm.unsqueeze(1).expand(-1, N, -1)  # [B, N, N]
        s_i = normalized_areas_log.unsqueeze(2).expand(-1, -1, N)
        s_j = normalized_areas_log.unsqueeze(1).expand(-1, N, -1)

        # 特征拼接 [B, N, N, 4]
        features = torch.stack([r_i, s_i, r_j, s_j], dim=-1)

        # 直接编码为标量偏置 [B, N, N, 1]
        bias = self.bias_encoder(features)  # [B, N, N, 1]

        # 调整维度 [B, 1, N, N] 以匹配注意力矩阵
        return bias.permute(0, 3, 1, 2) * self.shape_scale_weight


class LCAHilbertBiasWithShapeScale(nn.Module):
    """带形状-尺度修正的 LCA Hilbert 偏置 (I31, I98-3: 协议驱动配置)

    数学形式化
    ==========

    基础 LCA 偏置:
        B_LCA[i,j] = LCAEmbed(LCA(i,j))

    形状-尺度偏置:
        B_SS[i,j] = <ShapeScaleEncoder(c_i), ShapeScaleEncoder(c_j)>

    组合偏置:
        B_final[i,j] = B_LCA[i,j] + B_SS[i,j]
        其中 B_SS 包含 shape_scale_weight 缩放

    非零初始化 (I35-2):
        τ = 0.1 时，训练初期 B_SS 就有梯度回传，加速收敛

    I98-3: 协议驱动配置
        使用 ShapeScaleEncoderConfig 替代硬编码参数。

    属性
    ----
    lca_embedding : nn.Embedding
        LCA 深度嵌入层
    shape_scale_encoder : ShapeScaleEncoder
        形状-尺度编码器
    shape_scale_weight : nn.Parameter
        可学习组合权重
    config : ShapeScaleEncoderConfig
        编码器配置 (I98-3)
    """

    def __init__(
        self,
        dim: int,
        max_level: int,
        lca_embedding_dim: Optional[int] = None,
        enable_shape_scale: bool = True,
        shape_scale_dim: Optional[int] = None,
        shape_scale_hidden_dim: int = 64,
        shape_scale_config: Optional[ShapeScaleEncoderConfig] = None,
        use_fp16: bool = False,  # I104-3: FP16 存储选项
    ):
        """初始化带形状-尺度修正的 LCA Hilbert 偏置 (I35: 特征解耦重构, I98-3 协议驱动).

        参数
        ----
        dim : int
            注意力维度
        max_level : int
            最大四叉树深度
        lca_embedding_dim : int, optional
            LCA 嵌入维度，默认等于 dim
        enable_shape_scale : bool, optional
            是否启用形状-尺度修正，默认 True
        shape_scale_dim : int, optional
            形状-尺度嵌入维度，默认等于 dim
        shape_scale_hidden_dim : int, optional
            形状-尺度编码器隐藏层维度，默认 64 (I35: 与 ShapeScaleEncoder 默认值一致)
        shape_scale_config : ShapeScaleEncoderConfig, optional
            形状-尺度编码器配置 (I98-3)
        use_fp16 : bool, optional
            (I104-3) 是否使用 FP16 存储 LCA embedding，默认 False
        """
        super().__init__()
        self.use_fp16 = use_fp16

        # I98-3: 使用配置类
        if shape_scale_config is None:
            shape_scale_config = ShapeScaleEncoderConfig(hidden_dim=shape_scale_hidden_dim)
        self.config = shape_scale_config

        # LCA 嵌入层
        lca_embed_dim = lca_embedding_dim or dim
        self.lca_embedding = nn.Embedding(
            num_embeddings=max_level + 1,
            embedding_dim=lca_embed_dim
        )

        # I104-3: 转换为 FP16 以节省内存
        if use_fp16:
            self.lca_embedding.weight = nn.Parameter(
                self.lca_embedding.weight.to(torch.float16)
            )

        # 形状-尺度编码器 (I35: 统一使用特征解耦版本, I98-3: 使用配置类)
        self.enable_shape_scale = enable_shape_scale
        if enable_shape_scale:
            self.shape_scale_encoder = ShapeScaleEncoder(
                dim=shape_scale_dim or dim,
                config=shape_scale_config
            )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重。"""
        # LCA 嵌入: 深度越大（越邻近）偏置越高
        with torch.no_grad():
            for d in range(self.lca_embedding.num_embeddings):
                # 对数衰减初始化: depth d -> scale log(d+1)
                scale = 0.1 * (1 + torch.log(torch.tensor(d + 1.0)))
                self.lca_embedding.weight[d].fill_(scale)

    @torch._dynamo.disable  # I99-1: 排除 torch.compile 追踪，避免 Triton 编译错误
    def forward_from_regions(
        self,
        regions: torch.Tensor,
        image_size: int,
    ) -> torch.Tensor:
        """从 regions 计算 LCA Hilbert 偏置。

        参数
        ----
        regions : torch.Tensor
            区域边界张量，形状 [B, N, 4]
        image_size : int
            图像边长

        返回
        ----
        torch.Tensor
            偏置矩阵，形状 [B, H, N, N] 或 [B, dim, N, N]
        """
        # 委托给现有实现
        return self._compute_lca_bias_from_regions(regions, image_size)

    @torch._dynamo.disable  # I99-1: 排除 torch.compile 追踪，避免 Triton 编译错误
    def _compute_lca_bias_from_regions(
        self,
        regions: torch.Tensor,
        image_size: int,
    ) -> torch.Tensor:
        """从 regions 计算 LCA 偏置 (内部方法)。"""
        B, N, _ = regions.shape

        if N == 0:
            return torch.zeros(B, 1, N, N, device=regions.device)

        # 转换为整数坐标 (VectorizedPathEncoder 需要整数)
        regions_int = regions.long()

        # 从 regions 计算四叉树路径
        paths = VectorizedPathEncoder.compute_paths_from_regions(
            regions_int, image_size, self.lca_embedding.num_embeddings - 1
        )

        # 计算 LCA 深度矩阵
        lca_depths = VectorizedPathEncoder.compute_common_ancestor_depth(paths)
        lca_depths = lca_depths.clamp(0, self.lca_embedding.num_embeddings - 1)

        # 嵌入 [B, N, N, dim]
        bias = self.lca_embedding(lca_depths)

        # 调整形状 [B, dim, N, N]
        return bias.permute(0, 3, 1, 2)

    @torch._dynamo.disable  # I99-1: 排除 torch.compile 追踪，避免 Triton 编译错误
    def forward_with_shape_scale(
        self,
        regions: torch.Tensor,
        image_size: int,
    ) -> torch.Tensor:
        """带形状-尺度修正的偏置计算。

        参数
        ----
        regions : torch.Tensor
            区域边界张量，形状 [B, N, 4]
        image_size : int
            图像边长

        返回
        ----
        torch.Tensor
            组合偏置，形状 [B, dim, N, N]
        """
        # 1. LCA 偏置 [B, dim, N, N]
        lca_bias = self._compute_lca_bias_from_regions(regions, image_size)

        if not self.enable_shape_scale:
            return lca_bias

        # 2. 形状-尺度偏置 [B, 1, N, N] (I35 Phase 2: 直接计算)
        # 相比原方案 embed + 外积，FLOPs 减少 85%
        shape_bias = self.shape_scale_encoder.forward_with_bias(
            regions, (image_size, image_size)
        )

        # 3. 组合偏置: B_final = B_LCA + B_SS
        # shape_bias 已包含 shape_scale_weight 缩放
        combined_bias = lca_bias + shape_bias

        return combined_bias


# ==================== I31-3: 面积编码与仿射调制 ====================


class AreaEncoder(nn.Module):
    """面积编码器 (I31-3, I32-7, I98-3: 协议驱动配置化)

    数学形式化
    ==========

    面积归一化公式 (用户指定):

        .. math::
            f_{{area}} = \\frac{{\\log(s_{{patch}} + 1)}}{{\\log(S_{{total}} + 1)}}

    傅里叶特征 (I32-7 动态频率 + 软截断门控):

        .. math::
            f_k = \\pi \\cdot b^k \\cdot g_k(freq_k, L_{{norm}})

        其中:
            b = freq_base (频率基数，可配置)
            g_k = \\text{{CosineGate}}(freq_k, L_{{norm}})  (软截断门控)

    I98-3: 协议驱动配置
        使用 AreaEncoderConfig 替代硬编码参数。

    属性
    ----
    config : AreaEncoderConfig
        编码器配置 (I98-3)
    fourier_levels : int
        傅里叶频率级别数
    fourier_dim : int
        傅里叶特征维度 = 2 × fourier_levels
    fourier_proj : nn.Linear
        傅里叶特征投影层
    mlp : nn.Sequential
        MLP 投影层
    area_weight : nn.Parameter
        可学习权重 (零初始化)
    """

    def __init__(
        self,
        dim: int,
        config: Optional[AreaEncoderConfig] = None,
    ):
        """初始化面积编码器 (I98-3 协议驱动版本).

        参数
        ----
        dim : int
            输出嵌入维度
        config : AreaEncoderConfig, optional
            编码器配置，为 None 时使用默认配置
        """
        super().__init__()

        if config is None:
            config = AreaEncoderConfig()

        self.config = config
        self.dim = dim
        self.fourier_levels = config.fourier_levels
        self.freq_base = config.freq_base
        self.fourier_dim = config.fourier_levels * 2  # sin + cos

        # I98-3: 从配置读取 Nyquist 裁剪比例
        self.cutoff_ratio = config.cutoff_ratio

        # fourier_levels=0 时禁用编码器
        if config.fourier_levels > 0:
            # 傅里叶特征投影: 2L -> hidden
            self.fourier_proj = nn.Linear(self.fourier_dim, config.hidden_dim)

            # MLP 投影: hidden -> dim
            self.mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, dim)
            )
        else:
            # 禁用模式：创建占位符
            self.fourier_proj = None
            self.mlp = None

        # 可学习权重 (零初始化)
        self.area_weight = nn.Parameter(torch.zeros(1))

        # 初始化权重
        self._init_weights()

    def get_config(self) -> AreaEncoderConfig:
        """获取当前配置 (I98-3).

        返回
        ----
        AreaEncoderConfig
            当前配置，包含实际运行参数
        """
        return AreaEncoderConfig(
            fourier_levels=self.fourier_levels,
            freq_base=self.freq_base,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.dim,
            cutoff_ratio=self.cutoff_ratio,  # I98-3: 包含 cutoff_ratio
        )

    def _init_weights(self):
        """初始化权重。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _compute_nyquist_normalized_size(
        self,
        regions: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """计算归一化 Patch 尺寸 (用于 Nyquist 约束)。

        数学形式化
        ==========

        归一化尺寸:

            .. math::
                L_{{norm}} = \\frac{{L_{{patch}}}}{{L_{{image}}}}

        其中 L 为 patch 边长的几何平均:

            .. math::
                L_{{patch}} = \\sqrt{{w \\times h}}, \\quad L_{{image}} = \\sqrt{{W \\times H}}

        Nyquist 频率:

            .. math::
                \\omega_{{Nyquist}} = \\frac{{\\pi}}{{L_{{norm}}}}

        参数
        ----
        regions : torch.Tensor
            区域边界张量，形状 [B, N, 4]
        image_size : Tuple[int, int]
            (W, H) 图像尺寸

        返回
        ----
        torch.Tensor
            归一化尺寸 L_norm，形状 [B, N]
        """
        B, N, _ = regions.shape
        W, H = image_size

        # 计算 Patch 尺寸 [B, N]
        widths = torch.abs(regions[..., 2] - regions[..., 0])  # [B, N]
        heights = torch.abs(regions[..., 3] - regions[..., 1])  # [B, N]

        # Patch 边长的几何平均
        L_patch = torch.sqrt(widths * heights + EPS)  # [B, N]

        # 图像尺寸的几何平均
        L_image = math.sqrt(W * H)

        # 归一化尺寸
        L_norm = L_patch / L_image

        return L_norm

    def _compute_dynamic_fourier_features(
        self,
        area_scores: torch.Tensor,
        L_norm: torch.Tensor,
    ) -> torch.Tensor:
        """计算动态频率傅里叶特征 (I32-7 核心改进, P-OPT-5 向量化)。

        数学形式化
        ==========

        动态频率:

            .. math::
                f_k = \\pi \\cdot b^k, \\quad b = 2

        软截断门控 (Cosine Gate):

            .. math::
                \\omega_{{cutoff}} = 0.8 \\cdot \\omega_{{Nyquist}} = 0.8 \\cdot \\frac{{\\pi}}{{L_{{norm}}}}

                g_k = \\begin{cases}}
                    1 & f_k \\leq \\omega_{{cutoff}} \\\\
                    \\frac{{1}}{{2}}(1 + \\cos\\frac{{\\pi(f_k - \\omega_{{cutoff}})}}
                                  {{\\omega_{{Nyquist}} - \\omega_{{cutoff}}}}) & \\omega_{{cutoff}} < f_k < \\omega_{{Nyquist}} \\\\
                    0 & f_k \\geq \\omega_{{Nyquist}}
                \\end{cases}

        最终特征:

            .. math::
                \\gamma_k(f) = g_k \\cdot [\\sin(f_k \\cdot f), \\cos(f_k \\cdot f)]

        P-OPT-5 向量化改进:
            - 消除 Python for 循环，使用张量广播一次性计算所有频率
            - 复杂度: O(L) → O(1) 次 kernel 调用
            - 内存: 相同 (B, N, 2L)

        参数
        ----
        area_scores : torch.Tensor
            归一化面积分数，形状 [B, N]
        L_norm : torch.Tensor
            归一化 Patch 尺寸，形状 [B, N]

        返回
        ----
        torch.Tensor
            傅里叶特征，形状 [B, N, 2L]
        """
        B, N = area_scores.shape

        # Nyquist 频率: ω_Nyquist = π / L_norm
        # [B, N] -> [B, N, 1] 用于广播
        omega_nyquist = (math.pi / (L_norm + EPS)).unsqueeze(-1)  # [B, N, 1]

        # 截止频率: 从配置读取，默认 80% Nyquist
        # I98-3: omega_cutoff = cutoff_ratio * omega_nyquist
        # [B, N] -> [B, N, 1]
        omega_cutoff = omega_nyquist * self.cutoff_ratio  # [B, N, 1]

        # P-OPT-5: 预计算所有频率序列 (向量化关键)
        # [L] -> [1, 1, L]
        base_freqs = torch.tensor(
            [math.pi * (self.freq_base ** k) for k in range(self.fourier_levels)],
            device=area_scores.device,
            dtype=area_scores.dtype,
        ).view(1, 1, self.fourier_levels)  # [1, 1, L]

        # P-OPT-5: 一次性计算所有频率的门控 (向量化)
        # freq_k: [1, 1, L], omega_cutoff: [B, N, 1], omega_nyquist: [B, N, 1]
        # gate: [B, N, L]

        # 门控计算: 向量化 where 操作
        # gate = 1 if freq_k <= omega_cutoff (广播)
        # gate = 0.5 * (1 + cos(...)) if omega_cutoff < freq_k < omega_nyquist
        # gate = 0 if freq_k >= omega_nyquist

        # I102-2: 使用 clamp 替代 epsilon，保护 FP16 数值稳定性
        # 语义: 分母下界为 1e-6，确保门控函数行为可控
        denom = omega_nyquist - omega_cutoff  # [B, N, 1]
        denom_safe = denom.clamp(min=1e-6)    # FP16 安全下界

        gate_full = torch.where(
            base_freqs <= omega_cutoff,
            torch.ones(1, device=area_scores.device, dtype=area_scores.dtype),
            torch.where(
                base_freqs < omega_nyquist,
                0.5 * (1 + torch.cos(
                    math.pi * (base_freqs - omega_cutoff) / denom_safe
                )),
                torch.zeros(1, device=area_scores.device, dtype=area_scores.dtype)
            )
        )  # [B, N, L]

        # P-OPT-5: 一次性计算所有频率的傅里叶特征 (向量化)
        # area_scores: [B, N, 1] -> [B, N, L]
        area_expanded = area_scores.unsqueeze(-1)  # [B, N, 1]
        freq_times_area = base_freqs * area_expanded  # [B, N, L]

        # sin 和 cos 一次性计算
        sin_all = torch.sin(freq_times_area) * gate_full  # [B, N, L]
        cos_all = torch.cos(freq_times_area) * gate_full  # [B, N, L]

        # P-OPT-5: 交错 sin/cos 并展平 (消除 Python list append)
        # [B, N, L] + [B, N, L] -> [B, N, 2L]
        gamma = torch.stack([sin_all, cos_all], dim=-1).view(B, N, 2 * self.fourier_levels)

        return gamma

    def forward(
        self,
        regions: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """计算面积嵌入 (I98-3: 支持禁用).

        参数
        ----
        regions : torch.Tensor
            区域边界张量，形状 [B, N, 4]
            格式: [x1, y1, x2, y2]
        image_size : int or Tuple[int, int]
            图像尺寸，可以是整数或 (W, H) 元组

        返回
        ----
        torch.Tensor
            面积嵌入，形状 [B, N, dim]
        """
        B, N, _ = regions.shape

        # I98-3: fourier_levels=0 时禁用编码器，返回零张量
        if self.fourier_levels == 0:
            return torch.zeros(B, N, self.dim, device=regions.device)

        # 处理 image_size 格式：支持 int 或 (W, H) 元组
        if isinstance(image_size, int):
            image_size_tuple = (image_size, image_size)
        else:
            image_size_tuple = image_size
        W, H = image_size_tuple

        # 1. 计算归一化面积分数
        # I99-1 OPT: 使用 EPS 常量替代硬编码 1e-8
        area_scores = compute_normalized_area(regions, image_size, epsilon=EPS)
        # area_scores: [B, N]

        # 2. 计算归一化 Patch 尺寸 (用于动态频率)
        L_norm = self._compute_nyquist_normalized_size(regions, image_size_tuple)
        # L_norm: [B, N]

        # 3. 动态频率傅里叶特征编码 (I32-7)
        gamma = self._compute_dynamic_fourier_features(area_scores, L_norm)
        # gamma: [B, N, 2L]

        # 4. MLP 投影
        area_emb = self.mlp(self.fourier_proj(gamma))

        # 5. 应用可学习权重
        area_emb = area_emb * self.area_weight

        # I34-11: L2 归一化
        # 防止点积值域过大导致 sin 产生高频噪声
        area_emb = F.normalize(area_emb, p=2, dim=-1)

        return area_emb


class AffineModulatedBias(nn.Module):
    """仿射调制注意力偏置 (I31-3, I98-3: 协议驱动配置化)

    数学形式化
    ==========

    仿射调制公式 (用户指定):
        B_{\text{final}} = γ(s_i, s_j) ⊙ B_{\text{spatial}} + β(s_i, s_j)

    其中:
        γ(s_i, s_j) = σ(MLP_γ(p_s))     # 缩放因子
        β(s_i, s_j) = MLP_β(p_s)        # 偏置因子
        p_s = area_emb[i] · area_emb[j] # 面积相似性

    I98-3: 协议驱动配置
        使用 AttentionEncoderConfig 替代硬编码参数。

    属性
    ----
    config : AttentionEncoderConfig
        编码器配置 (I98-3)
    lca_embedding : nn.Embedding
        LCA 深度嵌入层
    area_encoder : AreaEncoder
        面积编码器
    scale_net : nn.Sequential
        缩放网络: area_sim → [0, 1]
    bias_net : nn.Sequential
        偏置网络: area_sim → (-1, 1)
    residual_alpha : nn.Parameter
        残差权重 (零初始化)
    """

    def __init__(
        self,
        dim: int,
        max_level: int,
        config: Optional[AttentionEncoderConfig] = None,
        use_fp16: bool = False,  # I104-3: FP16 存储选项
    ):
        """初始化仿射调制偏置 (I98-3 协议驱动版本).

        参数
        ----
        dim : int
            注意力维度
        max_level : int
            最大四叉树深度
        config : AttentionEncoderConfig, optional
            编码器配置，为 None 时使用默认配置
        use_fp16 : bool, optional
            (I104-3) 是否使用 FP16 存储 LCA embedding，默认 False
        """
        super().__init__()
        self.use_fp16 = use_fp16

        if config is None:
            config = AttentionEncoderConfig()

        self.config = config
        self.dim = dim
        self.max_level = max_level
        self.enable_area_modulation = config.area.fourier_levels > 0

        # I98-3: 从配置获取 scale_init_factor
        scale_init_factor = config.lca.scale_init_factor

        # 空间偏置 (LCA) - 始终使用 dim 作为嵌入维度以保持一致性
        # config.lca.embedding_dim 仅用于配置记录，实际使用 dim
        self.lca_embedding = nn.Embedding(
            num_embeddings=max_level + 1,
            embedding_dim=dim
        )

        # I104-3: 转换为 FP16 以节省内存
        if use_fp16:
            self.lca_embedding.weight = nn.Parameter(
                self.lca_embedding.weight.to(torch.float16)
            )

        # 面积编码器和调制网络
        if self.enable_area_modulation:
            # 使用配置的 AreaEncoder
            self.area_encoder = AreaEncoder(
                dim=dim,
                config=config.area
            )

            # 傅里叶特征维度
            fourier_dim = config.area.fourier_levels * 2  # sin + cos

            # 缩放网络: fourier_features → [0, 1]
            self.scale_net = nn.Sequential(
                nn.Linear(fourier_dim, dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, dim),
                nn.Sigmoid()  # γ ∈ (0, 1)
            )

            # 偏置网络: fourier_features → ℝ
            self.bias_net = nn.Sequential(
                nn.Linear(fourier_dim, dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, dim),
                nn.Tanh()  # β ∈ (-1, 1)
            )

            # 残差权重 (零初始化)
            self.residual_alpha = nn.Parameter(torch.zeros(1))

        # I31-P2: 添加 ShapeScale 编码器
        self.enable_shape_scale = config.shape_scale.enabled
        if self.enable_shape_scale:
            self.shape_scale_encoder = ShapeScaleEncoder(
                dim=dim,
                config=config.shape_scale
            )
            # 形状编码 → 偏置网络
            # Fourier 特征: 4 levels × 2 (sin/cos) = 8 维
            shape_fourier_dim = 8
            self.shape_bias_net = nn.Sequential(
                nn.Linear(shape_fourier_dim, shape_fourier_dim),
                nn.GELU(),
                nn.Linear(shape_fourier_dim, dim),
                nn.Tanh()  # β_shape ∈ (-1, 1)
            )
            # 形状调制权重
            self.shape_scale_alpha = nn.Parameter(torch.zeros(1))

        self._init_weights(scale_init_factor)

    def _init_weights(self, scale_init_factor: float = 0.1):
        """初始化权重 (I98-3: 使用可配置的 scale_init_factor)."""
        # LCA 嵌入: 深度越大（越邻近）偏置越高
        # I98-3: 使用配置的 scale_init_factor
        with torch.no_grad():
            for d in range(self.lca_embedding.num_embeddings):
                scale = scale_init_factor * (1 + torch.log(torch.tensor(d + 1.0)))
                self.lca_embedding.weight[d].fill_(scale)

    def _compute_lca_bias_from_regions(
        self,
        regions: torch.Tensor,
        image_size: int,
    ) -> torch.Tensor:
        """从 regions 计算 LCA 偏置 (内部方法)。"""
        B, N, _ = regions.shape

        if N == 0:
            return torch.zeros(B, 1, N, N, device=regions.device)

        # 转换为整数坐标
        regions_int = regions.long()

        # 从 regions 计算四叉树路径
        paths = VectorizedPathEncoder.compute_paths_from_regions(
            regions_int, image_size, self.max_level
        )

        # 计算 LCA 深度矩阵
        lca_depths = VectorizedPathEncoder.compute_common_ancestor_depth(paths)
        lca_depths = lca_depths.clamp(0, self.max_level)

        # 嵌入 [B, N, N, dim]
        bias = self.lca_embedding(lca_depths)

        # 调整形状 [B, dim, N, N]
        return bias.permute(0, 3, 1, 2)

    def forward(
        self,
        regions: torch.Tensor,
        image_size: int,
    ) -> torch.Tensor:
        """计算仿射调制偏置 (I31-P2: 添加 ShapeScale 支持).

        数学形式化
        ==========
        融合公式:
            B* = B_LCA + α_area * B_area + α_shape * B_shape

        其中:
            B_LCA: LCA 嵌入偏置 (已有)
            B_area: 面积调制偏置 (已有)
            B_shape: 形状-尺度编码偏置 (I31-P2: 新增)

        参数
        ----
        regions : torch.Tensor
            区域边界张量，形状 [B, N, 4]
        image_size : int
            图像边长

        返回
        ----
        torch.Tensor
            仿射调制偏置，形状 [B, dim, N, N]
        """
        B, N, _ = regions.shape

        # 1. 空间偏置 (LCA) [B, dim, N, N]
        lca_bias = self._compute_lca_bias_from_regions(regions, image_size)

        if not self.enable_area_modulation and not self.enable_shape_scale:
            return lca_bias

        # 2. 面积编码 [B, N, dim]
        # image_size 是 int，需要转换为 (W, H) 元组
        image_size_tuple = (image_size, image_size) if isinstance(image_size, int) else image_size
        area_emb = self.area_encoder(regions, image_size_tuple) if self.enable_area_modulation else None

        # I31-P2: 形状-尺度编码 [B, N, dim]
        shape_emb = self.shape_scale_encoder(regions, image_size_tuple) if self.enable_shape_scale else None

        # 3. 计算面积相似性矩阵 [B, N, N]
        # p_s[i,j] = area_emb[i] · area_emb[j]
        if area_emb is not None:
            area_sim = torch.bmm(area_emb, area_emb.transpose(-2, -1))
        else:
            area_sim = None

        # 4. 计算形状相似性矩阵 [B, N, N]
        # p_r[i,j] = shape_emb[i] · shape_emb[j]
        if shape_emb is not None:
            shape_sim = torch.bmm(shape_emb, shape_emb.transpose(-2, -1))
        else:
            shape_sim = None

        # 5. 傅里叶特征编码面积相似性
        if area_sim is not None:
            # I145: 向量化傅里叶特征 - 预计算频率张量，使用 tensor broadcasting
            # 原始实现 (Python循环):
            # for k in range(self.area_encoder.fourier_levels):
            #     freq = 2 ** k
            #     fourier_features.append(torch.sin(freq * math.pi * area_sim))
            #     fourier_features.append(torch.cos(freq * math.pi * area_sim))
            fourier_levels = self.area_encoder.fourier_levels
            # 预计算频率: [L]
            freqs = torch.tensor([2 ** k for k in range(fourier_levels)], device=area_sim.device, dtype=area_sim.dtype)
            freqs = freqs * math.pi  # [L]
            # 计算所有频率的 sin 和 cos: [2L, L] -> [2L] 通过广播
            # area_sim: [B, N, N], freqs: [L]
            # sin_features: [L, B, N, N], cos_features: [L, B, N, N]
            sin_features = torch.sin(freqs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * area_sim.unsqueeze(0))
            cos_features = torch.cos(freqs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * area_sim.unsqueeze(0))
            # 交错拼接: [B, N, N, 2L]
            fourier_features = torch.cat([sin_features, cos_features], dim=0).permute(1, 2, 3, 0)
            # I145: 添加溢出防护 - clamp 到合理范围
            fourier_features = fourier_features.clamp(-1e6, 1e6)
            gamma_features = fourier_features  # [B, N, N, 2L]

            # 仿射调制
            # 展平 [B, N, N, 2L] -> [B*N*N, 2L] 以便通过 Linear 层
            B, N, N, fourier_dim = gamma_features.shape
            gamma_flat = gamma_features.view(-1, fourier_dim)

            # 通过网络
            gamma = self.scale_net(gamma_flat)  # [B*N*N, dim]
            beta = self.bias_net(gamma_flat)    # [B*N*N, dim]

            # 恢复形状 [B, N, N, dim]
            gamma = gamma.view(B, N, N, self.dim)
            beta = beta.view(B, N, N, self.dim)

            # 调整维度以匹配 lca_bias: [B, dim, N, N]
            gamma = gamma.permute(0, 3, 1, 2)
            beta = beta.permute(0, 3, 1, 2)

            # 仿射变换
            modulated_area = gamma * lca_bias + beta

            # 残差连接
            combined_bias = lca_bias + self.residual_alpha * (modulated_area - lca_bias)
        else:
            combined_bias = lca_bias

        # I31-P2: 形状-尺度调制 (FIX: 向量化计算，避免 Python 循环)
        if shape_sim is not None:
            # 预计算固定 4 个频率: [4]
            shape_freqs = torch.tensor([2 ** k for k in range(4)], device=shape_sim.device, dtype=shape_sim.dtype)
            shape_freqs = shape_freqs * math.pi  # [4]
            # 向量化计算 sin/cos: [4, B, N, N]
            shape_sin = torch.sin(shape_freqs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * shape_sim.unsqueeze(0))
            shape_cos = torch.cos(shape_freqs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * shape_sim.unsqueeze(0))
            # 交错拼接: [B, N, N, 8]
            shape_gamma_features = torch.cat([shape_sin, shape_cos], dim=0).permute(1, 2, 3, 0)
            # 添加溢出防护
            shape_gamma_features = shape_gamma_features.clamp(-1e6, 1e6)

            # 展平并通过偏置网络
            B, N, N, shape_fourier_dim = shape_gamma_features.shape
            shape_flat = shape_gamma_features.view(-1, shape_fourier_dim)
            shape_beta = self.shape_bias_net(shape_flat)  # [B*N*N, dim]

            # 恢复形状 [B, dim, N, N]
            shape_beta = shape_beta.view(B, N, N, self.dim)
            shape_beta = shape_beta.permute(0, 3, 1, 2)

            # 形状调制: 直接加到偏置上
            # B_shape = shape_beta (与 LCA 偏置相加)
            combined_bias = combined_bias + self.shape_scale_alpha * shape_beta

        return combined_bias

