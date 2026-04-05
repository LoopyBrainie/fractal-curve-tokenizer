# -*- coding: utf-8 -*-
"""
分形感知 Transformer 模块

数学形式化
============

Transformer Block (I106-2 方案D):
    x' = x + DropPath(Attention(LN(x), L))
    x'' = x' + DropPath(FFN_d(x', L))

其中:
- Attention: HilbertAwareMultiScaleAttention (使用标准 LN + Hilbert Bias)
- FFN: AdaptiveFractalFeedForward (使用层级感知 LN: LN_d)
- L: levels_info 层级信息
- DropPath: 随机深度正则化

I106-2 归一化设计 (方案D):
    - Attention: LN(x) 标准 LayerNorm
      理由: Hilbert Bias 已处理不同深度 token 的尺度校准
    - FFN: LN_d(x) 层级感知 LayerNorm
      理由: FFN 是逐元素变换，需要层级特定的归一化参数

DropPath (Stochastic Depth):
    训练时: output = x * Bernoulli(1 - drop_prob) / (1 - drop_prob)
    推理时: output = x

层级感知聚合 (ARCH-R2):
    s_ℓ = σ(Embed_level(ℓ)) — 每个层级的 D 维缩放向量
    r = W₂ · ReLU(W₁ · x) — bottleneck 特征精炼
    x' = x + scale · (r ⊙ s_ℓ) — 层级感知的残差更新

注: GLOBAL_CONTEXT_SCALE 已废弃 (ARCH-R1)，全局上下文由 HilbertAwareAttention 隐式处理

复杂度分析
----------
FractalTransformerBlock:
    时间: O(B · N² · D) + O(B · N · D · D_ff)
          ├─ Attention: O(B · H · N² · d) = O(B · N² · D)  — QK^T 矩阵乘
          └─ FFN:       O(B · N · D · D_ff)                — 前馈网络
    空间: O(B · H · N²) + O(B · N · D_ff)
          ├─ Attention matrix: O(B · H · N²)
          └─ FFN 中间张量: O(B · N · D_ff)

FractalTransformer (L 层):
    时间: O(L · B · N² · D)  — 线性堆叠
    空间: O(B · H · N²)       — 无累积（逐层释放）

其中: B=batch, N=seq_len, D=dim, H=heads, d=dim_head, L=depth

类对照表
----------
+----------------------------------+----------------------------------+
| 类                                | 数学定义                           |
+==================================+==================================+
| DropPath                         | x → x * mask / keep_prob         |
| FractalTransformerBlock          | x → Attn + FFN + GlobalCtx       |
| FractalTransformer               | 堆叠 depth 个 TransformerBlock   |
+----------------------------------+----------------------------------+
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

from vit_pytorch.layers.attention.manifold_attention import ManifoldNativeAttention
from vit_pytorch.layers.ffn.swiglu import AdaptiveFractalFeedForward, FFNType
from vit_pytorch.core.levels_info import LevelsInfo


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """前向传播，应用随机路径丢弃。
        
        Args:
            x: 输入张量。
            
        Returns:
            经过 DropPath 处理后的张量。
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class FractalTransformerBlock(nn.Module):
    """Manifold-Native Transformer Block.

    使用 ManifoldNativeAttention 实现 Hilbert 带宽稀疏注意力。
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        max_level: int = 8,
        drop_path: float = 0.0,
        ffn_type: FFNType = 'swiglu_level',
        manifold_beta: float = 4.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_level = max_level
        self.manifold_beta = manifold_beta

        # Manifold-Native 注意力
        self.attention = ManifoldNativeAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            max_level=max_level,
            beta=manifold_beta,
            dropout=dropout,
            use_banded=True,
            use_fractal_residual=True,
        )

        self.ff = AdaptiveFractalFeedForward(
            dim=dim,
            hidden_dim=mlp_dim,
            dropout=dropout,
            max_level=max_level,
            ffn_type=ffn_type,
        )

        # STAB-5 方案 B+: 层级感知的残差门控 (P1-2 修复, P11-13 重命名)
        # 功能: 控制 Attention/FFN 输出对残差连接的贡献程度
        # 实现: gate(d) = sigmoid(Embedding(d)) * 2 ∈ [0, 2]
        #       x' = x + gate_1(d) * Attn(x)
        #       x'' = x' + gate_2(d) * FFN(x')
        #
        # I24-6 改进: 基于训练结果的智能初始化
        # 训练后学习到的门控模式:
        #   - depth=0 (全图): w ≈ 0.7 → 抑制全局信息
        #   - depth=1: w ≈ 1.0 → 保持原样
        #   - depth=2: w ≈ 0.85 → 轻微抑制
        #   - depth=3 (细粒度): w ≈ 0.65 → 抑制细节
        #
        # 使用 inverse_sigmoid 反算: sigmoid(x) * 2 = target → x = logit(target/2)
        # target=0.7 → x ≈ -0.36, target=1.0 → x = 0, target=0.65 → x ≈ -0.54
        #
        # Task 3 重构: 替换为 Sigmoid 门控
        # - 零初始化确保训练初期残差路径畅通 (sigmoid(0) = 0.5)
        # - sigmoid 激活值域 [0, 1]，梯度始终为正
        # - 添加梯度比例监控确保邻域路径梯度 >= 40%
        self._residual_gate = nn.Embedding(max_level + 1, 1)
        # I-NAN: 改为小值初始化，确保初始梯度流稳定
        nn.init.normal_(self._residual_gate.weight, mean=0, std=0.01)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # I106-2: 简化归一化层 (方案D)
        # - Attention: 使用标准 LayerNorm (Hilbert Bias 已处理尺度校准)
        # - FFN: 移除 Block 级别的归一化 (FFN 内部有自己的层级感知归一化)
        self.norm1 = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        levels_info: Optional[LevelsInfo] = None,
        attention_mask: Optional[torch.Tensor] = None,
        regions: Optional[torch.Tensor] = None,
        image_size: Optional[int] = None,
        geometry_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播。

        I98-4: levels_info 参数类型从 torch.Tensor 改为 LevelsInfo
        v5.0: 新增 geometry_emb 参数，用于 Attention 注入

        Args:
            x: 输入张量，形状为 [B, S, D]。
            levels_info: LevelsInfo 实例（可选）。
            attention_mask: 注意力掩码（可选）。
            regions: 区域边界张量，形状为 [B, N, 4]，格式 [x1, y1, x2, y2]。
            image_size: 图像边长，与 regions 配合使用。
            geometry_emb: 几何嵌入 (可选)，形状为 [B, S, D]，用于 Attention 注入

        Returns:
            输出张量，形状为 [B, S, D]。
        """
        # I98-4: 兼容 raw tensor 和 LevelsInfo 对象
        if isinstance(levels_info, torch.Tensor):
            # 转换为 LevelsInfo，确保数据类型为 Long
            if levels_info.dtype != torch.long:
                levels_info = levels_info.long()

            # 从数据形状推断 max_level: info_dim = max_level + 1
            info_dim = levels_info.shape[-1]
            inferred_max_level = info_dim - 1
            levels_info = LevelsInfo(data=levels_info, max_level=inferred_max_level)

        # Task 3 重构: 使用 Sigmoid 门控
        # gate ∈ [0, 1] 确保梯度方向始终正确
        # D3-AUDIT FIX: 物化 tensor 条件为 Python bool，消除 Graph Break
        has_levels = levels_info is not None and levels_info.data.numel() > 0
        if has_levels:
            depths = levels_info.depths  # (B, S)
            # I98-4: clamp depths to [0, max_level] to handle padding sentinel (-1)
            depths_clamped = depths.clamp(min=0, max=self.max_level)
            gate_raw = self._residual_gate(depths_clamped)  # (..., 1)
            gate = torch.sigmoid(gate_raw)  # (..., 1) ∈ [0, 1]

            # 调整形状以便广播: (B, S, 1) for element-wise multiplication
            # D3-AUDIT FIX: 物化 dim() 比较为 Python bool，消除 Graph Break
            is_2d = gate.dim() == 2
            if is_2d:
                # (S, 1) -> (1, S, 1)
                gate = gate.view(1, -1, 1)
            # gate 现在是 (B, S, 1) 或 (1, S, 1)
        else:
            # 无 levels_info 时使用深度 0 的默认权重 (零初始化 → sigmoid(0) = 0.5)
            gate = torch.sigmoid(self._residual_gate.weight[0])  # scalar ∈ [0, 1]
            gate = gate.view(1, 1, 1)  # (1, 1, 1) for broadcasting

        # I106-2: 使用标准 LayerNorm (替代层级感知归一化)
        # Hilbert Bias 已处理不同深度 token 的尺度校准
        norm1_x = self.norm1(x)
        attn_out = self.attention(
            norm1_x,
            levels_info=levels_info,
            attention_mask=attention_mask,
            regions=regions,
            image_size=image_size,
            geometry_emb=geometry_emb,
        )
        # Task 3 重构: 使用 Sigmoid 门控残差
        # gate ∈ [0, 1] → 直接使用 gate 确保梯度流通
        x = x + self.drop_path(attn_out * gate)

        # I106-2: FFN 跳过 Block 级别的归一化
        # FFN 内部有自己的层级感知归一化 (ffn_swiglu.py)
        ff_out = self.ff(x, levels_info)
        x = x + self.drop_path(ff_out * gate)

        return x


class FractalTransformer(nn.Module):
    """Manifold-Native Transformer Stack.

    使用 ManifoldNativeAttention 实现 Hilbert 带宽稀疏注意力。
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        max_level: int = 8,
        drop_path_rate: float = 0.1,
        ffn_type: FFNType = 'swiglu_level',
        use_checkpoint: bool = False,
        manifold_beta: float = 4.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = depth
        self.max_level = max_level
        self.ffn_type = ffn_type
        self.use_checkpoint = use_checkpoint
        self.manifold_beta = manifold_beta

        # P-OPT: Stochastic depth decay rule
        # D1-AUDIT FIX: 使用 .tolist() 避免 .item() 同步（__init__ 中调用，非 forward 热路径但仍需修复）
        self._drop_path_rates = torch.linspace(0, drop_path_rate, depth).tolist()

        self.layers = nn.ModuleList(
            [
                FractalTransformerBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    max_level=max_level,
                    drop_path=self._drop_path_rates[i],
                    ffn_type=ffn_type,
                    manifold_beta=manifold_beta,
                )
                for i in range(depth)
            ]
        )

        # ARCH-R1: 删除了冗余的 global_context_attn
        # 原因: HilbertAwareMultiScaleAttention 已经保留了 78.9% 的全局注意力权重
        # Hilbert Bias 只是软约束，不需要额外的全局注意力纠正

        # ARCH-R2 方案 B: 真正的层级感知聚合器 (I32-11 优化初始化)
        # 数学形式化:
        #   s_ℓ = σ(Embed_level(ℓ)) ∈ (0, 1)^D  — 每个层级的 D 维缩放向量
        #   r = W₂ · ReLU(W₁ · x)               — bottleneck 特征精炼
        #   x' = x + α · (r ⊙ s_ℓ)              — 层级感知的残差更新
        #
        # I32-11: 使用 Xavier/He 初始化保证方差一致性
        #         可学习对数尺度初始化为 softplus(γ) = 1.0
        #
        # 物理意义:
        #   - 浅层级 (level=0,1,2): 大区域，学习保留全局语义的特征维度
        #   - 深层级 (level=5,6,7): 小区域，学习增强局部细节的特征维度
        self._level_aggregator_scale = nn.Embedding(max_level + 1, dim)
        nn.init.xavier_uniform_(self._level_aggregator_scale.weight)  # I32-11: Xavier 初始化

        self._level_aggregator_bottleneck = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
        )
        # I32-11: 可学习对数尺度初始化
        # 目标: softplus(γ) = 1.0 (恒等变换初始)
        # 解: γ = ln(e^1 - 1) ≈ 1.3133
        self._aggregator_scale = nn.Parameter(torch.tensor(1.3133))  # softplus(1.3133) ≈ 1.0
        
        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        levels_info: Optional[LevelsInfo] = None,
        attention_mask: Optional[torch.Tensor] = None,
        regions: Optional[torch.Tensor] = None,
        image_size: Optional[int] = None,
        geometry_emb: Optional[torch.Tensor] = None,
        return_extra_info: bool = False,
    ) -> tuple[torch.Tensor, dict] | torch.Tensor:
        """前向传播。

        I98-4: levels_info 参数类型从 torch.Tensor 改为 LevelsInfo
        v5.0: 新增 geometry_emb 参数，用于 Attention 注入

        Args:
            x: 输入张量，形状为 [B, S, D]。
            levels_info: LevelsInfo 实例（可选）。
            attention_mask: 注意力掩码（可选）。
            regions: 区域边界张量，形状为 [B, N, 4]，格式 [x1, y1, x2, y2]。
            image_size: 图像边长，与 regions 配合使用。
            geometry_emb: 几何嵌入 (可选)，形状为 [B, S, D]
            return_extra_info: (I97-11) 是否返回额外信息。

        Returns:
            如果 return_extra_info=True: (output, extra_info)
            否则: output
        """
        # I98-4: 兼容 raw tensor 和 LevelsInfo 对象
        if isinstance(levels_info, torch.Tensor):
            # 转换为 LevelsInfo，确保数据类型为 Long
            if levels_info.dtype != torch.long:
                levels_info = levels_info.long()

            # 从数据形状推断 max_level: info_dim = max_level + 1
            info_dim = levels_info.shape[-1]
            inferred_max_level = info_dim - 1
            levels_info = LevelsInfo(data=levels_info, max_level=inferred_max_level)

        batch_size, seq_len, dim = x.shape

        # I100-6: 使用完整深度 num_layers (BUG FIX)
        # 原始设计 effective_depth = num_layers // 2 导致一半的 transformer 层
        # 从未被执行，这些层的参数永远不会收到梯度。
        # 这是一个严重的 bug，修复后所有创建的层都会被使用。
        effective_depth = self.num_layers
        extra_info = {'effective_depth': effective_depth}

        # 执行 transformer 层
        for i, layer in enumerate(self.layers):
            if i >= effective_depth:
                break
            if self.use_checkpoint and self.training:
                # Gradient checkpointing: 重新计算激活值以节省显存
                # Note: checkpoint 不支持关键字参数，需要使用位置参数
                # P11-3: 传递 regions 和 image_size
                # v5.0: 添加 geometry_emb 参数
                x = checkpoint(
                    layer, x, levels_info, attention_mask, regions, image_size, geometry_emb,
                    use_reentrant=False
                )
            else:
                x = layer(
                    x,
                    levels_info=levels_info,
                    attention_mask=attention_mask,
                    regions=regions,
                    image_size=image_size,
                    geometry_emb=geometry_emb,
                )

        # ARCH-R1: 删除了冗余的 global_context_attn 调用
        # HilbertAwareMultiScaleAttention 已经充分保留全局信息流

        # ARCH-R2 方案 B: 层级感知的特征聚合
        # D3-AUDIT FIX: 物化 tensor 条件为 Python bool，消除 Graph Break
        has_levels = levels_info is not None and levels_info.data.numel() > 0
        if has_levels:
            depths = levels_info.depths  # (B, S)
            # I98-4: clamp depths to [0, max_level] to handle padding sentinel (-1)
            depths_clamped = depths.clamp(min=0, max=self.max_level)
            scale = torch.sigmoid(self._level_aggregator_scale(depths_clamped))  # (..., D)
            
            # 调整形状以匹配 x: [B, S, D]
            # D3-AUDIT FIX: 物化 dim() 比较为 Python bool，消除 Graph Break
            is_2d_scale = scale.dim() == 2
            if is_2d_scale:
                # (S, D) -> (1, S, D) for broadcasting
                scale = scale.unsqueeze(0)
            
            refined = self._level_aggregator_bottleneck(x)  # (B, S, D)
            aggregated = refined * scale  # 层级感知的缩放
            # I32-11: 使用 softplus 约束 scale ∈ (0, +∞)
            x = x + F.softplus(self._aggregator_scale) * aggregated

        x = self.final_norm(x)

        # I97-11: 返回额外信息
        if return_extra_info:
            return x, extra_info
        return x
