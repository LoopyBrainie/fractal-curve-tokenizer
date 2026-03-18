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
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

from vit_pytorch.layers.attention.hilbert_bias import HilbertAwareMultiScaleAttention
from vit_pytorch.layers.attention.manifold_attention import ManifoldNativeAttention
from vit_pytorch.layers.ffn.swiglu import AdaptiveFractalFeedForward, FFNType
from vit_pytorch.core.config import AttentionEncoderConfig  # I98-3
from vit_pytorch.core.levels_info import LevelsInfo  # I98-4


class DropPath(nn.Module):
    r"""
    Drop paths (Stochastic Depth) per sample when applied in main path of residual blocks.

    During training, randomly drops entire residual branches to improve generalization.
    During inference, returns identity (no dropping) for deterministic behavior.

    See `Deep Networks with Stochastic Depth <https://arxiv.org/abs/1603.09382>`_ for details.

    .. note::
        This is also known as "Stochastic Depth" and is a regularization technique
        that helps train deeper networks by reducing vanishing gradient problems.
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        r"""
        Args:
            drop_prob (float): Drop probability for stochastic depth. Default: ``0.0``
            scale_by_keep (bool): Whether to scale the remaining paths by ``1 / (1 - drop_prob)``.
                Default: ``True``
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Applies stochastic depth to the input tensor.

        Args:
            x (Tensor): Input tensor of any shape

        Returns:
            Tensor: Output tensor of same shape as input, with paths randomly dropped during training

        Examples::

            >>> drop_path = DropPath(drop_prob=0.2)
            >>> x = torch.randn(2, 4, 64)  # [batch, seq, dim]
            >>> output = drop_path(x)
            >>> output.shape
            torch.Size([2, 4, 64])
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
    r"""
    Hierarchically aware transformer block with Hilbert-aware attention.

    This block combines Hilbert-aware attention with adaptive feed-forward networks,
    using level-dependent normalization for depth-aware processing.

    Architecture (I106-2 Scheme D):
        x' = x + DropPath(Attention(LN(x), L))
        x'' = x' + DropPath(FFN_d(x', L))

    Where:
        - Attention: :class:`~vit_pytorch.layers.attention.hilbert_bias.HilbertAwareMultiScaleAttention`
        - FFN: :class:`~vit_pytorch.layers.ffn.swiglu.AdaptiveFractalFeedForward`
        - L: levels_info for hierarchical information

    .. note::
        Parameter ``max_level`` should match ``tokenizer.max_level`` to ensure
        embedding table size matches the actual depth range used.

    Args:
        dim (int): Input/output dimension
        heads (int): Number of attention heads
        dim_head (int): Dimension per head
        mlp_dim (int): Feed-forward hidden dimension
        dropout (float): Dropout rate. Default: ``0.0``
        max_level (int): Maximum hierarchical level. Default: ``8``
        drop_path (float): DropPath rate for stochastic depth. Default: ``0.0``
        ffn_type (FFNType): FFN variant. Options: ``'gelu'``, ``'swiglu'``, ``'swiglu_level'``.
            Default: ``'swiglu_level'``
        use_affine_modulation (bool): Whether to use affine modulation bias. Default: ``True``
        fourier_levels (int): Number of Fourier frequency levels. Default: ``4``
        encoder_config (AttentionEncoderConfig, optional): Protocol-driven encoder configuration.
            Default: ``None``
        use_fp16 (bool): Use FP16 for LCA embeddings. Default: ``False``
        use_manifold_native (bool): Use Manifold-Native attention. Default: ``False``
        manifold_beta (float): Hilbert bandwidth coefficient. Default: ``4.0``
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        max_level: int = 8,  # P11-2: 默认改为 8，应由上层传入实际 max_level
        drop_path: float = 0.0,
        ffn_type: FFNType = 'swiglu_level',
        use_affine_modulation: bool = True,  # A17: 启用 ShapeScaleEncoder
        fourier_levels: int = 4,
        encoder_config: Optional["AttentionEncoderConfig"] = None,  # I98-3
        use_fp16: bool = False,  # I104-3: FP16 存储 LCA embedding
        use_manifold_native: bool = False,  # 新: 使用 Manifold-Native 注意力
        manifold_beta: float = 4.0,  # 新: Hilbert 带宽系数
    ):
        r"""
        See class docstring for parameters.
        """
        super().__init__()
        self.dim = dim
        self.max_level = max_level
        self.use_manifold_native = use_manifold_native
        self.manifold_beta = manifold_beta  # I-MANIFOLD: Hilbert 带宽系数

        # 选择注意力模块
        if use_manifold_native:
            # 新: Manifold-Native 注意力 (整合所有最佳实现)
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
        else:
            # 原有: HilbertAwareMultiScaleAttention
            self.attention = HilbertAwareMultiScaleAttention(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
                max_level=max_level,
                use_affine_modulation=use_affine_modulation,
                fourier_levels=fourier_levels,
                encoder_config=encoder_config,
                use_fp16=use_fp16,
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Forward pass of the fractal transformer block.

        Args:
            x (Tensor): Input tensor of shape :math:`(B, S, D)`
            levels_info (LevelsInfo, optional): Hierarchical level information. Default: ``None``
            attention_mask (Tensor, optional): Attention mask. Default: ``None``
            regions (Tensor, optional): Region boundary tensor of shape :math:`(B, N, 4)`
                with format :math:`[x_1, y_1, x_2, y_2]`. Default: ``None``
            image_size (int, optional): Image side length, used with ``regions``. Default: ``None``
            geometry_emb (Tensor, optional): Geometry embedding of shape :math:`(B, S, D)`
                for attention injection. Default: ``None``

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Tuple containing:
                - Output tensor of shape :math:`(B, S, D)`
                - Manifold bias tensor
                - Poincaré distances tensor

        Examples::

            >>> block = FractalTransformerBlock(dim=256, heads=8, dim_head=32, mlp_dim=512)
            >>> x = torch.randn(2, 64, 256)  # [batch, seq, dim]
            >>> output, manifold, distances = block(x)
            >>> output.shape
            torch.Size([2, 64, 256])
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
        if levels_info is not None and levels_info.data.numel() > 0:
            depths = levels_info.depths  # (B, S)
            # I98-4: clamp depths to [0, max_level] to handle padding sentinel (-1)
            depths_clamped = depths.clamp(min=0, max=self.max_level)
            gate_raw = self._residual_gate(depths_clamped)  # (..., 1)
            gate = torch.sigmoid(gate_raw)  # (..., 1) ∈ [0, 1]

            # 调整形状以便广播: (B, S, 1) for element-wise multiplication
            if gate.dim() == 2:
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
        # I171: attention 现在返回 (output, manifold_bias, poincare_distances)
        attn_out, manifold_bias, poincare_distances = self.attention(
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

        # I171: 返回输出 + 流形张量
        return x, manifold_bias, poincare_distances


class FractalTransformer(nn.Module):
    r"""
    High-level transformer stack with fractal hierarchical processing.

    This module stacks multiple :class:`FractalTransformerBlock` layers,
    adding level aggregation for enhanced hierarchical processing.
    Supports gradient checkpointing for memory-efficient training.

    Architecture:
        - Stacks ``depth`` fractal transformer blocks
        - Uses stochastic depth (DropPath) for regularization
        - Applies level-aware feature aggregation (ARCH-R2)
        - Returns manifold bias and Poincaré distances for analysis

    .. note::
        Parameter ``max_level`` should match ``tokenizer.max_level`` to ensure
        all sub-modules' embedding table sizes match the actual depth range.

    Args:
        dim (int): Input/output dimension
        depth (int): Number of transformer blocks
        heads (int): Number of attention heads
        dim_head (int): Dimension per head
        mlp_dim (int): Feed-forward hidden dimension
        dropout (float): Dropout rate. Default: ``0.0``
        max_level (int): Maximum hierarchical level. Default: ``8``
        drop_path_rate (float): Maximum DropPath rate (linearly increased). Default: ``0.1``
        ffn_type (FFNType): FFN variant. Options: ``'gelu'``, ``'swiglu'``, ``'swiglu_level'``.
            Default: ``'swiglu_level'``
        use_checkpoint (bool): Use gradient checkpointing to save memory. Default: ``False``
        use_affine_modulation (bool): Whether to use affine modulation bias. Default: ``True``
        fourier_levels (int): Number of Fourier frequency levels. Default: ``4``
        encoder_config (AttentionEncoderConfig, optional): Protocol-driven encoder configuration.
            Default: ``None``
        use_fp16 (bool): Use FP16 for LCA embeddings. Default: ``False``
        use_manifold_native (bool): Use Manifold-Native attention. Default: ``True``
        manifold_beta (float): Hilbert bandwidth coefficient. Default: ``4.0``
    """

    def __init__(
        self,
        dim: int,
        depth: int,  # 保持 depth 作为参数名以保持 API 兼容
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        max_level: int = 8,  # P11-2: 默认改为 8，应由上层传入实际 max_level
        drop_path_rate: float = 0.1,
        ffn_type: FFNType = 'swiglu_level',
        use_checkpoint: bool = False,
        use_affine_modulation: bool = True,  # A17: 启用 ShapeScaleEncoder
        fourier_levels: int = 4,
        encoder_config: Optional[AttentionEncoderConfig] = None,  # I98-3
        use_fp16: bool = False,  # I104-3: FP16 存储 LCA embedding
        use_manifold_native: bool = True,  # I-MANIFOLD: Manifold-Native 注意力
        manifold_beta: float = 4.0,  # I-MANIFOLD: Hilbert 带宽系数
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = depth  # 使用 num_layers 作为属性名
        self.max_level = max_level
        self.ffn_type = ffn_type
        self.use_checkpoint = use_checkpoint
        self.use_fp16 = use_fp16  # I104-3
        self.use_manifold_native = use_manifold_native  # I-MANIFOLD: Manifold-Native 注意力
        self.manifold_beta = manifold_beta  # I-MANIFOLD: Hilbert 带宽系数

        # P-OPT: Stochastic depth decay rule
        # 使用 torch.linspace 预计算，避免 numpy 依赖和 .tolist() 转换
        self.register_buffer('_drop_path_rates', torch.linspace(0, drop_path_rate, depth))

        self.layers = nn.ModuleList(
            [
                FractalTransformerBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    max_level=max_level,
                    drop_path=self._drop_path_rates[i].item(),
                    ffn_type=ffn_type,
                    use_affine_modulation=use_affine_modulation,
                    fourier_levels=fourier_levels,
                    encoder_config=encoder_config,  # I98-3
                    use_fp16=use_fp16,  # I104-3
                    use_manifold_native=use_manifold_native,  # I-MANIFOLD
                    manifold_beta=manifold_beta,  # I-MANIFOLD
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, dict]:
        r"""
        Forward pass of the fractal transformer stack.

        Args:
            x (Tensor): Input tensor of shape :math:`(B, S, D)`
            levels_info (LevelsInfo, optional): Hierarchical level information. Default: ``None``
            attention_mask (Tensor, optional): Attention mask. Default: ``None``
            regions (Tensor, optional): Region boundary tensor of shape :math:`(B, N, 4)`
                with format :math:`[x_1, y_1, x_2, y_2]`. Default: ``None``
            image_size (int, optional): Image side length, used with ``regions``. Default: ``None``
            geometry_emb (Tensor, optional): Geometry embedding of shape :math:`(B, S, D)`.
                Default: ``None``
            return_extra_info (bool): Whether to return extra information. Default: ``False``

        Returns:
            If ``return_extra_info=True``: Tuple of (output, extra_info_dict)
            Otherwise: Tuple of (output, manifold_bias, poincare_distances)

        Examples::

            >>> transformer = FractalTransformer(dim=256, depth=12, heads=8, dim_head=32, mlp_dim=512)
            >>> x = torch.randn(2, 64, 256)  # [batch, seq, dim]
            >>> output, manifold, distances = transformer(x)
            >>> output.shape
            torch.Size([2, 64, 256])
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

        # I100-6: 使用固定有效深度 num_layers // 2
        effective_depth = self.num_layers // 2
        extra_info = {'effective_depth': effective_depth}

        # I171: 收集所有 block 的流形张量
        all_manifold_biases = []
        all_poincare_distances = []

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
                # I171: layer 现在返回 (x, manifold_bias, poincare_distances)
                x, manifold_bias, poincare_distances = layer(
                    x,
                    levels_info=levels_info,
                    attention_mask=attention_mask,
                    regions=regions,
                    image_size=image_size,
                    geometry_emb=geometry_emb,
                )
                # 收集流形张量
                if manifold_bias is not None:
                    all_manifold_biases.append(manifold_bias)
                if poincare_distances is not None:
                    all_poincare_distances.append(poincare_distances)

        # I171: 聚合流形张量（取平均）
        final_manifold_bias = None
        final_poincare_distances = None
        if all_manifold_biases:
            final_manifold_bias = torch.stack(all_manifold_biases).mean(0)
        if all_poincare_distances:
            final_poincare_distances = torch.stack(all_poincare_distances).mean(0)

        # ARCH-R1: 删除了冗余的 global_context_attn 调用
        # HilbertAwareMultiScaleAttention 已经充分保留全局信息流

        # ARCH-R2 方案 B: 层级感知的特征聚合
        if levels_info is not None and levels_info.data.numel() > 0:
            depths = levels_info.depths  # (B, S)
            # I98-4: clamp depths to [0, max_level] to handle padding sentinel (-1)
            depths_clamped = depths.clamp(min=0, max=self.max_level)
            scale = torch.sigmoid(self._level_aggregator_scale(depths_clamped))  # (..., D)
            
            # 调整形状以匹配 x: [B, S, D]
            if scale.dim() == 2:
                # (S, D) -> (1, S, D) for broadcasting
                scale = scale.unsqueeze(0)
            
            refined = self._level_aggregator_bottleneck(x)  # (B, S, D)
            aggregated = refined * scale  # 层级感知的缩放
            # I32-11: 使用 softplus 约束 scale ∈ (0, +∞)
            x = x + F.softplus(self._aggregator_scale) * aggregated

        x = self.final_norm(x)

        # I171: 返回输出 + 流形张量
        # I97-11: 返回额外信息
        if return_extra_info:
            return x, extra_info
        # 返回三元组 (output, manifold_bias, poincare_distances)
        return x, final_manifold_bias, final_poincare_distances
