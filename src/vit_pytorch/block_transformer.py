# -*- coding: utf-8 -*-
"""
分形感知 Transformer 模块

数学形式化
============

Transformer Block:
    x' = x + DropPath(Attention(LN(x), L))
    x'' = x' + DropPath(FFN(LN(x'), L))

其中:
- Attention: HilbertAwareMultiScaleAttention
- FFN: AdaptiveFractalFeedForward  
- L: levels_info 层级信息
- DropPath: 随机深度正则化

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

from .attn_hilbert_bias import HilbertAwareMultiScaleAttention
from .ffn_swiglu import AdaptiveFractalFeedForward, FFNType
from .utils import extract_depths


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
    """Hierarchically aware transformer block extracted for reuse.
    
    This block combines Hilbert-aware attention with adaptive feed-forward,
    using level-dependent normalization for depth-aware processing.
    
    P11-2 修复: 参数 max_level 现在应传入与 tokenizer.max_depth 一致的值，
    而非硬编码的 50。这确保 Embedding 表大小与实际使用的深度范围匹配，
    减少约 90% 的参数浪费。
    
    P11-8 简化: 移除 hilbert_bias_mode 和 low_rank_r 参数，仅保留 LCA 模式。
    
    Args:
        dim: Input/output dimension.
        heads: Number of attention heads.
        dim_head: Dimension per head.
        mlp_dim: Feed-forward hidden dimension.
        dropout: Dropout rate.
        max_level: Maximum hierarchical level (P11-2: should match tokenizer.max_depth).
        drop_path: DropPath rate for stochastic depth.
        ffn_type: FFN variant ('gelu', 'swiglu', 'swiglu_level').
        lca_temperature: (P6-2) LCA bias temperature, default 1.5.
        learnable_temperature: (P6-2) Whether temperature is learnable.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        max_level: int = 8,  # P11-2: 默认改为 8，应由上层传入实际 max_depth
        drop_path: float = 0.0,
        ffn_type: FFNType = 'swiglu_level',
        lca_temperature: Optional[float] = 1.5,
        learnable_temperature: bool = True,
        use_affine_modulation: bool = True,  # A17: 启用 ShapeScaleEncoder
        fourier_levels: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.max_level = max_level

        self.attention = HilbertAwareMultiScaleAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            max_level=max_level,
            lca_temperature=lca_temperature,
            learnable_temperature=learnable_temperature,
            use_affine_modulation=use_affine_modulation,
            fourier_levels=fourier_levels,
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
        # I34-10 修复: 简化为单门控设计
        # - 降低参数量: 2×D → 1×D (减少50%)
        # - 零初始化确保训练初期残差路径畅通
        # - tanh激活支持双向调制 [-1, 1]
        self._residual_gate = nn.Embedding(max_level + 1, 1)
        nn.init.zeros_(self._residual_gate.weight)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # REFACTORED: Replaced ModuleList of LayerNorms with Embeddings for Gamma/Beta
        # This reduces parameters from 50*2*dim to 2*dim (plus embedding table)
        self.norm1_gamma = nn.Embedding(max_level + 1, dim)
        self.norm1_beta = nn.Embedding(max_level + 1, dim)
        self.norm2_gamma = nn.Embedding(max_level + 1, dim)
        self.norm2_beta = nn.Embedding(max_level + 1, dim)
        
        # Initialize to identity (gamma=1, beta=0)
        nn.init.ones_(self.norm1_gamma.weight)
        nn.init.zeros_(self.norm1_beta.weight)
        nn.init.ones_(self.norm2_gamma.weight)
        nn.init.zeros_(self.norm2_beta.weight)
        
        self.default_norm1 = nn.LayerNorm(dim)
        self.default_norm2 = nn.LayerNorm(dim)

    def _apply_level_aware_norm(
        self,
        x: torch.Tensor,
        levels_info: Optional[torch.Tensor],
        gamma_emb: nn.Embedding,
        beta_emb: nn.Embedding,
        default_norm: nn.LayerNorm,
    ) -> torch.Tensor:
        """应用层级感知的 LayerNorm。
        
        根据每个 token 的层级深度选择对应的 gamma 和 beta 参数。
        
        P11-16 改进: 当 levels_info 为 None 时，生成深度 0 的默认 levels_info，
        确保始终使用 level-aware norm，避免训练/推理行为不一致。
        
        Args:
            x: 输入张量，形状为 [B, S, D]。
            levels_info: 层级信息，可为 None（将使用深度 0 作为默认）。
            gamma_emb: Gamma 参数的嵌入表。
            beta_emb: Beta 参数的嵌入表。
            default_norm: 默认的 LayerNorm（现已弃用，保留用于向后兼容）。
            
        Returns:
            归一化后的张量，形状为 [B, S, D]。
        """
        # 验证输入维度
        if x.dim() != 3:
            raise ValueError(f"Expected x to be 3D [B, S, D], got {x.dim()}D with shape {x.shape}")

        batch_size, seq_len, dim = x.shape
        
        # P11-16: 当 levels_info 为 None 时，生成深度 0 的默认值
        # 这确保始终使用 level-aware norm，避免两种 norm 路径的行为差异
        if levels_info is None or levels_info.numel() == 0:
            # 创建全零 depths，表示所有 token 深度为 0
            depths = torch.zeros(batch_size, seq_len, dtype=torch.long, device=x.device)
            gamma = gamma_emb(depths)  # (B, S, dim)
            beta = beta_emb(depths)    # (B, S, dim)
        elif levels_info.dim() == 2:
            # Old behavior: (Seq, Info) -> broadcast to batch
            depths = extract_depths(levels_info, self.max_level) # (seq_len,)
            gamma = gamma_emb(depths).unsqueeze(0) # (1, seq_len, dim)
            beta = beta_emb(depths).unsqueeze(0) # (1, seq_len, dim)
        else:
            # New behavior: (Batch, Seq, Info)
            depths = extract_depths(levels_info, self.max_level) # (B, S)
            gamma = gamma_emb(depths) # (B, S, dim)
            beta = beta_emb(depths) # (B, S, dim)
        
        # Manual LayerNorm: (x - mean) / std * gamma + beta
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        
        return x_norm * gamma + beta

    def forward(
        self,
        x: torch.Tensor,
        levels_info: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        regions: Optional[torch.Tensor] = None,
        image_size: Optional[int] = None,
    ) -> torch.Tensor:
        """前向传播。
        
        P11-3 改进: 新增 regions 和 image_size 参数，用于直接从区域边界
        计算正确的四叉树 LCA 偏置，绕过 levels_info 中全为 0 的路径问题。
        
        Args:
            x: 输入张量，形状为 [B, S, D]。
            levels_info: 层级信息（可选），用于 depth 提取和 level bias。
            attention_mask: 注意力掩码（可选）。
            regions: (P11-3) 区域边界张量，形状为 [B, N, 4]，
                     格式 [x1, y1, x2, y2]，用于计算正确的 Hilbert LCA 偏置。
            image_size: (P11-3) 图像边长，与 regions 配合使用。
            
        Returns:
            输出张量，形状为 [B, S, D]。
        """
        # I34-10 修复: 使用单门控设计
        # gate ∈ [-1, 1] 支持双向调制 (抑制/增强)
        if levels_info is not None and levels_info.numel() > 0:
            depths = extract_depths(levels_info, self.max_level)  # (S,) or (B, S)
            gate_raw = self._residual_gate(depths)  # (..., 1)
            gate = torch.tanh(gate_raw)  # (..., 1) ∈ [-1, 1]

            # 调整形状以便广播: (B, S, 1) for element-wise multiplication
            if gate.dim() == 2:
                # (S, 1) -> (1, S, 1)
                gate = gate.view(1, -1, 1)
            # gate 现在是 (B, S, 1) 或 (1, S, 1)
        else:
            # 无 levels_info 时使用深度 0 的默认权重 (零初始化 → tanh(0) = 0)
            gate = torch.tanh(self._residual_gate.weight[0])  # scalar ∈ [-1, 1]
            gate = gate.view(1, 1, 1)  # (1, 1, 1) for broadcasting

        norm1_x = self._apply_level_aware_norm(x, levels_info, self.norm1_gamma, self.norm1_beta, self.default_norm1)
        attn_out = self.attention(
            norm1_x,
            levels_info=levels_info,
            attention_mask=attention_mask,
            regions=regions,
            image_size=image_size,
        )
        # I34-10: 使用单门控 (1 + gate) 确保残差连接始终畅通
        # gate ∈ [-1, 1] → (1 + gate) ∈ [0, 2]
        x = x + self.drop_path(attn_out * (1.0 + gate))

        norm2_x = self._apply_level_aware_norm(x, levels_info, self.norm2_gamma, self.norm2_beta, self.default_norm2)
        ff_out = self.ff(norm2_x, levels_info)
        x = x + self.drop_path(ff_out * (1.0 + gate))

        return x


class FractalTransformer(nn.Module):
    """High-level transformer stack coordinating block execution.
    
    This module stacks multiple FractalTransformerBlock layers,
    adding global context attention and level aggregation for enhanced
    hierarchical processing.
    
    Supports gradient checkpointing for memory-efficient training.
    
    P11-2 修复: 参数 max_level 现在应传入与 tokenizer.max_depth 一致的值，
    而非硬编码的 50。这确保所有子模块的 Embedding 表大小与实际使用的深度范围匹配。
    
    P11-8 简化: 移除 hilbert_bias_mode 和 low_rank_r 参数，仅保留 LCA 模式。
    
    Args:
        dim: Input/output dimension.
        depth: Number of transformer blocks.
        heads: Number of attention heads.
        dim_head: Dimension per head.
        mlp_dim: Feed-forward hidden dimension.
        dropout: Dropout rate.
        max_level: Maximum hierarchical level (P11-2: should match tokenizer.max_depth).
        drop_path_rate: Maximum DropPath rate (linearly increased).
        ffn_type: FFN variant ('gelu', 'swiglu', 'swiglu_level').
        use_checkpoint: Whether to use gradient checkpointing (saves memory).
        lca_temperature: (P6-2) LCA bias temperature, default 1.5.
        learnable_temperature: (P6-2) Whether temperature is learnable.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        max_level: int = 8,  # P11-2: 默认改为 8，应由上层传入实际 max_depth
        drop_path_rate: float = 0.1,
        ffn_type: FFNType = 'swiglu_level',
        use_checkpoint: bool = False,
        lca_temperature: Optional[float] = 1.5,
        learnable_temperature: bool = True,
        use_affine_modulation: bool = True,  # A17: 启用 ShapeScaleEncoder
        fourier_levels: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.max_level = max_level
        self.ffn_type = ffn_type
        self.use_checkpoint = use_checkpoint

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.layers = nn.ModuleList(
            [
                FractalTransformerBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    max_level=max_level,
                    drop_path=dpr[i],
                    ffn_type=ffn_type,
                    lca_temperature=lca_temperature,
                    learnable_temperature=learnable_temperature,
                    use_affine_modulation=use_affine_modulation,
                    fourier_levels=fourier_levels,
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
        levels_info: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        regions: Optional[torch.Tensor] = None,
        image_size: Optional[int] = None,
    ) -> torch.Tensor:
        """前向传播。
        
        P11-3 改进: 新增 regions 和 image_size 参数，用于直接从区域边界
        计算正确的四叉树 LCA 偏置，绕过 levels_info 中全为 0 的路径问题。
        
        Args:
            x: 输入张量，形状为 [B, S, D]。
            levels_info: 层级信息（可选），用于 depth 提取和 level bias。
            attention_mask: 注意力掩码（可选）。
            regions: (P11-3) 区域边界张量，形状为 [B, N, 4]，
                     格式 [x1, y1, x2, y2]，用于计算正确的 Hilbert LCA 偏置。
            image_size: (P11-3) 图像边长，与 regions 配合使用。
            
        Returns:
            输出张量，形状为 [B, S, D]。
        """
        batch_size, seq_len, dim = x.shape

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                # Gradient checkpointing: 重新计算激活值以节省显存
                # Note: checkpoint 不支持关键字参数，需要使用位置参数
                # P11-3: 传递 regions 和 image_size
                x = checkpoint(
                    layer, x, levels_info, attention_mask, regions, image_size, 
                    use_reentrant=False
                )
            else:
                x = layer(
                    x, 
                    levels_info=levels_info, 
                    attention_mask=attention_mask,
                    regions=regions,
                    image_size=image_size,
                )

        # ARCH-R1: 删除了冗余的 global_context_attn 调用
        # HilbertAwareMultiScaleAttention 已经充分保留全局信息流

        # ARCH-R2 方案 B: 层级感知的特征聚合
        if levels_info is not None and levels_info.numel() > 0:
            depths = extract_depths(levels_info, self.max_level)  # (S,) or (B, S)
            scale = torch.sigmoid(self._level_aggregator_scale(depths))  # (..., D)
            
            # 调整形状以匹配 x: [B, S, D]
            if scale.dim() == 2:
                # (S, D) -> (1, S, D) for broadcasting
                scale = scale.unsqueeze(0)
            
            refined = self._level_aggregator_bottleneck(x)  # (B, S, D)
            aggregated = refined * scale  # 层级感知的缩放
            # I32-11: 使用 softplus 约束 scale ∈ (0, +∞)
            x = x + F.softplus(self._aggregator_scale) * aggregated

        x = self.final_norm(x)
        return x
