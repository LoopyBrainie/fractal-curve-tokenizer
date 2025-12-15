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

全局上下文:
    global_ctx = mean(x) * GLOBAL_CONTEXT_SCALE
    x' = x + global_ctx

类对照表
----------
+----------------------------------+----------------------------------+
| 类                                | 数学定义                           |
+==================================+==================================+
| DropPath                         | x → x * mask / keep_prob         |
| EnhancedFractalTransformerBlock  | x → Attn + FFN + GlobalCtx       |
| EnhancedFractalTransformer       | 堆叠 depth 个 TransformerBlock   |
+----------------------------------+----------------------------------+
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

from .attention import HilbertAwareMultiScaleAttention
from .constants import GLOBAL_CONTEXT_SCALE
from .feedforward import AdaptiveFractalFeedForward, FFNType
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


class EnhancedFractalTransformerBlock(nn.Module):
    """Hierarchically aware transformer block extracted for reuse.
    
    This block combines Hilbert-aware attention with adaptive feed-forward,
    using level-dependent normalization for depth-aware processing.
    
    Args:
        dim: Input/output dimension.
        heads: Number of attention heads.
        dim_head: Dimension per head.
        mlp_dim: Feed-forward hidden dimension.
        dropout: Dropout rate.
        max_level: Maximum hierarchical level.
        drop_path: DropPath rate for stochastic depth.
        ffn_type: FFN variant ('gelu', 'swiglu', 'swiglu_level').
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        max_level: int = 50,
        drop_path: float = 0.0,
        ffn_type: FFNType = 'swiglu_level',
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
        )

        self.ff = AdaptiveFractalFeedForward(
            dim=dim,
            hidden_dim=mlp_dim,
            dropout=dropout,
            max_level=max_level,
            ffn_type=ffn_type,
        )

        self.residual_weights = nn.Parameter(torch.ones(2))
        
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
        
        Args:
            x: 输入张量，形状为 [B, S, D]。
            levels_info: 层级信息，可为 None。
            gamma_emb: Gamma 参数的嵌入表。
            beta_emb: Beta 参数的嵌入表。
            default_norm: 默认的 LayerNorm（当无层级信息时使用）。
            
        Returns:
            归一化后的张量，形状为 [B, S, D]。
        """
        if levels_info is None or levels_info.numel() == 0:
            return default_norm(x)

        # Vectorized implementation
        batch_size, seq_len, dim = x.shape
        
        # Handle both (Seq, Info) and (Batch, Seq, Info) shapes for levels_info
        if levels_info.dim() == 2:
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
    ) -> torch.Tensor:
        """前向传播。
        
        Args:
            x: 输入张量，形状为 [B, S, D]。
            levels_info: 层级信息（可选）。
            attention_mask: 注意力掩码（可选）。
            
        Returns:
            输出张量，形状为 [B, S, D]。
        """
        norm1_x = self._apply_level_aware_norm(x, levels_info, self.norm1_gamma, self.norm1_beta, self.default_norm1)
        attn_out = self.attention(norm1_x, levels_info, attention_mask)
        x = x + self.drop_path(attn_out * self.residual_weights[0])

        norm2_x = self._apply_level_aware_norm(x, levels_info, self.norm2_gamma, self.norm2_beta, self.default_norm2)
        ff_out = self.ff(norm2_x, levels_info)
        x = x + self.drop_path(ff_out * self.residual_weights[1])

        return x


class EnhancedFractalTransformer(nn.Module):
    """High-level transformer stack coordinating block execution.
    
    This module stacks multiple EnhancedFractalTransformerBlock layers,
    adding global context attention and level aggregation for enhanced
    hierarchical processing.
    
    Supports gradient checkpointing for memory-efficient training.
    
    Args:
        dim: Input/output dimension.
        depth: Number of transformer blocks.
        heads: Number of attention heads.
        dim_head: Dimension per head.
        mlp_dim: Feed-forward hidden dimension.
        dropout: Dropout rate.
        max_level: Maximum hierarchical level.
        drop_path_rate: Maximum DropPath rate (linearly increased).
        ffn_type: FFN variant ('gelu', 'swiglu', 'swiglu_level').
        use_checkpoint: Whether to use gradient checkpointing (saves memory).
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        max_level: int = 50,
        drop_path_rate: float = 0.1,
        ffn_type: FFNType = 'swiglu_level',
        use_checkpoint: bool = False,
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
                EnhancedFractalTransformerBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    max_level=max_level,
                    drop_path=dpr[i],
                    ffn_type=ffn_type,
                )
                for i in range(depth)
            ]
        )

        self.global_context_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.level_aggregator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
            nn.LayerNorm(dim),
        )
        self.final_norm = nn.LayerNorm(dim)
        self.depth_selector = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(dim, depth), nn.Sigmoid())

    def forward(
        self,
        x: torch.Tensor,
        levels_info: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_dynamic_depth: bool = False,
    ) -> torch.Tensor:
        """前向传播。
        
        Args:
            x: 输入张量，形状为 [B, S, D]。
            levels_info: 层级信息（可选）。
            attention_mask: 注意力掩码（可选）。
            use_dynamic_depth: 是否使用动态深度选择。
            
        Returns:
            输出张量，形状为 [B, S, D]。
        """
        batch_size, seq_len, dim = x.shape

        layer_weights = None
        if use_dynamic_depth:
            pooled = x.transpose(1, 2)
            layer_weights = self.depth_selector(pooled)

        for i, layer in enumerate(self.layers):
            if self.use_checkpoint and self.training:
                # Gradient checkpointing: 重新计算激活值以节省显存
                x = checkpoint(layer, x, levels_info, attention_mask, use_reentrant=False)
            else:
                x = layer(x, levels_info, attention_mask)

            if layer_weights is not None:
                weight = layer_weights[:, i : i + 1].unsqueeze(-1)
                x = x * weight

        if seq_len > 1:
            # 将 attention_mask (B, 1, 1, Seq) 转换为 key_padding_mask (B, Seq)
            # attention_mask: True = 保留, False = mask
            # key_padding_mask: True = mask, False = 保留 (相反的语义)
            key_padding_mask = None
            if attention_mask is not None:
                # attention_mask: (B, 1, 1, Seq) -> (B, Seq), 然后取反
                key_padding_mask = ~attention_mask.squeeze(1).squeeze(1).bool()
            
            global_context, _ = self.global_context_attn(x, x, x, key_padding_mask=key_padding_mask)
            x = x + global_context * GLOBAL_CONTEXT_SCALE

        if levels_info is not None and levels_info.numel() > 0:
            aggregated = self.level_aggregator(x)
            x = x + aggregated * 0.2

        x = self.final_norm(x)
        return x
