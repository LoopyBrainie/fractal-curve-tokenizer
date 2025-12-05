from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .attention import HilbertAwareMultiScaleAttention
from .feedforward import AdaptiveFractalFeedForward


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class EnhancedFractalTransformerBlock(nn.Module):
    """Hierarchically aware transformer block extracted for reuse."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0, max_level: int = 50, drop_path: float = 0.0):
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

        self.ff = AdaptiveFractalFeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout, max_level=max_level)

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
        if levels_info is None or levels_info.numel() == 0:
            return default_norm(x)

        # Vectorized implementation
        batch_size, seq_len, dim = x.shape
        
        # Handle both (Seq, Info) and (Batch, Seq, Info) shapes for levels_info
        if levels_info.dim() == 2:
            # Old behavior: (Seq, Info) -> broadcast to batch
            depths = levels_info[:, 0].clamp(0, self.max_level).long() # (seq_len,)
            gamma = gamma_emb(depths).unsqueeze(0) # (1, seq_len, dim)
            beta = beta_emb(depths).unsqueeze(0) # (1, seq_len, dim)
        else:
            # New behavior: (Batch, Seq, Info)
            depths = levels_info[:, :, 0].clamp(0, self.max_level).long() # (B, S)
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
        norm1_x = self._apply_level_aware_norm(x, levels_info, self.norm1_gamma, self.norm1_beta, self.default_norm1)
        attn_out = self.attention(norm1_x, levels_info, attention_mask)
        x = x + self.drop_path(attn_out * self.residual_weights[0])

        norm2_x = self._apply_level_aware_norm(x, levels_info, self.norm2_gamma, self.norm2_beta, self.default_norm2)
        ff_out = self.ff(norm2_x, levels_info)
        x = x + self.drop_path(ff_out * self.residual_weights[1])

        return x


class EnhancedFractalTransformer(nn.Module):
    """High-level transformer stack coordinating block execution."""

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
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.max_level = max_level

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
        batch_size, seq_len, dim = x.shape

        layer_weights = None
        if use_dynamic_depth:
            pooled = x.transpose(1, 2)
            layer_weights = self.depth_selector(pooled)

        for i, layer in enumerate(self.layers):
            x = layer(x, levels_info, attention_mask)

            if layer_weights is not None:
                weight = layer_weights[:, i : i + 1].unsqueeze(-1)
                x = x * weight

        if seq_len > 1:
            global_context, _ = self.global_context_attn(x, x, x)
            x = x + global_context * 0.1

        if levels_info is not None and levels_info.numel() > 0:
            aggregated = self.level_aggregator(x)
            x = x + aggregated * 0.2

        x = self.final_norm(x)
        return x
