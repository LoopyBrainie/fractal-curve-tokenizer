from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .attention import HilbertAwareMultiScaleAttention
from .feedforward import AdaptiveFractalFeedForward


class EnhancedFractalTransformerBlock(nn.Module):
    """Hierarchically aware transformer block extracted for reuse."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0, max_level: int = 50):
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
        self.level_aware_norm1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(max_level + 1)])
        self.level_aware_norm2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(max_level + 1)])
        self.default_norm1 = nn.LayerNorm(dim)
        self.default_norm2 = nn.LayerNorm(dim)

    def _apply_level_aware_norm(
        self,
        x: torch.Tensor,
        levels_info: Optional[torch.Tensor],
        norm_layers: nn.ModuleList,
        default_norm: nn.LayerNorm,
    ) -> torch.Tensor:
        if levels_info is None or levels_info.numel() == 0:
            return default_norm(x)

        batch_size, seq_len, _ = x.shape
        output = torch.zeros_like(x)
        depths = levels_info[:, 0].clamp(0, self.max_level)

        for i in range(seq_len):
            level_idx = int(depths[i].item())
            token = x[:, i : i + 1, :]
            normed_token = norm_layers[level_idx](token)
            output[:, i : i + 1, :] = normed_token

        return output

    def forward(
        self,
        x: torch.Tensor,
        levels_info: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        norm1_x = self._apply_level_aware_norm(x, levels_info, self.level_aware_norm1, self.default_norm1)
        attn_out = self.attention(norm1_x, levels_info, attention_mask)
        x = x + attn_out * self.residual_weights[0]

        norm2_x = self._apply_level_aware_norm(x, levels_info, self.level_aware_norm2, self.default_norm2)
        ff_out = self.ff(norm2_x, levels_info)
        x = x + ff_out * self.residual_weights[1]

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
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.max_level = max_level

        self.layers = nn.ModuleList(
            [
                EnhancedFractalTransformerBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    max_level=max_level,
                )
                for _ in range(depth)
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
