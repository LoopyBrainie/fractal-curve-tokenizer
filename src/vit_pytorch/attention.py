from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class HilbertAwareMultiScaleAttention(nn.Module):
    """Hilbert curve aware multi-scale attention.

    Encodes hierarchical depth and Hilbert path relationships to modulate attention
    weights. Extracted from the former monolithic fractal ViT module to improve
    composability and testing.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        max_level: int = 50,
        use_hilbert_bias: bool = True,
        use_level_scaling: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.max_level = max_level
        self.use_hilbert_bias = use_hilbert_bias
        self.use_level_scaling = use_level_scaling

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if use_hilbert_bias:
            self.hilbert_bias_network = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Linear(64, heads),
                nn.Tanh(),
            )
        else:
            self.hilbert_bias_network = None

        if use_level_scaling:
            self.level_scale_embedding = nn.Embedding(max_level + 1, heads)
            nn.init.constant_(self.level_scale_embedding.weight, 1.0)
            nn.init.normal_(self.level_scale_embedding.weight, std=0.1)
        else:
            self.level_scale_embedding = None

        self.scale_weights = nn.Parameter(torch.ones(heads))
        self.relative_pos_embedding = nn.Embedding(2 * max_level + 1, heads)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def _compute_hilbert_bias(self, levels_info: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.use_hilbert_bias or levels_info.numel() == 0:
            return None

        seq_len = levels_info.shape[0]
        device = levels_info.device

        if levels_info.shape[1] <= 1:
            return None

        paths = levels_info[:, 1:].float()
        bias_matrix = torch.zeros(self.heads, seq_len, seq_len, device=device)

        for i in range(seq_len):
            for j in range(seq_len):
                if i == j:
                    continue

                path_i = paths[i]
                path_j = paths[j]

                path_dist = torch.norm(path_i - path_j).unsqueeze(0)
                path_sim = F.cosine_similarity(path_i.unsqueeze(0), path_j.unsqueeze(0), dim=1)

                path_features = torch.cat([path_dist, path_sim])
                bias = self.hilbert_bias_network(path_features)
                bias_matrix[:, i, j] = bias

        return bias_matrix

    def _compute_level_bias(self, levels_info: torch.Tensor) -> Optional[torch.Tensor]:
        if levels_info.numel() == 0:
            return None

        depths = levels_info[:, 0]
        level_diff = depths.unsqueeze(0) - depths.unsqueeze(1)
        level_diff = level_diff.clamp(-self.max_level, self.max_level) + self.max_level

        rel_pos_bias = self.relative_pos_embedding(level_diff)
        return rel_pos_bias.permute(2, 0, 1)

    def forward(
        self,
        x: torch.Tensor,
        levels_info: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, _, _ = x.shape

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = dots * self.scale_weights.view(1, -1, 1, 1)

        if self.use_level_scaling and levels_info is not None and levels_info.numel() > 0:
            depths = levels_info[:, 0].clamp(0, self.max_level)
            level_scales = self.level_scale_embedding(depths)
            level_scales = level_scales.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
            dots = dots * level_scales

        if levels_info is not None:
            hilbert_bias = self._compute_hilbert_bias(levels_info)
            if hilbert_bias is not None:
                dots = dots + hilbert_bias.unsqueeze(0) * 0.1

            level_bias = self._compute_level_bias(levels_info)
            if level_bias is not None:
                dots = dots + level_bias.unsqueeze(0) * 0.05

        if attention_mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            dots.masked_fill_(~attention_mask.bool(), mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
