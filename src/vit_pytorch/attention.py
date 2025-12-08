# -*- coding: utf-8 -*-
"""Hilbert-aware multi-scale attention module.

This module implements HilbertAwareMultiScaleAttention, which encodes
hierarchical depth and Hilbert path relationships to modulate attention weights.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .constants import HILBERT_BIAS_SCALE, LEVEL_BIAS_SCALE
from .utils import extract_depths


class HilbertAwareMultiScaleAttention(nn.Module):
    """Hilbert 曲线感知的多尺度注意力机制。

    通过编码层级深度和 Hilbert 路径关系来调制注意力权重。
    
    Attributes:
        heads: 注意力头数
        dim_head: 每个头的维度
        max_level: 最大层级
        use_hilbert_bias: 是否使用 Hilbert 偏置
        use_level_scaling: 是否使用层级缩放
        scale: 注意力缩放因子
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
    ) -> None:
        """初始化 HilbertAwareMultiScaleAttention。
        
        Args:
            dim: 输入维度
            heads: 注意力头数
            dim_head: 每个头的维度
            dropout: Dropout 比率
            max_level: 最大层级
            use_hilbert_bias: 是否使用 Hilbert 路径偏置
            use_level_scaling: 是否使用层级缩放
        """
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
            self.hilbert_bias_network: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Linear(64, heads),
                nn.Tanh(),
            )
        else:
            self.hilbert_bias_network = None

        if use_level_scaling:
            self.level_scale_embedding: Optional[nn.Embedding] = nn.Embedding(max_level + 1, heads)
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
        """计算基于 Hilbert 路径的注意力偏置。
        
        Args:
            levels_info: 层级信息张量，形状为 (Seq, Info) 或 (Batch, Seq, Info)
            
        Returns:
            Hilbert 偏置张量，形状为 (H, S, S) 或 (B, H, S, S)，若无效则返回 None
        """
        if not self.use_hilbert_bias or levels_info.numel() == 0:
            return None

        # Type guard: guaranteed non-None when use_hilbert_bias is True
        assert self.hilbert_bias_network is not None

        device = levels_info.device

        if levels_info.dim() == 2:
            # Old behavior: (Seq, Info)
            seq_len = levels_info.shape[0]
            if levels_info.shape[1] <= 1:
                return None
            paths = levels_info[:, 1:].float() # (S, Path)
            path_i = paths.unsqueeze(1) # (S, 1, Path)
            path_j = paths.unsqueeze(0) # (1, S, Path)
            
            path_dist = torch.norm(path_i - path_j, dim=2)
            path_sim = F.cosine_similarity(path_i, path_j, dim=2, eps=1e-6)
            path_features = torch.stack([path_dist, path_sim], dim=-1) # (S, S, 2)
            
            bias = self.hilbert_bias_network(path_features) # (S, S, H)
            return bias.permute(2, 0, 1) # (H, S, S)
        else:
            # New behavior: (Batch, Seq, Info)
            batch_size, seq_len, info_dim = levels_info.shape
            if info_dim <= 1:
                return None
            
            paths = levels_info[:, :, 1:].float() # (B, S, Path)
            path_i = paths.unsqueeze(2) # (B, S, 1, Path)
            path_j = paths.unsqueeze(1) # (B, 1, S, Path)
            
            path_dist = torch.norm(path_i - path_j, dim=3) # (B, S, S)
            path_sim = F.cosine_similarity(path_i, path_j, dim=3, eps=1e-6) # (B, S, S)
            
            path_features = torch.stack([path_dist, path_sim], dim=-1) # (B, S, S, 2)
            
            bias = self.hilbert_bias_network(path_features) # (B, S, S, H)
            return bias.permute(0, 3, 1, 2) # (B, H, S, S)

    def _compute_level_bias(self, levels_info: torch.Tensor) -> Optional[torch.Tensor]:
        """计算基于层级差异的相对位置偏置。
        
        Args:
            levels_info: 层级信息张量，形状为 (Seq, Info) 或 (Batch, Seq, Info)
            
        Returns:
            层级偏置张量，形状为 (H, S, S) 或 (B, H, S, S)，若无效则返回 None
        """
        if levels_info.numel() == 0:
            return None

        if levels_info.dim() == 2:
            # Old behavior: (Seq, Info)
            depths = extract_depths(levels_info, self.max_level)
            level_diff = depths.unsqueeze(0) - depths.unsqueeze(1)
            level_diff = level_diff.clamp(-self.max_level, self.max_level) + self.max_level
            rel_pos_bias = self.relative_pos_embedding(level_diff)
            return rel_pos_bias.permute(2, 0, 1) # (H, S, S)
        else:
            # New behavior: (Batch, Seq, Info)
            depths = extract_depths(levels_info, self.max_level) # (B, S)
            level_diff = depths.unsqueeze(2) - depths.unsqueeze(1) # (B, S, S)
            level_diff = level_diff.clamp(-self.max_level, self.max_level) + self.max_level
            rel_pos_bias = self.relative_pos_embedding(level_diff) # (B, S, S, H)
            return rel_pos_bias.permute(0, 3, 1, 2) # (B, H, S, S)

    def forward(
        self,
        x: torch.Tensor,
        levels_info: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播。
        
        Args:
            x: 输入张量，形状为 [B, N, D]
            levels_info: 层级信息（可选）
            attention_mask: 注意力掩码（可选）
            
        Returns:
            输出张量，形状为 [B, N, D]
        """
        batch, _, _ = x.shape

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = dots * self.scale_weights.view(1, -1, 1, 1)

        if self.use_level_scaling and levels_info is not None and levels_info.numel() > 0:
            # Type guard: guaranteed non-None when use_level_scaling is True
            assert self.level_scale_embedding is not None
            
            depths = extract_depths(levels_info, self.max_level)
            if levels_info.dim() == 2:
                level_scales = self.level_scale_embedding(depths)
                level_scales = level_scales.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
            else:
                level_scales = self.level_scale_embedding(depths) # (B, S, H)
                level_scales = level_scales.permute(0, 2, 1).unsqueeze(-1) # (B, H, S, 1)
            
            dots = dots * level_scales

        if levels_info is not None:
            hilbert_bias = self._compute_hilbert_bias(levels_info)
            if hilbert_bias is not None:
                # hilbert_bias: (H, S, S) or (B, H, S, S)
                if hilbert_bias.dim() == 3:
                    dots = dots + hilbert_bias.unsqueeze(0) * HILBERT_BIAS_SCALE
                else:
                    dots = dots + hilbert_bias * HILBERT_BIAS_SCALE

            level_bias = self._compute_level_bias(levels_info)
            if level_bias is not None:
                # level_bias: (H, S, S) or (B, H, S, S)
                if level_bias.dim() == 3:
                    dots = dots + level_bias.unsqueeze(0) * LEVEL_BIAS_SCALE
                else:
                    dots = dots + level_bias * LEVEL_BIAS_SCALE

        if attention_mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            dots.masked_fill_(~attention_mask.bool(), mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
