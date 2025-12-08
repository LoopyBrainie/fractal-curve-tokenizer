# -*- coding: utf-8 -*-
"""Adaptive fractal-aware feed-forward network module.

This module implements AdaptiveFractalFeedForward, a feed-forward block
that is aware of the hierarchical structure from the fractal tokenizer,
applying depth-dependent scaling to the hidden representations.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import extract_depths


class AdaptiveFractalFeedForward(nn.Module):
    """Adaptive feed-forward block aware of tokenizer hierarchy.
    
    This module applies a feed-forward transformation to the input,
    with optional depth-aware level adaptation and feature gating.
    
    Args:
        dim: Input/output dimension.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout rate.
        max_level: Maximum hierarchical level for embeddings.
        use_level_adaptation: Whether to use level-aware adaptation.
        use_feature_gating: Whether to use feature gating.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        max_level: int = 50,
        use_level_adaptation: bool = True,
        use_feature_gating: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.max_level = max_level
        self.use_level_adaptation = use_level_adaptation
        self.use_feature_gating = use_feature_gating

        self.norm = nn.LayerNorm(dim)
        self.main_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

        if use_level_adaptation:
            # REFACTORED: Replaced 50 separate adapters with a single shared adapter + level embedding
            # This drastically reduces parameter count and enables vectorized execution.
            self.level_embedding: Optional[nn.Embedding] = nn.Embedding(max_level + 1, dim)
            self.shared_level_adapter: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(dim * 2, hidden_dim // 2), # Input: concatenated token + level_emb
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, dim),
                nn.Dropout(dropout),
            )
            self.level_mixing_weights: Optional[nn.Parameter] = nn.Parameter(torch.ones(max_level + 1))
        else:
            self.level_embedding = None
            self.shared_level_adapter = None
            self.level_mixing_weights = None

        if use_feature_gating:
            self.feature_gate: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, hidden_dim),
                nn.Sigmoid(),
            )
        else:
            self.feature_gate = None

        self.activation_selector = nn.Sequential(nn.Linear(dim, 3), nn.Softmax(dim=-1))

    def _apply_dynamic_activation(self, x: torch.Tensor, activation_weights: torch.Tensor) -> torch.Tensor:
        """应用动态加权的激活函数组合。
        
        根据输入计算的权重，混合 GELU、ReLU 和 Swish 三种激活函数。
        
        Args:
            x: 输入张量，形状为 [B, S, D]。
            activation_weights: 激活函数权重，形状为 [B, S, 3]。
            
        Returns:
            加权混合后的输出，形状为 [B, S, D]。
        """
        gelu_out = F.gelu(x)
        relu_out = F.relu(x)
        swish_out = x * torch.sigmoid(x)
        return (
            activation_weights[:, :, 0:1] * gelu_out
            + activation_weights[:, :, 1:2] * relu_out
            + activation_weights[:, :, 2:3] * swish_out
        )

    def forward(self, x: torch.Tensor, levels_info: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播。
        
        Args:
            x: 输入张量，形状为 [B, S, D]。
            levels_info: 层级信息，形状为 (S, Info) 或 (B, S, Info)。
            
        Returns:
            输出张量，形状为 [B, S, D]。
        """
        batch, seq_len, _ = x.shape
        x_norm = self.norm(x)

        main_out = self.main_net(x_norm)

        if self.use_level_adaptation and levels_info is not None and levels_info.numel() > 0:
            # Type guard: these are guaranteed non-None when use_level_adaptation is True
            assert self.level_embedding is not None
            assert self.shared_level_adapter is not None
            assert self.level_mixing_weights is not None
            
            if levels_info.dim() == 2:
                # Old behavior: (Seq, Info)
                depths = extract_depths(levels_info, self.max_level)
                level_embs = self.level_embedding(depths) # (Seq, Dim)
                level_embs = level_embs.unsqueeze(0).expand(batch, -1, -1) # (Batch, Seq, Dim)
                
                mixing_weights = F.softmax(self.level_mixing_weights[depths], dim=0)
                mixing_weights = mixing_weights.view(1, seq_len, 1)
            else:
                # New behavior: (Batch, Seq, Info)
                depths = extract_depths(levels_info, self.max_level) # (Batch, Seq)
                level_embs = self.level_embedding(depths) # (Batch, Seq, Dim)
                
                # Softmax across sequence dimension to match original behavior
                mixing_weights = F.softmax(self.level_mixing_weights[depths], dim=1) # (Batch, Seq)
                mixing_weights = mixing_weights.unsqueeze(-1) # (Batch, Seq, 1)
            
            # 3. Concatenate with input: (batch, seq_len, dim * 2)
            adapter_input = torch.cat([x_norm, level_embs], dim=-1)
            
            # 4. Pass through shared adapter
            level_adapted = self.shared_level_adapter(adapter_input)
            
            main_out = main_out * (1 - mixing_weights) + level_adapted * mixing_weights

        if self.use_feature_gating and self.feature_gate is not None:
            gates = self.feature_gate(x_norm)
            hidden = F.linear(x_norm, self.main_net[0].weight, self.main_net[0].bias)
            gated_hidden = hidden * gates
            activation_weights = self.activation_selector(x_norm)
            activated_hidden = self._apply_dynamic_activation(gated_hidden, activation_weights)
            main_out = F.linear(activated_hidden, self.main_net[3].weight, self.main_net[3].bias)
            main_out = self.main_net[4](main_out)

        return main_out
