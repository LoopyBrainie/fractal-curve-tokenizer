from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFractalFeedForward(nn.Module):
    """Adaptive feed-forward block aware of tokenizer hierarchy."""

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
            self.level_adapters = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim // 2, dim),
                        nn.Dropout(dropout),
                    )
                    for _ in range(max_level + 1)
                ]
            )
            self.level_mixing_weights = nn.Parameter(torch.ones(max_level + 1))
        else:
            self.level_adapters = None
            self.level_mixing_weights = None

        if use_feature_gating:
            self.feature_gate = nn.Sequential(
                nn.Linear(dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, hidden_dim),
                nn.Sigmoid(),
            )
        else:
            self.feature_gate = None

        self.activation_selector = nn.Sequential(nn.Linear(dim, 3), nn.Softmax(dim=-1))

    def _apply_dynamic_activation(self, x: torch.Tensor, activation_weights: torch.Tensor) -> torch.Tensor:
        gelu_out = F.gelu(x)
        relu_out = F.relu(x)
        swish_out = x * torch.sigmoid(x)
        return (
            activation_weights[:, :, 0:1] * gelu_out
            + activation_weights[:, :, 1:2] * relu_out
            + activation_weights[:, :, 2:3] * swish_out
        )

    def forward(self, x: torch.Tensor, levels_info: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        x_norm = self.norm(x)

        main_out = self.main_net(x_norm)

        if self.use_level_adaptation and levels_info is not None and levels_info.numel() > 0:
            depths = levels_info[:, 0].clamp(0, self.max_level)
            level_outputs = []
            for i, depth in enumerate(depths):
                depth_idx = int(depth.item())
                if i < seq_len:
                    token_input = x_norm[:, i : i + 1, :]
                    level_out = self.level_adapters[depth_idx](token_input)
                    level_outputs.append(level_out)

            if level_outputs:
                level_adapted = torch.cat(level_outputs, dim=1)
                mixing_weights = F.softmax(self.level_mixing_weights[depths], dim=0)
                mixing_weights = mixing_weights.view(1, seq_len, 1)
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
