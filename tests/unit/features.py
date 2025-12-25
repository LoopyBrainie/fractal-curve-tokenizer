# -*- coding: utf-8 -*-
"""
Token 特征计算模块

数学形式化
============

TokenFeatures 包含 4 类特征:

1. 统计特征 (stats ∈ R^{N × 2}):
   - variance: σ² = Var(token)
   - mean: μ = E[token]

2. 边缘特征 (edge ∈ R^{N × 1}):
   - edge_density = mean(|diff(token)|)
   表征局部梯度强度

3. 空间特征 (spatial ∈ R^{N × 2}):
   - height, width (归一化后的 patch 尺寸)

4. 层级特征 (level ∈ R^{N × 1}):
   - depth 值，用于层级感知处理

总特征维度: 6 维
用途: 分割决策网络的输入特征
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class TokenFeatures:
    """Container for token-level statistics used across tokenizer components."""

    stats: torch.Tensor
    edge: torch.Tensor
    spatial: torch.Tensor
    level: torch.Tensor


def compute_token_features(
    tokens: torch.Tensor,
    *,
    level: int,
    patch_size: Tuple[int, int],
) -> TokenFeatures:
    """Compute basic statistics for a batch of flattened tokens.

    Args:
        tokens: Flattened token tensor with shape [N, token_dim].
        level: Hierarchical depth associated with the tokens.
        patch_size: Original patch spatial dimensions (height, width).

    Returns:
        TokenFeatures with per-token statistics, edge proxy, spatial metadata and level encoding.
    """

    device = tokens.device
    batch_size = tokens.shape[0]
    patch_height, patch_width = patch_size

    if tokens.shape[-1] > 1:
        token_var = torch.var(tokens, dim=-1, keepdim=True)
    else:
        token_var = torch.zeros(batch_size, 1, device=device)

    token_mean = torch.mean(tokens, dim=-1, keepdim=True)

    if tokens.shape[-1] > 1:
        token_diff = torch.diff(tokens, dim=-1)
        edge_density = torch.mean(torch.abs(token_diff), dim=-1, keepdim=True)
    else:
        edge_density = torch.zeros(batch_size, 1, device=device)

    spatial_features = torch.tensor(
        [[patch_height, patch_width]], device=device, dtype=torch.float32
    ).expand(batch_size, -1)
    level_features = torch.tensor([[level]], device=device, dtype=torch.float32).expand(batch_size, -1)

    return TokenFeatures(
        stats=torch.cat([token_var, token_mean], dim=-1),
        edge=edge_density,
        spatial=spatial_features,
        level=level_features,
    )
