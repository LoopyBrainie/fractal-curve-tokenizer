from __future__ import annotations

from typing import List, Tuple

import torch


def pair(value: int | Tuple[int, int]) -> Tuple[int, int]:
    return value if isinstance(value, tuple) else (value, value)


def exists(value) -> bool:
    return value is not None


def default(value, default_value):
    return value if exists(value) else default_value


def create_attention_mask(levels_info: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Generate a soft attention prior based on hierarchical depth alignment."""
    if len(levels_info) == 0:
        return torch.empty(0, 0, 0, device=device)

    max_len = max(info.shape[0] for info in levels_info)
    batch_size = len(levels_info)

    mask = torch.ones(batch_size, max_len, max_len, device=device)

    for batch_index, level_info in enumerate(levels_info):
        seq_len = len(level_info)
        if seq_len == 0:
            continue

        for i in range(seq_len):
            for j in range(seq_len):
                if i < len(level_info) and j < len(level_info):
                    level_i = level_info[i, 0].item() if level_info[i].numel() > 0 else 0
                    level_j = level_info[j, 0].item() if level_info[j].numel() > 0 else 0

                    if level_i == level_j:
                        mask[batch_index, i, j] = 1.2
                    elif abs(level_i - level_j) == 1:
                        mask[batch_index, i, j] = 1.1
                    else:
                        mask[batch_index, i, j] = 1.0

    return mask
