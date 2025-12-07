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
    """Generate a soft attention prior based on hierarchical depth alignment.
    
    向量化实现，将 O(B × S²) 循环复杂度降低为批量张量操作。
    
    Args:
        levels_info: List of (N_i, info_len) tensors containing level information for each sample
        device: Target device for the output tensor
        
    Returns:
        (B, max_len, max_len) attention mask with values:
        - 1.2 for same level
        - 1.1 for adjacent levels (diff=1)
        - 1.0 for others or padding
    """
    if len(levels_info) == 0:
        return torch.empty(0, 0, 0, device=device)

    batch_size = len(levels_info)
    # 找到最大序列长度
    max_len = max((info.shape[0] for info in levels_info if info.numel() > 0), default=0)
    
    if max_len == 0:
        return torch.ones(batch_size, 1, 1, device=device)

    # 提取深度值并 pad 到 max_len，用 -1 表示 padding
    depths_list = []
    for info in levels_info:
        if info.numel() > 0:
            depth = info[:, 0].float()  # (N_i,)
            # Pad to max_len
            if depth.shape[0] < max_len:
                padding = torch.full((max_len - depth.shape[0],), -1.0, device=device)
                depth = torch.cat([depth, padding], dim=0)
        else:
            depth = torch.full((max_len,), -1.0, device=device)
        depths_list.append(depth)
    
    depths = torch.stack(depths_list)  # (B, max_len)
    
    # 向量化计算层级差异
    depth_i = depths.unsqueeze(2)  # (B, max_len, 1)
    depth_j = depths.unsqueeze(1)  # (B, 1, max_len)
    diff = torch.abs(depth_i - depth_j)  # (B, max_len, max_len)
    
    # 生成 mask 值：同级=1.2，相邻级=1.1，其他=1.0
    mask = torch.where(diff == 0, torch.tensor(1.2, device=device), 
           torch.where(diff == 1, torch.tensor(1.1, device=device), 
                       torch.tensor(1.0, device=device)))
    
    # 处理 padding 位置：padding 位置的 mask 设为 1.0
    valid_i = (depths >= 0).unsqueeze(2)  # (B, max_len, 1)
    valid_j = (depths >= 0).unsqueeze(1)  # (B, 1, max_len)
    valid_mask = valid_i & valid_j  # (B, max_len, max_len)
    
    # 对于 padding 位置，重置为 1.0
    mask = torch.where(valid_mask, mask, torch.tensor(1.0, device=device))

    return mask
