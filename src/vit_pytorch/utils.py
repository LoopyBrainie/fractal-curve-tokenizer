# -*- coding: utf-8 -*-
"""
工具函数模块

数学形式化
============

核心函数:
    extract_depths(L) → d ∈ Z^N
        从 levels_info 提取深度值，d_i = L[i, 0]
    
    normalize_levels_info(L, L_max) → L' ∈ Z^{N × info_len}
        归一化层级信息维度
    
    create_attention_mask(lengths, max_len) → M ∈ {0,1}^{B × N}
        M[b, i] = 1 if i < lengths[b] else 0
    
    sanitize_tensor(T) → T'
        T' = nan_to_num(T), 替换 NaN/Inf 保证数值稳定性

函数对照表
----------
+------------------------+-------------------------------+
| 函数                    | 数学定义                       |
+========================+===============================+
| pair(x)                | x → (x, x) if int else x      |
| extract_depths(L)      | L → L[:, 0].clamp(0, L_max)   |
| normalize_levels_info  | L → pad/truncate to info_len  |
| create_attention_mask  | lengths → bool mask           |
| sanitize_tensor        | T → nan_to_num(T)             |
+------------------------+-------------------------------+
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)


def pair(value: int | Tuple[int, int]) -> Tuple[int, int]:
    return value if isinstance(value, tuple) else (value, value)


def exists(value) -> bool:
    return value is not None


def default(value, default_value):
    return value if exists(value) else default_value


def sanitize_tensor(
    tensor: torch.Tensor,
    nan_value: float = 0.0,
    posinf_value: float = 1.0,
    neginf_value: float = -1.0,
) -> torch.Tensor:
    """清理张量中的 NaN 和 Inf 值。
    
    检查张量中是否存在 NaN 或 Inf 值，如果存在则替换为指定的默认值。
    这对于数值稳定性很重要，特别是在处理 softmax 或归一化操作时。
    
    Args:
        tensor: 输入张量
        nan_value: NaN 值的替换值，默认为 0.0
        posinf_value: 正无穷的替换值，默认为 1.0
        neginf_value: 负无穷的替换值，默认为 -1.0
        
    Returns:
        清理后的张量（如果没有 NaN/Inf 则返回原张量）
    """
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        logger.debug(
            "sanitize_tensor: Detected NaN/Inf in tensor with shape %s, replacing values",
            tuple(tensor.shape)
        )
        return torch.nan_to_num(
            tensor, nan=nan_value, posinf=posinf_value, neginf=neginf_value
        )
    return tensor


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


def extract_depths(levels_info: torch.Tensor, max_level: int) -> torch.Tensor:
    """统一从 levels_info 提取深度索引，自动处理 2D/3D 张量。
    
    Args:
        levels_info: 层级信息张量，形状为 (Seq, Info) 或 (Batch, Seq, Info)
        max_level: 最大层级值，用于 clamp
        
    Returns:
        深度索引张量，形状为 (Seq,) 或 (Batch, Seq)
    """
    if levels_info.dim() == 2:
        depths = levels_info[:, 0]
    else:
        depths = levels_info[:, :, 0]
    return depths.clamp(0, max_level).long()


def normalize_levels_info(levels_info: torch.Tensor) -> torch.Tensor:
    """规范化 levels_info 为 (Batch, Seq, Info) 格式。
    
    Args:
        levels_info: 层级信息张量，形状为 (Seq, Info) 或 (Batch, Seq, Info)
        
    Returns:
        规范化后的张量，形状为 (Batch, Seq, Info)
    """
    if levels_info.dim() == 2:
        return levels_info.unsqueeze(0)
    return levels_info
