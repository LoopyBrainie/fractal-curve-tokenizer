# -*- coding: utf-8 -*-
r"""
Utility functions module.

Mathematical Formulation
=========================

Core functions:
    create_attention_mask(lengths, max_len) → M ∈ {0,1}^{B × N}
        M[b, i] = 1 if i < lengths[b] else 0

    sanitize_tensor(T) → T'
        T' = nan_to_num(T), replaces NaN/Inf for numerical stability

Function Reference Table
-------------------------
+------------------------+-------------------------------+
| Function               | Mathematical Definition       |
+========================+===============================+
| pair(x)                | x → (x, x) if int else x      |
| create_attention_mask  | lengths → bool mask           |
| sanitize_tensor        | T → nan_to_num(T)             |
+------------------------+-------------------------------+
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Union

import torch

logger = logging.getLogger(__name__)


def pair(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    r"""
    Convert a value to a tuple of length 2.

    If the value is already a tuple, return it as-is.
    Otherwise, create a tuple with the value repeated twice.

    Args:
        value (int or Tuple[int, int]): Input value

    Returns:
        Tuple[int, int]: Tuple of (value, value) if input is int, otherwise the tuple itself

    Examples::

        >>> pair(224)
        (224, 224)
        >>> pair((384, 384))
        (384, 384)
    """
    return value if isinstance(value, tuple) else (value, value)


def exists(value) -> bool:
    r"""
    Check if a value is not None.

    Args:
        value: Any value to check

    Returns:
        bool: ``True`` if value is not ``None``, ``False`` otherwise

    Examples::

        >>> exists(None)
        False
        >>> exists(42)
        True
    """
    return value is not None


def default(value, default_value):
    r"""
    Return value if it exists (not None), otherwise return default_value.

    Args:
        value: Value to check
        default_value: Default value to return if value is None

    Returns:
        value if exists, otherwise default_value

    Examples::

        >>> default(None, 0)
        0
        >>> default(42, 0)
        42
    """
    return value if exists(value) else default_value


def sanitize_tensor(
    tensor: torch.Tensor,
    nan_value: float = 0.0,
    posinf_value: float = 1.0,
    neginf_value: float = -1.0,
) -> torch.Tensor:
    r"""
    Clean NaN and Inf values from a tensor.

    Checks if the tensor contains any NaN or Inf values, and replaces them
    with specified default values if present. This is important for numerical
    stability, especially when dealing with softmax or normalization operations.

    Args:
        tensor (Tensor): Input tensor
        nan_value (float): Replacement value for NaN. Default: ``0.0``
        posinf_value (float): Replacement value for positive infinity. Default: ``1.0``
        neginf_value (float): Replacement value for negative infinity. Default: ``-1.0``

    Returns:
        Tensor: Cleaned tensor (returns original if no NaN/Inf present)

    Examples::

        >>> x = torch.tensor([1.0, float('nan'), float('inf')])
        >>> sanitize_tensor(x)
        tensor([ 1.,  0.,  1.])
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
    r"""
    Generate a soft attention prior based on hierarchical depth alignment.

    Vectorized implementation that reduces O(B × S²) loop complexity to batch tensor operations.

    Args:
        levels_info (List[Tensor]): List of (N_i, info_len) tensors containing level information
            for each sample in the batch
        device (torch.device): Target device for the output tensor

    Returns:
        Tensor: Attention mask of shape :math:`(B, max\_len, max\_len)` with values:
            - ``1.2`` for same level (depth diff = 0)
            - ``1.1`` for adjacent levels (depth diff = 1)
            - ``1.0`` for other positions or padding
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
