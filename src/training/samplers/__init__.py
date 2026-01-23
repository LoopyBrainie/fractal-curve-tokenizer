# -*- coding: utf-8 -*-
"""类别平衡采样器模块

数学形式化
==========

**问题**: 标准均匀采样下，类别 c 的采样概率为:
    P_uniform(c) = n_c / N

其中 n_c 是类别 c 的样本数，N 是总样本数。

**解决方案**: 逆频率加权采样
    P_balanced(c) ∝ 1 / n_c^β

其中 β ∈ [0, 1] 控制平衡强度:
    - β = 0: 均匀采样 (无平衡)
    - β = 1: 完全逆频率采样 (每类等概率)
    - β = 0.5: 开方平衡 (推荐折中)

**样本级权重**:
    w_i = 1 / n_{c_i}^β

归一化后:
    P(sample i) = w_i / Σ_j w_j
"""

from .balanced_sampler import ClassBalancedSampler, ProgressiveSampler

import torch
from collections import Counter
from typing import Sequence


def compute_effective_sample_weights(
    class_counts: torch.Tensor,
    beta: float = 0.9999,
) -> torch.Tensor:
    """计算有效样本数权重 (Class-Balanced Loss)
    
    数学形式 (Cui et al., CVPR 2019):
        E_c = (1 - β^{n_c}) / (1 - β)  # 有效样本数
        w_c = 1 / E_c                   # 权重
    
    物理意义:
        当 β → 1 时，E_c → n_c (有效样本数等于实际样本数)
        当 β → 0 时，E_c → 1 (所有类别有效样本数相等)
    
    Args:
        class_counts: [num_classes] 每个类别的样本数
        beta: 平滑参数，推荐 0.9999
        
    Returns:
        [num_classes] 每个类别的权重
    """
    # 避免除零
    class_counts = class_counts.float().clamp(min=1)
    
    # E_c = (1 - β^{n_c}) / (1 - β)
    effective_num = (1.0 - torch.pow(beta, class_counts)) / (1.0 - beta)
    
    # w_c = 1 / E_c
    weights = 1.0 / effective_num
    
    # 归一化使平均权重为 1
    weights = weights / weights.mean()
    
    return weights


def get_class_counts(labels: Sequence[int], num_classes: int = None) -> torch.Tensor:
    """从标签序列获取类别统计
    
    Args:
        labels: 样本标签序列
        num_classes: 类别数量，None 则自动推断
        
    Returns:
        [num_classes] 每个类别的样本数
    """
    counter = Counter(labels)
    if num_classes is None:
        num_classes = max(counter.keys()) + 1
    
    counts = torch.zeros(num_classes, dtype=torch.long)
    for cls, count in counter.items():
        counts[cls] = count
    
    return counts


__all__ = [
    "ClassBalancedSampler",
    "ProgressiveSampler",
    "compute_effective_sample_weights",
    "get_class_counts",
]
