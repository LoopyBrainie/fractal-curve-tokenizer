# -*- coding: utf-8 -*-
"""
损失函数模块

提供类别平衡损失、Focal Loss 和资源感知损失。
"""

from .focal_loss import FocalLoss
from .balanced_ce import ClassBalancedCrossEntropy
from .resource_loss import ResourceAwareLoss, compute_depth_entropy, get_weighted_token_count

__all__ = [
    "FocalLoss",
    "ClassBalancedCrossEntropy",
    "ResourceAwareLoss",
    "compute_depth_entropy",
    "get_weighted_token_count",
]
