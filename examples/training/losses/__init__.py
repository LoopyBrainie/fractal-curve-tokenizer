# -*- coding: utf-8 -*-
"""类别平衡损失函数模块

数学形式化
==========

**Focal Loss** (Lin et al., ICCV 2017):
    L_focal = -Σ_c α_c (1 - p_c)^γ · y_c · log(p_c)

参数:
    - γ (gamma): 聚焦参数，γ > 0 时降低易分类样本权重
    - α_c (alpha): 类别平衡权重
    
效果分析:
    当 p_c → 1 (易分类): (1 - p_c)^γ → 0，损失被抑制
    当 p_c → 0 (难分类): (1 - p_c)^γ → 1，损失保持

**Class-Balanced CE** (Cui et al., CVPR 2019):
    L_cb = -Σ_c (1/E_c) · y_c · log(p_c)
    
其中 E_c = (1 - β^{n_c}) / (1 - β) 是有效样本数。

**组合损失**:
    L_total = Σ_i λ_i · L_i

**资源感知损失**:
    L_resource = λ_flops·L_flops + λ_token·L_token + λ_entropy·L_entropy
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# 导入已有的实现
from .focal_loss import FocalLoss
from .balanced_ce import ClassBalancedCrossEntropy, compute_class_weights_from_targets
from .resource_loss import ResourceAwareLoss, compute_depth_entropy, get_weighted_token_count
from .hilbert_hard_mining import (
    HilbertAwareHardMining,
    HilbertMiningWrapper,
    create_hilbert_mining_loss,
)


# ============================================================================
# ClassBalancedCE 别名
# ============================================================================

class ClassBalancedCE(nn.Module):
    """Class-Balanced Cross-Entropy Loss
    
    数学形式 (Cui et al., CVPR 2019):
        L = -(1/E_c) · log(p_c)
        E_c = (1 - β^{n_c}) / (1 - β)  # 有效样本数
    
    当 β → 1: E_c → n_c (标准逆频率权重)
    当 β → 0: E_c → 1 (无权重)
    
    Args:
        class_counts: [num_classes] 每个类别的样本数
        beta: 有效样本数参数，推荐 0.9999
        reduction: 'mean' | 'sum' | 'none'
        label_smoothing: 标签平滑系数
    """
    
    def __init__(
        self,
        class_counts: Tensor,
        beta: float = 0.9999,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        
        self.beta = beta
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        # 计算有效样本数权重
        weights = self._compute_weights(class_counts, beta)
        self.register_buffer("weights", weights)
    
    @staticmethod
    def _compute_weights(class_counts: Tensor, beta: float) -> Tensor:
        """计算类别权重"""
        class_counts = class_counts.float().clamp(min=1)
        
        # E_c = (1 - β^{n_c}) / (1 - β)
        effective_num = (1.0 - torch.pow(beta, class_counts)) / (1.0 - beta)
        
        # w_c = 1 / E_c
        weights = 1.0 / effective_num
        
        # 归一化
        weights = weights / weights.sum() * len(weights)
        
        return weights
    
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return F.cross_entropy(
            logits,
            targets,
            weight=self.weights,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
    
    def extra_repr(self) -> str:
        return f"beta={self.beta}, num_classes={len(self.weights)}"


class FocalClassBalancedLoss(nn.Module):
    """Focal Loss + Class-Balanced 权重的组合
    
    数学形式:
        L = -(1/E_c) · (1 - p_c)^γ · log(p_c)
    
    结合了:
    - Focal Loss 的难样本聚焦
    - Class-Balanced 的类别平衡
    
    Args:
        class_counts: [num_classes] 每个类别的样本数
        gamma: Focal Loss 聚焦参数
        beta: 有效样本数参数
        reduction: 'mean' | 'sum' | 'none'
    """
    
    def __init__(
        self,
        class_counts: Tensor,
        gamma: float = 2.0,
        beta: float = 0.9999,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        
        # 计算类别权重
        weights = ClassBalancedCE._compute_weights(class_counts, beta)
        
        self.focal = FocalLoss(
            gamma=gamma,
            alpha=weights,
            reduction=reduction,
        )
    
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.focal(logits, targets)


class CompositeLoss(nn.Module):
    """组合多个损失函数
    
    数学形式:
        L_total = Σ_i λ_i · L_i
    
    Args:
        losses: 损失函数列表
        weights: 权重列表，None 表示均等权重
    """
    
    def __init__(
        self,
        losses: list[nn.Module],
        weights: Optional[list[float]] = None,
    ) -> None:
        super().__init__()
        
        self.losses = nn.ModuleList(losses)
        
        if weights is None:
            weights = [1.0] * len(losses)
        
        if len(weights) != len(losses):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of losses ({len(losses)})"
            )
        
        self.register_buffer("loss_weights", torch.tensor(weights))
    
    def forward(self, *args, **kwargs) -> Tensor:
        """计算加权组合损失"""
        total_loss = 0.0
        
        for loss_fn, weight in zip(self.losses, self.loss_weights):
            total_loss = total_loss + weight * loss_fn(*args, **kwargs)
        
        return total_loss


__all__ = [
    # 核心损失函数
    "FocalLoss",
    "ClassBalancedCE",
    "ClassBalancedCrossEntropy",  # 别名
    "FocalClassBalancedLoss",
    "CompositeLoss",
    # 资源感知损失
    "ResourceAwareLoss",
    "compute_depth_entropy",
    "get_weighted_token_count",
    # I30-2: Hilbert-aware 困难样本挖掘
    "HilbertAwareHardMining",
    "HilbertMiningWrapper",
    "create_hilbert_mining_loss",
    # 工具函数
    "compute_class_weights_from_targets",
]
