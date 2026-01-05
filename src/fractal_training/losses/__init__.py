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
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification
    
    数学形式:
        L = -α_c · (1 - p_c)^γ · log(p_c)
    
    其中:
        - p_c: 真实类别 c 的预测概率
        - γ: 聚焦参数 (推荐 2.0)
        - α_c: 类别权重 (可选)
    
    Args:
        gamma: 聚焦参数，γ ≥ 0。γ = 0 时退化为标准 CE
        alpha: 类别权重，可以是:
            - None: 不使用类别权重
            - float: 所有类别使用相同权重
            - Tensor[num_classes]: 每个类别的权重
        reduction: 'mean' | 'sum' | 'none'
        label_smoothing: 标签平滑系数
        
    Example:
        >>> loss_fn = FocalLoss(gamma=2.0)
        >>> logits = torch.randn(32, 10)  # [batch, classes]
        >>> targets = torch.randint(0, 10, (32,))
        >>> loss = loss_fn(logits, targets)
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, Tensor]] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        # 处理 alpha
        if alpha is None:
            self.register_buffer("alpha", None)
        elif isinstance(alpha, (int, float)):
            self.register_buffer("alpha", torch.tensor([alpha]))
        else:
            self.register_buffer("alpha", alpha)
    
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits: [B, C] 未归一化的预测分数
            targets: [B] 类别标签 或 [B, C] one-hot/soft 标签
            
        Returns:
            标量损失值 (reduction='mean'|'sum') 或 [B] (reduction='none')
        """
        num_classes = logits.size(-1)
        
        # 计算 log_softmax
        log_probs = F.log_softmax(logits, dim=-1)  # [B, C]
        probs = torch.exp(log_probs)  # [B, C]
        
        # 处理标签
        if targets.dim() == 1:
            # 整数标签 → one-hot
            targets_one_hot = F.one_hot(targets, num_classes).float()  # [B, C]
        else:
            targets_one_hot = targets.float()
        
        # 标签平滑
        if self.label_smoothing > 0:
            smooth_targets = targets_one_hot * (1 - self.label_smoothing)
            smooth_targets = smooth_targets + self.label_smoothing / num_classes
            targets_one_hot = smooth_targets
        
        # Focal 权重: (1 - p_c)^γ
        # p_t = Σ_c y_c · p_c (真实类别的概率)
        p_t = (probs * targets_one_hot).sum(dim=-1)  # [B]
        focal_weight = (1 - p_t) ** self.gamma  # [B]
        
        # CE 损失 (per sample)
        ce_loss = -(targets_one_hot * log_probs).sum(dim=-1)  # [B]
        
        # Focal Loss
        focal_loss = focal_weight * ce_loss  # [B]
        
        # 类别权重 (alpha)
        if self.alpha is not None:
            if targets.dim() == 1:
                alpha_t = self.alpha[targets]  # [B]
            else:
                alpha_t = (self.alpha.unsqueeze(0) * targets_one_hot).sum(dim=-1)
            focal_loss = alpha_t * focal_loss
        
        # Reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
    
    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, reduction={self.reduction}"


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
        
    Example:
        >>> class_counts = torch.tensor([500, 50, 200, 100])  # 4 类
        >>> loss_fn = ClassBalancedCE(class_counts, beta=0.9999)
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
        """
        Args:
            logits: [B, C] 未归一化的预测分数
            targets: [B] 类别标签
            
        Returns:
            损失值
        """
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
        
    Example:
        >>> loss_fn = CompositeLoss(
        ...     losses=[FocalLoss(), nn.MSELoss()],
        ...     weights=[1.0, 0.1],
        ... )
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
        
        self.register_buffer("weights", torch.tensor(weights))
    
    def forward(self, *args, **kwargs) -> Tensor:
        """计算加权组合损失"""
        total_loss = 0.0
        
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss = total_loss + weight * loss_fn(*args, **kwargs)
        
        return total_loss
