# -*- coding: utf-8 -*-
"""
Focal Loss

数学形式化
============

Focal Loss 通过降低易分类样本的权重，使模型专注于困难样本：

    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
    
其中:
    p_t = {  p    if y = 1
          { 1-p   otherwise
          
    α_t: 类别平衡因子（可选）
    γ: 聚焦参数，γ ∈ [0, 5]
        - γ = 0: 退化为标准 CE
        - γ = 2: 推荐值
        - γ ↑: 更强地抑制易分类样本

梯度分析
--------
∂FL/∂z = α_t [(1-p_t)^γ - γ·p_t·(1-p_t)^(γ-1)] · (p_t - y)

当 p_t → 1 (易分类):
    (1-p_t)^γ → 0，梯度趋近 0
    
当 p_t → 0.5 (困难样本):
    梯度较大，模型专注学习

参考文献
--------
Lin et al. "Focal Loss for Dense Object Detection", ICCV 2017
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    数学公式:
        FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
    
    参数
    ----
    alpha : float, list, or tensor, optional
        类别平衡因子
        - None: 不使用类别权重
        - float: 所有类别使用相同权重
        - list/tensor: 每个类别的权重 (长度=num_classes)
    gamma : float, optional (default=2.0)
        聚焦参数，控制对易分类样本的抑制强度
        推荐范围: [0, 5]
    reduction : str, optional (default='mean')
        输出约简方式: 'none', 'mean', 'sum'
    label_smoothing : float, optional (default=0.0)
        标签平滑因子 ∈ [0, 1]
        
    示例
    ----
    >>> loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    >>> logits = torch.randn(10, 5)  # batch=10, classes=5
    >>> targets = torch.randint(0, 5, (10,))
    >>> loss = loss_fn(logits, targets)
    """
    
    def __init__(
        self,
        alpha: Optional[float | list | torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if not 0 <= label_smoothing < 1:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction}")
        
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        # 处理 alpha 参数
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            self.alpha = float(alpha)
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播
        
        参数
        ----
        logits : Tensor, shape [N, C]
            模型输出 logits (未经 softmax)
        targets : Tensor, shape [N]
            目标类别索引
            
        返回
        ----
        loss : Tensor
            Focal Loss 值
        """
        # 计算概率
        p = F.softmax(logits, dim=-1)  # [N, C]
        
        # 标签平滑
        if self.label_smoothing > 0:
            targets = self._smooth_labels(targets, logits.size(-1))
            ce_loss = -torch.sum(targets * torch.log_softmax(logits, dim=-1), dim=-1)  # [N]
            p_t = torch.sum(p * targets, dim=-1)  # [N]
        else:
            # 提取目标类别的概率 p_t
            ce_loss = F.cross_entropy(logits, targets, reduction="none")  # [N]
            p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)  # [N]
        
        # 计算 Focal权重: (1 - p_t)^γ
        focal_weight = (1 - p_t).pow(self.gamma)
        
        # 应用类别权重 α_t
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # 每个类别不同的权重
                if self.alpha.device != logits.device:
                    self.alpha = self.alpha.to(logits.device)
                alpha_t = self.alpha.gather(0, targets)  # [N]
            else:
                # 统一权重
                alpha_t = self.alpha
            
            focal_weight = alpha_t * focal_weight
        
        # Focal Loss = α_t (1-p_t)^γ CE(p_t)
        loss = focal_weight * ce_loss
        
        # 约简
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
    
    def _smooth_labels(self, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        标签平滑
        
        y_smooth = (1 - ε) * y + ε / C
        """
        confidence = 1.0 - self.label_smoothing
        smooth_label = torch.full(
            (targets.size(0), num_classes),
            self.label_smoothing / num_classes,
            dtype=torch.float32,
            device=targets.device
        )
        smooth_label.scatter_(1, targets.unsqueeze(1), confidence)
        return smooth_label
    
    def extra_repr(self) -> str:
        """打印额外信息"""
        return f"gamma={self.gamma}, alpha={self.alpha}, reduction={self.reduction}"
