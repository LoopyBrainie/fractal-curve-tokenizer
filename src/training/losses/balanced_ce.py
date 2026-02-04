# -*- coding: utf-8 -*-
"""
Class-Balanced Cross Entropy Loss

数学形式化
============

基于有效样本数（Effective Number of Samples）的类别平衡交叉熵：

有效样本数:
    E_n = (1 - β^n) / (1 - β)
    
其中:
    n: 该类别的样本数
    β ∈ [0, 1): 平衡参数
        - β = 0: E_n = 1 (完全不平衡)
        - β → 1: E_n → n (无平衡)
        - β = 0.999, 0.9999: 常用值

类别权重:
    w_c = 1 / E_{n_c}
    
Class-Balanced CE:
    L_CB = -Σ_c w_c y_c log(p_c)

直观理解
--------
有效样本数建模了样本的"边际效用递减"：
- 第 1 个样本: 完全新信息，权重 1
- 第 100 个样本: 信息重叠高，权重 < 1
- E_n 近似于"独立样本数"

参考文献
--------
Cui et al. "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalancedCrossEntropy(nn.Module):
    """
    基于有效样本数的类别平衡交叉熵
    
    数学公式:
        E_n = (1 - β^n) / (1 - β)
        w_c = 1 / E_{n_c}
        L = -Σ w_c y_c log(p_c)
    
    参数
    ----
    samples_per_class : list or array-like
        每个类别的样本数，长度 = num_classes
    beta : float, optional (default=0.9999)
        平衡参数 ∈ [0, 1)
        - 0.999: 强平衡
        - 0.9999: 中等平衡（推荐）
        - 0.99999: 弱平衡
    reduction : str, optional (default='mean')
        输出约简方式: 'none', 'mean', 'sum'
    label_smoothing : float, optional (default=0.0)
        标签平滑因子 ∈ [0, 1]
        
    示例
    ----
    >>> # Tiny-ImageNet: 200 类，不平衡分布
    >>> samples_per_class = [500, 450, 100, 80, ...]  # 200 个值
    >>> loss_fn = ClassBalancedCrossEntropy(samples_per_class, beta=0.9999)
    >>> 
    >>> logits = model(images)  # [N, 200]
    >>> targets = labels        # [N]
    >>> loss = loss_fn(logits, targets)
    """
    
    def __init__(
        self,
        samples_per_class: list | np.ndarray,
        beta: float = 0.9999,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        
        if not 0 <= beta < 1:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        if not 0 <= label_smoothing < 1:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction}")
        
        self.beta = beta
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        # 计算有效样本数 (I145: 添加 clamp 防止下溢)
        samples_per_class = np.array(samples_per_class, dtype=np.float64)
        # 当 beta^n 下溢到 0 时，使用极限公式: (1 - 0) / (1 - beta) = 1 / (1 - beta)
        beta_power = np.power(beta, samples_per_class)
        # clamp 防止数值下溢: 至少保留一个小 epsilon
        beta_power = np.clip(beta_power, a_min=1e-10, a_max=1.0)
        effective_num = (1.0 - beta_power) / (1.0 - beta)
        
        # 类别权重: w_c = 1 / E_n
        weights = 1.0 / effective_num
        
        # 归一化权重（可选，使平均权重 = 1）
        weights = weights / weights.mean()
        
        # 注册为 buffer（不参与梯度更新，但保存到模型）
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        
        self.num_classes = len(samples_per_class)
        self.samples_per_class = samples_per_class
    
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
            类别平衡交叉熵损失
        """
        # 使用 PyTorch 的 cross_entropy，传入权重
        loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weights,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )

        return loss

    @staticmethod
    def _compute_weights(class_counts: torch.Tensor, beta: float) -> torch.Tensor:
        """计算类别权重 (静态方法，兼容 ClassBalancedCE 接口)

        数学形式:
            E_c = (1 - β^{n_c}) / (1 - β)
            w_c = 1 / E_c

        Args:
            class_counts: [num_classes] 每个类别的样本数 (Tensor)
            beta: 有效样本数参数

        Returns:
            weights: [num_classes] 类别权重
        """
        class_counts = class_counts.float().clamp(min=1)

        # E_c = (1 - β^{n_c}) / (1 - β)
        effective_num = (1.0 - torch.pow(beta, class_counts)) / (1.0 - beta)

        # w_c = 1 / E_c
        weights = 1.0 / effective_num

        # 归一化
        weights = weights / weights.sum() * len(weights)

        return weights

    def get_weights(self) -> torch.Tensor:
        """
        获取类别权重
        
        返回
        ----
        weights : Tensor, shape [num_classes]
            类别权重向量
        """
        return self.weights.clone()
    
    def get_statistics(self) -> dict:
        """
        获取统计信息
        
        返回
        ----
        dict
            包含有效样本数、权重等统计信息
        """
        effective_num = (1.0 - np.power(self.beta, self.samples_per_class)) / (1.0 - self.beta)
        
        stats = {
            "num_classes": self.num_classes,
            "beta": self.beta,
            "samples_per_class": self.samples_per_class.tolist(),
            "effective_num": effective_num.tolist(),
            "weights": self.weights.cpu().numpy().tolist(),
        }
        
        # 头部/尾部统计
        sorted_indices = np.argsort(self.samples_per_class)
        head_classes = sorted_indices[-5:]  # 最多的 5 个类
        tail_classes = sorted_indices[:5]   # 最少的 5 个类
        
        stats["head_classes"] = {
            "indices": head_classes.tolist(),
            "samples": self.samples_per_class[head_classes].tolist(),
            "weights": self.weights[head_classes].cpu().numpy().tolist(),
        }
        
        stats["tail_classes"] = {
            "indices": tail_classes.tolist(),
            "samples": self.samples_per_class[tail_classes].tolist(),
            "weights": self.weights[tail_classes].cpu().numpy().tolist(),
        }
        
        return stats
    
    def extra_repr(self) -> str:
        """打印额外信息"""
        return (
            f"num_classes={self.num_classes}, beta={self.beta}, "
            f"reduction={self.reduction}, label_smoothing={self.label_smoothing}"
        )


def compute_class_weights_from_targets(
    targets: list | np.ndarray | torch.Tensor,
    beta: float = 0.9999,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """
    从目标标签自动计算类别权重
    
    工具函数，用于快速创建 ClassBalancedCrossEntropy
    
    参数
    ----
    targets : list, array, or tensor
        所有训练样本的标签
    beta : float, optional (default=0.9999)
        平衡参数
    num_classes : int, optional
        类别总数（默认自动推断）
        
    返回
    ----
    weights : Tensor, shape [num_classes]
        类别权重
        
    示例
    ----
    >>> targets = [0, 0, 0, 1, 1, 2]  # 3 个类
    >>> weights = compute_class_weights_from_targets(targets)
    >>> loss_fn = nn.CrossEntropyLoss(weight=weights)
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    targets = np.array(targets)
    
    # 统计每个类别的样本数
    if num_classes is None:
        num_classes = int(targets.max()) + 1
    
    samples_per_class = np.bincount(targets, minlength=num_classes)
    
    # 计算有效样本数
    effective_num = (1.0 - np.power(beta, samples_per_class)) / (1.0 - beta)
    weights = 1.0 / effective_num
    weights = weights / weights.mean()  # 归一化
    
    return torch.tensor(weights, dtype=torch.float32)
