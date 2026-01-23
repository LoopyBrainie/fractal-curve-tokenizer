# -*- coding: utf-8 -*-
"""
类别平衡采样器

数学形式化
============

Class-Balanced Sampling:
    采样概率: p(c) = (1/n_c^β) / Σ_j(1/n_j^β)
    
    其中:
        n_c: 类别 c 的样本数
        β ∈ [0, 1]: 平衡强度
            β = 0: 均匀采样 (忽略频率)
            β = 1: 完全逆频率采样
            
Progressive Sampling:
    β(t) = β_start · (1 - t/T) + β_end · (t/T)
    
    训练前期使用强平衡 (β_start=0.9)，后期减弱 (β_end=0.3)
    
参考文献
--------
1. Cui et al. "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
2. Lin et al. "Focal Loss for Dense Object Detection", ICCV 2017
"""

from __future__ import annotations

import math
from typing import Iterator, List

import numpy as np
import torch
from torch.utils.data import Sampler


class ClassBalancedSampler(Sampler):
    """
    类别平衡采样器
    
    根据类别频率的逆幂次进行加权采样，缓解数据不平衡问题。
    
    数学原理
    --------
    采样权重:
        w_i = (1 / n_{y_i}^β)
    
    采样概率:
        p_i = w_i / Σ_j w_j
        
    参数
    ----
    targets : List[int]
        每个样本的类别标签
    beta : float, optional (default=0.9)
        平衡强度 ∈ [0, 1]
        - 0: 均匀采样
        - 1: 完全逆频率
    replacement : bool, optional (default=True)
        是否有放回采样
        
    示例
    ----
    >>> targets = [0, 0, 0, 0, 1, 1, 2]  # 类别 0 占优
    >>> sampler = ClassBalancedSampler(targets, beta=0.9)
    >>> # 类别 2 的采样概率被提升
    """
    
    def __init__(
        self,
        targets: List[int],
        beta: float = 0.9,
        replacement: bool = True,
    ):
        if not 0 <= beta <= 1:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        
        self.targets = np.array(targets)
        self.beta = beta
        self.replacement = replacement
        self.num_samples = len(targets)
        
        # 计算类别频率
        self.classes, self.class_counts = np.unique(self.targets, return_counts=True)
        self.num_classes = len(self.classes)
        
        # 计算采样权重: w_c = 1 / n_c^β
        class_weights = 1.0 / np.power(self.class_counts, self.beta)
        
        # 为每个样本分配权重
        self.weights = np.zeros(self.num_samples, dtype=np.float64)
        for cls, cls_weight in zip(self.classes, class_weights):
            self.weights[self.targets == cls] = cls_weight
        
        # 归一化为概率
        self.weights /= self.weights.sum()
        
        # 统计信息
        self._compute_statistics()
    
    def _compute_statistics(self):
        """计算采样统计信息"""
        # 原始类别比例
        self.original_probs = self.class_counts / self.class_counts.sum()
        
        # 采样后类别比例（期望）
        self.sampled_probs = np.zeros(self.num_classes)
        for i, cls in enumerate(self.classes):
            self.sampled_probs[i] = self.weights[self.targets == cls].sum()
        
        # 频率提升比例
        self.boost_ratios = self.sampled_probs / (self.original_probs + 1e-10)
    
    def __iter__(self) -> Iterator[int]:
        """生成采样索引"""
        if self.replacement:
            # 有放回采样
            indices = np.random.choice(
                self.num_samples,
                size=self.num_samples,
                replace=True,
                p=self.weights
            )
        else:
            # 无放回采样（打乱）
            indices = np.argsort(np.random.random(self.num_samples))
        
        return iter(indices.tolist())
    
    def __len__(self) -> int:
        return self.num_samples
    
    def get_statistics(self) -> dict:
        """
        获取采样统计信息
        
        返回
        ----
        dict
            包含类别频率、采样概率、提升比例等统计信息
        """
        stats = {
            "num_classes": self.num_classes,
            "class_counts": self.class_counts.tolist(),
            "original_probs": self.original_probs.tolist(),
            "sampled_probs": self.sampled_probs.tolist(),
            "boost_ratios": self.boost_ratios.tolist(),
            "beta": self.beta,
        }
        
        # 添加头部/尾部统计
        sorted_indices = np.argsort(self.class_counts)
        head_classes = sorted_indices[-5:]  # 最多的 5 个类
        tail_classes = sorted_indices[:5]   # 最少的 5 个类
        
        stats["head_classes"] = {
            "indices": head_classes.tolist(),
            "counts": self.class_counts[head_classes].tolist(),
            "boost_ratios": self.boost_ratios[head_classes].tolist(),
        }
        
        stats["tail_classes"] = {
            "indices": tail_classes.tolist(),
            "counts": self.class_counts[tail_classes].tolist(),
            "boost_ratios": self.boost_ratios[tail_classes].tolist(),
        }
        
        return stats


class ProgressiveSampler(ClassBalancedSampler):
    """
    渐进式类别平衡采样器
    
    训练过程中动态调整平衡强度，前期强平衡（探索），后期弱平衡（收敛）。
    
    数学原理
    --------
    平衡强度调度:
        β(t) = β_start · (1 - t/T) + β_end · (t/T)
        
    其中:
        t: 当前 epoch
        T: 总 epochs
        
    参数
    ----
    targets : List[int]
        每个样本的类别标签
    beta_start : float, optional (default=0.9)
        初始平衡强度（强平衡）
    beta_end : float, optional (default=0.3)
        最终平衡强度（弱平衡）
    total_epochs : int, optional (default=100)
        总训练轮数
    current_epoch : int, optional (default=0)
        当前轮数（调用 set_epoch 更新）
        
    示例
    ----
    >>> sampler = ProgressiveSampler(targets, beta_start=0.9, beta_end=0.3, total_epochs=100)
    >>> for epoch in range(100):
    ...     sampler.set_epoch(epoch)
    ...     for batch in dataloader:
    ...         # 训练
    """
    
    def __init__(
        self,
        targets: List[int],
        beta_start: float = 0.9,
        beta_end: float = 0.3,
        total_epochs: int = 100,
        current_epoch: int = 0,
        replacement: bool = True,
    ):
        if not 0 <= beta_start <= 1:
            raise ValueError(f"beta_start must be in [0, 1], got {beta_start}")
        if not 0 <= beta_end <= 1:
            raise ValueError(f"beta_end must be in [0, 1], got {beta_end}")
        
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_epochs = total_epochs
        self.current_epoch = current_epoch
        
        # 计算当前 beta
        current_beta = self._compute_beta(current_epoch)
        
        # 初始化父类
        super().__init__(targets, beta=current_beta, replacement=replacement)
    
    def _compute_beta(self, epoch: int) -> float:
        """
        计算当前 epoch 的 beta 值
        
        使用线性插值: β(t) = β_start + (β_end - β_start) * (t / T)
        """
        if epoch >= self.total_epochs:
            return self.beta_end
        
        progress = epoch / self.total_epochs
        beta = self.beta_start + (self.beta_end - self.beta_start) * progress
        
        return beta
    
    def set_epoch(self, epoch: int):
        """
        更新当前 epoch 并重新计算权重
        
        参数
        ----
        epoch : int
            当前 epoch 编号
        """
        self.current_epoch = epoch
        new_beta = self._compute_beta(epoch)
        
        # 重新计算权重
        self.beta = new_beta
        
        # 重新计算类别权重
        class_weights = 1.0 / np.power(self.class_counts, self.beta)
        
        # 为每个样本分配权重
        self.weights = np.zeros(self.num_samples, dtype=np.float64)
        for cls, cls_weight in zip(self.classes, class_weights):
            self.weights[self.targets == cls] = cls_weight
        
        # 归一化
        self.weights /= self.weights.sum()
        
        # 更新统计信息
        self._compute_statistics()
    
    def get_schedule_info(self) -> dict:
        """
        获取调度信息
        
        返回
        ----
        dict
            包含 beta 调度曲线的信息
        """
        epochs = np.arange(self.total_epochs + 1)
        betas = [self._compute_beta(e) for e in epochs]
        
        return {
            "epochs": epochs.tolist(),
            "betas": betas,
            "current_epoch": self.current_epoch,
            "current_beta": self.beta,
        }
