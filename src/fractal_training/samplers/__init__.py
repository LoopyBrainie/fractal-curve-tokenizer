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

from __future__ import annotations

from collections import Counter
from typing import Iterator, List, Optional, Sequence

import torch
from torch.utils.data import Sampler


class ClassBalancedSampler(Sampler[int]):
    """类别平衡采样器
    
    数学形式:
        P(sample i) = w_i / Σ_j w_j
        w_i = 1 / n_{c_i}^β
    
    其中:
        - c_i: 样本 i 的类别
        - n_{c_i}: 类别 c_i 的样本数
        - β: 平衡强度参数
    
    Args:
        labels: 每个样本的类别标签
        num_samples: 每个 epoch 采样数量，None 表示与数据集大小相同
        beta: 平衡强度，β ∈ [0, 1]
            - 0: 均匀采样
            - 1: 完全逆频率
            - 0.5: 开方平衡 (推荐)
        replacement: 是否有放回采样
        
    Example:
        >>> labels = [0, 0, 0, 1, 2, 2]  # 类别分布: {0: 3, 1: 1, 2: 2}
        >>> sampler = ClassBalancedSampler(labels, beta=1.0)
        >>> # 类别0权重: 1/3, 类别1权重: 1/1, 类别2权重: 1/2
        >>> # 期望采样概率: 类别1 最高，类别0 最低
    """
    
    def __init__(
        self,
        labels: Sequence[int],
        num_samples: Optional[int] = None,
        beta: float = 0.5,
        replacement: bool = True,
    ) -> None:
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        
        self.labels = list(labels)
        self.num_samples = num_samples if num_samples is not None else len(labels)
        self.beta = beta
        self.replacement = replacement
        
        # 计算每个类别的样本数
        self.class_counts = Counter(labels)
        self.num_classes = len(self.class_counts)
        
        # 计算采样权重
        self._compute_weights()
    
    def _compute_weights(self) -> None:
        """计算每个样本的采样权重
        
        数学: w_i = 1 / n_{c_i}^β
        """
        if self.beta == 0.0:
            # 均匀采样
            self.weights = torch.ones(len(self.labels), dtype=torch.float64)
        else:
            weights = []
            for label in self.labels:
                n_c = self.class_counts[label]
                w = 1.0 / (n_c ** self.beta)
                weights.append(w)
            self.weights = torch.tensor(weights, dtype=torch.float64)
        
        # 无需归一化，torch.multinomial 会自动处理
    
    def __iter__(self) -> Iterator[int]:
        """生成采样索引"""
        indices = torch.multinomial(
            self.weights,
            num_samples=self.num_samples,
            replacement=self.replacement,
        )
        return iter(indices.tolist())
    
    def __len__(self) -> int:
        return self.num_samples
    
    def get_class_sampling_probs(self) -> torch.Tensor:
        """返回每个类别的期望采样概率 (用于验证)
        
        Returns:
            [num_classes] 张量，每个元素是对应类别的采样概率
        """
        # 计算每个类别的总权重
        class_total_weights = torch.zeros(self.num_classes, dtype=torch.float64)
        for i, label in enumerate(self.labels):
            class_total_weights[label] += self.weights[i]
        
        # 归一化
        total_weight = self.weights.sum()
        return (class_total_weights / total_weight).float()


class ProgressiveSampler(Sampler[int]):
    """渐进式类别平衡采样器
    
    数学形式:
        β(t) = β_max - (β_max - β_min) · (t / T)
    
    训练过程:
        - 初期 β 高 → 强平衡，加速尾部类别学习
        - 后期 β 低 → 弱平衡，恢复原始分布比例
    
    Args:
        labels: 每个样本的类别标签
        num_samples: 每个 epoch 采样数量
        beta_max: 初始平衡强度 (训练开始)
        beta_min: 最终平衡强度 (训练结束)
        total_epochs: 总训练轮数
        replacement: 是否有放回采样
    """
    
    def __init__(
        self,
        labels: Sequence[int],
        num_samples: Optional[int] = None,
        beta_max: float = 0.9,
        beta_min: float = 0.3,
        total_epochs: int = 100,
        replacement: bool = True,
    ) -> None:
        self.labels = list(labels)
        self.num_samples = num_samples if num_samples is not None else len(labels)
        self.beta_max = beta_max
        self.beta_min = beta_min
        self.total_epochs = total_epochs
        self.replacement = replacement
        
        self.class_counts = Counter(labels)
        self.num_classes = len(self.class_counts)
        
        # 当前 epoch (通过 set_epoch 更新)
        self._current_epoch = 0
        self._current_beta = beta_max
        
        # 初始化权重
        self._recompute_weights()
    
    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch，更新采样权重
        
        Args:
            epoch: 当前 epoch 编号 (从 0 开始)
        """
        self._current_epoch = epoch
        
        # 计算当前 β
        progress = min(epoch / max(self.total_epochs - 1, 1), 1.0)
        self._current_beta = self.beta_max - (self.beta_max - self.beta_min) * progress
        
        self._recompute_weights()
    
    def _recompute_weights(self) -> None:
        """根据当前 β 重新计算权重"""
        beta = self._current_beta
        
        if beta == 0.0:
            self.weights = torch.ones(len(self.labels), dtype=torch.float64)
        else:
            weights = []
            for label in self.labels:
                n_c = self.class_counts[label]
                w = 1.0 / (n_c ** beta)
                weights.append(w)
            self.weights = torch.tensor(weights, dtype=torch.float64)
    
    def __iter__(self) -> Iterator[int]:
        indices = torch.multinomial(
            self.weights,
            num_samples=self.num_samples,
            replacement=self.replacement,
        )
        return iter(indices.tolist())
    
    def __len__(self) -> int:
        return self.num_samples
    
    @property
    def current_beta(self) -> float:
        """当前的平衡强度"""
        return self._current_beta


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
