# -*- coding: utf-8 -*-
"""
Adaptive Focal Loss

数学形式化
============

AdaptiveFocalLossWrapper 在标准 Focal Loss 基础上增加自适应聚焦参数 γ：

    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

其中 γ 根据任务特征动态调整：

    γ = f(D)  # D ∈ {难度分布, 训练进度, 类别不平衡}

自适应模式
----------

1. difficulty 模式: γ ∝ 样本难度
   - 难度越高，γ 越大
   - 使用预测熵衡量难度: H = -Σ p_i log(p_i)

2. annealing 模式: γ ∝ 训练进度
   - 初期 γ 较低 (探索)
   - 末期 γ 较高 (利用)

3. combined 模式: 综合 difficulty 和 annealing
   - γ = γ_base × (1 + H_norm) × (1 + 0.5 × progress)

数学性质
--------

- γ ∈ [γ_min, γ_max] 有界
- 保留 Focal Loss 的所有数学性质
- 平滑过渡，无梯度突变

参考文献
--------

Lin et al. "Focal Loss for Dense Object Detection", ICCV 2017
Cui et al. "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .focal_loss import FocalLoss


class AdaptiveFocalLossWrapper(nn.Module):
    """任务自适应的 Focal Loss 包装器

    数学形式化
    ==========

    标准 Focal Loss:
        FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

    自适应 γ 计算:

        difficulty 模式:
            γ = γ_base × (1 + H / H_max)

        annealing 模式:
            γ = γ_min + (γ_max - γ_min) × (t / T)

        combined 模式:
            γ = γ_base × (1 + H / H_max) × (1 + 0.5 × t / T)

    其中:
        H = -Σ p_i log(p_i)  # 预测熵 (样本难度)
        H_max = log(C)        # 最大熵 (C = 类别数)
        t / T = 训练进度

    参数
    ----
    base_gamma : float, optional
        基础 γ 值，默认 2.0
    adaptive_mode : str, optional
        自适应模式:
        - "fixed": 固定 γ (退化为标准 FocalLoss)
        - "difficulty": 基于样本难度调整
        - "annealing": 基于训练进度退火
        - "combined": 综合 difficulty 和 annealing
    gamma_min : float, optional
        γ 最小值，默认 1.0
    gamma_max : float, optional
        γ 最大值，默认 5.0
    difficulty_window : int, optional
        难度统计滑动窗口大小，默认 100
    **focal_kwargs
        FocalLoss 的其他参数 (alpha, reduction, label_smoothing)

    示例
    ----
    >>> # difficulty 模式
    >>> loss_fn = AdaptiveFocalLossWrapper(
    ...     base_gamma=2.0,
    ...     adaptive_mode="difficulty",
    ...     alpha=0.25
    ... )
    >>> logits = torch.randn(10, 5)
    >>> targets = torch.randint(0, 5, (10,))
    >>> loss = loss_fn(logits, targets)  # γ 自动调整
    """

    def __init__(
        self,
        base_gamma: float = 2.0,
        adaptive_mode: str = "combined",
        gamma_min: float = 1.0,
        gamma_max: float = 5.0,
        difficulty_window: int = 100,
        **focal_kwargs,
    ):
        super().__init__()

        if gamma_min < 0:
            raise ValueError(f"gamma_min must be >= 0, got {gamma_min}")
        if gamma_max < gamma_min:
            raise ValueError(f"gamma_max must be >= gamma_min, got {gamma_max}")
        if adaptive_mode not in ["fixed", "difficulty", "annealing", "combined"]:
            raise ValueError(
                f"adaptive_mode must be one of 'fixed', 'difficulty', 'annealing', 'combined', "
                f"got {adaptive_mode}"
            )

        self.base_gamma = base_gamma
        self.adaptive_mode = adaptive_mode
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.difficulty_window = difficulty_window

        # 内部 FocalLoss (gamma=0，退化为 CE，然后由 forward 应用动态 gamma)
        self.focal_loss = FocalLoss(gamma=base_gamma, **focal_kwargs)

        # 难度统计 (滑动窗口) - 使用 buffer 以支持 torch.save/load
        self.register_buffer(
            'entropy_buffer', torch.zeros(difficulty_window)
        )
        self.register_buffer(
            'buffer_idx', torch.zeros(1, dtype=torch.long)
        )
        self.register_buffer(
            'filled_count', torch.zeros(1, dtype=torch.long)
        )

        # 当前 γ 值 (用于日志记录)
        self.register_buffer(
            'current_gamma', torch.tensor(base_gamma)
        )

        # 类别数缓存
        self._num_classes: Optional[int] = None

    def _compute_entropy(self, logits: torch.Tensor) -> float:
        """计算样本难度 (预测熵)

        数学形式:
            H(p) = -Σ_{c=1}^C p_c log(p_c)

        熵范围:
            0 ≤ H(p) ≤ log(C)

        其中:
            H = 0: 完全确定 (one-hot 预测)
            H = log(C): 完全不确定 (均匀分布)
        """
        p = F.softmax(logits, dim=-1)
        # 数值稳定性: log(p + eps)
        log_p = torch.log(p + 1e-8)
        entropy = -(p * log_p).sum(dim=-1).mean()
        return float(entropy)

    def _update_gamma(
        self,
        difficulty: float,
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
        num_classes: Optional[int] = None,
    ) -> float:
        """根据难度和进度更新 γ

        参数
        ----
        difficulty : float
            当前批次样本难度 (预测熵)
        epoch : int, optional
            当前 epoch
        total_epochs : int, optional
            总 epoch 数
        num_classes : int, optional
            类别数，用于计算最大熵

        返回
        ----
        float
            更新后的 γ 值
        """
        # 更新难度滑动窗口
        idx = int(self.buffer_idx.item()) % self.difficulty_window
        self.entropy_buffer[idx] = difficulty
        self.buffer_idx[0] = (idx + 1) % self.difficulty_window
        filled_count = int(self.filled_count.item()) + 1
        if filled_count > self.difficulty_window:
            filled_count = self.difficulty_window
        self.filled_count[0] = filled_count

        # 计算平滑后的难度
        filled = int(self.filled_count.item())
        if filled > 0:
            avg_difficulty = self.entropy_buffer[:filled].mean().item()
        else:
            avg_difficulty = difficulty

        # 计算 γ
        if self.adaptive_mode == "fixed":
            gamma = self.base_gamma

        elif self.adaptive_mode == "difficulty":
            # 基于难度: 难度越高，γ 越大
            if num_classes is not None:
                max_entropy = math.log(num_classes)
            else:
                max_entropy = 1.0  # 避免除零
            norm_entropy = avg_difficulty / max_entropy if max_entropy > 0 else 0.5
            gamma = self.base_gamma * (1.0 + norm_entropy)

        elif self.adaptive_mode == "annealing":
            # 基于进度: 退火
            if epoch is not None and total_epochs is not None and total_epochs > 0:
                progress = epoch / total_epochs
            else:
                progress = 0.5  # 默认中间值
            gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * progress

        elif self.adaptive_mode == "combined":
            # 综合模式
            # 难度因子
            if num_classes is not None:
                max_entropy = math.log(num_classes)
            else:
                max_entropy = 1.0
            norm_entropy = avg_difficulty / max_entropy if max_entropy > 0 else 0.5
            difficulty_factor = 1.0 + norm_entropy

            # 进度因子
            if epoch is not None and total_epochs is not None and total_epochs > 0:
                progress = epoch / total_epochs
            else:
                progress = 0.5
            annealing_factor = 1.0 + 0.5 * progress

            gamma = self.base_gamma * difficulty_factor * annealing_factor

        else:
            gamma = self.base_gamma

        # 钳制到有效范围
        gamma = float(
            torch.clip(torch.tensor(gamma), self.gamma_min, self.gamma_max)
        )

        # 更新当前 γ (使用 .item() 避免 0-dim tensor 索引问题)
        self.current_gamma = torch.tensor(gamma)

        return gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
    ) -> torch.Tensor:
        """前向传播

        参数
        ----
        logits : Tensor, shape [N, C]
            模型输出 logits (未经 softmax)
        targets : Tensor, shape [N]
            目标类别索引
        epoch : int, optional
            当前 epoch (用于 annealing 模式)
        total_epochs : int, optional
            总 epoch 数 (用于 annealing 模式)

        返回
        ----
        loss : Tensor
            Focal Loss 值
        """
        # 缓存类别数
        if self._num_classes is None:
            self._num_classes = logits.size(-1)

        # 计算难度
        difficulty = self._compute_entropy(logits)

        # 更新 γ
        if self.adaptive_mode != "fixed":
            gamma = self._update_gamma(
                difficulty, epoch, total_epochs, self._num_classes
            )
            # 临时修改内部 FocalLoss 的 gamma
            original_gamma = self.focal_loss.gamma
            self.focal_loss.gamma = gamma

        # 计算损失
        loss = self.focal_loss(logits, targets)

        # 恢复原始 γ (如果是 fixed 模式，不需要恢复)
        if self.adaptive_mode != "fixed":
            self.focal_loss.gamma = original_gamma

        return loss

    def get_current_gamma(self) -> float:
        """获取当前 γ 值"""
        if hasattr(self.current_gamma, 'item'):
            return float(self.current_gamma.item())
        return float(self.current_gamma)

    def get_average_difficulty(self) -> float:
        """获取平均样本难度"""
        filled = int(self.filled_count.item())
        if filled > 0:
            return float(self.entropy_buffer[:filled].mean().item())
        return 0.0

    def extra_repr(self) -> str:
        """打印额外信息"""
        return (
            f"base_gamma={self.base_gamma}, "
            f"adaptive_mode={self.adaptive_mode}, "
            f"gamma_range=[{self.gamma_min}, {self.gamma_max}], "
            f"current_gamma={self.get_current_gamma():.3f}"
        )


def create_adaptive_focal_loss(
    num_classes: int,
    base_gamma: float = 2.0,
    adaptive_mode: str = "combined",
    gamma_min: float = 1.0,
    gamma_max: float = 5.0,
    alpha: Optional[float | list | torch.Tensor] = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> AdaptiveFocalLossWrapper:
    """工厂函数: 创建自适应 Focal Loss

    参数
    ----
    num_classes : int
        类别数
    base_gamma : float, optional
        基础 γ 值
    adaptive_mode : str, optional
        自适应模式
    gamma_min : float, optional
        γ 最小值
    gamma_max : float, optional
        γ 最大值
    alpha : float, list, or tensor, optional
        类别平衡因子
    reduction : str, optional
        输出约简方式
    label_smoothing : float, optional
        标签平滑因子

    返回
    ----
    AdaptiveFocalLossWrapper
        配置好的损失函数
    """
    return AdaptiveFocalLossWrapper(
        base_gamma=base_gamma,
        adaptive_mode=adaptive_mode,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        alpha=alpha,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )


if __name__ == "__main__":
    # 演示代码
    print("Adaptive Focal Loss 演示")
    print("=" * 60)

    # 创建损失函数
    loss_fn = AdaptiveFocalLossWrapper(
        base_gamma=2.0,
        adaptive_mode="combined",
        gamma_min=1.0,
        gamma_max=5.0,
        alpha=0.25,
    )

    # 模拟训练
    logits = torch.randn(32, 10)  # batch=32, classes=10
    targets = torch.randint(0, 10, (32,))

    for epoch in range(1, 6):
        loss = loss_fn(logits, targets, epoch=epoch, total_epochs=5)
        gamma = loss_fn.get_current_gamma()
        difficulty = loss_fn.get_average_difficulty()
        print(
            f"Epoch {epoch}: loss={loss.item():.4f}, "
            f"γ={gamma:.3f}, difficulty={difficulty:.3f}"
        )

    print("\n不同模式对比:")
    for mode in ["fixed", "difficulty", "annealing", "combined"]:
        loss_fn = AdaptiveFocalLossWrapper(
            base_gamma=2.0,
            adaptive_mode=mode,
            gamma_min=1.0,
            gamma_max=5.0,
        )
        loss = loss_fn(logits, targets, epoch=3, total_epochs=5)
        print(f"  {mode}: γ={loss_fn.get_current_gamma():.3f}, loss={loss.item():.4f}")
