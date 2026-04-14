"""Learning Rate Scheduler Module

Provides learning rate schedulers with warmup and decay.
Following the three-layer parameter principle, this handles
Layer 3 (hyperparameters) for learning rate scheduling.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
import math
from dataclasses import dataclass


@dataclass
class LRSchedule:
    """Learning rate schedule specification

    Defines how LR changes over training.
    """
    warmup_epochs: int = 5
    warmup_start_lr: float = 1e-7
    base_lr: float = 5e-4
    min_lr: float = 1e-6
    decay_type: str = "cosine"  # "cosine", "linear", "step", "none"
    decay_epochs: Optional[List[int]] = None
    decay_mult: float = 0.1


class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine decay

    Mathematical forms:
    - Warmup (t <= warmup_epochs):
        lr(t) = warmup_start_lr + (base_lr - warmup_start_lr) * t / warmup_epochs

    - Cosine decay (t > warmup_epochs):
        lr(t) = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(π * (t - warmup_epochs) / (total_epochs - warmup_epochs)))

    V4: 支持按 param_group_name 为特定参数组设置 LR

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        warmup_start_lr: Starting LR during warmup
        base_lr: Maximum LR after warmup
        min_lr: Minimum LR after decay
        total_epochs: Total training epochs
        decay_epochs: Optional list of milestone epochs for step decay
        param_group_name: Optional name to target specific param group (via custom 'name' key)
    """

    def __init__(
        self,
        optimizer: Any,
        warmup_epochs: int = 5,
        warmup_start_lr: float = 1e-7,
        base_lr: float = 5e-4,
        min_lr: float = 1e-6,
        total_epochs: int = 100,
        decay_epochs: Optional[List[int]] = None,
        decay_mult: float = 0.1,
        param_group_name: Optional[str] = None,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.decay_epochs = decay_epochs or []
        self.decay_mult = decay_mult
        self.param_group_name = param_group_name

        # V4: 初始化时只设置目标参数组
        if param_group_name is None:
            self._set_lr(warmup_start_lr)
        else:
            self._set_lr(warmup_start_lr, param_group_name)

        # History
        self.lr_history: List[float] = []

    def _find_param_group(self, name: str) -> Optional[Dict[str, Any]]:
        """V4: 查找指定名称的参数组"""
        for param_group in self.optimizer.param_groups:
            if param_group.get("name") == name:
                return param_group
        return None

    def _set_lr(self, lr: float, param_group_name: Optional[str] = None):
        """Set LR for param groups

        V4: 如果 param_group_name 指定，则只更新该参数组
        """
        if param_group_name is not None:
            # V4: 只更新指定参数组
            target_group = self._find_param_group(param_group_name)
            if target_group is not None:
                target_group["lr"] = lr
        else:
            # 默认: 更新所有参数组
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

    def get_lr(self, epoch: int) -> float:
        """Get LR for given epoch

        Args:
            epoch: Current epoch (0-indexed)

        Returns:
            Learning rate for this epoch
        """
        # Warmup phase
        if epoch < self.warmup_epochs:
            # Linear warmup
            progress = epoch / max(self.warmup_epochs, 1)
            return self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * progress

        # Decay phase
        decay_epoch = epoch - self.warmup_epochs
        total_decay_epochs = self.total_epochs - self.warmup_epochs

        if self.decay_epochs:
            # Step decay
            step_mult = 1.0
            for milestone in self.decay_epochs:
                if epoch >= milestone:
                    step_mult *= self.decay_mult
            return max(self.base_lr * step_mult, self.min_lr)
        else:
            # Cosine decay
            if total_decay_epochs <= 0:
                return self.base_lr

            decay_progress = decay_epoch / total_decay_epochs
            cosine_factor = 0.5 * (1 + math.cos(math.pi * decay_progress))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine_factor

    def step(self, epoch: Optional[int] = None):
        """Update learning rate

        Args:
            epoch: Current epoch (auto-incremented if None)
        """
        if epoch is None:
            # Auto-increment
            epoch = len(self.lr_history)

        lr = self.get_lr(epoch)
        self._set_lr(lr)
        self.lr_history.append(lr)

        return lr

    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing"""
        return {
            "lr_history": self.lr_history,
            "warmup_epochs": self.warmup_epochs,
            "warmup_start_lr": self.warmup_start_lr,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "total_epochs": self.total_epochs,
            "decay_epochs": self.decay_epochs,
            "decay_mult": self.decay_mult,
            "param_group_name": self.param_group_name,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state from checkpoint"""
        self.lr_history = state_dict.get("lr_history", [])
        self.param_group_name = state_dict.get("param_group_name", None)
        # Restore current LR
        if self.lr_history:
            self._set_lr(self.lr_history[-1], self.param_group_name)

    def get_last_lr(self) -> float:
        """Get last learning rate"""
        return self.lr_history[-1] if self.lr_history else self.base_lr


class LinearWarmupScheduler:
    """Linear warmup then constant LR

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        warmup_start_lr: Starting LR
        base_lr: LR after warmup
    """

    def __init__(
        self,
        optimizer: Any,
        warmup_epochs: int = 5,
        warmup_start_lr: float = 1e-7,
        base_lr: float = 5e-4,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = base_lr

        self._set_lr(warmup_start_lr)
        self.lr_history: List[float] = []

    def _set_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = len(self.lr_history)

        if epoch < self.warmup_epochs:
            progress = epoch / max(self.warmup_epochs, 1)
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * progress
        else:
            lr = self.base_lr

        self._set_lr(lr)
        self.lr_history.append(lr)
        return lr

    def state_dict(self) -> Dict[str, Any]:
        return {"lr_history": self.lr_history}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.lr_history = state_dict.get("lr_history", [])
        if self.lr_history:
            self._set_lr(self.lr_history[-1])


class StepScheduler:
    """Step decay LR scheduler

    Args:
        optimizer: PyTorch optimizer
        step_size: Epochs between LR decay
        gamma: Multiplier for LR decay
        base_lr: Initial LR
    """

    def __init__(
        self,
        optimizer: Any,
        step_size: int = 30,
        gamma: float = 0.1,
        base_lr: float = 5e-4,
    ):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.base_lr = base_lr

        self._set_lr(base_lr)
        self.lr_history: List[float] = [base_lr]

    def _set_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = len(self.lr_history)

        # Calculate LR: base_lr * gamma^(epoch // step_size)
        decay_factor = self.gamma ** (epoch // self.step_size)
        lr = self.base_lr * decay_factor

        self._set_lr(lr)
        self.lr_history.append(lr)
        return lr

    def state_dict(self) -> Dict[str, Any]:
        return {"lr_history": self.lr_history}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.lr_history = state_dict.get("lr_history", [])
        if self.lr_history:
            self._set_lr(self.lr_history[-1])


def create_scheduler(
    optimizer: Any,
    scheduler_type: str = "warmup_cosine",
    total_epochs: int = 100,
    base_lr: float = 5e-4,
    min_lr: float = 1e-6,
    warmup_epochs: int = 5,
    warmup_start_lr: float = 1e-7,
    **kwargs,
) -> Any:
    """Factory function to create LR scheduler

    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler
            - "warmup_cosine": Linear warmup + cosine decay
            - "warmup_linear": Linear warmup + constant
            - "step": Step decay
            - "cosine": Cosine decay only (no warmup)
            - "none": Constant LR
        total_epochs: Total training epochs
        base_lr: Base/peak learning rate
        min_lr: Minimum learning rate
        warmup_epochs: Warmup epochs
        warmup_start_lr: Starting LR during warmup

    Returns:
        LR scheduler instance
    """
    scheduler_type = scheduler_type.lower()

    if scheduler_type == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            warmup_start_lr=warmup_start_lr,
            base_lr=base_lr,
            min_lr=min_lr,
            total_epochs=total_epochs,
            decay_epochs=kwargs.get("decay_epochs"),
            decay_mult=kwargs.get("decay_mult", 0.1),
        )
    elif scheduler_type == "warmup_linear":
        return LinearWarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            warmup_start_lr=warmup_start_lr,
            base_lr=base_lr,
        )
    elif scheduler_type == "step":
        return StepScheduler(
            optimizer=optimizer,
            step_size=kwargs.get("step_size", 30),
            gamma=kwargs.get("gamma", 0.1),
            base_lr=base_lr,
        )
    elif scheduler_type == "cosine":
        return WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_epochs=0,
            warmup_start_lr=base_lr,
            base_lr=base_lr,
            min_lr=min_lr,
            total_epochs=total_epochs,
        )
    elif scheduler_type == "none":
        # Return dummy scheduler that doesn't change LR
        return WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_epochs=total_epochs,
            warmup_start_lr=base_lr,
            base_lr=base_lr,
            min_lr=base_lr,
            total_epochs=total_epochs,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


__all__ = [
    "LRSchedule",
    "WarmupCosineScheduler",
    "LinearWarmupScheduler",
    "StepScheduler",
    "create_scheduler",
]
