"""Learning Rate Scheduler Module

Provides learning rate schedulers with warmup and decay.
Following the three-layer parameter principle, this handles
Layer 3 (hyperparameters) for learning rate scheduling.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any, TYPE_CHECKING
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


class FunctionalWarmupCosineScheduler:
    """Step-based LR scheduler with C⁰ + C¹ continuity

    Mathematically correct implementation that uses the passed step value
    directly, NOT internal state like len(lr_history).

    Mathematical forms:
    - Warmup (step < warmup_steps):
        lr(step) = min_lr + (base_lr - min_lr) * step / warmup_steps

    - Cosine decay (step >= warmup_steps):
        lr(step) = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(π * (step - warmup_steps) / (total_steps - warmup_steps)))

    The warmup→cosine transition is C¹ continuous:
    - At step=warmup_steps: lr = base_lr (both phases agree)
    - At step=warmup_steps: dlr/ds = (base_lr - min_lr) / warmup_steps (warmup derivative)
    - At step=warmup_steps: dlr/ds = (base_lr - min_lr) * (-π/2 / (total_steps - warmup_steps)) * (-sin(...)) = (base_lr - min_lr) / warmup_steps (cosine derivative)

    Args:
        optimizer: PyTorch optimizer
        base_lr: Maximum LR after warmup
        total_steps: Total training steps (epochs * steps_per_epoch)
        warmup_steps: Number of warmup steps
        min_lr: Minimum LR after decay
    """

    def __init__(
        self,
        optimizer: Any,
        base_lr: float = 5e-4,
        total_steps: int = 10000,
        warmup_steps: int = 500,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

        # Initialize all param groups to min_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = min_lr

    def get_lr_at_step(self, step: int) -> float:
        """Get LR at a specific step (pure function)

        Args:
            step: Global step counter

        Returns:
            Learning rate for this step
        """
        if step < self.warmup_steps:
            # Linear warmup phase
            return self.min_lr + (self.base_lr - self.min_lr) * (step / self.warmup_steps)

        # Cosine decay phase
        decay_steps = self.total_steps - self.warmup_steps
        curr_decay_step = step - self.warmup_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * curr_decay_step / decay_steps))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay

    def step(self, step: int) -> float:
        """Update LR based on global step

        Args:
            step: Global step counter (from TrainingState.global_step)

        Returns:
            The new learning rate
        """
        lr = self.get_lr_at_step(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing"""
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state from checkpoint"""
        self.base_lr = state_dict.get("base_lr", self.base_lr)
        self.total_steps = state_dict.get("total_steps", self.total_steps)
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        self.min_lr = state_dict.get("min_lr", self.min_lr)

    def get_last_lr(self) -> float:
        """Get last learning rate (approximation from base_lr)"""
        return self.base_lr


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
    total_steps: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
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
            - "functional_warmup_cosine": Step-based scheduler (recommended for batch-level stepping)
        total_epochs: Total training epochs
        base_lr: Base/peak learning rate
        min_lr: Minimum learning rate
        warmup_epochs: Warmup epochs
        warmup_start_lr: Starting LR during warmup
        total_steps: Total training steps (required for functional_warmup_cosine)
        steps_per_epoch: Steps per epoch (required for functional_warmup_cosine)

    Returns:
        LR scheduler instance
    """
    scheduler_type = scheduler_type.lower()

    if scheduler_type == "functional_warmup_cosine":
        if total_steps is None:
            if steps_per_epoch is None:
                raise ValueError("total_steps or steps_per_epoch required for functional_warmup_cosine")
            total_steps = total_epochs * steps_per_epoch
        warmup_steps = warmup_epochs * (steps_per_epoch or (total_steps // total_epochs))
        return FunctionalWarmupCosineScheduler(
            optimizer=optimizer,
            base_lr=base_lr,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr=min_lr,
        )
    elif scheduler_type == "warmup_cosine":
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
    "FunctionalWarmupCosineScheduler",
    "WarmupCosineScheduler",
    "LinearWarmupScheduler",
    "StepScheduler",
    "create_scheduler",
]
