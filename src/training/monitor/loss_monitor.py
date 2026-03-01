"""Loss Source Monitoring Module

Tracks detailed loss components and their sources during training.
Provides breakdown of total loss into individual components.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import torch


class LossMonitor:
    """Monitor loss components during training

    Tracks individual loss components for debugging and analysis.
    Useful for multi-task learning or models with auxiliary losses.

    Example:
        monitor = LossMonitor()

        # In training loop:
        total_loss = ce_loss + aux_loss * 0.5 + reg_loss * 0.1
        monitor.record({
            "cross_entropy": ce_loss.item(),
            "auxiliary": aux_loss.item(),
            "regularization": reg_loss.item(),
        })

    I150-3 ENHANCEMENT: 强制追踪所有 Loss 子项，包括权重为 0 的项
    """

    # 预定义的所有可能的损失项（即使权重为 0 也追踪）
    DEFAULT_LOSS_COMPONENTS = [
        "cross_entropy",
        "total",
        "budget_loss",
        "density_regularization",
        "elastic_budget",
        "sparsity_penalty",
        "semantic_loss",
        "diversity_loss",
        "reconstruction_loss",
        "consistency_loss",
        "entropy_loss",
        "quota_entropy",
        "jump_loss",
        "tree_constraint",
        "manifold_regularization",
    ]

    def __init__(self, force_track_all: bool = True):
        """Initialize LossMonitor

        Args:
            force_track_all: 如果为 True，初始化时创建所有默认损失项的容器
        """
        self.losses: Dict[str, List[float]] = {}
        self.step_losses: List[Dict[str, float]] = []
        self.force_track_all = force_track_all

        # I150-3: 预初始化所有可能的损失项，确保即使权重为 0 也能追踪
        if self.force_track_all:
            for name in self.DEFAULT_LOSS_COMPONENTS:
                self.losses[name] = []

    def record(self, loss_dict: Dict[str, float]) -> None:
        """Record loss components for one step

        Args:
            loss_dict: Dictionary of loss name -> loss value
        """
        # I150-3: 如果 force_track_all，确保所有默认项都被记录（即使值为0.0）
        if self.force_track_all:
            for name in self.DEFAULT_LOSS_COMPONENTS:
                if name not in loss_dict:
                    loss_dict[name] = 0.0

        self.step_losses.append(loss_dict)

        for name, value in loss_dict.items():
            if name not in self.losses:
                self.losses[name] = []
            self.losses[name].append(value)

    def get_average_components(self) -> Dict[str, float]:
        """Get average of each loss component

        Returns:
            Dictionary of loss name -> average loss
        """
        result = {}
        for name, values in self.losses.items():
            if values:
                result[name] = sum(values) / len(values)
        return result

    def get_total_loss(self) -> float:
        """Get sum of all loss components

        Returns:
            Total average loss
        """
        avg_components = self.get_average_components()
        return sum(avg_components.values())

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get detailed statistics for each loss component

        Returns:
            Dictionary of loss name -> {mean, std, min, max}
        """
        stats = {}
        for name, values in self.losses.items():
            if values:
                mean_val = sum(values) / len(values)
                std_val = (sum((x - mean_val)**2 for x in values) / len(values)) ** 0.5
                stats[name] = {
                    "mean": mean_val,
                    "std": std_val,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }
        return stats

    def get_step_history(self) -> List[Dict[str, float]]:
        """Get loss at each step

        Returns:
            List of loss dictionaries per step
        """
        return self.step_losses

    def get_component_contributions(self) -> Dict[str, float]:
        """Get percentage contribution of each component

        Returns:
            Dictionary of loss name -> percentage
        """
        total = self.get_total_loss()
        if total == 0:
            return {}

        avg_components = self.get_average_components()
        return {
            name: (value / total) * 100
            for name, value in avg_components.items()
        }

    def detect_loss_anomalies(self) -> Dict[str, List[int]]:
        """Detect anomalies in loss components

        Returns:
            Dictionary of issue type -> step indices
        """
        anomalies = {
            "nan": [],
            "inf": [],
            "exploding": [],  # > 1000
            "vanishing": [],  # < 1e-6
        }

        for step_idx, loss_dict in enumerate(self.step_losses):
            for name, value in loss_dict.items():
                if torch.isnan(torch.tensor(value)).any() if isinstance(value, (int, float)) else torch.isnan(value).any():
                    anomalies["nan"].append(step_idx)
                elif torch.isinf(torch.tensor(value)).any() if isinstance(value, (int, float)) else torch.isinf(value).any():
                    anomalies["inf"].append(step_idx)
                elif value > 1000:
                    anomalies["exploding"].append(step_idx)
                elif value < 1e-6:
                    anomalies["vanishing"].append(step_idx)

        return anomalies

    def reset(self) -> None:
        """Reset all recorded losses"""
        self.losses.clear()
        self.step_losses.clear()


class LossTracker:
    """Simple loss tracker with exponential moving average

    Tracks loss with EMA for smoother estimates.
    """

    def __init__(self, beta: float = 0.98):
        self.beta = beta
        self.value: Optional[float] = None
        self.count = 0

    def update(self, value: float) -> float:
        """Update with new value, return EMA

        Args:
            value: New loss value

        Returns:
            EMA of loss
        """
        self.count += 1
        if self.value is None:
            self.value = value
        else:
            self.value = self.beta * self.value + (1 - self.beta) * value
        return self.value

    @property
    def ema(self) -> Optional[float]:
        """Get current EMA value"""
        return self.value

    def reset(self) -> None:
        """Reset tracker"""
        self.value = None
        self.count = 0


class CombinedLossTracker:
    """Track multiple loss sources with individual EMAs

    Maintains separate EMA trackers for each loss component.
    """

    def __init__(self, components: List[str], beta: float = 0.98):
        self.trackers = {name: LossTracker(beta) for name in components}

    def update(self, losses: Dict[str, float]) -> Dict[str, float]:
        """Update all trackers

        Args:
            losses: Dictionary of loss name -> value

        Returns:
            EMA values for each component
        """
        result = {}
        for name, value in losses.items():
            if name in self.trackers:
                result[name] = self.trackers[name].update(value)
            else:
                # Create new tracker if not exists
                tracker = LossTracker()
                result[name] = tracker.update(value)
                self.trackers[name] = tracker
        return result

    def get_ema(self) -> Dict[str, float]:
        """Get all EMA values"""
        return {
            name: tracker.ema
            for name, tracker in self.trackers.items()
            if tracker.ema is not None
        }

    def reset(self) -> None:
        """Reset all trackers"""
        for tracker in self.trackers.values():
            tracker.reset()


__all__ = [
    "LossMonitor",
    "LossTracker",
    "CombinedLossTracker",
]
