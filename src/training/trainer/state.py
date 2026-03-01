"""Training State Module

Defines the training state machine for tracking all runtime state during training.
Enables seamless checkpoint/resume from any point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import torch


@dataclass
class TrainingState:
    """Training state machine

    Tracks all runtime state needed for training and checkpointing.
    This is Layer 2 (Variable) state that changes during training.

    Attributes:
        epoch: Current epoch (0-indexed)
        global_step: Total training steps
        best_metric: Best monitored metric value
        is_best: Whether current checkpoint is best
        optimizer_state: Optimizer state dict for resume
        scaler_state: GradScaler state dict for AMP resume
        scheduler_state: LR scheduler state dict
        sampler_state: Data sampler state (for distributed)
        metrics_history: History of all metrics
        warning_count: Count of numerical warnings
    """

    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0
    is_best: bool = False

    # State dicts for checkpoint resume
    optimizer_state: Optional[Dict[str, Any]] = None
    scaler_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    sampler_state: Optional[Dict[str, Any]] = None

    # Metrics history
    metrics_history: Dict[str, List[float]] = field(default_factory=dict)

    # Warning counters
    warning_count: int = 0
    nan_skip_count: int = 0

    def __post_init__(self):
        """Initialize default metrics history"""
        if not self.metrics_history:
            self.metrics_history = {
                "train_loss": [],
                "train_accuracy": [],
                "val_loss": [],
                "val_accuracy": [],
                "val_top5_accuracy": [],
                "learning_rate": [],
                "grad_norm": [],
            }

    def update_metric(self, name: str, value: float) -> None:
        """Update a metric in history"""
        if name not in self.metrics_history:
            self.metrics_history[name] = []
        self.metrics_history[name].append(value)

    def get_metric_history(self, name: str) -> list:
        """Get history for a specific metric"""
        return self.metrics_history.get(name, [])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for checkpointing"""
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "optimizer_state": self.optimizer_state,
            "scaler_state": self.scaler_state,
            "scheduler_state": self.scheduler_state,
            "sampler_state": self.sampler_state,
            "metrics_history": self.metrics_history,
            "warning_count": self.warning_count,
            "nan_skip_count": self.nan_skip_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TrainingState:
        """Create from dictionary (checkpoint loading)"""
        state = cls()
        state.epoch = d.get("epoch", 0)
        state.global_step = d.get("global_step", 0)
        state.best_metric = d.get("best_metric", 0.0)
        state.optimizer_state = d.get("optimizer_state")
        state.scaler_state = d.get("scaler_state")
        state.scheduler_state = d.get("scheduler_state")
        state.sampler_state = d.get("sampler_state")
        state.metrics_history = d.get("metrics_history", {})
        state.warning_count = d.get("warning_count", 0)
        state.nan_skip_count = d.get("nan_skip_count", 0)
        return state

    def increment_epoch(self) -> None:
        """Increment epoch counter"""
        self.epoch += 1

    def increment_step(self) -> None:
        """Increment global step counter"""
        self.global_step += 1

    def reset(self) -> None:
        """Reset state for new training run"""
        self.epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.is_best = False
        self.optimizer_state = None
        self.scaler_state = None
        self.scheduler_state = None
        self.sampler_state = None
        self.warning_count = 0
        self.nan_skip_count = 0
        # Keep metrics_history for reference


@dataclass
class EpochMetrics:
    """Metrics collected during one epoch

    This is the return type of train_one_epoch and evaluate functions.
    """

    # Loss
    loss: float = 0.0
    loss_components: Dict[str, float] = field(default_factory=dict)

    # Accuracy
    accuracy: float = 0.0
    top5_accuracy: Optional[float] = None

    # Token statistics (for Fractal ViT)
    avg_tokens: float = 0.0
    token_std: float = 0.0

    # Gradient statistics
    grad_norm: float = 0.0
    layer_grad_norms: Dict[str, float] = field(default_factory=dict)

    # Learning rate
    learning_rate: float = 0.0

    # Timing
    epoch_time: float = 0.0
    samples_per_second: float = 0.0

    # Numerical warnings
    nan_count: int = 0
    inf_count: int = 0
    skipped_steps: int = 0

    # Memory
    memory_allocated_mb: float = 0.0
    memory_reserved_mb: float = 0.0
    peak_memory_mb: float = 0.0

    # === 新增: 实验详细日志记录指标 ===

    # Splitter 统计
    splitter_logits_mean: float = 0.0
    splitter_logits_std: float = 0.0

    # 覆盖率
    active_ratio: float = 0.0

    # 流形统计
    manifold_bias_max: float = 0.0
    manifold_bias_min: float = 0.0
    manifold_bias_mean: float = 0.0
    manifold_bias_std: float = 0.0

    # Poincaré 距离统计
    poincare_dist_mean: float = 0.0
    poincare_dist_std: float = 0.0

    # 梯度比值
    backbone_grad_norm: float = 0.0
    splitter_grad_norm: float = 0.0
    backbone_vs_splitter_grad_ratio: float = 0.0

    # Bottleneck 层梯度
    entmax_grad_norm: float = 0.0
    manifold_decoder_grad_norm: float = 0.0

    # 损失项
    budget_penalty: float = 0.0
    consistency_loss: float = 0.0
    entropy_loss: float = 0.0

    # FLOPs 理论节省
    theoretical_flops_reduction: float = 0.0

    # I150-3 NEW: Splitter Logits 统计
    mean_abs_logits: float = 0.0
    budget_loss: float = 0.0  # I150-3 NEW: Elastic Budget 损失
    density_regularization: float = 0.0  # I150-3 NEW: 密度正则化损失

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging

        I150-3 ENHANCEMENT: 强制输出所有 Loss 子项，即使值为 0
        """
        result = {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "learning_rate": self.learning_rate,
            "grad_norm": self.grad_norm,
            "epoch_time": self.epoch_time,
            "samples_per_second": self.samples_per_second,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "skipped_steps": self.skipped_steps,
        }

        if self.top5_accuracy is not None:
            result["top5_accuracy"] = self.top5_accuracy
        if self.avg_tokens > 0:
            result["avg_tokens"] = self.avg_tokens
            result["token_std"] = self.token_std
        if self.layer_grad_norms:
            result["layer_grad_norms"] = self.layer_grad_norms

        # I150-3: 强制记录 loss_components（包括权重为 0 的项）
        if self.loss_components:
            result["loss_components"] = self.loss_components

        # 强制记录所有损失项（即使为 0）
        result["budget_loss"] = self.budget_loss
        result["density_regularization"] = self.density_regularization
        result["budget_penalty"] = self.budget_penalty
        result["consistency_loss"] = self.consistency_loss
        result["entropy_loss"] = self.entropy_loss

        # 新增: 实验详细日志指标
        if self.peak_memory_mb > 0:
            result["peak_memory_mb"] = self.peak_memory_mb

        # Splitter 统计（强制输出）
        result["splitter_logits_stats"] = {
            "mean": self.splitter_logits_mean,
            "std": self.splitter_logits_std,
        }
        result["mean_abs_logits"] = self.mean_abs_logits

        # 覆盖率
        result["active_ratio"] = self.active_ratio

        # 流形统计（强制输出）
        result["manifold_bias_stats"] = {
            "max": self.manifold_bias_max,
            "min": self.manifold_bias_min,
            "mean": self.manifold_bias_mean,
            "std": self.manifold_bias_std,
        }

        # Poincaré 距离统计（强制输出）
        result["poincare_dist_stats"] = {
            "mean": self.poincare_dist_mean,
            "std": self.poincare_dist_std,
        }

        # 梯度比值（强制输出）
        result["backbone_vs_splitter_grad_ratio"] = self.backbone_vs_splitter_grad_ratio
        result["backbone_grad_norm"] = self.backbone_grad_norm
        result["splitter_grad_norm"] = self.splitter_grad_norm

        # Bottleneck 层梯度（强制输出）
        result["bottleneck_layer_grad"] = {
            "entmax": self.entmax_grad_norm,
            "manifold_decoder": self.manifold_decoder_grad_norm,
        }

        # FLOPs 理论节省
        if self.theoretical_flops_reduction > 0:
            result["theoretical_flops_reduction"] = self.theoretical_flops_reduction

        return result


@dataclass
class EvaluationMetrics:
    """Metrics from evaluation"""

    loss: float
    accuracy: float
    top5_accuracy: Optional[float] = None
    ece: Optional[float] = None  # Expected Calibration Error
    per_class_accuracy: Optional[Dict[int, float]] = None
    confusion_matrix: Optional[torch.Tensor] = None
    num_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "num_samples": self.num_samples,
        }
        if self.top5_accuracy is not None:
            result["top5_accuracy"] = self.top5_accuracy
        if self.ece is not None:
            result["ece"] = self.ece
        if self.per_class_accuracy:
            result["per_class_accuracy"] = self.per_class_accuracy
        return result


__all__ = [
    "TrainingState",
    "EpochMetrics",
    "EvaluationMetrics",
]
