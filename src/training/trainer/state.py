"""Training State Module

Defines the training state machine for tracking all runtime state during training.
Enables seamless checkpoint/resume from any point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from collections import deque
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
    splitter_scheduler_state: Optional[Dict[str, Any]] = None  # V4: Splitter 独立 LR scheduler
    sampler_state: Optional[Dict[str, Any]] = None
    # PR4: Paced Window (T5) 已被铲除, staging_state_dict + paced_window_state 字段随之删除。

    # Metrics history
    metrics_history: Dict[str, List[float]] = field(default_factory=dict)

    # Warning counters
    warning_count: int = 0
    nan_skip_count: int = 0

    # Early stopping
    patience_counter: int = 0

    def __post_init__(self):
        """Initialize default metrics history with bounded deque"""
        # I-OOM FIX: 使用 deque(maxlen=1000) 防止无限增长
        maxlen = 1000
        if not self.metrics_history:
            self.metrics_history = {
                name: deque(maxlen=maxlen)
                for name in [
                    "train_loss", "train_accuracy", "val_loss", "val_accuracy",
                    "val_top5_accuracy", "learning_rate", "grad_norm",
                ]
            }

    def update_metric(self, name: str, value: float) -> None:
        """Update a metric in history with bounded deque"""
        # I-OOM FIX: 使用 deque(maxlen=1000) 防止无限增长
        if name not in self.metrics_history:
            self.metrics_history[name] = deque(maxlen=1000)
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
            "splitter_scheduler_state": self.splitter_scheduler_state,  # V4
            "sampler_state": self.sampler_state,
            "metrics_history": self.metrics_history,
            "warning_count": self.warning_count,
            "nan_skip_count": self.nan_skip_count,
            "patience_counter": self.patience_counter,
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
        state.splitter_scheduler_state = d.get("splitter_scheduler_state")  # V4
        state.sampler_state = d.get("sampler_state")
        state.metrics_history = d.get("metrics_history", {})
        state.warning_count = d.get("warning_count", 0)
        state.nan_skip_count = d.get("nan_skip_count", 0)
        state.patience_counter = d.get("patience_counter", 0)
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
        self.patience_counter = 0
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

    # 梯度比值
    backbone_grad_norm: float = 0.0
    splitter_grad_norm: float = 0.0
    backbone_vs_splitter_grad_ratio: float = 0.0

    # FLOPs 理论节省
    theoretical_flops_reduction: float = 0.0

    # I150-3 NEW: Splitter Logits 统计
    mean_abs_logits: float = 0.0
    raw_budget_error: float = 0.0  # D162: 重命名 (原 budget_loss)
    density_regularization: float = 0.0  # I150-3 NEW: 密度正则化损失

    # Layer-packaged auxiliary outputs (flattened).
    # Keys follow the convention "train/{layer}/{metric}",
    # e.g. "train/splitter/entropy", "train/attn_0/geometric_bias_mean".
    auxiliary_flat_metrics: Dict[str, float] = field(default_factory=dict)

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
        result["raw_budget_error"] = self.raw_budget_error
        result["density_regularization"] = self.density_regularization

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

        # 梯度比值（强制输出）
        result["backbone_vs_splitter_grad_ratio"] = self.backbone_vs_splitter_grad_ratio
        result["backbone_grad_norm"] = self.backbone_grad_norm
        result["splitter_grad_norm"] = self.splitter_grad_norm

        # FLOPs 理论节省
        if self.theoretical_flops_reduction > 0:
            result["theoretical_flops_reduction"] = self.theoretical_flops_reduction

        # Merge layer-packaged diagnostics (flattened auxiliary_outputs).
        # Keys are slash-namespaced so they cannot collide with existing fields.
        if self.auxiliary_flat_metrics:
            result.update(self.auxiliary_flat_metrics)

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
        # I-AUDIT: 输出 confusion_matrix（如果已计算）
        if self.confusion_matrix is not None:
            result["confusion_matrix"] = self.confusion_matrix.tolist()
        return result


__all__ = [
    "TrainingState",
    "EpochMetrics",
    "EvaluationMetrics",
]
