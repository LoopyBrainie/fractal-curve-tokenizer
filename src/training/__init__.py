"""Fractal Training Module

Completely重构的训练模块，与模型架构解耦。

模块结构:
- config.py: 训练配置
- trainer/: 训练和评估函数
- scheduler/: 学习率调度器
- monitor/: 数值监控和防御
- checkpoint/: 检查点保存和加载
- logging/: 日志和指标
"""

# === F-X1 PR0 DeprecationWarning shim (trainer refactor) ===
# 4 公共符号将在 PR0 删除: train_one_epoch_simple, evaluate_simple,
# GradientStatisticsTracker, MetricsTracker. 临时 shim 发出 DeprecationWarning。
import warnings as _fractal_training_warnings
_FX1_DEPRECATED = (
    "train_one_epoch_simple / evaluate_simple / GradientStatisticsTracker / "
    "MetricsTracker are deprecated and removed in PR0 of the trainer refactor "
    "(see plan fluffy-watching-turing.md §3 PR0). Use the modern callback-based "
    "trainers in src/training/callbacks/ (PR2+) or the FractalCurveViT model "
    "directly via `from vit_pytorch import FractalCurveViT`."
)
_fractal_training_warnings.warn(_FX1_DEPRECATED, DeprecationWarning, stacklevel=2)

# Stubs (return None on access; will be removed entirely after one version)
class _FX1DeprecatedStub:
    """Placeholder for symbols removed in PR0. Emits DeprecationWarning on access."""
    def __init__(self, name: str):
        self._name = name
    def __getattr__(self, _attr):
        _fractal_training_warnings.warn(
            f"{self._name} removed in PR0; see trainer refactor plan.",
            DeprecationWarning, stacklevel=2,
        )
        return None
    def __call__(self, *_args, **_kwargs):
        _fractal_training_warnings.warn(
            f"{self._name} removed in PR0; returning None.",
            DeprecationWarning, stacklevel=2,
        )
        return None


def _fx1_deprecated_factory(name: str):
    def _stub(*_args, **_kwargs):
        _fractal_training_warnings.warn(
            f"{name} removed in PR0; returning None.",
            DeprecationWarning, stacklevel=2,
        )
        return None
    _stub.__name__ = name
    return _stub


train_one_epoch_simple = _fx1_deprecated_factory("train_one_epoch_simple")
evaluate_simple = _fx1_deprecated_factory("evaluate_simple")
# === End F-X1 shim ===

from .config import (
    Config,
    TrainingHyperparams,
    NumericalConfig,
    CheckpointConfig,
    DataConfig,
    MixedPrecisionConfig,
    create_config,
)

from .trainer import (
    TrainingState,
    EpochMetrics,
    EvaluationMetrics,
    MixupCutmixLoss,
    compute_loss,
    train_one_epoch,
    evaluate,
    compute_ece_score,
)

from .scheduler import (
    LRSchedule,
    WarmupCosineScheduler,
    LinearWarmupScheduler,
    StepScheduler,
    create_scheduler,
)

from .monitor import (
    GradientMonitor,
    LossMonitor,
    LossTracker,
    CombinedLossTracker,
)

# PR2: NaNGuard + NaNDumpCallback replace numerical_defense triad
# (NumericalDefender / GradientValidator / NaNAutoInvestigation 已删除)
from .callbacks import (
    NaNGuard,
    NaNDumpCallback,
    TrainerCallback,
    TrainerContext,
)

from .checkpoint import (
    save_checkpoint,
    save_epoch_stats,
    load_checkpoint,
    load_model_weights,
    find_latest_checkpoint,
    find_best_checkpoint,
)

from .training_logs import (
    EpochLogger,
    compute_accuracy,
    compute_confusion_matrix,
    compute_per_class_accuracy,
    compute_ece,
    MetricsComputer,
)

__all__ = [
    # Config
    "Config",
    "TrainingHyperparams",
    "NumericalConfig",
    "CheckpointConfig",
    "DataConfig",
    "MixedPrecisionConfig",
    "create_config",
    # Trainer
    "TrainingState",
    "EpochMetrics",
    "EvaluationMetrics",
    "MixupCutmixLoss",
    "compute_loss",
    "train_one_epoch",
    "evaluate",
    "compute_ece_score",
    # Scheduler
    "LRSchedule",
    "WarmupCosineScheduler",
    "LinearWarmupScheduler",
    "StepScheduler",
    "create_scheduler",
    # Monitor
    "GradientMonitor",
    "LossMonitor",
    "LossTracker",
    "CombinedLossTracker",
    # PR2 numerical defense (replaces numerical_defense triad)
    "NaNGuard",
    "NaNDumpCallback",
    "TrainerCallback",
    "TrainerContext",
    # Checkpoint
    "save_checkpoint",
    "save_epoch_stats",
    "load_checkpoint",
    "load_model_weights",
    "find_latest_checkpoint",
    "find_best_checkpoint",
    # Logging
    "EpochLogger",
    "compute_accuracy",
    "compute_confusion_matrix",
    "compute_per_class_accuracy",
    "compute_ece",
    "MetricsComputer",
    # F-X1 DeprecationWarning shim (PR0; will be removed in PR0.1)
    "train_one_epoch_simple",
    "evaluate_simple",
]
