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
    AuxiliaryLossTracker,
    train_one_epoch,
    train_one_epoch_simple,
    evaluate,
    evaluate_simple,
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
    GradientStatisticsTracker,
    LossMonitor,
    LossTracker,
    CombinedLossTracker,
    AnomalyDetectionContext,
    GradientValidator,
    NumericalDefender,
    check_tensor_numerical_health,
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
    MetricsTracker,
    compute_accuracy,
    compute_confusion_matrix,
    compute_per_class_accuracy,
    compute_ece,
    compute_nll,
    compute_brier_score,
    compute_all_metrics,
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
    "AuxiliaryLossTracker",
    "train_one_epoch",
    "train_one_epoch_simple",
    "evaluate",
    "evaluate_simple",
    "compute_ece_score",
    # Scheduler
    "LRSchedule",
    "WarmupCosineScheduler",
    "LinearWarmupScheduler",
    "StepScheduler",
    "create_scheduler",
    # Monitor
    "GradientMonitor",
    "GradientStatisticsTracker",
    "LossMonitor",
    "LossTracker",
    "CombinedLossTracker",
    "AnomalyDetectionContext",
    "GradientValidator",
    "NumericalDefender",
    "check_tensor_numerical_health",
    # Checkpoint
    "save_checkpoint",
    "save_epoch_stats",
    "load_checkpoint",
    "load_model_weights",
    "find_latest_checkpoint",
    "find_best_checkpoint",
    # Logging
    "EpochLogger",
    "MetricsTracker",
    "compute_accuracy",
    "compute_confusion_matrix",
    "compute_per_class_accuracy",
    "compute_ece",
    "compute_nll",
    "compute_brier_score",
    "compute_all_metrics",
    "MetricsComputer",
]
