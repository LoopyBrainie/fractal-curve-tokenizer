# -*- coding: utf-8 -*-
"""Fractal Training System - 模块化训练框架

设计原则
========
1. **模型无关性**: 训练器不依赖具体模型实现，通过接口协议交互
2. **组件解耦**: Sampler, Loss, Metrics 可独立测试和替换
3. **配置驱动**: 通过 dataclass 配置而非硬编码
4. **数学可验证**: 所有组件有明确的数学形式化定义

架构
====
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Sampler    │     │     Loss     │     │   Metrics    │
│ (数据采样)   │     │  (损失计算)   │     │  (评估指标)   │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       └────────────────────┼────────────────────┘
                           │
                    ┌──────▼───────┐
                    │   Trainer    │
                    │  (训练协调)   │
                    └──────────────┘
```

使用示例
========
```python
from examples.training import (
    ClassBalancedSampler,
    FocalLoss,
    ClassificationMetrics,
    TrainerConfig,
    ModularTrainer,
)

# 配置
config = TrainerConfig(
    sampler_beta=0.5,
    focal_gamma=2.0,
    focal_alpha=None,  # 自动计算
)

# 创建组件
sampler = ClassBalancedSampler(labels, beta=config.sampler_beta)
loss_fn = FocalLoss(gamma=config.focal_gamma)
metrics = ClassificationMetrics(num_classes=200)

# 训练
trainer = ModularTrainer(model, train_loader, val_loader, optimizer, loss_fn, metrics)
trainer.fit()
```
"""

# Samplers
from .samplers import (
    ClassBalancedSampler,
    ProgressiveSampler,
    compute_effective_sample_weights,
    get_class_counts,
)

# Losses
from .losses import (
    FocalLoss,
    ClassBalancedCE,
    ClassBalancedCrossEntropy,
    CompositeLoss,
    FocalClassBalancedLoss,
    ResourceAwareLoss,
    compute_depth_entropy,
    get_weighted_token_count,
    compute_class_weights_from_targets,
    # I30-2: Hilbert-aware 困难样本挖掘
    HilbertAwareHardMining,
    HilbertMiningWrapper,
    create_hilbert_mining_loss,
    # 细粒度分类损失 (CUB-200)
    CenterLoss,
    AttentionEntropyLoss,
    FinegrainedLoss,
    FinegrainedLossConfig,
    create_finegrained_loss,
)

# Metrics
from .metrics import ClassificationMetrics, ResourceMetrics

# Schedulers
from .schedulers import (
    FLOPSConfig,
    compute_transformer_flops,
    FLOPSBudgetLoss,
    DepthWeightedBudgetLoss,
    BudgetScheduler,
    estimate_baseline_flops,
)

# Trainer
from .trainer import (
    TrainerConfig,
    TrainerState,
    Callback,
    CallbackContext,
    CallbackList,
    EarlyStoppingCallback,
    CheckpointCallback,
    LRSchedulerCallback,
    ProgressCallback,
    ModularTrainer,
    # CUB-200 细粒度分类训练器
    CUB200Trainer,
    CUB200TrainingConfig,
    CUB200EvalResult,
    create_cub200_trainer,
    get_cub200_augmentation,
)

# Config
from .config import (
    DataConfig,
    ModelConfig,
    LossConfig,
    BudgetConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    WandBConfig,
    EvaluationConfig,
    ExperimentConfig,
    ConfigLoader,
    save_config,
    create_default_configs,
)

# Callbacks (W&B integration)
from .callbacks import (
    WandBCallbackConfig,
    WandBCallback,
    SplitterHealthConfig,
    SplitterHealthCallback,
    LayeredEvaluationCallbackConfig,
    LayeredEvaluationCallback,
)

# Evaluation
from .evaluation_layers import (
    L1ClassificationMetrics,
    L2TokenizerMetrics,
    L3AttentionMetrics,
    L4RepresentationMetrics,
    L5EfficiencyMetrics,
    L6StabilityMetrics,
    L7SplitterMetrics,
    L8GradientFlowMetrics,
    LayeredEvaluationReport,
    ClassificationEvaluator,
    TokenizerEvaluator,
    AttentionEvaluator,
    RepresentationEvaluator,
    EfficiencyEvaluator,
    StabilityEvaluator,
    SplitterEvaluator,
    GradientFlowEvaluator,
)

from .layered_evaluator import (
    LayeredEvaluator,
    create_evaluator,
)

# Visualization
from .visualization import (
    plot_per_class_accuracy,
    plot_confusion_matrix,
    plot_class_distribution,
    plot_head_tail_comparison,
    plot_token_distribution,
    plot_depth_distribution,
    plot_splitter_health_timeline,
    plot_token_depth_heatmap,
    plot_flops_curve,
    plot_memory_curve,
    plot_resource_budget_comparison,
    plot_training_curves,
    VisualizationConfig,
    ExperimentVisualizer,
    auto_save_figure,
)

# Core
from .core import ModelResourceStats, compute_resource_stats_batch

__all__ = [
    # Samplers
    "ClassBalancedSampler",
    "ProgressiveSampler",
    "compute_effective_sample_weights",
    "get_class_counts",
    # Losses
    "FocalLoss",
    "ClassBalancedCE",
    "ClassBalancedCrossEntropy",
    "CompositeLoss",
    "FocalClassBalancedLoss",
    "ResourceAwareLoss",
    "compute_depth_entropy",
    "get_weighted_token_count",
    "compute_class_weights_from_targets",
    # I30-2: Hilbert-aware 困难样本挖掘
    "HilbertAwareHardMining",
    "HilbertMiningWrapper",
    "create_hilbert_mining_loss",
    # 细粒度分类损失 (CUB-200)
    "CenterLoss",
    "AttentionEntropyLoss",
    "FinegrainedLoss",
    "FinegrainedLossConfig",
    "create_finegrained_loss",
    # Metrics
    "ClassificationMetrics",
    "ResourceMetrics",
    # Schedulers
    "FLOPSConfig",
    "compute_transformer_flops",
    "FLOPSBudgetLoss",
    "DepthWeightedBudgetLoss",
    "BudgetScheduler",
    "estimate_baseline_flops",
    # Trainer
    "TrainerConfig",
    "TrainerState",
    "Callback",
    "CallbackContext",
    "CallbackList",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "LRSchedulerCallback",
    "ProgressCallback",
    "ModularTrainer",
    # CUB-200 细粒度分类训练器
    "CUB200Trainer",
    "CUB200TrainingConfig",
    "CUB200EvalResult",
    "create_cub200_trainer",
    "get_cub200_augmentation",
    # Config
    "DataConfig",
    "ModelConfig",
    "LossConfig",
    "BudgetConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "WandBConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ConfigLoader",
    "save_config",
    "create_default_configs",
    # Callbacks (W&B integration)
    "WandBCallbackConfig",
    "WandBCallback",
    "SplitterHealthConfig",
    "SplitterHealthCallback",
    "LayeredEvaluationCallbackConfig",
    "LayeredEvaluationCallback",
    # Evaluation
    "L1ClassificationMetrics",
    "L2TokenizerMetrics",
    "L3AttentionMetrics",
    "L4RepresentationMetrics",
    "L5EfficiencyMetrics",
    "L6StabilityMetrics",
    "L7SplitterMetrics",
    "L8GradientFlowMetrics",
    "LayeredEvaluationReport",
    "ClassificationEvaluator",
    "TokenizerEvaluator",
    "AttentionEvaluator",
    "RepresentationEvaluator",
    "EfficiencyEvaluator",
    "StabilityEvaluator",
    "SplitterEvaluator",
    "GradientFlowEvaluator",
    "LayeredEvaluator",
    "create_evaluator",
    # Visualization
    "plot_per_class_accuracy",
    "plot_confusion_matrix",
    "plot_class_distribution",
    "plot_head_tail_comparison",
    "plot_token_distribution",
    "plot_depth_distribution",
    "plot_splitter_health_timeline",
    "plot_token_depth_heatmap",
    "plot_flops_curve",
    "plot_memory_curve",
    "plot_resource_budget_comparison",
    "plot_training_curves",
    "VisualizationConfig",
    "ExperimentVisualizer",
    "auto_save_figure",
    # Core
    "ModelResourceStats",
    "compute_resource_stats_batch",
]
