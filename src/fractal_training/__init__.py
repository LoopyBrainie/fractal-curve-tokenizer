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
from fractal_training import (
    ClassBalancedSampler,
    FocalLoss,
    ClassificationMetrics,
    TrainerConfig,
    FractalViTTrainer,
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
trainer = FractalViTTrainer(model, sampler, loss_fn, metrics, config)
trainer.fit(num_epochs=100)
```
"""

from .samplers import ClassBalancedSampler, ProgressiveSampler
from .losses import FocalLoss, ClassBalancedCE, CompositeLoss, FocalClassBalancedLoss
from .metrics import ClassificationMetrics, ResourceMetrics
from .schedulers import (
    FLOPSConfig,
    compute_transformer_flops,
    FLOPSBudgetLoss,
    DepthWeightedBudgetLoss,
    BudgetScheduler,
    estimate_baseline_flops,
)
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
)
from .config import (
    DataConfig,
    ModelConfig,
    LossConfig,
    BudgetConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    WandBConfig,
    ExperimentConfig,
    ConfigLoader,
    save_config,
    create_default_configs,
)
from .callbacks import (
    WandBCallbackConfig,
    WandBCallback,
    SplitterHealthConfig,
    SplitterHealthCallback,
)
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

__all__ = [
    # Samplers
    "ClassBalancedSampler",
    "ProgressiveSampler",
    # Losses
    "FocalLoss",
    "ClassBalancedCE",
    "CompositeLoss",
    "FocalClassBalancedLoss",
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
    # Config
    "DataConfig",
    "ModelConfig",
    "LossConfig",
    "BudgetConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "WandBConfig",
    "ExperimentConfig",
    "ConfigLoader",
    "save_config",
    "create_default_configs",
    # Callbacks (W&B integration)
    "WandBCallbackConfig",
    "WandBCallback",
    "SplitterHealthConfig",
    "SplitterHealthCallback",
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
]
