r"""
Fractal Training Visualization Module
======================================

提供实验可追溯的可视化工具，用于分析训练过程和模型性能。

数学形式化
==========

**Per-Class Accuracy Heatmap**:
    可视化 $\{a_c\}_{c=1}^C$ 分布，其中 $a_c = \text{TP}_c / n_c$
    
    颜色映射: $c \mapsto \text{colormap}(a_c)$，红色表示低准确率，绿色表示高准确率

**Confusion Matrix**:
    归一化混淆矩阵 $\tilde{M}_{ij} = M_{ij} / \sum_j M_{ij}$
    
    对角线元素 $\tilde{M}_{ii}$ 表示类别 $i$ 的召回率

**Token Distribution**:
    经验分布 $\hat{P}(N=n) = \frac{\text{count}(N=n)}{B}$
    
    健康指标: 分布应在 $[N_{min}, N_{max}]$ 范围内有质量

**Depth Distribution**:
    深度占比 $p_d = N_d / \sum_{d'} N_{d'}$
    
    熵: $H = -\sum_d p_d \log p_d$（高熵表示多尺度利用均衡）

**Resource Curves**:
    FLOPS 随 epoch 变化: $F(t)$
    
    预算比: $r(t) = F(t) / F_{budget}$

设计原则
========
1. **独立于 W&B**: 可单独使用，也可集成到 WandBCallback
2. **自动保存**: 默认保存到 experiments/{run_id}/visualizations/
3. **自定义选项**: 所有参数可覆盖
4. **Seaborn + Matplotlib**: 美观且功能完整

使用示例
========
```python
from training.visualization import (
    plot_per_class_accuracy,
    plot_confusion_matrix,
    plot_token_distribution,
    plot_depth_distribution,
    plot_resource_curves,
    VisualizationConfig,
    ExperimentVisualizer,
)

# 独立使用
fig = plot_per_class_accuracy(per_class_acc, class_names=class_names)
fig.savefig("accuracy_heatmap.png")

# 自动化使用
visualizer = ExperimentVisualizer(
    save_dir="experiments/run_001/visualizations",
    config=VisualizationConfig(dpi=150, format="png"),
)
visualizer.plot_all(metrics_result, resource_result, token_counts)
```
"""

from .class_metrics import (
    plot_per_class_accuracy,
    plot_confusion_matrix,
    plot_class_distribution,
    plot_head_tail_comparison,
)

from .splitter_analysis import (
    plot_token_distribution,
    plot_depth_distribution,
    plot_splitter_health_timeline,
    plot_token_depth_heatmap,
)

from .resource_curves import (
    plot_flops_curve,
    plot_memory_curve,
    plot_resource_budget_comparison,
    plot_training_curves,
)

from .config import (
    VisualizationConfig,
    ExperimentVisualizer,
    auto_save_figure,
)

__all__ = [
    # Class metrics
    "plot_per_class_accuracy",
    "plot_confusion_matrix",
    "plot_class_distribution",
    "plot_head_tail_comparison",
    # Splitter analysis
    "plot_token_distribution",
    "plot_depth_distribution",
    "plot_splitter_health_timeline",
    "plot_token_depth_heatmap",
    # Resource curves
    "plot_flops_curve",
    "plot_memory_curve",
    "plot_resource_budget_comparison",
    "plot_training_curves",
    # Config & Utils
    "VisualizationConfig",
    "ExperimentVisualizer",
    "auto_save_figure",
]
