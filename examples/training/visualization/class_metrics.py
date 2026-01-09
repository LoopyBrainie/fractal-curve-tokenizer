r"""
类别指标可视化模块
==================

提供分类任务的可视化工具。

数学形式化
==========

**Per-Class Accuracy Heatmap**:
    输入: $\{a_c\}_{c=1}^C$，其中 $a_c = \text{TP}_c / n_c$
    
    可视化: $C \times 1$ 热图，颜色 $\propto a_c$
    
    颜色映射: RdYlGn (红色=低, 绿色=高)

**Confusion Matrix**:
    原始矩阵: $M_{ij} = \text{count}(y=i, \hat{y}=j)$
    
    归一化: $\tilde{M}_{ij} = M_{ij} / \sum_j M_{ij}$ (行归一化)
    
    对角线: $\tilde{M}_{ii}$ = 类别 $i$ 的召回率

**Head/Tail Analysis**:
    头部类别: $\mathcal{H} = \{c : n_c > \text{median}(n)\}$
    
    尾部类别: $\mathcal{T} = \{c : n_c \leq \text{median}(n)\}$
    
    准确率差异: $\Delta = \bar{a}_{\mathcal{H}} - \bar{a}_{\mathcal{T}}$
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from .config import VisualizationConfig


def _ensure_seaborn():
    """确保 seaborn 可用"""
    try:
        import seaborn as sns
        return sns
    except ImportError:
        raise ImportError(
            "seaborn is required for visualization. "
            "Install with: pip install seaborn"
        )


def plot_per_class_accuracy(
    per_class_accuracy: Dict[int, float],
    class_names: Optional[List[str]] = None,
    figsize: tuple = (14, 10),
    config: Optional['VisualizationConfig'] = None,
    sort_by_accuracy: bool = True,
    show_values: bool = True,
    highlight_threshold: float = 0.1,
) -> Figure:
    r"""绘制 Per-Class Accuracy 热图
    
    数学形式化:
        $a_c = \text{TP}_c / n_c$ 映射到颜色空间
        
        颜色: RdYlGn 发散色图
        - 红色: $a_c < 0.3$ (低准确率)
        - 黄色: $a_c \approx 0.5$ (中等)
        - 绿色: $a_c > 0.7$ (高准确率)
    
    Args:
        per_class_accuracy: {class_id: accuracy} 字典
        class_names: 类别名称列表
        figsize: 图像尺寸
        config: 可视化配置
        sort_by_accuracy: 是否按准确率排序
        show_values: 是否显示数值
        highlight_threshold: 高亮低于此阈值的类别
        
    Returns:
        matplotlib Figure 对象
    """
    sns = _ensure_seaborn()
    
    # 准备数据
    classes = list(per_class_accuracy.keys())
    accuracies = [per_class_accuracy[c] for c in classes]
    
    if class_names is None:
        labels = [f"Class {c}" for c in classes]
    else:
        labels = [class_names[c] if c < len(class_names) else f"Class {c}" 
                  for c in classes]
    
    # 排序
    if sort_by_accuracy:
        sorted_indices = np.argsort(accuracies)
        classes = [classes[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
    
    # 创建图像
    num_classes = len(classes)
    
    # 自动调整图像高度
    if num_classes > 50:
        figsize = (figsize[0], max(figsize[1], num_classes * 0.15))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热图
    cmap = config.cmap_diverging if config else "RdYlGn"
    
    # 使用水平条形图更适合大量类别
    colors = plt.colormaps[cmap](np.array(accuracies))
    bars = ax.barh(range(num_classes), accuracies, color=colors)
    
    # 设置标签
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(labels, fontsize=max(6, 10 - num_classes // 50))
    
    # 高亮低准确率类别
    for i, acc in enumerate(accuracies):
        if acc < highlight_threshold:
            ax.get_yticklabels()[i].set_color('red')
            ax.get_yticklabels()[i].set_fontweight('bold')
    
    # 显示数值
    if show_values:
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            width = bar.get_width()
            ax.text(
                width + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{acc:.1%}',
                ha='left', va='center',
                fontsize=max(5, 8 - num_classes // 100),
            )
    
    # 设置标题和标签
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Per-Class Accuracy Distribution', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.15)
    
    # 添加参考线
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='50%')
    mean_acc = np.mean(accuracies)
    ax.axvline(x=mean_acc, color='blue', linestyle='-', alpha=0.7, 
               label=f'Mean: {mean_acc:.1%}')
    ax.legend(loc='lower right')
    
    # 添加统计信息
    stats_text = (
        f"Classes: {num_classes}\n"
        f"Mean: {mean_acc:.2%}\n"
        f"Min: {min(accuracies):.2%}\n"
        f"Max: {max(accuracies):.2%}\n"
        f"Zero: {sum(1 for a in accuracies if a == 0)}"
    )
    ax.text(
        0.98, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )
    
    return fig


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    figsize: tuple = (12, 10),
    config: Optional['VisualizationConfig'] = None,
    show_values: bool = True,
    value_threshold: float = 0.01,
) -> Figure:
    r"""绘制混淆矩阵热图
    
    数学形式化:
        归一化: $\tilde{M}_{ij} = M_{ij} / \sum_j M_{ij}$
        
        对角线元素 $\tilde{M}_{ii}$ = 类别 $i$ 的召回率
        
        非对角线: 表示混淆模式
    
    Args:
        confusion_matrix: [C, C] 混淆矩阵
        class_names: 类别名称列表
        normalize: 是否归一化 (行归一化)
        figsize: 图像尺寸
        config: 可视化配置
        show_values: 是否显示数值
        value_threshold: 低于此值不显示数值
        
    Returns:
        matplotlib Figure 对象
    """
    sns = _ensure_seaborn()
    
    cm = confusion_matrix.copy()
    num_classes = cm.shape[0]
    
    # 归一化
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除零
        cm = cm.astype(float) / row_sums
    
    # 自动调整图像尺寸
    if num_classes > 50:
        figsize = (max(figsize[0], num_classes * 0.2), 
                   max(figsize[1], num_classes * 0.2))
        show_values = False  # 类别太多时不显示数值
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热图
    cmap = config.cmap_sequential if config else "Blues"
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    
    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Recall' if normalize else 'Count', rotation=-90, va="bottom")
    
    # 设置刻度
    if class_names is not None and num_classes <= 50:
        tick_labels = class_names
    elif num_classes <= 50:
        tick_labels = [str(i) for i in range(num_classes)]
    else:
        # 太多类别时只显示部分刻度
        tick_positions = np.linspace(0, num_classes - 1, min(20, num_classes), dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        tick_labels = None
    
    if tick_labels is not None:
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(tick_labels, fontsize=8)
    
    # 显示数值
    if show_values and num_classes <= 30:
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        
        for i in range(num_classes):
            for j in range(num_classes):
                if cm[i, j] > value_threshold:
                    ax.text(
                        j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=7,
                    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''), 
                 fontsize=14, fontweight='bold')
    
    # 计算统计信息
    diagonal_sum = np.trace(cm) if not normalize else np.mean(np.diag(cm))
    
    return fig


def plot_class_distribution(
    class_counts: Dict[int, int],
    per_class_accuracy: Optional[Dict[int, float]] = None,
    class_names: Optional[List[str]] = None,
    figsize: tuple = (14, 6),
    config: Optional['VisualizationConfig'] = None,
) -> Figure:
    """绘制类别样本分布图
    
    可视化每个类别的样本数量，可选叠加准确率信息。
    
    Args:
        class_counts: {class_id: count} 字典
        per_class_accuracy: {class_id: accuracy} 字典 (可选)
        class_names: 类别名称列表
        figsize: 图像尺寸
        config: 可视化配置
        
    Returns:
        matplotlib Figure 对象
    """
    sns = _ensure_seaborn()
    
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # 绘制样本数量柱状图
    x = np.arange(len(classes))
    bars = ax1.bar(x, counts, alpha=0.7, color='steelblue', label='Sample Count')
    
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Sample Count', color='steelblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    # 设置 x 轴标签
    if len(classes) <= 30:
        if class_names is not None:
            ax1.set_xticks(x)
            ax1.set_xticklabels(
                [class_names[c] if c < len(class_names) else str(c) for c in classes],
                rotation=45, ha='right', fontsize=8,
            )
        else:
            ax1.set_xticks(x)
            ax1.set_xticklabels([str(c) for c in classes], rotation=45, ha='right')
    else:
        ax1.set_xticks([])
    
    # 叠加准确率曲线 (如果提供)
    if per_class_accuracy is not None:
        ax2 = ax1.twinx()
        accuracies = [per_class_accuracy.get(c, 0) for c in classes]
        
        ax2.plot(x, accuracies, 'ro-', alpha=0.7, label='Accuracy', markersize=3)
        ax2.set_ylabel('Accuracy', color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 1.05)
        
        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax1.set_title('Class Distribution and Accuracy', fontsize=14, fontweight='bold')
    
    # 添加统计信息
    median_count = np.median(counts)
    ax1.axhline(y=median_count, color='gray', linestyle='--', alpha=0.5, 
                label=f'Median: {median_count:.0f}')
    
    return fig


def plot_head_tail_comparison(
    per_class_accuracy: Dict[int, float],
    class_counts: Dict[int, int],
    figsize: tuple = (10, 6),
    config: Optional['VisualizationConfig'] = None,
) -> Figure:
    r"""绘制头部/尾部类别准确率对比
    
    数学形式化:
        头部: $\mathcal{H} = \{c : n_c > \text{median}(n)\}$
        尾部: $\mathcal{T} = \{c : n_c \leq \text{median}(n)\}$
        
        平均准确率:
        $\bar{a}_{\mathcal{H}} = \frac{1}{|\mathcal{H}|} \sum_{c \in \mathcal{H}} a_c$
        
        差异: $\Delta = \bar{a}_{\mathcal{H}} - \bar{a}_{\mathcal{T}}$
    
    Args:
        per_class_accuracy: {class_id: accuracy} 字典
        class_counts: {class_id: count} 字典
        figsize: 图像尺寸
        config: 可视化配置
        
    Returns:
        matplotlib Figure 对象
    """
    sns = _ensure_seaborn()
    
    # 计算中位数
    counts = list(class_counts.values())
    median_count = np.median(counts)
    
    # 划分头部/尾部
    head_acc = []
    tail_acc = []
    
    for c in per_class_accuracy:
        count = class_counts.get(c, 0)
        acc = per_class_accuracy[c]
        
        if count > median_count:
            head_acc.append(acc)
        else:
            tail_acc.append(acc)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 左图: 箱线图对比
    ax1 = axes[0]
    data = [head_acc, tail_acc]
    labels = [f'Head\n({len(head_acc)} classes)', f'Tail\n({len(tail_acc)} classes)']
    
    bp = ax1.boxplot(data, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy Distribution by Class Frequency', fontsize=12)
    ax1.set_ylim(-0.05, 1.05)
    
    # 添加均值标注
    head_mean = np.mean(head_acc) if head_acc else 0
    tail_mean = np.mean(tail_acc) if tail_acc else 0
    ax1.axhline(y=head_mean, color='green', linestyle='--', alpha=0.7, xmin=0.1, xmax=0.4)
    ax1.axhline(y=tail_mean, color='red', linestyle='--', alpha=0.7, xmin=0.6, xmax=0.9)
    
    # 右图: 散点图
    ax2 = axes[1]
    
    classes = list(per_class_accuracy.keys())
    accuracies = [per_class_accuracy[c] for c in classes]
    counts_list = [class_counts.get(c, 0) for c in classes]
    
    # 颜色区分头部/尾部
    colors = ['green' if c > median_count else 'red' for c in counts_list]
    
    ax2.scatter(counts_list, accuracies, c=colors, alpha=0.6, s=30)
    ax2.axvline(x=median_count, color='gray', linestyle='--', alpha=0.5, 
                label=f'Median: {median_count:.0f}')
    
    ax2.set_xlabel('Sample Count', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy vs Sample Count', fontsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    
    # 总标题
    delta = head_mean - tail_mean
    fig.suptitle(
        f'Head vs Tail Analysis: Δ = {delta:+.1%}',
        fontsize=14, fontweight='bold', y=1.02,
    )
    
    plt.tight_layout()
    
    return fig
