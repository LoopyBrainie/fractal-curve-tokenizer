r"""
资源曲线可视化模块
==================

提供训练过程中资源使用的可视化工具。

数学形式化
==========

**FLOPS Curve**:
    $F(t)$: 第 $t$ 个 epoch 的 FLOPS
    
    预算比: $r(t) = F(t) / F_{budget}$
    
    违规检测: $\mathbb{1}[r(t) > 1]$

**Memory Curve**:
    $M(t)$: 第 $t$ 个 epoch 的内存使用 (MB)
    
    峰值检测: $M_{peak} = \max_t M(t)$

**Training Curves**:
    损失曲线: $\mathcal{L}(t)$
    
    准确率曲线: $a(t)$
    
    收敛检测: $|\mathcal{L}(t) - \mathcal{L}(t-k)| < \epsilon$
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


def plot_flops_curve(
    flops_values: List[float],
    budget: Optional[float] = None,
    figsize: tuple = (12, 5),
    config: Optional['VisualizationConfig'] = None,
) -> Figure:
    r"""绘制 FLOPS 使用曲线
    
    数学形式化:
        预算比: $r(t) = F(t) / F_{budget}$
        
        违规: $\mathbb{1}[r(t) > 1]$
    
    Args:
        flops_values: 每个 epoch 的 FLOPS 值
        budget: FLOPS 预算
        figsize: 图像尺寸
        config: 可视化配置
        
    Returns:
        matplotlib Figure 对象
    """
    sns = _ensure_seaborn()
    
    epochs = list(range(len(flops_values)))
    flops_values = np.array(flops_values)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 左图: FLOPS 曲线
    ax1 = axes[0]
    ax1.plot(epochs, flops_values / 1e9, 'b-', linewidth=2, marker='o', markersize=3,
             label='FLOPS')
    
    if budget is not None:
        ax1.axhline(y=budget / 1e9, color='red', linestyle='--', linewidth=2,
                    label=f'Budget: {budget/1e9:.1f}G')
        
        # 标记违规点
        violations = [e for e, f in zip(epochs, flops_values) if f > budget]
        if violations:
            ax1.scatter(violations, [flops_values[e] / 1e9 for e in violations],
                        c='red', s=100, marker='x', zorder=5, label='Violations')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('FLOPS (G)', fontsize=12)
    ax1.set_title('FLOPS Usage Over Training', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 右图: 预算比率
    ax2 = axes[1]
    
    if budget is not None and budget > 0:
        ratios = flops_values / budget
        
        # 使用颜色区分违规
        colors = ['red' if r > 1 else 'green' for r in ratios]
        ax2.bar(epochs, ratios, color=colors, alpha=0.7, edgecolor='black')
        
        ax2.axhline(y=1.0, color='black', linestyle='-', linewidth=2, label='Budget')
        ax2.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Budget Ratio', fontsize=12)
        ax2.set_title('FLOPS Budget Utilization', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        
        # 统计信息
        avg_ratio = np.mean(ratios)
        max_ratio = np.max(ratios)
        violation_rate = np.mean(ratios > 1)
        
        stats_text = (
            f"Avg Ratio: {avg_ratio:.2f}\n"
            f"Max Ratio: {max_ratio:.2f}\n"
            f"Violations: {violation_rate:.1%}"
        )
        ax2.text(
            0.98, 0.98, stats_text,
            transform=ax2.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )
    else:
        ax2.text(0.5, 0.5, 'No budget specified', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=12)
    
    plt.tight_layout()
    
    return fig


def plot_memory_curve(
    memory_values: List[float],
    budget: Optional[float] = None,
    figsize: tuple = (10, 5),
    config: Optional['VisualizationConfig'] = None,
) -> Figure:
    """绘制内存使用曲线
    
    Args:
        memory_values: 每个 epoch 的内存使用 (MB)
        budget: 内存预算 (MB)
        figsize: 图像尺寸
        config: 可视化配置
        
    Returns:
        matplotlib Figure 对象
    """
    sns = _ensure_seaborn()
    
    epochs = list(range(len(memory_values)))
    memory_values = np.array(memory_values)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制内存曲线
    ax.fill_between(epochs, memory_values, alpha=0.3, color='purple')
    ax.plot(epochs, memory_values, 'purple', linewidth=2, marker='s', markersize=3,
            label='Memory Usage')
    
    # 标记峰值
    peak_idx = np.argmax(memory_values)
    peak_value = memory_values[peak_idx]
    ax.scatter([peak_idx], [peak_value], c='red', s=100, marker='*', zorder=5,
               label=f'Peak: {peak_value:.0f} MB')
    
    if budget is not None:
        ax.axhline(y=budget, color='red', linestyle='--', linewidth=2,
                   label=f'Budget: {budget:.0f} MB')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Memory (MB)', fontsize=12)
    ax.set_title('Memory Usage Over Training', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 统计信息
    stats_text = (
        f"Mean: {np.mean(memory_values):.0f} MB\n"
        f"Peak: {peak_value:.0f} MB\n"
        f"Min: {np.min(memory_values):.0f} MB"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )
    
    return fig


def plot_resource_budget_comparison(
    flops_values: List[float],
    memory_values: List[float],
    flops_budget: Optional[float] = None,
    memory_budget: Optional[float] = None,
    figsize: tuple = (12, 5),
    config: Optional['VisualizationConfig'] = None,
) -> Figure:
    """绘制资源预算对比图
    
    Args:
        flops_values: 每个 epoch 的 FLOPS 值
        memory_values: 每个 epoch 的内存使用 (MB)
        flops_budget: FLOPS 预算
        memory_budget: 内存预算 (MB)
        figsize: 图像尺寸
        config: 可视化配置
        
    Returns:
        matplotlib Figure 对象
    """
    sns = _ensure_seaborn()
    
    epochs = list(range(len(flops_values)))
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 左图: FLOPS 比率
    ax1 = axes[0]
    if flops_budget and flops_budget > 0:
        ratios = [f / flops_budget for f in flops_values]
        colors = ['red' if r > 1 else 'green' for r in ratios]
        ax1.bar(epochs, ratios, color=colors, alpha=0.7)
        ax1.axhline(y=1.0, color='black', linestyle='-', linewidth=2)
        ax1.set_ylabel('FLOPS / Budget')
        ax1.set_title('FLOPS Budget Utilization')
    else:
        ax1.bar(epochs, [f / 1e9 for f in flops_values], color='blue', alpha=0.7)
        ax1.set_ylabel('FLOPS (G)')
        ax1.set_title('FLOPS Usage')
    ax1.set_xlabel('Epoch')
    
    # 右图: Memory 比率
    ax2 = axes[1]
    if memory_budget and memory_budget > 0:
        ratios = [m / memory_budget for m in memory_values]
        colors = ['red' if r > 1 else 'purple' for r in ratios]
        ax2.bar(epochs, ratios, color=colors, alpha=0.7)
        ax2.axhline(y=1.0, color='black', linestyle='-', linewidth=2)
        ax2.set_ylabel('Memory / Budget')
        ax2.set_title('Memory Budget Utilization')
    else:
        ax2.bar(epochs, memory_values, color='purple', alpha=0.7)
        ax2.set_ylabel('Memory (MB)')
        ax2.set_title('Memory Usage')
    ax2.set_xlabel('Epoch')
    
    plt.tight_layout()
    
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    figsize: tuple = (14, 10),
    config: Optional['VisualizationConfig'] = None,
) -> Figure:
    """绘制训练曲线
    
    Args:
        history: 训练历史 {metric_name: values_list}
        figsize: 图像尺寸
        config: 可视化配置
        
    Returns:
        matplotlib Figure 对象
    """
    sns = _ensure_seaborn()
    
    # 分类指标
    loss_keys = [k for k in history.keys() if 'loss' in k.lower()]
    acc_keys = [k for k in history.keys() if 'acc' in k.lower() or 'accuracy' in k.lower()]
    lr_keys = [k for k in history.keys() if 'lr' in k.lower() or 'learning_rate' in k.lower()]
    other_keys = [k for k in history.keys() 
                  if k not in loss_keys + acc_keys + lr_keys]
    
    # 确定子图布局
    n_plots = sum([
        len(loss_keys) > 0,
        len(acc_keys) > 0,
        len(lr_keys) > 0,
        len(other_keys) > 0,
    ])
    
    if n_plots == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No training history data', ha='center', va='center')
        return fig
    
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    
    plot_idx = 0
    
    # 绘制 Loss 曲线
    if loss_keys:
        ax = axes[plot_idx]
        for key in loss_keys:
            values = history[key]
            epochs = list(range(len(values)))
            label = key.replace('_', ' ').title()
            ax.plot(epochs, values, linewidth=2, marker='o', markersize=2, label=label)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # 绘制 Accuracy 曲线
    if acc_keys:
        ax = axes[plot_idx]
        for key in acc_keys:
            values = history[key]
            epochs = list(range(len(values)))
            label = key.replace('_', ' ').title()
            ax.plot(epochs, values, linewidth=2, marker='s', markersize=2, label=label)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training Accuracy', fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        plot_idx += 1
    
    # 绘制 Learning Rate 曲线
    if lr_keys:
        ax = axes[plot_idx]
        for key in lr_keys:
            values = history[key]
            epochs = list(range(len(values)))
            label = key.replace('_', ' ').title()
            ax.plot(epochs, values, linewidth=2, marker='^', markersize=2, label=label)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule', fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # 绘制其他指标
    if other_keys:
        ax = axes[plot_idx]
        for key in other_keys[:5]:  # 最多显示 5 个
            values = history[key]
            epochs = list(range(len(values)))
            label = key.replace('_', ' ').title()
            ax.plot(epochs, values, linewidth=2, markersize=2, label=label)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.set_title('Other Metrics', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # 隐藏未使用的子图
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    return fig


def plot_convergence_analysis(
    train_loss: List[float],
    val_loss: Optional[List[float]] = None,
    window_size: int = 5,
    figsize: tuple = (12, 5),
    config: Optional['VisualizationConfig'] = None,
) -> Figure:
    """绘制收敛分析图
    
    分析训练是否收敛，检测过拟合。
    
    Args:
        train_loss: 训练损失历史
        val_loss: 验证损失历史 (可选)
        window_size: 滑动窗口大小
        figsize: 图像尺寸
        config: 可视化配置
        
    Returns:
        matplotlib Figure 对象
    """
    sns = _ensure_seaborn()
    
    epochs = list(range(len(train_loss)))
    train_loss = np.array(train_loss)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 左图: Loss 曲线
    ax1 = axes[0]
    ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss')
    
    if val_loss is not None:
        val_loss = np.array(val_loss)
        ax1.plot(epochs, val_loss, 'r-', linewidth=2, label='Val Loss')
        
        # 检测过拟合
        if len(val_loss) > window_size:
            # 滑动窗口平均
            val_smooth = np.convolve(val_loss, np.ones(window_size)/window_size, mode='valid')
            train_smooth = np.convolve(train_loss, np.ones(window_size)/window_size, mode='valid')
            
            # 过拟合点: val 开始上升而 train 继续下降
            gap = val_smooth - train_smooth
            if len(gap) > 1 and gap[-1] > gap[0] * 1.5:
                ax1.axvline(x=len(train_loss) - len(gap) // 2, color='orange', 
                            linestyle='--', label='Potential Overfit')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 右图: Loss 变化率
    ax2 = axes[1]
    
    if len(train_loss) > 1:
        loss_diff = np.diff(train_loss)
        ax2.bar(epochs[1:], loss_diff, color='steelblue', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # 标记收敛 (连续 N 个 epoch 变化率接近 0)
        converged_epochs = []
        threshold = 0.001
        for i in range(len(loss_diff)):
            if abs(loss_diff[i]) < threshold:
                converged_epochs.append(i + 1)
        
        if len(converged_epochs) >= window_size:
            first_converge = converged_epochs[0]
            ax2.axvline(x=first_converge, color='green', linestyle='--', 
                        label=f'Converged @ {first_converge}')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss Change')
    ax2.set_title('Loss Convergence Rate', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig
