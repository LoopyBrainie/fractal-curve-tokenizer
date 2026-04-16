# -*- coding: utf-8 -*-
"""
综合指标仪表板

展示模型训练和评估的关键指标。
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def plot_metrics_dashboard(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: tuple = (16, 10)
):
    """
    绘制综合指标仪表板

    Args:
        metrics: 指标字典 {name: [values per epoch]}
        save_path: 保存路径 (可选)
        figsize: 图像大小
    """
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, n_metrics))

    for i, (name, values) in enumerate(metrics.items()):
        ax = axes[i]
        epochs = range(1, len(values) + 1)

        ax.plot(epochs, values, color=colors[i], linewidth=2, marker='o', markersize=4)
        ax.fill_between(epochs, values, alpha=0.3, color=colors[i])

        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(name, fontsize=10)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 标注最大值/最小值
        if 'loss' in name.lower():
            min_idx = np.argmin(values)
            ax.scatter([min_idx + 1], [values[min_idx]], c='red', s=100, zorder=5)
            ax.annotate(f'Min: {values[min_idx]:.4f}',
                       xy=(min_idx + 1, values[min_idx]),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, color='red')
        else:
            max_idx = np.argmax(values)
            ax.scatter([max_idx + 1], [values[max_idx]], c='green', s=100, zorder=5)
            ax.annotate(f'Max: {values[max_idx]:.4f}',
                       xy=(max_idx + 1, values[max_idx]),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, color='green')

    # 隐藏多余的子图
    for ax in axes[n_metrics:]:
        ax.axis('off')

    plt.suptitle('Model Metrics Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved metrics dashboard to {save_path}")

    plt.close(fig)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5)
):
    """
    绘制训练曲线

    Args:
        train_losses: 训练损失
        val_losses: 验证损失
        train_accs: 训练准确率 (可选)
        val_accs: 验证准确率 (可选)
        save_path: 保存路径 (可选)
        figsize: 图像大小
    """
    fig, axes = plt.subplots(1, 2 if train_accs else 1, figsize=figsize)

    epochs = range(1, len(train_losses) + 1)

    # =========================================================================
    # 损失曲线
    # =========================================================================
    ax = axes[0] if train_accs else axes
    ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4)
    ax.plot(epochs, val_losses, 'r-', linewidth=2, label='Val Loss', marker='s', markersize=4)
    ax.fill_between(epochs, train_losses, alpha=0.3, color='blue')
    ax.fill_between(epochs, val_losses, alpha=0.3, color='red')

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # 准确率曲线
    # =========================================================================
    if train_accs and val_accs:
        ax = axes[1]
        ax.plot(epochs, train_accs, 'b-', linewidth=2, label='Train Acc', marker='o', markersize=4)
        ax.plot(epochs, val_accs, 'r-', linewidth=2, label='Val Acc', marker='s', markersize=4)
        ax.fill_between(epochs, train_accs, alpha=0.3, color='blue')
        ax.fill_between(epochs, val_accs, alpha=0.3, color='red')

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    plt.close(fig)


def plot_performance_comparison(
    fractal_metrics: Dict[str, float],
    standard_metrics: Dict[str, float],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
):
    """
    绘制 Fractal ViT 与标准 ViT 的性能对比

    Args:
        fractal_metrics: Fractal ViT 指标
        standard_metrics: 标准 ViT 指标
        save_path: 保存路径 (可选)
        figsize: 图像大小
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 准备数据
    metrics = list(fractal_metrics.keys())
    fractal_values = list(fractal_metrics.values())
    standard_values = [standard_metrics.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, fractal_values, width, label='Fractal ViT',
                  color='green', edgecolor='black', alpha=0.7)
    bars2 = ax.bar(x + width/2, standard_values, width, label='Standard ViT',
                  color='blue', edgecolor='black', alpha=0.7)

    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标注
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved performance comparison to {save_path}")

    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Metrics Dashboard Demo")
    parser.add_argument("--save-path", type=str, default=None,
                       help="Path to save visualization")

    args = parser.parse_args()

    # 演示
    print("Metrics Dashboard Demo")
    print("Run with actual metrics for full visualization.")
