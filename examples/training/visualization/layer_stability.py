#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
L6 训练稳定性可视化层 (Training Stability Visualization)

数学形式化
============

本模块可视化 L6StabilityMetrics 的评估结果：

1. **权重范数分布** (Weight Norm Distribution)
   Frobenius 范数: $\|W\|_F = \sqrt{\sum_{ij} w_{ij}^2}$
   各模块权重范数对比

2. **数值健康检查** (Numerical Health Check)
   NaN/Inf 检测: $\exists w_{ij} \in \{\text{NaN}, \pm\infty\}$
   梯度爆炸/消失检测

3. **GumbelTopKSplitter 健康分析** (Splitter Health)
   温度参数状态、logits 分布

4. **训练曲线分析** (Training Curves)
   Loss 曲线、过拟合检测、学习率调度

5. **权重变化追踪** (Weight Change Tracking)
   $\Delta W = \|W_t - W_{t-1}\|_F$
   检测训练是否停滞

Author: GitHub Copilot
Date: 2026-01-11
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from .base import (
    FigureConfig,
    VisualizationLayer,
    VisualizationResult,
    safe_tight_layout,
)

# 尝试导入 evaluation_layers 中的数据类
try:
    from ..evaluation_layers import L6StabilityMetrics
except ImportError:
    L6StabilityMetrics = None


class L6StabilityVisualizer(VisualizationLayer):
    """L6 训练稳定性可视化器
    
    数学形式化：
        V₆: L6StabilityMetrics → {权重范数, 健康检查, Splitter分析, ...}
    
    输入：
        L6StabilityMetrics 包含:
        - weight_norm_stats: Dict[str, Dict[str, float]]
        - gradient_health_score: float
        - has_nan_weights: bool
        - has_inf_weights: bool
        - splitter_health_score: float
    
    额外输入：
        - training_history: List[Dict] - 训练历史记录
    
    输出：
        VisualizationResult 包含多个 Figure
    """
    
    LAYER_NAME = "L6_Stability"
    LAYER_DESCRIPTION = "训练稳定性可视化：权重分析、数值健康、训练曲线"
    
    def visualize(
        self,
        metrics: Any,  # L6StabilityMetrics
        class_names: Optional[List[str]] = None,
        training_history: Optional[List[Dict]] = None,
        **kwargs,
    ) -> VisualizationResult:
        """生成 L6 稳定性可视化
        
        Args:
            metrics: L6StabilityMetrics 评估结果
            class_names: 类别名称列表（未使用）
            training_history: 训练历史记录（可选）
            **kwargs: 额外参数
            
        Returns:
            VisualizationResult
        """
        result = VisualizationResult(layer_name=self.LAYER_NAME)
        
        # 1. 权重范数分析
        if metrics.weight_norm_stats:
            fig = self._plot_weight_norms(metrics.weight_norm_stats)
            result.figures.append(fig)
            result.names.append("L6_weight_norms")
            result.descriptions.append("各模块权重范数分布")
        
        # 2. 数值健康检查
        fig = self._plot_health_check(metrics)
        result.figures.append(fig)
        result.names.append("L6_health_check")
        result.descriptions.append("数值稳定性健康检查")
        
        # 3. 训练曲线（如果有历史记录）
        if training_history:
            fig = self._plot_training_curves(training_history)
            result.figures.append(fig)
            result.names.append("L6_training_curves")
            result.descriptions.append("训练曲线与过拟合分析")
        
        # 4. 综合摘要
        fig = self._plot_summary(metrics)
        result.figures.append(fig)
        result.names.append("L6_summary")
        result.descriptions.append("L6 训练稳定性摘要")
        
        return result
    
    def _plot_weight_norms(self, weight_norm_stats: Dict[str, Dict[str, float]]) -> Figure:
        """绘制权重范数分析
        
        数学形式：每个模块的 Frobenius 范数 norm_F(W)
        以及统计量（mean, std, min, max）
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        modules = list(weight_norm_stats.keys())
        
        # 提取各统计量
        means = [weight_norm_stats[m].get('mean', 0) for m in modules]
        stds = [weight_norm_stats[m].get('std', 0) for m in modules]
        mins = [weight_norm_stats[m].get('min', 0) for m in modules]
        maxs = [weight_norm_stats[m].get('max', 0) for m in modules]
        
        # 截断长模块名
        short_names = [m[-25:] if len(m) > 25 else m for m in modules]
        
        # 1. 条形图 + 误差棒
        ax1 = axes[0]
        y_pos = np.arange(len(modules))
        
        bars = ax1.barh(y_pos, means, xerr=stds, color='#3498db',
                       edgecolor='white', capsize=3, alpha=0.8)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(short_names, fontsize=max(6, 10 - len(modules) // 10))
        ax1.set_xlabel("Weight Norm (Frobenius)", fontsize=11)
        ax1.set_title("Mean Weight Norm by Module", fontsize=12, fontweight='bold')
        
        # 2. Min-Max 范围图
        ax2 = axes[1]
        
        for i, (mod, mn, mx) in enumerate(zip(short_names, mins, maxs)):
            ax2.plot([mn, mx], [i, i], 'o-', color='#e74c3c', markersize=6, linewidth=2)
            ax2.scatter([means[i]], [i], s=100, c='#3498db', zorder=5, marker='s')
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(short_names, fontsize=max(6, 10 - len(modules) // 10))
        ax2.set_xlabel("Weight Norm Range", fontsize=11)
        ax2.legend(['Range (Min-Max)', 'Mean'], loc='lower right')
        ax2.set_title("Weight Norm Range by Module", fontsize=12, fontweight='bold')
        
        fig.suptitle("L6: Weight Norm Analysis", fontsize=14, fontweight='bold')
        safe_tight_layout()
        return fig
    
    def _plot_health_check(self, metrics: Any) -> Figure:
        """绘制数值健康检查仪表盘"""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # 健康检查项
        checks = [
            ("NaN Weights", not metrics.has_nan_weights),
            ("Inf Weights", not metrics.has_inf_weights),
            ("Gradient Health", metrics.gradient_health_score > 0.5),
        ]
        
        # 颜色
        colors = ['#2ecc71' if passed else '#e74c3c' for _, passed in checks]
        labels = ['PASS' if passed else 'FAIL' for _, passed in checks]
        
        for ax, (name, passed), color, label in zip(axes, checks, colors, labels):
            # 圆形指示器
            circle = Circle((0.5, 0.5), 0.35, color=color, alpha=0.8)
            ax.add_patch(circle)
            
            ax.text(0.5, 0.5, label, ha='center', va='center',
                   fontsize=14, fontweight='bold', color='white')
            ax.text(0.5, 0.05, name, ha='center', va='bottom',
                   fontsize=11, fontweight='bold')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.axis('off')
        
        fig.suptitle("L6: Numerical Health Check", fontsize=14, fontweight='bold')
        safe_tight_layout()
        return fig
    
    def _plot_training_curves(self, training_history: List[Dict]) -> Figure:
        """绘制训练曲线
        
        数学形式：
            Loss 曲线、准确率曲线、过拟合检测
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = list(range(1, len(training_history) + 1))
        
        # 提取数据
        train_loss = [h.get('train_loss', np.nan) for h in training_history]
        val_loss = [h.get('val_loss', np.nan) for h in training_history]
        train_acc = [h.get('train_acc', np.nan) for h in training_history]
        val_acc = [h.get('val_acc', np.nan) for h in training_history]
        lr = [h.get('lr', np.nan) for h in training_history]
        
        # 1. Loss 曲线
        ax1 = axes[0, 0]
        ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Loss", fontsize=11)
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_title("Training & Validation Loss", fontsize=12, fontweight='bold')
        
        # 2. Accuracy 曲线
        ax2 = axes[0, 1]
        ax2.plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2)
        ax2.plot(epochs, val_acc, 'r-', label='Val Acc', linewidth=2)
        
        # 标记最佳点
        if not np.all(np.isnan(val_acc)):
            best_idx = np.nanargmax(val_acc)
            ax2.scatter([epochs[best_idx]], [val_acc[best_idx]], 
                       color='green', s=100, zorder=5, marker='★',
                       label=f'Best: {val_acc[best_idx]:.2f}%')
        
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Accuracy (%)", fontsize=11)
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_title("Training & Validation Accuracy", fontsize=12, fontweight='bold')
        
        # 3. 学习率曲线
        ax3 = axes[1, 0]
        ax3.plot(epochs, lr, 'g-', linewidth=2)
        ax3.set_xlabel("Epoch", fontsize=11)
        ax3.set_ylabel("Learning Rate", fontsize=11)
        ax3.set_yscale('log')
        ax3.grid(alpha=0.3)
        ax3.set_title("Learning Rate Schedule", fontsize=12, fontweight='bold')
        
        # 4. 过拟合指标 (Val/Train Loss Ratio)
        ax4 = axes[1, 1]
        train_loss_arr = np.array(train_loss)
        val_loss_arr = np.array(val_loss)
        
        # 避免除零
        valid_mask = (train_loss_arr > 0) & ~np.isnan(train_loss_arr) & ~np.isnan(val_loss_arr)
        if valid_mask.any():
            loss_ratio = np.where(valid_mask, val_loss_arr / train_loss_arr, np.nan)
            ax4.plot(epochs, loss_ratio, 'm-', linewidth=2)
            
            # 参考线
            ax4.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='No Overfit')
            ax4.axhline(1.5, color='orange', linestyle='--', alpha=0.5, label='Mild Overfit')
            ax4.axhline(2.0, color='red', linestyle='--', alpha=0.5, label='Severe Overfit')
        
        ax4.set_xlabel("Epoch", fontsize=11)
        ax4.set_ylabel("Val/Train Loss Ratio", fontsize=11)
        ax4.legend(loc='upper left')
        ax4.grid(alpha=0.3)
        ax4.set_ylim(0, 3)
        ax4.set_title("Overfitting Indicator", fontsize=12, fontweight='bold')
        
        fig.suptitle("L6: Training Curves & Stability", fontsize=14, fontweight='bold')
        safe_tight_layout()
        return fig
    
    def _plot_summary(self, metrics: Any) -> Figure:
        """绘制 L6 稳定性摘要"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        # 健康状态
        nan_status = "✗ DETECTED" if metrics.has_nan_weights else "✓ Clean"
        inf_status = "✗ DETECTED" if metrics.has_inf_weights else "✓ Clean"
        grad_status = f"{metrics.gradient_health_score:.2f}" + \
                     (" ✓" if metrics.gradient_health_score > 0.5 else " ✗")
        splitter_status = f"{metrics.splitter_health_score:.2f}" + \
                         (" ✓" if metrics.splitter_health_score > 0.5 else " ✗")
        
        # 权重范数摘要
        if metrics.weight_norm_stats:
            all_means = [s.get('mean', 0) for s in metrics.weight_norm_stats.values()]
            norm_summary = f"Mean: {np.mean(all_means):.4f}, Range: [{min(all_means):.4f}, {max(all_means):.4f}]"
        else:
            norm_summary = "N/A"
        
        summary_text = f"""
┌──────────────────────────────────────────────────────┐
│             L6 Training Stability Summary             │
├──────────────────────────────────────────────────────┤
│  Numerical Health                                     │
│    • NaN Weights: {nan_status:>20}                  │
│    • Inf Weights: {inf_status:>20}                  │
│    • Gradient Health Score: {grad_status:>15}          │
├──────────────────────────────────────────────────────┤
│  Splitter Health                                      │
│    • Splitter Score: {splitter_status:>18}             │
├──────────────────────────────────────────────────────┤
│  Weight Statistics                                    │
│    • {norm_summary:<45} │
│    • Modules Analyzed: {len(metrics.weight_norm_stats):>5}                       │
└──────────────────────────────────────────────────────┘

Overall Status: {"✓ HEALTHY" if not metrics.has_nan_weights and not metrics.has_inf_weights and metrics.gradient_health_score > 0.5 else "⚠ NEEDS ATTENTION"}
        """
        
        # 背景色基于健康状态
        bg_color = 'lightgreen' if not metrics.has_nan_weights and not metrics.has_inf_weights else 'lightyellow'
        
        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
               fontsize=10, va='center', ha='center',
               family='monospace',
               bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.8))
        
        return fig
