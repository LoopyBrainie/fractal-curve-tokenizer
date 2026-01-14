#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
L1 分类结果可视化层 (Classification Results Visualization)

数学形式化
============

本模块可视化 L1ClassificationMetrics 的评估结果：

1. **混淆矩阵** (Confusion Matrix)
   归一化: $\tilde{M}_{ij} = M_{ij} / \sum_j M_{ij}$
   对角线 $\tilde{M}_{ii}$ = 类别 $i$ 的召回率

2. **每类准确率** (Per-Class Accuracy)
   $a_c = \text{TP}_c / n_c$，按准确率排序展示

3. **校准曲线** (Calibration Curve / Reliability Diagram)
   ECE = $\sum_{m=1}^M \frac{|B_m|}{N} |acc(B_m) - conf(B_m)|$
   完美校准时，曲线应贴合 y=x 对角线

4. **Top 混淆类别对** (Top Confused Pairs)
   展示最常混淆的类别对 $(c_i, c_j, \text{count})$

5. **难样本类别** (Hardest Classes)
   展示错误率最高的类别

Author: GitHub Copilot
Date: 2026-01-11
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .base import (
    FigureConfig,
    VisualizationLayer,
    VisualizationResult,
    safe_tight_layout,
    truncate_labels,
)

# 尝试导入 evaluation_layers 中的数据类
try:
    from ..evaluation_layers import L1ClassificationMetrics
except ImportError:
    L1ClassificationMetrics = None


class L1ClassificationVisualizer(VisualizationLayer):
    """L1 分类结果可视化器
    
    数学形式化：
        V₁: L1ClassificationMetrics → {混淆矩阵, 每类准确率, 校准曲线, ...}
    
    输入：
        L1ClassificationMetrics 包含:
        - top1_accuracy, top5_accuracy
        - per_class_accuracy: Dict[int, float]
        - ece, mce (校准误差)
        - top_confused_pairs: List[Tuple[int, int, int]]
        - hardest_classes: List[Tuple[int, float]]
    
    输出：
        VisualizationResult 包含多个 Figure
    """
    
    LAYER_NAME = "L1_Classification"
    LAYER_DESCRIPTION = "分类性能可视化：混淆矩阵、每类准确率、校准分析"
    
    def visualize(
        self,
        metrics: Any,  # L1ClassificationMetrics
        class_names: Optional[List[str]] = None,
        confusion_matrix: Optional[np.ndarray] = None,
        calibration_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> VisualizationResult:
        """生成 L1 分类可视化
        
        Args:
            metrics: L1ClassificationMetrics 评估结果
            class_names: 类别名称列表
            confusion_matrix: 混淆矩阵 [C, C]（可选，用于详细可视化）
            calibration_data: 校准数据（可选）
                - bin_confidences: List[float]
                - bin_accuracies: List[float]
                - bin_counts: List[int]
            **kwargs: 额外参数
            
        Returns:
            VisualizationResult
        """
        result = VisualizationResult(layer_name=self.LAYER_NAME)
        
        # 1. 每类准确率
        if metrics.per_class_accuracy:
            fig = self._plot_per_class_accuracy(
                metrics.per_class_accuracy,
                class_names,
                metrics.top1_accuracy,
            )
            result.figures.append(fig)
            result.names.append("L1_per_class_accuracy")
            result.descriptions.append("每类准确率分布（按准确率排序）")
        
        # 2. 混淆矩阵
        if confusion_matrix is not None:
            fig = self._plot_confusion_matrix(
                confusion_matrix,
                class_names,
            )
            result.figures.append(fig)
            result.names.append("L1_confusion_matrix")
            result.descriptions.append("归一化混淆矩阵")
        
        # 3. 校准曲线
        if calibration_data is not None:
            fig = self._plot_calibration_curve(
                calibration_data,
                metrics.ece,
                metrics.mce,
            )
            result.figures.append(fig)
            result.names.append("L1_calibration_curve")
            result.descriptions.append("模型校准曲线与ECE")
        
        # 4. Top 混淆类别对
        if metrics.top_confused_pairs:
            fig = self._plot_top_confused_pairs(
                metrics.top_confused_pairs,
                class_names,
            )
            result.figures.append(fig)
            result.names.append("L1_confused_pairs")
            result.descriptions.append("最常混淆的类别对")
        
        # 5. 难样本类别
        if metrics.hardest_classes:
            fig = self._plot_hardest_classes(
                metrics.hardest_classes,
                class_names,
            )
            result.figures.append(fig)
            result.names.append("L1_hardest_classes")
            result.descriptions.append("错误率最高的类别")
        
        # 6. 综合摘要
        fig = self._plot_summary(metrics)
        result.figures.append(fig)
        result.names.append("L1_summary")
        result.descriptions.append("L1 分类性能摘要")
        
        return result
    
    def _plot_per_class_accuracy(
        self,
        per_class_acc: Dict[int, float],
        class_names: Optional[List[str]],
        overall_acc: float,
    ) -> Figure:
        """绘制每类准确率条形图
        
        数学形式：
            对每个类别 c，绘制 a_c = TP_c / n_c
            按 a_c 升序排列，便于识别弱类
        """
        # 排序
        classes = list(per_class_acc.keys())
        accuracies = [per_class_acc[c] for c in classes]
        sorted_indices = np.argsort(accuracies)
        
        sorted_classes = [classes[i] for i in sorted_indices]
        sorted_accs = [accuracies[i] for i in sorted_indices]
        
        n_classes = len(classes)
        
        # 如果类别太多（>50），只显示最差和最好的类
        show_top_bottom = n_classes > 50
        if show_top_bottom:
            n_show = 25  # 显示最差和最好各 25 个
            bottom_classes = sorted_classes[:n_show]
            bottom_accs = sorted_accs[:n_show]
            top_classes = sorted_classes[-n_show:]
            top_accs = sorted_accs[-n_show:]
            
            display_classes = bottom_classes + ['...'] + top_classes
            display_accs = bottom_accs + [np.nan] + top_accs
            sorted_classes = bottom_classes + top_classes
            sorted_accs = bottom_accs + top_accs
        else:
            display_classes = sorted_classes
            display_accs = sorted_accs
        
        # 标签
        if class_names:
            labels = []
            for c in display_classes:
                if c == '...':
                    labels.append(f'... ({n_classes - 50} classes omitted) ...')
                elif isinstance(c, int) and c < len(class_names):
                    labels.append(class_names[c])
                else:
                    labels.append(f"C{c}" if isinstance(c, int) else str(c))
        else:
            labels = []
            for c in display_classes:
                if c == '...':
                    labels.append(f'... ({n_classes - 50} classes omitted) ...')
                else:
                    labels.append(f"Class {c}")
        
        # 截断长标签
        labels = truncate_labels(labels, max_len=20)
        
        # 创建图表 - 限制最大高度
        n_display = len(display_accs)
        fig_height = min(20, max(6, n_display * 0.25))
        fig, ax = plt.subplots(figsize=(12, fig_height))
        
        # 颜色映射：低准确率红色，高准确率绿色
        cmap = plt.get_cmap("RdYlGn")
        # 过滤 NaN 值用于颜色映射
        valid_accs = [a for a in display_accs if not (isinstance(a, float) and np.isnan(a))]
        colors = []
        for a in display_accs:
            if isinstance(a, float) and np.isnan(a):
                colors.append('lightgray')
            else:
                colors.append(cmap(a))
        
        # 绘制水平条形图
        y_pos = np.arange(n_display)
        bar_accs = [0 if (isinstance(a, float) and np.isnan(a)) else a for a in display_accs]
        bars = ax.barh(y_pos, bar_accs, color=colors, edgecolor='white', linewidth=0.5)
        
        # 标签
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=max(6, 10 - n_display // 30))
        ax.set_xlabel("Accuracy", fontsize=11)
        ax.set_xlim(0, 1.1)
        
        # 参考线
        ax.axvline(overall_acc, color='blue', linestyle='--', linewidth=2, 
                   label=f'Overall: {overall_acc:.1%}')
        ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5, label='50%')
        
        # 数值标注
        for bar, acc in zip(bars, display_accs):
            if isinstance(acc, float) and np.isnan(acc):
                continue
            width = bar.get_width()
            color = 'red' if acc < 0.3 else 'black'
            ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                   f'{acc:.1%}', va='center', fontsize=max(5, 8 - n_display // 50),
                   color=color)
        
        # 统计信息
        zero_acc_count = sum(1 for a in valid_accs if a < 0.01)
        stats_text = (
            f"Classes: {n_classes}\n"
            f"Mean: {np.mean(valid_accs):.1%}\n"
            f"Std: {np.std(valid_accs):.2%}\n"
            f"Min: {min(valid_accs):.1%}\n"
            f"Zero (< 1%): {zero_acc_count}"
        )
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
               fontsize=9, va='bottom', ha='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.legend(loc='lower right')
        title = "Per-Class Accuracy Distribution"
        if show_top_bottom:
            title += f" (Top/Bottom {n_show} of {n_classes})"
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        safe_tight_layout()
        return fig
    
    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[List[str]],
    ) -> Figure:
        """绘制归一化混淆矩阵
        
        数学形式：归一化矩阵 M_norm[i,j] = M[i,j] / sum_j(M[i,j])
        使用行归一化，对角线表示召回率
        """
        n_classes_total = cm.shape[0]
        
        # 如果类别过多，采样显示
        max_display = 50
        if n_classes_total > max_display:
            # 均匀采样
            sample_indices = np.linspace(0, n_classes_total - 1, max_display, dtype=int)
            cm_display = cm[np.ix_(sample_indices, sample_indices)]
            n_classes = max_display
            sampled = True
        else:
            sample_indices = np.arange(n_classes_total)
            cm_display = cm
            n_classes = n_classes_total
            sampled = False
        
        # 行归一化
        row_sums = cm_display.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除零
        cm_normalized = cm_display / row_sums
        
        # 标签
        if class_names:
            labels = [class_names[sample_indices[i]] if sample_indices[i] < len(class_names) else f"C{sample_indices[i]}" 
                     for i in range(n_classes)]
        else:
            labels = [f"C{sample_indices[i]}" for i in range(n_classes)]
        labels = truncate_labels(labels, max_len=10)
        
        # 图表尺寸
        fig_size = max(8, min(n_classes * 0.4, 15))  # 限制最大尺寸
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        # 热图
        im = ax.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)
        
        # 坐标轴
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=max(5, 9 - n_classes // 20))
        ax.set_yticklabels(labels, fontsize=max(5, 9 - n_classes // 20))
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        
        # 数值标注（仅当类别数较少时）
        if n_classes <= 30:
            for i in range(n_classes):
                for j in range(n_classes):
                    value = cm_normalized[i, j]
                    if value > 0.01:  # 仅显示 > 1%
                        color = 'white' if value > 0.5 else 'black'
                        ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                               fontsize=max(5, 8 - n_classes // 10), color=color)
        
        # 颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Recall', fontsize=10)
        
        # 对角线准确率统计（使用完整矩阵计算）
        cm_full_normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        diagonal_acc = np.diag(cm_full_normalized)
        mean_recall = np.mean(diagonal_acc)
        
        title = f"Confusion Matrix (Normalized)\nMean Recall: {mean_recall:.1%}"
        if sampled:
            title = f"Confusion Matrix (Sampled {n_classes} of {n_classes_total})\nMean Recall: {mean_recall:.1%}"
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        safe_tight_layout()
        return fig
    
    def _plot_calibration_curve(
        self,
        calibration_data: Dict[str, Any],
        ece: float,
        mce: float,
    ) -> Figure:
        """绘制校准曲线（可靠性图）
        
        数学形式：
            X 轴: 预测置信度 bin 中心
            Y 轴: 该 bin 内的实际准确率
            
            完美校准: y = x (对角线)
            ECE = Σ (|B_m|/N) |acc(B_m) - conf(B_m)|
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        bin_confs = np.array(calibration_data.get('bin_confidences', []))
        bin_accs = np.array(calibration_data.get('bin_accuracies', []))
        bin_counts = np.array(calibration_data.get('bin_counts', []))
        
        if len(bin_confs) == 0:
            ax.text(0.5, 0.5, "No calibration data available",
                   ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return fig
        
        # 对角线（完美校准）
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        
        # 校准曲线
        valid_mask = bin_counts > 0
        ax.plot(bin_confs[valid_mask], bin_accs[valid_mask], 
               'bo-', label='Model', linewidth=2, markersize=8)
        
        # 误差区域（gap）
        for conf, acc in zip(bin_confs[valid_mask], bin_accs[valid_mask]):
            ax.fill_betweenx([min(conf, acc), max(conf, acc)], 
                            conf - 0.02, conf + 0.02,
                            alpha=0.3, color='red')
        
        # 柱状图显示样本分布
        ax2 = ax.twinx()
        total_samples = bin_counts.sum()
        if total_samples > 0:
            bin_width = 0.08
            ax2.bar(bin_confs, bin_counts / total_samples, 
                   width=bin_width, alpha=0.3, color='gray', label='Sample Dist.')
            ax2.set_ylabel('Sample Proportion', fontsize=10, color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')
            ax2.set_ylim(0, 0.5)
        
        # 标签和标题
        ax.set_xlabel('Mean Predicted Confidence', fontsize=11)
        ax.set_ylabel('Fraction of Positives (Accuracy)', fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)
        
        # ECE 和 MCE 标注
        ax.text(0.95, 0.05, f"ECE: {ece:.3f}\nMCE: {mce:.3f}",
               transform=ax.transAxes, fontsize=11,
               va='bottom', ha='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title("Reliability Diagram (Calibration Curve)", 
                    fontsize=13, fontweight='bold')
        
        safe_tight_layout()
        return fig
    
    def _plot_top_confused_pairs(
        self,
        confused_pairs: List[Tuple[int, int, int]],
        class_names: Optional[List[str]],
    ) -> Figure:
        """绘制最常混淆的类别对
        
        显示被混淆次数最多的 (true_class, pred_class) 对
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 取前 15 个
        top_pairs = confused_pairs[:15]
        
        if not top_pairs:
            ax.text(0.5, 0.5, "No confusion pairs to display",
                   ha='center', va='center', fontsize=12)
            return fig
        
        # 准备数据
        labels = []
        counts = []
        for true_c, pred_c, count in top_pairs:
            if class_names:
                true_name = class_names[true_c] if true_c < len(class_names) else f"C{true_c}"
                pred_name = class_names[pred_c] if pred_c < len(class_names) else f"C{pred_c}"
            else:
                true_name, pred_name = f"C{true_c}", f"C{pred_c}"
            labels.append(f"{true_name[:12]} -> {pred_name[:12]}")
            counts.append(count)
        
        # 条形图
        y_pos = np.arange(len(labels))
        colors = plt.get_cmap('Reds')(np.linspace(0.3, 0.9, len(labels)))[::-1]
        
        bars = ax.barh(y_pos, counts, color=colors, edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Confusion Count", fontsize=11)
        ax.invert_yaxis()
        
        # 数值标注
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                   str(count), va='center', fontsize=9)
        
        ax.set_title("Top Confused Class Pairs (True -> Predicted)",
                    fontsize=13, fontweight='bold')
        
        safe_tight_layout()
        return fig
    
    def _plot_hardest_classes(
        self,
        hardest: List[Tuple[int, float]],
        class_names: Optional[List[str]],
    ) -> Figure:
        """绘制错误率最高的类别"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        top_hard = hardest[:15]
        
        if not top_hard:
            ax.text(0.5, 0.5, "No hardest classes to display",
                   ha='center', va='center', fontsize=12)
            return fig
        
        labels = []
        error_rates = []
        for class_id, err_rate in top_hard:
            if class_names:
                name = class_names[class_id] if class_id < len(class_names) else f"C{class_id}"
            else:
                name = f"Class {class_id}"
            labels.append(name[:20])
            error_rates.append(err_rate)
        
        y_pos = np.arange(len(labels))
        colors = plt.get_cmap('Reds')(np.linspace(0.4, 0.9, len(labels)))[::-1]
        
        bars = ax.barh(y_pos, error_rates, color=colors, edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Error Rate", fontsize=11)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        
        for bar, rate in zip(bars, error_rates):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                   f'{rate:.1%}', va='center', fontsize=9)
        
        ax.set_title("Hardest Classes (Highest Error Rate)",
                    fontsize=13, fontweight='bold')
        
        safe_tight_layout()
        return fig
    
    def _plot_summary(self, metrics: Any) -> Figure:
        """绘制 L1 分类性能摘要"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        
        # 1. 整体指标雷达图/条形图
        ax1 = axes[0]
        metric_names = ['Top-1', 'Top-5', 'MCA']
        values = [metrics.top1_accuracy, metrics.top5_accuracy, metrics.mean_class_accuracy]
        
        colors = ['#2ecc71', '#3498db', '#9b59b6']
        bars = ax1.bar(metric_names, values, color=colors, edgecolor='white', linewidth=2)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Accuracy", fontsize=11)
        
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{val:.1%}', ha='center', fontsize=11, fontweight='bold')
        
        ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title("Overall Metrics", fontsize=12, fontweight='bold')
        
        # 2. 校准误差
        ax2 = axes[1]
        calib_names = ['ECE', 'MCE']
        calib_values = [metrics.ece, metrics.mce]
        
        colors_calib = ['#e74c3c' if v > 0.1 else '#2ecc71' for v in calib_values]
        bars2 = ax2.bar(calib_names, calib_values, color=colors_calib, edgecolor='white')
        ax2.set_ylim(0, max(0.3, max(calib_values) * 1.2))
        ax2.set_ylabel("Error", fontsize=11)
        
        for bar, val in zip(bars2, calib_values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', fontsize=11)
        
        ax2.axhline(0.05, color='green', linestyle='--', alpha=0.5, label='Good (<5%)')
        ax2.axhline(0.15, color='red', linestyle='--', alpha=0.5, label='Bad (>15%)')
        ax2.legend(fontsize=8)
        ax2.set_title("Calibration Error", fontsize=12, fontweight='bold')
        
        # 3. Per-class 准确率分布直方图
        ax3 = axes[2]
        if metrics.per_class_accuracy:
            accs = list(metrics.per_class_accuracy.values())
            ax3.hist(accs, bins=20, color='#3498db', edgecolor='white', alpha=0.8)
            ax3.axvline(np.mean(accs), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(accs):.1%}')
            ax3.set_xlabel("Accuracy", fontsize=11)
            ax3.set_ylabel("Count", fontsize=11)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, "No per-class data", ha='center', va='center')
        
        ax3.set_title("Per-Class Accuracy Dist.", fontsize=12, fontweight='bold')
        
        fig.suptitle("L1 Classification Performance Summary", fontsize=14, fontweight='bold')
        safe_tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
        return fig
