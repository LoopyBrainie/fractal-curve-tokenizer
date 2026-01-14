#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
L8 梯度流分析可视化层 (Gradient Flow Visualization)

数学形式化
============

本模块可视化 L8GradientFlowMetrics 的评估结果：

1. **组件梯度范数** (Component Gradient Norms)
   ||∇W_c||₂ = √(Σᵢ (∂L/∂wᵢ)²) for component c
   
2. **梯度健康检测** (Gradient Health)
   消失: ||∇W|| < ε_vanish (default 1e-7)
   爆炸: ||∇W|| > ε_explode (default 100)

3. **梯度流热图** (Gradient Flow Heatmap)
   按层展示梯度大小分布

4. **STE 梯度追踪** (Straight-Through Estimator)
   分割器 MLP → 阈值 → 配额 梯度链

Author: GitHub Copilot
Date: 2026-01-14
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .base import (
    FigureConfig,
    VisualizationLayer,
    VisualizationResult,
    safe_tight_layout,
)

try:
    from ..evaluation_layers import L8GradientFlowMetrics
except ImportError:
    L8GradientFlowMetrics = None


class L8GradientFlowVisualizer(VisualizationLayer):
    """L8 梯度流分析可视化器
    
    数学形式化：
        V₈: L8GradientFlowMetrics → {组件梯度, 健康检测, 梯度流热图, ...}
    """
    
    LAYER_NAME = "L8_GradientFlow"
    LAYER_DESCRIPTION = "梯度流分析可视化：组件梯度、消失/爆炸检测、STE追踪"
    
    def visualize(
        self,
        metrics: Any,  # L8GradientFlowMetrics
        history: Optional[Dict[str, List[float]]] = None,
        **kwargs,
    ) -> VisualizationResult:
        """生成 L8 梯度流可视化"""
        result = VisualizationResult(layer_name=self.LAYER_NAME)
        
        # 1. 组件梯度条形图
        if metrics.component_grad_norms:
            fig = self._plot_component_gradients(metrics)
            result.figures.append(fig)
            result.names.append("L8_component_gradients")
            result.descriptions.append("各组件梯度范数")
        
        # 2. 梯度健康仪表盘
        fig = self._plot_gradient_health(metrics)
        result.figures.append(fig)
        result.names.append("L8_gradient_health")
        result.descriptions.append("梯度健康状态检测")
        
        # 3. 关键组件梯度追踪
        fig = self._plot_key_components(metrics)
        result.figures.append(fig)
        result.names.append("L8_key_components")
        result.descriptions.append("关键组件梯度追踪")
        
        # 4. 问题检测报告
        if metrics.vanishing_gradients or metrics.exploding_gradients:
            fig = self._plot_problem_report(metrics)
            result.figures.append(fig)
            result.names.append("L8_problems")
            result.descriptions.append("梯度问题检测报告")
        
        # 5. 综合摘要
        fig = self._plot_summary_card(metrics)
        result.figures.append(fig)
        result.names.append("L8_summary")
        result.descriptions.append("L8 梯度流摘要")
        
        return result
    
    def _plot_component_gradients(self, metrics: Any) -> Figure:
        """组件梯度范数条形图"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 按梯度大小排序
        sorted_items = sorted(
            metrics.component_grad_norms.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]  # 只显示前 20 个
        
        names = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        # 颜色编码：正常=蓝，过小=红，过大=橙
        colors = []
        for v in values:
            if v < 1e-6:
                colors.append('red')
            elif v > 10:
                colors.append('orange')
            else:
                colors.append('steelblue')
        
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Gradient Norm ||grad W||")
        ax.set_title("Component Gradient Norms (Top 20)")
        ax.set_xscale('log')
        
        # 添加阈值线
        ax.axvline(x=1e-6, color='red', linestyle='--', alpha=0.7, label='Vanishing')
        ax.axvline(x=10, color='orange', linestyle='--', alpha=0.7, label='Exploding')
        ax.legend(loc='lower right')
        
        safe_tight_layout()
        self._add_watermark(fig)
        return fig
    
    def _plot_gradient_health(self, metrics: Any) -> Figure:
        """梯度健康仪表盘"""
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        
        # 左图：总梯度范数
        ax_total = axes[0]
        self._draw_gauge(
            ax_total,
            value=min(metrics.total_grad_norm, 100),  # 限制显示范围
            min_val=0,
            max_val=100,
            title="Total Grad Norm",
            unit="",
            thresholds=[(1, 'yellow'), (10, 'green'), (50, 'orange')],
        )
        ax_total.text(0.5, -0.15, f"Actual: {metrics.total_grad_norm:.4f}",
                     ha='center', transform=ax_total.transAxes, fontsize=10)
        
        # 中图：max/min 比值
        ax_ratio = axes[1]
        ratio = min(metrics.grad_norm_ratio, 1e6) if metrics.grad_norm_ratio > 0 else 1
        self._draw_gauge(
            ax_ratio,
            value=np.log10(ratio + 1),
            min_val=0,
            max_val=6,  # log10(1e6)
            title="Grad Ratio (log10)",
            unit="",
            thresholds=[(2, 'green'), (4, 'yellow'), (5, 'red')],
        )
        ax_ratio.text(0.5, -0.15, f"max/min = {metrics.grad_norm_ratio:.2e}",
                     ha='center', transform=ax_ratio.transAxes, fontsize=10)
        
        # 右图：问题统计
        ax_stats = axes[2]
        ax_stats.axis('off')
        
        n_vanish = len(metrics.vanishing_gradients)
        n_explode = len(metrics.exploding_gradients)
        
        if n_vanish == 0 and n_explode == 0:
            status = "[OK] Healthy"
            color = 'green'
        elif n_explode > 0:
            status = "[X] Exploding"
            color = 'red'
        else:
            status = "[!] Vanishing"
            color = 'orange'
        
        ax_stats.text(0.5, 0.7, status, ha='center', va='center',
                     fontsize=32, fontweight='bold', color=color,
                     transform=ax_stats.transAxes)
        ax_stats.text(0.5, 0.4, f"Vanishing: {n_vanish} params", ha='center',
                     fontsize=12, transform=ax_stats.transAxes,
                     color='orange' if n_vanish > 0 else 'gray')
        ax_stats.text(0.5, 0.25, f"Exploding: {n_explode} params", ha='center',
                     fontsize=12, transform=ax_stats.transAxes,
                     color='red' if n_explode > 0 else 'gray')
        
        safe_tight_layout()
        self._add_watermark(fig)
        return fig
    
    def _plot_key_components(self, metrics: Any) -> Figure:
        """关键组件梯度追踪"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 关键组件列表
        key_components = [
            ("Splitter MLP", metrics.splitter_mlp_grad_norm),
            ("Splitter Threshold", metrics.splitter_threshold_grad_norm),
            ("Splitter Quota", metrics.splitter_quota_grad_norm),
            ("Attention QKV", metrics.attention_qkv_grad_norm),
            ("LCA Embedding", metrics.lca_embed_grad_norm),
            ("Level Scale", metrics.level_scale_grad_norm),
        ]
        
        # 过滤有效值
        valid_components = [(name, val) for name, val in key_components if val > 0]
        
        if valid_components:
            names = [c[0] for c in valid_components]
            values = [c[1] for c in valid_components]
            
            colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
            
            bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=0.5)
            
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{val:.2e}",
                    ha='center', va='bottom', fontsize=9, rotation=45,
                )
            
            ax.set_ylabel("Gradient Norm ||grad W||")
            ax.set_title("Key Component Gradient Tracking\n(STE Flow: Splitter MLP -> Threshold -> Quota)")
            ax.set_yscale('log')
            plt.xticks(rotation=30, ha='right')
        else:
            ax.text(0.5, 0.5, "No key component gradient data", ha='center', va='center',
                   fontsize=14, transform=ax.transAxes)
            ax.axis('off')
        
        safe_tight_layout()
        self._add_watermark(fig)
        return fig
    
    def _plot_problem_report(self, metrics: Any) -> Figure:
        """梯度问题检测报告"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：消失梯度
        ax_vanish = axes[0]
        ax_vanish.axis('off')
        ax_vanish.set_title("Vanishing Gradient Detection", fontsize=14, fontweight='bold', color='orange')
        
        if metrics.vanishing_gradients:
            text = "\n".join(metrics.vanishing_gradients[:15])
            if len(metrics.vanishing_gradients) > 15:
                text += f"\n... and {len(metrics.vanishing_gradients) - 15} more"
            ax_vanish.text(0.05, 0.95, text, ha='left', va='top',
                          fontsize=9, transform=ax_vanish.transAxes,
                          fontfamily='monospace')
        else:
            ax_vanish.text(0.5, 0.5, "[OK] No vanishing gradients", ha='center', va='center',
                          fontsize=14, color='green', transform=ax_vanish.transAxes)
        
        # 右图：爆炸梯度
        ax_explode = axes[1]
        ax_explode.axis('off')
        ax_explode.set_title("Exploding Gradient Detection", fontsize=14, fontweight='bold', color='red')
        
        if metrics.exploding_gradients:
            text = "\n".join(metrics.exploding_gradients[:15])
            if len(metrics.exploding_gradients) > 15:
                text += f"\n... and {len(metrics.exploding_gradients) - 15} more"
            ax_explode.text(0.05, 0.95, text, ha='left', va='top',
                           fontsize=9, transform=ax_explode.transAxes,
                           fontfamily='monospace')
        else:
            ax_explode.text(0.5, 0.5, "[OK] No exploding gradients", ha='center', va='center',
                           fontsize=14, color='green', transform=ax_explode.transAxes)
        
        safe_tight_layout()
        self._add_watermark(fig)
        return fig
    
    def _plot_summary_card(self, metrics: Any) -> Figure:
        """L8 综合摘要卡片"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        ax.text(0.5, 0.95, "L8 Gradient Flow Summary", ha='center', va='top',
               fontsize=16, fontweight='bold', transform=ax.transAxes)
        
        # 健康状态
        n_vanish = len(metrics.vanishing_gradients)
        n_explode = len(metrics.exploding_gradients)
        
        if n_vanish == 0 and n_explode == 0:
            status = "[OK] Gradient Flow Healthy"
            status_color = 'green'
        elif n_explode > 0:
            status = "[X] Gradient Exploding"
            status_color = 'red'
        else:
            status = "[!] Gradient Vanishing"
            status_color = 'orange'
        
        ax.text(0.5, 0.82, status, ha='center', va='top',
               fontsize=18, fontweight='bold', color=status_color,
               transform=ax.transAxes)
        
        # 主要指标
        metrics_text = [
            f"Total Grad Norm: {metrics.total_grad_norm:.4f}",
            f"Max Grad Norm: {metrics.max_grad_norm:.4f}",
            f"Min Grad Norm: {metrics.min_grad_norm:.6f}",
            f"Max/Min Ratio: {metrics.grad_norm_ratio:.2e}",
            f"",
            f"Vanishing Params: {n_vanish}",
            f"Exploding Params: {n_explode}",
        ]
        
        y_start = 0.70
        for i, line in enumerate(metrics_text):
            ax.text(0.1, y_start - i * 0.08, line, ha='left', va='top',
                   fontsize=11, transform=ax.transAxes,
                   fontfamily='monospace')
        
        # 关键组件
        ax.text(0.55, 0.70, "Key Component Gradients:", ha='left', va='top',
               fontsize=12, fontweight='bold', transform=ax.transAxes)
        
        key_items = [
            ("Splitter MLP", metrics.splitter_mlp_grad_norm),
            ("Threshold", metrics.splitter_threshold_grad_norm),
            ("Quota", metrics.splitter_quota_grad_norm),
            ("Attn QKV", metrics.attention_qkv_grad_norm),
            ("LCA Embed", metrics.lca_embed_grad_norm),
        ]
        
        for i, (name, val) in enumerate(key_items):
            val_str = f"{val:.2e}" if val > 0 else "N/A"
            ax.text(0.60, 0.62 - i * 0.07, f"{name}: {val_str}", ha='left', va='top',
                   fontsize=10, transform=ax.transAxes)
        
        safe_tight_layout()
        self._add_watermark(fig)
        return fig
    
    def _draw_gauge(
        self,
        ax: plt.Axes,
        value: float,
        min_val: float,
        max_val: float,
        title: str,
        unit: str,
        thresholds: List[Tuple[float, str]],
    ) -> None:
        """绘制仪表盘"""
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        theta = np.linspace(np.pi, 0, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        ax.plot(x, y, 'lightgray', linewidth=20, solid_capstyle='round')
        
        normalized = (value - min_val) / (max_val - min_val + 1e-10)
        normalized = np.clip(normalized, 0, 1)
        
        color = 'gray'
        for thresh, c in sorted(thresholds):
            if value <= thresh:
                color = c
                break
        else:
            color = thresholds[-1][1] if thresholds else 'gray'
        
        theta_val = np.linspace(np.pi, np.pi - normalized * np.pi, 50)
        x_val = np.cos(theta_val)
        y_val = np.sin(theta_val)
        ax.plot(x_val, y_val, color=color, linewidth=18, solid_capstyle='round')
        
        pointer_angle = np.pi - normalized * np.pi
        ax.arrow(0, 0, 0.6 * np.cos(pointer_angle), 0.6 * np.sin(pointer_angle),
                head_width=0.1, head_length=0.05, fc='black', ec='black')
        ax.scatter([0], [0], s=100, c='black', zorder=5)
        
        ax.text(0, -0.2, f"{value:.2f} {unit}".strip(), ha='center', va='top',
               fontsize=14, fontweight='bold')
        ax.text(0, 1.3, title, ha='center', va='bottom', fontsize=12)
        ax.text(-1.1, 0, f"{min_val}", ha='center', va='center', fontsize=9)
        ax.text(1.1, 0, f"{max_val}", ha='center', va='center', fontsize=9)
