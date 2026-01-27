#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
L7 分割器专项可视化层 (Splitter Analysis Visualization)

数学形式化
============

本模块可视化 L7SplitterMetrics 的评估结果：

1. **温度调度** (Temperature Schedule)
   当前温度: τ(t) = τ_start × decay^t
   进度: progress = t / T_total

2. **可学习阈值** (Learnable Thresholds)
   每深度阈值: τ_d for d ∈ {0, ..., D}
   决策公式: split_d = 𝟙[score_d > τ_d]

3. **可学习配额** (Learnable Quotas)
   配额分布: q_d = softmax(quota_logits)_d
   配额熵: H_q = -Σ q_d log(q_d)

4. **MLP Logits 分布** (MLP Logits per Depth)
   每深度 MLP 输出统计: μ_d, σ_d
   健康指标: 各深度应有区分度

5. **决策置信度** (Decision Confidence)
   置信度: conf = |sigmoid(logit) - 0.5| × 2
   高置信度 = 明确的分割决策

Author: GitHub Copilot
Date: 2026-01-14
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .base import (
    FigureConfig,
    VisualizationLayer,
    VisualizationResult,
    safe_tight_layout,
)

# 尝试导入 evaluation_layers 中的数据类
try:
    from ..evaluation_layers import L7SplitterMetrics
except ImportError:
    L7SplitterMetrics = None


class L7SplitterVisualizer(VisualizationLayer):
    """L7 分割器专项可视化器
    
    数学形式化：
        V₇: L7SplitterMetrics → {温度进度, 阈值分布, 配额分布, MLP分析, ...}
    """
    
    LAYER_NAME = "L7_Splitter"
    LAYER_DESCRIPTION = "分割器专项可视化：温度调度、阈值、配额、决策分析"
    
    def visualize(
        self,
        metrics: Any,  # L7SplitterMetrics
        history: Optional[Dict[str, List[float]]] = None,
        **kwargs,
    ) -> VisualizationResult:
        """生成 L7 分割器可视化"""
        result = VisualizationResult(layer_name=self.LAYER_NAME)
        
        # 1. 温度与配额综合面板
        fig = self._plot_temperature_quota_panel(metrics)
        result.figures.append(fig)
        result.names.append("L7_temperature_quota")
        result.descriptions.append("温度调度与配额分布")
        
        # 2. 阈值分布
        if metrics.thresholds:
            fig = self._plot_thresholds(metrics.thresholds)
            result.figures.append(fig)
            result.names.append("L7_thresholds")
            result.descriptions.append("每深度可学习阈值")
        
        # 3. MLP Logits 分布
        # I101-4: decision_confidence_mean/selection_prob_mean 字段已移除，跳过可视化
        if metrics.gate_weight_stats:
            fig = self._plot_gate_weight_stats(metrics)
            result.figures.append(fig)
        
        # 5. I21 深度平衡诊断
        fig = self._plot_depth_balance_diagnosis(metrics)
        result.figures.append(fig)
        result.names.append("L7_depth_balance")
        result.descriptions.append("I21 深度平衡诊断")
        
        # 6. 综合摘要卡片
        fig = self._plot_summary_card(metrics)
        result.figures.append(fig)
        result.names.append("L7_summary")
        result.descriptions.append("L7 分割器摘要")
        
        return result
    
    def _plot_temperature_quota_panel(self, metrics: Any) -> Figure:
        """温度与配额综合面板"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图：温度仪表盘
        ax_temp = axes[0]
        self._draw_gauge(
            ax_temp,
            value=metrics.temperature,
            min_val=0.1,
            max_val=2.0,
            title="Gumbel Temperature",
            unit="T",
            thresholds=[(0.3, 'green'), (0.7, 'yellow'), (1.5, 'orange')],
        )
        
        if metrics.temperature_schedule_progress > 0:
            ax_temp.text(
                0.5, -0.1,
                f"退火进度: {metrics.temperature_schedule_progress:.1%}",
                ha='center', va='top',
                transform=ax_temp.transAxes,
                fontsize=10,
            )
        
        # 右图：配额分布饼图
        ax_quota = axes[1]
        if metrics.quotas:
            depths = sorted(metrics.quotas.keys())
            values = [metrics.quotas[d] for d in depths]
            labels = [f"深度 {d}" for d in depths]
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(depths)))
            
            ax_quota.pie(
                values, labels=labels, autopct='%1.1f%%',
                colors=colors, startangle=90,
            )
            ax_quota.set_title(f"可学习配额分布\n(熵 H={metrics.quota_entropy:.2f})")
        else:
            ax_quota.text(0.5, 0.5, "无配额数据", ha='center', va='center')
            ax_quota.set_title("可学习配额分布")
        
        safe_tight_layout()
        self._add_watermark(fig)
        return fig
    
    def _plot_thresholds(self, thresholds: Dict[int, float]) -> Figure:
        """每深度阈值条形图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        depths = sorted(thresholds.keys())
        values = [thresholds[d] for d in depths]
        
        bars = ax.bar(
            depths, values,
            color=plt.cm.coolwarm(np.linspace(0.2, 0.8, len(depths))),
            edgecolor='black', linewidth=0.5,
        )
        
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha='center', va='bottom', fontsize=9,
            )
        
        ax.set_xlabel("深度 (Depth)")
        ax.set_ylabel("阈值 (Threshold)")
        ax.set_title("可学习阈值 per Depth\n(高阈值 = 更难被选中)")
        ax.set_xticks(depths)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        safe_tight_layout()
        self._add_watermark(fig)
        return fig

    def _plot_depth_balance_diagnosis(self, metrics: Any) -> Figure:
        """I21 深度平衡诊断"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        status_items = [
            ("Log Compensation", metrics.log_compensation_enabled, 
             "启用 log(N_total/N_d) 补偿"),
            ("Depth KL Weight", metrics.depth_kl_weight > 0,
             f"KL 权重 = {metrics.depth_kl_weight}"),
            ("配额熵健康", metrics.quota_entropy > 0.5 if metrics.quota_entropy > 0 else None,
             f"H_quota = {metrics.quota_entropy:.2f}"),
            ("KL from Base", metrics.quota_kl_from_base < 1.0 if metrics.quota_kl_from_base > 0 else None,
             f"KL = {metrics.quota_kl_from_base:.2f}"),
        ]
        
        y_positions = np.arange(len(status_items))
        
        for i, (name, status, desc) in enumerate(status_items):
            if status is None:
                color = 'gray'
                symbol = '○'
            elif status:
                color = 'green'
                symbol = 'OK'
            else:
                color = 'red'
                symbol = 'X'
            
            ax.scatter([0.1], [y_positions[i]], s=500, c=color, marker='o', alpha=0.7)
            ax.text(0.1, y_positions[i], symbol, ha='center', va='center', 
                   fontsize=16, fontweight='bold', color='white')
            ax.text(0.25, y_positions[i], name, ha='left', va='center', fontsize=12, fontweight='bold')
            ax.text(0.25, y_positions[i] - 0.3, desc, ha='left', va='center', fontsize=9, color='gray')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(status_items) - 0.5)
        ax.set_title("I21 深度平衡诊断", fontsize=14, fontweight='bold')
        ax.axis('off')
        
        safe_tight_layout()
        self._add_watermark(fig)
        return fig
    
    def _plot_summary_card(self, metrics: Any) -> Figure:
        """L7 综合摘要卡片"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        ax.text(0.5, 0.95, "L7 Splitter Analysis Summary", ha='center', va='top',
               fontsize=16, fontweight='bold', transform=ax.transAxes)
        
        metrics_text = [
            f"Temperature tau = {metrics.temperature:.3f}",
            f"Annealing Progress = {metrics.temperature_schedule_progress:.1%}" if metrics.temperature_schedule_progress > 0 else "Annealing Progress = N/A",
            f"",
            # I101-4: mlp_logits_*/selection_prob_*/decision_confidence_* 字段已移除
            f"Gate Weight Stats: {len(metrics.gate_weight_stats)} depths tracked" if metrics.gate_weight_stats else "Gate Weight Stats: N/A",
            f"",
            f"Quota Entropy = {metrics.quota_entropy:.2f}" if metrics.quota_entropy > 0 else "Quota Entropy = N/A",
            f"Log Compensation = {'ON' if metrics.log_compensation_enabled else 'OFF'}",
            f"Depth KL Weight = {metrics.depth_kl_weight}",
        ]
        
        y_start = 0.80
        for i, line in enumerate(metrics_text):
            ax.text(0.1, y_start - i * 0.07, line, ha='left', va='top',
                   fontsize=11, transform=ax.transAxes,
                   fontfamily='monospace' if '=' in line else 'sans-serif')
        
        if metrics.quotas:
            ax.text(0.6, 0.80, "Quota Distribution:", ha='left', va='top',
                   fontsize=12, fontweight='bold', transform=ax.transAxes)
            for i, (d, q) in enumerate(sorted(metrics.quotas.items())):
                ax.text(0.65, 0.73 - i * 0.06, f"Depth {d}: {q:.1%}", ha='left', va='top',
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
        
        # 防止除零
        range_val = max_val - min_val
        if range_val <= 0:
            range_val = 1.0
        normalized = (value - min_val) / range_val
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
        
        ax.text(0, -0.2, f"{value:.3f} {unit}".strip(), ha='center', va='top',
               fontsize=14, fontweight='bold')
        ax.text(0, 1.3, title, ha='center', va='bottom', fontsize=12)
        ax.text(-1.1, 0, f"{min_val}", ha='center', va='center', fontsize=9)
        ax.text(1.1, 0, f"{max_val}", ha='center', va='center', fontsize=9)
