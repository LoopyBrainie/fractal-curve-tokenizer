#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L5 资源效率可视化层 (Resource Efficiency Visualization)

数学形式化
============

本模块可视化 L5EfficiencyMetrics 的评估结果：

1. **延迟分解** (Latency Breakdown)
   总延迟: $L_{total} = L_{tokenizer} + L_{transformer} + L_{head}$
   各部分占比分析

2. **吞吐量分析** (Throughput Analysis)
   吞吐量: $T = N_{samples} / t_{total}$ (samples/sec)

3. **内存效率** (Memory Efficiency)
   效率: $E_{mem} = Accuracy / Peak_{memory}$

4. **参数量分析** (Parameter Analysis)
   总参数、可训练参数、各模块占比

5. **效率-精度权衡** (Efficiency-Accuracy Trade-off)
   准确率 vs 延迟/内存/参数量

Author: GitHub Copilot
Date: 2026-01-11
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
    format_large_number,
)

# 尝试导入 evaluation_layers 中的数据类
try:
    from ..evaluation_layers import L5EfficiencyMetrics
except ImportError:
    L5EfficiencyMetrics = None


class L5EfficiencyVisualizer(VisualizationLayer):
    """L5 资源效率可视化器
    
    数学形式化：
        V₅: L5EfficiencyMetrics → {延迟分解, 内存分析, 参数量, ...}
    
    输入：
        L5EfficiencyMetrics 包含:
        - avg_latency_ms: float
        - tokenizer_latency_ms: float
        - transformer_latency_ms: float
        - head_latency_ms: float
        - throughput_samples_per_sec: float
        - peak_memory_mb: float
        - accuracy_per_gflops: float
        - accuracy_per_token: float
        - total_params: int
        - trainable_params: int
    
    输出：
        VisualizationResult 包含多个 Figure
    """
    
    LAYER_NAME = "L5_Efficiency"
    LAYER_DESCRIPTION = "资源效率可视化：延迟分解、内存分析、参数量"
    
    def visualize(
        self,
        metrics: Any,  # L5EfficiencyMetrics
        class_names: Optional[List[str]] = None,
        accuracy: Optional[float] = None,
        module_params: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> VisualizationResult:
        """生成 L5 效率可视化
        
        Args:
            metrics: L5EfficiencyMetrics 评估结果
            class_names: 类别名称列表（未使用）
            accuracy: 模型准确率（用于效率-精度分析）
            module_params: 各模块参数量 {模块名: 参数数}
            **kwargs: 额外参数
            
        Returns:
            VisualizationResult
        """
        result = VisualizationResult(layer_name=self.LAYER_NAME)
        
        # 1. 延迟分解
        fig = self._plot_latency_breakdown(metrics)
        result.figures.append(fig)
        result.names.append("L5_latency_breakdown")
        result.descriptions.append("推理延迟分解")
        
        # 2. 内存和吞吐量
        fig = self._plot_memory_throughput(metrics)
        result.figures.append(fig)
        result.names.append("L5_memory_throughput")
        result.descriptions.append("内存使用与吞吐量")
        
        # 3. 参数量分析
        fig = self._plot_parameter_analysis(metrics, module_params)
        result.figures.append(fig)
        result.names.append("L5_parameters")
        result.descriptions.append("模型参数量分析")
        
        # 4. 综合摘要
        fig = self._plot_summary(metrics, accuracy)
        result.figures.append(fig)
        result.names.append("L5_summary")
        result.descriptions.append("L5 资源效率摘要")
        
        return result
    
    def _plot_latency_breakdown(self, metrics: Any) -> Figure:
        """绘制延迟分解图
        
        数学形式：
            饼图和条形图展示各组件延迟占比
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 延迟数据
        components = ['Tokenizer', 'Transformer', 'Head', 'Other']
        latencies = [
            metrics.tokenizer_latency_ms,
            metrics.transformer_latency_ms,
            metrics.head_latency_ms,
            max(0, metrics.avg_latency_ms - metrics.tokenizer_latency_ms - 
                metrics.transformer_latency_ms - metrics.head_latency_ms)
        ]
        
        # 过滤零值
        valid_mask = np.array(latencies) > 0
        components = [c for c, v in zip(components, valid_mask) if v]
        latencies = [l for l, v in zip(latencies, valid_mask) if v]
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'][:len(components)]
        
        # 1. 饼图
        ax1 = axes[0]
        wedges, texts, autotexts = ax1.pie(
            latencies, labels=components, autopct='%1.1f%%',
            colors=colors, explode=[0.02] * len(components),
            startangle=90, textprops={'fontsize': 10}
        )
        ax1.set_title(f"Latency Breakdown\nTotal: {metrics.avg_latency_ms:.2f} ms",
                     fontsize=12, fontweight='bold')
        
        # 2. 条形图
        ax2 = axes[1]
        y_pos = np.arange(len(components))
        bars = ax2.barh(y_pos, latencies, color=colors, edgecolor='white')
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(components, fontsize=10)
        ax2.set_xlabel("Latency (ms)", fontsize=11)
        
        for bar, lat in zip(bars, latencies):
            ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    f'{lat:.2f} ms', va='center', fontsize=10)
        
        ax2.set_title("Component Latency", fontsize=12, fontweight='bold')
        
        fig.suptitle("L5: Inference Latency Analysis", fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _plot_memory_throughput(self, metrics: Any) -> Figure:
        """绘制内存和吞吐量分析"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. 内存使用
        ax1 = axes[0]
        memory_mb = metrics.peak_memory_mb
        
        # 仪表盘风格
        theta = np.linspace(0, np.pi, 100)
        ax1.fill_between(np.cos(theta), np.sin(theta), 0, color='lightgray', alpha=0.3)
        
        # 根据内存大小着色区域
        memory_levels = [(0, 2000, '#2ecc71'), (2000, 8000, '#f39c12'), (8000, 32000, '#e74c3c')]
        for low, high, color in memory_levels:
            t1 = np.pi * (1 - high / 32000)
            t2 = np.pi * (1 - low / 32000)
            theta_fill = np.linspace(max(0, t1), min(np.pi, t2), 20)
            ax1.fill_between(0.9 * np.cos(theta_fill), 0.9 * np.sin(theta_fill), 0,
                            color=color, alpha=0.3)
        
        # 指针
        normalized = min(1.0, memory_mb / 32000)
        angle = np.pi * (1 - normalized)
        ax1.arrow(0, 0, 0.7 * np.cos(angle), 0.7 * np.sin(angle),
                 head_width=0.08, head_length=0.05, fc='black', ec='black')
        ax1.scatter([0], [0], s=100, c='black', zorder=5)
        
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-0.2, 1.2)
        ax1.set_aspect('equal')
        ax1.axis('off')
        
        ax1.text(0, -0.15, f"Peak Memory: {memory_mb:.0f} MB", ha='center', fontsize=11)
        ax1.set_title("GPU Memory Usage", fontsize=12, fontweight='bold')
        
        # 2. 吞吐量
        ax2 = axes[1]
        throughput = metrics.throughput_samples_per_sec
        
        # 条形图
        ax2.bar(['Throughput'], [throughput], color='#3498db', edgecolor='white', width=0.4)
        ax2.text(0, throughput + 5, f'{throughput:.1f}\nsamples/sec',
                ha='center', fontsize=12, fontweight='bold')
        
        ax2.set_ylabel("Samples per Second", fontsize=11)
        ax2.set_ylim(0, throughput * 1.3)
        ax2.set_title("Inference Throughput", fontsize=12, fontweight='bold')
        
        fig.suptitle("L5: Memory & Throughput Analysis", fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _plot_parameter_analysis(
        self,
        metrics: Any,
        module_params: Optional[Dict[str, int]] = None,
    ) -> Figure:
        """绘制参数量分析"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. 总参数量
        ax1 = axes[0]
        total = metrics.total_params
        trainable = metrics.trainable_params
        frozen = total - trainable
        
        labels = ['Trainable', 'Frozen']
        sizes = [trainable, frozen]
        colors = ['#3498db', '#95a5a6']
        
        # 过滤零值
        valid = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
        if valid:
            labels, sizes, colors = zip(*valid)
            
            wedges, texts, autotexts = ax1.pie(
                sizes, labels=labels, autopct=lambda p: f'{p:.1f}%\n({format_large_number(int(p * total / 100))})',
                colors=colors, explode=[0.02] * len(sizes),
                startangle=90, textprops={'fontsize': 9}
            )
        
        ax1.set_title(f"Total Parameters: {format_large_number(total)}\n"
                     f"({total:,})",
                     fontsize=12, fontweight='bold')
        
        # 2. 模块分解（如果有）
        ax2 = axes[1]
        if module_params:
            modules = list(module_params.keys())
            params = list(module_params.values())
            
            # 排序
            sorted_idx = np.argsort(params)[::-1]
            modules = [modules[i] for i in sorted_idx]
            params = [params[i] for i in sorted_idx]
            
            # 取前 10 个
            modules = modules[:10]
            params = params[:10]
            
            y_pos = np.arange(len(modules))
            colors = plt.get_cmap('viridis')(np.linspace(0.2, 0.8, len(modules)))
            
            bars = ax2.barh(y_pos, params, color=colors, edgecolor='white')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(modules, fontsize=9)
            ax2.set_xlabel("Parameters", fontsize=11)
            
            for bar, p in zip(bars, params):
                ax2.text(bar.get_width() + total * 0.01, bar.get_y() + bar.get_height() / 2,
                        format_large_number(p), va='center', fontsize=9)
            
            ax2.set_title("Parameters by Module", fontsize=12, fontweight='bold')
        else:
            # 简单统计
            ax2.text(0.5, 0.5,
                    f"Parameter Summary\n\n"
                    f"Total: {format_large_number(total)}\n"
                    f"Trainable: {format_large_number(trainable)}\n"
                    f"Frozen: {format_large_number(frozen)}\n\n"
                    f"Trainable Ratio: {trainable / total * 100:.1f}%",
                    ha='center', va='center', fontsize=11,
                    transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow'))
            ax2.axis('off')
        
        fig.suptitle("L5: Parameter Analysis", fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _plot_summary(self, metrics: Any, accuracy: Optional[float] = None) -> Figure:
        """绘制 L5 效率摘要"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        acc_str = f"{accuracy:.1%}" if accuracy else "N/A"
        
        summary_text = f"""
┌──────────────────────────────────────────────────────┐
│              L5 Resource Efficiency Summary           │
├──────────────────────────────────────────────────────┤
│  Latency                                              │
│    • Total: {metrics.avg_latency_ms:>10.2f} ms                          │
│    • Tokenizer: {metrics.tokenizer_latency_ms:>10.2f} ms                      │
│    • Transformer: {metrics.transformer_latency_ms:>10.2f} ms                    │
│    • Head: {metrics.head_latency_ms:>10.2f} ms                            │
├──────────────────────────────────────────────────────┤
│  Throughput & Memory                                  │
│    • Throughput: {metrics.throughput_samples_per_sec:>10.1f} samples/sec           │
│    • Peak Memory: {metrics.peak_memory_mb:>10.0f} MB                       │
├──────────────────────────────────────────────────────┤
│  Parameters                                           │
│    • Total: {format_large_number(metrics.total_params):>12}                            │
│    • Trainable: {format_large_number(metrics.trainable_params):>12}                        │
├──────────────────────────────────────────────────────┤
│  Efficiency Metrics                                   │
│    • Accuracy: {acc_str:>10}                              │
│    • Acc/Token: {metrics.accuracy_per_token:>10.4f}                          │
│    • Acc/GFLOPs: {metrics.accuracy_per_gflops:>10.4f}                         │
└──────────────────────────────────────────────────────┘
        """
        
        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
               fontsize=10, va='center', ha='center',
               family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        
        return fig
