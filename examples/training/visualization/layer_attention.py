#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
L3 注意力机制可视化层 (Attention Mechanism Visualization)

数学形式化
============

本模块可视化 L3AttentionMetrics 的评估结果：

1. **注意力熵分布** (Attention Entropy Distribution)
   每层每头熵: $H_{attn}^{(l,h)} = -\frac{1}{N} \sum_i \sum_j A_{ij}^{(l,h)} \log A_{ij}^{(l,h)}$
   高熵 = 分散注意力，低熵 = 集中注意力

2. **Head 利用率热图** (Head Utilization Heatmap)
   利用率: $Util^{(l,h)} = \mathbb{1}[H_{attn}^{(l,h)} \in (\tau_{low}, \tau_{high})]$
   Dead Head = 熵过低（全集中）或过高（均匀随机）

3. **LCA Bias 效果可视化** (NEW)
   LCA(i,j) = depth(LCA(region_i, region_j))
   展示 Hilbert 邻域 token 的相对偏置

4. **CLS 注意力分析** (CLS Attention Analysis)
   CLS 关注范围: 哪些 token 被 CLS 重点关注
   覆盖率: 被关注的 token 比例

5. **跨层注意力演化** (Cross-Layer Attention Evolution)
   注意力模式如何随层深度变化

Author: GitHub Copilot
Date: 2026-01-11
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.colors import Normalize

from .base import (
    FigureConfig,
    VisualizationLayer,
    VisualizationResult,
    compute_entropy,
    safe_tight_layout,
)

# 尝试导入 evaluation_layers 中的数据类
try:
    from ..evaluation_layers import L3AttentionMetrics
except ImportError:
    L3AttentionMetrics = None


class L3AttentionVisualizer(VisualizationLayer):
    """L3 注意力机制可视化器
    
    数学形式化：
        V₃: L3AttentionMetrics → {熵分布, Head利用率, CLS分析, ...}
    
    输入：
        L3AttentionMetrics 包含:
        - per_layer_entropy: List[float]
        - avg_entropy: float
        - head_utilization: List[List[float]]  # [layer][head]
        - dead_head_ratio: float
        - cls_attention_coverage: float
        - cls_attention_entropy: float
    
    额外输入（用于详细可视化）:
        - attention_weights: List[np.ndarray]  # 每层的注意力权重
        - lca_bias_matrix: np.ndarray  # LCA 偏置矩阵
    
    输出：
        VisualizationResult 包含多个 Figure
    """
    
    LAYER_NAME = "L3_Attention"
    LAYER_DESCRIPTION = "注意力机制可视化：熵分布、Head 利用率、CLS 分析"
    
    def visualize(
        self,
        metrics: Any,  # L3AttentionMetrics
        class_names: Optional[List[str]] = None,
        attention_weights: Optional[List[np.ndarray]] = None,
        lca_bias_matrix: Optional[np.ndarray] = None,
        token_depths: Optional[np.ndarray] = None,
        **kwargs,
    ) -> VisualizationResult:
        """生成 L3 注意力可视化
        
        Args:
            metrics: L3AttentionMetrics 评估结果
            class_names: 类别名称列表（未使用）
            attention_weights: 每层注意力权重列表（可选）
            lca_bias_matrix: LCA 偏置矩阵（可选）
            token_depths: Token 深度数组（可选）
            **kwargs: 额外参数
            
        Returns:
            VisualizationResult
        """
        result = VisualizationResult(layer_name=self.LAYER_NAME)
        
        # 1. 每层注意力熵
        if metrics.per_layer_entropy:
            fig = self._plot_entropy_per_layer(
                metrics.per_layer_entropy,
                metrics.avg_entropy,
            )
            result.figures.append(fig)
            result.names.append("L3_entropy_per_layer")
            result.descriptions.append("每层注意力熵分布")
        
        # 2. Head 利用率热图
        if metrics.head_utilization:
            fig = self._plot_head_utilization(
                metrics.head_utilization,
                metrics.dead_head_ratio,
            )
            result.figures.append(fig)
            result.names.append("L3_head_utilization")
            result.descriptions.append("多头注意力利用率热图")
        
        # 3. CLS 注意力分析
        fig = self._plot_cls_attention(
            metrics.cls_attention_coverage,
            metrics.cls_attention_entropy,
            attention_weights,
        )
        result.figures.append(fig)
        result.names.append("L3_cls_attention")
        result.descriptions.append("CLS Token 注意力分析")
        
        # 4. LCA Bias 可视化（如果有）
        if lca_bias_matrix is not None:
            fig = self._plot_lca_bias(
                lca_bias_matrix,
                token_depths,
            )
            result.figures.append(fig)
            result.names.append("L3_lca_bias")
            result.descriptions.append("LCA Hilbert Bias 热图")
        
        # 5. LCA-注意力相关性分析（如果有）
        if lca_bias_matrix is not None and attention_weights is not None:
            fig = self._plot_lca_attention_correlation(
                lca_bias_matrix,
                attention_weights,
            )
            result.figures.append(fig)
            result.names.append("L3_lca_correlation")
            result.descriptions.append("LCA Bias 与注意力权重相关性")
        
        # 6. Per-Depth 注意力分析（如果有）
        if token_depths is not None and attention_weights is not None:
            fig = self._plot_per_depth_attention(
                attention_weights,
                token_depths,
            )
            result.figures.append(fig)
            result.names.append("L3_per_depth_attention")
            result.descriptions.append("不同深度 Token 的注意力模式")
        
        # 7. Hilbert 局部性分析（如果有）
        if hasattr(metrics, 'hilbert_locality_score'):
            fig = self._plot_hilbert_locality(
                metrics.hilbert_locality_score,
                metrics.hilbert_locality_by_layer if hasattr(metrics, 'hilbert_locality_by_layer') else None,
            )
            result.figures.append(fig)
            result.names.append("L3_hilbert_locality")
            result.descriptions.append("Hilbert 局部性评分")
        
        # 8. 综合摘要
        fig = self._plot_summary(metrics)
        result.figures.append(fig)
        result.names.append("L3_summary")
        result.descriptions.append("L3 注意力机制摘要")
        
        return result
    
    def _plot_entropy_per_layer(
        self,
        per_layer_entropy: List[float],
        avg_entropy: float,
    ) -> Figure:
        """绘制每层注意力熵
        
        数学形式：折线图表示 H(l) = mean_h(H(l,h))
        熵反映注意力的集中/分散程度
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        n_layers = len(per_layer_entropy)
        layers = np.arange(n_layers)
        
        # 1. 折线图
        ax1 = axes[0]
        ax1.plot(layers, per_layer_entropy, 'bo-', linewidth=2, markersize=8,
                label='Per-Layer Entropy')
        ax1.axhline(avg_entropy, color='red', linestyle='--', 
                   label=f'Mean: {avg_entropy:.3f}')
        
        # 填充区域表示异常范围
        ax1.fill_between(layers, 0, per_layer_entropy, alpha=0.3)
        
        ax1.set_xlabel("Layer", fontsize=11)
        ax1.set_ylabel("Attention Entropy", fontsize=11)
        ax1.set_xticks(layers)
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_title("Attention Entropy per Layer", fontsize=12, fontweight='bold')
        
        # 2. 柱状图
        ax2 = axes[1]
        colors = plt.get_cmap('coolwarm')(
            Normalize(vmin=min(per_layer_entropy) * 0.9, 
                     vmax=max(per_layer_entropy) * 1.1)(per_layer_entropy)
        )
        
        bars = ax2.bar(layers, per_layer_entropy, color=colors, edgecolor='white')
        ax2.axhline(avg_entropy, color='red', linestyle='--', linewidth=2)
        
        for bar, ent in zip(bars, per_layer_entropy):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{ent:.2f}', ha='center', fontsize=9)
        
        ax2.set_xlabel("Layer", fontsize=11)
        ax2.set_ylabel("Entropy", fontsize=11)
        ax2.set_xticks(layers)
        ax2.set_title("Entropy Comparison", fontsize=12, fontweight='bold')
        
        # 解读说明
        fig.text(0.5, 0.02, 
                "Low Entropy → Focused Attention | High Entropy → Distributed Attention",
                ha='center', fontsize=10, style='italic')
        
        fig.suptitle("L3: Attention Entropy Analysis", fontsize=14, fontweight='bold')
        safe_tight_layout(rect=[0, 0.05, 1, 0.95])
        return fig
    
    def _plot_head_utilization(
        self,
        head_utilization: List[List[float]],
        dead_head_ratio: float,
    ) -> Figure:
        """绘制 Head 利用率热图
        
        数学形式：
            热图表示每个 (layer, head) 的利用率
            Dead Head = 利用率 < 阈值
        """
        # 转换为 numpy 数组
        util_matrix = np.array(head_utilization)  # [n_layers, n_heads]
        n_layers, n_heads = util_matrix.shape
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 热图
        ax1 = axes[0]
        im = ax1.imshow(util_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax1.set_xlabel("Head", fontsize=11)
        ax1.set_ylabel("Layer", fontsize=11)
        ax1.set_xticks(np.arange(n_heads))
        ax1.set_yticks(np.arange(n_layers))
        ax1.set_xticklabels([f"H{i}" for i in range(n_heads)])
        ax1.set_yticklabels([f"L{i}" for i in range(n_layers)])
        
        # 颜色条
        cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('Utilization', fontsize=10)
        
        # 标记 dead heads
        for i in range(n_layers):
            for j in range(n_heads):
                if util_matrix[i, j] < 0.2:  # Dead head 阈值
                    ax1.text(j, i, '✗', ha='center', va='center', 
                            color='red', fontsize=12, fontweight='bold')
        
        ax1.set_title(f"Head Utilization Heatmap\n(Dead Head Ratio: {dead_head_ratio:.1%})",
                     fontsize=12, fontweight='bold')
        
        # 2. 每层平均 + 每 head 平均
        ax2 = axes[1]
        
        # 每层平均
        layer_means = util_matrix.mean(axis=1)
        ax2.barh(np.arange(n_layers) - 0.2, layer_means, height=0.35,
                color='#3498db', label='Layer Mean', alpha=0.8)
        
        # 每 head 平均（次坐标）
        ax2_twin = ax2.twiny()
        head_means = util_matrix.mean(axis=0)
        ax2_twin.bar(np.arange(n_heads), head_means, width=0.35,
                    color='#e74c3c', label='Head Mean', alpha=0.6)
        
        ax2.set_yticks(np.arange(n_layers))
        ax2.set_yticklabels([f"L{i}" for i in range(n_layers)])
        ax2.set_xlabel("Layer Mean Utilization", fontsize=11, color='#3498db')
        ax2.set_xlim(0, 1)
        
        ax2_twin.set_xticks(np.arange(n_heads))
        ax2_twin.set_xticklabels([f"H{i}" for i in range(n_heads)])
        ax2_twin.set_xlabel("Head", fontsize=11, color='#e74c3c')
        
        ax2.legend(loc='lower left')
        ax2.set_title("Mean Utilization", fontsize=12, fontweight='bold')
        
        fig.suptitle("L3: Multi-Head Attention Utilization", fontsize=14, fontweight='bold')
        safe_tight_layout()
        return fig
    
    def _plot_cls_attention(
        self,
        coverage: float,
        entropy: float,
        attention_weights: Optional[List[np.ndarray]] = None,
    ) -> Figure:
        """绘制 CLS Token 注意力分析
        
        数学形式：
            CLS 注意力: A_{0,:}^{(l)} (第 0 行，即 CLS 对其他 token 的注意力)
            覆盖率: 被显著关注的 token 比例
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. 覆盖率和熵仪表
        ax1 = axes[0]
        
        # 双条形图
        metrics_names = ['Coverage', 'Entropy']
        metrics_values = [coverage, entropy]
        colors = ['#3498db', '#9b59b6']
        
        bars = ax1.bar(metrics_names, metrics_values, color=colors, edgecolor='white')
        
        for bar, val in zip(bars, metrics_values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
        
        ax1.set_ylim(0, max(1.0, max(metrics_values) * 1.2))
        ax1.set_ylabel("Value", fontsize=11)
        
        # 参考线
        ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        
        ax1.set_title("CLS Attention Metrics", fontsize=12, fontweight='bold')
        
        # 2. 如果有注意力权重，显示 CLS 注意力分布
        ax2 = axes[1]
        
        if attention_weights is not None and len(attention_weights) > 0:
            # 取最后一层的 CLS 注意力
            last_layer_attn = attention_weights[-1]  # [n_heads, seq_len, seq_len] or similar
            
            if len(last_layer_attn.shape) >= 2:
                # 提取 CLS 行（假设 CLS 是第 0 个 token）
                if len(last_layer_attn.shape) == 3:
                    cls_attn = last_layer_attn.mean(axis=0)[0, 1:]  # 排除 CLS 自己
                else:
                    cls_attn = last_layer_attn[0, 1:]
                
                # 绘制注意力分布
                token_indices = np.arange(len(cls_attn))
                ax2.bar(token_indices, cls_attn, color='#3498db', alpha=0.7)
                ax2.set_xlabel("Token Index", fontsize=11)
                ax2.set_ylabel("CLS Attention", fontsize=11)
                
                # 标记高注意力 token
                threshold = np.percentile(cls_attn, 90)
                high_attn_mask = cls_attn > threshold
                ax2.scatter(token_indices[high_attn_mask], cls_attn[high_attn_mask],
                           color='red', s=50, zorder=5, label='Top 10%')
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, "Invalid attention shape", 
                        ha='center', va='center', transform=ax2.transAxes)
        else:
            # 显示占位文本
            ax2.text(0.5, 0.5, 
                    f"CLS Coverage: {coverage:.1%}\n"
                    f"CLS Entropy: {entropy:.3f}\n\n"
                    "(Detailed attention weights\nnot available)",
                    ha='center', va='center', fontsize=11,
                    transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
        
        ax2.set_title("CLS Attention Distribution (Last Layer)", fontsize=12, fontweight='bold')
        
        fig.suptitle("L3: CLS Token Attention Analysis", fontsize=14, fontweight='bold')
        safe_tight_layout()
        return fig
    
    def _plot_lca_bias(
        self,
        lca_bias_matrix: np.ndarray,
        token_depths: Optional[np.ndarray] = None,
    ) -> Figure:
        """绘制 LCA Hilbert Bias 热图
        
        数学形式：
            LCA(i,j) = depth(LCA(region_i, region_j))
            偏置值反映 Hilbert 曲线上的邻近关系
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. LCA Bias 热图
        ax1 = axes[0]
        n_tokens = lca_bias_matrix.shape[0]
        
        im = ax1.imshow(lca_bias_matrix, cmap='plasma', aspect='auto')
        ax1.set_xlabel("Token j", fontsize=11)
        ax1.set_ylabel("Token i", fontsize=11)
        
        cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('LCA Bias Value', fontsize=10)
        
        ax1.set_title(f"LCA Hilbert Bias Matrix\n({n_tokens} x {n_tokens})",
                     fontsize=12, fontweight='bold')
        
        # 2. 偏置分布 + 深度关系
        ax2 = axes[1]
        
        # 提取非对角元素
        mask = ~np.eye(n_tokens, dtype=bool)
        bias_values = lca_bias_matrix[mask]
        
        ax2.hist(bias_values, bins=30, color='#9b59b6', edgecolor='white', alpha=0.8)
        ax2.set_xlabel("LCA Bias Value", fontsize=11)
        ax2.set_ylabel("Frequency", fontsize=11)
        
        mean_bias = np.mean(bias_values)
        ax2.axvline(mean_bias, color='red', linestyle='--', 
                   label=f'Mean: {mean_bias:.3f}')
        ax2.legend()
        
        ax2.set_title("LCA Bias Distribution", fontsize=12, fontweight='bold')
        
        # 说明文字
        fig.text(0.5, 0.02,
                "LCA Bias: Higher values for Hilbert-adjacent tokens → "
                "Locality-preserving attention",
                ha='center', fontsize=10, style='italic')
        
        fig.suptitle("L3: LCA (Lowest Common Ancestor) Hilbert Bias",
                    fontsize=14, fontweight='bold')
        safe_tight_layout(rect=[0, 0.05, 1, 0.95])
        return fig
    
    def _plot_lca_attention_correlation(
        self,
        lca_bias_matrix: np.ndarray,
        attention_weights: List[np.ndarray],
    ) -> Figure:
        """绘制 LCA Bias 与注意力权重的相关性
        
        数学形式：
            ρ = corr(LCA_bias(i,j), A(i,j))
            高相关性表示注意力遵循 Hilbert 局部性
        
        Args:
            lca_bias_matrix: LCA 偏置矩阵 [N, N]
            attention_weights: 每层注意力权重
        """
        n_layers = len(attention_weights)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 提取 LCA 偏置的非对角元素
        n_tokens = lca_bias_matrix.shape[0]
        mask = ~np.eye(n_tokens, dtype=bool)
        lca_flat = lca_bias_matrix[mask]
        
        correlations = []
        
        # 1. 散点图（最后一层）
        ax1 = axes[0]
        if len(attention_weights) > 0:
            last_attn = attention_weights[-1]
            # 平均多头
            if len(last_attn.shape) == 3:
                last_attn = last_attn.mean(axis=0)
            # 确保形状匹配
            if last_attn.shape[0] == n_tokens:
                attn_flat = last_attn[mask]
                
                # 散点图
                ax1.scatter(lca_flat, attn_flat, alpha=0.3, s=5, c='#3498db')
                
                # 线性拟合
                if len(lca_flat) > 10:
                    try:
                        z = np.polyfit(lca_flat, attn_flat, 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(lca_flat.min(), lca_flat.max(), 100)
                        ax1.plot(x_line, p(x_line), 'r-', linewidth=2, label='Linear Fit')
                        
                        # 计算相关系数，防止 NaN
                        corr = np.corrcoef(lca_flat, attn_flat)[0, 1]
                        if np.isfinite(corr):
                            ax1.text(0.05, 0.95, f'rho = {corr:.3f}', transform=ax1.transAxes,
                                    fontsize=12, fontweight='bold', va='top')
                    except Exception:
                        pass  # 忽略拟合失败
                    
                ax1.legend()
        
        ax1.set_xlabel("LCA Bias", fontsize=11)
        ax1.set_ylabel("Attention Weight", fontsize=11)
        ax1.set_title("LCA-Attention Correlation (Last Layer)", fontsize=12, fontweight='bold')
        
        # 2. 每层相关系数
        ax2 = axes[1]
        for l, attn in enumerate(attention_weights):
            if len(attn.shape) == 3:
                attn = attn.mean(axis=0)
            if attn.shape[0] == n_tokens:
                attn_flat = attn[mask]
                try:
                    corr = np.corrcoef(lca_flat, attn_flat)[0, 1]
                    if not np.isfinite(corr):
                        corr = 0.0
                except Exception:
                    corr = 0.0
                correlations.append(corr)
            else:
                correlations.append(0.0)
        
        layers = np.arange(n_layers)
        colors = ['#2ecc71' if c > 0.3 else '#f39c12' if c > 0.1 else '#e74c3c' 
                 for c in correlations]
        ax2.bar(layers, correlations, color=colors, edgecolor='white')
        ax2.axhline(0.3, color='green', linestyle='--', alpha=0.5, label='Good (0.3)')
        ax2.axhline(0.1, color='orange', linestyle='--', alpha=0.5, label='Weak (0.1)')
        ax2.set_xlabel("Layer", fontsize=11)
        ax2.set_ylabel("Correlation ρ", fontsize=11)
        ax2.set_xticks(layers)
        ax2.legend(loc='lower right', fontsize=9)
        ax2.set_title("LCA-Attention Correlation per Layer", fontsize=12, fontweight='bold')
        
        # 3. 相关性仪表盘
        ax3 = axes[2]
        valid_corrs = [c for c in correlations if np.isfinite(c)]
        avg_corr = np.mean(valid_corrs) if valid_corrs else 0.0
        self._draw_correlation_gauge(ax3, avg_corr)
        ax3.set_title("Average LCA Correlation", fontsize=12, fontweight='bold')
        
        fig.suptitle("L3: LCA Bias - Attention Weight Correlation Analysis",
                    fontsize=14, fontweight='bold')
        safe_tight_layout()
        return fig
    
    def _draw_correlation_gauge(self, ax, value: float):
        """绘制相关性仪表盘"""
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 半圆仪表
        theta = np.linspace(0, np.pi, 100)
        r = 1.0
        
        # 背景弧（分三段颜色）
        segments = [
            (0, np.pi/3, '#e74c3c'),       # 红色：低相关
            (np.pi/3, 2*np.pi/3, '#f39c12'), # 橙色：中等
            (2*np.pi/3, np.pi, '#2ecc71'),   # 绿色：高相关
        ]
        
        for start, end, color in segments:
            t = np.linspace(start, end, 30)
            ax.fill_between(r * np.cos(t), [0]*len(t), r * np.sin(t),
                           color=color, alpha=0.3)
            ax.plot(r * np.cos(t), r * np.sin(t), color=color, linewidth=3)
        
        # 指针
        # 值范围 -1 到 1，映射到 0 到 π
        angle = np.pi * (1 - (value + 1) / 2)  # 反转方向
        pointer_len = 0.85
        ax.arrow(0, 0, pointer_len * np.cos(angle), pointer_len * np.sin(angle),
                head_width=0.08, head_length=0.05, fc='black', ec='black')
        
        # 数值显示
        ax.text(0, 0.3, f'{value:.3f}', ha='center', va='center',
               fontsize=20, fontweight='bold')
        
        # 状态文字
        if value > 0.3:
            status = "Strong Locality"
            color = '#2ecc71'
        elif value > 0.1:
            status = "Moderate"
            color = '#f39c12'
        else:
            status = "Weak Locality"
            color = '#e74c3c'
        
        ax.text(0, -0.3, status, ha='center', va='center',
               fontsize=12, fontweight='bold', color=color)
        
        # 标签
        ax.text(-1.2, 0, '-1', ha='center', fontsize=10)
        ax.text(1.2, 0, '1', ha='center', fontsize=10)
        ax.text(0, 1.2, '0', ha='center', fontsize=10)
    
    def _plot_per_depth_attention(
        self,
        attention_weights: List[np.ndarray],
        token_depths: np.ndarray,
    ) -> Figure:
        """绘制不同深度 Token 的注意力模式
        
        数学形式：
            A_{d→d'} = mean_{i:depth(i)=d, j:depth(j)=d'} A(i,j)
            展示 token 在不同 Quadtree 深度间的注意力交互
        
        Args:
            attention_weights: 每层注意力权重
            token_depths: 每个 token 的深度
        """
        unique_depths = np.unique(token_depths)
        n_depths = len(unique_depths)
        n_layers = len(attention_weights)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. 深度间注意力热图（最后一层）
        ax1 = axes[0]
        depth_attn_matrix = np.zeros((n_depths, n_depths))
        
        if len(attention_weights) > 0:
            last_attn = attention_weights[-1]
            if len(last_attn.shape) == 3:
                last_attn = last_attn.mean(axis=0)
            
            for i, d_i in enumerate(unique_depths):
                for j, d_j in enumerate(unique_depths):
                    mask_i = token_depths == d_i
                    mask_j = token_depths == d_j
                    
                    # 提取对应区块
                    block = last_attn[np.ix_(mask_i, mask_j)]
                    if block.size > 0:
                        depth_attn_matrix[i, j] = block.mean()
        
        im = ax1.imshow(depth_attn_matrix, cmap='Blues', aspect='auto')
        ax1.set_xlabel("Target Depth", fontsize=11)
        ax1.set_ylabel("Source Depth", fontsize=11)
        ax1.set_xticks(np.arange(n_depths))
        ax1.set_yticks(np.arange(n_depths))
        ax1.set_xticklabels([f"d={d}" for d in unique_depths])
        ax1.set_yticklabels([f"d={d}" for d in unique_depths])
        
        # 数值标注
        for i in range(n_depths):
            for j in range(n_depths):
                ax1.text(j, i, f'{depth_attn_matrix[i, j]:.3f}',
                        ha='center', va='center', fontsize=9)
        
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_title("Depth-to-Depth Attention (Last Layer)", fontsize=12, fontweight='bold')
        
        # 2. 每层深度注意力演化（堆叠面积图）
        ax2 = axes[1]
        
        # 计算每层每深度的平均接收注意力
        depth_receiving = np.zeros((n_layers, n_depths))
        for l, attn in enumerate(attention_weights):
            if len(attn.shape) == 3:
                attn = attn.mean(axis=0)
            for i, d in enumerate(unique_depths):
                mask = token_depths == d
                if mask.sum() > 0 and attn.shape[0] > mask.sum():
                    depth_receiving[l, i] = attn[:, mask].sum(axis=1).mean()
        
        # 堆叠面积图
        layers_x = np.arange(n_layers)
        colors = plt.get_cmap('viridis')(np.linspace(0, 1, n_depths))
        
        ax2.stackplot(layers_x, depth_receiving.T, labels=[f'd={d}' for d in unique_depths],
                     colors=colors, alpha=0.8)
        ax2.set_xlabel("Layer", fontsize=11)
        ax2.set_ylabel("Total Attention Received", fontsize=11)
        ax2.legend(loc='upper left', fontsize=9)
        ax2.set_title("Attention Flow by Depth", fontsize=12, fontweight='bold')
        
        # 3. 深度 Token 分布
        ax3 = axes[2]
        depth_counts = [np.sum(token_depths == d) for d in unique_depths]
        colors = plt.get_cmap('viridis')(np.linspace(0, 1, n_depths))
        
        bars = ax3.bar(unique_depths, depth_counts, color=colors, edgecolor='white')
        for bar, count in zip(bars, depth_counts):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(count), ha='center', fontsize=10)
        
        ax3.set_xlabel("Quadtree Depth", fontsize=11)
        ax3.set_ylabel("Token Count", fontsize=11)
        ax3.set_title("Token Distribution by Depth", fontsize=12, fontweight='bold')
        
        fig.suptitle("L3: Per-Depth Attention Analysis", fontsize=14, fontweight='bold')
        safe_tight_layout()
        return fig
    
    def _plot_hilbert_locality(
        self,
        locality_score: float,
        locality_by_layer: Optional[List[float]] = None,
    ) -> Figure:
        """绘制 Hilbert 局部性评分
        
        数学形式：
            Locality = mean_{i,j: |h(i)-h(j)|<k} A(i,j) / mean A
            其中 h(i) 是 token i 在 Hilbert 曲线上的位置
        
        Args:
            locality_score: 整体局部性评分
            locality_by_layer: 每层局部性评分
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. 整体局部性仪表
        ax1 = axes[0]
        self._draw_locality_gauge(ax1, locality_score)
        ax1.set_title("Overall Hilbert Locality", fontsize=12, fontweight='bold')
        
        # 2. 每层局部性
        ax2 = axes[1]
        if locality_by_layer:
            n_layers = len(locality_by_layer)
            layers = np.arange(n_layers)
            
            colors = ['#2ecc71' if s > 0.7 else '#f39c12' if s > 0.4 else '#e74c3c'
                     for s in locality_by_layer]
            
            bars = ax2.bar(layers, locality_by_layer, color=colors, edgecolor='white')
            ax2.axhline(0.7, color='green', linestyle='--', alpha=0.5, label='Good (0.7)')
            ax2.axhline(0.4, color='orange', linestyle='--', alpha=0.5, label='Weak (0.4)')
            
            for bar, score in zip(bars, locality_by_layer):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f'{score:.2f}', ha='center', fontsize=9)
            
            ax2.set_xlabel("Layer", fontsize=11)
            ax2.set_ylabel("Locality Score", fontsize=11)
            ax2.set_xticks(layers)
            ax2.set_ylim(0, 1.1)
            ax2.legend(loc='lower right', fontsize=9)
        else:
            ax2.text(0.5, 0.5, f"Overall Locality: {locality_score:.3f}\n\n(Per-layer data not available)",
                    ha='center', va='center', fontsize=12, transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        ax2.set_title("Locality by Layer", fontsize=12, fontweight='bold')
        
        # 说明
        fig.text(0.5, 0.02,
                "Hilbert Locality: Higher values indicate attention respects space-filling curve proximity",
                ha='center', fontsize=10, style='italic')
        
        fig.suptitle("L3: Hilbert Curve Locality Analysis", fontsize=14, fontweight='bold')
        safe_tight_layout(rect=[0, 0.05, 1, 0.95])
        return fig
    
    def _draw_locality_gauge(self, ax, value: float):
        """绘制局部性评分仪表盘"""
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 半圆仪表
        theta = np.linspace(0, np.pi, 100)
        r = 1.0
        
        # 背景弧（分三段颜色）
        segments = [
            (0, np.pi/3, '#e74c3c'),        # 红色：低
            (np.pi/3, 2*np.pi/3, '#f39c12'), # 橙色：中
            (2*np.pi/3, np.pi, '#2ecc71'),    # 绿色：高
        ]
        
        for start, end, color in segments:
            t = np.linspace(start, end, 30)
            ax.fill_between(r * np.cos(t), [0]*len(t), r * np.sin(t),
                           color=color, alpha=0.3)
            ax.plot(r * np.cos(t), r * np.sin(t), color=color, linewidth=3)
        
        # 指针（值范围 0 到 1，映射到 π 到 0）
        angle = np.pi * (1 - value)
        pointer_len = 0.85
        ax.arrow(0, 0, pointer_len * np.cos(angle), pointer_len * np.sin(angle),
                head_width=0.08, head_length=0.05, fc='black', ec='black')
        
        # 数值显示
        ax.text(0, 0.3, f'{value:.3f}', ha='center', va='center',
               fontsize=20, fontweight='bold')
        
        # 状态文字
        if value > 0.7:
            status = "Strong Locality"
            color = '#2ecc71'
        elif value > 0.4:
            status = "Moderate"
            color = '#f39c12'
        else:
            status = "Weak Locality"
            color = '#e74c3c'
        
        ax.text(0, -0.3, status, ha='center', va='center',
               fontsize=12, fontweight='bold', color=color)
        
        # 标签
        ax.text(-1.2, 0, '0', ha='center', fontsize=10)
        ax.text(1.2, 0, '1', ha='center', fontsize=10)
        ax.text(0, 1.2, '0.5', ha='center', fontsize=10)

    def _plot_summary(self, metrics: Any) -> Figure:
        """绘制 L3 注意力机制摘要"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # 构建摘要表格
        n_layers = len(metrics.per_layer_entropy) if metrics.per_layer_entropy else 0
        n_heads = len(metrics.head_utilization[0]) if metrics.head_utilization else 0
        
        # 获取 LCA 和 Hilbert 局部性信息
        lca_correlation = getattr(metrics, 'lca_attention_correlation', None)
        hilbert_locality = getattr(metrics, 'hilbert_locality_score', None)
        
        lca_str = f"{lca_correlation:>8.4f}" if lca_correlation is not None else "    N/A "
        locality_str = f"{hilbert_locality:>8.4f}" if hilbert_locality is not None else "    N/A "
        
        # 状态判定
        def get_status(value, thresholds, labels):
            if value is None:
                return "N/A", "gray"
            for thresh, label, color in thresholds:
                if value >= thresh:
                    return label, color
            return labels[-1][0], labels[-1][1]
        
        entropy_status, entropy_color = get_status(
            metrics.avg_entropy,
            [(0.5, "Balanced", "#2ecc71"), (0.2, "Focused", "#f39c12")],
            [("Very Focused", "#e74c3c")]
        )
        
        dead_head_status, dh_color = get_status(
            1 - metrics.dead_head_ratio,
            [(0.9, "Healthy", "#2ecc71"), (0.7, "Moderate", "#f39c12")],
            [("Concerning", "#e74c3c")]
        )
        
        summary_text = f"""
┌─────────────────────────────────────────────────────────────┐
│              L3 Attention Mechanism Summary                  │
├─────────────────────────────────────────────────────────────┤
│  Architecture                                                │
│    • Layers: {n_layers:>3}                                                │
│    • Heads per Layer: {n_heads:>3}                                       │
│    • Total Heads: {n_layers * n_heads:>5}                                          │
├─────────────────────────────────────────────────────────────┤
│  Attention Entropy                          [{entropy_status:^12}]   │
│    • Average Entropy: {metrics.avg_entropy:>8.4f}                               │
│    • Min Layer Entropy: {min(metrics.per_layer_entropy) if metrics.per_layer_entropy else 0:>8.4f}                             │
│    • Max Layer Entropy: {max(metrics.per_layer_entropy) if metrics.per_layer_entropy else 0:>8.4f}                             │
├─────────────────────────────────────────────────────────────┤
│  Head Utilization                           [{dead_head_status:^12}]   │
│    • Dead Head Ratio: {metrics.dead_head_ratio:>8.1%}                               │
│    • Active Heads: {int((1 - metrics.dead_head_ratio) * n_layers * n_heads):>5} / {n_layers * n_heads}                                  │
├─────────────────────────────────────────────────────────────┤
│  CLS Token Analysis                                          │
│    • CLS Coverage: {metrics.cls_attention_coverage:>8.1%}                                 │
│    • CLS Entropy: {metrics.cls_attention_entropy:>8.4f}                                │
├─────────────────────────────────────────────────────────────┤
│  Hilbert Locality Analysis (NEW)                             │
│    • LCA-Attention Correlation: {lca_str}                          │
│    • Hilbert Locality Score: {locality_str}                             │
└─────────────────────────────────────────────────────────────┘
        """
        
        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
               fontsize=11, va='center', ha='center',
               family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        
        return fig
