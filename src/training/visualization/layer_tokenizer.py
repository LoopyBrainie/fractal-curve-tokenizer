#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
L2 Tokenizer 行为可视化层 (Tokenizer Behavior Visualization)

数学形式化
============

本模块可视化 L2TokenizerMetrics 的评估结果：

1. **Token 数量分布** (Token Count Distribution)
   经验分布: $\hat{P}(N=n) = \frac{\text{count}(N=n)}{B}$
   健康指标: 分布应集中但有适度变化

2. **深度分布** (Depth Distribution)
   深度占比: $p_d = N_d / \sum_{d'} N_{d'}$
   深度熵: $H_d = -\sum_d p_d \log p_d$
   高熵 = 多尺度均衡利用

3. **GumbelTopKSplitter 决策分析** (NEW)
   分割概率: $\pi_k = \text{softmax}((\log\alpha_k + g_k) / \tau)$
   展示温度 τ 对决策硬度的影响

4. **空间覆盖率** (Spatial Coverage)
   覆盖率: $\text{Coverage} = \sum_i \text{Area}(R_i) / (H \times W)$
   应接近 1.0，过低表示遗漏区域

5. **内容-Token 相关性** (Content-Token Correlation)
   $\rho = \text{Corr}(\text{Complexity}(x), N_{tokens}(x))$
   正相关表明模型学会了自适应分割

6. **每类 Token 分析** (Per-Class Token Analysis)
   不同类别的平均 token 数，揭示类别复杂度差异

Author: GitHub Copilot
Date: 2026-01-11
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, FancyBboxPatch

from .base import (
    FigureConfig,
    VisualizationLayer,
    VisualizationResult,
    compute_entropy,
    safe_tight_layout,
    truncate_labels,
)

# 尝试导入 evaluation_layers 中的数据类
try:
    from ..evaluation_layers import L2TokenizerMetrics
except ImportError:
    L2TokenizerMetrics = None


class L2TokenizerVisualizer(VisualizationLayer):
    """L2 Tokenizer 行为可视化器
    
    数学形式化：
        V₂: L2TokenizerMetrics → {深度分布, Token分布, 空间分析, ...}
    
    输入：
        L2TokenizerMetrics 包含:
        - avg_tokens, min_tokens, max_tokens, std_tokens
        - depth_distribution: Dict[int, float]
        - depth_entropy: float
        - spatial_coverage_ratio: float
        - content_token_correlation: float
        - per_class_avg_tokens: Dict[int, float]
    
    输出：
        VisualizationResult 包含多个 Figure
    """
    
    LAYER_NAME = "L2_Tokenizer"
    LAYER_DESCRIPTION = "Tokenizer 行为可视化：深度分布、Token 统计、空间覆盖"
    
    def visualize(
        self,
        metrics: Any,  # L2TokenizerMetrics
        class_names: Optional[List[str]] = None,
        token_counts: Optional[List[int]] = None,
        sample_regions: Optional[List[np.ndarray]] = None,
        splitter_logits: Optional[np.ndarray] = None,
        **kwargs,
    ) -> VisualizationResult:
        """生成 L2 Tokenizer 可视化
        
        Args:
            metrics: L2TokenizerMetrics 评估结果
            class_names: 类别名称列表
            token_counts: 各样本的 token 数量列表（可选，用于分布图）
            sample_regions: 样本区域信息列表（可选，用于空间可视化）
            splitter_logits: GumbelTopKSplitter 的原始 logits（可选）
            **kwargs: 额外参数
            
        Returns:
            VisualizationResult
        """
        result = VisualizationResult(layer_name=self.LAYER_NAME)
        
        # 1. 深度分布
        if metrics.depth_distribution:
            fig = self._plot_depth_distribution(
                metrics.depth_distribution,
                metrics.depth_entropy,
            )
            result.figures.append(fig)
            result.names.append("L2_depth_distribution")
            result.descriptions.append("Token 深度（尺度）分布")
        
        # 2. Token 数量分布
        if token_counts is not None:
            fig = self._plot_token_distribution(
                token_counts,
                metrics.avg_tokens,
                metrics.std_tokens,
            )
            result.figures.append(fig)
            result.names.append("L2_token_distribution")
            result.descriptions.append("每样本 Token 数量分布")
        
        # 3. 每类 Token 分析
        if metrics.per_class_avg_tokens:
            fig = self._plot_per_class_tokens(
                metrics.per_class_avg_tokens,
                class_names,
            )
            result.figures.append(fig)
            result.names.append("L2_per_class_tokens")
            result.descriptions.append("各类别平均 Token 数")
        
        # 4. 空间覆盖分析
        if sample_regions is not None and len(sample_regions) > 0:
            fig = self._plot_spatial_coverage(
                sample_regions,
                metrics.spatial_coverage_ratio,
            )
            result.figures.append(fig)
            result.names.append("L2_spatial_coverage")
            result.descriptions.append("Token 空间覆盖可视化")
        
        # 5. 深度坍缩诊断 (新增)
        if hasattr(metrics, 'depth_kl_from_uniform'):
            fig = self._plot_depth_collapse_diagnosis(metrics)
            result.figures.append(fig)
            result.names.append("L2_depth_collapse")
            result.descriptions.append("深度坍缩诊断")
        
        # 6. Token 利用效率 (新增)
        if hasattr(metrics, 'token_utilization_score'):
            fig = self._plot_token_utilization(metrics)
            result.figures.append(fig)
            result.names.append("L2_token_utilization")
            result.descriptions.append("Token 利用效率分析")
        
        # 7. 深度贡献分析 (新增)
        if hasattr(metrics, 'depth_contribution') and metrics.depth_contribution:
            fig = self._plot_depth_contribution(metrics.depth_contribution)
            result.figures.append(fig)
            result.names.append("L2_depth_contribution")
            result.descriptions.append("深度贡献度分析")
        
        # 8. 综合摘要
        fig = self._plot_summary(metrics)
        result.figures.append(fig)
        result.names.append("L2_summary")
        result.descriptions.append("L2 Tokenizer 行为摘要")
        
        return result
    
    def _plot_depth_distribution(
        self,
        depth_dist: Dict[int, float],
        entropy: float,
    ) -> Figure:
        """绘制深度分布图
        
        数学形式：
            柱状图表示 p_d = N_d / sum(N_d')
            熵 H = -sum(p_d * log(p_d)) 衡量分布均匀性
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 排序深度
        depths = sorted(depth_dist.keys())
        percentages = [depth_dist[d] for d in depths]
        
        # 1. 条形图
        ax1 = axes[0]
        colors = plt.get_cmap('viridis')(np.linspace(0.2, 0.9, len(depths)))
        
        bars = ax1.bar(depths, percentages, color=colors, edgecolor='white', linewidth=1.5)
        ax1.set_xlabel("Depth (Scale Level)", fontsize=11)
        ax1.set_ylabel("Percentage", fontsize=11)
        ax1.set_ylim(0, max(percentages) * 1.2)
        
        # 数值标注
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                    f'{pct:.1%}', ha='center', fontsize=10)
        
        # 理想均匀分布参考线
        if len(depths) > 0:
            uniform_pct = 1.0 / len(depths)
            ax1.axhline(uniform_pct, color='red', linestyle='--', alpha=0.7,
                       label=f'Uniform: {uniform_pct:.1%}')
            ax1.legend()
        
        ax1.set_title("Depth Distribution", fontsize=12, fontweight='bold')
        
        # 2. 饼图
        ax2 = axes[1]
        labels = [f"D{d}" for d in depths]
        explode = [0.02] * len(depths)
        
        wedges, texts, autotexts = ax2.pie(
            percentages, labels=labels, autopct='%1.1f%%',
            colors=colors, explode=explode, startangle=90,
            textprops={'fontsize': 10}
        )
        
        # 熵标注
        max_entropy = np.log(len(depths)) if len(depths) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        ax2.text(0, -1.3, f"Entropy: {entropy:.3f}\n"
                          f"Normalized: {normalized_entropy:.2%}\n"
                          f"(Higher = More Balanced)",
                ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        ax2.set_title("Depth Proportion", fontsize=12, fontweight='bold')
        
        fig.suptitle("L2: Token Depth (Scale) Analysis", fontsize=14, fontweight='bold')
        safe_tight_layout()
        return fig
    
    def _plot_token_distribution(
        self,
        token_counts: List[int],
        avg: float,
        std: float,
    ) -> Figure:
        """绘制 Token 数量分布
        
        数学形式：
            直方图表示 P(N=n) 的经验分布
            均值和标准差反映 tokenizer 行为稳定性
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        token_counts = np.array(token_counts)
        
        # 1. 直方图
        ax1 = axes[0]
        n_bins = min(50, max(10, len(np.unique(token_counts))))
        
        ax1.hist(token_counts, bins=n_bins, color='#3498db', 
                edgecolor='white', alpha=0.8, density=True)
        ax1.axvline(avg, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {avg:.1f}')
        ax1.axvline(avg - std, color='orange', linestyle=':', alpha=0.7)
        ax1.axvline(avg + std, color='orange', linestyle=':', alpha=0.7,
                   label=f'+/-1 Std: {std:.1f}')
        
        ax1.set_xlabel("Token Count per Sample", fontsize=11)
        ax1.set_ylabel("Density", fontsize=11)
        ax1.legend()
        ax1.set_title("Token Count Distribution", fontsize=12, fontweight='bold')
        
        # 2. 箱线图 + 散点
        ax2 = axes[1]
        bp = ax2.boxplot(token_counts, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][0].set_alpha(0.6)
        
        # 抖动散点
        jitter = np.random.normal(1, 0.04, len(token_counts))
        sample_size = min(500, len(token_counts))
        indices = np.random.choice(len(token_counts), sample_size, replace=False)
        ax2.scatter(jitter[indices], token_counts[indices], 
                   alpha=0.3, s=10, color='blue')
        
        ax2.set_ylabel("Token Count", fontsize=11)
        ax2.set_xticklabels(['All Samples'])
        
        # 统计信息
        stats_text = (
            f"N: {len(token_counts)}\n"
            f"Mean: {avg:.1f}\n"
            f"Std: {std:.1f}\n"
            f"Min: {token_counts.min()}\n"
            f"Max: {token_counts.max()}\n"
            f"Median: {np.median(token_counts):.1f}"
        )
        ax2.text(1.4, 0.5, stats_text, transform=ax2.transAxes,
                fontsize=10, va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        ax2.set_title("Token Count Statistics", fontsize=12, fontweight='bold')
        
        fig.suptitle("L2: Token Count Analysis", fontsize=14, fontweight='bold')
        safe_tight_layout()
        return fig
    
    def _plot_per_class_tokens(
        self,
        per_class_tokens: Dict[int, float],
        class_names: Optional[List[str]],
    ) -> Figure:
        """绘制每类平均 Token 数
        
        揭示不同类别的复杂度差异：
        - 复杂类别（纹理丰富）应有更多 token
        - 简单类别（纯色/简单形状）应有较少 token
        """
        classes = list(per_class_tokens.keys())
        avg_tokens = [per_class_tokens[c] for c in classes]
        
        # 按 token 数排序
        sorted_indices = np.argsort(avg_tokens)
        sorted_classes = [classes[i] for i in sorted_indices]
        sorted_tokens = [avg_tokens[i] for i in sorted_indices]
        
        n_classes = len(classes)
        
        # 如果类别太多（>50），只显示最少和最多的类
        show_top_bottom = n_classes > 50
        if show_top_bottom:
            n_show = 25  # 显示最少和最多各 25 个
            bottom_classes = sorted_classes[:n_show]
            bottom_tokens = sorted_tokens[:n_show]
            top_classes = sorted_classes[-n_show:]
            top_tokens = sorted_tokens[-n_show:]
            
            display_classes = bottom_classes + top_classes
            display_tokens = bottom_tokens + top_tokens
            display_labels_raw = bottom_classes + ['...'] + top_classes
        else:
            display_classes = sorted_classes
            display_tokens = sorted_tokens
            display_labels_raw = sorted_classes
        
        # 标签
        if class_names:
            labels = []
            for c in display_labels_raw:
                if c == '...':
                    labels.append(f'... ({n_classes - 50} classes omitted) ...')
                elif isinstance(c, int) and c < len(class_names):
                    labels.append(class_names[c])
                else:
                    labels.append(f"C{c}" if isinstance(c, int) else str(c))
        else:
            labels = []
            for c in display_labels_raw:
                if c == '...':
                    labels.append(f'... ({n_classes - 50} classes omitted) ...')
                else:
                    labels.append(f"Class {c}")
        labels = truncate_labels(labels, max_len=15)
        
        # 创建图表 - 限制最大高度
        n_display = len(labels)
        fig_height = min(20, max(6, n_display * 0.25))
        fig, ax = plt.subplots(figsize=(12, fig_height))
        
        # 颜色映射：token 少的浅色，多的深色
        cmap = plt.get_cmap('YlOrRd')
        min_t, max_t = min(display_tokens), max(display_tokens)
        if max_t > min_t:
            colors = cmap((np.array(display_tokens) - min_t) / (max_t - min_t) * 0.8 + 0.1)
        else:
            colors = [cmap(0.5)] * len(display_tokens)
        
        # 处理分隔行
        if show_top_bottom:
            y_pos = list(range(n_show)) + [n_show] + list(range(n_show + 1, 2 * n_show + 1))
            bar_tokens = bottom_tokens + [0] + top_tokens
            bar_colors = list(colors[:n_show]) + ['lightgray'] + list(colors[n_show:])
        else:
            y_pos = np.arange(n_display)
            bar_tokens = display_tokens
            bar_colors = colors
        
        bars = ax.barh(y_pos, bar_tokens, color=bar_colors, edgecolor='white')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=max(6, 10 - n_display // 30))
        ax.set_xlabel("Average Token Count", fontsize=11)
        
        # 均值参考线
        mean_tokens = np.mean(display_tokens)
        ax.axvline(mean_tokens, color='blue', linestyle='--', 
                  label=f'Mean: {mean_tokens:.1f}')
        ax.legend(loc='lower right')
        
        # 数值标注
        for bar, tokens in zip(bars, bar_tokens):
            if tokens == 0:  # 跳过分隔行
                continue
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                   f'{tokens:.1f}', va='center', fontsize=max(6, 9 - n_display // 40))
        
        title = "Average Token Count per Class\n(Reflects Visual Complexity)"
        if show_top_bottom:
            title = f"Average Token Count per Class (Top/Bottom {n_show} of {n_classes})"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        safe_tight_layout()
        return fig
    
    def _plot_spatial_coverage(
        self,
        sample_regions: List[np.ndarray],
        coverage_ratio: float,
    ) -> Figure:
        """绘制空间覆盖可视化
        
        展示 token 在图像空间中的分布：
        - 完整覆盖 = 1.0
        - 过低覆盖可能表示遗漏重要区域
        """
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        axes = axes.flatten()
        
        # 显示前 8 个样本
        n_samples = min(8, len(sample_regions))
        
        for idx in range(n_samples):
            ax = axes[idx]
            regions = sample_regions[idx]  # [N, 4] - (x1, y1, x2, y2)
            
            if len(regions) == 0:
                ax.text(0.5, 0.5, "No regions", ha='center', va='center')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                continue
            
            # 假设归一化坐标 [0, 1]
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)  # 翻转 y 轴
            
            # 根据区域大小着色
            areas = (regions[:, 2] - regions[:, 0]) * (regions[:, 3] - regions[:, 1])
            cmap = plt.get_cmap('viridis')
            
            for i, (x1, y1, x2, y2) in enumerate(regions):
                w, h = x2 - x1, y2 - y1
                color = cmap(areas[i] / areas.max() if areas.max() > 0 else 0.5)
                rect = FancyBboxPatch((x1, y1), w, h,
                                      boxstyle="round,pad=0.01",
                                      facecolor=color, alpha=0.6,
                                      edgecolor='white', linewidth=0.5)
                ax.add_patch(rect)
            
            ax.set_title(f"Sample {idx+1}\n{len(regions)} tokens", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
        
        # 隐藏多余子图
        for idx in range(n_samples, 8):
            axes[idx].set_visible(False)
        
        fig.suptitle(f"L2: Spatial Coverage (Overall: {coverage_ratio:.1%})",
                    fontsize=14, fontweight='bold')
        safe_tight_layout()
        return fig
    
    def _plot_summary(self, metrics: Any) -> Figure:
        """绘制 L2 Tokenizer 行为摘要"""
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        
        # 1. Token 统计仪表盘
        ax1 = axes[0]
        labels = ['Avg', 'Min', 'Max', 'Std']
        values = [metrics.avg_tokens, metrics.min_tokens, 
                 metrics.max_tokens, metrics.std_tokens]
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        bars = ax1.bar(labels, values, color=colors, edgecolor='white')
        
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.1f}', ha='center', fontsize=11)
        
        ax1.set_ylabel("Token Count", fontsize=11)
        ax1.set_title("Token Statistics", fontsize=12, fontweight='bold')
        
        # 2. 深度熵仪表
        ax2 = axes[1]
        entropy = metrics.depth_entropy
        max_entropy = np.log(len(metrics.depth_distribution)) if metrics.depth_distribution else 1.0
        normalized = entropy / max_entropy if max_entropy > 0 else 0
        
        # 半圆仪表
        theta = np.linspace(0, np.pi, 100)
        ax2.fill_between(np.cos(theta), np.sin(theta), 0, color='lightgray', alpha=0.3)
        
        # 指针
        angle = np.pi * (1 - normalized)
        ax2.arrow(0, 0, 0.7 * np.cos(angle), 0.7 * np.sin(angle),
                 head_width=0.08, head_length=0.05, fc='red', ec='red')
        ax2.scatter([0], [0], s=100, c='black', zorder=5)
        
        ax2.set_xlim(-1.2, 1.2)
        ax2.set_ylim(-0.2, 1.2)
        ax2.set_aspect('equal')
        ax2.axis('off')
        
        ax2.text(0, -0.15, f"Depth Entropy: {entropy:.3f}\n(Normalized: {normalized:.1%})",
                ha='center', fontsize=11)
        ax2.set_title("Depth Distribution Balance", fontsize=12, fontweight='bold')
        
        # 3. 关键指标总结
        ax3 = axes[2]
        ax3.axis('off')
        
        # 文本表格
        info_text = (
            f"┌─────────────────────────────────┐\n"
            f"│  L2 Tokenizer Summary           │\n"
            f"├─────────────────────────────────┤\n"
            f"│  Avg Tokens: {metrics.avg_tokens:>8.1f}           │\n"
            f"│  Token Range: [{metrics.min_tokens}, {metrics.max_tokens}]        │\n"
            f"│  Token Std: {metrics.std_tokens:>8.2f}            │\n"
            f"│  Depth Entropy: {metrics.depth_entropy:>8.3f}        │\n"
            f"│  Coverage: {metrics.spatial_coverage_ratio:>8.1%}          │\n"
            f"|  Content-Token rho: {metrics.content_token_correlation:>6.3f}      |\n"
            f"└─────────────────────────────────┘"
        )
        
        ax3.text(0.5, 0.5, info_text, transform=ax3.transAxes,
                fontsize=11, va='center', ha='center',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        
        ax3.set_title("Key Metrics", fontsize=12, fontweight='bold')
        
        fig.suptitle("L2 Tokenizer Behavior Summary", fontsize=14, fontweight='bold')
        safe_tight_layout()
        return fig
    
    def _plot_depth_collapse_diagnosis(self, metrics: Any) -> Figure:
        """深度坍缩诊断可视化
        
        数学形式：
            KL_uniform = D_KL(π || U) = Σ_d π_d log(π_d / (1/D))
            collapse_detected = KL > 1.0 ∨ entropy_ratio < 0.3
        """
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        
        # 1. KL 散度仪表盘
        ax_kl = axes[0]
        kl_val = getattr(metrics, 'depth_kl_from_uniform', 0)
        self._draw_gauge(
            ax_kl,
            value=min(kl_val, 3.0),  # 限制显示范围
            min_val=0,
            max_val=3.0,
            title="KL from Uniform",
            thresholds=[(0.5, 'green'), (1.0, 'yellow'), (2.0, 'red')],
        )
        ax_kl.text(0.5, -0.15, f"KL = {kl_val:.3f}\n(阈值: 1.0)",
                  ha='center', transform=ax_kl.transAxes, fontsize=10)
        
        # 2. 熵比率仪表盘
        ax_entropy = axes[1]
        entropy_ratio = getattr(metrics, 'depth_entropy_ratio', 0)
        self._draw_gauge(
            ax_entropy,
            value=entropy_ratio,
            min_val=0,
            max_val=1.0,
            title="Entropy Ratio (H/H_max)",
            thresholds=[(0.3, 'red'), (0.5, 'yellow'), (0.7, 'green')],
        )
        ax_entropy.text(0.5, -0.15, f"Ratio = {entropy_ratio:.2%}\n(阈值: 30%)",
                       ha='center', transform=ax_entropy.transAxes, fontsize=10)
        
        # 3. 状态指示
        ax_status = axes[2]
        ax_status.axis('off')
        
        collapse = getattr(metrics, 'depth_collapse_detected', False)
        
        if collapse:
            status_text = "[!] Depth Collapse"
            status_color = 'red'
            detail = "Severe depth distribution imbalance\nMay cause loss of multi-scale features"
            recommendation = "Recommendations:\n- Enable LOG_COMPENSATION\n- Increase DEPTH_KL_WEIGHT\n- Check splitter training"
        else:
            status_text = "[OK] Depth Healthy"
            status_color = 'green'
            detail = "Depth distribution is balanced\nMulti-scale features normal"
            recommendation = ""
        
        ax_status.text(0.5, 0.75, status_text, ha='center', va='center',
                      fontsize=24, fontweight='bold', color=status_color,
                      transform=ax_status.transAxes)
        ax_status.text(0.5, 0.50, detail, ha='center', va='center',
                      fontsize=11, transform=ax_status.transAxes)
        ax_status.text(0.5, 0.20, recommendation, ha='center', va='center',
                      fontsize=10, color='gray', transform=ax_status.transAxes)
        
        fig.suptitle("Depth Collapse Diagnosis (I21)", fontsize=14, fontweight='bold')
        safe_tight_layout()
        self._add_watermark(fig)
        return fig
    
    def _plot_token_utilization(self, metrics: Any) -> Figure:
        """Token 利用效率分析
        
        数学形式：
            Score = (depth_entropy_ratio + |content_correlation| + adaptive_ratio) / 3
        """
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        
        # 1. 利用效率分数
        ax_score = axes[0]
        score = getattr(metrics, 'token_utilization_score', 0)
        self._draw_gauge(
            ax_score,
            value=score,
            min_val=0,
            max_val=1.0,
            title="Token 利用效率",
            thresholds=[(0.3, 'red'), (0.5, 'yellow'), (0.7, 'green')],
        )
        
        # 2. 冗余比例
        ax_redund = axes[1]
        redundancy = getattr(metrics, 'redundancy_ratio', 0)
        
        labels = ['有效', '冗余']
        sizes = [1 - redundancy, redundancy]
        colors = ['#2ecc71', '#e74c3c']
        
        ax_redund.pie(sizes, labels=labels, autopct='%1.1f%%',
                     colors=colors, startangle=90)
        ax_redund.set_title(f"Token 冗余估计\n(冗余 = {redundancy:.1%})")
        
        # 3. 自适应程度
        ax_adapt = axes[2]
        adaptive = getattr(metrics, 'adaptive_ratio', 0)
        
        ax_adapt.bar(['Adaptive Ratio'], [adaptive], color='steelblue', edgecolor='black')
        ax_adapt.axhline(y=0.2, color='green', linestyle='--', label='健康阈值')
        ax_adapt.set_ylabel("std / mean")
        ax_adapt.set_title(f"自适应程度\n(std/mean = {adaptive:.3f})")
        ax_adapt.set_ylim(0, max(adaptive * 1.5, 0.5))
        ax_adapt.legend()
        
        fig.suptitle("Token 利用效率分析", fontsize=14, fontweight='bold')
        safe_tight_layout()
        self._add_watermark(fig)
        return fig
    
    def _plot_depth_contribution(self, depth_contribution: Dict) -> Figure:
        """深度贡献度分析
        
        展示各深度对分类的贡献
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        depths = sorted(depth_contribution.keys())
        
        # 1. Token 数量占比
        ax_tokens = axes[0]
        token_ratios = [depth_contribution[d].get('token_ratio', 0) for d in depths]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(depths)))
        
        ax_tokens.bar(depths, token_ratios, color=colors, edgecolor='black')
        ax_tokens.set_xlabel("深度")
        ax_tokens.set_ylabel("Token 占比")
        ax_tokens.set_title("各深度 Token 数量占比")
        ax_tokens.set_xticks(depths)
        
        for d, ratio in zip(depths, token_ratios):
            ax_tokens.text(d, ratio + 0.01, f"{ratio:.1%}", ha='center', fontsize=9)
        
        # 2. 估计面积贡献
        ax_area = axes[1]
        area_ratios = [depth_contribution[d].get('estimated_area_ratio', 0) for d in depths]
        
        # 归一化
        total_area = sum(area_ratios)
        if total_area > 0:
            area_ratios = [a / total_area for a in area_ratios]
        
        ax_area.bar(depths, area_ratios, color=colors, edgecolor='black')
        ax_area.set_xlabel("深度")
        ax_area.set_ylabel("面积贡献比")
        ax_area.set_title("各深度估计面积贡献\n(深度越深, patch 越小)")
        ax_area.set_xticks(depths)
        
        for d, ratio in zip(depths, area_ratios):
            ax_area.text(d, ratio + 0.01, f"{ratio:.1%}", ha='center', fontsize=9)
        
        fig.suptitle("深度贡献度分析", fontsize=14, fontweight='bold')
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
        thresholds: list,
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
        
        ax.text(0, -0.2, f"{value:.3f}", ha='center', va='top',
               fontsize=14, fontweight='bold')
        ax.text(0, 1.3, title, ha='center', va='bottom', fontsize=12)
        ax.text(-1.1, 0, f"{min_val}", ha='center', va='center', fontsize=9)
        ax.text(1.1, 0, f"{max_val}", ha='center', va='center', fontsize=9)