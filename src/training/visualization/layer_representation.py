#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
L4 特征表示可视化层 (Feature Representation Visualization)

数学形式化
============

本模块可视化 L4RepresentationMetrics 的评估结果：

1. **t-SNE 特征降维** (t-SNE Feature Visualization)
   映射: $f: \mathbb{R}^D \to \mathbb{R}^2$
   目标: 保持高维空间中的局部邻居关系
   
   KL 散度最小化:
   $KL(P||Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$

2. **类别可分性热图** (Class Separability Heatmap)
   可分性: $S_c = \|\mu_c - \mu\| / \sigma_c$
   高可分性 = 类别中心远离全局中心，类内方差小

3. **Fisher 判别比** (Fisher Discriminant Ratio)
   $FDR = \text{tr}(S_B) / \text{tr}(S_W)$
   $S_B$ = 类间散度矩阵
   $S_W$ = 类内散度矩阵

4. **特征范数分布** (Feature Norm Distribution)
   分析 $\|f_i\|_2$ 的分布，检测特征坍塌

5. **类别中心距离矩阵** (Class Center Distance Matrix)
   $D_{ij} = \|\mu_i - \mu_j\|_2$
   用于识别易混淆的类别对

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
    safe_tight_layout,
    truncate_labels,
)

# 尝试导入 evaluation_layers 中的数据类
try:
    from ..evaluation_layers import L4RepresentationMetrics
except ImportError:
    L4RepresentationMetrics = None


class L4RepresentationVisualizer(VisualizationLayer):
    """L4 特征表示可视化器
    
    数学形式化：
        V₄: L4RepresentationMetrics → {t-SNE, 可分性热图, Fisher比, ...}
    
    输入：
        L4RepresentationMetrics 包含:
        - fisher_discriminant_ratio: float
        - feature_mean_norm: float
        - feature_std: float
        - per_class_separability: Dict[int, float]
        - avg_separability: float
    
    额外输入（用于详细可视化）:
        - features: np.ndarray [N, D] - 提取的特征
        - labels: np.ndarray [N] - 对应的标签
        - class_centers: np.ndarray [C, D] - 类别中心
    
    输出：
        VisualizationResult 包含多个 Figure
    """
    
    LAYER_NAME = "L4_Representation"
    LAYER_DESCRIPTION = "特征表示可视化：t-SNE、类别可分性、Fisher 判别比"
    
    def visualize(
        self,
        metrics: Any,  # L4RepresentationMetrics
        class_names: Optional[List[str]] = None,
        features: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        class_centers: Optional[np.ndarray] = None,
        **kwargs,
    ) -> VisualizationResult:
        """生成 L4 特征表示可视化
        
        Args:
            metrics: L4RepresentationMetrics 评估结果
            class_names: 类别名称列表
            features: 特征数组 [N, D]（可选，用于 t-SNE）
            labels: 标签数组 [N]（可选，用于 t-SNE）
            class_centers: 类别中心 [C, D]（可选）
            **kwargs: 额外参数
            
        Returns:
            VisualizationResult
        """
        result = VisualizationResult(layer_name=self.LAYER_NAME)
        
        # 1. t-SNE 可视化
        if features is not None and labels is not None:
            fig = self._plot_tsne(features, labels, class_names)
            if fig is not None:
                result.figures.append(fig)
                result.names.append("L4_tsne")
                result.descriptions.append("t-SNE 特征降维可视化")
        
        # 2. 类别可分性
        if metrics.per_class_separability:
            fig = self._plot_class_separability(
                metrics.per_class_separability,
                class_names,
                metrics.avg_separability,
            )
            result.figures.append(fig)
            result.names.append("L4_class_separability")
            result.descriptions.append("类别可分性分析")
        
        # 3. 类别中心距离矩阵
        if class_centers is not None:
            fig = self._plot_class_distance_matrix(class_centers, class_names)
            result.figures.append(fig)
            result.names.append("L4_class_distance")
            result.descriptions.append("类别中心距离矩阵")
        
        # 4. 特征范数分布
        if features is not None and len(features) > 0:
            fig = self._plot_feature_norm_distribution(features, labels, class_names)
            result.figures.append(fig)
            result.names.append("L4_feature_norms")
            result.descriptions.append("特征范数分布")
        
        # 5. 综合摘要
        fig = self._plot_summary(metrics)
        result.figures.append(fig)
        result.names.append("L4_summary")
        result.descriptions.append("L4 特征表示摘要")
        
        return result
    
    def _plot_tsne(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        class_names: Optional[List[str]],
    ) -> Optional[Figure]:
        """绘制 t-SNE 特征降维可视化
        
        数学形式：
            t-SNE 最小化 P (高维) 和 Q (低维) 分布的 KL 散度
            保持局部邻居结构
        """
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            print("[WARN] sklearn not installed, skipping t-SNE")
            return None
        
        # 采样（如果样本太多）
        n_samples = features.shape[0]
        max_samples = 5000
        
        if n_samples > max_samples:
            indices = np.random.choice(n_samples, max_samples, replace=False)
            features = features[indices]
            labels = labels[indices]
        
        # t-SNE 降维
        print(f"[INFO] Running t-SNE on {len(features)} samples...")
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        features_2d = tsne.fit_transform(features)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 获取唯一类别
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        # 颜色映射
        cmap = plt.get_cmap('tab20' if n_classes <= 20 else 'nipy_spectral')
        colors = {label: cmap(i / n_classes) for i, label in enumerate(unique_labels)}
        
        # 绘制散点
        for label in unique_labels:
            mask = labels == label
            name = class_names[label] if class_names and label < len(class_names) else f"C{label}"
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                      c=[colors[label]], label=name[:15], alpha=0.6, s=10)
        
        # 图例（如果类别数不太多）
        if n_classes <= 20:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                     fontsize=8, ncol=1 if n_classes <= 10 else 2)
        
        ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
        ax.set_title(f"t-SNE Feature Visualization\n({len(features)} samples, {n_classes} classes)",
                    fontsize=13, fontweight='bold')
        
        # 网格
        ax.grid(alpha=0.3)
        
        safe_tight_layout()
        return fig
    
    def _plot_class_separability(
        self,
        per_class_sep: Dict[int, float],
        class_names: Optional[List[str]],
        avg_sep: float,
    ) -> Figure:
        """绘制类别可分性分析
        
        数学形式：
            可分性 S_c = ||μ_c - μ|| / σ_c
            高可分性 = 类别远离中心且紧凑
        """
        # 排序
        classes = list(per_class_sep.keys())
        separabilities = [per_class_sep[c] for c in classes]
        sorted_indices = np.argsort(separabilities)
        
        sorted_classes = [classes[i] for i in sorted_indices]
        sorted_seps = [separabilities[i] for i in sorted_indices]
        
        # 处理类别过多的情况：只显示 top 和 bottom 各 25 个
        n_classes = len(classes)
        n_show = 25
        show_top_bottom = n_classes > 50
        
        if show_top_bottom:
            # 最低的 n_show 个
            bottom_classes = sorted_classes[:n_show]
            bottom_seps = sorted_seps[:n_show]
            # 最高的 n_show 个
            top_classes = sorted_classes[-n_show:]
            top_seps = sorted_seps[-n_show:]
            # 合并：bottom + separator + top
            display_classes = bottom_classes + [None] + top_classes
            display_seps = bottom_seps + [0] + top_seps  # separator row
            display_labels_raw = bottom_classes + [None] + top_classes
        else:
            display_classes = sorted_classes
            display_seps = sorted_seps
            display_labels_raw = sorted_classes
        
        n_display = len(display_classes)
        
        # 标签
        if class_names:
            labels = []
            for c in display_labels_raw:
                if c is None:
                    labels.append("...")
                elif c < len(class_names):
                    labels.append(class_names[c])
                else:
                    labels.append(f"C{c}")
        else:
            labels = []
            for c in display_labels_raw:
                if c is None:
                    labels.append("...")
                else:
                    labels.append(f"Class {c}")
        labels = truncate_labels(labels, max_len=15)
        
        # 创建图表
        fig_height = max(6, n_display * 0.25)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        
        # 颜色映射
        cmap = plt.get_cmap('RdYlGn')
        min_s, max_s = min(sorted_seps), max(sorted_seps)
        if max_s > min_s:
            colors = []
            for i, sep in enumerate(display_seps):
                if show_top_bottom and i == n_show:  # separator
                    colors.append('white')
                else:
                    colors.append(cmap((sep - min_s) / (max_s - min_s)))
        else:
            colors = [cmap(0.5)] * n_display
        
        y_pos = np.arange(n_display)
        bars = ax.barh(y_pos, display_seps, color=colors, edgecolor='white')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=max(6, 10 - n_display // 30))
        ax.set_xlabel("Separability Score", fontsize=11)
        
        # 均值参考线
        ax.axvline(avg_sep, color='blue', linestyle='--',
                  label=f'Mean: {avg_sep:.2f}')
        ax.legend(loc='lower right')
        
        # 数值标注
        for i, (bar, sep) in enumerate(zip(bars, display_seps)):
            if show_top_bottom and i == n_show:  # separator row
                continue
            if i < 5 or i >= len(bars) - 5:
                ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                       f'{sep:.2f}', va='center', fontsize=8)
        
        title = "Class Separability (Higher = More Separable)"
        if show_top_bottom:
            title = f"Class Separability (Top/Bottom {n_show} of {n_classes})"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        safe_tight_layout()
        return fig
    
    def _plot_class_distance_matrix(
        self,
        class_centers: np.ndarray,
        class_names: Optional[List[str]],
    ) -> Figure:
        """绘制类别中心距离矩阵
        
        数学形式：
            D_{ij} = ||μ_i - μ_j||_2
            小距离 = 易混淆的类别对
        """
        n_classes_total = class_centers.shape[0]
        
        # 如果类别过多，采样显示
        max_display = 50
        if n_classes_total > max_display:
            # 均匀采样
            sample_indices = np.linspace(0, n_classes_total - 1, max_display, dtype=int)
            class_centers_display = class_centers[sample_indices]
            n_classes = max_display
            sampled = True
        else:
            sample_indices = np.arange(n_classes_total)
            class_centers_display = class_centers
            n_classes = n_classes_total
            sampled = False
        
        # 计算距离矩阵
        dist_matrix = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(n_classes):
                dist_matrix[i, j] = np.linalg.norm(class_centers_display[i] - class_centers_display[j])
        
        # 标签
        if class_names:
            labels = [class_names[sample_indices[i]][:8] if sample_indices[i] < len(class_names) else f"C{sample_indices[i]}"
                     for i in range(n_classes)]
        else:
            labels = [f"C{sample_indices[i]}" for i in range(n_classes)]
        
        # 创建图表
        fig_size = max(8, min(n_classes * 0.3, 15))  # 限制最大尺寸
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        # 热图
        im = ax.imshow(dist_matrix, cmap='viridis_r', aspect='auto')
        
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(labels, rotation=45, ha='right', 
                          fontsize=max(5, 9 - n_classes // 20))
        ax.set_yticklabels(labels, fontsize=max(5, 9 - n_classes // 20))
        
        # 颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('L2 Distance', fontsize=10)
        
        # 标记最近的类别对（非对角线）
        mask = ~np.eye(n_classes, dtype=bool)
        min_dist = dist_matrix[mask].min()
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and dist_matrix[i, j] < min_dist * 1.5:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                              fill=False, edgecolor='red', linewidth=2))
        
        title = "Class Center Distance Matrix\n(Red = Most Similar Pairs)"
        if sampled:
            title = f"Class Center Distance Matrix (Sampled {n_classes} of {n_classes_total})"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        safe_tight_layout()
        return fig
    
    def _plot_feature_norm_distribution(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        class_names: Optional[List[str]],
    ) -> Figure:
        """绘制特征范数分布
        
        数学形式：norm(f_i) 的分布
        坍塌特征：范数接近 0 或方差极小
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 计算范数
        norms = np.linalg.norm(features, axis=1)
        
        # 1. 整体分布
        ax1 = axes[0]
        ax1.hist(norms, bins=50, color='#3498db', edgecolor='white', alpha=0.8)
        ax1.axvline(np.mean(norms), color='red', linestyle='--',
                   label=f'Mean: {np.mean(norms):.2f}')
        ax1.axvline(np.median(norms), color='orange', linestyle=':',
                   label=f'Median: {np.median(norms):.2f}')
        
        ax1.set_xlabel("Feature L2 Norm", fontsize=11)
        ax1.set_ylabel("Count", fontsize=11)
        ax1.legend()
        ax1.set_title("Feature Norm Distribution", fontsize=12, fontweight='bold')
        
        # 2. 每类范数箱线图（如果类别数合理）
        ax2 = axes[1]
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        if n_classes <= 30:
            class_norms = [norms[labels == c] for c in unique_labels]
            bp = ax2.boxplot(class_norms, patch_artist=True)
            
            # 着色
            colors = plt.get_cmap('viridis')(np.linspace(0.2, 0.8, n_classes))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            # 标签
            if class_names:
                tick_labels = [class_names[c][:8] if c < len(class_names) else f"C{c}"
                              for c in unique_labels]
            else:
                tick_labels = [f"C{c}" for c in unique_labels]
            
            ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
            ax2.set_ylabel("Feature Norm", fontsize=11)
            ax2.set_title("Per-Class Norm Distribution", fontsize=12, fontweight='bold')
        else:
            # 太多类别，显示统计摘要
            ax2.text(0.5, 0.5,
                    f"Too many classes ({n_classes}) for boxplot\n\n"
                    f"Norm Statistics:\n"
                    f"  Mean: {np.mean(norms):.3f}\n"
                    f"  Std: {np.std(norms):.3f}\n"
                    f"  Min: {np.min(norms):.3f}\n"
                    f"  Max: {np.max(norms):.3f}",
                    ha='center', va='center', fontsize=11,
                    transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow'))
            ax2.axis('off')
        
        fig.suptitle("L4: Feature Norm Analysis", fontsize=14, fontweight='bold')
        safe_tight_layout()
        return fig
    
    def _plot_summary(self, metrics: Any) -> Figure:
        """绘制 L4 特征表示摘要"""
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        
        # 1. Fisher 判别比仪表
        ax1 = axes[0]
        fdr = metrics.fisher_discriminant_ratio
        
        # 简单的仪表盘效果
        theta = np.linspace(0, np.pi, 100)
        ax1.fill_between(np.cos(theta), np.sin(theta), 0, color='lightgray', alpha=0.3)
        
        # 根据 FDR 值确定指针位置（假设好的 FDR > 1）
        normalized = min(1.0, fdr / 5.0)  # 归一化到 [0, 1]
        angle = np.pi * (1 - normalized)
        ax1.arrow(0, 0, 0.7 * np.cos(angle), 0.7 * np.sin(angle),
                 head_width=0.08, head_length=0.05, fc='red', ec='red')
        ax1.scatter([0], [0], s=100, c='black', zorder=5)
        
        # 区域着色
        for i, (start, end, color) in enumerate([
            (0, 0.33, '#e74c3c'),      # 差
            (0.33, 0.66, '#f39c12'),   # 中
            (0.66, 1.0, '#2ecc71'),    # 好
        ]):
            t1 = np.pi * (1 - end)
            t2 = np.pi * (1 - start)
            theta_fill = np.linspace(t1, t2, 20)
            ax1.fill_between(0.9 * np.cos(theta_fill), 0.9 * np.sin(theta_fill), 0,
                            color=color, alpha=0.3)
        
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-0.2, 1.2)
        ax1.set_aspect('equal')
        ax1.axis('off')
        
        ax1.text(0, -0.15, f"Fisher Discriminant Ratio\n{fdr:.3f}",
                ha='center', fontsize=11)
        ax1.set_title("Class Separability (FDR)", fontsize=12, fontweight='bold')
        
        # 2. 特征统计
        ax2 = axes[1]
        labels = ['Mean Norm', 'Feature Std', 'Avg Sep.']
        values = [metrics.feature_mean_norm, metrics.feature_std, metrics.avg_separability]
        
        colors = ['#3498db', '#9b59b6', '#2ecc71']
        bars = ax2.bar(labels, values, color=colors, edgecolor='white')
        
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', fontsize=10)
        
        ax2.set_ylabel("Value", fontsize=11)
        ax2.set_title("Feature Statistics", fontsize=12, fontweight='bold')
        
        # 3. 可分性分布
        ax3 = axes[2]
        if metrics.per_class_separability:
            seps = list(metrics.per_class_separability.values())
            ax3.hist(seps, bins=20, color='#2ecc71', edgecolor='white', alpha=0.8)
            ax3.axvline(metrics.avg_separability, color='red', linestyle='--',
                       label=f'Mean: {metrics.avg_separability:.2f}')
            ax3.set_xlabel("Separability", fontsize=11)
            ax3.set_ylabel("Count", fontsize=11)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, "No per-class data", ha='center', va='center',
                    transform=ax3.transAxes)
        
        ax3.set_title("Separability Distribution", fontsize=12, fontweight='bold')
        
        fig.suptitle("L4 Feature Representation Summary", fontsize=14, fontweight='bold')
        safe_tight_layout()
        return fig
