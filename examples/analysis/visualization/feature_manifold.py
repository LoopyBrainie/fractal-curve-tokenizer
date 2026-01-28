# -*- coding: utf-8 -*-
"""
Feature Manifold Visualization - 特征流形可视化

提供跨深度特征流形的 t-SNE/UMAP 可视化，验证 Coarse-to-Fine 特征分离。

数学背景
========
1. t-SNE/UMAP 降维:
   - t-SNE: 保留局部邻域结构，高维 → 2D
   - UMAP: 保持全局拓扑结构，更快

2. 深度分离度量:
   - 类间散度 (Between-class scatter): S_b = Σ_d N_d ||μ_d - μ||²
   - 类内散度 (Within-class scatter): S_w = Σ_d Σ_{x∈d} ||x - μ_d||²
   - Fisher 比: F = S_b / S_w

使用方法
========
    from examples.analysis.visualization.feature_manifold import plot_cross_depth_feature_manifold

    plot_cross_depth_feature_manifold(
        token_features=features,  # [N, D] token 特征
        token_depths=depths,      # [N] token 深度
        save_path="feature_manifold.png"
    )
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 导入统一的颜色工具
from ..utils.color_utils import get_depth_colors


def plot_cross_depth_feature_manifold(
    token_features: torch.Tensor,
    token_depths: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 12),
    method: str = 'tsne',  # 'tsne' or 'pca'
    perplexity: int = 30
):
    """
    跨深度特征流形可视化 - 验证 Coarse-to-Fine 特征分离

    数学背景
    =========
    1. t-SNE 降维:
       - 使用 t-分布核保持局部邻域结构
       - perplexity 参数控制有效邻域大小

    2. 深度分离度量:
       - 类内散度 (Within-class scatter): 衡量同深度 token 的聚集程度
       - 类间散度 (Between-class scatter): 衡量不同深度 centroids 的距离
       - Fisher 比: 分离度 / 紧密度，比值越大表示深度分离越好

    预期洞察
    ========
    - d=0 (coarse): 聚类在"轮廓/色彩"空间中心
    - d=3 (fine): 分散在"纹理/局部零件"空间
    - 深度递增时，特征方差通常增大

    Args:
        token_features: [N, D] token 特征
        token_depths: [N] token 深度
        save_path: 保存路径 (可选)
        figsize: 图像大小
        method: 降维方法 ('tsne' or 'pca')
        perplexity: t-SNE perplexity 参数
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    features = token_features.detach().float().cpu().numpy()
    depths = token_depths.cpu().numpy()
    unique_depths = np.unique(depths)
    max_depth = int(depths.max())

    # 统一颜色映射: 蓝色(d=0) → 红色(d=max_depth)
    depth_colors = get_depth_colors(max_depth, cmap='RdYlBu_r', alpha=0.8)
    depth_to_color = {d: depth_colors[d] for d in range(max_depth + 1)}

    # =========================================================================
    # 子图 1: t-SNE/PCA 投影
    # =========================================================================
    ax = axes[0, 0]

    if method == 'tsne':
        # t-SNE 降维
        perplexity = min(perplexity, len(features) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        coords_2d = tsne.fit_transform(features)
        method_label = 't-SNE'
    else:
        # PCA 降维
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(features)
        method_label = 'PCA'

    for d in unique_depths:
        mask = depths == d
        color = depth_to_color[d]
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=[color[:3]], s=50, alpha=0.7,
            label=f'd={d} (n={mask.sum()})',
            edgecolors='white', linewidths=0.5
        )

    ax.set_xlabel(f'{method_label} Dimension 1', fontsize=11)
    ax.set_ylabel(f'{method_label} Dimension 2', fontsize=11)
    ax.set_title(f'Cross-Depth Feature Manifold ({method_label})', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # 子图 2: 深度聚类紧密度 (类内散度)
    # =========================================================================
    ax = axes[0, 1]

    # 计算每个深度的聚类紧密度 (类内方差)
    cluster_tightness = []
    for d in unique_depths:
        mask = depths == d
        if mask.sum() > 1:
            cluster_features = features[mask]
            centroid = cluster_features.mean(axis=0)
            # 类内散度: 各点到 centroid 的平均距离
            tightness = np.linalg.norm(cluster_features - centroid, axis=1).mean()
            cluster_tightness.append(tightness)
        else:
            cluster_tightness.append(0)

    bars = ax.bar(
        [f'd={d}' for d in unique_depths],
        cluster_tightness,
        color=[depth_to_color[d][:3] for d in unique_depths],
        edgecolor='black', alpha=0.8
    )
    ax.set_xlabel('Depth', fontsize=11)
    ax.set_ylabel('Within-Cluster Variance', fontsize=11)
    ax.set_title('Cluster Tightness by Depth\n(Lower = Tighter)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, val in zip(bars, cluster_tightness):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # =========================================================================
    # 子图 3: 深度间分离度 (类间散度矩阵)
    # =========================================================================
    ax = axes[1, 0]

    # 计算深度间的 centroids
    centroids = {}
    for d in unique_depths:
        mask = depths == d
        if mask.sum() > 0:
            centroids[d] = features[mask].mean(axis=0)

    # 计算 centroids 间的欧几里得距离
    n_depths = len(unique_depths)
    separation_matrix = np.zeros((n_depths, n_depths))
    for i, d1 in enumerate(unique_depths):
        for j, d2 in enumerate(unique_depths):
            if d1 in centroids and d2 in centroids:
                separation_matrix[i, j] = np.linalg.norm(centroids[d1] - centroids[d2])

    im = ax.imshow(separation_matrix, cmap='YlOrRd', aspect='equal')
    ax.set_xticks(range(n_depths))
    ax.set_yticks(range(n_depths))
    ax.set_xticklabels([f'd={d}' for d in unique_depths])
    ax.set_yticklabels([f'd={d}' for d in unique_depths])
    ax.set_title('Inter-Depth Separation\n(Centroid Distance)', fontsize=12, fontweight='bold')

    # 添加数值标注
    for i in range(n_depths):
        for j in range(n_depths):
            color = 'white' if separation_matrix[i, j] > separation_matrix.max() / 2 else 'black'
            ax.text(j, i, f'{separation_matrix[i, j]:.2f}',
                   ha='center', va='center', fontsize=9, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label='Euclidean Distance')

    # =========================================================================
    # 子图 4: Fisher 比 (分离度/紧密度)
    # =========================================================================
    ax = axes[1, 1]
    ax.axis('off')

    # 计算 Fisher 比
    overall_centroid = features.mean(axis=0)
    between_var = 0
    within_var = 0

    for d in unique_depths:
        mask = depths == d
        cluster_features = features[mask]
        centroid = cluster_features.mean(axis=0)
        n_d = mask.sum()

        # 类间散度
        between_var += n_d * np.linalg.norm(centroid - overall_centroid) ** 2

        # 类内散度
        if n_d > 1:
            within_var += np.linalg.norm(cluster_features - centroid, axis=1).sum()

    fisher_ratio = between_var / (within_var + 1e-8)

    # 绘制 Fisher 比卡片
    card = plt.Rectangle((0.2, 0.6), 0.6, 0.25, transform=ax.transAxes,
                         facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=2)
    ax.add_patch(card)
    ax.text(0.5, 0.78, 'Fisher Ratio', transform=ax.transAxes, ha='center', fontsize=12,
           fontweight='bold', color='#1565c0')
    ax.text(0.5, 0.68, f'{fisher_ratio:.4f}', transform=ax.transAxes, ha='center',
           fontsize=24, fontweight='bold', color='#0d47a1')

    # 解释
    explanation = """
    Fisher Ratio = Between-Class Scatter / Within-Class Scatter

    Higher ratio indicates better depth separation:
    - d=0 features are tight and distant from other depths
    - d=4 features are spread but still separated
    """
    exp_card = plt.Rectangle((0.1, 0.1), 0.8, 0.4, transform=ax.transAxes,
                             facecolor='#fff3e0', edgecolor='#ff9800', linewidth=2)
    ax.add_patch(exp_card)
    ax.text(0.5, 0.42, explanation.strip(), transform=ax.transAxes, ha='center',
           va='top', fontsize=10, color='#e65100',
           bbox=dict(boxstyle='round', facecolor='none', edgecolor='none'))

    fig.suptitle('Cross-Depth Feature Manifold Analysis\n(Coarse-to-Fine Feature Separation)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved feature manifold to {save_path}")

    plt.close(fig)


def plot_depth_feature_statistics(
    token_features: torch.Tensor,
    token_depths: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5)
):
    """
    绘制深度特征的统计指标

    包含:
    1. 各深度特征范数分布
    2. 各深度特征方差
    3. 深度间余弦相似度热图

    Args:
        token_features: [N, D] token 特征
        token_depths: [N] token 深度
        save_path: 保存路径
        figsize: 图像大小
    """
    import torch.nn.functional as F

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    features = token_features.detach().float().cpu()
    depths = token_depths.cpu().numpy()
    unique_depths = np.unique(depths)
    max_depth = int(depths.max())

    # 统一颜色映射
    depth_colors = get_depth_colors(max_depth, cmap='RdYlBu_r', alpha=0.8)
    depth_to_color = {d: depth_colors[d][:3] for d in range(max_depth + 1)}

    # =========================================================================
    # 子图 1: 特征范数分布
    # =========================================================================
    ax = axes[0]

    norms = torch.norm(features, dim=1).numpy()

    for d in unique_depths:
        mask = depths == d
        ax.hist(norms[mask], bins=30, alpha=0.6, label=f'd={d}',
               color=depth_to_color[d], edgecolor='white')

    ax.set_xlabel('Feature Norm', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Feature Norm Distribution by Depth', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # 子图 2: 特征方差
    # =========================================================================
    ax = axes[1]

    variances = []
    for d in unique_depths:
        mask = depths == d
        if mask.sum() > 0:
            var = features[mask].var(dim=0).mean().item()
            variances.append(var)
        else:
            variances.append(0)

    bars = ax.bar([f'd={d}' for d in unique_depths], variances,
                  color=[depth_to_color[d] for d in unique_depths],
                  edgecolor='black', alpha=0.8)
    ax.set_xlabel('Depth', fontsize=11)
    ax.set_ylabel('Mean Feature Variance', fontsize=11)
    ax.set_title('Feature Variance by Depth', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, variances):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # =========================================================================
    # 子图 3: 深度间余弦相似度
    # =========================================================================
    ax = axes[2]

    # 计算深度 centroids
    centroids = {}
    for d in unique_depths:
        mask = depths == d
        if mask.sum() > 0:
            centroids[d] = F.normalize(features[mask].mean(dim=0), dim=0).numpy()

    # 计算 centroids 间的余弦相似度
    n_depths = len(unique_depths)
    similarity_matrix = np.zeros((n_depths, n_depths))
    for i, d1 in enumerate(unique_depths):
        for j, d2 in enumerate(unique_depths):
            if d1 in centroids and d2 in centroids:
                similarity_matrix[i, j] = np.dot(centroids[d1], centroids[d2])

    im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks(range(n_depths))
    ax.set_yticks(range(n_depths))
    ax.set_xticklabels([f'd={d}' for d in unique_depths])
    ax.set_yticklabels([f'd={d}' for d in unique_depths])
    ax.set_title('Depth Centroid Cosine Similarity', fontsize=12, fontweight='bold')

    for i in range(n_depths):
        for j in range(n_depths):
            color = 'white' if abs(similarity_matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                   ha='center', va='center', fontsize=9, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label='Cosine Similarity')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved depth feature statistics to {save_path}")

    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature Manifold Visualization Demo")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save visualization")
    parser.add_argument("--method", type=str, default='tsne', choices=['tsne', 'pca'],
                       help="Dimensionality reduction method")

    args = parser.parse_args()

    # 创建模拟数据测试
    print("Feature Manifold Visualization Demo")
    print("=" * 50)

    # 模拟 token 特征和深度
    n_tokens = 500
    dim = 384
    max_depth = 4

    # 生成模拟数据：不同深度有不同的特征分布
    features = []
    depths = []

    for d in range(max_depth + 1):
        n_d = n_tokens // (max_depth + 1) + (d * 20)  # 不同深度 token 数量不同
        # 每个深度的特征有特定的中心
        center = np.zeros(dim)
        center[d * 50:(d + 1) * 50] = 1.0  # 不同深度有不同的激活维度
        feat = np.random.randn(n_d, dim) + center
        features.append(feat)
        depths.extend([d] * n_d)

    features = torch.tensor(np.vstack(features), dtype=torch.float32)
    depths = torch.tensor(depths, dtype=torch.long)

    print(f"Generated {len(features)} tokens with depths 0-{max_depth}")

    # 运行可视化
    save_path = args.save_path

    plot_cross_depth_feature_manifold(
        features, depths,
        save_path=save_path,
        method=args.method
    )

    plot_depth_feature_statistics(
        features, depths,
        save_path=save_path.replace('.png', '_stats.png') if save_path else None
    )

    print("\nDone! Run with actual model features for full visualization.")
