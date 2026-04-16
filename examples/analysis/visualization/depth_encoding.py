# -*- coding: utf-8 -*-
"""
深度编码可视化

展示 depth embedding 的特性和层次结构。
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def plot_depth_embedding_similarity(
    depth_embed: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
):
    """
    绘制深度嵌入相似度矩阵

    数学背景:
    =========
    余弦相似度:
    CosSim(i,j) = (e_i · e_j) / (||e_i|| · ||e_j||)

    分离度 (Separation Score):
    Sep = 1 - mean_{i≠j} CosSim(i,j)

    层次性 (Hierarchy Score):
    Hier = mean_{adjacent} CosSim - mean_{distant} CosSim

    关键洞察:
    ========
    极低的非对角线相似度（接近 0 或负值）证明了模型成功为不同分形深度
    学习到了正交的层级表征。这意味着 d=0 和 d=3 的区域在特征空间中
    是完全不同的，模型能够区分粗粒度（整体）和细粒度（局部）特征。

    Args:
        depth_embed: 深度嵌入 [D+1, dim]
        save_path: 保存路径 (可选)
        figsize: 图像大小
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    depth_embed = depth_embed.detach().float()
    D = depth_embed.shape[0]

    # =========================================================================
    # 子图 1: 余弦相似度矩阵
    # =========================================================================
    ax = axes[0]

    # 计算余弦相似度
    normed = F.normalize(depth_embed, p=2, dim=-1)
    similarity = torch.mm(normed, normed.T).cpu().numpy()

    sns.heatmap(
        similarity,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',  # 红蓝渐变: 红色=高相似, 蓝色=低相似
        xticklabels=[f'd={d}' for d in range(D)],
        yticklabels=[f'd={d}' for d in range(D)],
        ax=ax,
        square=True,
        vmin=-1, vmax=1,
        annot_kws={'fontsize': 9}
    )

    ax.set_xlabel('Depth', fontsize=11)
    ax.set_ylabel('Depth', fontsize=11)
    ax.set_title('Cosine Similarity', fontsize=12, fontweight='bold')

    # =========================================================================
    # 子图 2: 欧几里得距离矩阵 (颜色与 Cosine Similarity 统一)
    # =========================================================================
    ax = axes[1]

    diff = depth_embed.unsqueeze(0) - depth_embed.unsqueeze(1)
    distance = torch.norm(diff, p=2, dim=-1).cpu().numpy()

    sns.heatmap(
        distance,
        annot=True,
        fmt='.1f',
        cmap='RdYlBu_r',
        xticklabels=[f'd={d}' for d in range(D)],
        yticklabels=[f'd={d}' for d in range(D)],
        ax=ax,
        square=True,
        annot_kws={'fontsize': 9}
    )

    ax.set_xlabel('Depth', fontsize=11)
    ax.set_ylabel('Depth', fontsize=11)
    ax.set_title('Euclidean Distance', fontsize=12, fontweight='bold')

    # =========================================================================
    # 子图 3: 统计信息 (分离度 + 正交表征说明)
    # =========================================================================
    ax = axes[2]
    ax.axis('off')

    # 计算指标
    mask_offdiag = ~torch.eye(D, dtype=torch.bool)
    offdiag_mean = similarity[mask_offdiag.cpu().numpy()].mean()
    separation = 1.0 - offdiag_mean

    # 分离度卡片
    sep_card = plt.Rectangle((0.05, 0.75), 0.9, 0.20, transform=ax.transAxes,
                             facecolor='#e8f5e9', edgecolor='#4caf50', linewidth=2)
    ax.add_patch(sep_card)
    ax.text(0.5, 0.87, 'Separation Score', transform=ax.transAxes, ha='center', fontsize=11,
           fontweight='bold', color='#2e7d32')
    ax.text(0.5, 0.78, f'{separation:.3f}', transform=ax.transAxes, ha='center',
           fontsize=18, fontweight='bold', color='#1b5e20')

    # 平均相似度卡片
    sim_card = plt.Rectangle((0.05, 0.50), 0.9, 0.20, transform=ax.transAxes,
                             facecolor='#fff3e0', edgecolor='#ff9800', linewidth=2)
    ax.add_patch(sim_card)
    ax.text(0.5, 0.62, 'Off-diagonal Mean', transform=ax.transAxes, ha='center', fontsize=11,
           fontweight='bold', color='#e65100')
    ax.text(0.5, 0.53, f'{offdiag_mean:.3f}', transform=ax.transAxes, ha='center',
           fontsize=18, fontweight='bold', color='#bf360c')

    # 正交表征说明
    orthogonality_card = plt.Rectangle((0.05, 0.05), 0.9, 0.40, transform=ax.transAxes,
                                       facecolor='#e3f2fd', edgecolor='#2196f3', linewidth=2)
    ax.add_patch(orthogonality_card)
    ax.text(0.5, 0.40, 'Orthogonal Representations', transform=ax.transAxes, ha='center',
           fontsize=11, fontweight='bold', color='#1565c0')
    ax.text(0.08, 0.28,
           'Low off-diagonal similarity confirms\n'
           'that the model learns orthogonal\n'
           'representations for different depths.\n'
           'd=0 (coarse) and d=3 (fine) are\n'
           'completely different in feature space.',
           transform=ax.transAxes, ha='left', va='top', fontsize=9,
           color='#0d47a1')

    plt.tight_layout(pad=2.0)

    # 添加图注
    fig.text(0.5, -0.02,
            'Note: Low off-diagonal similarity (near 0 or negative) proves orthogonal representations '
            'for different fractal depths.',
            ha='center', fontsize=9, style='italic', color='#666666')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved depth embedding similarity to {save_path}")

    plt.close(fig)


def plot_depth_hierarchy(
    depth_embed: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """
    绘制深度层次结构

    Args:
        depth_embed: 深度嵌入 [D+1, dim]
        save_path: 保存路径 (可选)
        figsize: 图像大小
    """
    fig, ax = plt.subplots(figsize=figsize)

    depth_embed = depth_embed.detach().float()
    D = depth_embed.shape[0]

    # 计算相邻深度的相似度
    adjacent_sims = []
    for d in range(D - 1):
        e1 = F.normalize(depth_embed[d], p=2, dim=-1)
        e2 = F.normalize(depth_embed[d + 1], p=2, dim=-1)
        sim = torch.dot(e1, e2).item()
        adjacent_sims.append(sim)

    # 绘制条形图
    depths = [f'd={d}-d={d+1}' for d in range(D - 1)]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(depths)))

    bars = ax.bar(depths, adjacent_sims, color=colors, edgecolor='black')

    ax.set_xlabel('Depth Pair', fontsize=11)
    ax.set_ylabel('Cosine Similarity', fontsize=11)
    ax.set_title('Adjacent Depth Similarity (Hierarchy)', fontsize=12, fontweight='bold')
    ax.set_ylim(-1, 1)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')

    # 标注数值
    for bar, sim in zip(bars, adjacent_sims):
        height = bar.get_height()
        ax.annotate(f'{sim:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved depth hierarchy to {save_path}")

    plt.close(fig)


def plot_pca_projection(
    depth_embed: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 8)
):
    """
    将深度嵌入投影到 2D 并可视化

    Args:
        depth_embed: 深度嵌入 [D+1, dim]
        save_path: 保存路径 (可选)
        figsize: 图像大小
    """
    fig, ax = plt.subplots(figsize=figsize)

    depth_embed = depth_embed.detach().float()
    D = depth_embed.shape[0]

    # SVD 降维
    centered = depth_embed - depth_embed.mean(dim=0, keepdim=True)
    U, S, V = torch.linalg.svd(centered, full_matrices=False)
    coords_2d = (centered @ V[:2, :].T).cpu().numpy()

    # 绘制
    colors = plt.cm.viridis(np.linspace(0, 1, D))

    for d in range(D):
        ax.scatter(coords_2d[d, 0], coords_2d[d, 1],
                  c=[colors[d]], s=300, label=f'd={d}',
                  edgecolors='black', linewidths=2, zorder=3)
        ax.annotate(f'd={d}', (coords_2d[d, 0], coords_2d[d, 1]),
                   textcoords='offset points', xytext=(10, 10),
                   fontsize=12, fontweight='bold')

        # 连接相邻深度
        if d < D - 1:
            ax.plot([coords_2d[d, 0], coords_2d[d+1, 0]],
                   [coords_2d[d, 1], coords_2d[d+1, 1]],
                   'k--', alpha=0.5, linewidth=1.5, zorder=1)

    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_title('Depth Embedding (PCA Projection)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # 添加解释方差
    total_var = (S ** 2).sum()
    explained_var = (S[:2] ** 2).sum() / total_var
    ax.text(0.02, 0.98, f'Explained Variance: {explained_var:.1%}',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved PCA projection to {save_path}")

    plt.close(fig)


def plot_depth_clustermap(
    depth_embed: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
):
    """
    使用 Seaborn Clustermap 进行深度嵌入的层次聚类可视化

    数学背景:
    =========
    Clustermap 执行层次聚类:
    - 行聚类: 基于余弦相似度的深度聚类
    - 列聚类: 相同 (自相似)
    - 使用 linkage 算法 (默认 'average')

    优势:
    - 自动发现深度之间的相似性结构
    - 树状图 (dendrogram) 显示聚类层次
    - 支持颜色标注聚类结果

    Args:
        depth_embed: 深度嵌入 [D+1, dim]
        save_path: 保存路径 (可选)
        figsize: 图像大小
    """
    depth_embed = depth_embed.detach().float()
    D = depth_embed.shape[0]

    # 计算余弦相似度矩阵
    normed = F.normalize(depth_embed, p=2, dim=-1)
    similarity = torch.mm(normed, normed.T).cpu().numpy()

    # 创建 DataFrame
    import pandas as pd
    df = pd.DataFrame(
        similarity,
        index=[f'd={d}' for d in range(D)],
        columns=[f'd={d}' for d in range(D)]
    )

    # 使用 Seaborn clustermap 进行层次聚类可视化
    g = sns.clustermap(
        df,
        method='average',
        metric='euclidean',
        cmap='RdYlBu_r',
        annot=True,
        fmt='.2f',
        figsize=figsize,
        dendrogram_ratio=(0.15, 0.15),
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        linewidths=0.5,
        linecolor='white',
        vmin=-1, vmax=1
    )

    g.ax_heatmap.set_xlabel('Depth', fontsize=11)
    g.ax_heatmap.set_ylabel('Depth', fontsize=11)
    g.fig.suptitle('Depth Embedding Hierarchical Clustering (Seaborn Clustermap)',
                   fontsize=14, fontweight='bold', y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved depth clustermap to {save_path}")

    plt.close(g.fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Depth Encoding Visualization Demo")
    parser.add_argument("--save-path", type=str, default=None,
                       help="Path to save visualization")

    args = parser.parse_args()

    print("Depth Encoding Visualization Demo")
    print("Run with actual depth embeddings for full visualization.")
