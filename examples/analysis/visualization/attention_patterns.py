# -*- coding: utf-8 -*-
"""
注意力模式可视化

展示 Hilbert 感知注意力的特性。
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def plot_attention_heatmap(
    attention: torch.Tensor,
    title: str = "Attention Heatmap",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
):
    """
    绘制注意力热图

    Args:
        attention: 注意力权重 [N, N] 或 [H, N, N]
        title: 标题
        save_path: 保存路径 (可选)
        figsize: 图像大小
    """
    fig, axes = plt.subplots(1, 2 if attention.dim() == 3 else 1,
                            figsize=(figsize[0], figsize[1] * 0.6) if attention.dim() == 3 else figsize)

    if attention.dim() == 3:
        # 平均所有 head
        attn_mean = attention.mean(dim=0).cpu().numpy()
        attn_std = attention.std(dim=0).cpu().numpy()

        ax = axes[0]
        im = ax.imshow(attn_mean, cmap='YlOrRd', aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f'{title} (Mean)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')

        ax = axes[1]
        im = ax.imshow(attn_std, cmap='Blues', aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f'{title} (Std)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
    else:
        ax = plt.gca() if attention.dim() == 2 else axes
        attn = attention.cpu().numpy()
        im = ax.imshow(attn, cmap='YlOrRd', aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention heatmap to {save_path}")

    plt.close(fig)


def plot_depth_attention_matrix(
    depth_matrix: torch.Tensor,
    title: str = "Depth-Pair Attention Matrix",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6)
):
    """
    绘制深度对聚合注意力矩阵

    数学背景:
    =========
    M[d_q, d_k] = mean_{i,j: d_i=d_q, d_j=d_k} A[i,j]

    对角线主导性表示同深度 token 间的注意力强度。

    Args:
        depth_matrix: 深度对聚合矩阵 [D+1, D+1]
        title: 标题
        save_path: 保存路径 (可选)
        figsize: 图像大小
    """
    fig, ax = plt.subplots(figsize=figsize)

    matrix = depth_matrix.cpu().numpy()
    D = matrix.shape[0]

    # 热图
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        xticklabels=[f'd={d}' for d in range(D)],
        yticklabels=[f'd={d}' for d in range(D)],
        ax=ax,
        square=True
    )

    ax.set_xlabel('Key Depth (d_k)', fontsize=11)
    ax.set_ylabel('Query Depth (d_q)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')

    # 计算并显示对角线主导性
    diag_sum = np.diag(matrix).sum()
    total_sum = matrix.sum()
    diagonal_dom = diag_sum / (total_sum + 1e-8)
    expected = 1.0 / D

    ax.text(0.02, 0.98, f'Diag Dominance: {diagonal_dom:.3f}\nExpected: {expected:.3f}',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved depth attention matrix to {save_path}")

    plt.close(fig)


def plot_head_comparison(
    attention_per_head: torch.Tensor,
    depths: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8)
):
    """
    绘制各 head 的注意力对比

    Args:
        attention_per_head: 每个 head 的注意力 [H, N, N]
        depths: token 深度 [N] (可选)
        save_path: 保存路径 (可选)
        figsize: 图像大小
    """
    H, N, _ = attention_per_head.shape

    n_cols = min(4, H)
    n_rows = (H + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    vmin = attention_per_head.min().item()
    vmax = attention_per_head.max().item()

    for h in range(H):
        ax = axes[h]
        attn = attention_per_head[h].cpu().numpy()

        im = ax.imshow(attn, cmap='YlOrRd', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f'Head {h}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')

        # 如果提供了深度，标注
        if depths is not None:
            ax.set_xticks(range(0, N, max(1, N // 8)))
            ax.set_yticks(range(0, N, max(1, N // 8)))

    # 隐藏多余的子图
    for ax in axes[H:]:
        ax.axis('off')

    fig.colorbar(im, ax=axes.tolist(), shrink=0.5, label='Attention')

    plt.suptitle('Per-Head Attention Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved head comparison to {save_path}")

    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Attention Visualization Demo")
    parser.add_argument("--save-path", type=str, default=None,
                       help="Path to save visualization")

    args = parser.parse_args()

    # 演示
    print("Attention Visualization Demo")
    print("Run with actual models for full visualization.")
