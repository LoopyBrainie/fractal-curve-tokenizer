# -*- coding: utf-8 -*-
"""
分词对比可视化

展示 Fractal Curve ViT 与标准 ViT 的分词策略差异。
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def plot_tokenization_comparison(
    image: torch.Tensor,
    fractal_tokens: torch.Tensor,
    standard_tokens: torch.Tensor,
    fractal_depths: Optional[torch.Tensor] = None,
    fractal_regions: Optional[List[Tuple[float, float, float, float]]] = None,
    fractal_centers: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 12)
):
    """
    对比分词结果

    数学背景:
    =========
    标准 ViT 分词:
    - 固定网格划分: H/p × W/p patches
    - 例如: 224×224, p=16 → 14×14 = 196 patches

    Fractal ViT 分词:
    - 自适应四叉树分割
    - 基于图像复杂度选择分割区域
    - 深度 d 的区域大小: image_size / 2^d

    Args:
        image: 输入图像 [C, H, W]
        fractal_tokens: Fractal ViT 的 token [N_fractal, D]
        standard_tokens: 标准 ViT 的 token [N_standard, D]
        fractal_depths: Fractal tokens 的深度 [N_fractal]
        save_path: 保存路径 (可选)
        figsize: 图像大小
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # 转换图像为 numpy
    img_np = image.permute(1, 2, 0).cpu().numpy()
    if img_np.shape[-1] == 1:
        img_np = img_np.squeeze()
        cmap = 'gray'
    else:
        cmap = None

    H, W = img_np.shape[:2] if cmap is None else img_np.shape

    # =========================================================================
    # 子图 1: 原始图像
    # =========================================================================
    ax = axes[0, 0]
    if cmap:
        ax.imshow(img_np, cmap=cmap)
    else:
        ax.imshow(img_np)
    ax.set_title('Original Image', fontsize=12, fontweight='bold')
    ax.axis('off')

    # =========================================================================
    # 子图 2: 标准 ViT 分词
    # =========================================================================
    ax = axes[0, 1]
    if cmap:
        ax.imshow(img_np, cmap=cmap)
    else:
        ax.imshow(img_np)

    N_std = standard_tokens.shape[0]
    grid_size = int(np.sqrt(N_std))
    patch_h = H // grid_size
    patch_w = W // grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            rect = patches.Rectangle(
                (j * patch_w, i * patch_h),
                patch_w, patch_h,
                linewidth=1.5,
                edgecolor='blue',
                facecolor='none',
                alpha=0.8
            )
            ax.add_patch(rect)

            # 标注 patch 索引
            cx, cy = j * patch_w + patch_w / 2, i * patch_h + patch_h / 2
            ax.text(cx, cy, f'{i * grid_size + j}', fontsize=6,
                   ha='center', va='center', color='white',
                   bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7))

    ax.set_title(f'Standard ViT: {N_std} Fixed Patches', fontsize=12, fontweight='bold')
    ax.axis('off')

    # =========================================================================
    # 子图 3: Fractal ViT 分词
    # =========================================================================
    ax = axes[0, 2]
    if cmap:
        ax.imshow(img_np, cmap=cmap)
    else:
        ax.imshow(img_np)

    N_frac = fractal_tokens.shape[0]
    depths = fractal_depths.cpu().numpy() if fractal_depths is not None else None
    max_depth = int(depths.max()) if depths is not None else 0
    colors = plt.cm.viridis(np.linspace(0, 1, max_depth + 1)) if max_depth > 0 else None

    if fractal_regions is not None:
        # 使用真实区域框
        for idx, region in enumerate(fractal_regions):
            x1, y1, x2, y2 = region
            depth = int(depths[idx]) if depths is not None and idx < len(depths) else 0
            color = colors[depth] if colors is not None else 'green'
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor='white',
                facecolor=color, alpha=0.4
            ))
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(cx, cy, str(depth), fontsize=7, ha='center', va='center',
                   color='white', fontweight='bold',
                   bbox=dict(boxstyle='circle', facecolor=color, alpha=0.8))
    else:
        # 退化为中心点可视化
        if fractal_centers is None:
            grid_size = max(1, int(np.ceil(np.sqrt(N_frac))))
            xs = (np.arange(N_frac) % grid_size) * (W / grid_size) + W / (2 * grid_size)
            ys = (np.arange(N_frac) // grid_size) * (H / grid_size) + H / (2 * grid_size)
            fractal_centers = np.stack([xs, ys], axis=1)

        for i in range(N_frac):
            depth = int(depths[i]) if depths is not None and i < len(depths) else 0
            color = colors[depth] if colors is not None else 'green'
            x, y = fractal_centers[i]
            ax.scatter([x], [y], c=[color], s=100, marker='o',
                      edgecolors='black', linewidths=1, zorder=5)
            ax.text(x, y, str(depth), fontsize=7, ha='center', va='center',
                   color='white', fontweight='bold',
                   bbox=dict(boxstyle='circle', facecolor=color, alpha=0.8))

    ax.set_title(f'Fractal ViT: {N_frac} Adaptive Tokens', fontsize=12, fontweight='bold')
    ax.axis('off')

    # =========================================================================
    # 子图 4: Token 数量对比
    # =========================================================================
    ax = axes[1, 0]
    models = ['Standard ViT', 'Fractal ViT']
    token_counts = [N_std, N_frac]
    colors_bar = ['blue', 'green']

    bars = ax.bar(models, token_counts, color=colors_bar, edgecolor='black', linewidth=2)
    ax.set_ylabel('Token Count', fontsize=11)
    ax.set_title('Token Count Comparison', fontsize=12, fontweight='bold')

    for bar, count in zip(bars, token_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               str(count), ha='center', va='bottom', fontsize=14, fontweight='bold')

    reduction = (1 - N_frac / N_std) * 100
    ax.annotate(f'-{reduction:.1f}%',
               xy=(0.5, max(N_std, N_frac) * 0.8),
               fontsize=16, color='red', fontweight='bold',
               ha='center')

    # =========================================================================
    # 子图 5: 深度分布 (Fractal ViT)
    # =========================================================================
    ax = axes[1, 1]
    if fractal_depths is not None:
        depths = fractal_depths.cpu().numpy()
        depth_counts = {}
        for d in depths:
            depth_counts[int(d)] = depth_counts.get(int(d), 0) + 1

        if depth_counts:
            depths_sorted = sorted(depth_counts.keys())
            counts = [depth_counts[d] for d in depths_sorted]
            depth_labels = [f'd={d}' for d in depths_sorted]
            depth_colors = plt.cm.viridis(np.linspace(0, 1, len(depth_labels)))

            bars = ax.bar(depth_labels, counts, color=depth_colors, edgecolor='black')
            ax.set_xlabel('Depth', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('Fractal ViT: Token Depth Distribution', fontsize=12, fontweight='bold')

            total = sum(counts)
            for bar, count in zip(bars, counts):
                pct = count / total * 100
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

    # =========================================================================
    # 子图 6: 分词策略说明
    # =========================================================================
    ax = axes[1, 2]
    ax.axis('off')

    depth_range_text = f"0 to {max_depth}" if max_depth > 0 else "N/A"
    explanation = f"""
    Tokenization Strategy Comparison
    =================================

    Standard ViT:
    • Grid size: {grid_size}×{grid_size} = {N_std} patches
    • Patch size: {patch_w}×{patch_h} pixels
    • Fixed, uniform coverage

    Fractal ViT:
    • Adaptive: {N_frac} tokens (varies per image)
    • Quadtree-based splitting
    • Depth: {depth_range_text}
    • Focus on complex regions

    Token Reduction: {reduction:.1f}%
    =================================

    Key Insight:
    Fractal ViT uses fewer tokens by
    focusing computational resources on
    image regions with high complexity.
    """

    ax.text(0.05, 0.95, explanation, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved tokenization comparison to {save_path}")

    plt.close(fig)


def plot_depth_distribution(
    depths: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 4)
):
    """
    绘制深度分布

    Args:
        depths: 深度 tensor [N]
        save_path: 保存路径 (可选)
        figsize: 图像大小
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    depths_np = depths.cpu().numpy()
    unique_depths = np.unique(depths_np)
    max_depth = int(depths_np.max())

    # =========================================================================
    # 子图 1: 直方图
    # =========================================================================
    ax = axes[0]
    ax.hist(depths_np, bins=max_depth + 1, range=(0, max_depth + 1),
           color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Depth', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Depth Distribution Histogram', fontsize=12, fontweight='bold')
    ax.set_xticks(range(max_depth + 1))

    # =========================================================================
    # 子图 2: 饼图
    # =========================================================================
    ax = axes[1]
    depth_counts = {}
    for d in depths_np:
        depth_counts[int(d)] = depth_counts.get(int(d), 0) + 1

    labels = [f'd={d}\n({c})' for d, c in sorted(depth_counts.items())]
    sizes = list(depth_counts.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        explode=[0.02] * len(sizes)
    )
    ax.set_title('Token Distribution by Depth', fontsize=12, fontweight='bold')

    # =========================================================================
    # 子图 3: 累积分布
    # =========================================================================
    ax = axes[2]
    sorted_depths = np.sort(depths_np)
    cumulative = np.arange(1, len(sorted_depths) + 1) / len(sorted_depths)

    for d in range(max_depth + 1):
        mask = sorted_depths == d
        if mask.any():
            ax.plot(sorted_depths[mask], cumulative[mask], 'o-',
                   label=f'd={d}', linewidth=2, markersize=4)

    ax.set_xlabel('Depth', fontsize=11)
    ax.set_ylabel('Cumulative Proportion', fontsize=11)
    ax.set_title('Cumulative Distribution by Depth', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved depth distribution to {save_path}")

    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tokenization Visualization Demo")
    parser.add_argument("--save-path", type=str, default=None,
                       help="Path to save visualization")

    args = parser.parse_args()

    # 演示
    print("Tokenization Visualization Demo")
    print("Run with actual models for full visualization.")
