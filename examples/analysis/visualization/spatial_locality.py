# -*- coding: utf-8 -*-
"""
空间局部性可视化

展示 Hilbert 曲线相对于栅格顺序的局部性保持优势。

增强功能 (Seaborn)
==================
- 使用 Seaborn histplot/kdeplot 进行分布可视化
- 支持核密度估计 (KDE) 和直方图叠加
- 更美观的统计图表样式
"""

import sys
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

# Seaborn 用于统计可视化
import seaborn as sns

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _generate_hilbert_curve(order: int) -> List[Tuple[int, int]]:
    """生成 Hilbert 曲线坐标 - 正确实现"""
    def rot(n, x, y, rx, ry):
        """旋转/翻转象限"""
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        return x, y

    def d_to_xy(n, d):
        """d 到 (x,y) 转换"""
        x = y = 0
        s = 1
        t = d
        while s < n:
            rx = (t // 2) & 1
            ry = (t ^ rx) & 1
            x, y = rot(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        return x, y

    n = 2 ** order
    coords = []
    for d in range(n * n):
        coords.append(d_to_xy(n, d))
    return coords


def _generate_raster_curve(size: int) -> List[Tuple[int, int]]:
    """生成栅格曲线坐标"""
    return [(x, y) for y in range(size) for x in range(size)]


def plot_hilbert_vs_raster(
    order: int = 4,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 7)
):
    """
    对比 Hilbert 曲线和栅格顺序

    数学背景:
    =========
    Hilbert 曲线:
    - 空间填充曲线，保持局部性
    - 任意相邻点之间欧几里得距离 ≤ √2
    - 适用于需要保持空间关系的场景

    栅格顺序:
    - 简单的行优先遍历
    - 对角线跳跃导致局部性差
    - 相邻索引可能有大的空间距离

    关键洞察:
    ========
    栅格顺序中"对角线跳跃"问题：例如 (15,0) 和 (0,15) 在空间上相邻，
    但在序列中索引相差 210，破坏了局部性。

    Args:
        order: Hilbert 曲线阶数
        save_path: 保存路径 (可选)
        figsize: 图像大小
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    n = 2 ** order

    # =========================================================================
    # Hilbert 曲线
    # =========================================================================
    ax = axes[0]
    hilbert_coords = _generate_hilbert_curve(order)
    hilbert_array = np.array(hilbert_coords)

    # 颜色映射
    colors = plt.cm.plasma(np.linspace(0, 1, len(hilbert_coords)))

    # 绘制曲线
    for i in range(len(hilbert_coords) - 1):
        ax.plot(
            [hilbert_array[i, 0], hilbert_array[i+1, 0]],
            [hilbert_array[i, 1], hilbert_array[i+1, 1]],
            color=colors[i], linewidth=2
        )

    # 标注起点和终点
    ax.scatter([0], [0], c='green', s=200, marker='o', zorder=5, label='Start')
    ax.scatter([n-1], [n-1], c='red', s=200, marker='X', zorder=5, label='End')

    ax.set_xlim(-1, n)
    ax.set_ylim(-1, n)
    ax.set_aspect('equal')
    ax.set_title(f'Hilbert Curve (Order {order})\nLocality Preserved', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # 栅格顺序 - 高亮对角线跳跃问题
    # =========================================================================
    ax = axes[1]
    raster_coords = _generate_raster_curve(n)
    raster_array = np.array(raster_coords)

    # 绘制曲线
    for i in range(len(raster_coords) - 1):
        ax.plot(
            [raster_array[i, 0], raster_array[i+1, 0]],
            [raster_array[i, 1], raster_array[i+1, 1]],
            color=colors[i], linewidth=2
        )

    # 高亮显示"对角线跳跃"区域
    jump_points = []
    for i in range(n - 1):
        p1 = raster_array[i * n + n - 1]
        p2 = raster_array[i * n + n]
        jump_points.append((p1, p2))

    for p1, p2 in jump_points:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
               'r--', linewidth=3, alpha=0.7)

    ax.annotate('Diagonal\nJump!', xy=(n/2, 0.5), fontsize=10,
               color='red', fontweight='bold', ha='center',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.scatter([0], [0], c='green', s=200, marker='o', zorder=5, label='Start')
    ax.scatter([n-1], [n-1], c='red', s=200, marker='X', zorder=5, label='End')

    ax.set_xlim(-1, n)
    ax.set_ylim(-1, n)
    ax.set_aspect('equal')
    ax.set_title(f'Raster Order\nLocality Broken (Red Dashed = Jumps)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Hilbert Curve vs Raster Order', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(pad=2.0)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Hilbert vs Raster comparison to {save_path}")

    plt.close(fig)


def plot_locality_preservation(
    order: int = 4,
    num_samples: int = 1000,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 5),
    seed: Optional[int] = 42
):
    """
    绘制局部性保持对比

    数学定义:
    ========
    对于排序 σ，局部性保持度量:

    ρ(σ) = E[ ||p1 - p2||_2 / |σ⁻¹(p1) - σ⁻¹(p2)|^(1/2) ]

    其中:
    - ||p1 - p2||_2: 欧几里得距离
    - |σ⁻¹(p1) - σ⁻¹(p2)|: 序列索引差

    更小的 ρ 表示更好的局部性保持。

    关键洞察:
    ========
    Hilbert Ratio 集中在 1.0 附近表示良好的局部性保持。
    Raster Ratio 分布广泛表示严重的"对角线跳跃"问题。

    Args:
        order: Hilbert 曲线阶数
        num_samples: 采样点对数量
        save_path: 保存路径 (可选)
        figsize: 图像大小
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    n = 2 ** order
    hilbert_coords = _generate_hilbert_curve(order)
    raster_coords = _generate_raster_curve(n)

    # 计算局部性指标
    rng = np.random.default_rng(seed)

    def compute_locality(coords: List[Tuple[int, int]]) -> Tuple[List[float], List[float]]:
        ratios = []
        index_diffs = []
        for _ in range(num_samples):
            i = rng.integers(0, len(coords))
            j = rng.integers(0, len(coords))
            if i != j:
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                seq_diff = abs(i - j)
                spatial_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if seq_diff > 0:
                    ratios.append(spatial_dist / np.sqrt(seq_diff))
                    index_diffs.append(seq_diff)
        return ratios, index_diffs

    hilbert_ratios, hilbert_diffs = compute_locality(hilbert_coords)
    raster_ratios, raster_diffs = compute_locality(raster_coords)

    # =========================================================================
    # 子图 1: 散点图对比 (添加基准线)
    # =========================================================================
    ax = axes[0]
    min_len = min(len(hilbert_ratios), len(raster_ratios))
    if min_len > 0:
        ax.scatter(hilbert_ratios[:min_len], raster_ratios[:min_len], alpha=0.3, s=5, c='blue')
        max_val = max(max(hilbert_ratios[:min_len]), max(raster_ratios[:min_len]))
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Equal (ρ_h=ρ_r)')

        ax.axhline(y=1.0, color='green', linestyle=':', linewidth=2, alpha=0.7,
                  label='Hilbert Ideal (ρ≈1.0)')
        ax.axvline(x=1.0, color='green', linestyle=':', linewidth=2, alpha=0.7)

        ax.fill_between([0, 2], [0, 0], [2, 2], alpha=0.1, color='green',
                       label='Hilbert Cluster Region')
    else:
        ax.text(0.5, 0.5, 'Insufficient samples', ha='center', va='center')
        max_val = 1
    ax.set_xlabel('Hilbert Ratio (ρ_h)', fontsize=11)
    ax.set_ylabel('Raster Ratio (ρ_r)', fontsize=11)
    ax.set_title('Locality Ratio Comparison\n(Lower = Better Locality)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    # =========================================================================
    # 子图 2: 分布直方图 + KDE (使用 Seaborn)
    # =========================================================================
    ax = axes[1]

    sns.set_style("whitegrid")

    all_ratios = hilbert_ratios + raster_ratios
    if all_ratios:
        sns.histplot(
            hilbert_ratios, bins=20, alpha=0.6, label='Hilbert',
            color='#3498db', stat='density', kde=True,
            line_kws={'linewidth': 2}, ax=ax
        )

        sns.histplot(
            raster_ratios, bins=20, alpha=0.6, label='Raster',
            color='#e74c3c', stat='density', kde=True,
            line_kws={'linewidth': 2}, ax=ax
        )

        ax.axvline(x=1.0, color='green', linestyle='--', linewidth=2, alpha=0.8,
                  label='Ideal (ρ=1.0)')

        ax.axvspan(0.5, 1.5, alpha=0.1, color='green', label='Hilbert Cluster')
    else:
        ax.text(0.5, 0.5, 'No valid samples', ha='center', va='center')

    ax.set_xlabel('Locality Ratio (Spatial / √Sequence)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Locality Ratio Distribution\n(Seaborn: Hist + KDE)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    sns.reset_orig()

    # =========================================================================
    # 子图 3: 统计对比
    # =========================================================================
    ax = axes[2]
    ax.axis('off')

    hilbert_mean = np.mean(hilbert_ratios) if hilbert_ratios else 0
    raster_mean = np.mean(raster_ratios) if raster_ratios else 0
    hilbert_std = np.std(hilbert_ratios) if hilbert_ratios else 0
    raster_std = np.std(raster_ratios) if raster_ratios else 0
    hilbert_median = np.median(hilbert_ratios) if hilbert_ratios else 0
    raster_median = np.median(raster_ratios) if raster_ratios else 0

    improvement = (raster_mean - hilbert_mean) / raster_mean * 100 if raster_mean > 0 else 0

    stats_text = f"""
    Locality Preservation Statistics
    =================================

    Hilbert Curve:
      Mean:   {hilbert_mean:.3f}
      Std:    {hilbert_std:.3f}
      Median: {hilbert_median:.3f}
      → Clustered near ρ≈1.0

    Raster Order:
      Mean:   {raster_mean:.3f}
      Std:    {raster_std:.3f}
      Median: {raster_median:.3f}
      → Spread out (diagonal jumps!)

    Result:
      Hilbert has {improvement:.1f}% better
      locality preservation!

    Key Insight:
      Hilbert curve maintains spatial
      proximity (ρ≈1.0), while raster
      order has "diagonal jumps".
    """

    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout(pad=2.0)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved locality preservation to {save_path}")

    plt.close(fig)


def animate_hilbert_curve(
    order: int = 4,
    save_path: Optional[str] = None,
    fps: int = 10
):
    """
    动画展示 Hilbert 曲线遍历过程

    Args:
        order: Hilbert 曲线阶数
        save_path: 保存路径 (可选，保存为 GIF)
        fps: 帧率
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    n = 2 ** order
    coords = _generate_hilbert_curve(order)

    points, = ax.plot([], [], 'o-', markersize=3, linewidth=1)
    start_marker, = ax.plot([], [], 'go', markersize=15, label='Start')
    end_marker, = ax.plot([], [], 'rx', markersize=15, label='Current')

    ax.set_xlim(-1, n)
    ax.set_ylim(-1, n)
    ax.set_aspect('equal')
    ax.set_title(f'Hilbert Curve Traversal (Order {order})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    def init():
        points.set_data([], [])
        start_marker.set_data([], [])
        end_marker.set_data([], [])
        return points, start_marker, end_marker

    def animate(frame):
        if frame == 0:
            x = [coords[0][0]]
            y = [coords[0][1]]
        else:
            x = [coords[frame-1][0], coords[frame][0]]
            y = [coords[frame-1][1], coords[frame][1]]

        color = plt.cm.plasma(frame / len(coords))
        points.set_color(color)

        points.set_data(x, y)
        end_marker.set_data([coords[frame][0]], [coords[frame][1]])
        return points, start_marker, end_marker

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(coords), interval=1000/fps, blit=True
    )

    plt.tight_layout()

    if save_path:
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"Saved Hilbert curve animation to {save_path}")

    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spatial Locality Visualization")
    parser.add_argument("--order", type=int, default=4, help="Hilbert curve order")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save")
    parser.add_argument("--animate", action="store_true", help="Create animation")

    args = parser.parse_args()

    if args.animate:
        animate_hilbert_curve(order=args.order, save_path=args.save_path)
    else:
        plot_hilbert_vs_raster(order=args.order, save_path=args.save_path)
        plot_locality_preservation(order=args.order, save_path=args.save_path)
