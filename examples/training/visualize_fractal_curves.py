#!/usr/bin/env python3
"""Fractal Curve 交互式可视化脚本

提供丰富的分形曲线可视化，包括：
1. Hilbert 曲线动画生成
2. 不同阶数的曲线对比
3. 2D → 1D 映射过程动画
4. 局部性保持的直观展示
5. 多尺度 tokenization 可视化

使用示例：
    # 生成静态可视化
    python visualize_fractal_curves.py --all
    
    # 生成 GIF 动画
    python visualize_fractal_curves.py --animate --order 4
    
    # 交互式探索 (需要 matplotlib 后端支持)
    python visualize_fractal_curves.py --interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vit_pytorch.hilbert import HilbertCurve

# 设置样式
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


# ============================================================================
# 颜色方案
# ============================================================================

def create_rainbow_cmap() -> LinearSegmentedColormap:
    """创建彩虹渐变色图"""
    colors = [
        (0.0, '#FF0000'),  # 红
        (0.17, '#FF7F00'),  # 橙
        (0.33, '#FFFF00'),  # 黄
        (0.5, '#00FF00'),  # 绿
        (0.67, '#0000FF'),  # 蓝
        (0.83, '#4B0082'),  # 靛
        (1.0, '#9400D3'),  # 紫
    ]
    return LinearSegmentedColormap.from_list(
        'rainbow_hilbert',
        [(pos, color) for pos, color in colors]
    )


RAINBOW_CMAP = create_rainbow_cmap()


# ============================================================================
# Hilbert 曲线生成
# ============================================================================

def generate_hilbert_points(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """生成 Hilbert 曲线的所有点坐标"""
    n = 2 ** order
    xs, ys = [], []
    
    for d in range(n * n):
        x, y = HilbertCurve.d_to_xy(n, d)
        xs.append(x)
        ys.append(y)
    
    return np.array(xs), np.array(ys)


def generate_hilbert_segments(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """生成 Hilbert 曲线的线段"""
    xs, ys = generate_hilbert_points(order)
    
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = np.linspace(0, 1, len(segments))
    
    return segments, colors


# ============================================================================
# 可视化：Hilbert 曲线生长动画
# ============================================================================

def visualize_hilbert_growth(
    order: int = 4,
    save_path: Optional[Path] = None,
    fps: int = 30,
    duration: float = 5.0,
) -> Optional[animation.FuncAnimation]:
    """生成 Hilbert 曲线生长动画
    
    Args:
        order: 曲线阶数
        save_path: 保存路径 (gif 或 mp4)
        fps: 帧率
        duration: 动画时长 (秒)
    """
    n = 2 ** order
    xs, ys = generate_hilbert_points(order)
    total_points = len(xs)
    
    n_frames = int(fps * duration)
    points_per_frame = max(1, total_points // n_frames)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_title(f'Hilbert Curve Growth (Order {order}: {n}×{n})', fontsize=14)
    
    line, = ax.plot([], [], lw=2, color='blue')
    head, = ax.plot([], [], 'ro', markersize=8)
    progress_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def init():
        line.set_data([], [])
        head.set_data([], [])
        progress_text.set_text('')
        return line, head, progress_text
    
    def animate(frame):
        end_idx = min((frame + 1) * points_per_frame, total_points)
        
        line.set_data(xs[:end_idx], ys[:end_idx])
        
        # 根据进度改变颜色
        progress = end_idx / total_points
        color = plt.cm.viridis(progress)
        line.set_color(color)
        
        if end_idx < total_points:
            head.set_data([xs[end_idx-1]], [ys[end_idx-1]])
        else:
            head.set_data([], [])
        
        progress_text.set_text(f'Progress: {end_idx}/{total_points} ({100*progress:.1f}%)')
        
        return line, head, progress_text
    
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=n_frames, interval=1000/fps, blit=True
    )
    
    if save_path:
        save_path = Path(save_path)
        print(f"[*] Saving animation to {save_path}...")
        
        if save_path.suffix == '.gif':
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps)
        
        anim.save(str(save_path), writer=writer, dpi=100)
        print(f"[OK] Animation saved to: {save_path}")
        plt.close()
    else:
        plt.show()
    
    return anim


# ============================================================================
# 可视化：曲线阶数对比
# ============================================================================

def visualize_order_comparison(
    orders: List[int] = [1, 2, 3, 4, 5],
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """对比不同阶数的 Hilbert 曲线"""
    n_orders = len(orders)
    
    fig, axes = plt.subplots(1, n_orders, figsize=(4 * n_orders, 4))
    if n_orders == 1:
        axes = [axes]
    
    for idx, order in enumerate(orders):
        ax = axes[idx]
        n = 2 ** order
        
        segments, colors = generate_hilbert_segments(order)
        
        lc = LineCollection(segments, cmap=RAINBOW_CMAP, norm=Normalize(0, 1))
        lc.set_array(colors)
        lc.set_linewidth(3 if order <= 2 else 2 if order <= 3 else 1.5 if order <= 4 else 1)
        
        ax.add_collection(lc)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_aspect('equal')
        
        # 添加网格
        if order <= 3:
            for i in range(n + 1):
                ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
                ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        
        ax.set_title(f'Order {order}\n{n}×{n} = {n*n} cells', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.suptitle('Hilbert Curve: Self-Similar Space-Filling Pattern', 
                fontsize=14, fontweight='bold', y=1.02)
    
    # 添加颜色条说明
    sm = plt.cm.ScalarMappable(cmap=RAINBOW_CMAP, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', 
                       fraction=0.05, pad=0.08, aspect=40)
    cbar.set_label('Traversal Progress (Start → End)', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Order comparison saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# 可视化：局部性保持
# ============================================================================

def visualize_locality_preservation(
    order: int = 4,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """可视化 Hilbert 曲线的局部性保持特性
    
    选择一个中心点，展示其在 1D 序列中的邻居在 2D 空间中的分布。
    """
    n = 2 ** order
    center_x, center_y = n // 2, n // 2
    center_d = HilbertCurve.xy_to_d(n, center_x, center_y)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 1. Hilbert 曲线 + 中心点
    ax1 = axes[0, 0]
    xs, ys = generate_hilbert_points(order)
    segments, colors = generate_hilbert_segments(order)
    
    lc = LineCollection(segments, cmap='viridis', norm=Normalize(0, 1), alpha=0.5)
    lc.set_array(colors)
    lc.set_linewidth(1)
    ax1.add_collection(lc)
    
    ax1.scatter([center_x], [center_y], color='red', s=200, zorder=10, 
                marker='*', edgecolors='white', linewidths=2,
                label=f'Center ({center_x}, {center_y})')
    
    ax1.set_xlim(-0.5, n - 0.5)
    ax1.set_ylim(-0.5, n - 0.5)
    ax1.set_aspect('equal')
    ax1.set_title('Hilbert Curve with Center Point', fontsize=12)
    ax1.legend(loc='upper right')
    
    # 2. 1D 邻居在 2D 空间中的分布
    ax2 = axes[0, 1]
    
    # 选择不同距离的 1D 邻居
    neighbor_ranges = [5, 10, 20, 50]
    colors_neighbors = plt.cm.cool(np.linspace(0.2, 0.8, len(neighbor_ranges)))
    
    for i, dist in enumerate(neighbor_ranges):
        neighbor_xs, neighbor_ys = [], []
        for delta in range(-dist, dist + 1):
            d = center_d + delta
            if 0 <= d < n * n:
                x, y = HilbertCurve.d_to_xy(n, d)
                neighbor_xs.append(x)
                neighbor_ys.append(y)
        
        ax2.scatter(neighbor_xs, neighbor_ys, color=colors_neighbors[i], 
                   s=30, alpha=0.7, label=f'±{dist} in 1D')
    
    ax2.scatter([center_x], [center_y], color='red', s=200, zorder=10, 
               marker='*', edgecolors='white', linewidths=2)
    
    ax2.set_xlim(-0.5, n - 0.5)
    ax2.set_ylim(-0.5, n - 0.5)
    ax2.set_aspect('equal')
    ax2.set_title('1D Neighbors in 2D Space (Hilbert)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=8)
    
    # 3. 2D 邻居的 1D 距离分布
    ax3 = axes[1, 0]
    
    d1_distances = []
    d2_distances = []
    
    for y in range(n):
        for x in range(n):
            d = HilbertCurve.xy_to_d(n, x, y)
            d1_dist = abs(d - center_d)
            d2_dist = abs(x - center_x) + abs(y - center_y)  # Manhattan
            d1_distances.append(d1_dist)
            d2_distances.append(d2_dist)
    
    ax3.scatter(d2_distances, d1_distances, alpha=0.3, s=5)
    ax3.set_xlabel('2D Manhattan Distance from Center', fontsize=10)
    ax3.set_ylabel('1D Hilbert Distance from Center', fontsize=10)
    ax3.set_title('2D Distance vs 1D Hilbert Distance', fontsize=12)
    
    # 添加理想线（完美局部性）
    max_d2 = max(d2_distances)
    ax3.plot([0, max_d2], [0, max_d2], 'r--', label='Perfect locality', alpha=0.5)
    ax3.legend()
    
    # 4. 对比：光栅扫描
    ax4 = axes[1, 1]
    
    d1_raster = []
    center_raster = center_y * n + center_x
    
    for y in range(n):
        for x in range(n):
            raster_d = y * n + x
            d1_raster.append(abs(raster_d - center_raster))
    
    ax4.scatter(d2_distances, d1_raster, alpha=0.3, s=5, color='orange')
    ax4.set_xlabel('2D Manhattan Distance from Center', fontsize=10)
    ax4.set_ylabel('1D Raster Distance from Center', fontsize=10)
    ax4.set_title('2D Distance vs 1D Raster Distance', fontsize=12)
    ax4.plot([0, max_d2], [0, max_d2], 'r--', label='Perfect locality', alpha=0.5)
    ax4.legend()
    
    fig.suptitle(f'Hilbert Curve Locality Preservation (Order {order})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Locality visualization saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# 可视化：四叉树递归结构
# ============================================================================

def visualize_quadtree_structure(
    max_depth: int = 4,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """可视化四叉树递归结构及其与 Hilbert 曲线的关系"""
    
    fig, axes = plt.subplots(1, max_depth, figsize=(4 * max_depth, 4))
    if max_depth == 1:
        axes = [axes]
    
    quadrant_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # 红、青、蓝、绿
    quadrant_names = ['Q0 (SW)', 'Q1 (SE)', 'Q2 (NW)', 'Q3 (NE)']
    
    for depth in range(1, max_depth + 1):
        ax = axes[depth - 1]
        n = 2 ** depth
        
        # 绘制 Hilbert 曲线
        xs, ys = generate_hilbert_points(depth)
        segments, colors = generate_hilbert_segments(depth)
        
        lc = LineCollection(segments, cmap='gray', norm=Normalize(0, 1), alpha=0.3)
        lc.set_array(colors)
        lc.set_linewidth(2)
        ax.add_collection(lc)
        
        # 绘制四叉树分割
        def draw_quadrants(x0, y0, size, current_depth, target_depth):
            if current_depth >= target_depth:
                return
            
            half = size // 2
            
            # 绘制分割线
            ax.axhline(y0 + half, xmin=(x0) / n, xmax=(x0 + size) / n,
                        color='black', linewidth=2, alpha=0.7)
            ax.axvline(x0 + half, ymin=(y0) / n, ymax=(y0 + size) / n,
                        color='black', linewidth=2, alpha=0.7)
            
            # 获取 Hilbert 顺序
            hilbert_order = HilbertCurve.get_quadrant_order(current_depth)
            
            # 递归绘制子象限
            quadrant_coords = [
                (x0, y0),           # SW (0)
                (x0 + half, y0),    # SE (1)
                (x0, y0 + half),    # NW (2)
                (x0 + half, y0 + half),  # NE (3)
            ]
            
            for q_idx, order_idx in enumerate(hilbert_order):
                qx, qy = quadrant_coords[order_idx]
                
                # 填充颜色
                if current_depth == target_depth - 1:
                    rect = Rectangle((qx, qy), half, half, 
                                    facecolor=quadrant_colors[q_idx % 4], 
                                    alpha=0.3, edgecolor='none')
                    ax.add_patch(rect)
                    
                    # 标注序号
                    ax.text(qx + half/2, qy + half/2, str(q_idx), 
                            ha='center', va='center', fontsize=10, fontweight='bold')
                
                draw_quadrants(qx, qy, half, current_depth + 1, target_depth)
        
        draw_quadrants(0, 0, n, 0, depth)
        
        ax.set_xlim(-0.1, n + 0.1)
        ax.set_ylim(-0.1, n + 0.1)
        ax.set_aspect('equal')
        ax.set_title(f'Depth {depth}: {n}×{n}', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 添加图例
    legend_patches = [mpatches.Patch(color=quadrant_colors[i], alpha=0.5, 
                                     label=f'Visit {i}: {quadrant_names[i]}')
                     for i in range(4)]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4, fontsize=9)
    
    fig.suptitle('Quadtree Structure with Hilbert Traversal Order', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Quadtree visualization saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# 可视化：2D → 1D 映射展开
# ============================================================================

def visualize_2d_to_1d_mapping(
    order: int = 3,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """可视化 2D 网格到 1D 序列的映射过程"""
    n = 2 ** order
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 2D 网格 + Hilbert 索引
    ax1 = axes[0]
    
    # 绘制网格
    for i in range(n + 1):
        ax1.axhline(i, color='gray', linewidth=0.5)
        ax1.axvline(i, color='gray', linewidth=0.5)
    
    # 填充每个格子的 Hilbert 索引
    for y in range(n):
        for x in range(n):
            d = HilbertCurve.xy_to_d(n, x, y)
            color = plt.cm.viridis(d / (n * n))
            rect = Rectangle((x, y), 1, 1, facecolor=color, alpha=0.7)
            ax1.add_patch(rect)
            ax1.text(x + 0.5, y + 0.5, str(d), ha='center', va='center', 
                    fontsize=8, color='white' if d / (n * n) > 0.5 else 'black')
    
    ax1.set_xlim(0, n)
    ax1.set_ylim(0, n)
    ax1.set_aspect('equal')
    ax1.set_title('2D Grid with Hilbert Index', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # 2. 连接线：2D → 1D
    ax2 = axes[1]
    ax2.set_xlim(0, 2)
    ax2.set_ylim(0, n * n)
    ax2.axis('off')
    ax2.set_title('Mapping', fontsize=12)
    
    # 简化：只显示部分连接
    sample_indices = list(range(0, n * n, max(1, n * n // 16)))
    for d in sample_indices:
        x, y = HilbertCurve.d_to_xy(n, d)
        
        # 2D 位置 (左侧)
        y_2d = (y + 0.5) / n * (n * n)
        
        # 1D 位置 (右侧)
        y_1d = d + 0.5
        
        color = plt.cm.viridis(d / (n * n))
        ax2.annotate('', xy=(1.8, y_1d), xytext=(0.2, y_2d),
                    arrowprops=dict(arrowstyle='->', color=color, alpha=0.5, lw=1))
    
    ax2.text(0.1, n * n / 2, '2D', fontsize=12, ha='center', va='center')
    ax2.text(1.9, n * n / 2, '1D', fontsize=12, ha='center', va='center')
    
    # 3. 1D 序列
    ax3 = axes[2]
    
    cell_height = 1
    for d in range(n * n):
        color = plt.cm.viridis(d / (n * n))
        rect = Rectangle((0, d), 1, cell_height, facecolor=color, 
                         edgecolor='white', linewidth=0.5)
        ax3.add_patch(rect)
        
        if n <= 4:  # 显示坐标
            x, y = HilbertCurve.d_to_xy(n, d)
            ax3.text(0.5, d + 0.5, f'{d}: ({x},{y})', ha='center', va='center', 
                    fontsize=7, color='white' if d / (n * n) > 0.5 else 'black')
    
    ax3.set_xlim(-0.1, 1.1)
    ax3.set_ylim(-0.5, n * n + 0.5)
    ax3.set_aspect(0.1)
    ax3.set_title('1D Sequence (Hilbert Order)', fontsize=12)
    ax3.set_ylabel('Token Index')
    ax3.set_xticks([])
    
    fig.suptitle(f'2D → 1D Hilbert Mapping (Order {order})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] 2D to 1D mapping saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# 可视化：多尺度层级结构
# ============================================================================

def visualize_multiscale_hierarchy(
    base_size: int = 64,
    patch_sizes: List[int] = [4, 8, 16],
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """可视化多尺度分层结构"""
    n_scales = len(patch_sizes)
    
    fig, axes = plt.subplots(2, n_scales, figsize=(5 * n_scales, 10))
    
    # 创建模拟图像（棋盘格）
    img = np.zeros((base_size, base_size))
    for i in range(base_size):
        for j in range(base_size):
            img[i, j] = ((i // 8) + (j // 8)) % 2
    
    for s, ps in enumerate(patch_sizes):
        grid_h, grid_w = base_size // ps, base_size // ps
        
        # 上排：显示 patch 划分
        ax_top = axes[0, s]
        ax_top.imshow(img, cmap='gray', alpha=0.5)
        
        # 绘制网格
        for i in range(grid_h + 1):
            ax_top.axhline(i * ps - 0.5, color='red', linewidth=1.5)
        for j in range(grid_w + 1):
            ax_top.axvline(j * ps - 0.5, color='red', linewidth=1.5)
        
        ax_top.set_title(f'Scale {s+1}: {ps}×{ps} patches\n({grid_h}×{grid_w} = {grid_h*grid_w} tokens)', 
                        fontsize=11)
        ax_top.axis('off')
        
        # 下排：Hilbert 遍历
        ax_bot = axes[1, s]
        ax_bot.imshow(img, cmap='gray', alpha=0.3)
        
        # 生成 Hilbert 路径
        grid_size = max(grid_h, grid_w)
        n = 1
        while n < grid_size:
            n *= 2
        
        path_points = []
        for d in range(n * n):
            x, y = HilbertCurve.d_to_xy(n, d)
            if x < grid_w and y < grid_h:
                cx = x * ps + ps // 2 - 0.5
                cy = y * ps + ps // 2 - 0.5
                path_points.append((cx, cy, d))
        
        if len(path_points) > 1:
            xs = [p[0] for p in path_points]
            ys = [p[1] for p in path_points]
            
            points = np.array([xs, ys]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            colors = np.linspace(0, 1, len(segments))
            
            lc = LineCollection(segments, cmap=RAINBOW_CMAP, 
                               norm=Normalize(0, 1), linewidths=2)
            lc.set_array(colors)
            ax_bot.add_collection(lc)
        
        ax_bot.set_xlim(-0.5, base_size - 0.5)
        ax_bot.set_ylim(base_size - 0.5, -0.5)
        ax_bot.set_title(f'Hilbert Order ({len(path_points)} tokens)', fontsize=11)
        ax_bot.axis('off')
    
    fig.suptitle(f'Multi-Scale Tokenization ({base_size}×{base_size} image)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Multi-scale hierarchy saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# 可视化：混合尺度（Mixed Level）分割演示
# ============================================================================

def visualize_mixed_level_segmentation(
    base_size: int = 128,
    patch_sizes: List[int] = [4, 8, 16],
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """可视化混合尺度分割的概念演示
    
    模拟模型如何根据区域复杂度自适应选择不同的 patch 大小：
    - 高复杂度区域 → 小 patch（精细分割）
    - 低复杂度区域 → 大 patch（粗糙分割）
    
    这是一个概念演示，不需要实际模型，使用合成数据展示原理。
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # ===== 创建模拟图像（带复杂度变化）=====
    np.random.seed(42)
    
    # 创建复杂度图 (0=简单, 1=复杂)
    complexity_map = np.zeros((base_size, base_size))
    
    # 中心区域高复杂度
    center = base_size // 2
    for i in range(base_size):
        for j in range(base_size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist < base_size // 4:
                complexity_map[i, j] = 0.9  # 高复杂度
            elif dist < base_size // 2:
                complexity_map[i, j] = 0.5  # 中等复杂度
            else:
                complexity_map[i, j] = 0.2  # 低复杂度
    
    # 左上角添加高复杂度区域
    complexity_map[:base_size//4, :base_size//4] = 0.85
    
    # 创建合成图像（基于复杂度添加细节）
    img = np.zeros((base_size, base_size, 3))
    for i in range(base_size):
        for j in range(base_size):
            if complexity_map[i, j] > 0.7:
                # 高频细节（棋盘格）
                img[i, j] = [(i + j) % 2 * 0.8 + 0.1] * 3
            elif complexity_map[i, j] > 0.4:
                # 中频细节（条纹）
                img[i, j] = [(i % 4 < 2) * 0.5 + 0.3] * 3
            else:
                # 低频（平滑）
                img[i, j] = [0.7, 0.8, 0.9]  # 浅蓝色背景
    
    # ===== 1. 原始图像 =====
    ax1 = axes[0, 0]
    ax1.imshow(img)
    ax1.set_title('Synthetic Image\n(with varying complexity)', fontsize=11)
    ax1.axis('off')
    
    # ===== 2. 复杂度热力图 =====
    ax2 = axes[0, 1]
    im = ax2.imshow(complexity_map, cmap='hot', vmin=0, vmax=1)
    ax2.set_title('Complexity Map\n(simulated edge density)', fontsize=11)
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, label='Complexity')
    
    # ===== 3. 尺度选择 =====
    # 根据复杂度分配尺度：高复杂度→小patch，低复杂度→大patch
    scale_colors = ['#FF6B6B', '#45B7D1', '#96CEB4']  # 红(4)、蓝(8)、绿(16)
    scale_labels = [f'{ps}×{ps}' for ps in patch_sizes]
    
    # 使用最小的 patch size 来划分决策网格
    min_ps = min(patch_sizes)
    grid_h, grid_w = base_size // min_ps, base_size // min_ps
    
    # 为每个网格位置计算平均复杂度并选择尺度
    scale_map = np.zeros((grid_h, grid_w), dtype=int)
    for i in range(grid_h):
        for j in range(grid_w):
            region = complexity_map[i*min_ps:(i+1)*min_ps, j*min_ps:(j+1)*min_ps]
            avg_complexity = np.mean(region)
            
            # 根据复杂度选择尺度
            if avg_complexity > 0.6:
                scale_map[i, j] = 0  # 最细 (4×4)
            elif avg_complexity > 0.35:
                scale_map[i, j] = 1  # 中等 (8×8)
            else:
                scale_map[i, j] = 2  # 最粗 (16×16)
    
    ax3 = axes[0, 2]
    scale_img = np.zeros((grid_h, grid_w, 3))
    for s in range(len(patch_sizes)):
        mask = scale_map == s
        color = np.array(plt.cm.colors.hex2color(scale_colors[s]))
        scale_img[mask] = color
    
    ax3.imshow(scale_img, interpolation='nearest', 
                extent=[0, base_size, base_size, 0])
    ax3.set_title('Scale Selection\n(based on complexity)', fontsize=11)
    ax3.axis('off')
    
    # 添加图例
    legend_patches = [mpatches.Patch(color=scale_colors[i], 
                                    label=f'{scale_labels[i]} (Scale {i+1})')
                    for i in range(len(patch_sizes))]
    ax3.legend(handles=legend_patches, loc='upper right', fontsize=8)
    
    # ===== 4. 混合分割叠加 =====
    ax4 = axes[1, 0]
    ax4.imshow(img, alpha=0.6)
    
    # 绘制混合 patch 边界
    for i in range(grid_h):
        for j in range(grid_w):
            scale = scale_map[i, j]
            ps = patch_sizes[scale]
            
            # 只在 patch 边界绘制
            x0, y0 = j * min_ps, i * min_ps
            
            # 检查是否是该尺度 patch 的左上角
            if scale == 0:  # 4×4, 每个格子都画
                rect = Rectangle((x0, y0), min_ps, min_ps, 
                                fill=False, edgecolor=scale_colors[0], 
                                linewidth=1.5, alpha=0.8)
                ax4.add_patch(rect)
            elif scale == 1:  # 8×8
                if i % 2 == 0 and j % 2 == 0:
                    rect = Rectangle((x0, y0), min_ps * 2, min_ps * 2, 
                                    fill=False, edgecolor=scale_colors[1], 
                                    linewidth=2, alpha=0.8)
                    ax4.add_patch(rect)
            else:  # 16×16
                if i % 4 == 0 and j % 4 == 0:
                    rect = Rectangle((x0, y0), min_ps * 4, min_ps * 4, 
                                    fill=False, edgecolor=scale_colors[2], 
                                    linewidth=2.5, alpha=0.8)
                    ax4.add_patch(rect)
    
    ax4.set_xlim(0, base_size)
    ax4.set_ylim(base_size, 0)
    ax4.set_title('Mixed-Level Segmentation\n(adaptive patches)', fontsize=11)
    ax4.axis('off')
    
    # ===== 5. Hilbert 遍历路径 - 展示不同尺度的像素区域大小 =====
    # 核心：每个 token 位置用方块大小表示其尺度（patch_size）
    ax5 = axes[1, 1]
    ax5.imshow(img, alpha=0.3)
    
    # 使用最细网格的 Hilbert 路径
    n = 1
    while n < grid_h:
        n *= 2
    
    # 按 Hilbert 顺序收集点，并绘制表示尺度的矩形
    hilbert_order = []
    for d in range(n * n):
        x, y = HilbertCurve.d_to_xy(n, d)
        if x < grid_w and y < grid_h:
            scale = scale_map[y, x]
            ps = patch_sizes[scale]
            # 像素坐标
            px = x * min_ps
            py = y * min_ps
            # 中心点（用于连线）
            cx = px + min_ps // 2
            cy = py + min_ps // 2
            hilbert_order.append((cx, cy, px, py, scale, ps))
    
    # 先画 Hilbert 连线（深色，更明显）
    if len(hilbert_order) > 1:
        for i in range(len(hilbert_order) - 1):
            cx1, cy1 = hilbert_order[i][0], hilbert_order[i][1]
            cx2, cy2 = hilbert_order[i+1][0], hilbert_order[i+1][1]
            ax5.plot([cx1, cx2], [cy1, cy2], 
                    color='#2C3E50', linewidth=1.2, alpha=0.85, zorder=1)
    
    # 绘制每个 token 的覆盖区域（用方块大小表示尺度）
    # 为避免重叠，只在每个 patch 的"起始位置"绘制
    drawn_patches = set()
    for idx, (cx, cy, px, py, scale, ps) in enumerate(hilbert_order):
        # 计算这个 token 对应的 patch 左上角（按其尺度对齐）
        patch_row = (py // ps) * ps
        patch_col = (px // ps) * ps
        patch_key = (patch_row, patch_col, scale)
        
        if patch_key not in drawn_patches:
            drawn_patches.add(patch_key)
            
            # 绘制 patch 区域（填充+边框）
            rect = Rectangle((patch_col, patch_row), ps, ps,
                            facecolor=scale_colors[scale], alpha=0.4,
                            edgecolor=scale_colors[scale], linewidth=2,
                            zorder=2)
            ax5.add_patch(rect)
            
            # 在 patch 中心标注序号（表示 Hilbert 遍历顺序）
            token_idx = len(drawn_patches)
            if ps >= 8:  # 只在较大的 patch 中显示序号
                ax5.text(patch_col + ps/2, patch_row + ps/2, 
                        str(token_idx), ha='center', va='center',
                        fontsize=7 if ps >= 16 else 5, fontweight='bold',
                        color='white', zorder=3)
    
    ax5.set_xlim(0, base_size)
    ax5.set_ylim(base_size, 0)
    ax5.set_title(f'Hilbert Traversal with Variable Patch Sizes\n'
                 f'(box size = patch size, {len(drawn_patches)} tokens)', fontsize=11)
    
    # 图例
    legend_patches = [mpatches.Patch(color=scale_colors[i], alpha=0.5,
                                     label=f'{patch_sizes[i]}×{patch_sizes[i]} px')
                     for i in range(len(patch_sizes))]
    ax5.legend(handles=legend_patches, loc='upper right', fontsize=8)
    ax5.axis('off')
    
    # ===== 6. Token 数量对比 =====
    ax6 = axes[1, 2]
    
    # 计算各种方案的 token 数
    fixed_tokens = [(base_size // ps) ** 2 for ps in patch_sizes]
    
    # 混合尺度 token 数估算
    mixed_tokens = 0
    for scale_idx, ps in enumerate(patch_sizes):
        count = np.sum(scale_map == scale_idx)
        tokens_per_region = (ps // min_ps) ** 2
        mixed_tokens += count // tokens_per_region
    
    labels = [f'Fixed {ps}×{ps}' for ps in patch_sizes] + ['Mixed Level']
    values = fixed_tokens + [mixed_tokens]
    colors = scale_colors + ['#FFD93D']  # 黄色表示混合
    
    bars = ax6.bar(labels, values, color=colors, edgecolor='black', linewidth=1)
    ax6.set_ylabel('Number of Tokens', fontsize=10)
    ax6.set_title('Token Count Comparison', fontsize=11)
    
    # 在柱状图上标注数值
    for bar, val in zip(bars, values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(int(val)), ha='center', va='bottom', fontsize=9)
    
    ax6.set_ylim(0, max(values) * 1.15)
    plt.setp(ax6.get_xticklabels(), rotation=15, ha='right')
    
    fig.suptitle(f'Mixed-Level (Adaptive) Tokenization Demo\n'
                f'Image Size: {base_size}×{base_size}, Patch Sizes: {patch_sizes}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Mixed-level segmentation saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# 生成所有可视化
# ============================================================================

def generate_all_visualizations(
    output_dir: Path,
    show: bool = False,
) -> None:
    """生成所有可视化"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Generating Fractal Curve Visualizations")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # 1. 阶数对比
    print("[1/7] Generating order comparison...")
    visualize_order_comparison(
        orders=[1, 2, 3, 4, 5],
        save_path=output_dir / "hilbert_order_comparison.png",
        show=show,
    )
    
    # 2. 局部性保持
    print("[2/7] Generating locality preservation...")
    visualize_locality_preservation(
        order=4,
        save_path=output_dir / "hilbert_locality.png",
        show=show,
    )
    
    # 3. 四叉树结构
    print("[3/7] Generating quadtree structure...")
    visualize_quadtree_structure(
        max_depth=4,
        save_path=output_dir / "quadtree_structure.png",
        show=show,
    )
    
    # 4. 2D → 1D 映射
    print("[4/7] Generating 2D to 1D mapping...")
    visualize_2d_to_1d_mapping(
        order=3,
        save_path=output_dir / "2d_to_1d_mapping.png",
        show=show,
    )
    
    # 5. 多尺度层级
    print("[5/7] Generating multi-scale hierarchy...")
    visualize_multiscale_hierarchy(
        base_size=64,
        patch_sizes=[4, 8, 16],
        save_path=output_dir / "multiscale_hierarchy.png",
        show=show,
    )
    
    # 6. 混合尺度分割演示
    print("[6/7] Generating mixed-level segmentation demo...")
    visualize_mixed_level_segmentation(
        base_size=64,
        patch_sizes=[4, 8, 16],
        save_path=output_dir / "mixed_level_segmentation.png",
        show=show,
    )
    
    # 7. 动画 (可选)
    print("[7/7] Generating growth animation...")
    try:
        visualize_hilbert_growth(
            order=4,
            save_path=output_dir / "hilbert_growth.gif",
            fps=20,
            duration=4.0,
        )
    except Exception as e:
        print(f"      [WARN] Animation generation failed: {e}")
        print("      (This may require additional dependencies like Pillow)")
    
    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Files saved to: {output_dir}")
    print(f"  - hilbert_order_comparison.png")
    print(f"  - hilbert_locality.png")
    print(f"  - quadtree_structure.png")
    print(f"  - 2d_to_1d_mapping.png")
    print(f"  - multiscale_hierarchy.png")
    print(f"  - mixed_level_segmentation.png")
    print(f"  - hilbert_growth.gif (if successful)")
    print(f"{'='*60}\n")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fractal Curve Visualization")
    
    # 模式
    parser.add_argument("--all", action="store_true",
                       help="Generate all visualizations")
    parser.add_argument("--animate", action="store_true",
                       help="Generate animation only")
    parser.add_argument("--comparison", action="store_true",
                       help="Generate order comparison")
    parser.add_argument("--locality", action="store_true",
                       help="Generate locality visualization")
    parser.add_argument("--quadtree", action="store_true",
                       help="Generate quadtree structure")
    parser.add_argument("--mapping", action="store_true",
                       help="Generate 2D to 1D mapping")
    parser.add_argument("--multiscale", action="store_true",
                       help="Generate multi-scale hierarchy")
    parser.add_argument("--mixed-level", action="store_true",
                       help="Generate mixed-level (adaptive) segmentation demo")
    
    # 参数
    parser.add_argument("--order", type=int, default=4,
                       help="Hilbert curve order")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--show", action="store_true",
                       help="Show plots interactively")
    
    args = parser.parse_args()
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "workspace" / "visualizations" / "fractal_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 如果没有指定任何选项，默认生成所有
    if not any([args.all, args.animate, args.comparison, args.locality, 
                args.quadtree, args.mapping, args.multiscale, args.mixed_level]):
        args.all = True
    
    if args.all:
        generate_all_visualizations(output_dir, show=args.show)
        return
    
    if args.animate:
        visualize_hilbert_growth(
            order=args.order,
            save_path=output_dir / f"hilbert_growth_order{args.order}.gif",
        )
    
    if args.comparison:
        visualize_order_comparison(
            save_path=output_dir / "hilbert_order_comparison.png",
            show=args.show,
        )
    
    if args.locality:
        visualize_locality_preservation(
            order=args.order,
            save_path=output_dir / f"hilbert_locality_order{args.order}.png",
            show=args.show,
        )
    
    if args.quadtree:
        visualize_quadtree_structure(
            save_path=output_dir / "quadtree_structure.png",
            show=args.show,
        )
    
    if args.mapping:
        visualize_2d_to_1d_mapping(
            order=min(args.order, 4),  # 限制阶数避免太密
            save_path=output_dir / f"2d_to_1d_mapping_order{min(args.order, 4)}.png",
            show=args.show,
        )
    
    if args.multiscale:
        visualize_multiscale_hierarchy(
            save_path=output_dir / "multiscale_hierarchy.png",
            show=args.show,
        )
    
    if args.mixed_level:
        visualize_mixed_level_segmentation(
            base_size=64,
            patch_sizes=[4, 8, 16],
            save_path=output_dir / "mixed_level_segmentation.png",
            show=args.show,
        )


if __name__ == "__main__":
    main()
