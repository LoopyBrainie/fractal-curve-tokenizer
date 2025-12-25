#!/usr/bin/env python3
"""Fractal Curve ViT 交互式可视化脚本

数学形式化
============

本脚本可视化 Fractal Curve ViT 的核心数学结构：

1. **Hilbert 曲线**:
   H: [0, n²) ↔ [0, n) × [0, n)
   局部性: ||p1 - p2||_2 ≤ C · |H⁻¹(p1) - H⁻¹(p2)|^(1/2)

2. **LCA (最低公共祖先) 偏置**:
   LCA(i, j) = 第一个不同象限的层级 ∈ [0, L]
   B[i,j] = LCAEmbed(LCA(i,j))

3. **Variable Depth Tokens (V3, 推荐)**:
   Regions = AdaptiveQuadtreeSplit(I)  # 内容自适应分割
   F = SharedConv(I)                    # 共享特征提取
   Token_i = Pool(F[R_i]) * σ_d + E_d  # 区域池化 + 深度编码
   
   优势:
   - 密集梯度流: 共享特征提取器所有路径都收到梯度
   - 无温度参数: 训练更稳定
   - 自适应分割: 根据图像内容动态决定分割深度

可视化功能
----------
1. Hilbert 曲线动画生成
2. 不同阶数的曲线对比  
3. LCA 距离矩阵可视化
4. 四叉树路径与偏置关系
5. 多尺度深度分布可视化
6. 注意力偏置矩阵可视化

使用示例：
    # 生成所有可视化
    python visualize_fractal_curves.py --all
    
    # 生成 LCA 偏置可视化
    python visualize_fractal_curves.py --lca-bias
    
    # 生成多尺度深度分布可视化
    python visualize_fractal_curves.py --depth-distribution
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
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
# 可视化：LCA (最低公共祖先) 偏置矩阵
# ============================================================================

def compute_quadtree_path(n: int, d: int, max_depth: int) -> List[int]:
    """计算 Hilbert 索引的四叉树路径.
    
    数学定义:
        path[ℓ] = 第 ℓ 层的象限索引 ∈ {0, 1, 2, 3}
        
    对于 n = 2^k，path 长度为 k
    """
    path = []
    x, y = HilbertCurve.d_to_xy(n, d)
    
    size = n
    for _ in range(max_depth):
        size //= 2
        if size == 0:
            break
        qx = 1 if x >= size else 0
        qy = 1 if y >= size else 0
        quadrant = qy * 2 + qx
        path.append(quadrant)
        x = x % size
        y = y % size
    
    return path


def compute_lca_depth(path_i: List[int], path_j: List[int]) -> int:
    """计算两条四叉树路径的 LCA 深度.
    
    数学定义:
        LCA(i, j) = min{ℓ : path_i[ℓ] ≠ path_j[ℓ]}
        若完全相同则返回 len(path)
    """
    min_len = min(len(path_i), len(path_j))
    for l in range(min_len):
        if path_i[l] != path_j[l]:
            return l
    return min_len


def visualize_lca_bias_matrix(
    order: int = 4,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """可视化 LCA 偏置矩阵.
    
    数学原理:
        LCA(i, j) = 两个 token 的最低公共祖先深度
        B[i,j] = LCAEmbed(LCA(i,j))
        
    LCA 值越小表示两个 token 在四叉树中越早分开，
    即它们在空间上相距越远。
    """
    n = 2 ** order
    num_tokens = n * n
    max_depth = order
    
    # 计算所有 token 的四叉树路径
    paths = []
    for d in range(num_tokens):
        path = compute_quadtree_path(n, d, max_depth)
        paths.append(path)
    
    # 计算 LCA 矩阵
    lca_matrix = np.zeros((num_tokens, num_tokens), dtype=int)
    for i in range(num_tokens):
        for j in range(num_tokens):
            lca_matrix[i, j] = compute_lca_depth(paths[i], paths[j])
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. LCA 矩阵热图
    ax1 = axes[0]
    im1 = ax1.imshow(lca_matrix, cmap='viridis', aspect='auto')
    ax1.set_title(f'LCA Depth Matrix ({n}×{n} = {num_tokens} tokens)', fontsize=12)
    ax1.set_xlabel('Token j (Hilbert order)')
    ax1.set_ylabel('Token i (Hilbert order)')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('LCA Depth')
    
    # 2. LCA 分布直方图
    ax2 = axes[1]
    unique, counts = np.unique(lca_matrix, return_counts=True)
    colors = plt.cm.viridis(unique / max(unique))
    bars = ax2.bar(unique, counts, color=colors, edgecolor='black')
    ax2.set_xlabel('LCA Depth', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Distribution of LCA Depths', fontsize=12)
    ax2.set_xticks(unique)
    
    # 标注
    for bar, u, c in zip(bars, unique, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{c}', ha='center', va='bottom', fontsize=8)
    
    # 3. LCA 与空间距离的关系
    ax3 = axes[2]
    
    # 采样一些 token 对
    sample_size = min(2000, num_tokens * num_tokens // 4)
    np.random.seed(42)
    i_samples = np.random.randint(0, num_tokens, sample_size)
    j_samples = np.random.randint(0, num_tokens, sample_size)
    
    lca_samples = []
    dist_samples = []
    for i, j in zip(i_samples, j_samples):
        xi, yi = HilbertCurve.d_to_xy(n, i)
        xj, yj = HilbertCurve.d_to_xy(n, j)
        spatial_dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
        lca_samples.append(lca_matrix[i, j])
        dist_samples.append(spatial_dist)
    
    ax3.scatter(lca_samples, dist_samples, alpha=0.3, s=5)
    ax3.set_xlabel('LCA Depth', fontsize=11)
    ax3.set_ylabel('Spatial Distance (Euclidean)', fontsize=11)
    ax3.set_title('LCA Depth vs Spatial Distance', fontsize=12)
    
    # 添加趋势线
    for lca_val in range(max_depth + 1):
        mask = np.array(lca_samples) == lca_val
        if mask.sum() > 0:
            avg_dist = np.mean(np.array(dist_samples)[mask])
            ax3.scatter([lca_val], [avg_dist], color='red', s=100, 
                       marker='x', zorder=5, linewidths=2)
    
    ax3.scatter([], [], color='red', marker='x', s=100, label='Mean', linewidths=2)
    ax3.legend()
    
    fig.suptitle('LCA (Lowest Common Ancestor) Bias Analysis\n'
                'B[i,j] = LCAEmbed(LCA(i,j)) captures hierarchical spatial relationships',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] LCA bias matrix saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# 可视化：Variable Depth Tokens 机制 (V3, 推荐)
# ============================================================================

def visualize_cross_scale_attention(
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """可视化 Variable Depth Tokens 多尺度分词机制 (V3).
    
    数学原理:
        1. 内容自适应分割:
           Regions = AdaptiveQuadtreeSplit(I)
           - 根据图像内容复杂度决定分割深度
           - 复杂区域 → 细粒度 patch (4x4)
           - 平滑区域 → 粗粒度 patch (16x16)
           
        2. 共享特征提取:
           F = SharedConv(I)
           - 单个卷积网络处理所有区域
           - 所有路径共享梯度
           
        3. 区域池化 + 深度编码:
           Token_i = Pool(F[R_i]) * σ_d + E_d
           - Pool: 根据区域大小自适应池化
           - σ_d: 深度相关缩放因子
           - E_d: 深度位置编码
    
    优势:
        - 密集梯度流: 共享特征提取器所有路径都收到梯度
        - 无温度参数: 训练更稳定
        - 自适应分割: 根据图像内容动态决定分割深度
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 设置尺度
    patch_sizes = [4, 8, 16]
    num_scales = len(patch_sizes)
    
    # 1. Variable Depth Tokens 架构示意图
    ax1 = axes[0, 0]
    ax1.axis('off')
    
    arch_text = """
    Variable Depth Tokens Architecture
    ====================================
    
    Input: Image I ∈ R^(H×W×C)
    
    Step 1: Adaptive Quadtree Split
      Regions = AdaptiveQuadtreeSplit(I)
      - Split based on content complexity
      - Complex → fine patches (4x4)
      - Smooth → coarse patches (16x16)
    
    Step 2: Shared Feature Extraction  
      F = SharedConv(I)  # [B, D, H/4, W/4]
      - Single convnet for all depths
    
    Step 3: Region Pooling + Depth Encoding
      For each region R_k at depth d:
        Token_k = Pool(F[R_k]) * σ_d + E_d
    
    Output: Variable-length tokens [B, N_var, D]
    """
    
    ax1.text(0.05, 0.95, arch_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#e8f4e8', alpha=0.9))
    ax1.set_title('Architecture Overview', fontsize=11)
    
    # 2. 梯度流示意图
    ax2 = axes[0, 1]
    
    # 模拟梯度流 - 所有区域都共享卷积权重
    grid_size = 5
    num_depths = 4  # 深度 0-3
    
    # 所有深度的区域都收到梯度 (通过共享卷积)
    gradient_density = np.ones((grid_size, num_depths)) * 0.8
    # 添加一些变化来显示不同区域的贡献
    np.random.seed(42)
    gradient_density += np.random.rand(grid_size, num_depths) * 0.2
    
    im2 = ax2.imshow(gradient_density.T, aspect='auto', cmap='Greens',
                     vmin=0, vmax=1)
    
    ax2.set_xlabel('Region Index')
    ax2.set_ylabel('Depth Level')
    ax2.set_yticks(range(num_depths))
    ax2.set_yticklabels([f'd={d}' for d in range(num_depths)])
    ax2.set_title('Gradient Flow: All Depths Share Gradients', fontsize=11)
    plt.colorbar(im2, ax=ax2, label='Gradient Magnitude')
    
    # 添加说明
    ax2.text(2.5, -0.8, 'SharedConv receives gradients\nfrom ALL depth levels', 
             ha='center', fontsize=9, style='italic')
    
    # 3. 深度分布示例
    ax3 = axes[0, 2]
    
    # 模拟不同区域的深度分布
    region_types = {
        'Edge (complex)': np.array([0.1, 0.2, 0.3, 0.4]),   # 偏向细粒度
        'Texture': np.array([0.2, 0.4, 0.3, 0.1]),          # 中等粒度
        'Background (smooth)': np.array([0.5, 0.3, 0.15, 0.05]), # 偏向粗粒度
    }
    
    x_pos = np.arange(num_depths)
    width = 0.25
    colors_regions = ['#FF6B6B', '#FFD93D', '#4ECDC4']
    
    for i, (region, weights) in enumerate(region_types.items()):
        ax3.bar(x_pos + i * width, weights, width,
               label=region, color=colors_regions[i], edgecolor='black')
    
    ax3.set_xlabel('Depth Level')
    ax3.set_ylabel('Proportion of Tokens')
    ax3.set_title('Depth Distribution by Region Type', fontsize=11)
    ax3.set_xticks(x_pos + width)
    ax3.set_xticklabels([f'd={d}\n({16//(2**d)}x{16//(2**d)})' for d in range(num_depths)])
    ax3.legend(fontsize=9)
    ax3.set_ylim(0, 0.6)
    
    # 4. 深度编码 (Depth Encoding) 的作用
    ax4 = axes[1, 0]
    
    # 模拟深度编码向量 (降维到 2D 可视化)
    np.random.seed(456)
    d_model = 8  # 简化示例
    depth_embeddings = np.random.randn(num_depths, d_model)
    
    # 使用 PCA 降到 2D
    from numpy.linalg import svd
    U, S, Vt = svd(depth_embeddings - depth_embeddings.mean(axis=0), full_matrices=False)
    depth_2d = U[:, :2] * S[:2]
    
    colors_depths = plt.cm.viridis(np.linspace(0.2, 0.8, num_depths))
    for d in range(num_depths):
        ps = 16 // (2 ** d)
        ax4.scatter(depth_2d[d, 0], depth_2d[d, 1], 
                   s=200, c=[colors_depths[d]], edgecolors='black', linewidths=2,
                   label=f'd={d} ({ps}x{ps})', zorder=10)
        ax4.annotate(f'd={d}', (depth_2d[d, 0], depth_2d[d, 1]),
                    textcoords='offset points', xytext=(10, 10), fontsize=10)
    
    ax4.set_xlabel('Principal Component 1')
    ax4.set_ylabel('Principal Component 2')
    ax4.set_title('Depth Embedding Visualization\n(learned to distinguish depths)', fontsize=11)
    ax4.legend(loc='lower right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # 5. 自适应分割示例
    ax5 = axes[1, 1]
    
    # 模拟 32x32 图像的自适应四叉树分割
    grid_size = 32
    
    # 创建示例分割 (复杂区域细分割，平滑区域粗分割)
    depth_map = np.zeros((grid_size, grid_size))
    
    # 模拟：边缘区域 (对角线附近) 细分割
    for i in range(grid_size):
        for j in range(grid_size):
            dist_to_diag = abs(i - j) / grid_size
            # 越靠近对角线，深度越大 (更细)
            if dist_to_diag < 0.1:
                depth_map[i, j] = 3  # 4x4 patches
            elif dist_to_diag < 0.25:
                depth_map[i, j] = 2  # 8x8 patches
            elif dist_to_diag < 0.5:
                depth_map[i, j] = 1  # 16x16 patches
            else:
                depth_map[i, j] = 0  # 32x32 patches (coarsest)
    
    # 添加一些噪声来模拟实际情况
    np.random.seed(789)
    noise = np.random.randint(-1, 2, (grid_size, grid_size))
    depth_map = np.clip(depth_map + noise * 0.3, 0, 3).astype(int)
    
    cmap = plt.cm.viridis
    im5 = ax5.imshow(depth_map, cmap=cmap, vmin=0, vmax=3)
    ax5.set_title('Adaptive Quadtree Depth Map\n(Complex regions → finer patches)', fontsize=11)
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    
    cbar = plt.colorbar(im5, ax=ax5, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['d=0 (16x16)', 'd=1 (8x8)', 'd=2 (4x4)', 'd=3 (2x2)'])
    
    # 6. Variable Depth Tokens 优势总结
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = """
    Variable Depth Tokens (V3) Advantages
    ======================================
    
    Architecture:
      - AdaptiveQuadtreeSplit for content-aware segmentation
      - SharedConv for unified feature extraction
      - Depth encoding for scale awareness
    
    Key Benefits:
      ✓ Dense Gradient Flow
        - All paths share the same convolution
        - Every depth level contributes to learning
      
      ✓ No Temperature Tuning
        - Deterministic splitting based on complexity
        - No Gumbel noise or annealing needed
      
      ✓ Content Adaptive
        - Complex regions → fine patches (more tokens)
        - Smooth regions → coarse patches (fewer tokens)
      
      ✓ Efficient Computation
        - Variable token count adapts to image content
        - Simpler architecture than attention-based fusion
    
    Mathematical Form:
      Token_k = Pool(F[R_k]) * σ_d + E_d
      where:
        F = SharedConv(I)        # shared features
        R_k = quadtree region    # adaptive split
        σ_d = depth scale factor # learnable
        E_d = depth embedding    # position encoding
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f0f8ff', alpha=0.9))
    ax6.set_title('V3 Advantages Summary', fontsize=11)
    
    fig.suptitle('Variable Depth Tokens Mechanism (V3)\n'
                'Content-Adaptive Multi-Scale Tokenization',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Variable Depth Tokens visualization saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# 可视化：Gumbel-Softmax 尺度选择过程 (V2, 已弃用)
# ============================================================================

def visualize_gumbel_softmax_decision(
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """可视化 Gumbel-Softmax 尺度选择机制.
    
    数学原理:
        π_hard = Gumbel-Softmax(logits, τ, hard=True)
        
        前向: argmax(logits + Gumbel_noise) → one-hot
        反向: 软梯度通过 softmax
        
    温度 τ 控制决策的"软硬程度"：
        τ → 0: 接近 argmax (硬决策)
        τ → ∞: 接近均匀分布 (软决策)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 设置尺度和 logits
    patch_sizes = [4, 8, 16]
    num_scales = len(patch_sizes)
    
    # 示例 logits (模拟不同复杂度区域)
    logits_examples = {
        'High Complexity': np.array([2.0, 0.5, -0.5]),  # 偏向小 patch
        'Medium Complexity': np.array([0.5, 1.5, 0.5]),  # 偏向中等 patch
        'Low Complexity': np.array([-0.5, 0.5, 2.0]),   # 偏向大 patch
    }
    
    # 1. 不同温度下的 softmax 分布
    ax1 = axes[0, 0]
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    logits = np.array([1.5, 1.0, 0.5])  # 示例 logits
    
    x_pos = np.arange(num_scales)
    width = 0.15
    
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(temperatures)))
    for i, tau in enumerate(temperatures):
        probs = np.exp(logits / tau)
        probs = probs / probs.sum()
        ax1.bar(x_pos + i * width, probs, width, 
               label=f'τ={tau}', color=colors[i], edgecolor='black')
    
    ax1.set_xlabel('Scale (patch size)')
    ax1.set_ylabel('Probability')
    ax1.set_title('Temperature Effect on Softmax\n(logits = [1.5, 1.0, 0.5])', fontsize=11)
    ax1.set_xticks(x_pos + width * 2)
    ax1.set_xticklabels([f'{ps}×{ps}' for ps in patch_sizes])
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 1)
    
    # 2. 温度退火曲线
    ax2 = axes[0, 1]
    epochs = np.linspace(0, 1, 100)
    tau_init, tau_min, tau_max = 2.0, 0.5, 5.0
    
    # 不同退火策略
    tau_linear = tau_max - (tau_max - tau_min) * epochs
    tau_cosine = tau_min + 0.5 * (tau_max - tau_min) * (1 + np.cos(np.pi * epochs))
    tau_exp = tau_max * (tau_min / tau_max) ** epochs
    
    ax2.plot(epochs, tau_linear, 'b-', linewidth=2, label='Linear')
    ax2.plot(epochs, tau_cosine, 'r-', linewidth=2, label='Cosine')
    ax2.plot(epochs, tau_exp, 'g--', linewidth=2, label='Exponential')
    ax2.axhline(tau_min, color='gray', linestyle=':', label=f'τ_min={tau_min}')
    ax2.set_xlabel('Training Progress')
    ax2.set_ylabel('Temperature (τ)')
    ax2.set_title('Temperature Annealing Schedules', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 不同复杂度区域的决策
    ax3 = axes[0, 2]
    
    tau = 1.0
    x_pos = np.arange(num_scales)
    width = 0.25
    
    colors_regions = ['#FF6B6B', '#FFD93D', '#4ECDC4']
    for i, (region, logits) in enumerate(logits_examples.items()):
        probs = np.exp(logits / tau)
        probs = probs / probs.sum()
        ax3.bar(x_pos + i * width, probs, width, 
               label=region, color=colors_regions[i], edgecolor='black')
    
    ax3.set_xlabel('Scale (patch size)')
    ax3.set_ylabel('Selection Probability')
    ax3.set_title('Scale Selection by Region Complexity\n(τ=1.0)', fontsize=11)
    ax3.set_xticks(x_pos + width)
    ax3.set_xticklabels([f'{ps}×{ps}' for ps in patch_sizes])
    ax3.legend(fontsize=9)
    ax3.set_ylim(0, 1)
    
    # 4. Gumbel 噪声的影响
    ax4 = axes[1, 0]
    
    np.random.seed(42)
    n_samples = 1000
    logits = np.array([1.0, 0.5, 0.3])
    tau = 0.5
    
    selections = []
    for _ in range(n_samples):
        gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, num_scales) + 1e-10) + 1e-10)
        noisy_logits = logits + gumbel_noise
        selection = np.argmax(noisy_logits)
        selections.append(selection)
    
    counts = [selections.count(i) for i in range(num_scales)]
    ax4.bar(range(num_scales), counts, color=colors_regions, edgecolor='black')
    ax4.set_xlabel('Selected Scale')
    ax4.set_ylabel('Count')
    ax4.set_title(f'Gumbel-Softmax Sampling (n={n_samples})\n(logits={logits.tolist()}, τ={tau})', fontsize=11)
    ax4.set_xticks(range(num_scales))
    ax4.set_xticklabels([f'{ps}×{ps}' for ps in patch_sizes])
    
    # 理论概率
    probs = np.exp(logits / tau)
    probs = probs / probs.sum()
    for i, p in enumerate(probs):
        ax4.axhline(p * n_samples, color='red', linestyle='--', alpha=0.7)
    ax4.plot([], [], 'r--', label='Theoretical')
    ax4.legend()
    
    # 5. Train vs Eval 行为对比
    ax5 = axes[1, 1]
    
    logits = np.array([1.2, 0.8, 0.4])
    
    # Train (hard=True): 多次采样
    np.random.seed(123)
    train_outputs = []
    for _ in range(5):
        gumbel = -np.log(-np.log(np.random.uniform(0, 1, num_scales) + 1e-10) + 1e-10)
        train_out = np.zeros(num_scales)
        train_out[np.argmax(logits + gumbel)] = 1.0
        train_outputs.append(train_out)
    
    # Eval: argmax (确定性)
    eval_out = np.zeros(num_scales)
    eval_out[np.argmax(logits)] = 1.0
    
    x = np.arange(num_scales)
    width = 0.12
    
    # 绘制多次 train 采样
    for i, out in enumerate(train_outputs):
        ax5.bar(x + i * width, out, width, alpha=0.6, 
               color='blue', edgecolor='blue', linewidth=0.5)
    
    # Eval 结果
    ax5.bar(x + 5 * width, eval_out, width, 
           color='green', edgecolor='black', label='Eval (argmax)')
    
    ax5.set_xlabel('Scale')
    ax5.set_ylabel('Output (one-hot)')
    ax5.set_title('Train (Gumbel) vs Eval (argmax)\n(hard=True ensures consistency)', fontsize=11)
    ax5.set_xticks(x + 2.5 * width)
    ax5.set_xticklabels([f'{ps}×{ps}' for ps in patch_sizes])
    ax5.bar([], [], color='blue', alpha=0.6, label='Train samples')
    ax5.legend()
    
    # 6. STE (Straight-Through Estimator) 说明
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    ste_text = """
    Straight-Through Estimator (STE)
    ================================
    
    Forward Pass:
        output = one_hot(argmax(logits + gumbel))
        
    Backward Pass:
        ∂L/∂logits ≈ ∂L/∂softmax(logits/τ)
        
    Key Insight:
        - Forward: Hard decision (discrete)
        - Backward: Soft gradient (continuous)
        
    Benefits:
        ✓ Train/Eval consistency (both use argmax)
        ✓ Gradient flow preserved
        ✓ Discrete token selection
        
    Mathematical Form:
        y = one_hot(argmax(x)) + softmax(x) - sg(softmax(x))
        
    where sg() is stop_gradient
    """
    
    ax6.text(0.05, 0.95, ste_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    fig.suptitle('Gumbel-Softmax Adaptive Scale Selection Mechanism',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Gumbel-Softmax visualization saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# 可视化：深度偏置预热 (Depth Bias Warmup v2.2)
# ============================================================================

def visualize_depth_bias_warmup(
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """可视化深度偏置预热机制 (v2.2).
    
    数学原理:
        在训练初期对小尺度（深层级）添加正偏置，引导模型探索细粒度特征。
        
        bias(t) = max_bias × (1 - adjusted_progress)^decay
        
        其中:
            adjusted_progress = (progress - warmup) / (1 - warmup), if progress > warmup
                              = 0, otherwise
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 参数设置
    epochs = np.linspace(0, 1, 100)
    max_bias = 2.0
    warmup_ratio = 0.2
    
    # 1. 不同衰减指数的偏置曲线
    ax1 = axes[0, 0]
    decay_powers = [1.0, 2.0, 3.0, 5.0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(decay_powers)))
    
    for decay, color in zip(decay_powers, colors):
        bias_values = []
        for progress in epochs:
            if progress < warmup_ratio:
                bias = max_bias
            else:
                adj_progress = (progress - warmup_ratio) / (1 - warmup_ratio)
                bias = max_bias * ((1 - adj_progress) ** decay)
            bias_values.append(bias)
        ax1.plot(epochs, bias_values, color=color, linewidth=2, label=f'decay={decay}')
    
    ax1.axvline(warmup_ratio, color='red', linestyle='--', alpha=0.5, label=f'warmup={warmup_ratio}')
    ax1.axhline(0.01, color='gray', linestyle=':', alpha=0.5, label='threshold=0.01')
    ax1.set_xlabel('Training Progress')
    ax1.set_ylabel('Depth Bias Strength')
    ax1.set_title('Depth Bias Decay Curves\n(different decay powers)', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, max_bias * 1.1)
    
    # 2. 尺度偏置权重分布
    ax2 = axes[0, 1]
    num_scales = 4
    patch_sizes = [4, 8, 16, 32]
    scale_weights = np.linspace(1.0, 0.0, num_scales)  # 小尺度权重大
    
    bars = ax2.bar(range(num_scales), scale_weights, 
                  color=plt.cm.Reds(np.linspace(0.8, 0.3, num_scales)),
                  edgecolor='black')
    ax2.set_xlabel('Scale Index')
    ax2.set_ylabel('Bias Weight')
    ax2.set_title('Scale Bias Weights\n(smaller patch → larger bias)', fontsize=11)
    ax2.set_xticks(range(num_scales))
    ax2.set_xticklabels([f'{ps}×{ps}' for ps in patch_sizes])
    
    for bar, w in zip(bars, scale_weights):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{w:.2f}', ha='center', fontsize=10)
    
    # 3. 训练过程中的实际偏置 (per scale)
    ax3 = axes[1, 0]
    
    decay = 2.0
    for s_idx, (ps, w) in enumerate(zip(patch_sizes, scale_weights)):
        effective_bias = []
        for progress in epochs:
            if progress < warmup_ratio:
                bias = max_bias
            else:
                adj_progress = (progress - warmup_ratio) / (1 - warmup_ratio)
                bias = max_bias * ((1 - adj_progress) ** decay)
            effective_bias.append(bias * w)
        
        color = plt.cm.Reds(1 - s_idx / num_scales)
        ax3.plot(epochs, effective_bias, color=color, linewidth=2, 
                label=f'{ps}×{ps} (w={w:.2f})')
    
    ax3.set_xlabel('Training Progress')
    ax3.set_ylabel('Effective Bias = bias × weight')
    ax3.set_title('Effective Bias per Scale\n(decay=2.0)', fontsize=11)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    
    # 4. 偏置对 logits 的影响
    ax4 = axes[1, 1]
    
    # 原始 logits (无偏好)
    original_logits = np.array([0.0, 0.0, 0.0, 0.0])
    
    # 不同训练阶段的偏置
    training_stages = [0.0, 0.1, 0.3, 0.5, 1.0]  # progress
    colors_stages = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(training_stages)))
    
    x_pos = np.arange(num_scales)
    width = 0.15
    
    for i, progress in enumerate(training_stages):
        if progress < warmup_ratio:
            bias = max_bias
        else:
            adj_progress = (progress - warmup_ratio) / (1 - warmup_ratio)
            bias = max_bias * ((1 - adj_progress) ** decay)
        
        biased_logits = original_logits + bias * scale_weights
        probs = np.exp(biased_logits)
        probs = probs / probs.sum()
        
        ax4.bar(x_pos + i * width, probs, width, 
               label=f'p={progress:.1f} (bias={bias:.2f})',
               color=colors_stages[i], edgecolor='black')
    
    ax4.set_xlabel('Scale (patch size)')
    ax4.set_ylabel('Selection Probability')
    ax4.set_title('Scale Selection Probability\n(how bias shifts preference)', fontsize=11)
    ax4.set_xticks(x_pos + width * 2)
    ax4.set_xticklabels([f'{ps}×{ps}' for ps in patch_sizes])
    ax4.legend(fontsize=8, loc='upper right')
    ax4.set_ylim(0, 0.8)
    
    fig.suptitle('Depth Bias Warmup (v2.2)\n'
                'Encourages exploration of fine-grained features early in training',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Depth bias warmup saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# 可视化：注意力偏置矩阵对比
# ============================================================================

def visualize_attention_bias_comparison(
    order: int = 3,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """对比不同 Hilbert Bias 模式的注意力矩阵.
    
    三种模式:
    1. lca: LCA 嵌入表 (推荐)
    2. low_rank: 低秩分解 B = ΦΨ^T
    3. hierarchical: 分层累加
    """
    n = 2 ** order
    num_tokens = n * n
    max_depth = order
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 计算四叉树路径
    paths = []
    for d in range(num_tokens):
        path = compute_quadtree_path(n, d, max_depth)
        paths.append(path)
    
    # 1. LCA 偏置矩阵 (真实)
    ax1 = axes[0, 0]
    lca_matrix = np.zeros((num_tokens, num_tokens))
    for i in range(num_tokens):
        for j in range(num_tokens):
            lca_matrix[i, j] = compute_lca_depth(paths[i], paths[j])
    
    im1 = ax1.imshow(lca_matrix, cmap='viridis')
    ax1.set_title('LCA Bias (Recommended)\nB[i,j] = LCAEmbed(LCA(i,j))', fontsize=10)
    ax1.set_xlabel('Token j')
    ax1.set_ylabel('Token i')
    plt.colorbar(im1, ax=ax1, label='LCA Depth')
    
    # 2. Hilbert 距离偏置
    ax2 = axes[0, 1]
    hilbert_dist_matrix = np.zeros((num_tokens, num_tokens))
    for i in range(num_tokens):
        for j in range(num_tokens):
            hilbert_dist_matrix[i, j] = abs(i - j)
    
    # 归一化到合理范围
    hilbert_dist_matrix = np.log1p(hilbert_dist_matrix)
    
    im2 = ax2.imshow(hilbert_dist_matrix, cmap='plasma')
    ax2.set_title('Hilbert Distance\nlog(1 + |i - j|)', fontsize=10)
    ax2.set_xlabel('Token j')
    ax2.set_ylabel('Token i')
    plt.colorbar(im2, ax=ax2, label='log(1 + dist)')
    
    # 3. 空间距离偏置
    ax3 = axes[0, 2]
    spatial_dist_matrix = np.zeros((num_tokens, num_tokens))
    for i in range(num_tokens):
        xi, yi = HilbertCurve.d_to_xy(n, i)
        for j in range(num_tokens):
            xj, yj = HilbertCurve.d_to_xy(n, j)
            spatial_dist_matrix[i, j] = np.sqrt((xi - xj)**2 + (yi - yj)**2)
    
    im3 = ax3.imshow(spatial_dist_matrix, cmap='inferno')
    ax3.set_title('Spatial (Euclidean) Distance\n||p_i - p_j||_2', fontsize=10)
    ax3.set_xlabel('Token j')
    ax3.set_ylabel('Token i')
    plt.colorbar(im3, ax=ax3, label='Distance')
    
    # 4. LCA vs Hilbert 距离相关性
    ax4 = axes[1, 0]
    
    # Flatten and sample
    lca_flat = lca_matrix.flatten()
    hilbert_flat = hilbert_dist_matrix.flatten()
    
    ax4.scatter(lca_flat, hilbert_flat, alpha=0.3, s=5)
    ax4.set_xlabel('LCA Depth')
    ax4.set_ylabel('log(1 + Hilbert Distance)')
    ax4.set_title('LCA vs Hilbert Distance', fontsize=10)
    
    # 5. LCA vs 空间距离相关性
    ax5 = axes[1, 1]
    spatial_flat = spatial_dist_matrix.flatten()
    
    ax5.scatter(lca_flat, spatial_flat, alpha=0.3, s=5, color='orange')
    ax5.set_xlabel('LCA Depth')
    ax5.set_ylabel('Spatial Distance')
    ax5.set_title('LCA vs Spatial Distance', fontsize=10)
    
    # 6. 参数量对比
    ax6 = axes[1, 2]
    
    # 估算不同方法的参数量
    heads = 8
    rank = 32
    path_dim = max_depth
    
    params = {
        'lca': (max_depth + 1) * heads,  # LCA 嵌入表
        'low_rank': 2 * path_dim * 64 + 64 * rank * heads,  # 两个 MLP
        'hierarchical': max_depth * (4 * 16 + 16 * heads),  # 每层一个小 MLP
    }
    
    methods = list(params.keys())
    values = [params[m] for m in methods]
    colors = ['#96CEB4', '#4ECDC4', '#45B7D1']
    
    bars = ax6.bar(methods, values, color=colors, edgecolor='black')
    ax6.set_ylabel('Parameter Count')
    ax6.set_title('Parameter Efficiency Comparison', fontsize=10)
    ax6.set_yscale('log')
    
    for bar, v in zip(bars, values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{v:,}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    fig.suptitle(f'Attention Bias Modes Comparison (Order {order}: {n}×{n} tokens)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Attention bias comparison saved to: {save_path}")
    
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

def _simulate_gumbel_softmax(
    logits: np.ndarray, 
    tau: float, 
    hard: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """模拟 Gumbel-Softmax 采样过程.
    
    数学形式:
        g_i ~ Gumbel(0, 1)  # Gumbel 噪声
        y_i = softmax((logits_i + g_i) / τ)  # 软采样
        
        if hard:
            z = one_hot(argmax(y))  # 硬决策
            return z - y.detach() + y  # STE 梯度
    
    Args:
        logits: [H, W, num_scales] 尺度 logits
        tau: Gumbel-Softmax 温度
        hard: 是否使用硬采样 (STE)
        seed: 随机种子
        
    Returns:
        weights: [H, W, num_scales] 尺度权重 (soft) 或 one-hot (hard)
    """
    np.random.seed(seed)
    
    # 添加 Gumbel 噪声: g = -log(-log(u)), u ~ Uniform(0, 1)
    u = np.random.uniform(0.001, 0.999, logits.shape)
    gumbel_noise = -np.log(-np.log(u))
    
    # 带噪声的 softmax
    noisy_logits = (logits + gumbel_noise) / tau
    # 数值稳定的 softmax
    exp_logits = np.exp(noisy_logits - np.max(noisy_logits, axis=-1, keepdims=True))
    soft_weights = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    if hard:
        # Straight-Through Estimator: 前向 argmax，反向软梯度
        hard_indices = np.argmax(soft_weights, axis=-1)
        hard_weights = np.eye(logits.shape[-1])[hard_indices]
        return hard_weights
    else:
        return soft_weights


def _compute_semantic_logits(
    img: np.ndarray,
    grid_h: int,
    grid_w: int,
    min_ps: int,
    num_scales: int,
    depth_bias: float = 0.0,
) -> np.ndarray:
    """模拟语义级复杂度估计 (ComplexityHead).
    
    数学形式:
        F_concat = Concat_{s}[Upsample(F_s, target_size)]  # 多尺度特征拼接
        logits = ComplexityHead(F_concat)                   # 轻量级预测头
        logits' = logits + depth_bias * scale_bias_weights  # 深度偏置
    
    这里用图像梯度作为"语义特征"的代理:
        - 高梯度区域 → 更可能选择小 patch (深层级)
        - 低梯度区域 → 更可能选择大 patch (浅层级)
    
    Args:
        img: [H, W, 3] 输入图像
        grid_h, grid_w: 决策网格大小
        min_ps: 最小 patch size
        num_scales: 尺度数量
        depth_bias: 深度探索偏置强度
        
    Returns:
        logits: [grid_h, grid_w, num_scales] 尺度 logits
    """
    # 1. 计算图像梯度作为"语义复杂度"代理
    gray = np.mean(img, axis=2)
    
    # Sobel 梯度
    gy = np.zeros_like(gray)
    gx = np.zeros_like(gray)
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    
    # 2. 下采样到网格级别
    complexity_grid = np.zeros((grid_h, grid_w))
    for i in range(grid_h):
        for j in range(grid_w):
            region = gradient_magnitude[i*min_ps:(i+1)*min_ps, j*min_ps:(j+1)*min_ps]
            complexity_grid[i, j] = np.mean(region)
    
    # 归一化到 [0, 1]
    if complexity_grid.max() > complexity_grid.min():
        complexity_grid = (complexity_grid - complexity_grid.min()) / (complexity_grid.max() - complexity_grid.min())
    
    # 3. 转换为尺度 logits
    # 高复杂度 → 偏好小 patch (index 0)
    # 低复杂度 → 偏好大 patch (index num_scales-1)
    logits = np.zeros((grid_h, grid_w, num_scales))
    
    for s in range(num_scales):
        # 尺度权重: 小 patch (s=0) 在高复杂度区域有更高 logit
        # 使用线性插值: scale_preference[s] = 1 - s / (num_scales - 1)
        scale_preference = 1.0 - s / (num_scales - 1) if num_scales > 1 else 0.5
        # logit = complexity * scale_preference - (1 - complexity) * (1 - scale_preference)
        logits[:, :, s] = complexity_grid * scale_preference * 3.0 - (1 - complexity_grid) * (1 - scale_preference) * 3.0
    
    # 4. 添加深度偏置 (v2.2 特性)
    # scale_bias_weights[s] = 1 - s / (num_scales - 1)，小尺度偏置大
    if depth_bias > 0.01:
        scale_bias_weights = np.linspace(1.0, 0.0, num_scales)
        logits = logits + depth_bias * scale_bias_weights
    
    return logits


def _enforce_quadtree_consistency(
    scale_map: np.ndarray,
    patch_sizes: List[int],
    min_ps: int,
) -> np.ndarray:
    """强制四叉树一致性约束（与 StreamingFractalTokenizerV2 对齐）.
    
    数学约束:
        若 scale_map[i,j] = k (选择尺度 k)
        则 Block(i,j,k) 内所有位置必须为 k
    
    这确保了**没有重叠**：每个像素区域只被一个 token 覆盖。
    
    Args:
        scale_map: [grid_h, grid_w] 每个位置的尺度索引
        patch_sizes: 多尺度 patch 大小列表
        min_ps: 最小 patch size
        
    Returns:
        一致性约束后的 scale_map（无重叠的四叉树分割）
    """
    result = scale_map.copy()
    grid_h, grid_w = scale_map.shape
    num_scales = len(patch_sizes)
    
    # 从粗尺度到细尺度遍历（跳过最细尺度）
    for scale_idx in range(num_scales - 1, 0, -1):
        ps = patch_sizes[scale_idx]
        block_size = ps // min_ps
        
        if block_size <= 1:
            continue
        
        # 遍历每个 block
        for by in range(0, grid_h, block_size):
            for bx in range(0, grid_w, block_size):
                by_end = min(by + block_size, grid_h)
                bx_end = min(bx + block_size, grid_w)
                
                block = result[by:by_end, bx:bx_end]
                
                # 如果 block 内有任何位置选择了当前粗尺度或更粗，整个 block 统一
                block_max = block.max()
                if block_max >= scale_idx:
                    result[by:by_end, bx:bx_end] = block_max
    
    return result


def visualize_mixed_level_segmentation(
    base_size: int = 128,
    patch_sizes: List[int] = [4, 8, 16],
    gumbel_tau: float = 2.0,
    depth_bias: float = 0.0,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """可视化混合尺度分割的概念演示（贴合项目数学形式化）.
    
    模拟 StreamingFractalTokenizerV2 的尺度选择机制：
    
    **核心数学形式**:
    
    1. **语义级复杂度估计** (v2.0):
       .. math::
           \\text{logits} = \\text{ComplexityHead}(\\text{Concat}_s[\\text{Upsample}(F_s)])
       
       这里用图像梯度作为语义特征的代理。
    
    2. **深度偏置 Warmup** (v2.2):
       .. math::
           \\text{logits}' = \\text{logits} + \\text{depth\\_bias} \\cdot \\text{scale\\_bias\\_weights}
       
       其中 scale_bias_weights = [1.0, 0.67, 0.33, 0.0] for 4 scales
    
    3. **Gumbel-Softmax 选择** (hard=True, STE):
       .. math::
           g_i \\sim \\text{Gumbel}(0, 1)
           \\\\
           \\pi_i = \\text{softmax}((\\text{logits}'_i + g_i) / \\tau)
           \\\\
           z = \\text{one\\_hot}(\\arg\\max \\pi)  \\quad \\text{(forward)}
    
    Args:
        base_size: 模拟图像大小
        patch_sizes: 多尺度 patch 大小 (从小到大排列)
        gumbel_tau: Gumbel-Softmax 温度 τ (高温→均匀分布，低温→确定性)
        depth_bias: 深度探索偏置强度 (>0 偏好小 patch)
        save_path: 保存路径
        show: 是否显示
        
    Returns:
        matplotlib Figure 对象
    
    Note:
        温度退火调度: τ 从 τ_max (5.0) 线性退火到 τ_min (0.5)
        深度偏置调度: bias 从 max_bias (2.0) 快速衰减到 0
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    num_scales = len(patch_sizes)
    scale_colors = ['#FF6B6B', '#45B7D1', '#96CEB4'][:num_scales]  # 红、蓝、绿
    scale_labels = [f'{ps}×{ps}' for ps in patch_sizes]
    
    # ===== 创建模拟图像（带复杂度变化）=====
    np.random.seed(42)
    
    # 创建合成图像：中心和左上角有高频细节
    img = np.zeros((base_size, base_size, 3))
    center = base_size // 2
    
    for i in range(base_size):
        for j in range(base_size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            
            if dist < base_size // 4:
                # 中心区域: 高频棋盘格
                img[i, j] = [(i + j) % 2 * 0.8 + 0.1] * 3
            elif dist < base_size // 2:
                # 过渡区域: 中频条纹
                img[i, j] = [(i % 4 < 2) * 0.5 + 0.3] * 3
            else:
                # 边缘区域: 低频平滑
                img[i, j] = [0.7, 0.8, 0.9]
            
            # 左上角: 高频细节
            if i < base_size // 4 and j < base_size // 4:
                img[i, j] = [(i + j) % 2 * 0.7 + 0.2] * 3
    
    # ===== 1. 原始图像 =====
    ax1 = axes[0, 0]
    ax1.imshow(img)
    ax1.set_title('Synthetic Image\n(varying complexity regions)', fontsize=11)
    ax1.axis('off')
    
    # ===== 计算语义级 logits 和 Gumbel-Softmax 选择 =====
    min_ps = min(patch_sizes)
    grid_h, grid_w = base_size // min_ps, base_size // min_ps
    
    # 计算语义级 logits (模拟 ComplexityHead)
    logits = _compute_semantic_logits(img, grid_h, grid_w, min_ps, num_scales, depth_bias)
    
    # Gumbel-Softmax 软权重 (用于可视化概率分布)
    soft_weights = _simulate_gumbel_softmax(logits, gumbel_tau, hard=False, seed=42)
    
    # Gumbel-Softmax 硬决策 (用于最终尺度选择)
    hard_weights = _simulate_gumbel_softmax(logits, gumbel_tau, hard=True, seed=42)
    scale_map_raw = np.argmax(hard_weights, axis=-1)  # [grid_h, grid_w]
    
    # ===== 应用四叉树一致性约束（与实际实现对齐）=====
    # 这确保了没有重叠：每个像素区域只被一个 token 覆盖
    scale_map = _enforce_quadtree_consistency(scale_map_raw, patch_sizes, min_ps)
    
    # ===== 2. Gumbel-Softmax 概率分布 =====
    ax2 = axes[0, 1]
    # 显示最细尺度 (index 0) 的选择概率
    prob_fine = soft_weights[:, :, 0]
    im = ax2.imshow(prob_fine, cmap='hot', vmin=0, vmax=1,
                   extent=[0, base_size, base_size, 0])
    ax2.set_title(f'Gumbel-Softmax P(fine scale)\n'
                 f'τ={gumbel_tau:.1f}, depth_bias={depth_bias:.1f}', fontsize=11)
    ax2.axis('off')
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046)
    cbar.set_label(f'P(patch={patch_sizes[0]}×{patch_sizes[0]})', fontsize=9)
    
    # ===== 3. 硬决策尺度选择 =====
    ax3 = axes[0, 2]
    scale_img = np.zeros((grid_h, grid_w, 3))
    for s in range(num_scales):
        mask = scale_map == s
        color = np.array(plt.cm.colors.hex2color(scale_colors[s]))
        scale_img[mask] = color
    
    ax3.imshow(scale_img, interpolation='nearest',
               extent=[0, base_size, base_size, 0])
    ax3.set_title('Scale Selection (hard=True, STE)\nz = one_hot(argmax π)', fontsize=11)
    ax3.axis('off')
    
    # 图例
    legend_patches = [mpatches.Patch(color=scale_colors[i],
                                    label=f'{scale_labels[i]} (scale {i})')
                     for i in range(num_scales)]
    ax3.legend(handles=legend_patches, loc='upper right', fontsize=8)
    
    # ===== 4. 混合分割叠加 =====
    ax4 = axes[1, 0]
    ax4.imshow(img, alpha=0.6)
    
    # 绘制混合 patch 边界
    for i in range(grid_h):
        for j in range(grid_w):
            scale = scale_map[i, j]
            ps = patch_sizes[scale]
            x0, y0 = j * min_ps, i * min_ps
            
            # 根据尺度决定是否绘制 (只在对齐边界绘制)
            if scale == 0:  # 最细尺度: 每个格子都画
                rect = Rectangle((x0, y0), min_ps, min_ps,
                                fill=False, edgecolor=scale_colors[0],
                                linewidth=1.5, alpha=0.8)
                ax4.add_patch(rect)
            elif scale == 1 and num_scales > 1:  # 中等尺度
                ratio = patch_sizes[1] // min_ps
                if i % ratio == 0 and j % ratio == 0:
                    rect = Rectangle((x0, y0), min_ps * ratio, min_ps * ratio,
                                    fill=False, edgecolor=scale_colors[1],
                                    linewidth=2, alpha=0.8)
                    ax4.add_patch(rect)
            elif scale == 2 and num_scales > 2:  # 最粗尺度
                ratio = patch_sizes[2] // min_ps
                if i % ratio == 0 and j % ratio == 0:
                    rect = Rectangle((x0, y0), min_ps * ratio, min_ps * ratio,
                                    fill=False, edgecolor=scale_colors[2],
                                    linewidth=2.5, alpha=0.8)
                    ax4.add_patch(rect)
    
    ax4.set_xlim(0, base_size)
    ax4.set_ylim(base_size, 0)
    ax4.set_title('Mixed-Level Segmentation\n(adaptive patch boundaries)', fontsize=11)
    ax4.axis('off')
    
    # ===== 5. Hilbert 遍历路径 (Variable Token 模式, 默认) =====
    # 关键：在 variable_tokens=True 模式下，一个 patch = 一个 token
    # Token 数量可变：N ∈ [N_min, N_max]
    ax5 = axes[1, 1]
    ax5.imshow(img, alpha=0.3)
    
    # 收集所有唯一的 patch（一个 patch = 一个 token）
    # 需要强制四叉树一致性：粗尺度区域内所有位置使用相同尺度
    unique_patches = []  # (patch_row, patch_col, scale, ps)
    visited = set()
    
    for i in range(grid_h):
        for j in range(grid_w):
            scale = scale_map[i, j]
            ps = patch_sizes[scale]
            # 计算该位置所属的 patch 左上角
            patch_row = (i * min_ps // ps) * ps
            patch_col = (j * min_ps // ps) * ps
            patch_key = (patch_row, patch_col, scale)
            
            if patch_key not in visited:
                visited.add(patch_key)
                # 计算 patch 中心用于 Hilbert 排序
                cx = patch_col + ps // 2
                cy = patch_row + ps // 2
                unique_patches.append((patch_row, patch_col, scale, ps, cx, cy))
    
    # 按 Hilbert 顺序排序这些 patch
    # 使用 patch 中心点的 Hilbert 距离作为排序键
    n = 1
    while n * min_ps < base_size:
        n *= 2
    
    def get_hilbert_distance(cx, cy):
        """计算像素坐标对应的 Hilbert 距离"""
        gx, gy = cx // min_ps, cy // min_ps
        gx = min(gx, n - 1)
        gy = min(gy, n - 1)
        return HilbertCurve.xy_to_d(n, gx, gy)
    
    sorted_patches = sorted(unique_patches, key=lambda p: get_hilbert_distance(p[4], p[5]))
    
    # 绘制 Hilbert 连线（连接 patch 中心）
    if len(sorted_patches) > 1:
        for i in range(len(sorted_patches) - 1):
            cx1, cy1 = sorted_patches[i][4], sorted_patches[i][5]
            cx2, cy2 = sorted_patches[i+1][4], sorted_patches[i+1][5]
            ax5.plot([cx1, cx2], [cy1, cy2],
                    color='#2C3E50', linewidth=1.5, alpha=0.9, zorder=1)
    
    # 绘制每个 token（一个 patch = 一个 token）
    for token_idx, (patch_row, patch_col, scale, ps, cx, cy) in enumerate(sorted_patches):
        # 绘制 patch 区域
        rect = Rectangle((patch_col, patch_row), ps, ps,
                        facecolor=scale_colors[scale], alpha=0.5,
                        edgecolor=scale_colors[scale], linewidth=2,
                        zorder=2)
        ax5.add_patch(rect)
        
        # 在 patch 中心标注 token 序号
        ax5.text(cx, cy, str(token_idx + 1),
                ha='center', va='center',
                fontsize=8 if ps >= 12 else 6, fontweight='bold',
                color='white', zorder=3,
                bbox=dict(boxstyle='circle,pad=0.15', facecolor=scale_colors[scale], 
                         edgecolor='white', linewidth=0.5, alpha=0.8))
    
    ax5.set_xlim(0, base_size)
    ax5.set_ylim(base_size, 0)
    ax5.set_title(f'Variable Token Mode (default): {len(sorted_patches)} tokens\n'
                 f'(1 patch = 1 token, Hilbert ordered)',
                 fontsize=10)
    ax5.legend(handles=[mpatches.Patch(color=c, alpha=0.5, label=f'{ps}×{ps}')
                       for c, ps in zip(scale_colors, patch_sizes)],
              loc='upper right', fontsize=8)
    ax5.axis('off')
    
    # ===== 6. Token 数量与温度效应对比 =====
    ax6 = axes[1, 2]
    
    # 计算不同温度下的尺度分布
    temps = [5.0, 2.0, 0.5]
    temp_distributions = []
    for tau in temps:
        hw = _simulate_gumbel_softmax(logits, tau, hard=True, seed=42)
        sm = np.argmax(hw, axis=-1)
        dist = [np.sum(sm == s) / sm.size for s in range(num_scales)]
        temp_distributions.append(dist)
    
    x = np.arange(num_scales)
    width = 0.25
    
    for i, (tau, dist) in enumerate(zip(temps, temp_distributions)):
        offset = (i - 1) * width
        bars = ax6.bar(x + offset, dist, width, label=f'τ={tau}',
                      color=plt.cm.Blues(0.3 + i * 0.25), edgecolor='black')
    
    ax6.set_xlabel('Scale Index', fontsize=10)
    ax6.set_ylabel('Selection Ratio', fontsize=10)
    ax6.set_title('Temperature Annealing Effect\n(τ: 5.0 → 0.5)', fontsize=11)
    ax6.set_xticks(x)
    ax6.set_xticklabels([f'{ps}×{ps}' for ps in patch_sizes])
    ax6.legend(title='Gumbel τ', fontsize=8)
    ax6.set_ylim(0, 1)
    
    # 添加数学公式注释
    ax6.text(0.02, 0.98, r'$\pi = \mathrm{softmax}(\frac{\mathrm{logits} + g}{\tau})$',
            transform=ax6.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(f'StreamingFractalTokenizerV2: Gumbel-Softmax Scale Selection\n'
                f'Image: {base_size}×{base_size}, Patches: {patch_sizes}, '
                f'τ={gumbel_tau:.1f}, depth_bias={depth_bias:.1f}',
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
    include_new: bool = True,
    include_v2_deprecated: bool = False,
) -> None:
    """生成所有可视化.
    
    Args:
        output_dir: 输出目录
        show: 是否交互显示
        include_new: 是否包含新增的高级可视化 (LCA, Cross-Scale, etc.)
        include_v2_deprecated: 是否包含 V2 已弃用的可视化 (Gumbel, Depth Bias)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_steps = 7  # 基础可视化
    if include_new:
        total_steps += 3  # LCA, Cross-Scale Attention, Bias Comparison
    if include_v2_deprecated:
        total_steps += 2  # Gumbel, Depth Bias
    
    print(f"\n{'='*60}")
    print("Generating Fractal Curve Visualizations")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"Include advanced visualizations: {include_new}")
    print(f"Include V2 deprecated (Gumbel, DepthBias): {include_v2_deprecated}")
    print(f"{'='*60}\n")
    
    step = 1
    
    # 1. 阶数对比
    print(f"[{step}/{total_steps}] Generating order comparison...")
    visualize_order_comparison(
        orders=[1, 2, 3, 4, 5],
        save_path=output_dir / "hilbert_order_comparison.png",
        show=show,
    )
    step += 1
    
    # 2. 局部性保持
    print(f"[{step}/{total_steps}] Generating locality preservation...")
    visualize_locality_preservation(
        order=4,
        save_path=output_dir / "hilbert_locality.png",
        show=show,
    )
    step += 1
    
    # 3. 四叉树结构
    print(f"[{step}/{total_steps}] Generating quadtree structure...")
    visualize_quadtree_structure(
        max_depth=4,
        save_path=output_dir / "quadtree_structure.png",
        show=show,
    )
    step += 1
    
    # 4. 2D → 1D 映射
    print(f"[{step}/{total_steps}] Generating 2D to 1D mapping...")
    visualize_2d_to_1d_mapping(
        order=3,
        save_path=output_dir / "2d_to_1d_mapping.png",
        show=show,
    )
    step += 1
    
    # 5. 多尺度层级
    print(f"[{step}/{total_steps}] Generating multi-scale hierarchy...")
    visualize_multiscale_hierarchy(
        base_size=64,
        patch_sizes=[4, 8, 16],
        save_path=output_dir / "multiscale_hierarchy.png",
        show=show,
    )
    step += 1
    
    # 6. 混合尺度分割演示
    print(f"[{step}/{total_steps}] Generating mixed-level segmentation demo...")
    visualize_mixed_level_segmentation(
        base_size=64,
        patch_sizes=[4, 8, 16],
        save_path=output_dir / "mixed_level_segmentation.png",
        show=show,
    )
    step += 1
    
    # 7. 动画 (可选)
    print(f"[{step}/{total_steps}] Generating growth animation...")
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
    step += 1
    
    # ========== 新增的高级可视化 (V3 推荐) ==========
    if include_new:
        # 8. LCA 偏置矩阵
        print(f"[{step}/{total_steps}] Generating LCA bias matrix...")
        visualize_lca_bias_matrix(
            order=4,
            save_path=output_dir / "lca_bias_matrix.png",
            show=show,
        )
        step += 1
        
        # 9. Cross-Scale Attention 机制 (V3 推荐)
        print(f"[{step}/{total_steps}] Generating Cross-Scale Attention visualization...")
        visualize_cross_scale_attention(
            save_path=output_dir / "cross_scale_attention.png",
            show=show,
        )
        step += 1
        
        # 10. 注意力偏置对比
        print(f"[{step}/{total_steps}] Generating attention bias comparison...")
        visualize_attention_bias_comparison(
            order=3,
            save_path=output_dir / "attention_bias_comparison.png",
            show=show,
        )
        step += 1
    
    # ========== V2 已弃用的可视化 ==========
    if include_v2_deprecated:
        # Gumbel-Softmax 机制 (V2, 已弃用)
        print(f"[{step}/{total_steps}] [V2 deprecated] Generating Gumbel-Softmax decision...")
        visualize_gumbel_softmax_decision(
            save_path=output_dir / "gumbel_softmax_decision.png",
            show=show,
        )
        step += 1
        
        # 深度偏置预热 (V2, 已弃用)
        print(f"[{step}/{total_steps}] [V2 deprecated] Generating depth bias warmup...")
        visualize_depth_bias_warmup(
            save_path=output_dir / "depth_bias_warmup.png",
            show=show,
        )
        step += 1
    
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
    if include_new:
        print(f"  - lca_bias_matrix.png")
        print(f"  - cross_scale_attention.png (V3)")
        print(f"  - attention_bias_comparison.png")
    if include_v2_deprecated:
        print(f"  - gumbel_softmax_decision.png (V2 deprecated)")
        print(f"  - depth_bias_warmup.png (V2 deprecated)")
    print(f"{'='*60}\n")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fractal Curve Visualization for ViT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all visualizations (V3 + basic)
  python visualize_fractal_curves.py --all

  # Generate all including V2 deprecated visualizations
  python visualize_fractal_curves.py --all --include-v2

  # Generate only basic visualizations (without LCA, Cross-Scale, etc.)
  python visualize_fractal_curves.py --all --no-advanced

  # Generate Cross-Scale Attention visualization (V3 recommended)
  python visualize_fractal_curves.py --cross-scale-attention --show

  # Generate V2 deprecated visualizations
  python visualize_fractal_curves.py --gumbel --depth-bias
        """
    )
    
    # 模式 - 基础
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
    
    # 模式 - 新增高级可视化
    parser.add_argument("--lca-bias", action="store_true",
                       help="Generate LCA bias matrix visualization")
    parser.add_argument("--cross-scale-attention", action="store_true",
                       help="[V3] Generate Cross-Scale Attention visualization (recommended)")
    parser.add_argument("--gumbel", action="store_true",
                       help="[V2] Generate Gumbel-Softmax decision visualization (deprecated)")
    parser.add_argument("--depth-bias", action="store_true",
                       help="[V2] Generate depth bias warmup visualization (deprecated)")
    parser.add_argument("--bias-comparison", action="store_true",
                       help="Generate attention bias modes comparison")
    
    # 参数
    parser.add_argument("--order", type=int, default=4,
                       help="Hilbert curve order (default: 4)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--show", action="store_true",
                       help="Show plots interactively")
    parser.add_argument("--no-advanced", action="store_true",
                       help="Exclude advanced visualizations when using --all")
    parser.add_argument("--include-v2", action="store_true",
                       help="Include V2 deprecated visualizations (Gumbel, Depth Bias) when using --all")
    
    args = parser.parse_args()
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "workspace" / "visualizations" / "fractal_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 如果没有指定任何选项，默认生成所有
    all_options = [
        args.all, args.animate, args.comparison, args.locality, 
        args.quadtree, args.mapping, args.multiscale, args.mixed_level,
        args.lca_bias, args.cross_scale_attention, args.gumbel, 
        args.depth_bias, args.bias_comparison
    ]
    if not any(all_options):
        args.all = True
    
    if args.all:
        generate_all_visualizations(
            output_dir, 
            show=args.show,
            include_new=not args.no_advanced,
            include_v2_deprecated=args.include_v2,
        )
        return
    
    # 基础可视化
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
    
    # 新增高级可视化
    if args.lca_bias:
        visualize_lca_bias_matrix(
            order=args.order,
            save_path=output_dir / f"lca_bias_matrix_order{args.order}.png",
            show=args.show,
        )
    
    if args.cross_scale_attention:
        visualize_cross_scale_attention(
            save_path=output_dir / "cross_scale_attention.png",
            show=args.show,
        )
    
    if args.gumbel:
        visualize_gumbel_softmax_decision(
            save_path=output_dir / "gumbel_softmax_decision.png",
            show=args.show,
        )
    
    if args.depth_bias:
        visualize_depth_bias_warmup(
            save_path=output_dir / "depth_bias_warmup.png",
            show=args.show,
        )
    
    if args.bias_comparison:
        visualize_attention_bias_comparison(
            order=min(args.order, 4),
            save_path=output_dir / f"attention_bias_comparison_order{min(args.order, 4)}.png",
            show=args.show,
        )


if __name__ == "__main__":
    main()
