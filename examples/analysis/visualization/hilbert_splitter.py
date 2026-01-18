# -*- coding: utf-8 -*-
"""
Hilbert Curve Splitter 分割与扫描策略可视化演示

数学背景
========
1. 四叉树递归分割:
   - 每个区域 R 可以分割为 4 个子区域 {R_00, R_01, R_10, R_11}
   - 深度 d 的区域大小: size = image_size / 2^d

2. Hilbert 曲线排序:
   - 每个区域有一个 Hilbert 索引 h ∈ [0, N)
   - 排序: tokens = sort(regions, by=hilbert_indices)

3. 空间局部性保持:
   - 空间上相邻的区域 → Hilbert 索引相近
   - 验证: |h1 - h2| small ⇒ ||p1 - p2|| small

增强功能 (Shapely)
=================
- 使用 Shapely 进行精确几何操作 (box, union, intersection)
- 四叉树区域可视化支持透明叠加
- 精确的边界碰撞检测
"""

import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon as MplPolygon

# Shapely 用于精确几何操作
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import unary_union

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# Fallback Hilbert 曲线实现（如果原生实现不可用）
# =============================================================================

def _xy_to_d_fallback(x: int, y: int, n: int) -> int:
    """将 2D 坐标转换为 Hilbert 曲线距离"""
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        s //= 2
    return d


def _d_to_xy_fallback(d: int, n: int) -> Tuple[int, int]:
    """将 Hilbert 曲线距离转换为 2D 坐标"""
    def rot(n, x, y, rx, ry):
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        return x, y

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


def generate_hilbert_curve(order: int) -> List[Tuple[int, int]]:
    """生成指定阶数的 Hilbert 曲线坐标序列"""
    n = 2 ** order
    coords = []
    for d in range(n * n):
        x, y = _d_to_xy_fallback(d, n)
        coords.append((x, y))
    return coords


# =============================================================================
# Shapely 几何工具函数 (增强功能)
# =============================================================================

def create_quadtree_regions_shapely(
    image_size: int,
    min_patch_size: int
) -> Dict[int, List[Polygon]]:
    """
    使用 Shapely 创建四叉树区域

    数学定义:
    =========
    对于深度 d，区域数量为 4^d，区域大小为 image_size / 2^d

    Shapely 优势:
    - 精确的边界表示 (浮点数坐标)
    - 支持几何操作 (union, intersection, buffer)
    - 高效的空间查询

    Args:
        image_size: 图像尺寸
        min_patch_size: 最小 patch 尺寸

    Returns:
        Dict: {depth: [Polygon, ...]}
    """
    max_depth = int(np.log2(image_size / min_patch_size))
    regions_by_depth = {}

    for depth in range(max_depth + 1):
        n_side = 2 ** depth
        region_size = image_size / n_side
        regions = []

        for i in range(n_side):
            for j in range(n_side):
                x1, y1 = j * region_size, i * region_size
                x2, y2 = x1 + region_size, y1 + region_size
                rect = box(x1, y1, x2, y2)
                regions.append(rect)

        regions_by_depth[depth] = regions

    return regions_by_depth


def check_overlap_shapely(polygons: List[Polygon]) -> bool:
    """
    使用 Shapely 检查多边形列表中是否有重叠

    数学原理:
    =========
    对于多边形 A, B:
    - overlap = A.intersection(B)
    - 若 overlap.area > 0，则存在重叠

    优化: 使用空间索引加速查询

    Args:
        polygons: 多边形列表

    Returns:
        bool: 是否有重叠
    """
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            if polygons[i].intersects(polygons[j]):
                intersection = polygons[i].intersection(polygons[j])
                if intersection.area > 1e-10:
                    return True
    return False


def merge_regions_shapely(polygons: List[Polygon]) -> Polygon:
    """
    使用 Shapely 合并多个多边形

    数学原理:
    =========
    union = A ∪ B ∪ C ∪ ...
    使用 shapely.ops.unary_union 进行高效合并

    Args:
        polygons: 多边形列表

    Returns:
        Polygon: 合并后的多边形
    """
    if len(polygons) == 0:
        return None
    if len(polygons) == 1:
        return polygons[0]
    return unary_union(polygons)


def get_region_boundary_shapely(polygon: Polygon, buffer: float = 0.5) -> Polygon:
    """
    使用 Shapely 获取多边形的边界带

    数学原理:
    =========
    boundary = polygon.exterior.buffer(buffer)
    - 向外扩展 buffer 距离
    - 减去内部区域得到边界带

    Args:
        polygon: 输入多边形
        buffer: 边界宽度

    Returns:
        Polygon: 边界带多边形
    """
    interior = polygon.buffer(-buffer)
    if interior.is_empty:
        return polygon.exterior.buffer(buffer / 2)
    boundary = polygon.difference(interior)
    return boundary


def shapely_polygon_to_mpl_patch(
    polygon: Polygon,
    edgecolor: str = 'black',
    facecolor: str = 'none',
    linewidth: float = 1.0,
    alpha: float = 1.0,
    fill: bool = True
) -> patches.Polygon:
    """
    将 Shapely Polygon 转换为 Matplotlib patch

    数学转换:
    =========
    Shapely Polygon exterior coords → Matplotlib Polygon vertices

    Args:
        polygon: Shapely 多边形
        edgecolor: 边框颜色
        facecolor: 填充颜色
        linewidth: 线宽
        alpha: 透明度
        fill: 是否填充

    Returns:
        patches.Polygon: Matplotlib 多边形
    """
    coords = list(polygon.exterior.coords)

    if fill:
        patch = patches.Polygon(
            coords,
            closed=True,
            edgecolor=edgecolor,
            facecolor=facecolor,
            linewidth=linewidth,
            alpha=alpha
        )
    else:
        patch = patches.Polygon(
            coords,
            closed=True,
            edgecolor=edgecolor,
            facecolor='none',
            linewidth=linewidth,
            alpha=alpha
        )

    return patch


def draw_shapely_regions(
    ax: plt.Axes,
    regions: List[Polygon],
    depth_colors: Dict[int, str],
    linewidths: Optional[List[float]] = None,
    alpha: float = 0.3,
    fill: bool = True
):
    """
    在 axes 上绘制 Shapely 多边形区域

    数学原理:
    =========
    对每个多边形:
    1. 转换为 matplotlib patch
    2. 添加到 axes

    优化:
    - 使用 PatchCollection 批量添加提高性能
    - 支持不同深度的颜色映射

    Args:
        ax: Matplotlib axes
        regions: Shapely 多边形列表
        depth_colors: 深度颜色映射
        linewidths: 线宽列表 (按深度)
        alpha: 填充透明度
        fill: 是否填充
    """
    if not regions:
        return

    patches_list = []
    color_list = []

    for polygon in regions:
        size = polygon.area ** 0.5
        if size > 0:
            depth = int(np.log2(64 / size))
        else:
            depth = 0

        color = depth_colors.get(depth, 'gray')
        linewidth = linewidths[depth] if linewidths and depth < len(linewidths) else 1.0

        patch = shapely_polygon_to_mpl_patch(
            polygon,
            edgecolor=color,
            facecolor=color if fill else 'none',
            linewidth=linewidth,
            alpha=alpha if fill else 1.0,
            fill=fill
        )
        patches_list.append(patch)
        color_list.append(color)

    for patch in patches_list:
        ax.add_patch(patch)


# =============================================================================
# 可视化函数
# =============================================================================

def visualize_quadtree_split(
    image: torch.Tensor,
    regions: List,
    depths: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
):
    """可视化四叉树分割结果"""
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # 子图 1: 原始图像
    ax = axes[0, 0]
    img_np = image.permute(1, 2, 0).cpu().numpy()
    if img_np.shape[2] == 1:
        img_np = img_np.squeeze()
        ax.imshow(img_np, cmap='gray')
    else:
        ax.imshow(img_np)
    ax.set_title('Original Image', fontsize=12, fontweight='bold')
    ax.axis('off')

    # 子图 2: 分割边界 + 深度热图
    ax = axes[0, 1]
    if img_np.shape[2] == 1:
        ax.imshow(img_np, cmap='gray')
    else:
        ax.imshow(img_np)

    max_depth = int(depths.max().item())
    colors = plt.cm.viridis(np.linspace(0, 1, max_depth + 1))

    for i, region in enumerate(regions):
        if hasattr(region, '__iter__'):
            x1, y1, x2, y2 = region
        else:
            continue

        d = int(depths[i].item())
        edge_color = colors[d]
        linewidth = 1.5 + d * 0.5

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=linewidth,
            edgecolor=edge_color,
            facecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(cx, cy, str(d), fontsize=8, ha='center', va='center',
                color='white', fontweight='bold',
                bbox=dict(boxstyle='circle', facecolor=edge_color, alpha=0.8))

    ax.set_title('Quadtree Split with Depth', fontsize=12, fontweight='bold')
    ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, max_depth))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Depth', shrink=0.8)

    # 子图 3: Hilbert 曲线扫描顺序
    ax = axes[0, 2]
    if img_np.shape[2] == 1:
        ax.imshow(img_np, cmap='gray')
    else:
        ax.imshow(img_np)

    order = int(math.log2(image.shape[-1]))
    n = 2 ** order
    coords = generate_hilbert_curve(order)
    coords_array = np.array(coords)
    ax.plot(coords_array[:, 0], coords_array[:, 1],
            'b-', linewidth=0.5, alpha=0.5)

    ax.scatter([0], [0], c='green', s=100, marker='o', label='Start (0)', zorder=5)
    ax.scatter([n-1], [n-1], c='red', s=100, marker='X', label='End (N-1)', zorder=5)

    ax.set_title('Hilbert Curve Scan Order', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.axis('off')

    # 子图 4: 深度直方图
    ax = axes[1, 0]
    depth_counts = torch.bincount(depths.long().clamp(0, max_depth))
    depth_range = range(len(depth_counts))

    bars = ax.bar(depth_range, depth_counts.cpu().numpy(), color=colors[:len(depth_counts)], edgecolor='black')
    ax.set_xlabel('Depth', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Depth Distribution Histogram', fontsize=12, fontweight='bold')
    ax.set_xticks(range(max_depth + 1))

    for bar, count in zip(bars, depth_counts.cpu().numpy()):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count.item()), ha='center', va='bottom', fontsize=9)

    # 子图 5: 各深度 token 占比饼图
    ax = axes[1, 1]
    labels = [f'd={d}\n({count.item()})' for d, count in enumerate(depth_counts) if count > 0]
    sizes = [count.item() for count in depth_counts if count > 0]
    pie_colors = [colors[d] for d in range(len(depth_counts)) if depth_counts[d] > 0]

    if sizes:
        ax.pie(
            sizes, labels=labels, colors=pie_colors,
            autopct='%1.1f%%', startangle=90,
            explode=[0.02] * len(sizes)
        )
        ax.set_title('Token Distribution by Depth', fontsize=12, fontweight='bold')

    # 子图 6: 空间局部性保持分析
    ax = axes[1, 2]

    num_samples = 500
    hilbert_dists = []
    euclidean_dists = []

    for _ in range(num_samples):
        i = np.random.randint(0, len(coords))
        j = np.random.randint(0, len(coords))
        if i != j:
            x1, y1 = coords[i]
            x2, y2 = coords[j]
            h_dist = abs(i - j)
            e_dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            hilbert_dists.append(h_dist)
            euclidean_dists.append(e_dist)

    ax.scatter(hilbert_dists, euclidean_dists, alpha=0.3, s=5, c='blue')
    ax.set_xlabel('Hilbert Index Difference', fontsize=11)
    ax.set_ylabel('Euclidean Distance', fontsize=11)
    ax.set_title('Locality Preservation: Hilbert vs Euclidean', fontsize=12, fontweight='bold')

    z = np.polyfit(hilbert_dists, euclidean_dists, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, max(hilbert_dists), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Linear fit')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved quadtree split visualization to {save_path}")

    plt.close(fig)
    return fig


def plot_hilbert_curve(
    order: int = 4,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """绘制 Hilbert 曲线"""
    n = 2 ** order
    coords = generate_hilbert_curve(order)

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0], figsize[1]//2 + 1))

    # 左图: 绘制曲线
    ax = axes[0]
    coords_array = np.array(coords)
    colors = plt.cm.plasma(np.linspace(0, 1, len(coords)))

    for i in range(len(coords) - 1):
        ax.plot(
            [coords_array[i, 0], coords_array[i+1, 0]],
            [coords_array[i, 1], coords_array[i+1, 1]],
            color=colors[i], linewidth=1.5
        )

    ax.scatter([0], [0], c='green', s=150, marker='o', zorder=5, label='Start')
    ax.scatter([n-1], [n-1], c='red', s=150, marker='X', zorder=5, label='End')

    step = len(coords) // 8
    for i in range(0, len(coords), step):
        x, y = coords[i]
        ax.scatter([x], [y], c='black', s=30, zorder=4, alpha=0.5)
        ax.annotate(str(i), (x+0.5, y+0.5), fontsize=7, alpha=0.7)

    ax.set_xlim(-1, n)
    ax.set_ylim(-1, n)
    ax.set_aspect('equal')
    ax.set_title(f'Hilbert Curve (Order {order}, n={n}x{n})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 右图: 对比栅格顺序
    ax = axes[1]

    raster_coords = [(x, y) for y in range(n) for x in range(n)]
    raster_array = np.array(raster_coords)

    for i in range(len(raster_coords) - 1):
        ax.plot(
            [raster_array[i, 0], raster_array[i+1, 0]],
            [raster_array[i, 1], raster_array[i+1, 1]],
            color=colors[i], linewidth=1.5
        )

    ax.scatter([0], [0], c='green', s=150, marker='o', zorder=5, label='Start')
    ax.scatter([n-1], [n-1], c='red', s=150, marker='X', zorder=5, label='End')

    ax.set_xlim(-1, n)
    ax.set_ylim(-1, n)
    ax.set_aspect('equal')
    ax.set_title(f'Raster Order (n={n}x{n})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Hilbert curve visualization to {save_path}")

    plt.close(fig)
    return fig


def compare_orderings(
    order: int = 4,
    num_pairs: int = 1000,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    seed: Optional[int] = 42
):
    """对比 Hilbert 顺序 vs 栅格顺序的局部性保持能力"""
    n = 2 ** order

    hilbert_coords = generate_hilbert_curve(order)
    raster_coords = [(x, y) for y in range(n) for x in range(n)]

    hilbert_ratios = []
    raster_ratios = []

    rng = np.random.default_rng(seed)

    for _ in range(num_pairs):
        i = rng.integers(0, n * n)
        j = rng.integers(0, n * n)

        if i != j:
            h_x1, h_y1 = hilbert_coords[i]
            h_x2, h_y2 = hilbert_coords[j]
            h_dist_idx = abs(i - j)
            h_euclidean = math.sqrt((h_x2 - h_x1)**2 + (h_y2 - h_y1)**2)
            if h_dist_idx > 0:
                hilbert_ratios.append(h_euclidean / math.sqrt(h_dist_idx))

            r_x1, r_y1 = raster_coords[i]
            r_x2, r_y2 = raster_coords[j]
            r_dist_idx = abs(i - j)
            r_euclidean = math.sqrt((r_x2 - r_x1)**2 + (r_y2 - r_y1)**2)
            if r_dist_idx > 0:
                raster_ratios.append(r_euclidean / math.sqrt(r_dist_idx))

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 左图: 散点图对比
    ax = axes[0]
    ax.scatter(hilbert_ratios, raster_ratios, alpha=0.3, s=5, c='blue')
    max_val = max(max(hilbert_ratios), max(raster_ratios))
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Equal')
    ax.set_xlabel('Hilbert Ratio', fontsize=11)
    ax.set_ylabel('Raster Ratio', fontsize=11)
    ax.set_title('Locality Preservation Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 中图: 直方图对比
    ax = axes[1]
    n_bins = min(20, len(set(hilbert_ratios) | set(raster_ratios)))
    ax.hist(hilbert_ratios, bins=n_bins, alpha=0.7, label='Hilbert', color='blue', density=True)
    ax.hist(raster_ratios, bins=n_bins, alpha=0.7, label='Raster', color='orange', density=True)
    ax.set_xlabel('Distance Ratio (Euclidean / Index^0.5)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Ratio Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右图: 统计信息
    ax = axes[2]
    ax.axis('off')

    stats_text = f"""
    Locality Preservation Statistics
    =================================

    Hilbert Curve (Order {order}):
    - Mean Ratio:      {np.mean(hilbert_ratios):.3f}
    - Std Ratio:       {np.std(hilbert_ratios):.3f}
    - Median Ratio:    {np.median(hilbert_ratios):.3f}

    Raster Order:
    - Mean Ratio:      {np.mean(raster_ratios):.3f}
    - Std Ratio:       {np.std(raster_ratios):.3f}
    - Median Ratio:    {np.median(raster_ratios):.3f}

    Conclusion:
    Hilbert curve has {np.mean(raster_ratios)/np.mean(hilbert_ratios):.2f}x
    better locality preservation than raster.
    """

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved order comparison to {save_path}")

    plt.close(fig)
    return fig


def visualize_splitter_decision(
    logits: torch.Tensor,
    depths: torch.Tensor,
    selected_mask: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8),
    seed: Optional[int] = 42
):
    """可视化 Splitter 的决策过程"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 子图 1: 评分分布
    ax = axes[0, 0]
    depths_np = depths.cpu().numpy()
    logits_np = logits.cpu().numpy()
    selected_np = selected_mask.cpu().numpy()

    unique_depths = np.unique(depths_np)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_depths)))

    for i, d in enumerate(unique_depths):
        mask = depths_np == d
        selected = selected_np[mask]
        scores = logits_np[mask]

        ax.scatter(scores, np.full(len(scores), i),
                  c=['green' if s else 'gray' for s in selected],
                  s=30, alpha=0.6, label=f'd={d}')

    ax.set_xlabel('Logits', fontsize=11)
    ax.set_ylabel('Depth', fontsize=11)
    ax.set_title('Split Decision: Logits by Depth', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    # 子图 2: 深度分布
    ax = axes[0, 1]
    depth_counts_all = np.bincount(depths_np, minlength=5)
    depth_counts_selected = np.bincount(depths_np[selected_np], minlength=5)

    x = np.arange(len(depth_counts_all))
    width = 0.35

    ax.bar(x - width/2, depth_counts_all, width, label='All Candidates', color='gray', alpha=0.5)
    ax.bar(x + width/2, depth_counts_selected, width, label='Selected', color='green', alpha=0.7)

    ax.set_xlabel('Depth', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Depth Distribution: All vs Selected', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 子图 3: 选择率随深度变化
    ax = axes[1, 0]
    select_rates = []
    for d in range(len(depth_counts_all)):
        if depth_counts_all[d] > 0:
            rate = depth_counts_selected[d] / depth_counts_all[d]
            select_rates.append(rate)
        else:
            select_rates.append(0)

    ax.bar(range(len(select_rates)), select_rates, color=colors, edgecolor='black')
    ax.set_xlabel('Depth', fontsize=11)
    ax.set_ylabel('Selection Rate', fontsize=11)
    ax.set_title('Selection Rate by Depth', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(select_rates)))
    ax.grid(True, alpha=0.3)

    # 子图 4: Gumbel 扰动效果
    ax = axes[1, 1]

    rng = np.random.default_rng(seed)
    num_samples = 1000
    gumbel_samples = -np.log(-np.log(rng.random(num_samples)))

    ax.hist(gumbel_samples, bins=50, density=True, alpha=0.7, color='purple')
    ax.axvline(x=0, color='red', linestyle='--', label='Zero')
    ax.set_xlabel('Gumbel Sample', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Gumbel Distribution (for exploration)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved splitter decision visualization to {save_path}")

    plt.close(fig)
    return fig


# =============================================================================
# 演示函数
# =============================================================================

def demo_hilbert_splitter():
    """Hilbert Splitter 分割与扫描策略演示"""
    print("\n" + "=" * 70)
    print(" Hilbert Curve Splitter: Split & Scan Strategy Demo")
    print("=" * 70)

    # 1. 演示 Hilbert 曲线
    print("\n[1] Hilbert Curve Generation")
    print("-" * 50)
    order = 4
    n = 2 ** order
    coords = generate_hilbert_curve(order)
    print(f"   Order: {order}, Grid: {n}x{n}")
    print(f"   Total points: {len(coords)}")
    print(f"   Start: {coords[0]}, End: {coords[-1]}")

    # 2. 可视化曲线
    print("\n[2] Plotting Hilbert Curve")
    print("-" * 50)
    plot_hilbert_curve(order=order)

    # 3. 对比排序策略
    print("\n[3] Comparing Orderings")
    print("-" * 50)
    compare_orderings(order=order)

    print("\n[4] Demo Complete!")
    print("=" * 70)


def demo_with_model(
    model: nn.Module,
    image: torch.Tensor,
    device: str = 'cuda'
):
    """使用模型进行演示"""
    print("\n" + "=" * 70)
    print(" Hilbert Splitter Demo with Model")
    print("=" * 70)

    model = model.to(device)
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        tokenizer = model.fractal_tokenizer
        print("\n[1] Running forward pass...")
        output = tokenizer(image)

        if hasattr(output, 'regions'):
            regions = output.regions
            depths = output.depths
            hilbert_indices = output.hilbert_indices

            print(f"   Number of tokens: {len(depths)}")
            print(f"   Depth distribution: {torch.bincount(depths.long()).tolist()}")

            print("\n[2] Visualizing quadtree split...")
            visualize_quadtree_split(image[0], regions, depths)

    print("\n[3] Demo Complete!")
    print("=" * 70)


def visualize_mixed_depth_regions(
    image_size: int = 64,
    min_patch_size: int = 4,
    image: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (20, 10),
    seed: Optional[int] = 42,
    max_depth_override: Optional[int] = 5
):
    """
    可视化混合深度的四叉树分割与 Hilbert 路径 - StreamingFractalTokenizerV3 架构

    核心概念 (StreamingFractalTokenizerV3):
    ======================================
    1. 并行 Top-K 选择 (Scheme D/E - GumbelTopKSplitter)
       - 所有候选区域一次性评估 (非递归)
       - Gumbel 扰动: g = -log(-log(u)), u ~ Uniform(0,1)
       - 温度 annealing: τ_start → τ_end

    2. Scheme E: 可学习配额分配 (Learnable Quota)
       - π_d = softmax(φ)  (φ: 可学习 logits)
       - K_d = round(π_d × K_total)
       - 最小配额: K_d ≥ QUOTA_MIN_PER_DEPTH (默认 2)

    3. 树一致性约束 (Tree Consistency Constraint)
       - 若子节点被选中，则父节点不能被选中
       - 公式: ∀i ∈ S: parent(i) ∉ S
       - 保证: ∑area(selected) ≤ image_size², 无重叠

    4. Hilbert 曲线排序
       - 每个区域中心点 → Hilbert 索引
       - 空间相邻 → 索引相近
       - 保持局部性用于注意力计算

    2x4 布局展示:
    =============
    Row 1: Complete Quadtree | Gumbel-Top-K | Tree Consistency | Hilbert Order
    Row 2: Scheme E Quota    | Depth Dist   | Gumbel Dist      | Pipeline

    Args:
        image_size: 图像尺寸 (默认 64)
        min_patch_size: 最小 patch 尺寸 (默认 4)
        image: 可选真实图像背景 (H, W, 3) 或 (H, W)
        save_path: 保存路径 (可选)
        figsize: 图像大小
    """
    raw_max_depth = int(np.log2(image_size / min_patch_size))
    max_depth = min(raw_max_depth, max_depth_override) if max_depth_override is not None else raw_max_depth

    fig, axes = plt.subplots(2, 4, figsize=figsize)

    # 颜色映射 - 与 depth_encoding.py 统一使用红蓝色系 (扩展到 0-5)
    cmap = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, max_depth + 1))
    depth_colors = {d: cmap[d] for d in range(max_depth + 1)}

    def _regions_for_depth(depth: int) -> List[Tuple[float, float, float, float]]:
        n_side = 2 ** depth
        region_size = image_size / n_side
        regions = []
        for i in range(n_side):
            for j in range(n_side):
                x1, y1 = i * region_size, j * region_size
                regions.append((x1, y1, x1 + region_size, y1 + region_size))
        return regions

    def _filter_regions(regions: List[Tuple[float, float, float, float]],
                        blocked: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
        filtered = []
        for r in regions:
            x1, y1, x2, y2 = r
            overlap = False
            for b in blocked:
                bx1, by1, bx2, by2 = b
                if not (x2 <= bx1 or x1 >= bx2 or y2 <= by1 or y1 >= by2):
                    overlap = True
                    break
            if not overlap:
                filtered.append(r)
        return filtered

    # 辅助函数：绘制图像背景
    def draw_image_background(ax, img, size):
        if img is not None:
            if img.ndim == 2:
                ax.imshow(img, cmap='gray', extent=[0, size, 0, size])
            else:
                ax.imshow(img, extent=[0, size, 0, size])
        else:
            ax.add_patch(patches.Rectangle((0, 0), size, size,
                                           linewidth=3, edgecolor='black',
                                           facecolor='#f8f9fa'))

    # =========================================================================
    # Row 1, Col 1: 完整的四叉树层级结构 (Complete Quadtree)
    # =========================================================================
    ax = axes[0, 0]
    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)
    ax.set_aspect('equal')
    total_candidates = int((4 ** (max_depth + 1) - 1) / 3)
    ax.set_title(f'Complete Quadtree Hierarchy\n(All {total_candidates} Candidates)',
                 fontsize=11, fontweight='bold')
    draw_image_background(ax, image, image_size)

    # d=0: 整幅图像
    ax.add_patch(patches.Rectangle((0, 0), image_size, image_size,
                                   linewidth=4, edgecolor=depth_colors[0],
                                   facecolor='none', linestyle='-'))

    # d=1: 四个 32x32 区域
    for i in range(2):
        for j in range(2):
            x1, y1 = i * (image_size / 2), j * (image_size / 2)
            ax.add_patch(patches.Rectangle((x1, y1), image_size / 2, image_size / 2,
                                           linewidth=2, edgecolor=depth_colors[1],
                                           facecolor='none', linestyle='--'))

    # d=2 与更深层只做示意（避免过密）
    if max_depth >= 2:
        region_size = image_size / 4
        for i in range(2):
            for j in range(2):
                x1, y1 = i * region_size, j * region_size
                ax.add_patch(patches.Rectangle((x1, y1), region_size, region_size,
                                               linewidth=1.2, edgecolor=depth_colors[2],
                                               facecolor=depth_colors[2], alpha=0.25))

    ax.text(image_size/2, -5,
            f'd=0..{max_depth} candidates = {total_candidates}',
            ha='center', fontsize=9)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # =========================================================================
    # Row 1, Col 2: Gumbel-Top-K 选择
    # =========================================================================
    ax = axes[0, 1]
    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)
    ax.set_aspect('equal')
    ax.set_title('Gumbel-Top-K Selection\n(Parallel Evaluation)', fontsize=11, fontweight='bold')
    draw_image_background(ax, image, image_size)

    rng = np.random.default_rng(seed)

    # 生成树一致的混合深度“叶节点”分区（覆盖整幅图像，深度 0-5）
    selected_by_depth: Dict[int, List[Tuple[float, float, float, float]]] = {d: [] for d in range(max_depth + 1)}

    def _split_square(x1: float, y1: float, size: float) -> List[Tuple[float, float, float, float]]:
        half = size / 2
        return [
            (x1, y1, x1 + half, y1 + half),
            (x1 + half, y1, x1 + size, y1 + half),
            (x1, y1 + half, x1 + half, y1 + size),
            (x1 + half, y1 + half, x1 + size, y1 + size),
        ]

    root = (0.0, 0.0, float(image_size), float(image_size))
    d1_quads = _split_square(root[0], root[1], image_size)

    q0, q1, q2, q3 = d1_quads

    selected_by_depth[1].append(q2)

    for r in _split_square(q3[0], q3[1], q3[2] - q3[0]):
        selected_by_depth[2].append(r)

    q0_d2 = _split_square(q0[0], q0[1], q0[2] - q0[0])
    for r2 in q0_d2:
        r3_list = _split_square(r2[0], r2[1], r2[2] - r2[0])
        for r3 in r3_list:
            if r3[0] >= (q0[0] + (q0[2] - q0[0]) / 2):
                for r4 in _split_square(r3[0], r3[1], r3[2] - r3[0]):
                    selected_by_depth[4].append(r4)
            else:
                selected_by_depth[3].append(r3)

    q1_d2 = _split_square(q1[0], q1[1], q1[2] - q1[0])
    for r2 in q1_d2:
        r3_list = _split_square(r2[0], r2[1], r2[2] - r2[0])
        for r3 in r3_list:
            r4_list = _split_square(r3[0], r3[1], r3[2] - r3[0])
            for r4 in r4_list:
                for r5 in _split_square(r4[0], r4[1], r4[2] - r4[0]):
                    selected_by_depth[5].append(r5)

    for depth, regions in selected_by_depth.items():
        for x1, y1, x2, y2 in regions:
            color = depth_colors.get(depth, 'gray')
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           linewidth=2, edgecolor='white',
                                           facecolor=color, alpha=0.55))

    # 添加 Gumbel 公式标注
    ax.text(0.02, 0.98, r'$s_i + g_i$, where $g \sim$ Gumbel',
           transform=ax.transAxes, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.text(0.02, 0.02, f'Leaf partition covers full image (d=1..{max_depth})',
            transform=ax.transAxes, fontsize=9, va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # =========================================================================
    # Row 1, Col 3: 树一致性约束
    # =========================================================================
    ax = axes[0, 2]
    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)
    ax.set_aspect('equal')
    ax.set_title('Tree Consistency Constraint\n(Parent-Child Mutual Exclusion)',
                 fontsize=11, fontweight='bold')
    draw_image_background(ax, image, image_size)

    parent_rect = patches.Rectangle((0, 0), 32, 32, linewidth=3,
                                    edgecolor=depth_colors[1], facecolor='none',
                                    linestyle='--', label='Parent (blocked)')
    ax.add_patch(parent_rect)

    for i in range(2):
        for j in range(2):
            x1, y1 = i * 16, j * 16
            ax.add_patch(patches.Rectangle((x1, y1), 16, 16,
                                           linewidth=2, edgecolor='white',
                                           facecolor=depth_colors[2], alpha=0.6))

    ax.annotate('If child selected\n→ parent BLOCKED',
               xy=(16, 16), xytext=(50, 50),
               fontsize=9, ha='center',
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))

    ax.text(0.02, 0.02, r'$\forall i \in S: parent(i) \notin S$',
           transform=ax.transAxes, fontsize=10, va='bottom',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # =========================================================================
    # Row 1, Col 4: Hilbert 曲线排序
    # =========================================================================
    ax = axes[0, 3]
    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)
    ax.set_aspect('equal')
    ax.set_title('Hilbert Curve Ordering\n(Spatial Locality Preserved)',
                 fontsize=11, fontweight='bold')
    draw_image_background(ax, image, image_size)

    def get_hilbert_index(x, y, order):
        n = 2 ** order
        xn = int(x * n / image_size)
        yn = int(y * n / image_size)
        return _xy_to_d_fallback(xn, yn, n)

    region_infos = []
    flat_selected = []
    for depth, regions in selected_by_depth.items():
        for r in regions:
            flat_selected.append((depth,) + r)

    for idx, region in enumerate(flat_selected):
        depth, x1, y1, x2, y2 = region
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        h_idx = get_hilbert_index(cx, cy, max_depth)
        region_infos.append((h_idx, idx, depth, cx, cy))

    region_infos.sort(key=lambda x: x[0])

    for h_idx, idx, depth, cx, cy in region_infos:
        color = depth_colors.get(depth, 'gray')
        x1, y1, x2, y2 = flat_selected[idx][1:]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       linewidth=1.5, edgecolor='white',
                                       facecolor=color, alpha=0.55))
        ax.text(cx, cy, f'{h_idx}', ha='center', va='center',
               fontsize=4, fontweight='bold', color='white')

    for i in range(len(region_infos) - 1):
        _, _, _, cx1, cy1 = region_infos[i]
        _, _, _, cx2, cy2 = region_infos[i + 1]
        ax.annotate('', xy=(cx2, cy2), xytext=(cx1, cy1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1))

    _, _, _, start_x, start_y = region_infos[0]
    _, _, _, end_x, end_y = region_infos[-1]
    ax.scatter([start_x], [start_y], c='lime', s=100, marker='o', zorder=10, edgecolors='black')
    ax.scatter([end_x], [end_y], c='red', s=100, marker='*', zorder=10, edgecolors='black')

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # =========================================================================
    # Row 2, Col 1: Scheme E 可学习配额
    # =========================================================================
    ax = axes[1, 0]
    ax.set_title('Scheme E: Learnable Quota\n(π = softmax(φ))', fontsize=11, fontweight='bold')

    K_total = sum(len(v) for v in selected_by_depth.values())
    depths_list = list(range(max_depth + 1))
    counts = np.array([len(selected_by_depth[d]) for d in depths_list], dtype=float)
    quota_probs = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)
    K_d = counts.astype(int)

    depths = [f'd={d}' for d in depths_list]
    colors = [depth_colors.get(d, '#7f8c8d') for d in depths_list]

    bars = ax.bar(depths, K_d, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)

    for bar, prob in zip(bars, quota_probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{bar.get_height()}\n({prob*100:.0f}%)',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Token Count (K_d)')
    ax.set_xlabel('Depth')
    ax.set_ylim(0, max(K_d) + 5)

    ax.text(0.02, 0.98, r'$\pi_d = \frac{e^{\phi_d}}{\sum_k e^{\phi_k}}$' + '\n' +
            r'$K_d = \text{round}(\pi_d \times K_{total})$',
           transform=ax.transAxes, fontsize=10, va='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # =========================================================================
    # Row 2, Col 2: 深度分布统计
    # =========================================================================
    ax = axes[1, 1]
    ax.set_title('Depth Distribution\n(Actual Token Count)', fontsize=11, fontweight='bold')

    depth_counts = {d: len(selected_by_depth[d]) for d in depths_list if len(selected_by_depth[d]) > 0}

    depths = sorted(depth_counts.keys())
    counts = [depth_counts[d] for d in depths]
    bar_colors = [depth_colors[d] for d in depths]

    bars = ax.bar([f'd={d}' for d in depths], counts, color=bar_colors,
                  edgecolor='black', linewidth=1.5)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
               str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Token Count')
    ax.set_xlabel('Depth')
    ax.set_ylim(0, max(counts) + 3)

    ax_inset = ax.inset_axes([0.55, 0.3, 0.4, 0.4])
    ax_inset.pie(counts, labels=[f'd={d}' for d in depths], colors=bar_colors,
                 autopct='%1.0f%%', startangle=90)
    ax_inset.set_title('Ratio')

    # =========================================================================
    # Row 2, Col 3: Gumbel 分布
    # =========================================================================
    ax = axes[1, 2]
    ax.set_title('Gumbel Distribution\n(Temperature Annealing)', fontsize=11, fontweight='bold')

    x = np.linspace(-4, 6, 200)
    temps = [1.0, 0.5, 0.1]
    colors_temp = ['blue', 'orange', 'red']

    for temp, color in zip(temps, colors_temp):
        pdf = (1/temp) * np.exp(-(x + np.exp(-x/temp)))
        ax.plot(x, pdf, label=f'τ={temp}', color=color, linewidth=2)

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Gumbel Sample')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Row 2, Col 4: 流水线图
    # =========================================================================
    ax = axes[1, 3]
    ax.set_title('Tokenizer Pipeline\n(StreamingFractalTokenizerV3)',
                 fontsize=11, fontweight='bold')
    ax.axis('off')

    pipeline_steps = [
        ('SharedConv\n[3→D]', '#3498db'),
        ('GumbelTopK\nSplitter', '#e74c3c'),
        ('Scheme E\nQuota', '#27ae60'),
        ('Tree\nConsistency', '#9b59b6'),
        ('Hilbert\nSort', '#f39c12'),
    ]

    n_steps = len(pipeline_steps)
    for i, (name, color) in enumerate(pipeline_steps):
        x = 0.1 + i * 0.18
        rect = patches.FancyBboxPatch((x, 0.4), 0.15, 0.2,
                                      boxstyle="round,pad=0.02",
                                      facecolor=color, edgecolor='black',
                                      linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x + 0.075, 0.5, name, ha='center', va='center',
               fontsize=8, fontweight='bold', color='white')

        if i < n_steps - 1:
            ax.annotate('', xy=(x + 0.18, 0.5), xytext=(x + 0.15, 0.5),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax.text(0.1, 0.25, '[B,3,H,W]', ha='center', fontsize=8)
    ax.text(0.1, 0.1, 'Image', ha='center', fontsize=9, fontweight='bold')

    ax.text(0.92, 0.25, '[B,N,D]', ha='center', fontsize=8)
    ax.text(0.92, 0.1, 'Tokens', ha='center', fontsize=9, fontweight='bold')

    ax.text(0.5, 0.85, 'O(1) parameters | 100% gradient flow',
           transform=ax.transAxes, ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(pad=2.0, h_pad=3.0)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved mixed depth visualization to {save_path}")

    plt.close(fig)
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hilbert Splitter Visualization Demo")
    parser.add_argument("--order", type=int, default=4, help="Hilbert curve order")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save visualizations")
    parser.add_argument("--no-show", action="store_true", help="Don't show plots, just save")

    args = parser.parse_args()

    if args.save_dir:
        import os
        os.makedirs(args.save_dir, exist_ok=True)

    save_path = lambda name: os.path.join(args.save_dir, name) if args.save_dir else None

    print("\n" + "=" * 70)
    print(" Hilbert Curve Splitter Visualization Demo")
    print("=" * 70)

    print("\n[1] Hilbert Curve")
    plot_hilbert_curve(order=args.order, save_path=save_path("hilbert_curve.png"))

    print("\n[2] Order Comparison")
    compare_orderings(order=args.order, save_path=save_path("order_comparison.png"))

    print("\n[3] Demo Complete!")
    print("=" * 70)
