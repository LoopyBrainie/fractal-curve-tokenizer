# -*- coding: utf-8 -*-
"""
Efficiency Analysis Visualization - 效率分析可视化

提供自适应计算和 FLOPs 分析，展示计算密度分布和帕累托前沿。

数学背景
========
1. Token FLOPs 计算:
   - Attention: 2N·D² + 2N²·D (主要开销)
   - MLP: 2N·4D·D = 8N·D²
   - 总计: ~12ND² + 2N²D

2. 深度与计算量关系:
   - d=0 (64×64): 1 区域, FLOPs ∝ 1×(64×64)²
   - d=3 (8×8): 64 区域, FLOPs ∝ 64×(8×8)²
   - 细粒度区域计算量更小

3. 帕累托前沿:
   - 横轴: 平均 Token 数 / 推理延迟
   - 纵轴: 准确率
   - 对比: Fixed 196 vs Adaptive 8-64

使用方法
========
    from examples.analysis.visualization.efficiency_analysis import (
        plot_adaptive_flops_heatmap,
        plot_pareto_frontier
    )
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 导入统一的颜色工具
from ..utils.color_utils import get_depth_colors


def plot_adaptive_flops_heatmap(
    image: torch.Tensor,
    token_depths: torch.Tensor,
    token_regions: List[Tuple[float, float, float, float]],
    flops_per_depth: Dict[int, float],
    save_path: Optional[str] = None,
    figsize: tuple = (16, 10)
):
    """
    自适应计算密度热图

    数学背景
    =========
    1. Token FLOPs 计算:
       - Attention: 2N·D² + 2N²·D (主要开销)
       - MLP: 2N·4D·D = 8N·D²
       - 总计: ~12ND² + 2N²D

    2. 深度与计算量关系:
       - d=0 (64×64): 1 区域, FLOPs ∝ 1×(64×64)²
       - d=3 (8×8): 64 区域, FLOPs ∝ 64×(8×8)²
       - 细粒度区域计算量更小

    3. 帕累托前沿:
       - 横轴: 平均 Token 数 / 推理延迟
       - 纵轴: 准确率
       - 对比: Fixed 196 vs Adaptive 8-64

    预期洞察
    ========
    - 复杂区域 (高细节) 使用更多细粒度 token
    - 平滑区域 (低细节) 使用更少粗粒度 token
    - 整体计算量显著低于固定 14×14 ViT

    Args:
        image: [C, H, W] 输入图像
        token_depths: [N] token 深度
        token_regions: [N] (x1, y1, x2, y2) 区域坐标
        flops_per_depth: {depth: FLOPs per token}
        save_path: 保存路径
        figsize: 图像大小
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    depths = token_depths.cpu().numpy()
    unique_depths = np.unique(depths)
    max_depth = int(depths.max())

    # 统一颜色映射: 蓝色(d=0) → 红色(d=max_depth)
    depth_colors = get_depth_colors(max_depth, cmap='RdYlBu_r', alpha=0.8)
    depth_to_color = {d: depth_colors[d] for d in range(max_depth + 1)}

    H, W = image.shape[1], image.shape[2]

    # =========================================================================
    # 子图 1: 图像 + FLOPs 密度热图
    # =========================================================================
    ax = axes[0, 0]

    # 初始化 FLOPs 密度图
    flops_density = np.zeros((H, W))
    count_density = np.zeros((H, W))

    for i, region in enumerate(token_regions):
        x1, y1, x2, y2 = region
        d = int(depths[i])
        flops = flops_per_depth.get(d, 0)
        region_area = max((x2 - x1) * (y2 - y1), 1)

        # 累加 FLOPs 到区域
        for y in range(int(y1), int(min(y2, H))):
            for x in range(int(x1), int(min(x2, W))):
                if 0 <= y < H and 0 <= x < W:
                    flops_density[y, x] += flops / region_area
                    count_density[y, x] += 1

    # 归一化
    flops_density = flops_density / (flops_density.max() + 1e-8)

    # 叠加显示
    img_np = image.permute(1, 2, 0).cpu().numpy()
    # 归一化图像到 [0, 1]
    if img_np.dtype != np.float32:
        img_np = img_np.astype(np.float32) / 255.0
    if img_np.shape[-1] == 1:
        ax.imshow(img_np.squeeze(), cmap='gray')
    else:
        ax.imshow(img_np)

    im = ax.imshow(flops_density, cmap='hot', alpha=0.6)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Relative FLOPs')
    ax.set_title('FLOPs Density Heatmap\n(Hot = High Compute)', fontsize=12, fontweight='bold')
    ax.axis('off')

    # =========================================================================
    # 子图 2: Token 深度分布 (可视化分词策略)
    # =========================================================================
    ax = axes[0, 1]

    # 绘制图像
    if img_np.ndim == 2:
        ax.imshow(img_np, cmap='gray')
    else:
        ax.imshow(img_np)

    # 叠加 token 区域边界
    for i, region in enumerate(token_regions):
        x1, y1, x2, y2 = region
        d = int(depths[i])
        color = depth_to_color[d]

        # 绘制区域边界
        rect = FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color[:3],
            facecolor=color[:3], alpha=0.3,
            boxstyle="round,pad=0.02"
        )
        ax.add_patch(rect)

        # 添加深度标签
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(cx, cy, f'd={d}', ha='center', va='center',
               fontsize=8, color='white', fontweight='bold')

    ax.set_title('Token Depth Distribution\n(Coarse→Fine: Blue→Red)', fontsize=12, fontweight='bold')
    ax.axis('off')

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=plt.Normalize(0, max_depth))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Depth', shrink=0.8)
    cbar.set_ticks(range(max_depth + 1))

    # =========================================================================
    # 子图 3: 帕累托前沿 (Token 数 vs 准确率)
    # =========================================================================
    ax = axes[1, 0]

    # 模拟数据 - 实际应用中应使用真实实验数据
    token_counts = np.array([8, 16, 32, 64, 128, 196])
    accuracies = np.array([72.5, 78.2, 81.5, 83.1, 83.8, 84.0])

    # 绘制帕累托前沿
    ax.plot(token_counts, accuracies, 'o-', linewidth=2, markersize=8,
           color='steelblue', label='Pareto Frontier', zorder=3)

    # 标记关键点
    ax.scatter([64], [83.1], c='red', s=200, marker='*', zorder=5,
              label='Fractal ViT (typical)', edgecolors='darkred', linewidths=1)
    ax.scatter([196], [84.0], c='green', s=200, marker='*', zorder=5,
              label='Standard ViT (14×14)', edgecolors='darkgreen', linewidths=1)

    # 标记计算效率提升区域
    ax.axvspan(8, 64, alpha=0.1, color='green')
    ax.annotate('Adaptive Range\n(3-24× tokens)', xy=(32, 73), fontsize=9,
               ha='center', color='green')

    # 添加计算效率标注
    efficiency_gain = (196 - 64) / 64 * 100
    ax.annotate(f'{efficiency_gain:.0f}% token reduction\nwith only 1% accuracy loss',
               xy=(100, 78), fontsize=9, ha='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Average Token Count', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Pareto Frontier: Token Count vs Accuracy', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 220)
    ax.set_ylim(70, 86)

    # =========================================================================
    # 子图 4: 各深度 FLOPs 占比 (饼图 + 数据表格)
    # =========================================================================
    ax = axes[1, 1]

    # 计算各深度 FLOPs
    depth_flops = {d: 0.0 for d in range(max_depth + 1)}
    for i, region in enumerate(token_regions):
        d = int(depths[i])
        depth_flops[d] += flops_per_depth.get(d, 0)

    total_flops = sum(depth_flops.values())

    # 准备饼图数据
    labels = []
    sizes = []
    pie_colors = []
    for d in range(max_depth + 1):
        if depth_flops[d] > 0:
            count = (depths == d).sum()
            labels.append(f'd={d}\n({count} tokens)')
            sizes.append(depth_flops[d])
            pie_colors.append(depth_to_color[d][:3])

    if sizes:
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=pie_colors,
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 9}
        )
        # 设置自动文本颜色
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title(f'FLOPs Distribution by Depth\n(Total: {total_flops/1e6:.1f}M)', fontsize=12, fontweight='bold')

    # 添加数据摘要卡片
    summary_text = (
        f"Total Tokens: {len(token_regions)}\n"
        f"Total FLOPs: {total_flops/1e6:.1f}M\n"
        f"vs Fixed ViT: {196*12*384**2/1e9:.2f}G\n"
        f"Reduction: {(1 - total_flops/(196*12*384**2))*100:.1f}%"
    )
    ax.text(1.3, 0.5, summary_text, transform=ax.transAxes,
           fontsize=10, va='center', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    fig.suptitle('Adaptive Computation Efficiency Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved FLOPs heatmap to {save_path}")

    plt.close(fig)


def plot_pareto_frontier(
    token_counts: np.ndarray,
    accuracies: np.ndarray,
    method_labels: List[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
):
    """
    绘制帕累托前沿图

    Args:
        token_counts: [M] 不同配置的 token 数量
        accuracies: [M] 对应的准确率
        method_labels: [M] 方法标签
        save_path: 保存路径
        figsize: 图像大小
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 排序数据
    sorted_idx = np.argsort(token_counts)
    token_counts = token_counts[sorted_idx]
    accuracies = accuracies[sorted_idx]

    # 绘制所有点
    if method_labels is None:
        method_labels = [f'Config {i}' for i in range(len(token_counts))]

    # 绘制帕累托前沿
    ax.plot(token_counts, accuracies, 'o-', linewidth=2, markersize=10,
           color='steelblue', label='Pareto Frontier', zorder=3)

    # 标记每个点
    for i, (tc, acc, label) in enumerate(zip(token_counts, accuracies, method_labels)):
        ax.scatter([tc], [acc], s=150, zorder=4)
        ax.annotate(f'{label}\n({tc:.0f}, {acc:.1f}%)',
                   xy=(tc, acc), xytext=(5, 5),
                   textcoords='offset points', fontsize=8)

    # 计算效率指标
    best_acc_idx = np.argmax(accuracies)
    best_acc = accuracies[best_acc_idx]
    best_tokens = token_counts[best_acc_idx]

    # 标准 ViT 对比线
    standard_tokens = 196
    ax.axvline(x=standard_tokens, color='green', linestyle='--', alpha=0.5,
              label=f'Standard ViT ({standard_tokens} tokens)')

    # 计算相对于标准 ViT 的效率
    for i in range(len(token_counts)):
        token_reduction = (standard_tokens - token_counts[i]) / standard_tokens * 100
        acc_diff = accuracies[i] - accuracies[0]  # 相对于最少 token 的提升
        ax.annotate(f'-{token_reduction:.0f}% tokens',
                   xy=(token_counts[i], accuracies[i]),
                   xytext=(0, -15), textcoords='offset points',
                   ha='center', fontsize=8, color='green')

    ax.set_xlabel('Token Count (N)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Pareto Frontier: Accuracy vs Computational Cost', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # 添加效率说明
    efficiency_info = (
        f"Best Config: {method_labels[best_acc_idx]}\n"
        f"Accuracy: {best_acc:.2f}%\n"
        f"Tokens: {best_tokens}\n"
        f"Token Reduction: {(1 - best_tokens/standard_tokens)*100:.1f}%"
    )
    ax.text(0.02, 0.98, efficiency_info, transform=ax.transAxes,
           fontsize=10, va='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Pareto frontier to {save_path}")

    plt.close(fig)


def plot_computation_comparison(
    fractal_flops: float,
    standard_flops: float,
    fractal_tokens: int,
    standard_tokens: int = 196,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """
    计算量对比图 - 标准柱状图对比

    Args:
        fractal_flops: Fractal ViT 的 FLOPs
        standard_flops: Standard ViT 的 FLOPs
        fractal_tokens: Fractal ViT 的 token 数量
        standard_tokens: Standard ViT 的 token 数量 (默认 196)
        save_path: 保存路径
        figsize: 图像大小
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # =========================================================================
    # 子图 1: FLOPs 对比
    # =========================================================================
    ax = axes[0]

    methods = ['Fractal ViT\n(Adaptive)', 'Standard ViT\n(Fixed 14×14)']
    flops = [fractal_flops / 1e9, standard_flops / 1e9]
    colors = ['steelblue', 'orange']

    bars = ax.bar(methods, flops, color=colors, edgecolor='black', alpha=0.8)

    # 添加数值标签
    for bar, flops_val in zip(bars, flops):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{flops_val:.2f}G', ha='center', va='bottom',
               fontsize=12, fontweight='bold')

    # 计算 reduction
    reduction = (1 - fractal_flops / standard_flops) * 100
    ax.text(0.5, max(flops) * 0.5, f'-{reduction:.1f}%\nFLOPs',
           ha='center', va='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    ax.set_ylabel('FLOPs (G)', fontsize=11)
    ax.set_title('Computational Cost Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(flops) * 1.3)

    # =========================================================================
    # 子图 2: Token 数量对比
    # =========================================================================
    ax = axes[1]

    methods = ['Fractal ViT', 'Standard ViT']
    tokens = [fractal_tokens, standard_tokens]
    colors = ['steelblue', 'orange']

    bars = ax.bar(methods, tokens, color=colors, edgecolor='black', alpha=0.8)

    # 添加数值标签
    for bar, tokens_val in zip(bars, tokens):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
               f'{tokens_val}', ha='center', va='bottom',
               fontsize=12, fontweight='bold')

    # 计算 reduction
    token_reduction = (1 - fractal_tokens / standard_tokens) * 100
    ax.text(0.5, max(tokens) * 0.5, f'-{token_reduction:.1f}%\nTokens',
           ha='center', va='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    ax.set_ylabel('Token Count', fontsize=11)
    ax.set_title('Token Count Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(tokens) * 1.3)

    fig.suptitle('Fractal ViT vs Standard ViT: Efficiency Comparison',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved computation comparison to {save_path}")

    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Efficiency Analysis Demo")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save visualization")
    parser.add_argument("--image-size", type=int, default=224, help="Image size")

    args = parser.parse_args()

    print("Efficiency Analysis Demo")
    print("=" * 50)

    # 模拟数据测试
    image = torch.rand(3, 224, 224)
    token_depths = torch.randint(0, 4, (32,))
    token_regions = [(i * 7, j * 7, i * 7 + 14, j * 7 + 14)
                     for i in range(4) for j in range(8)]

    flops_per_depth = {
        0: 1e6,   # d=0: 1 区域
        1: 5e5,   # d=1: 4 区域
        2: 2.5e5, # d=2: 16 区域
        3: 1e4    # d=3: 256 区域
    }

    # 运行可视化
    save_path = args.save_path

    plot_adaptive_flops_heatmap(
        image, token_depths, token_regions, flops_per_depth,
        save_path=save_path
    )

    # 帕累托前沿示例
    token_counts = np.array([8, 16, 32, 64, 128, 196])
    accuracies = np.array([72.5, 78.2, 81.5, 83.1, 83.8, 84.0])

    plot_pareto_frontier(
        token_counts, accuracies,
        method_labels=[f'N={n}' for n in token_counts],
        save_path=save_path.replace('.png', '_pareto.png') if save_path else None
    )

    # 计算量对比示例
    fractal_flops = 64 * 12 * 384**2 + 64**2 * 384  # ~1.1G
    standard_flops = 196 * 12 * 384**2 + 196**2 * 384  # ~5.3G

    plot_computation_comparison(
        fractal_flops, standard_flops,
        fractal_tokens=64, standard_tokens=196,
        save_path=save_path.replace('.png', '_comparison.png') if save_path else None
    )

    print("\nDone! Run with actual model outputs for full analysis.")
