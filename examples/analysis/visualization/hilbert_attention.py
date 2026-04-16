# -*- coding: utf-8 -*-
"""
Hilbert Attention Visualization - Hilbert 注意力可视化

提供 Hilbert 空间中的注意力分布分析，展示局部与全局注意力的对比。

数学背景
========
1. Hilbert 邻域定义:
   对于 Hilbert 索引 h，令 Hilbert 距离 HD(h₁, h₂) ≤ r 的区域为邻域

2. 局部/全局注意力比:
   Locality Ratio = Σ_{j ∈ N(i,r)} A[i,j] / Σ_j A[i,j]

3. Hilbert 曲线性质:
   - 相邻 Hilbert 索引在空间上也相邻
   - 适合捕捉图像的空间局部性

使用方法
========
    from examples.analysis.visualization.hilbert_attention import plot_hilbert_attention_map

    plot_hilbert_attention_map(
        attention=attn_weights,  # [H, N, N] 或 [N, N]
        hilbert_indices=h_indices,  # [N] Hilbert 索引
        save_path="hilbert_attention.png"
    )
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 导入 Hilbert 工具函数
from ..utils.hilbert_utils import d_to_xy, xy_to_d


def hilbert_distance(h1: int, h2: int, n_side: int) -> float:
    """
    计算两个 Hilbert 索引之间的欧几里得距离

    Args:
        h1, h2: Hilbert 索引
        n_side: 网格边长

    Returns:
        欧几里得距离
    """
    x1, y1 = d_to_xy(h1, n_side)
    x2, y2 = d_to_xy(h2, n_side)
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def plot_hilbert_attention_map(
    attention: torch.Tensor,
    hilbert_indices: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10)
):
    """
    Hilbert 注意力图可视化

    数学背景
    =========
    1. Hilbert 邻域注意力:
       对于 query token i，定义其 Hilbert 邻域为:
       N_r(i) = {j | HD(i, j) ≤ r}

       局部注意力比率:
       LocalityRatio(r) = Σ_{j ∈ N_r(i)} A[i,j] / Σ_j A[i,j]

    2. 与 Raster 顺序对比:
       - Raster 邻域: |i - j| ≤ r (一维索引邻域)
       - Hilbert 邻域: HD(i, j) ≤ r (空间邻域)

       Hilbert 邻域更符合图像的空间局部性

    3. 注意力对角线衰减:
       对角线强度 A[i, i+k] 随带宽 k 增加而衰减
       衰减率反映模型的空间注意力聚焦程度

    预期洞察
    ========
    - Fractal ViT: 注意力自然集中在 Hilbert 邻域内
    - 对角线附近注意力强度高，说明自注意力有效
    - 空间注意力分布与 Hilbert 曲线结构一致

    Args:
        attention: [N, N] 或 [H, N, N] 注意力矩阵
        hilbert_indices: [N] Hilbert 索引 (可选，从 token 位置计算)
        save_path: 保存路径
        figsize: 图像大小
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 处理注意力矩阵
    if attention.dim() == 3:
        attention = attention.mean(dim=0)  # 平均所有 head

    attention = attention.detach().float().cpu().numpy()
    N = attention.shape[0]
    n_side = int(np.sqrt(N))

    # 计算 Hilbert 索引 (如果未提供)
    if hilbert_indices is None:
        hilbert_indices = np.arange(N)
    else:
        hilbert_indices = hilbert_indices.cpu().numpy()

    # =========================================================================
    # 子图 1: 注意力热图 + Hilbert 邻域结构
    # =========================================================================
    ax = axes[0, 0]

    im = ax.imshow(attention, cmap='YlOrRd', aspect='equal')
    ax.set_xlabel('Key Index (Hilbert Order)', fontsize=11)
    ax.set_ylabel('Query Index (Hilbert Order)', fontsize=11)
    ax.set_title('Attention Matrix with Hilbert Structure', fontsize=12, fontweight='bold')

    # 高亮 Hilbert 分块结构 (4 个主要象限)
    block_size = N // 4
    block_colors = ['blue', 'green', 'orange', 'purple']

    for b, start in enumerate(range(0, N, block_size)):
        min(start + block_size, N)
        rect = plt.Rectangle(
            (start - 0.5, start - 0.5),
            block_size, block_size,
            fill=False, edgecolor=block_colors[b],
            linewidth=2, linestyle='--', alpha=0.7
        )
        ax.add_patch(rect)

    # 添加对角线标注
    ax.plot([0, N-1], [0, N-1], 'w--', linewidth=1, alpha=0.5, label='Diagonal')

    plt.colorbar(im, ax=ax, shrink=0.8, label='Attention Weight')

    # =========================================================================
    # 子图 2: Hilbert 邻域注意力覆盖率
    # =========================================================================
    ax = axes[0, 1]

    # 计算不同邻域半径下的局部注意力覆盖率
    radii = [1, 2, 4, 8]
    local_ratios = []

    for radius in radii:
        local_attention = []
        for i in range(N):
            # 找到 Hilbert 距离 ≤ radius 的邻居
            h_i = hilbert_indices[i]
            neighbors = []
            for j in range(N):
                if i != j:
                    dist = hilbert_distance(h_i, hilbert_indices[j], n_side)
                    if dist <= radius:
                        neighbors.append(j)

            if neighbors:
                local_sum = attention[i, neighbors].sum()
                total_sum = attention[i, :].sum()
                local_attention.append(local_sum / (total_sum + 1e-8))

        if local_attention:
            local_ratios.append(np.mean(local_attention))
        else:
            local_ratios.append(0)

    # 绘制柱状图
    bars = ax.bar([f'r={r}' for r in radii], local_ratios,
                  color='steelblue', edgecolor='black', alpha=0.8)

    ax.set_xlabel('Hilbert Neighborhood Radius', fontsize=11)
    ax.set_ylabel('Local Attention Ratio', fontsize=11)
    ax.set_title('Attention within Hilbert Neighborhood\n(Higher = Better Locality)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, val in zip(bars, local_ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 添加理论最大值参考线
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50% Reference')
    ax.legend(loc='upper right')

    # =========================================================================
    # 子图 3: 注意力对角线强度衰减
    # =========================================================================
    ax = axes[1, 0]

    # 计算不同带宽的对角线平均强度
    bandwidths = list(range(0, min(16, N // 2)))
    diag_means = []
    diag_stds = []

    for bw in bandwidths:
        diags = []
        for i in range(N - bw):
            diags.append(attention[i, i + bw])

        if diags:
            diag_means.append(np.mean(diags))
            diag_stds.append(np.std(diags))
        else:
            diag_means.append(0)
            diag_stds.append(0)

    ax.fill_between(
        bandwidths,
        np.array(diag_means) - np.array(diag_stds),
        np.array(diag_means) + np.array(diag_stds),
        alpha=0.2, color='steelblue', label='±1 Std'
    )
    ax.plot(bandwidths, diag_means, 'o-', linewidth=2, markersize=6,
           color='steelblue', label='Mean')

    ax.set_xlabel('Diagonal Bandwidth (|i - j|)', fontsize=11)
    ax.set_ylabel('Mean Attention', fontsize=11)
    ax.set_title('Attention Decay from Diagonal\n(Steep decay = Stronger locality)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(bandwidths)
    ax.legend(loc='upper right')

    # =========================================================================
    # 子图 4: 空间注意力分布 (2D 投影)
    # =========================================================================
    ax = axes[1, 1]

    # 选择中心 token 作为 query
    query_idx = N // 2
    query_h = hilbert_indices[query_idx]
    qx, qy = d_to_xy(query_h, n_side)

    # 绘制 Hilbert 网格背景
    ax.set_xlim(-1, n_side)
    ax.set_ylim(-1, n_side)
    ax.set_aspect('equal')
    ax.set_xlabel('x (Hilbert Coordinate)', fontsize=11)
    ax.set_ylabel('y (Hilbert Coordinate)', fontsize=11)
    ax.set_title(f'Spatial Attention Distribution\n(Query at ({qx}, {qy}))', fontsize=12, fontweight='bold')

    # 绘制所有 token 位置 (灰色背景)
    for h in range(N):
        x, y = d_to_xy(h, n_side)
        ax.scatter([x], [y], c='lightgray', s=30, alpha=0.3, zorder=1)

    # 高亮 query 位置 (蓝色星形)
    ax.scatter([qx], [qy], c='blue', s=300, marker='*', zorder=5,
              label='Query Token', edgecolors='darkblue', linewidths=1)

    # 绘制注意力权重 (红色圆点，大小和透明度反映权重)
    max_weight = attention[query_idx, :].max()
    for j in range(N):
        if j != query_idx:
            kx, ky = d_to_xy(hilbert_indices[j], n_side)
            weight = attention[query_idx, j] / (max_weight + 1e-8)
            ax.scatter([kx], [ky], c='red', s=weight * 200 + 10,
                      alpha=weight * 0.8 + 0.2, zorder=2,
                      edgecolors='none')

    # 添加颜色条表示注意力强度
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, label='Relative Attention')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

    ax.legend(loc='upper right')

    fig.suptitle('Hilbert Attention Analysis\n(Local vs Global Attention Distribution)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Hilbert attention map to {save_path}")

    plt.close(fig)


def compare_attention_patterns(
    fractal_attention: torch.Tensor,
    standard_attention: torch.Tensor,
    hilbert_indices: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 8)
):
    """
    对比 Fractal ViT 和 Standard ViT 的注意力模式

    Args:
        fractal_attention: [N, N] Fractal ViT 注意力
        standard_attention: [N, N] Standard ViT 注意力
        hilbert_indices: [N] Hilbert 索引
        save_path: 保存路径
        figsize: 图像大小
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # 处理注意力矩阵
    if fractal_attention.dim() == 3:
        fractal_attention = fractal_attention.mean(dim=0)
    if standard_attention.dim() == 3:
        standard_attention = standard_attention.mean(dim=0)

    fractal_attention = fractal_attention.detach().float().cpu().numpy()
    standard_attention = standard_attention.detach().float().cpu().numpy()

    N = fractal_attention.shape[0]
    n_side = int(np.sqrt(N))

    if hilbert_indices is None:
        hilbert_indices = np.arange(N)
    else:
        hilbert_indices = hilbert_indices.cpu().numpy()

    # =========================================================================
    # 子图 1 & 2: 注意力热图对比
    # =========================================================================
    for idx, (attn, title) in enumerate([
        (fractal_attention, 'Fractal ViT\n(Hilbert + LCA Bias)'),
        (standard_attention, 'Standard ViT\n(Fixed Patches)')
    ]):
        ax = axes[0, idx]
        im = ax.imshow(attn, cmap='YlOrRd', aspect='equal')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Key Index')
        ax.set_ylabel('Query Index')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # =========================================================================
    # 子图 3: 注意力分布对比直方图
    # =========================================================================
    ax = axes[0, 2]

    ax.hist(fractal_attention.flatten(), bins=50, alpha=0.6,
           label='Fractal ViT', color='steelblue', density=True)
    ax.hist(standard_attention.flatten(), bins=50, alpha=0.6,
           label='Standard ViT', color='orange', density=True)

    ax.set_xlabel('Attention Weight', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Attention Distribution Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # 子图 4: 局部注意力比率对比
    # =========================================================================
    ax = axes[1, 0]

    radii = [1, 2, 4, 8]
    fractal_local = []
    standard_local = []

    for radius in radii:
        fractal_local_attention = []
        standard_local_attention = []

        for attn, local_list in [(fractal_attention, fractal_local_attention),
                                  (standard_attention, standard_local_attention)]:
            for i in range(N):
                h_i = hilbert_indices[i]
                neighbors = []
                for j in range(N):
                    if i != j:
                        dist = hilbert_distance(h_i, hilbert_indices[j], n_side)
                        if dist <= radius:
                            neighbors.append(j)
                if neighbors:
                    local_sum = attn[i, neighbors].sum()
                    total_sum = attn[i, :].sum()
                    local_list.append(local_sum / (total_sum + 1e-8))

        fractal_local.append(np.mean(fractal_local_attention) if fractal_local_attention else 0)
        standard_local.append(np.mean(standard_local_attention) if standard_local_attention else 0)

    x = np.arange(len(radii))
    width = 0.35

    ax.bar(x - width/2, fractal_local, width, label='Fractal ViT',
          color='steelblue', edgecolor='black', alpha=0.8)
    ax.bar(x + width/2, standard_local, width, label='Standard ViT',
          color='orange', edgecolor='black', alpha=0.8)

    ax.set_xlabel('Hilbert Neighborhood Radius', fontsize=11)
    ax.set_ylabel('Local Attention Ratio', fontsize=11)
    ax.set_title('Local Attention Ratio\n(Higher = Better Spatial Locality)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'r={r}' for r in radii])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # 子图 5: 对角线衰减对比
    # =========================================================================
    ax = axes[1, 1]

    bandwidths = list(range(0, min(16, N // 2)))
    fractal_diag = []
    standard_diag = []

    for bw in bandwidths:
        fractal_diag.append(np.mean([fractal_attention[i, i+bw]
                                    for i in range(N-bw) if i+bw < N]))
        standard_diag.append(np.mean([standard_attention[i, i+bw]
                                     for i in range(N-bw) if i+bw < N]))

    ax.plot(bandwidths, fractal_diag, 'o-', linewidth=2, markersize=6,
           color='steelblue', label='Fractal ViT')
    ax.plot(bandwidths, standard_diag, 's-', linewidth=2, markersize=6,
           color='orange', label='Standard ViT')

    ax.set_xlabel('Diagonal Bandwidth', fontsize=11)
    ax.set_ylabel('Mean Attention', fontsize=11)
    ax.set_title('Diagonal Attention Decay\n(Steeper = More Local)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(bandwidths)

    # =========================================================================
    # 子图 6: 统计摘要
    # =========================================================================
    ax = axes[1, 2]
    ax.axis('off')

    # 计算统计指标
    fractal_sparsity = (fractal_attention < 0.01).sum() / fractal_attention.size
    standard_sparsity = (standard_attention < 0.01).sum() / standard_attention.size

    fractal_diag_mean = np.diag(fractal_attention).mean()
    standard_diag_mean = np.diag(standard_attention).mean()

    # 绘制统计卡片
    stats = [
        ('Metric', 'Fractal ViT', 'Standard ViT', 'Winner'),
        ('Diagonal Mean', f'{fractal_diag_mean:.4f}', f'{standard_diag_mean:.4f}',
         'Fractal' if fractal_diag_mean > standard_diag_mean else 'Standard'),
        ('Sparsity (1%)', f'{fractal_sparsity:.2%}', f'{standard_sparsity:.2%}',
         'Fractal' if fractal_sparsity > standard_sparsity else 'Standard'),
        ('Locality (r=2)', f'{fractal_local[1]:.3f}', f'{standard_local[1]:.3f}',
         'Fractal' if fractal_local[1] > standard_local[1] else 'Standard'),
    ]

    table_text = 'Attention Pattern Comparison\n' + '='*35 + '\n\n'
    for row in stats[1:]:
        table_text += f'{row[0]}: {row[1]} vs {row[2]} → {row[3]}\n'

    ax.text(0.1, 0.9, table_text, transform=ax.transAxes,
           fontsize=10, va='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Fractal ViT vs Standard ViT Attention Patterns',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention comparison to {save_path}")

    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hilbert Attention Visualization Demo")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save visualization")

    args = parser.parse_args()

    print("Hilbert Attention Visualization Demo")
    print("=" * 50)

    # 创建模拟注意力数据
    N = 64  # 8x8 tokens
    n_side = 8

    # 生成 Hilbert 曲线
    hilbert_order = int(np.log2(n_side))
    hilbert_indices = [xy_to_d(x, y, n_side) for y in range(n_side) for x in range(n_side)]
    hilbert_indices = np.array(hilbert_indices)

    # 模拟注意力矩阵 (对角线附近高注意力)
    attention = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            h_dist = hilbert_distance(hilbert_indices[i], hilbert_indices[j], n_side)
            attention[i, j] = np.exp(-h_dist / 2) + 0.1 * np.random.rand()

    # 归一化
    attention = attention / attention.sum(axis=1, keepdims=True)
    attention = torch.tensor(attention)

    # 运行可视化
    save_path = args.save_path

    plot_hilbert_attention_map(
        attention,
        hilbert_indices=torch.tensor(hilbert_indices),
        save_path=save_path
    )

    print("\nDone! Run with actual model attention weights for full visualization.")
