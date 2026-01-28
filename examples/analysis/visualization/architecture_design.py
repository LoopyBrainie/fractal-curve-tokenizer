# -*- coding: utf-8 -*-
"""
Architecture Design Visualization - 模型架构设计可视化

展示 Fractal Curve ViT 的核心架构设计，不依赖训练数据。
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as path_effects

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def plot_token_count_range(
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
):
    """
    可视化 Token 数量范围 (自适应 vs 固定)

    展示 Fractal ViT 如何根据图像复杂度动态调整 token 数量。

    数学对比:
    ========
    Standard ViT: N = (224/16)^2 = 196 (固定)
    Fractal ViT: N ∈ [8, 64] (自适应)

    核心区别:
    - Standard ViT: 固定 14x14 网格，每个 patch 16x16 像素
    - Fractal ViT: 并行选择 4^d 个候选区域，然后 Top-K 选择

    四叉树层级结构 (Complete Quadtree Hierarchy):
    ==============================================
    深度 d    | 区域数量 (4^d) | 区域大小 (64x64 图像示例)
    --------- | -------------- | -----------------------
    d=0       | 1              | 64×64
    d=1       | 4              | 32×32
    d=2       | 16             | 16×16
    d=3       | 64             | 8×8

    总候选数 = 1 + 4 + 16 + 64 = 85 个区域
    选择时：Gumbel-Top-K + 树一致性约束 (父子不能同时选)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # =================================================================
    # 子图 1: 标准 ViT 固定 token 数量 (14x14 网格)
    # =================================================================
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_title('Standard ViT\nFixed: 196 tokens', fontsize=12, fontweight='bold')

    # 14x14 网格展示
    n_tokens = 196
    grid_size = 14
    step = 10 / grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            rect = patches.Rectangle(
                (j * step + 1, i * step + 1),
                step - 1.5, step - 1.5,
                linewidth=1, edgecolor='#2980b9',
                facecolor='#3498db', alpha=0.7
            )
            ax.add_patch(rect)

    ax.text(5, -1.2, 'Patch size: 16x16\nGrid: 14x14', ha='center', fontsize=10)
    ax.axis('off')

    # =================================================================
    # 子图 2: Fractal ViT 四叉树层级结构 + 自适应选择
    # =================================================================
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_title('Fractal ViT\nQuadtree Hierarchy + Top-K (Leaf-Only)', fontsize=12, fontweight='bold')

    # 颜色映射 - 每种深度用不同颜色
    depth_colors = {0: '#e74c3c', 1: '#3498db', 2: '#27ae60', 3: '#9b59b6'}
    depth_labels = {0: 'd=0\n64×64', 1: 'd=1\n32×32', 2: 'd=2\n16×16', 3: 'd=3\n8×8'}

    # 统一坐标系统（外框 0.5~9.5）
    origin = 0.5
    full_size = 9.0

    def _region_coords(depth: int, i: int, j: int) -> tuple:
        n_side = 2 ** depth
        size = full_size / n_side
        x1 = origin + i * size
        y1 = origin + j * size
        x2 = x1 + size
        y2 = y1 + size
        return x1, y1, x2, y2

    # 绘制四叉树层级结构（从外到内）
    # d=0: 整个图像
    ax.add_patch(patches.Rectangle((origin, origin), full_size, full_size,
                                   linewidth=3, edgecolor=depth_colors[0],
                                   facecolor='none', linestyle='-'))

    # d=1: 四个 32x32 象限 (虚线)
    for i in range(2):
        for j in range(2):
            x1, y1, x2, y2 = _region_coords(1, j, i)
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor=depth_colors[1],
                                     facecolor='none', linestyle='--')
            ax.add_patch(rect)

    # d=2/d=3 的层级线框过于密集，这里仅展示 d=0/d=1 结构，
    # 具体的混合深度由“选中叶节点”高亮体现。

    # 标注每个层级的区域数量
    ax.text(5, 5, f'Total Candidates: 85\n(d=0:1, d=1:4, d=2:16, d=3:64)',
           ha='center', va='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    # 叶节点选择（树一致性：不允许父子重叠）
    # 目标：混合深度覆盖整幅图像，不留空洞
    selected = []
    # 将整幅图像按 d=1 分为四象限：
    # Q0(左下) 使用 d=1
    selected.append((1, 0, 0))

    # Q1(右下) 使用 d=2（4块）
    for i in [2, 3]:
        for j in [0, 1]:
            selected.append((2, i, j))

    # Q2(左上) 使用 d=3（16块，4x4）
    for i in range(4):
        for j in range(4, 8):
            selected.append((3, i, j))

    # Q3(右上) 使用 d=2（4块）
    for i in [2, 3]:
        for j in [2, 3]:
            selected.append((2, i, j))

    for depth, i, j in selected:
        x1, y1, x2, y2 = _region_coords(depth, i, j)
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor='white',
            facecolor=depth_colors[depth], alpha=0.65
        ))

    ax.text(5, 0.3, 'Leaf-only selection (tree-consistent)',
           ha='center', fontsize=8, color='#2c3e50')

    # 添加图例
    legend_elements = [
        patches.Patch(facecolor=depth_colors[d], edgecolor='white',
                      label=f'Selected d={d}', linewidth=1.0)
        for d in [1, 2, 3]
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7, framealpha=0.9)

    ax.axis('off')

    # =================================================================
    # 子图 3: Token 数量对比条形图
    # =================================================================
    ax = axes[2]

    labels = ['Standard ViT', 'Fractal ViT\n(Min)', 'Fractal ViT\n(Typical)', 'Fractal ViT\n(Max)']
    values = [196, 8, 32, 64]
    colors = ['#3498db', '#27ae60', '#2ecc71', '#1e8449']

    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Token Count', fontsize=11)
    ax.set_title('Token Count Comparison', fontsize=12, fontweight='bold')

    # 添加数值标签
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
               str(val), ha='center', fontsize=11, fontweight='bold')

    ax.set_ylim(0, 230)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(pad=2.0)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved token count range to {save_path}")

    plt.close(fig)
    return fig


def _common_prefix_length(i: int, j: int, width: int) -> int:
    """
    计算二进制表示的公共前缀长度。

    设 width 为固定比特宽度，公共前缀长度为：
    LCP(i, j) = width - bit_length(i XOR j)，若 i=j 则为 width。
    """
    xor = i ^ j
    if xor == 0:
        return width
    return width - xor.bit_length()


def plot_position_encoding_comparison(
    save_path: Optional[str] = None,
    figsize: tuple = (16, 8)
):
    """
    可视化位置编码参数效率对比 - FractalPositionEmbedding vs Standard ViT

    展示 O(N×D) vs O(log N) 参数效率差异，以及权重模式对比。

    数学背景:
    =========
    Standard ViT (绝对位置编码):
    - 参数数量: N × D (N tokens, D embedding dimension)
    - 复杂度: O(N×D)
    - 权重模式: 随机/无结构 (每个位置独立学习)
    - 注意力偏置: 无 (依赖相对位置编码时为 O(N²))

    Fractal ViT (LCA Hilbert 偏置 - VectorizedPathEncoder):
    - 参数数量: log₂(N) × τ_h (LCA 深度嵌入)
    - 复杂度: O(log N)
    - 注意力偏置: B[i,j] = LCAEmbed(LCA(i,j))
    - 权重模式: 结构化分块 (基于 LCA 深度的层级模式)

    VectorizedPathEncoder 原理:
    =========================
    1. Hilbert 索引 → 二进制路径: h_idx → b_1 b_2 ... b_τ
    2. 公共前缀长度 = LCA(i, j)
    3. LCA(i,j) 深度: 两个 Hilbert 索引的公共前缀位数
    4. 偏置: 基于公共前缀深度选择嵌入向量

    关键洞察:
    ========
    - Standard ViT: 每个 token 位置独立学习 D 维向量
    - Fractal ViT: 只需 log₂(N) 个深度嵌入向量，通过 LCA 计算偏置
    - 注意力偏置矩阵具有分块结构：相近位置有更近的 LCA

    增强功能:
    =========
    - 使用模拟的 LCA 分块结构展示注意力偏置矩阵
    - 添加 Hilbert 路径可视化说明
    - 展示实际 attention 偏置值分布

    Args:
        save_path: 保存路径 (可选)
        figsize: 图像大小
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # 颜色映射 - 与 depth_encoding.py 统一使用红蓝色系
    cmap = 'RdYlBu_r'

    # =========================================================================
    # Row 1, Col 1: Standard ViT 绝对位置编码
    # =========================================================================
    ax = axes[0, 0]
    ax.set_title('Standard ViT: Absolute Position Embedding\n(O(N×D) params)', fontsize=11, fontweight='bold')

    # 模拟 Standard ViT 的"随机/杂乱"权重模式
    n = 16  # tokens
    d = 64  # embedding dimension
    rng = np.random.default_rng(42)
    standard_weights = rng.normal(loc=0.3, scale=0.5, size=(n, d))

    im = ax.imshow(standard_weights, cmap='Blues', aspect='auto')
    ax.set_xlabel('Embedding Dimension (D)', fontsize=10)
    ax.set_ylabel('Token Position (N)', fontsize=10)
    ax.set_xticks(range(0, d, 16))
    ax.set_yticks(range(0, n, 4))

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Learnable Weight', fontsize=9)

    # 公式
    ax.text(0.5, -0.15, r'$P \in \mathbb{R}^{N \times D}$' + f'\nParams: {n*d:,}',
           transform=ax.transAxes, ha='center', fontsize=10)

    # =========================================================================
    # Row 1, Col 2: Fractal ViT LCA 注意力偏置矩阵 (结构化)
    # =========================================================================
    ax = axes[0, 1]
    ax.set_title('Fractal ViT: LCA Attention Bias Matrix\n(O(log N) params)', fontsize=11, fontweight='bold')

    # 计算模拟的 LCA 注意力偏置矩阵
    n_tokens = 16
    lca_bias = np.zeros((n_tokens, n_tokens))

    max_lca = int(np.log2(n_tokens))
    for i in range(n_tokens):
        for j in range(n_tokens):
            # LCA = 公共前缀长度 (固定宽度 max_lca)
            lca_depth = _common_prefix_length(i, j, max_lca)
            lca_bias[i, j] = lca_depth / max_lca

    # diag 设为最大值（自身位置）
    np.fill_diagonal(lca_bias, 1.0)

    im = ax.imshow(lca_bias, cmap=cmap, aspect='equal', vmin=0, vmax=1)
    ax.set_xlabel('Token Position j', fontsize=10)
    ax.set_ylabel('Token Position i', fontsize=10)
    ax.set_xticks(range(0, n_tokens, 4))
    ax.set_yticks(range(0, n_tokens, 4))

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Attention Bias (LCA Depth)', fontsize=9)

    # 标注分块结构
    ax.add_patch(plt.Rectangle((0, 0), 8, 8, fill=False, edgecolor='red', linewidth=2))
    ax.text(4, -1, 'Block 1', ha='center', fontsize=8, color='red')
    ax.add_patch(plt.Rectangle((8, 8), 8, 8, fill=False, edgecolor='red', linewidth=2))
    ax.text(12, 7, 'Block 2', ha='center', fontsize=8, color='red', va='bottom')

    # 公式
    ax.text(0.5, -0.15, r'$B[i,j] = \text{LCAEmbed}(\text{LCA}(i,j))$' + f'\nParams: {max_lca:,} vectors',
           transform=ax.transAxes, ha='center', fontsize=10)

    # =========================================================================
    # Row 1, Col 3: Hilbert 索引与 LCA 关系
    # =========================================================================
    ax = axes[0, 2]
    ax.set_title('Hilbert Index → LCA Computation\n(Binary Prefix Common Length)', fontsize=11, fontweight='bold')

    # 展示 Hilbert 索引的二进制表示
    h_indices = [0, 3, 12, 15]  # 示例索引
    bin_repr = [format(h, '04b') for h in h_indices]

    # 绘制表格
    ax.axis('off')
    table_data = [
        ['h_idx', 'Binary', 'LCA with h=0'],
        [str(h_indices[0]), bin_repr[0], '4 (self)'],
        [str(h_indices[1]), bin_repr[1], '2'],
        [str(h_indices[2]), bin_repr[2], '1'],
        [str(h_indices[3]), bin_repr[3], '0'],
    ]

    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.2, 0.3, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # 高亮 LCA 列
    for i in range(1, 5):
        table[(i, 2)].set_facecolor('#e8f5e9')

    # 公式
    ax.text(0.5, 0.02, r'$\text{LCA}(i,j) = \text{len}(\text{common prefix of bin}(i) \text{ and } \text{bin}(j))$',
           transform=ax.transAxes, ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # =========================================================================
    # Row 2, Col 1: Standard ViT vs Fractal ViT 注意力偏置对比
    # =========================================================================
    ax = axes[1, 0]
    ax.set_title('Attention Bias Pattern Comparison', fontsize=11, fontweight='bold')

    # 对比两种模式的注意力偏置值分布
    x = np.arange(n_tokens)
    width = 0.35

    # 提取一行的偏置值作为示例
    standard_row = standard_weights[0, :n_tokens] / standard_weights.max()
    fractal_row = lca_bias[0, :]

    ax.bar(x - width/2, standard_row, width, label='Standard ViT (Random)',
           color='#3498db', alpha=0.7)
    ax.bar(x + width/2, fractal_row, width, label='Fractal ViT (LCA)',
           color='#2ecc71', alpha=0.7)

    ax.set_xlabel('Token Position', fontsize=10)
    ax.set_ylabel('Attention Bias', fontsize=10)
    ax.set_xticks(range(0, n_tokens, 2))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # Row 2, Col 2: 参数数量对比 (对数坐标)
    # =========================================================================
    ax = axes[1, 1]
    ax.set_title('Parameter Efficiency (Log Scale)', fontsize=11, fontweight='bold')

    token_counts = [8, 16, 32, 64]
    embedding_dim = 384
    standard_params = [n * embedding_dim for n in token_counts]
    fractal_params = [int(np.log2(n)) * embedding_dim for n in token_counts]

    x = np.arange(len(token_counts))
    width = 0.35

    bars1 = ax.bar(x - width/2, standard_params, width, label='Standard ViT (N × D)',
                   color='#3498db', alpha=0.7)
    bars2 = ax.bar(x + width/2, fractal_params, width, label='Fractal ViT (log N × D)',
                   color='#2ecc71', alpha=0.7)

    ax.set_xlabel('Token Count (N)', fontsize=10)
    ax.set_ylabel('Position Parameters', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in token_counts])
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标注
    for bar, val in zip(bars1, standard_params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{val:,}', ha='center', va='bottom', fontsize=7, rotation=45)
    for bar, val in zip(bars2, fractal_params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{val}', ha='center', va='bottom', fontsize=7, rotation=45)

    # 标注压缩比
    compression = [s/f for s, f in zip(standard_params, fractal_params)]
    for i, comp in enumerate(compression):
        ax.annotate(f'{comp:.1f}×', xy=(i, fractal_params[i] * 1.5),
                   ha='center', fontsize=9, color='red', fontweight='bold')

    # =========================================================================
    # Row 2, Col 3: 关键优势总结
    # =========================================================================
    ax = axes[1, 2]
    ax.axis('off')
    ax.set_title('Key Advantages', fontsize=11, fontweight='bold')

    advantages = [
        ('O(1) Memory', 'Position bias uses constant\nmemory regardless of N'),
        ('100% Gradient Flow', 'No detached position embeddings,\nfull end-to-end learning'),
        ('Structured Prior', 'LCA depth encodes\nspatial hierarchy'),
        ('Variable N Support', 'Adaptive token count\n[8, 64] without retraining'),
    ]

    colors = ['#e3f2fd', '#e8f5e9', '#fff3e0', '#fce4ec']
    # 调整 y 位置以适应 tight_layout，避免超出边界
    # 4个框: y起始=0.82，间距=0.18，每个高度=0.16
    for i, (title, desc) in enumerate(advantages):
        y = 0.82 - i * 0.18
        rect = plt.Rectangle((0.05, y - 0.08), 0.9, 0.16, transform=ax.transAxes,
                             facecolor=colors[i], edgecolor='gray', linewidth=1)
        ax.add_patch(rect)
        ax.text(0.1, y, title, transform=ax.transAxes, fontsize=10,
               fontweight='bold', va='center')
        ax.text(0.1, y - 0.035, desc, transform=ax.transAxes, fontsize=8,
               va='center')

    # 使用 constrained_layout 避免文本被裁剪
    fig.tight_layout(pad=2.0, h_pad=3.0)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved position encoding comparison to {save_path}")

    plt.close(fig)


def plot_multi_scale_representation(
    save_path: Optional[str] = None,
    figsize: tuple = (16, 6)
):
    """
    可视化多尺度表示

    展示 depth 0-3 如何同时建模不同尺度的特征，以及 Overlay 叠加视图。

    核心概念:
    ========
    Fractal ViT 同时使用 4 个尺度的区域进行建模:
    - d=0: 1 个 64×64 区域 (整体视图)
    - d=1: 4 个 32×32 区域 (粗粒度)
    - d=2: 16 个 16×16 区域 (中粒度)
    - d=3: 64 个 8×8 区域 (细粒度)

    Overlay 视图:
    ============
    用线框粗细和颜色深浅同时表示 4 个深度层级，
    展示模型如何在不同尺度上进行空间建模。

    Args:
        save_path: 保存路径 (可选)
        figsize: 图像大小
    """
    fig, axes = plt.subplots(1, 5, figsize=figsize)  # 5个子图，额外一个用于Overlay

    image_size = 64
    # 使用与 depth_encoding.py 统一的红蓝色系
    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, 4))

    # 线宽随深度增加而减小 (d=0 粗线 → d=3 细线)
    linewidths = [4, 3, 2, 1]

    # =========================================================================
    # 子图 1-4: 各深度单独视图
    # =========================================================================
    for idx, depth in enumerate(range(4)):
        ax = axes[idx]
        ax.set_xlim(0, image_size)
        ax.set_ylim(0, image_size)
        ax.set_aspect('equal')

        # 计算该深度的区域大小
        n_side = 2 ** depth
        region_size = image_size // n_side

        # 绘制区域 (用线框表示)
        for i in range(n_side):
            for j in range(n_side):
                x1 = j * region_size
                y1 = i * region_size
                rect = patches.Rectangle(
                    (x1, y1), region_size, region_size,
                    linewidth=linewidths[idx], edgecolor=colors[idx],
                    facecolor=colors[idx], alpha=0.3
                )
                ax.add_patch(rect)

        scale_label = ['1× (64×64)', '1/4 (32×32)', '1/16 (16×16)', '1/64 (8×8)'][depth]
        ax.set_title(f'd={depth}: {scale_label}\n({4**depth} regions)', fontsize=11, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    # =========================================================================
    # 子图 5: Overlay 叠加视图 (新增)
    # =========================================================================
    ax = axes[4]
    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)
    ax.set_aspect('equal')
    ax.set_title('Overlay: All Depths Merged\n(Line width = depth)', fontsize=11, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # 绘制所有深度的线框叠加
    for depth in range(4):
        n_side = 2 ** depth
        region_size = image_size // n_side

        for i in range(n_side):
            for j in range(n_side):
                x1 = j * region_size
                y1 = i * region_size
                rect = patches.Rectangle(
                    (x1, y1), region_size, region_size,
                    linewidth=linewidths[depth], edgecolor=colors[depth],
                    facecolor='none', alpha=0.8
                )
                ax.add_patch(rect)

    # 添加图例
    legend_elements = [
        patches.Patch(facecolor='none', edgecolor=colors[d],
                      label=f'd={d} ({4**d} regions)',
                      linewidth=linewidths[d])
        for d in range(4)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    fig.suptitle('Multi-Scale Token Representation (Fractal ViT)', fontsize=14, fontweight='bold', y=1.03)

    plt.tight_layout(pad=2.0, h_pad=3.0)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved multi-scale representation to {save_path}")

    plt.close(fig)
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Architecture Design Visualization")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save visualizations")

    args = parser.parse_args()

    if args.save_dir:
        import os
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = lambda name: os.path.join(args.save_dir, name)
    else:
        save_path = lambda name: None

    print("\n[1] Token Count Range...")
    plot_token_count_range(save_path=save_path("token_count_range.png"))

    print("\n[2] Position Encoding Comparison...")
    plot_position_encoding_comparison(save_path=save_path("position_encoding_comparison.png"))

    print("\n[3] Multi-Scale Representation...")
    plot_multi_scale_representation(save_path=save_path("multi_scale_representation.png"))

    print("\nDone!")
