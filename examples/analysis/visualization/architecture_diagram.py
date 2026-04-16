# -*- coding: utf-8 -*-
"""
架构对比图可视化

展示 Fractal Curve ViT 与标准 ViT 的架构差异。
"""

import sys
from pathlib import Path
from typing import Optional

import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def plot_architecture_comparison(
    save_path: Optional[str] = None,
    figsize: tuple = (18, 12)
):
    """
    绘制 Fractal Curve ViT 与标准 ViT 的架构对比图

    数学背景:
    =========
    标准 ViT 架构:
    ┌─────────┐    ┌─────────────┐    ┌──────────────┐    ┌───────┐
    │  Image  │ →  │ Patch Embed │ →  │ Pos Encoding │ →  │ ViT   │
    │ [B,3,H,W]│    │  [B,N,D]    │    │   [B,N,D]    │    │ Blocks│
    └─────────┘    └─────────────┘    └──────────────┘    └───┬───┘
                                                                ↓
                                                          ┌───────┐
                                                          │  MLP  │
                                                          │ Head  │
                                                          └───────┘

    Fractal Curve ViT 架构:
    ┌─────────┐    ┌─────────────────┐    ┌──────────────────┐
    │  Image  │ →  │ Streaming       │ →  │ FractalPosition  │
    │ [B,3,H,W]│    │ FractalTokenizer│    │ Embedding        │
    └─────────┘    └─────────────────┘    └──────────────────┘
                            ↓                       ↑
                     ┌────────────────┐      ┌──────────────────┐
                     │ Hilbert Sort ──┼──────│ LCA + Depth Bias │
                     └────────────────┘      └──────────────────┘
                            ↓
                     ┌─────────────────────────────────────┐
                     │      FractalTransformer             │
                     │   (HilbertAwareAttention + SwiGLU)  │
                     └─────────────────────────────────────┘
                            ↓
                     ┌─────────────────────────────────────┐
                     │              MLP Head               │
                     └─────────────────────────────────────┘

    复杂度对比:
    ===========
    Standard ViT: O(N²) 位置参数 (N=196 tokens → ~38K params)
    Fractal ViT: O(log N) 位置参数 (N∈[8,64] → ~16 params)

    Args:
        save_path: 保存路径 (可选)
        figsize: 图像大小
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # =========================================================================
    # 标准 ViT 架构
    # =========================================================================
    ax = axes[0]

    # 定义组件
    components_standard = [
        {'name': 'Image\n[B,3,H,W]', 'pos': (0.5, 0.9), 'color': 'lightblue'},
        {'name': 'Patch Embed\n[B,N,D]\nN=196', 'pos': (0.5, 0.75), 'color': 'lightgreen'},
        {'name': 'Pos Encoding\n[B,N,D]\nO(N²) params', 'pos': (0.5, 0.6), 'color': 'lightyellow'},
        {'name': 'ViT Blocks\n×L', 'pos': (0.5, 0.4), 'color': 'lightcoral'},
        {'name': 'MLP Head\n[Classes]', 'pos': (0.5, 0.2), 'color': 'plum'},
    ]

    for comp in components_standard:
        rect = FancyBboxPatch(
            (comp['pos'][0] - 0.15, comp['pos'][1] - 0.05),
            0.3, 0.08,
            boxstyle="round,pad=0.02",
            facecolor=comp['color'],
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(comp['pos'][0], comp['pos'][1], comp['name'],
               ha='center', va='center', fontsize=9, fontweight='bold')

    # 绘制箭头
    for i in range(len(components_standard) - 1):
        y1 = components_standard[i]['pos'][1] - 0.05
        y2 = components_standard[i+1]['pos'][1] + 0.05
        ax.annotate('', xy=(0.5, y2), xytext=(0.5, y1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Standard ViT Architecture', fontsize=14, fontweight='bold')

    # 添加特点标注 (右侧)
    features_standard = [
        "• Fixed N = (H/P)² patches",
        "• Row-major position encoding",
        "• O(N²) position parameters (~50K)",
        "• Single-scale attention"
    ]
    for i, feature in enumerate(features_standard):
        ax.text(0.95, 0.95 - i * 0.08, feature,
               transform=ax.transAxes, fontsize=9,
               ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 添加 O(N²) 可视化标注 (移到 ViT Blocks 下方)
    ax.text(0.5, 0.32, 'O(N²) params', transform=ax.transAxes, ha='center',
           fontsize=10, color='red', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))

    # =========================================================================
    # Fractal Curve ViT 架构
    # =========================================================================
    ax = axes[1]

    # 定义组件
    components_fractal = [
        {'name': 'Image\n[B,3,H,W]', 'pos': (0.5, 0.95), 'color': 'lightblue'},
        {'name': 'Streaming\nFractalTokenizer', 'pos': (0.5, 0.78), 'color': 'lightgreen'},
        {'name': 'Fractal\nPosition Embedding', 'pos': (0.5, 0.6), 'color': 'lightyellow'},
        {'name': 'FractalTransformer\n×L', 'pos': (0.5, 0.4), 'color': 'lightcoral'},
        {'name': 'MLP Head\n[Classes]', 'pos': (0.5, 0.22), 'color': 'plum'},
    ]

    # 侧边标注 (Hilbert Sort 从右侧输出，LCA Bias 反馈到位置编码)
    side_components = [
        {'name': 'Hilbert\nSort', 'pos': (0.85, 0.78), 'color': 'lightskyblue'},
        {'name': 'LCA\nBias', 'pos': (0.15, 0.6), 'color': 'lightskyblue'},
    ]

    for comp in components_fractal:
        rect = FancyBboxPatch(
            (comp['pos'][0] - 0.18, comp['pos'][1] - 0.05),
            0.36, 0.09,
            boxstyle="round,pad=0.02",
            facecolor=comp['color'],
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(comp['pos'][0], comp['pos'][1], comp['name'],
               ha='center', va='center', fontsize=9, fontweight='bold')

    for comp in side_components:
        rect = FancyBboxPatch(
            (comp['pos'][0] - 0.1, comp['pos'][1] - 0.04),
            0.2, 0.08,
            boxstyle="round,pad=0.02",
            facecolor=comp['color'],
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(comp['pos'][0], comp['pos'][1], comp['name'],
               ha='center', va='center', fontsize=8, fontweight='bold')

    # 绘制箭头 (主流程: 向下)
    for i in range(len(components_fractal) - 1):
        y1 = components_fractal[i]['pos'][1] - 0.05
        y2 = components_fractal[i+1]['pos'][1] + 0.05
        ax.annotate('', xy=(0.5, y2), xytext=(0.5, y1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # 绘制 Hilbert Sort 箭头 (Tokenizer → Hilbert Sort)
    ax.annotate('', xy=(0.75, 0.78), xytext=(0.68, 0.78),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    # 绘制 LCA Bias 反馈箭头 (LCA Bias → Position Embedding)
    ax.annotate('', xy=(0.32, 0.6), xytext=(0.25, 0.6),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Fractal Curve ViT Architecture', fontsize=14, fontweight='bold')

    # 添加特点标注 (右侧)
    features_fractal = [
        "• Adaptive N ∈ [8, 64] tokens",
        "• Hilbert curve ordering",
        "• O(log N) position parameters (~16)",
        "• Multi-scale depth attention"
    ]
    for i, feature in enumerate(features_fractal):
        ax.text(0.02, 0.95 - i * 0.08, feature,
               transform=ax.transAxes, fontsize=9,
               ha='left', va='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # 添加 O(log N) 可视化标注 (移到 FractalTransformer 下方)
    ax.text(0.5, 0.32, 'O(log N) params', transform=ax.transAxes, ha='center',
           fontsize=10, color='green', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.8))

    plt.tight_layout(pad=2.0)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved architecture comparison to {save_path}")

    plt.close(fig)


def plot_model_components(
    model: nn.Module,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10)
):
    """
    绘制模型组件结构图

    Args:
        model: 模型实例
        save_path: 保存路径 (可选)
        figsize: 图像大小
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 提取组件信息
    components = []

    # 输入
    components.append({'name': 'Input\nImage', 'type': 'input', 'params': 0})

    # SharedConv
    if hasattr(model, 'fractal_tokenizer'):
        tokenizer = model.fractal_tokenizer
        if hasattr(tokenizer, 'shared_conv'):
            conv = tokenizer.shared_conv
            params = sum(p.numel() for p in conv.parameters())
            components.append({
                'name': f'SharedConv\n{conv[0].out_channels}ch',
                'type': 'conv',
                'params': params
            })

    # Splitter
    if hasattr(tokenizer, 'splitter'):
        splitter = tokenizer.splitter
        params = sum(p.numel() for p in splitter.parameters()) if hasattr(splitter, 'parameters') else 0
        components.append({
            'name': 'GumbelTopK\nSplitter',
            'type': 'splitter',
            'params': params
        })

    # Hilbert Attention
    if hasattr(tokenizer, 'hilbert_attention'):
        attn = tokenizer.hilbert_attention
        params = sum(p.numel() for p in attn.parameters())
        components.append({
            'name': f'HilbertAttn\n{attn.num_heads}heads',
            'type': 'attention',
            'params': params
        })

    # Transformer Blocks
    if hasattr(model, 'transformer'):
        transformer = model.transformer
        params = sum(p.numel() for p in transformer.parameters())
        components.append({
            'name': f'Transformer\n×{len(transformer.blocks)}blocks',
            'type': 'transformer',
            'params': params
        })

    # MLP Head
    if hasattr(model, 'mlp_head'):
        head = model.mlp_head
        params = sum(p.numel() for p in head.parameters())
        components.append({
            'name': 'MLP Head',
            'type': 'head',
            'params': params
        })

    # 绘制组件
    len(components)
    colors = {
        'input': 'lightblue',
        'conv': 'lightgreen',
        'splitter': 'lightyellow',
        'attention': 'lightcoral',
        'transformer': 'plum',
        'head': 'orange'
    }

    for i, comp in enumerate(components):
        x = 0.1 + (i % 4) * 0.25
        y = 0.7 - (i // 4) * 0.35

        rect = FancyBboxPatch(
            (x - 0.08, y - 0.08),
            0.16, 0.16,
            boxstyle="round,pad=0.02",
            facecolor=colors[comp['type']],
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(rect)

        # 参数数量格式化
        params_str = f"{comp['params']:,}" if comp['params'] > 0 else "N/A"
        ax.text(x, y, f"{comp['name']}\n({params_str})",
               ha='center', va='center', fontsize=8, fontweight='bold')

        # 绘制连接
        if i > 0:
            prev_x = 0.1 + ((i-1) % 4) * 0.25
            prev_y = 0.7 - ((i-1) // 4) * 0.35 - 0.08
            ax.annotate('', xy=(x, y + 0.08), xytext=(prev_x, prev_y),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # 总参数量
    total_params = sum(p.numel() for p in model.parameters())

    ax.text(0.5, 0.1, f"Total Parameters: {total_params:,}",
           ha='center', va='center', fontsize=12, fontweight='bold',
           transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Model Component Structure', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved model components to {save_path}")

    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Architecture Diagram Demo")
    parser.add_argument("--save-path", type=str, default=None,
                       help="Path to save visualization")

    args = parser.parse_args()

    plot_architecture_comparison(save_path=args.save_path)
