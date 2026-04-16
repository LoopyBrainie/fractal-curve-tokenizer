# -*- coding: utf-8 -*-
"""
Generate all architecture visualizations to workspace/visualizations

包含所有专家级可视化模块:
- feature_manifold.py: t-SNE/UMAP 跨深度特征流形分析
- hilbert_attention.py: Hilbert 空间注意力分布
- efficiency_analysis.py: FLOPs 热图和帕累托前沿
"""

import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

# 设置输出目录 (项目根目录下的 workspace/visualizations)
OUTPUT_DIR = PROJECT_ROOT / "workspace" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Output: {OUTPUT_DIR}")
print("=" * 70)

# Import modules
from examples.analysis import (
    demo_hilbert_splitter,
    plot_architecture_comparison,
    plot_hilbert_curve,
    compare_orderings,
    plot_locality_preservation,
    visualize_mixed_depth_regions,
    plot_depth_embedding_similarity,
    plot_token_count_range,
    plot_position_encoding_comparison,
    plot_multi_scale_representation,
    # New expert-level visualizations
    plot_cross_depth_feature_manifold,
    plot_hilbert_attention_map,
    plot_adaptive_flops_heatmap,
    plot_pareto_frontier,
    plot_computation_comparison,
)

# [1] Hilbert Splitter Demo
print("[1] Hilbert Splitter...")
demo_hilbert_splitter()

# [2] Hilbert Curve
print("[2] Hilbert Curve...")
plot_hilbert_curve(order=4, save_path=str(OUTPUT_DIR / "hilbert_curve.png"))

# [3] Orderings Comparison (Hilbert vs Raster)
print("[3] Orderings Comparison...")
compare_orderings(order=4, save_path=str(OUTPUT_DIR / "orderings_comparison.png"))

# [4] Locality Preservation
print("[4] Locality Preservation...")
plot_locality_preservation(order=4, num_samples=300, save_path=str(OUTPUT_DIR / "locality_preservation.png"))

# [5] Mixed Depth Regions (Correct quadtree structure)
print("[5] Mixed Depth Regions...")
visualize_mixed_depth_regions(image_size=64, min_patch_size=2, max_depth_override=5,
                              save_path=str(OUTPUT_DIR / "mixed_depth_regions.png"))

# [6] Architecture Comparison
print("[6] Architecture Comparison...")
plot_architecture_comparison(save_path=str(OUTPUT_DIR / "architecture_comparison.png"))

# [7] Architecture Design Visualizations (No training data)
print("[7] Token Count Range...")
plot_token_count_range(save_path=str(OUTPUT_DIR / "token_count_range.png"))

print("[8] Position Encoding Comparison...")
plot_position_encoding_comparison(save_path=str(OUTPUT_DIR / "position_encoding_comparison.png"))

print("[9] Multi-Scale Representation...")
plot_multi_scale_representation(save_path=str(OUTPUT_DIR / "multi_scale_representation.png"))

# [10] Depth Embedding Similarity
print("[10] Depth Embedding Similarity...")
sample_depth_embed = torch.randn(5, 128)
plot_depth_embedding_similarity(depth_embed=sample_depth_embed, save_path=str(OUTPUT_DIR / "depth_embedding_similarity.png"))

# ============================================================================
# [11-13] New Expert-Level Visualizations
# ============================================================================

# [11] Cross-Depth Feature Manifold (t-SNE/UMAP analysis)
print("\n[11] Cross-Depth Feature Manifold...")
n_tokens = 500
dim = 384
max_depth = 4

# Generate simulated features with depth-dependent structure
features = []
depths = []
for d in range(max_depth + 1):
    n_d = n_tokens // (max_depth + 1) + (d * 20)
    center = np.zeros(dim)
    center[d * 50:(d + 1) * 50] = 1.0
    feat = np.random.randn(n_d, dim) + center
    features.append(feat)
    depths.extend([d] * n_d)

features = torch.tensor(np.vstack(features), dtype=torch.float32)
depths = torch.tensor(depths, dtype=torch.long)

plot_cross_depth_feature_manifold(
    token_features=features,
    token_depths=depths,
    save_path=str(OUTPUT_DIR / "feature_manifold.png"),
    method='tsne'
)

# [12] Hilbert Attention Map
print("[12] Hilbert Attention Map...")
N = 64  # 8x8 tokens
n_side = 8

# Generate Hilbert indices
from examples.analysis.utils.hilbert_utils import xy_to_d
hilbert_indices = torch.tensor([xy_to_d(x % n_side, x // n_side, n_side) for x in range(N)])

# Simulate attention matrix (diagonal-heavy with Hilbert locality)
attention = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        # Hilbert distance-based attention
        x_i, y_i = i % n_side, i // n_side
        x_j, y_j = j % n_side, j // n_side
        h_dist = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
        attention[i, j] = np.exp(-h_dist / 2) + 0.05 * np.random.rand()

# Row-normalize
attention = attention / attention.sum(axis=1, keepdims=True)
attention = torch.tensor(attention)

plot_hilbert_attention_map(
    attention=attention,
    hilbert_indices=hilbert_indices,
    save_path=str(OUTPUT_DIR / "hilbert_attention_map.png")
)

# [13] Efficiency Analysis (FLOPs Heatmap + Pareto Frontier)
print("[13] Efficiency Analysis...")

# Simulate image and token data
image = torch.rand(3, 224, 224)
token_depths = torch.randint(0, 4, (32,))
token_regions = [(i * 7, j * 7, i * 7 + 14, j * 7 + 14)
                 for i in range(4) for j in range(8)]
flops_per_depth = {0: 1e6, 1: 5e5, 2: 2.5e5, 3: 1e4}

plot_adaptive_flops_heatmap(
    image=image,
    token_depths=token_depths,
    token_regions=token_regions,
    flops_per_depth=flops_per_depth,
    save_path=str(OUTPUT_DIR / "efficiency_analysis.png")
)

# [14] Pareto Frontier (Token Count vs Accuracy)
print("[14] Pareto Frontier...")
token_counts = np.array([8, 16, 32, 64, 128, 196])
accuracies = np.array([72.5, 78.2, 81.5, 83.1, 83.8, 84.0])

plot_pareto_frontier(
    token_counts=token_counts,
    accuracies=accuracies,
    method_labels=[f'N={n}' for n in token_counts],
    save_path=str(OUTPUT_DIR / "pareto_frontier.png")
)

# [15] Computation Comparison (Fractal vs Standard ViT)
print("[15] Computation Comparison...")
fractal_flops = 64 * 12 * 384**2 + 64**2 * 384  # ~1.1G
standard_flops = 196 * 12 * 384**2 + 196**2 * 384  # ~5.3G

plot_computation_comparison(
    fractal_flops=fractal_flops,
    standard_flops=standard_flops,
    fractal_tokens=64,
    standard_tokens=196,
    save_path=str(OUTPUT_DIR / "computation_comparison.png")
)

print("\n" + "=" * 70)
print("Output:", OUTPUT_DIR)
for f in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"  - {f.name}")
