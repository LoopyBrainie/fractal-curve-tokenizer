# -*- coding: utf-8 -*-
"""
Generate all architecture visualizations to workspace/visualizations
"""

import sys
from pathlib import Path

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

print("\n" + "=" * 70)
print("Output:", OUTPUT_DIR)
for f in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"  - {f.name}")
