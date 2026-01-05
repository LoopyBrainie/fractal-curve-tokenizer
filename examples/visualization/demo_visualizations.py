#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example Usage of Fractal ViT Visualization Suite

This script demonstrates how to use the comprehensive visualization tools
for analyzing the Fractal Curve Vision Transformer.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.vit_pytorch.model_fractal_vit import FractalCurveViT
from examples.visualization.seaborn_viz import SeabornVisualizer, compute_lca_depths_from_paths, extract_statistics
from examples.visualization.utils import create_synthetic_image, get_device


def main():
    """Main demonstration function."""
    print("=" * 80)
    print("Fractal Curve ViT Visualization Suite - Demo")
    print("=" * 80)
    
    # Setup
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Create output directory
    output_dir = Path("visualization_outputs")
    output_dir.mkdir(exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}")
    
    # Initialize model
    print("\n[1/5] Initializing Fractal ViT model...")
    model = FractalCurveViT(
        image_size=256,
        num_classes=1000,
        dim=384,
        depth=6,
        heads=6,
        mlp_dim=768,
        dropout=0.1,
        emb_dropout=0.1,
        tokenizer_type='streaming_v3'
    ).to(device)
    model.eval()
    print("✓ Model initialized")
    
    # Create synthetic test image
    print("\n[2/5] Generating synthetic test image...")
    img_tensor = create_synthetic_image(size=256).to(device)
    print(f"✓ Image shape: {img_tensor.shape}")
    
    # Tokenize image
    print("\n[3/5] Tokenizing image with adaptive quadtree...")
    with torch.no_grad():
        token_output = model.tokenizer(img_tensor)
        seq = token_output.sequences[0]
        tokens = seq.tokens
        levels_info = seq.get_levels()
    
    print(f"✓ Generated {tokens.shape[0]} tokens")
    print(f"  Token dimensions: {tokens.shape}")
    print(f"  Levels info shape: {levels_info.shape}")
    
    # Extract statistics
    stats = extract_statistics(token_output, levels_info)
    print(f"\n  Statistics:")
    print(f"    - Number of tokens: {stats['num_tokens']}")
    print(f"    - Mean depth: {stats['mean_depth']:.2f}")
    print(f"    - Std depth: {stats['std_depth']:.2f}")
    print(f"    - Entropy: {stats['entropy']:.3f} bits")
    print(f"    - Depth distribution: {stats['depth_distribution']}")
    
    # Initialize visualizer
    print("\n[4/5] Creating Seaborn visualizations...")
    viz = SeabornVisualizer(figsize=(12, 8), dpi=150)
    
    # Extract data
    depths = levels_info[:, 0].cpu().numpy()
    paths = levels_info[:, 1:].cpu()
    
    # Visualization 1: Depth Distribution
    print("  - Generating depth distribution plot...")
    fig1 = viz.plot_depth_distribution(
        depths,
        title="Fractal ViT: Token Depth Distribution",
        max_depth=4,
        show_entropy=True,
        save_path=output_dir / "depth_distribution.png"
    )
    print(f"    ✓ Saved to {output_dir / 'depth_distribution.png'}")
    
    # Visualization 2: LCA Bias Matrix
    print("  - Computing LCA depth matrix...")
    lca_matrix = compute_lca_depths_from_paths(paths)
    print("  - Generating LCA bias heatmap...")
    fig2 = viz.plot_lca_bias_matrix(
        lca_matrix,
        title="LCA Structural Bias Matrix",
        cmap="YlOrRd",
        save_path=output_dir / "lca_bias_matrix.png"
    )
    print(f"    ✓ Saved to {output_dir / 'lca_bias_matrix.png'}")
    
    # Visualization 3: Complexity Analysis (simulated)
    print("  - Generating complexity analysis...")
    # Simulate complexity values (in real usage, extract from tokenizer)
    complexities = np.random.beta(2, 2, size=len(depths)) * 0.8 + 0.1
    # Add correlation: higher complexity -> deeper splits
    complexities = complexities + depths * 0.05 + np.random.normal(0, 0.05, len(depths))
    complexities = np.clip(complexities, 0, 1)
    
    fig3 = viz.plot_complexity_analysis(
        complexities,
        depths,
        title="Complexity vs Depth Analysis",
        save_path=output_dir / "complexity_analysis.png"
    )
    print(f"    ✓ Saved to {output_dir / 'complexity_analysis.png'}")
    
    # Visualization 4: Comparative Analysis
    print("  - Generating comparative analysis (Fractal vs Standard ViT)...")
    fractal_stats = {
        'num_tokens': stats['num_tokens'],
        'entropy': stats['entropy'],
        'flops': 2.3e9  # Simulated
    }
    # Standard ViT with 16x16 patches on 256x256 image
    standard_stats = {
        'num_tokens': (256 // 16) ** 2,  # = 256
        'entropy': 0.0,  # Single depth
        'flops': 3.8e9  # Simulated (more tokens = more FLOPs)
    }
    
    fig4 = viz.plot_comparative_analysis(
        fractal_stats,
        standard_stats,
        metrics=['num_tokens', 'entropy', 'flops'],
        title="Fractal ViT vs Standard ViT Comparison",
        save_path=output_dir / "comparative_analysis.png"
    )
    print(f"    ✓ Saved to {output_dir / 'comparative_analysis.png'}")
    
    # Summary
    print("\n[5/5] Visualization complete!")
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    print(f"Total visualizations generated: 4")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"\nGenerated files:")
    print(f"  1. depth_distribution.png - Token depth histogram with entropy metrics")
    print(f"  2. lca_bias_matrix.png - LCA structural bias heatmap")
    print(f"  3. complexity_analysis.png - Complexity vs depth correlation study")
    print(f"  4. comparative_analysis.png - Fractal ViT vs Standard ViT comparison")
    
    print(f"\n{'=' * 80}")
    print("Key Insights:")
    print(f"{'=' * 80}")
    print(f"- Fractal ViT used {stats['num_tokens']} tokens vs {standard_stats['num_tokens']} in Standard ViT")
    token_reduction = (1 - stats['num_tokens'] / standard_stats['num_tokens']) * 100
    print(f"  → {token_reduction:.1f}% token reduction")
    print(f"- Depth entropy: {stats['entropy']:.3f} bits (multi-scale representation)")
    print(f"- Adaptive tokenization allocates tokens based on content complexity")
    print(f"- LCA bias provides hierarchical spatial structure with O(log N) parameters")
    
    print(f"\n{'=' * 80}")
    print("For More Information:")
    print(f"{'=' * 80}")
    print("- See examples/visualization/README.md for usage guide")
    print("- See examples/visualization/MATHEMATICAL_ANALYSIS.md for theoretical details")
    print("- Run 'uv run examples/visualization/app.py' for interactive dashboard")
    print("=" * 80)


if __name__ == "__main__":
    main()
