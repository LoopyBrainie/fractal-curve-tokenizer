"""
Visualization and Comparison of Scheme B vs Scheme C

This script provides visual analysis of the two adaptive splitting schemes:
- Scheme B: Balanced Greedy (2:1 constraint)
- Scheme C: Fixed Budget DP

Usage:
    python -m examples.visualize_adaptive_split --image path/to/image.jpg
    python -m examples.visualize_adaptive_split --random  # Use random image
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vit_pytorch.adaptive_split import (
    AdaptiveSplitConfig,
    SplitScheme,
    SplitResult,
    SplitToken,
    BalancedGreedySplitter,
    FixedBudgetDPSplitter,
    compare_schemes,
)


def create_test_images():
    """Create various test images for comparison."""
    images = {}
    
    # 1. Simple gradient
    gradient = torch.zeros(3, 224, 224)
    for i in range(224):
        gradient[:, :, i] = i / 223.0
    images['gradient'] = gradient
    
    # 2. Checkerboard
    checker = torch.zeros(3, 224, 224)
    for i in range(224):
        for j in range(224):
            if (i // 28 + j // 28) % 2 == 0:
                checker[:, i, j] = 1.0
    images['checkerboard'] = checker
    
    # 3. Mixed complexity
    mixed = torch.zeros(3, 224, 224)
    # Top-left: uniform
    mixed[:, :112, :112] = 0.5
    # Top-right: texture
    mixed[:, :112, 112:] = torch.rand(3, 112, 112) * 0.5 + 0.25
    # Bottom-left: edge
    mixed[:, 112:, :56] = 0.2
    mixed[:, 112:, 56:112] = 0.8
    # Bottom-right: gradient
    for i in range(112):
        mixed[:, 112:, 112+i] = i / 111.0
    images['mixed'] = mixed
    
    # 4. Natural-like (simulated)
    torch.manual_seed(42)
    natural = torch.rand(3, 224, 224) * 0.3 + 0.35
    # Add some structure
    natural[:, 50:150, 50:180] += 0.15
    natural[:, 100:200, 100:200] -= 0.15
    natural = natural.clamp(0, 1)
    images['natural'] = natural
    
    # 5. Sparse edges
    sparse = torch.ones(3, 224, 224) * 0.5
    sparse[:, 100:120, :] = 0.9  # horizontal line
    sparse[:, :, 100:120] = 0.1  # vertical line
    images['sparse_edges'] = sparse
    
    return images


def visualize_split_result(
    image: torch.Tensor,
    result: SplitResult,
    ax: plt.Axes,
    title: str,
    show_image: bool = True
):
    """Visualize split result on an image."""
    if image.dim() == 4:
        image = image.squeeze(0)
    
    H, W = image.shape[1], image.shape[2]
    
    # Show image
    if show_image:
        img_np = image.permute(1, 2, 0).numpy()
        if img_np.shape[2] == 1:
            img_np = img_np.squeeze(-1)
        ax.imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
    
    # Color by depth
    max_depth = max(t.depth for t in result.tokens)
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=max_depth)
    
    for token in result.tokens:
        r = token.region
        color = cmap(norm(token.depth))
        rect = mpatches.Rectangle(
            (r.x1, r.y1), r.width, r.height,
            linewidth=1,
            edgecolor=color,
            facecolor=(*color[:3], 0.2)
        )
        ax.add_patch(rect)
    
    # Stats
    depth_dist = result.depth_distribution
    depth_str = ", ".join([f"d{k}:{v}" for k, v in sorted(depth_dist.items())])
    ax.set_title(f"{title}\nTokens: {result.num_tokens}, Entropy: {result.depth_entropy:.2f}\n{depth_str}")
    ax.axis('off')


def compare_and_visualize(
    image: torch.Tensor,
    image_name: str,
    config_overrides: Optional[dict] = None
):
    """Compare schemes and create visualization."""
    overrides = config_overrides or {}
    
    # Run both schemes
    results = compare_schemes(image, overrides)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    if image.dim() == 4:
        image = image.squeeze(0)
    img_np = image.permute(1, 2, 0).numpy()
    axes[0].imshow(img_np)
    axes[0].set_title(f"Original: {image_name}")
    axes[0].axis('off')
    
    # Scheme B
    visualize_split_result(
        image, results['scheme_b'], axes[1],
        "Scheme B: Balanced Greedy"
    )
    
    # Scheme C
    visualize_split_result(
        image, results['scheme_c'], axes[2],
        f"Scheme C: Fixed Budget ({overrides.get('token_budget', 64)})"
    )
    
    plt.tight_layout()
    return fig, results


def print_comparison_stats(results: dict):
    """Print detailed comparison statistics."""
    print("\n" + "="*60)
    print("SCHEME COMPARISON STATISTICS")
    print("="*60)
    
    for scheme_name, result in results.items():
        print(f"\n{scheme_name.upper()}:")
        print(f"  Token count: {result.num_tokens}")
        print(f"  Depth entropy: {result.depth_entropy:.4f}")
        print(f"  Depth distribution: {result.depth_distribution}")
        
        # Compute Hilbert jump statistics
        if len(result.tokens) > 1:
            jumps = []
            for i in range(len(result.tokens) - 1):
                t1, t2 = result.tokens[i], result.tokens[i+1]
                c1, c2 = t1.region.center, t2.region.center
                jump = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5
                jumps.append(jump)
            print(f"  Hilbert jump (mean): {np.mean(jumps):.2f} px")
            print(f"  Hilbert jump (max): {max(jumps):.2f} px")
        
        # Complexity statistics
        complexities = [t.complexity for t in result.tokens]
        print(f"  Complexity (mean): {np.mean(complexities):.4f}")
        print(f"  Complexity (std): {np.std(complexities):.4f}")


def run_comprehensive_comparison():
    """Run comprehensive comparison on multiple test images."""
    print("Creating test images...")
    images = create_test_images()
    
    print("\nRunning comparisons...")
    
    # Different configurations to test
    configs = [
        {"tau_0": 0.10, "gamma": 0.80, "token_budget": 64},  # Aggressive
        {"tau_0": 0.15, "gamma": 0.85, "token_budget": 64},  # Balanced
        {"tau_0": 0.20, "gamma": 0.90, "token_budget": 64},  # Conservative
    ]
    
    for image_name, image in images.items():
        print(f"\n{'='*60}")
        print(f"Image: {image_name}")
        print(f"{'='*60}")
        
        for i, config in enumerate(configs):
            config_name = ["Aggressive", "Balanced", "Conservative"][i]
            print(f"\n--- Configuration: {config_name} ---")
            
            results = compare_schemes(image, config)
            
            print(f"Scheme B tokens: {results['scheme_b'].num_tokens}")
            print(f"Scheme C tokens: {results['scheme_c'].num_tokens}")
            print(f"Scheme B depth dist: {results['scheme_b'].depth_distribution}")
            print(f"Scheme C depth dist: {results['scheme_c'].depth_distribution}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and compare adaptive splitting schemes"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to input image"
    )
    parser.add_argument(
        "--random", action="store_true",
        help="Use random test image"
    )
    parser.add_argument(
        "--all-tests", action="store_true",
        help="Run comprehensive comparison on all test images"
    )
    parser.add_argument(
        "--budget", type=int, default=64,
        help="Token budget for Scheme C"
    )
    parser.add_argument(
        "--tau0", type=float, default=0.15,
        help="Root threshold tau_0"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.85,
        help="Threshold decay gamma"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save figure to path"
    )
    
    args = parser.parse_args()
    
    if args.all_tests:
        run_comprehensive_comparison()
        return
    
    # Create or load image
    if args.image:
        from PIL import Image
        import torchvision.transforms as T
        
        img = Image.open(args.image).convert('RGB')
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
        image = transform(img)
        image_name = Path(args.image).name
    else:
        # Use test image
        torch.manual_seed(42)
        images = create_test_images()
        image = images['mixed']
        image_name = "mixed_test"
    
    # Configuration
    config = {
        'tau_0': args.tau0,
        'gamma': args.gamma,
        'token_budget': args.budget,
    }
    
    # Compare and visualize
    fig, results = compare_and_visualize(image, image_name, config)
    print_comparison_stats(results)
    
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to {args.save}")
    
    plt.show()


if __name__ == "__main__":
    main()
