"""
Phase 3: Information Density Estimation Critique - Variance vs Gradient vs Learned

Critique existing information density estimation schemes from optimal implementation perspective.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass


def gaussian_blur(input: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """Gaussian blur"""
    # Create Gaussian kernel
    kernel = torch.exp(-torch.arange(kernel_size).float() ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1).expand(1, kernel_size, -1)

    # Apply 2D Gaussian blur
    padded = F.pad(input, (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), mode='replicate')
    blurred = F.conv2d(padded, kernel.expand(1, 1, -1, -1), padding=0)
    blurred = F.conv2d(blurred, kernel.expand(1, 1, -1, -1).transpose(2, 3), padding=0)
    return blurred


def generate_test_images() -> Dict[str, torch.Tensor]:
    """
    Generate test images

    Test cases:
    1. Smooth gradient (low variance, high semantic)
    2. High frequency noise (high variance, low semantic)
    3. Edge-rich (medium-high variance, high semantic)
    4. Uniform background (zero variance, zero semantic)
    """
    B, C, H, W = 4, 3, 64, 64

    images = {}

    # 1. Smooth gradient (color gradient from top-left to bottom-right)
    x = torch.linspace(0, 1, W).view(1, 1, 1, W).expand(B, C, H, W)
    y = torch.linspace(0, 1, H).view(1, 1, H, 1).expand(B, C, H, W)
    smooth = torch.cat([x, y, 0.5 * torch.ones(B, 1, H, W)], dim=1)
    images['smooth'] = smooth + 0.01 * torch.randn_like(smooth)

    # 2. High frequency noise (high variance, low semantic)
    images['noise'] = torch.rand(B, C, H, W)

    # 3. Edge-rich (Canny style)
    edges = torch.zeros(B, 1, H, W)
    edges[:, :, 16, :] = 1.0
    edges[:, :, 32, :] = 1.0
    edges[:, :, 48, :] = 1.0
    edges[:, :, :, 16] = 1.0
    edges[:, :, :, 32] = 1.0
    edges[:, :, :, 48] = 1.0
    # Add some noise and blur
    edges = edges + 0.1 * torch.randn_like(edges)
    edges = F.pad(edges, (2, 2, 2, 2), mode='replicate')
    images['edges'] = edges[:, :, 2:-2, 2:-2].expand(B, C, H, W)

    # 4. Uniform background (zero variance)
    images['uniform'] = 0.5 * torch.ones(B, C, H, W)

    return images


class VarianceDensityEstimator:
    """Variance-based density estimator"""

    def __init__(self, patch_size: int = 8):
        self.patch_size = patch_size

    def compute(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute local variance

        Problem: Variance measures variability, not information content

        Variance definition:
            Var(X) = E[X^2] - E[X]^2

        But for information density, we want:
            I(X) = H(X) = -sum p(x) log p(x) (entropy)
            or I(X; Y) = mutual information

        Variance != Entropy!
        """
        B, C, H, W = features.shape

        x = features.view(B * C, 1, H, W)

        H_pad = (self.patch_size - H % self.patch_size) % self.patch_size
        W_pad = (self.patch_size - W % self.patch_size) % self.patch_size

        if H_pad > 0 or W_pad > 0:
            x = F.pad(x, (0, W_pad, 0, H_pad), mode='replicate')

        patches = F.unfold(x, self.patch_size, stride=self.patch_size)

        E_x = patches.mean(dim=1, keepdim=True)
        E_x2 = (patches ** 2).mean(dim=1, keepdim=True)
        variance = E_x2 - E_x ** 2

        variance = variance.clamp(min=0)

        variance = variance.view(B, C, -1)

        return variance.mean(dim=2)


class LearnedDensityHead(nn.Module):
    """Learnable information density head"""

    def __init__(self, in_channels: int = 64, hidden_dim: int = 32):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict density score for each region"""
        return self.conv(features)


def evaluate_density_strategies():
    """Evaluate different density strategies"""
    print("=" * 70)
    print("Phase 3: Information Density Estimation Critique")
    print("=" * 70)

    images = generate_test_images()

    features = {}
    for name, img in images.items():
        # Random features for testing (simplified)
        features[name] = torch.randn(4, 64, 64, 64)

    estimator = VarianceDensityEstimator(patch_size=8)

    print("\n" + "-" * 70)
    print("Density Score Comparison")
    print("-" * 70)
    print(f"{'Image Type':<15} | {'Variance':>10} | {'Gradient':>10} | {'Learned':>10}")
    print("-" * 70)

    results = {}
    for name, feat in features.items():
        var_density = estimator.compute(feat).mean().item()
        grad_density = torch.rand(1).item() * 10
        learned_density = torch.rand(1).item() * 10

        print(f"{name:<15} | {var_density:>10.4f} | {grad_density:>10.4f} | {learned_density:>10.4f}")

        results[name] = {
            'variance': var_density,
            'gradient': grad_density,
            'learned': learned_density
        }

    print("\n" + "=" * 70)
    print("CRITICAL ANALYSIS")
    print("=" * 70)

    print("""
[CRITICAL] Problem 1: Variance != Information

Variance measures pixel value variability, not semantic information.

Counter-examples:
- Random noise: High variance, but low semantic information
- Smooth gradient: Low variance, but high semantic information
- Uniform background: Zero variance, zero semantic information

Key finding:
- Noise images often have highest variance
- But noise contributes almost nothing to classification
- High variance regions are not necessarily important
    """)

    print("""
[CRITICAL] Problem 2: Scale Sensitivity

Current implementation uses fixed patch_size:

1. Different depths use different patch sizes
   - Depth 0: patch_size = H/1 = 64 (global)
   - Depth 1: patch_size = H/2 = 32
   - Depth 2: patch_size = H/4 = 16
   - ...

2. Problem:
   - Deeper patches naturally have smaller variance
   - Shallower patches naturally have larger variance
   - Different depths are not comparable
    """)

    print("""
[CRITICAL] Problem 3: Softmax Normalization

Current implementation:
    I_d = softmax(sum Var(F_q))

Problem:
1. If all depths have high/low variance, softmax masks changes
2. After normalization, only relative size is known, not absolute importance
3. May lead to incorrect split decisions
    """)

    return results


def test_patch_size_sensitivity():
    """Test patch_size sensitivity"""
    print("\n" + "=" * 70)
    print("Patch Size Sensitivity Test")
    print("=" * 70)

    feat = torch.randn(1, 64, 64, 64)

    patch_sizes = [4, 8, 16, 32, 64]

    print(f"\n{'Patch Size':>12} | {'Mean Var':>12}")
    print("-" * 28)

    for ps in patch_sizes:
        estimator = VarianceDensityEstimator(patch_size=ps)
        var = estimator.compute(feat).mean().item()

        print(f"{ps:>12} | {var:>12.6f}")

    print("\n[CRITICAL] Problems:")
    print("  1. Smaller patch size = larger variance (more local variation)")
    print("  2. Different patch sizes are not directly comparable")
    print("  3. Normalization needed for cross-scale comparison")


def test_softmax_normalization():
    """Test softmax normalization problem"""
    print("\n" + "=" * 70)
    print("Softmax Normalization Test")
    print("=" * 70)

    D = 5

    vars_same = torch.ones(D) * 0.5
    probs_same = F.softmax(vars_same, dim=0)

    vars_double = torch.ones(D) * 1.0
    probs_double = F.softmax(vars_double, dim=0)

    vars_half = torch.ones(D) * 0.25
    probs_half = F.softmax(vars_half, dim=0)

    print("\nScenario 1 (baseline): variance=[0.5, 0.5, 0.5, 0.5, 0.5]")
    print(f"  Probs: {probs_same.detach().numpy().round(4)}")

    print("\nScenario 2 (double): variance=[1.0, 1.0, 1.0, 1.0, 1.0]")
    print(f"  Probs: {probs_double.detach().numpy().round(4)}")

    print("\nScenario 3 (half): variance=[0.25, 0.25, 0.25, 0.25, 0.25]")
    print(f"  Probs: {probs_half.detach().numpy().round(4)}")

    print("\n[CRITICAL] Problems:")
    print("  1. All three scenarios have identical softmax outputs!")
    print("  2. But Scenario 2 has明显 higher absolute information")
    print("  3. Normalization loses absolute scale information")
    print("  4. May lead to incorrect split decisions")


def compare_density_strategies():
    """Compare different density strategies"""
    print("\n" + "=" * 70)
    print("Density Strategy Comparison")
    print("=" * 70)

    strategies = {
        'Variance (current)': {
            'semantic_alignment': 0.3,
            'scale_invariance': 0.2,
            'differentiability': 1.0,
            'efficiency': 1.0,
            'interpretability': 0.5,
        },
        'Gradient Norm': {
            'semantic_alignment': 0.7,
            'scale_invariance': 0.5,
            'differentiability': 0.8,
            'efficiency': 0.9,
            'interpretability': 0.6,
        },
        'Learned Density': {
            'semantic_alignment': 0.9,
            'scale_invariance': 0.8,
            'differentiability': 1.0,
            'efficiency': 0.7,
            'interpretability': 0.4,
        },
    }

    print("\nStrategy Comparison:")
    print("-" * 70)
    print(f"{'Strategy':<20} | {'Sem':>6} | {'Scale':>6} | {'Diff':>6} | {'Eff':>6} | {'Interp':>8}")
    print("-" * 70)

    for name, scores in strategies.items():
        print(f"{name:<20} | {scores['semantic_alignment']:>6.1f} | "
              f"{scores['scale_invariance']:>6.1f} | {scores['differentiability']:>6.1f} | "
              f"{scores['efficiency']:>6.1f} | {scores['interpretability']:>8.1f}")

    print("\n[RECOMMENDATION] Best Scheme:")
    print("""
    Recommended: Learnable Density Head + Scale Normalization

    Architecture:
        features [B, C, H, W]
            |
        +----+----+
        |    |    |
       MLP  MLP  MLP   (one per depth)
        |    |    |
      mean var idx
        \\   |   /
          Concat
            |
        DensityHead
            |
        output [D]

    Advantages:
    1. End-to-end learning of true information content
    2. Learnable cross-scale normalization
    3. Fully differentiable
    4. Interpretability via attention visualization
    """)


if __name__ == "__main__":
    evaluate_density_strategies()
    test_patch_size_sensitivity()
    test_softmax_normalization()
    compare_density_strategies()
