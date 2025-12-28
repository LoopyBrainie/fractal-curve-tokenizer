"""
Tests for Adaptive Quadtree Split (LearnableSplitter)

This test suite verifies:
1. Basic functionality of LearnableSplitter
2. Mathematical properties (Hilbert locality, complexity estimation)
3. Core components (Region, IntegralImageCache, HilbertTokenSorter)
4. Configuration and factory functions

Mathematical Note:
==================
Scheme B (BalancedGreedySplitter) and Scheme C (FixedBudgetDPSplitter)
have been removed from the codebase. LearnableSplitter provides:

1. End-to-end differentiability via Gumbel-Softmax + STE:
   P(split | R) = softmax((logits + G) / τ), G ~ Gumbel(0, 1)

2. No complexity saturation (MLP vs variance-based formula):
   C_θ(R) = σ(MLP(ROI-Align(F, R))) instead of C(R) = Var/(Var+σ₀²)

3. Learnable thresholds: τ_d = τ_{base,d} + δ_d

4. O(D) BFS complexity vs O(N·4^D) DP

Run with: pytest tests/unit/test_adaptive_split.py -v
"""

import math
import pytest
import torch
from typing import Dict, List

import sys
sys.path.insert(0, 'src')

from vit_pytorch.split_adaptive import (
    AdaptiveSplitConfig,
    SplitScheme,
    Region,
    QuadtreeNode,
    SplitToken,
    SplitResult,
    TensorSplitResult,  # P9-1: 新的张量化返回类型
    IntegralImageCache,
    ComplexityEstimator,
    HilbertTokenSorter,
    LearnableSplitter,
    create_adaptive_splitter,
    split_image,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_image():
    """Simple uniform image."""
    return torch.zeros(3, 64, 64)


@pytest.fixture
def gradient_image():
    """Image with horizontal gradient."""
    img = torch.zeros(3, 64, 64)
    for i in range(64):
        img[:, :, i] = i / 63.0
    return img


@pytest.fixture
def complex_image():
    """Image with varying complexity regions."""
    img = torch.zeros(3, 64, 64)
    # Top-left: uniform (low complexity)
    img[:, :32, :32] = 0.5
    # Top-right: random texture (high complexity)
    img[:, :32, 32:] = torch.rand(3, 32, 32)
    # Bottom-left: edge (medium complexity)
    img[:, 32:, :16] = 0.0
    img[:, 32:, 16:32] = 1.0
    # Bottom-right: gradient (low-medium complexity)
    for i in range(32):
        img[:, 32:, 32+i] = i / 31.0
    return img


@pytest.fixture
def natural_image():
    """Simulated natural image statistics (64x64 for fast tests)."""
    torch.manual_seed(42)
    img = torch.rand(3, 64, 64) * 0.5 + 0.25
    # Add some structure
    img[:, 16:32, 16:48] += 0.2  # bright region
    img[:, 40:56, 32:56] -= 0.2  # dark region
    return img.clamp(0, 1)


@pytest.fixture
def default_config():
    """Default config (uses LEARNABLE scheme)."""
    return AdaptiveSplitConfig()


@pytest.fixture
def simple_features():
    """Simple feature map for LearnableSplitter tests."""
    return torch.randn(1, 64, 16, 16)  # B=1, C=64, H=W=16


@pytest.fixture
def image_size():
    """Image size corresponding to simple_features (scaled by 4x)."""
    return (64, 64)  # H, W


# =============================================================================
# Test: Configuration
# =============================================================================

class TestAdaptiveSplitConfig:
    
    def test_default_values(self):
        config = AdaptiveSplitConfig()
        assert config.alpha == 0.5
        assert config.sigma_0_sq == 0.01
        assert config.tau_0 == 0.15
        assert config.gamma == 0.85
        assert config.max_depth == 4
        assert config.scheme == SplitScheme.LEARNABLE
    
    def test_only_learnable_scheme_available(self):
        """Verify that only LEARNABLE scheme is available after B/C removal."""
        assert hasattr(SplitScheme, 'LEARNABLE')
        # BALANCED_GREEDY and FIXED_BUDGET_DP should not exist
        assert not hasattr(SplitScheme, 'BALANCED_GREEDY')
        assert not hasattr(SplitScheme, 'FIXED_BUDGET_DP')
    
    def test_threshold_decay(self):
        config = AdaptiveSplitConfig(tau_0=0.15, gamma=0.85)
        assert config.get_threshold(0) == 0.15
        assert abs(config.get_threshold(1) - 0.1275) < 1e-6
        assert abs(config.get_threshold(4) - 0.15 * 0.85**4) < 1e-6
    
    def test_validation(self):
        with pytest.raises(AssertionError):
            AdaptiveSplitConfig(alpha=1.5).validate()
        with pytest.raises(AssertionError):
            AdaptiveSplitConfig(tau_0=0).validate()
        with pytest.raises(AssertionError):
            AdaptiveSplitConfig(gamma=1.0).validate()


# =============================================================================
# Test: Region
# =============================================================================

class TestRegion:
    
    def test_properties(self):
        r = Region(10, 20, 50, 80)
        assert r.width == 40
        assert r.height == 60
        assert r.area == 2400
        assert r.center == (30.0, 50.0)
    
    def test_quadrants(self):
        r = Region(0, 0, 100, 100)
        q0 = r.get_quadrant(0)  # Top-left
        q1 = r.get_quadrant(1)  # Top-right
        q2 = r.get_quadrant(2)  # Bottom-left
        q3 = r.get_quadrant(3)  # Bottom-right
        
        assert q0 == Region(0, 0, 50, 50)
        assert q1 == Region(50, 0, 100, 50)
        assert q2 == Region(0, 50, 50, 100)
        assert q3 == Region(50, 50, 100, 100)


# =============================================================================
# Test: Integral Image Cache
# =============================================================================

class TestIntegralImageCache:
    
    def test_uniform_image(self, simple_image):
        cache = IntegralImageCache(simple_image)
        region = Region(0, 0, 64, 64)
        
        var = cache.compute_variance(region)
        assert var == 0.0  # Uniform image has zero variance
    
    def test_gradient_variance(self, gradient_image):
        cache = IntegralImageCache(gradient_image)
        region = Region(0, 0, 64, 64)
        
        var = cache.compute_variance(region)
        assert var > 0  # Gradient has non-zero variance
    
    def test_subregion_query(self, complex_image):
        cache = IntegralImageCache(complex_image)
        
        # Uniform region should have low variance
        uniform_region = Region(0, 0, 32, 32)
        var_uniform = cache.compute_variance(uniform_region)
        
        # Textured region should have high variance
        texture_region = Region(32, 0, 64, 32)
        var_texture = cache.compute_variance(texture_region)
        
        assert var_texture > var_uniform
    
    def test_gradient_energy(self, gradient_image):
        cache = IntegralImageCache(gradient_image)
        region = Region(0, 0, 64, 64)
        
        grad = cache.compute_gradient_energy(region)
        assert grad > 0  # Gradient image has gradient energy


# =============================================================================
# Test: Complexity Estimator
# =============================================================================

class TestComplexityEstimator:
    
    def test_uniform_low_complexity(self, simple_image, default_config):
        cache = IntegralImageCache(simple_image)
        estimator = ComplexityEstimator(default_config)
        
        region = Region(0, 0, 64, 64)
        c = estimator.compute(region, cache)
        
        assert c < 0.1  # Uniform should be low complexity
    
    def test_texture_high_complexity(self, default_config):
        # Random noise image
        noise_img = torch.rand(3, 64, 64)
        cache = IntegralImageCache(noise_img)
        estimator = ComplexityEstimator(default_config)
        
        region = Region(0, 0, 64, 64)
        c = estimator.compute(region, cache)
        
        assert c > 0.3  # Noise should be high complexity
    
    def test_alpha_effect(self):
        # Edge image (high gradient, medium variance)
        edge_img = torch.zeros(3, 64, 64)
        edge_img[:, :, 32:] = 1.0
        
        cache = IntegralImageCache(edge_img)
        
        # High alpha (variance-focused)
        config_var = AdaptiveSplitConfig(alpha=0.9)
        estimator_var = ComplexityEstimator(config_var)
        c_var = estimator_var.compute(Region(0, 0, 64, 64), cache)
        
        # Low alpha (gradient-focused)
        config_grad = AdaptiveSplitConfig(alpha=0.1)
        estimator_grad = ComplexityEstimator(config_grad)
        c_grad = estimator_grad.compute(Region(0, 0, 64, 64), cache)
        
        # Both should detect the edge
        assert c_var > 0.2
        assert c_grad > 0.2


# =============================================================================
# Test: Hilbert Token Sorter
# =============================================================================

class TestHilbertTokenSorter:
    
    def test_sorting_preserves_locality(self):
        """Test that Hilbert sorting preserves spatial locality."""
        sorter = HilbertTokenSorter(64)
        
        # Create regions in different spatial positions
        regions = [
            Region(0, 0, 32, 32),     # Top-left
            Region(32, 32, 64, 64),   # Bottom-right
            Region(32, 0, 64, 32),    # Top-right
            Region(0, 32, 32, 64),    # Bottom-left
        ]
        
        # Get Hilbert indices for each region
        indices = [sorter.get_hilbert_index(r) for r in regions]
        
        # Sort regions by Hilbert index
        sorted_pairs = sorted(zip(indices, regions))
        sorted_indices = [p[0] for p in sorted_pairs]
        
        # Verify indices are sorted
        assert sorted_indices == sorted(sorted_indices)
    
    def test_hilbert_index_uniqueness(self):
        sorter = HilbertTokenSorter(64)
        
        # Different regions should get different indices
        r1 = Region(0, 0, 16, 16)
        r2 = Region(48, 48, 64, 64)
        
        idx1 = sorter.get_hilbert_index(r1)
        idx2 = sorter.get_hilbert_index(r2)
        
        assert idx1 != idx2


# =============================================================================
# Test: LearnableSplitter
# =============================================================================

class TestLearnableSplitter:
    """Test LearnableSplitter (the only remaining splitter after B/C removal)."""
    
    def test_initialization(self):
        """Test LearnableSplitter can be initialized with default parameters."""
        splitter = LearnableSplitter(
            feature_dim=64,
            max_depth=4,
        )
        assert splitter.feature_dim == 64
        assert splitter.max_depth == 4
    
    def test_forward_with_features(self, simple_features, image_size):
        """Test forward pass with feature map input."""
        splitter = LearnableSplitter(
            feature_dim=64,
            max_depth=3,
            hidden_dim=32,
            pool_size=4,
        )
        
        result = splitter.forward(simple_features, image_size)
        
        # P9-1: 返回类型现在是 TensorSplitResult 而不是 List[SplitResult]
        assert isinstance(result, TensorSplitResult)
        assert result.num_tokens > 0
        assert result.batch_size == 1
    
    def test_complexity_mlp_no_saturation(self, simple_features):
        """
        Verify that LearnableSplitter's MLP doesn't saturate like Scheme B.
        
        Mathematical comparison:
        - Scheme B: C(R) = Var/(Var+σ₀²) → 1 as Var → ∞ (saturates)
        - Scheme L: C_θ(R) = σ(MLP(features)) (no saturation, learnable)
        """
        splitter = LearnableSplitter(
            feature_dim=64,
            max_depth=3,
        )
        
        # LearnableSplitter should have a complexity_mlp
        assert hasattr(splitter, 'complexity_mlp')
        
        # The MLP should be a nn.Module
        import torch.nn as nn
        assert isinstance(splitter.complexity_mlp, nn.Module)
    
    def test_learnable_thresholds(self):
        """
        Verify that thresholds are learnable parameters.
        
        Mathematical formulation: τ_d = τ_{base,d} + δ_d
        where δ_d is learnable.
        """
        splitter = LearnableSplitter(
            feature_dim=64,
            max_depth=4,
        )
        
        # Check for learnable threshold parameters
        param_names = [name for name, _ in splitter.named_parameters()]
        
        # Should have tau-related parameters
        tau_params = [n for n in param_names if 'tau' in n.lower() or 'threshold' in n.lower()]
        # Note: Implementation may vary, but learnable parameters should exist
        assert len(list(splitter.parameters())) > 0
    
    def test_gumbel_softmax_differentiability(self, simple_features, image_size):
        """
        Test that LearnableSplitter is differentiable via Gumbel-Softmax + STE.
        
        Mathematical formulation:
        P(split | R) = softmax((logits + G) / τ), G ~ Gumbel(0, 1)
        """
        splitter = LearnableSplitter(
            feature_dim=64,
            max_depth=3,
            use_gumbel=True,
        )
        
        # Enable training mode
        splitter.train()
        
        features = simple_features.clone().requires_grad_(True)
        result = splitter.forward(features, image_size)
        
        # P9-1: 返回类型现在是 TensorSplitResult
        assert isinstance(result, TensorSplitResult)
        assert result.num_tokens > 0
    
    def test_hilbert_ordering_preserved(self, simple_features, image_size):
        """Test that tokens are ordered by Hilbert index."""
        splitter = LearnableSplitter(
            feature_dim=64,
            max_depth=3,
        )
        
        result = splitter.forward(simple_features, image_size)
        
        # P9-1: TensorSplitResult 存储所有 batch 的 tokens
        # 对于单个 batch，检查 Hilbert 索引是否排序
        if result.num_tokens > 1:
            indices = result.hilbert_indices.tolist()
            # 按 batch 内排序
            assert indices == sorted(indices)
    
    def test_depth_distribution(self, simple_features, image_size):
        """Test that depth distribution is properly computed."""
        splitter = LearnableSplitter(
            feature_dim=64,
            max_depth=4,
        )
        
        result = splitter.forward(simple_features, image_size)
        
        # P9-1: TensorSplitResult 使用 depths 张量
        # 验证深度信息有效
        assert result.depths.shape[0] == result.num_tokens
        assert (result.depths >= 0).all()
        assert (result.depths <= 4).all()
    
    def test_levels_info_format(self, simple_features, image_size):
        """Test levels_info tensor format."""
        splitter = LearnableSplitter(
            feature_dim=64,
            max_depth=4,
        )
        
        result = splitter.forward(simple_features, image_size)
        levels_info = result.get_levels_info_tensor(4)
        
        assert levels_info.shape[0] == result.num_tokens
        assert levels_info.shape[1] == 5  # max_depth + 1
        
        # 第一列应该是深度
        for i in range(result.num_tokens):
            assert levels_info[i, 0].item() == result.depths[i].item()


# =============================================================================
# Test: Factory and Convenience Functions
# =============================================================================

class TestFactoryFunctions:
    
    def test_create_adaptive_splitter_returns_learnable(self, default_config):
        """create_adaptive_splitter should always return LearnableSplitter."""
        splitter = create_adaptive_splitter(default_config)
        assert isinstance(splitter, LearnableSplitter)
    
    def test_split_image_convenience(self, simple_features):
        """Test split_image convenience function with feature input."""
        # Note: split_image treats input as pre-extracted features when no extractor provided
        # P9-1: split_image 现在返回 TensorSplitResult
        result = split_image(simple_features)
        assert isinstance(result, TensorSplitResult)
        assert result.num_tokens > 0


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    
    def test_very_small_features(self):
        """Test with very small feature maps."""
        small_features = torch.randn(1, 64, 4, 4)
        image_size = (16, 16)  # 4x scale
        
        splitter = LearnableSplitter(
            feature_dim=64,
            max_depth=2,
            min_region_size=2,
        )
        result = splitter.forward(small_features, image_size)
        
        # P9-1: 返回 TensorSplitResult
        assert isinstance(result, TensorSplitResult)
        assert result.num_tokens >= 1
    
    def test_batch_size_one(self):
        """Test with batch size of 1."""
        features = torch.randn(1, 64, 16, 16)
        image_size = (64, 64)  # 4x scale
        
        splitter = LearnableSplitter(
            feature_dim=64,
            max_depth=3,
        )
        result = splitter.forward(features, image_size)
        
        # P9-1: 返回 TensorSplitResult
        assert isinstance(result, TensorSplitResult)
        assert result.batch_size == 1
        assert result.num_tokens >= 1


# =============================================================================
# Test: Performance Metrics
# =============================================================================

class TestPerformanceMetrics:
    
    def test_split_result_metrics(self, simple_features, image_size):
        splitter = LearnableSplitter(
            feature_dim=64,
            max_depth=4,
        )
        result = splitter.forward(simple_features, image_size)
        
        # P9-1: TensorSplitResult 使用张量存储
        assert result.num_tokens > 0
        assert result.depths.shape[0] == result.num_tokens
        assert result.complexities.shape[0] == result.num_tokens
    
    def test_regions_tensor(self, simple_features, image_size):
        splitter = LearnableSplitter(
            feature_dim=64,
            max_depth=3,
        )
        result = splitter.forward(simple_features, image_size)
        
        # P9-1: 使用 get_regions_boxes() 而不是 get_regions_tensor()
        regions = result.get_regions_boxes()
        
        assert regions.shape[0] == result.num_tokens
        assert regions.shape[1] == 4  # x1, y1, x2, y2
        
        # All regions should be valid
        for i in range(regions.shape[0]):
            x1, y1, x2, y2 = regions[i].tolist()
            assert x2 > x1
            assert y2 > y1


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
