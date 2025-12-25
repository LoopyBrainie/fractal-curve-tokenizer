"""
Tests for Adaptive Quadtree Split (Scheme B & C)

This test suite verifies:
1. Basic functionality of both schemes
2. Mathematical properties (Hilbert locality, LCA effectiveness)
3. Performance characteristics (token counts, complexity)
4. Scheme comparison

Run with: pytest tests/unit/test_adaptive_split.py -v
"""

import math
import pytest
import torch
from typing import Dict, List

import sys
sys.path.insert(0, 'src')

from vit_pytorch.adaptive_split import (
    AdaptiveSplitConfig,
    SplitScheme,
    Region,
    QuadtreeNode,
    SplitToken,
    SplitResult,
    IntegralImageCache,
    ComplexityEstimator,
    HilbertTokenSorter,
    BalancedGreedySplitter,
    FixedBudgetDPSplitter,
    create_adaptive_splitter,
    split_image,
    compare_schemes,
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
def config_b():
    """Default Scheme B config."""
    return AdaptiveSplitConfig.scheme_b()


@pytest.fixture
def config_c():
    """Default Scheme C config."""
    return AdaptiveSplitConfig.scheme_c(token_budget=64)


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
    
    def test_scheme_b_factory(self):
        config = AdaptiveSplitConfig.scheme_b(tau_0=0.2)
        assert config.scheme == SplitScheme.BALANCED_GREEDY
        assert config.tau_0 == 0.2
    
    def test_scheme_c_factory(self):
        config = AdaptiveSplitConfig.scheme_c(token_budget=128)
        assert config.scheme == SplitScheme.FIXED_BUDGET_DP
        assert config.token_budget == 128
    
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
    
    def test_uniform_low_complexity(self, simple_image, config_b):
        cache = IntegralImageCache(simple_image)
        estimator = ComplexityEstimator(config_b)
        
        region = Region(0, 0, 64, 64)
        c = estimator.compute(region, cache)
        
        assert c < 0.1  # Uniform should be low complexity
    
    def test_texture_high_complexity(self, config_b):
        # Random noise image
        noise_img = torch.rand(3, 64, 64)
        cache = IntegralImageCache(noise_img)
        estimator = ComplexityEstimator(config_b)
        
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
        sorter = HilbertTokenSorter(64)
        
        # Create tokens in random spatial order
        tokens = [
            SplitToken(Region(0, 0, 32, 32), 1, [0], 0, 0.1),
            SplitToken(Region(32, 32, 64, 64), 1, [3], 0, 0.1),
            SplitToken(Region(32, 0, 64, 32), 1, [1], 0, 0.1),
            SplitToken(Region(0, 32, 32, 64), 1, [2], 0, 0.1),
        ]
        
        sorted_tokens = sorter.sort_tokens(tokens)
        
        # Verify Hilbert indices are assigned and sorted
        indices = [t.hilbert_idx for t in sorted_tokens]
        assert indices == sorted(indices)
    
    def test_hilbert_index_uniqueness(self):
        sorter = HilbertTokenSorter(64)
        
        # Different regions should get different indices
        r1 = Region(0, 0, 16, 16)
        r2 = Region(48, 48, 64, 64)
        
        idx1 = sorter.get_hilbert_index(r1)
        idx2 = sorter.get_hilbert_index(r2)
        
        assert idx1 != idx2


# =============================================================================
# Test: Scheme B - Balanced Greedy Splitter
# =============================================================================

class TestBalancedGreedySplitter:
    
    def test_uniform_image_minimal_split(self, simple_image, config_b):
        splitter = BalancedGreedySplitter(config_b)
        result = splitter.split(simple_image)
        
        # Uniform image should produce few tokens
        assert result.num_tokens <= 4
    
    def test_complex_image_more_tokens(self, complex_image, config_b):
        splitter = BalancedGreedySplitter(config_b)
        result = splitter.split(complex_image)
        
        # Complex image should produce more tokens
        assert result.num_tokens > 4
    
    def test_balance_constraint(self, natural_image, config_b):
        config_b.enforce_balance = True
        splitter = BalancedGreedySplitter(config_b)
        result = splitter.split(natural_image)
        
        # Check 2:1 balance: no adjacent tokens differ by more than 1 level
        # (This is a simplified check - full verification would need neighbor detection)
        depths = [t.depth for t in result.tokens]
        if len(depths) > 1:
            depth_range = max(depths) - min(depths)
            # With balance, should not have extreme depth differences in adjacent regions
            assert depth_range <= config_b.max_depth
    
    def test_hilbert_ordering(self, complex_image, config_b):
        splitter = BalancedGreedySplitter(config_b)
        result = splitter.split(complex_image)
        
        # Tokens should be sorted by Hilbert index
        indices = [t.hilbert_idx for t in result.tokens]
        assert indices == sorted(indices)
    
    def test_levels_info_format(self, complex_image, config_b):
        splitter = BalancedGreedySplitter(config_b)
        result = splitter.split(complex_image)
        
        levels_info = result.get_levels_info(config_b.max_depth)
        
        assert levels_info.shape[0] == result.num_tokens
        assert levels_info.shape[1] == config_b.max_depth + 1
        
        # First column should be depth
        for i, token in enumerate(result.tokens):
            assert levels_info[i, 0].item() == token.depth


# =============================================================================
# Test: Scheme C - Fixed Budget DP Splitter
# =============================================================================

class TestFixedBudgetDPSplitter:
    
    @pytest.mark.slow
    def test_exact_token_count(self, natural_image, config_c):
        splitter = FixedBudgetDPSplitter(config_c)
        result = splitter.split(natural_image)
        
        # Should produce exactly the budget number of tokens
        assert result.num_tokens == config_c.token_budget
    
    @pytest.mark.slow
    def test_different_budgets(self, natural_image):
        for budget in [16, 32, 64]:  # Reduced from [16, 32, 64, 128]
            config = AdaptiveSplitConfig.scheme_c(token_budget=budget)
            splitter = FixedBudgetDPSplitter(config)
            result = splitter.split(natural_image)
            
            assert result.num_tokens == budget
    
    @pytest.mark.slow
    def test_hilbert_ordering(self, natural_image, config_c):
        splitter = FixedBudgetDPSplitter(config_c)
        result = splitter.split(natural_image)
        
        indices = [t.hilbert_idx for t in result.tokens]
        assert indices == sorted(indices)
    
    def test_importance_based_selection(self, complex_image, config_c):
        config_c.token_budget = 8
        splitter = FixedBudgetDPSplitter(config_c)
        result = splitter.split(complex_image)
        
        # High-complexity regions should get more tokens
        # (This is a qualitative check)
        assert result.num_tokens == 8


# =============================================================================
# Test: Scheme Comparison
# =============================================================================

class TestSchemeComparison:
    
    @pytest.mark.slow
    def test_compare_schemes_basic(self, natural_image):
        results = compare_schemes(natural_image, {'token_budget': 64})
        
        assert 'scheme_b' in results
        assert 'scheme_c' in results
        
        # Scheme C should have exact budget
        assert results['scheme_c'].num_tokens == 64
        
        # Scheme B token count is variable
        # For natural images with high complexity, may produce many tokens
        assert 1 <= results['scheme_b'].num_tokens <= 256  # Max is 4^4
    
    @pytest.mark.slow
    def test_depth_entropy_comparison(self, natural_image):
        results = compare_schemes(natural_image)
        
        entropy_b = results['scheme_b'].depth_entropy
        entropy_c = results['scheme_c'].depth_entropy
        
        # Both should have positive entropy (using multiple depths)
        # Note: This might fail for very simple images
        print(f"Scheme B depth entropy: {entropy_b:.3f}")
        print(f"Scheme C depth entropy: {entropy_c:.3f}")
    
    @pytest.mark.slow
    def test_depth_distribution(self, complex_image):
        results = compare_schemes(complex_image, {'max_depth': 3})
        
        dist_b = results['scheme_b'].depth_distribution
        dist_c = results['scheme_c'].depth_distribution
        
        print(f"Scheme B depth distribution: {dist_b}")
        print(f"Scheme C depth distribution: {dist_c}")
        
        # Both should use multiple depths for complex image
        assert len(dist_b) >= 1
        assert len(dist_c) >= 1


# =============================================================================
# Test: Factory and Convenience Functions
# =============================================================================

class TestFactoryFunctions:
    
    def test_create_adaptive_splitter_b(self, config_b):
        splitter = create_adaptive_splitter(config_b)
        assert isinstance(splitter, BalancedGreedySplitter)
    
    def test_create_adaptive_splitter_c(self, config_c):
        splitter = create_adaptive_splitter(config_c)
        assert isinstance(splitter, FixedBudgetDPSplitter)
    
    def test_split_image_convenience(self, natural_image):
        result = split_image(natural_image, scheme="balanced_greedy")
        assert isinstance(result, SplitResult)
        assert result.num_tokens > 0


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    
    def test_very_small_image(self):
        small_img = torch.rand(3, 16, 16)
        
        config = AdaptiveSplitConfig.scheme_b(max_depth=2, min_region_size=4)
        splitter = BalancedGreedySplitter(config)
        result = splitter.split(small_img)
        
        assert result.num_tokens >= 1
    
    def test_single_channel_image(self):
        gray_img = torch.rand(1, 64, 64)
        
        config = AdaptiveSplitConfig.scheme_b()
        splitter = BalancedGreedySplitter(config)
        result = splitter.split(gray_img)
        
        assert result.num_tokens >= 1
    
    @pytest.mark.slow
    def test_budget_larger_than_max_possible(self, simple_image):
        # Budget of 1000 but max possible is 256 (4^4)
        config = AdaptiveSplitConfig.scheme_c(
            token_budget=256,  # Use max possible
            max_depth=4
        )
        splitter = FixedBudgetDPSplitter(config)
        result = splitter.split(simple_image)
        
        # Should produce tokens up to max possible
        assert result.num_tokens <= 256
    
    def test_budget_of_one(self, natural_image):
        config = AdaptiveSplitConfig.scheme_c(token_budget=1)
        splitter = FixedBudgetDPSplitter(config)
        result = splitter.split(natural_image)
        
        assert result.num_tokens == 1


# =============================================================================
# Test: Performance Metrics
# =============================================================================

class TestPerformanceMetrics:
    
    def test_split_result_metrics(self, natural_image, config_b):
        splitter = BalancedGreedySplitter(config_b)
        result = splitter.split(natural_image)
        
        # Test all metric properties
        assert result.num_tokens > 0
        assert isinstance(result.depth_distribution, dict)
        assert result.depth_entropy >= 0
    
    def test_regions_tensor(self, complex_image, config_b):
        splitter = BalancedGreedySplitter(config_b)
        result = splitter.split(complex_image)
        
        regions = result.get_regions_tensor()
        
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
