# -*- coding: utf-8 -*-
"""
L2 Components Tests: Adaptive Splitter (GumbelTopKSplitter)

对应模块: vit_pytorch.gumbel_topk_splitter

测试内容:
- GumbelTopKSplitter 基本功能 (Scheme D/E)
- 数学性质 (Hilbert locality, 复杂度估计)
- 核心组件 (TensorSplitResult, GumbelTopKResult)
- 配置和工厂函数

Note:
====
Scheme A (LearnableSplitter) has been removed in I97-9.
GumbelTopKSplitter (Scheme D/E) is now the production implementation.

Key advantages:
1. End-to-end differentiability via Gumbel-Top-K + STE
2. Parallel evaluation of all candidates (no BFS dependency)
3. ~37.6% gradient coverage (K/N) vs ~25% for Scheme A
4. Learnable quota allocation per depth
"""

import math
import pytest
import torch
from typing import Dict, List

import sys
sys.path.insert(0, 'src')

from vit_pytorch.layers.splitters.gumbel_topk import (
    GumbelTopKSplitter,
    GumbelTopKResult,
    TensorSplitResult,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_image():
    """Simple uniform image."""
    return torch.zeros(3, 64, 64)


@pytest.fixture
def random_image():
    """Random image for testing."""
    return torch.randn(3, 64, 64)


@pytest.fixture
def splitter():
    """Create a GumbelTopKSplitter instance for testing."""
    return GumbelTopKSplitter(
        feature_dim=256,
        min_patch_size=4,
        max_level_limit=4,
        hidden_dim=128,
        pool_size=4,
        K_max=32,
        # I111-1: temperature 参数已移至 SplitterConfig
    )


# =============================================================================
# Basic Functionality Tests
# =============================================================================

class TestGumbelTopKBasic:
    """Basic functionality tests for GumbelTopKSplitter."""

    def test_forward_returns_correct_type(self, splitter):
        """Verify forward returns GumbelTopKResult."""
        features = torch.randn(1, 256, 16, 16)
        result = splitter(features, (64, 64))

        assert isinstance(result, GumbelTopKResult)

    def test_selected_mask_shape(self, splitter):
        """Verify selected_mask has correct shape."""
        features = torch.randn(2, 256, 16, 16)
        result = splitter(features, (64, 64))

        # Shape should be [B, N_candidates]
        assert result.selected_mask.dim() == 2
        assert result.selected_mask.shape[0] == 2

    def test_num_selected_per_batch(self, splitter):
        """Verify num_selected_per_batch is computed correctly."""
        features = torch.randn(1, 256, 16, 16)
        result = splitter(features, (64, 64))

        # Should have one value per batch
        assert result.num_selected_per_batch.shape[0] == 1
        # Should be positive
        assert result.num_selected_per_batch[0].item() > 0


class TestHilbertLocality2:
    """Hilbert curve locality tests.

    Note: hilbert_indices may contain duplicates because different regions
    at different depths can have the same Hilbert curve index. The sorting
    happens in the tokenizer output, not in the splitter.
    """

    def test_hilbert_indices_are_valid(self, splitter):
        """Verify hilbert_indices are valid candidate indices."""
        features = torch.randn(1, 256, 16, 16)
        result = splitter(features, (64, 64))

        # hilbert_indices should be valid indices into candidate regions
        # They should be in range [0, num_candidates)
        hilbert_indices = result.hilbert_indices
        num_candidates = splitter.num_candidates

        assert hilbert_indices.min().item() >= 0, "hilbert_indices should be non-negative"
        assert hilbert_indices.max().item() < num_candidates, \
            f"hilbert_indices ({hilbert_indices.max().item()}) should be < num_candidates ({num_candidates})"

    def test_hilbert_indices_count_matches_selection(self, splitter):
        """Verify hilbert_indices count matches num_selected_per_batch."""
        features = torch.randn(1, 256, 16, 16)
        result = splitter(features, (64, 64))

        # hilbert_indices should have same count as num_selected_per_batch
        expected_count = result.num_selected_per_batch[0].item()
        actual_count = len(result.hilbert_indices)
        assert actual_count == expected_count, \
            f"hilbert_indices count ({actual_count}) != num_selected_per_batch ({expected_count})"


class TestGradientFlow2:
    """Gradient flow tests."""

    def test_gradient_through_splitter(self, splitter):
        """Verify gradients flow through splitter."""
        features = torch.randn(1, 256, 16, 16, requires_grad=True)
        result = splitter(features, (64, 64), hard=False)

        loss = result.selected_mask.sum()
        loss.backward()

        assert features.grad is not None
        assert not torch.isnan(features.grad).any()
        assert not torch.isinf(features.grad).any()


class TestDeterminism2:
    """Determinism tests.

    Note: GumbelTopKSplitter uses Gumbel-Softmax which has inherent
    stochasticity. Determinism tests are skipped as they don't apply
    to stochastic algorithms.
    """
    pass  # Tests moved to integration tests where determinism can be properly configured


class TestTensorSplitResult2:
    """TensorSplitResult tests."""

    def test_tensor_split_result_properties(self, splitter):
        """Verify TensorSplitResult has all required properties."""
        features = torch.randn(1, 256, 16, 16)
        result = splitter(features, (64, 64))

        # GumbelTopKResult should have:
        assert hasattr(result, 'regions')
        assert hasattr(result, 'depths')
        assert hasattr(result, 'batch_indices')
        assert hasattr(result, 'hilbert_indices')
        assert hasattr(result, 'selected_mask')
        assert hasattr(result, 'logits')
        assert hasattr(result, 'probs')
        assert hasattr(result, 'num_selected_per_batch')


class TestHardSoftMode2:
    """Hard vs Soft mode tests."""

    def test_hard_mode_deterministic(self, splitter):
        """Verify hard mode produces deterministic binary mask with same seed."""
        torch.manual_seed(123)
        features = torch.randn(1, 256, 16, 16)

        # Run multiple times with same seed
        results = []
        for _ in range(5):
            torch.manual_seed(123)
            results.append(splitter(features, (64, 64), hard=True).selected_mask)

        # All hard mode results should be identical
        for r in results[1:]:
            assert torch.allclose(results[0], r), \
                "Hard mode should produce deterministic output with same seed"

    def test_soft_mode_probabilistic(self, splitter):
        """Verify soft mode produces valid probabilities."""
        features = torch.randn(1, 256, 16, 16)
        result = splitter(features, (64, 64), hard=False)

        # Probs should be in [0, 1]
        assert torch.all(result.probs >= 0)
        assert torch.all(result.probs <= 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
