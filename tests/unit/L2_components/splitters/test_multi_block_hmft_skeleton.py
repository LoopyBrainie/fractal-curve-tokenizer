"""v1.3 STANDARD: Multi-Block HMFT Splitter skeleton tests (B.3).

These are basic structural and forward-pass tests for the HMFT skeleton.
JVP-1 5-grid EAS verification (B.4) lives in test_multi_block_hmft_eas.py.
JVP-5 stress-scale verification (B.6) is in test_multi_block_hmft_poc.py.
"""
from __future__ import annotations

import pytest
import torch

from vit_pytorch.core.constants import HMFT_BLOCK_SIZES, HMFT_K_HARD_GLOBAL_POOL
from vit_pytorch.layers.splitters import (
    MultiBlockHMFTSplitter,
    MultiBlockHMFTSplitterConfig,
)
from vit_pytorch.core.splitter_protocol import validate_core_splitter


class TestMultiBlockHMFTSplitterSkeleton:
    """Skeleton-level tests for MultiBlockHMFTSplitter."""

    def test_hmft_block_sizes_constant(self):
        """The 5-bin block sizes are the design values (Power-of-4 alignment)."""
        assert HMFT_BLOCK_SIZES == (8, 16, 32, 64, 128)
        # Each h must be a power of 4
        for h in HMFT_BLOCK_SIZES:
            assert h & (h - 1) == 0, f"h={h} must be a power of 2"
            assert h % 4 == 0, f"h={h} must be divisible by 4 (Power-of-4)"

    def test_hmft_default_config(self):
        """Default config has the v1.3 STANDARD values."""
        cfg = MultiBlockHMFTSplitterConfig()
        assert cfg.block_sizes == HMFT_BLOCK_SIZES
        assert cfg.G_global_pool == HMFT_K_HARD_GLOBAL_POOL
        assert cfg.K_fixed == 16

    def test_hmft_skeleton_forward_n_64(self):
        """Forward succeeds on N=64 (smallest JVP-1 grid)."""
        splitter = MultiBlockHMFTSplitter()
        splitter.train()
        features = torch.randn(2, 256, 64, 64)
        result = splitter(features, image_size=(64, 64), hard=True)
        # Required SplitResult fields
        assert result.regions.ndim == 2
        assert result.regions.shape[-1] == 4
        assert result.hilbert_indices.ndim == 1
        assert result.batch_indices.ndim == 1
        # K is per-image, so M = B * K
        assert result.regions.shape[0] == 2 * 16

    def test_hmft_skeleton_forward_n_128(self):
        """Forward succeeds on N=128."""
        splitter = MultiBlockHMFTSplitter()
        features = torch.randn(1, 256, 128, 128)
        result = splitter(features, image_size=(128, 128), hard=True)
        assert result.regions.shape[0] == 1 * 16

    def test_hmft_skeleton_h_logits_is_5d(self):
        """The 5-bin h_logits parameter is a 5D learnable tensor."""
        splitter = MultiBlockHMFTSplitter()
        assert splitter.h_logits.shape == (5,)
        assert splitter.h_logits.requires_grad

    def test_hmft_skeleton_protocol_validation(self):
        """validate_core_splitter accepts the HMFT skeleton (no protocol violations)."""
        splitter = MultiBlockHMFTSplitter()
        # Should not raise
        validate_core_splitter(splitter, name="MultiBlockHMFTSplitter")

    def test_hmft_skeleton_diagnostics(self):
        """Diagnostics include h_probs over the 5 bins."""
        splitter = MultiBlockHMFTSplitter()
        diag = splitter.get_diagnostics()
        assert diag["splitter"] == "MultiBlockHMFTSplitter"
        assert len(diag["h_probs"]) == 5
        # h_probs sum to 1 (it's a softmax)
        assert abs(sum(diag["h_probs"]) - 1.0) < 1e-5
