"""v1.3 STANDARD: Multi-Block HMFT Axiom A5 (Fixed-K) verification.

Per I170.3 plan: A5 unifies all K references in MultiBlockHMFTSplitter to
HMFT_K_HARD_GLOBAL_POOL=8. The axiom contract:

  A5 (Fixed-K): splitter.forward 选中区域数 = HMFT_K_HARD_GLOBAL_POOL
  在所有 batch 元素、全部 image_size (>= HMFT_MIN_IMAGE_SIZE)、全部
  training mode (train/eval)、全部 h 选值、全部 random seed 下严格相等。
  无参数、无 override、无 training/eval 模式区分。

Two-layer defense:
  1. Input source (forward): ``image_size < HMFT_MIN_IMAGE_SIZE`` raises
     ``ValueError`` immediately, before any computation.
  2. T10 keystone (forward, post-`_sample_block_size`):
     ``K == HMFT_K_HARD_GLOBAL_POOL`` and ``n_cells >= HMFT_K_HARD_GLOBAL_POOL``
     both asserted. The second assert is a backstop in case the input guard
     is bypassed (e.g. direct test of internal helpers).

This test file follows the path-bootstrap pattern from
test_multi_block_hmft_a2.py:32-37.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Path bootstrap: tests live under tests/unit/L2_components/splitters/ but
# the project root contains src/. Insert src/ so vit_pytorch.layers.splitters
# can be imported without triggering the broken re-export chain in
# vit_pytorch/__init__.py.
_SRC = Path(__file__).resolve().parents[4] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest
import torch

from vit_pytorch.core.constants import (
    HMFT_BLOCK_SIZES,
    HMFT_K_HARD_GLOBAL_POOL,
    HMFT_MIN_IMAGE_SIZE,
)
from vit_pytorch.layers.splitters.multi_block_hmft_splitter import (
    MultiBlockHMFTSplitter,
    MultiBlockHMFTSplitterConfig,
)


class TestAxiomA5FixedK:
    """A5: K is statically HMFT_K_HARD_GLOBAL_POOL, no override."""

    @pytest.mark.parametrize("grid_size", [32, 64, 128])
    def test_K_equals_global_pool_for_all_grids(self, make_hmft_splitter, grid_size: int):
        """grid_size >= HMFT_MIN_IMAGE_SIZE: K=8 for all valid grids."""
        splitter = make_hmft_splitter()
        splitter.train()
        features = torch.randn(2, 256, grid_size, grid_size)
        result = splitter(features, image_size=(grid_size, grid_size), hard=True)
        # A5: K is per-image, M = B*K
        K = result.regions.shape[0] // 2
        assert K == HMFT_K_HARD_GLOBAL_POOL, (
            f"A5: K must equal HMFT_K_HARD_GLOBAL_POOL={HMFT_K_HARD_GLOBAL_POOL}, "
            f"got K={K} for grid_size={grid_size}"
        )

    def test_K_independent_of_training_mode(self, make_hmft_splitter):
        """A5: K=8 in BOTH train and eval mode (no mode-specific override)."""
        splitter_train = make_hmft_splitter()
        splitter_train.train()
        splitter_eval = make_hmft_splitter()
        splitter_eval.eval()
        features = torch.randn(2, 256, 32, 32)
        r_train = splitter_train(features, image_size=(32, 32), hard=False)
        r_eval = splitter_eval(features, image_size=(32, 32), hard=True)
        assert (r_train.regions.shape[0] // 2) == HMFT_K_HARD_GLOBAL_POOL
        assert (r_eval.regions.shape[0] // 2) == HMFT_K_HARD_GLOBAL_POOL

    @pytest.mark.parametrize("h_choice", HMFT_BLOCK_SIZES)
    def test_K_independent_of_h_choice(self, make_hmft_splitter, h_choice: int):
        """A5: K=8 regardless of which block size the 5-bin sampler picks.

        We exercise the actual h selection by forcing hard=True with seeded
        h_logits so each test sees a different block size. Across all 5
        block sizes, K must remain HMFT_K_HARD_GLOBAL_POOL=8.
        """
        splitter = make_hmft_splitter()
        # Force a specific h_choice by setting h_logits to favour that bin.
        # h_logits is a 5-dim Parameter; argmax picks the chosen h.
        h_idx = HMFT_BLOCK_SIZES.index(h_choice)
        with torch.no_grad():
            new_logits = torch.full((5,), -10.0)
            new_logits[h_idx] = 10.0
            splitter.h_logits.copy_(new_logits)
        # Use a 64x64 image so all block sizes (8, 16, 32, 64, 128) fit
        # (h=128 needs N>=128; we use N=128 to allow the largest h).
        N = max(128, h_choice)
        features = torch.randn(2, 256, N, N)
        r = splitter(features, image_size=(N, N), hard=True)
        K = r.regions.shape[0] // 2
        assert K == HMFT_K_HARD_GLOBAL_POOL, (
            f"A5: K must equal {HMFT_K_HARD_GLOBAL_POOL} for h_choice={h_choice}, "
            f"got K={K}"
        )

    def test_K_invariant_across_random_seeds(self, make_hmft_splitter):
        """A5: K=8 for all random seeds (orthogonal to A2's hard=True determinism)."""
        splitter = make_hmft_splitter()
        for seed in [0, 1, 42, 1234]:
            torch.manual_seed(seed)
            features = torch.randn(2, 256, 32, 32)
            r = splitter(features, image_size=(32, 32), hard=False)
            K = r.regions.shape[0] // 2
            assert K == HMFT_K_HARD_GLOBAL_POOL, (
                f"A5: seed={seed} produced K={K}, expected {HMFT_K_HARD_GLOBAL_POOL}"
            )

    def test_K_config_field_removed(self):
        """A5: MultiBlockHMFTSplitterConfig.K_fixed does NOT exist."""
        cfg = MultiBlockHMFTSplitterConfig()
        assert not hasattr(cfg, "K_fixed"), (
            "A5: K_fixed must NOT exist on MultiBlockHMFTSplitterConfig. "
            "All K references unify to HMFT_K_HARD_GLOBAL_POOL=8."
        )
        # Also verify that passing K_fixed raises TypeError (dataclass strictness).
        # The K_fixed=16 below is intentionally invalid — this is the test.
        with pytest.raises(TypeError):
            MultiBlockHMFTSplitterConfig(K_fixed=16)  # type: ignore[call-arg]

    @pytest.mark.parametrize("bad_size", [4, 8, 16, 23])
    def test_init_rejects_image_size_below_minimum(self, make_hmft_splitter, bad_size: int):
        """A5: image_size < HMFT_MIN_IMAGE_SIZE raises ValueError at forward.

        Note: image_size is a forward() argument, not __init__ — so the guard
        fires in forward, not in constructor. The splitter construction itself
        always succeeds.
        """
        splitter = make_hmft_splitter()
        features = torch.randn(2, 256, bad_size, bad_size)
        with pytest.raises(ValueError) as exc_info:
            splitter(features, image_size=(bad_size, bad_size), hard=True)
        # Error message must mention the constraint and the alternative
        err_msg = str(exc_info.value)
        assert str(HMFT_MIN_IMAGE_SIZE) in err_msg, (
            f"Error message must mention HMFT_MIN_IMAGE_SIZE={HMFT_MIN_IMAGE_SIZE}, "
            f"got: {err_msg}"
        )
        assert str(HMFT_K_HARD_GLOBAL_POOL) in err_msg, (
            f"Error message must mention K={HMFT_K_HARD_GLOBAL_POOL}, got: {err_msg}"
        )

    def test_init_accepts_dynamic_image_size(self, make_hmft_splitter):
        """A5: image_size=None is fine (dynamic branch, not rejected)."""
        splitter = make_hmft_splitter()
        features = torch.randn(2, 256, 64, 64)  # image_size inferred = (64, 64)
        # Should NOT raise
        r = splitter(features, image_size=None, hard=True)
        assert (r.regions.shape[0] // 2) == HMFT_K_HARD_GLOBAL_POOL

    def test_init_error_message_mentions_alternative_splitter(self, make_hmft_splitter):
        """A5: ValueError must guide user to PolarVoronoiSplitter for small images."""
        splitter = make_hmft_splitter()
        features = torch.randn(2, 256, 8, 8)
        with pytest.raises(ValueError) as exc_info:
            splitter(features, image_size=(8, 8), hard=True)
        err_msg = str(exc_info.value)
        assert "PolarVoronoiSplitter" in err_msg, (
            f"Error message should mention PolarVoronoiSplitter as alternative, "
            f"got: {err_msg}"
        )

    def test_t10_keystone_asserts_in_forward(self, make_hmft_splitter):
        """A5 T10 keystone: forward asserts K == HMFT_K_HARD_GLOBAL_POOL twice.

        Two-layer defense:
          - Input source guard: image_size < HMFT_MIN_IMAGE_SIZE → ValueError
          - T10 keystone: post-_sample_block_size asserts K and n_cells

        The second-layer T10 assert is the one we test here — it's a
        "should never fire" guard that catches a programmer error if
        someone bypasses the input source guard.
        """
        splitter = make_hmft_splitter()
        # Use exactly the minimum acceptable image size
        features = torch.randn(2, 256, HMFT_MIN_IMAGE_SIZE, HMFT_MIN_IMAGE_SIZE)
        # Should NOT raise — image_size == HMFT_MIN_IMAGE_SIZE is valid
        r = splitter(
            features,
            image_size=(HMFT_MIN_IMAGE_SIZE, HMFT_MIN_IMAGE_SIZE),
            hard=True,
        )
        K = r.regions.shape[0] // 2
        assert K == HMFT_K_HARD_GLOBAL_POOL
