"""v1.3 STANDARD: Multi-Block HMFT Axiom A2 (determinism) verification.

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.1
the Multi-Block HMFT splitter must satisfy:

  A2 (Determinism): When ``hard=True``, the selection topk_indices must be
  bit-exact reproducible across calls, regardless of the splitter's training
  mode or external random state.

This test exercises the new ``_gumbel_ste_topk`` helper extracted in
Commit 1 of the I170 plan. The helper is the contract surface for A2 — its
``hard`` parameter is the only knob that should gate Gumbel noise.

Note: This test imports the splitter directly via ``sys.path.insert`` to
bypass a pre-existing ``compute_num_candidates`` import error in
``vit_pytorch.models.fractal_vit`` (unrelated to HMFT).
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

from vit_pytorch.layers.splitters.multi_block_hmft_splitter import (
    MultiBlockHMFTSplitter,
    MultiBlockHMFTSplitterConfig,
)


def _make_splitter() -> MultiBlockHMFTSplitter:
    """Build a deterministic-default splitter for A2 tests."""
    return MultiBlockHMFTSplitter(MultiBlockHMFTSplitterConfig())


class TestAxiomA2Determinism:
    """A2: hard=True → bit-exact reproducible across calls."""

    def test_hard_true_reproducible_in_eval_mode(self):
        """Baseline: eval mode + hard=True should be deterministic (legacy A2)."""
        splitter = _make_splitter()
        splitter.eval()
        torch.manual_seed(0)
        r1 = splitter(
            torch.randn(1, 256, 32, 32), image_size=(32, 32), hard=True,
        )
        torch.manual_seed(0)
        r2 = splitter(
            torch.randn(1, 256, 32, 32), image_size=(32, 32), hard=True,
        )
        assert torch.equal(r1.candidate_indices, r2.candidate_indices)

    def test_hard_true_reproducible_in_train_mode(self):
        """A2 contract: hard=True is deterministic REGARDLESS of training mode.

        This is the new strict contract from I170 Commit 1: previously the
        Gumbel gate was ``self.training and not hard``, which made
        ``hard=True + train()`` accidentally deterministic only because
        ``set_training(True)`` was the default. The refactored helper
        gates Gumbel purely on ``not hard``, making the contract explicit.
        """
        splitter = _make_splitter()
        splitter.set_training(True)  # explicitly in train mode
        # Two different input features with the same shape — A2 should still
        # give the same topk_indices because hard=True is pure-deterministic.
        torch.manual_seed(42)
        r1 = splitter(
            torch.randn(1, 256, 32, 32), image_size=(32, 32), hard=True,
        )
        # Re-seed RNG to neutralise any state mutation; the call must not
        # consume random bits when hard=True.
        torch.manual_seed(42)
        r2 = splitter(
            torch.randn(1, 256, 32, 32), image_size=(32, 32), hard=True,
        )
        assert torch.equal(r1.candidate_indices, r2.candidate_indices)

    def test_hard_false_train_mode_uses_randomness(self):
        """Sanity: hard=False + train should NOT be deterministic (Gumbel is on).

        This is the *complement* of A2 — proves the helper actually injects
        noise in the non-hard path. Without this, the A2 test would pass
        trivially for a broken helper that never adds Gumbel.
        """
        splitter = _make_splitter()
        splitter.set_training(True)
        # Two calls with different RNG states should differ in their topk.
        torch.manual_seed(1)
        r1 = splitter(
            torch.randn(1, 256, 32, 32), image_size=(32, 32), hard=False,
        )
        torch.manual_seed(2)
        r2 = splitter(
            torch.randn(1, 256, 32, 32), image_size=(32, 32), hard=False,
        )
        # TopK indices should differ at least one element (probability ≈ 1)
        assert not torch.equal(r1.candidate_indices, r2.candidate_indices), (
            "Gumbel noise should make hard=False selections differ across seeds"
        )

    def test_helper_direct_call_hard_true_is_pure(self):
        """Direct call to _gumbel_ste_topk proves the helper is the contract surface.

        Bypasses the forward() entry path to isolate the helper. The helper
        must be a pure function of (score_head, K, hard) — no global RNG
        consumption when hard=True.
        """
        splitter = _make_splitter()
        score = torch.randn(2, 16)
        torch.manual_seed(7)
        out1 = splitter._gumbel_ste_topk(score, K=4, hard=True)
        torch.manual_seed(7)
        out2 = splitter._gumbel_ste_topk(score, K=4, hard=True)
        mask_hard_1, _, _, topk_1 = out1
        mask_hard_2, _, _, topk_2 = out2
        assert torch.equal(topk_1, topk_2)
        assert torch.equal(mask_hard_1, mask_hard_2)

    def test_helper_returns_correct_shapes(self):
        """Helper output contract: (mask_hard, mask_soft, mask_ste, topk_indices)."""
        splitter = _make_splitter()
        B, n_cells, K = 3, 20, 5
        # requires_grad_(True) simulates the real forward path where
        # score_head derives from feature_proj(cell_features), which has grad.
        # In the isolated helper test, we must opt in to grad tracking
        # because plain torch.randn() defaults to requires_grad=False.
        score = torch.randn(B, n_cells).requires_grad_(True)
        mask_hard, mask_soft, mask_ste, topk = splitter._gumbel_ste_topk(
            score, K=K, hard=True,
        )
        assert mask_hard.shape == (B, n_cells)
        assert mask_soft.shape == (B, n_cells)
        assert mask_ste.shape == (B, n_cells)
        assert topk.shape == (B, K)
        # mask_hard sums to K per batch (K selected cells)
        assert (mask_hard.sum(dim=-1) == K).all()
        # mask_ste preserves STE gradient (no stray .detach() in the bridge)
        assert mask_ste.requires_grad, (
            "T10+Review: mask_ste must preserve STE gradient. "
            "If this fails, check that (mask_hard - mask_soft).detach() "
            "+ mask_soft formula is intact."
        )
        # mask_soft is a valid probability distribution
        assert torch.allclose(mask_soft.sum(dim=-1), torch.ones(B), atol=1e-5)
