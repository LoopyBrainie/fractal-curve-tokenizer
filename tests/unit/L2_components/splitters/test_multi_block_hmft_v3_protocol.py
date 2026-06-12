"""v1.3 STANDARD: Multi-Block HMFT V3 protocol verification (I170 Commit 3).

Verifies the MultiBlockHMFTSplitter conforms to the V3 tokenizer protocol
that ``HilbertOptimalSplitter`` (H1SS) already satisfies:

  roi_features_raw:  [B, N, d_model]  pre-gather per-candidate pool
  roi_features:      [B, N, hidden_dim] post-projection LayerNorm'd
                     (T10 keystone — keeps feature_proj in autograd graph)
  mask_ste:          [B, N]  Gumbel-STE mask
  candidate_indices: [M]     selected token indices into the candidate pool

Plus two project rules (P0/P1 from review feedback):

  1. P0 STATIC REGISTRATION: feature_proj and roi_norm must be
     registered in __init__, NOT lazily built in forward. The optimizer
     snapshots ``model.parameters()`` at instantiation time, so any
     module created in forward() will never receive gradient updates.

  2. P1 NO POST-MUTATION: External code must not modify
     ``splitter._config.feature_dim`` after instantiation. nn.LayerNorm
     caches shape at init time and will not re-initialise.

This test imports the splitter directly via ``sys.path.insert`` to
bypass a pre-existing ``compute_num_candidates`` import error in
``vit_pytorch.models.fractal_vit`` (unrelated to HMFT).
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[4] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest
import torch

from vit_pytorch.layers.splitters.multi_block_hmft_splitter import (
    MultiBlockHMFTSplitter,
    MultiBlockHMFTSplitterConfig,
)


def _make_splitter(
    feature_dim_in: int = 256, feature_dim: int = 256, hidden_dim: int = 64,
) -> MultiBlockHMFTSplitter:
    return MultiBlockHMFTSplitter(MultiBlockHMFTSplitterConfig(
        feature_dim_in=feature_dim_in,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
    ))


class TestV3ProtocolShapeContract:
    """V3 protocol: shape contract on SplitResult fields."""

    def test_roi_features_raw_shape(self):
        """roi_features_raw must be [B, n_cells, feature_dim_in]."""
        splitter = _make_splitter()
        splitter.eval()
        features = torch.randn(2, 256, 32, 32)
        r = splitter(features, image_size=(32, 32), hard=True)
        assert r.roi_features_raw.dim() == 3
        assert r.roi_features_raw.shape[0] == 2  # B
        # n_cells depends on h chosen; just check it matches mask_ste
        assert r.roi_features_raw.shape[1] == r.mask_ste.shape[1]
        assert r.roi_features_raw.shape[2] == 256  # feature_dim_in

    def test_roi_features_post_projection_shape(self):
        """roi_features (post-projection, T10 keystone) must be [B, n_cells, feature_dim]."""
        splitter = _make_splitter(feature_dim=256, hidden_dim=64)
        splitter.eval()
        features = torch.randn(2, 256, 32, 32)
        r = splitter(features, image_size=(32, 32), hard=True)
        assert r.roi_features is not None, (
            "T10: roi_features must be set (post-projection field)"
        )
        assert r.roi_features.dim() == 3
        assert r.roi_features.shape[0] == 2
        assert r.roi_features.shape[1] == r.mask_ste.shape[1]
        assert r.roi_features.shape[2] == 256  # feature_dim (V3 d_model)

    def test_mask_ste_shape(self):
        """mask_ste must be [B, n_cells]."""
        splitter = _make_splitter()
        splitter.eval()
        features = torch.randn(2, 256, 32, 32)
        r = splitter(features, image_size=(32, 32), hard=True)
        assert r.mask_ste.dim() == 2
        assert r.mask_ste.shape[0] == 2  # B
        assert r.mask_ste.shape[1] == r.roi_features_raw.shape[1]  # n_cells

    def test_candidate_indices_shape(self):
        """candidate_indices must be [B * K]."""
        splitter = _make_splitter()
        splitter.eval()
        features = torch.randn(2, 256, 32, 32)
        r = splitter(features, image_size=(32, 32), hard=True)
        assert r.candidate_indices.dim() == 1
        # K ≤ K_fixed=16, so M = B*K ≤ 32
        assert r.candidate_indices.shape[0] == 2 * 16  # K=K_fixed for n_cells>=16

    def test_mask_ste_ste_preservation(self):
        """T10+Review: mask_ste must preserve STE gradient (no .detach())."""
        splitter = _make_splitter()
        splitter.train()
        features = torch.randn(1, 256, 32, 32).requires_grad_(True)
        r = splitter(features, image_size=(32, 32), hard=False)
        assert r.mask_ste.requires_grad, (
            "T10+Review: mask_ste must require grad (STE bridge intact)"
        )
        # Also verify the chain: backward through mask_ste reaches features
        r.mask_ste.sum().backward()
        assert features.grad is not None


class TestP0StaticRegistration:
    """P0: feature_proj and roi_norm must be statically registered in __init__.

    Forwarding this rule prevents the silent 'optimizer misses new params'
    failure mode where modules are lazily built in forward() and thus
    invisible to optimizer.param_groups.
    """

    def test_feature_proj_exists_at_init(self):
        """feature_proj must be a registered submodule after __init__."""
        splitter = _make_splitter()
        assert hasattr(splitter, "feature_proj")
        assert isinstance(splitter.feature_proj, torch.nn.Linear)
        # Default Linear: in_features=256, out_features=256 (config defaults)
        assert splitter.feature_proj.in_features == 256
        assert splitter.feature_proj.out_features == 256

    def test_roi_norm_exists_at_init(self):
        """roi_norm must be a registered submodule after __init__."""
        splitter = _make_splitter()
        assert hasattr(splitter, "roi_norm")
        assert isinstance(splitter.roi_norm, torch.nn.LayerNorm)
        assert splitter.roi_norm.normalized_shape == (256,)

    def test_param_count_unchanged_across_forward(self):
        """P0 critical: no new modules registered in forward()."""
        splitter = _make_splitter()
        # Snapshot parameters before forward
        params_before = list(splitter.parameters())
        n_before = len(params_before)
        # Snapshot named_modules
        modules_before = list(splitter.named_modules())
        n_mod_before = len(modules_before)

        # Run forward 3 times with different inputs
        for _ in range(3):
            features = torch.randn(2, 256, 32, 32)
            _ = splitter(features, image_size=(32, 32), hard=True)

        params_after = list(splitter.parameters())
        n_after = len(params_after)
        modules_after = list(splitter.named_modules())
        n_mod_after = len(modules_after)

        assert n_before == n_after, (
            f"P0 violation: param count changed from {n_before} to {n_after} "
            f"after forward — new modules were lazily registered!"
        )
        assert n_mod_before == n_mod_after, (
            f"P0 violation: module count changed from {n_mod_before} "
            f"to {n_mod_after} after forward"
        )

    def test_ste_gradient_flows_to_feature_proj(self):
        """Gradient must flow to feature_proj.weight (T10 keystone)."""
        splitter = _make_splitter()
        splitter.train()
        features = torch.randn(1, 256, 32, 32).requires_grad_(True)
        r = splitter(features, image_size=(32, 32), hard=False)
        # Use roi_features (post-projection) as the gradient target — it
        # explicitly depends on feature_proj
        r.roi_features.sum().backward()
        assert splitter.feature_proj.weight.grad is not None, (
            "T10: feature_proj.weight.grad is None — STE chain broken"
        )
        assert splitter.feature_proj.weight.grad.abs().sum().item() > 0
        assert splitter.roi_norm.weight.grad is not None


class TestP1NoPostMutation:
    """P1: External post-mutation of splitter._config must NOT work silently.

    The LayerNorm at __init__ caches its weight shape; mutating the
    config field afterwards doesn't reshape the LayerNorm. The fix is
    to disallow this pattern (the test verifies it would crash at
    forward time, not silently degrade).
    """

    def test_post_mutation_to_feature_dim_causes_runtime_error(self):
        """splitter._config.feature_dim = 64 should cause forward-time shape mismatch.

        This is the desired loud failure — preventing a future PR from
        silently breaking things by mutating config post-construction.
        """
        splitter = _make_splitter(feature_dim=256, hidden_dim=64)
        splitter.eval()
        # Mutate the config (the dangerous pattern)
        splitter._config.feature_dim = 64
        # Now forward should fail because roi_norm expects dim=256
        features = torch.randn(1, 256, 32, 32)
        with pytest.raises((RuntimeError, AssertionError, ValueError)):
            _ = splitter(features, image_size=(32, 32), hard=True)


class TestV3TokenizerBroadcastCompat:
    """V3 fast-path contract: roi_features_raw must be broadcastable with mask_ste.

    The tokenizer computes ``tokens_all = roi_features_raw * mask_ste.unsqueeze(-1)``,
    so the last two dims must align: roi_features_raw is [B, N, d] and
    mask_ste is [B, N].
    """

    def test_roi_features_raw_broadcasts_with_mask_ste(self):
        """roi_features_raw * mask_ste.unsqueeze(-1) must succeed."""
        splitter = _make_splitter()
        splitter.eval()
        features = torch.randn(2, 256, 32, 32)
        r = splitter(features, image_size=(32, 32), hard=True)
        # The exact broadcast that the tokenizer does at line 1021
        tokens_all = r.roi_features_raw * r.mask_ste.unsqueeze(-1)
        assert tokens_all.shape == r.roi_features_raw.shape

    def test_gather_with_candidate_indices_works(self):
        """tokens_all[batch_idx, candidate_indices, :] must produce [M, d_model]."""
        splitter = _make_splitter()
        splitter.eval()
        features = torch.randn(2, 256, 32, 32)
        r = splitter(features, image_size=(32, 32), hard=True)
        tokens_all = r.roi_features_raw * r.mask_ste.unsqueeze(-1)
        # Build batch indices for each candidate (B=2, K=16 → M=32)
        B = r.batch_indices.unique().numel()
        K = r.candidate_indices.shape[0] // B
        batch_idx = torch.arange(B).repeat_interleave(K)
        # Gather exactly as tokenizer.py:1024 does
        pooled = tokens_all[batch_idx, r.candidate_indices, :]
        assert pooled.shape == (B * K, r.roi_features_raw.shape[-1])
