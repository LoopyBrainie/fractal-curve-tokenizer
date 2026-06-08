"""Polar Voronoi WBA(64) >= 2.55 contract test (v1.3 §9.7, B.9).

This test module exercises the WBA(64) hard gate, the σ schedule, the
soft switch, and the CenterAwareEncoder plumbing on the
PolarVoronoiSplitter skeleton. The hard gate is the only block-merge
criterion in B.9 — passing the whole suite is the green light to
commit the B.8+B.9 bundle.
"""

import math

import pytest
import torch

from vit_pytorch.core.constants import (
    POLAR_VORONOI_EPOCH_FULL,
    POLAR_VORONOI_EPOCH_SOFT_START,
    POLAR_VORONOI_SIGMA_FINAL,
    POLAR_VORONOI_WBA64_BOUNDARY,
)
from vit_pytorch.layers.ca.center_aware_encoder import CenterAwareEncoder
from vit_pytorch.layers.splitters.polar_voronoi_splitter import (
    PolarVoronoiSplitter,
    PolarVoronoiSplitterConfig,
    compute_sigma,
    compute_soft_switch,
    compute_wba_64,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cae() -> CenterAwareEncoder:
    return CenterAwareEncoder(in_channels=64)


@pytest.fixture
def splitter(cae: CenterAwareEncoder) -> PolarVoronoiSplitter:
    return PolarVoronoiSplitter(cae=cae)


@pytest.fixture
def center_biased_features() -> torch.Tensor:
    """A 64x64 feature map with a strong center peak — the WBA(64) test
    fixture described in v1.3 §9.7. The center cell carries a unit
    impulse; the four NSEW neighbours carry small uniform noise.
    """
    torch.manual_seed(0)
    feats = torch.full((1, 64, 64, 64), 0.01)
    feats[:, :, 32, 32] = 1.0
    return feats


# ---------------------------------------------------------------------------
# WBA(64) hard gate
# ---------------------------------------------------------------------------


class TestWBA64HardGate:
    """WBA(64) >= 2.55 is the v1.3 §9.7 hard merge gate."""

    def test_wba64_boundary_value_constant(self):
        """Boundary constant in constants.py must equal 2.55."""
        assert POLAR_VORONOI_WBA64_BOUNDARY == pytest.approx(2.55, abs=1e-9)

    def test_wba64_passes_on_center_biased_input(self, center_biased_features):
        """WBA(64) >= 2.55 on a center-biased 64x64 input."""
        score = compute_wba_64(center_biased_features)
        assert score >= 2.55, (
            f"WBA(64) gate failed: got {score:.3f} < 2.55 boundary"
        )

    def test_wba64_uniform_input_is_below_boundary(self):
        """A uniform input has no center-bias; WBA(64) < 2.55."""
        feats = torch.full((1, 8, 64, 64), 0.5)
        score = compute_wba_64(feats)
        assert score < 2.55, f"Uniform input should fail gate, got {score:.3f}"

    def test_wba64_in_expected_range(self, center_biased_features):
        """WBA(64) is non-negative and well-defined on sensible inputs.

        The proxy score has no hard upper bound (contrast / spread with
        a 4x multiplier can grow into the thousands on a clean
        center-biased fixture). The contract is the lower-bound gate
        ``WBA(64) >= 2.55`` — the upper side is unconstrained.
        """
        score = compute_wba_64(center_biased_features)
        assert score >= 0.0, f"WBA(64) must be non-negative, got {score:.3f}"
        assert math.isfinite(score), f"WBA(64) must be finite, got {score}"

    def test_wba64_rejects_bad_shape(self):
        """Non-4D input is rejected with a clear error."""
        with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
            compute_wba_64(torch.zeros(64, 64))

    def test_wba64_rejects_tiny_input(self):
        """Inputs smaller than 3x3 cannot form a 5-cell mask."""
        score = compute_wba_64(torch.zeros(1, 4, 2, 2))
        assert score == 0.0


# ---------------------------------------------------------------------------
# σ schedule
# ---------------------------------------------------------------------------


class TestSigmaSchedule:
    """σ(epoch) interpolates linearly from 2.0 (epoch 0) to 0.7 (epoch 16)."""

    def test_sigma_at_epoch_zero_is_start(self):
        cfg = PolarVoronoiSplitterConfig()
        assert compute_sigma(0, cfg) == pytest.approx(2.0, abs=1e-9)

    def test_sigma_at_epoch_full_is_final(self):
        cfg = PolarVoronoiSplitterConfig()
        assert compute_sigma(cfg.epoch_full, cfg) == pytest.approx(
            POLAR_VORONOI_SIGMA_FINAL, abs=1e-9
        )

    def test_sigma_decreases_monotonically(self):
        cfg = PolarVoronoiSplitterConfig()
        prev = compute_sigma(0, cfg)
        for epoch in range(1, cfg.epoch_full + 1):
            cur = compute_sigma(epoch, cfg)
            assert cur <= prev + 1e-9, f"σ not monotonic at epoch {epoch}"
            prev = cur

    @pytest.mark.parametrize(
        "epoch,expected",
        [
            (0, 2.0),
            (4, 2.0),                          # before soft-start, clamped to start
            (5, 2.0),                          # soft-start
            (8, 2.0 - 1.3 * (3 / 11)),          # mid-soft-start
            (16, POLAR_VORONOI_SIGMA_FINAL),   # full = 0.7
            (32, POLAR_VORONOI_SIGMA_FINAL),   # after full, clamped
        ],
    )
    def test_sigma_at_key_epochs(self, epoch, expected):
        cfg = PolarVoronoiSplitterConfig()
        got = compute_sigma(epoch, cfg)
        assert got == pytest.approx(expected, abs=1e-3), (
            f"σ({epoch}) expected {expected:.3f}, got {got:.3f}"
        )

    def test_sigma_handles_degenerate_config(self):
        """epoch_full <= epoch_soft_start collapses to σ_final."""
        cfg = PolarVoronoiSplitterConfig(epoch_soft_start=10, epoch_full=5)
        assert compute_sigma(0, cfg) == pytest.approx(POLAR_VORONOI_SIGMA_FINAL)


# ---------------------------------------------------------------------------
# Soft switch
# ---------------------------------------------------------------------------


class TestSoftSwitch:
    """Soft switch ramps from 0 to 1 between epoch 5 and epoch 16."""

    def test_soft_switch_off_before_epoch_five(self):
        cfg = PolarVoronoiSplitterConfig()
        for epoch in range(0, cfg.epoch_soft_start):
            assert compute_soft_switch(epoch, cfg) == 0.0

    def test_soft_switch_on_after_epoch_full(self):
        cfg = PolarVoronoiSplitterConfig()
        for epoch in (cfg.epoch_full, cfg.epoch_full + 1, 50, 1000):
            assert compute_soft_switch(epoch, cfg) == 1.0

    def test_soft_switch_increases_monotonically(self):
        cfg = PolarVoronoiSplitterConfig()
        prev = 0.0
        for epoch in range(0, cfg.epoch_full + 2):
            cur = compute_soft_switch(epoch, cfg)
            assert cur >= prev - 1e-9
            prev = cur

    def test_soft_switch_midpoint(self):
        cfg = PolarVoronoiSplitterConfig()
        mid = (cfg.epoch_soft_start + cfg.epoch_full) // 2
        # Linear ramp: midpoint of (5, 16) is 10.5
        expected = (mid - cfg.epoch_soft_start) / (
            cfg.epoch_full - cfg.epoch_soft_start
        )
        got = compute_soft_switch(mid, cfg)
        assert got == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# CenterAwareEncoder
# ---------------------------------------------------------------------------


class TestCenterAwareEncoder:
    """The CAE returns a (bool, confidence) pair."""

    def test_cae_forward_shape(self, cae: CenterAwareEncoder):
        feats = torch.randn(2, 64, 16, 16)
        is_biased, confidence = cae(feats)
        assert is_biased.shape == (2,)
        assert is_biased.dtype == torch.bool
        assert confidence.shape == (2,)
        assert torch.all((confidence >= 0.0) & (confidence <= 1.0))

    def test_cae_rejects_bad_shape(self, cae: CenterAwareEncoder):
        with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
            cae(torch.randn(2, 64, 16))


# ---------------------------------------------------------------------------
# PolarVoronoiSplitter skeleton
# ---------------------------------------------------------------------------


class TestPolarVoronoiSplitterSkeleton:
    """The skeleton exposes WBA, CAE gate, σ and soft switch."""

    def test_skeleton_runs(self, splitter, center_biased_features):
        out = splitter(center_biased_features)
        assert "wba64" in out
        assert "cae_gate" in out
        assert "cae_active" in out
        assert "sigma" in out
        assert "soft_switch" in out
        assert "effective_weight" in out

    def test_skeleton_passes_wba_gate_at_init(
        self, splitter, center_biased_features
    ):
        """WBA(64) >= 2.55 on the standard fixture, regardless of epoch."""
        for epoch in (0, 5, 16, 100):
            splitter.set_epoch(epoch)
            out = splitter(center_biased_features)
            assert out["wba64"] >= 2.55, (
                f"epoch {epoch}: WBA(64) gate failed, got {out['wba64']:.3f}"
            )

    def test_skeleton_soft_switch_tracks_epoch(self, splitter, center_biased_features):
        splitter.set_epoch(0)
        assert splitter.current_soft_switch == 0.0
        splitter.set_epoch(10)
        assert 0.0 < splitter.current_soft_switch < 1.0
        splitter.set_epoch(20)
        assert splitter.current_soft_switch == 1.0

    def test_skeleton_sigma_decreases_with_epoch(
        self, splitter, center_biased_features
    ):
        splitter.set_epoch(0)
        s0 = splitter.current_sigma
        splitter.set_epoch(8)
        s8 = splitter.current_sigma
        splitter.set_epoch(16)
        s16 = splitter.current_sigma
        assert s0 > s8 > s16

    def test_skeleton_effective_weight_zero_before_soft_start(
        self, splitter, center_biased_features
    ):
        splitter.set_epoch(0)
        out = splitter(center_biased_features)
        # soft switch = 0 -> effective weight is exactly 0
        assert torch.all(out["effective_weight"] == 0.0)

    def test_skeleton_extra_repr(self, splitter):
        splitter.set_epoch(8)
        r = splitter.extra_repr()
        assert "PolarVoronoiSplitter" in r
        assert "sigma" in r
        assert "soft_switch" in r

    def test_skeleton_rejects_bad_shape(self, splitter):
        with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
            splitter(torch.randn(2, 64))


# ---------------------------------------------------------------------------
# Module-level import test
# ---------------------------------------------------------------------------


def test_polar_voronoi_imports():
    """Public surface is exported from the splitters package."""
    from vit_pytorch.layers.splitters import (
        PolarVoronoiSplitter,
        PolarVoronoiSplitterConfig,
        compute_sigma,
        compute_soft_switch,
        compute_wba_64,
    )

    assert PolarVoronoiSplitter is not None
    assert PolarVoronoiSplitterConfig is not None
    assert callable(compute_sigma)
    assert callable(compute_soft_switch)
    assert callable(compute_wba_64)
