"""v1.3 STANDARD: FMIG (Feature Map Integrity Guard) test.

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.6 Set B.
Channel variance threshold: 1e-6 (per v1.3 §9.6).
"""
import pytest
import torch
from vit_pytorch.core.outcome import Ok, Err
from vit_pytorch.core.static_guards.fmig import (
    check_feature_map_integrity,
    inject_orthogonal_noise,
    FMIGError,
    FMIG_CHANNEL_VARIANCE_THRESHOLD,
)


@pytest.mark.static_guard
class TestFMIGGuard:
    def test_fmig_passes_for_healthy_feature_map(self):
        feat = torch.randn(2, 64, 16, 16)  # Healthy variance
        result = check_feature_map_integrity(feat)
        assert isinstance(result, Ok)

    def test_fmig_detects_channel_collapse(self):
        # All channels identical (variance = 0)
        feat = torch.zeros(2, 64, 16, 16)
        result = check_feature_map_integrity(feat)
        assert isinstance(result, Err)
        assert result.error.kind == "channel_collapse"

    def test_fmig_detects_low_channel_variance(self):
        # Channel variance just below threshold
        feat = torch.randn(2, 64, 16, 16) * (1e-7)  # very small variance
        result = check_feature_map_integrity(feat)
        assert isinstance(result, Err)

    def test_fmig_injects_orthogonal_noise(self):
        feat_collapsed = torch.zeros(2, 64, 16, 16)
        feat_recovered = inject_orthogonal_noise(feat_collapsed, scale=1e-3)
        # After noise injection, channel variance should be > threshold
        result = check_feature_map_integrity(feat_recovered)
        assert isinstance(result, Ok)
        # The recovered tensor should NOT be the same as the input
        assert not torch.equal(feat_recovered, feat_collapsed)

    def test_fmig_default_threshold_is_1e_minus_6(self):
        assert FMIG_CHANNEL_VARIANCE_THRESHOLD == 1e-6
