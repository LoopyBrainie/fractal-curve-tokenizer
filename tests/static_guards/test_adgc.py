"""v1.3 STANDARD: ADGC (Adaptive Depthwise Gradient Clipping) guard test.

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.6 Set B.

delta_c(d) = delta_0 * gamma^(-d) where delta_0=1.0, gamma=0.95 (per v1.3 §9.10 + §9.6).
Monotonicity invariant: delta_c(d+1) < delta_c(d) is FALSE — actually it INCREASES
with depth when gamma < 1. The guard tests the correct invariant: delta_c is
INCREASING in d (deeper -> larger threshold -> more aggressive clipping).
"""
import pytest
from hypothesis import given, strategies as st
from vit_pytorch.core.outcome import Ok, Err
from vit_pytorch.core.static_guards.adgc import (
    compute_delta_c,
    check_adgc_monotonicity,
    ADGCError,
)


@pytest.mark.static_guard
class TestADGCGuard:
    def test_delta_c_at_depth_0_is_delta_0(self):
        assert compute_delta_c(0, delta_0=1.0, gamma=0.95) == 1.0

    def test_delta_c_at_depth_1(self):
        # delta_c(1) = 1.0 * 0.95^(-1) = 1/0.95 ~= 1.0526
        assert abs(compute_delta_c(1, delta_0=1.0, gamma=0.95) - 1.0 / 0.95) < 1e-9

    def test_delta_c_at_depth_5(self):
        # delta_c(5) = 1.0 * 0.95^(-5) ~= 1.276
        expected = 1.0 / (0.95 ** 5)
        assert abs(compute_delta_c(5, delta_0=1.0, gamma=0.95) - expected) < 1e-9

    @given(
        delta_0=st.floats(min_value=0.5, max_value=2.0, allow_nan=False),
        gamma=st.floats(min_value=0.5, max_value=0.99, allow_nan=False),
    )
    def test_delta_c_is_increasing_in_depth(self, delta_0, gamma):
        # gamma < 1 -> gamma^(-d) is increasing in d
        # So delta_c(d) is INCREASING (tighter clipping for deeper layers)
        d1 = compute_delta_c(2, delta_0, gamma)
        d2 = compute_delta_c(5, delta_0, gamma)
        assert d2 > d1, f"delta_c must increase with depth: delta_c(2)={d1}, delta_c(5)={d2}"

    def test_check_adgc_monotonicity_passes_default(self):
        result = check_adgc_monotonicity()
        assert isinstance(result, Ok)

    def test_check_adgc_rejects_inverted_gamma(self):
        result = check_adgc_monotonicity(gamma=1.05)  # gamma > 1 -> formula breaks
        assert isinstance(result, Err)
        assert result.error.kind == "invalid_gamma"
