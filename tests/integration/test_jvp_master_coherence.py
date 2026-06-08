"""v1.3 STANDARD Phase 0: JVP-6 Master Coherence consistency check.

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.6
the design must satisfy these 5 inter-JVP consistency invariants:

  1. JVP-1 (5-grid EAS) and JVP-5 (Stress-Scale N=64,128,256) token count
     bounds [K_min=8, K_max=64] are consistent (design range within safety bounds)
  2. JVP-2 (G1 precision ≥ 0.2%) and JVP-3 (G2 throughput ≥ 1.4×) thresholds
     are positive numerics (no contradictions)
  3. JVP-3 G2 GME floor (60%) and JVP-4 G3 GME floor (40%) define a meaningful
     "uncertainty band" [40%, 60%) where neither gate fires
  4. JVP-5 N values (64, 128, 256) are all powers-of-2 (2^6, 2^7, 2^8) and
     align with the Power-of-4 chunking theorem (each N is divisible by 4)
  5. Set A (classic 4) and Set B (geometric 4) static guard names are
     disjoint, confirming no shared state between the two guard families
"""
from __future__ import annotations

import pytest

from vit_pytorch.core import constants
from vit_pytorch.core.static_guards import GUARD_NAMES


# v1.3 STANDARD: design invariants
K_MIN_DESIGN = 8
K_MAX_DESIGN = 64

# v1.3 STANDARD: JVP-2 / JVP-3 / JVP-4 gates
G1_PRECISION_THRESHOLD = 0.002          # 0.2%
G2_THROUGHPUT_THRESHOLD = 1.4           # 1.4×
G2_GME_FLOOR = 0.60                     # 60%
G3_GME_FLOOR = 0.40                     # 40% (rollback below this)

# JVP-5 stress-scale N values
JVP5_STRESS_N = (64, 128, 256)

# 6 v1.3 STANDARD calibration values
V13_CALIBRATION_PARAM_NAMES = (
    "T_fatal",
    "alpha_tree",
    "alpha_skew",
    "gamma_kinetic",
    "delta_washout",
    "epsilon_leak_factor",
)

# v1.3 base calibration values
V13_BASE_CALIBRATION = {
    "T_fatal": 0.50,
    "alpha_tree": 1.20,
    "alpha_skew": 0.40,
    "gamma_kinetic": 0.15,
    "delta_washout": 0.05,
    "epsilon_leak_factor": 0.25,
}


def test_jvp1_jvp5_token_bounds_consistency():
    """Check 1: K_min=8 / K_max=64 design range fits within safety bounds."""
    assert K_MIN_DESIGN >= constants.K_MIN_HARD_LIMIT, (
        f"Design K_min={K_MIN_DESIGN} must be ≥ K_MIN_HARD_LIMIT="
        f"{constants.K_MIN_HARD_LIMIT}"
    )
    assert K_MAX_DESIGN <= constants.K_MAX_HARD_LIMIT, (
        f"Design K_max={K_MAX_DESIGN} must be ≤ K_MAX_HARD_LIMIT="
        f"{constants.K_MAX_HARD_LIMIT}"
    )
    # The adaptive design range is non-empty
    assert K_MIN_DESIGN < K_MAX_DESIGN


def test_jvp2_jvp3_thresholds_positive_no_contradiction():
    """Check 2: G1 (precision ≥ 0.2%) and G2 (throughput ≥ 1.4×) are well-defined."""
    assert G1_PRECISION_THRESHOLD > 0.0
    assert G2_THROUGHPUT_THRESHOLD > 1.0, (
        "G2 throughput must require strict improvement over baseline (1.0×)"
    )
    # G1 is a precision gain (small positive)
    assert G1_PRECISION_THRESHOLD < 0.10, (
        "G1 precision gain > 10% is implausibly large for a 3-gate system"
    )
    # G2 is a throughput gain (>= 40% improvement)
    assert 1.0 < G2_THROUGHPUT_THRESHOLD <= 5.0


def test_jvp3_jvp4_gme_floor_band_is_meaningful():
    """Check 3: G2 GME floor (60%) > G3 GME floor (40%) defines [40%, 60%) band.

    In the [40%, 60%) uncertainty band, neither G2 nor G3 fires — the system
    must enter the Paced Optimization Window (JVP-4) to make a decision.
    """
    assert G3_GME_FLOOR < G2_GME_FLOOR, (
        f"G3 GME floor {G3_GME_FLOOR} must be < G2 GME floor {G2_GME_FLOOR}"
    )
    # The band has non-zero width
    assert G2_GME_FLOOR - G3_GME_FLOOR >= 0.10, (
        "GME uncertainty band must be at least 10 percentage points wide"
    )
    # Both floors are valid probabilities
    assert 0.0 < G3_GME_FLOOR < G2_GME_FLOOR < 1.0


def test_jvp5_stress_scale_n_values_power_of_2_and_divisible_by_4():
    """Check 4: N=64, 128, 256 are powers-of-2 and divisible by 4.

    This is required by the Power-of-4 Alignment Chunking theorem:
    4^k chunk boundaries must align with quadtree nodes. For N=64 (= 4^3),
    the natural chunking is 4, 16, 64 cells — all power-of-4.
    """
    for n in JVP5_STRESS_N:
        # N must be a power of 2
        assert (n & (n - 1)) == 0, f"N={n} must be a power of 2"
        # N must be divisible by 4 (for Power-of-4 alignment)
        assert n % 4 == 0, f"N={n} must be divisible by 4 (Power-of-4 alignment)"
    # All N values distinct
    assert len(set(JVP5_STRESS_N)) == 3


def test_set_a_set_b_guard_names_disjoint():
    """Check 5: Set A (4 classic) and Set B (4 geometric) guards are disjoint.

    Set A: SMA, DGC, FFMI, PCAR (classic computational graph integrity)
    Set B: TMA, ADGC, FMIG, PCA (geometric Hilbert structure integrity)
    No shared state means independent CI-blocking enforcement.
    """
    set_a = {"SMA", "DGC", "FFMI", "PCAR"}
    set_b = {"TMA", "ADGC", "FMIG", "PCA"}
    assert set_a.isdisjoint(set_b), (
        f"Set A and Set B must be disjoint, found overlap: {set_a & set_b}"
    )
    # The 8-guard runner exposes all 8 names
    assert set_a | set_b == set(GUARD_NAMES), (
        f"GUARD_NAMES must equal Set A ∪ Set B; "
        f"missing: {(set_a | set_b) - set(GUARD_NAMES)}; "
        f"extra: {set(GUARD_NAMES) - (set_a | set_b)}"
    )
    assert len(GUARD_NAMES) == 8, (
        f"v1.3 STANDARD requires exactly 8 static guards, got {len(GUARD_NAMES)}"
    )


def test_jvp6_v13_calibration_params_consistent():
    """Check 6 (bonus): 6 v1.3 STANDARD calibration values are all in (0, 2].

    These are PoC calibration items per §9.10. A value > 2 would be a
    tuning red flag (likely means a base is miscalibrated).
    """
    for name in V13_CALIBRATION_PARAM_NAMES:
        assert name in V13_BASE_CALIBRATION
        v = V13_BASE_CALIBRATION[name]
        assert 0.0 < v <= 2.0, (
            f"Calibration value {name}={v} out of expected range (0, 2]"
        )
    # All 6 values are positive
    assert all(v > 0 for v in V13_BASE_CALIBRATION.values())
