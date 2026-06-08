"""v1.3 STANDARD: ADGC (Adaptive Depthwise Gradient Clipping) guard.

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.6 Set B.

delta_c(d) = delta_0 * gamma^(-d) where delta_0=1.0, gamma=0.95 (per v1.3 §9.10 + §9.6).
Monotonicity invariant: with gamma in (0, 1], delta_c is INCREASING in d
(deeper -> larger threshold -> more aggressive clipping).
"""
from __future__ import annotations
from typing import Tuple
from vit_pytorch.core.outcome import Ok, Err, Outcome, ConfigError


class ADGCError(ConfigError):
    """ADGC guard error. kind in {"invalid_gamma", "monotonicity_violation"}."""
    kind: str  # "invalid_gamma" | "monotonicity_violation"


def compute_delta_c(
    depth: int,
    delta_0: float = 1.0,
    gamma: float = 0.95,
) -> float:
    """Compute the ADGC threshold at the given depth.

    Args:
        depth: Quadtree depth (0 = root, 1 = first split, etc.)
        delta_0: Base threshold (default 1.0 per v1.3 §9.10)
        gamma: Decay factor (default 0.95 per v1.3 §9.10)

    Returns:
        Threshold delta_c(depth) = delta_0 * gamma^(-depth)
    """
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")
    if gamma <= 0 or gamma > 1.0:
        raise ValueError(f"gamma must be in (0, 1.0], got {gamma}")
    return delta_0 * (gamma ** (-depth))


def check_adgc_monotonicity(
    delta_0: float = 1.0,
    gamma: float = 0.95,
    depth_range: Tuple[int, int] = (0, 10),
) -> Outcome[None, ADGCError]:
    """Verify the ADGC threshold is monotonically increasing in depth.

    Per v1.3 §9.6: "Deep branches (higher d) get tighter clipping."
    With gamma=0.95<1, gamma^(-d) > 1 for d>0, so delta_c(d) > delta_0.
    Larger threshold = more aggressive clipping = "tighter".

    Returns:
        Ok if monotonicity holds across depth_range.
        Err if gamma is out of bounds or monotonicity is violated.
    """
    if gamma <= 0 or gamma > 1.0:
        return Err(ADGCError(
            kind="invalid_gamma",
            reason=f"gamma must be in (0, 1.0], got {gamma}",
        ))
    # Check monotonicity
    prev = compute_delta_c(depth_range[0], delta_0, gamma)
    for d in range(depth_range[0] + 1, depth_range[1] + 1):
        curr = compute_delta_c(d, delta_0, gamma)
        if curr <= prev:
            return Err(ADGCError(
                kind="monotonicity_violation",
                reason=f"delta_c({d-1})={prev} >= delta_c({d})={curr}",
            ))
        prev = curr
    return Ok(None)
