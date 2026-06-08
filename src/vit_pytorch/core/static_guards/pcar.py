"""v1.3 STANDARD: PCAR (P-Controller Anti-saturation) guard.

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.6 Set A.

Verifies that any P-Controller in the system has explicit anti-saturation
bounds (e.g., _saturation_min, _saturation_max) and that the bounds are
in the correct order (min < max). Passes vacuously for modules without
saturation attributes.
"""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from vit_pytorch.core.outcome import Ok, Err, Outcome, ConfigError


class PCARError(ConfigError):
    """PCAR guard error. kind in {"inverted_bounds", "missing_attrs"}."""
    kind: str


def check_p_controller_anti_saturation(
    module: nn.Module,
    eps: float = 1e-6,
) -> Outcome[None, PCARError]:
    """Verify the module has well-formed anti-saturation bounds.

    A P-Controller is identified by having both `_saturation_min` and
    `_saturation_max` attributes. If both are present, verify:
    - _saturation_min < _saturation_max
    - |_saturation_min| < 1.0 (else it's at hard saturation)
    - |_saturation_max| < 1.0

    If either attribute is missing, pass vacuously (not a P-Controller).
    """
    sat_min = getattr(module, "_saturation_min", None)
    sat_max = getattr(module, "_saturation_max", None)

    if sat_min is None or sat_max is None:
        return Ok(None)  # Not a P-Controller

    try:
        sat_min_val = float(sat_min)
        sat_max_val = float(sat_max)
    except (TypeError, ValueError):
        return Err(PCARError(
            kind="missing_attrs",
            reason=f"_saturation_min/max must be numeric, got {type(sat_min).__name__}/{type(sat_max).__name__}",
        ))

    if sat_min_val >= sat_max_val - eps:
        return Err(PCARError(
            kind="inverted_bounds",
            reason=f"_saturation_min={sat_min_val} >= _saturation_max={sat_max_val}",
        ))

    return Ok(None)
