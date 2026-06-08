"""v1.3 STANDARD: 8-guard runner (CI-blocking pre-PoC gate).

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.6.

Aggregates all 8 static code guards (4 Set A + 4 Set B) into a single
`run_all_guards()` function. The runner is the **CI-blocking pre-PoC gate**:
any single guard failure blocks the entire test session, not just the
PoC.

Usage:
    from vit_pytorch.core.static_guards import run_all_guards
    results = run_all_guards(model)
    if any(isinstance(r, Err) for r in results.values()):
        raise RuntimeError(f"Static guard failed: {results}")

Or via shell:
    bash scripts/verify_phase0.sh
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from vit_pytorch.core.outcome import Err, Ok, Outcome

# Import all 8 guards
from .sma import check_state_machine_alignment
from .dgc import check_dual_end_gradient_conservation
from .ffmi import check_fractal_feature_isolation
from .pcar import check_p_controller_anti_saturation
from .tma import check_telemetry_alignment
from .adgc import check_adgc_monotonicity
from .fmig import check_feature_map_integrity
from .pca import check_path_coordinate_alignment


# v1.3 STANDARD: 8 guards (4 Set A + 4 Set B), collision-free names per §9.6
GUARD_NAMES: Tuple[str, ...] = (
    "SMA",   # Set A: State Machine Alignment
    "DGC",   # Set A: Dual-end Gradient Conservation
    "FFMI",  # Set A: Fractal Feature Map Isolation
    "PCAR",  # Set A: P-Controller Anti-saturation
    "TMA",   # Set B: Telemetry Monitor Alignment
    "ADGC",  # Set B: Adaptive Depthwise Gradient Clipping
    "FMIG",  # Set B: Feature Map Integrity Guard
    "PCA",   # Set B: Path-Coordinate Alignment
)


def run_all_guards(
    model: nn.Module,
    *,
    input_shape: Tuple[int, ...] = (2, 8),
    feature_map_shape: Tuple[int, int, int, int] = (2, 64, 16, 16),
    dgc_input_shape: Tuple[int, ...] = (2, 8),
) -> Dict[str, Outcome[None, Any]]:
    """Run all 8 static code guards against a model.

    Args:
        model: The nn.Module to validate.
        input_shape: Shape for DGC's synthetic forward pass.
        feature_map_shape: Shape for FMIG's feature map check.
        dgc_input_shape: Alias for input_shape (DGC only).

    Returns:
        Dict mapping guard name to Outcome[None, error].
        Ok if the guard passes; Err if the guard fails.

    Example:
        >>> model = nn.Linear(8, 8)
        >>> results = run_all_guards(model)
        >>> all(r == Ok(None) or r.is_ok() for r in results.values())
        True
    """
    results: Dict[str, Outcome[None, Any]] = {}

    # Set A: classic control-flow/optimization guards

    # SMA: state machine alignment
    results["SMA"] = check_state_machine_alignment(model)

    # DGC: dual-end gradient conservation
    results["DGC"] = check_dual_end_gradient_conservation(model, input_shape=dgc_input_shape)

    # FFMI: fractal feature map isolation
    results["FFMI"] = check_fractal_feature_isolation(model)

    # PCAR: P-controller anti-saturation
    results["PCAR"] = check_p_controller_anti_saturation(model)

    # Set B: geometry/structure guards

    # TMA: telemetry monitor alignment
    results["TMA"] = check_telemetry_alignment(model)

    # ADGC: adaptive depthwise gradient clipping (parameter-only, no model input)
    results["ADGC"] = check_adgc_monotonicity()

    # FMIG: feature map integrity (synthetic feature map)
    synthetic_feat = torch.randn(*feature_map_shape)
    results["FMIG"] = check_feature_map_integrity(synthetic_feat)

    # PCA: path-coordinate alignment (synthetic round-trip)
    grid_size = 8
    n_samples = 16
    x = torch.randint(0, grid_size, (n_samples,), dtype=torch.long)
    y = torch.randint(0, grid_size, (n_samples,), dtype=torch.long)
    from vit_pytorch.core.curve_hilbert import HilbertCurve
    d = HilbertCurve.xy_to_d_batch(grid_size, x, y)
    results["PCA"] = check_path_coordinate_alignment(x, y, d, grid_size)

    return results


def assert_all_guards_pass(
    model: nn.Module,
    *,
    input_shape: Tuple[int, ...] = (2, 8),
    feature_map_shape: Tuple[int, int, int, int] = (2, 64, 16, 16),
) -> None:
    """Run all 8 guards and assert all pass. Raises RuntimeError on any failure.

    This is the CI-blocking assertion. If any guard fails, the test session
    is aborted.

    Args:
        model: The nn.Module to validate.
        input_shape: Shape for DGC's synthetic forward pass.
        feature_map_shape: Shape for FMIG's feature map check.

    Raises:
        RuntimeError: If any guard fails, with details about which guard
                      failed and the error.
    """
    results = run_all_guards(
        model,
        input_shape=input_shape,
        feature_map_shape=feature_map_shape,
    )
    failures = {name: result for name, result in results.items() if isinstance(result, Err)}
    if failures:
        details = "\n".join(
            f"  - {name}: {result.error}"
            for name, result in failures.items()
        )
        raise RuntimeError(
            f"Static code guard(s) FAILED (CI-blocking):\n{details}\n"
            f"Fix the guard errors before running PoC tests."
        )
