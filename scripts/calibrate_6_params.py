"""v1.3 STANDARD Phase 0: 6 base parameter values calibration script.

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.10.

The 6 base values are PoC calibration items (NOT final):
  - T_fatal = 0.50 (Hard Rollback threshold)
  - alpha_tree = 1.20 (A4 tree penalty coefficient)
  - alpha_skew = 0.40 (Q_balanced skew penalty)
  - gamma_kinetic = 0.15 (BV2 attention growth rate)
  - delta_washout = 0.05 (G1 WBA entropy washout)
  - epsilon_leak_factor = 0.25 (G2 RoPE MI one-strike factor)

Calibration policy (§9.10):
  - All 6 values are PoC calibration items
  - A value is "calibrated" when it survives a 5x5 grid search over ±50% range
  - If calibrated value deviates > 20% from base, design doc must be updated
  - If > 50%, new design version (v1.4 or later)

This script:
  1. Defines the 6 base values
  2. Generates the 5x5 grid for each parameter
  3. Documents the calibration methodology
  4. Provides a hook for actual grid search in Phase 1/2

Note: Running a 5^6 = 15,625-combination grid search with full quick-test
is impractical for Phase 0. The actual calibration happens in Phase 1/2
when the full system is operational.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class V13CalibrationValues:
    """v1.3 STANDARD: 6 base parameter values (per §9.10)."""
    T_fatal: float = 0.50
    alpha_tree: float = 1.20
    alpha_skew: float = 0.40
    gamma_kinetic: float = 0.15
    delta_washout: float = 0.05
    epsilon_leak_factor: float = 0.25

    def to_dict(self) -> Dict[str, float]:
        return {
            "T_fatal": self.T_fatal,
            "alpha_tree": self.alpha_tree,
            "alpha_skew": self.alpha_skew,
            "gamma_kinetic": self.gamma_kinetic,
            "delta_washout": self.delta_washout,
            "epsilon_leak_factor": self.epsilon_leak_factor,
        }


def generate_grid_5x5(base: float, range_factor: float = 0.5) -> List[float]:
    """Generate a 5-value grid centered on base, spanning ±range_factor * base.

    For T_fatal=0.5 and range_factor=0.5:
      values = [0.25, 0.375, 0.5, 0.625, 0.75]

    For alpha_tree=1.2:
      values = [0.6, 0.9, 1.2, 1.5, 1.8]

    For delta_washout=0.05 (a small value):
      values = [0.025, 0.0375, 0.05, 0.0625, 0.075]
    """
    if base == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    low = base * (1.0 - range_factor)
    high = base * (1.0 + range_factor)
    # 5 evenly-spaced values (base in the middle)
    return [low + (high - low) * i / 4.0 for i in range(5)]


def generate_all_grids(
    base_values: V13CalibrationValues,
    range_factor: float = 0.5,
) -> Dict[str, List[float]]:
    """Generate 5x5 grid for each of the 6 parameters.

    Returns:
        Dict mapping param name to 5-value grid.
    """
    return {
        name: generate_grid_5x5(value, range_factor)
        for name, value in base_values.to_dict().items()
    }


def total_combinations(grids: Dict[str, List[float]]) -> int:
    """Total grid search combinations (5^N where N = number of params)."""
    result = 1
    for grid in grids.values():
        result *= len(grid)
    return result


def main():
    """Print the 6 base values and 5x5 grids."""
    base = V13CalibrationValues()
    print("=" * 70)
    print("v1.3 STANDARD: 6 base parameter values (per §9.10)")
    print("=" * 70)
    print("\nBase values:")
    for name, value in base.to_dict().items():
        print(f"  {name:30s} = {value:.4f}")

    print("\n5x5 grid (±50% range) for each parameter:")
    grids = generate_all_grids(base)
    for name, grid in grids.items():
        formatted = ", ".join(f"{v:.4f}" for v in grid)
        print(f"  {name:30s} : [{formatted}]")

    total = total_combinations(grids)
    print(f"\nTotal grid search combinations: 5^6 = {total}")
    print("\n[Phase 0 Note] Actual grid search deferred to Phase 1/2 with full system.")
    print("[Phase 0 Note] Calibration policy: per §9.10, deviations > 20% require doc update,")
    print("[Phase 0 Note]                    deviations > 50% require new design version.")

    print("\n[Phase 1 Hook]")
    print("When Phase 1 components are operational, this script can be invoked as:")
    print("  for params in itertools.product(*grids.values()):")
    print("      run_quick_test(**params)")
    print("      record (params, top1_accuracy, throughput)")


if __name__ == "__main__":
    main()
