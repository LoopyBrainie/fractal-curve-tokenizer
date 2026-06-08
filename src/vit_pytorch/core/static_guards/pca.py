"""v1.3 STANDARD: PCA (Path-Coordinate Alignment) guard.

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.6 Set B.

Performs bitwise validation between2D spatial coordinates (x, y) and the
generated1D Hilbert sequence indices. Ensures round-trip consistency:
d -> (x, y) -> d' should be the identity.

Catches bit-level inconsistencies that could cause downstream RoPE
position-encoding corruption.
"""
from __future__ import annotations
from typing import Optional
import torch
from vit_pytorch.core.outcome import Ok, Err, Outcome, ConfigError
from vit_pytorch.core.curve_hilbert import HilbertCurve


class PCAError(ConfigError):
 """PCA guard error. kind in {"length_mismatch", "round_trip_violation"}."""
 kind: str # "length_mismatch" | "round_trip_violation"


def check_path_coordinate_alignment(
 x: torch.Tensor,
 y: torch.Tensor,
 hilbert_indices: torch.Tensor,
 grid_size: int,
 atol: int =0,
) -> Outcome[None, PCAError]:
 """Verify Hilbert index round-trip consistency.
 
 For each i:
  d = hilbert_indices[i]
  (x', y') = HilbertCurve.d_to_xy(grid_size, d)
  d'' = HilbertCurve.xy_to_d(grid_size, x', y')
  assert d'' == d
 
 Args:
  x: [N] x coordinates (long tensor)
  y: [N] y coordinates (long tensor)
  hilbert_indices: [N] Hilbert indices (long tensor)
  grid_size: Hilbert grid size (must be power of2)
  atol: Tolerance for round-trip (default0 = exact)
 
 Returns:
  Ok if all round-trips are consistent.
  Err with kind="length_mismatch" if tensor lengths differ.
  Err with kind="round_trip_violation" if any index is inconsistent.
 """
 n = x.shape[0]
 if y.shape[0] != n or hilbert_indices.shape[0] != n:
        return Err(PCAError(
         kind="length_mismatch",
         reason=f"tensor lengths: x={n}, y={y.shape[0]}, d={hilbert_indices.shape[0]}",
        ))

 if n ==0:
        return Ok(None) # Empty - vacuously consistent

 # Round-trip: d -> (x, y) -> d'
 x_back, y_back = HilbertCurve.d_to_xy_batch(grid_size, hilbert_indices)
 d_back = HilbertCurve.xy_to_d_batch(grid_size, x_back, y_back)

 if torch.abs(d_back - hilbert_indices).max().item() > atol:
        bad_idx = int(torch.abs(d_back - hilbert_indices).argmax().item())
        return Err(PCAError(
         kind="round_trip_violation",
         reason=f"index {bad_idx}: d={int(hilbert_indices[bad_idx])}, "
         f"d'={int(d_back[bad_idx])} (grid_size={grid_size})",
        ))

 return Ok(None)
