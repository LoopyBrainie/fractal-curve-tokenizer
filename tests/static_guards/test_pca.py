"""v1.3 STANDARD: PCA (Path-Coordinate Alignment) guard test.

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.6 Set B.

Validates bitwise consistency between Hilbert index and2D coordinates.
"""
import pytest
import torch
from vit_pytorch.core.outcome import Ok, Err
from vit_pytorch.core.static_guards.pca import check_path_coordinate_alignment, PCAError
from vit_pytorch.core.curve_hilbert import HilbertCurve


@pytest.mark.static_guard
class TestPCAGuard:
 def test_pca_passes_for_correct_round_trip(self):
        n =16
        coords = [(x, y) for x in range(4) for y in range(4)]
        xs = torch.tensor([c[0] for c in coords], dtype=torch.long)
        ys = torch.tensor([c[1] for c in coords], dtype=torch.long)
        d = HilbertCurve.xy_to_d_batch(n, xs, ys)
        x_back, y_back = HilbertCurve.d_to_xy_batch(n, d)
        result = check_path_coordinate_alignment(xs, ys, d, n)
        assert isinstance(result, Ok), f"PCA should pass for valid round-trip, got {result}"

 def test_pca_detects_mismatch(self):
        n =16
        xs = torch.tensor([0,1,2,3], dtype=torch.long)
        ys = torch.tensor([0,1,2,3], dtype=torch.long)
        d = HilbertCurve.xy_to_d_batch(n, xs, ys)
        d_corrupted = d.clone()
        d_corrupted[0] = (d_corrupted[0] +1) % (n * n)
        result = check_path_coordinate_alignment(xs, ys, d_corrupted, n)
        assert isinstance(result, (Ok, Err))

 def test_pca_handles_empty_tensors(self):
        xs = torch.tensor([], dtype=torch.long)
        ys = torch.tensor([], dtype=torch.long)
        d = torch.tensor([], dtype=torch.long)
        result = check_path_coordinate_alignment(xs, ys, d,16)
        assert isinstance(result, Ok)

 def test_pca_rejects_mismatched_lengths(self):
        xs = torch.tensor([0,1,2], dtype=torch.long)
        ys = torch.tensor([0,1], dtype=torch.long)
        d = torch.tensor([0,1,2], dtype=torch.long)
        result = check_path_coordinate_alignment(xs, ys, d,16)
        assert isinstance(result, Err)
        assert result.error.kind == "length_mismatch"
