"""v1.3 STANDARD Phase 0: Moon δ_2 formula numerical recalculation.

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.10:
   §9's δ_2 row was downgraded from MEDIUM confidence to PENDING.
   This script performs the numerical recalculation on 8x8, 32x32, 64x64 grids.

Reference: Moon, Jagadish, Faloutsos, Saltz (2001).
   "Analysis of the Clustering Properties of the Hilbert Space-Filling Curve"
   IEEE TKDE 13(1): 124-141.

Definition (Moon et al. 2001):
   For a space-filling curve P on a grid G of size n*n, the locality measure
   δ_2(P) is defined as:

       δ_2(P) = sup over query rectangles Q of
                (number of curve-segments crossing Q's boundary)
                / (total number of curve-segments in Q)

   For the Hilbert curve on an n*n grid, the known closed-form bound is:
       δ_2(H) ≈ 6.533 (Bauman 2006; HilbertNet ECCV 2022)

This script numerically computes δ_2(H) by sampling query rectangles
and computing the bad-jump ratio.
"""
from __future__ import annotations

import math
import sys
from typing import List, Tuple

import torch

# Make Hilbert curve importable
sys.path.insert(0, "src")
from vit_pytorch.core.curve_hilbert import HilbertCurve  # noqa: E402


def compute_hilbert_jumps(grid_size: int) -> torch.Tensor:
    """Compute pairwise |d_i - d_j| for adjacent Hilbert indices.

    Adjacency means: cells that share an edge in the 2D grid.
    Returns a tensor of shape [n*n] where entry i is the mean jump
    from cell i to its 2D-adjacent neighbors.
    """
    n = grid_size
    # Get Hilbert distance for each (x, y) cell
    xs = torch.arange(n).repeat(n, 1).flatten()
    ys = torch.arange(n).unsqueeze(1).expand(n, n).flatten()
    d = HilbertCurve.xy_to_d_batch(n, xs, ys)  # [n*n]

    # For each cell, find its 2D neighbors (up, down, left, right)
    jumps = []
    for i in range(n * n):
        xi, yi = int(xs[i]), int(ys[i])
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = xi + dx, yi + dy
            if 0 <= nx < n and 0 <= ny < n:
                n_idx = ny * n + nx
                neighbors.append(n_idx)
        if neighbors:
            neighbor_d = d[neighbors]
            mean_jump = (neighbor_d - d[i]).abs().float().mean().item()
            jumps.append(mean_jump)
    return torch.tensor(jumps)


def estimate_delta2(
    grid_size: int,
    n_query_rects: int = 200,
    seed: int = 42,
) -> float:
    """Numerically estimate δ_2(H) on a grid by sampling query rectangles.

    For each query rectangle Q (axis-aligned):
        - Count the Hilbert segments (jumps) inside Q that cross Q's boundary
        - Divide by total segments inside Q
        - Take the maximum over all sampled Q

    Returns the maximum (worst-case) bad-jump ratio observed.
    """
    import random
    random.seed(seed)

    n = grid_size
    xs = torch.arange(n).repeat(n, 1).flatten()
    ys = torch.arange(n).unsqueeze(1).expand(n, n).flatten()
    d = HilbertCurve.xy_to_d_batch(n, xs, ys)

    # Map (x, y) -> Hilbert d
    cell_d = {}
    for i in range(n * n):
        cell_d[(int(xs[i]), int(ys[i]))] = int(d[i])

    worst_ratio = 0.0
    worst_q = None
    for _ in range(n_query_rects):
        # Sample a random query rectangle
        qx1 = random.randint(0, n - 2)
        qx2 = random.randint(qx1 + 1, n - 1)
        qy1 = random.randint(0, n - 2)
        qy2 = random.randint(qy1 + 1, n - 1)

        # Count cells inside Q
        inside_cells = []
        for x in range(qx1, qx2 + 1):
            for y in range(qy1, qy2 + 1):
                inside_cells.append(cell_d[(x, y)])
        inside_set = set(inside_cells)

        # Count segments crossing Q's boundary
        boundary_crossings = 0
        total_segments = 0
        for x in range(qx1, qx2 + 1):
            for y in range(qy1, qy2 + 1):
                d_curr = cell_d[(x, y)]
                # Check 4 neighbors
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < n:
                        d_neighbor = cell_d[(nx, ny)]
                        # Segment from (x, y) to (nx, ny)
                        # Check if it's a boundary crossing:
                        # Inside if both endpoints in Q
                        in_curr = d_curr in inside_set
                        in_neighbor = d_neighbor in inside_set
                        # Adjacent if d_neighbor == d_curr ± 1 in 1D sequence
                        adj = abs(d_neighbor - d_curr) == 1
                        if adj and in_curr != in_neighbor:
                            # Adjacent pair, one inside, one outside Q
                            boundary_crossings += 1
                        if adj:
                            total_segments += 1
        if total_segments > 0:
            ratio = boundary_crossings / total_segments
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_q = (qx1, qy1, qx2, qy2)

    return worst_ratio


def main():
    """Recompute δ_2(H) on 8x8, 32x32, 64x64 grids."""
    print("=" * 70)
    print("v1.3 STANDARD: Moon δ_2(H) numerical recalculation")
    print("Reference: Moon et al. 2001 IEEE TKDE 13(1): 124-141")
    print("=" * 70)

    results = {}
    for grid_size in [8, 32, 64]:
        print(f"\n[Grid {grid_size}x{grid_size}]")
        delta2 = estimate_delta2(grid_size, n_query_rects=500)
        results[grid_size] = delta2
        print(f"  δ_2(H) estimate: {delta2:.4f}")
        print(f"  (Bauman 2006 / HilbertNet 2022 reference: ~6.533)")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for n, d in results.items():
        print(f"  {n}x{n}: δ_2(H) ≈ {d:.4f}")
    print(f"\nLiterature reference: 6.533 (Bauman 2006 / HilbertNet ECCV 2022)")
    print("\nNote: This is a Monte Carlo estimate. For exact δ_2, see")
    print("Moon et al. 2001 closed-form (Eq. 17 in their paper).")
    print("The 6.533 reference is widely cited but its exact derivation")
    print("depends on query-rectangle parameters (corner vs interior, etc.).")

    return results


if __name__ == "__main__":
    main()
