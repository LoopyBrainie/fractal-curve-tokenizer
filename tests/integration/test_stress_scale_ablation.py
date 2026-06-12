"""v1.3 STANDARD Phase 0: Stress-Scale Ablation test skeleton (JVP-5).

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.11.4:
  "Multi-Block HMFT must maintain token count in [K_min, K_max] = [8, 64]
   for N = 64, 128, 256 stress-scale ablation"

This test establishes the H1SS baseline before Phase 1 introduces
Multi-Block HMFT. Phase 1's tests/unit/L2_components/splitters/test_multi_block_hmft_poc.py
will reuse this skeleton with @pytest.mark.phase1.

Design token range (v1.3 §9):
  K_min = 8 (lower bound, even on uniform regions)
  K_max = 64 (upper bound, for complex regions)

Hard safety limits (constants.py):
  K_MIN_HARD_LIMIT = 8
  K_MAX_HARD_LIMIT = 8192
"""
from __future__ import annotations

import pytest
import torch

from vit_pytorch import FractalCurveViT


# v1.3 STANDARD: design working range (different from hard safety limits)
K_MIN_DESIGN = 8
K_MAX_DESIGN = 64

# JVP-5 ablation scales
STRESS_SCALE_GRIDS = [64, 128, 256]


@pytest.mark.parametrize("N", STRESS_SCALE_GRIDS)
def test_h1ss_stress_scale_baseline_forward(N: int):
    """H1SS forward succeeds at N=64, 128, 256.

    This is the JVP-5 baseline. Phase 1 will add a parallel test
    for Multi-Block HMFT that uses the same N values.
    """
    model = FractalCurveViT(
        image_size=N,
        num_classes=10,
        dim=64,
        mlp_dim=128,
        num_layers=2,
        heads=2,
    )
    model.train()
    x = torch.randn(2, 3, N, N)
    # Returns (ForwardOutput, MetricsTensors) per FractalCurveViT design
    forward_output = model(x)
    assert forward_output.logits.shape == (2, 10), (
        f"Expected logits shape (2, 10) at N={N}, got {forward_output.logits.shape}"
    )


@pytest.mark.parametrize("N", STRESS_SCALE_GRIDS)
def test_h1ss_stress_scale_token_count_bounds(N: int):
    """H1SS num_tokens stays in [K_min, K_max] = [8, 64] design range.

    Per v1.3 §9, adaptive tokenization must respect [8, 64] bounds
    regardless of input resolution. This is the A1-A5 axiom verification
    at scale.
    """
    model = FractalCurveViT(
        image_size=N,
        num_classes=10,
        dim=64,
        mlp_dim=128,
        num_layers=2,
        heads=2,
    )
    model.train()
    x = torch.randn(2, 3, N, N)
    forward_output = model(x)
    num_tokens = forward_output.num_tokens
    # num_tokens can be int, list[int], or tensor depending on impl branch.
    # Normalize to per-batch int values for bounds checking.
    if isinstance(num_tokens, torch.Tensor):
        per_batch = num_tokens.flatten().tolist()
    elif isinstance(num_tokens, int):
        per_batch = [num_tokens]
    else:
        per_batch = list(num_tokens)
    for batch_idx, nt in enumerate(per_batch):
        assert K_MIN_DESIGN <= nt <= K_MAX_DESIGN, (
            f"num_tokens[{batch_idx}]={nt} out of design range "
            f"[{K_MIN_DESIGN}, {K_MAX_DESIGN}] at N={N}"
        )


@pytest.mark.parametrize("N", STRESS_SCALE_GRIDS)
def test_h1ss_stress_scale_gradient_flow(N: int):
    """H1SS gradient flows end-to-end at N=64, 128, 256.

    Verifies STE gradient path is intact at scale (no broken backward
    when N grows). This protects against silent gradient corruption
    that would only surface in larger training runs.
    """
    model = FractalCurveViT(
        image_size=N,
        num_classes=10,
        dim=64,
        mlp_dim=128,
        num_layers=2,
        heads=2,
    )
    model.train()
    x = torch.randn(1, 3, N, N, requires_grad=True)
    forward_output = model(x)
    loss = forward_output.logits.sum()
    loss.backward()
    assert x.grad is not None, f"No gradient at input for N={N}"
    assert torch.isfinite(x.grad).all(), f"NaN/Inf in input gradient at N={N}"
