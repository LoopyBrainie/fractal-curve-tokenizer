# -*- coding: utf-8 -*-
"""Tests for Shadow Monitor (v1.3 §9.7 / B.11).

Validates:
  - test_shadow_monitor_wba_runs: wba_entropy is scalar tensor
  - test_shadow_monitor_mi_runs: mi_lower_bound is scalar tensor
  - test_shadow_monitor_non_blocking: does not break gradient flow
  - test_shadow_monitor_windowed: different window sizes give different WBA
"""

from __future__ import annotations

import math

import pytest
import torch

from vit_pytorch.core.shadow_monitor import ShadowMonitor


class TestShadowMonitor:
    """Shadow Monitor (G1 WBA + G2 RoPE InfoNCE) test suite."""

    @pytest.fixture
    def monitor(self) -> ShadowMonitor:
        return ShadowMonitor(
            window_size=16,
            delta_washout=0.05,
            epsilon_leak_factor=0.25,
        )

    @pytest.fixture
    def attention_scores(self) -> torch.Tensor:
        # [B, H, N, N] attention scores (pre-softmax logits)
        torch.manual_seed(0)
        return torch.randn(2, 4, 32, 32)

    @pytest.fixture
    def rope_embeddings(self) -> torch.Tensor:
        # [B, N, 2D] packed (Q_rope, K_rope)
        torch.manual_seed(1)
        return torch.randn(2, 32, 64)  # D = 32

    def test_shadow_monitor_wba_runs(
        self,
        monitor: ShadowMonitor,
        attention_scores: torch.Tensor,
        rope_embeddings: torch.Tensor,
    ) -> None:
        """G1 WBA entropy is a scalar tensor."""
        wba, _ = monitor(attention_scores, rope_embeddings, batch_size=2)
        assert isinstance(wba, torch.Tensor)
        assert wba.ndim == 0, f"wba_entropy must be scalar, got shape {tuple(wba.shape)}"
        # Entropy is non-negative and bounded by log(W) for the within-window
        # softmax. With W=16, the upper bound is ~2.77. Washout makes it slightly
        # smaller: ~2.63. Allow a generous headroom for FP drift.
        assert wba.item() >= 0.0
        assert wba.item() < math.log(16) + 1e-3

    def test_shadow_monitor_mi_runs(
        self,
        monitor: ShadowMonitor,
        attention_scores: torch.Tensor,
        rope_embeddings: torch.Tensor,
    ) -> None:
        """G2 InfoNCE MI lower bound is a scalar tensor."""
        _, mi = monitor(attention_scores, rope_embeddings, batch_size=2)
        assert isinstance(mi, torch.Tensor)
        assert mi.ndim == 0, f"mi_lower_bound must be scalar, got shape {tuple(mi.shape)}"
        # The value is non-negative: epsilon_leak_factor >= 0, log(B) >= log(2) > 0,
        # and InfoNCE loss is non-negative.
        assert mi.item() >= 0.0

    def test_shadow_monitor_non_blocking(
        self,
        attention_scores: torch.Tensor,
        rope_embeddings: torch.Tensor,
    ) -> None:
        """Shadow Monitor MUST NOT break gradient flow on its inputs.

        Scenario: input tensors carry `requires_grad=True`. The shadow monitor
        must return detached scalar tensors AND not raise an error trying to
        differentiate through a `no_grad` context. A correct implementation
        will not propagate gradients; this is verified by checking that
        calling `monitor(...)` does not produce a non-scalar gradient w.r.t.
        the inputs (because the returned tensors are detached from the graph).
        """
        # Sanity: a leaf tensor with requires_grad can be backwarded through.
        a = attention_scores.clone().detach().requires_grad_(True)
        r = rope_embeddings.clone().detach().requires_grad_(True)

        # Use a fresh, parameter-free monitor to avoid any side effects.
        m = ShadowMonitor(window_size=8, delta_washout=0.05, epsilon_leak_factor=0.25)
        wba, mi = m(a, r, batch_size=a.shape[0])

        # Both metrics are scalars, but we explicitly check they are detached:
        # the returned tensor's grad_fn must be None because @torch.no_grad
        # was used inside the forward pass.
        assert wba.grad_fn is None, "wba_entropy must be detached from autograd graph"
        assert mi.grad_fn is None, "mi_lower_bound must be detached from autograd graph"

        # Calling .backward() on the metric would fail if it were connected to
        # a graph — confirm that explicitly:
        with pytest.raises(RuntimeError):
            wba.backward()
        with pytest.raises(RuntimeError):
            mi.backward()

        # The original inputs must remain valid leaves with requires_grad=True
        # and an empty .grad buffer (no propagation happened).
        assert a.requires_grad is True
        assert r.requires_grad is True
        assert a.grad is None
        assert r.grad is None

    def test_shadow_monitor_windowed(
        self,
        attention_scores: torch.Tensor,
        rope_embeddings: torch.Tensor,
    ) -> None:
        """Different window sizes give different WBA entropy values.

        Mathematical justification: the within-window softmax normalizes over
        `W` elements. As W grows, more low-probability mass appears in the
        window, which monotonically increases the entropy of the within-
        window distribution. After multiplying by (1 - delta_washout), the
        ordering is preserved, so a larger W gives a larger WBA entropy.
        """
        m_small = ShadowMonitor(window_size=4, delta_washout=0.0)
        m_large = ShadowMonitor(window_size=16, delta_washout=0.0)

        wba_small, _ = m_small(attention_scores, rope_embeddings, batch_size=2)
        wba_large, _ = m_large(attention_scores, rope_embeddings, batch_size=2)

        # Sanity: both are scalars and finite.
        assert wba_small.ndim == 0
        assert wba_large.ndim == 0
        assert torch.isfinite(wba_small)
        assert torch.isfinite(wba_large)

        # Different windows must produce different entropies on the same input.
        # (Strictly: a larger window yields larger entropy because uniform
        # mass on more elements has higher entropy than concentrated mass on
        # fewer elements — see Cover & Thomas, Elements of Information Theory,
        # Theorem 2.7.2.)
        assert not torch.isclose(wba_small, wba_large, atol=1e-5), (
            f"WBA entropy unexpectedly identical for W=4 and W=16: "
            f"{wba_small.item()} vs {wba_large.item()}"
        )
        # In expectation W=16 > W=4 for non-pathological inputs.
        assert wba_large.item() > wba_small.item(), (
            f"W=16 WBA ({wba_large.item():.4f}) should exceed W=4 WBA "
            f"({wba_small.item():.4f})"
        )

    def test_shadow_monitor_default_constants(
        self,
        attention_scores: torch.Tensor,
        rope_embeddings: torch.Tensor,
    ) -> None:
        """Default constructor reads from v1.3 constants."""
        m = ShadowMonitor()  # no args
        assert m.window_size == 16
        assert abs(m.delta_washout - 0.05) < 1e-9
        assert abs(m.epsilon_leak_factor - 0.25) < 1e-9
        # Forward must still run.
        wba, mi = m(attention_scores, rope_embeddings, batch_size=2)
        assert wba.ndim == 0
        assert mi.ndim == 0

    def test_shadow_monitor_no_learnable_params(self) -> None:
        """Shadow Monitor must have no learnable parameters (observer-only)."""
        m = ShadowMonitor()
        params = list(m.parameters())
        assert len(params) == 0, (
            f"ShadowMonitor must be parameter-free, but has {len(params)} params"
        )
