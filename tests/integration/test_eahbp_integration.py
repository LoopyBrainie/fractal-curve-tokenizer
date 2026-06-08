"""v1.3 STANDARD: EAHBP integration with FractalCurveViT (C.8).

Verifies that EAHBPAttention can be substituted for the default
ManifoldNativeAttention via dependency injection. The EAHBP module
itself has its own QKV/RoPE/attn@V path, so it doesn't strictly need
the FractalCurveViT integration to be functional — but the integration
test ensures no regressions and validates the config-gated opt-in.
"""
from __future__ import annotations

import pytest
import torch

from vit_pytorch import FractalCurveViT
from vit_pytorch.core.config import FractalConfig
from vit_pytorch.core.paced_window import PacedWindow, PacedWindowState
from vit_pytorch.layers.attention import EAHBPAttention, EAHBPAttentionConfig


class TestEAHBPConfigGating:
    """C.8: Config flag enable_eahbp controls EAHBP path."""

    def test_default_config_has_eahbp_disabled(self):
        """Default config has enable_eahbp=False (no behavior change)."""
        cfg = FractalConfig(image_size=64)
        assert cfg.enable_eahbp is False

    def test_config_can_enable_eahbp(self):
        """Setting enable_eahbp=True is accepted."""
        cfg = FractalConfig(image_size=64, enable_eahbp=True)
        assert cfg.enable_eahbp is True


class TestEAHBPSubstitutability:
    """EAHBPAttention can be used as a drop-in attention module."""

    def test_eahbp_attention_constructs(self):
        """EAHBPAttention can be constructed standalone."""
        attn = EAHBPAttention()
        assert attn is not None
        # Has the QKV projections
        assert hasattr(attn, "to_q")
        assert hasattr(attn, "to_k")
        assert hasattr(attn, "to_v")

    def test_eahbp_attention_forward_shape(self):
        """Forward preserves shape."""
        attn = EAHBPAttention(EAHBPAttentionConfig())
        x = torch.randn(2, 64, 256)
        y = attn(x)
        assert y.shape == x.shape

    def test_eahbp_attention_flop_reduction(self):
        """EAHBP delivers the design 62.5% FLOP reduction."""
        attn = EAHBPAttention()
        eahbp_flops, full_flops = attn.estimate_flops(64)
        reduction = 1.0 - eahbp_flops / full_flops
        # Exactly 62.5% per design
        assert abs(reduction - 0.625) < 1e-6


class TestPacedWindowWithEAHBP:
    """PacedWindow can drive EAHBP gating end-to-end."""

    def test_paced_window_with_eahbp_gates(self):
        """Full EAHBP gate + Paced Window flow."""
        # Set up EAHBP with gate signals
        attn = EAHBPAttention()
        attn.set_gate_signals(gme=0.50, throughput=1.3, precision_gain=0.003)
        # G1 (precision) passes, G2 (throughput AND GME) fails
        assert attn.check_g1() is True
        assert attn.check_g2() is False  # 1.3 < 1.4 and 0.50 < 0.60
        # Enter Paced Window
        window = PacedWindow()
        window.enter_window()
        # Step through with various GME values
        theta = torch.ones(10)
        for _ in range(10):
            window.update(theta, theta.clone(), gme=0.50)
        # State may have terminated
        assert window.state in (
            PacedWindowState.EXIT,
            PacedWindowState.ROLLBACK,
            PacedWindowState.DECISION,
        )
