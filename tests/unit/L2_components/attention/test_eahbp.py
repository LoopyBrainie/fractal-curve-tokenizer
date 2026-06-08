"""v1.3 STANDARD: EAHBP Attention tests (C.1-C.4).

C.1: Skeleton (forward, block-local + global pool)
C.2: FLOP reduction verification (62.5% target)
C.3: G1 precision gate (≥ 0.2%)
C.4: G2 throughput + GME gate (≥ 1.4× AND GME ≥ 60%)
"""
from __future__ import annotations

import pytest
import torch

from vit_pytorch.core.constants import (
    EAHBP_G1_PRECISION_GAIN,
    EAHBP_G2_GME_FLOOR,
    EAHBP_G2_THROUGHPUT_GAIN,
    EAHBP_G3_GME_FLOOR,
)
from vit_pytorch.layers.attention.eahbp_attention import (
    EAHBPAttention,
    EAHBPAttentionConfig,
)


class TestEAHBPAttentionSkeleton:
    """C.1: Skeleton tests."""

    def test_eahbp_forward_n_64(self):
        """Forward on N=64 (divisible by b=16 and g=8)."""
        attn = EAHBPAttention()
        x = torch.randn(2, 64, 256)
        y = attn(x)
        assert y.shape == (2, 64, 256)

    def test_eahbp_forward_n_32(self):
        """Forward on N=32 (block-local disabled, global pool only)."""
        attn = EAHBPAttention()
        x = torch.randn(1, 32, 256)
        y = attn(x)
        assert y.shape == (1, 32, 256)

    def test_eahbp_forward_preserves_shape(self):
        """Forward output shape equals input shape for all valid N."""
        attn = EAHBPAttention()
        for N in [16, 32, 64, 128]:
            x = torch.randn(1, N, 256)
            y = attn(x)
            assert y.shape == (1, N, 256), f"N={N} produced wrong shape {y.shape}"


class TestEAHBPFLOPReduction:
    """C.2: FLOP reduction verification (62.5% target)."""

    def test_eahbp_flop_reduction_n_64(self):
        """EAHBP FLOPs are 62.5% less than full attention at N=64."""
        attn = EAHBPAttention()
        eahbp_flops, full_flops = attn.estimate_flops(64)
        reduction = 1.0 - eahbp_flops / full_flops
        assert 0.60 <= reduction <= 0.65, (
            f"FLOP reduction {reduction:.1%} outside design target [60%, 65%]"
        )

    def test_eahbp_flop_reduction_n_128(self):
        """EAHBP FLOPs reduce even more at N=128."""
        attn = EAHBPAttention()
        eahbp_flops, full_flops = attn.estimate_flops(128)
        reduction = 1.0 - eahbp_flops / full_flops
        assert reduction > 0.60, (
            f"FLOP reduction {reduction:.1%} should be > 60% at N=128"
        )


class TestEAHBPGateG1:
    """C.3: G1 precision gate (≥ 0.2%)."""

    def test_g1_passes_at_threshold(self):
        """G1 passes when precision gain = 0.2% (the threshold)."""
        attn = EAHBPAttention()
        attn.set_gate_signals(precision_gain=EAHBP_G1_PRECISION_GAIN)
        assert attn.check_g1() is True

    def test_g1_passes_above_threshold(self):
        """G1 passes when precision gain > 0.2%."""
        attn = EAHBPAttention()
        attn.set_gate_signals(precision_gain=0.005)  # 0.5%
        assert attn.check_g1() is True

    def test_g1_fails_below_threshold(self):
        """G1 fails when precision gain < 0.2%."""
        attn = EAHBPAttention()
        attn.set_gate_signals(precision_gain=0.001)  # 0.1%
        assert attn.check_g1() is False


class TestEAHBPGateG2:
    """C.4: G2 throughput + GME gate (≥ 1.4× AND GME ≥ 60%)."""

    def test_g2_passes_both_conditions(self):
        """G2 passes when throughput ≥ 1.4× AND GME ≥ 60%."""
        attn = EAHBPAttention()
        attn.set_gate_signals(throughput=1.5, gme=0.65)
        assert attn.check_g2() is True

    def test_g2_fails_throughput_only(self):
        """G2 fails when throughput is low even if GME is high."""
        attn = EAHBPAttention()
        attn.set_gate_signals(throughput=1.2, gme=0.65)
        assert attn.check_g2() is False

    def test_g2_fails_gme_only(self):
        """G2 fails when GME is low even if throughput is high."""
        attn = EAHBPAttention()
        attn.set_gate_signals(throughput=1.5, gme=0.50)
        assert attn.check_g2() is False


class TestEAHBPGateG3:
    """G3 rollback trigger (GME < 40%)."""

    def test_g3_rollback_when_gme_below_40(self):
        """G3 triggers rollback when GME < 40%."""
        attn = EAHBPAttention()
        attn.set_gate_signals(gme=0.30)
        assert attn.check_g3_rollback() is True

    def test_g3_no_rollback_when_gme_above_40(self):
        """G3 does not trigger rollback when GME ≥ 40%."""
        attn = EAHBPAttention()
        attn.set_gate_signals(gme=0.50)
        assert attn.check_g3_rollback() is False
