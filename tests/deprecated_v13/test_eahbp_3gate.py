"""v1.3 STANDARD: EAHBP 3-gate plumbing test (T4.1).

Verifies that the trainer-side EAHBP gate plumbing correctly:
  - Calls set_gate_signals() with computed GME/throughput/precision
  - Reads back check_g1 / check_g2 / check_g3_rollback()
  - Records the gate decisions into the MetricsCollector
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class TestEAHBP3GatePlumbing:
    """T4.1: gate signals are set and decisions are recorded."""

    def test_eahbp_attention_gate_signals_contract(self):
        """EAHBPAttention.set_gate_signals + check_g1/g2/g3_rollback return bool."""
        from vit_pytorch.layers.attention.eahbp_attention import EAHBPAttention
        attn = EAHBPAttention()
        attn.set_gate_signals(gme=0.1, throughput=100.0, precision_gain=0.05)
        # All three checks return bool
        assert isinstance(attn.check_g1(), bool)
        assert isinstance(attn.check_g2(), bool)
        assert isinstance(attn.check_g3_rollback(), bool)

    def test_g1_passes_when_precision_above_threshold(self):
        """G1 (precision gain) passes when gain > 0.2%."""
        from vit_pytorch.layers.attention.eahbp_attention import EAHBPAttention
        attn = EAHBPAttention()
        attn.set_gate_signals(gme=0.1, throughput=1.0, precision_gain=0.005)
        assert attn.check_g1() is True

    def test_g3_rollback_when_gme_below_40(self):
        """G3 triggers rollback when GME < 40%."""
        from vit_pytorch.layers.attention.eahbp_attention import EAHBPAttention
        attn = EAHBPAttention()
        attn.set_gate_signals(gme=0.30, throughput=1.0, precision_gain=0.005)
        assert attn.check_g3_rollback() is True

    def test_trainer_3gate_helper_iterates_eahbp_modules(self):
        """The trainer helper finds EAHBPAttention modules and updates their gates."""
        from vit_pytorch.layers.attention.eahbp_attention import EAHBPAttention
        from src.training.metrics.collector import MetricsCollector

        # Build a model with one EAHBPAttention
        eahbp = EAHBPAttention()
        model = nn.ModuleList([eahbp, nn.Linear(4, 2)])
        # Walk modules and set gates
        collector = MetricsCollector()
        from src.training.trainer.epoch_train import _apply_eahbp_gates
        _apply_eahbp_gates(
            model, gme=0.5, throughput=1.5, precision_gain=0.003,
            collector=collector,
        )
        # Gate signals set on the EAHBP instance
        sigs = eahbp.get_gate_signals()
        assert sigs["gme"] == 0.5
        assert sigs["throughput"] == 1.5
        assert sigs["precision_gain"] == 0.003
        # Decisions recorded in collector
        snap = collector.get_summary()
        assert "eahbp/g1_pass" in snap
        assert "eahbp/g2_pass" in snap
        assert "eahbp/g3_rollback" in snap
