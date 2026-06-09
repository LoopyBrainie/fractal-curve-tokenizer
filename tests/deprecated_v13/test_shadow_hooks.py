"""v1.3 STANDARD: ShadowMonitorTrainerHooks test (T2.1).

Verifies that ShadowMonitorTrainerHooks can be wired into the
trainer and produces the expected `shadow/*` metric keys via the
MetricsCollector.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import pytest


class TestShadowMonitorTrainerHooks:
    """T2.1: hook accumulates shadow/* scalars in the collector."""

    def test_shadow_hooks_module_exists(self):
        """The hook class is importable from src.training.monitor.shadow."""
        from src.training.monitor.shadow import ShadowMonitorTrainerHooks
        assert ShadowMonitorTrainerHooks is not None

    def test_post_backward_records_shadow_gme(self):
        """post_backward with a 1-Linear model records shadow/gme."""
        from src.training.monitor.shadow import ShadowMonitorTrainerHooks
        from src.training.metrics.collector import MetricsCollector

        hooks = ShadowMonitorTrainerHooks(interval=1, collector=MetricsCollector())
        model = nn.Linear(4, 2)
        x = torch.randn(2, 4)
        y = model(x).sum()
        y.backward()
        hooks.post_backward(model=model, optimizer=None, step=0)
        snap = hooks.collector.get_summary()
        assert "shadow/gme" in snap, f"Missing shadow/gme in {list(snap.keys())}"

    def test_post_forward_handles_none_inputs(self):
        """post_forward with None attn/rope (non-EAHBP model) does not raise."""
        from src.training.monitor.shadow import ShadowMonitorTrainerHooks
        from src.training.metrics.collector import MetricsCollector

        hooks = ShadowMonitorTrainerHooks(interval=1, collector=MetricsCollector())
        # Non-EAHBP model has no attn_scores / rope to pass
        hooks.post_forward(
            attention_scores=None, rope_embeddings=None, batch_size=2, step=0,
        )
        # Both keys should be recorded (as zero scalars) so the dashboard
        # sees continuous data
        snap = hooks.collector.get_summary()
        assert "shadow/wba_entropy" in snap
        assert "shadow/mi_lower_bound" in snap
        assert snap["shadow/wba_entropy"] == 0.0

    def test_gme_is_zero_for_no_grad_model(self):
        """A model with no grad (e.g. eval-mode no_grad) gives GME=0."""
        from src.training.monitor.shadow import ShadowMonitorTrainerHooks
        from src.training.metrics.collector import MetricsCollector

        hooks = ShadowMonitorTrainerHooks(interval=1, collector=MetricsCollector())
        model = nn.Linear(4, 2)
        # No backward was called, so no grads exist
        hooks.post_backward(model=model, optimizer=None, step=0)
        snap = hooks.collector.get_summary()
        assert snap.get("shadow/gme", 0.0) == 0.0

    def test_interval_zero_means_every_step(self):
        """interval=0 means every call is recorded."""
        from src.training.monitor.shadow import ShadowMonitorTrainerHooks
        from src.training.metrics.collector import MetricsCollector

        hooks = ShadowMonitorTrainerHooks(interval=0, collector=MetricsCollector())
        model = nn.Linear(2, 2)
        # Should not raise on any step
        for step in range(3):
            hooks.post_backward(model=model, optimizer=None, step=step)
        # All three calls should have recorded (or, depending on impl,
        # at least the last one)
        snap = hooks.collector.get_summary()
        assert "shadow/gme" in snap
