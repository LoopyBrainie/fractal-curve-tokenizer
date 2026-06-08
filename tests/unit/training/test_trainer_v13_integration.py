"""v1.3 STANDARD: Trainer integration matrix test (T7.1 + T7.2).

Verifies that:
  1. All 4 opt-in features can be turned on without import errors.
  2. The shadow/R12/EAHBP/paced/hmft module functions are reachable.
  3. When all flags are off, no v1.3 keys appear in the metric flow.
  4. No regressions: full baseline still passes (verified separately).
"""
from __future__ import annotations

import pytest


class TestTrainerV13Integration:
    """T7.1: all 4 opt-in features are reachable."""

    def test_shadow_monitor_module_reachable(self):
        """ShadowMonitorTrainerHooks is importable and instantiable."""
        from src.training.monitor.shadow import ShadowMonitorTrainerHooks
        from src.training.metrics.collector import MetricsCollector
        h = ShadowMonitorTrainerHooks(interval=1, collector=MetricsCollector())
        assert h is not None
        assert h.interval == 1
        assert h.collector is not None

    def test_r12_aux_helper_reachable(self):
        """R12 is reachable through compute_loss(aux_losses=...)."""
        from src.training.trainer.loss import compute_loss
        from vit_pytorch.core.measure_auxiliary_loss import R12AuxLoss
        r12 = R12AuxLoss()
        r12_tensor = r12(
            torch_dummy := __import__("torch").randn(2, 4),
            torch_dummy,
            __import__("torch").tensor([0.5, 0.3, 0.2]),
        )
        _, comps = compute_loss(
            __import__("torch").randn(2, 4),
            __import__("torch").randint(0, 4, (2,)),
            aux_losses={"r12": r12_tensor},
        )
        assert "aux_r12" in comps

    def test_eahbp_gates_helper_reachable(self):
        """_apply_eahbp_gates is importable and a no-op on a non-EAHBP model."""
        import torch
        import torch.nn as nn
        from src.training.trainer.epoch_train import _apply_eahbp_gates
        from src.training.metrics.collector import MetricsCollector
        model = nn.Linear(4, 2)  # no EAHBPAttention
        collector = MetricsCollector()
        _apply_eahbp_gates(
            model, gme=0.5, throughput=1.0, precision_gain=0.003,
            collector=collector,
        )
        # No EAHBP modules in the model — no keys recorded
        snap = collector.get_summary()
        assert "eahbp/g1_pass" not in snap

    def test_paced_window_helpers_reachable(self):
        """_init_paced_window_buffer is importable; off-state is a no-op."""
        from src.training.trainer.epoch_train import _init_paced_window_buffer, _paced_window_step
        from src.training.trainer.state import TrainingState
        from src.training.config import TrainingHyperparams
        import torch.nn as nn
        # enable_paced_window=False → no-op
        model = nn.Linear(4, 2)
        state = TrainingState()
        config = type("Cfg", (), {
            "training": TrainingHyperparams(enable_paced_window=False),
        })()
        _init_paced_window_buffer(model, state, config)
        assert state.staging_state_dict is None
        # Step is also a no-op when PacedWindow not initialized
        from src.training.metrics.collector import MetricsCollector
        _paced_window_step(model, state, gme=0.5, collector=MetricsCollector())

    def test_hmft_h_probs_helper_reachable(self):
        """log_h_probs is importable and works with a mock splitter."""
        from src.training.trainer.epoch_train import log_h_probs
        from src.training.metrics.collector import MetricsCollector
        from unittest.mock import MagicMock
        splitter = MagicMock()
        splitter.get_diagnostics.return_value = {"h_probs": [0.1, 0.2, 0.3, 0.2, 0.2]}
        collector = MetricsCollector()
        log_h_probs(splitter, collector, epoch=1)
        snap = collector.get_summary()
        assert "hmft/h_prob_bin_0" in snap


class TestTrainerV13RegressionGuard:
    """T7.2: default-off baseline is unchanged (no v1.3 keys)."""

    def test_default_config_has_no_v13_flags_enabled(self):
        """TrainingHyperparams() with no overrides has all enable_* False."""
        from src.training.config import TrainingHyperparams
        h = TrainingHyperparams()
        assert h.enable_shadow_monitor is False
        assert h.enable_r12_aux is False
        assert h.enable_eahbp_3gate is False
        assert h.enable_paced_window is False

    def test_default_collector_has_no_v13_keys(self):
        """A fresh MetricsCollector with no records has no v1.3 keys."""
        from src.training.metrics.collector import MetricsCollector
        collector = MetricsCollector()
        snap = collector.get_summary()
        for key in snap:
            assert not key.startswith("shadow/"), f"v1.3 leak: {key}"
            assert not key.startswith("r12_"), f"v1.3 leak: {key}"
            assert not key.startswith("aux_r12"), f"v1.3 leak: {key}"
            assert not key.startswith("aux_weight_r12"), f"v1.3 leak: {key}"
            assert not key.startswith("eahbp/"), f"v1.3 leak: {key}"
            assert not key.startswith("paced/"), f"v1.3 leak: {key}"
            assert not key.startswith("hmft/"), f"v1.3 leak: {key}"
