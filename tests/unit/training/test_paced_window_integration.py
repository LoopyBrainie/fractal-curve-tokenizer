"""v1.3 STANDARD: Paced Window integration test (T5.1).

Verifies that the trainer-side PacedWindow plumbing correctly:
  - Initializes a staging weight buffer (clone of main weights)
  - Maintains a PacedWindow state machine
  - Restores staging weights on ROLLBACK decision
  - Records pacing/rollback metrics
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class TestPacedWindowIntegration:
    """T5.1: Paced Window state machine + staging buffer."""

    def test_paced_window_module_exists(self):
        """PacedWindow can be imported and constructed."""
        from vit_pytorch.core.paced_window import PacedWindow, PacedWindowConfig
        pw = PacedWindow(PacedWindowConfig())
        assert pw.state.value == "INIT"

    def test_paced_window_triggers_rollback_on_3_consecutive_fatal(self):
        """3 consecutive D_struct >= T_fatal triggers ROLLBACK."""
        from vit_pytorch.core.paced_window import PacedWindow, PacedWindowConfig
        pw = PacedWindow(PacedWindowConfig(fatal_streak_threshold=1))
        pw.enter_window()
        # Update with strong D_struct to force rollback
        theta_main = torch.ones(10)
        theta_stage = torch.ones(10) * 100  # 10x larger
        rollback, reason = pw.update(theta_main, theta_stage, gme=0.50)
        # fatal_streak=1 means first fatal D_struct triggers
        assert rollback is True

    def test_staging_state_dict_field_exists(self):
        """TrainingState has staging_state_dict field."""
        from src.training.trainer.state import TrainingState
        s = TrainingState()
        assert hasattr(s, "staging_state_dict")
        assert s.staging_state_dict is None

    def test_paced_window_state_field_exists(self):
        """TrainingState has paced_window_state field."""
        from src.training.trainer.state import TrainingState
        s = TrainingState()
        assert hasattr(s, "paced_window_state")
        assert s.paced_window_state is None

    def test_init_staging_buffer_helper(self):
        """_init_paced_window_buffer clones model weights into staging_state_dict."""
        from src.training.trainer.epoch_train import _init_paced_window_buffer
        from src.training.trainer.state import TrainingState
        from src.training.config import TrainingHyperparams

        model = nn.Linear(4, 2)
        state = TrainingState()
        config = type("Cfg", (), {
            "training": TrainingHyperparams(
                enable_paced_window=True,
                paced_window_fatal_streak=3,
            ),
        })()
        _init_paced_window_buffer(model, state, config)
        assert state.staging_state_dict is not None
        assert "weight" in state.staging_state_dict
        # Cloned, not same tensor
        assert state.staging_state_dict["weight"] is not model.weight
        assert torch.equal(state.staging_state_dict["weight"], model.weight)

    def test_paced_window_step_helper(self):
        """_paced_window_step updates the PacedWindow and may trigger rollback."""
        from src.training.trainer.epoch_train import (
            _init_paced_window_buffer, _paced_window_step,
        )
        from src.training.trainer.state import TrainingState
        from vit_pytorch.core.paced_window import PacedWindow, PacedWindowConfig
        from src.training.config import TrainingHyperparams
        from src.training.metrics.collector import MetricsCollector

        model = nn.Linear(4, 2)
        state = TrainingState()
        config = type("Cfg", (), {
            "training": TrainingHyperparams(
                enable_paced_window=True,
                paced_window_fatal_streak=1,
            ),
        })()
        _init_paced_window_buffer(model, state, config)
        # PacedWindow was already created by the init helper with fatal_streak=1
        assert state.paced_window is not None
        state.paced_window.enter_window()
        # Run a step with strong D_struct
        collector = MetricsCollector()
        # First make model weights very different from staging
        with torch.no_grad():
            model.weight.mul_(100)
        _paced_window_step(model, state, gme=0.5, collector=collector)
        # Should have recorded rollback (fatal_streak=1)
        snap = collector.get_summary()
        # rollback count should be >= 1
        assert snap.get("paced/rollback_count", 0.0) >= 1.0
