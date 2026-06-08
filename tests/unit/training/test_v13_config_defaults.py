"""v1.3 STANDARD: Trainer config defaults test (T1.1).

Verifies that TrainingHyperparams has the v1.3 opt-in flags
all defaulting to False/their baseline values, ensuring zero
regression for users who don't opt in.
"""
from __future__ import annotations

import pytest


class TestV13ConfigDefaults:
    """v1.3 opt-in flags default to off / baseline values."""

    def test_v13_opt_in_flags_default_off(self):
        from src.training.config import TrainingHyperparams
        h = TrainingHyperparams()
        assert h.enable_shadow_monitor is False
        assert h.enable_r12_aux is False
        assert h.enable_eahbp_3gate is False
        assert h.enable_paced_window is False

    def test_v13_calibration_defaults(self):
        from src.training.config import TrainingHyperparams
        h = TrainingHyperparams()
        # R12 lambdas match vit_pytorch.core.constants defaults
        assert h.r12_lambda_tree == pytest.approx(0.10, abs=1e-9)
        assert h.r12_lambda_skew == pytest.approx(0.10, abs=1e-9)
        # Paced window fatal streak
        assert h.paced_window_fatal_streak == 3
        # Shadow monitor interval
        assert h.shadow_monitor_interval == 50

    def test_v13_config_preserves_existing_defaults(self):
        """Adding v1.3 fields must not break any existing field defaults."""
        from src.training.config import TrainingHyperparams
        h = TrainingHyperparams()
        # Sanity: existing fields still default
        assert h.num_epochs == 100
        assert h.batch_size == 128
        assert h.gradient_clip_norm == 5.0
        assert h.base_lr == pytest.approx(5e-4, abs=1e-9)
        assert h.budget_loss_weight == pytest.approx(0.05, abs=1e-9)
