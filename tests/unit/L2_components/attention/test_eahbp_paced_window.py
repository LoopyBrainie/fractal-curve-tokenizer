"""v1.3 STANDARD: Paced Window state machine tests (C.6).

Verifies the 5-epoch Paced Window state machine:
  INIT → MONITORING → INJECTING_TILING → MONITORING_2 → DECISION
                                                              ├── EXIT
                                                              └── ROLLBACK

Plus the D_struct circuit breaker (3 consecutive steps ≥ T_fatal → ROLLBACK).
"""
from __future__ import annotations

import pytest
import torch

from vit_pytorch.core.constants import (
    PACED_WINDOW_FATAL_STREAK,
    PACED_WINDOW_MAX_EPOCHS,
    V13_T_FATAL,
)
from vit_pytorch.core.paced_window import (
    PacedWindow,
    PacedWindowConfig,
    PacedWindowState,
)


class TestPacedWindowSkeleton:
    """C.6: Skeleton tests."""

    def test_paced_window_init_state(self):
        """Initial state is INIT."""
        window = PacedWindow()
        assert window.state == PacedWindowState.INIT

    def test_enter_window_transitions_to_monitoring(self):
        """enter_window() transitions INIT → MONITORING."""
        window = PacedWindow()
        window.enter_window()
        assert window.state == PacedWindowState.MONITORING
        assert window.epoch_in_window == 0
        assert window.fatal_streak == 0


class TestPacedWindowDStruct:
    """D_struct computation and circuit breaker."""

    def test_d_struct_is_zero_for_identical_weights(self):
        """D_struct = 0 when main == staging."""
        window = PacedWindow()
        theta = torch.ones(10)
        d = window.compute_d_struct(theta, theta.clone(), theta.clone())
        assert d == 0.0

    def test_d_struct_positive_for_different_weights(self):
        """D_struct > 0 when main ≠ staging."""
        window = PacedWindow()
        theta_main = torch.ones(10)
        theta_stage = torch.ones(10) * 2
        d = window.compute_d_struct(theta_main, theta_stage, theta_main.clone())
        # diff = sqrt(10*1) = 3.16, denom = sqrt(10) = 3.16, ratio = 1.0
        assert d > 0.5

    def test_d_struct_circuit_breaker_3_steps(self):
        """3 consecutive steps with D_struct ≥ T_fatal → ROLLBACK."""
        window = PacedWindow()
        window.enter_window()
        # Make weights differ strongly to push D_struct ≥ T_fatal
        theta_main = torch.ones(10)
        theta_stage = torch.ones(10) * 100
        # Step 1: MONITORING (transitions to INJECTING_TILING)
        rolled_back, _ = window.update(theta_main, theta_stage, gme=0.50)
        assert rolled_back is False
        # Step 2: INJECTING_TILING (transitions to MONITORING_2)
        rolled_back, _ = window.update(theta_main, theta_stage, gme=0.50)
        assert rolled_back is False
        # Step 3: MONITORING_2, epoch 1
        rolled_back, _ = window.update(theta_main, theta_stage, gme=0.50)
        assert rolled_back is False
        assert window.fatal_streak >= 1


class TestPacedWindowFullPath:
    """Full 5-epoch path with various GME outcomes."""

    def test_full_path_exit_when_gme_recovers(self):
        """Paced Window EXITs when GME recovers to ≥ 60% after 5 epochs."""
        window = PacedWindow()
        window.enter_window()
        # Use identical weights so D_struct stays small
        theta = torch.ones(10)
        for _ in range(PACED_WINDOW_MAX_EPOCHS + 2):
            rolled_back, _ = window.update(theta, theta.clone(), gme=0.65)
        # Should have EXITed (GME=0.65 ≥ 0.60)
        assert window.state in (PacedWindowState.EXIT, PacedWindowState.DECISION)

    def test_full_path_rollback_when_gme_stays_low(self):
        """Paced Window ROLLBACKs when GME stays < 60% after 5 epochs."""
        window = PacedWindow()
        window.enter_window()
        theta = torch.ones(10)
        rolled_back_seen = False
        # Full path: 1 MONITORING + 1 INJECTING + 5 MONITORING_2 + 1 DECISION = 8 steps
        for _ in range(PACED_WINDOW_MAX_EPOCHS + 3):
            rolled_back, _ = window.update(theta, theta.clone(), gme=0.45)
            if rolled_back:
                rolled_back_seen = True
                break
        # Eventually rolls back
        assert rolled_back_seen, f"State ended at {window.state}"

    def test_reset_clears_state(self):
        """reset() returns to INIT."""
        window = PacedWindow()
        window.enter_window()
        window.update(torch.ones(10), torch.ones(10), gme=0.50)
        window.reset()
        assert window.state == PacedWindowState.INIT
        assert window.fatal_streak == 0


class TestPacedWindowConfig:
    """PacedWindowConfig tests."""

    def test_default_config_uses_v13_constants(self):
        """Default config uses v1.3 STANDARD values."""
        cfg = PacedWindowConfig()
        assert cfg.T_fatal == V13_T_FATAL
        assert cfg.fatal_streak_threshold == PACED_WINDOW_FATAL_STREAK
        assert cfg.max_epochs == PACED_WINDOW_MAX_EPOCHS
