"""v1.3 STANDARD: Paced Optimization Window state machine (Phase 2 §9.4).

When EAHBP G2 fails (GME in [40%, 60%)), the system enters the
PACED_WINDOW state to give EAHBP a chance to converge:

  INIT → MONITORING → INJECTING_TILING → MONITORING_2 → DECISION
                                                              ├── EXIT (GME ≥ 60%, D_struct < T_fatal)
                                                              └── ROLLBACK (D_struct ≥ T_fatal for 3 consecutive steps)

D_struct = ||θ_main(t) - θ_stage(t)||² / (||θ_main(t-1)||² + ε)
       = 分布偏移度: main weight 与 staging weight 的相对偏差

Circuit breaker: D_struct ≥ T_fatal for PACED_WINDOW_FATAL_STREAK (3)
consecutive gradient steps → immediate Hard Rollback.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

from vit_pytorch.core.constants import (
    PACED_WINDOW_FATAL_STREAK,
    PACED_WINDOW_MAX_EPOCHS,
    V13_T_FATAL,
)


class PacedWindowState(str, Enum):
    """Paced Window state machine states."""
    INIT = "INIT"
    MONITORING = "MONITORING"
    INJECTING_TILING = "INJECTING_TILING"
    MONITORING_2 = "MONITORING_2"
    DECISION = "DECISION"
    EXIT = "EXIT"
    ROLLBACK = "ROLLBACK"


@dataclass
class PacedWindowConfig:
    """Configuration for PacedWindow."""
    T_fatal: float = V13_T_FATAL
    fatal_streak_threshold: int = PACED_WINDOW_FATAL_STREAK
    max_epochs: int = PACED_WINDOW_MAX_EPOCHS
    epsilon: float = 1e-8


class PacedWindow:
    """v1.3 STANDARD: Paced Optimization Window state machine.

    Tracks the state transitions for the 5-epoch EAHBP evaluation window
    when G2 fails (GME in [40%, 60%)). Computes D_struct on each gradient
    step and triggers Hard Rollback if D_struct ≥ T_fatal for 3 consecutive
    steps.

    Usage:
        window = PacedWindow()
        # Per step:
        should_rollback, reason = window.update(
            theta_main, theta_stage, gme
        )
        if should_rollback:
            # Hard rollback EAHBP
            ...
    """

    def __init__(self, config: Optional[PacedWindowConfig] = None) -> None:
        self._config = config or PacedWindowConfig()
        self._state: PacedWindowState = PacedWindowState.INIT
        self._epoch_in_window: int = 0
        self._fatal_streak: int = 0
        self._last_d_struct: float = 0.0
        self._last_theta_main_norm: float = 0.0
        self._d_struct_history: List[float] = []
        self._state_history: List[PacedWindowState] = [PacedWindowState.INIT]

    @property
    def state(self) -> PacedWindowState:
        return self._state

    @property
    def epoch_in_window(self) -> int:
        return self._epoch_in_window

    @property
    def last_d_struct(self) -> float:
        return self._last_d_struct

    @property
    def fatal_streak(self) -> int:
        return self._fatal_streak

    def enter_window(self) -> None:
        """Transition from INIT to MONITORING (called when G2 fails)."""
        if self._state == PacedWindowState.INIT:
            self._transition(PacedWindowState.MONITORING)
            self._epoch_in_window = 0
            self._fatal_streak = 0

    def compute_d_struct(
        self,
        theta_main: torch.Tensor,
        theta_stage: torch.Tensor,
        theta_main_prev: Optional[torch.Tensor] = None,
    ) -> float:
        """Compute D_struct = ||θ_main - θ_stage||² / (||θ_main_prev||² + ε).

        Args:
            theta_main: current main weights
            theta_stage: current staging weights
            theta_main_prev: previous-step main weights (or theta_main for first call)

        Returns:
            D_struct value (scalar)
        """
        diff = (theta_main - theta_stage).pow(2).sum().sqrt()
        if theta_main_prev is None:
            theta_main_prev = theta_main
        denom = theta_main_prev.pow(2).sum().sqrt() + self._config.epsilon
        d_struct = (diff / denom).item()
        self._last_d_struct = d_struct
        self._d_struct_history.append(d_struct)
        return d_struct

    def update(
        self,
        theta_main: torch.Tensor,
        theta_stage: torch.Tensor,
        gme: float,
        g2_gme_floor: float = 0.60,
        g3_gme_floor: float = 0.40,
    ) -> Tuple[bool, str]:
        """Process one gradient step in the Paced Window.

        Args:
            theta_main: current main weights
            theta_stage: current staging weights
            gme: gradient magnitude estimate
            g2_gme_floor: G2 GME threshold (default 0.60)
            g3_gme_floor: G3 GME threshold (default 0.40)

        Returns:
            (should_rollback, reason) — True if Hard Rollback required
        """
        if self._state == PacedWindowState.INIT:
            # Not in window yet
            return False, "Not in Paced Window"
        # Compute D_struct
        self.compute_d_struct(theta_main, theta_stage)
        # Update fatal streak
        if self._last_d_struct >= self._config.T_fatal:
            self._fatal_streak += 1
        else:
            self._fatal_streak = 0
        # State transitions
        if self._state == PacedWindowState.MONITORING:
            if self._fatal_streak >= self._config.fatal_streak_threshold:
                self._transition(PacedWindowState.ROLLBACK)
                return True, f"D_struct ≥ T_fatal for {self._fatal_streak} consecutive steps"
            self._transition(PacedWindowState.INJECTING_TILING)
            return False, "Injecting Tensor Tiling"
        elif self._state == PacedWindowState.INJECTING_TILING:
            self._transition(PacedWindowState.MONITORING_2)
            return False, "Tiling injected, monitoring 2"
        elif self._state == PacedWindowState.MONITORING_2:
            self._epoch_in_window += 1
            if self._epoch_in_window >= self._config.max_epochs:
                self._transition(PacedWindowState.DECISION)
            return False, f"Monitoring 2 (epoch {self._epoch_in_window}/{self._config.max_epochs})"
        elif self._state == PacedWindowState.DECISION:
            if gme >= g2_gme_floor:
                self._transition(PacedWindowState.EXIT)
                return False, f"GME={gme:.3f} ≥ {g2_gme_floor}, EXIT"
            else:
                self._transition(PacedWindowState.ROLLBACK)
                return True, f"GME={gme:.3f} < {g2_gme_floor} after {self._config.max_epochs} epochs"
        elif self._state in (PacedWindowState.EXIT, PacedWindowState.ROLLBACK):
            # Terminal states
            return self._state == PacedWindowState.ROLLBACK, f"Terminal state: {self._state}"
        return False, "Unknown state"

    def _transition(self, new_state: PacedWindowState) -> None:
        self._state = new_state
        self._state_history.append(new_state)

    def reset(self) -> None:
        """Reset to INIT (call after EXIT or ROLLBACK)."""
        self._state = PacedWindowState.INIT
        self._epoch_in_window = 0
        self._fatal_streak = 0
        self._last_d_struct = 0.0
        self._d_struct_history.clear()
        self._state_history = [PacedWindowState.INIT]

    def get_state_history(self) -> List[PacedWindowState]:
        """Return list of state transitions for diagnostics."""
        return list(self._state_history)

    def get_d_struct_history(self) -> List[float]:
        """Return D_struct values seen so far."""
        return list(self._d_struct_history)
