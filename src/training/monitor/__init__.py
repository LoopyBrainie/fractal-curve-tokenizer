"""Monitor Module

Numerical monitoring and defense.
"""

from .gradient_monitor import GradientMonitor
from .loss_monitor import LossMonitor, LossTracker, CombinedLossTracker
from .numerical_defense import (
    AnomalyDetectionContext,
    GradientValidator,
    NumericalDefender,
    ActivationStatsCollector,
    check_tensor_numerical_health,
)

# === PR1 (trainer refactor): UnifiedMonitor removed (Q5 decision: half-finished
# abstraction; replaces by layer-packaged auxiliary_outputs flowing into
# UnifiedMonitor (deleted PR1)) + per-layer GradientMonitor +
# numerical defense triad. See plan fluffy-watching-turing.md §3 PR1. ===

__all__ = [
    "GradientMonitor",
    "LossMonitor",
    "LossTracker",
    "CombinedLossTracker",
    "AnomalyDetectionContext",
    "GradientValidator",
    "NumericalDefender",
    "ActivationStatsCollector",
    "check_tensor_numerical_health",
]
