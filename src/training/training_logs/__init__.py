"""Logging Module

Epoch logging and metrics.
"""

# === F-X1 PR0: 4 dead symbols removed (0 callers + transitively dead) ===
# MetricsTracker: 0 callers (kept only as re-export)
# compute_nll / compute_brier_score: only called by compute_all_metrics
# compute_all_metrics: only called by MetricsComputer.compute()
#   => entire dead chain deleted; MetricsComputer kept per Q5
from .epoch_logger import EpochLogger
from .metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_per_class_accuracy,
    compute_ece,
    MetricsComputer,
)
from .pipeline import MetricsPipeline

__all__ = [
    "EpochLogger",
    "compute_accuracy",
    "compute_confusion_matrix",
    "compute_per_class_accuracy",
    "compute_ece",
    "MetricsComputer",
    "MetricsPipeline",
]
