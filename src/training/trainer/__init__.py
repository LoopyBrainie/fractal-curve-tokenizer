"""Trainer Module

Provides training and evaluation functions.
Completely decoupled from model architecture.
"""

from .state import TrainingState, EpochMetrics, EvaluationMetrics
from .loss import MixupCutmixLoss, compute_loss, AuxiliaryLossTracker
from .epoch_train import train_one_epoch, train_one_epoch_simple, GradBalancer
from .epoch_eval import evaluate, evaluate_simple, compute_ece_score

__all__ = [
    "TrainingState",
    "EpochMetrics",
    "EvaluationMetrics",
    "MixupCutmixLoss",
    "compute_loss",
    "AuxiliaryLossTracker",
    "GradBalancer",
    "train_one_epoch",
    "train_one_epoch_simple",
    "evaluate",
    "evaluate_simple",
    "compute_ece_score",
]
