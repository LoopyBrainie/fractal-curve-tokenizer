"""Trainer Module

Provides training and evaluation functions.
Completely decoupled from model architecture.
"""

# === F-X1 PR0 DeprecationWarning shim (trainer refactor) ===
# train_one_epoch_simple + evaluate_simple are removed in PR0.
import warnings as _trainer_warnings
_trainer_warnings.warn(
    "train_one_epoch_simple / evaluate_simple are deprecated and removed in PR0 "
    "of the trainer refactor (see plan fluffy-watching-turing.md §3 PR0).",
    DeprecationWarning, stacklevel=2,
)


def _trainer_deprecated_stub(name: str):
    def _stub(*_args, **_kwargs):
        _trainer_warnings.warn(
            f"{name} removed in PR0; returning None.",
            DeprecationWarning, stacklevel=2,
        )
        return None
    _stub.__name__ = name
    return _stub


train_one_epoch_simple = _trainer_deprecated_stub("train_one_epoch_simple")
evaluate_simple = _trainer_deprecated_stub("evaluate_simple")
# === End F-X1 shim ===

from .state import TrainingState, EpochMetrics, EvaluationMetrics
from .loss import MixupCutmixLoss, compute_loss, AuxiliaryLossTracker, UnifiedLoss
from .epoch_train import train_one_epoch
from .epoch_eval import evaluate, compute_ece_score

__all__ = [
    "TrainingState",
    "EpochMetrics",
    "EvaluationMetrics",
    "MixupCutmixLoss",
    "compute_loss",
    "AuxiliaryLossTracker",
    "UnifiedLoss",
    "train_one_epoch",
    "evaluate",
    "compute_ece_score",
    # F-X1 PR0 shim
    "train_one_epoch_simple",
    "evaluate_simple",
]
