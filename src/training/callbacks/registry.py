"""Callback registry / factory (PR4)

`build_callbacks(args, model, ctx)` constructs the default callback list
from CLI args + a fresh `TrainerContext`. The factory centralizes the
"which opt-in features are enabled" decision — previously scattered as
inline flags in `epoch_train.py`.

PR4 ships the factory. PR5c wires `train_one_epoch` to actually consume
`ctx.callbacks`; until then the factory is callable but no-op.

See plan fluffy-watching-turing.md §3 PR4.
"""

from __future__ import annotations

from typing import Any, List

from .base import TrainerCallback
from .fractal_tree_reg import FractalTreeRegCallback
from .hmft_h_probs import HMFTHProbsCallback
from .loss_components import LossComponentsAccumulator


def build_callbacks(config: Any) -> List[TrainerCallback]:
    """Construct the default callback list from `config` (PR5c+).

    Always-on:
      - LossComponentsAccumulator (PR4 contract, PR5c wires it)

    Opt-in (config flags, default off):
      - FractalTreeRegCallback (T3 R12, `config.training.enable_r12_aux`)
      - HMFTHProbsCallback (T6 HMFT, `config.training.enable_hmft_h_probs`, default-on)

    NaN / gradient monitoring is wired separately via `ctx.nan_guard`
    and `GradientMonitorCallback` — they are not constructed here.
    """
    cbs: List[TrainerCallback] = []
    training_cfg = getattr(config, "training", None)

    # Always-on: loss components accumulator
    cbs.append(LossComponentsAccumulator())

    # T3 R12 (opt-in via config.training.enable_r12_aux)
    if bool(getattr(training_cfg, "enable_r12_aux", False)):
        cbs.append(
            FractalTreeRegCallback(
                lambda_tree=float(getattr(training_cfg, "r12_lambda_tree", 1.0)),
                lambda_skew=float(getattr(training_cfg, "r12_lambda_skew", 1.0)),
            )
        )

    # T6 HMFT h_probs (default-on in v1.3 STANDARD)
    if bool(getattr(training_cfg, "enable_hmft_h_probs", True)):
        cbs.append(HMFTHProbsCallback())

    # Sort by priority (stable; preserves insertion order for ties)
    cbs.sort(key=lambda cb: cb.priority)
    return cbs


__all__ = ["build_callbacks"]
