"""T6 HMFT h_probs logging callback (PR4)

Per-epoch aggregation of the 5-bin HMFT block-size distribution, sourced
from `splitter.get_diagnostics()['h_probs']`. Writes scalar entries into
`ctx.metrics['hmft/h_prob_bin_{i}']` and `ctx.metrics['hmft/h_probs_epoch']`.

Default-on (v1.3 STANDARD): the skeleton (PR5c) always wires this callback
in via `build_callbacks()`.

Behavior change vs the pre-PR4 inline `log_h_probs`:
  - Old: `collector.record(...)` (PR1-removed MetricsCollector)
  - New: `ctx.metrics[name] = float(value)` (Q5 决策, flat dict)

See plan fluffy-watching-turing.md §3 PR4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import TrainerCallback, TrainerContext

if TYPE_CHECKING:
    pass


class HMFTHProbsCallback(TrainerCallback):
    """T6 HMFT h_probs 5-bin distribution per-epoch logger.

    Hooks `on_epoch_end` (after each epoch) to read
    `model.splitter.get_diagnostics()` and write into `ctx.metrics`.

    Silent no-op if the splitter doesn't expose `get_diagnostics` or the
    returned dict lacks the `h_probs` key — matching the pre-PR4 behavior.
    """

    priority: int = 10  # 后置 hook (after most metric callbacks)

    def on_epoch_end(self, ctx: TrainerContext) -> None:
        """Read splitter diagnostics → ctx.metrics (flat dict, Q5 决策)."""
        model = getattr(ctx, "model", None)
        if model is None:
            return
        splitter = getattr(model, "splitter", None)
        if splitter is None:
            return
        diag_fn = getattr(splitter, "get_diagnostics", None)
        if diag_fn is None:
            return
        try:
            diag = diag_fn() or {}
        except Exception:
            return
        if not isinstance(diag, dict):
            return
        h = diag.get("h_probs")
        if h is None:
            return
        for i, p in enumerate(h):
            try:
                ctx.metrics[f"hmft/h_prob_bin_{i}"] = float(p)
            except Exception:
                break
        try:
            ctx.metrics["hmft/h_probs_epoch"] = float(ctx.epoch)
        except Exception:
            pass


__all__ = ["HMFTHProbsCallback"]
