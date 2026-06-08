"""v1.3 STANDARD: ShadowMonitorTrainerHooks — wraps ShadowMonitor into the
trainer's hook lifecycle (post_forward / post_backward) without coupling
to the trainer's internal state.

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md
§9.7 the Shadow Monitor is a non-blocking observer:
  - G1 WBA local entropy (windowed boundary access)
  - G2 RoPE InfoNCE MI lower bound
  - GME (gradient magnitude estimate) computed in post_backward

Trainer integration (per docs/superpowers/plans/...phase T2):
  hooks = ShadowMonitorTrainerHooks(interval=50, collector=collector)
  ...
  hooks.post_forward(attn_scores, rope, B, state.global_step)
  loss.backward()
  hooks.post_backward(model, optimizer, state.global_step)

interval=0 means "every step" (no modulo gating).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from src.training.metrics.collector import MetricsCollector
    from vit_pytorch.core.shadow_monitor import ShadowMonitor


@dataclass
class ShadowMonitorTrainerHooks:
    """Trainer-side hook wrapper around the v1.3 ShadowMonitor.

    Mirrors the post_forward / post_backward pattern of `UnifiedMonitor`,
    but writes into a caller-provided `MetricsCollector` so the trainer
    owns the metric lifecycle.
    """
    interval: int = 50
    monitor: Optional["ShadowMonitor"] = None
    collector: Optional["MetricsCollector"] = None

    def __post_init__(self) -> None:
        # Lazy import to keep this module importable even if v1.3 deps
        # are not yet present (e.g. during the Phase 0 transition).
        if self.monitor is None:
            from vit_pytorch.core.shadow_monitor import ShadowMonitor
            self.monitor = ShadowMonitor()
        if self.collector is None:
            from src.training.metrics.collector import MetricsCollector
            self.collector = MetricsCollector()

    def _should_record(self, step: int) -> bool:
        if self.interval == 0:
            return True
        return (step % self.interval) == 0

    @torch.no_grad()
    def post_forward(
        self,
        attention_scores: Optional[Tensor],
        rope_embeddings: Optional[Tensor],
        batch_size: int,
        step: int,
    ) -> None:
        """Compute G1 WBA + G2 MI scalars (if step is on-interval).

        Both inputs may be None for non-EAHBP models — the hook records
        zero scalars in that case (no upstream attention to monitor).
        """
        if not self._should_record(step):
            return
        if attention_scores is None or rope_embeddings is None:
            # No EAHBP attention available; record zero scalars so downstream
            # dashboards see continuous (but uninformative) data.
            self.collector.record("shadow/wba_entropy", 0.0)
            self.collector.record("shadow/mi_lower_bound", 0.0)
            return
        wba, mi = self.monitor(attention_scores, rope_embeddings, batch_size)
        self.collector.record("shadow/wba_entropy", float(wba))
        self.collector.record("shadow/mi_lower_bound", float(mi))

    @torch.no_grad()
    def post_backward(
        self,
        model: torch.nn.Module,
        optimizer: Optional["Optimizer"],
        step: int,
    ) -> None:
        """Compute GME (gradient magnitude estimate) after backward."""
        if not self._should_record(step):
            return
        gme = self._compute_gme(model)
        self.collector.record("shadow/gme", gme)

    @staticmethod
    @torch.no_grad()
    def _compute_gme(model: torch.nn.Module) -> float:
        """GME = ||grad||_total / (||param||_total + 1e-12).

        A coarse signal of "how big is the latest update" relative to
        the parameter magnitudes. 0.0 if no gradients are present.
        """
        grad_sq_sum = 0.0
        param_sq_sum = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_sq_sum += float(p.grad.detach().pow(2).sum().item())
            param_sq_sum += float(p.detach().pow(2).sum().item())
        if param_sq_sum <= 0.0:
            return 0.0
        return (grad_sq_sum ** 0.5) / ((param_sq_sum ** 0.5) + 1e-12)
