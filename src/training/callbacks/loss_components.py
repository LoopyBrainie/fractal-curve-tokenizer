"""Loss components accumulator callback (PR4)

Receives per-batch `loss_components: dict[str, Tensor]` from the skeleton
(via `on_batch_end` once PR5c wires it) and accumulates into
`ctx.loss_components` for epoch-level logging.

The PR4 version is a passive accumulator — the skeleton has not yet
populated `ctx.loss_components` per batch (that's PR5c's wiring), so this
callback is effectively a no-op until then. We still create it now to
lock in the contract and prevent future drift.

See plan fluffy-watching-turing.md §3 PR4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import TrainerCallback, TrainerContext

if TYPE_CHECKING:
    from torch import Tensor


class LossComponentsAccumulator(TrainerCallback):
    """被动累加器: per-batch loss_components → ctx.loss_components。

    PR5c skeleton will:
        1. compute_loss returns (loss, components)
        2. skeleton calls `ctx.loss_components_batch = components` (临时)
        3. accumulator copies components → ctx.loss_components (epoch agg)

    PR4 ships the contract; PR5c implements the wiring.
    """

    priority: int = -10  # 早于其他后置 hook (在 on_batch_end 阶段)

    def on_batch_end(self, ctx: TrainerContext) -> None:
        """累加 per-batch components (skeleton-driven, PR5c 注入)。"""
        batch_components = getattr(ctx, "loss_components_batch", None)
        if not isinstance(batch_components, dict):
            return
        for name, tensor in batch_components.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            cur = ctx.loss_components.get(name)
            if cur is None:
                ctx.loss_components[name] = tensor.detach().clone()
            else:
                ctx.loss_components[name] = cur + tensor.detach()

    def on_epoch_end(self, ctx: TrainerContext) -> None:
        """epoch 末归一化 (per-batch mean = sum / num_batches)。"""
        num_batches = getattr(ctx, "epoch_num_batches", 0) or 0
        if num_batches <= 0:
            return
        for name, tensor in list(ctx.loss_components.items()):
            if isinstance(tensor, torch.Tensor):
                ctx.metrics[f"train/loss_component/{name}"] = float(
                    (tensor / num_batches).item()
                )


__all__ = ["LossComponentsAccumulator"]
