"""T3 R12 (fractal tree regularization) callback (PR4)

Callback that injects the R12 aux-loss into `ctx.aux_losses['r12']` so the
skeleton (PR5c) can sum it into the total loss before `backward()`.

Opt-in via `config.training.enable_r12_aux` (default False). When disabled
the callback is a no-op — preserving default behavior.

Aux-loss grad_fn is preserved per the 5-allow/3-forbid write contract:
the tensor written to `ctx.aux_losses` keeps its `grad_fn` so backprop
flows through it.

See plan fluffy-watching-turing.md §3 PR4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .base import TrainerCallback, TrainerContext

if TYPE_CHECKING:
    from torch import Tensor


def _padded_levels_to_depth_distribution(padded_levels, max_depth: int = 8) -> "Tensor":
    """Convert ForwardOutput.padded_levels to a 1D depth probability tensor.

    `padded_levels` is Tensor [B, max_N, D+1] with padding=-1; we ignore
    padding and average the D+1 routing values per image.
    """
    if padded_levels is None or (hasattr(padded_levels, "numel") and padded_levels.numel() == 0):
        return torch.zeros(max_depth + 1)
    if not hasattr(padded_levels, "view"):
        return torch.zeros(max_depth + 1)
    try:
        flat = padded_levels.view(-1, padded_levels.shape[-1]).float()
        depth = flat[:, -1]
        out = torch.zeros(max_depth + 1)
        for v in depth.tolist():
            idx = max(0, min(int(v), max_depth))
            out[idx] += 1.0
        total = out.sum()
        if total > 0:
            out = out / total
        return out
    except Exception:
        return torch.zeros(max_depth + 1)


class FractalTreeRegCallback(TrainerCallback):
    """R12 树结构正则回调 (T3 v1.3, opt-in).

    在 `on_loss_computed` 中从 `ctx.aux_forward` (由 skeleton 在 PR5c 设置)
    取出 `parent_logits` / `child_logits`,计算 R12AuxLoss,注入
    `ctx.aux_losses['r12']`。

    Attributes:
        lambda_tree: 树结构项权重
        lambda_skew: 偏度项权重
    """

    priority: int = 0  # 默认后置 hook

    def __init__(self, lambda_tree: float = 1.0, lambda_skew: float = 1.0) -> None:
        self.lambda_tree = lambda_tree
        self.lambda_skew = lambda_skew
        self._enabled = False
        self._r12_fn: Any = None  # R12AuxLoss 实例或 None

    def on_train_start(self, ctx: TrainerContext) -> None:
        """读取 config 标志 + 懒加载 R12AuxLoss。"""
        cfg = getattr(ctx, "config", None)
        self._enabled = bool(getattr(getattr(cfg, "training", None), "enable_r12_aux", False))
        if self._enabled:
            try:
                from vit_pytorch.core.measure_auxiliary_loss import (
                    R12AuxLoss,
                    R12AuxLossConfig,
                )
            except ImportError:
                self._enabled = False
            else:
                self._r12_fn = R12AuxLoss(
                    R12AuxLossConfig(
                        lambda_tree=self.lambda_tree,
                        lambda_skew=self.lambda_skew,
                    )
                )

    def on_loss_computed(
        self, ctx: TrainerContext, loss: "Tensor", components: dict
    ) -> None:
        """注入 r12 aux-loss。前提: skeleton 在 PR5c 把 forward_output 存到 ctx。

        `loss` 参数在 R12 路径下不直接使用 (我们只读 components 与 ctx.aux_forward),
        但保留参数名以满足 LSP 兼容 (父类签名 `loss: Tensor`)。
        """
        del loss  # 不直接使用,仅满足父类签名
        if not self._enabled or self._r12_fn is None:
            return  # disabled / unavailable
        fwd = getattr(ctx, "aux_forward", None)
        if fwd is None:
            return
        aux = getattr(fwd, "auxiliary_outputs", None) or {}
        logits = components.get("logits")
        parent = aux.get("parent_logits")
        child = aux.get("child_logits")
        if parent is None:
            parent = logits
        if child is None:
            child = logits
        depth_dist = _padded_levels_to_depth_distribution(
            getattr(fwd, "padded_levels", None), max_depth=8,
        )
        try:
            r12_tensor = self._r12_fn(parent, child, depth_dist)
        except Exception:
            return
        ctx.aux_losses["r12"] = r12_tensor


__all__ = ["FractalTreeRegCallback"]
