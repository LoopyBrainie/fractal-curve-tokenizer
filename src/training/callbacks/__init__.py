"""Callbacks Subpackage (PR2 of trainer refactor)

外延功能挂点。所有 trainer-side 可选 / opt-in 行为都通过 callback 注入。
骨架 (`src.training.trainer.epoch_train.train_one_epoch`) 显式
调用 4 个硬边界 hook (`on_loss_computed` / `pre_backward` / `post_backward` /
`on_batch_end` 等),callback 列表在 PR5 之前不会真正被骨架消费 — PR2
仅建立基础设施。

设计原则 (Q1 锁定方案):
  1. 改变张量生命周期 / 图依赖 / 硬件状态的代码 → 骨架硬编码
  2. 遥测 / 指标 / opt-in 实验性特性 → 回调化
  3. NaNGuard 是骨架硬依赖 (`ctx.nan_guard`),不通过 callback 列表注入
  4. AMP scaler / DDP sync / optimizer step → 骨架内联

=== PR4 ===
  T3 R12 (FractalTreeRegCallback) + T6 HMFT (HMFTHProbsCallback) 升格为 callback;
  LossComponentsAccumulator 接管 per-component 累加;`build_callbacks()` factory
  集中 CLI flag → callback 映射。

See:
  - plan: C:\\Users\\LamKo\\.claude\\plans\\fluffy-watching-turing.md §3 PR2 + PR4
  - design: docs/superpowers/specs/2026-06-08-fractal-vit-trainer-refactor-design.md
"""

from .base import (
    TrainerCallback,
    TrainerContext,
)
from .nan_guard import NaNGuard
from .nan_dump import NaNDumpCallback
from .gradient_monitor import GradientMonitorCallback
from .fractal_tree_reg import FractalTreeRegCallback
from .hmft_h_probs import HMFTHProbsCallback
from .loss_components import LossComponentsAccumulator
from .registry import build_callbacks

__all__ = [
    # PR2 single-def (TrainerContext shared across PR2+PR5)
    "TrainerCallback",
    "TrainerContext",
    # PR2 numerical defense triad
    "NaNGuard",
    "NaNDumpCallback",
    # PR3 gradient monitor
    "GradientMonitorCallback",
    # PR4 v1.3 opt-in callbacks (T3 R12, T6 HMFT)
    "FractalTreeRegCallback",
    "HMFTHProbsCallback",
    "LossComponentsAccumulator",
    "build_callbacks",
]
