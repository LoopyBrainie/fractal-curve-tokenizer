"""TrainerCallback ABC + TrainerContext dataclass (PR2 single-def)

**PR2 single-source-of-truth**: TrainerContext 在此定义,PR5 共享同一份定义。
避免 PR5 重复定义冲突 (F-X4 验证)。

5-allow / 3-forbid callback 写契约 (Q1 + beta-A-r2 D3 锁定):

| 允许写入 ctx | 禁止原地写入 | 禁止断开梯度链 |
|-------------|-------------|----------------|
| ctx.metrics (scalar) | model.* | loss.detach() 后再写 ctx |
| ctx.loss_components (tensor) | optimizer.* | aux_loss.detach() 在 on_loss_computed |
| ctx.aux_losses (tensor w/ grad_fn) | scaler.* | |
| ctx.should_skip_step (bool) | | |
| ctx.auxiliary_flat_metrics (scalar) | | |

NaNGuard / NaNDumpCallback 等"骨架硬依赖"通过 ctx.nan_guard 字段访问,
不通过 ctx.callbacks 列表注入。

See:
  - plan: C:\\Users\\LamKo\\.claude\\plans\\fluffy-watching-turing.md §3 PR2 + 5-allow/3-forbid table
  - design: docs/superpowers/specs/2026-06-08-fractal-vit-trainer-refactor-design.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer

if TYPE_CHECKING:
    from torch.cuda.amp import GradScaler

    from ..trainer.state import TrainingState


@dataclass
class TrainerContext:
    """跨回调共享的训练状态对象。指标以扁平 dict 流动 (Q5 决策)。

    Fields:
        model: nn.Module
        optimizer: torch.optim.Optimizer
        scaler: torch.cuda.amp.GradScaler (None if AMP disabled)
        scheduler: LR scheduler (Optional)
        state: TrainingState
        epoch: int
        global_step: int
        device: torch.device
        config: Config
        mixup: MixupCutmixLoss (Optional)
        nan_guard: NaNGuard (骨架硬依赖, 非 callback)
        callbacks: list[TrainerCallback] (按 priority 排序)
        amp: bool
        grad_clip: float

        # 扁平指标容器 (Q5 决策: 替代 MetricsCollector)
        metrics: dict[str, float]
        loss_components: dict[str, Tensor]
        auxiliary_flat_metrics: dict[str, float]
        aux_losses: dict[str, Tensor]            # on_loss_computed 注入,保留 grad_fn
        should_skip_step: bool
        warmup_params: dict[str, Any]
    """

    # === 硬依赖 (skeleton owns) ===
    model: torch.nn.Module
    optimizer: Optimizer
    scaler: Optional["GradScaler"] = None
    scheduler: Any = None
    state: Any = None                                # TrainingState (避免循环 import)
    nan_guard: Any = None                            # NaNGuard (避免循环 import)
    callbacks: list = field(default_factory=list)

    # === 标量状态 ===
    epoch: int = 0
    global_step: int = 0
    device: torch.device = torch.device("cuda")
    config: Any = None                               # Config
    mixup: Any = None                                # MixupCutmixLoss

    # === 配置转发 ===
    amp: bool = False
    grad_clip: float = 1.0

    # === 扁平指标容器 (Q5 决策: 替代 MetricsCollector) ===
    metrics: dict = field(default_factory=dict)
    loss_components: dict = field(default_factory=dict)
    auxiliary_flat_metrics: dict = field(default_factory=dict)
    aux_losses: dict = field(default_factory=dict)          # on_loss_computed 注入,保留 grad_fn
    should_skip_step: bool = False
    warmup_params: dict = field(default_factory=dict)


class TrainerCallback:
    """外延功能挂点。所有方法默认 no-op。

    priority 语义 (F-X4 + 6-gate 框架):
        priority < 0: 骨架硬边界回调 (loss_computed_boundary /
                       grad_accumulation_boundary / amp_step_boundary /
                       nan_guard / param_step_boundary) — 保留位
        priority == 0: 默认 (大多数 callback)
        priority > 0: 后置 hook (日志 / 监控)

    实际硬边界 hook 不通过本 ABC — 它们是骨架内联代码 (scaler.step、
    no_sync、optimizer.zero_grad 等),不能回调化。
    """

    priority: int = 0

    # === 训练生命周期 ===
    def on_train_start(self, ctx: TrainerContext) -> None:
        """训练开始时调用一次。"""

    def on_train_end(self, ctx: TrainerContext) -> None:
        """训练结束时调用一次。"""

    def on_epoch_start(self, ctx: TrainerContext) -> None:
        """每个 epoch 开始时调用。"""

    def on_epoch_end(self, ctx: TrainerContext) -> None:
        """每个 epoch 结束时调用。"""

    # === 批次生命周期 ===
    def on_batch_start(self, ctx: TrainerContext) -> None:
        """每个 batch 开始时调用。"""

    def on_batch_end(self, ctx: TrainerContext) -> None:
        """每个 batch 结束时调用 (在 optimizer.step 之后)。"""

    # === 损失合成边界 ===
    def on_loss_computed(
        self, ctx: TrainerContext, loss: Tensor, components: dict
    ) -> None:
        """loss 已计算, components dict 已就绪。callback 可注入 aux_losses
        到 `ctx.aux_losses`,骨架在 pre_backward 之前收口到总 loss。

        Args:
            ctx: 训练上下文
            loss: 主损失 (Tensor, scalar)
            components: dict of {name: Tensor}, 已包含 cross_entropy + 各 aux
        """

    # === 反向传播边界 ===
    def pre_backward(self, ctx: TrainerContext) -> None:
        """backward() 之前调用。"""

    def post_backward(self, ctx: TrainerContext) -> None:
        """backward() 之后调用 (在 is_healthy 判定之前)。

        注意: NaNGuard 是骨架硬依赖, 通过 ctx.nan_guard.is_healthy() 调用,
        不在 callback 列表中。本 hook 用于梯度监控等非数值防御用途。
        """

    # === 优化器 step 边界 ===
    def pre_step(self, ctx: TrainerContext) -> None:
        """optimizer.step() 之前调用 (在 grad clip 之后, scaler.step 之前)。"""

    def post_step(self, ctx: TrainerContext) -> None:
        """optimizer.step() 之后调用。"""

    # === 异常处理 ===
    def on_exception(self, ctx: TrainerContext, exc: BaseException) -> bool:
        """异常发生时调用。返回 True 表示已处理 (骨架 swallow),False 表示
        继续抛出。

        Returns:
            bool: True = 已处理,False = 未处理 (re-raise)
        """
        return False


__all__ = [
    "TrainerContext",
    "TrainerCallback",
]
