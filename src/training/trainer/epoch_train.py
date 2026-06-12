"""Training Epoch Skeleton (PR5c rewrite)

4-hook boundary pattern (Q1 + F-X2 locked design):
  1. Forward + loss computation (硬编码: AMP, mixup, OOM catch)
  2. callback: `on_loss_computed` → 注入 `ctx.aux_losses` (T3 R12 等)
  3. Backward + nan_guard (硬编码: scaler, grad clip, AMP protocol)
  4. callback: `on_batch_end` → 聚合 `ctx.metrics` / `ctx.loss_components`

形参收敛: 18+ → 6 (`model, dataloader, optimizer, scaler, state, ctx`)。
ctx 吸收: `scheduler, mixup, device, config, amp, grad_clip, callbacks, nan_guard`。

=== PR5c 关键迁移 ===
  - `loss_monitor`  → `LossComponentsAccumulator` callback (PR4 已就位)
  - `grad_monitor`  → 移除 (PR3 已迁至 `GradientMonitorCallback`; main.py 显式构建)
  - `defender`      → `ctx.nan_guard` (PR2 NaNGuard 骨架硬依赖)
  - `nan_investigator` → `NaNDumpCallback` (不在默认 list; --debug 才挂入)
  - `shadow_hooks`  → 删 (T2 cull per Q2)
  - `warmup_params` → 删 (PR5a dead writes 已清)
  - `collector`     → 删 (Q5: 指标走 ctx.metrics, 零抽象)

=== AMP 协议硬编码 (保留) ===
  1. `scaler.scale(loss).backward()` (or `loss.backward()` if no scaler)
  2. `scaler.unscale_(optimizer)` AFTER backward, BEFORE grad clip
  3. `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)`
  4. `scaler.step(optimizer)` / `scaler.update()` (ALL paths)

=== NaN 防御契约 (F-X3 + Q1 锁定) ===
  - `ctx.nan_guard.is_healthy(loss, model)` → True=proceed / False=skip
  - Skip 路径: `optimizer.zero_grad()` + `scaler.update()` + `state.nan_skip_count += 1`

=== 梯度累积契约 ===
  - `_is_accum_boundary = (batch_idx + 1) % accum_steps == 0`
  - 仅在 accum 边界执行 optimizer.step(); skip 路径必须 zero_grad (污染累积梯度)

See:
  - plan: C:\\Users\\LamKo\\.claude\\plans\\fluffy-watching-turing.md §3 PR5c + §7.2
  - design: docs/superpowers/specs/2026-06-08-fractal-vit-trainer-refactor-design.md
"""

from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import autocast as amp_autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from ..callbacks import TrainerContext
from .state import EpochMetrics, TrainingState
from .loss import compute_loss


def _to_device(batch, device):
    """Tuple/list batch → (images, labels) on `device` (non_blocking)。"""
    if isinstance(batch, (list, tuple)):
        return (
            batch[0].to(device, non_blocking=True),
            batch[1].to(device, non_blocking=True),
        )
    return batch.to(device, non_blocking=True), None


def _build_targets(images, labels, mixup, model):
    """Mixup / one-hot encode labels。返回 `targets` (one-hot) 或 `None`。

    MixupCutmixLoss.__call__(images, labels, apply_aug=True) → (mixed_images, mixed_labels)。
    """
    if mixup is not None and labels is not None:
        _, targets = mixup(images, labels, apply_aug=True)
        return targets
    if labels is None:
        return None
    n_classes = getattr(model, "num_classes", None)
    if n_classes is None:
        raise ValueError(
            "model.num_classes required for one-hot encoding when mixup disabled"
        )
    return torch.nn.functional.one_hot(labels, n_classes).float()


def _aggregate_epoch_metrics(
    ctx: TrainerContext,
    epoch_time: float,
    peak_memory_mb: float,
    skipped_steps: int,
    num_batches: int,
) -> EpochMetrics:
    """扁平 `ctx.metrics` dict → `EpochMetrics` (PR5c 收敛点)。

    `ctx.loss_components` 在 on_batch_end 阶段被 `LossComponentsAccumulator`
    累加为 sum, 此处除以 num_batches 转 mean; 写为 float 供 to_dict 序列化。
    """
    m = ctx.metrics
    samples = max(1, int(m.get("train/samples", 0)))
    return EpochMetrics(
        loss=m.get("train/loss", 0.0) / max(num_batches, 1),
        accuracy=m.get("train/accuracy", 0.0) / samples,
        grad_norm=m.get("train/grad_norm", 0.0) / max(num_batches, 1),
        learning_rate=ctx.optimizer.param_groups[0]["lr"],
        epoch_time=epoch_time,
        samples_per_second=samples / max(epoch_time, 1e-8),
        skipped_steps=skipped_steps,
        peak_memory_mb=peak_memory_mb,
        loss_components={
            k: (v / num_batches).item()
            for k, v in ctx.loss_components.items()
            if torch.is_tensor(v)
        } if ctx.loss_components else {},
        auxiliary_flat_metrics=dict(ctx.auxiliary_flat_metrics),
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    state: TrainingState,
    ctx: TrainerContext,
) -> EpochMetrics:
    """Train for one epoch — 4-hook skeleton (Q1 locked design)。

    Args:
        model: `nn.Module` (forward 返回 `(ForwardOutput, MetricsTensors)` tuple)
        dataloader: 训练 `DataLoader`
        optimizer: `torch.optim.Optimizer`
        scaler: `GradScaler` (None 表示 AMP 关闭)
        state: `TrainingState` (singleton, 原地修改)
        ctx: `TrainerContext` (callbacks, nan_guard, mixup, scheduler, amp, grad_clip)

    Returns:
        `EpochMetrics` 包含本 epoch 的聚合指标。
    """
    from tqdm import tqdm

    callbacks = sorted(ctx.callbacks, key=lambda cb: cb.priority)
    for cb in callbacks:
        cb.on_train_start(ctx)

    model.train()
    accum_steps: int = int(
        getattr(getattr(ctx.config, "training", None), "accumulation_steps", 1)
    )
    grad_clip: float = float(ctx.grad_clip) if ctx.grad_clip and ctx.grad_clip > 0 else 0.0
    epoch_start = time.time()
    peak_memory_mb = 0.0
    skipped_steps = 0
    num_batches = 0

    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Epoch {state.epoch}",
        leave=False,
    )
    for batch_idx, batch in pbar:
        for cb in callbacks:
            cb.on_batch_start(ctx)

        # === 1. 确定性数据流 ===
        images, labels = _to_device(batch, ctx.device)
        targets = _build_targets(images, labels, ctx.mixup, model)

        # === 2. 前向 (AMP 上下文 + OOM) ===
        with amp_autocast("cuda", enabled=ctx.amp):
            try:
                # 模型 forward 返回 TrainingStats; 旧 MetricsTensors 中包含的
                # splitter/active/flops 等统计在 layer-packaged 机制下
                # 已通过 forward_output.auxiliary_outputs 流入 auxiliary_flat_metrics,
                # 骨架不再需要解包第二元素 (PR5c 收敛)。
                forward_output = model(images)
            except torch.cuda.OutOfMemoryError as e:
                handled = False
                for cb in callbacks:
                    if cb.on_exception(ctx, e):
                        handled = True
                        break
                if not handled:
                    raise  # OOM 未被 callback 处理, 重新抛出

        # 提供 forward_output 给 on_loss_computed 回调 (T3 R12 FractalTreeRegCallback 读 parent_logits)
        ctx.aux_forward = forward_output

        # === 3. 损失合成 ===
        loss = torch.tensor(0.0, device=ctx.device, requires_grad=True)
        components: dict = {}
        if targets is not None:
            loss, components = compute_loss(forward_output.logits, targets)

        # 3a. on_loss_computed: 回调注入 aux_losses (T3 R12 等); grad_fn 保留 (F-X3)
        ctx.aux_losses.clear()
        ctx.loss_components_batch = components
        for cb in callbacks:
            cb.on_loss_computed(ctx, loss, components)
        for v in ctx.aux_losses.values():
            loss = loss + v

        # === 4. Backward (硬编码 scaler/AMP) ===
        loss_scaled = loss / accum_steps
        for cb in callbacks:
            cb.pre_backward(ctx)
        (scaler.scale(loss_scaled) if scaler else loss_scaled).backward()
        for cb in callbacks:
            cb.post_backward(ctx)

        # === 5. 数值防御 (硬依赖 nan_guard) + 梯度裁剪 + Optimizer step ===
        _is_accum_boundary = (batch_idx + 1) % accum_steps == 0
        should_skip = False
        if ctx.nan_guard is not None:
            should_skip = not ctx.nan_guard.is_healthy(loss, model)

        if scaler is not None:
            scaler.unscale_(optimizer)  # 必须在 grad clip 之前
        if grad_clip > 0 and _is_accum_boundary:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip,
            )
            if not torch.isfinite(grad_norm):
                should_skip = True
        else:
            grad_norm = torch.tensor(0.0)

        if not should_skip and _is_accum_boundary:
            if ctx.scheduler is not None:
                ctx.scheduler.step(state.global_step)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        elif should_skip:
            if ctx.scheduler is not None:
                ctx.scheduler.step(state.global_step)
            optimizer.zero_grad()  # NaN 污染累积梯度, 立即清零
            if scaler is not None:
                scaler.update()  # 即使 skip 也必须 update
            skipped_steps += 1
            state.nan_skip_count += 1

        # === 6. 指标累加 (扁平 ctx.metrics) ===
        if targets is not None and forward_output.logits is not None:
            if targets.dim() == 2:
                pred = forward_output.logits.argmax(dim=-1)
                tgt_cls = targets.argmax(dim=-1)
                correct = (pred == tgt_cls).sum().item()
            else:
                correct = (
                    forward_output.logits.argmax(dim=-1) == targets
                ).sum().item()
            ctx.metrics["train/accuracy"] = (
                ctx.metrics.get("train/accuracy", 0.0) + correct
            )
            ctx.metrics["train/samples"] = (
                ctx.metrics.get("train/samples", 0) + images.size(0)
            )

        if forward_output.num_tokens is not None:
            ntok = forward_output.num_tokens
            if isinstance(ntok, torch.Tensor):
                ntok = ntok.detach().float().mean().item()
            elif isinstance(ntok, (list, tuple)):
                ntok = float(sum(ntok) / len(ntok))
            else:
                ntok = float(ntok)
            ctx.metrics["train/num_tokens"] = (
                ctx.metrics.get("train/num_tokens", 0.0) + ntok
            )

        ctx.metrics["train/loss"] = ctx.metrics.get("train/loss", 0.0) + loss.item()
        grad_norm_val = (
            grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
        )
        ctx.metrics["train/grad_norm"] = (
            ctx.metrics.get("train/grad_norm", 0.0) + grad_norm_val
        )

        # 显存峰值
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024**2
            if peak > peak_memory_mb:
                peak_memory_mb = peak

        # === 7. Batch end callbacks (LossComponentsAccumulator 在此累加, priority=-10) ===
        for cb in callbacks:
            cb.on_batch_end(ctx)
        state.increment_step()
        num_batches += 1

    # === 8. Epoch end ===
    epoch_time = time.time() - epoch_start
    ctx.metrics["train/num_batches"] = num_batches
    ctx.metrics["train/epoch_time"] = epoch_time
    ctx.metrics["train/skipped_steps"] = skipped_steps
    ctx.metrics["train/peak_memory_mb"] = peak_memory_mb
    ctx.epoch_num_batches = num_batches  # LossComponentsAccumulator.on_epoch_end 读取

    for cb in callbacks:
        cb.on_epoch_end(ctx)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for cb in callbacks:
        cb.on_train_end(ctx)

    return _aggregate_epoch_metrics(
        ctx, epoch_time, peak_memory_mb, skipped_steps, num_batches,
    )


__all__ = ["train_one_epoch"]
