"""Training Epoch Module

Implements independent train_one_epoch function.
Following the three-layer parameter principle, this module
handles Layer 3 (hyperparameters) for training loop.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
import time
import torch
import torch.nn as nn
from torch.amp import autocast as amp_autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from ..config import Config
from .state import TrainingState, EpochMetrics
from .loss import MixupCutmixLoss, compute_loss
from ..monitor.gradient_monitor import GradientMonitor
from ..monitor.loss_monitor import LossMonitor
from ..monitor.numerical_defense import NumericalDefender, NaNAutoInvestigation, dump_debug_info


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    state: TrainingState,
    config: Config,
    device: torch.device,
    scheduler: Optional[Any] = None,
    mixup_cutmix: Optional[MixupCutmixLoss] = None,
    debug_dir: Optional[str] = None,
) -> EpochMetrics:
    """Train for one epoch

    This function is COMPLETELY INDEPENDENT from model architecture.
    It only receives model through the generic nn.Module interface.

    Args:
        model: Neural network model
        dataloader: Training data loader
        optimizer: Optimizer
        scaler: GradScaler for AMP (None if AMP disabled)
        state: Training state
        config: Training configuration
        device: Device to train on
        scheduler: Optional LR scheduler
        mixup_cutmix: Optional Mixup/Cutmix loss handler

    Returns:
        EpochMetrics with training statistics
    """
    model.train()

    # Initialize monitors
    grad_monitor = GradientMonitor(
        model=model,
        record_layer_norms=config.numerical.record_layer_grad_norms,
        hooks_enabled=config.numerical.record_grad_norms,
    )
    loss_monitor = LossMonitor()
    defender = NumericalDefender(
        model=model,
        detect_anomaly=config.numerical.detect_anomaly,
        skip_on_nan=config.numerical.skip_on_nan_grad,
    )

    # I-NAN: 初始化 NaN 自动取证器
    # debug_dir 默认为实验目录下的 debug 子目录
    _debug_dir = debug_dir if debug_dir else "experiments/debug"
    nan_investigator = NaNAutoInvestigation(
        model=model,
        debug_dir=_debug_dir,
        enabled=True,  # 始终启用，用于捕获第一次 NaN
    )

    # 用于记录输入数据统计（用于调试）
    _input_stats: Dict[str, float] = {}

    # Metrics accumulators
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_tokens = 0.0
    total_grad_norm = 0.0
    num_batches = 0
    skipped_steps = 0

    # 新增: 实验详细日志指标 accumulators
    total_splitter_logits_mean = 0.0
    total_splitter_logits_std = 0.0
    total_active_ratio = 0.0
    total_manifold_bias_max = 0.0
    total_manifold_bias_min = 0.0
    total_manifold_bias_mean = 0.0
    total_manifold_bias_std = 0.0
    total_poincare_dist_mean = 0.0
    total_poincare_dist_std = 0.0
    total_backbone_grad_norm = 0.0
    total_splitter_grad_norm = 0.0
    total_entmax_grad_norm = 0.0
    total_manifold_decoder_grad_norm = 0.0
    total_budget_penalty = 0.0
    total_consistency_loss = 0.0
    total_entropy_loss = 0.0
    total_theoretical_flops_reduction = 0.0
    total_mean_abs_logits = 0.0  # I150-3 NEW: Splitter Logits 平均绝对值
    total_budget_loss = 0.0  # I150-3 NEW: Elastic Budget 损失
    total_density_regularization = 0.0  # I150-3 NEW: 密度正则化损失

    # 显存峰值
    peak_memory_mb = 0.0

    epoch_start_time = time.time()

    # Training loop
    for batch_idx, batch in enumerate(dataloader):
        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            images = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)
        else:
            images = batch.to(device, non_blocking=True)
            labels = None

        # I-NAN: 记录输入数据统计（用于 NaN 调试）
        _input_stats = {
            "images_mean": float(images.mean().detach()),
            "images_std": float(images.std().detach()),
            "images_min": float(images.min().detach()),
            "images_max": float(images.max().detach()),
            "images_has_nan": bool(torch.isnan(images).any().detach()),
            "images_has_inf": bool(torch.isinf(images).any().detach()),
        }

        # Apply Mixup/Cutmix if enabled
        # FIX: 修复运算符优先级问题，需要用括号明确分组
        apply_mixup = (
            mixup_cutmix is not None and
            (config.training.mixup_alpha > 0 or config.training.cutmix_alpha > 0)
        )

        if mixup_cutmix is not None and labels is not None:
            images, targets = mixup_cutmix(images, labels, apply_aug=apply_mixup)
        else:
            targets = torch.nn.functional.one_hot(labels, model.num_classes).float() if labels is not None else None

        # Forward pass with AMP
        with amp_autocast('cuda', enabled=config.amp.enabled):
            outputs = model(images)

            # Handle TrainingStats from Fractal ViT
            if hasattr(outputs, 'logits'):
                logits = outputs.logits

                # Extract token info if available
                if hasattr(outputs, 'num_tokens'):
                    num_tokens_raw = outputs.num_tokens
                    # Handle different types (int, tensor, list)
                    if isinstance(num_tokens_raw, torch.Tensor):
                        batch_tokens = num_tokens_raw.float().mean().item()
                    elif isinstance(num_tokens_raw, (int, float)):
                        batch_tokens = float(num_tokens_raw)
                    else:
                        batch_tokens = float(sum(num_tokens_raw) / len(num_tokens_raw))
                    total_tokens += batch_tokens

                # 新增: 提取实验详细日志指标
                if hasattr(outputs, 'splitter_logits_mean'):
                    total_splitter_logits_mean += outputs.splitter_logits_mean
                    total_splitter_logits_std += outputs.splitter_logits_std
                if hasattr(outputs, 'active_ratio'):
                    total_active_ratio += outputs.active_ratio
                if hasattr(outputs, 'manifold_bias_max'):
                    total_manifold_bias_max += outputs.manifold_bias_max
                    total_manifold_bias_min += outputs.manifold_bias_min
                    total_manifold_bias_mean += outputs.manifold_bias_mean
                    total_manifold_bias_std += outputs.manifold_bias_std
                if hasattr(outputs, 'poincare_dist_mean'):
                    total_poincare_dist_mean += outputs.poincare_dist_mean
                    total_poincare_dist_std += outputs.poincare_dist_std
                if hasattr(outputs, 'budget_penalty'):
                    total_budget_penalty += abs(outputs.budget_penalty)
                if hasattr(outputs, 'consistency_loss'):
                    total_consistency_loss += outputs.consistency_loss
                if hasattr(outputs, 'entropy_loss'):
                    total_entropy_loss += outputs.entropy_loss
                if hasattr(outputs, 'theoretical_flops_reduction'):
                    total_theoretical_flops_reduction += outputs.theoretical_flops_reduction
                if hasattr(outputs, 'mean_abs_logits'):
                    total_mean_abs_logits += outputs.mean_abs_logits
                if hasattr(outputs, 'budget_loss'):
                    total_budget_loss += outputs.budget_loss
                if hasattr(outputs, 'density_regularization'):
                    total_density_regularization += outputs.density_regularization
            else:
                logits = outputs

            # Compute loss
            if targets is not None:
                loss, loss_components = compute_loss(logits, targets)
            else:
                # Fallback if no targets
                loss = torch.tensor(0.0, device=device)

        # Record loss components
        if config.numerical.record_loss_components:
            loss_components["total"] = loss.item()
            loss_monitor.record(loss_components)

        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient monitoring
        if config.numerical.record_grad_norms:
            grad_norm = grad_monitor.compute_total_grad_norm()
            total_grad_norm += grad_norm

            # 新增: 计算 backbone vs splitter 梯度比值
            layer_norms = grad_monitor.compute_grad_norms()
            backbone_norm = 0.0
            splitter_norm = 0.0
            for name, norm in layer_norms.items():
                # Splitter 参数通常包含 "splitter" 或 "tokenizer" 在名称中
                if 'splitter' in name.lower() or 'tokenizer' in name.lower() or 'gumbel' in name.lower():
                    splitter_norm += norm
                else:
                    backbone_norm += norm
            total_backbone_grad_norm += backbone_norm
            total_splitter_grad_norm += splitter_norm

            # I150-3 NEW: 计算关键组件（ManifoldDecoder等）的梯度
            component_norms = grad_monitor.compute_component_grad_norms()
            for comp_name, norm in component_norms.items():
                if comp_name == 'manifold_decoder':
                    total_manifold_decoder_grad_norm += norm
                elif comp_name == 'entmax':
                    total_entmax_grad_norm += norm

        # 显存峰值监控
        if torch.cuda.is_available():
            current_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            if current_memory_mb > peak_memory_mb:
                peak_memory_mb = current_memory_mb

        # I-NAN: 计算裁剪前的梯度范数（用于 NaN 调试）
        pre_clip_grad_norm = 0.0
        try:
            pre_clip_grad_norm = sum(
                p.grad.norm().item()
                for p in model.parameters()
                if p.grad is not None
            ) or 0.0
        except Exception:
            pass

        # Numerical defense
        should_skip = defender.post_backward()

        if should_skip:
            # Gradient clipping
            if config.training.gradient_clip_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.training.gradient_clip_norm,
                )

            # Update scheduler BEFORE optimizer step
            if scheduler is not None:
                scheduler.step(state.global_step)

            # Optimizer step
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        else:
            # I-NAN: 检测到 NaN！触发自动取证
            # 收集 Splitter logits 统计（如果有）
            splitter_logits_stats: Dict[str, float] = {}
            if hasattr(outputs, 'logits') and outputs.logits is not None:
                lgt = outputs.logits.detach()
                splitter_logits_stats = {
                    "logits_mean": float(lgt.mean()),
                    "logits_std": float(lgt.std()),
                    "logits_min": float(lgt.min()),
                    "logits_max": float(lgt.max()),
                    "logits_has_nan": bool(torch.isnan(lgt).any()),
                    "logits_has_inf": bool(torch.isinf(lgt).any()),
                }

            # 获取训练环境信息
            current_lr = optimizer.param_groups[0]["lr"]
            amp_loss_scale = float(scaler.get_scale()) if scaler is not None else None

            # 获取 Loss Components
            current_loss_components = loss_monitor.get_last_components() if hasattr(loss_monitor, 'get_last_components') else {}

            # 触发 NaN 自动取证
            debug_path = nan_investigator.investigate(
                epoch=state.epoch,
                step=state.global_step,
                loss_value=loss.item(),
                pre_clip_grad_norm=pre_clip_grad_norm,
                input_stats=_input_stats,
                splitter_logits_stats=splitter_logits_stats,
                amp_loss_scale=amp_loss_scale,
                learning_rate=current_lr,
                loss_components=current_loss_components,
            )
            if debug_path:
                print(f"[CRITICAL] NaN detected! Debug info: {debug_path}")

            # Skip this step due to numerical issues
            # Still need to update scheduler even when skipping
            if scheduler is not None:
                scheduler.step(state.global_step)
            optimizer.zero_grad()
            skipped_steps += 1
            state.nan_skip_count += 1

        # Note: Scheduler is now updated inside the conditional block above

        # Compute accuracy (if targets available)
        if targets is not None and logits is not None:
            if targets.dim() == 2:
                # Mixed labels from Mixup/Cutmix - compute approximate accuracy
                pred = logits.argmax(dim=-1)
                target_cls = targets.argmax(dim=-1)
                correct = (pred == target_cls).sum().item()
            else:
                correct = (logits.argmax(dim=-1) == targets).sum().item()

            batch_size = images.size(0)
            total_correct += correct
            total_samples += batch_size

        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1
        state.increment_step()

        # Logging
        if (batch_idx + 1) % config.training.log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Step [{batch_idx + 1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"LR: {current_lr:.2e} "
                  f"Grad: {grad_norm:.4f}" if config.numerical.record_grad_norms else "")

    epoch_time = time.time() - epoch_start_time

    # Compute final metrics
    avg_loss = total_loss / max(num_batches, 1)
    accuracy = total_correct / max(total_samples, 1) if total_samples > 0 else 0.0
    avg_grad_norm = total_grad_norm / max(num_batches, 1)
    avg_tokens = total_tokens / max(num_batches, 1)
    samples_per_second = total_samples / max(epoch_time, 1e-8)

    # Get loss components
    loss_components = loss_monitor.get_average_components()

    # Get layer gradient norms
    layer_grad_norms = grad_monitor.get_layer_statistics()

    # 新增: 计算平均指标
    avg_splitter_logits_mean = total_splitter_logits_mean / max(num_batches, 1)
    avg_splitter_logits_std = total_splitter_logits_std / max(num_batches, 1)
    avg_active_ratio = total_active_ratio / max(num_batches, 1)
    avg_manifold_bias_max = total_manifold_bias_max / max(num_batches, 1)
    avg_manifold_bias_min = total_manifold_bias_min / max(num_batches, 1)
    avg_manifold_bias_mean = total_manifold_bias_mean / max(num_batches, 1)
    avg_manifold_bias_std = total_manifold_bias_std / max(num_batches, 1)
    avg_poincare_dist_mean = total_poincare_dist_mean / max(num_batches, 1)
    avg_poincare_dist_std = total_poincare_dist_std / max(num_batches, 1)
    avg_backbone_grad_norm = total_backbone_grad_norm / max(num_batches, 1)
    avg_splitter_grad_norm = total_splitter_grad_norm / max(num_batches, 1)
    avg_entmax_grad_norm = total_entmax_grad_norm / max(num_batches, 1)  # I150-3 NEW
    avg_manifold_decoder_grad_norm = total_manifold_decoder_grad_norm / max(num_batches, 1)  # I150-3 NEW
    avg_budget_penalty = total_budget_penalty / max(num_batches, 1)
    avg_consistency_loss = total_consistency_loss / max(num_batches, 1)
    avg_entropy_loss = total_entropy_loss / max(num_batches, 1)
    avg_theoretical_flops_reduction = total_theoretical_flops_reduction / max(num_batches, 1)
    avg_mean_abs_logits = total_mean_abs_logits / max(num_batches, 1)  # I150-3 NEW
    avg_budget_loss = total_budget_loss / max(num_batches, 1)  # I150-3 NEW
    avg_density_regularization = total_density_regularization / max(num_batches, 1)  # I150-3 NEW

    # 计算梯度比值
    backbone_vs_splitter_ratio = 0.0
    if avg_splitter_grad_norm > 0:
        backbone_vs_splitter_ratio = avg_backbone_grad_norm / avg_splitter_grad_norm

    # Create metrics
    metrics = EpochMetrics(
        loss=avg_loss,
        loss_components=loss_components,
        accuracy=accuracy,
        avg_tokens=avg_tokens,
        grad_norm=avg_grad_norm,
        layer_grad_norms={k: v.get("mean", 0.0) for k, v in layer_grad_norms.items()},
        learning_rate=optimizer.param_groups[0]["lr"],
        epoch_time=epoch_time,
        samples_per_second=samples_per_second,
        skipped_steps=skipped_steps,
        peak_memory_mb=peak_memory_mb,
        splitter_logits_mean=avg_splitter_logits_mean,
        splitter_logits_std=avg_splitter_logits_std,
        active_ratio=avg_active_ratio,
        manifold_bias_max=avg_manifold_bias_max,
        manifold_bias_min=avg_manifold_bias_min,
        manifold_bias_mean=avg_manifold_bias_mean,
        manifold_bias_std=avg_manifold_bias_std,
        poincare_dist_mean=avg_poincare_dist_mean,
        poincare_dist_std=avg_poincare_dist_std,
        backbone_grad_norm=avg_backbone_grad_norm,
        splitter_grad_norm=avg_splitter_grad_norm,
        entmax_grad_norm=avg_entmax_grad_norm,  # I150-3 NEW
        manifold_decoder_grad_norm=avg_manifold_decoder_grad_norm,  # I150-3 NEW
        backbone_vs_splitter_grad_ratio=backbone_vs_splitter_ratio,
        budget_penalty=avg_budget_penalty,
        consistency_loss=avg_consistency_loss,
        entropy_loss=avg_entropy_loss,
        theoretical_flops_reduction=avg_theoretical_flops_reduction,
        mean_abs_logits=avg_mean_abs_logits,  # I150-3 NEW
        budget_loss=avg_budget_loss,  # I150-3 NEW
        density_regularization=avg_density_regularization,  # I150-3 NEW
    )

    # Memory stats
    if torch.cuda.is_available():
        metrics.memory_allocated_mb = torch.cuda.memory_allocated() / 1024**2
        metrics.memory_reserved_mb = torch.cuda.memory_reserved() / 1024**2
        # Reset peak memory for next epoch
        torch.cuda.reset_peak_memory_stats()

    return metrics


def train_one_epoch_simple(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 1,
    gradient_clip_norm: float = 1.0,
    log_interval: int = 50,
) -> Dict[str, float]:
    """Simplified training loop (minimal version)

    For quick testing without full config.

    Args:
        model: Model to train
        dataloader: Data loader
        optimizer: Optimizer
        device: Device
        num_epochs: Number of epochs
        gradient_clip_norm: Gradient clipping threshold
        log_interval: Logging interval

    Returns:
        Dictionary of metrics
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            images = batch[0].to(device)
            labels = batch[1].to(device)

            optimizer.zero_grad()

            outputs = model(images)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits

            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()

            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % log_interval == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] "
                      f"Batch [{batch_idx + 1}] "
                      f"Loss: {loss.item():.4f}")

    return {
        "avg_loss": total_loss / max(num_batches, 1),
    }


__all__ = [
    "train_one_epoch",
    "train_one_epoch_simple",
]
