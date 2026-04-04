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
    collector: Optional[Any] = None,  # NEW: Optional MetricsCollector
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

    # Initialize monitors (use collector if provided)
    grad_monitor = GradientMonitor(
        model=model,
        record_layer_norms=config.numerical.record_layer_grad_norms,
        hooks_enabled=config.numerical.record_grad_norms,
        collector=collector,
    )
    # I-NAN: 注册梯度 hooks 以启用 layer_norms 追踪
    if config.numerical.record_grad_norms:
        grad_monitor.register_hooks(model)
    loss_monitor = LossMonitor(collector=collector)
    defender = NumericalDefender(
        model=model,
        detect_anomaly=config.numerical.detect_anomaly,
        skip_on_nan=config.numerical.skip_on_nan_grad,
        collector=collector,
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
    # I-AUDIT: 使用 None 初始值，只有在收到有效数据时才累加
    total_splitter_logits_mean: Optional[float] = None
    total_splitter_logits_std: Optional[float] = None
    _splitter_logits_count: int = 0  # I-AUDIT: 跟踪有效值数量
    total_active_ratio: Optional[float] = None
    _active_ratio_count: int = 0
    total_manifold_bias_max: Optional[float] = None
    total_manifold_bias_min: Optional[float] = None
    total_manifold_bias_mean: Optional[float] = None
    total_manifold_bias_std: Optional[float] = None
    _manifold_bias_count: int = 0
    total_poincare_dist_mean: Optional[float] = None
    total_poincare_dist_std: Optional[float] = None
    _poincare_dist_count: int = 0
    total_backbone_grad_norm: Optional[float] = None
    _backbone_grad_count: int = 0
    total_splitter_grad_norm: Optional[float] = None
    _splitter_grad_count: int = 0
    total_entmax_grad_norm: Optional[float] = None
    _entmax_grad_count: int = 0
    total_manifold_decoder_grad_norm: Optional[float] = None
    _manifold_decoder_grad_count: int = 0
    total_budget_penalty: Optional[float] = None
    _budget_penalty_count: int = 0
    total_consistency_loss: Optional[float] = None
    _consistency_loss_count: int = 0
    total_entropy_loss: Optional[float] = None
    _entropy_loss_count: int = 0
    total_theoretical_flops_reduction: Optional[float] = None
    _theoretical_flops_count: int = 0
    total_mean_abs_logits: Optional[float] = None  # I150-3 NEW: Splitter Logits 平均绝对值
    _mean_abs_logits_count: int = 0
    total_budget_loss: Optional[float] = None  # I150-3 NEW: Elastic Budget 损失
    _budget_loss_count: int = 0
    total_density_regularization: Optional[float] = None  # I150-3 NEW: 密度正则化损失
    _density_reg_count: int = 0

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
                # I-AUDIT: 使用计数器跟踪有效值数量，避免平均值计算时除以错误分母
                if outputs.splitter_logits_mean is not None:
                    total_splitter_logits_mean = (total_splitter_logits_mean or 0.0) + outputs.splitter_logits_mean
                    total_splitter_logits_std = (total_splitter_logits_std or 0.0) + (outputs.splitter_logits_std or 0.0)
                    _splitter_logits_count += 1
                if outputs.active_ratio is not None:
                    total_active_ratio = (total_active_ratio or 0.0) + outputs.active_ratio
                    _active_ratio_count += 1
                if outputs.manifold_bias_max is not None:
                    total_manifold_bias_max = (total_manifold_bias_max or 0.0) + outputs.manifold_bias_max
                    total_manifold_bias_min = (total_manifold_bias_min or 0.0) + (outputs.manifold_bias_min or 0.0)
                    total_manifold_bias_mean = (total_manifold_bias_mean or 0.0) + (outputs.manifold_bias_mean or 0.0)
                    total_manifold_bias_std = (total_manifold_bias_std or 0.0) + (outputs.manifold_bias_std or 0.0)
                    _manifold_bias_count += 1
                if outputs.poincare_dist_mean is not None:
                    total_poincare_dist_mean = (total_poincare_dist_mean or 0.0) + outputs.poincare_dist_mean
                    total_poincare_dist_std = (total_poincare_dist_std or 0.0) + (outputs.poincare_dist_std or 0.0)
                    _poincare_dist_count += 1
                if outputs.budget_penalty is not None:
                    total_budget_penalty = (total_budget_penalty or 0.0) + abs(outputs.budget_penalty)
                    _budget_penalty_count += 1
                if outputs.consistency_loss is not None:
                    total_consistency_loss = (total_consistency_loss or 0.0) + outputs.consistency_loss
                    _consistency_loss_count += 1
                if outputs.entropy_loss is not None:
                    total_entropy_loss = (total_entropy_loss or 0.0) + outputs.entropy_loss
                    _entropy_loss_count += 1
                if outputs.theoretical_flops_reduction is not None:
                    total_theoretical_flops_reduction = (total_theoretical_flops_reduction or 0.0) + outputs.theoretical_flops_reduction
                    _theoretical_flops_count += 1
                if outputs.mean_abs_logits is not None:
                    total_mean_abs_logits = (total_mean_abs_logits or 0.0) + outputs.mean_abs_logits
                    _mean_abs_logits_count += 1
                if outputs.budget_loss is not None:
                    total_budget_loss = (total_budget_loss or 0.0) + outputs.budget_loss
                    _budget_loss_count += 1
                if outputs.density_regularization is not None:
                    total_density_regularization = (total_density_regularization or 0.0) + outputs.density_regularization
                    _density_reg_count += 1
            else:
                logits = outputs

            # Compute loss
            if targets is not None:
                # I-AUDIT: 传递 H1SS 辅助损失给 compute_loss
                aux_losses = outputs.auxiliary_losses if hasattr(outputs, 'auxiliary_losses') else None
                loss, loss_components = compute_loss(logits, targets, aux_losses=aux_losses)
            else:
                # Fallback if no targets
                loss = torch.tensor(0.0, device=device)

        # Record loss components
        if config.numerical.record_loss_components:
            loss_components["total"] = loss.item()
            # I-AUDIT: 使用 is not None 检查，TrainingStats 字段现在是 Optional[float] = None
            if outputs.budget_loss is not None:
                loss_components["budget_loss"] = outputs.budget_loss
            if outputs.density_regularization is not None:
                loss_components["density_regularization"] = outputs.density_regularization
            if outputs.consistency_loss is not None:
                loss_components["consistency_loss"] = outputs.consistency_loss
            if outputs.entropy_loss is not None:
                loss_components["entropy_loss"] = outputs.entropy_loss
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

            # I150-3 FIX: 计算 backbone 和 splitter 的梯度范数（按参数名前缀分类）
            # 按照日志系统规范: backbone_grad_norm 和 splitter_grad_norm 是 TrainingStats 直接字段
            layer_norms = grad_monitor.compute_grad_norms()
            if layer_norms:
                backbone_grad_sq = 0.0
                splitter_grad_sq = 0.0
                entmax_grad_sq = 0.0
                manifold_decoder_grad_sq = 0.0
                for param_name, norm in layer_norms.items():
                    if param_name.startswith("splitter."):
                        splitter_grad_sq += norm ** 2
                    elif "entmax" in param_name.lower() or "bottleneck" in param_name.lower():
                        # I150-3: 追踪 entmax 和 bottleneck 梯度
                        entmax_grad_sq += norm ** 2
                    elif "geo_decoder" in param_name.lower() or "manifold_decoder" in param_name.lower():
                        # I150-3: 追踪 manifold_decoder 梯度
                        manifold_decoder_grad_sq += norm ** 2
                    else:
                        backbone_grad_sq += norm ** 2
                backbone_grad_norm = backbone_grad_sq ** 0.5
                splitter_grad_norm = splitter_grad_sq ** 0.5
                entmax_grad_norm = entmax_grad_sq ** 0.5 if entmax_grad_sq > 0 else 0.0
                manifold_decoder_grad_norm = manifold_decoder_grad_sq ** 0.5 if manifold_decoder_grad_sq > 0 else 0.0
                # 累加到 epoch 统计
                if total_backbone_grad_norm is None:
                    total_backbone_grad_norm = backbone_grad_norm
                    total_splitter_grad_norm = splitter_grad_norm
                    total_entmax_grad_norm = entmax_grad_norm
                    total_manifold_decoder_grad_norm = manifold_decoder_grad_norm
                    _backbone_grad_count = 1
                    _splitter_grad_count = 1
                    _entmax_grad_count = 1
                    _manifold_decoder_grad_count = 1
                else:
                    total_backbone_grad_norm += backbone_grad_norm
                    total_splitter_grad_norm += splitter_grad_norm
                    total_entmax_grad_norm += entmax_grad_norm
                    total_manifold_decoder_grad_norm += manifold_decoder_grad_norm
                    _backbone_grad_count += 1
                    _splitter_grad_count += 1
                    _entmax_grad_count += 1
                    _manifold_decoder_grad_count += 1

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
            # 收集 Classification logits 统计（outputs.logits 是分类 logits，不是 splitter 内部 logits）
            classification_logits_stats: Dict[str, float] = {}
            if hasattr(outputs, 'logits') and outputs.logits is not None:
                lgt = outputs.logits.detach()
                classification_logits_stats = {
                    "logits_mean": float(lgt.mean()),
                    "logits_std": float(lgt.std()),
                    "logits_min": float(lgt.min()),
                    "logits_max": float(lgt.max()),
                    "logits_has_nan": bool(torch.isnan(lgt).any()),
                    "logits_has_inf": bool(torch.isinf(lgt).any()),
                }

            # 收集特征模长统计 (mlp_head 前的特征)
            feature_stats: Dict[str, float] = {}
            if hasattr(outputs, 'features') and outputs.features is not None:
                feat = outputs.features.detach()
                feature_stats = {
                    "mean": float(feat.mean()),
                    "std": float(feat.std()),
                    "min": float(feat.min()),
                    "max": float(feat.max()),
                    "norm": float(feat.norm()),
                    "has_nan": bool(torch.isnan(feat).any()),
                    "has_inf": bool(torch.isinf(feat).any()),
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
                classification_logits_stats=classification_logits_stats,
                feature_stats=feature_stats,
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

    # P4-A FIX: 从 layer_grad_norms（累积的hook数据）计算 manifold_decoder 梯度范数
    # 问题根源: compute_grad_norms() 返回的 param.grad.norm() 与 hooks 累积的数据不一致
    # 解决: 直接从 layer_grad_norms 的均值统计计算，确保与 stats 文件中的 layer_grad_norms 一致
    manifold_decoder_grad_sq = 0.0
    for param_name, stats in layer_grad_norms.items():
        if isinstance(stats, dict) and "mean" in stats:
            mean_norm = stats["mean"]
        elif isinstance(stats, (int, float)):
            mean_norm = stats  # 有些可能是直接的值
        else:
            continue
        if "geo_decoder" in param_name.lower() or "manifold_decoder" in param_name.lower():
            manifold_decoder_grad_sq += mean_norm ** 2
    avg_manifold_decoder_grad_norm = manifold_decoder_grad_sq ** 0.5 if manifold_decoder_grad_sq > 0 else 0.0

    # 新增: 计算平均指标
    # I-AUDIT: 使用 None 检查，只有在有数据时才计算平均值
    def safe_avg(total, num_batches):
        """安全计算平均值，如果 total 为 None 则返回 None"""
        if total is None or num_batches == 0:
            return None
        return total / num_batches

    # I-AUDIT: 使用各自的计数器计算平均值，避免除以错误分母
    avg_splitter_logits_mean = safe_avg(total_splitter_logits_mean, _splitter_logits_count)
    avg_splitter_logits_std = safe_avg(total_splitter_logits_std, _splitter_logits_count)
    avg_active_ratio = safe_avg(total_active_ratio, _active_ratio_count)
    avg_manifold_bias_max = safe_avg(total_manifold_bias_max, _manifold_bias_count)
    avg_manifold_bias_min = safe_avg(total_manifold_bias_min, _manifold_bias_count)
    avg_manifold_bias_mean = safe_avg(total_manifold_bias_mean, _manifold_bias_count)
    avg_manifold_bias_std = safe_avg(total_manifold_bias_std, _manifold_bias_count)
    avg_poincare_dist_mean = safe_avg(total_poincare_dist_mean, _poincare_dist_count)
    avg_poincare_dist_std = safe_avg(total_poincare_dist_std, _poincare_dist_count)
    avg_backbone_grad_norm = safe_avg(total_backbone_grad_norm, _backbone_grad_count)
    avg_splitter_grad_norm = safe_avg(total_splitter_grad_norm, _splitter_grad_count)
    avg_entmax_grad_norm = safe_avg(total_entmax_grad_norm, _entmax_grad_count)  # I150-3 NEW
    # P4-A FIX: 使用 epoch 级别从 layer_grad_norms（hooks 累积数据）计算的 manifold_decoder 梯度范数
    # 注意: per-step compute_grad_norms() 与 hooks 累积数据不一致，改用 get_layer_statistics() 的数据
    # avg_manifold_decoder_grad_norm 已在 line 472 从 layer_grad_norms 计算，直接使用
    # (line 495 被注释，不再用 per-step 的污染数据覆盖正确值)
    # avg_manifold_decoder_grad_norm = safe_avg(total_manifold_decoder_grad_norm, _manifold_decoder_grad_count)
    avg_budget_penalty = safe_avg(total_budget_penalty, _budget_penalty_count)
    avg_consistency_loss = safe_avg(total_consistency_loss, _consistency_loss_count)
    avg_entropy_loss = safe_avg(total_entropy_loss, _entropy_loss_count)
    avg_theoretical_flops_reduction = safe_avg(total_theoretical_flops_reduction, _theoretical_flops_count)
    avg_mean_abs_logits = safe_avg(total_mean_abs_logits, _mean_abs_logits_count)  # I150-3 NEW
    avg_budget_loss = safe_avg(total_budget_loss, _budget_loss_count)  # I150-3 NEW
    avg_density_regularization = safe_avg(total_density_regularization, _density_reg_count)  # I150-3 NEW

    # 计算梯度比值
    backbone_vs_splitter_ratio = None
    if avg_splitter_grad_norm is not None and avg_splitter_grad_norm > 0:
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
