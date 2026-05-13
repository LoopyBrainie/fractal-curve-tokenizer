"""Training Epoch Module

Implements independent train_one_epoch function.
Following the three-layer parameter principle, this module
handles Layer 3 (hyperparameters) for training loop.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List, Union
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
from ..monitor.numerical_defense import NumericalDefender, NaNAutoInvestigation


def _get_flatten_layer_outputs():
    """Lazy import to avoid module resolution order issues in containers."""
    from vit_pytorch.core.layer_output import flatten_layer_outputs
    return flatten_layer_outputs


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
    warmup_params: Optional[dict] = None,  # Phase 4: only {'tau': current_tau}
    # I-OOM FIX: Monitors now passed from outside to prevent O(N^2) hook leak
    grad_monitor: Optional[GradientMonitor] = None,
    loss_monitor: Optional[LossMonitor] = None,
    defender: Optional[NumericalDefender] = None,
    nan_investigator: Optional[NaNAutoInvestigation] = None,
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
    from tqdm import tqdm

    model.train()

    # I-OOM FIX: Only create monitors if not provided from outside
    # This prevents O(N^2) hook leak where each epoch created new hooks
    if grad_monitor is None:
        grad_monitor = GradientMonitor(
            model=model,
            record_layer_norms=config.numerical.record_layer_grad_norms,
            hooks_enabled=config.numerical.record_grad_norms,
            collector=collector,
        )
        # I-NAN: 注册梯度 hooks 以启用 layer_norms 追踪
        if config.numerical.record_grad_norms:
            grad_monitor.register_hooks(model)

    if loss_monitor is None:
        loss_monitor = LossMonitor(collector=collector)

    if defender is None:
        defender = NumericalDefender(
            model=model,
            detect_anomaly=config.numerical.detect_anomaly,
            skip_on_nan=config.numerical.skip_on_nan_grad,
            collector=collector,
        )
        defender.register_discovery_hooks()

    # I-NAN: 初始化 NaN 自动取证器
    # debug_dir 默认为实验目录下的 debug 子目录
    # I-OOM FIX: Only create if not provided
    if nan_investigator is None:
        _debug_dir = debug_dir if debug_dir else "experiments/debug"
        nan_investigator = NaNAutoInvestigation(
            model=model,
            debug_dir=_debug_dir,
            enabled=True,  # 始终启用，用于捕获第一次 NaN
        )

    # 用于记录输入数据统计（用于调试）
    _input_stats: Dict[str, float] = {}

    # Metrics accumulators (I-OPT: tensor accumulation, single .item() at epoch end)
    total_loss: Union[torch.Tensor, float] = 0.0
    total_correct: Union[torch.Tensor, int] = 0
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
    total_backbone_grad_norm: Optional[float] = None
    _backbone_grad_count: int = 0
    total_splitter_grad_norm: Optional[float] = None
    _splitter_grad_count: int = 0
    total_theoretical_flops_reduction: Optional[float] = None
    _theoretical_flops_count: int = 0
    total_mean_abs_logits: Optional[float] = None  # I150-3 NEW: Splitter Logits 平均绝对值
    _mean_abs_logits_count: int = 0
    total_raw_budget_error: Optional[float] = None  # D162: 重命名 (原 budget_loss)
    _raw_budget_error_count: int = 0
    total_density_regularization: Optional[float] = None  # I150-3 NEW: 密度正则化损失
    _density_reg_count: int = 0

    # Auxiliary outputs accumulator: maps flat key → list of per-batch values.
    # Populated by flatten_layer_outputs() on each forward pass.
    _aux_flat_accum: Dict[str, List[float]] = {}

    # 显存峰值
    peak_memory_mb = 0.0

    epoch_start_time = time.time()

    # Training loop
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {state.epoch}", leave=False)
    for batch_idx, batch in pbar:
        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            images = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)
        else:
            images = batch.to(device, non_blocking=True)
            labels = None

        # I-NAN: 记录输入数据统计（用于 NaN 调试）- D1-SYNC: 延迟 .item() 到 investigate()
        _input_stats = {
            "images_mean": images.mean().detach(),
            "images_std": images.std().detach(),
            "images_min": images.min().detach(),
            "images_max": images.max().detach(),
            "images_has_nan": torch.isnan(images).any().detach(),
            "images_has_inf": torch.isinf(images).any().detach(),
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
            # P0 FIX: num_classes fallback - 从 model 属性获取，若无则报错
            if labels is not None:
                model_classes = getattr(model, 'num_classes', None)
                if model_classes is None:
                    raise ValueError(
                        "model.num_classes is required for one-hot encoding when labels provided. "
                        "Ensure model architecture has num_classes attribute set."
                    )
                targets = torch.nn.functional.one_hot(labels, model_classes).float()
            else:
                targets = None

        # Forward pass with AMP
        with amp_autocast('cuda', enabled=config.amp.enabled):
            # MEM-OOM FIX (Suspect 3): OOM 诊断 try-except，快速定位"罪魁祸首"
            # 当 OOM 发生时，打印 Batch 形状和 Token 数量，帮助定位异常样本
            try:
                outputs = model(images)
            except torch.cuda.OutOfMemoryError as e:
                # OOM 遥测: 打印致命调试信息
                print(f"\n[CRITICAL] CUDA OOM at Step {state.global_step}, Batch {batch_idx}")
                print(f"  images.shape: {images.shape}")
                k_info = f", splitter K_max={model.splitter.K_max}" if hasattr(model, 'splitter') else ""
                print(f"  Batch size: {images.shape[0]}, Image size: {images.shape[2:]}{k_info}")

                # 导出完整的显存分配摘要到文本文件，供事后分析 (Post-mortem)
                try:
                    oom_file = f"oom_summary_step_{state.global_step}.txt"
                    with open(oom_file, "w") as f:
                        f.write(torch.cuda.memory_summary(device=images.device))
                    print(f"  OOM memory summary saved to: {oom_file}")
                except Exception:
                    pass  # 不因日志问题影响异常传播

                torch.cuda.empty_cache()
                raise

            # Handle TrainingStats from Fractal ViT
            if hasattr(outputs, 'logits'):
                logits = outputs.logits

                # Extract token info if available
                # I-OPT: 延迟 .item()，使用 tensor 累加模式
                if hasattr(outputs, 'num_tokens'):
                    num_tokens_raw = outputs.num_tokens
                    # Handle different types (int, tensor, list)
                    if isinstance(num_tokens_raw, torch.Tensor):
                        # I-OPT: 直接累加 tensor，不调用 .item() 强制同步
                        total_tokens += num_tokens_raw.float().mean().detach()
                    elif isinstance(num_tokens_raw, (int, float)):
                        total_tokens += float(num_tokens_raw)
                    else:
                        total_tokens += float(sum(num_tokens_raw) / len(num_tokens_raw))

                # 新增: 提取实验详细日志指标
                # I-AUDIT: 使用计数器跟踪有效值数量，避免平均值计算时除以错误分母
                if outputs.splitter_logits_mean is not None:
                    total_splitter_logits_mean = (total_splitter_logits_mean or 0.0) + outputs.splitter_logits_mean.detach()
                    total_splitter_logits_std = (total_splitter_logits_std or 0.0) + (outputs.splitter_logits_std.detach() if outputs.splitter_logits_std is not None else 0.0)
                    _splitter_logits_count += 1
                if outputs.active_ratio is not None:
                    total_active_ratio = (total_active_ratio or 0.0) + outputs.active_ratio.detach()
                    _active_ratio_count += 1
                if outputs.theoretical_flops_reduction is not None:
                    total_theoretical_flops_reduction = (total_theoretical_flops_reduction or 0.0) + outputs.theoretical_flops_reduction.detach()
                    _theoretical_flops_count += 1
                if outputs.mean_abs_logits is not None:
                    total_mean_abs_logits = (total_mean_abs_logits or 0.0) + outputs.mean_abs_logits.detach()
                    _mean_abs_logits_count += 1
                if outputs.raw_budget_error is not None:
                    total_raw_budget_error = (total_raw_budget_error or 0.0) + outputs.raw_budget_error.detach()
                    _raw_budget_error_count += 1
                if outputs.density_regularization is not None:
                    total_density_regularization = (total_density_regularization or 0.0) + outputs.density_regularization.detach()
                    _density_reg_count += 1

                # auxiliary_outputs flattening (layer-packaged → trainer-unpacked)
                if hasattr(outputs, 'auxiliary_outputs') and outputs.auxiliary_outputs:
                    _flat = _get_flatten_layer_outputs()(outputs.auxiliary_outputs, prefix="train")
                    for _k, _v in _flat.items():
                        if _k not in _aux_flat_accum:
                            _aux_flat_accum[_k] = []
                        # FIX: flatten_layer_outputs 返回 Dict[str, float]，但防御性检查 tensor
                        if isinstance(_v, torch.Tensor):
                            _aux_flat_accum[_k].append(_v.detach().to('cpu', non_blocking=True))
                        else:
                            _aux_flat_accum[_k].append(_v)
            else:
                logits = outputs

            # Compute loss
            if targets is not None:
                loss, loss_components = compute_loss(logits, targets)
            else:
                # Fallback if no targets
                loss = torch.tensor(0.0, device=device)
                loss_components = {}

# Backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 将 loss component 记录从 forward 路径移到此处，确保 backward 可以先完成
        if config.numerical.record_loss_components and targets is not None:
            loss_components_float = {}
            for k, v in loss_components.items():
                if isinstance(v, torch.Tensor):
                    loss_components_float[k] = v.detach().item()
                else:
                    loss_components_float[k] = v
            loss_components_float["total"] = loss.detach().item()
            # I-AUDIT: 使用 is not None 检查，TrainingStats 字段现在是 Optional[float] = None
            if outputs.raw_budget_error is not None:
                loss_components_float["raw_budget_error"] = outputs.raw_budget_error.detach().item() if isinstance(outputs.raw_budget_error, torch.Tensor) else outputs.raw_budget_error
            if outputs.density_regularization is not None:
                loss_components_float["density_regularization"] = outputs.density_regularization.detach().item() if isinstance(outputs.density_regularization, torch.Tensor) else outputs.density_regularization
            loss_monitor.record(loss_components_float)

        # I-OOM FIX: 调用 finalize 将 GPU tensors 转为 Python floats，释放显存
        if grad_monitor is not None:
            grad_monitor.finalize()

        # MEMORY DIAGNOSTIC: 每10步监控显存，定位暴涨时刻
        if torch.cuda.is_available() and (batch_idx + 1) % 10 == 0:
            allocated_mb = torch.cuda.memory_allocated() / 1024**2
            reserved_mb = torch.cuda.memory_reserved() / 1024**2
            max_allocated_mb = torch.cuda.max_memory_allocated() / 1024**2
            # 获取 num_tokens (从 forward 时获取的 outputs)
            # I-OPT: 使用 .detach().item() 避免阻塞 backward 后的 GPU 流水线
            num_tokens_info = ""
            if hasattr(outputs, 'num_tokens') and outputs.num_tokens is not None:
                ntok = outputs.num_tokens
                if isinstance(ntok, torch.Tensor):
                    ntok = ntok.detach().float().mean().item()
                num_tokens_info = f", tokens={ntok:.0f}"
            print(f"  [MEM] Step {batch_idx+1}: alloc={allocated_mb:.1f}MB, reserved={reserved_mb:.1f}MB, peak={max_allocated_mb:.1f}MB{num_tokens_info}")

            # P4-Fix: 每 50 步清理显存碎片，防止 reserved 持续增长
            if (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()

        # Gradient monitoring
        if config.numerical.record_grad_norms:
            grad_norm = grad_monitor.compute_total_grad_norm()
            total_grad_norm += grad_norm

            # I150-3 FIX: 计算 backbone 和 splitter 的梯度范数（按参数名前缀分类）
            # 按照日志系统规范: backbone_grad_norm 和 splitter_grad_norm 是 TrainingStats 直接字段
            layer_norms = grad_monitor.compute_grad_norms()
            # P1 FIX: 如果 Hook 失效，添加手动审计
            if not layer_norms and (batch_idx + 1) % config.training.log_interval == 0:
                print(f"[CRITICAL] Gradient hooks failed at Step {state.global_step}. Manual Audit:")
                for name, param in model.named_parameters():
                    if "splitter" in name:
                        g_norm = param.grad.norm().item() if param.grad is not None else "NONE"
                        print(f"  - {name}: grad_norm = {g_norm}")
            if layer_norms:
                backbone_grad_sq = 0.0
                splitter_grad_sq = 0.0
                for param_name, norm in layer_norms.items():
                    if param_name.startswith("splitter."):
                        splitter_grad_sq += norm ** 2
                    else:
                        backbone_grad_sq += norm ** 2
                backbone_grad_norm = backbone_grad_sq ** 0.5
                splitter_grad_norm = splitter_grad_sq ** 0.5
                # 累加到 epoch 统计
                if total_backbone_grad_norm is None:
                    total_backbone_grad_norm = backbone_grad_norm
                    total_splitter_grad_norm = splitter_grad_norm
                    _backbone_grad_count = 1
                    _splitter_grad_count = 1
                else:
                    total_backbone_grad_norm += backbone_grad_norm
                    total_splitter_grad_norm += splitter_grad_norm
                    _backbone_grad_count += 1
                    _splitter_grad_count += 1

            # C1: Spectral Norm 监控 - 每 100 步计算一次 Geometry 模块的谱范数
            # 谱范数 = 权重矩阵的最大奇异值，反映结构健康度
            # 异常阈值: >10 或突然翻倍预警
            if (state.global_step + 1) % (config.training.log_interval * 10) == 0:
                geo_spec_norms = grad_monitor.compute_geometry_spectral_norms()
                if geo_spec_norms:
                    for name, spec_norm in geo_spec_norms.items():
                        if spec_norm > 10.0:
                            print(f"  [WARN] Spectral norm explosion: {name} = {spec_norm:.2f}")

        # 显存峰值监控
        if torch.cuda.is_available():
            current_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            if current_memory_mb > peak_memory_mb:
                peak_memory_mb = current_memory_mb

        # I-NAN FIX: 重构 GradScaler 逻辑（方案 A）
        # 核心原则: PyTorch 官方推荐的健壮 AMP 写法
        # 1. unscale_ 必须在所有操作之前执行，让梯度恢复正常范围
        if scaler is not None:
            scaler.unscale_(optimizer)

        # 2. 鲁棒的梯度统计 (FP32 计算，防止 L2-norm 平方和溢出)
        # 使用 torch.no_grad() 包裹，避免污染计算图
        with torch.no_grad():
            try:
                device_for_norm = next((p.device for p in model.parameters() if p.grad is not None), torch.device('cpu'))
                total_norm_sq = torch.tensor(0.0, device=device_for_norm, dtype=torch.float32)
                for p in model.parameters():
                    if p.grad is not None:
                        # 强制 float32，防止平方和溢出
                        param_norm = p.grad.detach().float().pow(2).sum()
                        total_norm_sq += param_norm
                pre_clip_grad_norm = torch.sqrt(total_norm_sq).clamp(min=1e-6)
            except Exception:
                pre_clip_grad_norm = torch.tensor(float('nan'), device=device_for_norm)

        # 3. 梯度防御检查（使用恢复后的真实梯度）
        # BUG FIX: post_backward() 返回 True=可以继续, False=应该跳过
        # 原代码错误地将 True 当作"跳过"导致误报
        is_finite = torch.isfinite(pre_clip_grad_norm)
        if defender:
            # post_backward(): True= proceed(不skip), False= skip
            defender_says_skip = not defender.post_backward()
        else:
            defender_says_skip = False
        should_skip_step = not is_finite or defender_says_skip

        # A-NAN FIX: 修复 GradScaler "更新死锁"
        # 核心原则：无论是否 skip，scaler.update() 都必须在所有路径执行
        # 否则 loss_scale 永不下降，NaN 会像幽灵一样在计算图中循环

        if not should_skip_step:
            # Gradient clipping (P0-Fix: 使用配置值替代硬编码max_norm=1.0)
            if config.training.gradient_clip_norm > 0:
                # P0 修复: 改用 config.gradient_clip_norm，允许临时放宽到 1000 进行诊断
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config.training.gradient_clip_norm,
                )
                # A-NAN FIX: 检查裁剪后的梯度是否有限
                if not torch.isfinite(grad_norm):
                    print(f"Warning: Gradient norm is {grad_norm}, skipping step!")
                    should_skip_step = True  # 转换为 skip

            if not should_skip_step:
                # Update scheduler BEFORE optimizer step
                if scheduler is not None:
                    scheduler.step(state.global_step)

                # Optimizer step
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()  # A-NAN FIX: 正常路径也必须 update
                else:
                    optimizer.step()
                optimizer.zero_grad()

            # A-NAN FIX: scaler.update() 在所有路径后执行（已在上方正常路径执行）
            # 如果进入 skip 分支，下方的 scaler.update() 会执行

        if should_skip_step:
            # I-NAN: 检测到 NaN 或梯度异常！触发自动取证并 skip
            # 收集 Classification logits 统计（outputs.logits 是分类 logits，不是 splitter 内部 logits）
            classification_logits_stats: Dict[str, Any] = {}
            if hasattr(outputs, 'logits') and outputs.logits is not None:
                lgt = outputs.logits.detach()
                classification_logits_stats = {
                    "logits_mean": lgt.mean(),
                    "logits_std": lgt.std(),
                    "logits_min": lgt.min(),
                    "logits_max": lgt.max(),
                    "logits_has_nan": torch.isnan(lgt).any(),
                    "logits_has_inf": torch.isinf(lgt).any(),
                }

            # 收集特征模长统计 (mlp_head 前的特征) - D1-SYNC: 延迟 .item()
            feature_stats: Dict[str, Any] = {}
            if hasattr(outputs, 'features') and outputs.features is not None:
                feat = outputs.features.detach()
                feature_stats = {
                    "mean": feat.mean(),
                    "std": feat.std(),
                    "min": feat.min(),
                    "max": feat.max(),
                    "norm": feat.norm(),
                    "has_nan": torch.isnan(feat).any(),
                    "has_inf": torch.isinf(feat).any(),
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
            if scaler is not None:
                scaler.update()  # 即使 skip，也必须 update 以便自动降 scale
            skipped_steps += 1
            state.nan_skip_count += 1

        # Note: Scheduler is now updated inside the conditional block above

        # Compute accuracy (if targets available)
        # I-OPT: 延迟 .item()，使用 tensor 累加
        if targets is not None and logits is not None:
            if targets.dim() == 2:
                # Mixed labels from Mixup/Cutmix - compute approximate accuracy
                pred = logits.argmax(dim=-1)
                target_cls = targets.argmax(dim=-1)
                correct = (pred == target_cls).sum()
            else:
                correct = (logits.argmax(dim=-1) == targets).sum()

            batch_size = images.size(0)
            total_correct += correct.detach()
            total_samples += batch_size

        # I-OPT: 累积 tensor，不在循环内 .item()
        total_loss += loss.detach()
        num_batches += 1
        state.increment_step()

        # Logging
        if (batch_idx + 1) % config.training.log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            # I-OPT:延迟 .item() 到打印前，减少 GPU→CPU 同步频率
            grad_str = f"Grad: {grad_norm:.4f}" if config.numerical.record_grad_norms and grad_norm is not None else ""
            print(f"  Step [{batch_idx + 1}/{len(dataloader)}] "
                  f"Loss: {loss.detach().item():.4f} "
                  f"LR: {current_lr:.2e} "
                  f"{grad_str}")

    epoch_time = time.time() - epoch_start_time

    # Compute final metrics
    # I-OPT: 在 epoch 结束时一次性 .item()，避免每 batch 同步
    avg_loss = (total_loss / max(num_batches, 1)).item() if isinstance(total_loss, torch.Tensor) else total_loss / max(num_batches, 1)
    accuracy = (total_correct / max(total_samples, 1)).item() if isinstance(total_correct, torch.Tensor) else total_correct / max(total_samples, 1)
    avg_grad_norm = (total_grad_norm / max(num_batches, 1)).item() if isinstance(total_grad_norm, torch.Tensor) else total_grad_norm / max(num_batches, 1)
    avg_tokens = (total_tokens / max(num_batches, 1)).item() if isinstance(total_tokens, torch.Tensor) else total_tokens / max(num_batches, 1)
    samples_per_second = total_samples / max(epoch_time, 1e-8)

    # Get loss components
    loss_components = loss_monitor.get_average_components()

    # Get layer gradient norms
    layer_grad_norms = grad_monitor.get_layer_statistics()

    # P0: Layer-wise SNR 监控 - 诊断梯度展平和 Rank Collapse
    # SNR = mean / (std + 1e-6), SNR < 0.05 说明噪声主导
    layer_snr = {}
    low_snr_layers = []
    for param_name, stats in layer_grad_norms.items():
        if isinstance(stats, dict) and "mean" in stats and "std" in stats:
            snr = stats["mean"] / (stats["std"] + 1e-6)
            layer_snr[param_name] = snr
            if snr < 0.05:
                low_snr_layers.append((param_name, snr))

    if low_snr_layers and state.global_step % config.training.log_interval == 0:
        print(f"  [Layer-wise SNR] Low SNR detected ({len(low_snr_layers)} layers < 0.05):")
        for param_name, snr in sorted(low_snr_layers, key=lambda x: x[1])[:5]:
            print(f"    - {param_name[:60]}: SNR={snr:.4f}")

    # 新增: 计算平均指标
    def safe_avg(total, num_batches):
        """安全计算平均值，如果 total 为 None 则返回 None"""
        if total is None or num_batches == 0:
            return None
        return total / num_batches

    avg_splitter_logits_mean = safe_avg(total_splitter_logits_mean, _splitter_logits_count)
    avg_splitter_logits_std = safe_avg(total_splitter_logits_std, _splitter_logits_count)
    avg_active_ratio = safe_avg(total_active_ratio, _active_ratio_count)
    avg_backbone_grad_norm = safe_avg(total_backbone_grad_norm, _backbone_grad_count)
    avg_splitter_grad_norm = safe_avg(total_splitter_grad_norm, _splitter_grad_count)
    avg_theoretical_flops_reduction = safe_avg(total_theoretical_flops_reduction, _theoretical_flops_count)
    avg_mean_abs_logits = safe_avg(total_mean_abs_logits, _mean_abs_logits_count)
    avg_raw_budget_error = safe_avg(total_raw_budget_error, _raw_budget_error_count)
    avg_density_regularization = safe_avg(total_density_regularization, _density_reg_count)

    # Average auxiliary flat metrics across all batches.
    auxiliary_flat_metrics: Dict[str, float] = {
        k: sum(vs) / len(vs) for k, vs in _aux_flat_accum.items() if vs
    }

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
        backbone_grad_norm=avg_backbone_grad_norm,
        splitter_grad_norm=avg_splitter_grad_norm,
        backbone_vs_splitter_grad_ratio=backbone_vs_splitter_ratio,
        theoretical_flops_reduction=avg_theoretical_flops_reduction,
        mean_abs_logits=avg_mean_abs_logits,
        raw_budget_error=avg_raw_budget_error,
        density_regularization=avg_density_regularization,
        auxiliary_flat_metrics=auxiliary_flat_metrics,
    )

    # Memory stats
    if torch.cuda.is_available():
        metrics.memory_allocated_mb = torch.cuda.memory_allocated() / 1024**2
        metrics.memory_reserved_mb = torch.cuda.memory_reserved() / 1024**2
        # Reset peak memory for next epoch
        torch.cuda.reset_peak_memory_stats()

    # Phase 4: 仅记录 τ
    if warmup_params:
        metrics.current_tau = warmup_params.get('tau', 1.0)

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
            images = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)

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
