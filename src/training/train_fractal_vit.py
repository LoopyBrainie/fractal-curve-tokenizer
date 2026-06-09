#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FractalCurveViT 训练入口脚本

支持完整的 CLI 参数，与 shell 训练脚本完全对齐。
调用方式: python -m src.training.train_fractal_vit [args]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import random
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Add src directory to path for vit_pytorch imports
def _setup_path():
    src_dir = Path(__file__).parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
_setup_path()

from .config import create_config  # noqa: E402
from .trainer import (  # noqa: E402
    TrainingState,
    train_one_epoch,
    evaluate,
    MixupCutmixLoss,
)
from .scheduler import create_scheduler  # noqa: E402
# PR5c: monitor/ 子包已删除, loss_monitor → LossComponentsAccumulator (PR4 收敛)
# grad_monitor → GradientMonitorCallback (PR3 收敛), defender → NaNGuard (PR2 保留)
from .callbacks import (  # noqa: E402
    NaNGuard,
    TrainerContext,
    build_callbacks,
)
from .checkpoint import (  # noqa: E402
    save_checkpoint,
    load_checkpoint,
)
from .training_logs import EpochLogger  # noqa: E402


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_experiment_dir(base_dir: str = "./experiments", experiment_name: Optional[str] = None) -> Path:
    """Create experiment directory with standard naming convention

    Args:
        base_dir: Base experiments directory
        experiment_name: Custom experiment name. If None, generates fractal_vit_YYYYMMDD_HHMMSS

    Returns:
        Path to created experiment directory

    Example:
        experiments/fractal_vit_20260226_143000/
            ├── logs/
            │   └── config.json
            ├── checkpoints/
            │   └── best.pth
            └── training_history.json
    """
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"fractal_vit_{timestamp}"

    exp_dir = Path(base_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)

    return exp_dir


def configure_cuda():
    """Configure CUDA optimizations"""
    if not torch.cuda.is_available():
        return

    # MEM-OOM FIX (Suspect 2): 开启可扩展段分配器，缓解动态 Token 长度导致的显存碎片化
    # PyTorch CUDA 分配器在不断申请/释放不同大小的张量后会产生碎片，
    # expandable_segments:True 让分配器使用可扩展段，减少碎片化
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "expandable_segments:True" + ("," + alloc_conf if alloc_conf else "")
        )

    # TF32 for Ampere+ GPUs
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.allow_tf32 = True
        # B7.11 FIX: 仅在非确定性模式下启用 benchmark，避免与 set_seed 矛盾
        if not torch.backends.cudnn.deterministic:
            torch.backends.cudnn.benchmark = True


def create_model(args, device: torch.device) -> nn.Module:
    """Create FractalCurveViT model from CLI arguments

    Args:
        args: Parsed command-line arguments
        device: Device to create model on

    Returns:
        FractalCurveViT model
    """
    from vit_pytorch import FractalCurveViT

    # Parse boolean flags
    use_checkpoint = getattr(args, 'gradient_checkpoint', False)
    channels_last = getattr(args, 'channels_last', False)

    # Parse splitter type
    splitter_type = getattr(args, 'splitter_type', 'hilbert_optimal')

    # Parse quota learnable
    quota_learnable = getattr(args, 'quota_learnable', 'disable')
    if quota_learnable == 'enable':
        quota_learnable = True
    elif quota_learnable == 'disable':
        quota_learnable = False

    # Parse image_size
    image_size = args.image_size
    if image_size is not None:
        image_size_str = str(image_size).lower()
        if image_size_str == 'none':
            image_size = None  # Dynamic resolution
        else:
            try:
                image_size = int(image_size_str)
            except ValueError:
                raise ValueError(f"Invalid image_size: {image_size}. Must be an integer or 'none'.")

    # Build model kwargs from args
    model_kwargs = {
        'image_size': image_size,
        'num_classes': args.num_classes,
        'dim': args.dim,
        'num_layers': args.num_layers,
        'heads': args.heads,
        'mlp_dim': args.mlp_dim,
        'dim_head': args.dim_head if args.dim_head is not None else args.dim // args.heads,
        'channels': getattr(args, 'channels', 3),
        'min_patch_size': args.min_patch_size,

        # Dropout
        'tokenizer_dropout': getattr(args, 'tokenizer_dropout', 0.0),
        'transformer_dropout': getattr(args, 'transformer_dropout', 0.0),
        'emb_dropout': getattr(args, 'emb_dropout', 0.0),
        'drop_path_rate': getattr(args, 'drop_path', 0.0),

        # Pooling
        'pool': getattr(args, 'pool', 'weighted'),

        # Tokenizer config
        'target_ratio': getattr(args, 'target_ratio', 0.5),

        # Splitter config
        'splitter_type': splitter_type,
        'K_min_abs': getattr(args, 'K_min_abs', getattr(args, 'K_min', 8)),
        'splitter_token_ratio_min': getattr(args, 'splitter_token_ratio_min', 0.02),
        'splitter_token_ratio_max': getattr(args, 'splitter_token_ratio_max', 0.15),

        # Splitter temperature
        'splitter_temp_start': getattr(args, 'splitter_temp_start', 1.0),
        'splitter_temp_end': getattr(args, 'splitter_temp_end', 0.5),

        # Quota learning
        'quota_learnable': quota_learnable,
        'quota_entropy_weight': getattr(args, 'quota_entropy_weight', 0.01),

        # I167-1: Distance Decay Convolution
        'use_distance_decay_conv': not getattr(args, 'no_distance_decay_conv', False),
        # I167-4: SDS Regularization
        'use_sds_regularization': getattr(args, 'use_sds_regularization', False),
        'sds_lambda': getattr(args, 'sds_lambda', 0.1),

        # P6-1: depth_scale_range - 使用 sigmoid 参数化防止 CUDA 梯度爆炸
        'depth_scale_range': (0.5, 2.0),

        # FFN type
        'ffn_type': getattr(args, 'ffn_type', 'swiglu'),

        # Use checkpoint
        'use_checkpoint': use_checkpoint,
    }

    # Add optional parameters if provided
    if hasattr(args, 'use_pattern_encoder') and args.use_pattern_encoder:
        model_kwargs['use_pattern_encoder'] = True

    if hasattr(args, 'use_geometry_field') and args.use_geometry_field:
        model_kwargs['use_geometry_field'] = True

    print("Creating FractalCurveViT with args:")
    for k, v in model_kwargs.items():
        print(f"  {k}: {v}")

    model = FractalCurveViT(**model_kwargs)

    # Apply optimizations
    if getattr(args, 'compile', False):
        print("Compiling model with torch.compile...")
        # I164-1: 使用 mode='default' 替代 'reduce-overhead'
        # 'reduce-overhead' 启用 CUDA Graphs，与 gradient_checkpointing 不兼容
        # 'default' 禁用 CUDA Graphs，避免动态形状导致的 index out of bounds
        model = torch.compile(model, mode='default')

    if channels_last:
        print("Converting to channels_last memory format...")
        model = model.to(memory_format=torch.channels_last)

    # D4-AUDIT FIX: 使用 non_blocking=True 配合 DataLoader pin_memory
    return model.to(device, non_blocking=True)


# Global cache for dummy dataset labels to ensure train/val consistency
_dummy_dataset_cache: dict = {}


def create_dataloader(args, split: str = 'train') -> DataLoader:
    """Create dataloader for specified dataset

    Args:
        args: Parsed command-line arguments
        split: 'train' or 'val'

    Returns:
        DataLoader
    """
    dataset_name = args.dataset.lower() if hasattr(args, 'dataset') else 'dummy'

    if dataset_name == 'dummy' or hasattr(args, 'quick_test') and args.quick_test:
        global _dummy_dataset_cache
        # Use cached labels to ensure train/val consistency
        cache_key = f"dummy_{args.num_classes}_{getattr(args, 'num_train_samples', 10000)}_{getattr(args, 'num_val_samples', 1000)}"
        if cache_key not in _dummy_dataset_cache:
            _dummy_dataset_cache[cache_key] = {
                'train': torch.randint(0, args.num_classes, (getattr(args, 'num_train_samples', 10000),)).tolist(),
                'val': torch.randint(0, args.num_classes, (getattr(args, 'num_val_samples', 1000),)).tolist(),
            }
        return create_dummy_dataloader(args, split, _dummy_dataset_cache[cache_key])

    # Hugging Face dataset: use --use-hf-dataset flag to enable HF loading
    # When enabled, loads 'zh-plus/tiny-imagenet' from Hugging Face Hub
    use_hf = getattr(args, 'use_hf_dataset', False)

    # Get image size from args (auto-set by get_dataset_info) or handle dynamic resolution
    image_size = args.image_size
    if hasattr(image_size, 'lower') and str(image_size).lower() == 'none':
        image_size = 224  # Dynamic resolution fallback
    elif image_size is not None:
        image_size = int(image_size)

    # Hugging Face dataset path
    hf_path = None
    if use_hf:
        try:
            from .data import create_hf_dataset, get_hf_path
            hf_path = get_hf_path(dataset_name)
            if hf_path is None:
                print(f"Warning: {dataset_name} does not support HF loading, using local data")
                use_hf = False
        except ImportError:
            print("Warning: HF datasets not available, falling back to local data")
            use_hf = False

    if use_hf and hf_path:
        # HF dataset: loads from Hugging Face Hub using the mapped hf_path
        dataset = create_hf_dataset(
            name=hf_path,
            split=split,
            image_size=image_size,
            augment=split == 'train' and not getattr(args, 'no_augment', False),
            shuffle=(split == 'train'),  # Shuffle only for training
            seed=args.seed,
        )
    else:
        # Import dataset loaders
        try:
            from .data import create_dataset, get_transforms
        except ImportError:
            print("Warning: Dataset loading not available, using dummy data")
            return create_dummy_dataloader(args, split)

        transforms = get_transforms(
            dataset=dataset_name,
            split=split,
            image_size=image_size,
            augment=split == 'train' and not getattr(args, 'no_augment', False),
        )

        # Create dataset
        dataset = create_dataset(
            name=dataset_name,
            split=split,
            transform=transforms,
            root=getattr(args, 'data_root', './data'),
        )

    # Create dataloader
    batch_size = args.batch_size
    num_workers = getattr(args, 'num_workers', 4)

    # Streaming datasets (HF in streaming mode) don't support multi-processing
    # because they can't be randomly accessed by index
    if use_hf and hf_path:
        num_workers = 0

    # Shuffle strategy:
    # - HF streaming datasets: shuffle handled internally by hf_dataset.shuffle() in create_hf_dataset
    # - Local datasets: shuffle only for training split to ensure randomization
    should_shuffle = (split == 'train') and not use_hf

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )


def create_dummy_dataloader(args, split: str = 'train', label_cache: Optional[dict] = None) -> DataLoader:
    """Create a dummy dataloader for testing

    Args:
        args: Parsed command-line arguments
        split: 'train' or 'val'
        label_cache: Pre-generated labels to ensure train/val consistency

    Returns:
        DataLoader with dummy data
    """
    batch_size = args.batch_size

    # Use cached labels if provided (for train/val consistency)
    if label_cache is not None:
        num_samples = len(label_cache[split])
        labels = label_cache[split]
    else:
        num_samples = getattr(args, 'num_train_samples', 10000) if split == 'train' else getattr(args, 'num_val_samples', 1000)
        labels = None

    # Parse image_size
    image_size = args.image_size
    if image_size is not None:
        image_size_str = str(image_size).lower()
        if image_size_str == 'none':
            image_size = 64  # Default for dynamic resolution
        else:
            try:
                image_size = int(image_size_str)
            except ValueError:
                image_size = 64  # Fallback

    channels = getattr(args, 'channels', 3)
    num_classes = args.num_classes

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, labels):
            # Use provided labels or generate new ones
            self.labels = labels if labels is not None else torch.randint(0, num_classes, (num_samples,)).tolist()

        def __len__(self):
            return num_samples

        def __getitem__(self, idx):
            return (
                torch.randn(channels, image_size, image_size),
                self.labels[idx]
            )

    return DataLoader(
        DummyDataset(labels),
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=0,
        pin_memory=True,
    )


def verify_optimizer_coverage(model: nn.Module, optimizer) -> bool:
    """V3: 打印模型参数与优化器的覆盖情况

    用于检测是否存在未被优化器覆盖的参数（特别是 splitter 子模块）。

    Args:
        model: FractalCurveViT 模型
        optimizer: 优化器实例

    Returns:
        True if all parameters are covered, False otherwise
    """
    optimizer_params = {id(p) for group in optimizer.param_groups for p in group["params"]}

    print("\n" + "=" * 60)
    print("OPTIMIZER PARAMETER COVERAGE CHECK (V3)")
    print("=" * 60)

    all_covered = True
    for name, param in model.named_parameters():
        if param.requires_grad:
            covered = id(param) in optimizer_params
            status = "[OK]" if covered else "[MISSING]"
            print(f"{status} {name}: {param.shape}")
            if not covered:
                all_covered = False

    # 特别检查 splitter 子模块
    if hasattr(model, "splitter"):
        splitter = model.splitter
        print("\n--- Splitter Submodules ---")
        for subname, submodule in splitter.named_modules():
            if len(list(submodule.parameters())) > 0:
                sub_params = sum(p.numel() for p in submodule.parameters() if p.requires_grad)
                # 检查是否有任何参数未被覆盖
                sub_covered = any(id(p) in optimizer_params for p in submodule.parameters() if p.requires_grad)
                status = "[OK]" if sub_covered else "[MISSING]"
                print(f"  {status} {subname}: {sub_params} params")

    print("=" * 60)
    if not all_covered:
        print("WARNING: Some parameters are NOT in optimizer!")
    else:
        print("OK: All parameters are covered by optimizer.")
    return all_covered


def _update_tau_only(model, epoch, tau_start=1.0, tau_end=0.1, tau_epochs=20):
    """τ 线性退火: Phase 4 唯一保留的调度器。

    Args:
        model: FractalCurveViT 模型
        epoch: 当前 epoch
        tau_start: 初始温度 (default: 1.0)
        tau_end: 最终温度 (default: 0.1)
        tau_epochs: 退火 epoch 数 (default: 20)

    Returns:
        {'tau': current_tau}
    """
    progress = min(1.0, epoch / tau_epochs)
    tau = tau_start + (tau_end - tau_start) * progress
    if hasattr(model, 'splitter') and hasattr(model.splitter, 'set_temperature'):
        model.splitter.set_temperature(tau)
    return {'tau': tau}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    """Main training function

    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        args: Command-line arguments
        device: Device to train on

    Returns:
        Final metrics dictionary
    """
    # Create experiment directory with standard naming convention
    # Format: experiments/fractal_vit_YYYYMMDD_HHMMSS/
    if args.resume:
        # When resuming, use the existing experiment directory
        output_dir = Path(args.resume).parent.parent
    else:
        # Generate new experiment directory name
        output_dir = create_experiment_dir(
            base_dir=getattr(args, 'experiments_dir', './experiments'),
            experiment_name=getattr(args, 'experiment_name', None)
        )

    # Subdirectories
    logs_dir = output_dir / "logs"
    checkpoints_dir = output_dir / "checkpoints"

    # Initialize state
    state = TrainingState()
    # patience_counter is now a TrainingState dataclass field (default=0)

    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device=str(device))
        state = TrainingState.from_dict(checkpoint.get("training_state", {}))
        # Load model weights
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed from epoch {state.epoch}")

    # Create config from args
    config = create_config(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        base_lr=args.lr,
        weight_decay=args.weight_decay,
        device=str(device),
    )

    # Override config with args
    config.training.warmup_epochs = getattr(args, 'warmup_epochs', 5)
    config.training.min_lr = getattr(args, 'min_lr', 1e-6)
    config.training.mixup_alpha = getattr(args, 'mixup_alpha', 0.0)
    config.training.cutmix_alpha = getattr(args, 'cutmix_alpha', 0.0)
    config.training.label_smoothing = getattr(args, 'label_smoothing', 0.0)
    config.training.gradient_clip_norm = getattr(args, 'gradient_clip', 1.0)
    config.numerical.detect_anomaly = getattr(args, 'detect_anomaly', False)
    config.numerical.skip_on_nan_grad = getattr(args, 'skip_on_nan', True)
    config.training.log_interval = getattr(args, 'log_interval', 10)
    config.output_dir = str(output_dir)
    config.checkpoint.checkpoint_dir = str(checkpoints_dir)
    config.amp.enabled = getattr(args, 'use_amp', False)
    config.data.dataset = args.dataset

    # Save config to logs/config.json
    config.save(str(logs_dir / "config.json"))

    print(f"\n{'='*60}")
    print(f"Experiment: {output_dir.name}")
    print(f"Logs: {logs_dir}")
    print(f"Checkpoints: {checkpoints_dir}")
    print(f"{'='*60}\n")

    # B1: 拓扑冷启动隔离 - 三参数组分离 (精确子模块路径匹配)
    # Fix 47: 替换弱字符串匹配 "geo" in n 为精确模块路径，消除假阴性/假阳性
    # geometry 组: GeometryEncoder (path/rotation/area) + fusion.geo_norm + fusion._semantic_ratio
    # splitter 组: splitter 内除 geometry 以外的所有参数 (feature_proj, depth_embedding, roi_norm, conv1d, logit_scale)
    # backbone 组: transformer + mlp_head + 其他非 splitter 参数
    _geo_prefixes = ('splitter.geometry_encoder.', 'splitter.fusion.geo_norm', 'splitter.fusion._semantic_ratio')
    _splitter_prefix = 'splitter.'
    geometry_params = [p for n, p in model.named_parameters() if n.startswith(_geo_prefixes)]
    splitter_params = [p for n, p in model.named_parameters()
                       if n.startswith(_splitter_prefix) and not n.startswith(_geo_prefixes)]
    backbone_params = [p for n, p in model.named_parameters()
                       if not n.startswith(_splitter_prefix)]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config.training.base_lr, 'name': 'backbone'},
        {'params': splitter_params, 'lr': config.training.base_lr * 0.1, 'name': 'splitter'},
        {'params': geometry_params, 'lr': config.training.base_lr * 0.1, 'name': 'geometry'},
    ], weight_decay=config.training.weight_decay)

    # V3: Safety Check - 验证优化器覆盖所有模型参数
    verify_optimizer_coverage(model, optimizer)

    # Resume optimizer state if available
    if state.optimizer_state:
        optimizer.load_state_dict(state.optimizer_state)

    # A1: 计算 steps_per_epoch 用于 FunctionalWarmupCosineScheduler
    steps_per_epoch = len(train_loader)

    # A1: 使用 FunctionalWarmupCosineScheduler (step-based, 避免时间基准 bug)
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type="functional_warmup_cosine",
        total_epochs=config.training.num_epochs,
        base_lr=config.training.base_lr,
        min_lr=config.training.min_lr,
        warmup_epochs=config.training.warmup_epochs,
        total_steps=config.training.num_epochs * steps_per_epoch,
        steps_per_epoch=steps_per_epoch,
    )

    # Resume scheduler state if available
    if state.scheduler_state:
        scheduler.load_state_dict(state.scheduler_state)

    # Create GradScaler (保守初始化，防止 warmup 期 GradScaler collapse)
    # init_scale=2048: 从 65536 降至 2048，崩溃阶梯从 16 步降到 5 步
    # growth_interval=500: 更长的稳定观察期，防止 scale 过快反弹
    scaler = GradScaler(
        init_scale=2048.0,
        growth_interval=500,
        backoff_factor=0.5,
    ) if config.amp.enabled else None

    # Resume scaler state if available
    if scaler and state.scaler_state:
        scaler.load_state_dict(state.scaler_state)

    # Create Mixup/Cutmix
    mixup_cutmix = None
    if config.training.mixup_alpha > 0 or config.training.cutmix_alpha > 0:
        mixup_cutmix = MixupCutmixLoss(
            num_classes=model.num_classes if hasattr(model, 'num_classes') else args.num_classes,
            label_smoothing=config.training.label_smoothing,
            mixup_alpha=config.training.mixup_alpha,
            cutmix_alpha=config.training.cutmix_alpha,
            mixup_prob=getattr(args, 'mixup_prob', 0.5),
        )

    # Create monitors - I-OOM FIX: 移到循环外，只创建一次
    # PR5c: grad_monitor / loss_monitor → callbacks (PR3/PR4); defender → NaNGuard (PR2)
    defender = NaNGuard(
        model=model,
        detect_anomaly=config.numerical.detect_anomaly,
        skip_on_nan=config.numerical.skip_on_nan_grad,
    )
    defender.register_discovery_hooks()

    # Create logger
    logger = EpochLogger(output_dir=str(output_dir))

    # Move model to device
    # D4-AUDIT FIX: 使用 non_blocking=True 配合 DataLoader pin_memory
    model = model.to(device, non_blocking=True)

    # P0: Pre-flight Check（零时刻诊断）
    print("\n" + "=" * 60)
    print("PRE-FLIGHT CHECK (零时刻诊断)")
    print("=" * 60)

    # Gradient Check: backbone_grad_norm / splitter_grad_norm 初始比值
    model.eval()
    try:
        from vit_pytorch.models.fractal_vit import TrainingStats

        # B7.10 FIX: 从 dataloader 获取实际图像尺寸，而非硬编码 64×64
        _sample_batch = next(iter(train_loader))
        _actual_h, _actual_w = _sample_batch[0].shape[2], _sample_batch[0].shape[3]
        test_input = torch.randn(2, 3, _actual_h, _actual_w).to(device)
        test_target = torch.randint(0, model.num_classes if hasattr(model, 'num_classes') else 10, (2,)).to(device)
        test_output = model(test_input)
        if isinstance(test_output, TrainingStats):
            loss = F.cross_entropy(test_output.logits, test_target)
        elif isinstance(test_output, dict) and 'logits' in test_output:
            loss = F.cross_entropy(test_output['logits'], test_target)
        else:
            loss = F.cross_entropy(test_output, test_target)
        loss.backward()

        backbone_grad_norm = 0.0
        splitter_grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                if 'splitter' in name:
                    splitter_grad_norm += grad_norm ** 2
                else:
                    backbone_grad_norm += grad_norm ** 2

        backbone_grad_norm = backbone_grad_norm ** 0.5
        splitter_grad_norm = splitter_grad_norm ** 0.5
        grad_ratio = splitter_grad_norm / (backbone_grad_norm + 1e-8)

        print(f"[Pre-flight] Backbone grad norm: {backbone_grad_norm:.4f}")
        print(f"[Pre-flight] Splitter grad norm: {splitter_grad_norm:.4f}")
        print(f"[Pre-flight] Grad ratio (splitter/backbone): {grad_ratio:.4f}")
        if grad_ratio > 1.5:
            print(f"  WARNING: 梯度比异常({grad_ratio:.2f} > 1.5)")

        model.zero_grad()
    except Exception as e:
        print(f"[Pre-flight] Gradient check skipped: {e}")

    model.train()
    print("=" * 60 + "\n")

    # Get eval interval
    eval_interval = getattr(args, 'eval_interval', 1)
    patience = getattr(args, 'patience', 999)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {config.training.num_epochs} epochs")
    print(f"Dataset: {args.dataset}")
    print("Model: FractalCurveViT")
    print(f"{'='*60}\n")

    for epoch in range(state.epoch, config.training.num_epochs):
        state.epoch = epoch

        # Phase 4: τ 线性退火 — 唯一保留的调度器
        # B7.9 FIX: 传递 CLI 参数，避免硬编码默认值覆盖用户意图
        warmup_params = _update_tau_only(
            model, epoch,
            tau_end=getattr(args, 'splitter_temp_end', 0.5),
            tau_epochs=getattr(args, 'tau_epochs', 20),
        )

        # PR5c: 一次性构建 TrainerContext (callbacks + nan_guard 注入)
        # 在 epoch loop 外构建可避免每 epoch 重建 callback 列表 (PR4 收敛)
        if epoch == state.epoch:
            # 仅首 epoch 构建, 后续 epoch 复用同一 ctx
            callbacks = build_callbacks(config)
            ctx = TrainerContext(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                state=state,
                nan_guard=defender,
                callbacks=callbacks,
                epoch=epoch,
                global_step=state.global_step,
                device=device,
                config=config,
                mixup=mixup_cutmix,
                amp=config.amp.enabled,
                grad_clip=float(config.training.gradient_clip_norm),
            )
        else:
            ctx.epoch = epoch  # 同步 ctx.epoch
            ctx.global_step = state.global_step

        # PR5c skeleton: 4-hook, 6-param 签名
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            state=state,
            ctx=ctx,
        )

        # PR5c: grad_monitor / loss_monitor reset 走 callback 内部, 骨架不再管
        defender.clear()  # I-SLOW FIX: 清理 NaNGuard._pending_stats_tensors 中的 GPU tensor 引用

        # Evaluate
        eval_metrics = None
        if (epoch + 1) % eval_interval == 0:
            # Confusion matrix 保存目录
            cm_dir = str(output_dir / "confusion_matrices" / f"epoch_{epoch+1:04d}")

            eval_result = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                config=config,
                num_classes=model.num_classes,
                save_confusion_matrix_dir=cm_dir,
            )
            eval_metrics = eval_result.to_dict()

        # Phase 4: 简化日志 — 仅记录 τ
        extra_logs = {
            "splitter/temperature": warmup_params.get("tau", 1.0),
        }

        # Log epoch
        logger.log(
            epoch=epoch + 1,
            train_metrics=train_metrics.to_dict(),
            eval_metrics=eval_metrics,
            extra=extra_logs,
        )

        # Update state
        state.optimizer_state = optimizer.state_dict()
        if scaler:
            state.scaler_state = scaler.state_dict()
        state.scheduler_state = scheduler.state_dict()

        # Check if best (handle first epoch: initialize best_metric to -inf for "max" mode)
        is_best = False
        if eval_metrics and config.checkpoint.save_best:
            # Initialize best_metric if first epoch (0.0 is ambiguous for accuracy)
            if state.best_metric == 0.0 and config.checkpoint.monitor_mode == "max":
                state.best_metric = float('-inf')  # Initialize to negative infinity

            metric_value = eval_metrics.get(config.checkpoint.monitor_metric, 0.0)
            if config.checkpoint.monitor_mode == "max":
                is_best = metric_value > state.best_metric
            else:
                is_best = metric_value < state.best_metric

            if is_best:
                state.best_metric = metric_value
                print(f"New best metric: {metric_value:.4f}")

        # ==============================================================================
        # FIX 46: Max/Min感知的高鲁棒性 Early Stopping 检查体系
        # ==============================================================================
        if eval_metrics and patience < 999:
            monitor_metric = config.checkpoint.monitor_metric

            if monitor_metric not in eval_metrics:
                print(
                    f"[WARNING] Early stopping 无法在 eval_metrics 中找到监控指标 '{monitor_metric}'。"
                    f"当前可用指标: {list(eval_metrics.keys())}。跳过本轮检查。"
                )
            else:
                metric_value = eval_metrics[monitor_metric]
                monitor_mode = config.checkpoint.monitor_mode

                if monitor_mode == "max":
                    improved = metric_value > state.best_metric
                elif monitor_mode == "min":
                    improved = metric_value < state.best_metric
                else:
                    raise ValueError(f"未知的 monitor_mode 级别: {monitor_mode}, 必须为 'max' 或 'min'")

                if improved:
                    state.patience_counter = 0
                else:
                    state.patience_counter = state.patience_counter + 1
                    if state.patience_counter >= patience:
                        print(
                            f"[EARLY STOPPING] 指标 '{monitor_metric}' 已连续 {state.patience_counter} 个 Epoch 未改善。"
                            f"当前最佳值: {state.best_metric:.6f}, 当前触发值: {metric_value:.6f}。训练提前终止。"
                        )
                        break

        # Save checkpoint: separate concerns
        # 1. Save epoch checkpoint every N epochs (controlled by save_interval)
        # 2. Save best.pth only when is_best=True (handled in save_checkpoint)
        # 3. Save last.pth every time (handled in save_checkpoint)
        is_last_epoch = (epoch + 1) >= config.training.num_epochs
        save_interval = getattr(args, 'save_interval', 10)
        should_save_epoch = is_last_epoch or (epoch + 1) % save_interval == 0

        if should_save_epoch:
            # Save epoch checkpoint + last.pth (always)
            # is_best determines if best.pth is also saved
            save_checkpoint(
                checkpoint_dir=config.checkpoint.checkpoint_dir,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics={"train": train_metrics.to_dict(), "eval": eval_metrics or {}},
                scheduler_state=state.scheduler_state,
                scaler_state=state.scaler_state,
                training_state=state.to_dict(),
                is_best=is_best,  # Only True when truly a new record
                save_epoch_checkpoint=True,
            )
        elif is_best:
            # Not on save interval, but new best → only update best.pth and last.pth
            save_checkpoint(
                checkpoint_dir=config.checkpoint.checkpoint_dir,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics={"train": train_metrics.to_dict(), "eval": eval_metrics or {}},
                scheduler_state=state.scheduler_state,
                scaler_state=state.scaler_state,
                training_state=state.to_dict(),
                is_best=True,  # Only True when truly a new record
                save_epoch_checkpoint=False,  # Don't save epoch checkpoint, just best.pth + last.pth
            )

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best metric: {state.best_metric:.4f}")
    print(f"{'='*60}\n")

    # Save training history in standard format
    history = logger.get_history()
    _save_training_history(history, output_dir)

    return {
        "best_metric": state.best_metric,
        "history": history,
    }


def _tensor_to_serializable(obj: Any) -> Any:
    """Recursively convert Tensor objects to JSON-serializable Python types.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, torch.Tensor):
        if obj.dim() == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    elif isinstance(obj, dict):
        return {k: _tensor_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_tensor_to_serializable(item) for item in obj]
    elif isinstance(obj, float):
        return obj
    elif isinstance(obj, int):
        return obj
    elif obj is None:
        return None
    else:
        try:
            return float(obj)
        except (TypeError, ValueError):
            return str(obj)


def _save_training_history(history: list, output_dir: Path) -> None:
    """Save training history in standard experiments format

    Args:
        history: List of epoch stats from EpochLogger
        output_dir: Experiment directory
    """
    # Convert to standard format matching existing experiments
    # Format: [{"epoch": 1, "train_loss": ..., "train_acc": ..., "val_loss": ..., "val_acc": ..., "lr": ..., "time": ...}]
    training_history = []

    for stats in history:
        entry = {"epoch": stats.get("epoch", 0)}

        # Extract train metrics
        if "train" in stats:
            train = stats["train"]
            entry["train_loss"] = train.get("loss", 0.0)
            entry["train_acc"] = train.get("accuracy", 0.0) * 100  # Convert to percentage
            entry["train_grad_norm"] = train.get("grad_norm", 0.0)

            # Extract loss components
            if "loss_components" in train:
                for k, v in train["loss_components"].items():
                    entry[f"train_{k}"] = v

            # Extract splitter logits stats
            if "splitter_logits_stats" in train:
                for k, v in train["splitter_logits_stats"].items():
                    entry[f"splitter_logits_{k}"] = v

            # Extract manifold bias stats
            if "manifold_bias_stats" in train:
                for k, v in train["manifold_bias_stats"].items():
                    entry[f"manifold_bias_{k}"] = v

            # Extract poincare dist stats
            if "poincare_dist_stats" in train:
                for k, v in train["poincare_dist_stats"].items():
                    entry[f"poincare_dist_{k}"] = v

            # Extract grad ratio
            entry["backbone_vs_splitter_grad_ratio"] = train.get("backbone_vs_splitter_grad_ratio", 0.0)
            entry["backbone_grad_norm"] = train.get("backbone_grad_norm", 0.0)
            entry["splitter_grad_norm"] = train.get("splitter_grad_norm", 0.0)

            # Extract active ratio
            entry["train_active_ratio"] = train.get("active_ratio", 0.0)

        # Extract eval metrics
        if "eval" in stats:
            eval_stats = stats["eval"]
            entry["val_loss"] = eval_stats.get("loss", 0.0)
            entry["val_acc"] = eval_stats.get("accuracy", 0.0) * 100  # Convert to percentage
            entry["val_ece"] = eval_stats.get("ece", None)  # Expected Calibration Error

        # Extract learning rate from extra if available
        if "extra" in stats and "lr" in stats["extra"]:
            entry["lr"] = stats["extra"]["lr"]

        # Extract time if available
        if "extra" in stats and "time" in stats["extra"]:
            entry["time"] = stats["extra"]["time"]

        training_history.append(entry)

    # Save to training_history.json
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        # Convert Tensor objects to JSON-serializable types
        serializable_history = _tensor_to_serializable(training_history)
        json.dump(serializable_history, f, indent=2)

    print(f"Training history saved to: {history_path}")


def add_args(parser: argparse.ArgumentParser):
    """Add all CLI arguments to the parser

    Args:
        parser: ArgumentParser instance
    """
    # ==================== Dataset ====================
    parser.add_argument('--dataset', type=str, default='tiny-imagenet',
                        help='Dataset name: cifar10, cifar100, tiny-imagenet, cub200, imagenet, or HF path like zh-plus/tiny-imagenet')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--data-root', type=str, default='./data',
                        help='Root directory for datasets')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--use-hf-dataset', action='store_true',
                        help='Use Hugging Face zh-plus/tiny-imagenet instead of local dataset')

    # ==================== Model Architecture ====================
    # I-OPT: RTX 4070 8GB + BS=192 最优配置 (2026-04-05)
    # 计算依据: dim=512, L=12, heads=8, mlp_dim=2048
    # 显存占用: ~2GB (24%)，留有充足余量供200 epochs训练
    # 参数量: 50.3M (适合 Tiny-ImageNet 200类)
    parser.add_argument('--dim', type=int, default=512,
                        help='Model embedding dimension (default: 512 for RTX 4070 8GB)')
    parser.add_argument('--num-layers', type=int, default=12,
                        help='Number of transformer layers (default: 12)')
    parser.add_argument('--heads', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--mlp-dim', type=int, default=2048,
                        help='MLP hidden dimension (default: 2048 = 4*dim)')
    parser.add_argument('--dim-head', type=int, default=None,
                        help='Per-head dimension (default: dim/heads)')
    parser.add_argument('--min-patch-size', type=int, default=4,
                        help='Minimum patch size')
    parser.add_argument('--channels', type=int, default=3,
                        help='Number of input channels')

    # ==================== Pooling & FFN ====================
    parser.add_argument('--pool', type=str, default='weighted',
                        help='Pooling type: cls, mean, weighted')
    parser.add_argument('--ffn-type', type=str, default='swiglu',
                        help='FFN type: swiglu, geglu')

    # ==================== Dropout ====================
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='General dropout')
    parser.add_argument('--tokenizer-dropout', type=float, default=0.0,
                        help='Tokenizer dropout (should be 0 for deterministic)')
    parser.add_argument('--transformer-dropout', type=float, default=0.0,
                        help='Transformer dropout')
    parser.add_argument('--emb-dropout', type=float, default=0.0,
                        help='Embedding dropout')
    parser.add_argument('--drop-path', type=float, default=0.0,
                        help='Stochastic depth drop path rate')

    # ==================== Tokenizer ====================
    parser.add_argument('--token-coverage-min', type=float, default=0.01,
                        help='Minimum token coverage ratio')
    parser.add_argument('--token-coverage-max', type=float, default=None,
                        help='Maximum token coverage ratio')
    parser.add_argument('--target-ratio', type=float, default=0.25,
                        help='Target token ratio (default: 0.25, 增加token数量缓解信息瓶颈)')

    # ==================== Splitter ====================
    parser.add_argument('--splitter-type', type=str, default='hilbert_optimal',
                        help='Splitter type: hilbert_optimal (其他类型已废弃)')
    parser.add_argument('--K-min-abs', type=int, default=8,
                        help='Absolute minimum token count')
    parser.add_argument('--K-max', type=int, default=None,
                        help='Maximum token count')
    parser.add_argument('--K-min', type=int, default=None,
                        help='Alias for K-min-abs')
    parser.add_argument('--splitter-token-ratio-min', type=float, default=0.02,
                        help='Minimum token ratio per depth')
    parser.add_argument('--splitter-token-ratio-max', type=float, default=0.15,
                        help='Maximum token ratio per depth')

    # ==================== Splitter Temperature ====================
    parser.add_argument('--splitter-temp-start', type=float, default=1.0,
                        help='Starting temperature for splitter')
    parser.add_argument('--splitter-temp-end', type=float, default=0.5,
                        help='Ending temperature for splitter')
    parser.add_argument('--splitter-temp-warmup', type=int, default=0,
                        help='Number of warmup epochs for temperature')

    # ==================== Quota Learning ====================
    parser.add_argument('--quota-learnable', type=str, default='disable',
                        help='Enable learnable quota: enable, disable')
    parser.add_argument('--quota-entropy-weight', type=float, default=0.01,
                        help='Weight for quota entropy loss')
    parser.add_argument('--quota-align-weight', type=float, default=0.0,
                        help='Weight for quota alignment loss')
    parser.add_argument('--quota-align-mode', type=str, default='fixed',
                        help='Quota alignment mode: fixed, curriculum')

    # ==================== I167-1: Distance Decay Convolution ====================
    parser.add_argument('--use-distance-decay-conv', action='store_true',
                        help='Enable distance decay convolution (I167-1)')
    parser.add_argument('--no-distance-decay-conv', action='store_true',
                        help='Disable distance decay convolution, use standard Conv1D')

    # ==================== I167-4: SDS Regularization ====================
    parser.add_argument('--use-sds-regularization', action='store_true',
                        help='Enable SDS regularization (I167-4)')
    parser.add_argument('--sds-lambda', type=float, default=0.1,
                        help='SDS regularization strength (I167-4)')

    # ==================== Encoders ====================
    parser.add_argument('--use-area-encoding', action='store_true',
                        help='Enable area encoding')
    parser.add_argument('--fourier-levels', type=int, default=4,
                        help='Number of Fourier levels for area encoding')
    parser.add_argument('--use-pattern-encoder', action='store_true',
                        help='Enable pattern encoder')
    parser.add_argument('--use-geometry-field', action='store_true',
                        help='Enable geometry field')

    # ==================== Training ====================
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=8e-5,
                        help='Base learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.1,
                        help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=15,
                        help='Number of warmup epochs')
    parser.add_argument('--warmup-start-lr', type=float, default=1e-7,
                        help='Starting learning rate for warmup')
    parser.add_argument('--gradient-clip', type=float, default=5.0,
                        help='Gradient clipping norm')
    parser.add_argument('--accum-steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Label smoothing')

    # ==================== Mixup/Cutmix ====================
    parser.add_argument('--mixup-alpha', type=float, default=0.0,
                        help='Mixup alpha')
    parser.add_argument('--cutmix-alpha', type=float, default=0.0,
                        help='Cutmix alpha')
    parser.add_argument('--mixup-prob', type=float, default=0.5,
                        help='Probability of applying mixup/cutmix')

    # ==================== Soft Entropy ====================
    parser.add_argument('--include-soft-entropy', action='store_true',
                        help='Include soft entropy loss')
    parser.add_argument('--soft-entropy-mode', type=str, default='maximize',
                        help='Soft entropy mode: maximize, minimize')
    parser.add_argument('--soft-entropy-weight', type=float, default=0.0,
                        help='Soft entropy loss weight')

    # ==================== Elastic Budget ====================
    parser.add_argument('--include-elastic-budget', action='store_true',
                        help='Include elastic budget loss')
    parser.add_argument('--elastic-coverage-min', type=float, default=0.03,
                        help='Minimum elastic coverage')
    parser.add_argument('--elastic-coverage-max', type=float, default=0.25,
                        help='Maximum elastic coverage')
    parser.add_argument('--elastic-lambda-over', type=float, default=0.1,
                        help='Elastic budget over-penalty weight')
    parser.add_argument('--elastic-lambda-under', type=float, default=0.01,
                        help='Elastic budget under-penalty weight')
    parser.add_argument('--elastic-lambda-collapse', type=float, default=0.0,
                        help='Collapse penalty weight')

    # ==================== Auxiliary Loss ====================
    parser.add_argument('--aux-loss-warmup-epochs', type=int, default=0,
                        help='Warmup epochs for auxiliary losses')

    # ==================== Optimization ====================
    parser.add_argument('--use-amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--gradient-checkpoint', action='store_true',
                        help='Use gradient checkpointing')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile')
    parser.add_argument('--channels-last', action='store_true',
                        help='Use channels_last memory format')

    # ==================== Training Control ====================
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='Evaluation interval (epochs)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Training log interval (steps)')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Checkpoint save interval (epochs)')
    parser.add_argument('--patience', type=int, default=999,
                        help='Early stopping patience')
    parser.add_argument('--num-train-samples', type=int, default=10000,
                        help='Number of training samples (dummy mode)')
    parser.add_argument('--num-val-samples', type=int, default=1000,
                        help='Number of validation samples (dummy mode)')

    # ==================== System ====================
    parser.add_argument('--experiments-dir', type=str, default='./experiments',
                        help='Base experiments directory')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Custom experiment name (default: fractal_vit_YYYYMMDD_HHMMSS)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (deprecated, use --experiments-dir)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Checkpoint directory (auto-managed in experiments)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    # ==================== Quick Test ====================
    parser.add_argument('--quick-test', action='store_true',
                        help='Run quick test with dummy data')

    # ==================== v1.3 STANDARD: opt-in features ====================
    v13 = parser.add_argument_group("v1.3 STANDARD (opt-in)")
    v13.add_argument('--enable-shadow-monitor', action='store_true',
                     help='Enable ShadowMonitor hooks (WBA entropy + RoPE MI)')
    v13.add_argument('--enable-r12-aux', action='store_true',
                     help='Enable R12 hierarchical aux loss on top of CE')
    v13.add_argument('--enable-eahbp-3gate', action='store_true',
                     help='Enable EAHBP G1/G2/G3 gate signals (rollback-aware)')
    v13.add_argument('--enable-paced-window', action='store_true',
                     help='Enable PacedWindow state machine for staged weight rollbacks')
    v13.add_argument('--shadow-monitor-interval', type=int, default=50,
                     help='Shadow Monitor step interval')
    v13.add_argument('--r12-lambda-tree', type=float, default=0.10,
                     help='R12 L_tree coefficient (overrides default 0.10)')
    v13.add_argument('--r12-lambda-skew', type=float, default=0.10,
                     help='R12 L_skew coefficient (overrides default 0.10)')
    v13.add_argument('--paced-window-fatal-streak', type=int, default=3,
                     help='PacedWindow D_struct fatal streak (consecutive steps)')


def build_parser() -> argparse.ArgumentParser:
    """Construct the full ArgumentParser (extracted from main for testability)."""
    parser = argparse.ArgumentParser(
        description="FractalCurveViT Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_args(parser)
    return parser


def main():
    """Main entry point"""
    parser = build_parser()
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    configure_cuda()

    # TorchDynamo 优化：降低缓存限制强制回收，减少显存泄漏
    # P3-Fix: cache_size_limit=16（强制回收旧缓存，防止无限增长）
    # 注意：capture_scalar_outputs=True 已禁用，因其会导致 inductor C++ 代码生成 bug
    # （zuf0/zuf1 等符号变量未正确声明，导致 C++ 编译失败）
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 16
    # torch._dynamo.config.capture_scalar_outputs = True  # 已禁用：会导致 C++ 编译崩溃
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Auto-detect dataset info (image_size, num_classes) from dataset
    # If --use-hf-dataset is set, we use the HF path for metadata
    try:
        from .data import get_dataset_info, get_hf_path
        # Determine which dataset name to query for metadata
        if getattr(args, 'use_hf_dataset', False):
            hf_path = get_hf_path(args.dataset)
            if hf_path:
                dataset_for_info = hf_path
            else:
                print(f"Warning: {args.dataset} does not support HF loading, using local data")
                dataset_for_info = args.dataset
        else:
            dataset_for_info = args.dataset
        dataset_info = get_dataset_info(dataset_for_info)
        args.image_size = dataset_info['image_size']
        args.num_classes = dataset_info['num_classes']
        print(f"\nDataset: {dataset_for_info}")
        print(f"Image size: {args.image_size}, Num classes: {args.num_classes}")
    except Exception as e:
        print(f"Warning: Could not auto-detect dataset info: {e}")
        print("Using defaults: image_size=64, num_classes=200")
        args.image_size = 64
        args.num_classes = 200

    # Auto-adjust dim to ensure dim_per_subspace is even for fractal RoPE
    # Formula: dim_per_subspace = (heads * dim_head) // max_level
    # where dim_head = dim // heads, max_level = ceil(log2(image_size / min_patch_size))
    # For fractal RoPE to work, dim_per_subspace must be even
    if args.dim is not None and args.image_size is not None:
        import math
        min_patch = getattr(args, 'min_patch_size', 4)
        heads = getattr(args, 'heads', 8)
        computed_max_level = math.ceil(math.log2(args.image_size // min_patch))
        if computed_max_level > 0:
            dim_head = args.dim // heads
            inner_dim = heads * dim_head
            dim_per_subspace = inner_dim // computed_max_level
            if dim_per_subspace % 2 == 1:
                # dim_per_subspace is odd - adjust dim to make it even
                # Find the nearest lower dim that gives even dim_per_subspace
                for new_dim in range(args.dim, 63, -2):  # Step by 2 to keep even/odd consistent
                    new_dim_head = new_dim // heads
                    if new_dim_head <= 0:
                        continue
                    new_inner_dim = heads * new_dim_head
                    new_dim_per_subspace = new_inner_dim // computed_max_level
                    if new_dim_per_subspace % 2 == 0:
                        print(f"\n[AUTO-ADJUST] dim_per_subspace={dim_per_subspace} (odd) -> adjusting dim from {args.dim} to {new_dim} for fractal RoPE compatibility")
                        args.dim = new_dim
                        args.mlp_dim = new_dim * 4  # Maintain mlp_ratio=4
                        break

    # Quick test mode: override with smaller model parameters
    if getattr(args, 'quick_test', False):
        args.dim = 128
        args.num_layers = 2
        args.heads = 4
        args.mlp_dim = 512
        args.num_train_samples = 100
        args.num_val_samples = 500  # Quick-test用小样本验证评估管线，真实训练用完整测试集
        # P1-2 FIX: BS=4 is too small for ViT, use BS=16 (gradient accumulation will be added separately)
        args.batch_size = 16
        args.epochs = 30  # Warmup test: 30 epochs covers Stage 1 (0-9) + Stage 2 (9-30) + Stage 3 (30+)
        args.compile = False  # P6-1: Disable compile on Windows (torch.compile has unicode path issue)
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True  # Fallback to eager for function-level compile decorators
        print(f"\n[Quick Test Mode] Using small model: dim={args.dim}, layers={args.num_layers}, epochs={args.epochs}, batch_size={args.batch_size}")

    # Create model
    print("\nCreating model...")
    model = create_model(args, device)

    # Count parameters
    if hasattr(model, 'num_parameters'):
        num_params = model.num_parameters()
    else:
        num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = create_dataloader(args, split='train')
    val_loader = create_dataloader(args, split='val')
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Train
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        device=device,
    )


if __name__ == "__main__":
    main()
