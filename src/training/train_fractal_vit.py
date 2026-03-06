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
import time
import random
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Add src directory to path for vit_pytorch imports
def _setup_path():
    src_dir = Path(__file__).parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
_setup_path()

from .config import Config, create_config
from .trainer import (
    TrainingState,
    train_one_epoch,
    evaluate,
    MixupCutmixLoss,
)
from .scheduler import create_scheduler
from .monitor import (
    GradientMonitor,
    LossMonitor,
    NumericalDefender,
)
from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
)
from .logging import EpochLogger


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

    # TF32 for Ampere+ GPUs
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.allow_tf32 = True
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
    compile_model = getattr(args, 'compile', False)
    channels_last = getattr(args, 'channels_last', False)

    # Parse splitter type
    splitter_type = getattr(args, 'splitter_type', 'gumbel_topk')

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

        # P6-1: depth_scale_range - 使用 sigmoid 参数化防止 CUDA 梯度爆炸
        'depth_scale_range': (0.5, 2.0),

        # FFN type
        'ffn_type': getattr(args, 'ffn_type', 'swiglu_level'),

        # Use checkpoint
        'use_checkpoint': use_checkpoint,
    }

    # Add optional parameters if provided
    if hasattr(args, 'use_area_encoding') and args.use_area_encoding:
        model_kwargs['use_area_encoding'] = True
        model_kwargs['fourier_levels'] = getattr(args, 'fourier_levels', 4)

    if hasattr(args, 'use_pattern_encoder') and args.use_pattern_encoder:
        model_kwargs['use_pattern_encoder'] = True

    if hasattr(args, 'use_geometry_field') and args.use_geometry_field:
        model_kwargs['use_geometry_field'] = True

    print(f"Creating FractalCurveViT with args:")
    for k, v in model_kwargs.items():
        print(f"  {k}: {v}")

    model = FractalCurveViT(**model_kwargs)

    # Apply optimizations
    if compile_model:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')

    if channels_last:
        print("Converting to channels_last memory format...")
        model = model.to(memory_format=torch.channels_last)

    return model.to(device)


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

    # Import dataset loaders
    try:
        from .data import create_dataset, get_transforms
    except ImportError:
        print("Warning: Dataset loading not available, using dummy data")
        return create_dummy_dataloader(args, split)

    # Get transforms
    image_size = args.image_size
    if image_size is not None and str(image_size).lower() == 'none':
        image_size = 224  # Default for dynamic resolution
    elif image_size is not None:
        image_size = int(image_size)  # Convert string to int

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

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
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
    state.patience_counter = 0  # Initialize early stopping counter

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

    # Save config to logs/config.json
    config.save(str(logs_dir / "config.json"))

    print(f"\n{'='*60}")
    print(f"Experiment: {output_dir.name}")
    print(f"Logs: {logs_dir}")
    print(f"Checkpoints: {checkpoints_dir}")
    print(f"{'='*60}\n")

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.base_lr,
        weight_decay=config.training.weight_decay,
    )

    # Resume optimizer state if available
    if state.optimizer_state:
        optimizer.load_state_dict(state.optimizer_state)

    # Create scheduler
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type="warmup_cosine",
        total_epochs=config.training.num_epochs,
        base_lr=config.training.base_lr,
        min_lr=config.training.min_lr,
        warmup_epochs=config.training.warmup_epochs,
        warmup_start_lr=getattr(args, 'warmup_start_lr', config.training.base_lr * 0.1),
    )

    # Resume scheduler state if available
    if state.scheduler_state:
        scheduler.load_state_dict(state.scheduler_state)

    # Create GradScaler
    scaler = GradScaler() if config.amp.enabled else None

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

    # Create monitors
    grad_monitor = GradientMonitor(
        model=model,
        record_layer_norms=config.numerical.record_layer_grad_norms,
    )
    loss_monitor = LossMonitor()
    defender = NumericalDefender(
        model=model,
        detect_anomaly=config.numerical.detect_anomaly,
        skip_on_nan=config.numerical.skip_on_nan_grad,
    )

    # Create logger
    logger = EpochLogger(output_dir=str(output_dir))

    # Move model to device
    model = model.to(device)

    # Get eval interval
    eval_interval = getattr(args, 'eval_interval', 1)
    patience = getattr(args, 'patience', 999)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {config.training.num_epochs} epochs")
    print(f"Dataset: {args.dataset}")
    print(f"Model: FractalCurveViT")
    print(f"{'='*60}\n")

    for epoch in range(state.epoch, config.training.num_epochs):
        state.epoch = epoch

        # Train one epoch
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            state=state,
            config=config,
            device=device,
            scheduler=scheduler,
            mixup_cutmix=mixup_cutmix,
            debug_dir=str(output_dir / "debug"),
        )

        # Reset monitors
        grad_monitor.reset()
        loss_monitor.reset()

        # Evaluate
        eval_metrics = None
        if (epoch + 1) % eval_interval == 0:
            eval_result = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                config=config,
            )
            eval_metrics = eval_result.to_dict()

        # Log epoch
        logger.log(
            epoch=epoch + 1,
            train_metrics=train_metrics.to_dict(),
            eval_metrics=eval_metrics,
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
                state.best_metric = -1.0  # Initialize to negative for accuracy

            metric_value = eval_metrics.get(config.checkpoint.monitor_metric, 0.0)
            if config.checkpoint.monitor_mode == "max":
                is_best = metric_value > state.best_metric
            else:
                is_best = metric_value < state.best_metric

            if is_best:
                state.best_metric = metric_value
                print(f"New best metric: {metric_value:.4f}")

        # Early stopping check
        if eval_metrics and patience < 999:
            metric_value = eval_metrics.get(config.checkpoint.monitor_metric, 0.0)
            if state.best_metric > 0 or config.checkpoint.monitor_mode == "min":
                if metric_value >= state.best_metric:
                    state.patience_counter = 0
                else:
                    state.patience_counter = getattr(state, 'patience_counter', 0) + 1
                    if state.patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

        # Save checkpoint (always save on last epoch)
        is_last_epoch = (epoch + 1) >= config.training.num_epochs
        save_interval = getattr(args, 'save_interval', 10)
        if is_last_epoch or is_best or (epoch + 1) % save_interval == 0:
            checkpoint_path = save_checkpoint(
                checkpoint_dir=config.checkpoint.checkpoint_dir,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics={"train": train_metrics.to_dict(), "eval": eval_metrics or {}},
                scheduler_state=state.scheduler_state,
                scaler_state=state.scaler_state,
                training_state=state.to_dict(),
                is_best=is_best,
            )

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best metric: {state.best_metric:.4f}")
    print(f"{'='*60}\n")

    # Save training history in standard format
    history = logger.get_history()
    _save_training_history(history, output_dir)

    return {
        "best_metric": state.best_metric,
        "history": history,
    }


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

        # Extract eval metrics
        if "eval" in stats:
            eval_stats = stats["eval"]
            entry["val_loss"] = eval_stats.get("loss", 0.0)
            entry["val_acc"] = eval_stats.get("accuracy", 0.0) * 100  # Convert to percentage

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
        json.dump(training_history, f, indent=2)

    print(f"Training history saved to: {history_path}")


def add_args(parser: argparse.ArgumentParser):
    """Add all CLI arguments to the parser

    Args:
        parser: ArgumentParser instance
    """
    # ==================== Dataset ====================
    parser.add_argument('--dataset', type=str, default='tiny-imagenet',
                        help='Dataset name: cifar10, cifar100, tiny-imagenet, cub200, imagenet')
    parser.add_argument('--image-size', type=str, default='64',
                        help='Image size (integer or "none" for dynamic resolution)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--data-root', type=str, default='./data',
                        help='Root directory for datasets')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--num-classes', type=int, default=200,
                        help='Number of classes')

    # ==================== Model Architecture ====================
    parser.add_argument('--dim', type=int, default=384,
                        help='Model embedding dimension')
    parser.add_argument('--num-layers', type=int, default=8,
                        help='Number of transformer layers')
    parser.add_argument('--heads', type=int, default=6,
                        help='Number of attention heads')
    parser.add_argument('--mlp-dim', type=int, default=1536,
                        help='MLP hidden dimension')
    parser.add_argument('--dim-head', type=int, default=None,
                        help='Per-head dimension (default: dim/heads)')
    parser.add_argument('--min-patch-size', type=int, default=4,
                        help='Minimum patch size')
    parser.add_argument('--channels', type=int, default=3,
                        help='Number of input channels')

    # ==================== Pooling & FFN ====================
    parser.add_argument('--pool', type=str, default='weighted',
                        help='Pooling type: cls, mean, weighted')
    parser.add_argument('--ffn-type', type=str, default='swiglu_level',
                        help='FFN type: swiglu, swiglu_level, geglu')

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
    parser.add_argument('--target-ratio', type=float, default=0.5,
                        help='Target token ratio')

    # ==================== Splitter ====================
    parser.add_argument('--splitter-type', type=str, default='gumbel_topk',
                        help='Splitter type: gumbel_topk, deterministic_neighbor, hilbert_optimal')
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
    parser.add_argument('--gradient-clip', type=float, default=1.0,
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


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="FractalCurveViT Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_args(parser)
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    configure_cuda()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Quick test mode: override with smaller model parameters
    if getattr(args, 'quick_test', False):
        args.dim = 128
        args.num_layers = 2
        args.heads = 4
        args.mlp_dim = 512
        args.num_train_samples = 100
        args.num_val_samples = 50
        args.batch_size = 4
        print(f"\n[Quick Test Mode] Using small model: dim={args.dim}, layers={args.num_layers}")

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
