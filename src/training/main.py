"""Main Training Entry Point

Integrates all training modules into a complete training pipeline.
Following the three-layer parameter principle, this only handles
Layer 3 (hyperparameters) for training, decoupled from model architecture.
"""

from __future__ import annotations

import argparse
import sys
import random
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Add project root to path
def _setup_path():
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
_setup_path()

from .config import Config, create_config  # noqa: E402
from .trainer import (  # noqa: E402
    TrainingState,
    train_one_epoch,
    evaluate,
    MixupCutmixLoss,
)
from .scheduler import create_scheduler  # noqa: E402
from .monitor import (  # noqa: E402
    GradientMonitor,
    LossMonitor,
    NumericalDefender,
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


def create_dummy_dataloader(
    batch_size: int = 128,
    num_samples: int = 1000,
    image_size: int = 64,
    channels: int = 3,
    num_classes: int = 200,
    num_workers: int = 0,
) -> DataLoader:
    """Create a dummy dataloader for testing"""

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return num_samples

        def __getitem__(self, idx):
            return (
                torch.randn(channels, image_size, image_size),
                torch.randint(0, num_classes, (1,)).item()
            )

    return DataLoader(
        DummyDataset(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


def create_model(
    num_classes: int = 200,
    device: str = "cuda",
) -> nn.Module:
    """Create a dummy model for testing

    In real usage, this would load FractalCurveViT from vit_pytorch.
    """
    # This is a placeholder - in real usage, import from vit_pytorch
    # from vit_pytorch import FractalCurveViT

    class DummyModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.num_classes = num_classes
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.fc = nn.Linear(64, num_classes)

        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=[2, 3])
            return self.fc(x)

    return DummyModel(num_classes).to(device)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    device: torch.device,
    resume_from: Optional[str] = None,
) -> Dict[str, Any]:
    """Main training function

    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to train on
        resume_from: Optional checkpoint path to resume from

    Returns:
        Final metrics dictionary
    """
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize state
    state = TrainingState()

    # Resume from checkpoint if specified
    if resume_from:
        checkpoint = load_checkpoint(resume_from, device=str(device))
        state = TrainingState.from_dict(checkpoint.get("training_state", {}))
        # Load model weights
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed from epoch {state.epoch}")

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
        warmup_start_lr=config.training.warmup_start_lr,
    )

    # Resume scheduler state if available
    if state.scheduler_state:
        scheduler.load_state_dict(state.scheduler_state)

    # Create GradScaler (保守初始化，防止 warmup 期 GradScaler collapse)
    # init_scale=2048: 崩溃阶梯从 16 步降到 5 步
    # growth_interval=500: 更长的稳定观察期
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
            num_classes=model.num_classes if hasattr(model, 'num_classes') else 200,
            label_smoothing=config.training.label_smoothing,
            mixup_alpha=config.training.mixup_alpha,
            cutmix_alpha=config.training.cutmix_alpha,
            mixup_prob=config.training.mixup_cutmix_prob,
        )

    # Create monitors
    grad_monitor = GradientMonitor(
        model=model,
        record_layer_norms=config.numerical.record_layer_grad_norms,
    )
    loss_monitor = LossMonitor()
    NumericalDefender(
        model=model,
        detect_anomaly=config.numerical.detect_anomaly,
        skip_on_nan=config.numerical.skip_on_nan_grad,
    )

    # Create logger
    logger = EpochLogger(output_dir=str(output_dir))

    # Move model to device
    model = model.to(device)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {config.training.num_epochs} epochs")
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
        )

        # Reset monitors
        grad_monitor.reset()
        loss_monitor.reset()

        # Evaluate
        eval_metrics = None
        if (epoch + 1) % config.training.eval_interval == 0:
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

        # Check if best
        is_best = False
        if eval_metrics and config.checkpoint.save_best:
            # Initialize best_metric if first epoch (0.0 is ambiguous for accuracy)
            if state.best_metric == 0.0 and config.checkpoint.monitor_mode == "max":
                state.best_metric = float('-inf')
            metric_value = eval_metrics.get(config.checkpoint.monitor_metric, 0.0)
            if config.checkpoint.monitor_mode == "max":
                is_best = metric_value > state.best_metric
            else:
                is_best = metric_value < state.best_metric

            if is_best:
                state.best_metric = metric_value

        # Save checkpoint: separate concerns
        # 1. Save epoch checkpoint every N epochs (controlled by save_interval)
        # 2. Save best.pth only when is_best=True (handled in save_checkpoint)
        # 3. Save last.pth every time (handled in save_checkpoint)
        save_interval = config.checkpoint.save_interval
        should_save_epoch = (epoch + 1) % save_interval == 0 or (epoch + 1) >= config.training.num_epochs

        if should_save_epoch:
            save_checkpoint(
                checkpoint_dir=config.checkpoint.checkpoint_dir,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics={"train": train_metrics.to_dict(), "eval": eval_metrics or {}},
                scheduler_state=state.scheduler_state,
                scaler_state=state.scaler_state,
                training_state=state.to_dict(),
                is_best=is_best,
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
                is_best=True,
                save_epoch_checkpoint=False,
            )

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best metric: {state.best_metric:.4f}")
    print(f"{'='*60}\n")

    return {
        "best_metric": state.best_metric,
        "history": logger.get_history(),
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Fractal Training")

    # Training hyperparameters
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--base-lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)

    # Mixup/Cutmix
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)

    # Numerical
    parser.add_argument("--detect-anomaly", action="store_true")
    parser.add_argument("--skip-on-nan-grad", action="store_true", default=True)

    # Model
    parser.add_argument("--num-classes", type=int, default=200)

    # Data
    parser.add_argument("--num-train-samples", type=int, default=10000)
    parser.add_argument("--num-val-samples", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=64)

    # System
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None)

    # Quick test
    parser.add_argument("--quick-test", action="store_true", help="Run quick test with dummy data")

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    configure_cuda()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Create config
    config = create_config(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        base_lr=args.base_lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )
    config.training.warmup_epochs = args.warmup_epochs
    config.training.min_lr = args.min_lr
    config.training.mixup_alpha = args.mixup_alpha
    config.training.cutmix_alpha = args.cutmix_alpha
    config.numerical.detect_anomaly = args.detect_anomaly
    config.numerical.skip_on_nan_grad = args.skip_on_nan_grad
    config.output_dir = args.output_dir
    config.checkpoint.checkpoint_dir = args.checkpoint_dir

    # Save config
    config.save(Path(args.output_dir) / "config.json")

    # Create model
    print("Creating model...")
    model = create_model(num_classes=args.num_classes, device=str(device))

    # Create dataloaders
    if args.quick_test:
        print("Running quick test with dummy data...")
        train_loader = create_dummy_dataloader(
            batch_size=args.batch_size,
            num_samples=args.num_train_samples,
            image_size=args.image_size,
            num_classes=args.num_classes,
        )
        val_loader = create_dummy_dataloader(
            batch_size=args.batch_size,
            num_samples=args.num_val_samples,
            image_size=args.image_size,
            num_classes=args.num_classes,
        )
    else:
        # In real usage, load actual datasets here
        raise NotImplementedError("Please use --quick-test for now")

    # Train
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
