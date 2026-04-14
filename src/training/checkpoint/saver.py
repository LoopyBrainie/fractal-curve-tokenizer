"""Checkpoint Saver Module

Saves training checkpoints with complete state.
Following the three-layer parameter principle, this module
handles Layer 3 (hyperparameters) for checkpointing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import torch
import json


def save_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    scheduler_state: Optional[Dict[str, Any]] = None,
    scaler_state: Optional[Dict[str, Any]] = None,
    training_state: Optional[Dict[str, Any]] = None,
    is_best: bool = False,
    filename: Optional[str] = None,
    save_epoch_checkpoint: bool = True,
) -> str:
    """Save training checkpoint

    Saves complete training state for resume:
    - Model state dict
    - Optimizer state dict
    - Scheduler state dict
    - GradScaler state dict
    - Training metrics

    Args:
        checkpoint_dir: Directory to save checkpoint
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Dictionary of metrics
        scheduler_state: Optional LR scheduler state
        scaler_state: Optional GradScaler state
        training_state: Optional full training state
        is_best: Whether this is the best checkpoint
        filename: Optional custom filename
        save_epoch_checkpoint: Whether to save epoch checkpoint (default True)
            Set to False when only updating best.pth/last.pth without epoch checkpoint

    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Prepare checkpoint dict
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    # Add optional states
    if scheduler_state is not None:
        checkpoint["scheduler_state_dict"] = scheduler_state

    if scaler_state is not None:
        checkpoint["scaler_state_dict"] = scaler_state

    if training_state is not None:
        checkpoint["training_state"] = training_state

    # Save regular checkpoint (epoch checkpoint)
    checkpoint_path = None
    if save_epoch_checkpoint:
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pth"
        checkpoint_path = checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"[CHECKPOINT] Saved: {checkpoint_path}")

    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / "best.pth"
        torch.save(checkpoint, best_path)
        print(f"[CHECKPOINT] Saved best: {best_path}")

    # Save last checkpoint
    last_path = checkpoint_dir / "last.pth"
    torch.save(checkpoint, last_path)
    print(f"[CHECKPOINT] Saved last: {last_path}")

    return str(checkpoint_path) if checkpoint_path else str(last_path)


def save_epoch_stats(
    output_dir: str,
    epoch: int,
    train_metrics: Dict[str, Any],
    eval_metrics: Dict[str, Any],
) -> str:
    """Save epoch statistics to JSON

    Args:
        output_dir: Directory to save stats
        epoch: Current epoch
        train_metrics: Training metrics
        eval_metrics: Evaluation metrics

    Returns:
        Path to saved stats
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge metrics
    stats = {
        "epoch": epoch,
        "train": train_metrics,
        "eval": eval_metrics,
    }

    # Save to JSON
    stats_path = output_dir / f"epoch_{epoch:04d}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return str(stats_path)


__all__ = [
    "save_checkpoint",
    "save_epoch_stats",
]
