"""Checkpoint Loader Module

Loads training checkpoints with complete state recovery.
Following the three-layer parameter principle, this module
handles Layer 3 (hyperparameters) for checkpoint loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn


def load_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
    load_optimizer: bool = True,
    load_scheduler: bool = True,
    load_scaler: bool = True,
) -> Dict[str, Any]:
    """Load training checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load to
        load_optimizer: Whether to load optimizer state
        load_scheduler: Whether to load scheduler state
        load_scaler: Whether to load GradScaler state

    Returns:
        Dictionary with checkpoint contents:
        - model_state_dict
        - optimizer_state_dict
        - scheduler_state_dict
        - scaler_state_dict
        - epoch
        - metrics
        - training_state
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[CHECKPOINT] Loading: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Extract components
    result = {
        "model_state_dict": checkpoint.get("model_state_dict"),
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
    }

    if load_optimizer:
        result["optimizer_state_dict"] = checkpoint.get("optimizer_state_dict")

    if load_scheduler:
        result["scheduler_state_dict"] = checkpoint.get("scheduler_state_dict")

    if load_scaler:
        result["scaler_state_dict"] = checkpoint.get("scaler_state_dict")

    result["training_state"] = checkpoint.get("training_state")

    print(f"[CHECKPOINT] Loaded epoch {result['epoch']}, metrics: {result['metrics']}")

    return result


def load_model_weights(
    model: nn.Module,
    checkpoint_path: str,
    device: str = "cpu",
    strict: bool = False,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load only model weights from checkpoint

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint
        device: Device to load to
        strict: Whether to strictly enforce key matching

    Returns:
        Tuple of (model, metadata dict)
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Get model state
    model_state = checkpoint.get("model_state_dict", checkpoint)

    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=strict)

    if missing_keys:
        print(f"[CHECKPOINT] Missing keys: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"[CHECKPOINT] Unexpected keys: {unexpected_keys[:5]}...")

    # Return metadata
    metadata = {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
    }

    return model, metadata


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in directory

    Args:
        checkpoint_dir: Directory to search

    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    # Check for 'last.pth' first
    last_path = checkpoint_dir / "last.pth"
    if last_path.exists():
        return str(last_path)

    # Find all checkpoints
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))

    if not checkpoints:
        return None

    # Sort by epoch (extract number from filename)
    checkpoints.sort(key=lambda p: int(p.stem.split("_")[-1]))

    return str(checkpoints[-1])


def find_best_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the best checkpoint in directory

    Args:
        checkpoint_dir: Directory to search

    Returns:
        Path to best checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)

    best_path = checkpoint_dir / "best.pth"
    if best_path.exists():
        return str(best_path)

    return None


__all__ = [
    "load_checkpoint",
    "load_model_weights",
    "find_latest_checkpoint",
    "find_best_checkpoint",
]
