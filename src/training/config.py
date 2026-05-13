"""Training Configuration Module

This module provides training-specific configuration dataclasses.
Following the three-layer parameter principle:
- Layer 1 (Parameters): Model architecture - handled by ModelGene
- Layer 2 (Variable): Runtime parameters (K, max_level) - computed internally
- Layer 3 (Hyper): Training hyperparameters - defined here

This config ONLY contains Layer 3 (hyperparameters) for training,
decoupled from model architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import json


@dataclass
class TrainingHyperparams:
    """Training hyperparameters - Layer 3 (Hyper parameters)

    These are FIXED training settings that don't change during training.
    Following the three-layer parameter principle, these are hyper-parameters
    that should be consistent across an experiment run.

    Attributes:
        num_epochs: Total training epochs
        batch_size: Batch size per iteration
        accumulation_steps: Gradient accumulation steps
        gradient_clip_norm: Gradient clipping threshold
        warmup_epochs: LR warmup epochs
        min_lr: Minimum learning rate after warmup
        weight_decay: Weight decay coefficient
        label_smoothing: Label smoothing factor
    """
    # Core training
    num_epochs: int = 100
    batch_size: int = 128
    accumulation_steps: int = 1
    gradient_clip_norm: float = 5.0

    # Learning rate
    base_lr: float = 5e-4
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    warmup_start_lr: float = 1e-7

    # V4: Splitter 独立学习率倍数
    # Splitter 使用 base_lr * splitter_lr_multiplier 以保持足够的路径选择能力
    splitter_lr_multiplier: float = 5.0

    # V2: Budget loss weight for BPE regularization (梯度平衡)
    # 目标: 使 Budget 梯度达到 CE 梯度的 10-15% (约 0.5-0.8 梯度范数)
    budget_loss_weight: float = 0.05

    # Regularization
    weight_decay: float = 0.05
    label_smoothing: float = 0.0

    # Mixup/Cutmix
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    mixup_cutmix_prob: float = 0.5

    # Logging
    log_interval: int = 50
    eval_interval: int = 1
    checkpoint_interval: int = 10


@dataclass
class NumericalConfig:
    """Numerical stability and monitoring configuration

    Controls gradient anomaly detection and detailed numerical monitoring.
    """
    # Anomaly detection
    detect_anomaly: bool = False
    check_gradients: bool = True
    skip_on_nan_grad: bool = True

    # Gradient monitoring
    record_grad_norms: bool = True
    record_layer_grad_norms: bool = True

    # Loss monitoring
    record_loss_components: bool = True


@dataclass
class CheckpointConfig:
    """Checkpoint saving and loading configuration"""
    checkpoint_dir: str = "./checkpoints"
    save_best: bool = True
    save_last: bool = True
    save_interval: int = 10
    monitor_metric: str = "val_accuracy"
    monitor_mode: str = "max"  # "max" or "min"


@dataclass
class DataConfig:
    """Data loading configuration"""
    dataset: str = "tiny-imagenet"
    data_dir: str = "./data"
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    image_size: int = 64
    augment: bool = True
    auto_augment: Optional[str] = "rand-m9-mstd0.5"


@dataclass
class MixedPrecisionConfig:
    """Mixed precision training configuration"""
    enabled: bool = True
    opt_level: str = "O1"  # O1 or O2
    loss_scale: Optional[float] = None  # None = dynamic


@dataclass
class Config:
    """Main training configuration container

    This config is COMPLETELY DECOUPLED from model architecture.
    It only contains training-specific hyperparameters.

    The model architecture (dim, depth, heads, etc.) should be passed
    separately and managed by the caller.
    """

    # Hyperparameters
    training: TrainingHyperparams = field(default_factory=TrainingHyperparams)

    # Numerical settings
    numerical: NumericalConfig = field(default_factory=NumericalConfig)

    # Checkpoint settings
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    # Data settings
    data: DataConfig = field(default_factory=DataConfig)

    # Mixed precision
    amp: MixedPrecisionConfig = field(default_factory=MixedPrecisionConfig)

    # Experiment
    seed: int = 42
    output_dir: str = "./outputs"
    device: str = "cuda"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        result = {}
        for key in ["training", "numerical", "checkpoint", "data", "amp"]:
            obj = getattr(self, key)
            if hasattr(obj, "__dataclass_fields__"):
                for field_name in obj.__dataclass_fields__:
                    result[f"{key}.{field_name}"] = getattr(obj, field_name)
            else:
                result[key] = obj
        result["seed"] = self.seed
        result["output_dir"] = self.output_dir
        result["device"] = self.device
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Config:
        """Create config from dictionary"""
        config = cls()

        # Parse nested config
        for key in ["training", "numerical", "checkpoint", "data", "amp"]:
            if hasattr(config, key):
                obj = getattr(config, key)
                if hasattr(obj, "__dataclass_fields__"):
                    for field_name in obj.__dataclass_fields__:
                        full_key = f"{key}.{field_name}"
                        if full_key in d:
                            setattr(obj, field_name, d[full_key])

        if "seed" in d:
            config.seed = d["seed"]
        if "output_dir" in d:
            config.output_dir = d["output_dir"]
        if "device" in d:
            config.device = d["device"]

        return config

    def save(self, path: str) -> None:
        """Save config to JSON file"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> Config:
        """Load config from JSON file"""
        with open(path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)


# Convenience function to create config
def create_config(
    num_epochs: int = 100,
    batch_size: int = 128,
    base_lr: float = 5e-4,
    weight_decay: float = 0.05,
    device: str = "cuda",
    **kwargs
) -> Config:
    """Create a training config with common defaults"""
    config = Config()
    config.training.num_epochs = num_epochs
    config.training.batch_size = batch_size
    config.training.base_lr = base_lr
    config.training.weight_decay = weight_decay
    config.device = device

    # Apply any additional kwargs
    for key, value in kwargs.items():
        if hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config, key):
            setattr(config, key, value)

    return config


# I147: Additional config classes expected by tests
@dataclass
class ModelArchitectureConfig:
    """Model architecture configuration (I147)"""
    focal_gamma: float = 0.0
    use_focal_loss: bool = False


@dataclass
class OptimizerConfig:
    """Optimizer configuration (I147)"""
    lr: float = 5e-4


@dataclass
class LossConfig:
    """Loss configuration (I147)"""
    type: str = "cross_entropy"


__all__ = [
    "Config",
    "TrainingHyperparams",
    "NumericalConfig",
    "CheckpointConfig",
    "DataConfig",
    "MixedPrecisionConfig",
    "create_config",
    "ModelArchitectureConfig",
    "OptimizerConfig",
    "LossConfig",
]
