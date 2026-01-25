# -*- coding: utf-8 -*-
"""
Tiny-ImageNet Optimal Training Configuration (RTX 4070 Laptop)
===============================================================

Mathematical Formalization for 100 Epoch Training with:
- gradient-checkpoint
- --compile
- --channels-last
- Maximum batch_size = 192

Math Derivation Summary:
1. Model Parameters: P = depth × 16 × dim² + 2.5 × dim × num_classes
2. Memory Budget: M_total = 2P + 8P + B × L × S × D × 13 × checkpoint_factor × amp_factor
3. Learning Rate: lr = 3e-4 × (batch_eff / 256)
4. P/N Ratio: R = P / N_samples ∈ [150, 400] for ViT

Author: Claude Code
Date: 2026-01-20
"""

import math
from dataclasses import dataclass, field
from typing import Tuple, Optional
import torch

# =============================================================================
# I98-6: 统一配置入口
# 使用 ModelArchitectureConfig 作为统一的模型配置类
# =============================================================================
from src.training.config import ModelArchitectureConfig as ModelConfig


# =============================================================================
# 2. Tokenizer Configuration
# =============================================================================

@dataclass
class TokenizerConfig:
    """
    Fractal Tokenizer Configuration

    Math Derivation:
    - N_max_patches = (image_size / min_patch_size)²
    - N_tokens = round(coverage × N_max_patches)
    - coverage ∈ [0.01, 0.05] for Tiny-ImageNet

    For image_size=64, min_patch_size=4:
    - N_max_patches = (64/4)² = 256
    - N_tokens ∈ [8, 64] (clamped by K_min_abs, K_max)
    """
    image_size: int = 64
    min_patch_size: int = 4
    K_min_abs: int = 8       # Absolute minimum tokens
    K_max: int = 64          # Maximum tokens
    coverage_min: float = 0.01
    coverage_max: float = 0.05

    @property
    def patches_per_side(self) -> int:
        return self.image_size // self.min_patch_size

    @property
    def N_max_patches(self) -> int:
        return self.patches_per_side ** 2

    @property
    def N_tokens_min(self) -> int:
        return max(self.K_min_abs, int(self.coverage_min * self.N_max_patches))

    @property
    def N_tokens_max(self) -> int:
        return min(self.K_max, int(self.coverage_max * self.N_max_patches))

    @property
    def N_tokens_typical(self) -> int:
        return (self.N_tokens_min + self.N_tokens_max) // 2


# =============================================================================
# 3. Memory Budget Calculation
# =============================================================================

def calc_memory_budget(
    params: int,
    batch_size: int,
    seq_len: int,
    dim: int,
    depth: int,
    heads: int,
    use_checkpoint: bool = True,
    use_amp: bool = True,
    use_channels_last: bool = True,
) -> dict:
    """
    Calculate training memory budget (I101-2 修正版)

    Math Formulas:
    - M_params = P × 4 bytes (FP32) or × 2 (FP16)
    - M_grad = P × 4 bytes (FP32) or × 2 (FP16)
    - M_optimizer = P × 8 bytes (AdamW m+v states)
    - M_activations = B × L × S × D × 8 × checkpoint_factor × amp_factor
    - M_attention = B × H × S × S × 2 × depth × 0.5 (FP16, with checkpoint)
    - M_lca_bias = B × H × S × S × 4 (FP32, LCA bias matrix)
    - M_levels_info = B × S × (D+1) × 4 (levels_info data)
    - M_data = B × C × H × W × 4 bytes

    Key Fix (I101-2):
    - Original formula missed O(N²) attention and LCA bias matrices
    - For N=256, H=8, depth=10, batch=64:
      - Attention matrix: ~655 MB (FP16)
      - LCA bias matrix: ~128 MB (FP32)
      - Total missing: ~783 MB!

    Optimization Factors:
    - checkpoint_factor = 0.35 (65% savings with gradient checkpointing)
    - amp_factor = 0.5 (FP16 computation)
    - channels_last_factor = 0.9 (memory layout optimization)

    Returns:
        Memory budget in MB (detailed breakdown)
    """
    import math

    checkpoint_factor = 0.35 if use_checkpoint else 1.0
    amp_factor = 0.5 if use_amp else 1.0
    channels_last_factor = 0.9 if use_channels_last else 1.0

    # Parameters (FP32)
    M_params = params * 4 / (1024 ** 2)

    # Gradients (FP32)
    M_grad = params * 4 / (1024 ** 2)

    # Optimizer states (AdamW: m + v)
    M_optimizer = params * 8 / (1024 ** 2)

    # I101-2 Fix: Activations (corrected formula)
    # Q, K, V, O projections + FFN activations = 8 × dim per token
    M_activations = (
        batch_size * seq_len * dim * 8 * 4 / (1024 ** 2)  # FP32 baseline
        * depth * checkpoint_factor * amp_factor * channels_last_factor
    )

    # I101-2 Fix: Attention matrices (B×H×N×N×2 bytes per head, FP16)
    # Standard ViT attention: QK^T matrix per layer per head
    # With gradient checkpointing, only need to recompute during backward
    M_attention = (
        batch_size * heads * seq_len * seq_len * 2 / (1024 ** 2)  # FP16
        * depth * 0.5  # checkpoint saves ~50% of attention memory
        * amp_factor
    )

    # I101-2 Fix: LCA bias matrix (B×H×N×N×4 bytes, FP32 required for stability)
    # This is the Fractal ViT specific O(N²) memory consumer
    M_lca_bias = (
        batch_size * heads * seq_len * seq_len * 4 / (1024 ** 2)  # FP32
    )

    # I101-2 Fix: levels_info data (B×N×(D+1)×4 bytes)
    # D = log2(N) is the maximum quadtree depth
    D = max(1, int(math.log2(seq_len))) if seq_len > 0 else 1
    M_levels_info = (
        batch_size * seq_len * (D + 1) * 4 / (1024 ** 2)
    )

    # Input data (64x64 RGB)
    M_data = batch_size * 3 * 64 * 64 * 4 / (1024 ** 2)

    # CUDA overhead
    M_cuda = 500

    # Calculate total with all terms
    total = (
        M_params + M_grad + M_optimizer +
        M_activations + M_attention + M_lca_bias +
        M_levels_info + M_data + M_cuda
    )

    return {
        "params_MB": M_params,
        "grad_MB": M_grad,
        "optimizer_MB": M_optimizer,
        "activations_MB": M_activations,
        "attention_MB": M_attention,      # I101-2: was missing!
        "lca_bias_MB": M_lca_bias,        # I101-2: was missing!
        "levels_info_MB": M_levels_info,  # I101-2: was missing!
        "data_MB": M_data,
        "cuda_MB": M_cuda,
        "total_MB": total,
        "utilization_pct": total / 8192 * 100,
    }


# =============================================================================
# 4. Learning Rate Schedule
# =============================================================================

@dataclass
class LRScheduleConfig:
    """
    Learning Rate Schedule Configuration

    Math Formulas:
    - Linear Warmup: lr(t) = lr_max × t / warmup_steps, t ∈ [0, warmup_steps]
    - Cosine Decay: lr(t) = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t / T))
    - Linear Scaling: lr = base_lr × (batch_eff / 256)

    For Tiny-ImageNet (100K samples), batch=192:
    - lr = 3e-4 × (192 / 256) = 2.25e-4
    - warmup_steps = 10 × (100000 / 192) ≈ 5208
    """
    base_lr: float = 3e-4      # Base LR for batch=256
    batch_size: int = 192
    warmup_epochs: int = 10
    total_epochs: int = 100
    num_samples: int = 100000
    min_lr_ratio: float = 0.01

    @property
    def eff_batch_size(self) -> int:
        return self.batch_size

    @property
    def scaled_lr(self) -> float:
        """LR scaled by effective batch size"""
        return self.base_lr * (self.eff_batch_size / 256)

    @property
    def min_lr(self) -> float:
        return self.scaled_lr * self.min_lr_ratio

    @property
    def steps_per_epoch(self) -> int:
        return self.num_samples // self.eff_batch_size

    @property
    def warmup_steps(self) -> int:
        return self.warmup_epochs * self.steps_per_epoch

    @property
    def total_steps(self) -> int:
        return self.total_epochs * self.steps_per_epoch


# =============================================================================
# 5. Regularization Configuration
# =============================================================================

@dataclass
class RegularizationConfig:
    """
    Regularization Configuration

    Math Formulas:
    - DropPath: rate = drop_path_rate × layer / depth (linear increase)
    - Label Smoothing: target = (1 - ε) × y_true + ε / num_classes
    - Weight Decay: AdamW decoupled wd

    For 100 epochs, depth=10:
    - drop_path_rate ∈ [0.05, 0.2] for ViT
    - label_smoothing = 0.1 (ε = 0.1)
    """
    drop_path_rate: float = 0.15
    dropout: float = 0.1
    label_smoothing: float = 0.1
    weight_decay: float = 0.1

    @property
    def attention_dropout(self) -> float:
        return 0.0  # ViT standard

    @property
    def embedding_dropout(self) -> float:
        return self.dropout


# =============================================================================
# 6. Optimization Configuration
# =============================================================================

@dataclass
class OptimizationConfig:
    """
    PyTorch 2.4+ Optimization Configuration

    Configuration for:
    - gradient-checkpoint: Reduces activation memory by ~65%
    - torch.compile: ~30% speedup with CUDA graphs
    - channels-last: ~20% memory savings, better performance
    - AMP (FP16): 50% memory reduction for activations
    """
    use_checkpoint: bool = True
    use_compile: bool = True
    use_channels_last: bool = True
    use_amp: bool = True
    compile_mode: str = "reduce-overhead"  # Training optimal
    cudnn_benchmark: bool = True
    tf32_enabled: bool = True

    @property
    def backend_flags(self) -> dict:
        """PyTorch backend configuration"""
        return {
            "cudnn.benchmark": self.cudnn_benchmark,
            "cudnn.allow_tf32": self.tf32_enabled,
            "matmul.allow_tf32": self.tf32_enabled,
        }


# =============================================================================
# 7. Complete Training Configuration
# =============================================================================

@dataclass
class TinyImageNetTrainingConfig:
    """
    Complete Tiny-ImageNet Training Configuration (RTX 4070 Laptop)

    Mathematically derived for 100 epochs with:
    - gradient-checkpoint
    - --compile
    - --channels-last
    - batch_size = 192

    Validation:
    - Parameters: 41.4M (P/N = 414, acceptable for ViT with strong regularization)
    - Memory: ~1.2GB (15% of 8GB VRAM)
    - Learning Rate: 2.25e-4 (scaled from base 3e-4)
    """
    # Model Architecture
    model: ModelConfig = field(default_factory=ModelConfig)

    # Tokenizer
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)

    # Learning Rate
    lr: LRScheduleConfig = field(default_factory=LRScheduleConfig)

    # Regularization
    reg: RegularizationConfig = field(default_factory=RegularizationConfig)

    # Optimization
    opt: OptimizationConfig = field(default_factory=OptimizationConfig)

    # Training Settings
    epochs: int = 100
    gradient_accumulation: int = 1
    gradient_clip: float = 1.0

    # Data
    num_workers: int = 4
    val_split: float = 0.1

    def validate(self) -> dict:
        """Validate configuration against hardware constraints"""
        mem = calc_memory_budget(
            params=self.model.params_total,
            batch_size=self.lr.batch_size,
            seq_len=self.tokenizer.N_tokens_typical,
            dim=self.model.dim,
            depth=self.model.depth,
            heads=self.model.heads,
            use_checkpoint=self.opt.use_checkpoint,
            use_amp=self.opt.use_amp,
            use_channels_last=self.opt.use_channels_last,
        )

        pn_ratio = self.model.params_total / self.lr.num_samples

        return {
            "memory": mem,
            "pn_ratio": pn_ratio,
            "is_feasible": mem["utilization_pct"] < 80,
            "is_optimal": 150 <= pn_ratio <= 500,
        }

    def __str__(self) -> str:
        """String representation"""
        validation = self.validate()
        mem = validation["memory"]

        return f"""
================================================================================
Tiny-ImageNet Training Configuration (RTX 4070 Laptop)
================================================================================

Model Architecture:
  - dim: {self.model.dim}
  - depth: {self.model.depth}
  - heads: {self.model.heads} (dim_head={self.model.dim_head})
  - mlp_dim: {self.model.mlp_dim} (ratio={self.model.mlp_ratio:.2f})
  - Parameters: {self.model.params_total/1e6:.2f}M

Tokenizer:
  - image_size: {self.tokenizer.image_size}
  - min_patch_size: {self.tokenizer.min_patch_size}
  - tokens: [{self.tokenizer.N_tokens_min}, {self.tokenizer.N_tokens_max}] (typ={self.tokenizer.N_tokens_typical})
  - patches: {self.tokenizer.N_max_patches}

Learning Rate:
  - base_lr: {self.lr.base_lr}
  - scaled_lr: {self.lr.scaled_lr:.2e}
  - warmup_epochs: {self.lr.warmup_epochs}
  - min_lr: {self.lr.min_lr:.2e}
  - total_steps: {self.lr.total_steps}

Regularization:
  - drop_path: {self.reg.drop_path_rate}
  - dropout: {self.reg.dropout}
  - label_smoothing: {self.reg.label_smoothing}
  - weight_decay: {self.reg.weight_decay}

Optimization:
  - checkpoint: {self.opt.use_checkpoint}
  - compile: {self.opt.use_compile} (mode={self.opt.compile_mode})
  - channels_last: {self.opt.use_channels_last}
  - AMP: {self.opt.use_amp}
  - TF32: {self.opt.tf32_enabled}

Validation:
  - P/N ratio: {validation['pn_ratio']:.2f}
  - Memory: {mem['total_MB']:.0f}MB ({mem['utilization_pct']:.1f}%)
  - Feasible: {'YES' if validation['is_feasible'] else 'NO'}
================================================================================
"""


# =============================================================================
# 8. Command Generation
# =============================================================================

def generate_training_command(config: TinyImageNetTrainingConfig) -> str:
    """Generate training command from configuration"""
    cmd = [
        "uv run python examples/training/train_fractal_vit.py",
        f"--dataset tiny-imagenet",
        f"--image-size {config.tokenizer.image_size}",
        f"--batch-size {config.lr.batch_size}",
        f"--epochs {config.epochs}",
        f"--dim {config.model.dim}",
        f"--depth {config.model.depth}",
        f"--heads {config.model.heads}",
        f"--mlp-dim {config.model.mlp_dim}",
        f"--learning-rate {config.lr.scaled_lr:.2e}",
        f"--warmup-epochs {config.lr.warmup_epochs}",
        f"--dropout {config.reg.dropout}",
        f"--drop-path {config.reg.drop_path_rate}",
        f"--label-smoothing {config.reg.label_smoothing}",
        f"--weight-decay {config.reg.weight_decay}",
        f"--min-patch-size {config.tokenizer.min_patch_size}",
        f"--K-min {config.tokenizer.K_min_abs}",
        f"--K-max {config.tokenizer.K_max}",
    ]

    if config.opt.use_checkpoint:
        cmd.append("--gradient-checkpoint")
    if config.opt.use_compile:
        cmd.append("--compile")
    if config.opt.use_channels_last:
        cmd.append("--channels-last")
    if config.opt.use_amp:
        cmd.append("--use-amp")
    if config.opt.tf32_enabled:
        cmd.append("--tf32")

    return " \\\n  ".join(cmd)


# =============================================================================
# 9. Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Tiny-ImageNet Optimal Training Configuration")
    print("RTX 4070 Laptop | 100 Epochs | Batch=192")
    print("=" * 80)

    # Create configuration
    config = TinyImageNetTrainingConfig()

    # Print configuration
    print(config)

    # Print training command
    print("\nTraining Command:")
    print("-" * 80)
    print(generate_training_command(config))
    print("-" * 80)

    # Alternative: Higher capacity model
    print("\n\nAlternative: Higher Capacity Model (dim=384, depth=12)")
    print("-" * 80)

    config_large = TinyImageNetTrainingConfig()
    config_large.model = ModelConfig(
        num_classes=200,
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=384 * 4,
    )

    print(config_large)

    validation = config_large.validate()
    if validation['is_feasible']:
        print("\nTraining Command (Large Model):")
        print(generate_training_command(config_large))
    else:
        print("WARNING: Configuration exceeds memory budget!")
