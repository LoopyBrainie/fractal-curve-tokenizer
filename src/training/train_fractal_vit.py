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
import math
import os
import sys
import random
from pathlib import Path
from typing import Optional, Dict, Any, List
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

from .config import create_config
from .trainer import (
    TrainingState,
    train_one_epoch,
    evaluate,
    MixupCutmixLoss,
    GradBalancer,
)
from .scheduler import create_scheduler, WarmupCosineScheduler
from .monitor import (
    GradientMonitor,
    LossMonitor,
    NumericalDefender,
)
from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
)
from .training_logs import EpochLogger


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
        'ffn_type': getattr(args, 'ffn_type', 'swiglu_level'),

        # Use checkpoint
        'use_checkpoint': use_checkpoint,
    }

    # Add optional parameters if provided
    if hasattr(args, 'use_area_encoding') and args.use_area_encoding:
        model_kwargs['use_area_encoding'] = True
        # Note: fourier_levels is not a FractalCurveViT parameter

    if hasattr(args, 'use_pattern_encoder') and args.use_pattern_encoder:
        model_kwargs['use_pattern_encoder'] = True

    if hasattr(args, 'use_geometry_field') and args.use_geometry_field:
        model_kwargs['use_geometry_field'] = True

    print("Creating FractalCurveViT with args:")
    for k, v in model_kwargs.items():
        print(f"  {k}: {v}")

    model = FractalCurveViT(**model_kwargs)

    # Apply optimizations
    if compile_model:
        print("Compiling model with torch.compile...")
        # I164-1: 使用 mode='default' 替代 'reduce-overhead'
        # 'reduce-overhead' 启用 CUDA Graphs，与 gradient_checkpointing 不兼容
        # 'default' 禁用 CUDA Graphs，避免动态形状导致的 index out of bounds
        model = torch.compile(model, mode='default')

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

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # HF datasets already handle shuffling internally
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


class ExplorationInjector:
    """V3: 连续 N epoch 活跃度冻结时触发温度扰动

    检测 active_ratio 是否在连续 patience 个 epoch 内保持冻结（方差 < threshold）。
    当验证准确率较低时说明模型陷入局部最优，此时临时提升 τ 打破僵局。

    数学形式:
       冻结检测: var(active_ratio_history[-patience:]) < freeze_threshold
       触发条件: frozen AND val_acc < exploration_threshold
       扰动幅度: τ_boost = tau * tau_boost_factor

    Args:
        patience: 冻结检测窗口大小 (default: 3)
        tau_boost_factor: τ 提升倍数 (default: 1.5)
        freeze_threshold: 冻结判定方差阈值 (default: 0.0001)
        exploration_threshold: 探索触发验证准确率阈值 (default: 0.05)
    """

    def __init__(
        self,
        patience: int = 3,
        tau_boost_factor: float = 1.5,
        freeze_threshold: float = 0.0001,
        exploration_threshold: float = 0.05,
    ):
        self.patience = patience
        self.tau_boost_factor = tau_boost_factor
        self.freeze_threshold = freeze_threshold
        self.exploration_threshold = exploration_threshold

        self.active_ratio_history: List[float] = []
        self.triggered_count: int = 0

    def update(self, active_ratio: float, val_acc: float, current_tau: float) -> float:
        """检测冻结状态并返回扰动后的温度

        Args:
            active_ratio: 当前 epoch 的 active_ratio
            val_acc: 当前 epoch 的验证准确率 (0-1)
            current_tau: 当前温度 τ

        Returns:
            扰动后的温度（如果触发则提升，否则返回原值）
        """
        self.active_ratio_history.append(active_ratio)

        if len(self.active_ratio_history) > self.patience:
            self.active_ratio_history.pop(0)

        # 检查冻结条件
        if len(self.active_ratio_history) == self.patience:
            variance = np.var(self.active_ratio_history)
            if variance < self.freeze_threshold and val_acc < self.exploration_threshold:
                self.triggered_count += 1
                return current_tau * self.tau_boost_factor

        return current_tau

    def get_status(self) -> Dict[str, Any]:
        """返回当前状态"""
        return {
            "triggered_count": self.triggered_count,
            "history_len": len(self.active_ratio_history),
        }


class GradAwareExplorationInjector:
    """V4: 梯度感知的探索注入学

    当检测到 splitter 梯度下降时，自动降低 τ 开启路径搜索。
    这是对原有 ExplorationInjector 的增强，补充了梯度监控维度。

    数学形式:
       梯度下降检测: splitter_grad_norm / prev_grad < grad_drop_threshold
       触发条件: 连续 patience 次下降
       扰动动作: τ_new = max(1.0, τ - 0.2)

    Args:
        patience: 连续下降检测窗口大小 (default: 2)
        grad_drop_threshold: 梯度下降判定阈值 (default: 0.5)
    """

    def __init__(
        self,
        patience: int = 2,
        grad_drop_threshold: float = 0.5,
    ):
        self.patience = patience
        self.grad_drop_threshold = grad_drop_threshold

        self.prev_splitter_grad: Optional[float] = None
        self.consecutive_drop_count: int = 0

    def update(
        self,
        splitter_grad_norm: float,
        current_tau: float,
        active_ratio: float = 0.0,
        val_acc: float = 0.0,
    ) -> float:
        """检测 splitter 梯度下降并返回调整后的温度

        Args:
            splitter_grad_norm: 当前 splitter 梯度范数
            current_tau: 当前温度 τ
            active_ratio: 当前 active_ratio (未使用，为兼容性保留)
            val_acc: 当前验证准确率 (未使用，为兼容性保留)

        Returns:
            调整后的温度（如果触发下降检测则降低，否则返回原值）
        """
        if self.prev_splitter_grad is None:
            self.prev_splitter_grad = splitter_grad_norm
            return current_tau

        grad_ratio = splitter_grad_norm / max(self.prev_splitter_grad, 1e-8)

        if grad_ratio < self.grad_drop_threshold:
            self.consecutive_drop_count += 1
        else:
            self.consecutive_drop_count = 0

        # V4: 如果连续下降，强制降低 τ 开启探索
        if self.consecutive_drop_count >= self.patience:
            new_tau = max(1.0, current_tau - 0.2)  # 最低降至 1.0
            if new_tau < current_tau:
                print(f"  [GradAwareExplorationInjector] τ forced down: {current_tau:.3f} -> {new_tau:.3f}")
                self.consecutive_drop_count = 0  # 重置计数
                self.prev_splitter_grad = splitter_grad_norm
                return new_tau

        self.prev_splitter_grad = splitter_grad_norm
        return current_tau

    def get_status(self) -> Dict[str, Any]:
        """返回当前状态"""
        return {
            "consecutive_drop_count": self.consecutive_drop_count,
            "prev_grad": self.prev_splitter_grad,
        }


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


def _update_fractal_hyperparams(
    model,
    epoch: int,
    warmup_epochs: int,
    config,
) -> dict:
    """更新 Fractal ViT 动态超参数（三阶段 warmup V2）

    V2 改进：
        - Power-Law Annealing: 使用 (0.5 * (1 - cos(πt)))^1.5 加速后期收敛
        - Diversity 提前介入: Stage 2 即开启 diversity 惩罚，防止死路径

    三阶段设计：
        Stage 1 (0.0-0.3): 纯探索阶段，高活跃度，无 budget 约束
        Stage 2 (0.3-1.0): Power-Law 压缩，逐渐引入 budget 约束 + diversity
        Stage 3 (1.0+):    结构巩固阶段，启用完整约束

    Args:
        model: FractalCurveViT 模型
        epoch: 当前 epoch
        warmup_epochs: warmup 总 epoch 数
        config: 配置对象（需包含 config.training.budget_loss_weight）

    Returns:
        包含 stage, target_ratio, budget_weight, tau, logits_diversity 的字典
    """
    warmup_progress = epoch / max(warmup_epochs, 1)  # [0, 1]
    # V2: 从 config.training 读取（而不是 getattr）
    budget_loss_weight_target = config.training.budget_loss_weight

    def cosine_progress(t: float) -> float:
        """S 型曲线进度因子: t=0 → 0.0, t=1 → 1.0（单向递增，避免悬崖）"""
        return 0.5 * (1 - math.cos(math.pi * t))

    def power_law_progress(t: float, power: float = 1.5) -> float:
        """V2: Power-Law 加速进度因子

        公式: (0.5 * (1 - cos(πt)))^power
        效果: 在 t → 1 时加速收敛，产生更强的向心压缩力
        """
        return cosine_progress(t) ** power

    # Stage 判断
    if warmup_progress < 0.3:
        # Stage 1: Stochastic Exploration（纯 CE 梯度，无 budget 约束）
        stage = 1
        target_ratio = 0.5
        budget_weight = 0.0
        tau = 2.0
        logits_diversity = False
    elif warmup_progress < 1.0:
        # Stage 2: Power-Law Budget Annealing（V2: 加速收敛 + diversity 提前开启）
        stage = 2
        stage_progress = (warmup_progress - 0.3) / 0.7  # [0, 1]
        # V2: 使用 Power-Law 加速: factor^1.5
        progress = power_law_progress(stage_progress, power=1.5)  # 单调递增: 0.0 → 1.0
        target_ratio = 0.5 + (0.1 - 0.5) * progress  # 0.5 → 0.1
        budget_weight = budget_loss_weight_target * progress  # 0.0 → 0.20
        tau = 2.0 + (1.0 - 2.0) * progress  # 2.0 → 1.0
        # V2: Diversity 在 Stage 2 即开启，辅助压缩过程中的路径筛选
        logits_diversity = True
    else:
        # Stage 3: Structural Consolidation（锁定目标）
        stage = 3
        target_ratio = 0.1
        budget_weight = budget_loss_weight_target
        tau = 1.0
        logits_diversity = True

    # 更新 model splitter 参数
    if hasattr(model, 'splitter'):
        splitter = model.splitter
        if hasattr(splitter, 'set_temperature'):
            splitter.set_temperature(tau)
        if hasattr(splitter, 'set_target_ratio'):
            splitter.set_target_ratio(target_ratio)
        if hasattr(splitter, 'set_budget_weight'):
            splitter.set_budget_weight(budget_weight)
        if hasattr(splitter, 'set_logits_diversity'):
            splitter.set_logits_diversity(logits_diversity)

    # V3: τ Floor - 防止 τ 过早降至 1.0，与 α 形成双重退火坍缩
    # α 在 epoch 25 后才稳定到 1.5，所以 τ 也应该延迟到 epoch 25 后才降至 1.0
    tau_floor = 1.2
    if epoch < 25:
        tau = max(tau, tau_floor)

    return {
        'stage': stage,
        'target_ratio': target_ratio,
        'budget_weight': budget_weight,
        'tau': tau,
        'logits_diversity': logits_diversity,
    }


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
    # V2: Budget loss weight for gradient balancing
    config.training.budget_loss_weight = getattr(args, 'budget_loss_weight', 0.20)
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

    # V4: Splitter 独立学习率 - 先收集 splitter 参数 id，再从主参数组排除
    splitter_param_ids = set(id(p) for p in model.splitter.parameters() if p.requires_grad)
    backbone_params = [p for p in model.parameters() if id(p) not in splitter_param_ids]

    optimizer = optim.AdamW(
        backbone_params,
        lr=config.training.base_lr,
        weight_decay=config.training.weight_decay,
    )

    # V4: Splitter 独立学习率 - 添加专用参数组
    splitter_lr = config.training.base_lr * config.training.splitter_lr_multiplier
    optimizer.add_param_group({
        'params': [p for p in model.splitter.parameters() if p.requires_grad],
        'lr': splitter_lr,
        'name': 'splitter',
        'weight_decay': config.training.weight_decay,  # Splitter 也使用相同的 weight_decay
    })

    # V3: Safety Check - 验证优化器覆盖所有模型参数
    verify_optimizer_coverage(model, optimizer)

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

    # V4: Splitter 独立学习率 scheduler
    splitter_lr = config.training.base_lr * config.training.splitter_lr_multiplier
    splitter_scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=config.training.warmup_epochs,
        warmup_start_lr=config.training.warmup_start_lr * config.training.splitter_lr_multiplier,
        base_lr=splitter_lr,
        min_lr=config.training.min_lr * config.training.splitter_lr_multiplier,
        total_epochs=config.training.num_epochs,
        param_group_name='splitter',
    )

    # Resume scheduler state if available
    if state.scheduler_state:
        scheduler.load_state_dict(state.scheduler_state)
        # V4: 恢复 splitter scheduler 状态
        if hasattr(state, 'splitter_scheduler_state') and state.splitter_scheduler_state:
            splitter_scheduler.load_state_dict(state.splitter_scheduler_state)

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
    NumericalDefender(
        model=model,
        detect_anomaly=config.numerical.detect_anomaly,
        skip_on_nan=config.numerical.skip_on_nan_grad,
    )

    # Create logger
    logger = EpochLogger(output_dir=str(output_dir))

    # V3: 初始化 GradBalancer 和 ExplorationInjector
    grad_balancer = GradBalancer(
        beta=0.95,
        eta=0.1,
        budget_weight_target=config.training.budget_loss_weight,
        min_weight=0.01,
    )
    exploration_injector = ExplorationInjector(
        patience=3,
        tau_boost_factor=1.5,
        freeze_threshold=0.0001,
        exploration_threshold=0.05,
    )
    # V4: 梯度感知的探索注入
    grad_aware_injector = GradAwareExplorationInjector(
        patience=2,
        grad_drop_threshold=0.5,
    )

    # Move model to device
    model = model.to(device)

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

        # I-BUGFIX: 调用 splitter.set_epoch() 更新课程学习进度
        # 修复 K_min 钳制 Bug：_current_K 之前从未被更新，导致 avg_tokens 永远 = K_min = 8
        if hasattr(model, 'splitter') and hasattr(model.splitter, 'set_epoch'):
            model.splitter.set_epoch(epoch)

        # BPE-style 三阶段动态探索 Warmup
        # 更新 splitter 的温度、目标比例、budget 权重等参数
        warmup_params = _update_fractal_hyperparams(
            model=model,
            epoch=epoch,
            warmup_epochs=config.training.warmup_epochs,
            config=config,
        )

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
            splitter_scheduler=splitter_scheduler,  # V4: Splitter 独立 LR scheduler
            mixup_cutmix=mixup_cutmix,
            debug_dir=str(output_dir / "debug"),
            warmup_params=warmup_params,
            grad_balancer=grad_balancer,
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

            # V3: Exploration Injection - 检测活跃度冻结并临时提升 τ
            val_acc = eval_metrics.get("accuracy", 0.0)
            active_ratio = train_metrics.to_dict().get("active_ratio", 0.0)
            boosted_tau = exploration_injector.update(
                active_ratio=active_ratio,
                val_acc=val_acc,
                current_tau=warmup_params["tau"],
            )
            if boosted_tau != warmup_params["tau"]:
                print(f"  [ExplorationInjector] τ boosted: {warmup_params['tau']:.3f} -> {boosted_tau:.3f}")
                # 下个 epoch 会通过 set_temperature 应用（需要存储状态）
                # 临时修改 warmup_params 中的 tau 值供下次 _update_fractal_hyperparams 读取
                warmup_params["tau"] = boosted_tau

            # V4: Grad-Aware Exploration Injection - 基于 splitter 梯度的 τ 调整
            splitter_grad_norm = train_metrics.to_dict().get("splitter_grad_norm", 0.0)
            grad_aware_tau = grad_aware_injector.update(
                splitter_grad_norm=splitter_grad_norm,
                current_tau=warmup_params["tau"],
            )
            if grad_aware_tau != warmup_params["tau"]:
                print(f"  [GradAwareExplorationInjector] τ adjusted: {warmup_params['tau']:.3f} -> {grad_aware_tau:.3f}")
                warmup_params["tau"] = grad_aware_tau

        # V4: 构建 extra 日志字典，包含 BPE 内部变量
        extra_logs = {
            "bpe/current_stage": warmup_params.get("stage", 0),
            "bpe/target_ratio": warmup_params.get("target_ratio", 0.25),
            "bpe/budget_weight": warmup_params.get("budget_weight", 0.0),
            "splitter/alpha_actual": model.splitter.entmax_alpha if hasattr(model.splitter, 'entmax_alpha') else None,
            "splitter/temperature": warmup_params.get("tau", 1.0),
        }

        # Log epoch
        logger.log(
            epoch=epoch + 1,
            train_metrics=train_metrics.to_dict(),
            eval_metrics=eval_metrics,
            extra=extra_logs,  # V4: BPE 内部变量日志
        )

        # Update state
        state.optimizer_state = optimizer.state_dict()
        if scaler:
            state.scaler_state = scaler.state_dict()
        state.scheduler_state = scheduler.state_dict()
        state.splitter_scheduler_state = splitter_scheduler.state_dict()  # V4: Splitter 独立 LR

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
    # V2: Budget loss weight for gradient balancing (BPE regularization)
    parser.add_argument('--budget-loss-weight', type=float, default=0.20,
                        help='Budget loss weight for BPE regularization (default: 0.20)')
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

    # TorchDynamo 优化：增加缓存上限减少重编译导致的显存泄漏
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 64
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
