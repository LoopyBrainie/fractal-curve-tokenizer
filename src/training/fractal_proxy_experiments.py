# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Fractal Proxy Experiments - Three experiment modes

Mode A: Sparse Subset Ablation (SSA)
    - SubsetSampler extracts N classes from CUB-200
    - Grid search over sparsity_ratio x gumbel_tau
    - Record average token activation count

Mode B: Frozen-Encoder Reconstruction (FER)
    - Freeze Transformer Encoder
    - Train only Learnable Splitter and ShapeScaleEncoder
    - FeatureReconstructionLoss: Reconstruct ViT feature statistics

Mode C: Geometric Jigsaw Proxy (GJP)
    - Self-supervised: Randomly shuffle fractal hierarchy order
    - Predict relative Hilbert indices for fractal patches

Usage:
    # SSA grid search
    uv run python src/training/fractal_proxy_experiments.py --mode ssa \
        --sparsity-ratios 0.3,0.5,0.7 --gumbel-taus 0.5,1.0,2.0

    # FER (optional pretrained)
    uv run python src/training/fractal_proxy_experiments.py --mode fer \
        --pretrained path/to/checkpoint.pt

    # GJP
    uv run python src/training/fractal_proxy_experiments.py --mode gjp \
        --shuffle-group 4
"""

import sys
from pathlib import Path

# Add src directory to path (ensure vit_pytorch is importable)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
src_PATH = PROJECT_ROOT / "src"
if str(src_PATH) not in sys.path:
    sys.path.insert(0, str(src_PATH))

import argparse
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

logger = logging.getLogger(__name__)

# =============================================================================
# PART 1: Configuration and Constants
# =============================================================================

class ExperimentMode(Enum):
    """Three experiment modes"""
    SSA = "sparse_subset_ablation"    # Sparse subset ablation
    FER = "frozen_encoder_reconstruction"  # Frozen encoder reconstruction
    GJP = "geometric_jigsaw_proxy"    # Geometric jigsaw proxy

    @classmethod
    def from_string(cls, s: str) -> "ExperimentMode":
        """Support abbreviations: ssa -> SSA, fer -> FER, gjp -> GJP"""
        s = s.lower()
        mapping = {
            "ssa": cls.SSA,
            "fer": cls.FER,
            "gjp": cls.GJP,
            "sparse_subset_ablation": cls.SSA,
            "frozen_encoder_reconstruction": cls.FER,
            "geometric_jigsaw_proxy": cls.GJP,
        }
        if s not in mapping:
            raise ValueError(f"'{s}' is not a valid experiment mode. Valid options: {list(mapping.keys())}")
        return mapping[s]


@dataclass
class ProxyExperimentConfig:
    """Experiment config - Three-layer parameter structure

    L1 Data Config (Data):
        - batch_size: Batch size
        - num_workers: Data loader workers

    L2 Model Architecture Config (Model):
        - dim: Embedding dimension
        - num_layers: Transformer layers (original depth)
        - heads: Attention heads
        - min_patch_size: Min patch size
        - max_level: Max split depth (original max_fractal_depth)

    L3 Training Config (Training):
        - learning_rate: Learning rate
        - weight_decay: Weight decay
        - use_amp: Use mixed precision
        - gradient_accumulation_steps: Gradient accumulation steps
    """
    # L1: Data config
    mode: ExperimentMode = ExperimentMode.SSA
    experiment_name: str = "default_experiment"
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 4

    # L2: Model architecture config
    dim: int = 384
    num_layers: int = 8  # Original depth (I145: unified naming)
    heads: int = 6
    min_patch_size: int = 4
    max_level: int = 8  # Original max_fractal_depth

    # L3: Training config
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    use_amp: bool = True
    use_compile: bool = False
    gradient_accumulation_steps: int = 4
    log_interval: int = 10  # Logging interval for batches

    # Runtime config (not in three-layer params)
    device: str = "cuda"

    # Model path
    pretrained_path: Optional[str] = None

    # Experiment-specific config
    ssa_config: Optional["SSAConfig"] = None
    fer_config: Optional["FERConfig"] = None
    gjp_config: Optional["GJPConfig"] = None


@dataclass
class SSAConfig:
    """Sparse Subset Ablation config"""
    num_classes_subset: int = 10  # Extract N classes
    sparsity_ratios: Tuple[float, ...] = (0.3, 0.5, 0.7)
    gumbel_taus: Tuple[float, ...] = (0.5, 1.0, 2.0)

    # Record metrics
    record_per_layer_tokens: bool = True
    record_token_density: bool = True
    record_depth_variance: bool = True

    # Quick mode
    quick_test: bool = False  # Reduce iterations for quick validation


@dataclass
class FERConfig:
    """Frozen Encoder Reconstruction config

    Note:
        - learning_rate and weight_decay inherited from ProxyExperimentConfig
        - Avoid duplication, maintain three-layer parameter consistency
    """
    # Reconstruction targets
    reconstruct_mean: bool = True
    reconstruct_var: bool = True
    reconstruct_channel_corr: bool = True

    # Loss weights
    mean_weight: float = 1.0
    var_weight: float = 1.0
    corr_weight: float = 0.5

    # Training config (use ProxyExperimentConfig's learning_rate and weight_decay)
    epochs: int = 10  # Original frozen_encoder_epochs

    # ShapeScaleEncoder config
    shape_encoder_names: Tuple[str, ...] = (
        'shape_scale_encoder', 'shape_encoder', 'scale_encoder'
    )


@dataclass
class GJPConfig:
    """Geometric Jigsaw Proxy config

    Note:
        - learning_rate and weight_decay inherited from ProxyExperimentConfig
        - Avoid duplication, maintain three-layer parameter consistency
    """
    shuffle_group_size: int = 4  # Shuffle group size

    # Hilbert index related
    max_relative_distance: int = 8  # Max relative distance
    predict_relative_offset: bool = True

    # Loss weights
    jigsaw_weight: float = 1.0
    entropy_weight: float = 0.1

    # Training config (use ProxyExperimentConfig's learning_rate and weight_decay)
    epochs: int = 10  # Original num_epochs

    # Quick mode
    quick_test: bool = False  # Reduce iterations for quick validation


# =============================================================================
# PART 2: Basic Architecture
# =============================================================================

@dataclass
class TokenMetrics:
    """Token metrics data structure"""
    # Core metrics (must record)
    token_density: float  # Active token ratio per layer
    recursive_depth_variance: float  # Depth variance

    # Extended metrics
    num_tokens_per_depth: Dict[int, int] = field(default_factory=dict)
    avg_tokens_per_sample: float = 0.0
    hilbert_locality_score: float = 0.0
    batch_size: int = 1


class TokenMetricsCollector:
    """Token metrics collector - extract from TrainingStats"""

    def __init__(self, max_depth: int = 8):
        self.max_depth = max_depth

    def compute_metrics(
        self,
        stats,
        num_total_patches: int,
        batch_size: int = 1,
    ) -> TokenMetrics:
        """Compute metrics from TrainingStats (I139: adapt new interface)"""

        # I139: Support TrainingStats or legacy aux_infos dict
        # I141: num_tokens can be int, List[int], or torch.Tensor
        if hasattr(stats, 'num_tokens'):
            # TrainingStats mode
            num_tokens = stats.num_tokens
            if isinstance(num_tokens, torch.Tensor):
                # I141: GPU tensor to scalar
                num_tokens = num_tokens.sum().item()  # Total tokens in batch
            elif isinstance(num_tokens, list):
                # I141: List type, sum
                num_tokens = sum(num_tokens)  # Total tokens in batch

            # Get levels_used from split_info
            split_info = getattr(stats, 'split_info', {})
            levels_list = split_info.get('levels_list', None)

            # Compute depth variance
            if levels_list:
                all_levels = []
                for levels in levels_list:
                    if levels.numel() > 0:
                        all_levels.extend(levels[:, 0].cpu().tolist())
                depth_variance = self._compute_depth_variance(all_levels) if all_levels else 0.0
            else:
                depth_variance = 0.0

            # Depth distribution
            depth_dist = getattr(stats, 'depth_distribution', {})
            if isinstance(depth_dist, dict):
                num_tokens_per_depth = {k: int(v * num_tokens) for k, v in depth_dist.items()}
            else:
                num_tokens_per_depth = {}
        else:
            # Legacy aux_infos dict mode (backward compatible)
            aux_infos = stats
            num_tokens = aux_infos.get("num_tokens", 0)
            if isinstance(num_tokens, torch.Tensor):
                num_tokens = num_tokens.item()

            levels_used = aux_infos.get("levels_used", None)
            if levels_used is not None:
                if isinstance(levels_used, torch.Tensor):
                    levels_used = levels_used.cpu().tolist()
                depth_variance = self._compute_depth_variance(levels_used)
            else:
                depth_variance = 0.0

            levels_distribution = aux_infos.get("levels_distribution", None)
            if levels_distribution is not None:
                if isinstance(levels_distribution, torch.Tensor):
                    levels_distribution = levels_distribution.cpu().tolist()
                num_tokens_per_depth = {
                    i: levels_distribution[i] for i in range(len(levels_distribution))
                }
            else:
                num_tokens_per_depth = {}

        # Token Density: active token / total patch
        token_density = num_tokens / num_total_patches if num_total_patches > 0 else 0.0

        return TokenMetrics(
            token_density=token_density,
            recursive_depth_variance=depth_variance,
            num_tokens_per_depth=num_tokens_per_depth,
            avg_tokens_per_sample=num_tokens / batch_size if batch_size > 0 else 0.0,
            hilbert_locality_score=0.0,  # TrainingStats does not provide this field
            batch_size=batch_size,
        )

    def _compute_depth_variance(self, levels: List[int]) -> float:
        """Compute depth variance"""
        if not levels:
            return 0.0

        levels_tensor = torch.tensor(levels, dtype=torch.float32)
        return levels_tensor.var().item()

    def aggregate_metrics(self, metrics_list: List[TokenMetrics]) -> Dict[str, float]:
        """Aggregate metrics from multiple batches"""
        if not metrics_list:
            return {}

        return {
            "token_density_mean": sum(m.token_density for m in metrics_list) / len(metrics_list),
            "token_density_std": torch.tensor([m.token_density for m in metrics_list]).std().item(),
            "depth_variance_mean": sum(m.recursive_depth_variance for m in metrics_list) / len(metrics_list),
            "depth_variance_std": torch.tensor([m.recursive_depth_variance for m in metrics_list]).std().item(),
            "avg_tokens_per_sample": sum(m.avg_tokens_per_sample for m in metrics_list) / len(metrics_list),
        }


class MemoryManager:
    """Memory management - satisfy engineering constraints"""

    def __init__(self, config: ProxyExperimentConfig):
        self.config = config
        self.accum_steps = config.gradient_accumulation_steps
        self._last_reserved: Optional[float] = None

    def maybe_clear_cache(self, epoch: int, force: bool = False) -> None:
        """
        Call torch.cuda.empty_cache() at end of epoch
        Also called after Graph Break warnings
        """
        if torch.cuda.is_available():
            # Clear at end of each epoch, or force clear
            if force or epoch % 1 == 0:
                torch.cuda.empty_cache()

    def check_memory_leak(self) -> Dict[str, float]:
        """
        Check memory leak - detect non-detached tensors
        Use VectorizationAuditor pattern
        """
        if not torch.cuda.is_available():
            return {}

        memory_stats = {
            "allocated_GB": torch.cuda.memory_allocated() / 1024**3,
            "reserved_GB": torch.cuda.memory_reserved() / 1024**3,
        }

        # Warning: if reserved keeps growing, there's a leak
        if self._last_reserved is not None:
            growth_ratio = memory_stats["reserved_GB"] / self._last_reserved
            if growth_ratio > 1.1:  # Growth exceeds 10%
                logger.warning(
                    f"Potential memory leak: reserved increased from "
                    f"{self._last_reserved:.2f}GB to {memory_stats['reserved_GB']:.2f}GB "
                    f"(ratio: {growth_ratio:.2f})"
                )
        self._last_reserved = memory_stats["reserved_GB"]

        return memory_stats

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory status"""
        if not torch.cuda.is_available():
            return {}

        return {
            "allocated_GB": torch.cuda.memory_allocated() / 1024**3,
            "reserved_GB": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated_GB": torch.cuda.max_memory_allocated() / 1024**3,
        }


class ExperimentWrapper:
    """
    Experiment wrapper - decoupled design, no modification to core model library

    Encapsulate model via dependency injection and attribute access, supports:
    - Freeze/unfreeze modules
    - Set Gumbel temperature
    - Set sparsity ratio
    - Auto-detect ShapeScaleEncoder
    """

    def __init__(
        self,
        model: nn.Module,
        config: ProxyExperimentConfig,
        metrics_collector: TokenMetricsCollector,
    ):
        self.model = model
        self.config = config
        self.metrics_collector = metrics_collector
        self.device = torch.device(config.device)

        # Extract tokenizer and splitter
        self.tokenizer = getattr(model, 'tokenizer', None)
        self.splitter = getattr(self.tokenizer, 'splitter', None) if self.tokenizer else None

        # Auto-detect ShapeScaleEncoder (FER mode)
        self.shape_scale_encoder: Optional[nn.Module] = None
        if config.mode == ExperimentMode.FER:
            self._detect_shape_encoder()

        # Move to device
        self.model = self.model.to(self.device)

    def _detect_shape_encoder(self) -> None:
        """Auto-detect ShapeScaleEncoder (multiple candidate names)"""
        if self.tokenizer is None:
            return

        fer_config = self.config.fer_config or FERConfig()
        for name in fer_config.shape_encoder_names:
            encoder = getattr(self.tokenizer, name, None)
            if encoder is not None:
                self.shape_scale_encoder = encoder
                logger.info(f"Detected ShapeScaleEncoder: {name}")
                break

        if self.shape_scale_encoder is None:
            logger.warning("ShapeScaleEncoder not found in tokenizer")

    def wrap_model_for_experiment(self) -> nn.Module:
        """Wrap model for experiment needs"""
        if self.config.mode == ExperimentMode.FER:
            self._freeze_encoder()
        return self.model

    def unwrap_model(self) -> nn.Module:
        """Restore model to original state"""
        if self.config.mode == ExperimentMode.FER:
            self._unfreeze_encoder()
        return self.model

    def _freeze_encoder(self) -> None:
        """Freeze Transformer Encoder"""
        encoder = getattr(self.model, 'transformer', None)
        if encoder is not None:
            for param in encoder.parameters():
                param.requires_grad = False
            logger.info("Transformer Encoder frozen")

            # Ensure tokenizer and splitter are trainable
            if self.tokenizer:
                for param in self.tokenizer.parameters():
                    param.requires_grad = True
            if self.splitter:
                for param in self.splitter.parameters():
                    param.requires_grad = True
            if self.shape_scale_encoder:
                for param in self.shape_scale_encoder.parameters():
                    param.requires_grad = True

    def _unfreeze_encoder(self) -> None:
        """Unfreeze Encoder (after FER pretraining)"""
        encoder = getattr(self.model, 'transformer', None)
        if encoder is not None:
            for param in encoder.parameters():
                param.requires_grad = True
            logger.info("Transformer Encoder unfrozen")

    def set_gumbel_temperature(self, tau: float) -> None:
        """Set Gumbel temperature (SSA experiment)"""
        if self.splitter is not None and hasattr(self.splitter, 'log_temperature'):
            self.splitter.log_temperature.data = torch.tensor(tau).log().to(self.device)
            logger.debug(f"Gumbel temperature set to {tau}")

    def set_sparsity_ratio(self, ratio: float) -> None:
        """Set sparsity ratio (modify token_coverage_max)

        I33: 使用相对预算 API 替代绝对 K 值
        公式: token_coverage_max = base_ratio * ratio
        """
        if self.splitter is not None:
            # I33: 从 config 获取基准覆盖率
            base_coverage = getattr(self.splitter, '_token_coverage_max', 0.25)
            if hasattr(self.splitter, 'config') and self.splitter.config is not None:
                base_coverage = getattr(self.splitter.config, 'token_coverage_max', 0.25)

            # 根据稀疏比率调整覆盖率
            new_coverage = base_coverage * ratio
            # 确保覆盖率在合理范围内 [0.01, 0.5]
            new_coverage = max(0.01, min(0.5, new_coverage))

            # 更新 config 中的 token_coverage_max
            if hasattr(self.splitter, 'config') and self.splitter.config is not None:
                self.splitter.config.token_coverage_max = new_coverage
            else:
                # 备选：直接更新内部变量
                if hasattr(self.splitter, '_token_coverage_max'):
                    self.splitter._token_coverage_max = new_coverage

            logger.debug(f"Sparsity ratio {ratio} -> token_coverage_max = {new_coverage:.4f}")

    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """Get trainable parameter list (FER mode)"""
        params = []

        if self.splitter is not None:
            params.extend(p for p in self.splitter.parameters() if p.requires_grad)

        if self.shape_scale_encoder is not None:
            params.extend(p for p in self.shape_scale_encoder.parameters() if p.requires_grad)

        # If no modules specified, return all parameters
        if not params:
            params = list(self.model.parameters())

        return params


class BaseExperiment(ABC):
    """Experiment base class"""

    def __init__(
        self,
        model: nn.Module,
        config: ProxyExperimentConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.metrics_collector = TokenMetricsCollector(
            max_depth=config.max_level  # 原 max_fractal_depth
        )

        self.wrapper = ExperimentWrapper(
            model=model,
            config=config,
            metrics_collector=self.metrics_collector,
        )

        self.memory_manager = MemoryManager(config)
        self.device = torch.device(config.device)

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """运行实验，必须由子类实现"""
        pass

    def _handle_compile_warning(self, e: Exception) -> None:
        """处理 torch.compile 的 Graph Break 警告"""
        logger.warning(
            f"torch.compile Graph Break detected: {e}. "
            "Consider optimizing data flow or disabling compile."
        )
        # 尝试优化数据流
        if hasattr(self, '_optimize_dataflow'):
            self._optimize_dataflow()

    def _optimize_dataflow(self) -> None:
        """优化数据流以减少 Graph Break"""
        # 确保输入是 contiguous
        if hasattr(self, '_input_buffer'):
            self._input_buffer = self._input_buffer.contiguous()


# =============================================================================
# PART 3: Mode A - SSA (Sparse Subset Ablation)
# =============================================================================

class SubsetSampler(Sampler):
    """
    Extract N classes from CUB-200 - streaming sampling

    Prohibit full load: use generator pattern, support large datasets
    """

    def __init__(
        self,
        dataset: Dataset,
        num_classes: int = 10,
        class_ids: Optional[List[int]] = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.num_classes = num_classes
        self.shuffle = shuffle

        # 如果未指定，随机选择 N 个类
        if class_ids is None:
            all_classes = list(range(200))  # CUB-200 有 200 类
            rng = torch.Generator().manual_seed(seed)
            self.class_ids = sorted(
                torch.randperm(len(all_classes), generator=rng)[:num_classes].tolist()
            )
        else:
            self.class_ids = sorted(class_ids)

        # 建立类到样本索引的映射
        self._build_class_to_indices()

        # 生成采样顺序
        self._sample_order = self._generate_sample_order()

    def _build_class_to_indices(self) -> None:
        """构建类到样本索引的映射 (延迟构建以支持流式)"""
        self.class_to_indices: Dict[int, List[int]] = {
            c: [] for c in self.class_ids
        }

        # 流式遍历数据集
        for idx in range(len(self.dataset)):
            try:
                _, label = self.dataset[idx]
                if label in self.class_ids:
                    self.class_to_indices[label].append(idx)
            except Exception:
                # 某些数据集可能不支持索引访问
                continue

    def _generate_sample_order(self) -> List[int]:
        """生成采样顺序"""
        indices = []
        for class_id in self.class_ids:
            class_indices = self.class_to_indices.get(class_id, [])
            if self.shuffle:
                rng = torch.Generator().manual_seed(
                    self.config.seed if hasattr(self, 'config') else 42
                )
                perm = torch.randperm(len(class_indices), generator=rng)
                class_indices = [class_indices[i] for i in perm.tolist()]
            indices.extend(class_indices)
        return indices

    def __len__(self) -> int:
        return len(self._sample_order)

    def __iter__(self):
        """生成器模式 - 禁止全量加载"""
        return iter(self._sample_order)


class SparseAblationExperiment(BaseExperiment):
    """
    Mode A: Sparse Subset Ablation experiment

    Objective: test sparsity_ratio and gumbel_tau impact on token activation
    """

    def __init__(
        self,
        model: nn.Module,
        config: ProxyExperimentConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        super().__init__(model, config, train_loader, val_loader)

        self.ssa_config = config.ssa_config or SSAConfig()

        # 确保模型已包装
        self.wrapper.wrap_model_for_experiment()

    def run(self) -> Dict[Tuple[float, float], Dict[str, float]]:
        """
        Grid search: iterate over sparsity_ratio and gumbel_tau

        Returns:
            Dict[(sparsity, tau)] -> Dict[metric] average results
        """
        results = {}

        for sparsity in self.ssa_config.sparsity_ratios:
            for tau in self.ssa_config.gumbel_taus:
                key = (sparsity, tau)
                logger.info(f"Running SSA: sparsity={sparsity}, tau={tau}")

                # Set experiment parameters
                self.wrapper.set_sparsity_ratio(sparsity)
                self.wrapper.set_gumbel_temperature(tau)

                # Run single config experiment
                run_results = self._run_single_config()
                results[key] = run_results

                # Clear memory
                self.memory_manager.maybe_clear_cache(0, force=True)

                if self.ssa_config.quick_test:
                    break  # Quick test mode runs only one config
            if self.ssa_config.quick_test:
                break

        return results

    def _run_single_config(self) -> Dict[str, float]:
        """Run single config experiment"""
        metrics_history = []

        max_iter = 10 if self.ssa_config.quick_test else len(self.train_loader)

        for batch_idx, (imgs, labels) in enumerate(self.train_loader):
            if batch_idx >= max_iter:
                break

            imgs = imgs.to(self.device)

            # Forward - disable torch.compile entirely for stability
            try:
                # Use inference_mode to disable gradients and compile
                with torch.inference_mode():
                    stats = self.wrapper.model(imgs)
            except Exception as e:
                self._handle_compile_warning(e)
                continue

            # I139: Extract from TrainingStats logits
            logits = stats.logits if hasattr(stats, 'logits') else stats

            # Collect metrics
            img_size = imgs.shape[2:]
            patch_size = 4  # 默认 patch size
            num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

            metrics = self.metrics_collector.compute_metrics(
                stats,
                num_total_patches=num_patches,
                batch_size=imgs.shape[0],
            )

            metrics_history.append({
                "token_density": metrics.token_density,
                "depth_variance": metrics.recursive_depth_variance,
                "avg_tokens": metrics.avg_tokens_per_sample,
            })

            if batch_idx % self.config.log_interval == 0:
                logger.info(
                    f"  [Batch {batch_idx}] "
                    f"Token_Density={metrics.token_density:.3f}, "
                    f"Recursive_Depth_Variance={metrics.recursive_depth_variance:.3f}, "
                    f"Avg_Tokens={metrics.avg_tokens_per_sample:.1f}"
                )

        # 聚合结果
        if not metrics_history:
            return {"token_density": 0.0, "depth_variance": 0.0, "avg_tokens": 0.0}

        aggregated = self.metrics_collector.aggregate_metrics([
            TokenMetrics(
                token_density=m["token_density"],
                recursive_depth_variance=m["depth_variance"],
                avg_tokens_per_sample=m["avg_tokens"],
            )
            for m in metrics_history
        ])

        return {
            "token_density": aggregated.get("token_density_mean", 0.0),
            "depth_variance": aggregated.get("depth_variance_mean", 0.0),
            "avg_tokens": aggregated.get("avg_tokens_per_sample", 0.0),
        }


# =============================================================================
# PART 4: Mode B - FER (Frozen Encoder Reconstruction)
# =============================================================================

class FeatureReconstructionLoss(nn.Module):
    """
    Reconstruct ViT feature statistics

    数学形式化:
        L_fer = λ_mean × L_mean + λ_var × L_var + λ_corr × L_corr

    其中:
        L_mean = ||μ_recon - μ_original||²
        L_var = ||σ²_recon - σ²_original||²
        L_corr = ||Σ_recon - Σ_original||_F (Frobenius 范数)
    """

    def __init__(self, config: FERConfig, feat_dim: int = 384):
        super().__init__()
        self.config = config
        self.feat_dim = feat_dim

    def forward(
        self,
        original_features: torch.Tensor,
        reconstructed_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute reconstruction loss

        Args:
            original_features: [B, N, D] Original ViT features
            reconstructed_features: [B, N, D] Reconstructed features

        Returns:
            loss: Total loss
            info: Component statistics
        """
        info = {}

        # Ensure features are contiguous
        original_features = original_features.contiguous()
        reconstructed_features = reconstructed_features.contiguous()

        # 1. Mean reconstruction loss
        if self.config.reconstruct_mean:
            original_mean = original_features.mean(dim=(0, 1))  # [D]
            recon_mean = reconstructed_features.mean(dim=(0, 1))
            mean_loss = F.mse_loss(recon_mean, original_mean, reduction='mean')
            info["mean_loss"] = mean_loss.item()
        else:
            mean_loss = torch.tensor(0.0, device=original_features.device)

        # 2. Variance reconstruction loss
        if self.config.reconstruct_var:
            original_var = original_features.var(dim=(0, 1))  # [D]
            recon_var = reconstructed_features.var(dim=(0, 1))
            var_loss = F.mse_loss(recon_var, original_var, reduction='mean')
            info["var_loss"] = var_loss.item()
        else:
            var_loss = torch.tensor(0.0, device=original_features.device)

        # 3. Channel correlation reconstruction loss
        if self.config.reconstruct_channel_corr:
            original_flat = original_features.flatten(0, 1)  # [B*N, D]
            recon_flat = reconstructed_features.flatten(0, 1)

            # Center
            original_flat = original_flat - original_flat.mean(dim=0, keepdim=True)
            recon_flat = recon_flat - recon_flat.mean(dim=0, keepdim=True)

            # Compute covariance matrix
            # cov = (X^T X) / (n - 1)
            n = original_flat.shape[0]
            original_cov = (original_flat.mT @ original_flat) / (n - 1 + 1e-6)
            recon_cov = (recon_flat.mT @ recon_flat) / (n - 1 + 1e-6)

            corr_loss = F.mse_loss(recon_cov, original_cov)
            info["corr_loss"] = corr_loss.item()
        else:
            corr_loss = torch.tensor(0.0, device=original_features.device)

        # Total loss
        total_loss = (
            self.config.mean_weight * mean_loss +
            self.config.var_weight * var_loss +
            self.config.corr_weight * corr_loss
        )

        info["total_loss"] = total_loss.item()

        return total_loss, info


class FrozenEncoderExperiment(BaseExperiment):
    """
    Mode B: Frozen Encoder Reconstruction experiment

    Objective: freeze Encoder, train only Splitter + ShapeScaleEncoder
    """

    def __init__(
        self,
        model: nn.Module,
        config: ProxyExperimentConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        reference_model: Optional[nn.Module] = None,
    ):
        super().__init__(model, config, train_loader, val_loader)

        self.fer_config = config.fer_config or FERConfig()
        self.reference_model = reference_model

        # Wrap model（会冻结 encoder）
        self.wrapper.wrap_model_for_experiment()

        # 损失函数
        feat_dim = getattr(config, 'dim', 384)
        self.recon_loss_fn = FeatureReconstructionLoss(
            config=self.fer_config,
            feat_dim=feat_dim,
        )

        # Optimizer (只训练 Splitter 和 ShapeScaleEncoder)
        self.optimizer = self._create_optimizer()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer - optimize only Splitter and ShapeScaleEncoder"""
        params = self.wrapper.get_trainable_params()

        if not params:
            logger.warning("No trainable parameters found")
            return torch.optim.Adam([])  # 空参数

        logger.info(f"FER: 找到 {len(params)} 组可训练参数")

        return torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,  # 从 ProxyExperimentConfig 继承
            weight_decay=self.config.weight_decay,
        )

    def run(self) -> Dict[str, Any]:
        """Run FER experiment"""
        history = {
            "recon_loss": [],
            "token_density": [],
            "depth_variance": [],
            "memory_stats": [],
        }

        self.model.train()

        num_epochs = self.fer_config.epochs  # 原 frozen_encoder_epochs
        max_iter = 10 if getattr(self.fer_config, 'quick_test', False) else None

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            epoch_metrics = []

            for batch_idx, (imgs, _) in enumerate(self.train_loader):
                if max_iter and batch_idx >= max_iter:
                    break

                imgs = imgs.to(self.device)

                # 1. 获取Original ViT features (冻结的 encoder)
                with torch.no_grad():
                    if self.reference_model is not None:
                        original_stats = self.reference_model(imgs)
                    else:
                        # Use current model but encoder frozen
                        original_stats = self.wrapper.model(imgs)

                    # Extract original features
                    original_features = original_stats.transformer_tokens
                    if original_features is None:
                        original_features = original_stats.features

                    if original_features is None:
                        logger.warning("无法Extract original features，跳过 batch")
                        continue

                # 2. Forward pass (train splitter)
                try:
                    if self.config.use_compile:
                        # 使用 torch._dynamo.disable 避免 graph break
                        stats = torch._dynamo.disable(self.wrapper.model)(imgs)
                    else:
                        stats = self.wrapper.model(imgs)
                except Exception as e:
                    self._handle_compile_warning(e)
                    continue

                # I139: Extract from TrainingStats
                # I139: FER 需要使用 transformer_tokens 作为 reconstructed features
                recon_features = getattr(stats, 'transformer_tokens', None)
                if recon_features is None:
                    recon_features = getattr(stats, 'features', None)

                if recon_features is None:
                    logger.debug(f"TrainingStats 可用字段: {[k for k in dir(stats) if not k.startswith('_')]}")
                    logger.warning("无法获取 reconstructed features，跳过 batch")
                    continue

                # 4. Compute reconstruction loss
                loss, info = self.recon_loss_fn(original_features, recon_features)

                # 5. Gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

                # 6. Backward pass
                loss.backward()

                # 7. Gradient accumulation步骤
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_loss += info["total_loss"]
                num_batches += 1

                # Collect metrics
                img_size = imgs.shape[2:]
                patch_size = 4
                num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

                metrics = self.metrics_collector.compute_metrics(
                    stats,
                    num_total_patches=num_patches,
                    batch_size=imgs.shape[0],
                )
                epoch_metrics.append(metrics)

            if num_batches == 0:
                logger.warning(f"Epoch {epoch+1} 没有有效 batch")
                continue

            avg_loss = epoch_loss / num_batches
            aggregated = self.metrics_collector.aggregate_metrics(epoch_metrics)

            history["recon_loss"].append(avg_loss)
            history["token_density"].append(aggregated.get("token_density_mean", 0.0))
            history["depth_variance"].append(aggregated.get("depth_variance_mean", 0.0))
            history["memory_stats"].append(self.memory_manager.get_memory_stats())

            logger.info(
                f"FER Epoch {epoch+1}/{num_epochs}: "
                f"loss={avg_loss:.4f}, "
                f"Token_Density={aggregated.get('token_density_mean', 0):.3f}, "
                f"Recursive_Depth_Variance={aggregated.get('depth_variance_mean', 0):.3f}"
            )

            # 显存清理
            self.memory_manager.maybe_clear_cache(epoch, force=True)

        return history


# =============================================================================
# PART 5: Mode C - GJP (Geometric Jigsaw Proxy)
# =============================================================================

class HilbertIndexShuffler:
    """Hilbert index shuffler - for geometric jigsaw task"""

    def __init__(self, group_size: int = 4, max_depth: int = 8):
        self.group_size = group_size
        self.max_depth = max_depth

    def shuffle_within_groups(
        self,
        levels_info_data: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Shuffle topology order within fractal hierarchy

        数学形式化:
            For each group of G tokens, randomly permute order
            Keep hierarchy structure, only change Hilbert index order

        Args:
            levels_info_data: [B, N, D+1] Depth + path info
            batch_indices: [N] Batch index for each token

        Returns:
            shuffled_data: Shuffled levels_info
            shuffle_indices: Original -> shuffled mapping
        """
        B, N, D = levels_info_data.shape
        device = levels_info_data.device

        shuffled_data = levels_info_data.clone()
        shuffle_indices = torch.arange(N, device=device)

        # Process by batch
        for b in range(B):
            batch_mask = batch_indices == b
            batch_positions = torch.where(batch_mask)[0]

            if len(batch_positions) >= self.group_size:
                # Group shuffle
                num_groups = len(batch_positions) // self.group_size

                for g in range(num_groups):
                    start = g * self.group_size
                    end = start + self.group_size
                    group = batch_positions[start:end]

                    # Randomly shuffle within group
                    perm = torch.randperm(self.group_size, device=device)
                    shuffled_data[b, group] = levels_info_data[b, group[perm]]

        return shuffled_data, shuffle_indices

    @staticmethod
    def compute_hilbert_indices(
        depths: torch.Tensor,
        paths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Hilbert index from depths and paths

        Args:
            depths: [B, N] Depth values
            paths: [B, N, D] Quadrant path

        Returns:
            hilbert_indices: [B, N] Hilbert curve index
        """
        B, N, D = paths.shape

        hilbert_indices = torch.zeros(B, N, dtype=torch.long, device=paths.device)

        for b in range(B):
            for n in range(N):
                d = int(depths[b, n])
                if d >= 0:
                    # 从路径计算 Hilbert 索引
                    path_int = 0
                    for t in range(min(d, D)):
                        quadrant = int(paths[b, n, t])
                        path_int = path_int * 4 + quadrant
                    hilbert_indices[b, n] = path_int

        return hilbert_indices


class GeometricJigsawLoss(nn.Module):
    """
    Geometric jigsaw loss - predict relative Hilbert index

    数学形式化:
        L_jigsaw = -Σ_i Σ_j 1[|h_i - h_j| < d_max] × log P(offset_ij)

    其中:
        h_i, h_j: token i, j 的 Hilbert 索引
        d_max: 最大相对距离阈值
        P(offset): 相对偏移的预测概率
    """

    def __init__(self, config: GJPConfig, feat_dim: int = 384):
        super().__init__()
        self.config = config
        self.feat_dim = feat_dim
        self.max_offset = config.max_relative_distance

        # 预测头：Predict relative offset category
        self.offset_predictor = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, config.max_relative_distance + 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,  # [B, N, D]
        levels_info: torch.Tensor,  # [B, N, D+1]
        original_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute jigsaw loss

        Args:
            tokens: [B, N, D] Fractal token features
            levels_info: [B, N, D+1] Depth + path info
            original_indices: Original Hilbert index (用于Compute relative offset)

        Returns:
            loss: 拼图损失
            info: 统计信息
        """
        info = {}
        B, N, D = tokens.shape

        if N < 2:
            info["jigsaw_loss"] = 0.0
            return torch.tensor(0.0, device=tokens.device), info

        # 确保连续
        tokens = tokens.contiguous()
        levels_info = levels_info.contiguous()

        # 确保 offset_predictor 在正确设备上
        self.offset_predictor = self.offset_predictor.to(tokens.device)

        # 1. 计算 Hilbert 索引
        if original_indices is None:
            depths = levels_info[:, :, 0]  # [B, N]
            paths = levels_info[:, :, 1:]  # [B, N, D]
            hilbert_indices = HilbertIndexShuffler.compute_hilbert_indices(depths, paths)
        else:
            hilbert_indices = original_indices

        # 2. Compute relative offset (分类任务)
        target_pairs = []
        offset_labels = []

        for b in range(B):
            for i in range(N):
                for j in range(i + 1, min(i + self.max_offset * 2 + 1, N)):
                    # Only consider tokens of same level
                    if levels_info[b, i, 0] == levels_info[b, j, 0]:
                        offset = abs(int(hilbert_indices[b, i]) - int(hilbert_indices[b, j]))
                        if offset <= self.max_offset:
                            target_pairs.append((b, i, j))
                            offset_labels.append(min(offset, self.max_offset))

        if not target_pairs:
            info["jigsaw_loss"] = 0.0
            info["num_pairs"] = 0
            return torch.tensor(0.0, device=tokens.device), info

        # 3. Build contrastive features
        pair_features = []
        for b, i, j in target_pairs:
            pair_feat = torch.cat([tokens[b, i], tokens[b, j]], dim=-1)
            pair_features.append(pair_feat)

        pair_features = torch.stack(pair_features)  # [P, 2*D]

        # 4. Predict offset
        offset_logits = self.offset_predictor(pair_features)  # [P, max_offset+1]

        # 5. Classification loss
        offset_labels_tensor = torch.tensor(offset_labels, device=tokens.device)
        jigsaw_loss = F.cross_entropy(offset_logits, offset_labels_tensor)

        info["jigsaw_loss"] = jigsaw_loss.item()
        info["num_pairs"] = len(target_pairs)
        info["accuracy"] = (offset_logits.argmax(dim=-1) == offset_labels_tensor).float().mean().item()

        # 6. Entropy regularization
        if self.config.entropy_weight > 0:
            probs = F.softmax(offset_logits, dim=-1)
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
            jigsaw_loss = jigsaw_loss + self.config.entropy_weight * entropy
            info["entropy"] = entropy.item()

        return jigsaw_loss, info


class GeometricJigsawExperiment(BaseExperiment):
    """
    Mode C: Geometric Jigsaw Proxy experiment

    Objective: self-supervised - predict relative Hilbert index for fractal patches
    """

    def __init__(
        self,
        model: nn.Module,
        config: ProxyExperimentConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        super().__init__(model, config, train_loader, val_loader)

        self.gjp_config = config.gjp_config or GJPConfig()

        # Wrap model
        self.wrapper.wrap_model_for_experiment()

        # Components
        self.shuffler = HilbertIndexShuffler(
            group_size=self.gjp_config.shuffle_group_size,
            max_depth=config.max_level,  # 原 max_fractal_depth
        )

        feat_dim = getattr(config, 'dim', 384)
        self.jigsaw_loss_fn = GeometricJigsawLoss(
            config=self.gjp_config,
            feat_dim=feat_dim,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,  # 从 ProxyExperimentConfig 继承
            weight_decay=self.config.weight_decay,
        )

    def run(self) -> Dict[str, Any]:
        """Run GJP experiment"""
        history = {
            "jigsaw_loss": [],
            "token_density": [],
            "depth_variance": [],
            "accuracy": [],
        }

        self.model.train()

        num_epochs = self.gjp_config.epochs  # 原 num_epochs
        max_iter = 10 if getattr(self.gjp_config, 'quick_test', False) else None

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0
            epoch_metrics = []

            for batch_idx, (imgs, _) in enumerate(self.train_loader):
                if max_iter and batch_idx >= max_iter:
                    break

                imgs = imgs.to(self.device)

                # 1. Forward pass
                try:
                    if self.config.use_compile:
                        # 使用 torch._dynamo.disable 避免 graph break
                        stats = torch._dynamo.disable(self.wrapper.model)(imgs)
                    else:
                        stats = self.wrapper.model(imgs)
                except Exception as e:
                    self._handle_compile_warning(e)
                    continue

                # I139: Extract transformer tokens and levels_info
                transformer_tokens = getattr(stats, 'transformer_tokens', None)
                if transformer_tokens is None:
                    logger.debug(f"TrainingStats 可用字段: {[k for k in dir(stats) if not k.startswith('_')]}")
                    logger.warning("GJP mode: transformer_tokens unavailable, skipping batch")
                    continue

                # Extract levels_info from split_info
                split_info = getattr(stats, 'split_info', {})
                levels_list = split_info.get('levels_list', None)
                if levels_list is None or len(levels_list) == 0:
                    logger.warning("GJP mode: levels_list unavailable, skipping batch")
                    continue

                # Stack levels_list to create levels_info [B, N, D+1]
                # levels_list is a list of tensors, each [N, D+1] where first col is depth
                # Ensure tensors are on the same device as tokens
                levels_info = torch.stack(
                    [l.to(transformer_tokens.device) for l in levels_list], dim=0
                )  # [B, N, D+1]

                # 3. Compute jigsaw loss
                loss, info = self.jigsaw_loss_fn(
                    tokens=transformer_tokens,
                    levels_info=levels_info,
                )

                # 4. Gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

                # 5. Backward pass
                loss.backward()

                # 6. Gradient accumulation步骤
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_loss += info.get("jigsaw_loss", 0.0)
                epoch_acc += info.get("accuracy", 0.0)
                num_batches += 1

                # Collect metrics
                img_size = imgs.shape[2:]
                patch_size = 4
                num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

                metrics = self.metrics_collector.compute_metrics(
                    stats,
                    num_total_patches=num_patches,
                    batch_size=imgs.shape[0],
                )
                epoch_metrics.append(metrics)

            if num_batches == 0:
                logger.warning(f"Epoch {epoch+1} 没有有效 batch")
                continue

            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches
            aggregated = self.metrics_collector.aggregate_metrics(epoch_metrics)

            history["jigsaw_loss"].append(avg_loss)
            history["accuracy"].append(avg_acc)
            history["token_density"].append(aggregated.get("token_density_mean", 0.0))
            history["depth_variance"].append(aggregated.get("depth_variance_mean", 0.0))

            logger.info(
                f"GJP Epoch {epoch+1}/{num_epochs}: "
                f"loss={avg_loss:.4f}, "
                f"accuracy={avg_acc:.3f}, "
                f"Token_Density={aggregated.get('token_density_mean', 0):.3f}"
            )

            self.memory_manager.maybe_clear_cache(epoch, force=True)

        return history


# =============================================================================
# PART 6: Main Entry
# =============================================================================

def create_experiment(
    mode: ExperimentMode,
    model: nn.Module,
    config: ProxyExperimentConfig,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    **kwargs,
) -> BaseExperiment:
    """Factory function: create experiment instance"""

    if mode == ExperimentMode.SSA:
        return SparseAblationExperiment(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
        )
    elif mode == ExperimentMode.FER:
        reference_model = kwargs.get("reference_model")
        return FrozenEncoderExperiment(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            reference_model=reference_model,
        )
    elif mode == ExperimentMode.GJP:
        return GeometricJigsawExperiment(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )


def main():
    """Main entry"""
    setup_logging(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Fractal Proxy Experiments - 三种实验模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 基础参数
    parser.add_argument("--mode", type=str,
                        default="sparse_subset_ablation",
                        help="实验模式 (ssa/fer/gjp 或 sparse_subset_ablation/frozen_encoder_reconstruction/geometric_jigsaw_proxy)")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径 (可选)")
    parser.add_argument("--output-dir", type=str, default="./experiments",
                        help="输出目录")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="预训练模型路径 (FER 可选)")

    # SSA 参数
    parser.add_argument("--num-classes", type=int, default=10,
                        help="SSA: 提取的类别数")
    parser.add_argument("--sparsity-ratios", type=str, default="0.3,0.5,0.7",
                        help="SSA: 稀疏比率列表")
    parser.add_argument("--gumbel-taus", type=str, default="0.5,1.0,2.0",
                        help="SSA: Gumbel 温度列表")
    parser.add_argument("--quick-test", action="store_true",
                        help="快速测试模式 (减少迭代次数)")

    # FER/GJP 实验配置 (L3 训练参数)
    parser.add_argument("--epochs", type=int, default=10,
                        help="实验训练轮数 (FER/GJP)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="权重衰减")

    # FER 参数
    parser.add_argument("--fer-epochs", type=int, default=None,
                        help="FER: 训练轮数 (覆盖 --epochs)")

    # GJP 参数
    parser.add_argument("--shuffle-group", type=int, default=4,
                        help="GJP: 打乱组大小")
    parser.add_argument("--gjp-epochs", type=int, default=None,
                        help="GJP: 训练轮数 (覆盖 --epochs)")

    # L1 数据集参数
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "cub200"],
                        help="数据集选择 (默认: cifar10)")

    # L2 模型架构参数
    parser.add_argument("--dim", type=int, default=384,
                        help="嵌入维度")
    parser.add_argument("--num-layers", type=int, default=8,
                        help="Transformer 层数 (原 depth)")
    parser.add_argument("--heads", type=int, default=6,
                        help="注意力头数")
    parser.add_argument("--min-patch-size", type=int, default=4,
                        help="最小 patch 大小")
    parser.add_argument("--max-level", type=int, default=8,
                        help="最大分割深度 (原 max_fractal_depth)")

    # 通用参数
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true",
                        help="禁用 AMP")
    parser.add_argument("--no-compile", action="store_true",
                        help="禁用 torch.compile")

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 构建配置
    mode = ExperimentMode.from_string(args.mode)

    # 使用命令行参数覆盖默认值
    fer_epochs = args.fer_epochs if args.fer_epochs else args.epochs
    gjp_epochs = args.gjp_epochs if args.gjp_epochs else args.epochs

    config = ProxyExperimentConfig(
        mode=mode,
        experiment_name=f"{mode.value}_{args.seed}",
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # L2 模型架构
        dim=args.dim,
        num_layers=args.num_layers,
        heads=args.heads,
        min_patch_size=args.min_patch_size,
        max_level=args.max_level,
        # L3 训练配置
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        pretrained_path=args.pretrained,
        use_amp=not args.no_amp,
        use_compile=not args.no_compile,
    )

    if mode == ExperimentMode.SSA:
        config.ssa_config = SSAConfig(
            num_classes_subset=args.num_classes,
            sparsity_ratios=tuple(map(float, args.sparsity_ratios.split(","))),
            gumbel_taus=tuple(map(float, args.gumbel_taus.split(","))),
            quick_test=args.quick_test,
        )
    elif mode == ExperimentMode.FER:
        config.fer_config = FERConfig(
            epochs=fer_epochs,
        )
    elif mode == ExperimentMode.GJP:
        config.gjp_config = GJPConfig(
            shuffle_group_size=args.shuffle_group,
            epochs=gjp_epochs,
        )

    # 加载模型
    from vit_pytorch import FractalCurveViT

    model = FractalCurveViT(
        num_classes=200,
        dim=config.dim,
        num_layers=config.num_layers,  # I145: depth -> num_layers
        heads=config.heads,
    )

    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=args.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Load pretrained model: {args.pretrained}")

    # Load data (support CIFAR-10/100 and CUB-200)
    from torchvision import datasets, transforms

    dataset_name = args.dataset
    logger.info(f"Loading dataset: {dataset_name}")

    if dataset_name.startswith("cifar"):
        # CIFAR-10/100: 32x32 images, smaller model works well
        num_classes = 100 if dataset_name == "cifar100" else 10
        image_size = 32

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            ),
        ])

        train_dataset = getattr(datasets, dataset_name.upper())(
            root="./data",
            train=True,
            download=True,
            transform=transform,
        )
        logger.info(f"CIFAR dataset loaded: {len(train_dataset)} samples, {num_classes} classes")

    elif dataset_name == "cub200":
        # CUB-200: 224x224 images, requires larger model
        data_root = Path("./data/CUB-200-2011")
        train_dir = data_root / "CUB_200_2011" / "train"

        if not train_dir.exists():
            raise FileNotFoundError(
                f"CUB-200 dataset not found at {train_dir}. "
                "Please download and extract the CUB-200-2011 dataset."
            )

        num_classes = 200
        image_size = 224

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        train_dataset = datasets.ImageFolder(str(train_dir), transform=transform)
        logger.info(f"CUB-200 dataset loaded: {len(train_dataset)} samples, {num_classes} classes")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 流式加载不需要多进程
        pin_memory=True,
        drop_last=True,
    )

    # Create and run experiment
    experiment = create_experiment(
        mode=mode,
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=None,
    )

    logger.info(f"开始运行 {mode.value} 实验...")

    if mode == ExperimentMode.SSA:
        results = experiment.run()

        # Print result table
        print("\n" + "=" * 60)
        print("SSA Results - Token Activation Analysis")
        print("=" * 60)
        print(f"{'Sparsity':<10} {'Tau':<8} {'Token_Density':<15} {'Depth_Variance':<15} {'Avg_Tokens':<10}")
        print("-" * 60)
        for (sparsity, tau), metrics in results.items():
            print(f"{sparsity:<10} {tau:<8} {metrics['token_density']:<15.3f} "
                  f"{metrics['depth_variance']:<15.3f} {metrics['avg_tokens']:<10.1f}")
        print("=" * 60)

    else:
        history = experiment.run()
        print("\n" + "=" * 60)
        print(f"{mode.value} Results")
        print("=" * 60)

        if mode == ExperimentMode.FER:
            print(f"Final Recon Loss: {history['recon_loss'][-1]:.4f}")
            print(f"Final Token Density: {history['token_density'][-1]:.3f}")
        else:
            print(f"Final Jigsaw Loss: {history['jigsaw_loss'][-1]:.4f}")
            print(f"Final Accuracy: {history['accuracy'][-1]:.3f}")
            print(f"Final Token Density: {history['token_density'][-1]:.3f}")

        print("=" * 60)


if __name__ == "__main__":
    main()
