# -*- coding: utf-8 -*-
"""
Fractal Proxy Experiments - 三种实验模式

Mode A: Sparse Subset Ablation (SSA)
    - SubsetSampler 从 CUB-200 提取 N 个类
    - 网格搜索 sparsity_ratio × gumbel_tau
    - 记录平均 Token 激活数

Mode B: Frozen-Encoder Reconstruction (FER)
    - 冻结 Transformer Encoder
    - 只训练 Learnable Splitter 和 ShapeScaleEncoder
    - FeatureReconstructionLoss: 重构 ViT 特征统计特性

Mode C: Geometric Jigsaw Proxy (GJP)
    - 自监督任务：随机打乱分形层级拓扑顺序
    - 预测分形 Patch 的相对希尔伯特索引

Usage:
    # SSA 网格搜索
    uv run python src/training/fractal_proxy_experiments.py --mode ssa \
        --sparsity-ratios 0.3,0.5,0.7 --gumbel-taus 0.5,1.0,2.0

    # FER (可选预训练)
    uv run python src/training/fractal_proxy_experiments.py --mode fer \
        --pretrained path/to/checkpoint.pt

    # GJP
    uv run python src/training/fractal_proxy_experiments.py --mode gjp \
        --shuffle-group 4
"""

from __future__ import annotations

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
# PART 1: 配置与常量
# =============================================================================

class ExperimentMode(Enum):
    """三种实验模式"""
    SSA = "sparse_subset_ablation"    # 稀疏子集消融
    FER = "frozen_encoder_reconstruction"  # 冻结编码器重构
    GJP = "geometric_jigsaw_proxy"    # 几何拼图代理


@dataclass
class ProxyExperimentConfig:
    """实验统一配置"""
    # 基础配置
    mode: ExperimentMode = ExperimentMode.SSA
    experiment_name: str = "default_experiment"
    seed: int = 42

    # 显存管理
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_fractal_depth: int = 8  # 硬上限，防止死循环递归

    # 日志配置
    log_interval: int = 50
    save_dir: Optional[Path] = None

    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    use_compile: bool = False  # 可选启用 torch.compile

    # 模型路径
    pretrained_path: Optional[str] = None

    # 实验特定配置
    ssa_config: Optional["SSAConfig"] = None
    fer_config: Optional["FERConfig"] = None
    gjp_config: Optional["GJPConfig"] = None


@dataclass
class SSAConfig:
    """Sparse Subset Ablation 配置"""
    num_classes_subset: int = 10  # 提取 N 个类
    sparsity_ratios: Tuple[float, ...] = (0.3, 0.5, 0.7)
    gumbel_taus: Tuple[float, ...] = (0.5, 1.0, 2.0)

    # 记录指标
    record_per_layer_tokens: bool = True
    record_token_density: bool = True
    record_depth_variance: bool = True

    # 快速模式
    quick_test: bool = False  # 减少迭代次数用于快速验证


@dataclass
class FERConfig:
    """Frozen Encoder Reconstruction 配置"""
    # 重构目标
    reconstruct_mean: bool = True
    reconstruct_var: bool = True
    reconstruct_channel_corr: bool = True

    # 损失权重
    mean_weight: float = 1.0
    var_weight: float = 1.0
    corr_weight: float = 0.5

    # 训练配置
    learning_rate: float = 1e-4
    frozen_encoder_epochs: int = 10
    weight_decay: float = 0.01

    # ShapeScaleEncoder 配置
    shape_encoder_names: Tuple[str, ...] = (
        'shape_scale_encoder', 'shape_encoder', 'scale_encoder'
    )


@dataclass
class GJPConfig:
    """Geometric Jigsaw Proxy 配置"""
    shuffle_group_size: int = 4  # 每组 shuffle 大小

    # Hilbert 索引相关
    max_relative_distance: int = 8  # 最大相对距离
    predict_relative_offset: bool = True

    # 损失权重
    jigsaw_weight: float = 1.0
    entropy_weight: float = 0.1

    # 训练配置
    learning_rate: float = 1e-4
    num_epochs: int = 10
    weight_decay: float = 0.01


# =============================================================================
# PART 2: 基础架构
# =============================================================================

@dataclass
class TokenMetrics:
    """Token 指标数据结构"""
    # 核心指标 (必须记录)
    token_density: float  # 每层活跃 Token 比例
    recursive_depth_variance: float  # 深度方差

    # 扩展指标
    num_tokens_per_depth: Dict[int, int] = field(default_factory=dict)
    avg_tokens_per_sample: float = 0.0
    hilbert_locality_score: float = 0.0
    batch_size: int = 1


class TokenMetricsCollector:
    """Token 指标收集器 - 从 model.forward() aux_infos 提取"""

    def __init__(self, max_depth: int = 8):
        self.max_depth = max_depth

    def compute_metrics(
        self,
        aux_infos: Dict[str, Any],
        num_total_patches: int,
        batch_size: int = 1,
    ) -> TokenMetrics:
        """从 aux_infos 计算指标"""

        # 1. Token Density: 活跃 token / 总 patch
        num_tokens = aux_infos.get("num_tokens", 0)
        if isinstance(num_tokens, torch.Tensor):
            num_tokens = num_tokens.item()

        token_density = num_tokens / num_total_patches if num_total_patches > 0 else 0.0

        # 2. Recursive Depth Variance
        levels_used = aux_infos.get("levels_used", None)
        if levels_used is not None:
            if isinstance(levels_used, torch.Tensor):
                levels_used = levels_used.cpu().tolist()
            depth_variance = self._compute_depth_variance(levels_used)
        else:
            depth_variance = 0.0

        # 3. 每层 Token 分布
        levels_distribution = aux_infos.get("levels_distribution", None)
        if levels_distribution is not None:
            if isinstance(levels_distribution, torch.Tensor):
                levels_distribution = levels_distribution.cpu().tolist()
            num_tokens_per_depth = {
                i: levels_distribution[i] for i in range(len(levels_distribution))
            }
        else:
            num_tokens_per_depth = {}

        # 4. Hilbert Locality Score (可选)
        hilbert_locality = aux_infos.get("hilbert_locality_score", 0.0)
        if isinstance(hilbert_locality, torch.Tensor):
            hilbert_locality = hilbert_locality.item()

        return TokenMetrics(
            token_density=token_density,
            recursive_depth_variance=depth_variance,
            num_tokens_per_depth=num_tokens_per_depth,
            avg_tokens_per_sample=num_tokens / batch_size if batch_size > 0 else 0.0,
            hilbert_locality_score=hilbert_locality,
            batch_size=batch_size,
        )

    def _compute_depth_variance(self, levels: List[int]) -> float:
        """计算深度方差"""
        if not levels:
            return 0.0

        levels_tensor = torch.tensor(levels, dtype=torch.float32)
        return levels_tensor.var().item()

    def aggregate_metrics(self, metrics_list: List[TokenMetrics]) -> Dict[str, float]:
        """聚合多个 batch 的指标"""
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
    """显存管理 - 满足工程约束"""

    def __init__(self, config: ProxyExperimentConfig):
        self.config = config
        self.accum_steps = config.gradient_accumulation_steps
        self._last_reserved: Optional[float] = None

    def maybe_clear_cache(self, epoch: int, force: bool = False) -> None:
        """
        Epoch 结束时调用 torch.cuda.empty_cache()
        Also called after Graph Break warnings
        """
        if torch.cuda.is_available():
            # 每个 epoch 结束时清理，或强制清理
            if force or epoch % 1 == 0:
                torch.cuda.empty_cache()

    def check_memory_leak(self) -> Dict[str, float]:
        """
        检查显存泄漏 - 检测未 detach 张量
        使用 VectorizationAuditor 的模式
        """
        if not torch.cuda.is_available():
            return {}

        memory_stats = {
            "allocated_GB": torch.cuda.memory_allocated() / 1024**3,
            "reserved_GB": torch.cuda.memory_reserved() / 1024**3,
        }

        # 警告：如果 reserved 持续增长，说明有泄漏
        if self._last_reserved is not None:
            growth_ratio = memory_stats["reserved_GB"] / self._last_reserved
            if growth_ratio > 1.1:  # 增长超过 10%
                logger.warning(
                    f"Potential memory leak: reserved increased from "
                    f"{self._last_reserved:.2f}GB to {memory_stats['reserved_GB']:.2f}GB "
                    f"(ratio: {growth_ratio:.2f})"
                )
        self._last_reserved = memory_stats["reserved_GB"]

        return memory_stats

    def get_memory_stats(self) -> Dict[str, float]:
        """获取当前显存状态"""
        if not torch.cuda.is_available():
            return {}

        return {
            "allocated_GB": torch.cuda.memory_allocated() / 1024**3,
            "reserved_GB": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated_GB": torch.cuda.max_memory_allocated() / 1024**3,
        }


class ExperimentWrapper:
    """
    实验包装器 - 解耦设计，不修改核心模型库

    通过依赖注入和属性访问封装模型，支持：
    - 冻结/解冻模块
    - 设置 Gumbel 温度
    - 设置稀疏比率
    - 自动检测 ShapeScaleEncoder
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

        # 提取 tokenizer 和 splitter
        self.tokenizer = getattr(model, 'tokenizer', None)
        self.splitter = getattr(self.tokenizer, 'splitter', None) if self.tokenizer else None

        # 自动检测 ShapeScaleEncoder (FER 模式)
        self.shape_scale_encoder: Optional[nn.Module] = None
        if config.mode == ExperimentMode.FER:
            self._detect_shape_encoder()

        # 移动到设备
        self.model = self.model.to(self.device)

    def _detect_shape_encoder(self) -> None:
        """自动检测 ShapeScaleEncoder (多候选名)"""
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
        """包装模型以适应实验需求"""
        if self.config.mode == ExperimentMode.FER:
            self._freeze_encoder()
        return self.model

    def unwrap_model(self) -> nn.Module:
        """恢复模型原始状态"""
        if self.config.mode == ExperimentMode.FER:
            self._unfreeze_encoder()
        return self.model

    def _freeze_encoder(self) -> None:
        """冻结 Transformer Encoder"""
        encoder = getattr(self.model, 'transformer', None)
        if encoder is not None:
            for param in encoder.parameters():
                param.requires_grad = False
            logger.info("Transformer Encoder 已冻结")

            # 确保 tokenizer 和 splitter 可训练
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
        """解冻 Encoder (FER 预训练后)"""
        encoder = getattr(self.model, 'transformer', None)
        if encoder is not None:
            for param in encoder.parameters():
                param.requires_grad = True
            logger.info("Transformer Encoder 已解冻")

    def set_gumbel_temperature(self, tau: float) -> None:
        """设置 Gumbel 温度 (SSA 实验)"""
        if self.splitter is not None and hasattr(self.splitter, 'log_temperature'):
            self.splitter.log_temperature.data = torch.tensor(tau).log().to(self.device)
            logger.debug(f"Gumbel 温度设置为 {tau}")

    def set_sparsity_ratio(self, ratio: float) -> None:
        """设置稀疏比率 (修改 K_max)"""
        if self.splitter is not None:
            original_K_max = getattr(self.splitter, 'K_max', 64)
            new_K_max = max(8, int(original_K_max * ratio))
            self.splitter.K_max = new_K_max
            logger.debug(f"稀疏比率 {ratio} -> K_max = {new_K_max}")

    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """获取可训练参数列表 (FER 模式)"""
        params = []

        if self.splitter is not None:
            params.extend(p for p in self.splitter.parameters() if p.requires_grad)

        if self.shape_scale_encoder is not None:
            params.extend(p for p in self.shape_scale_encoder.parameters() if p.requires_grad)

        # 如果没有指定模块，返回所有参数
        if not params:
            params = list(self.model.parameters())

        return params


class BaseExperiment(ABC):
    """实验基类"""

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
            max_depth=config.max_fractal_depth
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
    从 CUB-200 提取 N 个类 - 流式采样

    禁止全量加载：使用生成器模式，支持大数据集
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
    Mode A: Sparse Subset Ablation 实验

    目标：测试 sparsity_ratio 和 gumbel_tau 对 Token 激活的影响
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
        网格搜索: 遍历 sparsity_ratio 和 gumbel_tau

        Returns:
            Dict[(sparsity, tau)] -> Dict[metric] 平均结果
        """
        results = {}

        for sparsity in self.ssa_config.sparsity_ratios:
            for tau in self.ssa_config.gumbel_taus:
                key = (sparsity, tau)
                logger.info(f"Running SSA: sparsity={sparsity}, tau={tau}")

                # 设置实验参数
                self.wrapper.set_sparsity_ratio(sparsity)
                self.wrapper.set_gumbel_temperature(tau)

                # 运行单配置实验
                run_results = self._run_single_config()
                results[key] = run_results

                # 清理显存
                self.memory_manager.maybe_clear_cache(0, force=True)

                if self.ssa_config.quick_test:
                    break  # 快速测试模式只跑一个配置
            if self.ssa_config.quick_test:
                break

        return results

    def _run_single_config(self) -> Dict[str, float]:
        """运行单配置实验"""
        metrics_history = []

        max_iter = 10 if self.ssa_config.quick_test else len(self.train_loader)

        for batch_idx, (imgs, labels) in enumerate(self.train_loader):
            if batch_idx >= max_iter:
                break

            imgs = imgs.to(self.device)

            # Forward
            try:
                if self.config.use_compile:
                    with torch.compiler.disable():
                        logits, aux_infos = self.wrapper.model(imgs)
                else:
                    logits, aux_infos = self.wrapper.model(imgs)
            except Exception as e:
                self._handle_compile_warning(e)
                continue

            # 收集指标
            img_size = imgs.shape[2:]
            patch_size = 4  # 默认 patch size
            num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

            metrics = self.metrics_collector.compute_metrics(
                aux_infos,
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
    重构 ViT 特征统计特性

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
        计算重构损失

        Args:
            original_features: [B, N, D] 原始 ViT 特征
            reconstructed_features: [B, N, D] 重构特征

        Returns:
            loss: 总损失
            info: 各分量统计
        """
        info = {}

        # 确保特征是连续的
        original_features = original_features.contiguous()
        reconstructed_features = reconstructed_features.contiguous()

        # 1. 均值重构损失
        if self.config.reconstruct_mean:
            original_mean = original_features.mean(dim=(0, 1))  # [D]
            recon_mean = reconstructed_features.mean(dim=(0, 1))
            mean_loss = F.mse_loss(recon_mean, original_mean)
            info["mean_loss"] = mean_loss.item()
        else:
            mean_loss = torch.tensor(0.0, device=original_features.device)

        # 2. 方差重构损失
        if self.config.reconstruct_var:
            original_var = original_features.var(dim=(0, 1))  # [D]
            recon_var = reconstructed_features.var(dim=(0, 1))
            var_loss = F.mse_loss(recon_var, original_var)
            info["var_loss"] = var_loss.item()
        else:
            var_loss = torch.tensor(0.0, device=original_features.device)

        # 3. 通道相关性重构损失
        if self.config.reconstruct_channel_corr:
            original_flat = original_features.flatten(0, 1)  # [B*N, D]
            recon_flat = reconstructed_features.flatten(0, 1)

            # 中心化
            original_flat = original_flat - original_flat.mean(dim=0, keepdim=True)
            recon_flat = recon_flat - recon_flat.mean(dim=0, keepdim=True)

            # 计算协方差矩阵
            # cov = (X^T X) / (n - 1)
            n = original_flat.shape[0]
            original_cov = (original_flat.T @ original_flat) / (n - 1 + 1e-6)
            recon_cov = (recon_flat.T @ recon_flat) / (n - 1 + 1e-6)

            corr_loss = F.mse_loss(recon_cov, original_cov)
            info["corr_loss"] = corr_loss.item()
        else:
            corr_loss = torch.tensor(0.0, device=original_features.device)

        # 总损失
        total_loss = (
            self.config.mean_weight * mean_loss +
            self.config.var_weight * var_loss +
            self.config.corr_weight * corr_loss
        )

        info["total_loss"] = total_loss.item()

        return total_loss, info


class FrozenEncoderExperiment(BaseExperiment):
    """
    Mode B: Frozen Encoder Reconstruction 实验

    目标：冻结 Encoder，只训练 Splitter + ShapeScaleEncoder
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

        # 包装模型（会冻结 encoder）
        self.wrapper.wrap_model_for_experiment()

        # 损失函数
        feat_dim = getattr(config, 'dim', 384)
        self.recon_loss_fn = FeatureReconstructionLoss(
            config=self.fer_config,
            feat_dim=feat_dim,
        )

        # 优化器 (只训练 Splitter 和 ShapeScaleEncoder)
        self.optimizer = self._create_optimizer()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器 - 只优化 Splitter 和 ShapeScaleEncoder"""
        params = self.wrapper.get_trainable_params()

        if not params:
            logger.warning("未找到可训练参数")
            return torch.optim.Adam([])  # 空参数

        logger.info(f"FER: 找到 {len(params)} 组可训练参数")

        return torch.optim.AdamW(
            params,
            lr=self.fer_config.learning_rate,
            weight_decay=self.fer_config.weight_decay,
        )

    def run(self) -> Dict[str, Any]:
        """运行 FER 实验"""
        history = {
            "recon_loss": [],
            "token_density": [],
            "depth_variance": [],
            "memory_stats": [],
        }

        self.model.train()

        num_epochs = self.fer_config.frozen_encoder_epochs
        max_iter = 10 if getattr(self.ssa_config, 'quick_test', False) else None

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            epoch_metrics = []

            for batch_idx, (imgs, _) in enumerate(self.train_loader):
                if max_iter and batch_idx >= max_iter:
                    break

                imgs = imgs.to(self.device)

                # 1. 获取原始 ViT 特征 (冻结的 encoder)
                with torch.no_grad():
                    if self.reference_model is not None:
                        _, original_aux = self.reference_model(imgs, return_aux_info=True)
                    else:
                        # 使用当前模型但 encoder 冻结
                        _, original_aux = self.wrapper.model(imgs, return_aux_info=True)

                    # 提取原始特征
                    original_features = original_aux.get("transformer_tokens")
                    if original_features is None:
                        original_features = original_aux.get("features")

                    if original_features is None:
                        logger.warning("无法提取原始特征，跳过 batch")
                        continue

                # 2. 前向传播 (训练 splitter)
                try:
                    if self.config.use_compile:
                        with torch.compiler.disable():
                            logits, aux_infos = self.wrapper.model(imgs)
                    else:
                        logits, aux_infos = self.wrapper.model(imgs)
                except Exception as e:
                    self._handle_compile_warning(e)
                    continue

                # 3. 获取重构特征
                recon_features = aux_infos.get("reconstructed_features", logits)

                # 4. 计算重构损失
                loss, info = self.recon_loss_fn(original_features, recon_features)

                # 5. 梯度累积
                loss = loss / self.config.gradient_accumulation_steps

                # 6. 反向传播
                loss.backward()

                # 7. 梯度累积步骤
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_loss += info["total_loss"]
                num_batches += 1

                # 收集指标
                img_size = imgs.shape[2:]
                patch_size = 4
                num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

                metrics = self.metrics_collector.compute_metrics(
                    aux_infos,
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
    """Hilbert 索引打乱器 - 用于几何拼图任务"""

    def __init__(self, group_size: int = 4, max_depth: int = 8):
        self.group_size = group_size
        self.max_depth = max_depth

    def shuffle_within_groups(
        self,
        levels_info_data: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        在分形层级内打乱拓扑顺序

        数学形式化:
            对每组 G 个 token，随机排列顺序
            保持层级结构不变，只改变 Hilbert 索引排序

        Args:
            levels_info_data: [B, N, D+1] 深度+路径信息
            batch_indices: [N] 每个 token 的 batch 索引

        Returns:
            shuffled_data: 打乱后的 levels_info
            shuffle_indices: 原始 -> 打乱后 的映射
        """
        B, N, D = levels_info_data.shape
        device = levels_info_data.device

        shuffled_data = levels_info_data.clone()
        shuffle_indices = torch.arange(N, device=device)

        # 按 batch 处理
        for b in range(B):
            batch_mask = batch_indices == b
            batch_positions = torch.where(batch_mask)[0]

            if len(batch_positions) >= self.group_size:
                # 分组打乱
                num_groups = len(batch_positions) // self.group_size

                for g in range(num_groups):
                    start = g * self.group_size
                    end = start + self.group_size
                    group = batch_positions[start:end]

                    # 随机打乱组内顺序
                    perm = torch.randperm(self.group_size, device=device)
                    shuffled_data[b, group] = levels_info_data[b, group[perm]]

        return shuffled_data, shuffle_indices

    @staticmethod
    def compute_hilbert_indices(
        depths: torch.Tensor,
        paths: torch.Tensor,
    ) -> torch.Tensor:
        """
        从 depths 和 paths 计算 Hilbert 索引

        Args:
            depths: [B, N] 深度值
            paths: [B, N, D] 四象限路径

        Returns:
            hilbert_indices: [B, N] Hilbert 曲线索引
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
    几何拼图损失 - 预测相对 Hilbert 索引

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

        # 预测头：预测相对偏移类别
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
        计算拼图损失

        Args:
            tokens: [B, N, D] 分形 token 特征
            levels_info: [B, N, D+1] 深度+路径信息
            original_indices: 原始 Hilbert 索引 (用于计算相对偏移)

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

        # 1. 计算 Hilbert 索引
        if original_indices is None:
            depths = levels_info[:, :, 0]  # [B, N]
            paths = levels_info[:, :, 1:]  # [B, N, D]
            hilbert_indices = HilbertIndexShuffler.compute_hilbert_indices(depths, paths)
        else:
            hilbert_indices = original_indices

        # 2. 计算相对偏移 (分类任务)
        target_pairs = []
        offset_labels = []

        for b in range(B):
            for i in range(N):
                for j in range(i + 1, min(i + self.max_offset * 2 + 1, N)):
                    # 只考虑同层级的 token
                    if levels_info[b, i, 0] == levels_info[b, j, 0]:
                        offset = abs(int(hilbert_indices[b, i]) - int(hilbert_indices[b, j]))
                        if offset <= self.max_offset:
                            target_pairs.append((b, i, j))
                            offset_labels.append(min(offset, self.max_offset))

        if not target_pairs:
            info["jigsaw_loss"] = 0.0
            info["num_pairs"] = 0
            return torch.tensor(0.0, device=tokens.device), info

        # 3. 构建对比特征
        pair_features = []
        for b, i, j in target_pairs:
            pair_feat = torch.cat([tokens[b, i], tokens[b, j]], dim=-1)
            pair_features.append(pair_feat)

        pair_features = torch.stack(pair_features)  # [P, 2*D]

        # 4. 预测偏移
        offset_logits = self.offset_predictor(pair_features)  # [P, max_offset+1]

        # 5. 分类损失
        offset_labels_tensor = torch.tensor(offset_labels, device=tokens.device)
        jigsaw_loss = F.cross_entropy(offset_logits, offset_labels_tensor)

        info["jigsaw_loss"] = jigsaw_loss.item()
        info["num_pairs"] = len(target_pairs)
        info["accuracy"] = (offset_logits.argmax(dim=-1) == offset_labels_tensor).float().mean().item()

        # 6. 熵正则化
        if self.config.entropy_weight > 0:
            probs = F.softmax(offset_logits, dim=-1)
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
            jigsaw_loss = jigsaw_loss + self.config.entropy_weight * entropy
            info["entropy"] = entropy.item()

        return jigsaw_loss, info


class GeometricJigsawExperiment(BaseExperiment):
    """
    Mode C: Geometric Jigsaw Proxy 实验

    目标：自监督任务 - 预测分形 Patch 的相对 Hilbert 索引
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

        # 包装模型
        self.wrapper.wrap_model_for_experiment()

        # 组件
        self.shuffler = HilbertIndexShuffler(
            group_size=self.gjp_config.shuffle_group_size,
            max_depth=config.max_fractal_depth,
        )

        feat_dim = getattr(config, 'dim', 384)
        self.jigsaw_loss_fn = GeometricJigsawLoss(
            config=self.gjp_config,
            feat_dim=feat_dim,
        )

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.gjp_config.learning_rate,
            weight_decay=self.gjp_config.weight_decay,
        )

    def run(self) -> Dict[str, Any]:
        """运行 GJP 实验"""
        history = {
            "jigsaw_loss": [],
            "token_density": [],
            "depth_variance": [],
            "accuracy": [],
        }

        self.model.train()

        num_epochs = self.gjp_config.num_epochs
        max_iter = 10 if getattr(self.ssa_config, 'quick_test', False) else None

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0
            epoch_metrics = []

            for batch_idx, (imgs, _) in enumerate(self.train_loader):
                if max_iter and batch_idx >= max_iter:
                    break

                imgs = imgs.to(self.device)

                # 1. 前向传播
                try:
                    if self.config.use_compile:
                        with torch.compiler.disable():
                            logits, aux_infos = self.wrapper.model(imgs)
                    else:
                        logits, aux_infos = self.wrapper.model(imgs)
                except Exception as e:
                    self._handle_compile_warning(e)
                    continue

                # 2. 提取 levels_info
                levels_info = aux_infos.get("levels_info")
                if levels_info is None:
                    logger.warning("无法获取 levels_info，跳过 batch")
                    continue

                # 3. 计算拼图损失
                loss, info = self.jigsaw_loss_fn(
                    tokens=logits,
                    levels_info=levels_info,
                )

                # 4. 梯度累积
                loss = loss / self.config.gradient_accumulation_steps

                # 5. 反向传播
                loss.backward()

                # 6. 梯度累积步骤
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_loss += info.get("jigsaw_loss", 0.0)
                epoch_acc += info.get("accuracy", 0.0)
                num_batches += 1

                # 收集指标
                img_size = imgs.shape[2:]
                patch_size = 4
                num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

                metrics = self.metrics_collector.compute_metrics(
                    aux_infos,
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
# PART 6: 主入口
# =============================================================================

def create_experiment(
    mode: ExperimentMode,
    model: nn.Module,
    config: ProxyExperimentConfig,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    **kwargs,
) -> BaseExperiment:
    """工厂函数：创建实验实例"""

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
    """设置日志"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )


def main():
    """主入口"""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Fractal Proxy Experiments - 三种实验模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 基础参数
    parser.add_argument("--mode", type=str,
                        choices=["ssa", "fer", "gjp"],
                        default="ssa", help="实验模式")
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

    # FER 参数
    parser.add_argument("--fer-lr", type=float, default=1e-4,
                        help="FER: 学习率")
    parser.add_argument("--fer-epochs", type=int, default=10,
                        help="FER: 训练轮数")

    # GJP 参数
    parser.add_argument("--shuffle-group", type=int, default=4,
                        help="GJP: 打乱组大小")
    parser.add_argument("--gjp-epochs", type=int, default=10,
                        help="GJP: 训练轮数")

    # 通用参数
    parser.add_argument("--batch-size", type=int, default=32)
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
    mode = ExperimentMode(args.mode)

    config = ProxyExperimentConfig(
        mode=mode,
        batch_size=args.batch_size,
        device=args.device,
        pretrained_path=args.pretrained,
        use_amp=not args.no_amp,
        use_compile=not args.no_compile,
        experiment_name=f"{mode.value}_{args.seed}",
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
            learning_rate=args.fer_lr,
            frozen_encoder_epochs=args.fer_epochs,
        )
    elif mode == ExperimentMode.GJP:
        config.gjp_config = GJPConfig(
            shuffle_group_size=args.shuffle_group,
            num_epochs=args.gjp_epochs,
        )

    # 加载模型
    from vit_pytorch import FractalCurveViT

    model = FractalCurveViT(
        num_classes=200,
        dim=384,
        depth=8,
        heads=6,
    )

    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=args.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"加载预训练模型: {args.pretrained}")

    # 加载数据 (使用流式加载)
    try:
        from training.datasets import StreamingCUB200Dataset

        train_dataset = StreamingCUB200Dataset(
            root="./data/CUB-200-2011",
            train=True,
            transform=None,
        )
        logger.info(f"CUB-200 流式数据集加载成功: {len(train_dataset)} 样本")
    except ImportError:
        # 回退到标准数据集
        from torchvision.datasets import CUB200
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        train_dataset = CUB200(
            root="./data/CUB-200-2011",
            train=True,
            download=False,
            transform=transform,
        )
        logger.info(f"CUB-200 标准数据集加载成功: {len(train_dataset)} 样本")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 流式加载不需要多进程
        pin_memory=True,
        drop_last=True,
    )

    # 创建并运行实验
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

        # 打印结果表格
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
