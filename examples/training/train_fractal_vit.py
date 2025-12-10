#!/usr/bin/env python3
"""Lightweight Fractal ViT training entry point for quick experiments."""

from __future__ import annotations

# 必须在导入其他模块之前设置这些环境变量
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
# 减少 OpenMP 线程数以避免 CPU 过载
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import argparse
import json
import multiprocessing
import random
import signal
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

try:
    from torch.amp.autocast_mode import autocast as _autocast
    from torch.amp.grad_scaler import GradScaler as _GradScaler
    _HAS_TORCH_AMP = True
except ImportError:  # pragma: no cover - fallback for older torch versions
    from torch.cuda.amp import autocast as _autocast
    from torch.cuda.amp import GradScaler as _GradScaler
    _HAS_TORCH_AMP = False

if TYPE_CHECKING:
    from torch.cuda.amp.grad_scaler import GradScaler as GradScalerType
else:
    GradScalerType = Any

from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder, CocoDetection, Caltech256
from tqdm import tqdm
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vit_pytorch.fractal_vit import NextGenerationFractalViT

import multiprocessing

# ============================================================================
# DataLoader 优化配置
# ============================================================================

def _diagnose_dataloader_config(num_workers: int, batch_size: int) -> None:
    """诊断并提供 DataLoader 配置建议。"""
    try:
        import psutil
    except ImportError:
        # psutil 不是必需的依赖
        return
    
    # 检测共享内存大小
    try:
        shm_stats = psutil.disk_usage('/dev/shm')
        shm_gb = shm_stats.total / (1024**3)
        shm_free_gb = shm_stats.free / (1024**3)
        
        print(f"\n📊 DataLoader Configuration:")
        print(f"  Workers: {num_workers}")
        print(f"  Batch size: {batch_size}")
        print(f"  Shared memory: {shm_gb:.1f}GB total, {shm_free_gb:.1f}GB free")
        
        # 估算内存需求
        # 每个 worker 大约需要 batch_size * image_size * channels * sizeof(float32) 的内存
        estimated_mb_per_worker = batch_size * 64 * 64 * 3 * 4 / (1024**2) * 2  # *2 for prefetch
        total_estimated_mb = estimated_mb_per_worker * num_workers
        
        print(f"  Estimated memory per worker: ~{estimated_mb_per_worker:.0f}MB")
        print(f"  Total estimated: ~{total_estimated_mb:.0f}MB")
        
        # 提供建议
        if total_estimated_mb > shm_free_gb * 1024 * 0.8:
            print(f"  ⚠️  WARNING: Workers may exceed shared memory!")
            recommended = max(1, int(shm_free_gb * 1024 * 0.8 / estimated_mb_per_worker))
            print(f"  💡 Recommended: --num-workers {recommended}")
        elif num_workers > 8 and shm_gb < 4:
            print(f"  ⚠️  WARNING: High worker count with limited shared memory")
            print(f"  💡 Consider: --num-workers 4-8 for --shm-size=4g")
        else:
            print(f"  ✓ Configuration looks good")
        
        print()
    except Exception:
        # psutil 可能不可用或在某些环境下失败
        pass


def get_optimal_num_workers() -> int:
    """获取最优的 worker 数量。
    
    考虑因素:
    - CPU 核心数
    - 共享内存大小（容器环境重要）
    - 系统类型（容器 vs 本地）
    
    容器建议:
    - --shm-size=4g: 推荐 4-8 workers
    - --shm-size=8g: 推荐 8-12 workers
    - 不足 2g: 推荐 0-2 workers
    """
    cpu_count = multiprocessing.cpu_count()
    
    # 检测是否在容器中（通过 /.dockerenv 或 cgroup）
    in_container = os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')
    
    if in_container:
        # 容器环境：更保守的设置
        # 共享内存通常是瓶颈，不是 CPU
        return min(max(2, cpu_count // 3), 8)
    else:
        # 本地环境：可以更激进
        return min(max(2, cpu_count // 2), 8)


# ============================================================================
# 自适应 Tokenization 监控
# ============================================================================

@dataclass
class TokenizationStats:
    """每个 batch 的 tokenization 统计数据。"""
    avg_tokens_per_image: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    std_tokens: float = 0.0
    levels_used: List[int] = field(default_factory=list)
    max_level_reached: int = 0
    split_decisions: int = 0  # 总分割决策次数
    split_ratio: float = 0.0  # 选择分割的比例


class AdaptiveTokenMonitor:
    """监控自适应 tokenization 特性的工具类。
    
    用于验证模型是否按预期实现了基于 Hilbert curve 的自适应分词：
    1. 简单图像应生成较少的 tokens
    2. 复杂图像应生成较多的 tokens  
    3. 分割决策应随训练进行而优化
    4. 不同层级的使用应反映图像复杂度
    """
    
    def __init__(self) -> None:
        self.epoch_stats: Dict[int, List[TokenizationStats]] = defaultdict(list)
        self.complexity_correlation: List[Tuple[float, int]] = []  # (图像复杂度, token数)
        self.split_ratio_history: List[float] = []  # 每个 epoch 的平均分割比例
    
    @torch.no_grad()
    def analyze_batch(
        self,
        model: nn.Module,
        images: torch.Tensor,
        epoch: int,
    ) -> TokenizationStats:
        """分析一个 batch 的 tokenization 行为。"""
        tokenizer = getattr(model, "tokenizer", None) or getattr(model, "fractal_tokenizer", None)
        if tokenizer is None:
            # SimpleFractalViT wraps enhanced_model
            enhanced = getattr(model, "enhanced_model", None)
            if enhanced:
                tokenizer = getattr(enhanced, "tokenizer", None)
        
        if tokenizer is None:
            return TokenizationStats()
        
        # 获取 tokenization 结果
        output = tokenizer.tokenize(images)
        sequences = output.sequences
        
        token_counts = [seq.tokens.shape[0] for seq in sequences]
        levels_used_all: List[int] = []
        max_level = 0
        
        for seq in sequences:
            levels = seq.metadata.get("levels", None)
            if levels is not None and levels.numel() > 0:
                depths = levels[:, 0].tolist()
                levels_used_all.extend(depths)
                max_level = max(max_level, max(depths))
        
        # 计算图像复杂度（使用方差作为代理）
        for i, seq in enumerate(sequences):
            img_var = images[i].var().item()
            self.complexity_correlation.append((img_var, token_counts[i]))
        
        # 计算分割决策统计
        split_decisions = len(getattr(tokenizer, "saved_log_probs", []))
        # 注意：saved_log_probs 可能在 clear_saved_actions 后为空
        
        stats = TokenizationStats(
            avg_tokens_per_image=float(np.mean(token_counts)),
            min_tokens=int(np.min(token_counts)),
            max_tokens=int(np.max(token_counts)),
            std_tokens=float(np.std(token_counts)),
            levels_used=sorted(list(set(levels_used_all))),
            max_level_reached=max_level,
            split_decisions=split_decisions,
        )
        
        self.epoch_stats[epoch].append(stats)
        return stats
    
    def get_epoch_summary(self, epoch: int) -> Dict[str, Any]:
        """获取一个 epoch 的汇总统计。"""
        stats_list = self.epoch_stats.get(epoch, [])
        if not stats_list:
            return {}
        
        avg_tokens = np.mean([s.avg_tokens_per_image for s in stats_list])
        std_tokens = np.mean([s.std_tokens for s in stats_list])
        min_tokens = min(s.min_tokens for s in stats_list)
        max_tokens = max(s.max_tokens for s in stats_list)
        all_levels = set()
        for s in stats_list:
            all_levels.update(s.levels_used)
        max_level = max(s.max_level_reached for s in stats_list)
        
        return {
            "epoch": epoch,
            "avg_tokens_per_image": float(avg_tokens),
            "token_std_across_images": float(std_tokens),
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "levels_used": sorted(list(all_levels)),
            "max_level_reached": max_level,
            "token_range": max_tokens - min_tokens,
            "is_adaptive": max_tokens > min_tokens,  # 关键指标
        }
    
    def compute_complexity_correlation(self) -> float:
        """计算图像复杂度与 token 数量的相关性。
        
        正相关表示复杂图像产生更多 tokens（期望行为）。
        """
        if len(self.complexity_correlation) < 10:
            return 0.0
        
        complexities = [c[0] for c in self.complexity_correlation]
        token_counts = [c[1] for c in self.complexity_correlation]
        
        # Pearson 相关系数
        corr = np.corrcoef(complexities, token_counts)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
    
    def get_full_report(self) -> Dict[str, Any]:
        """生成完整的监控报告。"""
        epochs = sorted(self.epoch_stats.keys())
        epoch_summaries = [self.get_epoch_summary(e) for e in epochs]
        
        complexity_corr = self.compute_complexity_correlation()
        
        # 检查自适应性趋势
        token_ranges = [s.get("token_range", 0) for s in epoch_summaries]
        adaptivity_trend = "improving" if len(token_ranges) > 1 and token_ranges[-1] > token_ranges[0] else "stable"
        
        return {
            "epoch_summaries": epoch_summaries,
            "complexity_token_correlation": complexity_corr,
            "adaptivity_trend": adaptivity_trend,
            "is_working_as_expected": complexity_corr > 0.1,  # 期望正相关
            "diagnosis": self._generate_diagnosis(complexity_corr, epoch_summaries),
        }
    
    def _generate_diagnosis(self, corr: float, summaries: List[Dict]) -> str:
        """生成诊断信息。"""
        messages = []
        
        if corr > 0.3:
            messages.append("[OK] 自适应分词工作正常：复杂图像产生更多 tokens")
        elif corr > 0.1:
            messages.append("[~] 自适应分词有轻微效果，但可能需要更多训练")
        elif corr > -0.1:
            messages.append("[X] 自适应分词效果不明显：token 数与图像复杂度无关")
        else:
            messages.append("[X] 异常：复杂图像反而产生更少 tokens")
        
        if summaries:
            last = summaries[-1]
            if last.get("token_range", 0) > 0:
                messages.append(f"[OK] 不同图像产生不同数量的 tokens (范围: {last.get('min_tokens')}-{last.get('max_tokens')})")
            else:
                messages.append("[X] 所有图像产生相同数量的 tokens（可能是非自适应模式）")
            
            if len(last.get("levels_used", [])) > 1:
                messages.append(f"[OK] 使用了多个层级: {last.get('levels_used')}")
            else:
                messages.append("[~] 只使用了单一层级")
        
        return "\n".join(messages)


def analyze_adaptive_tokenization(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_samples: int = 100,
) -> Dict[str, Any]:
    """分析模型的自适应 tokenization 特性。
    
    创建简单和复杂图像，验证 tokenization 是否有差异。
    """
    model.eval()
    tokenizer = getattr(model, "tokenizer", None) or getattr(model, "fractal_tokenizer", None)
    if tokenizer is None:
        enhanced = getattr(model, "enhanced_model", None)
        if enhanced:
            tokenizer = getattr(enhanced, "tokenizer", None)
    
    if tokenizer is None:
        return {"error": "无法找到 tokenizer"}
    
    results = {
        "simple_images": [],
        "complex_images": [],
        "real_images": [],
    }
    
    # 获取一个 batch 来确定图像尺寸
    sample_batch = next(iter(loader))[0]
    _, c, h, w = sample_batch.shape
    
    with torch.no_grad():
        # 1. 测试简单图像（均匀颜色）
        simple_img = torch.zeros(1, c, h, w, device=device)
        output = tokenizer.tokenize(simple_img)
        simple_tokens = output.sequences[0].tokens.shape[0]
        results["simple_images"].append(simple_tokens)
        
        # 2. 测试复杂图像（随机噪声）
        complex_img = torch.randn(1, c, h, w, device=device)
        output = tokenizer.tokenize(complex_img)
        complex_tokens = output.sequences[0].tokens.shape[0]
        results["complex_images"].append(complex_tokens)
        
        # 3. 测试真实图像
        count = 0
        for images, _ in loader:
            images = images.to(device)
            output = tokenizer.tokenize(images)
            for seq in output.sequences:
                results["real_images"].append(seq.tokens.shape[0])
                count += 1
                if count >= num_samples:
                    break
            if count >= num_samples:
                break
    
    analysis = {
        "simple_image_tokens": int(np.mean(results["simple_images"])),
        "complex_image_tokens": int(np.mean(results["complex_images"])),
        "real_image_tokens_mean": float(np.mean(results["real_images"])),
        "real_image_tokens_std": float(np.std(results["real_images"])),
        "real_image_tokens_min": int(np.min(results["real_images"])),
        "real_image_tokens_max": int(np.max(results["real_images"])),
        "is_adaptive": int(np.mean(results["complex_images"])) > int(np.mean(results["simple_images"])),
        "adaptivity_ratio": float(np.mean(results["complex_images"])) / max(float(np.mean(results["simple_images"])), 1),
    }
    
    return analysis


class CocoClassificationWrapper(CocoDetection):
    """Wrapper for COCO dataset to behave like a classification dataset.
    It selects the category of the largest object in the image as the label.
    """
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super().__init__(root, annFile, transform, target_transform, transforms)
        # Map COCO category IDs (non-contiguous) to 0-79
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        # Target is a list of dicts. Pick the largest object.
        if not target:
            # No object? Return a dummy label (0) or handle as background
            label = 0
        else:
            # Find largest area
            best_obj = max(target, key=lambda x: x['area'])
            cat_id = best_obj['category_id']
            label = self.cat2label.get(cat_id, 0)
        
        return img, label


class TinyImageNetVal(Dataset):
    """Custom Dataset for Tiny ImageNet Validation Set."""
    def __init__(self, root: Path, class_to_idx: Dict[str, int], transform=None):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.annotations_file = self.root / "val_annotations.txt"
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.data = []
        
        if not self.annotations_file.exists():
             raise FileNotFoundError(f"Validation annotations not found: {self.annotations_file}")

        with open(self.annotations_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    self.data.append((parts[0], parts[1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, cls_name = self.data[idx]
        img_path = self.images_dir / img_name
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # Some validation classes might not be in train if dataset is corrupted, but usually safe
        label = self.class_to_idx[cls_name]
        return img, label


def autocast_context(device: torch.device, enabled: bool):
    if _HAS_TORCH_AMP:
        return _autocast(device.type, enabled=enabled)  # type: ignore[call-arg]
    return _autocast(enabled=enabled)  # type: ignore[call-arg]


def create_grad_scaler(use_amp: bool, device: torch.device) -> GradScalerType:
    enabled = use_amp and device.type == "cuda"
    if _HAS_TORCH_AMP:
        if device.type == "cuda":
            return _GradScaler("cuda", enabled=enabled)  # type: ignore[misc]
        return _GradScaler(enabled=enabled)
    return _GradScaler(enabled=enabled)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    image_size: int
    channels: int
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    dataset_cls: type


@dataclass
class ExperimentPaths:
    experiment_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    visuals_dir: Path
    workspace_models_dir: Path
    workspace_results_dir: Path
    workspace_visuals_dir: Path
    timestamp: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device option: {requested}")

    if requested == "cpu":
        print("Using CPU for training (requested).")
        return torch.device("cpu")

    if requested == "cuda":
        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device_index)
            print(f"Using CUDA device (requested): {device_name}")
            return torch.device("cuda")
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")

    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        print(f"Auto-selected CUDA device: {device_name}")
        return torch.device("cuda")

    print("CUDA not available; using CPU.")
    return torch.device("cpu")


def prepare_experiment_paths(prefix: str) -> ExperimentPaths:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = PROJECT_ROOT / "experiments" / f"{prefix}_{timestamp}"
    checkpoints_dir = experiment_dir / "checkpoints"
    logs_dir = experiment_dir / "logs"
    visuals_dir = experiment_dir / "visualizations"

    for path in (experiment_dir, checkpoints_dir, logs_dir, visuals_dir):
        path.mkdir(parents=True, exist_ok=True)

    workspace_root = PROJECT_ROOT / "workspace"
    workspace_models_dir = workspace_root / "models" / "fractal_vit"
    workspace_results_dir = workspace_root / "results"
    workspace_visuals_dir = workspace_root / "visualizations"
    for path in (workspace_models_dir, workspace_results_dir, workspace_visuals_dir):
        path.mkdir(parents=True, exist_ok=True)

    return ExperimentPaths(
        experiment_dir=experiment_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        visuals_dir=visuals_dir,
        workspace_models_dir=workspace_models_dir,
        workspace_results_dir=workspace_results_dir,
        workspace_visuals_dir=workspace_visuals_dir,
        timestamp=timestamp,
    )


def get_dataset_spec(dataset: str) -> DatasetSpec:
    dataset = dataset.lower()
    specs: Dict[str, DatasetSpec] = {
        "cifar10": DatasetSpec(
            name="CIFAR10",
            num_classes=10,
            image_size=32,
            channels=3,
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
            dataset_cls=CIFAR10,
        ),
        "cifar100": DatasetSpec(
            name="CIFAR100",
            num_classes=100,
            image_size=32,
            channels=3,
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761),
            dataset_cls=CIFAR100,
        ),
        "mnist": DatasetSpec(
            name="MNIST",
            num_classes=10,
            image_size=28,
            channels=1,
            mean=(0.1307,),
            std=(0.3081,),
            dataset_cls=MNIST,
        ),
        "imagenet": DatasetSpec(
            name="ImageNet",
            num_classes=1000,
            image_size=224,
            channels=3,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            dataset_cls=ImageFolder,
        ),
        "coco": DatasetSpec(
            name="COCO",
            num_classes=80,
            image_size=224,
            channels=3,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            dataset_cls=CocoClassificationWrapper,
        ),
        "caltech256": DatasetSpec(
            name="Caltech256",
            num_classes=257,
            image_size=224,
            channels=3,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            dataset_cls=Caltech256,
        ),
        "tiny-imagenet": DatasetSpec(
            name="TinyImageNet",
            num_classes=200,
            image_size=64,
            channels=3,
            mean=(0.4802, 0.4481, 0.3975),
            std=(0.2302, 0.2265, 0.2262),
            dataset_cls=TinyImageNetVal, # Placeholder, handled specially
        ),
    }

    if dataset not in specs:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return specs[dataset]


# ============================================================================
# Dataset Auto-Download Functions
# ============================================================================

def download_with_progress(url: str, dest: Path, desc: str = "Downloading") -> None:
    """Download file with progress bar."""
    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = min(int(count * block_size * 100 / total_size), 100)
            bar_length = 50
            filled = int(bar_length * percent / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            sys.stdout.write(f'\r{desc}: [{bar}] {percent}%')
            sys.stdout.flush()
    
    print(f"\n{desc} from {url}")
    urlretrieve(url, dest, progress_hook)
    print()  # New line after progress


def auto_download_tiny_imagenet(data_root: Path) -> bool:
    """Automatically download and prepare Tiny ImageNet dataset.
    
    Args:
        data_root: Root directory where the dataset will be stored
        
    Returns:
        True if successful or already exists, False if failed
    """
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    
    # Check if already exists
    if train_dir.exists() and val_dir.exists():
        num_train_classes = len(list(train_dir.glob('*')))
        if num_train_classes >= 200:  # Tiny ImageNet has 200 classes
            print(f"✓ Tiny ImageNet already exists at {data_root}")
            return True
    
    print("=" * 70)
    print("Tiny ImageNet not found. Downloading automatically...")
    print("=" * 70)
    
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = data_root / "tiny-imagenet-200.zip"
    extract_dir = data_root / "tiny-imagenet-200"
    
    try:
        # Create data directory
        data_root.mkdir(parents=True, exist_ok=True)
        
        # Download
        if not zip_path.exists():
            download_with_progress(url, zip_path, "Downloading Tiny ImageNet (~237 MB)")
        else:
            print(f"✓ ZIP file already exists: {zip_path}")
        
        # Extract
        if not extract_dir.exists():
            print("Extracting archive...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract with progress
                members = zip_ref.namelist()
                for i, member in enumerate(members):
                    if i % 100 == 0:
                        percent = int((i / len(members)) * 100)
                        sys.stdout.write(f'\rExtracting: {percent}%')
                        sys.stdout.flush()
                    zip_ref.extract(member, data_root)
                print('\rExtracting: 100%')
        
        # Move directories
        src_train = extract_dir / "train"
        src_val = extract_dir / "val"
        
        if not train_dir.exists() and src_train.exists():
            print(f"Moving train/ to {train_dir}")
            shutil.move(str(src_train), str(train_dir))
        
        if not val_dir.exists() and src_val.exists():
            print(f"Moving val/ to {val_dir}")
            shutil.move(str(src_val), str(val_dir))
        
        # Clean up
        print("Cleaning up temporary files...")
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        if zip_path.exists():
            zip_path.unlink()
        
        print("=" * 70)
        print("✓ Tiny ImageNet download and setup complete!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading Tiny ImageNet: {e}")
        print("\nPlease download manually:")
        print(f"1. Visit: {url}")
        print(f"2. Extract to: {data_root}")
        return False


def check_and_download_dataset(dataset_name: str, data_root: Path) -> bool:
    """Check if dataset exists and download if necessary.
    
    Args:
        dataset_name: Name of the dataset
        data_root: Root directory for dataset storage
        
    Returns:
        True if dataset is ready, False otherwise
    """
    if dataset_name == "tiny-imagenet":
        return auto_download_tiny_imagenet(data_root)
    
    # For other datasets (CIFAR, MNIST), torchvision handles download automatically
    return True


def build_transforms(spec: DatasetSpec) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    构建数据增强管道
    
    根据不同数据集选择合适的增强策略：
    - MNIST: 简单增强（灰度图不适合颜色变换）
    - CIFAR10/100: 使用 CIFAR10 AutoAugment 策略
    - ImageNet 类: 使用 IMAGENET AutoAugment 策略
    """
    # 数据集到 AutoAugment 策略的映射
    augment_policies = {
        "CIFAR10": transforms.AutoAugmentPolicy.CIFAR10,
        "CIFAR100": transforms.AutoAugmentPolicy.CIFAR10,  # CIFAR10 策略对 CIFAR100 也有效
        "MNIST": None,  # MNIST 不使用 AutoAugment
        "ImageNet": transforms.AutoAugmentPolicy.IMAGENET,
        "COCO": transforms.AutoAugmentPolicy.IMAGENET,
        "Caltech256": transforms.AutoAugmentPolicy.IMAGENET,
        "TinyImageNet": transforms.AutoAugmentPolicy.IMAGENET,
    }
    
    policy = augment_policies.get(spec.name)
    
    if spec.name.lower() == "mnist":
        # MNIST: 灰度图，使用简单增强
        train_ops = [
            transforms.Resize(32),
            transforms.RandomRotation(10),  # 轻微旋转
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
        ]
        test_ops = [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
        ]
    else:
        # 彩色图像：使用更丰富的增强策略
        train_ops = [
            transforms.Resize(spec.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(spec.image_size, padding=4),
        ]
        
        # 根据数据集选择对应的 AutoAugment 策略
        if policy is not None:
            train_ops.append(transforms.AutoAugment(policy))
        
        train_ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
            transforms.RandomErasing(p=0.25),  # 随机擦除
        ])
        
        test_ops = [
            transforms.Resize(spec.image_size),
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
        ]
    
    return transforms.Compose(train_ops), transforms.Compose(test_ops)


def create_dataloaders(
    spec: DatasetSpec,
    batch_size: int,
    val_split: float,
    subset_size: Optional[int],
    num_workers: int,
    pin_memory: bool,
    data_root_override: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if subset_size is not None and subset_size < 2:
        raise ValueError("subset_size must be at least 2 when provided")

    if data_root_override:
        data_root = Path(data_root_override)
    else:
        data_root = PROJECT_ROOT / "workspace" / "data"
    
    data_root.mkdir(parents=True, exist_ok=True)

    train_transform, test_transform = build_transforms(spec)
    
    # Helper for creating loaders with optimized settings
    use_persistent_workers = num_workers > 0
    prefetch = 2 if num_workers > 0 else None
    
    def make_loader(dataset: Dataset, sampler_indices: np.ndarray, shuffle: bool = False) -> DataLoader:
        common_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": use_persistent_workers,
            "prefetch_factor": prefetch,
            "drop_last": False,
        }
        if shuffle:
            return DataLoader(dataset, shuffle=True, **common_kwargs)
        sampler = SubsetRandomSampler(sampler_indices.tolist())
        return DataLoader(dataset, sampler=sampler, **common_kwargs)

    if spec.name == "Caltech256":
        # Caltech256 does not have a standard train/test split. We perform a random split.
        # We load the dataset twice: once with train transforms, once with test transforms.
        full_train_set = spec.dataset_cls(root=str(data_root), transform=train_transform, download=True)
        full_test_set = spec.dataset_cls(root=str(data_root), transform=test_transform, download=True)
        
        num_total = len(full_train_set)
        indices = np.arange(num_total)
        np.random.shuffle(indices)
        
        if subset_size is not None:
            indices = indices[:subset_size]
            num_total = len(indices)
            
        # Split: 10% Test, Val split from args, rest Train
        test_split = 0.1
        test_count = max(1, int(num_total * test_split))
        val_count = max(1, int(num_total * val_split))
        train_count = num_total - test_count - val_count
        
        if train_count <= 0:
             raise ValueError("Dataset too small for the requested split sizes.")

        test_indices = indices[:test_count]
        val_indices = indices[test_count : test_count + val_count]
        train_indices = indices[test_count + val_count :]
        
        train_loader = make_loader(full_train_set, train_indices)
        val_loader = make_loader(full_test_set, val_indices) # Use test transform for val
        test_loader = make_loader(full_test_set, test_indices)
        
        print(f"Loaded {spec.name} → train: {len(train_indices)}, val: {len(val_indices)}, test: {len(test_indices)}")
        return train_loader, val_loader, test_loader

    if spec.name == "TinyImageNet":
        train_dir = data_root / "train"
        val_dir = data_root / "val"
        
        # Try auto-download if not exists
        if not train_dir.exists() or not val_dir.exists():
            print(f"\nTiny ImageNet not found at {data_root}")
            if not auto_download_tiny_imagenet(data_root):
                # If auto-download failed, show manual instructions
                error_msg = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    Tiny ImageNet Dataset Not Found                      ║
╚══════════════════════════════════════════════════════════════════════════╝

Automatic download failed. Please download manually:

Expected directory structure:
  {data_root}/
    ├── train/           # 100,000 training images (200 classes, 500 each)
    └── val/             # 10,000 validation images

Manual download instructions:

1. Download the dataset:
   wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
   
   OR visit: http://cs231n.stanford.edu/tiny-imagenet-200.zip

2. Extract and prepare:
   unzip tiny-imagenet-200.zip
   mv tiny-imagenet-200/train {data_root}/train
   mv tiny-imagenet-200/val {data_root}/val

3. For Windows (PowerShell):
   Invoke-WebRequest -Uri "http://cs231n.stanford.edu/tiny-imagenet-200.zip" -OutFile "tiny-imagenet-200.zip"
   Expand-Archive -Path "tiny-imagenet-200.zip" -DestinationPath "."
   Move-Item "tiny-imagenet-200\\train" "{data_root}\\train"
   Move-Item "tiny-imagenet-200\\val" "{data_root}\\val"

Note: The dataset is ~237 MB compressed, ~500 MB uncompressed.

Alternatively, use a different dataset like CIFAR-10 or CIFAR-100:
  python examples/training/train_fractal_vit.py --dataset cifar10 --quick-test
"""
                raise FileNotFoundError(error_msg)
        
        # Verify directories exist after download attempt
        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError(f"Failed to prepare Tiny ImageNet at {data_root}")
        
        # Train set is standard ImageFolder
        train_dataset = ImageFolder(str(train_dir), transform=train_transform)
        
        # Val set needs custom loader to handle annotations file
        # We pass train_dataset.class_to_idx to ensure class mapping consistency
        val_dataset = TinyImageNetVal(val_dir, train_dataset.class_to_idx, transform=test_transform)
        
        # Tiny ImageNet 'test' set has no labels, so we use 'val' set for testing as well
        test_dataset = val_dataset 
        
        # Create loaders with optimized settings
        # Note: Tiny ImageNet Val is already a separate split, so we don't need to split train_dataset
        # unless user wants to carve out a validation set from train.
        # Standard practice: Train on 'train', Evaluate on 'val'.
        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": num_workers > 0,
            "prefetch_factor": 2 if num_workers > 0 else None,
        }
        
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
        
        print(f"Loaded {spec.name} → train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
        return train_loader, val_loader, test_loader

    if spec.name == "ImageNet":
        # Expects root to have train/val folders
        train_dir = data_root / "train"
        val_dir = data_root / "val"
        if not train_dir.exists() or not val_dir.exists():
            error_msg = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                     ImageNet Dataset Not Found                          ║
╚══════════════════════════════════════════════════════════════════════════╝

Expected directory structure:
  {data_root}/
    ├── train/           # 1,281,167 training images (1000 classes)
    └── val/             # 50,000 validation images

ImageNet requires registration and manual download:

1. Register and download from:
   https://image-net.org/download.php
   
2. Download ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar

3. Extract and organize:
   mkdir -p {data_root}/train {data_root}/val
   tar -xf ILSVRC2012_img_train.tar -C {data_root}/train
   tar -xf ILSVRC2012_img_val.tar -C {data_root}/val

Note: ImageNet is very large (~150 GB). Consider using:
  - Tiny ImageNet (64x64, 200 classes, ~500 MB)
  - CIFAR-100 (32x32, 100 classes, ~170 MB)
  - CIFAR-10 (32x32, 10 classes, ~170 MB)

Quick test with smaller dataset:
  python examples/training/train_fractal_vit.py --dataset cifar10 --quick-test
"""
            raise FileNotFoundError(error_msg)
        train_dataset = spec.dataset_cls(root=str(train_dir), transform=train_transform)
        test_dataset = spec.dataset_cls(root=str(val_dir), transform=test_transform)
    elif spec.name == "COCO":
        # Expects standard COCO structure
        train_img_dir = data_root / "train2017"
        val_img_dir = data_root / "val2017"
        train_ann = data_root / "annotations" / "instances_train2017.json"
        val_ann = data_root / "annotations" / "instances_val2017.json"
        
        if not train_img_dir.exists() or not train_ann.exists():
            error_msg = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                      COCO Dataset Not Found                              ║
╚══════════════════════════════════════════════════════════════════════════╝

Expected directory structure:
  {data_root}/
    ├── train2017/       # 118,287 training images
    ├── val2017/         # 5,000 validation images
    └── annotations/
        ├── instances_train2017.json
        └── instances_val2017.json

Download COCO 2017 dataset:

1. Download images and annotations:
   wget http://images.cocodataset.org/zips/train2017.zip
   wget http://images.cocodataset.org/zips/val2017.zip
   wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

2. Extract:
   unzip train2017.zip -d {data_root}
   unzip val2017.zip -d {data_root}
   unzip annotations_trainval2017.zip -d {data_root}

3. For Windows (PowerShell):
   Invoke-WebRequest -Uri "http://images.cocodataset.org/zips/train2017.zip" -OutFile "train2017.zip"
   Invoke-WebRequest -Uri "http://images.cocodataset.org/zips/val2017.zip" -OutFile "val2017.zip"
   Invoke-WebRequest -Uri "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" -OutFile "annotations.zip"
   Expand-Archive train2017.zip -DestinationPath "{data_root}"
   Expand-Archive val2017.zip -DestinationPath "{data_root}"
   Expand-Archive annotations.zip -DestinationPath "{data_root}"

Note: COCO is large (~25 GB). Consider starting with smaller datasets:
  python examples/training/train_fractal_vit.py --dataset cifar10 --quick-test
"""
            raise FileNotFoundError(error_msg)

        train_dataset = spec.dataset_cls(root=str(train_img_dir), annFile=str(train_ann), transform=train_transform)
        test_dataset = spec.dataset_cls(root=str(val_img_dir), annFile=str(val_ann), transform=test_transform)
    else:
        train_dataset = spec.dataset_cls(root=str(data_root), train=True, download=True, transform=train_transform)
        test_dataset = spec.dataset_cls(root=str(data_root), train=False, download=True, transform=test_transform)

    indices = np.arange(len(train_dataset))
    np.random.shuffle(indices)
    if subset_size is not None:
        indices = indices[:subset_size + max(1, int(subset_size * val_split))]

    val_count = max(1, int(len(indices) * val_split))
    if val_count >= len(indices):
        val_count = max(1, len(indices) - 1)

    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    if len(train_indices) == 0:
        train_indices = val_indices[:1]
        val_indices = val_indices[1:]

    train_loader = make_loader(train_dataset, train_indices)
    val_loader = make_loader(train_dataset, val_indices)
    test_loader = make_loader(test_dataset, np.arange(len(test_dataset)), shuffle=True)

    print(
        f"Loaded {spec.name} → train: {len(train_indices)}, val: {len(val_indices)}, test: {len(test_dataset)}"
    )
    return train_loader, val_loader, test_loader


def _get_bias_mode_description(mode: str) -> str:
    """获取 bias_mode 的描述信息。"""
    descriptions = {
        'original': 'Original full computation (O(S²) memory)',
        'low_rank': 'Low-rank factorization (O(S·r) memory, 5-8× faster, recommended)',
        'hierarchical': 'Hierarchical computation (O(S²·log L) time)',
    }
    return descriptions.get(mode, 'Unknown mode')


def build_model(args: argparse.Namespace, spec: DatasetSpec) -> nn.Module:
    """构建 NextGenerationFractalViT 模型。
    
    注意: SimpleFractalViT 已被移除，所有训练都使用 NextGenerationFractalViT。
    --use-simple 参数已废弃但仍被接受以保持向后兼容。
    """
    # 警告已废弃的参数
    if getattr(args, 'use_simple', False):
        print("⚠️  Warning: --use-simple is deprecated and will be ignored.")
        print("   All training now uses NextGenerationFractalViT.")
    
    model = NextGenerationFractalViT(
        image_size=max(spec.image_size, 32),
        num_classes=spec.num_classes,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.dim * 2,
        pool=args.pool,
        channels=spec.channels,
        dim_head=args.dim_head,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
        min_patch_size=(4, 4),
        max_level=args.max_level,
        learnable_split=not args.no_learnable_split,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: NextGenerationFractalViT")
    print(f"Parameters: total={total_params:,}, trainable={trainable_params:,}")
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    scaler: GradScalerType,
    gradient_clip: float,
    baseline_ema: Optional[float] = None,
    accum_steps: int = 1,
) -> Tuple[float, float, float]:
    """
    训练一个 epoch
    
    Args:
        baseline_ema: REINFORCE 基线的指数移动平均值，用于减少方差
        accum_steps: P1 优化 - 梯度累积步数，用于模拟更大 batch size
    
    Returns:
        (avg_loss, accuracy, updated_baseline_ema)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
    
    # 初始化基线 EMA
    if baseline_ema is None:
        baseline_ema = 0.0
    ema_decay = 0.99  # EMA 衰减系数

    for step, (data, target) in enumerate(progress):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # P1 优化：梯度累积 - 仅在累积周期开始时清零
        if step % accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)
        
        with autocast_context(device, scaler.is_enabled()):
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            ce_loss = F.cross_entropy(output, target)
            
            # 添加 Tokenizer 辅助 Loss (REINFORCE + 正则化)
            aux_loss = torch.tensor(0.0, device=device)
            if hasattr(model, "get_tokenizer_loss"):
                # 计算 REINFORCE 奖励：使用负损失作为奖励
                # 损失越低 -> 奖励越高 -> 鼓励当前的分割策略
                with torch.no_grad():
                    reward = -ce_loss.item()
                    # 更新基线 EMA
                    baseline_ema = ema_decay * baseline_ema + (1 - ema_decay) * reward
                
                # 获取策略梯度损失
                aux_loss = model.get_tokenizer_loss(
                    reward=reward,
                    baseline=baseline_ema,
                    entropy_coef=0.01,
                )
                
                # 清空 tokenizer 缓存，为下一个 batch 做准备
                if hasattr(model, "clear_tokenizer_cache"):
                    model.clear_tokenizer_cache()
            
            # P1 优化：梯度累积 - 损失除以累积步数
            loss = (ce_loss + aux_loss) / accum_steps

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            # P1 优化：仅在累积周期结束时更新
            if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
                if gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            # P1 优化：仅在累积周期结束时更新
            if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

        # 记录未缩放的损失用于监控
        running_loss += loss.item() * accum_steps
        preds = output.argmax(dim=1)
        correct += preds.eq(target).sum().item()
        total += target.size(0)
        progress.set_postfix(loss=f"{loss.item() * accum_steps:.4f}", acc=f"{100.0 * correct / max(total, 1):.1f}%")

    avg_loss = running_loss / max(len(loader), 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy, baseline_ema


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str,
    use_amp: bool = False,
) -> Tuple[float, float]:
    """P1 优化：评估阶段支持 AMP，加速 20-30%"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(loader, desc=desc, leave=False):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # P1 优化：评估时也使用 AMP
            with autocast_context(device, use_amp):
                output = model(data)
                if isinstance(output, tuple):
                    output = output[0]
                loss = F.cross_entropy(output, target)
            
            running_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += target.size(0)

    avg_loss = running_loss / max(len(loader), 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


def get_detailed_config(args: argparse.Namespace, model: nn.Module, device: torch.device) -> Dict[str, Any]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    config = {
        "arguments": vars(args),
        "model_statistics": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_class": model.__class__.__name__,
        },
        "system_information": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    if torch.cuda.is_available():
        config["system_information"]["cuda_device_name"] = torch.cuda.get_device_name(0)

    return config


def save_history(
    paths: ExperimentPaths,
    config: Dict[str, Any],
    history: Dict[str, list],
    summary: Dict[str, float],
) -> None:
    payload = {
        "config": config,
        "history": history,
        "summary": summary,
        "timestamp": paths.timestamp,
    }
    history_path = paths.experiment_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    # Also save detailed config to logs
    logs_config_path = paths.logs_dir / "experiment_config.json"
    with logs_config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    workspace_summary_path = paths.workspace_results_dir / f"fractal_vit_simple_{paths.timestamp}.json"
    with workspace_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def save_best_checkpoint(paths: ExperimentPaths, checkpoint: Dict[str, object]) -> None:
    best_path = paths.checkpoints_dir / "best.pth"
    torch.save(checkpoint, best_path)

    workspace_path = paths.workspace_models_dir / f"fractal_vit_simple_best_{paths.timestamp}.pth"
    torch.save(checkpoint, workspace_path)


def plot_curves(paths: ExperimentPaths, history: Dict[str, list]) -> None:
    if len(history["train_loss"]) < 2:
        return
    try:
        import matplotlib.pyplot as plt

        epochs = range(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        axes[0].plot(epochs, history["train_loss"], label="train")
        axes[0].plot(epochs, history["val_loss"], label="val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("epoch")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(epochs, history["train_acc"], label="train")
        axes[1].plot(epochs, history["val_acc"], label="val")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("epoch")
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()
        exp_path = paths.visuals_dir / "training_curves.png"
        plt.savefig(exp_path, dpi=150, bbox_inches="tight")
        workspace_path = paths.workspace_visuals_dir / f"fractal_vit_simple_curves_{paths.timestamp}.png"
        plt.savefig(workspace_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        print("matplotlib is not installed, skipping curve export")


def plot_tokenization_analysis(
    paths: ExperimentPaths,
    monitor: AdaptiveTokenMonitor,
    final_analysis: Dict[str, Any],
) -> None:
    """绘制自适应 tokenization 分析图表。"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Token 数量随 epoch 变化
        epoch_summaries = monitor.get_full_report().get("epoch_summaries", [])
        if epoch_summaries:
            epochs = [s["epoch"] for s in epoch_summaries]
            avg_tokens = [s["avg_tokens_per_image"] for s in epoch_summaries]
            min_tokens = [s["min_tokens"] for s in epoch_summaries]
            max_tokens = [s["max_tokens"] for s in epoch_summaries]
            
            ax = axes[0, 0]
            ax.fill_between(epochs, min_tokens, max_tokens, alpha=0.3, label="Token Range")
            ax.plot(epochs, avg_tokens, "b-", linewidth=2, label="Avg Tokens")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Tokens per Image")
            ax.set_title("Token Count Over Training")
            ax.legend()
            ax.grid(True)
        
        # 2. Simple vs Complex image token comparison
        ax = axes[0, 1]
        simple = final_analysis.get("simple_image_tokens", 0)
        complex_ = final_analysis.get("complex_image_tokens", 0)
        bars = ax.bar(["Simple\n(Uniform)", "Complex\n(Noise)"], [simple, complex_], color=["green", "red"])
        ax.set_ylabel("Token Count")
        ax.set_title("Adaptivity: Simple vs Complex Images")
        for bar, val in zip(bars, [simple, complex_]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   str(val), ha='center', va='bottom', fontsize=12)
        
        # 3. Image complexity vs Token count scatter plot
        ax = axes[1, 0]
        if monitor.complexity_correlation:
            complexities = [c[0] for c in monitor.complexity_correlation[-500:]]  # Last 500 samples
            token_counts = [c[1] for c in monitor.complexity_correlation[-500:]]
            ax.scatter(complexities, token_counts, alpha=0.5, s=10)
            ax.set_xlabel("Image Variance (Complexity Proxy)")
            ax.set_ylabel("Token Count")
            corr = monitor.compute_complexity_correlation()
            ax.set_title(f"Complexity-Token Correlation (r={corr:.3f})")
            ax.grid(True)
        
        # 4. Level usage distribution
        ax = axes[1, 1]
        if epoch_summaries:
            last_summary = epoch_summaries[-1]
            levels = last_summary.get("levels_used", [])
            if levels:
                # Count usage per level
                level_counts: Dict[int, int] = defaultdict(int)
                for stats in monitor.epoch_stats.get(last_summary["epoch"], []):
                    for level in stats.levels_used:
                        level_counts[level] += 1
                
                if level_counts:
                    sorted_levels = sorted(level_counts.keys())
                    counts = [level_counts[l] for l in sorted_levels]
                    ax.bar([f"Level {l}" for l in sorted_levels], counts)
                    ax.set_ylabel("Usage Count")
                    ax.set_title("Level Distribution (Last Epoch)")
                    ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存
        exp_path = paths.visuals_dir / "tokenization_analysis.png"
        plt.savefig(exp_path, dpi=150, bbox_inches="tight")
        workspace_path = paths.workspace_visuals_dir / f"tokenization_analysis_{paths.timestamp}.png"
        plt.savefig(workspace_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        print(f"Tokenization analysis saved to: {exp_path}")
        
    except ImportError:
        print("matplotlib is not installed, skipping tokenization analysis plot")


def main() -> None:
    # =========================================================================
    # Multiprocessing 配置
    # =========================================================================
    # 配置 multiprocessing 启动方法:
    # - Windows: 必须使用 'spawn'
    # - Linux/Mac: 优先使用 'fork'（更快），在 CUDA 环境下使用 'spawn'
    try:
        current_method = multiprocessing.get_start_method(allow_none=True)
        if current_method is None:
            # 在 Linux 上，fork 比 spawn 快得多，且更稳定
            if sys.platform == 'linux' and not torch.cuda.is_available():
                multiprocessing.set_start_method('fork')
            else:
                multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # 已经设置过了
    
    parser = argparse.ArgumentParser(description="Quick Fractal ViT trainer")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "mnist", "imagenet", "coco", "caltech256", "tiny-imagenet"], default="cifar10")
    parser.add_argument("--data-root", type=str, default=None, help="Path to dataset root directory")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--subset-size", type=int, default=None, help="Limit training samples for quick iterations")
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim-head", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--emb-dropout", type=float, default=0.1)
    parser.add_argument("--max-level", type=int, default=4)
    parser.add_argument("--pool", choices=["cls", "mean"], default="cls")
    parser.add_argument("--use-simple", action="store_true", help="[DEPRECATED] Ignored, always uses NextGenerationFractalViT")
    parser.add_argument("--no-learnable-split", action="store_true")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps for larger effective batch size")
    parser.add_argument("--num-workers", type=int, default=-1, help="Number of DataLoader workers. -1 for auto-detect.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bias-mode",
        choices=["original", "low_rank", "hierarchical"],
        default="low_rank",
        help="Hilbert bias computation mode: 'original' (high memory), 'low_rank' (memory efficient, default), 'hierarchical' (interpretable).",
    )
    parser.add_argument(
        "--low-rank-r",
        type=int,
        default=32,
        help="Rank for low-rank Hilbert bias factorization (only used when --bias-mode=low_rank).",
    )
    # --force-next-gen 已移除，现在始终使用 NextGenerationFractalViT
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Select training device; 'auto' prefers CUDA when available.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    # 自动检测最优 worker 数量
    if args.num_workers < 0:
        args.num_workers = get_optimal_num_workers()
        print(f"Auto-detected num_workers: {args.num_workers}")
    
    # 诊断 DataLoader 配置
    if args.num_workers > 0:
        _diagnose_dataloader_config(args.num_workers, args.batch_size)

    if args.quick_test:
        if args.epochs == parser.get_default("epochs"):
            args.epochs = 5
        if args.subset_size is None:
            args.subset_size = 512
        # quick-test 模式下使用单线程以避免 multiprocessing 问题
        args.num_workers = 0
        print("Quick-test mode: epochs capped, subset sampling enabled, single-threaded loading")

    device = resolve_device(args.device)

    # P0 优化：启用 cuDNN benchmark 模式（固定输入尺寸时加速 5-15%）
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        # 可选：启用 TF32 以获得更快的矩阵运算（Ampere+ GPU）
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        print("CUDA optimizations enabled: cudnn.benchmark=True, TF32=True")

    spec = get_dataset_spec(args.dataset)
    paths = prepare_experiment_paths("fractal_vit")

    # CPU 模式下自动降低模型复杂度以加速训练
    if device.type == "cpu":
        print("CPU detected; reducing model complexity for faster training.")
        args.dim = min(args.dim, 192)
        args.depth = min(args.depth, 6)
        args.heads = min(args.heads, 6)
        args.dim_head = min(args.dim_head, 32)
        print(f"  Adjusted: dim={args.dim}, depth={args.depth}, heads={args.heads}")

    model = build_model(args, spec)

    # Print bias mode information
    bias_mode_info = {
        "original": "Original (O(S²) memory, high accuracy baseline)",
        "low_rank": "Low-rank factorization (O(S·r) memory, 5-8× faster, recommended)",
        "hierarchical": "Hierarchical decomposition (O(S·L) memory, interpretable)",
    }
    print(f"Using Hilbert bias mode: {args.bias_mode} - {bias_mode_info.get(args.bias_mode, 'Unknown')}")
    if args.bias_mode == "low_rank":
        print(f"Low-rank factorization rank: {args.low_rank_r}")

    model = model.to(device)

    # P0 优化：torch.compile() 加速（PyTorch 2.0+，预期 15-40% 加速）
    # 注意：compiled_model 用于前向传播，原始 model 用于保存状态等
    compiled_model = None
    if hasattr(torch, 'compile') and device.type == 'cuda' and not args.quick_test:
        try:
            compiled_model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled with torch.compile(mode='reduce-overhead')")
        except Exception as e:
            print(f"torch.compile() failed, using eager mode: {e}")
            compiled_model = None
    
    # 使用编译模型进行训练（如果可用）
    # type: ignore 用于抑制 torch.compile 返回类型的静态分析警告
    train_model: nn.Module = compiled_model if compiled_model is not None else model  # type: ignore[assignment]

    # Generate detailed config
    detailed_config = get_detailed_config(args, model, device)
    # Save immediately to logs for reference
    with (paths.logs_dir / "experiment_config.json").open("w", encoding="utf-8") as f:
        json.dump(detailed_config, f, indent=2)

    train_loader, val_loader, test_loader = create_dataloaders(
        spec,
        batch_size=args.batch_size,
        val_split=args.val_split,
        subset_size=args.subset_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        data_root_override=args.data_root,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    
    # Warmup + Cosine Annealing
    warmup_epochs = min(5, args.epochs // 2)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=args.lr * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
    scaler = create_grad_scaler(args.use_amp, device)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_state: Optional[Dict[str, object]] = None
    baseline_ema: Optional[float] = None  # REINFORCE 基线
    
    # 初始化自适应 tokenization 监控器
    token_monitor = AdaptiveTokenMonitor()
    monitor_interval = max(1, args.epochs // 10)  # 每 10% 的 epochs 监控一次

    start_time = time.time()
    interrupted = False  # 跟踪是否被中断
    
    # 训练前分析初始 tokenization 特性
    print("\n=== 训练前 Tokenization 分析 ===")
    initial_analysis = analyze_adaptive_tokenization(train_model, val_loader, device, num_samples=50)
    print(f"简单图像 tokens: {initial_analysis.get('simple_image_tokens', 'N/A')}")
    print(f"复杂图像 tokens: {initial_analysis.get('complex_image_tokens', 'N/A')}")
    print(f"自适应性: {'[YES]' if initial_analysis.get('is_adaptive') else '[NO]'}")
    print(f"自适应比率: {initial_analysis.get('adaptivity_ratio', 0):.2f}x")
    print()
    
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc, baseline_ema = train_one_epoch(
                train_model,
                train_loader,
                optimizer,
                device,
                epoch,
                args.epochs,
            scaler,
            args.gradient_clip,
            baseline_ema=baseline_ema,
            accum_steps=args.accum_steps,
        )
        val_loss, val_acc = evaluate(train_model, val_loader, device, desc="val", use_amp=args.use_amp)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{args.epochs} | LR: {current_lr:.2e} | "
            f"Train: loss={train_loss:.4f}, acc={train_acc:.2f}% | "
            f"Val: loss={val_loss:.4f}, acc={val_acc:.2f}%"
        )
        
        # 定期监控 tokenization 特性
        if epoch % monitor_interval == 0 or epoch == 1:
            # 在验证集上采样分析
            sample_batch = next(iter(val_loader))[0].to(device)
            stats = token_monitor.analyze_batch(train_model, sample_batch, epoch)
            print(f"  [Token Monitor] avg={stats.avg_tokens_per_image:.1f}, "
                  f"range=[{stats.min_tokens}-{stats.max_tokens}], "
                  f"levels={stats.levels_used}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_acc": best_val_acc,
                "args": vars(args),
            }
            save_best_checkpoint(paths, best_state)
            print(f"New best validation accuracy: {best_val_acc:.2f}% (epoch {epoch})")

    except KeyboardInterrupt:
        # =========================================================================
        # 优雅中断处理：Ctrl+C 时保存当前状态
        # =========================================================================
        interrupted = True
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("⚠️  训练被中断 (Ctrl+C)")
        print("=" * 70)
        print("正在保存当前训练状态...")
        
        # 保存中断时的检查点
        interrupt_checkpoint = {
            "epoch": len(history["train_loss"]),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_acc": best_val_acc,
            "args": vars(args),
            "interrupted": True,
        }
        interrupt_path = paths.checkpoints_dir / "interrupted.pth"
        torch.save(interrupt_checkpoint, interrupt_path)
        print(f"  ✓ 中断检查点已保存: {interrupt_path}")
        
        # 保存训练历史
        if history["train_loss"]:
            summary = {
                "best_val_acc": best_val_acc,
                "test_acc": None,  # 未完成测试
                "epochs_ran": len(history["train_loss"]),
                "elapsed_seconds": elapsed,
                "interrupted": True,
                "tokenization_analysis": {
                    "initial": initial_analysis,
                    "final": None,
                },
            }
            save_history(paths, detailed_config, history, summary)
            print(f"  ✓ 训练历史已保存")
            
            # 尝试绘制曲线
            try:
                plot_curves(paths, history)
                print(f"  ✓ 训练曲线已保存")
            except Exception:
                pass
        
        print(f"\n训练统计:")
        print(f"  已完成 epochs: {len(history['train_loss'])}/{args.epochs}")
        print(f"  最佳验证准确率: {best_val_acc:.2f}%")
        print(f"  已用时间: {elapsed:.1f}s")
        print(f"  实验目录: {paths.experiment_dir}")
        print("\n要恢复训练，请从检查点加载模型状态。")
        print("=" * 70)
        sys.exit(0)

    elapsed = time.time() - start_time
    if best_state is not None:
        model.load_state_dict(best_state["model"])  # type: ignore[arg-type]

    test_loss, test_acc = evaluate(train_model, test_loader, device, desc="test", use_amp=args.use_amp)
    
    # 训练后分析 tokenization 特性
    print("\n=== 训练后 Tokenization 分析 ===")
    final_analysis = analyze_adaptive_tokenization(train_model, val_loader, device, num_samples=50)
    print(f"简单图像 tokens: {final_analysis.get('simple_image_tokens', 'N/A')}")
    print(f"复杂图像 tokens: {final_analysis.get('complex_image_tokens', 'N/A')}")
    print(f"真实图像 tokens: 均值={final_analysis.get('real_image_tokens_mean', 0):.1f}, "
          f"范围=[{final_analysis.get('real_image_tokens_min', 0)}-{final_analysis.get('real_image_tokens_max', 0)}]")
    print(f"自适应性: {'[YES]' if final_analysis.get('is_adaptive') else '[NO]'}")
    print(f"自适应比率: {final_analysis.get('adaptivity_ratio', 0):.2f}x")
    
    # 生成完整监控报告
    monitor_report = token_monitor.get_full_report()
    print(f"\n=== 自适应 Tokenization 诊断 ===")
    print(monitor_report.get("diagnosis", "无诊断信息"))
    print(f"复杂度-Token 相关系数: {monitor_report.get('complexity_token_correlation', 0):.3f}")
    
    summary = {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "epochs_ran": len(history["train_loss"]),
        "elapsed_seconds": elapsed,
        "tokenization_analysis": {
            "initial": initial_analysis,
            "final": final_analysis,
            "complexity_correlation": monitor_report.get("complexity_token_correlation", 0),
            "is_working_as_expected": monitor_report.get("is_working_as_expected", False),
        },
    }
    save_history(paths, detailed_config, history, summary)
    plot_curves(paths, history)
    
    # 绘制 tokenization 分析图表
    plot_tokenization_analysis(paths, token_monitor, final_analysis)

    print("\nTraining finished")
    print(f"Best val accuracy: {best_val_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Artifacts: {paths.experiment_dir}")


if __name__ == "__main__":
    # 对于使用 spawn 的 multiprocessing，必须有 __main__ 保护
    # 这可以防止在 worker 进程中重新执行主代码
    multiprocessing.freeze_support()  # Windows 支持
    main()