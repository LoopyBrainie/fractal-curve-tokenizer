#!/usr/bin/env python3
"""Fractal ViT Training Script - V3 Variable Depth Tokens

特性：
1. StreamingFractalTokenizerV3：Variable Depth Tokens 自适应多尺度 (唯一支持)
   - 使用 AdaptiveQuadtreeSplit 进行内容自适应分割
   - 共享卷积特征提取 + 深度编码 + ROI-Align 池化
2. SwiGLU FFN：现代化前馈网络
3. Hilbert 曲线重排序：保持空间局部性
4. AMP 混合精度训练

V3 优势：
- 密集梯度流：共享特征提取器所有路径都收到梯度
- 无温度参数：训练更稳定
- 自适应分割：根据图像内容动态决定分割深度
- 深度编码：Token 携带尺度信息

使用示例：
    # CIFAR-10 快速测试
    python train_fractal_vit.py --quick-test --use-amp
    
    # Tiny ImageNet 完整训练 (推荐配置)
    python train_fractal_vit.py --dataset tiny-imagenet --epochs 100 --dim 256 \
        --depth 8 --heads 8 --dropout 0.1 --drop-path 0.1 --use-amp

注意：V1 和 V2 已从代码库中完全删除，当前仅支持 streaming_v3。
"""

from __future__ import annotations

import os
import sys
import platform
import multiprocessing as _mp

# 强制 spawn 方法（CUDA + 容器必需）
try:
    _mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# CUDA 内存优化
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512,expandable_segments:True')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

# 抑制 torch.compile 的符号形状警告和 checkpoint autocast 废弃警告
import warnings
warnings.filterwarnings('ignore', message='.*is not in var_ranges.*')
warnings.filterwarnings('ignore', message='.*defaulting to unknown range.*')
warnings.filterwarnings('ignore', message='.*torch.cpu.amp.autocast.*is deprecated.*', category=FutureWarning)

import argparse
import json
import multiprocessing
import random
import shutil
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# 分层采样
try:
    from sklearn.model_selection import StratifiedShuffleSplit
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def stratified_split(
    indices: np.ndarray, 
    labels: np.ndarray, 
    val_ratio: float, 
    seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """手动实现分层划分 (不依赖 sklearn).
    
    确保每个类别在训练/验证集中的比例相同。
    """
    np.random.seed(seed)
    unique_classes = np.unique(labels)
    train_idx_list = []
    val_idx_list = []
    
    for c in unique_classes:
        class_indices = indices[labels == c]
        np.random.shuffle(class_indices)
        
        val_count = max(1, int(len(class_indices) * val_ratio))
        val_idx_list.append(class_indices[:val_count])
        train_idx_list.append(class_indices[val_count:])
    
    train_idx = np.concatenate(train_idx_list)
    val_idx = np.concatenate(val_idx_list)
    
    # 再次打乱
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    
    return train_idx, val_idx

import logging
# 抑制 torch.compile 的符号形状警告
logging.getLogger('torch.fx.experimental.symbolic_shapes').setLevel(logging.ERROR)
logging.getLogger('torch._dynamo').setLevel(logging.ERROR)

# AMP 兼容层
try:
    from torch.amp import autocast, GradScaler
    _NEW_AMP = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler  # type: ignore
    _NEW_AMP = False

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vit_pytorch import FractalCurveViT


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class DatasetSpec:
    """数据集规格"""
    name: str
    num_classes: int
    image_size: int
    channels: int
    mean: Tuple[float, ...]
    std: Tuple[float, ...]


@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据集
    dataset: str
    batch_size: int
    num_workers: int
    val_split: float
    subset_size: Optional[int]
    
    # 模型
    dim: int
    depth: int
    heads: int
    mlp_dim: int
    dim_head: int
    max_level: int
    num_scales: int
    pool: str
    ffn_type: str
    hilbert_bias_mode: str
    
    # Tokenizer 配置 (V3 Variable Depth Tokens)
    tokenizer_type: str  # 'streaming_v3' (唯一支持)
    
    # V3 高级分割参数 (Adaptive Quadtree Split)
    split_scheme: str  # 'balanced_greedy' 或 'fixed_budget_dp'
    target_tokens: Optional[int]  # 目标 token 数量
    complexity_alpha: float  # 复杂度函数方差权重 α ∈ [0,1]
    split_tau0: float  # 根节点阈值 τ₀
    split_gamma: float  # 阈值衰减因子 γ
    enforce_balance: bool  # 是否强制 2:1 平衡约束
    domain_preset: Optional[str]  # 域适应预设
    
    # 训练
    epochs: int
    learning_rate: float
    weight_decay: float
    dropout: float
    emb_dropout: float
    drop_path: float
    label_smoothing: float
    gradient_clip: float
    use_amp: bool
    accum_steps: int
    warmup_epochs: int
    gradient_checkpoint: bool
    compile_model: bool
    channels_last: bool
    
    # 早停
    patience: int
    min_delta: float
    
    # Mixup/CutMix
    mixup_alpha: float
    cutmix_alpha: float
    mixup_prob: float
    
    # 系统
    seed: int
    device: str


# ============================================================================
# Mixup/CutMix 实现
# ============================================================================

class MixupCutmix:
    """Mixup 和 CutMix 数据增强
    
    参考: 
    - Mixup: https://arxiv.org/abs/1710.09412
    - CutMix: https://arxiv.org/abs/1905.04899
    """
    
    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        prob: float = 0.5,
        num_classes: int = 10,
        label_smoothing: float = 0.0,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
    
    def __call__(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用 Mixup 或 CutMix
        
        Args:
            images: [B, C, H, W] 图像张量
            labels: [B] 标签张量
            
        Returns:
            mixed_images: 混合后的图像
            mixed_labels: 混合后的 one-hot 标签 [B, num_classes]
        """
        batch_size = images.size(0)
        device = images.device
        
        # 检查 label 范围，防止越界
        assert labels.min() >= 0, f"Label 包含负值: min={labels.min().item()}"
        assert labels.max() < self.num_classes, f"Label 越界: max={labels.max().item()} >= {self.num_classes}"
        
        # 转换为 one-hot 并应用 label smoothing
        labels_one_hot = F.one_hot(labels, self.num_classes).float()
        if self.label_smoothing > 0:
            labels_one_hot = labels_one_hot * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
        
        # 随机决定是否应用增强
        if random.random() > self.prob:
            return images, labels_one_hot
        
        # 随机选择 Mixup 或 CutMix
        use_cutmix = random.random() > 0.5 and self.cutmix_alpha > 0
        
        if use_cutmix:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1.0
        
        # 随机打乱索引
        index = torch.randperm(batch_size, device=device)
        
        if use_cutmix:
            # CutMix: 随机裁剪区域
            _, _, H, W = images.shape
            cut_h = int(H * np.sqrt(1 - lam))
            cut_w = int(W * np.sqrt(1 - lam))
            
            cx = random.randint(0, W)
            cy = random.randint(0, H)
            
            x1 = max(0, cx - cut_w // 2)
            x2 = min(W, cx + cut_w // 2)
            y1 = max(0, cy - cut_h // 2)
            y2 = min(H, cy + cut_h // 2)
            
            mixed_images = images.clone()
            mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
            
            # 重新计算 lambda 基于实际裁剪区域
            lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
        else:
            # Mixup: 线性混合
            mixed_images = lam * images + (1 - lam) * images[index]
        
        # 混合标签
        mixed_labels = lam * labels_one_hot + (1 - lam) * labels_one_hot[index]
        
        return mixed_images, mixed_labels


def mixup_criterion(
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """计算 Mixup/CutMix 的交叉熵损失
    
    Args:
        outputs: [B, C] 模型输出 logits
        targets: [B, C] one-hot 或 soft 标签
        
    Returns:
        损失标量
    """
    # 检查 logits 是否包含 NaN/Inf
    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
        raise ValueError(f"Logits 包含 NaN/Inf: nan={torch.isnan(outputs).sum()}, inf={torch.isinf(outputs).sum()}")
    
    # 数值稳定的 log_softmax
    log_probs = F.log_softmax(outputs, dim=1)
    
    # 确保 targets 归一化且非负
    targets = targets.clamp(min=0)
    targets = targets / (targets.sum(dim=1, keepdim=True) + 1e-8)
    
    loss = -(targets * log_probs).sum(dim=1).mean()
    
    # 检查 loss 是否为 NaN
    if torch.isnan(loss):
        raise ValueError("Loss 为 NaN，可能是 logits 过大或标签问题")
    
    return loss


# ============================================================================
# 数据集配置
# ============================================================================

DATASETS = {
    "cifar10": DatasetSpec("CIFAR10", 10, 32, 3, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": DatasetSpec("CIFAR100", 100, 32, 3, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "mnist": DatasetSpec("MNIST", 10, 28, 1, (0.1307,), (0.3081,)),
    "tiny-imagenet": DatasetSpec("TinyImageNet", 200, 64, 3, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


# ============================================================================
# 环境检测
# ============================================================================

def detect_environment() -> Dict[str, Any]:
    """检测运行环境"""
    env = {
        'in_container': os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv'),
        'platform': platform.system(),
        'cpu_count': multiprocessing.cpu_count(),
        'recommended_workers': 4,
    }
    
    if env['platform'] == 'Linux':
        try:
            import psutil
            shm = psutil.disk_usage('/dev/shm')
            shm_gb = shm.total / (1024**3)
            if shm_gb >= 8:
                env['recommended_workers'] = min(12, env['cpu_count'])
            elif shm_gb >= 4:
                env['recommended_workers'] = min(8, env['cpu_count'])
        except ImportError:
            pass
    
    return env


def print_environment_info(env: Dict[str, Any]) -> None:
    """打印环境信息"""
    print("\n" + "="*70)
    print("Environment")
    print("="*70)
    print(f"  Platform: {env['platform']}")
    print(f"  Container: {'Yes' if env['in_container'] else 'No'}")
    print(f"  CPU Cores: {env['cpu_count']}")
    print(f"  Recommended Workers: {env['recommended_workers']}")
    print("="*70 + "\n")


# ============================================================================
# 工具函数
# ============================================================================

def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_amp_context(device: torch.device, enabled: bool):
    """获取 AMP autocast 上下文"""
    if _NEW_AMP:
        return autocast('cuda', enabled=enabled)
    else:
        return autocast(enabled=enabled)


def create_grad_scaler(enabled: bool) -> GradScaler:
    """创建 GradScaler"""
    if _NEW_AMP:
        return GradScaler('cuda', enabled=enabled)
    else:
        return GradScaler(enabled=enabled)


# ============================================================================
# Tiny ImageNet 下载
# ============================================================================

def download_with_progress(url: str, dest: Path, desc: str = "Downloading") -> bool:
    """带进度条的下载函数"""
    import urllib.request
    
    try:
        # 获取文件大小
        with urllib.request.urlopen(url, timeout=30) as response:
            total_size = int(response.headers.get('Content-Length', 0))
        
        # 下载
        downloaded = 0
        block_size = 8192
        
        with urllib.request.urlopen(url, timeout=30) as response:
            with open(dest, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        f.write(buffer)
                        downloaded += len(buffer)
                        pbar.update(len(buffer))
        
        return True
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_tiny_imagenet(data_root: Path) -> bool:
    """下载并设置 Tiny ImageNet
    
    数据集信息:
    - 200 类，每类 500 张训练图像
    - 训练集: 100,000 张 64x64 图像
    - 验证集: 10,000 张图像
    - 测试集: 10,000 张图像（无标签）
    
    下载源:
    - 主源: Stanford CS231n
    - 大小: ~237MB
    """
    target_dir = data_root / "tiny-imagenet-200"
    
    # 检查是否已存在
    if (target_dir / "train").exists() and (target_dir / "val").exists():
        train_classes = len(list((target_dir / "train").iterdir()))
        val_has_classes = any((target_dir / "val").iterdir())
        if train_classes >= 200 and val_has_classes:
            print(f"[OK] Tiny ImageNet already exists at {target_dir}")
            return True
    
    print("\n" + "="*60)
    print("Downloading Tiny ImageNet Dataset")
    print("="*60)
    print(f"  Target: {target_dir}")
    print(f"  Size: ~237MB")
    print("="*60 + "\n")
    
    zip_path = data_root / "tiny-imagenet-200.zip"
    
    # 检查已缓存的 zip 是否有效
    if zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # 验证 zip 文件
                if zf.testzip() is not None:
                    raise zipfile.BadZipFile("Corrupted zip file")
                if len(zf.namelist()) < 100:  # Tiny ImageNet 应该有很多文件
                    raise zipfile.BadZipFile("Incomplete zip file")
            print(f"[OK] Using cached zip: {zip_path}")
        except (zipfile.BadZipFile, Exception) as e:
            print(f"[WARN] Cached zip is invalid: {e}")
            print("[*] Removing corrupted file and re-downloading...")
            zip_path.unlink()
    
    # 尝试多个下载源
    urls = [
        "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
        "https://image-net.org/data/tiny-imagenet-200.zip",
    ]
    
    if not zip_path.exists():
        download_success = False
        for i, url in enumerate(urls):
            print(f"[{i+1}/{len(urls)}] Trying: {url}")
            if download_with_progress(url, zip_path, "Tiny ImageNet"):
                # 验证下载的文件
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        if zf.testzip() is not None:
                            raise zipfile.BadZipFile("Downloaded file is corrupted")
                    download_success = True
                    print("[OK] Download complete and verified")
                    break
                except zipfile.BadZipFile as e:
                    print(f"[WARN] Downloaded file is invalid: {e}")
                    if zip_path.exists():
                        zip_path.unlink()
            print(f"[WARN] Failed, trying next source...")
        
        if not download_success:
            print("\n[ERROR] All download sources failed.")
            print("Please download manually from:")
            print("  http://cs231n.stanford.edu/tiny-imagenet-200.zip")
            print(f"And place it at: {zip_path}")
            return False
    
    # 解压
    print("\nExtracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            total = len(zf.namelist())
            with tqdm(total=total, desc="Extracting", unit="files") as pbar:
                for member in zf.namelist():
                    zf.extract(member, data_root)
                    pbar.update(1)
        print("[OK] Extraction complete")
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        return False
    
    # 组织验证集（原始格式是所有图片在一个文件夹）
    val_dir = target_dir / "val"
    val_images_dir = val_dir / "images"
    
    if val_images_dir.exists():
        print("\nOrganizing validation set by class...")
        val_annotations = val_dir / "val_annotations.txt"
        
        if val_annotations.exists():
            # 读取标注
            with open(val_annotations, 'r') as f:
                lines = f.readlines()
            
            # 按类别组织
            for line in tqdm(lines, desc="Organizing"):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_name, class_id = parts[0], parts[1]
                    class_dir = val_dir / class_id / "images"
                    class_dir.mkdir(parents=True, exist_ok=True)
                    src = val_images_dir / img_name
                    dst = class_dir / img_name
                    if src.exists() and not dst.exists():
                        shutil.move(str(src), str(dst))
            
            # 删除原始 images 文件夹
            if val_images_dir.exists():
                shutil.rmtree(val_images_dir)
            
            print("[OK] Validation set organized")
        else:
            print("[WARN] val_annotations.txt not found")
    
    # 验证
    train_classes = len(list((target_dir / "train").iterdir()))
    val_classes = len([d for d in (target_dir / "val").iterdir() if d.is_dir()])
    print(f"\n[OK] Dataset ready:")
    print(f"  Train classes: {train_classes}")
    print(f"  Val classes: {val_classes}")
    
    # 清理 zip
    if zip_path.exists():
        zip_path.unlink()
        print("[OK] Cleaned up zip file")
    
    return True


# ============================================================================
# 数据加载
# ============================================================================

def create_dataloaders(
    spec: DatasetSpec,
    config: TrainingConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器"""
    
    # 数据增强
    if spec.name == "MNIST":
        train_tf = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
        ])
    elif spec.name == "TinyImageNet":
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=8),
            transforms.RandAugment(num_ops=2, magnitude=9),  # Phase 2: RandAugment
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
            transforms.RandomErasing(p=0.25),  # Cutout-like augmentation
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(spec.image_size, padding=4),
            transforms.RandAugment(num_ops=2, magnitude=9),  # Phase 2: RandAugment
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
            transforms.RandomErasing(p=0.25),  # Cutout-like augmentation
        ])
    
    test_tf = transforms.Compose([
        transforms.Resize(max(spec.image_size, 32)),
        transforms.ToTensor(),
        transforms.Normalize(spec.mean, spec.std),
    ])
    
    # 加载数据
    data_root = PROJECT_ROOT / "data"
    data_root.mkdir(exist_ok=True)
    
    if spec.name == "CIFAR10":
        train_ds = datasets.CIFAR10(data_root, train=True, download=True, transform=train_tf)
        test_ds = datasets.CIFAR10(data_root, train=False, download=True, transform=test_tf)
    elif spec.name == "CIFAR100":
        train_ds = datasets.CIFAR100(data_root, train=True, download=True, transform=train_tf)
        test_ds = datasets.CIFAR100(data_root, train=False, download=True, transform=test_tf)
    elif spec.name == "MNIST":
        train_ds = datasets.MNIST(data_root, train=True, download=True, transform=train_tf)
        test_ds = datasets.MNIST(data_root, train=False, download=True, transform=test_tf)
    elif spec.name == "TinyImageNet":
        if not download_tiny_imagenet(data_root):
            raise FileNotFoundError("Failed to download Tiny ImageNet")
        train_dir = data_root / "tiny-imagenet-200" / "train"
        test_dir = data_root / "tiny-imagenet-200" / "val"
        train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
        test_ds = datasets.ImageFolder(str(test_dir), transform=test_tf)
    else:
        raise ValueError(f"Unknown dataset: {spec.name}")
    
    # 划分 (使用分层采样确保类别平衡)
    n_samples = len(train_ds)
    indices = np.arange(n_samples)
    
    # 获取所有标签用于分层采样
    if hasattr(train_ds, 'targets'):
        all_labels = np.array(train_ds.targets)
    elif hasattr(train_ds, 'labels'):
        all_labels = np.array(train_ds.labels)
    else:
        # ImageFolder 需要遍历
        all_labels = np.array([train_ds.samples[i][1] for i in range(n_samples)])
    
    if config.subset_size:
        indices = indices[:config.subset_size]
        all_labels = all_labels[:config.subset_size]
    
    # 分层采样：确保训练/验证集中每个类别比例相同
    num_classes = len(np.unique(all_labels))
    if num_classes > 1:
        if HAS_SKLEARN:
            val_size = max(1, int(len(indices) * config.val_split))
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=config.seed)
            train_idx, val_idx = next(sss.split(indices, all_labels[indices] if config.subset_size else all_labels))
            train_idx = indices[train_idx]
            val_idx = indices[val_idx]
            print(f"[OK] 使用 sklearn 分层采样划分训练/验证集")
        else:
            # 使用手动分层划分
            train_idx, val_idx = stratified_split(indices, all_labels, config.val_split, config.seed)
            print(f"[OK] 使用手动分层采样划分训练/验证集 (确保类别平衡)")
    else:
        # 单类别数据集 - 简单随机划分
        np.random.shuffle(indices)
        val_size = max(1, int(len(indices) * config.val_split))
        train_idx, val_idx = indices[val_size:], indices[:val_size]
    
    # DataLoader 参数
    mp_context = 'spawn' if config.num_workers > 0 else None
    loader_kwargs = {
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'pin_memory': config.num_workers > 0,
        'multiprocessing_context': mp_context,
        'persistent_workers': config.num_workers > 1,
    }
    if config.num_workers > 0:
        loader_kwargs['prefetch_factor'] = 4
    
    train_loader = DataLoader(train_ds, sampler=SubsetRandomSampler(train_idx), **loader_kwargs)
    val_loader = DataLoader(train_ds, sampler=SubsetRandomSampler(val_idx), **loader_kwargs)
    
    test_kwargs = loader_kwargs.copy()
    test_kwargs['shuffle'] = False
    test_loader = DataLoader(test_ds, **test_kwargs)
    
    print(f"[OK] Data: train={len(train_idx)}, val={len(val_idx)}, test={len(test_ds)}")
    
    return train_loader, val_loader, test_loader


# ============================================================================
# CUDA Prefetcher
# ============================================================================

class CudaPrefetcher:
    """CUDA 异步数据预取器"""
    
    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
        
    def __iter__(self):
        self.loader_iter = iter(self.loader)
        self.preload()
        return self
    
    def preload(self):
        try:
            self.next_batch = next(self.loader_iter)
        except StopIteration:
            self.next_batch = None
            return
        
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.next_data = (
                    self.next_batch[0].to(self.device, non_blocking=True),
                    self.next_batch[1].to(self.device, non_blocking=True),
                )
        else:
            self.next_data = self.next_batch
    
    def __next__(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        
        if self.next_batch is None:
            raise StopIteration
        
        data = self.next_data
        self.preload()
        return data
    
    def __len__(self):
        return len(self.loader)


# ============================================================================
# 训练函数
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    config: TrainingConfig,
    mixup_fn: Optional[MixupCutmix] = None,
    num_classes: int = 10,
    profile: bool = False,
) -> Tuple[float, float, Dict[str, float]]:
    """训练一个 epoch"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    
    batch_times, data_times, forward_times = [], [], []
    entropy_losses = []  # P1-5: 收集熵损失用于统计
    cuda_mem_peak = 0.0
    use_mixup = mixup_fn is not None
    nan_count = 0  # NaN 计数器
    
    data_iter = CudaPrefetcher(loader, device) if device.type == 'cuda' else loader
    pbar = tqdm(data_iter, desc="Train", total=len(loader))
    data_start = time.time()
    
    for i, batch in enumerate(pbar):
        data_time = time.time() - data_start
        data_times.append(data_time)
        batch_start = time.time()
        
        imgs, labels = batch
        if device.type != 'cuda':
            imgs = imgs.to(device)
            labels = labels.to(device)
        
        # 检查 label 范围
        if labels.min() < 0 or labels.max() >= num_classes:
            print(f"\n[WARN] Label 范围异常: min={labels.min().item()}, max={labels.max().item()}, num_classes={num_classes}")
            continue
        
        # 应用 Mixup/CutMix
        mixed_labels: Optional[torch.Tensor] = None
        if use_mixup and mixup_fn is not None:
            imgs, mixed_labels = mixup_fn(imgs, labels)
        
        forward_start = time.time()
        with get_amp_context(device, config.use_amp):
            outs, _ = model(imgs, return_aux_info=True)
            
            # 检查 logits 范围，防止爆炸
            if torch.isnan(outs).any() or torch.isinf(outs).any():
                nan_count += 1
                if nan_count <= 3:
                    print(f"\n[WARN] Logits 包含 NaN/Inf (batch {i}), 跳过此 batch")
                if nan_count > 10:
                    raise RuntimeError(f"连续出现 {nan_count} 次 NaN，训练终止")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            if use_mixup and mixed_labels is not None:
                # 使用混合标签的交叉熵
                ce_loss = mixup_criterion(outs, mixed_labels) / config.accum_steps
            else:
                ce_loss = F.cross_entropy(outs, labels, label_smoothing=config.label_smoothing) / config.accum_steps
            
            # P1-5 修复: 收集熵正则化损失
            # 熵损失鼓励尺度分布多样性，防止 CrossScaleAttention 崩塌到单一尺度
            entropy_loss = None
            if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'get_entropy_loss'):
                entropy_loss = model.tokenizer.get_entropy_loss()
            
            if entropy_loss is not None:
                loss = ce_loss + entropy_loss / config.accum_steps
                entropy_losses.append(entropy_loss.item())  # P1-5: 记录熵损失
            else:
                loss = ce_loss
        
        # 检查 loss 是否为 NaN
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            if nan_count <= 3:
                print(f"\n[WARN] Loss 为 NaN/Inf (batch {i}), 跳过此 batch")
            if nan_count > 10:
                raise RuntimeError(f"连续出现 {nan_count} 次 NaN loss，训练终止")
            optimizer.zero_grad(set_to_none=True)
            continue
        
        nan_count = 0  # 重置计数器
        
        forward_time = time.time() - forward_start
        forward_times.append(forward_time)
        
        scaler.scale(loss).backward()
        
        if (i + 1) % config.accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item() * config.accum_steps
        _, pred = outs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        
        batch_times.append(time.time() - batch_start)
        
        if device.type == 'cuda':
            cuda_mem_peak = max(cuda_mem_peak, torch.cuda.max_memory_allocated() / 1024**3)
        
        if profile and (i < 5 or i % 100 == 0):
            pbar.set_postfix(
                loss=f'{loss.item()*config.accum_steps:.3f}',
                acc=f'{100.*correct/total:.1f}%',
                data=f'{data_time*1000:.0f}ms',
                fwd=f'{forward_time*1000:.0f}ms'
            )
        else:
            pbar.set_postfix(loss=f'{loss.item()*config.accum_steps:.4f}', acc=f'{100.*correct/total:.1f}%')
        
        data_start = time.time()
    
    perf_stats = {
        'avg_batch_time': np.mean(batch_times) if batch_times else 0,
        'avg_data_time': np.mean(data_times) if data_times else 0,
        'avg_forward_time': np.mean(forward_times) if forward_times else 0,
        'throughput': total / sum(batch_times) if batch_times else 0,
        'cuda_mem_peak_gb': cuda_mem_peak,
        # P1-5: 添加熵统计
        'avg_entropy_loss': np.mean(entropy_losses) if entropy_losses else None,
    }
    
    # P1-5: 获取当前尺度熵值用于监控
    if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'get_scale_entropy'):
        scale_entropy = model.tokenizer.get_scale_entropy()
        if scale_entropy is not None:
            perf_stats['scale_entropy'] = scale_entropy
    
    return total_loss / len(loader), 100.0 * correct / total, perf_stats


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    num_classes: int = 10,
    return_per_class: bool = False,
) -> Tuple[float, float, Optional[Dict[str, Any]]]:
    """评估
    
    Args:
        model: 模型
        loader: 数据加载器
        device: 设备
        use_amp: 是否使用混合精度
        num_classes: 类别数
        return_per_class: 是否返回逐类别统计
        
    Returns:
        (loss, accuracy, per_class_stats)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    nan_batches = 0
    
    # 逐类别统计
    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)
    
    for batch in tqdm(loader, desc="Eval"):
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        with get_amp_context(device, use_amp):
            outs, _ = model(imgs, return_aux_info=True)
            
            # 检查 logits 是否有问题
            if torch.isnan(outs).any() or torch.isinf(outs).any():
                nan_batches += 1
                continue
            
            loss = F.cross_entropy(outs, labels)
        
        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
        else:
            nan_batches += 1
            continue
            
        _, pred = outs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        
        # 逐类别统计
        for c in range(num_classes):
            mask = labels == c
            class_total[c] += mask.sum()
            class_correct[c] += (pred[mask] == c).sum()
    
    if nan_batches > 0:
        print(f"[WARN] 评估时跳过 {nan_batches} 个包含 NaN 的 batch")
    
    if total == 0:
        return float('inf'), 0.0, None
    
    # 计算逐类别准确率
    per_class_stats = None
    if return_per_class:
        class_correct = class_correct.cpu().numpy()
        class_total = class_total.cpu().numpy()
        class_acc = np.divide(class_correct, class_total, out=np.zeros_like(class_correct), where=class_total > 0) * 100
        
        per_class_stats = {
            'class_accuracy': class_acc.tolist(),
            'class_correct': class_correct.tolist(),
            'class_total': class_total.tolist(),
            'worst_classes': np.argsort(class_acc)[:10].tolist(),
            'best_classes': np.argsort(class_acc)[-10:][::-1].tolist(),
            'accuracy_std': float(np.std(class_acc[class_total > 0])),
            'accuracy_min': float(np.min(class_acc[class_total > 0])) if np.any(class_total > 0) else 0.0,
            'accuracy_max': float(np.max(class_acc[class_total > 0])) if np.any(class_total > 0) else 0.0,
        }
    
    return total_loss / max(len(loader) - nan_batches, 1), 100.0 * correct / total, per_class_stats


@torch.no_grad()
def verify_train_eval_consistency(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
) -> Dict[str, Any]:
    """验证模型在 train/eval 模式下的输出一致性.
    
    关键检查:
    1. train/eval 输出差异是否在可接受范围内
    2. 尺度/深度选择是否稳定
    
    Returns:
        一致性报告字典
    """
    report = {
        'passed': True,
        'checks': {},
        'warnings': [],
        'tokenizer_type': config.tokenizer_type,
    }
    
    # 获取一个 batch 用于测试
    sample_batch = next(iter(loader))
    imgs = sample_batch[0][:4].to(device)  # 只用 4 张图
    
    # 检查 1: train/eval 输出差异
    model.eval()
    with get_amp_context(device, config.use_amp):
        out_eval, aux_eval = model(imgs, return_aux_info=True)
    
    model.train()
    with get_amp_context(device, config.use_amp):
        out_train, aux_train = model(imgs, return_aux_info=True)
    model.eval()  # 恢复 eval 模式
    
    # 计算输出差异
    output_diff = (out_eval - out_train).abs()
    max_diff = output_diff.max().item()
    mean_diff = output_diff.mean().item()
    
    # V3 Variable Depth: 理论上 train/eval 应完全一致 (确定性分割)
    threshold = 0.01
    output_check = {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'threshold': threshold,
        'passed': max_diff < threshold,
    }
    report['checks']['output_consistency'] = output_check
    
    if output_check['passed']:
        print(f"  [OK] 输出一致性: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    else:
        report['warnings'].append(f"[WARN] train/eval 输出差异较大 (max={max_diff:.4f})，意外的输出差异")
        print(f"  [WARN] 输出差异: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    
    # 检查 3: 尺度选择稳定性 (多次推理应产生相同结果)
    if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'compute_scale_distribution'):
        dist1 = model.tokenizer.compute_scale_distribution(imgs)
        dist2 = model.tokenizer.compute_scale_distribution(imgs)
        
        # 比较两次的尺度比例
        scale_stable = all(
            abs(dist1['scale_ratios'][ps] - dist2['scale_ratios'][ps]) < 0.001
            for ps in dist1['scale_ratios']
        )
        
        stability_check = {
            'run1': dist1['scale_ratios'],
            'run2': dist2['scale_ratios'],
            'passed': scale_stable,
        }
        report['checks']['scale_stability'] = stability_check
        
        if scale_stable:
            print(f"  [OK] 尺度选择稳定 (eval 模式下确定性)")
        else:
            report['warnings'].append("[WARN] 尺度选择不稳定，可能存在随机性")
            report['passed'] = False
    
    # 打印 Tokenizer 信息
    print(f"  [OK] Tokenizer: {config.tokenizer_type} (Variable Depth Tokens)")
    
    # 总结
    print()
    if report['passed']:
        print("  [PASSED] 一致性检查通过: 训练成果可正确体现在推理中")
    else:
        print("  [FAILED] 一致性检查警告:")
        for w in report['warnings']:
            print(f"     {w}")
    
    return report


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fractal ViT Training")
    
    # 数据集
    parser.add_argument("--dataset", type=str, default="cifar10", 
                       choices=["cifar10", "cifar100", "mnist", "tiny-imagenet"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Number of workers (auto-detect if not set)")
    parser.add_argument("--val-split", type=float, default=0.05)  # T3: 减少验证集，增加训练数据
    parser.add_argument("--subset-size", type=int, default=None)
    
    # 模型
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim-head", type=int, default=32)
    parser.add_argument("--max-level", type=int, default=4)
    parser.add_argument("--num-scales", type=int, default=3,
                       help="Number of scales for multi-scale tokenizer")
    parser.add_argument("--pool", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--ffn-type", type=str, default="swiglu_level",
                       choices=["gelu", "swiglu", "swiglu_level"])
    parser.add_argument("--hilbert-bias-mode", type=str, default="lca",
                       choices=["lca", "low_rank", "hierarchical"],
                       help="Hilbert bias mode (lca recommended, ~100 params)")
    parser.add_argument("--gradient-checkpoint", action="store_true",
                       help="Enable gradient checkpointing to save memory")
    parser.add_argument("--compile", action="store_true",
                       help="Use torch.compile for faster training (PyTorch 2.0+)")
    parser.add_argument("--channels-last", action="store_true",
                       help="Use channels-last memory format for faster convolutions")
    
    # Tokenizer 类型
    parser.add_argument("--tokenizer-type", type=str, default="streaming_v3",
                       choices=["streaming_v3"],
                       help="Tokenizer type: streaming_v3 (Variable Depth Tokens, only supported)")
    
    # V3 Tokenizer 高级参数 (Adaptive Quadtree Split)
    parser.add_argument("--split-scheme", type=str, default="balanced_greedy",
                       choices=["balanced_greedy", "fixed_budget_dp"],
                       help="Split scheme: balanced_greedy (Scheme B) or fixed_budget_dp (Scheme C)")
    parser.add_argument("--target-tokens", type=int, default=None,
                       help="Target token count per image (None = adaptive)")
    parser.add_argument("--complexity-alpha", type=float, default=0.5,
                       help="Complexity function variance weight alpha in [0,1] (0.5 = balanced)")
    parser.add_argument("--split-tau0", type=float, default=0.15,
                       help="Root threshold tau_0 for adaptive splitting")
    parser.add_argument("--split-gamma", type=float, default=0.85,
                       help="Threshold decay factor gamma in (0,1) per depth")
    parser.add_argument("--enforce-balance", action="store_true", default=True,
                       help="Enforce 2:1 balance constraint in balanced_greedy")
    parser.add_argument("--no-enforce-balance", action="store_false", dest="enforce_balance")
    parser.add_argument("--domain-preset", type=str, default=None,
                       choices=["natural", "medical", "satellite", "document"],
                       help="Use domain-specific preset for split parameters")
    
    # 训练
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03,
                       help="Weight decay (default: 0.03)")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate (default: 0.1)")
    parser.add_argument("--emb-dropout", type=float, default=0.1)
    parser.add_argument("--drop-path", type=float, default=0.1,
                       help="Drop path (stochastic depth) rate")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                       help="Label smoothing factor (default: 0.1)")
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--accum-steps", type=int, default=1)
    
    # 早停
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--min-delta", type=float, default=0.001,
                       help="Minimum improvement for early stopping")
    
    # Mixup/CutMix
    parser.add_argument("--mixup-alpha", type=float, default=0.4,
                       help="Mixup alpha (default: 0.4, 0 to disable)")
    parser.add_argument("--cutmix-alpha", type=float, default=1.0,
                       help="CutMix alpha (default: 1.0, 0 to disable)")
    parser.add_argument("--mixup-prob", type=float, default=0.5,
                       help="Probability of applying Mixup/CutMix (default: 0.5)")
    
    # 系统
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick-test", action="store_true")
    
    args = parser.parse_args()
    
    # 环境检测
    env = detect_environment()
    print_environment_info(env)
    
    # 自动设置 workers
    if args.num_workers is None:
        args.num_workers = env['recommended_workers']
        print(f"Auto-detected num_workers: {args.num_workers}")
    
    # Quick test
    if args.quick_test:
        args.epochs = 3
        args.subset_size = 256
        print("[*] Quick test mode\n")
    
    # 初始化
    set_seed(args.seed)
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    spec = DATASETS[args.dataset]
    
    config = TrainingConfig(
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        subset_size=args.subset_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.dim * 4,
        dim_head=args.dim_head,
        max_level=args.max_level,
        num_scales=args.num_scales,
        pool=args.pool,
        ffn_type=args.ffn_type,
        hilbert_bias_mode=args.hilbert_bias_mode,
        # Tokenizer 配置 (V3)
        tokenizer_type=args.tokenizer_type,
        # V3 高级分割参数
        split_scheme=args.split_scheme,
        target_tokens=args.target_tokens,
        complexity_alpha=args.complexity_alpha,
        split_tau0=args.split_tau0,
        split_gamma=args.split_gamma,
        enforce_balance=args.enforce_balance,
        domain_preset=args.domain_preset,
        # 训练配置
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
        drop_path=args.drop_path,
        label_smoothing=args.label_smoothing,
        gradient_clip=args.gradient_clip,
        use_amp=args.use_amp,
        accum_steps=args.accum_steps,
        warmup_epochs=args.warmup_epochs,
        gradient_checkpoint=args.gradient_checkpoint,
        compile_model=getattr(args, 'compile', False),
        channels_last=getattr(args, 'channels_last', False),
        patience=args.patience,
        min_delta=args.min_delta,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mixup_prob=args.mixup_prob,
        seed=args.seed,
        device=str(device),
    )
    
    # 创建自定义 Tokenizer (支持高级分割参数)
    from vit_pytorch.tokenizer_streaming import StreamingFractalTokenizerV3
    from vit_pytorch.split_adaptive import AdaptiveSplitConfig, SplitScheme
    
    # 域适应预设
    split_kwargs: Dict[str, Any] = {
        'max_depth': config.num_scales - 1,
        'alpha': config.complexity_alpha,
        'tau_0': config.split_tau0,
        'gamma': config.split_gamma,
        'enforce_balance': config.enforce_balance,
        'target_tokens': config.target_tokens,
    }
    
    if config.domain_preset:
        preset_map = {
            'natural': AdaptiveSplitConfig.natural_images,
            'medical': AdaptiveSplitConfig.medical_images,
            'satellite': AdaptiveSplitConfig.satellite_images,
            'document': AdaptiveSplitConfig.document_images,
        }
        if config.domain_preset in preset_map:
            split_config = preset_map[config.domain_preset](**split_kwargs)
            print(f"[OK] Using domain preset: {config.domain_preset}")
            print(f"     sigma_0^2={split_config.sigma_0_sq:.4f}, g_0^2={split_config.g_0_sq:.4f}, alpha={split_config.alpha:.2f}")
    
    tokenizer = StreamingFractalTokenizerV3(
        image_size=max(spec.image_size, 32),
        channels=spec.channels,
        d_model=config.dim,
        base_patch_size=4,
        max_depth=config.num_scales - 1,
        use_hilbert_order=True,
        split_scheme=config.split_scheme,
        target_tokens=config.target_tokens,
        complexity_alpha=config.complexity_alpha,
        enforce_balance=config.enforce_balance,
    )
    
    # 创建模型 (V3 Variable Depth Tokens)
    model_kwargs = dict(
        image_size=max(spec.image_size, 32),
        num_classes=spec.num_classes,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        mlp_dim=config.mlp_dim,
        pool=config.pool,
        channels=spec.channels,
        dim_head=config.dim_head,
        dropout=config.dropout,
        emb_dropout=config.emb_dropout,
        drop_path_rate=config.drop_path,
        min_patch_size=(4, 4),
        max_level=config.max_level,
        use_checkpoint=config.gradient_checkpoint,
        ffn_type=config.ffn_type,
        # 使用自定义 tokenizer (支持高级分割参数)
        tokenizer=tokenizer,
        num_scales=config.num_scales,
        # Hilbert Bias 配置
        hilbert_bias_mode=config.hilbert_bias_mode,
    )
    
    model = FractalCurveViT(**model_kwargs).to(device)
    
    # 打印模型信息
    params = sum(p.numel() for p in model.parameters())
    split_info = f"{config.split_scheme}"
    if config.target_tokens:
        split_info += f", target={config.target_tokens}"
    tokenizer_name = f'StreamingFractalTokenizerV3 ({split_info})'
    
    print(f"\n{'='*70}")
    print(f"Model: FractalCurveViT")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"FFN Type: {config.ffn_type}")
    print(f"Hilbert Bias: {config.hilbert_bias_mode}")
    print(f"Parameters: {params:,}")
    print(f"Gradient Checkpoint: {config.gradient_checkpoint}")
    print(f"Compile Model: {config.compile_model}")
    print(f"Channels Last: {config.channels_last}")
    print(f"{'='*70}\n")
    
    # 数据加载
    train_loader, val_loader, test_loader = create_dataloaders(spec, config)
    
    # 优化器 (使用 fused 版本加速)
    use_fused = device.type == 'cuda' and hasattr(torch.optim.AdamW, 'fused')
    try:
        optimizer = AdamW(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay,
            fused=use_fused
        )
        if use_fused:
            print("[OK] Using fused AdamW optimizer")
    except TypeError:
        # 旧版本 PyTorch 不支持 fused 参数
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # 学习率调度
    warmup = min(args.warmup_epochs, config.epochs // 2)
    warmup_sch = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup)
    cosine_sch = CosineAnnealingLR(optimizer, T_max=config.epochs - warmup, eta_min=config.learning_rate * 0.01)
    scheduler = SequentialLR(optimizer, [warmup_sch, cosine_sch], milestones=[warmup])
    
    scaler = create_grad_scaler(config.use_amp)
    
    # CUDA 优化
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Channels Last 内存格式 (卷积加速)
    if config.channels_last and device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
        print("[OK] Using channels-last memory format")
    
    # torch.compile 编译优化 (PyTorch 2.0+)
    # 注意: mode='reduce-overhead' 使用 CUDA graphs，但不兼容动态缓存操作
    # 使用 mode='default' 更稳定，编译时间更短
    if config.compile_model:
        try:
            model = torch.compile(
                model, 
                mode='default',
                fullgraph=False,
                dynamic=False,  # 固定输入尺寸时设为 False 更快
            )
            print("[OK] Model compiled with torch.compile (mode=default)")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")
    
    # 创建 Mixup/CutMix 增强器
    use_mixup = config.mixup_alpha > 0 or config.cutmix_alpha > 0
    mixup_fn = None
    if use_mixup:
        mixup_fn = MixupCutmix(
            mixup_alpha=config.mixup_alpha,
            cutmix_alpha=config.cutmix_alpha,
            prob=config.mixup_prob,
            num_classes=spec.num_classes,
            label_smoothing=config.label_smoothing,
        )
    
    # 训练
    print("="*70)
    print("TRAINING START")
    print("="*70)
    
    if config.compile_model:
        print("[INFO] First batch will be slow due to JIT compilation (1-3 minutes)...")
    
    print()
    
    # 编译预热: 在正式训练前触发 JIT 编译
    if config.compile_model:
        print("[INFO] Warming up compiled model...")
        try:
            warmup_batch = next(iter(train_loader))
            if isinstance(warmup_batch, (list, tuple)):
                warmup_imgs = warmup_batch[0][:2].to(device)  # 只用2个样本
            else:
                warmup_imgs = warmup_batch[:2].to(device)
            if config.channels_last:
                warmup_imgs = warmup_imgs.to(memory_format=torch.channels_last)
            with torch.no_grad():
                with get_amp_context(device, config.use_amp):
                    _ = model(warmup_imgs)
            del warmup_imgs
            torch.cuda.empty_cache()
            print("[OK] Compilation complete!")
        except Exception as e:
            print(f"[WARN] Warmup failed: {e}")
    
    best_val = 0.0
    patience_counter = 0
    early_stopped = False
    
    exp_dir = PROJECT_ROOT / "experiments" / f"fractal_vit_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    
    # 保存配置
    with open(exp_dir / "logs" / "config.json", 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    history = []
    
    print(f"[INFO] Early stopping: patience={config.patience}, min_delta={config.min_delta}")
    print(f"[INFO] Regularization: dropout={config.dropout}, weight_decay={config.weight_decay}")
    print(f"[INFO] Label smoothing: {config.label_smoothing}")
    print(f"[INFO] Mixup/CutMix: alpha={config.mixup_alpha}/{config.cutmix_alpha}, prob={config.mixup_prob}\n")
    
    for epoch in range(1, config.epochs + 1):
        start = time.time()
        
        train_loss, train_acc, perf_stats = train_epoch(
            model, train_loader, optimizer, device, scaler, config,
            mixup_fn=mixup_fn,
            num_classes=spec.num_classes,
            profile=(epoch == 1)
        )
        
        val_loss, val_acc, _ = evaluate(model, val_loader, device, config.use_amp, spec.num_classes)
        
        scheduler.step()
        
        # 获取详细训练状态 (关键追踪参数)
        training_stats = None
        scale_distribution = None
        if hasattr(model, 'tokenizer'):
            if hasattr(model.tokenizer, 'get_training_stats'):
                training_stats = model.tokenizer.get_training_stats()
            
            # 每 10 个 epoch 或最后一个 epoch 计算尺度分布
            if hasattr(model.tokenizer, 'compute_scale_distribution') and (epoch % 10 == 0 or epoch == config.epochs):
                # 使用 val_loader 的一个 batch 计算尺度分布
                sample_batch = next(iter(val_loader))
                sample_imgs = sample_batch[0][:8].to(device)  # 只用 8 张图
                scale_distribution = model.tokenizer.compute_scale_distribution(sample_imgs)
        
        epoch_time = time.time() - start
        
        # 记录历史
        history_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time,
        }
        if training_stats is not None:
            history_entry['training_stats'] = training_stats
        if scale_distribution is not None:
            history_entry['scale_distribution'] = scale_distribution
        history.append(history_entry)
        
        print(f"\nEpoch {epoch}/{config.epochs}:")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.2f}%")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.2f}%")
        print(f"  Time:  {epoch_time:.1f}s, Throughput: {perf_stats['throughput']:.1f} samples/s")
        
        # 显示尺度分布 (每 10 epoch)
        if scale_distribution is not None:
            ratios = scale_distribution['scale_ratios']
            entropy = scale_distribution['entropy']
            max_entropy = scale_distribution['max_entropy']
            dominant = scale_distribution['dominant_scale']
            ratio_str = ", ".join([f"p{ps}:{r*100:.1f}%" for ps, r in ratios.items()])
            print(f"  Scales: {ratio_str}")
            print(f"  Entropy: {entropy:.3f}/{max_entropy:.3f} ({entropy/max_entropy*100:.1f}%), Dominant: {dominant}px")
        
        if epoch == 1:
            data_pct = perf_stats['avg_data_time'] / perf_stats['avg_batch_time'] * 100 if perf_stats['avg_batch_time'] > 0 else 0
            fwd_pct = perf_stats['avg_forward_time'] / perf_stats['avg_batch_time'] * 100 if perf_stats['avg_batch_time'] > 0 else 0
            print(f"  Perf:  data={data_pct:.1f}%, fwd={fwd_pct:.1f}%, mem={perf_stats['cuda_mem_peak_gb']:.2f}GB")
        
        # 保存最佳
        if val_acc > best_val + config.min_delta:
            best_val = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': asdict(config),
            }, exp_dir / "checkpoints" / "best.pth")
            print(f"  [*] Best model saved: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"  [!] No improvement ({patience_counter}/{config.patience})")
            if patience_counter >= config.patience:
                print(f"\n[EARLY STOPPING] No improvement for {config.patience} epochs.")
                print(f"[EARLY STOPPING] Best val acc: {best_val:.2f}% at epoch {epoch - patience_counter}")
                early_stopped = True
                break
    
    # 保存训练历史
    with open(exp_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # ========== Train/Eval 一致性验证 ==========
    print("\n" + "="*70)
    print("TRAIN/EVAL CONSISTENCY CHECK")
    print("="*70 + "\n")
    
    consistency_report = verify_train_eval_consistency(model, val_loader, device, config)
    
    # 保存一致性报告
    with open(exp_dir / "logs" / "consistency_report.json", 'w') as f:
        json.dump(consistency_report, f, indent=2)
    
    # 测试
    print("\n" + "="*70)
    print("TESTING")
    print("="*70 + "\n")
    
    ckpt = torch.load(exp_dir / "checkpoints" / "best.pth", weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    
    test_loss, test_acc, per_class_stats = evaluate(
        model, test_loader, device, config.use_amp, 
        spec.num_classes, return_per_class=True
    )
    
    print(f"Test: loss={test_loss:.4f}, acc={test_acc:.2f}%")
    
    # 逐类别评估诊断
    if per_class_stats:
        print(f"\n{'='*70}")
        print("PER-CLASS ACCURACY ANALYSIS")
        print(f"{'='*70}")
        print(f"  Accuracy range: {per_class_stats['accuracy_min']:.1f}% - {per_class_stats['accuracy_max']:.1f}%")
        print(f"  Accuracy std:   {per_class_stats['accuracy_std']:.1f}%")
        print(f"  Worst 5 classes: {per_class_stats['worst_classes'][:5]}")
        print(f"  Best 5 classes:  {per_class_stats['best_classes'][:5]}")
        
        # 检测类别不平衡
        if per_class_stats['accuracy_std'] > 20:
            print(f"\n  [WARN] 类别不平衡警告: std={per_class_stats['accuracy_std']:.1f}% > 20%")
            print(f"         请检查数据加载和模型架构")
    
    if early_stopped:
        print(f"[*] Training stopped early at epoch {epoch}/{config.epochs}")
    print(f"\n[OK] Results saved to: {exp_dir}")
    
    with open(exp_dir / "results.json", 'w') as f:
        results = {
            'best_val_acc': best_val,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'total_epochs': epoch if early_stopped else config.epochs,
            'early_stopped': early_stopped,
            'best_epoch': epoch - patience_counter if early_stopped else epoch,
            'consistency_passed': consistency_report.get('passed', None),
            'tokenizer_type': config.tokenizer_type,
        }
        if per_class_stats:
            results['per_class_stats'] = {
                'accuracy_std': per_class_stats['accuracy_std'],
                'accuracy_min': per_class_stats['accuracy_min'],
                'accuracy_max': per_class_stats['accuracy_max'],
                'worst_classes': per_class_stats['worst_classes'],
                'best_classes': per_class_stats['best_classes'],
            }
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
