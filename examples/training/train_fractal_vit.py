#!/usr/bin/env python3
"""Fractal ViT Training Script - Optimized for Experimentation

重写版本 (2025-12-11):
- 完全采用 NextGenerationFractalViT 和 ffn_type 参数
- 详细的实验日志和指标追踪
- 支持 SwiGLU + Level Adaptation 等 FFN 变体
- 改进的 tokenization 监控和分析
- 优化的内存管理和训练效率

主要改进:
1. 移除 SimpleFractalViT，统一使用 NextGenerationFractalViT
2. 新增 --ffn-type 参数（gelu / swiglu / swiglu_level）
3. 结构化的实验日志（JSON 格式）
4. 详细的 tokenization 自适应性分析
5. 改进的错误处理和训练中断恢复
"""

from __future__ import annotations

import os
import sys

# ============================================================================
# 环境配置（必须在导入 torch 之前）
# ============================================================================

# 强制设置 multiprocessing 启动方法为 spawn（CUDA 兼容）
import multiprocessing as _mp
try:
    _mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # 已经设置过

_max_split_mb = 512
for i, arg in enumerate(sys.argv):
    if arg == '--max-split-size-mb' and i + 1 < len(sys.argv):
        try:
            _max_split_mb = int(sys.argv[i + 1])
        except ValueError:
            pass

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 
                      f'max_split_size_mb:{_max_split_mb},expandable_segments:True')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import argparse
import json
import multiprocessing
import random
import shutil
import time
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

# 项目路径设置
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vit_pytorch import NextGenerationFractalViT


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
    pool: str
    ffn_type: str
    learnable_split: bool
    gradient_checkpoint: bool
    
    # 训练
    epochs: int
    learning_rate: float
    weight_decay: float
    dropout: float
    emb_dropout: float
    gradient_clip: float
    use_amp: bool
    accum_steps: int
    
    # 系统
    seed: int
    device: str


@dataclass
class TokenStats:
    """Tokenization 统计"""
    avg_tokens: float
    min_tokens: int
    max_tokens: int
    std_tokens: float
    levels_used: List[int]


@dataclass
class EpochMetrics:
    """Epoch 指标"""
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float
    time: float
    tokens: Optional[TokenStats] = None


# ============================================================================
# Tiny ImageNet 下载和组织
# ============================================================================

def download_and_setup_tiny_imagenet(data_root: Path) -> bool:
    """下载并设置 Tiny ImageNet 数据集
    
    Returns:
        bool: True if successful, False otherwise
    """
    target_dir = data_root / "tiny-imagenet-200"
    
    # 检查是否已存在且完整
    if (target_dir / "train").exists() and (target_dir / "val").exists():
        # 快速验证：检查训练集是否有足够的类别
        train_classes = len(list((target_dir / "train").iterdir()))
        if train_classes >= 200:
            return True
    
    print("\n" + "="*70)
    print("Downloading Tiny ImageNet...")
    print("="*70)
    
    zip_path = data_root / "tiny-imagenet-200.zip"
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    expected_size = 248100043  # ~237 MB
    
    # 检查现有 zip 文件是否有效
    if zip_path.exists():
        print(f"Found existing zip: {zip_path}")
        
        # 验证文件大小
        actual_size = zip_path.stat().st_size
        if actual_size < expected_size * 0.95:  # 允许 5% 误差
            print(f"⚠️  File size mismatch (expected ~{expected_size}, got {actual_size})")
            print("Removing corrupted file and re-downloading...")
            zip_path.unlink()
        else:
            # 验证是否为有效 zip 文件
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    # 测试 zip 文件完整性
                    if zf.testzip() is not None:
                        print("⚠️  Zip file is corrupted")
                        print("Removing corrupted file and re-downloading...")
                        zip_path.unlink()
                    else:
                        print("✓ Zip file verified")
            except zipfile.BadZipFile:
                print("⚠️  Invalid zip file")
                print("Removing corrupted file and re-downloading...")
                zip_path.unlink()
    
    # 下载
    if not zip_path.exists():
        print(f"\nDownloading from {url}...")
        print("This may take several minutes (~237 MB)...")
        
        try:
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading") as pbar:
                def reporthook(block_num, block_size, total_size):
                    if pbar.total is None and total_size > 0:
                        pbar.total = total_size
                    pbar.update(block_size)
                
                urllib.request.urlretrieve(url, zip_path, reporthook=reporthook)
            
            # 验证下载的文件
            downloaded_size = zip_path.stat().st_size
            print(f"\n✓ Downloaded {downloaded_size:,} bytes")
            
            if downloaded_size < expected_size * 0.95:
                print(f"⚠️  Downloaded file seems incomplete (expected ~{expected_size:,} bytes)")
                return False
                
        except Exception as e:
            print(f"\n✗ Download failed: {e}")
            if zip_path.exists():
                zip_path.unlink()
            return False
    
    # 解压
    print("\nExtracting archive...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 显示解压进度
            members = zip_ref.namelist()
            for member in tqdm(members, desc="Extracting"):
                zip_ref.extract(member, data_root)
        print(f"✓ Extracted to {target_dir}")
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        # 清理部分解压的文件
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        return False
    
    # 组织验证集（Tiny ImageNet 的 val 目录需要重新组织）
    val_dir = target_dir / "val"
    val_images_dir = val_dir / "images"
    
    if val_images_dir.exists():
        print("\nOrganizing validation set...")
        
        # 读取验证集标注
        val_annotations = val_dir / "val_annotations.txt"
        if val_annotations.exists():
            # 创建类别目录
            img_to_class = {}
            with open(val_annotations, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        img_name = parts[0]
                        class_id = parts[1]
                        img_to_class[img_name] = class_id
            
            # 移动图像到对应类别目录
            for img_name, class_id in tqdm(img_to_class.items(), desc="Organizing"):
                class_dir = val_dir / class_id / "images"
                class_dir.mkdir(parents=True, exist_ok=True)
                
                src = val_images_dir / img_name
                dst = class_dir / img_name
                
                if src.exists() and not dst.exists():
                    shutil.move(str(src), str(dst))
            
            # 删除原始 images 目录
            if val_images_dir.exists() and not any(val_images_dir.iterdir()):
                val_images_dir.rmdir()
            
            print("✓ Validation set organized")
    
    # 清理 zip 文件（可选）
    if zip_path.exists():
        try:
            zip_path.unlink()
            print(f"✓ Cleaned up {zip_path.name}")
        except:
            pass
    
    print("="*70)
    print("✓ Tiny ImageNet setup complete!\n")
    return True


# ============================================================================
# 数据集配置
# ============================================================================

DATASETS = {
    "cifar10": DatasetSpec(
        name="CIFAR10",
        num_classes=10,
        image_size=32,
        channels=3,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
    "cifar100": DatasetSpec(
        name="CIFAR100",
        num_classes=100,
        image_size=32,
        channels=3,
        mean=(0.5071, 0.4865, 0.4409),
        std=(0.2673, 0.2564, 0.2762),
    ),
    "mnist": DatasetSpec(
        name="MNIST",
        num_classes=10,
        image_size=28,
        channels=1,
        mean=(0.1307,),
        std=(0.3081,),
    ),
    "tiny-imagenet": DatasetSpec(
        name="TinyImageNet",
        num_classes=200,
        image_size=64,
        channels=3,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
}


# ============================================================================
# Tokenization 监控
# ============================================================================

class TokenizationMonitor:
    """监控 tokenization 自适应性"""
    
    def __init__(self):
        self.epoch_stats: Dict[int, List[TokenStats]] = defaultdict(list)
        self.variance_token_pairs: List[Tuple[float, int]] = []
    
    @torch.no_grad()
    def analyze_batch(self, model: nn.Module, images: torch.Tensor, epoch: int) -> TokenStats:
        """分析一个 batch"""
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            return TokenStats(0, 0, 0, 0, [])
        
        output = tokenizer.tokenize(images)
        sequences = output.sequences
        
        token_counts = [seq.tokens.shape[0] for seq in sequences]
        all_levels = []
        
        for seq in sequences:
            levels = seq.metadata.get("levels", None)
            if levels is not None and levels.numel() > 0:
                all_levels.extend(levels[:, 0].tolist())
        
        # 记录 variance-token 关系
        for i, count in enumerate(token_counts):
            var = images[i].var().item()
            self.variance_token_pairs.append((var, count))
        
        stats = TokenStats(
            avg_tokens=float(np.mean(token_counts)),
            min_tokens=int(np.min(token_counts)),
            max_tokens=int(np.max(token_counts)),
            std_tokens=float(np.std(token_counts)),
            levels_used=sorted(list(set(all_levels))) if all_levels else [],
        )
        
        self.epoch_stats[epoch].append(stats)
        return stats
    
    def compute_correlation(self) -> float:
        """计算 variance-token 相关性"""
        if len(self.variance_token_pairs) < 10:
            return 0.0
        vars_list = [v for v, _ in self.variance_token_pairs]
        tokens_list = [t for _, t in self.variance_token_pairs]
        corr = np.corrcoef(vars_list, tokens_list)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
    
    def get_report(self) -> Dict[str, Any]:
        """生成报告"""
        summaries = []
        for epoch in sorted(self.epoch_stats.keys()):
            stats_list = self.epoch_stats[epoch]
            summaries.append({
                "epoch": epoch,
                "avg_tokens": float(np.mean([s.avg_tokens for s in stats_list])),
                "token_range": [
                    min(s.min_tokens for s in stats_list),
                    max(s.max_tokens for s in stats_list),
                ],
            })
        
        return {
            "epoch_summaries": summaries,
            "correlation": self.compute_correlation(),
            "is_adaptive": self.compute_correlation() > 0.1,
        }


# ============================================================================
# 实验管理
# ============================================================================

class ExperimentManager:
    """实验管理器"""
    
    def __init__(self, base_dir: Path, name: str):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.exp_dir = base_dir / f"{name}_{timestamp}"
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.log_dir = self.exp_dir / "logs"
        
        for d in [self.ckpt_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.metrics: List[EpochMetrics] = []
        self.start_time = time.time()
    
    def save_config(self, config: TrainingConfig):
        """保存配置"""
        path = self.log_dir / "config.json"
        with open(path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
    
    def log_epoch(self, metrics: EpochMetrics):
        """记录 epoch"""
        self.metrics.append(metrics)
        
        # 打印
        print(f"\n{'='*70}")
        print(f"Epoch {metrics.epoch}")
        print(f"{'='*70}")
        print(f"Train: loss={metrics.train_loss:.4f}, acc={metrics.train_acc:.2f}%")
        print(f"Val:   loss={metrics.val_loss:.4f}, acc={metrics.val_acc:.2f}%")
        print(f"LR: {metrics.lr:.2e}, Time: {metrics.time:.1f}s")
        if metrics.tokens:
            t = metrics.tokens
            print(f"Tokens: avg={t.avg_tokens:.1f}, range=[{t.min_tokens}-{t.max_tokens}]")
        print(f"{'='*70}\n")
        
        # 保存
        path = self.log_dir / "metrics.json"
        with open(path, 'w') as f:
            json.dump([asdict(m) for m in self.metrics], f, indent=2)
    
    def save_checkpoint(self, model: nn.Module, optimizer, name: str, **kwargs):
        """保存检查点"""
        path = self.ckpt_dir / f"{name}.pth"
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            **kwargs
        }
        torch.save(checkpoint, path)
    
    def log_final(self, test_loss: float, test_acc: float, token_report: Dict):
        """记录最终结果"""
        best_val = max(m.val_acc for m in self.metrics)
        
        final = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_val_acc": best_val,
            "total_time": time.time() - self.start_time,
            "tokenization": token_report,
        }
        
        path = self.log_dir / "final.json"
        with open(path, 'w') as f:
            json.dump(final, f, indent=2)
        
        print(f"\n{'='*70}")
        print("FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Best Val: {best_val:.2f}%")
        print(f"Test: {test_acc:.2f}%")
        print(f"Time: {final['total_time']:.1f}s")
        print(f"Adaptive: {token_report.get('is_adaptive', False)}")
        print(f"Correlation: {token_report.get('correlation', 0):.3f}")
        print(f"{'='*70}\n")


# ============================================================================
# 工具函数
# ============================================================================

def set_seed(seed: int):
    """设置种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(config: TrainingConfig, spec: DatasetSpec) -> NextGenerationFractalViT:
    """创建模型"""
    model = NextGenerationFractalViT(
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
        min_patch_size=(4, 4),
        max_level=config.max_level,
        learnable_split=config.learnable_split,
        use_checkpoint=config.gradient_checkpoint,
        ffn_type=config.ffn_type,  # type: ignore
    )
    
    params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*70}")
    print(f"Model: NextGenerationFractalViT")
    print(f"FFN Type: {config.ffn_type}")
    print(f"Gradient Checkpoint: {config.gradient_checkpoint}")
    print(f"Parameters: {params:,}")
    print(f"{'='*70}\n")
    
    return model


def create_dataloaders(
    spec: DatasetSpec,
    batch_size: int,
    val_split: float,
    subset_size: Optional[int],
    num_workers: int,
    pin_memory: bool,
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
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(spec.image_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
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
        # Tiny ImageNet 自动下载和组织
        train_dir = data_root / "tiny-imagenet-200" / "train"
        test_dir = data_root / "tiny-imagenet-200" / "val"
        
        if not train_dir.exists() or not test_dir.exists():
            print("\n⚠️  Tiny ImageNet not found, attempting automatic download...")
            success = download_and_setup_tiny_imagenet(data_root)
            if not success:
                raise FileNotFoundError(
                    f"\nAutomatic download failed. Please manually download:\n"
                    f"URL: http://cs231n.stanford.edu/tiny-imagenet-200.zip\n"
                    f"Extract to: {data_root}\n"
                    f"Expected structure:\n"
                    f"  {data_root}/tiny-imagenet-200/train/n01443537/images/*.JPEG\n"
                    f"  {data_root}/tiny-imagenet-200/val/n01443537/images/*.JPEG\n"
                )
        
        train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
        test_ds = datasets.ImageFolder(str(test_dir), transform=test_tf)
    else:
        raise ValueError(f"Unknown dataset: {spec.name}")
    
    # 划分
    indices = np.arange(len(train_ds))
    np.random.shuffle(indices)
    if subset_size:
        indices = indices[:subset_size]
    
    val_size = max(1, int(len(indices) * val_split))
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    
    # 创建 loader
    # 强制使用 spawn（已在顶部设置，这里确保兼容性）
    mp_context = 'spawn' if num_workers > 0 else None
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=num_workers,
        pin_memory=pin_memory,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,  # 保持 worker 进程
    )
    
    val_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_idx),
        num_workers=num_workers,
        pin_memory=pin_memory,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
    )
    
    print(f"✓ Data: train={len(train_idx)}, val={len(val_idx)}, test={len(test_ds)}\n")
    
    return train_loader, val_loader, test_loader


# ============================================================================
# 训练函数
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,  # torch.amp.GradScaler (避免类型检查问题)
    clip: float,
    accum: int,
    use_amp: bool,
) -> Tuple[float, float, Dict[str, float]]:
    """训练一个 epoch，返回 (loss, accuracy, perf_stats)"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
    
    # 性能监控
    batch_times = []
    data_times = []
    cuda_mem_peak = 0.0
    
    pbar = tqdm(loader, desc="Train")
    data_start = time.time()
    
    for i, (imgs, labels) in enumerate(pbar):
        data_times.append(time.time() - data_start)
        batch_start = time.time()
        
        # non_blocking=True 实现异步数据传输
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            outs, _ = model(imgs, return_aux_info=True)
            loss = F.cross_entropy(outs, labels) / accum
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item() * accum
        _, pred = outs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        
        batch_times.append(time.time() - batch_start)
        
        # 更新 CUDA 内存峰值
        if device.type == 'cuda':
            cuda_mem_peak = max(cuda_mem_peak, torch.cuda.max_memory_allocated() / 1024**3)
        
        pbar.set_postfix(loss=f'{loss.item()*accum:.4f}', acc=f'{100.*correct/total:.1f}%')
        data_start = time.time()
    
    # 性能统计
    perf_stats = {
        'avg_batch_time': np.mean(batch_times) if batch_times else 0.0,
        'avg_data_time': np.mean(data_times) if data_times else 0.0,
        'throughput': total / sum(batch_times) if batch_times else 0.0,  # samples/sec
        'cuda_mem_peak_gb': cuda_mem_peak,
    }
    
    return total_loss / len(loader), 100.0 * correct / total, perf_stats


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    """评估"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(loader, desc="Eval")
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            outs, _ = model(imgs, return_aux_info=True)
            loss = F.cross_entropy(outs, labels)
        
        total_loss += loss.item()
        _, pred = outs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        
        pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{100.*correct/total:.1f}%')
    
    return total_loss / len(loader), 100.0 * correct / total


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="Fractal ViT Training")
    
    # 数据集
    parser.add_argument("--dataset", type=str, default="cifar10", 
                       choices=["cifar10", "cifar100", "mnist", "tiny-imagenet"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--subset-size", type=int, default=None)
    
    # 模型
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim-head", type=int, default=32)
    parser.add_argument("--max-level", type=int, default=4)
    parser.add_argument("--pool", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--ffn-type", type=str, default="swiglu_level",
                       choices=["gelu", "swiglu", "swiglu_level"])
    parser.add_argument("--no-learnable-split", action="store_true")
    parser.add_argument("--gradient-checkpoint", action="store_true",
                       help="Use gradient checkpointing to save memory (slower but ~40%% less VRAM)")
    
    # 训练
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--emb-dropout", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=10,
                       help="Number of warmup epochs (default: 10)")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--accum-steps", type=int, default=1)
    
    # 系统
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--quick-test", action="store_true")
    
    # 性能优化
    parser.add_argument("--compile", action="store_true",
                       help="Use torch.compile() for model optimization (PyTorch 2.0+)")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead",
                       choices=["default", "reduce-overhead", "max-autotune"],
                       help="torch.compile mode (default: reduce-overhead)")
    
    args = parser.parse_args()
    
    # Quick test
    if args.quick_test:
        args.epochs = 5
        args.subset_size = 512
        args.num_workers = 0
        print("⚡ Quick test mode\n")
    
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
        mlp_dim=args.dim * 2,
        dim_head=args.dim_head,
        max_level=args.max_level,
        pool=args.pool,
        ffn_type=args.ffn_type,
        learnable_split=not args.no_learnable_split,
        gradient_checkpoint=args.gradient_checkpoint,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
        gradient_clip=args.gradient_clip,
        use_amp=args.use_amp,
        accum_steps=args.accum_steps,
        seed=args.seed,
        device=str(device),
    )
    
    # 实验管理
    exp_mgr = ExperimentManager(PROJECT_ROOT / "experiments", "fractal_vit")
    exp_mgr.save_config(config)
    print(f"✓ Experiment: {exp_mgr.exp_dir}\n")
    
    # CUDA 优化
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        
        # 启用 TF32 加速（Ampere 及以上 GPU）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        torch.cuda.empty_cache()
        
        # 输出 GPU 信息
        gpu_name = torch.cuda.get_device_name(0)
        gpu_cap = torch.cuda.get_device_capability(0)
        print(f"✓ CUDA optimized: {gpu_name}")
        print(f"  - Compute Capability: {gpu_cap[0]}.{gpu_cap[1]}")
        print(f"  - TF32 Enabled: {gpu_cap[0] >= 8}")
        print(f"  - cuDNN Benchmark: True\n")
    
    # 模型和数据
    model = create_model(config, spec).to(device)
    
    # torch.compile() 优化 (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        print(f"⚡ Compiling model with mode='{args.compile_mode}'...")
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print("  ✓ Model compiled successfully\n")
        except Exception as e:
            print(f"  ⚠ Compilation failed: {e}")
            print("  → Falling back to eager mode\n")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        spec, config.batch_size, config.val_split, config.subset_size,
        config.num_workers, device.type == "cuda"
    )
    
    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, 
                     weight_decay=config.weight_decay)
    
    # Warmup 策略：默认 10 轮
    warmup = min(args.warmup_epochs, config.epochs // 2)
    warmup_sch = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup)
    cosine_sch = CosineAnnealingLR(optimizer, T_max=config.epochs - warmup, 
                                   eta_min=config.learning_rate * 0.01)
    scheduler = SequentialLR(optimizer, [warmup_sch, cosine_sch], milestones=[warmup])
    
    print(f"✓ Optimizer: AdamW (lr={config.learning_rate:.2e}, wd={config.weight_decay})")
    print(f"✓ Scheduler: {warmup} warmup epochs + cosine annealing\n")
    
    scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp)
    
    # Tokenization 监控
    tok_monitor = TokenizationMonitor()
    monitor_freq = max(1, config.epochs // 10)
    
    # 显存清理函数
    def clear_cuda_cache():
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
    # 训练
    print(f"{'='*70}")
    print("TRAINING START")
    print(f"{'='*70}")
    if config.gradient_checkpoint:
        print("⚡ Gradient Checkpointing: ENABLED (saves ~40% VRAM)")
    print("")
    
    best_val = 0.0
    clear_cuda_cache()  # 训练前清理
    
    try:
        for epoch in range(1, config.epochs + 1):
            start = time.time()
            
            train_loss, train_acc, perf_stats = train_epoch(
                model, train_loader, optimizer, device, scaler,
                config.gradient_clip, config.accum_steps, config.use_amp
            )
            
            val_loss, val_acc = evaluate(model, val_loader, device, config.use_amp)
            scheduler.step()
            
            # Token 监控
            tok_stats = None
            if epoch % monitor_freq == 0 or epoch == 1:
                sample = next(iter(val_loader))[0].to(device)
                tok_stats = tok_monitor.analyze_batch(model, sample, epoch)
            
            # 记录
            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=optimizer.param_groups[0]['lr'],
                time=time.time() - start,
                tokens=tok_stats,
            )
            exp_mgr.log_epoch(metrics)
            
            # 显示性能统计（每 5 轮或第一轮）
            if epoch == 1 or epoch % 5 == 0:
                print(f"  📊 Perf: {perf_stats['throughput']:.1f} samples/s, "
                      f"mem={perf_stats['cuda_mem_peak_gb']:.2f}GB")
            
            # 保存最佳
            if val_acc > best_val:
                best_val = val_acc
                exp_mgr.save_checkpoint(model, optimizer, "best", val_acc=val_acc, epoch=epoch)
                print(f"✓ Best saved: {val_acc:.2f}%\n")
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted\n")
    
    # 测试
    print(f"{'='*70}")
    print("TESTING")
    print(f"{'='*70}\n")
    
    ckpt = torch.load(exp_mgr.ckpt_dir / "best.pth")
    model.load_state_dict(ckpt['model_state_dict'])
    
    test_loss, test_acc = evaluate(model, test_loader, device, config.use_amp)
    tok_report = tok_monitor.get_report()
    
    exp_mgr.log_final(test_loss, test_acc, tok_report)
    print(f"✓ Results: {exp_mgr.log_dir}\n")


if __name__ == "__main__":
    # Multiprocessing 已在顶部强制设置为 spawn
    main()
