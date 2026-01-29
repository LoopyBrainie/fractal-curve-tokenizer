# -*- coding: utf-8 -*-
"""
Hilbert vs Raster 消融实验 (I25-2)

实验设计:
┌──────────────┬────────────────────────────────────────────────────────────┐
│ 模式         │ 说明                                                        │
├──────────────┼────────────────────────────────────────────────────────────┤
│ Standard ViT | 标准 ViT (16x16 patch) - 基线，对比分形 tokenization 收益     │
│ Hilbert      | FractalCurveViT + Hilbert 排序 - 核心假设验证                │
│ Raster       | FractalCurveViT + Raster 排序 - Hilbert 对照组              │
└──────────────┴────────────────────────────────────────────────────────────┘

控制变量:
- Hilbert vs Raster: 唯一区别 use_hilbert_encoding=True/False
- Standard vs Fractal: 参数量/计算量同级别 (公平对比)

默认配置 (v3.0 2026-01-18):
┌─────────────────────────────────────────────────────────────────────────────┐
│ 模型: dim=384, depth=12 → P=32.7M (Tiny-ImageNet 调优)                       │
│ 动态分辨率: image_size=64, min_patch_size=4 → max_depth=4                    │
│ Token: K∈[16,64] (4:1 压缩比)                                               │
│ I31: use_area_encoding=True, fourier_levels=4                              │
│ I78 优化: channels-last ✓, torch.compile ✓                                  │
│ 训练: batch=64, lr=1e-3, epochs=100                                         │
│ 正则: dropout=0.2, drop_path=0.2, weight_decay=0.1                          │
└─────────────────────────────────────────────────────────────────────────────┘

使用示例:
  # 完整消融实验 (3 模式 × 3 次 = 9 次训练)
  uv run python tests/benchmarks/ablation_hilbert_curve.py --all --runs 3

  # 只运行 Hilbert vs Raster (核心假设验证)
  uv run python tests/benchmarks/ablation_hilbert_curve.py --modes hilbert raster --runs 3

  # 快速测试 (10 epochs, 1 run)
  uv run python tests/benchmarks/ablation_hilbert_curve.py --quick

  # 保存结果到 JSON
  uv run python tests/benchmarks/ablation_hilbert_curve.py --output results.json

  # 禁用性能优化 (CPU 或调试模式)
  uv run python tests/benchmarks/ablation_hilbert_curve.py --all --no-channels-last --no-compile

  # I31 面积编码消融
  uv run python tests/benchmarks/ablation_hilbert_curve.py --modes hilbert --no-area-encoding
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import zipfile
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torchvision import datasets, transforms

# Handle import paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from einops import rearrange, repeat
from vit_pytorch import FractalCurveViT


# ============================================================================
# 数据集加载 (复用 examples/training/train_fractal_vit.py 的方法)
# ============================================================================

def organize_tiny_imagenet_val_set(val_dir: Path) -> None:
    """组织 Tiny-ImageNet 验证集为 ImageFolder 兼容格式

    原始结构: val/images/*.JPEG + val/val_annotations.txt
    目标结构: val/类别名/images/*.JPEG

    这是 examples/training/train_fractal_vit.py 中验证集准备的标准方法
    """
    val_images_dir = val_dir / "images"

    if not val_images_dir.exists():
        return  # 已经组织好了

    val_annotations = val_dir / "val_annotations.txt"
    if not val_annotations.exists():
        print("[WARN] val_annotations.txt not found, skipping validation set organization")
        return

    print("[INFO] Organizing validation set by class...")

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


def download_tiny_imagenet(data_root: Path) -> bool:
    """下载并设置 Tiny ImageNet (从 examples/training/train_fractal_vit.py 复制)

    数据集信息:
    - 200 类，每类 500 张训练图像
    - 训练集: 100,000 张 64x64 图像
    - 验证集: 10,000 张图像
    """
    target_dir = data_root / "tiny-imagenet-200"

    # 检查是否已存在
    if (target_dir / "train").exists() and (target_dir / "val").exists():
        train_classes = len(list((target_dir / "train").iterdir()))
        val_has_classes = any((target_dir / "val").iterdir())
        if train_classes >= 200 and val_has_classes:
            print(f"[OK] Tiny ImageNet already exists at {target_dir}")
            # 自动组织验证集
            organize_tiny_imagenet_val_set(target_dir / "val")
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
                if zf.testzip() is not None:
                    raise zipfile.BadZipFile("Corrupted zip file")
                if len(zf.namelist()) < 100:
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
    print(f"[*] Extracting to {target_dir}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_root)
        print("[OK] Extraction complete")

        # 验证解压结果
        if not (target_dir / "train").exists():
            raise FileNotFoundError("train directory not found after extraction")

        # 自动组织验证集
        organize_tiny_imagenet_val_set(target_dir / "val")

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

    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        return False

    return True


def download_with_progress(url: str, dest: Path, name: str) -> bool:
    """带进度条下载"""
    try:
        print(f"[*] Downloading {name}...")

        # 设置请求头模拟浏览器
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        with urllib.request.urlopen(req, timeout=300) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            block_size = 8192

            with open(dest, 'wb') as f:
                downloaded = 0
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)

                    if total_size > 0:
                        percent = 100.0 * downloaded / total_size
                        print(f"\r  Progress: {percent:.1f}% ({downloaded/1024/1024:.1f}MB)",
                              end='', flush=True)

        print()  # 换行
        return dest.exists() and dest.stat().st_size > 0

    except Exception as e:
        print(f"  Error: {e}")
        if dest.exists():
            dest.unlink()
        return False


def get_tiny_imagenet_loaders(
    batch_size: int = 64,
    image_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, int]:
    """获取 Tiny-ImageNet 数据加载器 (复用 examples/training/train_fractal_vit.py 的方法)

    Returns:
        train_loader, val_loader, num_classes
    """
    data_root = PROJECT_ROOT / "data"
    data_root.mkdir(exist_ok=True)

    # 下载数据集 (自动组织验证集)
    if not download_tiny_imagenet(data_root):
        raise FileNotFoundError("Failed to download Tiny ImageNet")

    tiny_imagenet_dir = data_root / "tiny-imagenet-200"

    # 数据增强 (与 examples/training/train_fractal_vit.py 一致)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(image_size, padding=8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(image_size + 8),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 加载数据集 (直接使用 ImageFolder，验证集已自动组织)
    train_dir = tiny_imagenet_dir / "train"
    val_dir = tiny_imagenet_dir / "val"

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # 验证数据集加载成功
    if len(train_ds) == 0:
        raise ValueError("训练集为空!")
    if len(val_ds) == 0:
        raise ValueError("验证集为空!")

    print(f"[INFO] Dataset loaded: train={len(train_ds)}, val={len(val_ds)}")

    return train_loader, val_loader, 200


# ============================================================================
# Standard ViT 基线模型 (参考: https://github.com/lucidrains/vit-pytorch)
# ============================================================================

class FeedForward(nn.Module):
    """前馈网络 (GELU 激活)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """多头自注意力."""

    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float = 0.25):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    """Transformer 编码器."""

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_ratio: float, dropout: float = 0.25):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        hidden_dim = int(dim * mlp_ratio)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads, dim_head, dropout),
                FeedForward(dim, hidden_dim, dropout),
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)

        return self.norm(x)


class StandardViT(nn.Module):
    """标准 ViT 基线模型."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.to_patch_embedding = nn.Conv2d(3, dim, patch_size, patch_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim // heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 3, H, W] -> [B, dim, h_patches, w_patches]
        x = self.to_patch_embedding(x)
        # [B, dim, h_patches, w_patches] -> [B, num_patches, dim]
        x = rearrange(x, 'b d h w -> b (h w) d')

        # 添加 cls token (标准 ViT 做法: prepend cls_token)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, dim]

        # 添加位置编码 (在 cls_token 之后)
        x = x + self.pos_embedding[:, :x.shape[1]]

        x = self.transformer(x)

        # 分类
        return self.mlp_head(x[:, 0])


# ============================================================================
# 实验配置
# ============================================================================

@dataclass
class ExperimentConfig:
    """消融实验配置

    I78 动态分辨率:
    - image_size=64 表示固定分辨率
    - max_depth = floor(log2(64/4)) = 4

    I31 面积编码:
    - use_area_encoding=True 启用位置编码增强
    - fourier_levels=4 控制傅里叶特征级别数

    默认配置优化:
    - channels-last: 默认启用 (CUDA 环境)
    - compile: 默认启用 (首次运行后加速)
    - 适合 RTX 4070 (8GB VRAM) 的 batch_size=64
    """
    name: str
    mode: str  # 'standard', 'hilbert', 'raster'
    description: str

    # 模型超参数 (最佳实践 - Tiny-ImageNet 调优)
    dim: int = 384
    depth: int = 12
    heads: int = 8
    mlp_ratio: float = 4.0

    # 正则化 (最佳实践 - I30-3)
    dropout: float = 0.2
    drop_path: float = 0.2
    weight_decay: float = 0.1

    # 训练超参数 (适合 8GB VRAM)
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    min_lr: float = 1e-6
    warmup_epochs: int = 5

    # 数据集
    image_size: int = 64
    num_classes: int = 200

    # I78 动态分辨率 Tokenizer 配置
    min_patch_size: int = 4  # 最小 patch 大小
    K_min: int = 16          # Token 数量下限 (信息论: log2(200) × 2 ≈ 16)
    K_max: int = 64          # Token 数量上限 (4:1 压缩比)

    # I31 面积编码配置 (2026-01-18)
    use_area_encoding: bool = True
    fourier_levels: int = 4

    # 性能优化 (I78 - 默认启用)
    use_channels_last: bool = True   # channels-last 内存格式 (~20% VRAM 节省)
    use_compile: bool = True         # torch.compile 优化 (~30% 训练加速)

    def __post_init__(self):
        if self.mode not in ['standard', 'hilbert', 'raster']:
            raise ValueError(f"Unknown mode: {self.mode}")


@dataclass
class RunResult:
    """单次运行结果"""
    run_id: int
    mode: str
    seed: int

    # 动态字段
    train_losses: List[float] = field(default_factory=list)
    train_accs: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_accs: List[float] = field(default_factory=list)

    # 汇总字段
    best_val_acc: float = 0.0
    best_epoch: int = 0
    final_val_acc: float = 0.0
    avg_epoch_time: float = 0.0
    throughput: float = 0.0
    total_params: int = 0


@dataclass
class ModeStatistics:
    """某个模式的统计结果"""
    results: List[RunResult]

    @property
    def mean_val_acc(self) -> float:
        return np.mean([r.best_val_acc for r in self.results])

    @property
    def std_val_acc(self) -> float:
        return np.std([r.best_val_acc for r in self.results])

    @property
    def mean_train_acc(self) -> float:
        return np.mean([r.train_accs[-1] if r.train_accs else 0 for r in self.results])

    @property
    def mean_epoch_time(self) -> float:
        return np.mean([r.avg_epoch_time for r in self.results])

    @property
    def throughput(self) -> float:
        return np.mean([r.throughput for r in self.results])

    @property
    def best_epoch(self) -> int:
        epochs = [r.best_epoch for r in self.results]
        return int(np.mean(epochs)) if epochs else 0

    @property
    def total_params(self) -> int:
        return self.results[0].total_params if self.results else 0


# ============================================================================
# 模型创建
# ============================================================================

def create_model(config: ExperimentConfig) -> nn.Module:
    """根据配置创建模型.

    设计原则:
    - Standard ViT: 参数量/计算量与 Fractal ViT 同级别 (公平对比基础)
    - Hilbert vs Raster: 唯一区别是 use_hilbert_encoding=True/False (严格控制变量)

    Args:
        config: 实验配置

    Returns:
        模型实例
    """
    if config.mode == 'standard':
        # Standard ViT 基线 (参数量与 Fractal ViT 同级别)
        return StandardViT(
            image_size=config.image_size,
            patch_size=16,  # 标准 16x16 patch
            num_classes=config.num_classes,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
        )
    else:
        # FractalCurveViT (I78 动态分辨率 + I31 面积编码)
        # Hilbert vs Raster: 唯一区别 use_hilbert_encoding
        return FractalCurveViT(
            image_size=config.image_size,
            num_classes=config.num_classes,
            dim=config.dim,
            num_layers=config.num_layers,
            heads=config.heads,
            mlp_dim=int(config.dim * config.mlp_ratio),
            # 正则化 (I30-3 最佳实践)
            dropout=config.dropout,
            drop_path_rate=config.drop_path,
            # 池化 (I30-11)
            pool="weighted",
            # I78: 动态分辨率 Tokenizer 配置
            min_patch_size=config.min_patch_size,
            K_min=config.K_min,
            K_max=config.K_max,
            # I31: 面积编码配置 (2026-01-18)
            use_area_encoding=config.use_area_encoding,
            fourier_levels=config.fourier_levels,
            # 核心变量: 排序方式 (Hilbert vs Raster)
            use_hilbert_encoding=(config.mode == 'hilbert'),
        )


# ============================================================================
# 训练与评估 (复用 examples/training/trainer/__init__.py 的 ModularTrainer.validate)
# ============================================================================

def set_seed(seed: int):
    """设置随机种子以确保可复现性."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = True,
    accumulation_steps: int = 1,
    epoch: int = 0,
    total_epochs: int = 0,
) -> Tuple[float, float]:
    """单轮训练 (参考 ModularTrainer.train_epoch)

    Returns:
        (avg_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    scaler = GradScaler('cuda') if use_amp else None

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=True)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # I78: 保持 channels-last 格式
        if images.dim() == 4:
            images = images.to(memory_format=torch.channels_last)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast('cuda', enabled=True):
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                if accumulation_steps > 1:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        # 统计
        batch_loss = loss.detach().item() * accumulation_steps
        total_loss += batch_loss

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 更新进度条
        current_acc = 100.0 * correct / max(total, 1)
        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'acc': f'{current_acc:.1f}%'
        })

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """验证 (完全复制 examples/training/trainer/__init__.py ModularTrainer.validate)

    关键点:
    1. 验证禁用 AMP (autocast enabled=False) - 确保指标精度
    2. 使用 detach().item() - 支持 torch.compile
    3. 禁用 channels-last - 避免精度问题

    Returns:
        (avg_loss, accuracy, num_batches)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    pbar = tqdm(val_loader, desc="Validating", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 验证禁用 AMP 以确保指标精度 (I78: 使用 detach().item() 支持 torch.compile)
        with autocast(device_type=device.type, enabled=False):
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

        total_loss += loss.detach().item()
        num_batches += 1

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 更新进度条
        current_acc = 100.0 * correct / max(total, 1)
        pbar.set_postfix({'acc': f'{current_acc:.1f}%'})

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy, num_batches


def run_experiment(
    config: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    run_id: int,
    seed: int,
    verbose: bool = True,
) -> RunResult:
    """运行单次实验.

    使用优化后的训练循环，支持:
    - I78 channels-last 内存格式 (~20% VRAM 节省)
    - I78 torch.compile 优化 (~30% 训练加速)
    - 混合精度训练 (AMP)

    Args:
        config: 实验配置
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 计算设备
        run_id: 运行编号
        seed: 随机种子
        verbose: 是否打印详细信息

    Returns:
        RunResult: 单次运行结果
    """
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    # 设置随机种子
    set_seed(seed)

    if verbose:
        print(f"\n{'='*60}")
        print(f"运行 {run_id}: {config.name} (mode={config.mode}, seed={seed})")
        if config.use_channels_last and device.type == 'cuda':
            print(f"  [I78] channels-last: ON")
        if config.use_compile:
            print(f"  [I78] torch.compile: ON")
        print(f"{'='*60}")

    # 创建模型
    model = create_model(config)

    # I78: channels-last 内存格式 (仅 CUDA 支持)
    if config.use_channels_last and device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
        if verbose:
            print("  [I78] 启用 channels-last 内存格式")

    model = model.to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())

    if verbose:
        print(f"  参数量: {total_params:,}")

    # 优化器 (使用最佳实践配置)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # 学习率调度器
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.epochs,
        T_mult=1,
        eta_min=config.min_lr,
    )

    # 结果记录
    result = RunResult(run_id=run_id, mode=config.mode, seed=seed)
    result.total_params = total_params

    epoch_times = []
    best_val_acc = 0.0
    best_epoch = 0

    # 训练循环
    for epoch in range(config.epochs):
        epoch_start = time.time()

        # Warmup 阶段
        if epoch < config.warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.learning_rate * (epoch + 1) / config.warmup_epochs

        # 训练 (参考 ModularTrainer.train_epoch)
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            use_amp=(device.type == 'cuda'),
            accumulation_steps=1,
            epoch=epoch,
            total_epochs=config.epochs,
        )

        # 验证 (完全复制 ModularTrainer.validate)
        val_loss, val_acc, num_batches = validate(
            model=model,
            val_loader=val_loader,
            device=device,
        )

        # 更新学习率
        if epoch >= config.warmup_epochs:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # 记录结果
        result.train_losses.append(train_loss)
        result.train_accs.append(train_acc)
        result.val_losses.append(val_loss)
        result.val_accs.append(val_acc)

        # 记录最佳验证准确率
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

        # 每10个epoch打印一次汇总信息
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config.epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.1f}%, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.1f}% [{epoch_time:.1f}s]")

    # 汇总结果
    result.best_val_acc = best_val_acc
    result.best_epoch = best_epoch
    result.final_val_acc = result.val_accs[-1] if result.val_accs else 0.0
    result.avg_epoch_time = np.mean(epoch_times)

    # 计算吞吐量
    num_epochs = len(result.train_losses)
    total_images = len(train_loader.dataset) * num_epochs
    total_time = sum(epoch_times)
    result.throughput = total_images / total_time if total_time > 0 else 0.0

    if verbose:
        print(f"\n  最佳验证准确率: {result.best_val_acc:.2f}% (Epoch {result.best_epoch+1})")
        print(f"  平均 Epoch 时间: {result.avg_epoch_time:.1f}s")
        print(f"  吞吐量: {result.throughput:.1f} images/sec")

    return result


# ============================================================================
# 统计分析
# ============================================================================

def compute_mode_statistics(results: List[RunResult]) -> ModeStatistics:
    """计算某个模式的统计结果"""
    return ModeStatistics(results=results)


def analyze_results(mode_results: Dict[str, ModeStatistics]) -> Dict[str, Dict[str, Any]]:
    """分析实验结果

    Args:
        mode_results: {mode: ModeStatistics}

    Returns:
        分析结果字典
    """
    analysis = {}

    for mode, stats in mode_results.items():
        analysis[mode] = {
            'mean_val_acc': stats.mean_val_acc,
            'std_val_acc': stats.std_val_acc,
            'mean_train_acc': stats.mean_train_acc,
            'mean_epoch_time': stats.mean_epoch_time,
            'throughput': stats.throughput,
            'best_epoch': stats.best_epoch,
            'total_params': stats.total_params,
        }

    return analysis


def print_analysis(analysis: Dict[str, Dict[str, Any]]) -> None:
    """打印分析结果"""
    print("\n" + "=" * 60)
    print("实验结果分析")
    print("=" * 60)

    # 排名
    sorted_modes = sorted(analysis.items(), key=lambda x: x[1]['mean_val_acc'], reverse=True)
    print("\n总体结果排名 (按验证准确率):")
    for i, (mode, stats) in enumerate(sorted_modes, 1):
        print(f"  {i}. {mode.upper()}: {stats['mean_val_acc']:.2f}% ± {stats['std_val_acc']:.2f}%")

    # 详细统计
    print("\n详细统计:")
    for mode, stats in analysis.items():
        print(f"\n  [{mode.upper()}]")
        print(f"    验证准确率: {stats['mean_val_acc']:.2f}% ± {stats['std_val_acc']:.2f}%")
        print(f"    训练准确率: {stats['mean_train_acc']:.2f}%")
        print(f"    平均 Epoch 时间: {stats['mean_epoch_time']:.1f}s")
        print(f"    吞吐量: {stats['throughput']:.1f} images/sec")
        print(f"    最佳 epoch: {stats['best_epoch']}")
        print(f"    参数量: {stats['total_params']:,}")


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hilbert vs Raster vs Standard ViT 消融实验 (I25-2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
控制变量实验设计:
  - Hilbert vs Raster: 唯一区别 use_hilbert_encoding=True/False
  - Standard vs Fractal: 参数量/计算量同级别

推荐实验流程:
  1. 快速验证: --quick
  2. 核心对比: --modes hilbert raster --runs 3
  3. 完整消融: --all --runs 3
  4. I31 消融: --modes hilbert --no-area-encoding

性能优化:
  - 默认启用 channels-last + torch.compile (CUDA)
  - 使用 --no-channels-last --no-compile 禁用

示例:
  # 完整消融实验 (3 模式 × 3 次 = 9 次训练)
  uv run python tests/benchmarks/ablation_hilbert_curve.py --all --runs 3

  # 只运行 Hilbert vs Raster (核心假设验证)
  uv run python tests/benchmarks/ablation_hilbert_curve.py --modes hilbert raster --runs 3

  # 快速测试 (1 epoch, 1 run)
  uv run python tests/benchmarks/ablation_hilbert_curve.py --quick

  # 保存结果
  uv run python tests/benchmarks/ablation_hilbert_curve.py --output results.json

  # 禁用性能优化 (CPU 或调试模式)
  uv run python tests/benchmarks/ablation_hilbert_curve.py --all --no-channels-last --no-compile

  # I31 面积编码消融
  uv run python tests/benchmarks/ablation_hilbert_curve.py --modes hilbert --no-area-encoding
        """
    )

    # 运行模式
    parser.add_argument(
        '--all', action='store_true',
        help='运行所有三种模式 (standard, hilbert, raster)'
    )
    parser.add_argument(
        '--modes', nargs='+',
        choices=['standard', 'hilbert', 'raster'],
        default=['hilbert'],
        help='运行模式 (默认: hilbert)'
    )
    parser.add_argument(
        '--runs', type=int, default=3,
        help='每个模式的独立运行次数 (默认: 3)'
    )

    # 训练参数
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='训练 epochs 数 (默认: 100)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='批次大小 (默认: 64)'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='学习率 (默认: 1e-3)'
    )

    # I78 性能优化参数 (默认启用)
    parser.add_argument(
        '--channels-last', action='store_true', dest='use_channels_last',
        help='启用 channels-last 内存格式 (约20%% VRAM 节省)'
    )
    parser.add_argument(
        '--no-channels-last', action='store_false', dest='use_channels_last',
        help='禁用 channels-last'
    )
    parser.add_argument(
        '--compile', action='store_true', dest='use_compile',
        help='启用 torch.compile 优化 (约30%% 训练加速)'
    )
    parser.add_argument(
        '--no-compile', action='store_false', dest='use_compile',
        help='禁用 torch.compile'
    )

    # I31 面积编码参数
    parser.add_argument(
        '--area-encoding', action='store_true', dest='use_area_encoding',
        help='启用面积编码 (位置编码增强)'
    )
    parser.add_argument(
        '--no-area-encoding', action='store_false', dest='use_area_encoding',
        help='禁用面积编码 (用于 I31 消融实验)'
    )
    parser.add_argument(
        '--fourier-levels', type=int, default=4,
        help='傅里叶特征级别数 (默认: 4)'
    )

    # 其他参数
    parser.add_argument(
        '--quick', action='store_true',
        help='快速测试模式 (10 epochs, 1 run)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='结果输出文件路径 (JSON)'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='设备 (cuda/cpu/auto)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='基础随机种子 (默认: 42)'
    )

    # 设置默认参数
    parser.set_defaults(
        use_channels_last=True,
        use_compile=True,
        use_area_encoding=True,
    )

    args = parser.parse_args()

    # 快速测试模式覆盖
    if args.quick:
        args.epochs = 10
        args.runs = 1
        print("快速测试模式: 10 epochs, 1 run per mode")

    # 处理 --all 参数
    if args.all:
        args.modes = ['standard', 'hilbert', 'raster']

    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    # I78 性能优化状态
    print(f"\n[I78] 性能优化:")
    print(f"  channels-last: {'ON' if args.use_channels_last and device.type == 'cuda' else 'OFF (CPU)'}")
    print(f"  torch.compile: {'ON' if args.use_compile else 'OFF'}")

    # I31 面积编码配置
    print(f"\n[I31] 面积编码:")
    print(f"  use_area_encoding: {args.use_area_encoding}")
    print(f"  fourier_levels: {args.fourier_levels}")

    # 加载数据 (自动下载并组织验证集)
    print("\n加载 Tiny-ImageNet 数据...")
    print("  (自动组织验证集格式)")
    try:
        train_loader, val_loader, num_classes = get_tiny_imagenet_loaders(
            batch_size=args.batch_size,
            image_size=64,
        )
    except Exception as e:
        print(f"\n[ERROR] 数据集准备失败: {e}")
        print("\n请手动下载 Tiny-ImageNet:")
        print("  1. 下载: http://cs231n.stanford.edu/tiny-imagenet-200.zip")
        print(f"  2. 解压到: {PROJECT_ROOT / 'data' / 'tiny-imagenet-200'}")
        return

    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")

    # 创建基础配置
    base_config = ExperimentConfig(
        name="Ablation",
        mode="standard",  # 会被覆盖
        description="基础配置",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_classes=num_classes,
        # I78 性能优化
        use_channels_last=args.use_channels_last and device.type == 'cuda',
        use_compile=args.use_compile,
        # I31 面积编码
        use_area_encoding=args.use_area_encoding,
        fourier_levels=args.fourier_levels,
    )

    # 运行实验
    all_results: Dict[str, List[RunResult]] = {mode: [] for mode in args.modes}

    for mode in args.modes:
        print(f"\n{'#'*60}")
        print(f"# 模式: {mode.upper()}")
        print(f"{'#'*60}")

        config = ExperimentConfig(
            name=f"{mode.upper()}",
            mode=mode,
            description=f"{mode} 模式",
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_classes=num_classes,
            # I78 性能优化
            use_channels_last=args.use_channels_last and device.type == 'cuda',
            use_compile=args.use_compile,
            # I31 面积编码
            use_area_encoding=args.use_area_encoding,
            fourier_levels=args.fourier_levels,
        )

        for run_id in range(1, args.runs + 1):
            seed = args.seed + run_id  # 每个运行使用不同种子

            try:
                result = run_experiment(
                    config, train_loader, val_loader, device,
                    run_id, seed, verbose=True
                )
                all_results[mode].append(result)
            except Exception as e:
                print(f"\n[ERROR] 运行 {run_id} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue

    # 统计分析
    mode_results = {}
    for mode, results in all_results.items():
        if results:
            mode_results[mode] = compute_mode_statistics(results)

    if mode_results:
        analysis = analyze_results(mode_results)
        print_analysis(analysis)

        # 保存结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        exp_name = f"ablation_hilbert_{timestamp}"

        # 创建 experiments 目录
        EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
        EXPERIMENTS_DIR.mkdir(exist_ok=True)
        exp_dir = EXPERIMENTS_DIR / exp_name
        exp_dir.mkdir(exist_ok=True)

        # 保存实验数据 (JSON)
        output_data = {
            'config': asdict(base_config),
            'mode_results': {
                mode: {
                    'statistics': asdict(stat),
                    'runs': [asdict(r) for r in stat.results]
                }
                for mode, stat in mode_results.items()
            },
            'analysis': analysis,
            'timestamp': timestamp,
            'modes': list(args.modes),
        }
        data_path = exp_dir / "experiment_data.json"
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        # 保存实验日志
        log_content = []
        log_content.append("=" * 60)
        log_content.append("Hilbert vs Raster 消融实验日志")
        log_content.append(f"时间戳: {timestamp}")
        log_content.append("=" * 60)
        log_content.append(f"\n[I78] 性能优化:")
        log_content.append(f"  channels-last: {'ON' if args.use_channels_last and device.type == 'cuda' else 'OFF'}")
        log_content.append(f"  torch.compile: {'ON' if args.use_compile else 'OFF'}")
        log_content.append(f"\n[I31] 面积编码:")
        log_content.append(f"  use_area_encoding: {args.use_area_encoding}")
        log_content.append(f"  fourier_levels: {args.fourier_levels}")
        log_content.append(f"\n训练配置:")
        log_content.append(f"  epochs: {args.epochs}")
        log_content.append(f"  batch_size: {args.batch_size}")
        log_content.append(f"  learning_rate: {args.lr}")
        log_content.append(f"  runs per mode: {args.runs}")
        log_content.append(f"\n运行模式: {', '.join(args.modes)}")
        log_content.append("\n" + "=" * 60)
        log_content.append("实验结果分析")
        log_content.append("=" * 60)

        # 添加分析结果
        if analysis:
            log_content.append(f"\n总体结果排名 (按验证准确率):")
            sorted_modes = sorted(analysis.items(), key=lambda x: x[1]['mean_val_acc'], reverse=True)
            for i, (mode, stats) in enumerate(sorted_modes, 1):
                log_content.append(f"  {i}. {mode.upper()}: {stats['mean_val_acc']:.2f}% ± {stats['std_val_acc']:.2f}%")

            log_content.append(f"\n详细统计:")
            for mode, stats in analysis.items():
                log_content.append(f"\n  [{mode.upper()}]")
                log_content.append(f"    验证准确率: {stats['mean_val_acc']:.2f}% ± {stats['std_val_acc']:.2f}%")
                log_content.append(f"    训练准确率: {stats['mean_train_acc']:.2f}%")
                log_content.append(f"    平均 Epoch 时间: {stats['mean_epoch_time']:.1f}s")
                log_content.append(f"    吞吐量: {stats['throughput']:.1f} images/sec")
                log_content.append(f"    最佳 epoch: {stats['best_epoch']}")
                log_content.append(f"    参数量: {stats['total_params']:,}")

        log_content.append("\n" + "=" * 60)
        log_content.append("消融实验结论")
        log_content.append("=" * 60)

        # 生成消融实验结论
        if 'hilbert' in mode_results and 'raster' in mode_results:
            hilbert_acc = mode_results['hilbert'].mean_val_acc
            raster_acc = mode_results['raster'].mean_val_acc
            diff = hilbert_acc - raster_acc

            log_content.append(f"\n[Hilbert vs Raster 对比]")
            log_content.append(f"  Hilbert:  {hilbert_acc:.2f}%")
            log_content.append(f"  Raster:   {raster_acc:.2f}%")
            log_content.append(f"  差异:     {diff:+.2f}%")

            if diff > 1.0:
                log_content.append(f"  结论: Hilbert 排序显著优于 Raster (+{diff:.2f}%)")
            elif diff < -1.0:
                log_content.append(f"  结论: Raster 排序优于 Hilbert ({diff:.2f}%)")
            else:
                log_content.append(f"  结论: Hilbert 与 Raster 差异不显著 (|diff| < 1%)")

        if 'standard' in mode_results:
            for fractal_mode in ['hilbert', 'raster']:
                if fractal_mode in mode_results:
                    standard_acc = mode_results['standard'].mean_val_acc
                    fractal_acc = mode_results[fractal_mode].mean_val_acc
                    diff = fractal_acc - standard_acc

                    log_content.append(f"\n[Standard vs {fractal_mode.upper()}]")
                    log_content.append(f"  Standard: {standard_acc:.2f}%")
                    log_content.append(f"  {fractal_mode.upper()}: {fractal_acc:.2f}%")
                    log_content.append(f"  差异:     {diff:+.2f}%")

                    if diff > 1.0:
                        log_content.append(f"  结论: 分形 tokenization 提升 {diff:.2f}%")
                    elif diff < -1.0:
                        log_content.append(f"  结论: 分形 tokenization 下降 {abs(diff):.2f}%")
                    else:
                        log_content.append(f"  结论: 分形 tokenization 差异不显著")

        log_content.append("\n" + "=" * 60)

        # 保存日志
        log_path = exp_dir / "experiment_log.txt"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(log_content))

        # 打印日志内容
        print('\n'.join(log_content))

        print(f"\n实验数据已保存到: {data_path}")
        print(f"实验日志已保存到: {log_path}")

        # 保存到 args.output (如果指定)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"结果已保存到: {args.output}")

        # 运行最终评估 - 消融实验关键指标对比
        print("\n" + "=" * 60)
        print("最终评估报告")
        print("=" * 60)

        # 计算并输出消融实验所需的关键数据
        eval_results = {
            'hilbert_vs_raster': {},
            'fractal_vs_standard': {},
        }

        if 'hilbert' in mode_results and 'raster' in mode_results:
            hilbert = mode_results['hilbert']
            raster = mode_results['raster']
            eval_results['hilbert_vs_raster'] = {
                'hilbert_val_acc': hilbert.mean_val_acc,
                'raster_val_acc': raster.mean_val_acc,
                'improvement': hilbert.mean_val_acc - raster.mean_val_acc,
                'hilbert_throughput': hilbert.throughput,
                'raster_throughput': raster.throughput,
                'hilbert_best_epoch': hilbert.best_epoch,
                'raster_best_epoch': raster.best_epoch,
            }
            print(f"\n[Hilbert vs Raster 评估]")
            print(f"  验证准确率: Hilbert={hilbert.mean_val_acc:.2f}%, Raster={raster.mean_val_acc:.2f}%")
            print(f"  提升幅度: {hilbert.mean_val_acc - raster.mean_val_acc:+.2f}%")
            print(f"  吞吐量: Hilbert={hilbert.throughput:.1f}, Raster={raster.throughput:.1f} images/sec")

        if 'standard' in mode_results:
            for mode in ['hilbert', 'raster']:
                if mode in mode_results:
                    standard = mode_results['standard']
                    fractal = mode_results[mode]
                    eval_results['fractal_vs_standard'][mode] = {
                        f'{mode}_val_acc': fractal.mean_val_acc,
                        'standard_val_acc': standard.mean_val_acc,
                        'improvement': fractal.mean_val_acc - standard.mean_val_acc,
                        f'{mode}_params': fractal.total_params,
                        'standard_params': standard.total_params,
                    }
                    print(f"\n[Standard vs {mode.upper()} 评估]")
                    print(f"  验证准确率: Standard={standard.mean_val_acc:.2f}%, {mode.upper()}={fractal.mean_val_acc:.2f}%")
                    print(f"  提升幅度: {fractal.mean_val_acc - standard.mean_val_acc:+.2f}%")
                    print(f"  参数量: Standard={standard.total_params:,}, {mode.upper()}={fractal.total_params:,}")

        # 保存评估结果
        eval_path = exp_dir / "evaluation_results.json"
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        print(f"\n评估结果已保存到: {eval_path}")


if __name__ == '__main__':
    main()
