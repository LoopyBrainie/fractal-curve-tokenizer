# -*- coding: utf-8 -*-
"""
Hilbert vs Raster 消融实验 (I25-2)

实验设计:
┌──────────────┬────────────────────────────────────────────────────────────┐
│ 模式         │ 说明                                                         │
├──────────────┼────────────────────────────────────────────────────────────┤
│ Standard ViT | 标准 ViT (16x16 patch) - 基线，对比分形 tokenization 收益      │
│ Hilbert      | FractalCurveViT + Hilbert 排序 - 核心假设验证                  │
│ Raster       | FractalCurveViT + Raster 排序 - Hilbert 对照组                 │
└──────────────┴────────────────────────────────────────────────────────────┘

控制变量:
- dropout = 0.25
- drop_path = 0.25
- weight_decay = 0.15
- pool = weighted
- epochs = 60

严格控制:
- Hilbert vs Raster: 唯一区别 use_hilbert_encoding=True/False
- Standard vs Fractal: 参数量/计算量同级别 (公平对比)

参考实现: https://github.com/lucidrains/vit-pytorch

Usage:
    # 完整消融实验 (3 模式 × 3 次 = 9 次训练)
    uv run python tests/benchmarks/ablation_hilbert_curve.py --all --epochs 60

    # 只运行 Hilbert vs Raster (核心对比)
    uv run python tests/benchmarks/ablation_hilbert_curve.py --modes hilbert raster --runs 3

    # 快速测试
    uv run python tests/benchmarks/ablation_hilbert_curve.py --quick

    # 保存结果
    uv run python tests/benchmarks/ablation_hilbert_curve.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Handle import paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "examples" / "training"))

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from vit_pytorch import FractalCurveViT


# ============================================================================
# Standard ViT 基线模型 (参考: https://github.com/lucidrains/vit-pytorch)
# ============================================================================

class FeedForward(nn.Module):
    """前馈网络 (GELU 激活)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """多头自注意力机制."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.25):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    """Transformer 编码器."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class StandardViT(nn.Module):
    """标准 Vision Transformer (固定 patch 大小).

    参考: https://github.com/lucidrains/vit-pytorch

    特点:
    - 固定 16x16 patch 划分
    - 可学习位置编码
    - CLS token 用于分类
    - GELU 激活函数

    参数量级别与 Fractal ViT 相当 (~31M)，用于公平对比。
    """

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 200,
        dim: int = 320,
        depth: int = 12,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim

        # 验证图像尺寸可被 patch 整除
        assert image_size % patch_size == 0, "图像尺寸必须能被 patch 大小整除"

        # Patch 数量和维度
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        # Patch 嵌入 (使用 Rearrange 替代 Conv2d)
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_size,
                p2=patch_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # 位置编码和 CLS token
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, dim) * 0.02
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer 编码器
        mlp_dim = int(dim * mlp_ratio)
        self.transformer = Transformer(
            dim, depth, heads, dim // heads, mlp_dim, dropout
        )

        # 分类头
        self.mlp_head = nn.Linear(dim, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.zeros_(self.mlp_head.weight)
        nn.init.zeros_(self.mlp_head.bias)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        batch = img.shape[0]

        # Patch 嵌入
        x = self.to_patch_embedding(img)

        # 添加 CLS token
        cls_tokens = repeat(self.cls_token, "... d -> b ... d", b=batch)
        x = torch.cat((cls_tokens, x), dim=1)

        # 添加位置编码
        seq_len = x.shape[1]
        x = x + self.pos_embedding[:, :seq_len]
        x = self.dropout(x)

        # Transformer 编码
        x = self.transformer(x)

        # 分类
        x = x[:, 0]
        return self.mlp_head(x)


# ============================================================================
# 实验配置
# ============================================================================

@dataclass
class ExperimentConfig:
    """消融实验配置"""
    name: str
    mode: str  # 'standard', 'hilbert', 'raster'
    description: str

    # 模型超参数 (最佳实践)
    dim: int = 320
    depth: int = 12
    heads: int = 8
    mlp_ratio: float = 4.0

    # 正则化 (最佳实践 - I30-3)
    dropout: float = 0.25
    drop_path: float = 0.25
    weight_decay: float = 0.15

    # 训练超参数
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3
    min_lr: float = 1e-6
    warmup_epochs: int = 5

    # 数据集
    image_size: int = 64
    num_classes: int = 200

    def __post_init__(self):
        if self.mode not in ['standard', 'hilbert', 'raster']:
            raise ValueError(f"Unknown mode: {self.mode}")


@dataclass
class RunResult:
    """单次运行结果"""
    run_id: int
    mode: str
    seed: int

    # 训练指标
    train_losses: List[float] = field(default_factory=list)
    train_accs: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_accs: List[float] = field(default_factory=list)

    # 最终指标
    best_val_acc: float = 0.0
    best_epoch: int = 0
    final_val_acc: float = 0.0

    # 计算指标
    total_params: int = 0
    avg_epoch_time: float = 0.0
    throughput: float = 0.0


@dataclass
class ModeResult:
    """单个模式的聚合结果"""
    mode: str
    results: List[RunResult] = field(default_factory=list)

    # 聚合统计
    mean_acc: float = 0.0
    std_acc: float = 0.0
    best_acc: float = 0.0
    ci_95: Tuple[float, float] = (0.0, 0.0)


# ============================================================================
# 数据集下载工具 (从 train_fractal_vit.py 复制)
# ============================================================================

def download_with_progress(url: str, dest: Path, desc: str = "Downloading") -> bool:
    """带进度条的下载函数"""
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            total_size = int(response.headers.get('Content-Length', 0))

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
    """下载并设置 Tiny ImageNet (从 train_fractal_vit.py 复制)

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
        if not (target_dir / "val").exists():
            raise FileNotFoundError("val directory not found after extraction")

        return True
    except Exception as e:
        print(f"\n[ERROR] Extraction failed: {e}")
        return False


# ============================================================================
# 数据加载
# ============================================================================

def get_tiny_imagenet_loaders(
    batch_size: int = 64,
    image_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, int]:
    """获取 Tiny-ImageNet 数据加载器 (复用 train_fractal_vit.py 的数据增强策略)

    Returns:
        train_loader, val_loader, num_classes
    """
    data_root = PROJECT_ROOT / "data"
    data_root.mkdir(exist_ok=True)

    # 下载数据集
    if not download_tiny_imagenet(data_root):
        raise FileNotFoundError("Failed to download Tiny ImageNet")

    tiny_imagenet_dir = data_root / "tiny-imagenet-200"

    # 数据增强 (与 train_fractal_vit.py 一致)
    mean = [0.4802, 0.4481, 0.3975]
    std = [0.2302, 0.2265, 0.2262]

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

    # 加载数据集
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
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, 200


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
        # Standard ViT 基线
        # 参数量级别: ~31M (与 Fractal ViT 相当)
        return StandardViT(
            image_size=config.image_size,
            patch_size=16,  # 固定 16x16 patch
            num_classes=config.num_classes,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
        )
    else:
        # FractalCurveViT
        # Hilbert vs Raster: 唯一区别 use_hilbert_encoding
        return FractalCurveViT(
            image_size=config.image_size,
            num_classes=config.num_classes,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            mlp_dim=int(config.dim * config.mlp_ratio),
            # 正则化 (I30-3 最佳实践)
            dropout=config.dropout,
            drop_path_rate=config.drop_path,
            # 池化 (I30-11)
            pool="weighted",
            # 核心变量: 排序方式
            use_hilbert_encoding=(config.mode == 'hilbert'),
            # 其他配置 (固定)
            min_patch_size=4,
            max_depth_hard_limit=8,
        )


# ============================================================================
# 训练与评估
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[float, float]:
    """训练一个 epoch.

    Returns:
        avg_loss, accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """评估模型.

    Returns:
        avg_loss, accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


def set_seed(seed: int):
    """设置随机种子以确保可复现性."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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
    # 设置随机种子
    set_seed(seed)

    if verbose:
        print(f"\n{'='*60}")
        print(f"运行 {run_id}: {config.name} (mode={config.mode}, seed={seed})")
        print(f"{'='*60}")

    # 创建模型
    model = create_model(config)
    model = model.to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())

    if verbose:
        print(f"参数量: {total_params:,}")

    # 优化器 (使用最佳实践配置)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # 学习率调度 (余弦退火 + Warmup)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.epochs,
        T_mult=1,
        eta_min=config.min_lr,
    )

    # 混合精度训练 (如果可用)
    scaler = None
    if device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()

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
            # 线性 warmup
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.learning_rate * (epoch + 1) / config.warmup_epochs

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, device)

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config.epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.1f}%, "
                  f"Val Acc={val_acc:.1f}% [{epoch_time:.1f}s]")

    # 汇总结果
    result.best_val_acc = best_val_acc
    result.best_epoch = best_epoch
    result.final_val_acc = result.val_accs[-1] if result.val_accs else 0.0
    result.avg_epoch_time = np.mean(epoch_times)

    # 计算吞吐量
    total_images = len(train_loader.dataset) * config.epochs
    total_time = sum(epoch_times)
    result.throughput = total_images / total_time if total_time > 0 else 0.0

    if verbose:
        print(f"\n最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch+1})")
        print(f"平均 Epoch 时间: {result.avg_epoch_time:.1f}s")
        print(f"吞吐量: {result.throughput:.1f} images/sec")

    return result


# ============================================================================
# 统计分析
# ============================================================================

def compute_mode_statistics(results: List[RunResult]) -> ModeResult:
    """计算单个模式的统计信息.

    Args:
        results: 该模式下所有运行的结果

    Returns:
        ModeResult: 聚合统计结果
    """
    if not results:
        return ModeResult(mode="")

    mode_result = ModeResult(mode=results[0].mode)

    # 收集所有最佳验证准确率
    best_accs = [r.best_val_acc for r in results]

    # 计算统计量
    mode_result.results = results
    mode_result.mean_acc = np.mean(best_accs)
    mode_result.std_acc = np.std(best_accs)
    mode_result.best_acc = np.max(best_accs)

    # 95% 置信区间
    if len(best_accs) > 1:
        from scipy import stats as scipy_stats
        mean = np.mean(best_accs)
        std = np.std(best_accs, ddof=1)
        se = std / np.sqrt(len(best_accs))
        ci = scipy_stats.t.interval(0.95, df=len(best_accs)-1, loc=mean, scale=se)
        mode_result.ci_95 = (ci[0], ci[1])
    else:
        mode_result.ci_95 = (best_accs[0], best_accs[0])

    return mode_result


def compute_effect_size(acc1: List[float], acc2: List[float]) -> float:
    """计算 Cohen's d 效应量.

    Args:
        acc1: 第一组的准确率列表
        acc2: 第二组的准确率列表

    Returns:
        Cohen's d 值
    """
    n1, n2 = len(acc1), len(acc2)
    if n1 < 2 or n2 < 2:
        return 0.0

    mean1, mean2 = np.mean(acc1), np.mean(acc2)
    var1, var2 = np.var(acc1, ddof=1), np.var(acc2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-8:
        return 0.0

    return (mean1 - mean2) / pooled_std


def compute_paired_ttest(acc1: List[float], acc2: List[float]) -> float:
    """计算配对 t-test p 值.

    Args:
        acc1: 第一组的准确率列表
        acc2: 第二组的准确率列表

    Returns:
        p 值
    """
    from scipy import stats as scipy_stats

    if len(acc1) != len(acc2) or len(acc1) < 2:
        return 1.0

    _, p_value = scipy_stats.ttest_rel(acc1, acc2)
    return p_value if not np.isnan(p_value) else 1.0


def analyze_results(
    mode_results: Dict[str, ModeResult]
) -> Dict[str, Any]:
    """分析消融实验结果.

    Args:
        mode_results: 各模式的聚合结果

    Returns:
        分析报告字典
    """
    analysis = {
        'summary': {},
        'comparisons': {},
        'recommendations': [],
    }

    # 提取摘要
    for mode, result in mode_results.items():
        accs = [r.best_val_acc for r in result.results]
        analysis['summary'][mode] = {
            'mean_acc': result.mean_acc,
            'std_acc': result.std_acc,
            'best_acc': result.best_acc,
            'ci_95': result.ci_95,
            'runs': len(accs),
            'total_params': result.results[0].total_params if result.results else 0,
            'throughput': result.results[0].throughput if result.results else 0,
        }

    # 模式间比较
    modes = list(mode_results.keys())

    # Hilbert vs Raster
    if 'hilbert' in modes and 'raster' in modes:
        hilbert_accs = [r.best_val_acc for r in mode_results['hilbert'].results]
        raster_accs = [r.best_val_acc for r in mode_results['raster'].results]

        diff = np.mean(hilbert_accs) - np.mean(raster_accs)
        p_value = compute_paired_ttest(hilbert_accs, raster_accs)
        effect = compute_effect_size(hilbert_accs, raster_accs)

        analysis['comparisons']['hilbert_vs_raster'] = {
            'hilbert_mean': np.mean(hilbert_accs),
            'raster_mean': np.mean(raster_accs),
            'improvement': diff,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'cohens_d': effect,
            'effect_interpretation': '大' if abs(effect) > 0.8 else ('中' if abs(effect) > 0.5 else '小'),
        }

    # Hilbert vs Standard
    if 'hilbert' in modes and 'standard' in modes:
        hilbert_accs = [r.best_val_acc for r in mode_results['hilbert'].results]
        standard_accs = [r.best_val_acc for r in mode_results['standard'].results]

        diff = np.mean(hilbert_accs) - np.mean(standard_accs)
        p_value = compute_paired_ttest(hilbert_accs, standard_accs)
        effect = compute_effect_size(hilbert_accs, standard_accs)

        analysis['comparisons']['hilbert_vs_standard'] = {
            'hilbert_mean': np.mean(hilbert_accs),
            'standard_mean': np.mean(standard_accs),
            'improvement': diff,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'cohens_d': effect,
            'effect_interpretation': '大' if abs(effect) > 0.8 else ('中' if abs(effect) > 0.5 else '小'),
        }

    # Raster vs Standard
    if 'raster' in modes and 'standard' in modes:
        raster_accs = [r.best_val_acc for r in mode_results['raster'].results]
        standard_accs = [r.best_val_acc for r in mode_results['standard'].results]

        diff = np.mean(raster_accs) - np.mean(standard_accs)
        p_value = compute_paired_ttest(raster_accs, standard_accs)
        effect = compute_effect_size(raster_accs, standard_accs)

        analysis['comparisons']['raster_vs_standard'] = {
            'raster_mean': np.mean(raster_accs),
            'standard_mean': np.mean(standard_accs),
            'improvement': diff,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'cohens_d': effect,
            'effect_interpretation': '大' if abs(effect) > 0.8 else ('中' if abs(effect) > 0.5 else '小'),
        }

    # 生成建议
    if 'hilbert_vs_raster' in analysis['comparisons']:
        comp = analysis['comparisons']['hilbert_vs_raster']
        if comp['significant'] and comp['improvement'] > 0.5:
            analysis['recommendations'].append(
                f"✅ Hilbert 排序有效: 提升 {comp['improvement']:.2f}%, p={comp['p_value']:.4f}"
            )
        elif comp['significant'] and comp['improvement'] < -0.5:
            analysis['recommendations'].append(
                f"⚠️ Raster 优于 Hilbert: 差异 {abs(comp['improvement']):.2f}%, p={comp['p_value']:.4f}"
            )
        else:
            analysis['recommendations'].append(
                f"⚠️ Hilbert vs Raster 无显著差异: {comp['improvement']:+.2f}%, p={comp['p_value']:.4f}"
            )

    return analysis


def print_analysis(analysis: Dict[str, Any]) -> None:
    """打印分析结果."""
    print("\n" + "="*70)
    print("                Hilbert vs Raster vs Standard ViT 消融实验报告")
    print("="*70)

    # 摘要表格
    print("\n📊 实验结果摘要:")
    print("-"*70)
    print(f"{'模式':<15} {'最佳准确率':>12} {'平均准确率':>12} {'标准差':>10} {'参数量':>12}")
    print("-"*70)

    for mode, summary in analysis['summary'].items():
        mode_name = {'standard': 'Standard ViT', 'hilbert': 'Hilbert', 'raster': 'Raster'}.get(mode, mode)
        print(f"{mode_name:<15} {summary['best_acc']:>11.2f}% {summary['mean_acc']:>11.2f}% "
              f"± {summary['std_acc']:>8.2f}% {summary['total_params']:>12,}")
    print("-"*70)

    # 模式间比较
    print("\n🔬 统计检验结果:")
    print("-"*70)

    for name, comp in analysis['comparisons'].items():
        print(f"\n{name.replace('_', ' ').title()}:")
        print(f"  差异: {comp['improvement']:+.2f}%")
        print(f"  p-value: {comp['p_value']:.4f} ({'显著' if comp['significant'] else '不显著'})")
        print(f"  Cohen's d: {comp['cohens_d']:.2f} ({comp['effect_interpretation']}效应)")

    # 建议
    print("\n💡 结论与建议:")
    print("-"*70)
    for rec in analysis['recommendations']:
        print(f"  {rec}")

    print("\n" + "="*70)


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hilbert vs Raster vs Standard ViT 消融实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整消融实验 (3 模式 × 3 次 = 9 次训练)
  uv run python tests/benchmarks/ablation_hilbert_curve.py --all --epochs 60

  # 只运行 Hilbert vs Raster
  uv run python tests/benchmarks/ablation_hilbert_curve.py --modes hilbert raster --runs 3

  # 快速测试
  uv run python tests/benchmarks/ablation_hilbert_curve.py --quick

  # 保存结果
  uv run python tests/benchmarks/ablation_hilbert_curve.py --output results.json
        """
    )

    # 运行模式
    parser.add_argument(
        '--all', action='store_true',
        help='运行所有三种模式 (standard, hilbert, raster)'
    )
    parser.add_argument(
        '--modes', nargs='+', default=['standard', 'hilbert', 'raster'],
        choices=['standard', 'hilbert', 'raster'],
        help='要运行的模式 (默认: all)'
    )
    parser.add_argument(
        '--runs', type=int, default=3,
        help='每个模式的独立运行次数 (默认: 3)'
    )

    # 训练参数
    parser.add_argument(
        '--epochs', type=int, default=60,
        help='训练 epochs 数 (默认: 60)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='批次大小 (默认: 64)'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='学习率 (默认: 1e-3)'
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

    args = parser.parse_args()

    # 快速测试模式覆盖
    if args.quick:
        args.epochs = 10
        args.runs = 1
        print("快速测试模式: 10 epochs, 1 run per mode")

    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 加载数据 (自动下载)
    print("\n加载 Tiny-ImageNet 数据...")
    print("  (如需手动下载，请参考 --help)")
    try:
        train_loader, val_loader, num_classes = get_tiny_imagenet_loaders(
            batch_size=args.batch_size,
            image_size=64,
        )
    except (FileNotFoundError, Exception) as e:
        print(f"\n[ERROR] 数据集准备失败: {e}")
        print("\n请手动下载 Tiny-ImageNet:")
        print("  1. 下载: http://cs231n.stanford.edu/tiny-imagenet-200.zip")
        print(f"  2. 解压到: {PROJECT_ROOT / 'data' / 'tiny-imagenet-200'}")
        return

    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")

    # 创建配置
    base_config = ExperimentConfig(
        name="Ablation",
        mode="standard",  # 会被覆盖
        description="基础配置",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_classes=num_classes,
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
        )

        for run_id in range(1, args.runs + 1):
            seed = args.seed + run_id  # 每个运行使用不同种子

            try:
                result = run_experiment(
                    config, train_loader, val_loader, device,
                    run_id=run_id, seed=seed, verbose=True
                )
                all_results[mode].append(result)
            except Exception as e:
                print(f"\n❌ 运行 {run_id} 失败: {e}")
                import traceback
                traceback.print_exc()

    # 分析结果
    if all_results and any(len(results) > 0 for results in all_results.values()):
        mode_results = {}
        for mode, results in all_results.items():
            if results:
                mode_results[mode] = compute_mode_statistics(results)

        if mode_results:
            analysis = analyze_results(mode_results)
            print_analysis(analysis)

            # 保存结果
            if args.output:
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
                }
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
