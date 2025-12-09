#!/usr/bin/env python3
"""CNN Ablation Benchmark: 对比有/无 CNN 特征对分割决策的影响。

此脚本用于验证在 Tokenizer 中使用 CNN 特征是否能带来显著收益。

实验设置:
- 基线 (Baseline): 仅使用手工特征 (6维: level, height, width, variance, mean, edge_density)
- 对照组 (With CNN): 手工特征 + CNN 特征 (6+32=38维)

评估指标:
- 训练损失收敛速度
- 验证准确率
- Token 数量分布
- 训练时间 (每 epoch)

使用方法:
    python tests/benchmarks/benchmark_cnn_ablation.py --dataset cifar10 --epochs 20
    python tests/benchmarks/benchmark_cnn_ablation.py --dataset tiny-imagenet --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vit_pytorch.fractal_vit import NextGenerationFractalViT


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    use_cnn: bool
    learnable_split: bool = True
    
    # 模型参数
    dim: int = 192
    depth: int = 6
    heads: int = 6
    max_level: int = 4
    
    # 训练参数
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 0.01
    epochs: int = 20
    
    # 其他
    seed: int = 42


@dataclass
class ExperimentResult:
    """实验结果"""
    config: ExperimentConfig
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    token_stats: List[Dict[str, Any]] = field(default_factory=list)
    best_val_acc: float = 0.0
    total_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "name": self.config.name,
                "use_cnn": self.config.use_cnn,
                "learnable_split": self.config.learnable_split,
                "dim": self.config.dim,
                "depth": self.config.depth,
                "epochs": self.config.epochs,
            },
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "epoch_times": self.epoch_times,
            "token_stats": self.token_stats,
            "best_val_acc": self.best_val_acc,
            "total_time": self.total_time,
        }


def set_seed(seed: int) -> None:
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(
    dataset_name: str,
    batch_size: int,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """创建数据加载器
    
    Returns:
        (train_loader, val_loader, num_classes, image_size)
    """
    if dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        num_classes = 10
        image_size = 32
        dataset_cls = CIFAR10
    elif dataset_name == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        num_classes = 100
        image_size = 32
        dataset_cls = CIFAR100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    data_root = PROJECT_ROOT / "workspace" / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    
    train_dataset = dataset_cls(root=str(data_root), train=True, download=True, transform=train_transform)
    val_dataset = dataset_cls(root=str(data_root), train=False, download=True, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, num_classes, image_size


def create_model(
    config: ExperimentConfig,
    num_classes: int,
    image_size: int,
    device: torch.device,
) -> nn.Module:
    """创建模型"""
    model = NextGenerationFractalViT(
        image_size=image_size,
        num_classes=num_classes,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        mlp_dim=config.dim * 2,
        channels=3,
        dim_head=32,
        dropout=0.1,
        emb_dropout=0.1,
        min_patch_size=(4, 4),
        max_level=config.max_level,
        learnable_split=config.learnable_split,
        # 关键参数: 是否使用 CNN
        # 需要通过 tokenizer 传递
    )
    
    # 设置 tokenizer 的 use_cnn 参数
    if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'use_cnn'):
        model.tokenizer.use_cnn = config.use_cnn
        
        # 如果需要 CNN 但还没初始化
        if config.use_cnn and model.tokenizer.cnn_encoder is None:
            from vit_pytorch.fractal_curve_tokenizer import MiniCNN, LearnableSplitDecision
            model.tokenizer.cnn_encoder = MiniCNN(in_channels=3, hidden_dim=16, out_dim=32).to(device)
            model.tokenizer.split_decision = LearnableSplitDecision(
                patch_features=6, cnn_features=32, hidden_dim=64
            ).to(device)
    
    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, List[int]]:
    """训练一个 epoch
    
    Returns:
        (avg_loss, token_counts_per_image)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    token_counts: List[int] = []
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        output = model(images)
        if isinstance(output, tuple):
            output = output[0]
        
        loss = F.cross_entropy(output, labels)
        
        # 添加 tokenizer 辅助损失
        if hasattr(model, "get_tokenizer_loss"):
            with torch.no_grad():
                reward = -loss.item()
            aux_loss = model.get_tokenizer_loss(reward=reward, baseline=0.0, entropy_coef=0.01)
            loss = loss + aux_loss
            if hasattr(model, "clear_tokenizer_cache"):
                model.clear_tokenizer_cache()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1), token_counts


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """评估模型
    
    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        output = model(images)
        if isinstance(output, tuple):
            output = output[0]
        
        loss = F.cross_entropy(output, labels)
        total_loss += loss.item()
        
        preds = output.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / max(len(loader), 1)
    accuracy = 100.0 * correct / max(total, 1)
    
    return avg_loss, accuracy


@torch.no_grad()
def analyze_tokenization(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_samples: int = 100,
) -> Dict[str, Any]:
    """分析 tokenization 统计"""
    model.eval()
    tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
    
    if tokenizer is None:
        return {"error": "No tokenizer found"}
    
    token_counts: List[int] = []
    sample_count = 0
    
    for images, _ in loader:
        images = images.to(device)
        output = tokenizer.tokenize(images)
        
        for seq in output.sequences:
            token_counts.append(seq.tokens.shape[0])
            sample_count += 1
            if sample_count >= num_samples:
                break
        
        if sample_count >= num_samples:
            break
    
    if not token_counts:
        return {"error": "No tokens generated"}
    
    return {
        "mean_tokens": float(np.mean(token_counts)),
        "std_tokens": float(np.std(token_counts)),
        "min_tokens": int(np.min(token_counts)),
        "max_tokens": int(np.max(token_counts)),
        "token_range": int(np.max(token_counts) - np.min(token_counts)),
    }


def run_experiment(
    config: ExperimentConfig,
    dataset_name: str,
    device: torch.device,
) -> ExperimentResult:
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"实验: {config.name}")
    print(f"使用 CNN: {config.use_cnn}")
    print(f"{'='*60}")
    
    set_seed(config.seed)
    
    # 创建数据
    train_loader, val_loader, num_classes, image_size = create_dataloaders(
        dataset_name, config.batch_size
    )
    
    # 创建模型
    model = create_model(config, num_classes, image_size, device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {total_params:,} (可训练: {trainable_params:,})")
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr * 0.01)
    
    # 结果记录
    result = ExperimentResult(config=config)
    start_time = time.time()
    
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        
        # 训练
        train_loss, _ = train_one_epoch(model, train_loader, optimizer, device)
        
        # 验证
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # 记录结果
        result.train_losses.append(train_loss)
        result.val_losses.append(val_loss)
        result.val_accuracies.append(val_acc)
        result.epoch_times.append(epoch_time)
        
        if val_acc > result.best_val_acc:
            result.best_val_acc = val_acc
        
        print(f"Epoch {epoch}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")
        
        # 定期分析 tokenization
        if epoch % 5 == 0 or epoch == 1:
            token_stats = analyze_tokenization(model, val_loader, device)
            token_stats["epoch"] = epoch
            result.token_stats.append(token_stats)
            print(f"  Token stats: mean={token_stats.get('mean_tokens', 'N/A'):.1f}, "
                  f"range=[{token_stats.get('min_tokens', 'N/A')}-{token_stats.get('max_tokens', 'N/A')}]")
    
    result.total_time = time.time() - start_time
    
    return result


def compare_results(results: List[ExperimentResult]) -> None:
    """对比实验结果"""
    print("\n" + "="*80)
    print("实验对比结果")
    print("="*80)
    
    print(f"\n{'实验名称':<25} | {'最佳验证准确率':>12} | {'平均 Epoch 时间':>12} | {'总时间':>10}")
    print("-" * 70)
    
    for result in results:
        avg_epoch_time = np.mean(result.epoch_times)
        print(f"{result.config.name:<25} | "
              f"{result.best_val_acc:>11.2f}% | "
              f"{avg_epoch_time:>11.1f}s | "
              f"{result.total_time:>9.1f}s")
    
    # 收敛速度对比
    print("\n收敛速度对比 (达到各准确率阈值的 epoch):")
    thresholds = [50, 60, 70, 80]
    
    print(f"{'实验名称':<25}", end=" | ")
    for t in thresholds:
        print(f"{t}% acc", end=" | ")
    print()
    print("-" * 70)
    
    for result in results:
        print(f"{result.config.name:<25}", end=" | ")
        for t in thresholds:
            epoch = next((i+1 for i, acc in enumerate(result.val_accuracies) if acc >= t), "-")
            print(f"{str(epoch):>6}", end=" | ")
        print()
    
    # Token 统计对比
    print("\n最终 Token 统计:")
    print(f"{'实验名称':<25} | {'平均 Token':>10} | {'Token 范围':>12}")
    print("-" * 55)
    
    for result in results:
        if result.token_stats:
            last_stats = result.token_stats[-1]
            mean_tokens = last_stats.get('mean_tokens', 'N/A')
            token_range = f"[{last_stats.get('min_tokens', '?')}-{last_stats.get('max_tokens', '?')}]"
            print(f"{result.config.name:<25} | {mean_tokens:>10.1f} | {token_range:>12}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CNN Ablation Benchmark")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=str, default=None, help="保存结果的 JSON 文件路径")
    args = parser.parse_args()
    
    # 设备选择
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 定义实验配置
    configs = [
        ExperimentConfig(
            name="Baseline (无 CNN)",
            use_cnn=False,
            learnable_split=True,
            dim=args.dim,
            depth=args.depth,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        ),
        ExperimentConfig(
            name="With CNN (有 CNN)",
            use_cnn=True,
            learnable_split=True,
            dim=args.dim,
            depth=args.depth,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        ),
    ]
    
    # 运行实验
    results: List[ExperimentResult] = []
    for config in configs:
        result = run_experiment(config, args.dataset, device)
        results.append(result)
    
    # 对比结果
    compare_results(results)
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "workspace" / "results" / "benchmarks"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"cnn_ablation_{args.dataset}_{timestamp}.json"
    
    output_data = {
        "dataset": args.dataset,
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "results": [r.to_dict() for r in results],
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
