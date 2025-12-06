#!/usr/bin/env python3
"""Lightweight Fractal ViT training entry point for quick experiments."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

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

from vit_pytorch.fractal_vit import NextGenerationFractalViT, SimpleFractalViT


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


def build_transforms(spec: DatasetSpec) -> Tuple[transforms.Compose, transforms.Compose]:
    if spec.name.lower() == "mnist":
        train_ops = [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(spec.mean, spec.std)]
        test_ops = train_ops.copy()
    else:
        # 增强的数据增强策略：Mixup/CutMix 需要在 Batch 层面做，这里做基础增强
        # 引入 AutoAugment 或 RandAugment
        train_ops = [
            transforms.Resize(spec.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(spec.image_size, padding=4), # 增加 RandomCrop
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10), # 引入 AutoAugment
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
            transforms.RandomErasing(p=0.25), # 引入 RandomErasing
        ]
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
    
    # Helper for creating loaders
    def make_loader(dataset: Dataset, sampler_indices: np.ndarray, shuffle: bool = False) -> DataLoader:
        if shuffle:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        sampler = SubsetRandomSampler(sampler_indices.tolist())
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

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
        if not train_dir.exists() or not val_dir.exists():
             raise FileNotFoundError(f"Tiny ImageNet requires 'train' and 'val' folders in {data_root}")
        
        # Train set is standard ImageFolder
        train_dataset = ImageFolder(str(train_dir), transform=train_transform)
        
        # Val set needs custom loader to handle annotations file
        # We pass train_dataset.class_to_idx to ensure class mapping consistency
        val_dataset = TinyImageNetVal(val_dir, train_dataset.class_to_idx, transform=test_transform)
        
        # Tiny ImageNet 'test' set has no labels, so we use 'val' set for testing as well
        test_dataset = val_dataset 
        
        # Create loaders
        # Note: Tiny ImageNet Val is already a separate split, so we don't need to split train_dataset
        # unless user wants to carve out a validation set from train.
        # Standard practice: Train on 'train', Evaluate on 'val'.
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        
        print(f"Loaded {spec.name} → train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
        return train_loader, val_loader, test_loader

    if spec.name == "ImageNet":
        # Expects root to have train/val folders
        train_dir = data_root / "train"
        val_dir = data_root / "val"
        if not train_dir.exists() or not val_dir.exists():
             raise FileNotFoundError(f"ImageNet requires 'train' and 'val' folders in {data_root}")
        train_dataset = spec.dataset_cls(root=str(train_dir), transform=train_transform)
        test_dataset = spec.dataset_cls(root=str(val_dir), transform=test_transform)
    elif spec.name == "COCO":
        # Expects standard COCO structure
        train_img_dir = data_root / "train2017"
        val_img_dir = data_root / "val2017"
        train_ann = data_root / "annotations" / "instances_train2017.json"
        val_ann = data_root / "annotations" / "instances_val2017.json"
        
        if not train_img_dir.exists() or not train_ann.exists():
             raise FileNotFoundError(f"COCO requires train2017/val2017 and annotations in {data_root}")

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


def build_model(args: argparse.Namespace, spec: DatasetSpec) -> nn.Module:
    model_kwargs = {
        "image_size": max(spec.image_size, 32),
        "num_classes": spec.num_classes,
        "dim": args.dim,
        "depth": args.depth,
        "heads": args.heads,
        "mlp_dim": args.dim * 2,
        "channels": spec.channels,
        "dropout": args.dropout,
        "emb_dropout": args.emb_dropout,
        "min_patch_size": (4, 4),
        "max_level": args.max_level,
    }
    if args.use_simple:
        simple_keys = {
            "image_size",
            "num_classes",
            "dim",
            "depth",
            "heads",
            "mlp_dim",
            "channels",
            "dropout",
            "emb_dropout",
            "min_patch_size",
            "max_level",
        }
        simple_kwargs = {k: v for k, v in model_kwargs.items() if k in simple_keys}
        simple_kwargs["pool"] = args.pool
        simple_kwargs["dim_head"] = args.dim_head
        model = SimpleFractalViT(**simple_kwargs)
    else:
        model = NextGenerationFractalViT(
            **model_kwargs,
            pool=args.pool,
            dim_head=args.dim_head,
            learnable_split=not args.no_learnable_split,
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: total={total_params:,}, trainable={trainable_params:,}")
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
) -> Tuple[float, float, float]:
    """
    训练一个 epoch
    
    Args:
        baseline_ema: REINFORCE 基线的指数移动平均值，用于减少方差
    
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

    for data, target in progress:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()
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
            
            loss = ce_loss + aux_loss

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        running_loss += loss.item()
        preds = output.argmax(dim=1)
        correct += preds.eq(target).sum().item()
        total += target.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0 * correct / max(total, 1):.1f}%")

    avg_loss = running_loss / max(len(loader), 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy, baseline_ema


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, desc: str) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(loader, desc=desc, leave=False):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
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


def main() -> None:
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
    parser.add_argument("--use-simple", action="store_true")
    parser.add_argument("--no-learnable-split", action="store_true")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable-hilbert-bias",
        action="store_true",
        help="Turn off Hilbert-path attention bias to speed up CPU training.",
    )
    parser.add_argument(
        "--force-next-gen",
        action="store_true",
        help="Keep NextGeneration model even on CPU (may be very slow).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Select training device; 'auto' prefers CUDA when available.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    if args.quick_test:
        if args.epochs == parser.get_default("epochs"):
            args.epochs = 5
        if args.subset_size is None:
            args.subset_size = 512
        print("Quick-test mode: epochs capped and subset sampling enabled")

    device = resolve_device(args.device)

    spec = get_dataset_spec(args.dataset)
    paths = prepare_experiment_paths("fractal_vit_simple")

    if device.type == "cpu" and not args.use_simple and not args.force_next_gen:
        print("CPU detected; switching to SimpleFractalViT with lighter configuration for faster training.")
        args.use_simple = True
        args.dim = min(args.dim, 128)
        args.depth = min(args.depth, 4)
        args.heads = min(args.heads, 4)
        args.dim_head = min(args.dim_head, 32)

    model = build_model(args, spec)

    disable_hilbert_bias = args.disable_hilbert_bias or device.type == "cpu"
    if disable_hilbert_bias:
        for layer in getattr(getattr(model, "transformer", None), "layers", []):
            attention = getattr(layer, "attention", None)
            if attention is not None and hasattr(attention, "use_hilbert_bias"):
                attention.use_hilbert_bias = False
        if device.type == "cpu" and not args.disable_hilbert_bias:
            print("Hilbert-path attention bias disabled automatically for CPU training.")

    model = model.to(device)

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

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, baseline_ema = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            args.epochs,
            scaler,
            args.gradient_clip,
            baseline_ema=baseline_ema,
        )
        val_loss, val_acc = evaluate(model, val_loader, device, desc="val")
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

    elapsed = time.time() - start_time
    if best_state is not None:
        model.load_state_dict(best_state["model"])  # type: ignore[arg-type]

    test_loss, test_acc = evaluate(model, test_loader, device, desc="test")
    summary = {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "epochs_ran": len(history["train_loss"]),
        "elapsed_seconds": elapsed,
    }
    save_history(paths, detailed_config, history, summary)
    plot_curves(paths, history)

    print("Training finished")
    print(f"Best val accuracy: {best_val_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Artifacts: {paths.experiment_dir}")


if __name__ == "__main__":
    main()