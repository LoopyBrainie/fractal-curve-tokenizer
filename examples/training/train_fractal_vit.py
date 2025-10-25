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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from tqdm import tqdm
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vit_pytorch.fractal_vit import NextGenerationFractalViT, SimpleFractalViT


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
    }

    if dataset not in specs:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return specs[dataset]


def build_transforms(spec: DatasetSpec) -> Tuple[transforms.Compose, transforms.Compose]:
    if spec.name.lower() == "mnist":
        train_ops = [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(spec.mean, spec.std)]
        test_ops = train_ops.copy()
    else:
        train_ops = [
            transforms.Resize(spec.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
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
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if subset_size is not None and subset_size < 2:
        raise ValueError("subset_size must be at least 2 when provided")

    data_root = PROJECT_ROOT / "workspace" / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    train_transform, test_transform = build_transforms(spec)
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
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)

    for data, target in progress:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast_context(device, scaler.is_enabled()):
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            loss = F.cross_entropy(output, target)

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
    return avg_loss, accuracy


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


def save_history(paths: ExperimentPaths, args: argparse.Namespace, history: Dict[str, list], summary: Dict[str, float]) -> None:
    payload = {
        "arguments": vars(args),
        "history": history,
        "summary": summary,
        "timestamp": paths.timestamp,
    }
    history_path = paths.experiment_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

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
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "mnist"], default="cifar10")
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

    train_loader, val_loader, test_loader = create_dataloaders(
        spec,
        batch_size=args.batch_size,
        val_split=args.val_split,
        subset_size=args.subset_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(args.epochs // 4, 1),
        T_mult=2,
        eta_min=args.lr * 0.01,
    )
    scaler = create_grad_scaler(args.use_amp, device)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_state: Optional[Dict[str, object]] = None

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            args.epochs,
            scaler,
            args.gradient_clip,
        )
        val_loss, val_acc = evaluate(model, val_loader, device, desc="val")
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{args.epochs} → train: loss={train_loss:.4f}, acc={train_acc:.2f}% | val: loss={val_loss:.4f}, acc={val_acc:.2f}%"
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
    save_history(paths, args, history, summary)
    plot_curves(paths, history)

    print("Training finished")
    print(f"Best val accuracy: {best_val_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Artifacts: {paths.experiment_dir}")


if __name__ == "__main__":
    main()