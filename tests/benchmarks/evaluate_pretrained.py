#!/usr/bin/env python3
"""Evaluate pretrained Fractal ViT models and generate visualizations.

This script loads trained .pth model checkpoints and evaluates them with:
- Performance metrics (forward/backward pass timing)
- Tokenization analysis (adaptive token counts, level distribution)
- Accuracy evaluation on test data
- Visualization of results (attention maps, token distributions, etc.)

Usage:
    python -m tests.benchmarks.evaluate_pretrained --checkpoint path/to/best.pth
    python -m tests.benchmarks.evaluate_pretrained --checkpoint path/to/best.pth --visualize --dataset cifar10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vit_pytorch.fractal_vit import NextGenerationFractalViT, SimpleFractalViT

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import torchvision.transforms as transforms
    from torchvision.datasets import CIFAR10, CIFAR100, MNIST
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


# ============================================================================
# Data Classes for Results
# ============================================================================

@dataclass
class TokenizationResult:
    """Tokenization analysis result for a single image."""
    num_tokens: int = 0
    levels_used: List[int] = field(default_factory=list)
    max_level: int = 0
    patch_sizes: List[Tuple[int, int]] = field(default_factory=list)
    complexity_score: float = 0.0


@dataclass
class EvaluationMetrics:
    """Overall evaluation metrics."""
    # Accuracy
    accuracy: float = 0.0
    top5_accuracy: float = 0.0
    
    # Performance
    forward_time_ms: float = 0.0
    total_time_ms: float = 0.0
    images_per_second: float = 0.0
    
    # Tokenization
    avg_tokens: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    token_std: float = 0.0
    
    # Model info
    num_parameters: int = 0
    model_type: str = ""
    checkpoint_path: str = ""
    epoch: int = 0
    val_acc_from_checkpoint: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'accuracy': self.accuracy,
            'top5_accuracy': self.top5_accuracy,
            'forward_time_ms': self.forward_time_ms,
            'total_time_ms': self.total_time_ms,
            'images_per_second': self.images_per_second,
            'avg_tokens': self.avg_tokens,
            'min_tokens': self.min_tokens,
            'max_tokens': self.max_tokens,
            'token_std': self.token_std,
            'num_parameters': self.num_parameters,
            'model_type': self.model_type,
            'checkpoint_path': self.checkpoint_path,
            'epoch': self.epoch,
            'val_acc_from_checkpoint': self.val_acc_from_checkpoint,
        }


# ============================================================================
# Model Loading
# ============================================================================

def infer_mlp_dim_from_state_dict(state_dict: Dict[str, Any], default_dim: int) -> int:
    """Infer mlp_dim from the saved state_dict.
    
    The FeedForward layer's first linear weight has shape [mlp_dim, dim].
    """
    for key in state_dict.keys():
        if 'ff' in key and 'main_net.0.weight' in key:
            return state_dict[key].shape[0]
    return default_dim * 4  # fallback


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a pretrained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: Device to load the model on
        
    Returns:
        Tuple of (model, checkpoint_info)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract args/config
    args = checkpoint.get('args', {})
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    
    state_dict = checkpoint['model']
    
    # Determine model type and create model
    use_simple = args.get('use_simple', True)
    force_next_gen = args.get('force_next_gen', False)
    
    # Get dataset info for num_classes
    dataset_name = args.get('dataset', 'cifar10')
    num_classes_map = {
        'cifar10': 10,
        'cifar100': 100,
        'mnist': 10,
        'imagenet': 1000,
        'caltech256': 257,
    }
    num_classes = num_classes_map.get(dataset_name, 10)
    
    # Get image size from dataset
    image_size_map = {
        'cifar10': 32,
        'cifar100': 32,
        'mnist': 28,
        'imagenet': 224,
        'caltech256': 224,
    }
    image_size = image_size_map.get(dataset_name, 32)
    
    # Model parameters
    dim = args.get('dim', 128)
    depth = args.get('depth', 4)
    heads = args.get('heads', 4)
    dim_head = args.get('dim_head', 32)
    dropout = args.get('dropout', 0.1)
    emb_dropout = args.get('emb_dropout', 0.1)
    max_level = args.get('max_level', 4)
    pool = args.get('pool', 'cls')
    channels = 1 if dataset_name == 'mnist' else 3
    
    # Infer mlp_dim from saved weights
    mlp_dim = infer_mlp_dim_from_state_dict(state_dict, dim)
    
    if use_simple and not force_next_gen:
        model = SimpleFractalViT(
            image_size=image_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            max_level=max_level,
        )
        model_type = "SimpleFractalViT"
    else:
        learnable_split = not args.get('no_learnable_split', False)
        model = NextGenerationFractalViT(
            image_size=image_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            max_level=max_level,
            learnable_split=learnable_split,
        )
        model_type = "NextGenerationFractalViT"
    
    # Load weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    checkpoint_info = {
        'args': args,
        'epoch': checkpoint.get('epoch', 0),
        'val_acc': checkpoint.get('val_acc', 0.0),
        'model_type': model_type,
        'num_classes': num_classes,
        'image_size': image_size,
        'dataset': dataset_name,
        'channels': channels,
    }
    
    return model, checkpoint_info


# ============================================================================
# Dataset Loading
# ============================================================================

def get_test_dataloader(
    dataset_name: str,
    batch_size: int = 32,
    num_samples: Optional[int] = None,
    data_root: Optional[str] = None,
) -> DataLoader:
    """Get test dataloader for a dataset.
    
    Args:
        dataset_name: Name of the dataset (cifar10, cifar100, mnist)
        batch_size: Batch size
        num_samples: Optional number of samples to use (for quick testing)
        data_root: Optional data root directory
        
    Returns:
        DataLoader for test set
    """
    if not HAS_TORCHVISION:
        raise ImportError("torchvision is required for dataset loading")
    
    # Use project root data directory by default
    if data_root:
        root = data_root
    else:
        root = str(PROJECT_ROOT / "data")
    
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        # Try loading first, download if needed
        try:
            dataset = CIFAR10(root=root, train=False, download=False, transform=transform)
        except RuntimeError:
            print(f"    Dataset not found locally, downloading to {root}...")
            dataset = CIFAR10(root=root, train=False, download=True, transform=transform)
    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        try:
            dataset = CIFAR100(root=root, train=False, download=False, transform=transform)
        except RuntimeError:
            print(f"    Dataset not found locally, downloading to {root}...")
            dataset = CIFAR100(root=root, train=False, download=True, transform=transform)
    elif dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        try:
            dataset = MNIST(root=root, train=False, download=False, transform=transform)
        except RuntimeError:
            print(f"    Dataset not found locally, downloading to {root}...")
            dataset = MNIST(root=root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    if num_samples is not None and num_samples < len(dataset):
        indices = list(range(num_samples))
        dataset = Subset(dataset, indices)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )


# ============================================================================
# Evaluation Functions
# ============================================================================

@torch.no_grad()
def analyze_tokenization(
    model: nn.Module,
    images: torch.Tensor,
) -> List[TokenizationResult]:
    """Analyze tokenization for a batch of images."""
    results = []
    
    # Get tokenizer
    tokenizer = getattr(model, "tokenizer", None) or getattr(model, "fractal_tokenizer", None)
    if tokenizer is None:
        enhanced = getattr(model, "enhanced_model", None)
        if enhanced:
            tokenizer = getattr(enhanced, "tokenizer", None)
    
    if tokenizer is None:
        # Return default results if no tokenizer found
        for _ in range(images.size(0)):
            results.append(TokenizationResult())
        return results
    
    # Analyze each image
    for i in range(images.size(0)):
        img = images[i:i+1]
        
        # Compute image complexity (gradient magnitude)
        if img.dim() == 4:
            gray = img.mean(dim=1, keepdim=True)
        else:
            gray = img
        
        dx = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
        dy = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
        complexity = (dx.mean() + dy.mean()).item() / 2
        
        # Tokenize and get info
        try:
            tokens, info = tokenizer(img)
            num_tokens = tokens.size(1)
            levels_info = info.get('levels_info', [])
            levels_used = [li.get('level', 0) for li in levels_info] if levels_info else []
            max_level = max(levels_used) if levels_used else 0
            
            results.append(TokenizationResult(
                num_tokens=num_tokens,
                levels_used=levels_used,
                max_level=max_level,
                complexity_score=complexity,
            ))
        except Exception:
            results.append(TokenizationResult(complexity_score=complexity))
    
    return results


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    analyze_tokens: bool = True,
) -> EvaluationMetrics:
    """Evaluate model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to run on
        analyze_tokens: Whether to analyze tokenization
        
    Returns:
        EvaluationMetrics with results
    """
    model.eval()
    
    correct = 0
    top5_correct = 0
    total = 0
    total_time = 0.0
    
    all_token_counts: List[int] = []
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)
        
        # Time forward pass
        start = time.perf_counter()
        outputs = model(images)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = time.perf_counter() - start
        total_time += elapsed
        
        # Compute accuracy
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        
        # Top-5 accuracy
        if outputs.size(1) >= 5:
            _, top5_pred = outputs.topk(5, dim=1)
            top5_correct += sum(labels[i] in top5_pred[i] for i in range(batch_size))
        else:
            top5_correct += correct
        
        total += batch_size
        
        # Analyze tokenization (on first batch only to save time)
        if analyze_tokens and len(all_token_counts) == 0:
            token_results = analyze_tokenization(model, images)
            all_token_counts.extend([r.num_tokens for r in token_results])
    
    # Compute metrics
    metrics = EvaluationMetrics()
    metrics.accuracy = 100.0 * correct / total
    metrics.top5_accuracy = 100.0 * top5_correct / total
    metrics.total_time_ms = total_time * 1000
    metrics.forward_time_ms = metrics.total_time_ms / len(dataloader)
    metrics.images_per_second = total / total_time
    
    # Token statistics
    if all_token_counts:
        metrics.avg_tokens = np.mean(all_token_counts)
        metrics.min_tokens = min(all_token_counts)
        metrics.max_tokens = max(all_token_counts)
        metrics.token_std = np.std(all_token_counts)
    
    # Model info
    metrics.num_parameters = sum(p.numel() for p in model.parameters())
    
    return metrics


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_tokenization_analysis(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    output_path: Path,
    num_samples: int = 8,
) -> None:
    """Plot tokenization analysis for sample images."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping visualization")
        return
    
    model.eval()
    
    # Select samples
    num_samples = min(num_samples, images.size(0))
    sample_images = images[:num_samples]
    sample_labels = labels[:num_samples]
    
    # Create figure
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2.5, 5))
    
    for i in range(num_samples):
        img = sample_images[i]
        label = sample_labels[i].item()
        
        # Original image
        img_np = img.cpu().numpy()
        if img_np.shape[0] == 1:
            img_np = img_np[0]  # Grayscale
        else:
            img_np = img_np.transpose(1, 2, 0)  # CHW -> HWC
        
        # Denormalize for visualization
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        axes[0, i].imshow(img_np, cmap='gray' if len(img_np.shape) == 2 else None)
        axes[0, i].set_title(f"{class_names[label]}", fontsize=9)
        axes[0, i].axis('off')
        
        # Tokenization analysis
        token_results = analyze_tokenization(model, img.unsqueeze(0).to(next(model.parameters()).device))
        result = token_results[0]
        
        # Create bar chart of levels used
        if result.levels_used:
            level_counts = {}
            for lv in result.levels_used:
                level_counts[lv] = level_counts.get(lv, 0) + 1
            levels = list(level_counts.keys())
            counts = list(level_counts.values())
            axes[1, i].bar(levels, counts, color='steelblue')
            axes[1, i].set_xlabel('Level')
            axes[1, i].set_ylabel('Count')
        else:
            axes[1, i].text(0.5, 0.5, f"Tokens: {result.num_tokens}", 
                          ha='center', va='center', fontsize=10)
        axes[1, i].set_title(f"Tokens: {result.num_tokens}", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path / 'tokenization_analysis.png', dpi=150)
    plt.close()
    print(f"Saved tokenization analysis to {output_path / 'tokenization_analysis.png'}")


def plot_complexity_vs_tokens(
    model: nn.Module,
    images: torch.Tensor,
    output_path: Path,
) -> None:
    """Plot relationship between image complexity and token count."""
    if not HAS_MATPLOTLIB:
        return
    
    model.eval()
    device = next(model.parameters()).device
    
    # Analyze all images
    token_results = analyze_tokenization(model, images.to(device))
    
    complexities = [r.complexity_score for r in token_results]
    token_counts = [r.num_tokens for r in token_results]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(complexities, token_counts, alpha=0.6, c='steelblue')
    plt.xlabel('Image Complexity (Gradient Magnitude)')
    plt.ylabel('Number of Tokens')
    plt.title('Image Complexity vs Token Count')
    
    # Add trend line
    if len(set(token_counts)) > 1:
        z = np.polyfit(complexities, token_counts, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(complexities), max(complexities), 100)
        plt.plot(x_line, p(x_line), "r--", alpha=0.8, label='Trend')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'complexity_vs_tokens.png', dpi=150)
    plt.close()
    print(f"Saved complexity analysis to {output_path / 'complexity_vs_tokens.png'}")


def plot_confusion_matrix(
    predictions: List[int],
    labels: List[int],
    class_names: List[str],
    output_path: Path,
) -> None:
    """Plot confusion matrix."""
    if not HAS_MATPLOTLIB:
        return
    
    num_classes = len(class_names)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for pred, label in zip(predictions, labels):
        matrix[label, pred] += 1
    
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha='right', fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix.png', dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {output_path / 'confusion_matrix.png'}")


def plot_performance_summary(
    metrics: EvaluationMetrics,
    checkpoint_info: Dict[str, Any],
    output_path: Path,
) -> None:
    """Plot performance summary."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Accuracy
    axes[0].bar(['Top-1', 'Top-5'], [metrics.accuracy, metrics.top5_accuracy], color=['steelblue', 'lightsteelblue'])
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy')
    axes[0].set_ylim(0, 100)
    for i, v in enumerate([metrics.accuracy, metrics.top5_accuracy]):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center')
    
    # Throughput
    axes[1].bar(['Images/sec'], [metrics.images_per_second], color='forestgreen')
    axes[1].set_ylabel('Images per Second')
    axes[1].set_title('Throughput')
    axes[1].text(0, metrics.images_per_second + 5, f'{metrics.images_per_second:.1f}', ha='center')
    
    # Model info
    info_text = f"""Model: {checkpoint_info['model_type']}
Dataset: {checkpoint_info['dataset']}
Epochs: {checkpoint_info['epoch']}
Parameters: {metrics.num_parameters:,}
Checkpoint Acc: {checkpoint_info['val_acc']:.1f}%
Eval Accuracy: {metrics.accuracy:.2f}%"""
    
    axes[2].text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                 fontfamily='monospace', transform=axes[2].transAxes)
    axes[2].axis('off')
    axes[2].set_title('Model Info')
    
    plt.tight_layout()
    plt.savefig(output_path / 'performance_summary.png', dpi=150)
    plt.close()
    print(f"Saved performance summary to {output_path / 'performance_summary.png'}")


# ============================================================================
# Main
# ============================================================================

def run_evaluation(
    checkpoint_path: str,
    output_dir: Optional[str] = None,
    visualize: bool = True,
    num_samples: Optional[int] = None,
    batch_size: int = 32,
) -> EvaluationMetrics:
    """Run full evaluation of a pretrained model.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint
        output_dir: Output directory for results and visualizations
        visualize: Whether to generate visualizations
        num_samples: Number of samples to evaluate (None for all)
        batch_size: Batch size for evaluation
        
    Returns:
        EvaluationMetrics with results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("Pretrained Model Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    
    # Setup output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Path(checkpoint_path).parent.parent / 'evaluation'
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n[1] Loading checkpoint...")
    model, checkpoint_info = load_checkpoint(checkpoint_path, device)
    print(f"    Model type: {checkpoint_info['model_type']}")
    print(f"    Dataset: {checkpoint_info['dataset']}")
    print(f"    Epoch: {checkpoint_info['epoch']}")
    print(f"    Checkpoint val_acc: {checkpoint_info['val_acc']:.2f}%")
    
    # Get dataloader
    print("\n[2] Loading test data...")
    dataloader = get_test_dataloader(
        checkpoint_info['dataset'],
        batch_size=batch_size,
        num_samples=num_samples,
    )
    print(f"    Test samples: {len(dataloader.dataset)}")
    
    # Class names for visualization
    class_names_map = {
        'cifar10': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck'],
        'cifar100': [f'class_{i}' for i in range(100)],
        'mnist': [str(i) for i in range(10)],
    }
    class_names = class_names_map.get(checkpoint_info['dataset'], [])
    
    # Evaluate
    print("\n[3] Evaluating model...")
    metrics = evaluate_model(model, dataloader, device, analyze_tokens=True)
    metrics.model_type = checkpoint_info['model_type']
    metrics.checkpoint_path = checkpoint_path
    metrics.epoch = checkpoint_info['epoch']
    metrics.val_acc_from_checkpoint = checkpoint_info['val_acc']
    
    print(f"\n    Accuracy: {metrics.accuracy:.2f}%")
    print(f"    Top-5 Accuracy: {metrics.top5_accuracy:.2f}%")
    print(f"    Throughput: {metrics.images_per_second:.1f} images/sec")
    print(f"    Avg tokens: {metrics.avg_tokens:.1f}")
    print(f"    Parameters: {metrics.num_parameters:,}")
    
    # Generate visualizations
    if visualize and HAS_MATPLOTLIB:
        print("\n[4] Generating visualizations...")
        
        # Get sample batch for visualization
        sample_iter = iter(dataloader)
        sample_images, sample_labels = next(sample_iter)
        
        # Tokenization analysis
        plot_tokenization_analysis(
            model, sample_images, sample_labels,
            class_names, output_path, num_samples=8
        )
        
        # Complexity vs tokens
        plot_complexity_vs_tokens(model, sample_images, output_path)
        
        # Performance summary
        plot_performance_summary(metrics, checkpoint_info, output_path)
        
        # Confusion matrix (for small number of samples)
        if num_samples and num_samples <= 1000:
            predictions = []
            labels_list = []
            model.eval()
            with torch.no_grad():
                for images, labels in dataloader:
                    outputs = model(images.to(device))
                    _, preds = outputs.max(1)
                    predictions.extend(preds.cpu().tolist())
                    labels_list.extend(labels.tolist())
            plot_confusion_matrix(predictions, labels_list, class_names, output_path)
    
    # Save results
    results_path = output_path / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'metrics': metrics.to_dict(),
            'checkpoint_info': {k: v for k, v in checkpoint_info.items() 
                               if not isinstance(v, (torch.Tensor, np.ndarray))},
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained Fractal ViT models")
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to the .pth checkpoint file')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Output directory for results and visualizations')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--num-samples', '-n', type=int, default=None,
                       help='Number of samples to evaluate (default: all)')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    run_evaluation(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        visualize=args.visualize,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
