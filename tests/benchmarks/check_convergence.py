"""
Convergence Analysis for Fractal ViT Models.

This module provides tools to evaluate and compare the training
convergence characteristics of Fractal ViT models.

Key Metrics:
1. Convergence speed (epochs to reach target accuracy/loss)
2. Training stability (loss variance, gradient statistics)
3. Overfitting detection (train-val gap)
4. Learning dynamics (loss improvement rate)
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Handle import paths
try:
    from vit_pytorch import FractalCurveViT
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
    from vit_pytorch import FractalCurveViT

from .benchmark_metrics import (
    ConvergenceMetrics,
    Timer,
    compute_statistics,
)


# ============================================================================
# Synthetic Dataset Generators
# ============================================================================

def create_color_classification_dataset(
    num_samples: int = 500,
    image_size: int = 32,
    num_classes: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a synthetic dataset for color-based classification."""
    images = []
    labels = []
    
    for i in range(num_samples):
        label = i % num_classes
        img = torch.zeros(3, image_size, image_size)
        
        if label == 0:  # Red dominant
            img[0] = torch.rand(image_size, image_size) * 0.5 + 0.5
            img[1] = torch.rand(image_size, image_size) * 0.3
            img[2] = torch.rand(image_size, image_size) * 0.3
        elif label == 1:  # Green dominant
            img[0] = torch.rand(image_size, image_size) * 0.3
            img[1] = torch.rand(image_size, image_size) * 0.5 + 0.5
            img[2] = torch.rand(image_size, image_size) * 0.3
        elif label == 2:  # Blue dominant
            img[0] = torch.rand(image_size, image_size) * 0.3
            img[1] = torch.rand(image_size, image_size) * 0.3
            img[2] = torch.rand(image_size, image_size) * 0.5 + 0.5
        else:  # Mixed
            base = torch.rand(image_size, image_size) * 0.5 + 0.25
            img[0] = base + torch.randn(image_size, image_size) * 0.1
            img[1] = base + torch.randn(image_size, image_size) * 0.1
            img[2] = base + torch.randn(image_size, image_size) * 0.1
        
        images.append(img.clamp(0, 1))
        labels.append(label)
    
    return torch.stack(images), torch.tensor(labels)


def create_pattern_classification_dataset(
    num_samples: int = 500,
    image_size: int = 32,
    num_classes: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a synthetic dataset for pattern-based classification."""
    images = []
    labels = []
    
    for i in range(num_samples):
        label = i % num_classes
        img = torch.zeros(3, image_size, image_size)
        
        color1 = torch.rand(3)
        color2 = torch.rand(3)
        
        if label == 0:  # Horizontal stripes
            stripe_width = np.random.randint(2, 8)
            for y in range(image_size):
                color = color1 if (y // stripe_width) % 2 == 0 else color2
                img[:, y, :] = color.view(3, 1)
                
        elif label == 1:  # Vertical stripes
            stripe_width = np.random.randint(2, 8)
            for x in range(image_size):
                color = color1 if (x // stripe_width) % 2 == 0 else color2
                img[:, :, x] = color.view(3, 1)
                
        elif label == 2:  # Checkerboard
            block_size = np.random.randint(4, 12)
            for y in range(0, image_size, block_size):
                for x in range(0, image_size, block_size):
                    color = color1 if ((y // block_size) + (x // block_size)) % 2 == 0 else color2
                    y_end = min(y + block_size, image_size)
                    x_end = min(x + block_size, image_size)
                    img[:, y:y_end, x:x_end] = color.view(3, 1, 1)
                    
        else:  # Diagonal
            for y in range(image_size):
                for x in range(image_size):
                    color = color1 if (x + y) % 8 < 4 else color2
                    img[:, y, x] = color
        
        img += torch.randn_like(img) * 0.05
        images.append(img.clamp(0, 1))
        labels.append(label)
    
    return torch.stack(images), torch.tensor(labels)


def create_complexity_classification_dataset(
    num_samples: int = 500,
    image_size: int = 32,
    num_classes: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a dataset where class correlates with image complexity."""
    images = []
    labels = []
    
    for i in range(num_samples):
        label = i % num_classes
        img = torch.zeros(3, image_size, image_size)
        
        if label == 0:  # Simple
            color = torch.rand(3)
            img = color.view(3, 1, 1).expand(3, image_size, image_size).clone()
            img += torch.randn_like(img) * 0.02
            
        elif label == 1:  # Medium
            num_blocks = np.random.randint(4, 9)
            for _ in range(num_blocks):
                x = np.random.randint(0, image_size - 8)
                y = np.random.randint(0, image_size - 8)
                w = np.random.randint(4, 12)
                h = np.random.randint(4, 12)
                color = torch.rand(3)
                x_end = min(x + w, image_size)
                y_end = min(y + h, image_size)
                img[:, y:y_end, x:x_end] = color.view(3, 1, 1)
                
        else:  # Complex
            img = torch.rand(3, image_size, image_size)
            for y in range(image_size):
                for x in range(image_size):
                    if (x + y) % 2 == 0:
                        img[:, y, x] *= 0.7
            img += torch.randn_like(img) * 0.15
        
        images.append(img.clamp(0, 1))
        labels.append(label)
    
    return torch.stack(images), torch.tensor(labels)


# ============================================================================
# Training Loop
# ============================================================================

@dataclass
class TrainingHistory:
    """Records training history for analysis."""
    
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)
    gradient_norms: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'gradient_norms': self.gradient_norms,
            'epoch_times': self.epoch_times,
        }


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute total gradient norm."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    return np.sqrt(total_norm)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> TrainingHistory:
    """Train a model and record history."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    history = TrainingHistory()
    
    for epoch in range(num_epochs):
        with Timer(sync_cuda=torch.cuda.is_available()) as timer:
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            epoch_grad_norms = []
            
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                epoch_grad_norms.append(compute_gradient_norm(model))
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                train_correct += (outputs.argmax(dim=1) == labels).sum().item()
                train_total += images.size(0)
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * images.size(0)
                    val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                    val_total += images.size(0)
            
            val_loss /= val_total
            val_acc = val_correct / val_total
        
        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)
        history.train_accuracies.append(train_acc)
        history.val_accuracies.append(val_acc)
        history.gradient_norms.append(float(np.mean(epoch_grad_norms)))
        history.epoch_times.append(timer.elapsed_ms)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")
    
    return history


# ============================================================================
# Convergence Analysis
# ============================================================================

def analyze_convergence(
    history: TrainingHistory,
    target_accuracy: float = 0.9,
    convergence_threshold: float = 0.01,
) -> ConvergenceMetrics:
    """Analyze training history to compute convergence metrics."""
    metrics = ConvergenceMetrics()
    
    if history.train_losses:
        metrics.initial_loss = history.train_losses[0]
        metrics.final_train_loss = history.train_losses[-1]
        
    if history.val_losses:
        metrics.final_val_loss = history.val_losses[-1]
        metrics.best_val_loss = min(history.val_losses)
        
    if history.train_accuracies:
        metrics.final_train_accuracy = history.train_accuracies[-1]
        
    if history.val_accuracies:
        metrics.final_val_accuracy = history.val_accuracies[-1]
        metrics.best_val_accuracy = max(history.val_accuracies)
    
    for i, acc in enumerate(history.val_accuracies):
        if acc >= target_accuracy:
            metrics.epochs_to_convergence = i + 1
            break
    else:
        metrics.epochs_to_convergence = len(history.val_accuracies)
    
    if len(history.train_losses) > 5:
        recent_losses = history.train_losses[-5:]
        metrics.loss_variance = float(np.var(recent_losses))
    
    if history.gradient_norms:
        grad_stats = compute_statistics(history.gradient_norms)
        metrics.gradient_norm_mean = grad_stats['mean']
        metrics.gradient_norm_std = grad_stats['std']
    
    if len(history.train_losses) > 1:
        loss_diff = metrics.initial_loss - metrics.final_train_loss
        metrics.loss_improvement_rate = loss_diff / len(history.train_losses)
    
    return metrics


def detect_overfitting(
    history: TrainingHistory,
    gap_threshold: float = 0.1,
) -> Dict[str, Any]:
    """Detect overfitting from training history."""
    result: Dict[str, Any] = {
        'is_overfitting': False,
        'overfitting_epoch': None,
        'max_gap': 0.0,
        'final_gap': 0.0,
    }
    
    if not history.train_losses or not history.val_losses:
        return result
    
    gaps = [val - train for train, val in 
            zip(history.train_losses, history.val_losses)]
    
    result['max_gap'] = max(gaps) if gaps else 0.0
    result['final_gap'] = gaps[-1] if gaps else 0.0
    
    for i, gap in enumerate(gaps):
        if gap > gap_threshold:
            result['is_overfitting'] = True
            result['overfitting_epoch'] = i + 1
            break
    
    return result


# ============================================================================
# Comparison Functions
# ============================================================================

def compare_model_convergence(
    models: Dict[str, nn.Module],
    dataset_generator: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
    num_epochs: int = 50,
    batch_size: int = 32,
    num_runs: int = 3,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compare convergence of multiple models on the same dataset."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results: Dict[str, Dict[str, Any]] = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating: {model_name}")
        run_metrics = []
        
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}")
            
            images, labels = dataset_generator()
            n_train = int(len(images) * 0.8)
            train_dataset = TensorDataset(images[:n_train], labels[:n_train])
            val_dataset = TensorDataset(images[n_train:], labels[n_train:])
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Create fresh model
            model_copy = create_model_copy(model_name, model)
            
            history = train_model(
                model_copy, train_loader, val_loader,
                num_epochs=num_epochs, device=device, verbose=False
            )
            
            metrics = analyze_convergence(history)
            overfitting = detect_overfitting(history)
            
            run_metrics.append({
                'convergence': metrics.to_dict(),
                'overfitting': overfitting,
                'history': history.to_dict(),
            })
        
        results[model_name] = {
            'runs': run_metrics,
            'summary': aggregate_metrics(run_metrics),
        }
    
    return results


def create_model_copy(model_name: str, template: nn.Module) -> nn.Module:
    """Create a fresh copy of a model.
    
    Note: Since the model classes don't store all constructor parameters as attributes,
    we use the available attributes and sensible defaults for the rest.
    """
    # Default values that match the model class defaults
    DEFAULT_DEPTH = 6
    DEFAULT_HEADS = 8
    DEFAULT_MLP_DIM = 1024
    DEFAULT_CHANNELS = 3
    
    # All models now use FractalCurveViT
    num_classes = template.num_classes
    return FractalCurveViT(
        image_size=template.image_size,
        num_classes=num_classes,
        dim=template.dim,
        num_layers=DEFAULT_DEPTH,
        heads=DEFAULT_HEADS,
        mlp_dim=DEFAULT_MLP_DIM,
        channels=DEFAULT_CHANNELS,
        max_level=template.max_level,
    )


def aggregate_metrics(run_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across multiple runs."""
    if not run_metrics:
        return {}
    
    summary: Dict[str, Any] = {}
    
    convergence_keys = [
        'final_train_loss', 'final_val_loss', 'best_val_loss',
        'epochs_to_convergence', 'final_train_accuracy', 'final_val_accuracy',
        'best_val_accuracy', 'loss_improvement_rate',
    ]
    
    for key in convergence_keys:
        values = [r['convergence'][key] for r in run_metrics if key in r['convergence']]
        if values:
            stats = compute_statistics(values)
            summary[key] = {
                'mean': stats['mean'],
                'std': stats['std'],
                'min': stats['min'],
                'max': stats['max'],
            }
    
    overfitting_count = sum(1 for r in run_metrics if r['overfitting']['is_overfitting'])
    summary['overfitting_rate'] = overfitting_count / len(run_metrics)
    
    return summary


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_convergence_benchmark(
    image_size: int = 32,
    num_classes: int = 4,
    num_epochs: int = 30,
    num_runs: int = 3,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run comprehensive convergence benchmark."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("Convergence Benchmark")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Image size: {image_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Runs: {num_runs}")
    
    # Note: SimpleFractalViT has been removed, using only FractalCurveViT
    models = {
        'FractalCurveViT': FractalCurveViT(
            image_size=image_size,
            num_classes=num_classes,
            dim=64,
            num_layers=4,
            heads=4,
            mlp_dim=256,
            channels=3,
            max_level=2,
        ),
    }
    
    results: Dict[str, Any] = {
        'config': {
            'image_size': image_size,
            'num_classes': num_classes,
            'num_epochs': num_epochs,
            'num_runs': num_runs,
            'device': str(device),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'datasets': {},
    }
    
    datasets = {
        'color_classification': lambda: create_color_classification_dataset(
            num_samples=400, image_size=image_size, num_classes=num_classes
        ),
        'pattern_classification': lambda: create_pattern_classification_dataset(
            num_samples=400, image_size=image_size, num_classes=num_classes
        ),
    }
    
    for dataset_name, generator in datasets.items():
        print(f"\n--- Dataset: {dataset_name} ---")
        
        comparison = compare_model_convergence(
            models, generator,
            num_epochs=num_epochs,
            num_runs=num_runs,
            device=device,
        )
        
        results['datasets'][dataset_name] = comparison
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"convergence_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for dataset_name, comparison in results['datasets'].items():
        print(f"\n{dataset_name}:")
        for model_name, model_results in comparison.items():
            summary = model_results['summary']
            print(f"  {model_name}:")
            if 'best_val_accuracy' in summary:
                print(f"    Best val accuracy: {summary['best_val_accuracy']['mean']:.3f} "
                      f"(+/- {summary['best_val_accuracy']['std']:.3f})")
            if 'epochs_to_convergence' in summary:
                print(f"    Epochs to converge: {summary['epochs_to_convergence']['mean']:.1f}")
            if 'overfitting_rate' in summary:
                print(f"    Overfitting rate: {summary['overfitting_rate']:.0%}")
    
    return results


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run convergence benchmarks')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                        help='Directory to save results')
    parser.add_argument('--image-size', type=int, default=32,
                        help='Image size')
    parser.add_argument('--num-epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--num-runs', type=int, default=3,
                        help='Number of runs per configuration')
    
    args = parser.parse_args()
    
    run_convergence_benchmark(
        image_size=args.image_size,
        num_epochs=args.num_epochs,
        num_runs=args.num_runs,
        output_dir=Path(args.output_dir),
    )
