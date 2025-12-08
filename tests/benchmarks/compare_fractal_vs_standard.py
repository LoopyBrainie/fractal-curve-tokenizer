"""
Compare Fractal ViT vs Standard ViT.

This module provides comprehensive comparison between Fractal ViT models
and a standard fixed-grid ViT baseline.

Comparison Metrics:
1. Tokenization efficiency (tokens per image, adaptivity)
2. Computational cost (speed, memory, parameters)
3. Training convergence (accuracy, loss)
4. Robustness to image complexity
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Handle import paths
try:
    from vit_pytorch import SimpleFractalViT, NextGenerationFractalViT
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
    from vit_pytorch import SimpleFractalViT, NextGenerationFractalViT

from .benchmark_metrics import (
    ComputationalMetrics,
    Timer,
    count_parameters,
    get_model_memory_mb,
    get_peak_memory_mb,
    reset_memory_stats,
    compute_statistics,
)

from .benchmark_fractal_vit import (
    generate_simple_images,
    generate_complex_images,
)

from .check_convergence import (
    TrainingHistory,
    train_model,
    analyze_convergence,
    create_color_classification_dataset,
)


# ============================================================================
# Standard ViT Baseline
# ============================================================================

class StandardViT(nn.Module):
    """Standard Vision Transformer with fixed-grid patch tokenization."""
    
    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        num_classes: int = 10,
        dim: int = 64,
        depth: int = 4,
        heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = dim
        self.depth = depth
        self.heads = heads
        
        self.patch_embed = nn.Conv2d(
            in_channels, dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, dim) * 0.02
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = x + self.pos_embed
        
        x = self.transformer(x)
        
        x = self.norm(x[:, 0])
        x = self.head(x)
        
        return x
    
    def count_tokens(self, x: torch.Tensor) -> int:
        return self.num_patches


# ============================================================================
# Comparison Utilities
# ============================================================================

def create_comparison_models(
    image_size: int = 64,
    num_classes: int = 10,
    dim: int = 64,
    depth: int = 4,
    heads: int = 4,
) -> Dict[str, nn.Module]:
    """Create all models for comparison with similar configurations."""
    patch_size = max(image_size // 8, 4)
    
    return {
        'StandardViT': StandardViT(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
        ),
        'SimpleFractalViT': SimpleFractalViT(
            image_size=image_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=dim * 4,
            channels=3,
            max_level=3,
        ),
        'NextGenerationFractalViT': NextGenerationFractalViT(
            image_size=image_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=dim * 4,
            channels=3,
            max_level=3,
        ),
    }


def benchmark_model_performance(
    model: nn.Module,
    images: torch.Tensor,
    num_runs: int = 10,
    warmup_runs: int = 3,
) -> ComputationalMetrics:
    """Benchmark a model's computational performance."""
    device = images.device
    batch_size = images.shape[0]
    model = model.to(device)
    model.eval()
    
    total_params, trainable_params = count_parameters(model)
    model_memory = get_model_memory_mb(model)
    
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(images)
    
    forward_times = []
    reset_memory_stats()
    
    for _ in range(num_runs):
        with Timer() as timer:
            with torch.no_grad():
                _ = model(images)
        forward_times.append(timer.elapsed_ms)
    
    peak_memory = get_peak_memory_mb()
    
    model.train()
    backward_times = []
    
    for _ in range(num_runs):
        model.zero_grad()
        outputs = model(images)
        loss = outputs.mean()
        
        with Timer() as timer:
            loss.backward()
        backward_times.append(timer.elapsed_ms)
    
    forward_stats = compute_statistics(forward_times)
    backward_stats = compute_statistics(backward_times)
    
    return ComputationalMetrics(
        forward_pass_time_ms=forward_stats['mean'],
        backward_pass_time_ms=backward_stats['mean'],
        total_time_ms=forward_stats['mean'] + backward_stats['mean'],
        images_per_second=1000 * batch_size / forward_stats['mean'] if forward_stats['mean'] > 0 else 0,
        peak_memory_mb=peak_memory,
        model_memory_mb=model_memory,
        total_parameters=total_params,
        trainable_parameters=trainable_params,
    )


def compare_tokenization(
    models: Dict[str, nn.Module],
    simple_images: torch.Tensor,
    complex_images: torch.Tensor,
) -> Dict[str, Dict[str, Any]]:
    """Compare tokenization behavior across models."""
    results: Dict[str, Dict[str, Any]] = {}
    
    for name, model in models.items():
        model.eval()
        
        with torch.no_grad():
            if hasattr(model, 'tokenizer'):
                tokenizer = model.tokenizer
                simple_tokens = []
                complex_tokens = []
                
                for img in simple_images:
                    output = tokenizer.tokenize(img.unsqueeze(0))
                    if hasattr(output, 'tokens') and len(output.tokens) > 0:
                        simple_tokens.append(output.tokens[0].shape[0])
                    else:
                        simple_tokens.append(1)
                
                for img in complex_images:
                    output = tokenizer.tokenize(img.unsqueeze(0))
                    if hasattr(output, 'tokens') and len(output.tokens) > 0:
                        complex_tokens.append(output.tokens[0].shape[0])
                    else:
                        complex_tokens.append(1)
                        
            elif hasattr(model, 'count_tokens'):
                token_count = model.count_tokens(simple_images)
                simple_tokens = [token_count] * len(simple_images)
                complex_tokens = [token_count] * len(complex_images)
            else:
                simple_tokens = [64] * len(simple_images)
                complex_tokens = [64] * len(complex_images)
        
        simple_stats = compute_statistics([float(c) for c in simple_tokens])
        complex_stats = compute_statistics([float(c) for c in complex_tokens])
        
        results[name] = {
            'simple_images': {
                'mean_tokens': simple_stats['mean'],
                'std_tokens': simple_stats['std'],
                'min_tokens': simple_stats['min'],
                'max_tokens': simple_stats['max'],
            },
            'complex_images': {
                'mean_tokens': complex_stats['mean'],
                'std_tokens': complex_stats['std'],
                'min_tokens': complex_stats['min'],
                'max_tokens': complex_stats['max'],
            },
            'adaptivity_ratio': (
                complex_stats['mean'] / simple_stats['mean'] 
                if simple_stats['mean'] > 0 else 1.0
            ),
            'is_adaptive': simple_stats['mean'] != complex_stats['mean'],
        }
    
    return results


def compare_training_convergence(
    models: Dict[str, nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 30,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compare training convergence across models."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results: Dict[str, Dict[str, Any]] = {}
    
    for name, model in models.items():
        print(f"  Training {name}...")
        
        model_copy = create_model_copy(name, model)
        
        history = train_model(
            model_copy, train_loader, val_loader,
            num_epochs=num_epochs, device=device, verbose=False
        )
        
        convergence = analyze_convergence(history)
        
        results[name] = {
            'convergence': convergence.to_dict(),
            'history': history.to_dict(),
        }
    
    return results


def create_model_copy(name: str, model: nn.Module) -> nn.Module:
    """Create a fresh copy of a model.
    
    Note: Since the model classes don't store all constructor parameters as attributes,
    we use the available attributes and sensible defaults for the rest.
    """
    # Default values that match the model class defaults
    DEFAULT_DEPTH = 6
    DEFAULT_HEADS = 8
    DEFAULT_MLP_DIM = 1024
    DEFAULT_CHANNELS = 3
    
    if name == 'StandardViT':
        return StandardViT(
            image_size=model.image_size,
            patch_size=model.patch_size,
            in_channels=DEFAULT_CHANNELS,
            num_classes=model.head.out_features,
            dim=model.dim,
            depth=model.depth,
            heads=model.heads,
        )
    elif name == 'SimpleFractalViT':
        enhanced = model.enhanced_model
        num_classes = enhanced.num_classes
        return SimpleFractalViT(
            image_size=enhanced.image_size,
            num_classes=num_classes,
            dim=enhanced.dim,
            depth=DEFAULT_DEPTH,
            heads=DEFAULT_HEADS,
            mlp_dim=DEFAULT_MLP_DIM,
            channels=DEFAULT_CHANNELS,
            max_level=enhanced.max_level,
        )
    else:  # NextGenerationFractalViT
        num_classes = model.num_classes
        return NextGenerationFractalViT(
            image_size=model.image_size,
            num_classes=num_classes,
            dim=model.dim,
            depth=DEFAULT_DEPTH,
            heads=DEFAULT_HEADS,
            mlp_dim=DEFAULT_MLP_DIM,
            channels=DEFAULT_CHANNELS,
            max_level=model.max_level,
        )


# ============================================================================
# Full Comparison Suite
# ============================================================================

@dataclass
class ComparisonSummary:
    """Summary of model comparison results."""
    
    fastest_forward: str = ""
    fastest_backward: str = ""
    most_memory_efficient: str = ""
    fewest_parameters: str = ""
    most_adaptive: str = ""
    best_accuracy: str = ""
    fastest_convergence: str = ""
    fractal_vs_standard_speedup: float = 1.0
    fractal_vs_standard_memory: float = 1.0
    fractal_adaptivity_ratio: float = 1.0


def run_full_comparison(
    image_size: int = 64,
    num_classes: int = 10,
    batch_size: int = 8,
    num_epochs: int = 30,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run comprehensive comparison of all model variants."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("Fractal ViT vs Standard ViT Comparison")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    
    print("\n[1] Creating models...")
    models = create_comparison_models(
        image_size=image_size,
        num_classes=num_classes,
    )
    
    for name, model in models.items():
        total, _ = count_parameters(model)
        print(f"  {name}: {total:,} parameters")
    
    results: Dict[str, Any] = {
        'config': {
            'image_size': image_size,
            'num_classes': num_classes,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'device': str(device),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'models': {
            name: {
                'parameters': count_parameters(model)[0],
                'trainable_parameters': count_parameters(model)[1],
            }
            for name, model in models.items()
        },
    }
    
    print("\n[2] Generating test images...")
    simple_images = generate_simple_images(batch_size, image_size).to(device)
    complex_images = generate_complex_images(batch_size, image_size).to(device)
    mixed_images = torch.cat([simple_images[:batch_size//2], complex_images[:batch_size//2]])
    
    print("\n[3] Comparing tokenization...")
    tok_comparison = compare_tokenization(models, simple_images, complex_images)
    results['tokenization'] = tok_comparison
    
    for name, tok in tok_comparison.items():
        print(f"  {name}:")
        print(f"    Simple: {tok['simple_images']['mean_tokens']:.1f} tokens")
        print(f"    Complex: {tok['complex_images']['mean_tokens']:.1f} tokens")
        print(f"    Adaptive: {'Yes' if tok['is_adaptive'] else 'No'}")
    
    print("\n[4] Benchmarking performance...")
    perf_results: Dict[str, Dict[str, Any]] = {}
    
    for name, model in models.items():
        print(f"  Benchmarking {name}...")
        metrics = benchmark_model_performance(model.to(device), mixed_images)
        perf_results[name] = metrics.to_dict()
    
    results['performance'] = perf_results
    
    print("\n  Performance Summary:")
    print(f"  {'Model':<25} {'Forward(ms)':<12} {'Backward(ms)':<14} {'Memory(MB)':<12}")
    print("  " + "-" * 63)
    for name, perf in perf_results.items():
        print(f"  {name:<25} {perf['forward_pass_time_ms']:<12.2f} "
              f"{perf['backward_pass_time_ms']:<14.2f} {perf['peak_memory_mb']:<12.1f}")
    
    print("\n[5] Comparing training convergence...")
    
    images, labels = create_color_classification_dataset(
        num_samples=400,
        image_size=image_size,
        num_classes=num_classes,
    )
    
    n_train = int(len(images) * 0.8)
    train_dataset = TensorDataset(images[:n_train], labels[:n_train])
    val_dataset = TensorDataset(images[n_train:], labels[n_train:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    conv_results = compare_training_convergence(
        models, train_loader, val_loader,
        num_epochs=num_epochs, device=device
    )
    results['convergence'] = {
        name: res['convergence']
        for name, res in conv_results.items()
    }
    
    print("\n  Convergence Summary:")
    print(f"  {'Model':<25} {'Best Val Acc':<12} {'Final Loss':<12} {'Epochs':<8}")
    print("  " + "-" * 57)
    for name, res in conv_results.items():
        conv = res['convergence']
        print(f"  {name:<25} {conv['best_val_accuracy']:<12.3f} "
              f"{conv['final_val_loss']:<12.4f} {conv['epochs_to_convergence']:<8d}")
    
    print("\n[6] Computing summary...")
    summary = compute_comparison_summary(results)
    results['summary'] = {
        'fastest_forward': summary.fastest_forward,
        'fastest_backward': summary.fastest_backward,
        'most_memory_efficient': summary.most_memory_efficient,
        'fewest_parameters': summary.fewest_parameters,
        'most_adaptive': summary.most_adaptive,
        'best_accuracy': summary.best_accuracy,
        'fastest_convergence': summary.fastest_convergence,
        'fractal_vs_standard_speedup': summary.fractal_vs_standard_speedup,
        'fractal_adaptivity_ratio': summary.fractal_adaptivity_ratio,
    }
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("Final Summary")
    print("=" * 70)
    print(f"  Fastest forward pass: {summary.fastest_forward}")
    print(f"  Most memory efficient: {summary.most_memory_efficient}")
    print(f"  Most adaptive tokenization: {summary.most_adaptive}")
    print(f"  Best validation accuracy: {summary.best_accuracy}")
    print(f"  Fractal adaptivity ratio: {summary.fractal_adaptivity_ratio:.2f}x")
    
    return results


def compute_comparison_summary(results: Dict[str, Any]) -> ComparisonSummary:
    """Compute summary statistics from comparison results."""
    summary = ComparisonSummary()
    
    perf = results.get('performance', {})
    tok = results.get('tokenization', {})
    conv = results.get('convergence', {})
    
    if perf:
        summary.fastest_forward = min(
            perf.keys(),
            key=lambda k: perf[k]['forward_pass_time_ms']
        )
        summary.fastest_backward = min(
            perf.keys(),
            key=lambda k: perf[k]['backward_pass_time_ms']
        )
        summary.most_memory_efficient = min(
            perf.keys(),
            key=lambda k: perf[k]['peak_memory_mb']
        )
        summary.fewest_parameters = min(
            perf.keys(),
            key=lambda k: perf[k]['total_parameters']
        )
    
    if tok:
        summary.most_adaptive = max(
            tok.keys(),
            key=lambda k: tok[k]['adaptivity_ratio']
        )
        
        fractal_models = [k for k in tok.keys() if 'Fractal' in k]
        if fractal_models:
            summary.fractal_adaptivity_ratio = max(
                tok[k]['adaptivity_ratio'] for k in fractal_models
            )
    
    if conv:
        summary.best_accuracy = max(
            conv.keys(),
            key=lambda k: conv[k]['best_val_accuracy']
        )
        summary.fastest_convergence = min(
            conv.keys(),
            key=lambda k: conv[k]['epochs_to_convergence']
        )
    
    if 'StandardViT' in perf and 'SimpleFractalViT' in perf:
        std_time = perf['StandardViT']['forward_pass_time_ms']
        fractal_time = perf['SimpleFractalViT']['forward_pass_time_ms']
        if fractal_time > 0:
            summary.fractal_vs_standard_speedup = std_time / fractal_time
    
    return summary


def plot_comparison_results(
    results: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> None:
    """Generate visualization plots for comparison results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = list(results.get('performance', {}).keys())
    if not models:
        return
    
    perf = results['performance']
    forward_times = [perf[m]['forward_pass_time_ms'] for m in models]
    backward_times = [perf[m]['backward_pass_time_ms'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax1 = axes[0, 0]
    ax1.bar(x - width/2, forward_times, width, label='Forward', color='steelblue')
    ax1.bar(x + width/2, backward_times, width, label='Backward', color='coral')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Inference Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('ViT', '\nViT') for m in models], fontsize=8)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    ax2 = axes[0, 1]
    params = [perf[m]['total_parameters'] / 1e6 for m in models]
    memory = [perf[m]['peak_memory_mb'] for m in models]
    
    ax2_twin = ax2.twinx()
    bars1 = ax2.bar(x - width/2, params, width, label='Parameters (M)', color='forestgreen')
    bars2 = ax2_twin.bar(x + width/2, memory, width, label='Peak Memory (MB)', color='purple')
    
    ax2.set_ylabel('Parameters (Millions)', color='forestgreen')
    ax2_twin.set_ylabel('Peak Memory (MB)', color='purple')
    ax2.set_title('Model Size Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace('ViT', '\nViT') for m in models], fontsize=8)
    ax2.legend([bars1, bars2], ['Parameters (M)', 'Peak Memory (MB)'], loc='upper right')
    
    ax3 = axes[1, 0]
    tok = results.get('tokenization', {})
    if tok:
        simple_tokens = [tok[m]['simple_images']['mean_tokens'] for m in models]
        complex_tokens = [tok[m]['complex_images']['mean_tokens'] for m in models]
        
        ax3.bar(x - width/2, simple_tokens, width, label='Simple Images', color='lightblue')
        ax3.bar(x + width/2, complex_tokens, width, label='Complex Images', color='darkblue')
        ax3.set_ylabel('Mean Tokens per Image')
        ax3.set_title('Tokenization Adaptivity')
        ax3.set_xticks(x)
        ax3.set_xticklabels([m.replace('ViT', '\nViT') for m in models], fontsize=8)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
    
    ax4 = axes[1, 1]
    conv_data = results.get('convergence', {})
    if conv_data:
        colors = ['blue', 'green', 'red']
        for i, model in enumerate(models):
            if model in conv_data:
                best_acc = conv_data[model].get('best_val_accuracy', 0)
                epochs = conv_data[model].get('epochs_to_convergence', 0)
                
                ax4.bar(i, best_acc, color=colors[i % len(colors)], alpha=0.7)
                ax4.annotate(f'{epochs} epochs', (i, best_acc + 0.02), ha='center', fontsize=8)
        
        ax4.set_ylabel('Best Validation Accuracy')
        ax4.set_title('Training Convergence')
        ax4.set_xticks(range(len(models)))
        ax4.set_xticklabels([m.replace('ViT', '\nViT') for m in models], fontsize=8)
        ax4.set_ylim(0, 1.1)
        ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'comparison_results.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_dir / 'comparison_results.png'}")
    
    plt.close()


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Fractal ViT vs Standard ViT')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                        help='Directory to save results')
    parser.add_argument('--image-size', type=int, default=64,
                        help='Image size')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='Number of classes')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization plots')
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    results = run_full_comparison(
        image_size=args.image_size,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        output_dir=output_dir,
    )
    
    if args.plot:
        plot_comparison_results(results, output_dir=output_dir)
