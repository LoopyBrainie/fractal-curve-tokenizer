"""
Fractal ViT Tokenizer Benchmarks.

This module provides comprehensive benchmarks for evaluating the
Fractal ViT tokenizer's adaptive tokenization capabilities.

Benchmarks:
1. Tokenization efficiency on different image types
2. Adaptive token distribution analysis
3. Comparison with fixed-grid tokenization
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Handle import paths
try:
    from vit_pytorch import SimpleFractalViT, NextGenerationFractalViT
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
    from vit_pytorch import SimpleFractalViT, NextGenerationFractalViT

from .benchmark_metrics import (
    TokenizationMetrics,
    ComputationalMetrics,
    Timer,
    compute_batch_complexity,
    compute_correlation,
    compute_statistics,
    count_parameters,
    get_model_memory_mb,
    get_peak_memory_mb,
    reset_memory_stats,
)


# ============================================================================
# Test Image Generators
# ============================================================================

def generate_simple_images(batch_size: int, image_size: int = 64) -> torch.Tensor:
    """Generate simple images with uniform regions."""
    images = torch.zeros(batch_size, 3, image_size, image_size)
    for i in range(batch_size):
        color = torch.rand(3, 1, 1)
        images[i] = color.expand(3, image_size, image_size)
        images[i] += torch.randn_like(images[i]) * 0.01
    return images.clamp(0, 1)


def generate_complex_images(batch_size: int, image_size: int = 64) -> torch.Tensor:
    """Generate complex images with detailed textures."""
    images = torch.zeros(batch_size, 3, image_size, image_size)
    for i in range(batch_size):
        for y in range(0, image_size, 4):
            for x in range(0, image_size, 4):
                color = torch.rand(3)
                images[i, :, y:y+4, x:x+4] = color.view(3, 1, 1)
        images[i] += torch.randn_like(images[i]) * 0.2
        images[i, :, ::2, :] *= 0.8
        images[i, :, :, ::3] *= 0.9
    return images.clamp(0, 1)


def generate_mixed_complexity_images(
    batch_size: int, 
    image_size: int = 64
) -> Tuple[torch.Tensor, List[str]]:
    """Generate images with varying complexity."""
    images = []
    labels = []
    
    for i in range(batch_size):
        complexity = i % 4
        if complexity == 0:
            img = torch.ones(3, image_size, image_size) * torch.rand(3, 1, 1)
            labels.append('very_simple')
        elif complexity == 1:
            img = torch.zeros(3, image_size, image_size)
            for c in range(3):
                for y in range(image_size):
                    img[c, y, :] = y / image_size
            labels.append('simple')
        elif complexity == 2:
            img = torch.zeros(3, image_size, image_size)
            block_size = 16
            for y in range(0, image_size, block_size):
                for x in range(0, image_size, block_size):
                    img[:, y:y+block_size, x:x+block_size] = torch.rand(3, 1, 1)
            labels.append('medium')
        else:
            img = torch.zeros(3, image_size, image_size)
            for y in range(image_size):
                for x in range(image_size):
                    if (x + y) % 2 == 0:
                        img[:, y, x] = torch.rand(3)
                    else:
                        img[:, y, x] = 1 - torch.rand(3) * 0.5
            img += torch.randn_like(img) * 0.1
            labels.append('complex')
        
        images.append(img.clamp(0, 1))
    
    return torch.stack(images), labels


# ============================================================================
# Model Benchmarks
# ============================================================================

def benchmark_fractal_vit(
    model: nn.Module,
    images: torch.Tensor,
    num_runs: int = 10,
    warmup_runs: int = 3,
) -> Tuple[ComputationalMetrics, Dict[str, Any]]:
    """Benchmark a Fractal ViT model's performance."""
    device = images.device
    batch_size = images.shape[0]
    model = model.to(device)
    model.eval()
    
    total_params, trainable_params = count_parameters(model)
    model_memory = get_model_memory_mb(model)
    
    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(images)
    
    # Forward pass timing
    forward_times = []
    reset_memory_stats()
    
    for _ in range(num_runs):
        with Timer() as timer:
            with torch.no_grad():
                _ = model(images)
        forward_times.append(timer.elapsed_ms)
    
    peak_memory = get_peak_memory_mb()
    
    # Backward pass timing
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
    
    metrics = ComputationalMetrics(
        forward_pass_time_ms=forward_stats['mean'],
        backward_pass_time_ms=backward_stats['mean'],
        total_time_ms=forward_stats['mean'] + backward_stats['mean'],
        images_per_second=1000 * batch_size / forward_stats['mean'] if forward_stats['mean'] > 0 else 0,
        peak_memory_mb=peak_memory,
        model_memory_mb=model_memory,
        total_parameters=total_params,
        trainable_parameters=trainable_params,
    )
    
    additional_info = {
        'model_type': model.__class__.__name__,
        'batch_size': batch_size,
        'device': str(device),
        'forward_time_std': forward_stats['std'],
        'backward_time_std': backward_stats['std'],
    }
    
    return metrics, additional_info


def benchmark_simple_fractal_vit(
    image_size: int = 64,
    batch_size: int = 4,
    num_runs: int = 10,
    warmup_runs: int = 3,
    device: Optional[torch.device] = None
) -> Tuple[ComputationalMetrics, Dict[str, Any]]:
    """Benchmark SimpleFractalViT performance."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SimpleFractalViT(
        image_size=image_size,
        num_classes=10,
        dim=64,
        depth=4,
        heads=4,
        mlp_dim=256,
        channels=3,
        max_level=3,
    ).to(device)
    
    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    return benchmark_fractal_vit(model, images, num_runs, warmup_runs)


def benchmark_next_gen_fractal_vit(
    image_size: int = 64,
    batch_size: int = 4,
    num_runs: int = 10,
    warmup_runs: int = 3,
    device: Optional[torch.device] = None
) -> Tuple[ComputationalMetrics, Dict[str, Any]]:
    """Benchmark NextGenerationFractalViT performance."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = NextGenerationFractalViT(
        image_size=image_size,
        num_classes=10,
        dim=64,
        depth=4,
        heads=4,
        mlp_dim=256,
        channels=3,
        max_level=3,
    ).to(device)
    
    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    return benchmark_fractal_vit(model, images, num_runs, warmup_runs)


# ============================================================================
# Tokenization Analysis
# ============================================================================

def analyze_tokenization(
    model: nn.Module,
    simple_images: torch.Tensor,
    complex_images: torch.Tensor,
) -> TokenizationMetrics:
    """Analyze tokenization behavior on simple vs complex images."""
    model.eval()
    device = next(model.parameters()).device
    
    simple_images = simple_images.to(device)
    complex_images = complex_images.to(device)
    
    # Count tokens from tokenizer if available
    simple_token_counts = []
    complex_token_counts = []
    
    with torch.no_grad():
        if hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
            
            # Process simple images
            for img in simple_images:
                output = tokenizer.tokenize(img.unsqueeze(0))
                if hasattr(output, 'tokens') and len(output.tokens) > 0:
                    simple_token_counts.append(output.tokens[0].shape[0])
                else:
                    simple_token_counts.append(1)
            
            # Process complex images
            for img in complex_images:
                output = tokenizer.tokenize(img.unsqueeze(0))
                if hasattr(output, 'tokens') and len(output.tokens) > 0:
                    complex_token_counts.append(output.tokens[0].shape[0])
                else:
                    complex_token_counts.append(1)
        else:
            # Default fixed tokenization
            simple_token_counts = [64] * len(simple_images)
            complex_token_counts = [64] * len(complex_images)
    
    all_counts = simple_token_counts + complex_token_counts
    all_complexities = (compute_batch_complexity(simple_images) + 
                       compute_batch_complexity(complex_images))
    
    stats = compute_statistics([float(c) for c in all_counts])
    
    return TokenizationMetrics(
        total_tokens=sum(all_counts),
        tokens_per_image=stats['mean'],
        min_tokens=int(stats['min']),
        max_tokens=int(stats['max']),
        std_tokens=stats['std'],
        adaptivity_ratio=stats['max'] / max(stats['min'], 1),
        complexity_correlation=compute_correlation(all_complexities, [float(c) for c in all_counts]),
    )


# ============================================================================
# Benchmark Suite
# ============================================================================

def run_tokenization_benchmark_suite(
    image_sizes: Optional[List[int]] = None,
    batch_sizes: Optional[List[int]] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Run comprehensive tokenization benchmarks."""
    if image_sizes is None:
        image_sizes = [32, 64]
    if batch_sizes is None:
        batch_sizes = [1, 4]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results: Dict[str, Any] = {
        'device': str(device),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'benchmarks': []
    }
    
    for image_size in image_sizes:
        for batch_size in batch_sizes:
            print(f"Benchmarking: image_size={image_size}, batch_size={batch_size}")
            
            model = SimpleFractalViT(
                image_size=image_size,
                num_classes=10,
                dim=64,
                depth=4,
                heads=4,
                mlp_dim=256,
                channels=3,
                max_level=3,
            ).to(device)
            
            for img_type, generator in [
                ('simple', generate_simple_images),
                ('complex', generate_complex_images),
            ]:
                images = generator(batch_size, image_size).to(device)
                metrics, info = benchmark_fractal_vit(model, images)
                
                results['benchmarks'].append({
                    'image_size': image_size,
                    'batch_size': batch_size,
                    'image_type': img_type,
                    'computation': metrics.to_dict(),
                    'info': info,
                })
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"tokenization_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")
    
    return results


def run_model_benchmark_suite(
    image_size: int = 64,
    batch_sizes: Optional[List[int]] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Run comprehensive model benchmarks."""
    if batch_sizes is None:
        batch_sizes = [1, 4, 8]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results: Dict[str, Any] = {
        'device': str(device),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'image_size': image_size,
        'models': {}
    }
    
    for model_type in ['SimpleFractalViT', 'NextGenerationFractalViT']:
        results['models'][model_type] = []
        
        for batch_size in batch_sizes:
            print(f"Benchmarking: {model_type}, batch_size={batch_size}")
            
            if model_type == 'SimpleFractalViT':
                metrics, info = benchmark_simple_fractal_vit(
                    image_size=image_size,
                    batch_size=batch_size,
                    device=device
                )
            else:
                metrics, info = benchmark_next_gen_fractal_vit(
                    image_size=image_size,
                    batch_size=batch_size,
                    device=device
                )
            
            results['models'][model_type].append({
                'batch_size': batch_size,
                'metrics': metrics.to_dict(),
                'info': info,
            })
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"model_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")
    
    return results


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Fractal ViT benchmarks')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                        help='Directory to save results')
    parser.add_argument('--image-sizes', type=int, nargs='+', default=[32, 64],
                        help='Image sizes to test')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4],
                        help='Batch sizes to test')
    parser.add_argument('--tokenizer-only', action='store_true',
                        help='Only run tokenizer benchmarks')
    parser.add_argument('--model-only', action='store_true',
                        help='Only run model benchmarks')
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("Fractal ViT Benchmark Suite")
    print("=" * 60)
    
    if not args.model_only:
        print("\n[1] Running Tokenization Benchmarks...")
        tok_results = run_tokenization_benchmark_suite(
            image_sizes=args.image_sizes,
            batch_sizes=args.batch_sizes,
            output_dir=output_dir
        )
        
        print("\nTokenization Benchmark Summary:")
        for bench in tok_results['benchmarks']:
            comp = bench['computation']
            print(f"  {bench['image_type']:8s} | size={bench['image_size']:3d} | "
                  f"batch={bench['batch_size']:2d} | "
                  f"time={comp['forward_pass_time_ms']:.2f}ms")
    
    if not args.tokenizer_only:
        print("\n[2] Running Model Benchmarks...")
        model_results = run_model_benchmark_suite(
            image_size=args.image_sizes[0] if args.image_sizes else 64,
            batch_sizes=args.batch_sizes,
            output_dir=output_dir
        )
        
        print("\nModel Benchmark Summary:")
        for model_type, benchmarks in model_results['models'].items():
            print(f"\n  {model_type}:")
            for bench in benchmarks:
                m = bench['metrics']
                print(f"    batch={bench['batch_size']:2d} | "
                      f"forward={m['forward_pass_time_ms']:.2f}ms | "
                      f"backward={m['backward_pass_time_ms']:.2f}ms | "
                      f"params={m['total_parameters']:,}")
    
    print("\n" + "=" * 60)
    print("Benchmarks complete!")
    print("=" * 60)
