"""
Comprehensive Benchmark Metrics for Fractal ViT Tokenizer.

This module defines evaluation metrics and utilities for benchmarking
the Fractal ViT tokenizer against baseline approaches.

Metrics Categories:
1. Tokenization Efficiency - tokens per image, adaptivity ratio
2. Hilbert Locality Preservation - spatial coherence metrics
3. Computational Performance - speed, memory, throughput
4. Token Distribution Analysis - multi-scale statistics
"""

from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Generator

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TokenizationMetrics:
    """Metrics for evaluating tokenization quality."""
    
    total_tokens: int = 0
    tokens_per_image: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    std_tokens: float = 0.0
    adaptivity_ratio: float = 0.0
    complexity_correlation: float = 0.0
    level_distribution: Dict[int, float] = field(default_factory=dict)
    locality_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_tokens': self.total_tokens,
            'tokens_per_image': self.tokens_per_image,
            'min_tokens': self.min_tokens,
            'max_tokens': self.max_tokens,
            'std_tokens': self.std_tokens,
            'adaptivity_ratio': self.adaptivity_ratio,
            'complexity_correlation': self.complexity_correlation,
            'level_distribution': self.level_distribution,
            'locality_score': self.locality_score,
        }


@dataclass
class ComputationalMetrics:
    """Metrics for evaluating computational performance."""
    
    tokenization_time_ms: float = 0.0
    forward_pass_time_ms: float = 0.0
    backward_pass_time_ms: float = 0.0
    total_time_ms: float = 0.0
    images_per_second: float = 0.0
    tokens_per_second: float = 0.0
    peak_memory_mb: float = 0.0
    model_memory_mb: float = 0.0
    activation_memory_mb: float = 0.0
    total_parameters: int = 0
    trainable_parameters: int = 0
    flops_estimate: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tokenization_time_ms': self.tokenization_time_ms,
            'forward_pass_time_ms': self.forward_pass_time_ms,
            'backward_pass_time_ms': self.backward_pass_time_ms,
            'total_time_ms': self.total_time_ms,
            'images_per_second': self.images_per_second,
            'tokens_per_second': self.tokens_per_second,
            'peak_memory_mb': self.peak_memory_mb,
            'model_memory_mb': self.model_memory_mb,
            'activation_memory_mb': self.activation_memory_mb,
            'total_parameters': self.total_parameters,
            'trainable_parameters': self.trainable_parameters,
            'flops_estimate': self.flops_estimate,
        }


@dataclass
class ConvergenceMetrics:
    """Metrics for evaluating training convergence."""
    
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    best_val_loss: float = float('inf')
    epochs_to_convergence: int = 0
    final_train_accuracy: float = 0.0
    final_val_accuracy: float = 0.0
    best_val_accuracy: float = 0.0
    loss_variance: float = 0.0
    gradient_norm_mean: float = 0.0
    gradient_norm_std: float = 0.0
    initial_loss: float = 0.0
    loss_improvement_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'final_train_loss': self.final_train_loss,
            'final_val_loss': self.final_val_loss,
            'best_val_loss': self.best_val_loss,
            'epochs_to_convergence': self.epochs_to_convergence,
            'final_train_accuracy': self.final_train_accuracy,
            'final_val_accuracy': self.final_val_accuracy,
            'best_val_accuracy': self.best_val_accuracy,
            'loss_variance': self.loss_variance,
            'gradient_norm_mean': self.gradient_norm_mean,
            'gradient_norm_std': self.gradient_norm_std,
            'initial_loss': self.initial_loss,
            'loss_improvement_rate': self.loss_improvement_rate,
        }


@dataclass
class ComparisonResult:
    """Result of comparing Fractal ViT vs baseline."""
    
    model_name: str = ""
    tokenization: TokenizationMetrics = field(default_factory=TokenizationMetrics)
    computation: ComputationalMetrics = field(default_factory=ComputationalMetrics)
    convergence: ConvergenceMetrics = field(default_factory=ConvergenceMetrics)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'tokenization': self.tokenization.to_dict(),
            'computation': self.computation.to_dict(),
            'convergence': self.convergence.to_dict(),
        }


class Timer:
    """High-precision timer for benchmarking."""
    
    def __init__(self, sync_cuda: bool = True):
        self.sync_cuda = sync_cuda and torch.cuda.is_available()
        self.elapsed_ms: float = 0.0
        self.start: float = 0.0
        
    def __enter__(self) -> "Timer":
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self
        
    def __exit__(self, *args: Any) -> None:
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000


@contextmanager
def measure_peak_memory() -> Generator[None, None, None]:
    """Context manager to measure peak GPU memory usage."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_peak_memory_mb() -> float:
    """Get peak GPU memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def reset_memory_stats() -> None:
    """Reset memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def compute_image_complexity(image: torch.Tensor) -> float:
    """Compute complexity score for an image using gradient magnitude."""
    if image.dim() == 4:
        image = image[0]
    if image.shape[0] == 3:
        gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
    else:
        gray = image[0]
    dx = gray[:, 1:] - gray[:, :-1]
    dy = gray[1:, :] - gray[:-1, :]
    grad_mag = torch.sqrt(dx[:, :-1] ** 2 + dy[:-1, :] ** 2 + 1e-8)
    return float(grad_mag.mean())


def compute_batch_complexity(images: torch.Tensor) -> List[float]:
    """Compute complexity for a batch of images."""
    return [compute_image_complexity(img) for img in images]


def compute_hilbert_locality_score(
    token_positions: List[Tuple[int, int]],
    token_indices: List[int]
) -> float:
    """Compute how well the Hilbert curve preserves spatial locality."""
    if len(token_positions) < 2:
        return 1.0
    sorted_pairs = sorted(zip(token_indices, token_positions), key=lambda x: x[0])
    positions = [p for _, p in sorted_pairs]
    total_distance = 0.0
    for i in range(1, len(positions)):
        dx = abs(positions[i][0] - positions[i-1][0])
        dy = abs(positions[i][1] - positions[i-1][1])
        distance = np.sqrt(dx**2 + dy**2)
        total_distance += distance
    avg_distance = total_distance / (len(positions) - 1)
    max_possible_distance = np.sqrt(2) * max(
        max(p[0] for p in positions),
        max(p[1] for p in positions)
    )
    if max_possible_distance == 0:
        return 1.0
    locality_score = 1.0 - (avg_distance - 1.0) / max_possible_distance
    return max(0.0, min(1.0, locality_score))


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device
) -> int:
    """Estimate FLOPs for a forward pass (rough approximation)."""
    total_params, _ = count_parameters(model)
    batch_size = input_shape[0]
    return 2 * total_params * batch_size


def get_model_memory_mb(model: nn.Module) -> float:
    """Estimate model memory usage in MB."""
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()
    return total_bytes / (1024 ** 2)


def compute_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    x_arr = np.array(x)
    y_arr = np.array(y)
    x_mean = np.mean(x_arr)
    y_mean = np.mean(y_arr)
    numerator = np.sum((x_arr - x_mean) * (y_arr - y_mean))
    denominator = np.sqrt(np.sum((x_arr - x_mean)**2) * np.sum((y_arr - y_mean)**2))
    if denominator < 1e-10:
        return 0.0
    return float(numerator / denominator)


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute basic statistics for a list of values."""
    if not values:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
    }
