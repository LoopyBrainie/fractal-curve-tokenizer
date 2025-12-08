"""
Fractal ViT Benchmarks Package.

This package provides comprehensive benchmarking tools for evaluating
Fractal ViT tokenization and model performance.

Modules:
- benchmark_metrics: Core metrics definitions and utilities
- benchmark_fractal_vit: Tokenizer and model benchmarks
- check_convergence: Training convergence analysis
- compare_fractal_vs_standard: Comparison with standard ViT
- evaluate_pretrained: Evaluate pretrained .pth models

Usage:
    # Run tokenization benchmarks
    python -m tests.benchmarks.benchmark_fractal_vit
    
    # Run convergence analysis
    python -m tests.benchmarks.check_convergence
    
    # Run full comparison
    python -m tests.benchmarks.compare_fractal_vs_standard
    
    # Evaluate pretrained model
    python -m tests.benchmarks.evaluate_pretrained --checkpoint path/to/best.pth --visualize
"""

from .benchmark_metrics import (
    TokenizationMetrics,
    ComputationalMetrics,
    ConvergenceMetrics,
    ComparisonResult,
    Timer,
    compute_image_complexity,
    compute_batch_complexity,
    compute_correlation,
    compute_statistics,
    count_parameters,
    get_model_memory_mb,
    get_peak_memory_mb,
    reset_memory_stats,
)

__all__ = [
    # Metrics
    'TokenizationMetrics',
    'ComputationalMetrics',
    'ConvergenceMetrics',
    'ComparisonResult',
    'Timer',
    
    # Utilities
    'compute_image_complexity',
    'compute_batch_complexity',
    'compute_correlation',
    'compute_statistics',
    'count_parameters',
    'get_model_memory_mb',
    'get_peak_memory_mb',
    'reset_memory_stats',
]
