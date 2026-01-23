"""
Fractal ViT Benchmarks Package - Non-Pytest Scripts

警告: 此目录包含独立运行脚本，不属于 pytest 测试套件。
这些脚本用于性能基准测试和 ablation 研究，需独立运行。

运行方式:
    python -m tests.benchmarks.benchmark_fractal_vit
    python -m tests.benchmarks.check_convergence
    python -m tests.benchmarks.compare_fractal_vs_standard

NOT part of pytest - DO NOT run with: pytest tests/benchmarks/

Modules:
- benchmark_metrics: Core metrics definitions and utilities
- benchmark_fractal_vit: Tokenizer and model benchmarks
- check_convergence: Training convergence analysis
- compare_fractal_vs_standard: Comparison with standard ViT
- evaluate_pretrained: Evaluate pretrained .pth models
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
