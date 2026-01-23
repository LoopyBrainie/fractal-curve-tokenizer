# -*- coding: utf-8 -*-
"""
Performance Benchmarks

性能基准测试套件 (与 pytest 分离).

这些脚本用于性能分析和优化验证，不应在 CI 中作为 pytest 运行.

Usage:
    python benchmarks/runner.py --suite fractal_vit --quick
    python benchmarks/runner.py --output results/
"""

from .config import BenchmarkConfig
from .metrics import BenchmarkMetrics, PerformanceReport

__all__ = ["BenchmarkConfig", "BenchmarkMetrics", "PerformanceReport"]
