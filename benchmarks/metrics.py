# -*- coding: utf-8 -*-
"""
Benchmark Metrics

性能指标定义和计算.

包含:
- BenchmarkMetrics: 性能指标收集器
- PerformanceReport: 性能报告生成器
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import statistics


@dataclass
class BenchmarkMetrics:
    """性能指标收集器.

    用于收集和计算性能指标.
    """
    # 基础指标
    iterations: List[float] = field(default_factory=list)  # 每次迭代的时间
    memory_usage: List[float] = field(default_factory=list)  # 内存使用 (MB)
    gpu_memory: List[float] = field(default_factory=list)  # GPU 显存 (MB)

    # 派生指标
    @property
    def mean_time(self) -> float:
        """平均时间 (秒)."""
        if not self.iterations:
            return 0.0
        return statistics.mean(self.iterations)

    @property
    def std_time(self) -> float:
        """时间标准差."""
        if len(self.iterations) < 2:
            return 0.0
        return statistics.stdev(self.iterations)

    @property
    def min_time(self) -> float:
        """最小时间."""
        return min(self.iterations) if self.iterations else 0.0

    @property
    def max_time(self) -> float:
        """最大时间."""
        return max(self.iterations) if self.iterations else 0.0

    @property
    def throughput(self) -> float:
        """吞吐量 (samples/sec)."""
        if self.mean_time == 0:
            return 0.0
        return 1.0 / self.mean_time

    @property
    def mean_memory(self) -> float:
        """平均内存使用."""
        return statistics.mean(self.memory_usage) if self.memory_usage else 0.0

    def record_iteration(self, duration: float, memory: Optional[float] = None,
                         gpu_memory: Optional[float] = None) -> None:
        """记录一次迭代."""
        self.iterations.append(duration)
        if memory is not None:
            self.memory_usage.append(memory)
        if gpu_memory is not None:
            self.gpu_memory.append(gpu_memory)

    def reset(self) -> None:
        """重置所有指标."""
        self.iterations.clear()
        self.memory_usage.clear()
        self.gpu_memory.clear()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典."""
        return {
            "mean_time": self.mean_time,
            "std_time": self.std_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "throughput": self.throughput,
            "mean_memory": self.mean_memory,
            "iterations_count": len(self.iterations),
        }


@dataclass
class PerformanceReport:
    """性能报告生成器.

    生成格式化的性能报告.
    """
    metrics: Dict[str, BenchmarkMetrics] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_benchmark(self, name: str, metrics: BenchmarkMetrics) -> None:
        """添加基准测试结果."""
        self.metrics[name] = metrics

    def set_metadata(self, key: str, value: Any) -> None:
        """设置元数据."""
        self.metadata[key] = value

    def generate_markdown(self) -> str:
        """生成 Markdown 格式报告."""
        lines = ["# Performance Benchmark Report\n"]

        # 元数据
        if self.metadata:
            lines.append("## Environment\n")
            for key, value in self.metadata.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        # 各基准结果
        lines.append("## Results\n")
        for name, metrics in self.metrics.items():
            lines.append(f"### {name}")
            lines.append(f"- Mean Time: {metrics.mean_time*1000:.2f} ms")
            lines.append(f"- Std Time:  {metrics.std_time*1000:.2f} ms")
            lines.append(f"- Min/Max:   {metrics.min_time*1000:.2f} / {metrics.max_time*1000:.2f} ms")
            lines.append(f"- Throughput: {metrics.throughput:.2f} samples/sec")
            if metrics.mean_memory > 0:
                lines.append(f"- Memory: {metrics.mean_memory:.2f} MB")
            lines.append("")

        return "\n".join(lines)

    def generate_summary(self) -> str:
        """生成摘要字符串."""
        parts = []
        for name, metrics in self.metrics.items():
            parts.append(f"{name}: {metrics.mean_time*1000:.2f}ms +/- {metrics.std_time*1000:.2f}ms")
        return " | ".join(parts)
