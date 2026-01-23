# -*- coding: utf-8 -*-
"""
Benchmark Configuration

性能基准测试配置.

包含:
- BenchmarkConfig: 基准配置类
- 预设配置模板
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BenchmarkConfig:
    """基准测试配置.

    Attributes:
        output_dir: 结果输出目录
        warmup_iterations: 预热迭代次数
        benchmark_iterations: 基准迭代次数
        device: 运行设备 ("cpu" 或 "cuda")
        precision: 计算精度 ("fp32" 或 "amp")
        log_level: 日志级别
    """
    output_dir: Path = Path("benchmark_results")
    warmup_iterations: int = 5
    benchmark_iterations: int = 20
    device: str = "cpu"
    precision: str = "fp32"
    log_level: str = "INFO"

    # 模型特定配置
    model_configs: dict = field(default_factory=lambda: {
        "small": {"dim": 64, "depth": 2, "heads": 4},
        "medium": {"dim": 256, "depth": 6, "heads": 8},
        "large": {"dim": 512, "depth": 12, "heads": 16},
    })

    @classmethod
    def from_file(cls, config_path: Path) -> "BenchmarkConfig":
        """从 JSON 文件加载配置."""
        with open(config_path) as f:
            data = json.load(f)
        return cls(**data)

    def to_file(self, config_path: Path) -> None:
        """保存配置到 JSON 文件."""
        config_dict = {
            "output_dir": str(self.output_dir),
            "warmup_iterations": self.warmup_iterations,
            "benchmark_iterations": self.benchmark_iterations,
            "device": self.device,
            "precision": self.precision,
            "log_level": self.log_level,
            "model_configs": self.model_configs,
        }
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)


# 预设配置模板
PRESETS = {
    "quick": BenchmarkConfig(
        warmup_iterations=1,
        benchmark_iterations=3,
        device="cpu",
    ),
    "standard": BenchmarkConfig(
        warmup_iterations=5,
        benchmark_iterations=10,
    ),
    "detailed": BenchmarkConfig(
        warmup_iterations=10,
        benchmark_iterations=30,
    ),
    "cuda_full": BenchmarkConfig(
        warmup_iterations=10,
        benchmark_iterations=50,
        device="cuda",
        precision="amp",
    ),
}


def get_preset(name: str) -> BenchmarkConfig:
    """获取预设配置."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]
