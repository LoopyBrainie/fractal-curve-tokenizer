# -*- coding: utf-8 -*-
"""
Demos Package - 演示脚本

提供独立运行的可视化演示脚本。

使用方法
========
    python -m examples.analysis.demos.hilbert_demo --order 4
    python -m examples.analysis.demos.architecture_demo
    python -m examples.analysis.demos.locality_demo
"""

from .hilbert_demo import main as hilbert_demo
from .architecture_demo import main as architecture_demo
from .depth_demo import main as depth_demo
from .locality_demo import main as locality_demo
from .attention_demo import main as attention_demo

__all__ = [
    "hilbert_demo",
    "architecture_demo",
    "depth_demo",
    "locality_demo",
    "attention_demo",
]
