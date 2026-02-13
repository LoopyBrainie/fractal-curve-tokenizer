# -*- coding: utf-8 -*-
"""
向后兼容模块 - 旧路径导入重定向

此模块将旧路径的导入重定向到新的模块位置。
I162-1: 模块重构后保留旧导入路径的兼容性
"""
from vit_pytorch.layers.splitters.gumbel_topk import (
    GumbelTopKSplitter,
    TensorSplitResult,
    GumbelTopKResult,
    DepthMonitor,
    create_gumbel_topk_from_config,
)

__all__ = [
    "GumbelTopKSplitter",
    "TensorSplitResult",
    "GumbelTopKResult",
    "DepthMonitor",
    "create_gumbel_topk_from_config",
]
