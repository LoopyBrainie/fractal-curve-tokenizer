# -*- coding: utf-8 -*-
"""
Training Core Components

提供训练系统的核心组件，包括资源统计、配置管理和训练器基类。
"""

from .resource_stats import ModelResourceStats, compute_resource_stats_batch

# 从同级模块导入以便统一访问
# 注意：这些模块在 examples/training 下的新位置

__all__ = [
    "ModelResourceStats",
    "compute_resource_stats_batch",
]
