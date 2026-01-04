# -*- coding: utf-8 -*-
"""
Training Core Components

提供训练系统的核心组件，包括资源统计、配置管理和训练器基类。
"""

from .resource_stats import ModelResourceStats, compute_resource_stats_batch

__all__ = [
    "ModelResourceStats",
    "compute_resource_stats_batch",
]
