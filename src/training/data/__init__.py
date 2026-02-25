# -*- coding: utf-8 -*-
"""Training Data - 数据加载和增强模块

本模块包含数据集加载、预处理和增强功能。

主要组件:
- transforms.py: 数据增强 (MixupCutmix, DatasetSpec, DATASETS)
- dataloaders.py: 数据加载器创建

使用示例:
    from training.data import DATASETS, DatasetSpec
    from training.data import MixupCutmix, mixup_criterion
    from training.data import create_dataloaders
"""

from .transforms import (
    DatasetSpec,
    DATASETS,
    MixupCutmix,
    mixup_criterion,
)

# dataloaders.py will be added in future

__all__ = [
    "DatasetSpec",
    "DATASETS",
    "MixupCutmix",
    "mixup_criterion",
    # Future: create_dataloaders,
]
