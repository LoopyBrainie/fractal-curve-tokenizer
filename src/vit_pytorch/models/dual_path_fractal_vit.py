# -*- coding: utf-8 -*-
"""
双路径Fractal ViT架构 (I162-1)

⚠️ 已废弃模块
============
此模块已整合到 fractal_vit.py 中。
请使用:

    from vit_pytorch import FractalCurveViT

或启用双路径模式:

    from vit_pytorch import FractalCurveViT
    model = FractalCurveViT(use_pattern_plugin=True)

保留此文件仅为向后兼容。
"""

from __future__ import annotations

# 向后兼容导入 - 从主模块重新导出
from vit_pytorch.models.fractal_vit import (
    TrainingStats,
)

# 保留 TrainingStatsV2 别名以保持兼容性
TrainingStatsV2 = TrainingStats

# 向后兼容导入 - Pattern Encoder 相关类
# 这些类现在定义在 core.pattern_encoder 中
