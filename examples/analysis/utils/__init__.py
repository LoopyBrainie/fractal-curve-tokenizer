# -*- coding: utf-8 -*-
"""
Utils Package - 工具函数包

提供 Fractal Curve ViT 分析模块的通用工具函数。

模块列表
========
- hilbert_utils: Hilbert 曲线工具函数
- color_utils: 颜色映射工具
- math_docs: 数学公式文档

使用方法
========
    from examples.analysis.utils import hilbert_utils
    from examples.analysis.utils.color_utils import get_depth_colors
"""

from .hilbert_utils import (
    d_to_xy,
    xy_to_d,
    generate_hilbert_curve,
    generate_raster_curve,
    verify_hilbert_properties,
)

from .color_utils import (
    get_depth_colors,
    get_attention_colors,
)

__all__ = [
    # hilbert_utils
    "d_to_xy",
    "xy_to_d",
    "generate_hilbert_curve",
    "generate_raster_curve",
    "verify_hilbert_properties",
    # color_utils
    "get_depth_colors",
    "get_attention_colors",
]
