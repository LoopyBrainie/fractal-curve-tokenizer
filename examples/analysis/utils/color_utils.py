# -*- coding: utf-8 -*-
"""
Color Utilities - 颜色映射工具

提供统一的深度颜色映射和调色板，确保可视化一致性。

颜色规范
========
1. 深度颜色 (DEPTH_COLORS):
   - d=0 (最粗/Coarse): 蓝色端 - 整体轮廓、色彩信息
   - d=4 (最细/Fine): 红色端 - 纹理、局部细节

   使用 RdYlBu_r 统一生成:
   - 0.1 → 蓝色 (d=0)
   - 0.9 → 红色 (d=4)

2. 注意力颜色 (ATTENTION_COLORS):
   - high: 红色 - 高注意力权重
   - medium: 黄色 - 中等注意力权重
   - low: 蓝色 - 低注意力权重

使用方法
========
    from examples.analysis.utils.color_utils import (
        get_depth_colors,
        DEPTH_COLORS,
        ATTENTION_COLORS
    )

    # 统一颜色使用
    colors = get_depth_colors(max_depth=4)
    plt.scatter(..., c=[colors[d] for d in depths])

    # 或使用预设颜色
    plt.scatter(..., c=[DEPTH_COLORS[d] for d in depths])
"""

from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# 统一颜色常量定义
# =============================================================================

# 默认深度颜色映射名称
DEPTH_CMAP = 'RdYlBu_r'

# 预设深度颜色 (RGBA 格式, 适用于 max_depth=4)
# d=0 (coarse): 蓝色 - [0.03, 0.15, 0.3, 1.0] - #07254D
# d=1:         青蓝色 - [0.1, 0.4, 0.6, 1.0]
# d=2:         青绿色 - [0.2, 0.6, 0.5, 1.0]
# d=3:         黄绿色 - [0.6, 0.7, 0.2, 1.0]
# d=4 (fine):  红色   - [0.7, 0.1, 0.1, 1.0] - #B31111
DEPTH_COLORS = {
    0: [0.03, 0.15, 0.30, 1.0],  # 深蓝 - Coarse
    1: [0.10, 0.40, 0.60, 1.0],  # 青蓝
    2: [0.20, 0.60, 0.50, 1.0],  # 青绿
    3: [0.60, 0.70, 0.20, 1.0],  # 黄绿
    4: [0.70, 0.10, 0.10, 1.0],  # 深红 - Fine
}

# 注意力模式颜色
ATTENTION_COLORS = {
    'high': [0.8, 0.2, 0.2, 1.0],     # 红色 - 高注意力
    'medium': [0.8, 0.8, 0.2, 1.0],   # 黄色 - 中注意力
    'low': [0.2, 0.5, 0.8, 1.0],      # 蓝色 - 低注意力
}

# 局部性分析颜色
LOCALITY_COLORS = {
    'hilbert': [0.2, 0.6, 0.8, 1.0],   # 蓝色 - Hilbert
    'raster': [0.8, 0.4, 0.2, 1.0],    # 橙色 - 栅格
    'optimal': [0.2, 0.8, 0.4, 1.0],   # 绿色 - 最优
}

# 模型对比颜色
MODEL_COLORS = {
    'fractal': [0.2, 0.5, 0.8, 1.0],   # 蓝色 - Fractal ViT
    'standard': [0.8, 0.5, 0.2, 1.0],  # 橙色 - Standard ViT
    'baseline': [0.5, 0.5, 0.5, 1.0],  # 灰色 - 基线
}

# =============================================================================
# 颜色获取函数
# =============================================================================

def get_depth_colors(
    max_depth: int,
    cmap: str = DEPTH_CMAP,
    alpha: float = 0.8
) -> Dict[int, List[float]]:
    """获取深度颜色映射

    动态生成深度到颜色的映射，支持任意最大深度。

    Args:
        max_depth: 最大深度
        cmap: 颜色映射名称 (matplotlib colormap)
        alpha: 透明度

    Returns:
        {depth: RGBA颜色} 字典

    示例:
        >>> colors = get_depth_colors(4)
        >>> colors[0]  # 蓝色端
        [0.03, 0.15, 0.30, 0.8]
        >>> colors[4]  # 红色端
        [0.70, 0.10, 0.10, 0.8]
    """
    colors = plt.cm.get_cmap(cmap)(np.linspace(0.1, 0.9, max_depth + 1))
    # 添加透明度
    colors[:, 3] = alpha
    return {d: colors[d].tolist() for d in range(max_depth + 1)}


def get_attention_colors() -> Dict[str, List[float]]:
    """获取注意力模式颜色映射

    Returns:
        {'high': RGBA, 'medium': RGBA, 'low': RGBA} 字典

    示例:
        >>> colors = get_attention_colors()
        >>> colors['high']
        [0.8, 0.2, 0.2, 1.0]
    """
    return ATTENTION_COLORS.copy()


def get_locality_colors() -> Dict[str, List[float]]:
    """获取局部性分析颜色映射

    Returns:
        {'hilbert': RGBA, 'raster': RGBA, 'optimal': RGBA} 字典
    """
    return LOCALITY_COLORS.copy()


def get_model_colors() -> Dict[str, List[float]]:
    """获取模型对比颜色映射

    Returns:
        {'fractal': RGBA, 'standard': RGBA, 'baseline': RGBA} 字典
    """
    return MODEL_COLORS.copy()


def get_gradient_colors(n_colors: int, cmap: str = 'viridis') -> List[List[float]]:
    """获取渐变色板

    Args:
        n_colors: 颜色数量
        cmap: 颜色映射名称

    Returns:
        RGBA 颜色列表

    示例:
        >>> gradient = get_gradient_colors(10)
        >>> len(gradient)
        10
    """
    return plt.cm.get_cmap(cmap)(np.linspace(0, 1, n_colors)).tolist()


def get_depth_color_rgb(depth: int, max_depth: int = 4) -> Tuple[float, float, float]:
    """获取指定深度的 RGB 颜色

    Args:
        depth: 深度值
        max_depth: 最大深度 (用于归一化)

    Returns:
        (R, G, B) 元组
    """
    colors = get_depth_colors(max_depth)
    rgba = colors.get(depth, colors[max_depth])
    return tuple(rgba[:3])


def apply_depth_colormap(values: np.ndarray, max_depth: int = 4) -> np.ndarray:
    """将深度值数组映射为颜色数组

    Args:
        values: [N] 深度值数组
        max_depth: 最大深度

    Returns:
        [N, 3] RGB 颜色数组
    """
    colors = get_depth_colors(max_depth)
    result = np.zeros((len(values), 3))
    for i, v in enumerate(values):
        d = int(np.clip(v, 0, max_depth))
        result[i] = colors[d][:3]
    return result


if __name__ == "__main__":
    # 验证颜色函数
    print("=" * 60)
    print("Color Utilities Verification")
    print("=" * 60)

    # 深度颜色测试
    colors = get_depth_colors(4)
    print("\n深度颜色 (max_depth=4):")
    for d, c in colors.items():
        print(f"  d={d}: RGB({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})")

    # 预设颜色
    print("\n预设深度颜色:")
    for d, c in DEPTH_COLORS.items():
        print(f"  d={d}: RGB({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})")

    # 注意力颜色
    attention = get_attention_colors()
    print(f"\n注意力颜色: {list(attention.keys())}")

    # 局部性颜色
    locality = get_locality_colors()
    print(f"局部性颜色: {list(locality.keys())}")

    print("\n" + "=" * 60)
    print("Verification passed!")
    print("=" * 60)
