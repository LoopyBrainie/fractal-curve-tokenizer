# -*- coding: utf-8 -*-
"""
Hilbert Curve Utilities - Hilbert 曲线工具函数

提供跨模块复用的 Hilbert 曲线生成和转换函数。

数学背景
========
Hilbert 曲线是一种空间填充曲线，提供 2D 网格到 1D 序列的双射:
    H: [0, n²) ↔ [0, n) × [0, n)

局部性保证:
    ‖H(d1) - H(d2)‖_2 ≤ √(2 × |d1 - d2|)

函数列表
========
- d_to_xy(d, n): 将 Hilbert 距离转换为 2D 坐标
- xy_to_d(x, y, n): 将 2D 坐标转换为 Hilbert 距离
- generate_hilbert_curve(order): 生成指定阶数的 Hilbert 曲线坐标序列
- generate_raster_curve(size): 生成栅格曲线坐标序列
"""

from typing import List, Tuple


def d_to_xy(d: int, n: int) -> Tuple[int, int]:
    """将 Hilbert 曲线距离转换为 2D 坐标 (Butz 算法)

    Args:
        d: Hilbert 距离 (0 到 n²-1)
        n: 网格边长 (2 的幂次)

    Returns:
        (x, y) 坐标

    示例:
        >>> d_to_xy(0, 4)
        (0, 0)
        >>> d_to_xy(1, 4)
        (0, 1)
    """
    def rot(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
        """旋转/翻转象限"""
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        return x, y

    x = y = 0
    s = 1
    t = d
    while s < n:
        rx = (t // 2) & 1
        ry = (t ^ rx) & 1
        x, y = rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def xy_to_d(x: int, y: int, n: int) -> int:
    """将 2D 坐标转换为 Hilbert 曲线距离

    Args:
        x: x 坐标 (0 到 n-1)
        y: y 坐标 (0 到 n-1)
        n: 网格边长 (2 的幂次)

    Returns:
        Hilbert 距离 (0 到 n²-1)

    示例:
        >>> xy_to_d(0, 0, 4)
        0
        >>> xy_to_d(0, 1, 4)
        1
    """
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        s //= 2
    return d


def generate_hilbert_curve(order: int) -> List[Tuple[int, int]]:
    """生成指定阶数的 Hilbert 曲线坐标序列

    Args:
        order: Hilbert 曲线阶数

    Returns:
        坐标列表，每个元素为 (x, y)

    示例:
        >>> curve = generate_hilbert_curve(2)
        >>> len(curve)
        16
        >>> curve[0]
        (0, 0)
    """
    n = 2 ** order
    coords = []
    for d in range(n * n):
        coords.append(d_to_xy(d, n))
    return coords


def generate_raster_curve(size: int) -> List[Tuple[int, int]]:
    """生成栅格曲线坐标序列 (行优先)

    Args:
        size: 网格边长

    Returns:
        坐标列表，每个元素为 (x, y)

    示例:
        >>> curve = generate_raster_curve(4)
        >>> len(curve)
        16
        >>> curve[0]
        (0, 0)
        >>> curve[1]
        (1, 0)
    """
    return [(x, y) for y in range(size) for x in range(size)]


def verify_hilbert_properties(order: int) -> bool:
    """验证 Hilbert 曲线的数学性质

    Args:
        order: Hilbert 曲线阶数

    Returns:
        所有性质验证通过返回 True

    验证项目:
        1. 逆变换: xy_to_d(d_to_xy(d, n), n) == d
        2. 唯一性: 所有坐标唯一
        3. 范围: 坐标在 [0, n) 范围内
    """
    n = 2 ** order
    coords = generate_hilbert_curve(order)

    # 1. 验证逆变换
    for d, (x, y) in enumerate(coords):
        if xy_to_d(x, y, n) != d:
            print(f"逆变换失败 at d={d}, ({x}, {y})")
            return False

    # 2. 验证唯一性
    if len(coords) != len(set(coords)):
        print("坐标不唯一")
        return False

    # 3. 验证范围
    for x, y in coords:
        if not (0 <= x < n and 0 <= y < n):
            print(f"坐标超出范围: ({x}, {y})")
            return False

    return True


if __name__ == "__main__":
    # 验证不同阶数的 Hilbert 曲线
    for order in range(1, 6):
        n = 2 ** order
        print(f"Hilbert 曲线 order={order}, n={n}: 验证 {verify_hilbert_properties(order)}")
