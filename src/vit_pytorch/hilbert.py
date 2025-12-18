# -*- coding: utf-8 -*-
"""
Hilbert 曲线核心工具模块

数学形式化
============

Hilbert 曲线是一种空间填充曲线，提供 2D 网格到 1D 序列的双射：

    H: [0, n²) ↔ [0, n) × [0, n)

其中 n = 2^k 是网格边长。

核心性质
--------
1. 局部性保持: ||p1 - p2||_2 ≤ C · |H⁻¹(p1) - H⁻¹(p2)|^(1/2)
2. 自相似性: 每个象限内部是旋转/翻转的子曲线
3. 双射: xy_to_d 和 d_to_xy 互为逆操作

函数对照表
----------
+---------------------------+-----------------------------------+
| 函数                       | 数学定义                          |
+===========================+===================================+
| xy_to_d(n, x, y)          | H⁻¹: (x, y) → d ∈ [0, n²)       |
| d_to_xy(n, d)             | H: d → (x, y) ∈ [0, n)²         |
| get_quadrant_order(l,r)   | Q_l(r): level × ratio → [0,3]⁴   |
| adaptive_mapping(h, w)    | M: (h, w) → 象限遍历索引          |
+---------------------------+-----------------------------------+

算法复杂度: O(log n) 每次映射
缓存策略: LRU 缓存 1024 个结果
"""

from __future__ import annotations

import functools
from functools import lru_cache
from typing import List, Literal, Tuple

import torch._dynamo

Orientation = Literal["up", "right", "down", "left"]

# 创建兼容 torch.compile 的缓存装饰器
# lru_cache 与 torch.compile 不兼容，需要使用 dynamo.disable 排除这些函数
def _dynamo_safe_lru_cache(maxsize: int = 128):
    """LRU 缓存装饰器，兼容 torch.compile."""
    def decorator(func):
        cached = lru_cache(maxsize=maxsize)(func)
        # 使用 dynamo.disable 排除此函数，避免 torch.compile 追踪
        return torch._dynamo.disable(cached)
    return decorator


class HilbertCurve:
    """Hilbert 曲线核心实现类"""

    # 四种基本方向对应的遍历顺序
    # 索引映射: 0=左上, 1=右上, 2=左下, 3=右下
    BASE_ORDERS = {
        "up": [2, 0, 1, 3],  # 左下→左上→右上→右下 (U形向上开口)
        "right": [0, 2, 3, 1],  # 左上→左下→右下→右上 (U形向右开口)
        "down": [1, 3, 2, 0],  # 右上→右下→左下→左上 (U形向下开口)
        "left": [3, 1, 0, 2],  # 右下→右上→左上→左下 (U形向左开口)
    }

    # 递归时子象限的方向变换映射
    ORIENTATION_MAP = {
        "up": ["left", "up", "up", "right"],
        "right": ["up", "right", "right", "down"],
        "down": ["right", "down", "down", "left"],
        "left": ["down", "left", "left", "up"],
    }

    @staticmethod
    @_dynamo_safe_lru_cache(maxsize=1024)
    def xy_to_d(n: int, x: int, y: int) -> int:
        """
        将 2D 坐标转换为 Hilbert 曲线距离
        
        Args:
            n: 曲线阶数 (网格大小为 n × n，n 必须是 2 的幂)
            x: x 坐标 (0 到 n-1)
            y: y 坐标 (0 到 n-1)
            
        Returns:
            在 Hilbert 曲线上的距离 (0 到 n²-1)
        """
        d = 0
        s = n // 2
        
        while s > 0:
            rx = 1 if (x & s) > 0 else 0
            ry = 1 if (y & s) > 0 else 0
            d += s * s * ((3 * rx) ^ ry)
            
            # 旋转坐标
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            s //= 2
            
        return d

    @staticmethod
    @_dynamo_safe_lru_cache(maxsize=1024)
    def d_to_xy(n: int, d: int) -> Tuple[int, int]:
        """
        将 Hilbert 曲线距离转换为 2D 坐标
        
        Args:
            n: 曲线阶数 (网格大小为 n × n，n 必须是 2 的幂)
            d: Hilbert 曲线距离 (0 到 n²-1)
            
        Returns:
            (x, y) 坐标元组
        """
        x = y = 0
        s = 1
        
        while s < n:
            rx = 1 & (d // 2)
            ry = 1 & (d ^ rx)
            
            # 旋转坐标
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            
            x += s * rx
            y += s * ry
            d //= 4
            s *= 2
            
        return x, y

    @classmethod
    def get_base_order(cls, orientation: Orientation = "up") -> List[int]:
        """获取指定方向的基础遍历顺序"""
        return cls.BASE_ORDERS.get(orientation, cls.BASE_ORDERS["up"]).copy()

    @classmethod
    def get_quadrant_order(
        cls,
        level: int,
        aspect_ratio: float = 1.0,
        orientation: Orientation | None = None,
    ) -> List[int]:
        """
        获取四象限的 Hilbert 遍历顺序
        
        Args:
            level: 当前递归层级
            aspect_ratio: 宽高比 (w/h)
            orientation: 显式指定方向，None 时根据 level 自动确定
            
        Returns:
            [0,1,2,3] 的某种排列，表示遍历顺序
            索引映射: 0=左上, 1=右上, 2=左下, 3=右下
        """
        # 确定方向
        if orientation is None:
            orientations: List[Orientation] = ["up", "right", "down", "left"]
            orientation = orientations[level % 4]
        
        base_order = cls.get_base_order(orientation)
        
        # 根据宽高比调整
        if aspect_ratio > 1.6:  # 宽矩形
            return cls._adjust_for_wide(base_order, level)
        elif aspect_ratio < 0.625:  # 高矩形
            return cls._adjust_for_tall(base_order, level)
        
        return base_order

    @staticmethod
    def _adjust_for_wide(base_order: List[int], level: int) -> List[int]:
        """为宽矩形调整顺序，优化水平连续性"""
        if level % 2 == 0:
            return [2, 0, 1, 3]  # 左下→左上→右上→右下
        return [0, 2, 3, 1]  # 左上→左下→右下→右上

    @staticmethod
    def _adjust_for_tall(base_order: List[int], level: int) -> List[int]:
        """为高矩形调整顺序，优化垂直连续性"""
        if level % 2 == 0:
            return [0, 1, 3, 2]  # 左上→右上→右下→左下
        return [2, 3, 1, 0]  # 左下→右下→右上→左上

    @classmethod
    def adaptive_mapping(cls, h: int, w: int) -> List[int]:
        """
        为任意尺寸的 2x2 patch 网格生成自适应 Hilbert 映射
        
        Args:
            h: 网格高度
            w: 网格宽度
            
        Returns:
            四个象限按 Hilbert 顺序排列的索引列表
        """
        # 象限坐标映射: 0=左上, 1=右上, 2=左下, 3=右下
        coords = [
            (0, 1, 0),  # 左上
            (1, 1, 1),  # 右上
            (0, 0, 2),  # 左下
            (1, 0, 3),  # 右下
        ]
        
        # 计算 Hilbert 距离
        max_dim = max(h, w, 2)
        # 找到最小的 2 的幂次方 >= max_dim
        n = 1
        while n < max_dim:
            n *= 2
        
        # 计算每个象限的 Hilbert 距离
        distances = []
        for x, y, idx in coords:
            # 归一化坐标
            norm_x = (x * (n - 1)) // max(w - 1, 1) if w > 1 else x
            norm_y = (y * (n - 1)) // max(h - 1, 1) if h > 1 else y
            dist = cls.xy_to_d(n, norm_x, norm_y)
            distances.append((dist, idx))
        
        distances.sort()
        return [idx for _, idx in distances]

    @classmethod
    def generate_curve_points(cls, order: int = 2) -> List[Tuple[int, int]]:
        """
        生成 n 阶 Hilbert 曲线的所有坐标点
        
        Args:
            order: 曲线阶数 (生成 2^order × 2^order 网格)
            
        Returns:
            按 Hilbert 顺序排列的坐标点列表
        """
        n = 1 << order  # 2^order
        return [cls.d_to_xy(n, d) for d in range(n * n)]


# 便捷函数
def xy_to_hilbert_distance(n: int, x: int, y: int) -> int:
    """将 2D 坐标转换为 Hilbert 距离（便捷函数）"""
    return HilbertCurve.xy_to_d(n, x, y)


def hilbert_distance_to_xy(n: int, d: int) -> Tuple[int, int]:
    """将 Hilbert 距离转换为 2D 坐标（便捷函数）"""
    return HilbertCurve.d_to_xy(n, d)


def get_quadrant_order(level: int, h: int, w: int) -> List[int]:
    """
    获取四象限遍历顺序（便捷函数）
    
    Args:
        level: 递归层级
        h: 高度
        w: 宽度
        
    Returns:
        四象限的遍历顺序
    """
    aspect_ratio = w / h if h > 0 else 1.0
    return HilbertCurve.get_quadrant_order(level, aspect_ratio)


# ==============================================================================
# PseudoHilbertCurve: 任意尺寸矩形的 Pseudo-Hilbert 扫描
# ==============================================================================
#
# 数学形式化 (Zhang & Kamata, 2007)
# ===================================
#
# 对于 H × W 矩形区域，Pseudo-Hilbert 扫描定义为:
#
#     PH_{H,W}: [0, H×W) → [0, H) × [0, W)
#
# 递归定义:
#     1. 如果 H = W = 2^k: 使用标准 Hilbert 曲线
#     2. 如果 H > W: 水平分割为上下两部分，递归处理并连接
#     3. 如果 W > H: 垂直分割为左右两部分，递归处理并连接
#     4. 如果 H = W 且 H ≠ 2^k: 任意分割后递归
#
# 局部性保证:
#     对于相邻扫描点 p_i, p_{i+1}:
#     ||p_i - p_{i+1}||_2 ≤ √2 × max(H, W) / 2^⌊log₂ min(H, W)⌋
#
# 与标准 Hilbert 对比:
#     - 标准 Hilbert: 严格要求 n = 2^k，最大跳跃 √2
#     - Pseudo-Hilbert: 支持任意 H × W，最大跳跃约 1.5√2
#
# 混合策略阈值 (padding_ratio):
#     ρ* = 4/3 ≈ 1.333
#     当 padding_ratio < ρ* 时使用 Standard Hilbert + Padding
#     当 padding_ratio ≥ ρ* 时使用 Pseudo-Hilbert
# ==============================================================================


def _is_power_of_2(n: int) -> bool:
    """检查 n 是否为 2 的幂次方"""
    return n > 0 and (n & (n - 1)) == 0


def _next_power_of_2(n: int) -> int:
    """返回大于等于 n 的最小 2 的幂次方"""
    if n <= 0:
        return 1
    if _is_power_of_2(n):
        return n
    p = 1
    while p < n:
        p *= 2
    return p


class PseudoHilbertCurve:
    """任意尺寸矩形的 Pseudo-Hilbert 扫描.
    
    基于 Zhang & Kamata (2007) 的递归区域细分算法。
    
    核心特性:
    1. 支持任意 H × W 尺寸 (无需 2^k 约束)
    2. 对于 2^k × 2^k 情况，退化为标准 Hilbert 曲线
    3. 保持良好的局部性 (相邻扫描点在空间上接近)
    4. O(H × W) 时间复杂度生成完整序列
    
    混合策略:
    使用 PADDING_RATIO_THRESHOLD = 4/3 ≈ 1.333 决定:
    - 当填充开销 < 阈值时: 使用 Standard Hilbert + Padding
    - 当填充开销 ≥ 阈值时: 使用 Pseudo-Hilbert
    
    使用示例:
        # 标准 2^k 情况 (退化为 Hilbert)
        points = PseudoHilbertCurve.scan(16, 16)
        
        # 非 2^k 情况 (使用 Pseudo-Hilbert)
        points = PseudoHilbertCurve.scan(12, 12)
        
        # 非正方形
        points = PseudoHilbertCurve.scan(30, 20)
    """
    
    # 混合策略阈值: ρ* = 4/3
    # 数学推导: 基于局部性损失分析
    # L_pad(ρ) = √2 + 4(√ρ - 1)² vs L_pseudo ≈ 1.49
    # 解方程得 ρ* ≈ 1.30，取四叉树自然边界 4/3
    PADDING_RATIO_THRESHOLD: float = 4 / 3
    
    @classmethod
    @_dynamo_safe_lru_cache(maxsize=256)
    def scan(cls, h: int, w: int) -> Tuple[Tuple[int, int], ...]:
        """生成 H × W 矩形的 Pseudo-Hilbert 扫描序列.
        
        使用 LRU 缓存避免重复计算常见尺寸。
        
        Args:
            h: 矩形高度 (行数)
            w: 矩形宽度 (列数)
            
        Returns:
            按 Pseudo-Hilbert 顺序排列的 (x, y) 坐标元组的元组
            其中 x ∈ [0, w), y ∈ [0, h)
            
        Examples:
            >>> PseudoHilbertCurve.scan(2, 2)
            ((0, 0), (0, 1), (1, 1), (1, 0))
            
            >>> len(PseudoHilbertCurve.scan(12, 12))
            144
        """
        if h <= 0 or w <= 0:
            return ()
        
        # 自动选择策略
        return cls._scan_with_strategy(h, w)
    
    @classmethod
    def _scan_with_strategy(cls, h: int, w: int) -> Tuple[Tuple[int, int], ...]:
        """根据混合策略选择最优扫描方法.
        
        混合策略决策:
        1. 计算扩展到 2^k 的 padding_ratio
        2. 如果 ratio < THRESHOLD: 使用 Standard Hilbert + 过滤
        3. 否则: 使用 Pseudo-Hilbert 递归
        """
        n = _next_power_of_2(max(h, w))
        padding_ratio = (n * n) / (h * w)
        
        if h == w and _is_power_of_2(h):
            # 完美 2^k 正方形: 直接使用标准 Hilbert
            return cls._standard_hilbert(h)
        
        if padding_ratio < cls.PADDING_RATIO_THRESHOLD:
            # 低填充开销: 使用标准 Hilbert + 过滤
            return cls._hilbert_with_filter(h, w, n)
        else:
            # 高填充开销: 使用 Pseudo-Hilbert
            return cls._pseudo_hilbert_recursive(h, w)
    
    @classmethod
    def _standard_hilbert(cls, n: int) -> Tuple[Tuple[int, int], ...]:
        """标准 2^k Hilbert 曲线."""
        points = []
        for d in range(n * n):
            x, y = HilbertCurve.d_to_xy(n, d)
            points.append((x, y))
        return tuple(points)
    
    @classmethod
    def _hilbert_with_filter(
        cls, h: int, w: int, n: int
    ) -> Tuple[Tuple[int, int], ...]:
        """标准 Hilbert 曲线 + 过滤有效点."""
        points = []
        for d in range(n * n):
            x, y = HilbertCurve.d_to_xy(n, d)
            if x < w and y < h:
                points.append((x, y))
        return tuple(points)
    
    @classmethod
    def _pseudo_hilbert_recursive(cls, h: int, w: int) -> Tuple[Tuple[int, int], ...]:
        """Pseudo-Hilbert 递归实现.
        
        递归策略:
        1. 基础情况: 1×1 → 返回 [(0, 0)]
        2. 基础情况: 2^k × 2^k → 使用标准 Hilbert
        3. H > W: 水平分割
        4. W ≥ H: 垂直分割
        """
        # 基础情况: 1×1
        if h == 1 and w == 1:
            return ((0, 0),)
        
        # 基础情况: 1 × W (单行)
        if h == 1:
            return tuple((x, 0) for x in range(w))
        
        # 基础情况: H × 1 (单列)
        if w == 1:
            return tuple((0, y) for y in range(h))
        
        # 2^k × 2^k 正方形: 使用标准 Hilbert
        if h == w and _is_power_of_2(h):
            return cls._standard_hilbert(h)
        
        # 递归分割
        if h > w:
            # 水平分割: 上下两部分
            return cls._split_horizontal(h, w)
        else:
            # 垂直分割: 左右两部分
            return cls._split_vertical(h, w)
    
    @classmethod
    def _split_horizontal(cls, h: int, w: int) -> Tuple[Tuple[int, int], ...]:
        """水平分割 (上下两部分).
        
        将 H × W 区域分为:
        - 下半部分: h1 × W (y ∈ [0, h1))
        - 上半部分: h2 × W (y ∈ [h1, h))
        
        连接策略: 下 → 上 (保持 y 连续性)
        """
        h1 = h // 2
        h2 = h - h1
        
        # 递归处理下半部分
        lower = cls._pseudo_hilbert_recursive(h1, w)
        
        # 递归处理上半部分 (需要 y 偏移)
        upper_raw = cls._pseudo_hilbert_recursive(h2, w)
        upper = tuple((x, y + h1) for x, y in upper_raw)
        
        # 检查连接点是否需要翻转
        # 目标: lower 的最后一个点与 upper 的第一个点尽量接近
        if len(lower) > 0 and len(upper) > 0:
            lower_end = lower[-1]
            upper_start = upper[0]
            upper_end = upper[-1]
            
            # 计算距离
            dist_normal = abs(lower_end[0] - upper_start[0]) + abs(lower_end[1] - upper_start[1])
            dist_flipped = abs(lower_end[0] - upper_end[0]) + abs(lower_end[1] - upper_end[1])
            
            if dist_flipped < dist_normal:
                # 翻转上半部分以优化连接
                upper = upper[::-1]
        
        return lower + upper
    
    @classmethod
    def _split_vertical(cls, h: int, w: int) -> Tuple[Tuple[int, int], ...]:
        """垂直分割 (左右两部分).
        
        将 H × W 区域分为:
        - 左半部分: H × w1 (x ∈ [0, w1))
        - 右半部分: H × w2 (x ∈ [w1, w))
        
        连接策略: 左 → 右 (保持 x 连续性)
        """
        w1 = w // 2
        w2 = w - w1
        
        # 递归处理左半部分
        left = cls._pseudo_hilbert_recursive(h, w1)
        
        # 递归处理右半部分 (需要 x 偏移)
        right_raw = cls._pseudo_hilbert_recursive(h, w2)
        right = tuple((x + w1, y) for x, y in right_raw)
        
        # 检查连接点是否需要翻转
        if len(left) > 0 and len(right) > 0:
            left_end = left[-1]
            right_start = right[0]
            right_end = right[-1]
            
            dist_normal = abs(left_end[0] - right_start[0]) + abs(left_end[1] - right_start[1])
            dist_flipped = abs(left_end[0] - right_end[0]) + abs(left_end[1] - right_end[1])
            
            if dist_flipped < dist_normal:
                right = right[::-1]
        
        return left + right
    
    @classmethod
    def xy_to_d(cls, h: int, w: int, x: int, y: int) -> int:
        """将 2D 坐标转换为 Pseudo-Hilbert 距离.
        
        Args:
            h: 矩形高度
            w: 矩形宽度
            x: x 坐标 (0 到 w-1)
            y: y 坐标 (0 到 h-1)
            
        Returns:
            在 Pseudo-Hilbert 曲线上的距离
        """
        points = cls.scan(h, w)
        try:
            return points.index((x, y))
        except ValueError:
            raise ValueError(f"坐标 ({x}, {y}) 不在 {h}×{w} 网格范围内")
    
    @classmethod
    def d_to_xy(cls, h: int, w: int, d: int) -> Tuple[int, int]:
        """将 Pseudo-Hilbert 距离转换为 2D 坐标.
        
        Args:
            h: 矩形高度
            w: 矩形宽度
            d: Pseudo-Hilbert 距离 (0 到 h*w-1)
            
        Returns:
            (x, y) 坐标元组
        """
        points = cls.scan(h, w)
        if d < 0 or d >= len(points):
            raise ValueError(f"距离 {d} 超出范围 [0, {len(points)})")
        return points[d]
    
    @classmethod
    def compute_locality(cls, h: int, w: int) -> float:
        """计算扫描序列的平均局部性 (用于分析).
        
        局部性 = 相邻扫描点的平均欧氏距离
        理想 Hilbert: ~√2 ≈ 1.414
        
        Returns:
            平均相邻点距离
        """
        points = cls.scan(h, w)
        if len(points) < 2:
            return 0.0
        
        total_dist = 0.0
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            total_dist += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        
        return total_dist / (len(points) - 1)
    
    @classmethod
    def clear_cache(cls) -> None:
        """清空 LRU 缓存."""
        cls.scan.cache_clear()