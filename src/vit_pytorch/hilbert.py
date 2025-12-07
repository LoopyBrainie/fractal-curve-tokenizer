"""
Hilbert 曲线核心工具模块

统一的 Hilbert 曲线实现，提供：
- 坐标到距离的映射 (xy_to_d)
- 距离到坐标的映射 (d_to_xy)
- 象限遍历顺序生成
- 自适应 Hilbert 映射

基于标准 Hilbert 曲线算法实现，支持非正方形区域的自适应处理。
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Literal, Tuple

Orientation = Literal["up", "right", "down", "left"]


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
    @lru_cache(maxsize=1024)
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
    @lru_cache(maxsize=1024)
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
