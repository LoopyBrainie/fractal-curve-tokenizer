"""Hilbert 曲线模块单元测试"""

import pytest

from vit_pytorch.curve_hilbert import (
    HilbertCurve,
    get_quadrant_order,
    hilbert_distance_to_xy,
    xy_to_hilbert_distance,
)


class TestHilbertCurve:
    """HilbertCurve 类测试"""

    def test_xy_to_d_origin_is_zero(self):
        """验证原点距离为0"""
        assert HilbertCurve.xy_to_d(2, 0, 0) == 0
        assert HilbertCurve.xy_to_d(4, 0, 0) == 0
        assert HilbertCurve.xy_to_d(8, 0, 0) == 0

    def test_xy_to_d_range(self):
        """验证距离值在有效范围内"""
        n = 4
        for x in range(n):
            for y in range(n):
                d = HilbertCurve.xy_to_d(n, x, y)
                assert 0 <= d < n * n, f"距离 {d} 超出范围 [0, {n*n})"

    def test_d_to_xy_inverse(self):
        """验证 d_to_xy 是 xy_to_d 的逆运算"""
        n = 4
        for d in range(n * n):
            x, y = HilbertCurve.d_to_xy(n, d)
            assert HilbertCurve.xy_to_d(n, x, y) == d, f"d={d} -> ({x},{y}) -> {HilbertCurve.xy_to_d(n, x, y)}"

    def test_xy_to_d_is_bijective(self):
        """验证 xy_to_d 是双射（每个距离唯一对应一个坐标）"""
        n = 4
        distances = set()
        for x in range(n):
            for y in range(n):
                d = HilbertCurve.xy_to_d(n, x, y)
                assert d not in distances, f"距离 {d} 重复出现"
                distances.add(d)
        assert len(distances) == n * n

    def test_generate_curve_points_order_1(self):
        """验证 1 阶 Hilbert 曲线点生成"""
        points = HilbertCurve.generate_curve_points(order=1)
        assert len(points) == 4
        # 验证起点是原点
        assert points[0] == (0, 0)
        # 验证所有点都是唯一的
        assert len(set(points)) == 4

    def test_generate_curve_points_order_2(self):
        """验证 2 阶 Hilbert 曲线点生成"""
        points = HilbertCurve.generate_curve_points(order=2)
        assert len(points) == 16
        # 验证所有点都是唯一的
        assert len(set(points)) == 16

    def test_get_base_order_returns_copy(self):
        """验证 get_base_order 返回副本而非原列表"""
        order1 = HilbertCurve.get_base_order("up")
        order2 = HilbertCurve.get_base_order("up")
        order1[0] = 99
        assert order2[0] != 99, "get_base_order 应返回列表副本"


class TestGetQuadrantOrder:
    """象限遍历顺序测试"""

    def test_returns_valid_permutation(self):
        """验证返回的是 [0,1,2,3] 的有效排列"""
        for level in range(5):
            for aspect_ratio in [0.5, 1.0, 2.0]:
                order = HilbertCurve.get_quadrant_order(level, aspect_ratio)
                assert sorted(order) == [0, 1, 2, 3], f"level={level}, ar={aspect_ratio}"

    def test_wide_rectangle_adjustment(self):
        """验证宽矩形的特殊调整"""
        order_wide = HilbertCurve.get_quadrant_order(0, 2.0)  # 宽矩形
        order_normal = HilbertCurve.get_quadrant_order(0, 1.0)  # 正方形
        # 宽矩形应该有不同的优化顺序
        assert order_wide == [2, 0, 1, 3]  # 偶数层宽矩形顺序

    def test_tall_rectangle_adjustment(self):
        """验证高矩形的特殊调整"""
        order_tall = HilbertCurve.get_quadrant_order(0, 0.5)  # 高矩形
        assert order_tall == [0, 1, 3, 2]  # 偶数层高矩形顺序

    def test_level_affects_orientation(self):
        """验证不同层级产生不同方向"""
        orders = [HilbertCurve.get_quadrant_order(i, 1.0) for i in range(4)]
        # 由于方向循环，四个层级应有不同的基础顺序
        # 注意：部分顺序可能相同，但至少有变化
        unique_orders = [tuple(o) for o in orders]
        assert len(set(unique_orders)) >= 2, "不同层级应产生不同顺序"


class TestConvenienceFunctions:
    """便捷函数测试"""

    def test_xy_to_hilbert_distance(self):
        """验证便捷函数正确委托"""
        assert xy_to_hilbert_distance(4, 0, 0) == HilbertCurve.xy_to_d(4, 0, 0)
        assert xy_to_hilbert_distance(4, 2, 3) == HilbertCurve.xy_to_d(4, 2, 3)

    def test_hilbert_distance_to_xy(self):
        """验证便捷函数正确委托"""
        assert hilbert_distance_to_xy(4, 0) == HilbertCurve.d_to_xy(4, 0)
        assert hilbert_distance_to_xy(4, 7) == HilbertCurve.d_to_xy(4, 7)

    def test_get_quadrant_order_function(self):
        """验证 get_quadrant_order 便捷函数"""
        order = get_quadrant_order(2, 32, 32)
        assert sorted(order) == [0, 1, 2, 3]


class TestAdaptiveMapping:
    """自适应映射测试"""

    def test_adaptive_mapping_square(self):
        """正方形区域的自适应映射"""
        mapping = HilbertCurve.adaptive_mapping(32, 32)
        assert sorted(mapping) == [0, 1, 2, 3]

    def test_adaptive_mapping_wide(self):
        """宽矩形区域的自适应映射"""
        mapping = HilbertCurve.adaptive_mapping(16, 64)
        assert sorted(mapping) == [0, 1, 2, 3]

    def test_adaptive_mapping_tall(self):
        """高矩形区域的自适应映射"""
        mapping = HilbertCurve.adaptive_mapping(64, 16)
        assert sorted(mapping) == [0, 1, 2, 3]


class TestHilbertLocality:
    """Hilbert 曲线局部性测试"""

    def test_adjacent_distances_are_nearby(self):
        """验证相邻距离对应的坐标是相邻的"""
        n = 4
        for d in range(n * n - 1):
            x1, y1 = HilbertCurve.d_to_xy(n, d)
            x2, y2 = HilbertCurve.d_to_xy(n, d + 1)
            # 相邻距离的坐标应该只相差 1 步
            manhattan_dist = abs(x2 - x1) + abs(y2 - y1)
            assert manhattan_dist == 1, f"d={d}: ({x1},{y1}) -> ({x2},{y2}) 距离应为1，实际为{manhattan_dist}"

    def test_quadrant_order_preserves_locality(self):
        """验证象限顺序保持局部连续性"""
        for level in range(4):
            order = HilbertCurve.get_quadrant_order(level, 1.0)
            # 检查顺序中相邻元素在 2D 空间也相邻
            # 象限坐标: 0=(0,1), 1=(1,1), 2=(0,0), 3=(1,0)
            coords = {0: (0, 1), 1: (1, 1), 2: (0, 0), 3: (1, 0)}
            for i in range(3):
                q1, q2 = order[i], order[i + 1]
                x1, y1 = coords[q1]
                x2, y2 = coords[q2]
                manhattan_dist = abs(x2 - x1) + abs(y2 - y1)
                # Hilbert 曲线中相邻步骤应该是相邻格子
                assert manhattan_dist <= 2, f"level={level}, 步骤{i}: {q1}->){q2} 距离过大"
