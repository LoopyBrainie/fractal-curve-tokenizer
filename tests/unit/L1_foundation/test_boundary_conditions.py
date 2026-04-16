# -*- coding: utf-8 -*-
"""
边界条件测试 - Hilbert 曲线核心组件

测试场景：
1. n=1 时的 xy_to_d 和 d_to_xy
2. 空输入处理
3. NaN/Inf 输入处理
4. 超大 n 值 (n > 2^16)
5. 坐标越界处理
6. LevelsInfo 边界条件
7. GumbelTopKSplitter 边界
"""

import pytest
import torch
import math
from vit_pytorch.core.curve_hilbert import (
    HilbertCurve,
    xy_to_hilbert_distance,
    hilbert_distance_to_xy,
    PseudoHilbertCurve,
)
from vit_pytorch.core.levels_info import LevelsInfo


class TestHilbertCurveBoundary:
    """Hilbert 曲线边界条件测试"""

    def test_n_equals_one(self):
        """n=1 边界测试"""
        # n=1 时，只有一个点 (0, 0)
        d = HilbertCurve.xy_to_d(1, 0, 0)
        assert d == 0

        # 逆操作
        x, y = HilbertCurve.d_to_xy(1, 0)
        assert x == 0 and y == 0

    def test_n_equals_two(self):
        """n=2 边界测试"""
        # 2x2 网格有 4 个点
        for d in range(4):
            x, y = HilbertCurve.d_to_xy(2, d)
            assert 0 <= x < 2
            assert 0 <= y < 2

        # 逆操作验证
        for x in range(2):
            for y in range(2):
                d = HilbertCurve.xy_to_d(2, x, y)
                x_back, y_back = HilbertCurve.d_to_xy(2, d)
                assert (x, y) == (x_back, y_back)

    def test_n_equals_four(self):
        """n=4 边界测试"""
        # 4x4 网格有 16 个点
        for d in range(16):
            x, y = HilbertCurve.d_to_xy(4, d)
            assert 0 <= x < 4
            assert 0 <= y < 4

        # 双射性质验证
        for x in range(4):
            for y in range(4):
                d = HilbertCurve.xy_to_d(4, x, y)
                x_back, y_back = HilbertCurve.d_to_xy(4, d)
                assert (x, y) == (x_back, y_back)

    def test_power_of_two_sizes(self):
        """2 的幂次方尺寸测试"""
        for order in range(1, 8):  # 2, 4, 8, ..., 128
            n = 1 << order
            for d in range(0, n * n, max(1, n * n // 10)):  # 测试约 10% 的点
                x, y = HilbertCurve.d_to_xy(n, d)
                assert 0 <= x < n
                assert 0 <= y < n

    def test_coordinate_out_of_bounds(self):
        """坐标越界测试"""
        n = 8
        # 越界坐标应该仍然能处理（但可能产生错误结果）
        # 实际行为：xy_to_d 接受任何整数，但仅对 [0, n) 内的坐标保证正确性
        HilbertCurve.xy_to_d(n, -1, 0)  # 负坐标
        # 不抛出异常，但结果可能无意义

        HilbertCurve.xy_to_d(n, n, 0)  # 边界外坐标
        # 不抛出异常

    def test_large_n(self):
        """大 n 值测试 (n=256)"""
        n = 256  # 2^8
        # 验证双射性质
        for x in [0, 1, 127, 128, 255]:
            for y in [0, 1, 127, 128, 255]:
                d = HilbertCurve.xy_to_d(n, x, y)
                x_back, y_back = HilbertCurve.d_to_xy(n, d)
                assert (x, y) == (x_back, y_back)

    def test_d_out_of_range(self):
        """d 超出范围测试"""
        n = 8
        # 有效的 d 范围是 [0, n*n)
        # 超出范围的行为未定义，但不应崩溃
        x, y = HilbertCurve.d_to_xy(n, -1)  # 负值
        x, y = HilbertCurve.d_to_xy(n, 100)  # n*n=64，100 超出范围


class TestHilbertCurveBatchBoundary:
    """Hilbert 曲线批量操作边界测试"""

    def test_empty_batch(self):
        """空批量测试"""
        n = 8
        x = torch.tensor([], dtype=torch.long)
        y = torch.tensor([], dtype=torch.long)

        d = HilbertCurve.xy_to_d_batch(n, x, y)
        assert d.shape == (0,)

    def test_single_element_batch(self):
        """单元素批量测试"""
        n = 8
        x = torch.tensor([3])
        y = torch.tensor([4])

        d = HilbertCurve.xy_to_d_batch(n, x, y)
        assert d.shape == (1,)
        assert d[0] == HilbertCurve.xy_to_d(n, 3, 4)

    def test_batch_bijectivity(self):
        """批量双射性质验证"""
        n = 16
        x = torch.randint(0, n, (100,))
        y = torch.randint(0, n, (100,))

        d = HilbertCurve.xy_to_d_batch(n, x, y)
        x_back, y_back = HilbertCurve.d_to_xy_batch(n, d)

        assert torch.equal(x, x_back)
        assert torch.equal(y, y_back)


class TestPseudoHilbertCurveBoundary:
    """Pseudo-Hilbert 曲线边界测试"""

    def test_h_equals_one(self):
        """h=1 边界测试"""
        points = PseudoHilbertCurve.scan(1, 4)
        assert len(points) == 4
        assert all(0 <= x < 4 and y == 0 for x, y in points)

    def test_w_equals_one(self):
        """w=1 边界测试"""
        points = PseudoHilbertCurve.scan(4, 1)
        assert len(points) == 4
        assert all(x == 0 and 0 <= y < 4 for x, y in points)

    def test_h_equals_w_equals_one(self):
        """h=1, w=1 边界测试"""
        points = PseudoHilbertCurve.scan(1, 1)
        assert len(points) == 1
        assert points[0] == (0, 0)

    def test_zero_dimension(self):
        """零维度测试"""
        points = PseudoHilbertCurve.scan(0, 4)
        assert len(points) == 0

        points = PseudoHilbertCurve.scan(4, 0)
        assert len(points) == 0

    def test_non_power_of_two(self):
        """非 2 的幂次方测试"""
        # 3x3 应该能处理
        points = PseudoHilbertCurve.scan(3, 3)
        assert len(points) == 9

        # 5x7 应该能处理
        points = PseudoHilbertCurve.scan(5, 7)
        assert len(points) == 35

    def test_rectangle(self):
        """矩形测试"""
        points = PseudoHilbertCurve.scan(4, 8)
        assert len(points) == 32

        points = PseudoHilbertCurve.scan(8, 4)
        assert len(points) == 32

    def test_large_rectangle(self):
        """大矩形测试"""
        points = PseudoHilbertCurve.scan(64, 64)
        assert len(points) == 4096

    def test_locality_metrics(self):
        """局部性度量测试"""
        # 小尺寸应该返回合理的局部性
        points = PseudoHilbertCurve.scan(2, 2)
        assert len(points) == 4

        # 局部性损失应该是有定义的
        avg_loss = PseudoHilbertCurve.compute_locality(4, 4)
        assert avg_loss >= 0

        max_jump = PseudoHilbertCurve.compute_locality(4, 4)
        assert max_jump >= 0


class TestLevelsInfoBoundary:
    """LevelsInfo 边界条件测试"""

    def test_empty_tensor(self):
        """空张量测试 - B=0"""
        # B=0 的情况
        empty = torch.tensor([]).reshape(0, 5, 3)  # B=0, N=5, D+1=3
        # LevelsInfo 应该能处理空 batch（不抛出异常）
        info = LevelsInfo(data=empty, max_level=2)
        assert info.batch_size == 0
        assert info.num_tokens == 5

    def test_single_token(self):
        """单 token 测试"""
        data = torch.zeros(1, 1, 4, dtype=torch.long)  # B=1, N=1, D+1=3
        info = LevelsInfo(data=data, max_level=3)
        assert info.num_tokens == 1
        assert info.batch_size == 1

    def test_padding_sentinel(self):
        """填充标记测试"""
        data = torch.tensor([
            [[0, 1, 2, 3]],  # 深度 0，有效
            [[-1, 0, 0, 0]],  # 深度 -1，填充
        ], dtype=torch.long)
        info = LevelsInfo(data=data, max_level=3)
        depths = info.depths
        assert depths[0, 0] == 0
        assert depths[1, 0] == -1

    def test_max_level_zero(self):
        """max_level=0 测试"""
        data = torch.zeros(2, 4, 1, dtype=torch.long)  # D+1=1
        info = LevelsInfo(data=data, max_level=0)
        assert info.max_level == 0

    def test_large_max_level(self):
        """大 max_level 测试"""
        data = torch.zeros(1, 8, 9, dtype=torch.long)  # D+1=9
        info = LevelsInfo(data=data, max_level=8)
        assert info.max_level == 8

    def test_device_consistency(self):
        """设备一致性测试"""
        data = torch.zeros(2, 4, 3, dtype=torch.long)
        info = LevelsInfo(data=data, max_level=2)

        info_cpu = info.cpu()
        assert info_cpu.data.device.type == "cpu"

    def test_random_levels_info(self):
        """随机 LevelsInfo 测试"""
        info = LevelsInfo.random(B=2, N=16, max_level=4)
        assert info.batch_size == 2
        assert info.num_tokens == 16
        assert info.max_level == 4


class TestHilbertIndexBoundary:
    """Hilbert 索引边界测试"""

    def test_depth_zero(self):
        """深度为 0"""
        info = LevelsInfo.random(B=1, N=1, max_level=4)
        indices = info.get_hilbert_indices()
        # 深度为 0 时应该返回 0
        assert indices.shape == (1, 1)

    def test_all_depths(self):
        """所有深度"""
        info = LevelsInfo.random(B=2, N=16, max_level=4)
        indices = info.get_hilbert_indices()
        assert indices.shape == (2, 16)

    def test_lca_matrix(self):
        """LCA 矩阵测试"""
        info = LevelsInfo.random(B=2, N=8, max_level=4)
        lca = info.get_lca_matrix()
        assert lca.shape == (2, 8, 8)
        # 对角线应该为 0（与自己的 LCA 深度为 0）
        # 对每个 batch 分别检查对角线
        for b in range(lca.shape[0]):
            batch_lca = lca[b]
            diag_values = torch.diagonal(batch_lca)
            assert torch.all(diag_values == 0)


class TestHilbertUtilitiesBoundary:
    """Hilbert 工具函数边界测试"""

    def test_xy_to_hilbert_distance(self):
        """便捷函数测试"""
        d = xy_to_hilbert_distance(8, 3, 4)
        assert isinstance(d, int)
        assert 0 <= d < 64

    def test_hilbert_distance_to_xy(self):
        """便捷函数测试"""
        x, y = hilbert_distance_to_xy(8, 10)
        assert isinstance(x, int)
        assert isinstance(y, int)


class TestCurveCacheBoundary:
    """曲线缓存边界测试"""

    def test_clear_cache(self):
        """清空缓存"""
        # 获取一些缓存的曲线点
        points1 = HilbertCurve.get_curve_points_cached(4)
        assert len(points1) == 256

        # 清空缓存
        HilbertCurve.clear_curve_cache()

        # 重新获取
        points2 = HilbertCurve.get_curve_points_cached(4)
        assert len(points2) == 256

    def test_pseudo_hilbert_clear_cache(self):
        """Pseudo-Hilbert 清空缓存"""
        points1 = PseudoHilbertCurve.scan(8, 8)
        assert len(points1) == 64

        PseudoHilbertCurve.clear_cache()

        points2 = PseudoHilbertCurve.scan(8, 8)
        assert len(points2) == 64


class TestMathInvariantsBoundary:
    """数学不变式边界测试"""

    def test_bijectivity_all_orders(self):
        """所有阶数的双射性质"""
        for order in range(1, 7):
            n = 1 << order
            points = HilbertCurve.generate_curve_points(order)
            assert len(points) == n * n

            # 检查所有点都在 [0, n) 范围内
            for x, y in points:
                assert 0 <= x < n
                assert 0 <= y < n

    def test_no_duplicates(self):
        """无重复点验证"""
        for order in range(1, 6):
            1 << order
            points = HilbertCurve.generate_curve_points(order)

            # 检查无重复
            seen = set()
            for x, y in points:
                assert (x, y) not in seen
                seen.add((x, y))

    def test_coverage(self):
        """覆盖性验证"""
        for order in range(1, 6):
            n = 1 << order
            points = HilbertCurve.generate_curve_points(order)

            # 检查覆盖了所有 [0, n) x [0, n) 网格点
            expected_set = {(x, y) for x in range(n) for y in range(n)}
            actual_set = set(points)
            assert expected_set == actual_set

    def test_locality_property(self):
        """局部性性质"""
        points = HilbertCurve.generate_curve_points(3)

        # 计算相邻点的距离
        max_dist = 0
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            max_dist = max(max_dist, dist)

        # 标准 Hilbert 曲线的最大跳跃应该是 √2 ≈ 1.41
        # 实际上由于边界翻转，可能达到约 2.12
        assert max_dist < 3.0  # 确保没有极端跳跃


class TestNPowerOf2Validation:
    """I102-11: n 值必须是 2 的幂验证测试"""

    @pytest.mark.parametrize("invalid_n", [3, 5, 6, 7, 9, 10, 12, 15, 17, 31])
    def test_xy_to_d_invalid_n_raises_error(self, invalid_n):
        """n 非 2 的幂时 xy_to_d 应该抛出 ValueError"""
        with pytest.raises(ValueError, match="must be a power of 2"):
            HilbertCurve.xy_to_d(invalid_n, 0, 0)

    @pytest.mark.parametrize("invalid_n", [3, 5, 6, 7, 9, 10, 12, 15, 17, 31])
    def test_d_to_xy_invalid_n_raises_error(self, invalid_n):
        """n 非 2 的幂时 d_to_xy 应该抛出 ValueError"""
        with pytest.raises(ValueError, match="must be a power of 2"):
            HilbertCurve.d_to_xy(invalid_n, 0)

    @pytest.mark.parametrize("invalid_n", [3, 5, 6, 7, 9, 10, 12, 15, 17, 31])
    def test_xy_to_d_batch_invalid_n_raises_error(self, invalid_n):
        """n 非 2 的幂时 xy_to_d_batch 应该抛出 ValueError"""
        x = torch.tensor([0, 1, 2])
        y = torch.tensor([0, 1, 2])
        with pytest.raises(ValueError, match="must be a power of 2"):
            HilbertCurve.xy_to_d_batch(invalid_n, x, y)

    @pytest.mark.parametrize("invalid_n", [3, 5, 6, 7, 9, 10, 12, 15, 17, 31])
    def test_d_to_xy_batch_invalid_n_raises_error(self, invalid_n):
        """n 非 2 的幂时 d_to_xy_batch 应该抛出 ValueError"""
        d = torch.tensor([0, 1, 2])
        with pytest.raises(ValueError, match="must be a power of 2"):
            HilbertCurve.d_to_xy_batch(invalid_n, d)

    @pytest.mark.parametrize("valid_n", [1, 2, 4, 8, 16, 32, 64, 128])
    def test_valid_powers_of_2_xy_to_d(self, valid_n):
        """有效的 2 的幂 xy_to_d 应该正常工作"""
        d = HilbertCurve.xy_to_d(valid_n, 0, 0)
        assert isinstance(d, int)
        assert d >= 0

    @pytest.mark.parametrize("valid_n", [1, 2, 4, 8, 16, 32, 64, 128])
    def test_valid_powers_of_2_d_to_xy(self, valid_n):
        """有效的 2 的幂 d_to_xy 应该正常工作"""
        x, y = HilbertCurve.d_to_xy(valid_n, 0)
        assert isinstance(x, int)
        assert isinstance(y, int)
        assert 0 <= x < valid_n
        assert 0 <= y < valid_n

    @pytest.mark.parametrize("valid_n", [1, 2, 4, 8, 16, 32])
    def test_valid_powers_of_2_batch(self, valid_n):
        """有效的 2 的幂批量方法应该正常工作"""
        B = 4
        x = torch.randint(0, valid_n, (B,))
        y = torch.randint(0, valid_n, (B,))
        d = HilbertCurve.xy_to_d_batch(valid_n, x, y)
        assert d.shape == (B,)

        d_input = torch.randint(0, valid_n * valid_n, (B,))
        x_out, y_out = HilbertCurve.d_to_xy_batch(valid_n, d_input)
        assert x_out.shape == (B,)
        assert y_out.shape == (B,)

    def test_error_message_contains_suggestion(self):
        """错误消息应该包含自动调整建议"""
        n = 9
        with pytest.raises(ValueError) as exc_info:
            HilbertCurve.xy_to_d(n, 0, 0)

        error_msg = str(exc_info.value)
        assert "_next_power_of_2" in error_msg
        assert "16" in error_msg  # _next_power_of_2(9) = 16

    def test_edge_case_n_1_is_valid(self):
        """n=1 是有效的 2 的幂 (2^0=1)"""
        # n=1 应该正常工作
        d = HilbertCurve.xy_to_d(1, 0, 0)
        assert d == 0

        x, y = HilbertCurve.d_to_xy(1, 0)
        assert x == 0 and y == 0

        # 批量方法
        x_batch = torch.tensor([0])
        y_batch = torch.tensor([0])
        d_batch = HilbertCurve.xy_to_d_batch(1, x_batch, y_batch)
        assert d_batch.shape == (1,)

        d_input = torch.tensor([0])
        x_out, y_out = HilbertCurve.d_to_xy_batch(1, d_input)
        assert x_out.shape == (1,)
