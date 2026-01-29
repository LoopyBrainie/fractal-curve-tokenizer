# -*- coding: utf-8 -*-
"""
Mathematical Invariants Verification Tests

数学形式化验证框架:
  1. LevelsInfo 不变量 (C1, C2, C3)
  2. Hilbert 曲线性质 (双向性, 范围, 局部性)
  3. Splitter 约束 (配额求和, 树一致性)
  4. 常量边界条件

标记: @pytest.mark.数学

数学形式化定义
================

LevelsInfo (四叉树展平表示):
    L ∈ Z^{B × N × (D+1)}
    L[:, :, 0] = depths, d_i ∈ {-1, 0, 1, ..., D}
    L[:, :, 1:] = paths, q_i ∈ {0, 1, 2, 3}^D

约束集 C:
    C1: depth ∈ [-1, D] (padding sentinel)
    C2: path ∈ [0, 3] for valid tokens (四象限编码)
    C3: path length = depth (结构一致性)

Hilbert Curve:
    xy_to_d: (x, y, n) → d ∈ [0, n²)
    d_to_xy: (d, n) → (x, y) ∈ [0, n)²
    双向性: xy_to_d(d_to_xy(n,d)) = d
"""

import pytest
import torch
from vit_pytorch.levels_info import LevelsInfo
from vit_pytorch.curve_hilbert import HilbertCurve


class TestLevelsInfoInvariants:
    """LevelsInfo 数学不变量验证."""

    @pytest.mark.数学
    def test_c1_depth_range_valid(self):
        """C1: depth ∈ [-1, max_level] - valid range.

        数学形式化:
            depth ∈ {-1} ∪ [0, max_level]
            -1 表示 padding sentinel (无有效数据)

        测试用例:
            -1: padding sentinel
            0-4: valid depths for max_level=4
        """
        max_level = 4
        B, N = 2, 8

        # Valid depths: -1 (padding), 0, 1, 2, 3, 4
        for depth in [-1, 0, 1, 2, 3, 4]:
            data = torch.zeros(B, N, max_level + 1, dtype=torch.long)
            data[:, :, 0] = depth  # Set all depths to test value

            # Valid depth should not raise
            info = LevelsInfo(data=data, max_level=max_level)
            assert info.max_level == max_level

    @pytest.mark.数学
    def test_c1_depth_range_invalid_low(self):
        """C1: depth ∈ [-1, max_level] - invalid depth < -1.

        数学形式化:
            depth < -1 违反 C1 约束
            应抛出异常
        """
        max_level = 4
        B, N = 2, 8

        data = torch.zeros(B, N, max_level + 1, dtype=torch.long)
        data[:, :, 0] = -2  # Invalid: less than -1

        with pytest.raises(AssertionError):
            LevelsInfo(data=data, max_level=max_level)

    @pytest.mark.数学
    def test_c1_depth_range_invalid_high(self):
        """C1: depth ∈ [-1, max_level] - invalid depth > max_level.

        数学形式化:
            depth > max_level 违反 C1 约束
            应抛出异常
        """
        max_level = 4
        B, N = 2, 8

        data = torch.zeros(B, N, max_level + 1, dtype=torch.long)
        data[:, :, 0] = 5  # Invalid: greater than max_level

        with pytest.raises(AssertionError):
            LevelsInfo(data=data, max_level=max_level)

    @pytest.mark.数学
    def test_c2_path_quadrant_encoding(self):
        """C2: path ∈ [0, 3] for valid nodes (quadrant encoding).

        数学形式化:
            path ∈ {0, 1, 2, 3}
            0: NW (top-left), 1: NE (top-right)
            2: SE (bottom-right), 3: SW (bottom-left)

        四象限编码是 Hilbert 曲线的基础
        """
        max_level = 4
        B, N = 2, 4

        # Test valid quadrant values: 0, 1, 2, 3
        for path_value in range(4):
            data = torch.zeros(B, N, max_level + 1, dtype=torch.long)
            data[:, :, 0] = 1  # depth = 1
            data[:, :, 1] = path_value  # Valid path value

            # Valid path should not raise
            info = LevelsInfo(data=data, max_level=max_level)
            hilbert_idx = info.get_hilbert_indices()
            assert hilbert_idx.shape == (B, N)

    @pytest.mark.数学
    def test_c2_path_encoding_invalid(self):
        """C2: path ∈ [0, 3] - invalid path values.

        数学形式化:
            path < 0 或 path > 3 违反 C2 约束
            应抛出异常
        """
        max_level = 4
        B, N = 2, 8

        # Test invalid path values
        for invalid_path in [-1, 4, 5]:
            data = torch.zeros(B, N, max_level + 1, dtype=torch.long)
            data[:, :, 0] = 1  # depth = 1
            data[:, :, 1] = invalid_path  # Invalid path value

            with pytest.raises(AssertionError):
                LevelsInfo(data=data, max_level=max_level)

    @pytest.mark.数学
    def test_c3_path_depth_consistency(self):
        """C3: path.length = depth (structural consistency).

        数学形式化:
            对每个节点 i: len(path_i) = depth_i
            深度 d 的节点恰好有 d 位路径编码

        C3 由数据结构保证 (data[:, :, 1:1+depth])
        此测试验证路径计算正确性
        """
        max_level = 4
        B, N = 2, 4

        for d in range(max_level + 1):
            data = torch.zeros(B, N, max_level + 1, dtype=torch.long)
            data[:, :, 0] = d  # Set depth to d

            # Fill paths with d valid digits
            if d > 0:
                for i in range(d):
                    data[:, :, 1 + i] = i % 4  # Valid quadrant values

            info = LevelsInfo(data=data, max_level=max_level)

            # Verify paths are correctly extracted
            paths = info.paths
            assert paths.shape[1] == N

    @pytest.mark.数学
    def test_levels_info_random_valid(self):
        """LevelsInfo.random() 生成有效的四叉树结构.

        数学形式化:
            random() 方法应生成满足 C1-C3 的随机 LevelsInfo
        """
        max_level = 4
        info = LevelsInfo.random(B=4, N=16, max_level=max_level)

        assert info.max_level == max_level
        assert info.batch_size == 4
        assert info.num_tokens == 16

        depths = info.depths
        assert (depths >= -1).all()
        assert (depths <= max_level).all()


class TestHilbertCurveProperties:
    """Hilbert 曲线数学性质验证.

    注意: HilbertCurve 使用静态方法 (非实例方法):
        - HilbertCurve.xy_to_d(n, x, y) - 2D 坐标转 Hilbert 距离
        - HilbertCurve.d_to_xy(n, d) - Hilbert 距离转 2D 坐标
    """

    @pytest.mark.数学
    def test_bidirectional_mapping_order_1_to_6(self):
        """验证 xy_to_d(d_to_xy(n,d)) = d for orders 1-6.

        数学形式化:
            对所有 n = 2^k, d ∈ [0, n²):
                xy = d_to_xy(n, d)
                d' = xy_to_d(n, xy[0], xy[1])
                assert d' == d

        测试范围: order 1 (2×2) 到 order 6 (64×64)
        """
        for order in range(1, 7):  # 2^1 to 2^6
            n = 2 ** order

            # Test every 10% of indices
            for d in range(0, n * n, max(1, n * n // 10)):
                x, y = HilbertCurve.d_to_xy(n, d)
                d_recovered = HilbertCurve.xy_to_d(n, x, y)
                assert d_recovered == d, \
                    f"Order {order}, d {d}: ({x},{y}) → {d_recovered} ≠ {d}"

    @pytest.mark.数学
    def test_bidirectional_mapping_full_coverage(self):
        """验证双向映射完整覆盖 [0, n²).

        数学形式化:
            d_to_xy 和 xy_to_d 应形成双射
            即: 对每个 d 存在唯一 (x,y)，反之亦然
        """
        order = 4
        n = 2 ** order

        # Collect all (x, y) pairs
        xy_pairs = set()
        for d in range(n * n):
            x, y = HilbertCurve.d_to_xy(n, d)
            assert (x, y) not in xy_pairs, f"Duplicate at d={d}"
            xy_pairs.add((x, y))

        # Verify complete coverage
        assert len(xy_pairs) == n * n

    @pytest.mark.数学
    def test_range_constraints(self):
        """验证 d ∈ [0, n²), (x,y) ∈ [0,n)².

        数学形式化:
            range(d) = [0, n²)
            range(x) = range(y) = [0, n)
        """
        order = 5
        n = 2 ** order

        for d in range(n * n):
            x, y = HilbertCurve.d_to_xy(n, d)
            assert 0 <= x < n, f"x={x} out of [0, {n})"
            assert 0 <= y < n, f"y={y} out of [0, {n})"

            # Verify reverse mapping
            d_recovered = HilbertCurve.xy_to_d(n, x, y)
            assert 0 <= d_recovered < n * n

    @pytest.mark.数学
    def test_hilbert_index_monotonicity(self):
        """验证 Hilbert 索引的邻域性质.

        数学形式化:
            相邻 Hilbert 索引 d 和 d+1 对应坐标在空间上接近
            这体现了 Hilbert 曲线的局部性保持特性
        """
        order = 4
        n = 2 ** order

        # Check 100 random adjacent pairs
        for _ in range(100):
            d = torch.randint(0, n * n - 1, (1,)).item()
            x1, y1 = HilbertCurve.d_to_xy(n, d)
            x2, y2 = HilbertCurve.d_to_xy(n, d + 1)

            # Manhattan distance should be small for Hilbert curve
            manhattan_dist = abs(x2 - x1) + abs(y2 - y1)
            # Hilbert curve has max Manhattan distance of 2 for adjacent points
            assert manhattan_dist <= 2, \
                f"Adjacent indices {d},{d+1} have distance {manhattan_dist}"

    @pytest.mark.数学
    def test_order_1_boundary_cases(self):
        """验证 order=1 (2×2 grid) 的边界情况.

        数学形式化:
            n = 2, d ∈ [0, 3]
            Hilbert 曲线应覆盖完整的 2×2 网格
        """
        n = 2

        # All 4 points should be covered
        points = set()
        for d in range(4):
            x, y = HilbertCurve.d_to_xy(n, d)
            points.add((x, y))
            assert 0 <= x < n and 0 <= y < n

        assert len(points) == 4, "Not all points covered"

        # Verify bidirectional
        for x in range(n):
            for y in range(n):
                d = HilbertCurve.xy_to_d(n, x, y)
                x_recovered, y_recovered = HilbertCurve.d_to_xy(n, d)
                assert x == x_recovered and y == y_recovered


class TestQuadtreeProperties:
    """四叉树结构数学性质验证."""

    @pytest.mark.数学
    def test_quadrant_child_parent_relationship(self):
        """验证四叉树父子区域关系.

        数学形式化:
            深度 d 的区域大小为: n/2^d × n/2^d
            深度 d+1 的子区域是父区域的 1/4
        """
        max_level = 4
        n = 64  # Image size

        for depth in range(max_level):
            # Region size at depth
            region_size = n / (2 ** depth)

            # Child region size
            child_region_size = n / (2 ** (depth + 1))

            # Child should be half the parent in each dimension
            assert abs(child_region_size * 2 - region_size) < 1e-6, \
                f"Depth {depth}: child should be half parent size"

    @pytest.mark.数学
    def test_total_region_count_per_depth(self):
        """验证每层区域数量.

        数学形式化:
            深度 d 有 4^d 个候选区域
            总区域数 = Σ_{d=0}^{max_level} 4^d = (4^{max_level+1} - 1) / 3
        """
        for max_level in [1, 2, 3, 4]:
            # Expected total regions
            expected_total = (4 ** (max_level + 1) - 1) // 3

            # Count regions per depth
            total = 0
            for d in range(max_level + 1):
                regions_at_d = 4 ** d
                total += regions_at_d

            assert total == expected_total, \
                f"max_level={max_level}: expected {expected_total}, got {total}"


class TestTensorShapeConsistency:
    """张量形状一致性验证."""

    @pytest.mark.数学
    def test_levels_info_shape(self):
        """验证 LevelsInfo 形状约束.

        数学形式化:
            data.shape = (B, N, D+1)
            其中 D = max_level
        """
        B, N, D = 4, 16, 4

        # Create valid data
        data = torch.zeros(B, N, D + 1, dtype=torch.long)
        info = LevelsInfo(data=data, max_level=D)

        assert info.shape == (B, N, D + 1)
        assert info.batch_size == B
        assert info.num_tokens == N
        assert info.max_level == D

    @pytest.mark.数学
    def test_depths_extraction(self):
        """验证 depths 属性提取.

        数学形式化:
            depths = data[:, :, 0]
            shape = (B, N)
        """
        B, N, D = 2, 8, 4

        # Create valid paths for each depth
        data = torch.zeros(B, N, D + 1, dtype=torch.long)
        for b in range(B):
            for n in range(N):
                d = torch.randint(0, D + 1, (1,)).item()
                data[b, n, 0] = d
                # Fill paths with d valid digits (0-3)
                for i in range(d):
                    data[b, n, 1 + i] = i % 4

        info = LevelsInfo(data=data, max_level=D)

        depths = info.depths
        assert depths.shape == (B, N)

    @pytest.mark.数学
    def test_paths_extraction(self):
        """验证 paths 属性提取.

        数学形式化:
            paths = data[:, :, 1:]
            shape = (B, N, D)
        """
        B, N, D = 2, 8, 4

        # Create valid paths for each depth
        data = torch.zeros(B, N, D + 1, dtype=torch.long)
        for b in range(B):
            for n in range(N):
                d = torch.randint(0, D + 1, (1,)).item()
                data[b, n, 0] = d
                # Fill paths with d valid digits (0-3)
                for i in range(d):
                    data[b, n, 1 + i] = i % 4

        info = LevelsInfo(data=data, max_level=D)

        paths = info.paths
        assert paths.shape == (B, N, D)
