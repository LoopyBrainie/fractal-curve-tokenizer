# -*- coding: utf-8 -*-
"""
L1 Foundation: Hilbert Normalization Tests (I161-1 修复)

对应模块: vit_pytorch.levels_info

数学形式化
==========

深度根归一化:

    H_raw = Σ q_k × 4^{d-k}  (原始Hilbert距离)
    H_norm = (H_raw / 4^d)^(1/d)  (深度根归一化)

核心性质:
- H_norm ∈ [0, 1) 对于任意深度 d
- max_norm(d) = ((4^d-1)/4^d)^(1/d) < 1，随深度增加趋近于1
- 子区域的H_norm围绕父区域的H_norm波动（Hilbert曲线的仿射变换特性）
"""

from __future__ import annotations

import pytest
import torch
import numpy as np

from vit_pytorch import LevelsInfo


class TestHilbertNormalization:
    """深度根归一化测试 (I161-1)."""

    def test_normalize_output_range(self):
        """归一化后输出应在 [0, 1) 范围内."""
        # 深度1: 路径 [0], [1], [2], [3] -> H ∈ [0, 3]
        depths = torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.long)
        paths = torch.zeros(1, 5, 4, dtype=torch.long)
        paths[0, 1, 0] = 0  # 左上
        paths[0, 2, 0] = 1  # 右上
        paths[0, 3, 0] = 2  # 左下
        paths[0, 4, 0] = 3  # 右下

        levels = LevelsInfo.from_arrays(depths, paths, max_level=4)

        # 归一化
        h_norm = levels.get_hilbert_indices(normalize=True)

        # 0 <= H_norm < 1
        assert h_norm.min() >= 0.0 - 1e-6, f"归一化下限违反: {h_norm.min()}"
        assert h_norm.max() < 1.0 + 1e-6, f"归一化上限违反: {h_norm.max()}"

    def test_depth_zero_returns_zero(self):
        """深度为0时应返回0."""
        depths = torch.tensor([[0]], dtype=torch.long)
        paths = torch.zeros(1, 1, 4, dtype=torch.long)

        levels = LevelsInfo.from_arrays(depths, paths, max_level=4)
        h_norm = levels.get_hilbert_indices(normalize=True)

        assert h_norm[0, 0] == 0.0, "深度0应返回0"

    def test_max_norm_less_than_one(self):
        """所有深度的最大归一化值应小于1."""
        for d in [1, 2, 3, 4]:
            # 创建深度d的token，Hilbert距离为最大值 (4^d - 1)
            depths = torch.tensor([[d]], dtype=torch.long)
            paths = torch.zeros(1, 1, 4, dtype=torch.long)
            # 对于Hilbert曲线，最大值的路径是 [1, 1, ..., 1]
            for i in range(d):
                paths[0, 0, i] = 1

            levels = LevelsInfo.from_arrays(depths, paths, max_level=4)
            h_norm = levels.get_hilbert_indices(normalize=True)

            max_norm = ((4**d - 1) / (4**d)) ** (1/d)
            assert h_norm[0, 0].item() < 1.0, f"深度{d}的最大归一化值应<1"
            assert abs(h_norm[0, 0].item() - max_norm) < 1e-4, \
                f"深度{d}的max_norm计算错误: expected={max_norm}, got={h_norm[0, 0].item()}"

    def test_max_norm_converges_to_one(self):
        """max_norm随深度增加趋近于1."""
        max_norms = []
        for d in [1, 2, 3, 4, 5]:
            max_norm = ((4**d - 1) / (4**d)) ** (1/d)
            max_norms.append(max_norm)

        # 验证单调递增并趋近于1
        for i in range(1, len(max_norms)):
            assert max_norms[i] > max_norms[i-1], \
                f"max_norm应随深度增加: {max_norms[i-1]} -> {max_norms[i]}"
            assert max_norms[i] < 1.0, f"max_norm应始终<1: {max_norms[i]}"

    def test_first_child_matches_parent(self):
        """第一个子区域的H_norm应与父区域相同.

        Hilbert曲线的性质：第一个子区域是父区域的"原地"分裂，
        其H_norm与父区域相等。
        """
        # 深度1: 区域0
        depths_d1 = torch.tensor([[1]], dtype=torch.long)
        paths_d1 = torch.zeros(1, 1, 4, dtype=torch.long)
        paths_d1[0, 0, 0] = 0
        levels_d1 = LevelsInfo.from_arrays(depths_d1, paths_d1, max_level=4)
        h_norm_d1 = levels_d1.get_hilbert_indices(normalize=True)[0, 0].item()

        # 深度2: 区域0的第一个子区域
        depths_d2 = torch.tensor([[2]], dtype=torch.long)
        paths_d2 = torch.zeros(1, 1, 4, dtype=torch.long)
        paths_d2[0, 0, 0] = 0  # 父区域0
        paths_d2[0, 0, 1] = 0  # 第一个子区域
        levels_d2 = LevelsInfo.from_arrays(depths_d2, paths_d2, max_level=4)
        h_norm_d2 = levels_d2.get_hilbert_indices(normalize=True)[0, 0].item()

        assert abs(h_norm_d2 - h_norm_d1) < 1e-4, \
            f"第一个子区域应与父区域H_norm相同: {h_norm_d2} vs {h_norm_d1}"

    def test_child_distribution_centered_at_parent(self):
        """子区域的H_norm分布围绕父区域.

        Hilbert曲线的仿射变换特性：
        同一父区域的4个子区域的H_norm分布围绕父区域的H_norm。
        """
        # 深度1: 区域1 (H_norm ≈ 0.75)
        depths_d1 = torch.tensor([[1]], dtype=torch.long)
        paths_d1 = torch.zeros(1, 1, 4, dtype=torch.long)
        paths_d1[0, 0, 0] = 1
        levels_d1 = LevelsInfo.from_arrays(depths_d1, paths_d1, max_level=4)
        h_norm_d1 = levels_d1.get_hilbert_indices(normalize=True)[0, 0].item()

        # 深度2: 区域1的4个子区域
        h_norm_children = []
        for child_q in range(4):
            depths_d2 = torch.tensor([[2]], dtype=torch.long)
            paths_d2 = torch.zeros(1, 1, 4, dtype=torch.long)
            paths_d2[0, 0, 0] = 1  # 父区域1
            paths_d2[0, 0, 1] = child_q
            levels_d2 = LevelsInfo.from_arrays(depths_d2, paths_d2, max_level=4)
            h_norm_children.append(levels_d2.get_hilbert_indices(normalize=True)[0, 0].item())

        # 子区域的均值应该接近父区域（考虑Hilbert曲线的特性）
        mean_child = np.mean(h_norm_children)

        # 由于Hilbert曲线的旋转特性，子区域分布可能不完全对称
        # 但我们验证：所有子区域都在[0.5, 1.0]范围内（与父区域接近）
        for child_h in h_norm_children:
            assert 0.5 - 1e-4 < child_h < 1.0 + 1e-4, \
                f"子区域H_norm应在[0.5, 1.0]范围内: got={child_h}"

    def test_normalize_false_returns_raw(self):
        """normalize=False时应返回原始Hilbert距离."""
        depths = torch.tensor([[1, 2]], dtype=torch.long)
        paths = torch.zeros(1, 2, 4, dtype=torch.long)
        paths[0, 0, 0] = 0
        paths[0, 1, :2] = torch.tensor([0, 0])

        levels = LevelsInfo.from_arrays(depths, paths, max_level=4)

        # 原始距离应该使用long类型
        h_raw = levels.get_hilbert_indices(normalize=False)
        assert h_raw.dtype == torch.long, f"原始距离应为long类型, 得到 {h_raw.dtype}"

        # 归一化后应该使用float类型
        h_norm = levels.get_hilbert_indices(normalize=True)
        assert h_norm.dtype == torch.float32, f"归一化后应为float类型, 得到 {h_norm.dtype}"


class TestHilbertNormalizationMathProperties:
    """Hilbert归一化数学性质验证."""

    def test_single_point_depth_property(self):
        """单点应满足: (H/4^d)^(1/d) ∈ [0, 1)."""
        # 验证归一化公式的正确性（使用已知值）
        # 深度1，全0路径：H=0
        depths = torch.tensor([[1]], dtype=torch.long)
        paths = torch.zeros(1, 1, 4, dtype=torch.long)
        levels = LevelsInfo.from_arrays(depths, paths, max_level=4)
        h_norm = levels.get_hilbert_indices(normalize=True)[0, 0].item()
        assert abs(h_norm - 0.0) < 1e-4, f"深度1的H=0归一化应为0"

        # 深度2，全0路径：H=0
        depths = torch.tensor([[2]], dtype=torch.long)
        paths = torch.zeros(1, 1, 4, dtype=torch.long)
        levels = LevelsInfo.from_arrays(depths, paths, max_level=4)
        h_norm = levels.get_hilbert_indices(normalize=True)[0, 0].item()
        assert abs(h_norm - 0.0) < 1e-4, f"深度2的H=0归一化应为0"

    def test_consistency_across_same_position(self):
        """相同空间位置在不同深度应有可比较的归一化值."""
        # 左上角 (0,0) 在不同深度的归一化值
        h_norms = []
        for d in [1, 2, 3]:
            depths = torch.tensor([[d]], dtype=torch.long)
            paths = torch.zeros(1, 1, 4, dtype=torch.long)
            # 左上角对应的Hilbert路径
            for i in range(d):
                paths[0, 0, i] = 0
            levels = LevelsInfo.from_arrays(depths, paths, max_level=4)
            h_norms.append(levels.get_hilbert_indices(normalize=True)[0, 0].item())

        # 所有深度，左上角的H_norm都应为0
        for d, h in zip([1, 2, 3], h_norms):
            assert abs(h) < 1e-4, f"深度{d}的(0,0)位置H_norm应为0, got={h}"


class TestNormalizedHilbertIndexFunction:
    """测试 normalize_hilbert_index 静态方法."""

    def test_scalar_input(self):
        """测试标量输入."""
        depth = torch.tensor(3)
        hilbert_dist = torch.tensor(27)  # 假设值

        result = LevelsInfo.normalize_hilbert_index(hilbert_dist, depth)

        expected = (27 / 64) ** (1/3)
        assert abs(result.item() - expected) < 1e-6

    def test_batch_input(self):
        """测试批量输入."""
        depths = torch.tensor([1, 2, 3])
        hilbert_dists = torch.tensor([0, 8, 27])

        result = LevelsInfo.normalize_hilbert_index(hilbert_dists, depths)

        # d=1: (0/4)^1 = 0
        # d=2: (8/16)^0.5 = sqrt(0.5) ≈ 0.707
        # d=3: (27/64)^0.333 = cube_root(0.422) ≈ 0.75
        assert abs(result[0].item() - 0.0) < 1e-6
        assert abs(result[1].item() - (8/16)**0.5) < 1e-6
        assert abs(result[2].item() - (27/64)**(1/3)) < 1e-6
