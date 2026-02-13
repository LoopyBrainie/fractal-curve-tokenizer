# -*- coding: utf-8 -*-
"""
depth_utils.py 单元测试

测试内容：
1. compute_max_depth 边界条件 (min_dim < min_patch_size)
2. compute_actual_min_patch 正确性
3. compute_region_shape_scale 形状/尺度计算
4. compute_normalized_area 面积归一化
5. 形状-尺度相似性计算
6. 空输入、NaN输入处理
"""

import pytest
import torch
import math
from vit_pytorch.core.depth_utils import (
    compute_max_depth,
    compute_actual_min_patch,
    compute_patch_sizes,
    compute_depth_distribution,
    compute_total_candidates,
    compute_region_shape_scale,
    compute_shape_scale_similarity,
    compute_normalized_area,
    compute_area_similarity,
)


class TestComputeMaxDepth:
    """compute_max_depth 函数测试"""

    def test_standard_case_64x64(self):
        """标准情况: 64x64 图像, min_patch_size=4"""
        result = compute_max_depth((64, 64), 4)
        assert result == 4  # 64/2^4 = 4

    def test_standard_case_224x224(self):
        """标准情况: 224x224 图像, min_patch_size=4"""
        result = compute_max_depth((224, 224), 4)
        # ceil(log2(224/4)) = ceil(log2(56)) = ceil(5.81) = 6
        assert result == 6

    def test_standard_case_512x512(self):
        """标准情况: 512x512 图像, min_patch_size=4"""
        result = compute_max_depth((512, 512), 4)
        assert result == 7  # 512/2^7 = 4

    def test_min_dim_less_than_min_patch(self):
        """边界情况: min_dim < min_patch_size"""
        result = compute_max_depth((16, 16), 32)
        assert result == 0  # 应该返回 0

    def test_min_patch_size_equals_min_dim(self):
        """边界情况: min_patch_size == min_dim"""
        result = compute_max_depth((32, 32), 32)
        assert result == 0  # 32/2^0 = 32

    def test_hard_limit(self):
        """硬上限测试"""
        result = compute_max_depth((64, 64), 4, hard_limit=3)
        assert result == 3  # 被硬上限限制

    def test_negative_min_patch_size_raises(self):
        """负的 min_patch_size 应该抛出异常"""
        with pytest.raises(ValueError):
            compute_max_depth((64, 64), -1)

    def test_zero_min_patch_size_raises(self):
        """零的 min_patch_size 应该抛出异常"""
        with pytest.raises(ValueError):
            compute_max_depth((64, 64), 0)

    def test_non_square_image(self):
        """非正方形图像测试"""
        result = compute_max_depth((64, 128), 4)
        # min_dim = 64, 64/4 = 16, log2(16) = 4
        assert result == 4


class TestComputeActualMinPatch:
    """compute_actual_min_patch 函数测试"""

    def test_exact_division(self):
        """精确除法情况"""
        result = compute_actual_min_patch((64, 64), 4)
        assert result == 4  # 64/16 = 4

    def test_inexact_division(self):
        """非精确除法情况"""
        result = compute_actual_min_patch((224, 224), 4)
        assert result == 14  # 224/16 = 14 (floor)

    def test_large_depth(self):
        """大深度情况"""
        result = compute_actual_min_patch((512, 512), 7)
        assert result == 4  # 512/128 = 4

    def test_minimum_one(self):
        """最小值保证为 1"""
        result = compute_actual_min_patch((64, 64), 10)
        # 64/1024 = 0, 但应该返回 1
        assert result == 1


class TestComputePatchSizes:
    """compute_patch_sizes 函数测试"""

    def test_basic_case(self):
        """基本情况"""
        sizes = compute_patch_sizes((64, 64), 4, base_patch_size=4)
        assert sizes == (4, 8, 16, 32, 64)

    def test_different_base_size(self):
        """不同基础大小"""
        sizes = compute_patch_sizes((64, 64), 2, base_patch_size=8)
        assert sizes == (8, 16, 32)

    def test_depth_zero(self):
        """深度为 0"""
        sizes = compute_patch_sizes((64, 64), 0, base_patch_size=4)
        assert sizes == (4,)


class TestComputeDepthDistribution:
    """compute_depth_distribution 函数测试"""

    def test_basic_case(self):
        """基本情况"""
        dist = compute_depth_distribution((64, 64), 3)
        assert dist == (1, 4, 16, 64)

    def test_max_depth_four(self):
        """最大深度为 4"""
        dist = compute_depth_distribution((224, 224), 4)
        assert dist == (1, 4, 16, 64, 256)


class TestComputeTotalCandidates:
    """compute_total_candidates 函数测试"""

    def test_geometric_series(self):
        """几何级数验证"""
        # max_depth=3: 1 + 4 + 16 + 64 = 85
        result = compute_total_candidates((64, 64), 3)
        assert result == 85

    def test_max_depth_four(self):
        """最大深度为 4"""
        # 1 + 4 + 16 + 64 + 256 = 341
        result = compute_total_candidates((224, 224), 4)
        assert result == 341


class TestComputeRegionShapeScale:
    """compute_region_shape_scale 函数测试"""

    def test_square_region(self):
        """正方形区域"""
        regions = torch.tensor([[10, 10, 50, 50]])  # 40x40 区域
        aspect, area = compute_region_shape_scale(regions, (64, 64))
        # log(40/40) = 0, (40/64)*(40/64) ≈ 0.39
        assert torch.allclose(aspect, torch.tensor([0.0]), atol=1e-4)
        assert area.item() > 0.3 and area.item() < 0.5

    def test_landscape_region(self):
        """横向矩形区域"""
        regions = torch.tensor([[10, 10, 60, 30]])  # 50x20 区域
        aspect, area = compute_region_shape_scale(regions, (64, 64))
        # log(50/20) = log(2.5) > 0
        assert aspect.item() > 0

    def test_portrait_region(self):
        """纵向矩形区域"""
        regions = torch.tensor([[10, 10, 30, 50]])  # 20x40 区域
        aspect, area = compute_region_shape_scale(regions, (64, 64))
        # log(20/40) = log(0.5) < 0
        assert aspect.item() < 0

    def test_empty_input(self):
        """空输入处理"""
        empty = torch.tensor([]).reshape(0, 4)
        aspect, area = compute_region_shape_scale(empty, (64, 64))
        assert aspect.numel() == 0
        assert area.numel() == 0

    def test_2d_input(self):
        """2D 输入处理 (单个区域)"""
        regions = torch.tensor([[10, 10, 50, 30]])  # [N, 4]
        aspect, area = compute_region_shape_scale(regions, (64, 64))
        assert aspect.dim() == 1
        assert area.dim() == 1

    def test_batch_input(self):
        """批量输入"""
        regions = torch.tensor([
            [[10, 10, 50, 30]],
            [[10, 10, 50, 30]]
        ])  # [B, N, 4]
        aspect, area = compute_region_shape_scale(regions, (64, 64))
        assert aspect.dim() == 2
        assert area.dim() == 2


class TestComputeShapeScaleSimilarity:
    """compute_shape_scale_similarity 函数测试"""

    def test_identical_regions(self):
        """相同区域相似性为 1"""
        aspect = torch.tensor([0.0, 0.0])
        area = torch.tensor([0.25, 0.25])
        sim = compute_shape_scale_similarity(aspect, area)
        assert torch.allclose(torch.diag(sim), torch.ones(2))

    def test_different_regions(self):
        """不同区域相似性"""
        aspect = torch.tensor([0.0, 1.0])  # 不同纵横比
        area = torch.tensor([0.25, 0.25])  # 相同面积
        sim = compute_shape_scale_similarity(aspect, area)
        assert sim[0, 1] < 1.0  # 应该小于 1

    def test_empty_input(self):
        """空输入处理"""
        empty = torch.tensor([])
        sim = compute_shape_scale_similarity(empty, empty)
        assert sim.numel() == 0


class TestComputeNormalizedArea:
    """compute_normalized_area 函数测试"""

    def test_full_image(self):
        """整幅图像"""
        regions = torch.tensor([[0, 0, 64, 64]])  # 整幅图像
        area = compute_normalized_area(regions, (64, 64))
        # log(4096+1)/log(4096+1) ≈ 1.0
        assert torch.allclose(area, torch.tensor([1.0]), atol=1e-4)

    def test_quarter_image(self):
        """四分之一图像"""
        regions = torch.tensor([[0, 0, 32, 32]])  # 32x32 区域
        area = compute_normalized_area(regions, (64, 64))
        # log(1024+1)/log(4096+1) < 1.0
        assert area.item() < 1.0 and area.item() > 0.0

    def test_empty_input(self):
        """空输入处理"""
        empty = torch.tensor([]).reshape(0, 4)
        area = compute_normalized_area(empty, (64, 64))
        assert area.numel() == 0

    def test_single_int_size(self):
        """单个整数图像尺寸"""
        regions = torch.tensor([[0, 0, 32, 32]])
        area = compute_normalized_area(regions, 64)  # (64, 64) 的简写
        assert area.numel() == 1


class TestComputeAreaSimilarity:
    """compute_area_similarity 函数测试"""

    def test_identical_areas(self):
        """相同面积相似性为 1"""
        areas = torch.tensor([0.25, 0.25])
        sim = compute_area_similarity(areas)
        assert torch.allclose(torch.diag(sim), torch.ones(2))

    def test_different_areas(self):
        """不同面积相似性"""
        areas = torch.tensor([0.1, 0.5])
        sim = compute_area_similarity(areas)
        assert sim[0, 1] < 1.0

    def test_empty_input(self):
        """空输入处理"""
        empty = torch.tensor([])
        sim = compute_area_similarity(empty)
        assert sim.numel() == 0


class TestDepthUtilsIntegration:
    """深度工具函数集成测试"""

    def test_consistency_max_depth_and_candidates(self):
        """max_depth 和候选数量的自洽性"""
        for max_depth in range(5):
            dist = compute_depth_distribution((64, 64), max_depth)
            total = compute_total_candidates((64, 64), max_depth)
            expected = sum(dist)
            assert total == expected

    def test_patch_sizes_monotonic(self):
        """patch_sizes 单调递增"""
        sizes = compute_patch_sizes((64, 64), 4)
        for i in range(len(sizes) - 1):
            assert sizes[i] < sizes[i + 1]

    def test_shape_scale_roundtrip(self):
        """形状-尺度计算往返测试"""
        regions = torch.tensor([[10, 10, 50, 30], [20, 20, 60, 50]])
        aspect1, area1 = compute_region_shape_scale(regions, (64, 64))
        sim = compute_shape_scale_similarity(aspect1, area1)
        # 对角线应为 1
        assert torch.allclose(torch.diag(sim), torch.ones(2), atol=1e-5)
