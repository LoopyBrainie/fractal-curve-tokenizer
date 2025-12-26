# -*- coding: utf-8 -*-
"""
Pseudo-Hilbert 曲线单元测试

测试 ARCH-P2-3: 任意分辨率图像支持

测试内容:
1. 扫描覆盖性: 所有点被访问且无重复
2. 局部性保持: 相邻扫描点的空间距离
3. 2^k 兼容性: 退化为标准 Hilbert 曲线
4. 混合策略: 阈值正确选择
5. 坐标转换: xy_to_d 和 d_to_xy 互逆
6. 边界情况: 1×1, 单行/单列
7. 非正方形: H ≠ W 情况
"""

import math
from typing import Set, Tuple

import pytest
import torch

from vit_pytorch.curve_hilbert import (
    HilbertCurve,
    PseudoHilbertCurve,
    _is_power_of_2,
    _next_power_of_2,
)
from vit_pytorch.curve_hilbert_indexer import HilbertIndexer, HilbertPathCache
from vit_pytorch.config_fractal import FractalConfig


class TestHelperFunctions:
    """测试辅助函数"""
    
    def test_is_power_of_2(self):
        """测试 2 的幂次方判断"""
        # 是 2 的幂
        for k in range(10):
            assert _is_power_of_2(2 ** k), f"2^{k} = {2**k} 应该是 2 的幂"
        
        # 不是 2 的幂
        non_powers = [0, 3, 5, 6, 7, 9, 10, 12, 15, 17, 100]
        for n in non_powers:
            assert not _is_power_of_2(n), f"{n} 不应该是 2 的幂"
    
    def test_next_power_of_2(self):
        """测试下一个 2 的幂次方"""
        test_cases = [
            (0, 1), (1, 1), (2, 2), (3, 4), (4, 4),
            (5, 8), (7, 8), (8, 8), (9, 16), (15, 16),
            (16, 16), (17, 32), (100, 128),
        ]
        for n, expected in test_cases:
            result = _next_power_of_2(n)
            assert result == expected, f"next_power_of_2({n}) = {result}, expected {expected}"


class TestPseudoHilbertCurve:
    """测试 PseudoHilbertCurve 类"""
    
    @pytest.mark.parametrize("h,w", [
        (1, 1),    # 最小情况
        (2, 2),    # 最小 2^k
        (4, 4),    # 小 2^k
        (8, 8),    # 中等 2^k
        (16, 16),  # 标准 2^k
        (3, 3),    # 非 2^k 正方形
        (5, 5),
        (6, 6),
        (7, 7),
        (12, 12),  # 典型非 2^k
        (15, 15),  # 接近 2^k
    ])
    def test_scan_coverage(self, h: int, w: int):
        """验证扫描覆盖所有点且无重复"""
        points = PseudoHilbertCurve.scan(h, w)
        
        # 点数正确
        assert len(points) == h * w, f"{h}×{w} 网格应有 {h*w} 个点，实际 {len(points)}"
        
        # 无重复
        point_set: Set[Tuple[int, int]] = set(points)
        assert len(point_set) == h * w, f"存在重复点"
        
        # 所有点在有效范围内
        for x, y in points:
            assert 0 <= x < w, f"x={x} 超出范围 [0, {w})"
            assert 0 <= y < h, f"y={y} 超出范围 [0, {h})"
    
    @pytest.mark.parametrize("h,w", [
        (2, 3),    # 宽矩形
        (3, 2),    # 高矩形
        (4, 6),
        (6, 4),
        (10, 15),
        (15, 10),
        (30, 20),  # 大矩形
    ])
    def test_scan_coverage_rect(self, h: int, w: int):
        """验证非正方形矩形的扫描覆盖"""
        points = PseudoHilbertCurve.scan(h, w)
        
        assert len(points) == h * w
        assert len(set(points)) == h * w
        
        for x, y in points:
            assert 0 <= x < w
            assert 0 <= y < h
    
    @pytest.mark.parametrize("n", [2, 4, 8, 16])
    def test_2k_compatibility(self, n: int):
        """验证 2^k 情况与标准 Hilbert 一致"""
        pseudo_points = PseudoHilbertCurve.scan(n, n)
        
        # 生成标准 Hilbert 点
        hilbert_points = tuple(
            HilbertCurve.d_to_xy(n, d) for d in range(n * n)
        )
        
        assert pseudo_points == hilbert_points, (
            f"2^k = {n} 情况下 Pseudo-Hilbert 应与标准 Hilbert 一致"
        )
    
    def test_locality_preservation(self):
        """验证局部性保持"""
        # 测试多个尺寸的局部性
        test_cases = [
            (4, 4),   # 2^k
            (8, 8),   # 2^k
            (12, 12), # 非 2^k，使用 Pseudo-Hilbert
            (15, 15), # 非 2^k，使用 Hilbert + Padding
        ]
        
        for h, w in test_cases:
            locality = PseudoHilbertCurve.compute_locality(h, w)
            
            # 理想 Hilbert 局部性约为 √2 ≈ 1.414
            # 允许一定容差，Pseudo-Hilbert 可能略高
            assert locality < 3.0, (
                f"{h}×{w} 局部性 {locality:.2f} 过高 (应 < 3.0)"
            )
            
            # 对于 2^k，应该非常接近理想值
            if _is_power_of_2(h) and h == w:
                assert locality < 1.5, (
                    f"2^k = {h} 局部性 {locality:.2f} 应接近 √2 ≈ 1.414"
                )
    
    def test_locality_pseudo_vs_filter(self):
        """验证 Pseudo-Hilbert 比过滤方法局部性更好"""
        # 12×12: padding_ratio = 256/144 ≈ 1.78 > 4/3
        # 应该使用 Pseudo-Hilbert
        h, w = 12, 12
        
        locality = PseudoHilbertCurve.compute_locality(h, w)
        
        # 根据分析，Pseudo-Hilbert 局部性约 1.49
        # 过滤方法局部性约 1.87
        assert locality < 1.7, f"12×12 Pseudo-Hilbert 局部性应 < 1.7，实际 {locality:.2f}"
    
    @pytest.mark.parametrize("h,w", [
        (4, 4), (8, 8), (12, 12), (10, 15),
    ])
    def test_xy_to_d_d_to_xy_inverse(self, h: int, w: int):
        """验证 xy_to_d 和 d_to_xy 互为逆操作"""
        for y in range(h):
            for x in range(w):
                d = PseudoHilbertCurve.xy_to_d(h, w, x, y)
                x_back, y_back = PseudoHilbertCurve.d_to_xy(h, w, d)
                assert (x, y) == (x_back, y_back), (
                    f"坐标 ({x}, {y}) → d={d} → ({x_back}, {y_back}) 不匹配"
                )
    
    def test_d_to_xy_all_distances(self):
        """验证所有距离值的转换"""
        h, w = 5, 7
        n = h * w
        
        for d in range(n):
            x, y = PseudoHilbertCurve.d_to_xy(h, w, d)
            assert 0 <= x < w
            assert 0 <= y < h
    
    def test_invalid_coordinates(self):
        """测试无效坐标的错误处理"""
        with pytest.raises(ValueError, match="不在.*范围内"):
            PseudoHilbertCurve.xy_to_d(4, 4, 5, 0)
        
        with pytest.raises(ValueError, match="不在.*范围内"):
            PseudoHilbertCurve.xy_to_d(4, 4, 0, 5)
    
    def test_invalid_distance(self):
        """测试无效距离的错误处理"""
        with pytest.raises(ValueError, match="超出范围"):
            PseudoHilbertCurve.d_to_xy(4, 4, 16)  # 0-15 有效
        
        with pytest.raises(ValueError, match="超出范围"):
            PseudoHilbertCurve.d_to_xy(4, 4, -1)
    
    def test_empty_grid(self):
        """测试空网格"""
        assert PseudoHilbertCurve.scan(0, 0) == ()
        assert PseudoHilbertCurve.scan(0, 5) == ()
        assert PseudoHilbertCurve.scan(5, 0) == ()
    
    def test_single_row_column(self):
        """测试单行/单列"""
        # 单行
        row = PseudoHilbertCurve.scan(1, 5)
        assert row == tuple((x, 0) for x in range(5))
        
        # 单列
        col = PseudoHilbertCurve.scan(5, 1)
        assert col == tuple((0, y) for y in range(5))
    
    def test_threshold_value(self):
        """验证阈值常量"""
        assert abs(PseudoHilbertCurve.PADDING_RATIO_THRESHOLD - 4/3) < 1e-10
    
    def test_cache_clearing(self):
        """测试缓存清理"""
        # 生成一些缓存
        PseudoHilbertCurve.scan(10, 10)
        PseudoHilbertCurve.scan(12, 12)
        
        # 清理缓存
        PseudoHilbertCurve.clear_cache()
        
        # 应该仍然正常工作
        result = PseudoHilbertCurve.scan(10, 10)
        assert len(result) == 100


class TestHilbertPathCacheIntegration:
    """测试 HilbertPathCache 与 Pseudo-Hilbert 的集成"""
    
    def setup_method(self):
        """每个测试前清理缓存"""
        HilbertPathCache.clear_cache()
    
    @pytest.mark.parametrize("h,w", [
        (4, 4), (8, 8), (16, 16),  # 2^k
        (12, 12), (6, 6), (10, 10),  # 非 2^k
        (5, 8), (8, 5),  # 非正方形
    ])
    def test_cache_returns_correct_size(self, h: int, w: int):
        """验证缓存返回正确大小的张量"""
        hilbert_to_raster, quadtree_paths = HilbertPathCache.get_or_compute(h, w, 8)
        
        n_tokens = h * w
        assert hilbert_to_raster.shape == (n_tokens,)
        assert quadtree_paths.shape == (n_tokens, 8)
    
    def test_cache_raster_indices_valid(self):
        """验证光栅索引有效"""
        h, w = 12, 12
        hilbert_to_raster, _ = HilbertPathCache.get_or_compute(h, w, 8)
        
        # 所有索引应在 [0, h*w) 范围内
        assert hilbert_to_raster.min() >= 0
        assert hilbert_to_raster.max() < h * w
        
        # 无重复索引
        assert len(hilbert_to_raster.unique()) == h * w
    
    def test_cache_consistency_with_pseudo_hilbert(self):
        """验证缓存与 PseudoHilbertCurve 一致"""
        h, w = 12, 12
        
        # 从缓存获取
        hilbert_to_raster, _ = HilbertPathCache.get_or_compute(h, w, 8)
        
        # 直接从 PseudoHilbertCurve 获取
        points = PseudoHilbertCurve.scan(h, w)
        expected = torch.tensor([y * w + x for x, y in points], dtype=torch.long)
        
        assert torch.equal(hilbert_to_raster, expected)


class TestHilbertIndexerIntegration:
    """测试 HilbertIndexer 与 Pseudo-Hilbert 的集成"""
    
    @pytest.mark.parametrize("grid_size", [4, 8, 16, 12, 6, 10])
    def test_get_hilbert_order(self, grid_size: int):
        """验证 HilbertIndexer.get_hilbert_order"""
        indices = HilbertIndexer.get_hilbert_order(grid_size)
        
        n = grid_size * grid_size
        assert indices.shape == (n,)
        assert indices.min() >= 0
        assert indices.max() < n
        assert len(indices.unique()) == n
    
    @pytest.mark.parametrize("h,w", [(4, 6), (6, 4), (10, 15)])
    def test_get_hilbert_order_rect(self, h: int, w: int):
        """验证 HilbertIndexer.get_hilbert_order_rect"""
        indices = HilbertIndexer.get_hilbert_order_rect(h, w)
        
        n = h * w
        assert indices.shape == (n,)
        assert indices.min() >= 0
        assert indices.max() < n
        assert len(indices.unique()) == n


class TestFractalConfigIntegration:
    """测试 FractalConfig 与 Pseudo-Hilbert 的集成"""
    
    def test_2k_config_standard_hilbert(self):
        """验证 2^k 配置使用标准 Hilbert"""
        config = FractalConfig(64, 4)  # grid_size = 16 = 2^4
        
        assert config.grid_size == 16
        assert not config.uses_pseudo_hilbert
    
    def test_non_2k_low_padding_uses_standard(self):
        """验证低填充率使用标准 Hilbert"""
        # 60/4 = 15, 扩展到 16
        # padding_ratio = 256/225 ≈ 1.14 < 4/3
        config = FractalConfig(60, 4)
        
        assert config.grid_size == 15
        assert not config.uses_pseudo_hilbert
    
    def test_non_2k_high_padding_uses_pseudo(self):
        """验证高填充率使用 Pseudo-Hilbert"""
        # 48/4 = 12, 扩展到 16
        # padding_ratio = 256/144 ≈ 1.78 > 4/3
        config = FractalConfig(48, 4)
        
        assert config.grid_size == 12
        assert config.uses_pseudo_hilbert
    
    def test_config_repr_shows_strategy(self):
        """验证 __repr__ 显示 Hilbert 策略"""
        config = FractalConfig(48, 4)  # 使用 Pseudo-Hilbert
        repr_str = repr(config)
        
        assert "uses_pseudo_hilbert=True" in repr_str
        assert "非 2^k" in repr_str
    
    @pytest.mark.parametrize("image_size,patch_size,expected_pseudo", [
        (64, 4, False),   # 16 = 2^4
        (32, 4, False),   # 8 = 2^3
        (60, 4, False),   # 15, ratio=1.14 < 4/3
        (56, 4, False),   # 14, ratio=1.31 < 4/3
        (48, 4, True),    # 12, ratio=1.78 > 4/3
        (100, 10, True),  # 10, ratio=2.56 > 4/3
        (90, 10, True),   # 9, ratio=1.78 > 4/3
    ])
    def test_pseudo_hilbert_selection(
        self, image_size: int, patch_size: int, expected_pseudo: bool
    ):
        """验证 Pseudo-Hilbert 选择逻辑"""
        config = FractalConfig(image_size, patch_size)
        assert config.uses_pseudo_hilbert == expected_pseudo, (
            f"image_size={image_size}, patch_size={patch_size}: "
            f"expected uses_pseudo_hilbert={expected_pseudo}, "
            f"got {config.uses_pseudo_hilbert}"
        )
    
    def test_no_2k_constraint_error(self):
        """验证非 2^k 不再抛出错误"""
        # 以前这些配置会抛出 ValueError
        # 现在应该正常工作
        configs = [
            (60, 4),   # grid_size = 15
            (48, 4),   # grid_size = 12
            (100, 10), # grid_size = 10
            (90, 9),   # grid_size = 10
        ]
        
        for image_size, patch_size in configs:
            config = FractalConfig(image_size, patch_size)
            assert config.grid_size > 0
            assert config.num_tokens > 0


class TestEndToEndIntegration:
    """端到端集成测试"""
    
    def test_feature_reordering_non_2k(self):
        """测试非 2^k 特征图的重排序"""
        B, D, H, W = 2, 64, 12, 12
        features = torch.randn(B, D, H, W)
        
        # 使用 HilbertIndexer 重排序
        reordered = HilbertIndexer.reorder_to_hilbert(features, H, W)
        
        assert reordered.shape == (B, H * W, D)
        
        # 验证所有值都被保留
        original_flat = features.flatten(2).transpose(1, 2)  # [B, H*W, D]
        for b in range(B):
            original_set = set(original_flat[b, :, 0].tolist())
            reordered_set = set(reordered[b, :, 0].tolist())
            assert original_set == reordered_set


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
