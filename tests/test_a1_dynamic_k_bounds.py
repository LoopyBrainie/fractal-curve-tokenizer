"""
A1: 动态 K 边界约束 - 单元测试

Tests for:
- K_max = α × N (α = 5%)
- K_min = β × N (β = 0.5%)
- 硬上限 K_max <= 4096
- 硬下限 K_min >= 8
- K_max / K_min ≈ 10

Mathematical properties:
    K_min(N) = max(8, ceil(0.005 × N))
    K_max(N) = min(4096, ceil(0.05 × N))
"""

import math
import pytest


class TestA1DynamicKBounds:
    """A1: 动态 K 边界约束验证"""

    def test_dynamic_formula(self):
        """验证动态公式 K_max = 5%×N, K_min = 0.5%×N"""
        from vit_pytorch.constants import (
            K_MAX_SAMPLE_RATIO,
            K_MIN_SAMPLE_RATIO,
            K_MAX_HARD_LIMIT,
            K_MIN_HARD_LIMIT,
        )

        # 期望参数值
        assert K_MAX_SAMPLE_RATIO == 0.05
        assert K_MIN_SAMPLE_RATIO == 0.005
        assert K_MAX_HARD_LIMIT == 4096
        assert K_MIN_HARD_LIMIT == 8

    def test_dynamic_k_bounds_shallow_depth(self):
        """验证浅层深度 (D=4, N=341) 的动态边界"""
        from vit_pytorch.constants import (
            K_MAX_SAMPLE_RATIO,
            K_MIN_SAMPLE_RATIO,
            K_MAX_HARD_LIMIT,
            K_MIN_HARD_LIMIT,
        )

        N = 341  # D=4

        K_min = max(K_MIN_HARD_LIMIT, int(math.ceil(K_MIN_SAMPLE_RATIO * N)))
        K_max = min(K_MAX_HARD_LIMIT, int(math.ceil(K_MAX_SAMPLE_RATIO * N)))

        # 验证计算
        # ceil(0.005 × 341) = 2，但被硬下限覆盖为 8
        assert K_min == 8  # 被 K_MIN_HARD_LIMIT 覆盖
        # ceil(0.05 × 341) = ceil(17.05) = 18
        assert K_max == 18
        assert K_max >= K_min
        assert K_min >= K_MIN_HARD_LIMIT

    def test_dynamic_k_bounds_medium_depth(self):
        """验证中层深度 (D=6, N=5461) 的动态边界"""
        from vit_pytorch.constants import (
            K_MAX_SAMPLE_RATIO,
            K_MIN_SAMPLE_RATIO,
            K_MAX_HARD_LIMIT,
            K_MIN_HARD_LIMIT,
        )

        N = 5461  # D=6

        K_min = max(K_MIN_HARD_LIMIT, int(math.ceil(K_MIN_SAMPLE_RATIO * N)))
        K_max = min(K_MAX_HARD_LIMIT, int(math.ceil(K_MAX_SAMPLE_RATIO * N)))

        # 验证计算
        # 注意: 实际实现使用 K_COVERAGE_MIN=0.01，测试使用 K_MIN_SAMPLE_RATIO=0.005
        # 这里测试的是常量定义，实际行为由 _token_coverage_min 决定
        assert K_min == 28  # ceil(0.005 × 5461) = ceil(27.305) = 28
        assert K_max == 274  # ceil(0.05 × 5461) = ceil(273.05) = 274
        assert K_max >= K_min

    def test_dynamic_k_bounds_deep_depth(self):
        """验证深层深度 (D=8, N=87381) 的动态边界"""
        from vit_pytorch.constants import (
            K_MAX_SAMPLE_RATIO,
            K_MIN_SAMPLE_RATIO,
            K_MAX_HARD_LIMIT,
            K_MIN_HARD_LIMIT,
        )

        N = 87381  # D=8

        K_min = max(K_MIN_HARD_LIMIT, int(math.ceil(K_MIN_SAMPLE_RATIO * N)))
        K_max = min(K_MAX_HARD_LIMIT, int(math.ceil(K_MAX_SAMPLE_RATIO * N)))

        # 验证计算
        assert K_min == 437  # ceil(0.005 × 87381) = ceil(436.905) = 437
        # ceil(0.05 × 87381) = 4370，但被硬上限截断为 4096
        assert K_max == 4096  # 被 K_MAX_HARD_LIMIT 覆盖
        assert K_max >= K_min

    def test_ratio_constraint(self):
        """验证 K_max / K_min ≈ 10 的比例约束 (未达到硬限制时)"""
        from vit_pytorch.constants import (
            K_MAX_SAMPLE_RATIO,
            K_MIN_SAMPLE_RATIO,
            K_MAX_HARD_LIMIT,
            K_MIN_HARD_LIMIT,
        )

        # 测试不同深度的 N 值 (排除被硬限制截断的情况)
        test_cases = [
            (1365, 7, 69),   # D=5: 比例 9.9
            (5461, 28, 274), # D=6: 比例 9.8
            (21845, 110, 1093),  # D=7: 比例 9.9
        ]

        for N, expected_min, expected_max in test_cases:
            K_min = max(K_MIN_HARD_LIMIT, int(math.ceil(K_MIN_SAMPLE_RATIO * N)))
            K_max = min(K_MAX_HARD_LIMIT, int(math.ceil(K_MAX_SAMPLE_RATIO * N)))

            # 验证未达到硬限制
            assert K_max < K_MAX_HARD_LIMIT, f"N={N}: K_max={K_max} 被硬上限截断"

            ratio = K_max / K_min

            # 比例应在 8-12 范围内 (允许 ±20% 误差)
            assert 8 <= ratio <= 12, \
                f"N={N}: K_min={K_min}, K_max={K_max}, ratio={ratio:.2f} 不在 [8, 12] 范围内"

    def test_hard_limit(self):
        """验证硬上限约束 (K_max 被截断)"""
        from vit_pytorch.constants import (
            K_MAX_SAMPLE_RATIO,
            K_MIN_SAMPLE_RATIO,
            K_MAX_HARD_LIMIT,
            K_MIN_HARD_LIMIT,
        )

        # 非常大的 N 应该被硬上限截断
        N = 1000000  # 1M 候选区域

        K_min = max(K_MIN_HARD_LIMIT, int(math.ceil(K_MIN_SAMPLE_RATIO * N)))
        K_max = min(K_MAX_HARD_LIMIT, int(math.ceil(K_MAX_SAMPLE_RATIO * N)))

        # K_max 应该被硬上限截断
        assert K_max == K_MAX_HARD_LIMIT, f"K_max={K_max} != {K_MAX_HARD_LIMIT}"
        # K_min = ceil(0.005 × 1000000) = 5000 > 8，所以不会被硬下限覆盖
        assert K_min == 5000, f"K_min={K_min} != 5000"

    def test_small_n(self):
        """验证小 N 值的边界情况 (K_min 被硬下限覆盖)"""
        from vit_pytorch.constants import (
            K_MAX_SAMPLE_RATIO,
            K_MIN_SAMPLE_RATIO,
            K_MAX_HARD_LIMIT,
            K_MIN_HARD_LIMIT,
        )

        # 非常小的 N (N=100 < K_MIN_HARD_LIMIT / β = 8 / 0.005 = 1600)
        N = 100

        K_min = max(K_MIN_HARD_LIMIT, int(math.ceil(K_MIN_SAMPLE_RATIO * N)))
        K_max = min(K_MAX_HARD_LIMIT, int(math.ceil(K_MAX_SAMPLE_RATIO * N)))

        # K_min 应该被硬下限覆盖: ceil(0.005 × 100) = 1 < 8
        assert K_min == K_MIN_HARD_LIMIT, f"K_min={K_min} != {K_MIN_HARD_LIMIT}"
        # K_max 应该小于 4096: ceil(0.05 × 100) = 5
        assert K_max == 5, f"K_max={K_max} != 5"
        # 注意: 当 N 非常小时，K_max < K_min 是正常行为
        # 因为硬下限约束 K_min >= 8，而小 N 时采样率计算出的 K_max 可能更小


class TestA1SplitterIntegration:
    """A1: GumbelTopKSplitter 集成测试"""

    def test_splitter_has_dynamic_k_bounds_method(self):
        """验证 Splitter 有 _get_dynamic_k_bounds 方法"""
        from vit_pytorch.gumbel_topk_splitter import GumbelTopKSplitter
        from vit_pytorch.constants import (
            K_MAX_SAMPLE_RATIO,
            K_MIN_SAMPLE_RATIO,
        )

        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_depth_limit=8,
        )

        # 验证方法存在且可调用
        assert hasattr(splitter, '_get_dynamic_k_bounds')
        assert callable(splitter._get_dynamic_k_bounds)

        # 验证返回值
        K_min, K_max = splitter._get_dynamic_k_bounds(5461)
        assert isinstance(K_min, int)
        assert isinstance(K_max, int)
        assert K_max >= K_min

    def test_splitter_dynamic_bounds_different_depths(self):
        """验证不同深度使用不同的动态边界"""
        from vit_pytorch.gumbel_topk_splitter import GumbelTopKSplitter

        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_depth_limit=8,
        )

        # 测试不同深度的边界
        # 注意: 实际实现使用 K_COVERAGE_MIN=0.01, K_COVERAGE_BASE=0.05
        depths_and_expected = [
            (341, (8, 18)),     # D=4: ceil(0.01×341)=4→8(硬下限), ceil(0.05×341)=18
            (5461, (55, 274)),  # D=6: ceil(0.01×5461)=55, ceil(0.05×5461)=274
            (21845, (219, 1093)), # D=7: ceil(0.01×21845)=219, ceil(0.05×21845)=1093
        ]

        for N, (exp_min, exp_max) in depths_and_expected:
            K_min, K_max = splitter._get_dynamic_k_bounds(N)
            assert K_min == exp_min, f"N={N}: K_min={K_min} != {exp_min}"
            assert K_max == exp_max, f"N={N}: K_max={K_max} != {exp_max}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
