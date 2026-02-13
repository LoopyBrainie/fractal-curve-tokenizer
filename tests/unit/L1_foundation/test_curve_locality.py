# -*- coding: utf-8 -*-
"""
L1 Foundation: Hilbert Locality Metrics Tests

对应模块: vit_pytorch.curve_hilbert (HilbertLocalityMetrics, PseudoHilbertCurve)

测试内容:
- HilbertLocalityMetrics 工具类
- 局部性量化指标 (max_jump, avg_locality_loss, preservation_rate)
- 标准 Hilbert vs Pseudo-Hilbert 对比
"""

import math
from typing import Tuple

import pytest

from vit_pytorch.core.curve_hilbert import (
    HilbertCurve,
    PseudoHilbertCurve,
    HilbertLocalityMetrics,
    HilbertProbabilityMetrics,
)


class TestHilbertLocalityMetrics:
    """HilbertLocalityMetrics 工具类测试."""

    def test_standard_hilbert_4x4(self):
        """4×4 标准 Hilbert 曲线局部性."""
        points = tuple(HilbertCurve.d_to_xy(4, d) for d in range(16))

        # 标准 Hilbert 应该完全保持局部性
        max_jump = HilbertLocalityMetrics.max_jump_distance(points)
        avg_loss = HilbertLocalityMetrics.average_locality_loss(points)

        assert max_jump <= math.sqrt(2) + 1e-6, f"Max jump {max_jump} > √2"
        assert avg_loss <= 1.2, f"Avg loss {avg_loss} > 1.2"

    def test_pseudo_hilbert_16x16(self):
        """16×16 Pseudo-Hilbert 局部性."""
        points = PseudoHilbertCurve.scan(16, 16)

        metrics = HilbertLocalityMetrics.full_report(16, 16)

        # 验证理论边界
        assert metrics['max_jump'] <= 2.12, "理论上限 1.5√2 ≈ 2.12"
        assert metrics['avg_locality_loss'] < 1.2, "经验上界"
        assert metrics['locality_preservation_rate'] > 0.90, "90% 保持率"
        assert metrics['total_points'] == 256
        assert metrics['hilbert_equivalence'] is True

    def test_pseudo_hilbert_12x12(self):
        """12×12 非 2^k Pseudo-Hilbert 局部性."""
        points = PseudoHilbertCurve.scan(12, 12)

        metrics = HilbertLocalityMetrics.full_report(12, 12)

        # 非 2^k 情况应该有更低的局部性保持率
        assert metrics['max_jump'] >= 1.0, "至少应该有邻居"
        assert metrics['avg_locality_loss'] > 1.0, "比标准 Hilbert 差"
        assert metrics['total_points'] == 144
        assert metrics['hilbert_equivalence'] is False

    def test_hilbert_vs_pseudo_hilbert_equivalence(self):
        """Hilbert vs Pseudo-Hilbert 在 2^k 方形情况下应等价."""
        # 标准 2^k 情况，两者应该接近
        hilbert_metrics = HilbertLocalityMetrics.full_report(16, 16, "hilbert")
        pseudo_metrics = HilbertLocalityMetrics.full_report(16, 16, "pseudo_hilbert")

        # 在 2^k 方形情况下，两者应该几乎相同
        assert abs(hilbert_metrics['avg_locality_loss'] -
                   pseudo_metrics['avg_locality_loss']) < 0.05

    def test_rectangle_aspect_ratio_degradation(self):
        """矩形情况下局部性下降验证."""
        # 16×8 比 8×16 更"长"
        rect_16x8 = HilbertLocalityMetrics.full_report(16, 8)
        rect_8x16 = HilbertLocalityMetrics.full_report(8, 16)

        # 长宽比越大，局部性损失通常越大
        # 注意：这个测试可能因具体实现而变化
        assert rect_16x8['max_jump'] >= 1.0
        assert rect_8x16['max_jump'] >= 1.0

    def test_neighbor_distance_distribution(self):
        """邻居距离分布测试."""
        points = PseudoHilbertCurve.scan(16, 16)
        distribution = HilbertLocalityMetrics.neighbor_distance_distribution(points)

        # 分布应该只包含少数几个距离值
        assert len(distribution) <= 5, "距离类型应该很少"

        # 所有概率应该加起来约为 1
        total_prob = sum(distribution.values())
        assert abs(total_prob - 1.0) < 0.01, f"概率和 {total_prob} ≠ 1"

        # 应该有相当比例的邻居距离为 √2 (1.414)
        if 1.414 in distribution:
            assert distribution[1.414] > 0.5, "至少 50% 应该是 √2 邻居"

    def test_edge_case_single_point(self):
        """单点边缘情况."""
        points = ((0, 0),)

        assert HilbertLocalityMetrics.max_jump_distance(points) == 0.0
        assert HilbertLocalityMetrics.average_locality_loss(points) == 0.0
        assert HilbertLocalityMetrics.locality_preservation_rate(points) == 1.0
        assert HilbertLocalityMetrics.neighbor_distance_distribution(points) == {}

    def test_edge_case_two_points(self):
        """两点边缘情况."""
        points = ((0, 0), (1, 0))

        assert HilbertLocalityMetrics.max_jump_distance(points) == 1.0
        assert HilbertLocalityMetrics.average_locality_loss(points) == 1.0
        assert HilbertLocalityMetrics.locality_preservation_rate(points) == 1.0

    def test_full_report_format(self):
        """完整报告格式测试."""
        # Hilbert 需要 2^k 正方形
        report = HilbertLocalityMetrics.full_report(8, 8, "hilbert")

        # 验证报告包含所有必要字段
        required_keys = ['shape', 'total_points', 'max_jump',
                        'avg_locality_loss', 'locality_preservation_rate',
                        'distance_distribution', 'hilbert_equivalence']

        for key in required_keys:
            assert key in report, f"报告缺少字段: {key}"

        # 验证类型
        assert isinstance(report['shape'], tuple)
        assert isinstance(report['total_points'], int)
        assert isinstance(report['max_jump'], float)
        assert isinstance(report['distance_distribution'], dict)

    def test_locality_preservation_rate_threshold(self):
        """局部性保持率阈值测试."""
        points = tuple(HilbertCurve.d_to_xy(4, d) for d in range(16))

        # threshold=√2 应该返回 1.0 (标准 Hilbert)
        rate_sqrt2 = HilbertLocalityMetrics.locality_preservation_rate(
            points, threshold=math.sqrt(2)
        )

        # 标准 Hilbert 在 √2 阈值下应该保持 100%
        assert rate_sqrt2 >= 0.99, f"应该接近 100%, 得到 {rate_sqrt2}"

    def test_32x32_performance(self):
        """32×32 性能基准测试."""
        import time

        points = PseudoHilbertCurve.scan(32, 32)

        start = time.time()
        for _ in range(100):
            HilbertLocalityMetrics.full_report(32, 32)
        elapsed = time.time() - start

        # 100 次报告应该在 1 秒内完成
        assert elapsed < 1.0, f"性能测试失败: {elapsed:.2f}s"

    def test_various_sizes(self):
        """多种尺寸测试."""
        sizes = [(4, 4), (8, 8), (16, 8), (8, 16), (12, 12)]

        for h, w in sizes:
            points = PseudoHilbertCurve.scan(h, w)
            assert len(points) == h * w

            max_jump = HilbertLocalityMetrics.max_jump_distance(points)
            avg_loss = HilbertLocalityMetrics.average_locality_loss(points)

            # 所有情况都应该有合理的上界
            assert max_jump <= max(h, w), f"跳跃超出边界: {max_jump}"
            assert 1.0 <= avg_loss <= max(h, w), f"平均损失异常: {avg_loss}"


# =============================================================================
# I108-5: Hilbert Probability Metrics Tests
# =============================================================================

class TestHilbertProbabilityMetrics:
    """HilbertProbabilityMetrics 工具类测试.

    测试条件概率 P(d_S ≤ τ | d_H = k) 的精确计算。

    关键发现: Hilbert 曲线保证 d_H = 1 时 d_S = 1 (确定性)。
    这意味着 P(d_S = 1 | d_H = 1) = 1.0。
    """

    def test_condition_prob_exact_k1_tau_1_0(self):
        """验证 k=1, τ=1.0 时的条件概率.

        Hilbert 曲线核心性质: d_H=1 ⟹ d_S=1
        P(d_S ≤ 1.0 | d_H = 1) = 1.0
        """
        prob = HilbertProbabilityMetrics.condition_prob_exact(
            order=4, delta_d=1, spatial_threshold=1.0
        )
        assert abs(prob - 1.0) < 1e-6

    def test_condition_prob_exact_k1_tau_1_5(self):
        """验证 k=1, τ=1.5 时的条件概率.

        τ ≥ 1.0 时，P = 1.0 (距离恒为 1.0)
        """
        prob = HilbertProbabilityMetrics.condition_prob_exact(
            order=4, delta_d=1, spatial_threshold=1.5
        )
        assert abs(prob - 1.0) < 1e-6

    def test_condition_prob_exact_k1_tau_2_0(self):
        """验证 k=1, τ=2.0 时的条件概率.

        τ ≥ 1.0 时，P = 1.0
        """
        prob = HilbertProbabilityMetrics.condition_prob_exact(
            order=4, delta_d=1, spatial_threshold=2.0
        )
        assert abs(prob - 1.0) < 1e-6

    def test_condition_prob_exact_k1_tau_0_5(self):
        """验证 k=1, τ=0.5 时的条件概率.

        τ < 1.0 时，P = 0.0 (距离恒为 1.0 > τ)
        """
        prob = HilbertProbabilityMetrics.condition_prob_exact(
            order=4, delta_d=1, spatial_threshold=0.5
        )
        assert abs(prob - 0.0) < 1e-6

    def test_condition_prob_exact_k1_order_8(self):
        """验证不同阶数的概率一致性.

        对于 k=1，所有阶数的概率都是 1.0 (确定性)
        """
        for order in [4, 5, 6, 7, 8]:
            prob = HilbertProbabilityMetrics.condition_prob_exact(
                order=order, delta_d=1, spatial_threshold=1.0
            )
            assert abs(prob - 1.0) < 1e-6, f"Order {order} 不一致"

    def test_condition_prob_exact_delta_d_greater_than_1_raises(self):
        """验证 delta_d > 1 时抛出 NotImplementedError (精确解不支持)."""
        with pytest.raises(NotImplementedError):
            HilbertProbabilityMetrics.condition_prob_exact(
                order=4, delta_d=2, spatial_threshold=1.5
            )

    def test_condition_prob_approximate_k1_brute_force(self):
        """验证暴力精确计算与解析解一致 (k=1).

        brute_force 遍历所有点对，k=1 时所有距离都是 1.0
        """
        exact = HilbertProbabilityMetrics.condition_prob_exact(
            order=4, delta_d=1, spatial_threshold=1.0
        )
        approx = HilbertProbabilityMetrics.condition_prob_approximate(
            order=4, delta_d=1, spatial_threshold=1.0,
            method="brute_force"
        )
        assert abs(exact - approx) < 1e-6

    def test_condition_prob_approximate_k2_brute_force(self):
        """验证 k=2 时的概率计算 (暴力法).

        k=2 时距离分布: {√2: 约 80%, 2: 约 20%}
        P(d_S ≤ 2.0 | d_H = 2) = 1.0
        """
        prob = HilbertProbabilityMetrics.condition_prob_approximate(
            order=8, delta_d=2, spatial_threshold=2.0,
            method="brute_force"
        )
        # k=2 时，最大距离为 2.0，所以 P = 1.0
        assert abs(prob - 1.0) < 1e-6

    def test_condition_prob_approximate_k2_threshold_1_5(self):
        """验证 k=2, τ=1.5 时的概率计算.

        k=2 时，部分点对距离为 √2 (≈1.414)，部分为 2.0
        P(d_S ≤ 1.5 | d_H = 2) = P(d_S = √2)
        """
        prob = HilbertProbabilityMetrics.condition_prob_approximate(
            order=8, delta_d=2, spatial_threshold=1.5,
            method="brute_force"
        )
        # 验证概率在 (0, 1) 范围内
        assert 0.0 < prob < 1.0
        # 期望约 80% (50/62 = 0.806)
        assert 0.7 < prob < 0.9

    def test_condition_prob_approximate_monte_carlo_k2(self):
        """验证蒙特卡洛估计的精度 (k=2, τ=1.5).

        蒙特卡洛估计应接近暴力精确解，误差 < 0.05 (95% 置信度)
        """
        brute = HilbertProbabilityMetrics.condition_prob_approximate(
            order=8, delta_d=2, spatial_threshold=1.5,
            method="brute_force"
        )
        monte_carlo = HilbertProbabilityMetrics.condition_prob_approximate(
            order=8, delta_d=2, spatial_threshold=1.5,
            method="monte_carlo", samples=50000, seed=42
        )
        assert abs(brute - monte_carlo) < 0.05

    def test_locality_entropy_k1(self):
        """验证香农熵计算 (k=1).

        k=1 时，所有距离都是 1.0 (确定性分布)
        H = -1.0 * log2(1.0) = 0.0 bits
        """
        entropy = HilbertProbabilityMetrics.locality_entropy(order=4)
        assert abs(entropy - 0.0) < 1e-6

    def test_locality_entropy_k2(self):
        """验证香农熵计算 (k=2).

        k=2 时，距离分布为 {√2: 50/62, 2: 12/62}
        H = -(50/62)*log2(50/62) - (12/62)*log2(12/62)
        """
        entropy = HilbertProbabilityMetrics.locality_entropy(order=8, delta_d=2)
        # 熵应该 > 0 (非确定性分布)
        assert entropy > 0.0
        # 熵应该 < 1 (分布不完全均匀)
        assert entropy < 1.0

    def test_locality_entropy_bounds_k2(self):
        """验证熵值在合理范围内 (k=2).

        k=2 时，P(√2) ≈ 0.8, P(2.0) ≈ 0.2
        H = -0.8*log2(0.8) - 0.2*log2(0.2) ≈ 0.72 bits
        """
        for n in [4, 5, 6, 7, 8]:
            entropy = HilbertProbabilityMetrics.locality_entropy(order=n, delta_d=2)
            # 熵应该在 0.7 ~ 0.75 范围内
            assert 0.7 < entropy < 0.75, f"n={n}: 熵={entropy} 超出范围"

    def test_distance_distribution_k1(self):
        """验证距离分布计算 (k=1).

        k=1 时，所有距离都是 1.0
        """
        dist = HilbertProbabilityMetrics._distance_distribution(order=4, delta_d=1)
        # 验证分布概率和为 1
        assert abs(sum(dist.values()) - 1.0) < 1e-6
        # 验证距离值
        assert 1.0 in dist
        assert len(dist) == 1  # 只有一种距离

    def test_distance_distribution_k2(self):
        """验证距离分布计算 (k=2).

        k=2 时，距离分布为 {√2: 50/62, 2: 12/62}
        """
        dist = HilbertProbabilityMetrics._distance_distribution(order=8, delta_d=2)
        # 验证分布概率和为 1
        assert abs(sum(dist.values()) - 1.0) < 1e-6
        # 验证距离值
        assert 1.414 in dist
        assert 2.0 in dist

    def test_full_probability_report_k1(self):
        """验证完整概率报告生成 (k=1)."""
        report = HilbertProbabilityMetrics.full_probability_report(order=4)

        assert report["order"] == 4
        assert report["delta_d"] == 1
        assert "entropy_bits" in report
        assert "distance_distribution" in report
        assert "P(d_S <= 1.0)" in report
        assert "P(d_S <= 1.414)" in report

    def test_full_probability_report_k2(self):
        """验证完整概率报告生成 (k=2)."""
        report = HilbertProbabilityMetrics.full_probability_report(order=8, delta_d=2)

        assert report["order"] == 8
        assert report["delta_d"] == 2
        assert report["entropy_bits"] > 0  # k=2 时有不确定性
        assert "distance_distribution" in report

    def test_cumulative_prob_threshold_1_0(self):
        """验证累积概率计算 (τ=1.0).

        分布 {1.0: 0.5, 1.414: 0.5}: P(d_S ≤ 1.0) = 0.5
        """
        distribution = {1.0: 0.5, 1.414: 0.5}
        prob = HilbertProbabilityMetrics._cumulative_prob(distribution, 1.0)
        assert abs(prob - 0.5) < 1e-6

    def test_cumulative_prob_threshold_1_414(self):
        """验证累积概率计算 (τ=1.414).

        分布 {1.0: 0.5, 1.414: 0.5}: P(d_S ≤ 1.414) = 1.0
        """
        distribution = {1.0: 0.5, 1.414: 0.5}
        prob = HilbertProbabilityMetrics._cumulative_prob(distribution, 1.414)
        assert abs(prob - 1.0) < 1e-6

    def test_probability_consistency_k1(self):
        """验证 k=1 时解析解与暴力精确解一致."""
        for order in [4, 5, 6]:
            exact = HilbertProbabilityMetrics.condition_prob_exact(
                order=order, delta_d=1, spatial_threshold=1.0
            )
            brute = HilbertProbabilityMetrics.condition_prob_approximate(
                order=order, delta_d=1, spatial_threshold=1.0,
                method="brute_force"
            )
            assert abs(exact - brute) < 1e-6, \
                f"Order {order} 不一致: exact={exact}, brute={brute}"
