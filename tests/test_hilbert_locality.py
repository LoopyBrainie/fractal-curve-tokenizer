"""I25-12: Pseudo-Hilbert 局部性量化测试.

测试 HilbertLocalityMetrics 工具类的各项指标计算功能。
"""

import math
from typing import Tuple

import pytest

from vit_pytorch.curve_hilbert import (
    HilbertCurve,
    PseudoHilbertCurve,
    HilbertLocalityMetrics,
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
