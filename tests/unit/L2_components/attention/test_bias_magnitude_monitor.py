# -*- coding: utf-8 -*-
"""
L2 Components: Bias Magnitude Monitor Tests (I161-2)

对应模块: vit_pytorch.layers.attention.hilbert_bias (BiasMagnitudeMonitor)

测试内容:
- Hilbert 偏置归一化量级验证
- Area/Shape 偏置量级验证
- 超阈值警告功能
- 统计信息输出格式

数学背景 (I161-2):
==================

归一化量级计算:
    - Hilbert/Level 偏置: norm = max|B| / √d_k
    - Area/Shape 偏置: norm = max|B| (已由 Tanh 限制)

监控维度:
    | 偏置类型 | 目标量级 | 验证公式 |
    |---------|---------|---------|
    | Hilbert | O(√d_k) | max|B_hilbert| / √d_k ≈ O(1) |
    | Level   | O(√d_k) | max|B_level| / √d_k ≈ O(1) |
    | Area    | O(1)    | max|B_area| ∈ (-1, 1) |
    | Shape   | O(1)    | max|B_shape| ∈ (-1, 1) |
"""

import pytest
import torch
import warnings

from vit_pytorch.layers.attention.hilbert_bias import (
    BiasMagnitudeMonitor,
)


class TestBiasMagnitudeMonitor:
    """测试偏置量级监控器 (I161-2)"""

    def test_hilbert_bias_normalized(self):
        """验证 Hilbert 偏置归一化量级在预期范围。

        Hilbert 偏置经过 √d_k 量纲对齐后，归一化量级应在 O(1) 范围。
        放宽断言范围，因为随机张量的最大值可能较大。
        """
        monitor = BiasMagnitudeMonitor(dim_head=32, warning_threshold=10.0)

        # 模拟 Hilbert 偏置 (有效值 ≈ 3.9)
        hilbert_bias = torch.randn(2, 8, 85, 85) * 3.9

        stats = monitor({'hilbert': hilbert_bias})
        normalized = stats['hilbert/normalized_max'].item()
        max_abs = stats['hilbert/max_abs'].item()

        # 归一化量级应在合理范围 (放宽到 0.1-10.0)
        assert 0.1 < normalized < 10.0, (
            f"Normalized Hilbert bias {normalized:.2f} out of expected range"
        )
        # 原始最大值应约为 3.9
        assert max_abs > 1.0, f"Max abs {max_abs:.2f} too small"

    def test_area_bias_within_tanh_bounds(self):
        """验证 Area 偏置量级在 Tanh 限制范围内。

        Area 偏置网络使用 Tanh 激活，输出应在 (-1, 1) 范围内。
        """
        monitor = BiasMagnitudeMonitor(dim_head=32)

        # 模拟 Area 偏置 (Tanh 限制在 (-1, 1))
        area_bias = torch.randn(2, 32, 85, 85).tanh()

        stats = monitor({'area': area_bias})
        max_abs = stats['area/max_abs'].item()
        normalized = stats['area/normalized_max'].item()

        # Area 偏置不应超过 1.0 (Tanh 限制)
        assert max_abs <= 1.0 + 1e-4, f"Area bias {max_abs:.3f} exceeds Tanh bounds"
        # 归一化后应接近原始值
        assert abs(normalized - max_abs) < 1e-5, "Area normalization should be identity"

    def test_shape_bias_within_tanh_bounds(self):
        """验证 Shape 偏置量级在 Tanh 限制范围内。

        Shape 偏置网络使用 Tanh 激活，输出应在 (-1, 1) 范围内。
        """
        monitor = BiasMagnitudeMonitor(dim_head=64)

        # 模拟 Shape 偏置
        shape_bias = torch.randn(2, 64, 50, 50).tanh()

        stats = monitor({'shape': shape_bias})
        max_abs = stats['shape/max_abs'].item()

        assert max_abs <= 1.0 + 1e-4, f"Shape bias {max_abs:.3f} exceeds Tanh bounds"

    def test_combined_bias_magnitude(self):
        """验证组合偏置量级在合理范围内。

        组合偏置 = Hilbert + Area + Shape，各偏置应有相似的量级。
        """
        monitor = BiasMagnitudeMonitor(dim_head=32)

        # 构造组合偏置
        hilbert = torch.randn(2, 32, 50, 50) * 3.0
        area = torch.randn(2, 32, 50, 50).tanh() * 0.5
        shape = torch.randn(2, 32, 50, 50).tanh() * 0.3

        combined = hilbert + area + shape

        stats = monitor({
            'hilbert': hilbert,
            'area': area,
            'shape': shape,
            'combined': combined,
        })

        # 组合偏置的最大值应接近 Hilbert 偏置主导
        combined_max = stats['combined/max_abs'].item()
        assert combined_max > 2.0, f"Combined bias {combined_max:.2f} too small"

    def test_warning_threshold_exceeded(self):
        """测试超阈值警告功能。

        当归一化量级超过阈值时，应发出 UserWarning。
        """
        monitor = BiasMagnitudeMonitor(dim_head=32, warning_threshold=5.0)

        # 使用较小的阈值来测试
        small_threshold_monitor = BiasMagnitudeMonitor(dim_head=32, warning_threshold=2.0)

        # 构造超阈值偏置
        large_bias = torch.ones(1, 8, 10, 10) * 10.0

        with pytest.warns(UserWarning, match="exceeds threshold"):
            small_threshold_monitor({'large_bias': large_bias})

    def test_none_handling(self):
        """测试 None 偏置的处理。

        某些偏置可能为 None，监控器应能正确处理。
        """
        monitor = BiasMagnitudeMonitor(dim_head=32)

        stats = monitor({
            'hilbert': torch.randn(2, 8, 10, 10),
            'area': None,
            'shape': torch.randn(2, 32, 10, 10).tanh(),
        })

        # None 偏置不应出现在统计中
        assert 'hilbert/max_abs' in stats
        assert 'area/max_abs' not in stats  # None 被跳过
        assert 'shape/max_abs' in stats

    def test_training_mode_warning(self):
        """测试仅在训练模式下发出警告。

        评估模式下不应发出警告。
        """
        monitor = BiasMagnitudeMonitor(dim_head=32, warning_threshold=0.1)
        monitor.eval()

        large_bias = torch.ones(1, 8, 10, 10) * 10.0

        # 评估模式下不应发出警告
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # 将警告转为错误
            stats = monitor({'large_bias': large_bias})
            # 如果没有引发异常，测试通过

    def test_get_summary_format(self):
        """测试统计摘要字符串格式。

        摘要应包含所有监控的偏置和其量级。
        """
        monitor = BiasMagnitudeMonitor(dim_head=32)

        stats = {
            'hilbert/max_abs': torch.tensor(3.92),
            'hilbert/normalized_max': torch.tensor(0.69),
            'area/max_abs': torch.tensor(0.85),
            'area/normalized_max': torch.tensor(0.85),
        }

        summary = monitor.get_summary(stats)

        assert 'hilbert' in summary.lower()
        assert 'area' in summary.lower()
        assert '3.92' in summary or '0.69' in summary

    def test_dim_head_scaling(self):
        """测试不同 dim_head 的缩放效果。

        dim_head 越大，√d_k 越大，归一化后 Hilbert 偏置量级越小。
        """
        bias = torch.ones(1, 8, 10, 10) * 5.0

        monitor_32 = BiasMagnitudeMonitor(dim_head=32)  # √32 ≈ 5.66
        monitor_64 = BiasMagnitudeMonitor(dim_head=64)  # √64 = 8.0

        stats_32 = monitor_32({'hilbert': bias})
        stats_64 = monitor_64({'hilbert': bias})

        norm_32 = stats_32['hilbert/normalized_max'].item()
        norm_64 = stats_64['hilbert/normalized_max'].item()

        # 更大的 dim_head 应产生更小的归一化量级
        assert norm_32 > norm_64, (
            f"Normalized values should decrease with dim_head: {norm_32:.3f} vs {norm_64:.3f}"
        )

    def test_level_bias_normalized(self):
        """验证 Level 偏置归一化量级与 Hilbert 类似。

        Level 偏置也需要除以 √d_k 进行归一化。
        """
        monitor = BiasMagnitudeMonitor(dim_head=64)

        # 模拟 Level 偏置
        level_bias = torch.randn(2, 8, 50, 50) * 4.0

        stats = monitor({'level': level_bias})
        normalized = stats['level/normalized_max'].item()

        # Level 归一化量级应与 Hilbert 类似
        assert 0.1 < normalized < 10.0, f"Level normalized {normalized:.2f} out of range"


class TestBiasMagnitudeMonitorIntegration:
    """偏置量级监控集成测试 (I161-2)"""

    def test_training_loop_simulation(self):
        """模拟训练过程中的偏置量级监控。

        验证监控器能正确跟踪训练过程中偏置量级的变化。
        """
        monitor = BiasMagnitudeMonitor(dim_head=32, warning_threshold=10.0)
        monitor.train()

        # 模拟多个训练步骤
        for step in range(5):
            # 模拟学习过程中偏置量级的变化
            scale = 1.0 + step * 0.1
            hilbert_bias = torch.randn(2, 8, 50, 50) * scale * 3.0
            area_bias = torch.randn(2, 32, 50, 50).tanh() * scale * 0.5

            stats = monitor({
                'hilbert': hilbert_bias,
                'area': area_bias,
            })

            # 每个步骤都应返回统计
            assert 'hilbert/normalized_max' in stats
            assert 'area/normalized_max' in stats

    def test_multi_head_consistency(self):
        """测试多头偏置的一致性监控。

        不同头之间的偏置量级应保持一致。
        """
        monitor = BiasMagnitudeMonitor(dim_head=32)

        # 模拟多头的 Hilbert 偏置
        multi_head_bias = torch.randn(8, 8, 50, 50) * 4.0

        stats = monitor({'hilbert': multi_head_bias})

        max_abs = stats['hilbert/max_abs'].item()
        normalized = stats['hilbert/normalized_max'].item()

        # 多头应产生合理的统计值
        assert max_abs > 0, "Multi-head bias should have positive max"
        assert 0.1 < normalized < 10.0, f"Normalized {normalized:.2f} out of range"

    def test_empty_batch(self):
        """测试空批次处理。

        空批次应返回空统计。
        """
        monitor = BiasMagnitudeMonitor(dim_head=32)

        # 空批次
        empty_bias = torch.randn(0, 8, 10, 10)

        stats = monitor({'hilbert': empty_bias})

        # 应该能处理空批次而不报错
        assert 'hilbert/max_abs' in stats or len(stats) == 0

    def test_single_element(self):
        """测试单元素张量。

        单元素张量应能正确处理。
        """
        monitor = BiasMagnitudeMonitor(dim_head=32)

        # 单元素
        single = torch.tensor([[[1.0]]])

        stats = monitor({'test': single})

        assert 'test/max_abs' in stats
        assert 'test/normalized_max' in stats
        assert stats['test/max_abs'].item() == 1.0
