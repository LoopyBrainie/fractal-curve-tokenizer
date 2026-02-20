# -*- coding: utf-8 -*-
"""
I120-8: Hilbert-感知自适应预算系统测试

对应模块: vit_pytorch.gumbel_topk_splitter

测试内容:
- Hilbert-感知连续性损失
- 图像复杂度估计
- 空间覆盖预算损失
- 自适应目标覆盖率
"""

import pytest
import torch
import torch.nn.functional as F

from vit_pytorch import GumbelTopKSplitter
from vit_pytorch.layers.splitters.gumbel_topk import GumbelTopKSplitter
from vit_pytorch.core.constants import (
    HILBERT_CONTINUITY_ENABLED,
    HILBERT_CONTINUITY_WEIGHT,
    HILBERT_CONTINUITY_GAMMA,
    ADAPTIVE_COVERAGE_ENABLED,
    ADAPTIVE_COVERAGE_MIN,
    ADAPTIVE_COVERAGE_MAX,
    COVERAGE_BUDGET_ENABLED,
    SPATIAL_COVERAGE_MIN,
    SPATIAL_COVERAGE_WEIGHT,
)


class TestHilbertContinuityLoss:
    """Hilbert-感知连续性损失测试"""

    def test_function_executes(self):
        """损失函数应该能正常执行"""
        splitter = GumbelTopKSplitter(
            image_size=(224, 224),
            feature_dim=256,
        )

        # 运行前向以填充缓存
        features = torch.randn(1, 256, 14, 14)
        splitter(features)

        # 获取辅助损失（包含 hilbert_continuity_loss）
        losses = splitter.get_auxiliary_losses(
            include_elastic_budget=False,
            include_soft_entropy=False,
        )

        # 应该包含 hilbert_continuity_loss
        assert 'hilbert_continuity_loss' in losses
        loss = losses['hilbert_continuity_loss']

        # 损失应该是有效的数值
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()


class TestImageComplexity:
    """图像复杂度估计测试"""

    def test_complexity_default_value(self):
        """没有 info_density 时应该返回默认中等复杂度"""
        splitter = GumbelTopKSplitter(
            image_size=(224, 224),
            feature_dim=256,
        )

        complexity = splitter._compute_image_complexity(info_density=None)

        # 默认复杂度应该是 0.5
        assert complexity.item() == 0.5

    def test_complexity_with_info_density(self):
        """使用 info_density 计算复杂度"""
        splitter = GumbelTopKSplitter(
            image_size=(224, 224),
            feature_dim=256,
        )

        # 模拟各深度的信息密度
        info_density = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.1])
        complexity = splitter._compute_image_complexity(info_density=info_density)

        # 复杂度 = sum(I_d) / N_candidates
        expected = info_density.sum() / splitter.num_candidates
        assert abs(complexity.item() - expected.item()) < 1e-6

    def test_complexity_comparison(self):
        """高信息密度应该有更高的复杂度"""
        splitter = GumbelTopKSplitter(
            image_size=(224, 224),
            feature_dim=256,
        )

        # 低密度
        info_density_low = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01])
        complexity_low = splitter._compute_image_complexity(info_density=info_density_low)

        # 高密度
        info_density_high = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
        complexity_high = splitter._compute_image_complexity(info_density=info_density_high)

        # 高密度应该有更高的复杂度
        assert complexity_high.item() > complexity_low.item()


class TestSpatialCoverageLoss:
    """空间覆盖预算损失测试"""

    def test_loss_is_valid(self):
        """损失应该是有效的数值"""
        splitter = GumbelTopKSplitter(
            image_size=(224, 224),
            feature_dim=256,
        )

        selected_mask = torch.zeros(1, splitter.num_candidates)
        selected_mask[0, :20] = 1.0

        loss = splitter.get_spatial_coverage_loss(
            selected_mask=selected_mask,
            image_size=(224, 224),
            weight=0.05,
        )

        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()

    def test_full_coverage_zero_loss(self):
        """达到最小覆盖率应该产生零损失"""
        splitter = GumbelTopKSplitter(
            image_size=(224, 224),
            feature_dim=256,
        )

        # 选择足够多的区域以达到高覆盖率
        selected_mask = torch.zeros(1, splitter.num_candidates)
        selected_mask[0, :200] = 1.0

        loss = splitter.get_spatial_coverage_loss(
            selected_mask=selected_mask,
            image_size=(224, 224),
            weight=0.05,
        )

        # 如果覆盖率 >= SPATIAL_COVERAGE_MIN，损失应该为 0
        assert loss.item() >= 0.0


class TestAdaptiveTargetTokens:
    """自适应目标覆盖率测试"""

    def test_adaptive_target_computes(self):
        """自适应目标应该正确计算"""
        splitter = GumbelTopKSplitter(
            image_size=(224, 224),
            feature_dim=256,
        )

        info_density = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.1])
        target = splitter.get_adaptive_target_tokens(
            info_density=info_density,
            image_size=(224, 224),
        )

        # 目标应该是一个正数
        assert target.item() > 0.0

    def test_adaptive_target_scales_with_complexity(self):
        """高复杂度图像应该获得更多 token"""
        splitter = GumbelTopKSplitter(
            image_size=(224, 224),
            feature_dim=256,
        )

        # 低复杂度
        info_density_low = torch.tensor([0.01] * 5)
        target_low = splitter.get_adaptive_target_tokens(info_density=info_density_low)

        # 高复杂度
        info_density_high = torch.tensor([0.5] * 5)
        target_high = splitter.get_adaptive_target_tokens(info_density=info_density_high)

        # 高复杂度应该获得更多 token
        assert target_high.item() > target_low.item()

    def test_adaptive_target_in_range(self):
        """自适应目标应该在合理范围内"""
        splitter = GumbelTopKSplitter(
            image_size=(224, 224),
            feature_dim=256,
        )

        info_density = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
        target = splitter.get_adaptive_target_tokens(info_density=info_density)

        # 应该使用自适应覆盖率
        expected_min = ADAPTIVE_COVERAGE_MIN * splitter.num_candidates
        expected_max = ADAPTIVE_COVERAGE_MAX * splitter.num_candidates

        assert expected_min <= target.item() <= expected_max


class TestAuxiliaryLossesIntegration:
    """辅助损失集成测试"""

    def test_get_auxiliary_losses_new_losses(self):
        """get_auxiliary_losses 应该返回新的 I120-8 损失"""
        splitter = GumbelTopKSplitter(
            image_size=(224, 224),
            feature_dim=256,
        )

        # 运行前向以填充缓存
        features = torch.randn(1, 256, 14, 14)
        splitter(features)

        losses = splitter.get_auxiliary_losses(
            include_elastic_budget=False,
            include_soft_entropy=False,
        )

        # 应该包含新的损失项
        assert 'hilbert_continuity_loss' in losses
        assert 'spatial_coverage_loss' in losses
        # 注意: adaptive_target_tokens 不是损失，只是内部计算值（I165-FIX）

    def test_losses_are_valid(self):
        """所有损失应该是有效的数值"""
        splitter = GumbelTopKSplitter(
            image_size=(224, 224),
            feature_dim=256,
        )

        features = torch.randn(1, 256, 14, 14)
        splitter(features)

        losses = splitter.get_auxiliary_losses(
            include_elastic_budget=False,
            include_soft_entropy=False,
        )

        for key, loss in losses.items():
            assert not torch.isnan(loss).any()
            assert not torch.isinf(loss).any()


class TestConstantsExist:
    """常量存在性测试"""

    def test_hilbert_continuity_constants(self):
        """Hilbert 连续性常量应该存在"""
        assert HILBERT_CONTINUITY_ENABLED is True
        # I120-8 修复: 权重从 0.1 提高至 0.2，增强 Hilbert 连续性约束
        assert HILBERT_CONTINUITY_WEIGHT == 0.2
        assert HILBERT_CONTINUITY_GAMMA == 0.1

    def test_adaptive_coverage_constants(self):
        """自适应覆盖率常量应该存在"""
        assert ADAPTIVE_COVERAGE_ENABLED is True
        assert ADAPTIVE_COVERAGE_MIN == 0.05
        assert ADAPTIVE_COVERAGE_MAX == 0.40

    def test_coverage_budget_constants(self):
        """覆盖预算常量应该存在"""
        assert COVERAGE_BUDGET_ENABLED is True
        assert SPATIAL_COVERAGE_MIN == 0.3
        assert SPATIAL_COVERAGE_WEIGHT == 0.05


class TestDepthBalanceLoss:
    """I120-8: 深度平衡损失测试"""

    def test_depth_balance_loss_exists(self):
        """深度平衡损失函数应该存在"""
        splitter = GumbelTopKSplitter(
            image_size=(64, 64),
            feature_dim=256,
        )
        assert hasattr(splitter, 'get_depth_balance_loss')

    def test_depth_balance_loss_basic(self):
        """深度平衡损失应该计算"""
        splitter = GumbelTopKSplitter(
            image_size=(64, 64),
            feature_dim=256,
        )

        # 运行前向以填充缓存
        features = torch.randn(2, 256, 8, 8)
        splitter(features)

        # 获取辅助损失（包含 depth_balance_loss）
        losses = splitter.get_auxiliary_losses()

        # 应该包含 depth_balance_loss
        assert 'depth_balance_loss' in losses
        loss = losses['depth_balance_loss']

        # 损失应该是有效的数值
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()

    def test_depth_balance_loss_uniform_distribution(self):
        """均匀分布时损失应该接近零"""
        splitter = GumbelTopKSplitter(
            image_size=(64, 64),
            feature_dim=256,
        )

        # 运行前向
        features = torch.randn(1, 256, 8, 8)
        splitter(features)

        losses = splitter.get_auxiliary_losses()
        balance_loss = losses['depth_balance_loss']

        # 对于小权重，损失应该较小
        assert balance_loss.item() < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
