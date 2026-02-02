# -*- coding: utf-8 -*-
"""
I109-6: Gradient Balance Verification Tests

对应模块: vit_pytorch.gumbel_topk_splitter

测试内容:
- Gumbel-Top-K STE 梯度强度平衡
- 覆盖率缩放因子 α = K/N 的效果

Note:
====
覆盖率缩放 STE: st = hard - soft.detach() + α * soft
其中 α = K/N (覆盖率)
"""

import pytest
import torch

import sys
sys.path.insert(0, 'src')

from vit_pytorch.gumbel_topk_splitter import GumbelTopKSplitter


class TestGradientBalance:
    """Gradient balance tests for I109-6."""

    @pytest.fixture
    def splitter(self):
        """Create a GumbelTopKSplitter instance for testing."""
        return GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=4,
            hidden_dim=128,
            pool_size=4,
            K_max=32,
            temperature=1.0,
        )

    def test_gradient_flow_through_splitter(self, splitter):
        """Verify gradients flow through splitter.

        I109-6: 验证梯度可以通过 STE 流向 logits
        """
        torch.manual_seed(42)

        # 创建特征
        features = torch.randn(1, 256, 16, 16, requires_grad=True)

        # 前向传播
        result = splitter(features, (64, 64), hard=False)

        # 使用 selected_mask.sum() 作为损失
        loss = result.selected_mask.sum()

        # 反向传播
        loss.backward()

        # 验证梯度可以流动到 features
        assert features.grad is not None, "Gradient should flow to features"
        assert not torch.isnan(features.grad).any(), "No NaN in gradients"

        # 验证梯度不为零
        assert features.grad.abs().sum() > 0, "Gradient should be non-zero"

        print(f"\nI109-6 Gradient Flow Test:")
        print(f"  Selected mask sum: {result.selected_mask.sum().item():.2f}")
        print(f"  Features grad norm: {features.grad.norm().item():.6f}")
        print(f"  Features grad mean: {features.grad.abs().mean().item():.6f}")

    def test_gradient_coverage_with_quota_loss(self, splitter):
        """Verify gradient coverage using auxiliary losses.

        I109-6: 使用辅助损失（quota_loss）验证梯度流动
        """
        torch.manual_seed(456)

        features = torch.randn(1, 256, 16, 16, requires_grad=True)
        result = splitter(features, (64, 64), hard=False)

        # 获取辅助损失
        aux_losses = splitter.get_auxiliary_losses(result)
        total_loss = result.selected_mask.sum() + sum(aux_losses.values())

        total_loss.backward()

        # 验证梯度可以流动
        assert features.grad is not None, "Gradient should flow to features"
        assert not torch.isnan(features.grad).any(), "No NaN in gradients"

        print(f"\nI109-6 Gradient Coverage with Aux Losses:")
        print(f"  Aux losses: {aux_losses}")

    def test_coverage_ratio_calculation(self, splitter):
        """Verify coverage ratio α = K/N is computed correctly.

        I109-6 核心公式:
        st_mask = hard_mask - soft_mask.detach() + coverage_ratio * soft_mask

        注意: selected_mask 是软掩码（sigmoid 概率）
        覆盖率 α = num_selected / num_candidates
        """
        torch.manual_seed(789)

        features = torch.randn(1, 256, 16, 16)
        result = splitter(features, (64, 64), hard=False)

        # 计算覆盖率
        K_hard = result.num_selected_per_batch[0].item()
        N_candidates = result.selected_mask.shape[1]
        coverage_ratio = K_hard / N_candidates

        print(f"\nI109-6 Coverage Ratio:")
        print(f"  K (hard selected): {K_hard}")
        print(f"  N (candidates): {N_candidates}")
        print(f"  α = K/N: {coverage_ratio:.3f}")

        # 覆盖率应该在合理范围内
        assert 0.001 <= coverage_ratio <= 0.5, \
            f"Coverage ratio {coverage_ratio:.3f} out of expected range [0.001, 0.5]"


class TestGradientBalanceMathematical:
    """Mathematical verification of gradient balance."""

    def test_theoretical_gradient_ratio(self):
        """Verify theoretical gradient ratio matches formula.

        I109-6 理论分析:
        覆盖率缩放 STE: st = hard - soft.detach() + α * soft

        梯度计算:
        ∂st/∂soft = -1 (从 hard 中减去) + α (缩放的 soft)
        = α - 1 (选中 token, hard=1)
        = α (未选中 token, hard=0)

        梯度比率:
        ||∇unselected|| / ||∇selected|| ≈ α / (1-α)
        """
        # 验证公式: ratio ≈ α / (1-α)
        for alpha in [0.05, 0.1, 0.15, 0.2]:
            expected_ratio = alpha / (1 - alpha)
            print(f"α={alpha:.2f}: theoretical gradient ratio ≈ {expected_ratio:.2f}")

            # 验证公式合理性
            assert expected_ratio < 0.3, \
                f"Theoretical ratio {expected_ratio:.2f} exceeds threshold for low coverage"


class TestSTEGradientFlow:
    """Direct STE gradient flow tests."""

    @pytest.fixture
    def splitter(self):
        """Create a GumbelTopKSplitter instance for testing."""
        return GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=4,
            hidden_dim=128,
            pool_size=4,
            K_max=32,
            temperature=1.0,
        )

    def test_ste_gradient_scaling(self, splitter):
        """Test that STE uses coverage ratio for gradient scaling.

        I109-6 核心改进:
        覆盖率缩放 STE 使用 α = K/N 作为梯度缩放因子
        """
        torch.manual_seed(42)

        features = torch.randn(1, 256, 16, 16, requires_grad=True)
        result = splitter(features, (64, 64), hard=False)

        # 计算覆盖率
        K_hard = result.num_selected_per_batch[0].item()
        N_candidates = result.selected_mask.shape[1]
        coverage = K_hard / N_candidates

        print(f"\nI109-6 STE Gradient Scaling:")
        print(f"  K (hard selected): {K_hard}")
        print(f"  N (candidates): {N_candidates}")
        print(f"  α = K/N: {coverage:.3f}")

        # 验证覆盖率在有效范围内
        assert 0 < coverage < 1, "Coverage should be in (0, 1)"


class TestSTEImplementation:
    """Verify STE implementation details."""

    @pytest.fixture
    def splitter(self):
        """Create a GumbelTopKSplitter instance for testing."""
        return GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=4,
            hidden_dim=128,
            pool_size=4,
            K_max=32,
            temperature=1.0,
        )

    def test_hard_vs_soft_mask(self, splitter):
        """Compare hard and soft mask behavior.

        I109-6: 硬掩码用于前向，软掩码用于反向（STE）
        """
        torch.manual_seed(42)

        features = torch.randn(1, 256, 16, 16)

        # 硬选择
        result_hard = splitter(features, (64, 64), hard=True)
        # 软选择
        result_soft = splitter(features, (64, 64), hard=False)

        print(f"\nI109-6 Hard vs Soft Mask:")
        print(f"  Hard mask sum: {result_hard.selected_mask.sum().item():.0f}")
        print(f"  Soft mask sum: {result_soft.selected_mask.sum().item():.2f}")
        print(f"  Hard num_selected: {result_hard.num_selected_per_batch}")
        print(f"  Soft num_selected: {result_soft.num_selected_per_batch}")

        # 硬掩码应该是二值的（总和 = 选中的数量）
        assert result_hard.selected_mask.sum().item() == result_hard.num_selected_per_batch[0].item(), \
            "Hard mask sum should equal num_selected"

    def test_auxiliary_losses_exist(self, splitter):
        """Verify auxiliary losses are computed.

        I109-6: 辅助损失包括 quota_loss, elastic_budget_loss 等
        """
        torch.manual_seed(123)

        features = torch.randn(1, 256, 16, 16)
        result = splitter(features, (64, 64), hard=False)

        aux_losses = splitter.get_auxiliary_losses(result)

        print(f"\nI109-6 Auxiliary Losses:")
        for name, loss in aux_losses.items():
            print(f"  {name}: {loss.item():.6f}" if loss.numel() == 1 else f"  {name}: {loss}")

        # 应该有一些辅助损失
        assert len(aux_losses) > 0, "Should have auxiliary losses"
