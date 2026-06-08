"""
R12 Auxiliary Loss unit tests (B.12).

Covers:
- test_l_tree_hinge_squared_formula: 验证 L_tree = mean(relu(parent-child-margin)^2)
- test_l_tree_zero_when_axiom_holds: 当 child >= parent + margin 时 L_tree = 0
- test_l_skew_kl_to_balanced: KL 散度计算正确
- test_combined_loss_is_scalar: forward 返回标量
"""

import math

import pytest
import torch

from vit_pytorch.core.measure_auxiliary_loss import R12AuxLoss, R12AuxLossConfig


class TestR12AuxLoss:
    """R12 Auxiliary Loss 测试套件"""

    @pytest.fixture
    def loss_module(self):
        return R12AuxLoss(R12AuxLossConfig(lambda_tree=0.10, lambda_skew=0.10))

    def test_l_tree_hinge_squared_formula(self, loss_module):
        """L_tree = mean(relu(parent - child - margin)^2)."""
        parent = torch.tensor([2.0, 0.5, 1.0])
        child = torch.tensor([0.0, 0.5, 1.5])
        # margin = 0.0 (default); differences: 2.0, 0.0, -0.5
        # relu: 2.0, 0.0, 0.0; squared: 4.0, 0.0, 0.0; mean = 4/3
        expected = 4.0 / 3.0
        l_tree = loss_module.compute_l_tree(parent, child)
        assert torch.isclose(l_tree, torch.tensor(expected), atol=1e-6)

    def test_l_tree_zero_when_axiom_holds(self, loss_module):
        """当 child >= parent + margin 时 L_tree = 0（公理 A4 满足）."""
        parent = torch.tensor([1.0, 0.5, 0.0])
        child = torch.tensor([2.0, 0.5, 1.0])
        # differences: -1.0, 0.0, -1.0 → relu 全部 ≤ 0 → mean = 0
        l_tree = loss_module.compute_l_tree(parent, child)
        assert l_tree.item() == 0.0

    def test_l_tree_with_margin(self):
        """margin > 0 时 hinge 边界偏移."""
        mod = R12AuxLoss(R12AuxLossConfig(margin=0.5))
        parent = torch.tensor([1.0, 0.5])
        child = torch.tensor([0.0, 0.5])
        # relu(1.0-0.0-0.5)=0.5; relu(0.5-0.5-0.5)=0 → squared: 0.25, 0 → mean=0.125
        l_tree = mod.compute_l_tree(parent, child)
        assert torch.isclose(l_tree, torch.tensor(0.125), atol=1e-6)

    def test_l_skew_kl_to_balanced(self, loss_module):
        """L_skew = KL(P || Q_balanced) + eps 当 P=Q 时退化为 eps."""
        # Q_balanced(0..3): 4^0, 4^-1, 4^-2, 4^-3 = [1, 1/4, 1/16, 1/64]
        # normalized: 64/85, 16/85, 4/85, 1/85
        q = torch.tensor([1.0, 1 / 4, 1 / 16, 1 / 64])
        q = q / q.sum()
        l_skew_same = loss_module.compute_l_skew(q)
        # 当 P == Q 时，KL=0，L_skew=eps
        assert torch.isclose(l_skew_same, torch.tensor(1e-6), atol=1e-7)

        # 偏离 balanced → KL > 0
        p_skewed = torch.tensor([1.0, 0.0, 0.0, 0.0])
        l_skew_off = loss_module.compute_l_skew(p_skewed)
        # p_log(1) - log(64/85) > 0
        expected_kl = 1.0 * math.log(85 / 64)
        assert l_skew_off.item() > expected_kl * 0.5  # rough sanity check

    def test_combined_loss_is_scalar(self, loss_module):
        """forward 返回标量 = λ_tree·L_tree + λ_skew·L_skew."""
        parent = torch.tensor([1.0, 0.0])
        child = torch.tensor([0.0, 0.0])
        depth_dist = torch.tensor([0.5, 0.3, 0.2])

        total = loss_module(parent, child, depth_dist)
        assert total.dim() == 0
        assert total.item() >= 0.0
        # 拆解验证
        l_tree = loss_module.compute_l_tree(parent, child).item()
        l_skew = loss_module.compute_l_skew(depth_dist).item()
        expected = 0.10 * l_tree + 0.10 * l_skew
        assert torch.isclose(total, torch.tensor(expected), atol=1e-5)

    def test_gradient_flow(self, loss_module):
        """确保 R12 支持梯度反传."""
        parent = torch.tensor([1.0], requires_grad=True)
        child = torch.tensor([0.0], requires_grad=True)
        depth_dist = torch.tensor([0.5, 0.3, 0.2], requires_grad=True)
        total = loss_module(parent, child, depth_dist)
        total.backward()
        assert parent.grad is not None
        assert child.grad is not None
        assert depth_dist.grad is not None
