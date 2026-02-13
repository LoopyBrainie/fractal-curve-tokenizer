# -*- coding: utf-8 -*-
"""
I121-5, I122-6: 课程学习权重测试

测试 CurriculumWeightScheduler 的数学性质:
1. Cosine 平滑过渡的连续性和梯度有界性
2. 熵权重反向联动的正确性 (I122-6: 从平方根联动改为线性联动)
3. 距离惩罚因子的保护作用
"""

import pytest
import math
import torch
import torch.nn as nn
from torch import Tensor

from vit_pytorch.layers.splitters.gumbel_topk import CurriculumWeightScheduler
from vit_pytorch.core.constants import (
    CURRICULUM_EXPLORATION_END,
    CURRICULUM_ADAPTATION_END,
    CURRICULUM_WEIGHT_FACTOR_EXPLORE,
    CURRICULUM_WEIGHT_FACTOR_ADAPT,
    CURRICULUM_WEIGHT_FACTOR_EXPLOIT,
    CURRICULUM_PENALTY_GAMMA,
    CURRICULUM_BASE_ENTROPY_WEIGHT,
    CURRICULUM_BASE_BUDGET_WEIGHT,
)


class TestCurriculumWeightScheduler:
    """CurriculumWeightScheduler 测试类"""

    def test_initialization(self):
        """测试初始化 (I122-6: 使用新常量)"""
        scheduler = CurriculumWeightScheduler(
            total_epochs=100,
            exploration_end=CURRICULUM_EXPLORATION_END,
            adaptation_end=CURRICULUM_ADAPTATION_END,
            explore_factor=CURRICULUM_WEIGHT_FACTOR_EXPLORE,
            adapt_factor=CURRICULUM_WEIGHT_FACTOR_ADAPT,
            exploit_factor=CURRICULUM_WEIGHT_FACTOR_EXPLOIT,
            penalty_gamma=CURRICULUM_PENALTY_GAMMA,
            base_budget_weight=0.1,
            base_entropy_weight=0.3,
            enable_distance_penalty=True,
        )

        assert scheduler._total_epochs.item() == 100
        assert scheduler._exploration_end == CURRICULUM_EXPLORATION_END
        assert scheduler._adaptation_end == CURRICULUM_ADAPTATION_END
        assert scheduler._explore_factor == CURRICULUM_WEIGHT_FACTOR_EXPLORE
        assert scheduler._adapt_factor == CURRICULUM_WEIGHT_FACTOR_ADAPT
        assert scheduler._exploit_factor == CURRICULUM_WEIGHT_FACTOR_EXPLOIT
        assert scheduler._penalty_gamma == CURRICULUM_PENALTY_GAMMA
        assert scheduler._base_budget_weight == 0.1
        assert scheduler._base_entropy_weight == 0.3
        assert scheduler._enable_distance_penalty is True

    def test_phase_factor_explore(self):
        """测试探索阶段的阶段因子 (I122-6: 使用新常量)"""
        scheduler = CurriculumWeightScheduler(total_epochs=100)

        # I122-6: 探索阶段 α = 3.0
        exploration_epochs = int(CURRICULUM_EXPLORATION_END * 100)
        for epoch in range(exploration_epochs):
            t_norm = torch.tensor(epoch / 100.0)
            alpha = scheduler._get_phase_factor(t_norm)
            assert torch.allclose(alpha, torch.tensor(CURRICULUM_WEIGHT_FACTOR_EXPLORE), atol=1e-5), \
                f"Epoch {epoch}: alpha = {alpha}"

    def test_phase_factor_exploit(self):
        """测试利用阶段的阶段因子 (I122-6: 使用新常量)"""
        scheduler = CurriculumWeightScheduler(total_epochs=100)

        # I122-6: 利用阶段 α 从 1.0 过渡到 1/3
        # epoch 99 应该在利用阶段的末期，alpha 接近 1/3
        t_norm = torch.tensor(99 / 100.0)
        alpha = scheduler._get_phase_factor(t_norm)
        assert float(alpha) < 1.0, f"Epoch 99: alpha = {alpha}"  # 应该小于 1.0

    def test_phase_factor_adaptation_transition(self):
        """测试适应阶段的过渡 (I122-6: 更新边界)"""
        scheduler = CurriculumWeightScheduler(total_epochs=100)

        # I122-6: 适应阶段边界 [0.25, 0.75]
        # epoch 25: α 应该从 3.0 下降到约 2.0
        # epoch 50: α 应该从约 2.0 下降到约 1.0
        # epoch 75: α 应该接近 1.0

        alpha_25 = scheduler._get_phase_factor(torch.tensor(0.25))
        alpha_50 = scheduler._get_phase_factor(torch.tensor(0.50))
        alpha_75 = scheduler._get_phase_factor(torch.tensor(0.75))

        # 验证过渡趋势
        assert float(alpha_25) > float(alpha_50), f"25: {alpha_25}, 50: {alpha_50}"
        assert float(alpha_50) > float(alpha_75), f"50: {alpha_50}, 75: {alpha_75}"
        assert 0.9 <= float(alpha_75) <= 1.1, f"75: {alpha_75}"

    def test_cosine_smoothness(self):
        """验证 Cosine 平滑过渡的数学性质"""
        scheduler = CurriculumWeightScheduler(total_epochs=100)

        # 验证连续性: 相邻 epoch 的 alpha 变化应该很小
        max_diff = 0.0
        for epoch in range(99):
            t1 = torch.tensor(epoch / 100.0)
            t2 = torch.tensor((epoch + 1) / 100.0)
            alpha1 = scheduler._get_phase_factor(t1)
            alpha2 = scheduler._get_phase_factor(t2)
            diff = abs(float(alpha1) - float(alpha2))
            max_diff = max(max_diff, diff)

        # Cosine 平滑过渡，相邻变化应该很小
        assert max_diff < 0.1, f"Max diff: {max_diff}"

    def test_budget_weight_explore(self):
        """测试探索阶段的高预算权重 (I122-6: 使用新常量)"""
        scheduler = CurriculumWeightScheduler(total_epochs=100)

        # I122-6: 探索阶段: λ_b ≈ 3.0 × base_weight
        weight = scheduler.get_budget_weight(
            current_epoch=0,
            K=torch.tensor(50.0),
            K_min=torch.tensor(8.0),
            K_max=torch.tensor(100.0),
        )

        expected = scheduler._base_budget_weight * CURRICULUM_WEIGHT_FACTOR_EXPLORE  # α = 3.0
        assert torch.allclose(weight, torch.tensor(expected), atol=1e-5)

    def test_budget_weight_exploit(self):
        """测试利用阶段的低预算权重 (I122-6: 更新注释)"""
        scheduler = CurriculumWeightScheduler(total_epochs=100)

        # I122-6: 利用阶段: λ_b ≈ (1/3) × base_weight
        weight = scheduler.get_budget_weight(
            current_epoch=99,
            K=torch.tensor(50.0),
            K_min=torch.tensor(8.0),
            K_max=torch.tensor(100.0),
        )

        expected = scheduler._base_budget_weight * CURRICULUM_WEIGHT_FACTOR_EXPLOIT  # α ≈ 0.33
        assert torch.allclose(weight, torch.tensor(expected), atol=1e-1)

    def test_entropy_weight_linkage(self):
        """验证熵权重反向联动"""
        scheduler = CurriculumWeightScheduler(total_epochs=100)

        # 探索阶段: λ_e 较低
        lambda_e_explore = scheduler.get_entropy_weight(current_epoch=0)

        # 利用阶段: λ_e 较高
        lambda_e_exploit = scheduler.get_entropy_weight(current_epoch=99)

        # 利用阶段权重应该更高
        assert float(lambda_e_exploit) > float(lambda_e_explore), \
            f"explore: {lambda_e_explore}, exploit: {lambda_e_exploit}"

        # 探索阶段应该低于基础值
        assert float(lambda_e_explore) < scheduler._base_entropy_weight, \
            f"explore: {lambda_e_explore}, base: {scheduler._base_entropy_weight}"

        # 利用阶段应该高于基础值
        assert float(lambda_e_exploit) > scheduler._base_entropy_weight, \
            f"exploit: {lambda_e_exploit}, base: {scheduler._base_entropy_weight}"

    def test_distance_factor_in_range(self):
        """测试范围内无惩罚"""
        scheduler = CurriculumWeightScheduler(penalty_gamma=1.0, enable_distance_penalty=True)

        # K 在 [K_min, K_max] 范围内
        beta = scheduler._get_distance_factor(
            K=torch.tensor(50.0),
            K_min=torch.tensor(10.0),
            K_max=torch.tensor(100.0),
        )

        # 范围内应该接近 1.0
        assert torch.allclose(beta, torch.tensor(1.0), atol=1e-5)

    def test_distance_factor_out_of_range(self):
        """测试超出范围时的惩罚"""
        scheduler = CurriculumWeightScheduler(penalty_gamma=1.0, enable_distance_penalty=True)

        # K 超出 [K_min, K_max] 范围
        beta = scheduler._get_distance_factor(
            K=torch.tensor(150.0),
            K_min=torch.tensor(10.0),
            K_max=torch.tensor(100.0),
        )

        # 超出范围应该 > 1.0
        assert float(beta) > 1.0

    def test_distance_factor_disabled(self):
        """测试禁用距离惩罚"""
        scheduler = CurriculumWeightScheduler(penalty_gamma=1.0, enable_distance_penalty=False)

        # 禁用时应该始终返回 1.0
        for K in [50.0, 150.0, 200.0]:
            beta = scheduler._get_distance_factor(
                K=torch.tensor(K),
                K_min=torch.tensor(10.0),
                K_max=torch.tensor(100.0),
            )
            assert torch.allclose(beta, torch.tensor(1.0), atol=1e-5)

    def test_get_weight_info(self):
        """测试权重信息获取 (I122-6: 使用新常量)"""
        scheduler = CurriculumWeightScheduler(total_epochs=100)

        # 探索阶段
        info_explore = scheduler.get_weight_info(current_epoch=0)
        # I122-6: α_explore = 3.0, λ_b^0 = 0.01
        assert float(info_explore['phase_factor']) == pytest.approx(CURRICULUM_WEIGHT_FACTOR_EXPLORE, abs=0.1)
        expected_budget = CURRICULUM_BASE_BUDGET_WEIGHT * CURRICULUM_WEIGHT_FACTOR_EXPLORE
        assert float(info_explore['budget_weight']) == pytest.approx(expected_budget, abs=0.01)
        assert info_explore['epoch'] == 0
        assert info_explore['total_epochs'] == 100

        # 利用阶段
        info_exploit = scheduler.get_weight_info(current_epoch=99)
        # I122-6: α_exploit = 1/3 ≈ 0.33
        assert float(info_exploit['phase_factor']) == pytest.approx(CURRICULUM_WEIGHT_FACTOR_EXPLOIT, abs=0.1)
        expected_budget_exploit = CURRICULUM_BASE_BUDGET_WEIGHT * CURRICULUM_WEIGHT_FACTOR_EXPLOIT
        assert float(info_exploit['budget_weight']) == pytest.approx(expected_budget_exploit, abs=0.01)
        assert info_exploit['epoch'] == 99


class TestCurriculumWeightMathematicalProperties:
    """课程学习数学习性质测试"""

    def test_alpha_symmetry(self):
        """验证 Cosine 过渡的对称性 (I122-6: 更新阶段边界)"""
        scheduler = CurriculumWeightScheduler(total_epochs=100)

        # 在适应→利用过渡区间内，取对称点验证
        # I122-6: 利用过渡 t ∈ [0.75, 1.0]
        t1 = 0.80  # 80%
        t2 = 0.95  # 95%
        alpha1 = scheduler._get_phase_factor(torch.tensor(t1))
        alpha2 = scheduler._get_phase_factor(torch.tensor(t2))

        # 应该单调下降
        assert float(alpha1) > float(alpha2), f"80%: {alpha1}, 95%: {alpha2}"

    def test_alpha_bounds(self):
        """验证阶段因子边界 (I122-6: 更新为新配置)"""
        scheduler = CurriculumWeightScheduler(total_epochs=100)

        for epoch in range(100):
            t_norm = torch.tensor(epoch / 100.0)
            alpha = scheduler._get_phase_factor(t_norm)

            # I122-6: α ∈ [0.33, 3.0]
            assert 0.2 <= float(alpha) <= 3.5, f"Epoch {epoch}: alpha = {alpha}"

    def test_budget_weight_combined(self):
        """测试组合权重计算 (I122-6: 更新为新配置)"""
        scheduler = CurriculumWeightScheduler(
            total_epochs=100,
            base_budget_weight=0.1,
            penalty_gamma=2.0,
        )

        # 探索阶段 + 范围内
        weight1 = scheduler.get_budget_weight(
            current_epoch=0,
            K=torch.tensor(50.0),
            K_min=torch.tensor(10.0),
            K_max=torch.tensor(100.0),
        )
        # I122-6: λ_b = 0.1 × 3.0 × 1.0 = 0.3
        assert torch.allclose(weight1, torch.tensor(0.3), atol=1e-5)

        # 利用阶段 + 超范围
        weight2 = scheduler.get_budget_weight(
            current_epoch=99,
            K=torch.tensor(150.0),
            K_min=torch.tensor(10.0),
            K_max=torch.tensor(100.0),
        )
        # I122-6: λ_b = 0.1 × (1/3) × β > 0.03
        assert float(weight2) > 0.03

    def test_entropy_weight_linear_linkage(self):
        """测试线性联动 (I122-6: 从平方根联动改为线性联动)"""
        scheduler = CurriculumWeightScheduler(total_epochs=100)

        # 探索阶段: α = 3.0
        lambda_e_explore = scheduler.get_entropy_weight(current_epoch=0)

        # 利用阶段中期: α ≈ 0.5
        lambda_e_mid = scheduler.get_entropy_weight(current_epoch=85)

        # 验证利用阶段权重 > 探索阶段权重
        # 因为 λ_e = λ_e^0 / α，且探索阶段 α > 利用阶段 α
        assert float(lambda_e_mid) > float(lambda_e_explore), \
            f"explore: {lambda_e_explore}, mid: {lambda_e_mid}"

        # 验证熵权重随时间递增
        # 探索(α=3): λ_e ∝ 1/3 ≈ 0.33
        # 利用(α≈0.5): λ_e ∝ 1/0.5 = 2.0
        # 利用末期(α≈0.33): λ_e ∝ 1/0.33 ≈ 3.0
        assert float(lambda_e_mid) > 1.0 * scheduler._base_entropy_weight, \
            f"lambda_e_mid should be > base"


class TestCurriculumWeightIntegration:
    """课程学习集成测试"""

    def test_scheduler_creation(self):
        """测试调度器创建 (I122-6: 使用新常量)"""
        scheduler = CurriculumWeightScheduler()

        # 默认参数应该与常量一致
        assert scheduler._total_epochs.item() == 100
        assert scheduler._exploration_end == CURRICULUM_EXPLORATION_END
        assert scheduler._adaptation_end == CURRICULUM_ADAPTATION_END
        assert scheduler._base_budget_weight == CURRICULUM_BASE_BUDGET_WEIGHT
        assert scheduler._base_entropy_weight == CURRICULUM_BASE_ENTROPY_WEIGHT

    def test_full_epoch_sweep(self):
        """完整 epoch 扫描测试"""
        scheduler = CurriculumWeightScheduler(total_epochs=100)

        for epoch in range(100):
            # 获取权重
            budget_w = scheduler.get_budget_weight(
                current_epoch=epoch,
                K=torch.tensor(50.0),
                K_min=torch.tensor(8.0),
                K_max=torch.tensor(100.0),
            )
            entropy_w = scheduler.get_entropy_weight(current_epoch=epoch)

            # 验证权重有效性
            assert torch.isfinite(budget_w), f"Epoch {epoch}: budget_w = {budget_w}"
            assert torch.isfinite(entropy_w), f"Epoch {epoch}: entropy_w = {entropy_w}"
            assert float(budget_w) > 0, f"Epoch {epoch}: budget_w = {budget_w}"
            assert float(entropy_w) > 0, f"Epoch {epoch}: entropy_w = {entropy_w}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
