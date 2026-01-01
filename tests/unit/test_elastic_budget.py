# -*- coding: utf-8 -*-
"""
P10-9: Elastic Budget 弹性预算机制单元测试

测试用例:
    1. get_soft_token_count 可微分性验证
    2. get_elastic_budget_loss Dead Zone 行为
    3. 非对称惩罚比例验证
    4. 梯度流验证
    5. get_auxiliary_losses 集成

数学形式化:
==========

Elastic Budget 损失函数:
    L_elastic = λ_over · ReLU(N - N_max)² / N_max + λ_under · ReLU(N_min - N) / N_max

区间行为:
    | 区间           | 损失 | 梯度方向 | 行为     |
    |----------------|------|----------|----------|
    | N < N_min      | > 0  | ∂L/∂N < 0 | 软约束增加 |
    | N ∈ [N_min, N_max] | = 0 | 0 | Dead Zone |
    | N > N_max      | > 0  | ∂L/∂N > 0 | 二次惩罚 |
"""

import pytest
import torch
import torch.nn as nn
import math

from vit_pytorch.split_adaptive import LearnableSplitter


class TestGetSoftTokenCount:
    """软 Token 计数测试组."""

    @pytest.fixture
    def splitter(self) -> LearnableSplitter:
        """创建测试用分割器."""
        return LearnableSplitter(
            feature_dim=64,
            max_depth=3,
            min_region_size=4,
        )

    def test_soft_token_count_shape(self, splitter: LearnableSplitter):
        """测试软 Token 计数返回标量."""
        soft_count = splitter.get_soft_token_count(batch_size=1)
        assert soft_count.ndim == 0  # 标量
        assert soft_count.dtype == torch.float32

    def test_soft_token_count_batch_scaling(self, splitter: LearnableSplitter):
        """测试软 Token 计数随 batch size 线性缩放."""
        count_1 = splitter.get_soft_token_count(batch_size=1)
        count_2 = splitter.get_soft_token_count(batch_size=2)
        count_4 = splitter.get_soft_token_count(batch_size=4)
        
        # 期望线性关系
        assert torch.isclose(count_2, count_1 * 2, rtol=0.01)
        assert torch.isclose(count_4, count_1 * 4, rtol=0.01)

    def test_soft_token_count_bounds(self, splitter: LearnableSplitter):
        """测试软 Token 计数在合理范围内."""
        # 对于 max_depth=3:
        # 最少 tokens = 1 (不分割)
        # 最多 tokens = 4^3 = 64 (全分割)
        soft_count = splitter.get_soft_token_count(batch_size=1)
        
        assert soft_count >= 1.0, "至少 1 个 token"
        assert soft_count <= 64.0, "最多 4^3 = 64 个 tokens"

    def test_soft_token_count_with_cached_probs(self, splitter: LearnableSplitter):
        """测试使用缓存概率时的梯度流."""
        splitter.train()
        
        # 模拟前向传播后缓存的分割概率
        splitter._cached_split_probs = {
            0: torch.tensor([0.6, 0.7], requires_grad=True),
            1: torch.tensor([0.5, 0.5, 0.4, 0.6], requires_grad=True),
            2: torch.tensor([0.3] * 8, requires_grad=True),
        }
        
        soft_count = splitter.get_soft_token_count(batch_size=1)
        
        # 应该保留梯度
        assert soft_count.requires_grad

    def test_soft_token_count_no_cache_eval(self, splitter: LearnableSplitter):
        """测试无缓存时使用阈值先验 (P10-10 修复后始终有梯度)."""
        splitter.eval()
        splitter._cached_split_probs = {}  # 清空缓存
        
        soft_count = splitter.get_soft_token_count(batch_size=1)
        
        # P10-10 修复: 阈值先验替代 EMA，始终有梯度
        # 即使在 eval 模式且无缓存时，也可以通过阈值先验获得梯度信号
        assert soft_count.requires_grad


class TestGetElasticBudgetLoss:
    """Elastic Budget 损失测试组."""

    @pytest.fixture
    def splitter(self) -> LearnableSplitter:
        """创建测试用分割器."""
        return LearnableSplitter(
            feature_dim=64,
            max_depth=3,
            min_region_size=4,
        )

    def test_dead_zone_zero_loss(self, splitter: LearnableSplitter):
        """测试 Dead Zone 内损失为零."""
        N_min, N_max = 32, 256
        
        # Dead Zone 中点
        N = torch.tensor(128.0)
        loss = splitter.get_elastic_budget_loss(N, N_min, N_max)
        assert loss.item() == 0.0
        
        # Dead Zone 边界
        N_lower = torch.tensor(32.0)
        N_upper = torch.tensor(256.0)
        loss_lower = splitter.get_elastic_budget_loss(N_lower, N_min, N_max)
        loss_upper = splitter.get_elastic_budget_loss(N_upper, N_min, N_max)
        assert loss_lower.item() == 0.0
        assert loss_upper.item() == 0.0

    def test_over_budget_quadratic_penalty(self, splitter: LearnableSplitter):
        """测试超出上界的二次惩罚."""
        N_min, N_max = 32, 256
        lambda_over = 0.1
        
        # N = 300 > N_max = 256
        N = torch.tensor(300.0)
        loss = splitter.get_elastic_budget_loss(
            N, N_min, N_max, 
            lambda_over=lambda_over, lambda_under=0.01
        )
        
        # 期望: λ_over * (N - N_max)² / N_max = 0.1 * (44)² / 256
        expected = lambda_over * (300 - 256) ** 2 / 256
        assert torch.isclose(loss, torch.tensor(expected), rtol=0.01)

    def test_under_budget_linear_penalty(self, splitter: LearnableSplitter):
        """测试低于下界的线性惩罚."""
        N_min, N_max = 32, 256
        lambda_under = 0.01
        
        # N = 20 < N_min = 32
        N = torch.tensor(20.0)
        loss = splitter.get_elastic_budget_loss(
            N, N_min, N_max, 
            lambda_over=0.1, lambda_under=lambda_under
        )
        
        # 期望: λ_under * (N_min - N) / N_max = 0.01 * (32 - 20) / 256
        expected = lambda_under * (32 - 20) / 256
        assert torch.isclose(loss, torch.tensor(expected), rtol=0.01)

    def test_asymmetric_ratio(self, splitter: LearnableSplitter):
        """测试非对称惩罚比例 (λ_over >> λ_under)."""
        N_min, N_max = 32, 256
        lambda_over, lambda_under = 0.1, 0.01  # 10:1 比例
        
        # 超出和低于相同数量
        excess = 50  # 超出/低于 50 tokens
        N_over = torch.tensor(float(N_max + excess))
        N_under = torch.tensor(float(N_min - excess))
        
        loss_over = splitter.get_elastic_budget_loss(
            N_over, N_min, N_max, lambda_over=lambda_over, lambda_under=lambda_under
        )
        loss_under = splitter.get_elastic_budget_loss(
            N_under, N_min, N_max, lambda_over=lambda_over, lambda_under=lambda_under
        )
        
        # 超出的惩罚应远大于低于的惩罚
        # over: 0.1 * 50² / 256 = 0.977
        # under: 0.01 * 50 / 256 = 0.00195
        assert loss_over > 100 * loss_under  # 非对称性

    def test_gradient_directions(self, splitter: LearnableSplitter):
        """测试梯度方向符合设计意图."""
        N_min, N_max = 32, 256
        
        # N > N_max: 梯度应鼓励减少 N
        N_over = torch.tensor(300.0, requires_grad=True)
        loss_over = splitter.get_elastic_budget_loss(N_over, N_min, N_max)
        loss_over.backward()
        assert N_over.grad > 0  # ∂L/∂N > 0 → 减少 N 可降低 loss
        
        # N < N_min: 梯度应鼓励增加 N
        N_under = torch.tensor(20.0, requires_grad=True)
        loss_under = splitter.get_elastic_budget_loss(N_under, N_min, N_max)
        loss_under.backward()
        assert N_under.grad < 0  # ∂L/∂N < 0 → 增加 N 可降低 loss
        
        # N ∈ Dead Zone: 梯度为零
        N_dead = torch.tensor(128.0, requires_grad=True)
        loss_dead = splitter.get_elastic_budget_loss(N_dead, N_min, N_max)
        loss_dead.backward()
        assert N_dead.grad == 0.0  # Dead Zone


class TestGetAuxiliaryLossesWithElasticBudget:
    """get_auxiliary_losses 集成测试."""

    @pytest.fixture
    def splitter(self) -> LearnableSplitter:
        """创建测试用分割器."""
        return LearnableSplitter(
            feature_dim=64,
            max_depth=3,
            min_region_size=4,
        )

    def test_elastic_budget_included(self, splitter: LearnableSplitter):
        """测试 Elastic Budget 损失被正确包含."""
        losses = splitter.get_auxiliary_losses(
            include_elastic_budget=True,
            batch_size=2,
            elastic_N_min=32,
            elastic_N_max=256,
        )
        
        assert 'elastic_budget_loss' in losses
        assert losses['elastic_budget_loss'].ndim == 0  # 标量

    def test_elastic_budget_excluded_by_default(self, splitter: LearnableSplitter):
        """测试 Elastic Budget 默认不包含."""
        losses = splitter.get_auxiliary_losses()
        
        assert 'elastic_budget_loss' not in losses
        assert 'barrier_loss' in losses

    def test_all_losses_differentiable(self, splitter: LearnableSplitter):
        """测试所有损失可微分."""
        splitter.train()
        
        # 模拟缓存
        splitter._cached_split_probs = {
            0: torch.tensor([0.6], requires_grad=True),
            1: torch.tensor([0.5] * 2, requires_grad=True),
            2: torch.tensor([0.4] * 4, requires_grad=True),
        }
        
        losses = splitter.get_auxiliary_losses(
            include_elastic_budget=True,
            batch_size=1,
            elastic_N_min=4,
            elastic_N_max=32,
        )
        
        # 验证可反向传播
        total_loss = sum(losses.values())
        total_loss.backward()
        
        # 验证缓存概率收到梯度
        for d, probs in splitter._cached_split_probs.items():
            assert probs.grad is not None


class TestEndToEndGradientFlow:
    """端到端梯度流测试."""

    def test_gradient_flows_to_splitter_params(self):
        """验证 Elastic Budget 梯度流向分割器参数."""
        splitter = LearnableSplitter(
            feature_dim=64,
            max_depth=2,
            min_region_size=4,
        )
        splitter.train()
        
        # 创建需要梯度的输入
        features = torch.randn(1, 64, 8, 8, requires_grad=True)
        
        # 前向传播
        from vit_pytorch.split_adaptive import TensorSplitResult
        result = splitter(features, (64, 64))
        
        # 计算 Elastic Budget 损失
        soft_count = splitter.get_soft_token_count(batch_size=1)
        elastic_loss = splitter.get_elastic_budget_loss(
            soft_count, N_min=4, N_max=16
        )
        
        # 反向传播
        elastic_loss.backward()
        
        # 验证阈值偏移收到梯度
        assert splitter.threshold_offsets.grad is not None


class TestHilbertConstraintCompliance:
    """Hilbert Curve 约束合规性测试."""

    def test_power_of_4_recommendation(self):
        """测试 N_max 应为 4^m 的建议."""
        valid_N_max = [4, 16, 64, 256, 1024]  # 4^1, 4^2, 4^3, 4^4, 4^5
        
        for N in valid_N_max:
            # 验证是 4 的幂
            assert N & (N - 1) == 0 or N == 4, f"{N} should be power of 4"
            # 额外验证: log4(N) 是整数
            log4 = math.log(N) / math.log(4)
            assert abs(log4 - round(log4)) < 1e-10

    def test_typical_configurations(self):
        """测试典型配置的合理性."""
        configs = [
            # (image_size, N_target, N_min, N_max)
            (64, 64, 32, 256),    # Tiny ImageNet
            (224, 196, 98, 784),  # ImageNet (aligned with ViT-B/16)
        ]
        
        for img_size, N_target, N_min, N_max in configs:
            # Dead Zone 应包含 N_target
            assert N_min <= N_target <= N_max
            # 动态范围合理 (不超过 8x)
            assert N_max / N_min <= 8
