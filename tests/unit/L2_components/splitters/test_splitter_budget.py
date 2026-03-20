# -*- coding: utf-8 -*-
"""
Splitter Budget Tests - 整合自多个来源

迁移来源:
- test_i147_refactoring.py: TestSimplifiedBudgetLoss
- L3_critique/test_quota_gradient.py: TestContinuousQuotaAllocator
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np


class TestSimplifiedBudgetLoss:
    """简化的预算损失验证 (从 test_i147_refactoring.py 迁移)"""

    def test_relative_error_loss_zero_at_target(self):
        """验证 K = K_t 时损失 = 0"""
        from vit_pytorch.layers.splitters.gumbel_topk import GumbelTopKSplitter
        from vit_pytorch.core.config import SplitterConfig

        config = SplitterConfig(max_level_limit=4)
        splitter = GumbelTopKSplitter(config=config, image_size=(64, 64))

        # K = K_t 时，损失应该接近 0
        K = 32.0
        K_t = 32.0
        N = 85  # candidate_count

        diff = abs(K - K_t) / N  # 相对误差
        delta = 0.5

        if diff <= delta:
            loss = 0.5 * diff ** 2
        else:
            loss = delta * diff - 0.5 * delta ** 2

        assert loss < 1e-10, f"Loss should be 0 at K=K_t: {loss}"

    def test_relative_error_loss_positive_away_from_target(self):
        """验证偏离目标时损失 > 0"""
        K_t = 32.0
        N = 85
        delta = 0.5

        for K in [16, 48, 64, 128]:
            diff = abs(K - K_t) / N
            if diff <= delta:
                loss = 0.5 * diff ** 2
            else:
                loss = delta * diff - 0.5 * delta ** 2

            assert loss > 0, f"Loss should be > 0 when K != K_t: K={K}"

    def test_gradient_direction(self):
        """验证梯度方向正确"""
        K = torch.tensor(48.0, requires_grad=True)  # K > K_t
        K_t = 32.0
        N = 85

        diff = abs(K - K_t) / N
        delta = 0.5
        diff = diff.clamp(min=0.0, max=1.0)

        if diff <= delta:
            loss = 0.5 * diff ** 2
        else:
            loss = delta * diff - 0.5 * delta ** 2

        loss.backward()

        # K > K_t: 梯度应该 > 0 (指向减少 K)
        assert K.grad.item() > 0, f"Gradient should be > 0 when K > K_t"

        # K < K_t
        K = torch.tensor(16.0, requires_grad=True)
        diff = abs(K - K_t) / N
        diff = diff.clamp(min=0.0, max=1.0)

        if diff <= delta:
            loss = 0.5 * diff ** 2
        else:
            loss = delta * diff - 0.5 * delta ** 2

        loss.backward()

        # K < K_t: 梯度应该 < 0 (指向增加 K)
        assert K.grad.item() < 0, f"Gradient should be < 0 when K < K_t"

    def test_huber_boundary_behavior(self):
        """验证 Huber 边界行为"""
        K_t = 32.0
        N = 85
        delta_norm = 0.5  # 归一化后的 delta

        # 小偏差 (在二次区域)
        K_small = 33.0
        diff_small = abs(K_small - K_t) / N
        loss_small = 0.5 * diff_small ** 2

        # 大偏差 (在边界区域)
        K_large = 64.0
        diff_large = abs(K_large - K_t) / N
        loss_large = delta_norm * diff_large - 0.5 * delta_norm ** 2

        # 大偏差的损失应该大于小偏差
        assert loss_large > loss_small, \
            f"Large deviation should have larger loss: large={loss_large}, small={loss_small}"


# =============================================================================
# I113-17: ContinuousQuotaAllocator 测试 (从 L3_critique/test_quota_gradient.py 迁移)
# =============================================================================

class TestContinuousQuotaAllocator:
    """I113-17: 连续松弛配额分配器测试"""

    def test_allocator_creation(self):
        """测试分配器创建"""
        from vit_pytorch.layers.splitters.gumbel_topk import ContinuousQuotaAllocator

        allocator = ContinuousQuotaAllocator(D=8, tau=1.0)

        assert allocator.D == 8
        assert allocator.temperature.item() == 1.0
        assert allocator.quota_logits.shape == (8,)

    def test_forward_backward(self):
        """测试前向和反向传播"""
        from vit_pytorch.layers.splitters.gumbel_topk import ContinuousQuotaAllocator

        allocator = ContinuousQuotaAllocator(D=8, tau=1.0)

        K_hard, K_soft = allocator.forward(K=64)

        # 验证形状
        assert K_hard.shape == (8,)
        assert K_soft.shape == (8,)

        # 验证硬配额是整数
        assert K_hard.dtype == torch.int64

        # 验证软配额是浮点数
        assert K_soft.dtype == torch.float32

        # 验证总和 (硬配额)
        assert K_hard.sum().item() == 64

    def test_gradient_flow(self):
        """测试梯度流是否正确传递"""
        from vit_pytorch.layers.splitters.gumbel_topk import ContinuousQuotaAllocator

        allocator = ContinuousQuotaAllocator(D=8, tau=1.0)

        # 启用梯度
        K_hard, K_soft = allocator.forward(K=64)

        # 使用软配额计算损失 - 目标是均匀分布
        target = torch.ones(8) * 8
        loss = F.mse_loss(K_soft, target)

        # 反向传播
        loss.backward()

        # 验证梯度存在且非零
        grad = allocator.quota_logits.grad
        assert grad is not None
        assert not torch.allclose(grad, torch.zeros_like(grad)), "梯度应该非零"

    def test_temperature_warmup(self):
        """测试温度warmup"""
        from vit_pytorch.layers.splitters.gumbel_topk import ContinuousQuotaAllocator

        # tau_warmup_steps=100: warmup 从 1.0 线性增加到 2.0
        # tau_decay_steps=100: warmup 完成后开始 cosine 衰减到 tau_min=0.1
        allocator = ContinuousQuotaAllocator(
            D=8, tau=1.0, tau_warmup_steps=100, tau_decay_steps=100,
            tau_min=0.1, tau_max=2.0, enable_warmup=True
        )

        # 初始温度 (step 0: warmup 起点)
        initial_tau = allocator.temperature.item()
        assert initial_tau >= 1.0, f"初始温度 {initial_tau} 应该 >= 1.0"

        # step 50: warmup 中间点 (温度应该 > 1.0 但 < 2.0)
        for _ in range(50):
            allocator.step()
        mid_tau = allocator.temperature.item()
        assert mid_tau > 1.0, f"warmup 中间温度 {mid_tau} 应该 > 1.0"

        # step 100: warmup 结束 (温度应该 = tau_max = 2.0)
        for _ in range(50):
            allocator.step()
        final_tau = allocator.temperature.item()
        assert final_tau == 2.0, f"warmup 结束时温度 {final_tau} 应该 = tau_max = 2.0"

        # step 150: decay 开始 (温度应该开始下降)
        for _ in range(50):
            allocator.step()
        decay_tau = allocator.temperature.item()
        assert decay_tau < 2.0, f"decay 开始后温度 {decay_tau} 应该 < 2.0"

    def test_multi_step_training(self):
        """模拟多步训练"""
        from vit_pytorch.layers.splitters.gumbel_topk import ContinuousQuotaAllocator

        allocator = ContinuousQuotaAllocator(D=8, tau=1.0, enable_warmup=False)

        optimizer = torch.optim.Adam(allocator.parameters(), lr=0.1)

        losses = []
        grad_norms = []

        for step in range(100):
            optimizer.zero_grad()

            K_hard, K_soft = allocator.forward(K=64)

            # 目标: 均匀分布 - 使用软配额保持梯度
            target = torch.ones(8) * 8
            loss = F.mse_loss(K_soft, target)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            grad_norms.append(allocator.quota_logits.grad.norm().item())

        # 验证损失下降
        assert losses[-1] < losses[0], "损失应该下降"


class TestQuotaGradientAnalysis:
    """配额分配梯度质量分析 (从 L3_critique/test_quota_gradient.py 迁移)"""

    def test_gradient_bias_analysis(self):
        """分析 STE 梯度偏差"""
        D = 8
        K = 64
        p = np.random.rand(D)
        p = p / p.sum()

        eps = 1e-5

        # STE 代理梯度
        grad_ste = np.eye(D) * K

        # 有限差分计算真实梯度
        def largest_remainder_method(p, K):
            floor_quota = np.floor(p * K).astype(int)
            remainders = p * K - floor_quota
            remaining = K - floor_quota.sum()

            if remaining > 0:
                indices = np.argsort(-remainders)[:remaining]
                floor_quota[indices] += 1
            return floor_quota

        grad_true = np.zeros((D, D))
        for e in range(D):
            p_eps = p.copy()
            p_eps[e] += eps
            p_eps = np.clip(p_eps, 0, 1)
            p_eps = p_eps / p_eps.sum()

            k_base = largest_remainder_method(p, K)
            k_eps = largest_remainder_method(p_eps, K)
            grad_true[:, e] = (k_eps - k_base) / eps

        grad_bias = grad_ste - grad_true
        bias_mean = np.abs(grad_bias).mean()

        # STE 偏差应该显著
        assert bias_mean > 0, f"STE gradient bias should be significant: {bias_mean:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
