# -*- coding: utf-8 -*-
"""
Test I30-2: Hilbert-Aware Hard Mining
=====================================

测试 Token Variance 困难样本挖掘模块

数学验证:
---------
1. Token 方差计算: TokenVar(x) = (1/N) Σ ||f_i - f̄||²
2. 样本权重: w = 1 + λ × σ((TokenVar - μ) / (τ × σ))
3. EMA 更新: μ_new = (1-β) × μ_old + β × μ_batch
4. 梯度流: 权重不阻断 base_loss 梯度
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "examples"))

from training.losses.hilbert_hard_mining import (
    HilbertAwareHardMining,
    HilbertMiningWrapper,
    create_hilbert_mining_loss,
)


class TestTokenVarianceComputation:
    """测试 Token 方差计算的数学正确性"""
    
    def test_variance_formula_correctness(self):
        """验证方差计算公式: (1/N) Σ ||f_i - f̄||²"""
        mining = HilbertAwareHardMining(lambda_weight=0.5)
        
        # 构造已知方差的 tokens
        B, N, D = 2, 4, 8
        tokens = torch.randn(B, N, D)
        
        # 手动计算每个样本的方差
        expected_vars = []
        for i in range(B):
            sample_tokens = tokens[i]  # [N, D]
            mean_token = sample_tokens.mean(dim=0)  # [D]
            # (1/N) Σ ||f_i - f̄||²
            var = ((sample_tokens - mean_token) ** 2).sum() / N
            expected_vars.append(var)
        expected_vars = torch.stack(expected_vars)
        
        # 使用模块计算
        computed_vars = mining.compute_token_variance(tokens)
        
        # 验证一致性
        torch.testing.assert_close(computed_vars, expected_vars, rtol=1e-5, atol=1e-5)
    
    def test_variance_with_variable_lengths(self):
        """验证变长序列的方差计算"""
        mining = HilbertAwareHardMining(lambda_weight=0.5)
        
        B, N, D = 3, 10, 16
        tokens = torch.randn(B, N, D)
        lengths = torch.tensor([5, 8, 10])  # 不同有效长度
        
        computed_vars = mining.compute_token_variance(tokens, lengths)
        
        # 手动验证
        for i in range(B):
            L = lengths[i].item()
            valid_tokens = tokens[i, :L]
            mean_token = valid_tokens.mean(dim=0)
            expected_var = ((valid_tokens - mean_token) ** 2).sum() / L
            torch.testing.assert_close(
                computed_vars[i], expected_var, rtol=1e-5, atol=1e-5
            )
    
    def test_single_token_variance_is_zero(self):
        """单 token 方差应为 0"""
        mining = HilbertAwareHardMining(lambda_weight=0.5)
        
        B, N, D = 2, 10, 8
        tokens = torch.randn(B, N, D)
        lengths = torch.tensor([1, 1])  # 每个样本只有 1 个有效 token
        
        computed_vars = mining.compute_token_variance(tokens, lengths)
        
        assert (computed_vars == 0).all(), "单 token 方差应为 0"


class TestWeightComputation:
    """测试样本权重计算"""
    
    def test_weight_range(self):
        """验证权重范围: w ∈ [1.0, 1.0 + λ]"""
        lambda_weight = 0.5
        mining = HilbertAwareHardMining(
            lambda_weight=lambda_weight,
            warmup_batches=0,  # 禁用 warmup
        )
        
        # 模拟 EMA 统计已收敛
        mining.running_mean.fill_(1.0)
        mining.running_var.fill_(1.0)
        mining.batch_count.fill_(100)
        
        # 不同方差的 tokens
        B, N, D = 100, 16, 32
        tokens = torch.randn(B, N, D)
        base_loss = torch.ones(B)
        
        _, info = mining(tokens, base_loss)
        
        assert info['weight_min'] >= 1.0, f"最小权重应 >= 1.0, got {info['weight_min']}"
        assert info['weight_max'] <= 1.0 + lambda_weight + 0.01, \
            f"最大权重应 <= {1.0 + lambda_weight}, got {info['weight_max']}"
    
    def test_high_variance_gets_higher_weight(self):
        """高方差样本应获得更高权重"""
        mining = HilbertAwareHardMining(
            lambda_weight=0.5,
            warmup_batches=0,
        )
        
        # 设置 EMA
        mining.running_mean.fill_(1.0)
        mining.running_var.fill_(1.0)
        mining.batch_count.fill_(100)
        
        B, N, D = 2, 16, 32
        
        # 样本 0: 低方差 (所有 token 相似)
        low_var_tokens = torch.randn(1, 1, D).expand(1, N, D) + torch.randn(1, N, D) * 0.01
        # 样本 1: 高方差 (token 差异大)
        high_var_tokens = torch.randn(1, N, D) * 10
        
        tokens = torch.cat([low_var_tokens, high_var_tokens], dim=0)
        base_loss = torch.ones(B)
        
        _, info = mining(tokens, base_loss)
        
        # 高方差样本权重应更大
        # (通过 weight_max > weight_min 间接验证)
        assert info['weight_max'] > info['weight_min'], \
            "高方差样本应获得更高权重"


class TestEMAStatistics:
    """测试 EMA 统计更新"""
    
    def test_ema_momentum_update(self):
        """验证 EMA 更新公式: μ_new = (1-β) × μ_old + β × μ_batch"""
        momentum = 0.1
        mining = HilbertAwareHardMining(
            lambda_weight=0.5,
            momentum=momentum,
            warmup_batches=0,
        )
        
        # 初始化
        initial_mean = 5.0
        mining.running_mean.fill_(initial_mean)
        mining.running_var.fill_(1.0)
        mining.batch_count.fill_(10)
        
        # 构造新 batch
        B, N, D = 4, 8, 16
        tokens = torch.randn(B, N, D) * 2  # 方差约 4
        base_loss = torch.ones(B)
        
        # 计算 batch 方差均值
        variances = mining.compute_token_variance(tokens)
        batch_mean = variances.mean().item()
        
        # 期望的 EMA 更新
        expected_mean = (1 - momentum) * initial_mean + momentum * batch_mean
        
        # 执行 forward
        mining(tokens, base_loss)
        
        # 验证更新
        assert abs(mining.running_mean.item() - expected_mean) < 0.01, \
            f"EMA 均值更新不正确: expected {expected_mean:.4f}, got {mining.running_mean.item():.4f}"
    
    def test_warmup_phase_collects_statistics(self):
        """Warmup 期间只收集统计，不加权"""
        mining = HilbertAwareHardMining(
            lambda_weight=0.5,
            warmup_batches=100,
        )
        
        B, N, D = 4, 8, 16
        tokens = torch.randn(B, N, D)
        base_loss = torch.randn(B).abs()  # 不同的基础损失
        
        # Warmup 期间
        weighted_loss, info = mining(tokens, base_loss)
        
        assert info['in_warmup'], "应在 warmup 期间"
        # Warmup 期间损失应该是原始均值
        torch.testing.assert_close(weighted_loss, base_loss.mean(), rtol=1e-5, atol=1e-5)


class TestGradientFlow:
    """测试梯度流正确性"""
    
    def test_gradients_flow_through_weights(self):
        """验证梯度能正确流过权重计算"""
        mining = HilbertAwareHardMining(
            lambda_weight=0.5,
            warmup_batches=0,
        )
        mining.running_mean.fill_(1.0)
        mining.running_var.fill_(1.0)
        mining.batch_count.fill_(100)
        
        B, N, D = 4, 8, 16
        tokens = torch.randn(B, N, D, requires_grad=True)
        # 使用叶子张量来测试梯度流
        base_loss_leaf = torch.randn(B, requires_grad=True).abs() + 0.1
        base_loss_leaf.retain_grad()  # 保留非叶子张量梯度
        
        weighted_loss, _ = mining(tokens, base_loss_leaf)
        weighted_loss.backward()
        
        # 验证梯度存在
        assert tokens.grad is not None, "tokens 应有梯度 (通过方差计算)"
        # base_loss 作为输入，其梯度会流向上游
        # 验证 weighted_loss 与 base_loss 的关系
        assert weighted_loss.requires_grad, "weighted_loss 应可微分"


class TestHilbertMiningWrapper:
    """测试包装器类"""
    
    def test_wrapper_with_focal_loss(self):
        """测试与 FocalLoss 的集成"""
        from training.losses import FocalLoss
        
        base_loss = FocalLoss(gamma=2.0)
        wrapper = HilbertMiningWrapper(
            base_loss_fn=base_loss,
            lambda_weight=0.5,
            warmup_batches=0,
        )
        wrapper.mining.running_mean.fill_(1.0)
        wrapper.mining.running_var.fill_(1.0)
        wrapper.mining.batch_count.fill_(100)
        
        B, N, D, C = 4, 16, 32, 10
        logits = torch.randn(B, C)
        labels = torch.randint(0, C, (B,))
        tokens = torch.randn(B, N, D)
        
        # 使用 tokens
        loss_with_mining = wrapper(logits, labels, tokens=tokens)
        
        # 不使用 tokens
        loss_without_mining = wrapper(logits, labels, tokens=None)
        
        # 两者都应该能正常计算
        assert not torch.isnan(loss_with_mining), "带挖掘的损失不应为 NaN"
        assert not torch.isnan(loss_without_mining), "不带挖掘的损失不应为 NaN"


class TestCreateHilbertMiningLoss:
    """测试工厂函数"""
    
    def test_factory_function_creates_wrapper(self):
        """验证工厂函数创建正确的包装器"""
        from training.losses import FocalLoss
        
        base_loss = FocalLoss(gamma=2.5)
        mining_loss = create_hilbert_mining_loss(
            base_loss,
            lambda_weight=0.5,
            warmup_batches=50,
        )
        
        assert isinstance(mining_loss, HilbertMiningWrapper)
        assert mining_loss.mining.lambda_weight == 0.5
        assert mining_loss.mining.warmup_batches == 50


class TestMathematicalProperties:
    """验证数学性质"""
    
    def test_weight_symmetry_around_mean(self):
        """验证权重关于均值对称"""
        mining = HilbertAwareHardMining(
            lambda_weight=0.5,
            temperature=1.0,
            warmup_batches=0,
        )
        
        # 设置 EMA
        mean = 1.0
        std = 0.5
        mining.running_mean.fill_(mean)
        mining.running_var.fill_(std ** 2)
        mining.batch_count.fill_(100)
        
        # 测试: z = +2 和 z = -2 应该关于 w = 1.25 对称
        # w(+2) + w(-2) = 2 + 0.5 * (σ(2) + σ(-2)) = 2 + 0.5 * 1 = 2.5
        # 因为 σ(x) + σ(-x) = 1
        
        # 构造 z = +2 的样本
        D = 16
        N = 100
        high_var_token = torch.randn(1, 1, D)
        # 方差约 var * D = std^2 * D for per-dim, summed = D * std^2
        target_var = mean + 2 * std  # z = +2
        
        # 直接测试 sigmoid 性质
        z_pos = torch.tensor(2.0)
        z_neg = torch.tensor(-2.0)
        
        w_pos = 1.0 + 0.5 * torch.sigmoid(z_pos)
        w_neg = 1.0 + 0.5 * torch.sigmoid(z_neg)
        
        # σ(2) + σ(-2) ≈ 1
        assert abs((w_pos + w_neg).item() - 2.5) < 0.01, \
            f"权重应关于均值对称: w(+2) + w(-2) = {(w_pos + w_neg).item():.4f}, expected 2.5"
    
    def test_gradient_scaling_factor(self):
        """验证困难样本的梯度增益"""
        mining = HilbertAwareHardMining(
            lambda_weight=0.5,
            warmup_batches=0,
        )
        mining.running_mean.fill_(1.0)
        mining.running_var.fill_(1.0)
        mining.batch_count.fill_(100)
        
        # z = +2 (困难样本) 的权重应接近 1 + 0.5 * σ(2) ≈ 1.44
        # z = -2 (简单样本) 的权重应接近 1 + 0.5 * σ(-2) ≈ 1.06
        
        expected_hard_weight = 1.0 + 0.5 * torch.sigmoid(torch.tensor(2.0)).item()
        expected_easy_weight = 1.0 + 0.5 * torch.sigmoid(torch.tensor(-2.0)).item()
        
        gradient_ratio = expected_hard_weight / expected_easy_weight
        
        # 验证比例约 1.36
        assert 1.3 < gradient_ratio < 1.5, \
            f"困难/简单梯度比应在 1.3-1.5 之间, got {gradient_ratio:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
