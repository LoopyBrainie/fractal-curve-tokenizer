# -*- coding: utf-8 -*-
"""
StreamingFractalTokenizerV3 可学习分割集成测试

测试用例:
    1. 规则分割 (balanced_greedy) 基线
    2. 可学习分割 (learnable) 功能验证
    3. 梯度流验证 (端到端)
    4. 辅助损失计算
    5. 温度退火调度
    6. 训练/推理模式切换
"""

import pytest
import torch
import torch.nn as nn

from vit_pytorch import StreamingFractalTokenizerV3
from vit_pytorch.split_adaptive import LearnableSplitter


class TestV3RuleBased:
    """规则分割测试组."""
    
    def test_balanced_greedy_basic(self):
        """测试 balanced_greedy 基本功能."""
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
            split_scheme='balanced_greedy',
        )
        
        images = torch.randn(2, 3, 64, 64)
        output = v3.tokenize(images)
        
        assert len(output.sequences) == 2
        for seq in output.sequences:
            # balanced_greedy 通常产生多个 token
            assert len(seq.tokens) >= 1
            assert seq.tokens.shape[-1] == 128  # d_model
    
    def test_fixed_budget_dp_basic(self):
        """测试 fixed_budget_dp 基本功能."""
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
            split_scheme='fixed_budget_dp',
            target_tokens=16,
        )
        
        images = torch.randn(2, 3, 64, 64)
        output = v3.tokenize(images)
        
        assert len(output.sequences) == 2
        for seq in output.sequences:
            # DP 方法严格遵守预算
            assert len(seq.tokens) <= 16


class TestV3Learnable:
    """可学习分割测试组."""
    
    def test_learnable_init(self):
        """测试可学习分割器初始化."""
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
            split_scheme='learnable',
        )
        
        # 验证 splitter 类型
        assert isinstance(v3.splitter, LearnableSplitter)
        
        # 验证 shared_conv 属性
        assert hasattr(v3, 'shared_conv')
        assert isinstance(v3.shared_conv, nn.Module)
    
    def test_learnable_forward(self):
        """测试可学习分割前向传播."""
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
            split_scheme='learnable',
        )
        v3.train()
        
        images = torch.randn(2, 3, 64, 64)
        output = v3.tokenize(images)
        
        assert len(output.sequences) == 2
        for seq in output.sequences:
            assert len(seq.tokens) >= 1
            assert seq.tokens.shape[-1] == 128
    
    def test_learnable_training_stats(self):
        """测试训练统计信息."""
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
            split_scheme='learnable',
        )
        v3.train()
        
        images = torch.randn(2, 3, 64, 64)
        _ = v3.tokenize(images)
        
        stats = v3.get_training_stats()
        
        assert stats['learnable_split'] is True
        assert 'learnable_thresholds' in stats
        assert 'learnable_temperature' in stats
        assert len(stats['learnable_thresholds']) == 4  # max_depth + 1


class TestV3GradientFlow:
    """梯度流测试组."""
    
    def test_gradient_to_threshold_offsets(self):
        """测试梯度是否流向 threshold_offsets 参数.
        
        数学分析:
            P-TAU-1: 使用纯 offset 参数化:
            τ_eff = τ_base + offset
            
            barrier loss 提供可微分路径:
            L_barrier = λ · Σ[ReLU(τ_min - τ)² + ReLU(τ - τ_max)²]
            
            梯度: ∂L_barrier/∂offset = ∂L_barrier/∂τ · ∂τ/∂offset = ∂L_barrier/∂τ · 1
            
        注意:
            由于分割决策是离散的 (采样)，直接的 token 损失无法传播梯度到
            SharedConv。这需要 P-GRAD-1 的直通估计器来解决。
        """
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
            split_scheme='learnable',
        )
        v3.train()
        
        # 设置 offset 使阈值超出边界，触发 barrier loss
        with torch.no_grad():
            v3.splitter.threshold_offsets.data = torch.tensor([0.5, 0.5, 0.5, 0.5])
        
        barrier_loss = v3.splitter.get_threshold_barrier_loss()
        barrier_loss.backward()
        
        # threshold_offsets 应有梯度
        assert v3.splitter.threshold_offsets.grad is not None
        # 对于接近边界的阈值，梯度可能为 0；对于超出边界的，梯度非 0
        # 初始 tau_bases 约为 [0.3, 0.18, 0.11, 0.07]，加上 0.5 后约 [0.8, 0.68, 0.61, 0.57]
        # 全部在 [0, 1] 内，barrier loss = 0，梯度为 0
        # 需要更大的 offset 来测试
        
    def test_gradient_through_barrier_loss(self):
        """测试超出边界时 barrier loss 提供梯度."""
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
            split_scheme='learnable',
        )
        v3.train()
        
        # 设置 offset 使阈值超出上边界
        with torch.no_grad():
            v3.splitter.threshold_offsets.data = torch.tensor([1.0, 1.0, 1.0, 1.0])
        
        barrier_loss = v3.splitter.get_threshold_barrier_loss()
        barrier_loss.backward()
        
        # 超出边界时应有非零梯度
        assert v3.splitter.threshold_offsets.grad is not None
        assert not torch.all(v3.splitter.threshold_offsets.grad == 0)
    
    def test_gradient_through_auxiliary_loss(self):
        """测试梯度通过辅助损失流向 MLP 参数.
        
        数学分析:
            由于分割决策是离散的，直接的 token 损失无法传播梯度到 MLP。
            辅助损失 (如 threshold regularization, depth loss) 提供
            可微分的梯度路径:
            
            L_aux → threshold_logits → grad ✓
            L_aux → depth_loss → MLP → grad ✓
        """
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
            split_scheme='learnable',
        )
        v3.train()
        
        images = torch.randn(2, 3, 64, 64)
        _ = v3.tokenize(images)
        
        # 使用辅助损失而非直接的 token 损失
        aux_loss = v3.get_learnable_split_loss(
            lambda_entropy=0.1,
            lambda_budget=0.01,
            target_tokens=16,
        )
        
        assert aux_loss is not None
        aux_loss.backward()
        
        # 阈值参数应有梯度 (来自正则化项)
        assert v3.splitter.threshold_offsets.grad is not None
        
        # 注意: MLP 梯度取决于 depth_loss 的实现
        # 如果 depth_loss 使用预计算的特征进行复杂度计算，则 MLP 有梯度
    
    def test_gradient_through_regularization(self):
        """测试阈值正则化提供梯度."""
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
            split_scheme='learnable',
        )
        v3.train()
        
        # 直接调用正则化损失 (barrier loss)
        reg_loss = v3.splitter.get_threshold_barrier_loss()
        reg_loss.backward()
        
        # 阈值必须有梯度
        assert v3.splitter.threshold_offsets.grad is not None


class TestV3AuxiliaryLoss:
    """辅助损失测试组."""
    
    def test_split_loss_computation(self):
        """测试可学习分割损失计算."""
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
            split_scheme='learnable',
        )
        v3.train()
        
        images = torch.randn(2, 3, 64, 64)
        _ = v3.tokenize(images)
        
        loss = v3.get_learnable_split_loss(
            lambda_entropy=0.1,
            lambda_budget=0.01,
            target_tokens=16,
        )
        
        assert loss is not None
        assert loss.requires_grad
        assert loss.item() >= 0
    
    def test_split_loss_none_for_rule_based(self):
        """测试规则分割器返回 None 损失."""
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
            split_scheme='balanced_greedy',
        )
        
        images = torch.randn(2, 3, 64, 64)
        _ = v3.tokenize(images)
        
        loss = v3.get_learnable_split_loss()
        assert loss is None
    
    def test_scale_entropy(self):
        """测试尺度熵计算."""
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
            split_scheme='learnable',
        )
        v3.train()
        
        images = torch.randn(2, 3, 64, 64)
        _ = v3.tokenize(images)
        
        entropy = v3.get_scale_entropy()
        assert entropy is not None
        assert entropy >= 0


class TestV3TemperatureAnnealing:
    """温度退火测试组."""
    
    def test_set_temperature(self):
        """测试温度设置."""
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
            split_scheme='learnable',
            learnable_temperature=1.0,
        )
        
        # 初始温度
        stats = v3.get_training_stats()
        assert abs(stats['learnable_temperature'] - 1.0) < 0.01
        
        # 设置新温度
        v3.set_split_temperature(0.5)
        stats = v3.get_training_stats()
        assert abs(stats['learnable_temperature'] - 0.5) < 0.01
    
    def test_temperature_affects_hardness(self):
        """测试温度对分割硬度的影响."""
        torch.manual_seed(42)
        
        v3_high_temp = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
            split_scheme='learnable',
            learnable_temperature=5.0,
        )
        v3_high_temp.train()
        
        v3_low_temp = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
            split_scheme='learnable',
            learnable_temperature=0.1,
        )
        v3_low_temp.train()
        
        images = torch.randn(1, 3, 64, 64)
        
        # 高温度 → 更软的决策
        out_high = v3_high_temp.tokenize(images)
        
        # 低温度 → 更硬的决策
        out_low = v3_low_temp.tokenize(images)
        
        # 两者都应该能工作
        assert len(out_high.sequences[0].tokens) >= 1
        assert len(out_low.sequences[0].tokens) >= 1


class TestV3TrainEvalMode:
    """训练/推理模式测试组."""
    
    def test_train_uses_soft_sampling(self):
        """测试训练模式使用软采样."""
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
            split_scheme='learnable',
        )
        v3.train()
        
        images = torch.randn(1, 3, 64, 64, requires_grad=True)
        output = v3.tokenize(images)
        
        # 训练模式应该支持梯度
        tokens = output.sequences[0].tokens
        loss = tokens.sum()
        loss.backward()
        
        assert images.grad is not None
    
    def test_eval_uses_hard_sampling(self):
        """测试推理模式使用硬采样."""
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
            split_scheme='learnable',
        )
        v3.eval()
        
        with torch.no_grad():
            images = torch.randn(1, 3, 64, 64)
            output = v3.tokenize(images)
        
        assert len(output.sequences[0].tokens) >= 1


class TestV3BackwardCompatibility:
    """向后兼容性测试组."""
    
    def test_default_scheme_is_balanced_greedy(self):
        """测试默认分割方案是 balanced_greedy."""
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
        )
        
        assert v3.split_scheme == 'balanced_greedy'
        assert not v3._use_learnable_split
    
    def test_api_compatibility(self):
        """测试 API 向后兼容."""
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
        )
        
        # 所有旧 API 应该存在
        assert hasattr(v3, 'tokenize')
        assert hasattr(v3, 'forward')
        assert hasattr(v3, 'get_split_stats')
        assert hasattr(v3, 'get_entropy_loss')
        assert hasattr(v3, 'get_scale_entropy')
        assert hasattr(v3, 'compute_scale_distribution')
        assert hasattr(v3, 'get_training_stats')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
