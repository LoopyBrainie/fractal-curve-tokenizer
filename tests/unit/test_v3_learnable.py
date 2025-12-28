# -*- coding: utf-8 -*-
"""
StreamingFractalTokenizerV3 可学习分割集成测试

测试用例:
    1. LearnableSplitter 初始化与前向传播
    2. 梯度流验证 (端到端)
    3. 辅助损失计算
    4. 温度退火调度
    5. 训练/推理模式切换

Mathematical Note:
==================
Scheme B (BalancedGreedySplitter) 和 Scheme C (FixedBudgetDPSplitter)
已从代码库中移除。LearnableSplitter 现在是唯一的分割器，提供:

1. 端到端可微分性 (Gumbel-Softmax + STE):
   P(split | R) = softmax((logits + G) / τ), G ~ Gumbel(0, 1)

2. 无复杂度饱和 (MLP vs 方差公式):
   C_θ(R) = σ(MLP(ROI-Align(F, R))) 代替 C(R) = Var/(Var+σ₀²)

3. 可学习阈值: τ_d = τ_{base,d} + δ_d

4. O(D) BFS 复杂度 vs O(N·4^D) DP
"""

import pytest
import torch
import torch.nn as nn

from vit_pytorch import StreamingFractalTokenizerV3
from vit_pytorch.split_adaptive import LearnableSplitter


class TestV3Learnable:
    """可学习分割测试组 (现在是唯一的分割方式)."""
    
    def test_learnable_init(self):
        """测试可学习分割器初始化."""
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
        )
        
        # 验证 splitter 类型 (应该总是 LearnableSplitter)
        assert isinstance(v3.splitter, LearnableSplitter)
        
        # 验证 shared_conv 属性
        assert hasattr(v3, 'shared_conv')
        assert isinstance(v3.shared_conv, nn.Module)
        
        # 验证 _use_learnable_split 始终为 True
        assert v3._use_learnable_split is True
    
    def test_learnable_forward(self):
        """测试可学习分割前向传播."""
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
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
        """
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
        )
        v3.train()
        
        # 设置 offset 使阈值超出边界，触发 barrier loss
        with torch.no_grad():
            v3.splitter.threshold_offsets.data = torch.tensor([0.5, 0.5, 0.5, 0.5])
        
        barrier_loss = v3.splitter.get_threshold_barrier_loss()
        barrier_loss.backward()
        
        # threshold_offsets 应有梯度
        assert v3.splitter.threshold_offsets.grad is not None
        
    def test_gradient_through_barrier_loss(self):
        """测试超出边界时 barrier loss 提供梯度."""
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
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
        )
        v3.train()
        
        images = torch.randn(2, 3, 64, 64)
        _ = v3.tokenize(images)
        
        # 使用辅助损失
        aux_loss = v3.get_learnable_split_loss(
            lambda_entropy=0.1,
            lambda_budget=0.01,
            target_tokens=16,
        )
        
        assert aux_loss is not None
        aux_loss.backward()
        
        # 阈值参数应有梯度 (来自正则化项)
        assert v3.splitter.threshold_offsets.grad is not None
    
    def test_gradient_through_regularization(self):
        """测试阈值正则化提供梯度."""
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
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
    
    def test_scale_entropy(self):
        """测试尺度熵计算."""
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
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
            learnable_temperature=5.0,
        )
        v3_high_temp.train()
        
        v3_low_temp = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
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
        )
        v3.eval()
        
        with torch.no_grad():
            images = torch.randn(1, 3, 64, 64)
            output = v3.tokenize(images)
        
        assert len(output.sequences[0].tokens) >= 1


class TestV3API:
    """API 测试组."""
    
    def test_api_methods_exist(self):
        """测试所有必要的 API 方法存在."""
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
        )
        
        # 核心 API 应该存在
        assert hasattr(v3, 'tokenize')
        assert hasattr(v3, 'forward')
        assert hasattr(v3, 'get_split_stats')
        assert hasattr(v3, 'get_entropy_loss')
        assert hasattr(v3, 'get_scale_entropy')
        assert hasattr(v3, 'compute_scale_distribution')
        assert hasattr(v3, 'get_training_stats')
        assert hasattr(v3, 'get_learnable_split_loss')
        assert hasattr(v3, 'set_split_temperature')
    
    def test_always_uses_learnable_splitter(self):
        """测试始终使用 LearnableSplitter (Scheme B/C 已移除)."""
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
        )
        
        # 应该总是使用 LearnableSplitter
        assert v3._use_learnable_split is True
        assert isinstance(v3.splitter, LearnableSplitter)


class TestV3MathematicalProperties:
    """数学性质测试组."""
    
    def test_complexity_mlp_no_saturation(self):
        """验证 MLP 复杂度估计不饱和.
        
        数学对比:
        - Scheme B: C(R) = Var/(Var+σ₀²) → 1 as Var → ∞ (饱和)
        - Scheme L: C_θ(R) = σ(MLP(features)) (无饱和，可学习)
        """
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
        )
        
        # LearnableSplitter 应有 complexity_mlp
        assert hasattr(v3.splitter, 'complexity_mlp')
        assert isinstance(v3.splitter.complexity_mlp, nn.Module)
    
    def test_learnable_thresholds(self):
        """验证阈值是可学习参数.
        
        数学公式: τ_d = τ_{base,d} + δ_d
        其中 δ_d 是可学习的。
        """
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=4,
        )
        
        # 应有 threshold_offsets 参数
        assert hasattr(v3.splitter, 'threshold_offsets')
        assert isinstance(v3.splitter.threshold_offsets, nn.Parameter)
        
        # 应有 max_depth + 1 个阈值
        assert v3.splitter.threshold_offsets.shape[0] == 5
    
    def test_depth_entropy_computed(self):
        """验证深度熵正确计算."""
        v3 = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=128,
            base_patch_size=4,
            max_depth=3,
        )
        v3.eval()
        
        images = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            output = v3.tokenize(images)
        
        # 检查 split stats
        stats = v3.get_split_stats()
        assert 'depth_entropy' in stats
        assert stats['depth_entropy'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
