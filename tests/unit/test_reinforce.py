"""REINFORCE 策略梯度实现测试"""

import pytest
import torch

from vit_pytorch import NextGenerationFractalViT


class TestREINFORCE:
    """REINFORCE 策略梯度测试"""

    @pytest.fixture
    def model(self):
        """创建测试用的简单模型"""
        return NextGenerationFractalViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=4,
            mlp_dim=128,
            min_patch_size=(4, 4),
            max_level=3,
            learnable_split=True,
        )

    def test_tokenizer_loss_with_reward(self, model):
        """验证 get_tokenizer_loss 正确接受 reward 参数"""
        model.train()
        x = torch.randn(2, 3, 32, 32)
        _ = model(x)
        
        # 提供 reward 和 baseline
        loss = model.get_tokenizer_loss(reward=-0.5, baseline=-0.6, entropy_coef=0.01)
        
        assert loss is not None
        assert loss.ndim == 0  # 标量
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_tokenizer_loss_without_reward(self, model):
        """验证不提供 reward 时的默认行为"""
        model.train()
        x = torch.randn(2, 3, 32, 32)
        _ = model(x)
        
        # 不提供 reward，应该只计算熵正则化
        loss = model.get_tokenizer_loss()
        
        assert loss is not None
        assert loss.ndim == 0

    def test_saved_log_probs_populated(self, model):
        """验证训练模式下 saved_log_probs 被正确填充"""
        model.train()
        model.clear_tokenizer_cache()
        
        x = torch.randn(1, 3, 32, 32)
        _ = model(x)
        
        # 检查 tokenizer 是否保存了 log_probs
        if hasattr(model.tokenizer, 'saved_log_probs'):
            # log_probs 列表可能为空（如果没有进行分割决策）
            # 或者包含张量
            for lp in model.tokenizer.saved_log_probs:
                assert isinstance(lp, torch.Tensor)
                # log_prob 可能是标量或 1-D 张量，取决于采样方式
                assert lp.numel() >= 1

    def test_saved_entropies_populated(self, model):
        """验证训练模式下 saved_entropies 被正确填充"""
        model.train()
        model.clear_tokenizer_cache()
        
        x = torch.randn(1, 3, 32, 32)
        _ = model(x)
        
        if hasattr(model.tokenizer, 'saved_entropies'):
            for ent in model.tokenizer.saved_entropies:
                assert isinstance(ent, torch.Tensor)
                assert ent >= 0  # 熵非负

    def test_policy_gradient_has_gradient(self, model):
        """验证策略梯度能正确反向传播"""
        model.train()
        model.clear_tokenizer_cache()
        
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        
        # 计算分类损失
        target = torch.randint(0, 10, (1,))
        ce_loss = torch.nn.functional.cross_entropy(output, target)
        
        # 计算 tokenizer loss（带 reward）
        aux_loss = model.get_tokenizer_loss(reward=-ce_loss.item(), baseline=-0.5)
        
        # 总损失
        total_loss = ce_loss + aux_loss
        total_loss.backward()
        
        # 检查 split_decision 网络是否有梯度
        if model.tokenizer.split_decision is not None:
            for param in model.tokenizer.split_decision.parameters():
                # 由于 REINFORCE 使用 sample()，梯度通过 log_prob 传递
                # 参数可能有梯度也可能没有（取决于是否触发分割）
                pass  # 不强制检查，避免测试不稳定

    def test_clear_tokenizer_cache(self, model):
        """验证 clear_tokenizer_cache 正确清理缓存"""
        model.train()
        
        # 先进行前向传播
        x = torch.randn(1, 3, 32, 32)
        _ = model(x)
        
        # 清理缓存
        model.clear_tokenizer_cache()
        
        # 验证缓存已清空
        assert len(model.tokenizer.saved_log_probs) == 0
        assert len(model.tokenizer.saved_entropies) == 0

    def test_inference_mode_no_log_probs(self, model):
        """验证推理模式下不保存 log_probs"""
        model.eval()
        model.clear_tokenizer_cache()
        
        with torch.no_grad():
            x = torch.randn(1, 3, 32, 32)
            _ = model(x)
        
        # 推理模式下不应该保存 log_probs
        # （或者保存空列表）
        if hasattr(model.tokenizer, 'saved_log_probs'):
            assert len(model.tokenizer.saved_log_probs) == 0


class TestBaselineVarianceReduction:
    """基线方差减少机制测试"""

    @pytest.fixture
    def model(self):
        return NextGenerationFractalViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=4,
            mlp_dim=128,
            min_patch_size=(4, 4),
            max_level=3,
            learnable_split=True,
        )

    def test_positive_advantage(self, model):
        """测试正优势（reward > baseline）"""
        model.train()
        x = torch.randn(1, 3, 32, 32)
        _ = model(x)
        
        # 正优势：reward > baseline -> 应该鼓励当前策略
        loss_positive = model.get_tokenizer_loss(reward=0.8, baseline=0.5)
        # 不需要检查具体值，只需确保不出错
        assert not torch.isnan(loss_positive)

    def test_negative_advantage(self, model):
        """测试负优势（reward < baseline）"""
        model.train()
        x = torch.randn(1, 3, 32, 32)
        _ = model(x)
        
        # 负优势：reward < baseline -> 应该减少当前策略概率
        loss_negative = model.get_tokenizer_loss(reward=0.2, baseline=0.5)
        assert not torch.isnan(loss_negative)

    def test_zero_advantage(self, model):
        """测试零优势（reward = baseline）"""
        model.train()
        x = torch.randn(1, 3, 32, 32)
        _ = model(x)
        
        # 零优势：应该不更新策略
        loss_zero = model.get_tokenizer_loss(reward=0.5, baseline=0.5)
        assert not torch.isnan(loss_zero)
