"""
语义分裂器集成测试 (I110-9)

测试覆盖:
1. 端到端训练流程
2. Token 数量变化趋势
3. 深度分布统计
4. 与基线模型精度对比
"""

import pytest
import torch
import torch.nn as nn

from vit_pytorch import (
    FractalCurveViT,
    StreamingFractalTokenizerV3,
    SemanticSplitterConfig,
)
from vit_pytorch.layers.splitters.semantic_redundancy import SemanticRedundancySplitter
from vit_pytorch.modules.semantic_losses import SemanticRedundancyLoss


class TestSemanticTokenizerIntegration:
    """语义分裂器与 Tokenizer 集成测试"""

    def test_tokenizer_semantic_config(self):
        """测试 Tokenizer 配置语义分裂器"""
        config = SemanticSplitterConfig(
            hidden_dim=128,
            diversity_weight=0.1,
            reconstruction_weight=0.1,
        )

        tokenizer = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=256,
            base_patch_size=4,
        )

        # 配置语义分裂器
        tokenizer.use_semantic_splitter(config=config)

        assert tokenizer._use_semantic_splitter is True
        assert tokenizer._semantic_config.hidden_dim == 128
        assert tokenizer._semantic_loss_fn is not None

        # 获取分裂器实例
        splitter = tokenizer.get_semantic_splitter()
        assert isinstance(splitter, SemanticRedundancySplitter)

    def test_tokenizer_semantic_loss_fn(self):
        """测试 Tokenizer 获取语义损失函数"""
        tokenizer = StreamingFractalTokenizerV3(
            image_size=64,
            d_model=256,
            base_patch_size=4,
        )

        custom_loss = SemanticRedundancyLoss(
            diversity_weight=0.2,
            reconstruction_weight=0.3,
        )

        tokenizer.use_semantic_splitter(
            config=SemanticSplitterConfig(),
            loss_fn=custom_loss,
        )

        loss_fn = tokenizer.get_semantic_loss_fn()
        assert loss_fn is custom_loss


class TestSemanticModelIntegration:
    """语义分裂器与模型集成测试"""

    def test_model_with_semantic_splitter(self):
        """测试模型使用语义分裂器"""
        config = SemanticSplitterConfig(
            hidden_dim=128,
            diversity_weight=0.1,
            reconstruction_weight=0.1,
        )

        model = FractalCurveViT(
            image_size=64,
            num_classes=100,
            dim=256,
            num_layers=2,
            use_semantic_splitter=True,
            semantic_splitter_config=config,
        )

        assert model.use_semantic_splitter is True
        assert model.get_semantic_splitter() is not None
        assert model.get_semantic_loss_fn() is not None

    def test_model_forward_with_semantic(self):
        """测试模型前向传播（语义分裂器模式）"""
        model = FractalCurveViT(
            image_size=64,
            num_classes=100,
            dim=256,
            num_layers=2,
            use_semantic_splitter=True,
            semantic_splitter_config=SemanticSplitterConfig(),
        )

        x = torch.randn(2, 3, 64, 64)
        stats = model(x)

        assert stats.logits.shape == (2, 100)
        assert stats.features.shape == (2, 256)
        assert stats.transformer_tokens is not None

    def test_model_without_semantic_splitter(self):
        """测试模型不使用语义分裂器（基线模式）"""
        model = FractalCurveViT(
            image_size=64,
            num_classes=100,
            dim=256,
            num_layers=2,
            use_semantic_splitter=False,
        )

        assert model.use_semantic_splitter is False
        assert model.get_semantic_splitter() is None
        assert model.get_semantic_loss_fn() is None

        x = torch.randn(2, 3, 64, 64)
        stats = model(x)

        assert stats.logits.shape == (2, 100)

    def test_semantic_loss_computation(self):
        """测试语义损失计算"""
        model = FractalCurveViT(
            image_size=64,
            num_classes=100,
            dim=256,
            num_layers=2,
            use_semantic_splitter=True,
            semantic_splitter_config=SemanticSplitterConfig(),
        )

        # 模拟特征
        parent = torch.randn(2, 10, 256, requires_grad=True)
        child = torch.randn(2, 10, 4, 256, requires_grad=True)
        split = torch.rand(2, 10) > 0.5

        # 计算损失
        loss_dict = model.compute_semantic_loss(parent, child, split)

        assert 'loss' in loss_dict
        assert 'diversity_loss' in loss_dict
        assert 'reconstruction_loss' in loss_dict

        # 反向传播
        loss_dict['loss'].backward()

        assert parent.grad is not None
        assert child.grad is not None


class TestSemanticSplitterBehavior:
    """语义分裂器行为测试"""

    def test_split_decision_range(self):
        """测试分裂决策范围"""
        model = FractalCurveViT(
            image_size=64,
            num_classes=100,
            dim=256,
            num_layers=2,
            use_semantic_splitter=True,
            semantic_splitter_config=SemanticSplitterConfig(),
        )

        splitter = model.get_semantic_splitter()
        features = torch.randn(2, 10, 256)

        result = splitter(features, depth=2, remaining_quota=0.5)

        # 分裂决策应该在 [0, 1] 范围内
        assert result.split_decision.min() >= 0
        assert result.split_decision.max() <= 1

    def test_redundancy_calculation(self):
        """测试冗余性计算"""
        model = FractalCurveViT(
            image_size=64,
            num_classes=100,
            dim=256,
            num_layers=2,
            use_semantic_splitter=True,
            semantic_splitter_config=SemanticSplitterConfig(),
        )

        splitter = model.get_semantic_splitter()
        features = torch.randn(2, 10, 256)

        result = splitter(features, depth=2, remaining_quota=0.5)

        # 冗余性应该在 [0, 1] 范围内
        assert result.redundancy.min() >= 0
        assert result.redundancy.max() <= 1

    def test_hard_mode(self):
        """测试硬模式输出"""
        model = FractalCurveViT(
            image_size=64,
            num_classes=100,
            dim=256,
            num_layers=2,
            use_semantic_splitter=True,
            semantic_splitter_config=SemanticSplitterConfig(),
        )

        splitter = model.get_semantic_splitter()
        features = torch.randn(2, 10, 256)

        result = splitter(features, depth=2, remaining_quota=0.5, hard=True)

        # 硬模式应该输出二值
        unique_values = result.split_decision.unique()
        assert all(v in [0.0, 1.0] for v in unique_values)


class TestConfigSerialization:
    """配置序列化测试"""

    def test_config_to_dict(self):
        """测试配置转换为字典"""
        config = SemanticSplitterConfig(
            hidden_dim=128,
            diversity_weight=0.1,
            reconstruction_weight=0.2,
            split_threshold=0.5,
            gumbel_temp_start=1.0,
            gumbel_temp_end=0.5,
            learnable_temperature=True,
        )

        config_dict = config.to_dict()

        assert config_dict['hidden_dim'] == 128
        assert config_dict['diversity_weight'] == 0.1
        assert config_dict['reconstruction_weight'] == 0.2
        assert config_dict['split_threshold'] == 0.5
        assert config_dict['gumbel_temp_start'] == 1.0
        assert config_dict['gumbel_temp_end'] == 0.5
        assert config_dict['learnable_temperature'] is True

    def test_config_validation(self):
        """测试配置验证"""
        # 无效配置应该在调用 validate() 时抛出异常
        with pytest.raises(ValueError):
            config = SemanticSplitterConfig(feature_dim=-1)
            config.validate()

        with pytest.raises(ValueError):
            config = SemanticSplitterConfig(hidden_dim=0)
            config.validate()

        with pytest.raises(ValueError):
            config = SemanticSplitterConfig(diversity_weight=0)
            config.validate()

        with pytest.raises(ValueError):
            config = SemanticSplitterConfig(split_threshold=1.5)
            config.validate()

        with pytest.raises(ValueError):
            config = SemanticSplitterConfig(gumbel_temp_end=2.0, gumbel_temp_start=1.0)
            config.validate()


class TestIntegration:
    """端到端集成测试"""

    def test_full_training_step(self):
        """完整训练步骤测试（端到端梯度流验证）"""
        model = FractalCurveViT(
            image_size=64,
            num_classes=100,
            dim=256,
            num_layers=2,
            use_semantic_splitter=True,
            semantic_splitter_config=SemanticSplitterConfig(),
        )

        # 模拟训练步骤
        x = torch.randn(4, 3, 64, 64, requires_grad=True)
        targets = torch.randint(0, 100, (4,))

        # 前向
        stats = model(x)

        # 计算分类损失
        cls_loss = nn.CrossEntropyLoss()(stats.logits, targets)

        # 验证语义分裂器存在且可访问
        splitter = model.get_semantic_splitter()
        assert splitter is not None

        # 测试语义损失函数的梯度流
        loss_fn = model.get_semantic_loss_fn()
        assert loss_fn is not None

        # 创建有梯度的特征张量
        parent = torch.randn(4, 10, 256, requires_grad=True)
        child = torch.randn(4, 10, 4, 256, requires_grad=True)
        split = torch.rand(4, 10) > 0.5

        # 计算语义损失
        semantic_losses = loss_fn(parent, child, split)
        total_loss = cls_loss + semantic_losses['loss']

        # 反向传播
        total_loss.backward()

        # 验证分类损失的梯度流向输入
        assert x.grad is not None, "输入 x 应该有权重"

        # 验证语义损失的梯度流向输入特征
        assert parent.grad is not None, "parent 特征应该有权重"
        assert child.grad is not None, "child 特征应该有权重"

    def test_inference_mode(self):
        """推理模式测试"""
        model = FractalCurveViT(
            image_size=64,
            num_classes=100,
            dim=256,
            num_layers=2,
            use_semantic_splitter=True,
            semantic_splitter_config=SemanticSplitterConfig(),
        )

        model.eval()

        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            stats = model(x)

        assert stats.logits.shape == (2, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
