# -*- coding: utf-8 -*-
"""
L4 Application Tests: 架构一致性 (I112)

测试内容:
- 模型 forward 返回 TrainingStats 类型
- TrainingStats 包含所有必需字段
- 训练器能正确处理模型输出
- TrainingStats.validate() 数学约束

对应模块: vit_pytorch.model_fractal_vit
"""

import pytest
import torch

from vit_pytorch import FractalCurveViT
from vit_pytorch.model_fractal_vit import TrainingStats


class TestModelOutputInterface:
    """模型输出接口测试."""

    def test_model_returns_trainingstats_type(self):
        """验证模型 forward 返回 TrainingStats 类型."""
        model = FractalCurveViT(
            image_size=224,
            num_classes=1000,
            dim=256,
            depth=4,
            heads=4,
        )
        x = torch.randn(2, 3, 224, 224)

        output = model(x)

        assert isinstance(output, TrainingStats), \
            f"模型应返回 TrainingStats，实际返回 {type(output)}"

    def test_trainingstats_required_fields(self):
        """验证 TrainingStats 包含所有必需字段."""
        model = FractalCurveViT(
            image_size=224,
            num_classes=1000,
            dim=256,
            depth=4,
            heads=4,
        )
        x = torch.randn(2, 3, 224, 224)

        output = model(x)

        required_fields = [
            'logits',
            'num_tokens',
            'depth_used',
            'depth_distribution',
            'features',
            'transformer_tokens',
        ]
        for field in required_fields:
            assert hasattr(output, field), f"TrainingStats 缺少字段: {field}"
            assert getattr(output, field) is not None, f"字段 {field} 不能为 None"

    def test_trainingstats_optional_fields_exist(self):
        """验证 TrainingStats 可选字段存在且有默认值."""
        model = FractalCurveViT(
            image_size=224,
            num_classes=1000,
            dim=256,
            depth=4,
            heads=4,
        )
        x = torch.randn(2, 3, 224, 224)

        output = model(x)

        # 可选字段应有默认值
        assert hasattr(output, 'shared_features')
        assert hasattr(output, 'splitter_entropy')
        assert hasattr(output, 'temperature')
        assert hasattr(output, 'aux_infos')
        assert hasattr(output, 'ema_stats')
        assert hasattr(output, 'split_info')

    def test_trainingstats_tensor_shapes(self):
        """验证 TrainingStats Tensor 形状正确."""
        batch_size = 4
        num_classes = 100
        dim = 256
        model = FractalCurveViT(
            image_size=224,
            num_classes=num_classes,
            dim=dim,
            depth=4,
            heads=4,
        )
        x = torch.randn(batch_size, 3, 224, 224)

        output = model(x)

        assert output.logits.shape == (batch_size, num_classes)
        assert output.features.shape == (batch_size, dim)
        assert output.transformer_tokens.ndim == 3  # [B, N, dim]
        assert output.num_tokens > 0
        assert 0 <= output.depth_used <= 50


class TestTrainingStatsValidation:
    """TrainingStats 数学约束验证测试."""

    def test_validate_valid_stats(self):
        """验证有效统计能通过验证."""
        valid_stats = TrainingStats(
            logits=torch.randn(2, 1000),
            num_tokens=100,
            depth_used=4,
            depth_distribution={0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4},
            features=torch.randn(2, 512),
            transformer_tokens=torch.randn(2, 100, 512),
        )
        # 不应抛出异常
        valid_stats.validate()

    def test_validate_invalid_num_tokens(self):
        """验证无效 num_tokens 会被检测."""
        invalid_stats = TrainingStats(
            logits=torch.randn(2, 1000),
            num_tokens=5000,  # > 4096
            depth_used=4,
            depth_distribution={0: 1.0},
            features=torch.randn(2, 512),
            transformer_tokens=torch.randn(2, 100, 512),
        )
        with pytest.raises(AssertionError, match="Token 数异常"):
            invalid_stats.validate()

    def test_validate_invalid_depth(self):
        """验证无效 depth_used 会被检测."""
        invalid_stats = TrainingStats(
            logits=torch.randn(2, 1000),
            num_tokens=100,
            depth_used=60,  # > 50
            depth_distribution={0: 1.0},
            features=torch.randn(2, 512),
            transformer_tokens=torch.randn(2, 100, 512),
        )
        with pytest.raises(AssertionError, match="深度越界"):
            invalid_stats.validate()

    def test_validate_unnormalized_distribution(self):
        """验证未归一化分布会被检测."""
        invalid_stats = TrainingStats(
            logits=torch.randn(2, 1000),
            num_tokens=100,
            depth_used=2,
            depth_distribution={0: 0.5, 1: 0.3},  # 总和 0.8 != 1.0
            features=torch.randn(2, 512),
            transformer_tokens=torch.randn(2, 100, 512),
        )
        with pytest.raises(AssertionError, match="分布未归一化"):
            invalid_stats.validate()


class TestBackwardCompatibility:
    """向后兼容性测试."""

    def test_trainingstats_with_optional_fields(self):
        """验证 TrainingStats 可接受所有可选字段."""
        stats = TrainingStats(
            logits=torch.randn(2, 1000),
            num_tokens=100,
            depth_used=4,
            depth_distribution={0: 1.0},
            features=torch.randn(2, 512),
            transformer_tokens=torch.randn(2, 100, 512),
            shared_features=torch.randn(2, 128, 14, 14),
            splitter_entropy=0.5,
            temperature=0.8,
            aux_infos=[{'num_tokens': 100}],
            ema_stats=torch.randn(10),
            split_info={'region_0': [0, 1, 2]},
        )

        assert stats.shared_features is not None
        assert stats.splitter_entropy == 0.5
        assert stats.temperature == 0.8
        assert stats.aux_infos == [{'num_tokens': 100}]
        assert stats.ema_stats is not None
        assert stats.split_info == {'region_0': [0, 1, 2]}

    def test_trainingstats_default_values(self):
        """验证 TrainingStats 默认值正确."""
        stats = TrainingStats(
            logits=torch.randn(2, 1000),
            num_tokens=100,
            depth_used=4,
            depth_distribution={0: 1.0},
            features=torch.randn(2, 512),
            transformer_tokens=torch.randn(2, 100, 512),
        )

        assert stats.shared_features is None
        assert stats.splitter_entropy == 0.0
        assert stats.temperature == 1.0
        assert stats.aux_infos is None
        assert stats.ema_stats is None
        assert stats.split_info == {}
