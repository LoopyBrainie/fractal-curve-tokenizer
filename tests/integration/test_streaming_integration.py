# -*- coding: utf-8 -*-
"""ARCH-P1 集成测试: StreamingFractalTokenizer 与 FractalCurveViT 的集成.

测试目标：
1. 验证 tokenizer_type 参数正确选择 tokenizer
2. 验证 streaming tokenizer 路径的前向传播
3. 验证 get_tokenizer_loss() 在 streaming 模式下返回零
4. 验证输出形状正确性
5. 验证梯度正确流动（Gumbel-Softmax 可微分）
"""

import pytest
import torch
import torch.nn as nn

from vit_pytorch import FractalCurveViT
from vit_pytorch.streaming_tokenizer import (
    StreamingFractalTokenizer,
    StreamingFractalTokenizerV2,
)


class TestTokenizerTypeSelection:
    """测试 tokenizer_type 参数的选择逻辑."""

    def test_streaming_tokenizer_selection(self) -> None:
        """测试 streaming tokenizer 被正确选择."""
        model = FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=2,
            mlp_dim=128,
            min_patch_size=(4, 4),
            max_level=5,
            tokenizer_type="streaming",
            num_scales=3,
        )
        
        assert model.tokenizer_type == "streaming"
        assert model._is_streaming
        assert isinstance(model.tokenizer, StreamingFractalTokenizer)

    def test_streaming_v2_tokenizer_selection(self) -> None:
        """测试 streaming_v2 tokenizer 被正确选择."""
        model = FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=2,
            mlp_dim=128,
            min_patch_size=(4, 4),
            max_level=5,
            tokenizer_type="streaming_v2",
            num_scales=3,
            streaming_tau=0.5,
        )
        
        assert model.tokenizer_type == "streaming_v2"
        assert model._is_streaming
        assert isinstance(model.tokenizer, StreamingFractalTokenizerV2)

    def test_default_tokenizer_is_streaming_v2(self) -> None:
        """测试默认 tokenizer 是 streaming_v2."""
        model = FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=2,
            mlp_dim=128,
            min_patch_size=(4, 4),
            max_level=5,
        )
        
        assert model.tokenizer_type == "streaming_v2"
        assert model._is_streaming


class TestForwardPass:
    """测试不同 tokenizer_type 的前向传播."""

    @pytest.fixture
    def batch_input(self) -> torch.Tensor:
        """创建测试输入."""
        return torch.randn(2, 3, 32, 32)

    def test_streaming_forward(self, batch_input: torch.Tensor) -> None:
        """测试 streaming tokenizer 的前向传播."""
        model = FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=2,
            mlp_dim=128,
            min_patch_size=(4, 4),
            max_level=5,
            tokenizer_type="streaming",
            num_scales=2,
        )
        
        output = model(batch_input)
        
        assert output.shape == (2, 10)
        assert not torch.isnan(output).any()

    def test_streaming_v2_forward(self, batch_input: torch.Tensor) -> None:
        """测试 streaming_v2 tokenizer 的前向传播."""
        model = FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=2,
            mlp_dim=128,
            min_patch_size=(4, 4),
            max_level=5,
            tokenizer_type="streaming_v2",
            num_scales=2,
        )
        
        output = model(batch_input)
        
        assert output.shape == (2, 10)
        assert not torch.isnan(output).any()


class TestTokenizerLoss:
    """测试 get_tokenizer_loss() 在不同 tokenizer 下的行为."""

    def test_streaming_tokenizer_loss_is_zero(self) -> None:
        """测试 streaming tokenizer 返回零损失."""
        model = FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=2,
            mlp_dim=128,
            min_patch_size=(4, 4),
            max_level=5,
            tokenizer_type="streaming",
        )
        
        x = torch.randn(2, 3, 32, 32)
        _ = model(x)
        
        loss = model.get_tokenizer_loss(reward=1.0)
        
        assert loss.item() == 0.0

    def test_streaming_v2_tokenizer_loss_is_zero(self) -> None:
        """测试 streaming_v2 tokenizer 返回零损失."""
        model = FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=2,
            mlp_dim=128,
            min_patch_size=(4, 4),
            max_level=5,
            tokenizer_type="streaming_v2",
        )
        
        x = torch.randn(2, 3, 32, 32)
        _ = model(x)
        
        loss = model.get_tokenizer_loss(reward=1.0)
        
        assert loss.item() == 0.0


class TestGradientFlow:
    """测试梯度流动."""

    def test_streaming_gradient_flows(self) -> None:
        """测试 streaming tokenizer 的梯度正确流动."""
        model = FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=2,
            mlp_dim=128,
            min_patch_size=(4, 4),
            max_level=5,
            tokenizer_type="streaming",
        )
        
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        has_gradient = False
        for name, param in model.tokenizer.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_gradient = True
                break
        
        assert has_gradient, "At least one tokenizer parameter should receive gradients"

    def test_streaming_v2_gumbel_softmax_gradient(self) -> None:
        """测试 streaming_v2 的 Gumbel-Softmax 梯度流动."""
        model = FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=2,
            mlp_dim=128,
            min_patch_size=(4, 4),
            max_level=5,
            tokenizer_type="streaming_v2",
        )
        
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # 验证复杂度头有梯度（重构后使用 complexity_head 替代 complexity_estimator）
        has_gradient = False
        for param in model.tokenizer.complexity_head.parameters():
            if param.grad is not None:
                has_gradient = True
                break
        
        assert has_gradient, "Complexity head should receive gradients"


class TestOutputConsistency:
    """测试输出一致性和形状正确性."""

    def test_aux_info_with_streaming(self) -> None:
        """测试 streaming tokenizer 的辅助信息输出."""
        model = FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=2,
            mlp_dim=128,
            min_patch_size=(4, 4),
            max_level=5,
            tokenizer_type="streaming",
        )
        
        x = torch.randn(2, 3, 32, 32)
        output, aux_info = model(x, return_aux_info=True)
        
        assert output.shape == (2, 10)
        assert len(aux_info) == 2
        assert "num_tokens" in aux_info[0]

    def test_features_with_streaming(self) -> None:
        """测试 streaming tokenizer 的特征输出."""
        model = FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=2,
            mlp_dim=128,
            min_patch_size=(4, 4),
            max_level=5,
            tokenizer_type="streaming",
        )
        
        x = torch.randn(2, 3, 32, 32)
        output, features = model(x, return_features=True)
        
        assert output.shape == (2, 10)
        assert len(features) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
