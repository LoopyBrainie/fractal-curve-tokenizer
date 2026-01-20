"""I30-11: 混合池化策略单元测试

验证三种池化策略 (cls, mean, weighted) 的数学正确性。
"""
import pytest
import torch

from vit_pytorch import FractalCurveViT


class TestPoolingStrategies:
    """池化策略单元测试类"""

    @pytest.fixture
    def model_cls(self):
        """创建使用 cls 池化的模型"""
        return FractalCurveViT(
            image_size=224,
            num_classes=100,
            dim=384,
            depth=2,
            heads=4,
            pool="cls",
        )

    @pytest.fixture
    def model_mean(self):
        """创建使用 mean 池化的模型"""
        return FractalCurveViT(
            image_size=224,
            num_classes=100,
            dim=384,
            depth=2,
            heads=4,
            pool="mean",
        )

    @pytest.fixture
    def model_weighted(self):
        """创建使用 weighted 池化的模型"""
        return FractalCurveViT(
            image_size=224,
            num_classes=100,
            dim=384,
            depth=2,
            heads=4,
            pool="weighted",
        )

    @pytest.fixture
    def sample_input(self):
        """创建样本输入"""
        batch_size = 2
        seq_len = 10
        dim = 384
        return torch.randn(batch_size, seq_len, dim)

    @pytest.fixture
    def sample_mask(self):
        """创建样本 padding mask (全部有效)"""
        batch_size = 2
        seq_len = 10
        return torch.zeros(batch_size, seq_len, dtype=torch.bool)

    @pytest.fixture
    def sample_mask_with_padding(self):
        """创建带 padding 的 mask"""
        return torch.tensor([
            [False, False, False, False, False, True, True, True, True, True],
            [False, False, False, False, False, False, False, True, True, True],
        ])

    @pytest.fixture
    def sample_split_probs(self):
        """创建样本分割概率"""
        batch_size = 2
        seq_len = 9  # 排除 CLS 后 9 个 tokens
        return torch.rand(batch_size, seq_len)

    def test_cls_pooling_basic(self, model_cls, sample_input):
        """测试 CLS Token 池化基础功能"""
        batch_size = sample_input.shape[0]
        key_padding_mask = torch.zeros(batch_size, sample_input.shape[1], dtype=torch.bool)

        result = model_cls._apply_pooling(sample_input, key_padding_mask)

        expected = sample_input[:, 0]  # 只取 CLS token
        assert result.shape == (batch_size, 384)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_cls_pooling_with_padding(self, model_cls, sample_input, sample_mask_with_padding):
        """测试 CLS Token 池化带 padding mask"""
        result = model_cls._apply_pooling(sample_input, sample_mask_with_padding)
        expected = sample_input[:, 0]

        assert result.shape == (2, 384)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_mean_pooling_basic(self, model_mean, sample_input, sample_mask):
        """测试平均池化基础功能"""
        batch_size = sample_input.shape[0]
        result = model_mean._apply_pooling(sample_input, sample_mask)

        expected = sample_input[:, 1:].mean(dim=1)  # 排除 CLS 后平均
        assert result.shape == (batch_size, 384)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_mean_pooling_with_padding(self, model_mean, sample_input, sample_mask_with_padding):
        """测试平均池化带 padding mask"""
        result = model_mean._apply_pooling(sample_input, sample_mask_with_padding)

        # 手动计算期望值
        token_x = sample_input[:, 1:]  # [B, 9, D]
        token_mask = ~sample_mask_with_padding[:, 1:]  # [B, 9]
        masked_x = token_x * token_mask.unsqueeze(-1).float()
        expected = masked_x.sum(dim=1) / token_mask.sum(dim=1, keepdim=True).float()

        assert result.shape == (2, 384)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_weighted_pooling_basic(self, model_weighted, sample_input, sample_mask, sample_split_probs):
        """测试加权池化基础功能"""
        batch_size = sample_input.shape[0]
        result = model_weighted._apply_pooling(sample_input, sample_mask, sample_split_probs)

        # 手动计算期望值
        token_x = sample_input[:, 1:]  # [B, 9, D]
        token_probs = sample_split_probs  # [B, 9]
        weights = token_probs / token_probs.sum(dim=-1, keepdim=True)
        expected = (token_x * weights.unsqueeze(-1)).sum(dim=1)

        assert result.shape == (batch_size, 384)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_weighted_pooling_with_padding(self, model_weighted, sample_input, sample_mask_with_padding, sample_split_probs):
        """测试加权池化带 padding mask"""
        result = model_weighted._apply_pooling(sample_input, sample_mask_with_padding, sample_split_probs)

        # 手动计算期望值 (带 mask)
        token_x = sample_input[:, 1:]  # [B, 9, D]
        token_mask = ~sample_mask_with_padding[:, 1:]  # [B, 9]
        token_probs = sample_split_probs * token_mask.float()
        weights = token_probs / token_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        expected = (token_x * weights.unsqueeze(-1)).sum(dim=1)

        assert result.shape == (2, 384)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_weighted_pooling_fallback_to_mean(self, model_weighted, sample_input, sample_mask):
        """测试 weighted 模式在 split_probs=None 时回退到 mean"""
        result = model_weighted._apply_pooling(sample_input, sample_mask, split_probs=None)

        expected = sample_input[:, 1:].mean(dim=1)
        assert result.shape == (2, 384)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_weighted_pooling_uniform_weights(self, model_weighted, sample_input, sample_mask):
        """测试加权池化在均匀权重下等价于平均池化"""
        batch_size = sample_input.shape[0]
        uniform_probs = torch.ones(batch_size, 9) / 9  # 均匀权重

        result = model_weighted._apply_pooling(sample_input, sample_mask, uniform_probs)
        expected = sample_input[:, 1:].mean(dim=1)

        assert result.shape == (batch_size, 384)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_weighted_pooling_single_token(self, model_weighted, sample_input, sample_mask):
        """测试加权池化只有一个有效 token 时"""
        batch_size = sample_input.shape[0]
        # 只让第一个 token 有概率
        probs = torch.zeros(batch_size, 9)
        probs[:, 0] = 1.0

        result = model_weighted._apply_pooling(sample_input, sample_mask, probs)
        expected = sample_input[:, 1]  # 只取第一个 token

        assert result.shape == (batch_size, 384)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_weighted_pooling_gradient_flow(self, model_weighted, sample_input, sample_mask, sample_split_probs):
        """测试加权池化的梯度流动"""
        sample_input.requires_grad_(True)
        sample_split_probs.requires_grad_(True)

        result = model_weighted._apply_pooling(sample_input, sample_mask, sample_split_probs)
        loss = result.sum()
        loss.backward()

        assert sample_input.grad is not None
        assert sample_split_probs.grad is not None
        assert not torch.all(sample_input.grad == 0)
        assert not torch.all(sample_split_probs.grad == 0)

    def test_pooling_output_shape(self, model_cls, model_mean, model_weighted, sample_input, sample_mask, sample_split_probs):
        """测试所有池化策略输出形状一致性"""
        result_cls = model_cls._apply_pooling(sample_input, sample_mask)
        result_mean = model_mean._apply_pooling(sample_input, sample_mask)
        result_weighted = model_weighted._apply_pooling(sample_input, sample_mask, sample_split_probs)

        assert result_cls.shape == result_mean.shape == result_weighted.shape
        assert result_cls.shape == (2, 384)

    def test_pooling_with_different_batch_sizes(self, model_weighted):
        """测试不同 batch size 下的池化"""
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 10, 384)
            probs = torch.rand(batch_size, 9)
            mask = torch.zeros(batch_size, 10, dtype=torch.bool)

            result = model_weighted._apply_pooling(x, mask, probs)
            assert result.shape == (batch_size, 384)

    def test_pooling_with_different_dims(self, sample_mask, sample_split_probs):
        """测试不同特征维度下的池化"""
        for dim in [128, 256, 512, 768]:
            model = FractalCurveViT(
                image_size=224,
                num_classes=100,
                dim=dim,
                depth=2,
                heads=4,
                pool="weighted",
            )
            x = torch.randn(2, 10, dim)
            probs = torch.rand(2, 9)

            result = model._apply_pooling(x, sample_mask, probs)
            assert result.shape == (2, dim)


class TestPoolingIntegration:
    """池化策略集成测试"""

    def test_model_forward_with_different_pooling(self):
        """测试模型前向传播在不同池化策略下正常"""
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)

        for pool_type in ["cls", "mean", "weighted"]:
            model = FractalCurveViT(
                image_size=224,
                num_classes=100,
                dim=128,
                depth=2,
                heads=4,
                pool=pool_type,
            )
            output = model(images)
            assert output.shape == (batch_size, 100)

    def test_weighted_pooling_integration(self):
        """测试 weighted 池化在完整模型中正常工作"""
        batch_size = 2
        images = torch.randn(batch_size, 3, 64, 64)

        model = FractalCurveViT(
            image_size=64,
            num_classes=100,
            dim=128,
            depth=2,
            heads=4,
            pool="weighted",
        )

        # 前向传播
        output, aux_infos = model(images, return_aux_info=True)

        # 验证输出
        assert output.shape == (batch_size, 100)
        assert len(aux_infos) == batch_size
        assert "num_tokens" in aux_infos[0]
        assert "levels_used" in aux_infos[0]
