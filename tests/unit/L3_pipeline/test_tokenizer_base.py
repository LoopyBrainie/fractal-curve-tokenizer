# -*- coding: utf-8 -*-
"""
L3 Pipeline Tests: Tokenizer Base Components

对应模块:
- vit_pytorch.curve_hilbert_indexer (HilbertIndexer)
- vit_pytorch.embed_multiscale_patch (MultiScalePatchEncoder)

测试内容:
1. HilbertIndexer - Hilbert 索引器
   H: [0, n²) → [0, n) × [0, n)

2. MultiScalePatchEncoder - 多尺度特征提取
   F_s = Conv(I, kernel=s)
"""

import pytest
import torch

from vit_pytorch import (
    StreamingFractalTokenizerV3,
    HilbertIndexer,
    MultiScalePatchEncoder,
    TokenizerOutput,
)
from vit_pytorch.gumbel_topk_splitter import GumbelTopKSplitter


# I98-1: Pipeline API 辅助函数
def create_v3_pipeline(image_size, max_level=4):
    """创建 V3 tokenizer-splitter pipeline (I98-1 解耦架构)."""
    tokenizer = StreamingFractalTokenizerV3(
        image_size=image_size,
        channels=3,
        d_model=64,
        base_patch_size=4,
        max_level=max_level,
    )
    splitter = GumbelTopKSplitter(
        feature_dim=64,
        min_patch_size=4,
        max_level_limit=max_level,
        hidden_dim=64,
        intermediate_dim=64,
        pool_size=4,
        K_min=8,
        K_max=256,
        use_dynamic_k=True,
    )
    return tokenizer, splitter


def run_v3_pipeline(tokenizer, splitter, images, hard=True):
    """运行完整的 V3 tokenization pipeline."""
    features = tokenizer.patch_embed.shared_conv(images)
    split_result = splitter(features, images.shape[2:], hard=hard)
    output = tokenizer.tokenize(images, split_result)
    return output


class TestHilbertIndexer:
    """测试 Hilbert 索引器."""

    def test_get_hilbert_order_power_of_2(self):
        """测试 2 的幂次网格的 Hilbert 顺序."""
        order = HilbertIndexer.get_hilbert_order(4)
        assert len(order) == 16
        assert set(order.tolist()) == set(range(16))

    def test_get_hilbert_order_small(self):
        """测试小网格."""
        order = HilbertIndexer.get_hilbert_order(2)
        assert len(order) == 4
        assert set(order.tolist()) == set(range(4))

    def test_get_hilbert_order_cached(self):
        """测试缓存功能."""
        order1 = HilbertIndexer.get_hilbert_order(4)
        order2 = HilbertIndexer.get_hilbert_order(4)
        assert torch.equal(order1, order2)

    def test_reorder_to_hilbert_square(self):
        """测试正方形特征图的重排."""
        B, D, H, W = 2, 32, 4, 4
        features = torch.randn(B, D, H, W)

        reordered = HilbertIndexer.reorder_to_hilbert(features, H, W)

        assert reordered.shape == (B, H * W, D)

    def test_reorder_to_hilbert_preserves_content(self):
        """验证重排后内容不变（只是顺序变化）."""
        B, D, H, W = 1, 4, 2, 2
        features = torch.arange(D * H * W).float().view(1, D, H, W)

        reordered = HilbertIndexer.reorder_to_hilbert(features, H, W)

        original_flat = features.flatten(2).sort(dim=2)[0]
        reordered_flat = reordered.transpose(1, 2).sort(dim=2)[0]
        assert torch.allclose(original_flat, reordered_flat)


class TestMultiScalePatchEncoder:
    """测试多尺度编码器."""

    def test_basic_encoding(self):
        """测试基础编码功能."""
        encoder = MultiScalePatchEncoder(
            channels=3,
            d_model=64,
            patch_sizes=(4, 8),
        )

        images = torch.randn(2, 3, 32, 32)
        features_dict = encoder(images)

        assert 4 in features_dict
        assert 8 in features_dict

        feat_4, (h_4, w_4) = features_dict[4]
        assert feat_4.shape == (2, 64, 8, 8)

        feat_8, (h_8, w_8) = features_dict[8]
        assert feat_8.shape == (2, 64, 4, 4)

    def test_image_too_small(self):
        """测试图像小于 patch 尺寸的情况."""
        encoder = MultiScalePatchEncoder(
            channels=3,
            d_model=64,
            patch_sizes=(8, 16),
        )

        images = torch.randn(2, 3, 12, 12)
        features_dict = encoder(images)

        assert 8 in features_dict
        assert 16 not in features_dict


class TestStreamingFractalTokenizerV3:
    """测试 V3 Variable Depth Tokenizer.

    I98-1: 使用 Pipeline API (tokenizer + splitter)
    """

    @pytest.fixture
    def pipeline(self):
        """创建 V3 pipeline (I98-1 解耦架构)."""
        return create_v3_pipeline(image_size=32, max_level=4)

    def test_tokenize_basic(self, pipeline):
        """测试 V3 基础 tokenization (Pipeline API)."""
        tokenizer, splitter = pipeline
        images = torch.randn(2, 3, 32, 32)
        output = run_v3_pipeline(tokenizer, splitter, images)

        assert isinstance(output, TokenizerOutput)
        assert len(output) == 2

    def test_gradient_flow_v3(self, pipeline):
        """测试 V3 梯度流动 (Pipeline API)."""
        tokenizer, splitter = pipeline
        images = torch.randn(1, 3, 32, 32, requires_grad=True)

        tokenizer.train()
        output = run_v3_pipeline(tokenizer, splitter, images)

        loss = output.sequences[0].tokens.sum()
        loss.backward()

        assert images.grad is not None
        assert not torch.isnan(images.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
