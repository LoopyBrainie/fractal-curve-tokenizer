# -*- coding: utf-8 -*-
"""Unit tests for Tokenizer core components.

数学形式化验证
==============

Tokenizer 核心组件测试:

1. HilbertIndexer - Hilbert 索引器
   H: [0, n²) → [0, n) × [0, n)
   
2. MultiScalePatchEncoder - 多尺度特征提取
   F_s = Conv(I, kernel=s)
   
3. StreamingFractalTokenizer - 流式 tokenizer
4. StreamingFractalTokenizerV3 - Variable Depth tokenizer
"""

import pytest
import torch

from vit_pytorch import (
    StreamingFractalTokenizer,
    StreamingFractalTokenizerV3,
    HilbertIndexer,
    MultiScalePatchEncoder,
    TokenizerOutput,
)


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


class TestStreamingFractalTokenizer:
    """测试流式 Tokenizer."""
    
    @pytest.fixture
    def tokenizer(self):
        return StreamingFractalTokenizer(
            image_size=32,
            channels=3,
            d_model=64,
            patch_sizes=(4, 8),
            primary_scale=0,
        )
    
    def test_tokenize_basic(self, tokenizer):
        """测试基础 tokenization."""
        images = torch.randn(2, 3, 32, 32)
        output = tokenizer.tokenize(images)
        
        assert isinstance(output, TokenizerOutput)
        assert len(output) == 2
        
        for seq in output:
            assert seq.tokens.shape == (64, 64)
            assert seq.get_levels() is not None
    
    def test_tokenize_output_format(self, tokenizer):
        """测试输出格式与原接口兼容."""
        images = torch.randn(1, 3, 32, 32)
        output = tokenizer.tokenize(images)
        
        legacy = output.to_legacy()
        assert len(legacy.tokens) == 1
        assert len(legacy.levels) == 1
        
        assert legacy.tokens[0].shape[0] == legacy.levels[0].shape[0]
    
    def test_forward_equals_tokenize(self, tokenizer):
        """测试 forward 和 tokenize 等价."""
        images = torch.randn(1, 3, 32, 32)
        
        tokenizer.eval()
        with torch.no_grad():
            output1 = tokenizer.tokenize(images)
            output2 = tokenizer.forward(images)
        
        assert torch.equal(output1.sequences[0].tokens, output2.sequences[0].tokens)
    
    def test_levels_info_structure(self, tokenizer):
        """测试 levels_info 的结构."""
        images = torch.randn(1, 3, 32, 32)
        output = tokenizer.tokenize(images)
        
        levels_info = output.sequences[0].get_levels()
        assert levels_info is not None
        
        depths = levels_info[:, 0]
        assert depths.min() >= 0
        assert depths.max() <= tokenizer.max_level
    
    def test_hilbert_order_disabled(self):
        """测试禁用 Hilbert 顺序."""
        tokenizer = StreamingFractalTokenizer(
            image_size=32,
            channels=3,
            d_model=64,
            patch_sizes=(4,),
            use_hilbert_order=False,
        )
        
        images = torch.randn(1, 3, 32, 32)
        output = tokenizer.tokenize(images)
        
        assert len(output) == 1
        assert output.sequences[0].tokens.shape[0] == 64
    
    def test_different_image_sizes(self):
        """测试不同图像尺寸."""
        tokenizer = StreamingFractalTokenizer(
            image_size=64,
            channels=3,
            d_model=64,
            patch_sizes=(8,),
        )
        
        images = torch.randn(1, 3, 48, 48)
        output = tokenizer.tokenize(images)
        
        assert output.sequences[0].tokens.shape[0] == 36


class TestStreamingFractalTokenizerV3:
    """测试 V3 Variable Depth Tokenizer."""
    
    @pytest.fixture
    def tokenizer_v3(self):
        return StreamingFractalTokenizerV3(
            image_size=32,
            channels=3,
            d_model=64,
            base_patch_size=4,
            max_depth=4,
        )
    
    def test_tokenize_basic(self, tokenizer_v3):
        """测试 V3 基础 tokenization."""
        images = torch.randn(2, 3, 32, 32)
        output = tokenizer_v3.tokenize(images)
        
        assert isinstance(output, TokenizerOutput)
        assert len(output) == 2
    
    def test_gradient_flow_v3(self, tokenizer_v3):
        """测试 V3 梯度流动."""
        images = torch.randn(1, 3, 32, 32, requires_grad=True)
        
        tokenizer_v3.train()
        output = tokenizer_v3.tokenize(images)
        
        loss = output.sequences[0].tokens.sum()
        loss.backward()
        
        assert images.grad is not None
        assert not torch.isnan(images.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


