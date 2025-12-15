# -*- coding: utf-8 -*-
"""Unit tests for StreamingFractalTokenizer.

验证流式统一 Tokenizer 的核心功能：
1. 基础 tokenization 功能
2. Hilbert 顺序重排
3. 与原接口的兼容性
4. 多尺度特征提取
"""

import pytest
import torch

from vit_pytorch import (
    StreamingFractalTokenizer,
    StreamingFractalTokenizerV2,
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
        assert set(order.tolist()) == set(range(16))  # 包含所有索引
    
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
        
        # 展平后的内容应该是相同的（只是顺序不同）
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
        
        # 检查输出尺寸
        feat_4, (h_4, w_4) = features_dict[4]
        assert feat_4.shape == (2, 64, 8, 8)  # 32/4 = 8
        
        feat_8, (h_8, w_8) = features_dict[8]
        assert feat_8.shape == (2, 64, 4, 4)  # 32/8 = 4
    
    def test_image_too_small(self):
        """测试图像小于 patch 尺寸的情况."""
        encoder = MultiScalePatchEncoder(
            channels=3,
            d_model=64,
            patch_sizes=(8, 16),
        )
        
        images = torch.randn(2, 3, 12, 12)
        features_dict = encoder(images)
        
        # 只有 patch_size=8 有效
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
            primary_scale=0,  # 使用 patch_size=4
        )
    
    def test_tokenize_basic(self, tokenizer):
        """测试基础 tokenization."""
        images = torch.randn(2, 3, 32, 32)
        output = tokenizer.tokenize(images)
        
        assert isinstance(output, TokenizerOutput)
        assert len(output) == 2
        
        # 检查每个序列
        for seq in output:
            # patch_size=4, image=32 → grid=8x8=64 tokens
            assert seq.tokens.shape == (64, 64)  # (num_tokens, d_model)
            assert seq.get_levels() is not None
    
    def test_tokenize_output_format(self, tokenizer):
        """测试输出格式与原接口兼容."""
        images = torch.randn(1, 3, 32, 32)
        output = tokenizer.tokenize(images)
        
        # 测试 to_legacy 方法
        legacy = output.to_legacy()
        assert len(legacy.tokens) == 1
        assert len(legacy.levels) == 1
        
        # tokens 和 levels 维度匹配
        assert legacy.tokens[0].shape[0] == legacy.levels[0].shape[0]
    
    def test_forward_equals_tokenize(self, tokenizer):
        """测试 forward 和 tokenize 等价."""
        images = torch.randn(1, 3, 32, 32)
        
        # 使用 eval 模式避免 Dropout 随机性
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
        
        # levels_info[:, 0] 是深度
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
        assert output.sequences[0].tokens.shape[0] == 64  # 8x8=64
    
    def test_different_image_sizes(self):
        """测试不同图像尺寸."""
        tokenizer = StreamingFractalTokenizer(
            image_size=64,
            channels=3,
            d_model=64,
            patch_sizes=(8,),
        )
        
        # 使用与 image_size 不同的实际输入
        images = torch.randn(1, 3, 48, 48)
        output = tokenizer.tokenize(images)
        
        # 48/8 = 6, 6x6 = 36 tokens
        assert output.sequences[0].tokens.shape[0] == 36
    
    def test_invalid_input_dimension(self, tokenizer):
        """测试无效输入维度."""
        with pytest.raises(ValueError, match="expects 4D input"):
            tokenizer.tokenize(torch.randn(3, 32, 32))  # 3D instead of 4D
    
    def test_gpu_if_available(self, tokenizer):
        """测试 GPU 支持（如果可用）."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        tokenizer = tokenizer.cuda()
        images = torch.randn(2, 3, 32, 32).cuda()
        
        output = tokenizer.tokenize(images)
        
        assert output.sequences[0].tokens.device.type == 'cuda'


class TestStreamingFractalTokenizerV2:
    """测试带区域自适应的 Tokenizer V2."""
    
    @pytest.fixture
    def tokenizer_v2(self):
        return StreamingFractalTokenizerV2(
            image_size=32,
            channels=3,
            d_model=64,
            patch_sizes=(4, 8),
            gumbel_temperature=1.0,
        )
    
    def test_tokenize_basic(self, tokenizer_v2):
        """测试基础 tokenization."""
        images = torch.randn(2, 3, 32, 32)
        output = tokenizer_v2.tokenize(images)
        
        assert isinstance(output, TokenizerOutput)
        assert len(output) == 2
    
    def test_training_vs_eval_mode(self, tokenizer_v2):
        """测试训练和推理模式的差异."""
        images = torch.randn(1, 3, 32, 32)
        
        # 训练模式
        tokenizer_v2.train()
        output_train = tokenizer_v2.tokenize(images)
        
        # 推理模式
        tokenizer_v2.eval()
        with torch.no_grad():
            output_eval = tokenizer_v2.tokenize(images)
        
        # 两种模式都应该产生有效输出
        assert output_train.sequences[0].tokens.shape == output_eval.sequences[0].tokens.shape
    
    def test_complexity_estimator(self, tokenizer_v2):
        """测试复杂度估计器."""
        images = torch.randn(1, 3, 32, 32)
        
        # 访问内部方法
        scale_weights = tokenizer_v2._compute_scale_weights(images)
        
        # 权重应该在 [0, 1] 范围，且沿尺度维度求和为 1
        assert scale_weights.shape[1] == len(tokenizer_v2.patch_sizes)
        assert scale_weights.min() >= 0
        # 对于 Gumbel-Softmax（非硬模式），权重和接近 1
        assert torch.allclose(scale_weights.sum(dim=1), torch.ones_like(scale_weights.sum(dim=1)), atol=0.1)


class TestIntegration:
    """集成测试."""
    
    def test_streaming_tokenizer_with_transformer_input(self):
        """测试 Tokenizer 输出可以作为 Transformer 输入."""
        tokenizer = StreamingFractalTokenizer(
            image_size=32,
            channels=3,
            d_model=64,
            patch_sizes=(4,),
        )
        
        images = torch.randn(2, 3, 32, 32)
        output = tokenizer.tokenize(images)
        
        # 获取 tokens 和 levels
        legacy = output.to_legacy()
        
        # Pad 到相同长度（模拟 Transformer 输入准备）
        tokens_padded = torch.nn.utils.rnn.pad_sequence(
            legacy.tokens, batch_first=True
        )
        levels_padded = torch.nn.utils.rnn.pad_sequence(
            legacy.levels, batch_first=True
        )
        
        # 验证形状
        B, S, D = tokens_padded.shape
        assert B == 2
        assert D == 64
        assert levels_padded.shape[0] == B
        assert levels_padded.shape[1] == S
    
    def test_compare_output_structure_with_legacy(self):
        """验证输出结构与原 Tokenizer 兼容."""
        from vit_pytorch import FractalHilbertTokenizer
        
        # 原 tokenizer
        legacy_tokenizer = FractalHilbertTokenizer(
            min_patch_size=(4, 4),
            max_level=10,
            learnable_split=False,
        )
        
        # 新 tokenizer
        streaming_tokenizer = StreamingFractalTokenizer(
            image_size=32,
            channels=3,
            d_model=48,  # 3 * 4 * 4 = 48 (匹配原 tokenizer 输出维度)
            patch_sizes=(4,),
        )
        
        images = torch.randn(1, 3, 32, 32)
        
        # 两个 tokenizer 都应该产生 TokenizerOutput
        output_legacy = legacy_tokenizer.tokenize(images)
        output_streaming = streaming_tokenizer.tokenize(images)
        
        assert isinstance(output_legacy, TokenizerOutput)
        assert isinstance(output_streaming, TokenizerOutput)
        
        # 都应该有 levels 元数据
        assert output_legacy.sequences[0].get_levels() is not None
        assert output_streaming.sequences[0].get_levels() is not None
