# -*- coding: utf-8 -*-
"""Unit tests for StreamingFractalTokenizer and StreamingFractalTokenizerV3.

数学形式化验证
==============

流式 Tokenizer 将图像映射到变长 token 序列:

    T: I → (tokens, levels_info)
    
其中:
    I ∈ R^{B × C × H × W}      输入图像
    tokens ∈ R^{B × N × D}     Token 嵌入
    levels_info ∈ Z^{B × N × L} 层级信息

测试覆盖:
1. HilbertPathCache - Hilbert 路径缓存机制
2. HilbertIndexer - Hilbert 索引重排
3. MultiScalePatchEncoder - 多尺度特征提取
4. StreamingFractalTokenizer - V1 流式 tokenizer
5. StreamingFractalTokenizerV3 - V3 Variable Depth tokenizer
"""

import pytest
import torch

from vit_pytorch import (
    StreamingFractalTokenizer,
    StreamingFractalTokenizerV3,
    HilbertIndexer,
    HilbertPathCache,
    MultiScalePatchEncoder,
    TokenizerOutput,
)


class TestHilbertPathCache:
    """测试统一的 Hilbert 路径缓存.
    
    数学性质:
    - Hilbert 曲线: H: [0, n²) → [0, n) × [0, n)
    - 双射性: H 是一一映射
    - 局部性: |H(d₁) - H(d₂)|₂ ≤ C·|d₁ - d₂|^(1/2)
    """
    
    def setup_method(self):
        """每个测试前清空缓存."""
        HilbertPathCache.clear_cache()
    
    def test_basic_cache(self):
        """测试基本缓存功能."""
        hilbert_to_raster, paths = HilbertPathCache.get_or_compute(4, 4, 8)
        
        assert hilbert_to_raster.shape == (16,)
        assert paths.shape == (16, 8)
        assert set(hilbert_to_raster.tolist()) == set(range(16))
    
    def test_cache_hit(self):
        """测试缓存命中."""
        h2r1, p1 = HilbertPathCache.get_or_compute(4, 4, 8)
        h2r2, p2 = HilbertPathCache.get_or_compute(4, 4, 8)
        
        assert torch.equal(h2r1, h2r2)
        assert torch.equal(p1, p2)
    
    def test_non_square_grid(self):
        """测试非正方形网格."""
        hilbert_to_raster, paths = HilbertPathCache.get_or_compute(4, 8, 8)
        
        assert hilbert_to_raster.shape == (32,)  # 4 * 8 = 32
        assert paths.shape == (32, 8)
    
    def test_quadtree_paths_values(self):
        """测试四叉树路径值在 [0, 3] 范围内.
        
        四叉树路径: q_l ∈ {0, 1, 2, 3}
        0=左上, 1=右上, 2=左下, 3=右下
        """
        _, paths = HilbertPathCache.get_or_compute(8, 8, 8)
        
        assert paths.min() >= 0
        assert paths.max() <= 3
    
    def test_hilbert_to_raster_bijection(self):
        """测试 Hilbert 到光栅映射是双射."""
        hilbert_to_raster, _ = HilbertPathCache.get_or_compute(4, 4, 8)
        
        # 应该是 [0, 15] 的排列
        sorted_indices = hilbert_to_raster.sort().values
        expected = torch.arange(16)
        assert torch.equal(sorted_indices, expected)
    
    def test_cache_eviction(self):
        """测试缓存淘汰机制."""
        # 填充缓存到上限
        for i in range(70):  # 超过 64 的上限
            HilbertPathCache.get_or_compute(2 + i % 30, 2 + i % 30, 8)
        
        # 缓存大小不应超过上限
        assert len(HilbertPathCache._cache) <= HilbertPathCache._max_cache_size
    
    def test_clear_cache(self):
        """测试缓存清空."""
        HilbertPathCache.get_or_compute(4, 4, 8)
        assert len(HilbertPathCache._cache) > 0
        
        HilbertPathCache.clear_cache()
        assert len(HilbertPathCache._cache) == 0


class TestHilbertIndexer:
    """测试 Hilbert 索引器.
    
    功能: 将 2D 特征图按 Hilbert 顺序重排为 1D 序列
    """
    
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
        
        # 展平后的内容应该是相同的（只是顺序不同）
        original_flat = features.flatten(2).sort(dim=2)[0]
        reordered_flat = reordered.transpose(1, 2).sort(dim=2)[0]
        assert torch.allclose(original_flat, reordered_flat)


class TestMultiScalePatchEncoder:
    """测试多尺度编码器.
    
    数学: F_s = Conv(I, kernel=s) ∈ R^{B × D × H/s × W/s}
    """
    
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
    """测试流式 Tokenizer V1.
    
    数学形式化:
    - 输入: I ∈ R^{B × C × H × W}
    - 输出: tokens ∈ R^{B × N × D}, levels_info ∈ Z^{B × N × L}
    - N = (H/s) × (W/s) 固定 token 数量
    """
    
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
    
    def test_invalid_input_dimension(self, tokenizer):
        """测试无效输入维度."""
        with pytest.raises(ValueError, match="expects 4D input"):
            tokenizer.tokenize(torch.randn(3, 32, 32))
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_if_available(self, tokenizer):
        """测试 GPU 支持."""
        tokenizer = tokenizer.cuda()
        images = torch.randn(2, 3, 32, 32).cuda()
        
        output = tokenizer.tokenize(images)
        
        assert output.sequences[0].tokens.device.type == 'cuda'


class TestStreamingFractalTokenizerV3:
    """测试 V3 Variable Depth Tokenizer.
    
    数学形式化:
    - 自适应分割: R → {R₁, ..., Rₖ} (四叉树)
    - Variable tokens: N ∈ [N_min, N_max]
    - 深度调制: t_i = Pool(F[R_i]) · σ_d + E_d
    """
    
    @pytest.fixture
    def tokenizer_v3(self):
        return StreamingFractalTokenizerV3(
            image_size=32,
            channels=3,
            d_model=64,
            base_patch_size=4,
            max_depth=4,
        )
    
    def test_basic_forward(self, tokenizer_v3):
        """测试基础前向传播."""
        images = torch.randn(2, 3, 32, 32)
        output = tokenizer_v3.tokenize(images)
        
        assert isinstance(output, TokenizerOutput)
        assert len(output) == 2
        
        for seq in output.sequences:
            assert seq.tokens.dim() == 2
            assert seq.tokens.shape[1] == 64
    
    def test_levels_info_format(self, tokenizer_v3):
        """测试 levels_info 格式正确."""
        images = torch.randn(1, 3, 32, 32)
        output = tokenizer_v3.tokenize(images)
        
        levels_info = output.sequences[0].get_levels()
        if levels_info is not None:
            num_tokens = output.sequences[0].tokens.shape[0]
            assert levels_info.shape[0] == num_tokens
            
            depths = levels_info[:, 0]
            assert depths.min() >= 0
    
    def test_gradient_flow(self, tokenizer_v3):
        """测试梯度流动."""
        images = torch.randn(1, 3, 32, 32, requires_grad=True)
        
        tokenizer_v3.train()
        output = tokenizer_v3.tokenize(images)
        
        loss = output.sequences[0].tokens.sum()
        loss.backward()
        
        assert images.grad is not None
        assert not torch.isnan(images.grad).any()
    
    def test_metadata_contains_num_tokens(self, tokenizer_v3):
        """测试元数据包含 token 数量."""
        images = torch.randn(1, 3, 32, 32)
        output = tokenizer_v3.tokenize(images)
        
        metadata = output.sequences[0].metadata
        # num_tokens 在 split_stats 子字典中
        assert 'split_stats' in metadata
        assert 'num_tokens' in metadata['split_stats']
        assert metadata['split_stats']['num_tokens'] == output.sequences[0].tokens.shape[0]
        """测试 Tokenizer 输出可以作为 Transformer 输入."""
        tokenizer = StreamingFractalTokenizer(
            image_size=32,
            channels=3,
            d_model=64,
            patch_sizes=(4,),
        )
        
        images = torch.randn(2, 3, 32, 32)
        output = tokenizer.tokenize(images)
        
        legacy = output.to_legacy()
        
        tokens_padded = torch.nn.utils.rnn.pad_sequence(
            legacy.tokens, batch_first=True
        )
        levels_padded = torch.nn.utils.rnn.pad_sequence(
            legacy.levels, batch_first=True
        )
        
        B, S, D = tokens_padded.shape
        assert B == 2
        assert D == 64
        assert levels_padded.shape[0] == B
        assert levels_padded.shape[1] == S
    
    def test_v1_and_v3_output_structure_compatible(self):
        """验证 V1 和 V3 tokenizer 的输出结构兼容."""
        streaming_v1 = StreamingFractalTokenizer(
            image_size=32,
            channels=3,
            d_model=64,
            patch_sizes=(4,),
        )
        
        streaming_v3 = StreamingFractalTokenizerV3(
            image_size=32,
            channels=3,
            d_model=64,
            base_patch_size=4,
            max_depth=3,
        )
        
        images = torch.randn(1, 3, 32, 32)
        
        output_v1 = streaming_v1.tokenize(images)
        output_v3 = streaming_v3.tokenize(images)
        
        assert isinstance(output_v1, TokenizerOutput)
        assert isinstance(output_v3, TokenizerOutput)
        
        assert output_v1.sequences[0].tokens.shape[1] == output_v3.sequences[0].tokens.shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
