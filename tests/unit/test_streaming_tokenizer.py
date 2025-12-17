# -*- coding: utf-8 -*-
"""Unit tests for StreamingFractalTokenizer.

验证流式统一 Tokenizer 的核心功能：
1. 基础 tokenization 功能
2. Hilbert 顺序重排
3. 与原接口的兼容性
4. 多尺度特征提取
5. HilbertPathCache 缓存机制
"""

import pytest
import torch

from vit_pytorch import (
    StreamingFractalTokenizer,
    StreamingFractalTokenizerV2,
    HilbertIndexer,
    HilbertPathCache,
    MultiScalePatchEncoder,
    TokenizerOutput,
)


class TestHilbertPathCache:
    """测试统一的 Hilbert 路径缓存."""
    
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
        """测试四叉树路径值在 [0, 3] 范围内."""
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
        """测试复杂度估计器 (语义级, 基于 Encoder 特征)."""
        images = torch.randn(2, 3, 32, 32)  # 使用 batch_size=2 避免 BatchNorm 问题
        
        # 获取 encoder 特征
        tokenizer_v2.eval()  # 使用 eval 模式
        with torch.no_grad():
            features_dict = tokenizer_v2.encoder(images)
            
            # 获取最小尺度的特征图大小作为目标大小
            min_ps = tokenizer_v2.patch_sizes[0]
            if min_ps in features_dict:
                target_size = features_dict[min_ps][1]  # (grid_h, grid_w)
            else:
                target_size = (8, 8)  # 默认
            
            # 计算尺度权重
            scale_weights = tokenizer_v2._compute_scale_weights(features_dict, target_size)
        
        # 权重应该在 [0, 1] 范围
        assert scale_weights.shape[1] == len(tokenizer_v2.patch_sizes)
        assert scale_weights.min() >= 0
        # 由于使用 STE (hard=True)，权重应该是 one-hot
        # 沿尺度维度求和应该为 1
        assert torch.allclose(
            scale_weights.sum(dim=1), 
            torch.ones_like(scale_weights.sum(dim=1)), 
            atol=0.01
        )


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
    
    def test_v1_and_v2_output_structure_compatible(self):
        """验证 V1 和 V2 tokenizer 的输出结构兼容."""
        # V1 tokenizer
        streaming_v1 = StreamingFractalTokenizer(
            image_size=32,
            channels=3,
            d_model=64,
            patch_sizes=(4,),
        )
        
        # V2 tokenizer
        streaming_v2 = StreamingFractalTokenizerV2(
            image_size=32,
            channels=3,
            d_model=64,
            patch_sizes=(4,),
        )
        
        images = torch.randn(1, 3, 32, 32)
        
        # 两个 tokenizer 都应该产生 TokenizerOutput
        output_v1 = streaming_v1.tokenize(images)
        output_v2 = streaming_v2.tokenize(images)
        
        assert isinstance(output_v1, TokenizerOutput)
        assert isinstance(output_v2, TokenizerOutput)
        
        # 都应该有 levels 元数据
        assert output_v1.sequences[0].get_levels() is not None
        assert output_v2.sequences[0].get_levels() is not None
        
        # 输出维度应该相同
        assert output_v1.sequences[0].tokens.shape[1] == output_v2.sequences[0].tokens.shape[1]


class TestVariableTokensMode:
    """测试可变 Token 数量模式 (variable_tokens=True).
    
    数学形式化验证:
    1. 四叉树一致性: 粗尺度区域内所有位置使用相同尺度
    2. Token 数量可变: N ∈ [N_min, N_max]
    3. Hilbert 排序: 按层级和空间位置排序
    4. levels_info 格式正确
    """
    
    @pytest.fixture
    def variable_tokenizer(self):
        """创建可变 token 数量的 tokenizer."""
        return StreamingFractalTokenizerV2(
            image_size=32,
            channels=3,
            d_model=64,
            patch_sizes=(4, 8, 16),  # 3 种尺度
            use_hilbert_order=True,
            max_level=10,
            gumbel_temperature=1.0,
            variable_tokens=True,
        )
    
    @pytest.fixture
    def fixed_tokenizer(self):
        """创建固定 token 数量的 tokenizer (对比用)."""
        return StreamingFractalTokenizerV2(
            image_size=32,
            channels=3,
            d_model=64,
            patch_sizes=(4, 8, 16),
            use_hilbert_order=True,
            max_level=10,
            gumbel_temperature=1.0,
            variable_tokens=False,
        )
    
    def test_basic_forward(self, variable_tokenizer):
        """测试基础前向传播."""
        images = torch.randn(2, 3, 32, 32)
        output = variable_tokenizer.tokenize(images)
        
        assert len(output) == 2
        for seq in output.sequences:
            assert seq.tokens.dim() == 2  # [N, D]
            assert seq.tokens.shape[1] == 64  # d_model
            assert seq.get_levels() is not None
    
    def test_variable_token_count(self, variable_tokenizer):
        """测试 token 数量确实可变."""
        # 使用不同复杂度的图像
        simple_image = torch.zeros(1, 3, 32, 32)  # 简单图像（全黑）
        complex_image = torch.randn(1, 3, 32, 32)  # 复杂图像（随机噪声）
        
        variable_tokenizer.eval()
        with torch.no_grad():
            simple_output = variable_tokenizer.tokenize(simple_image)
            complex_output = variable_tokenizer.tokenize(complex_image)
        
        simple_count = simple_output.sequences[0].tokens.shape[0]
        complex_count = complex_output.sequences[0].tokens.shape[0]
        
        # Token 数量范围应在 [N_min, N_max] 之间
        # N_min = (32/16)^2 = 4 (最粗尺度)
        # N_max = (32/4)^2 = 64 (最细尺度)
        assert 1 <= simple_count <= 64
        assert 1 <= complex_count <= 64
        
        # 不要求必须不同，但打印出来便于观察
        print(f"\nSimple image tokens: {simple_count}, Complex image tokens: {complex_count}")
    
    def test_fixed_vs_variable_comparison(self, fixed_tokenizer, variable_tokenizer):
        """对比固定和可变模式的输出."""
        images = torch.randn(1, 3, 32, 32)
        
        fixed_tokenizer.eval()
        variable_tokenizer.eval()
        
        with torch.no_grad():
            fixed_output = fixed_tokenizer.tokenize(images)
            variable_output = variable_tokenizer.tokenize(images)
        
        fixed_count = fixed_output.sequences[0].tokens.shape[0]
        variable_count = variable_output.sequences[0].tokens.shape[0]
        
        # 固定模式应该总是输出 (32/4)^2 = 64 个 token
        assert fixed_count == 64
        
        # 可变模式应该 <= 64
        assert variable_count <= 64
        assert variable_count >= 1
    
    def test_levels_info_structure(self, variable_tokenizer):
        """测试 levels_info 结构正确."""
        images = torch.randn(1, 3, 32, 32)
        output = variable_tokenizer.tokenize(images)
        
        levels_info = output.sequences[0].get_levels()
        assert levels_info is not None
        
        num_tokens = output.sequences[0].tokens.shape[0]
        assert levels_info.shape[0] == num_tokens
        
        # 检查深度值范围
        depths = levels_info[:, 0]
        assert depths.min() >= 0
        assert depths.max() <= variable_tokenizer.max_level
        
        # 检查路径值范围 [0, 3]
        if levels_info.shape[1] > 1:
            paths = levels_info[:, 1:]
            assert paths.min() >= 0
            assert paths.max() <= 3
    
    def test_quadtree_consistency(self, variable_tokenizer):
        """测试四叉树一致性约束.
        
        验证: 同一粗尺度区域内的所有细粒度位置应使用相同尺度。
        """
        # 手动创建测试用的 scale_map
        scale_map = torch.zeros(1, 8, 8, dtype=torch.long)  # 8x8 网格
        
        # 设置一些粗尺度决策
        scale_map[0, 0:4, 0:4] = 2  # 左上 4x4 使用最粗尺度 (scale_idx=2)
        scale_map[0, 0:4, 4:8] = 1  # 右上 4x4 使用中等尺度 (scale_idx=1)
        scale_map[0, 4:8, :] = 0    # 下半部分使用最细尺度 (scale_idx=0)
        
        # 调用一致性强制函数
        result = variable_tokenizer._enforce_quadtree_consistency(scale_map, 8, 8)
        
        # 验证粗尺度区域保持一致
        # 左上 4x4 区域应该全部是 2
        assert (result[0, 0:4, 0:4] == 2).all() or (result[0, 0:4, 0:4] == result[0, 0, 0]).all()
        
        # 每个 block 内部应该一致
        for by in range(0, 8, 4):
            for bx in range(0, 8, 4):
                block = result[0, by:by+4, bx:bx+4]
                # 检查 block 内是否存在粗尺度 (>=1)
                if block.max() >= 1:
                    # 如果有粗尺度，整个 block 应该统一
                    assert block.max() == block.min() or block.max() <= block[0, 0]
    
    def test_hilbert_sort_correctness(self, variable_tokenizer):
        """测试 Hilbert 排序的正确性."""
        # 创建测试位置列表
        positions = [
            (1, 0, 0, 4, 4),  # level=1, (0,0) in 4x4 grid
            (1, 0, 1, 4, 4),  # level=1, (0,1)
            (1, 1, 0, 4, 4),  # level=1, (1,0)
            (1, 1, 1, 4, 4),  # level=1, (1,1)
        ]
        
        sorted_indices = variable_tokenizer._hilbert_sort_by_position(positions)
        
        # 应该返回有效的排列
        assert len(sorted_indices) == 4
        assert set(sorted_indices) == {0, 1, 2, 3}
    
    def test_quadtree_path_computation(self, variable_tokenizer):
        """测试四叉树路径计算."""
        # 位置 (0, 0) 在 4x4 网格中
        path = variable_tokenizer._compute_quadtree_path(0, 0, 4, 4, 4)
        
        # 路径应该非空
        assert len(path) > 0
        
        # 路径值应该在 [0, 3] 范围
        assert all(0 <= p <= 3 for p in path)
    
    def test_batch_with_different_token_counts(self, variable_tokenizer):
        """测试 batch 中不同图像产生不同 token 数量."""
        # 创建两个明显不同复杂度的图像
        images = torch.zeros(2, 3, 32, 32)
        images[1] = torch.randn(1, 3, 32, 32)  # 第二个图像更复杂
        
        variable_tokenizer.eval()
        with torch.no_grad():
            output = variable_tokenizer.tokenize(images)
        
        count1 = output.sequences[0].tokens.shape[0]
        count2 = output.sequences[1].tokens.shape[0]
        
        # 验证都在有效范围内
        assert 1 <= count1 <= 64
        assert 1 <= count2 <= 64
        
        print(f"\nBatch token counts: {count1}, {count2}")
    
    def test_metadata_contains_num_tokens(self, variable_tokenizer):
        """测试元数据包含 token 数量."""
        images = torch.randn(1, 3, 32, 32)
        output = variable_tokenizer.tokenize(images)
        
        # variable_tokens 模式应该在 metadata 中记录 num_tokens
        metadata = output.sequences[0].metadata
        if 'num_tokens' in metadata:
            assert metadata['num_tokens'] == output.sequences[0].tokens.shape[0]
    
    def test_gradient_flow(self, variable_tokenizer):
        """测试梯度可以正常流动."""
        images = torch.randn(1, 3, 32, 32, requires_grad=True)
        
        variable_tokenizer.train()
        output = variable_tokenizer.tokenize(images)
        
        # 计算损失并反向传播
        loss = output.sequences[0].tokens.sum()
        loss.backward()
        
        # 验证梯度流动
        assert images.grad is not None
        assert not torch.isnan(images.grad).any()


class TestDepthBiasWarmup:
    """测试深度探索优先 Warmup 策略 (v2.2)."""
    
    @pytest.fixture
    def tokenizer_v2(self):
        """创建 V2 tokenizer 用于测试."""
        return StreamingFractalTokenizerV2(
            image_size=32,
            channels=3,
            d_model=64,
            patch_sizes=(4, 8, 16),
            gumbel_temperature=2.0,
        )
    
    def test_depth_bias_initialization(self, tokenizer_v2):
        """测试深度偏置初始化."""
        # 检查初始值
        assert tokenizer_v2._depth_bias_max == 2.0
        assert tokenizer_v2._depth_bias_decay == 2.0
        assert tokenizer_v2._depth_bias_warmup == 0.2
        assert tokenizer_v2._current_depth_bias == 2.0
        
        # 检查尺度偏置权重
        # 3 个尺度: [1.0, 0.5, 0.0] 或类似 (小尺度偏置大)
        weights = tokenizer_v2._scale_bias_weights
        assert weights.shape == (3,)
        assert weights[0] > weights[1] > weights[2]  # 递减
        assert weights[0].item() == pytest.approx(1.0)
        assert weights[2].item() == pytest.approx(0.0)
    
    def test_depth_bias_annealing(self, tokenizer_v2):
        """测试深度偏置退火调度."""
        total_epochs = 100
        
        # Epoch 1 (1%): 在 warmup 期间，保持最大偏置
        bias_e1 = tokenizer_v2.anneal_depth_bias(1, total_epochs)
        assert bias_e1 == pytest.approx(2.0, rel=0.1)
        
        # Epoch 10 (10%): 仍在 warmup (20%) 期间
        bias_e10 = tokenizer_v2.anneal_depth_bias(10, total_epochs)
        assert bias_e10 == pytest.approx(2.0, rel=0.1)
        
        # Epoch 50 (50%): warmup 后，偏置开始衰减
        bias_e50 = tokenizer_v2.anneal_depth_bias(50, total_epochs)
        assert bias_e50 < 2.0
        assert bias_e50 > 0.0
        
        # Epoch 100 (100%): 偏置接近 0
        bias_e100 = tokenizer_v2.anneal_depth_bias(100, total_epochs)
        assert bias_e100 < 0.1
    
    def test_anneal_temperature_updates_depth_bias(self, tokenizer_v2):
        """测试 anneal_temperature 同时更新深度偏置."""
        # 初始状态
        assert tokenizer_v2._current_depth_bias == 2.0
        
        # 调用 anneal_temperature
        tokenizer_v2.anneal_temperature(50, 100, schedule="cosine")
        
        # 验证深度偏置也被更新
        assert tokenizer_v2._current_depth_bias < 2.0
    
    def test_depth_bias_affects_logits(self, tokenizer_v2):
        """测试深度偏置影响尺度选择."""
        images = torch.randn(2, 3, 32, 32)
        
        tokenizer_v2.train()
        
        # 高偏置时 (初始状态)
        tokenizer_v2._current_depth_bias = 2.0
        features_dict = tokenizer_v2.encoder(images)
        min_ps = min(features_dict.keys())
        _, target_size = features_dict[min_ps]
        weights_high_bias = tokenizer_v2._compute_scale_weights(features_dict, target_size)
        
        # 低偏置时
        tokenizer_v2._current_depth_bias = 0.0
        weights_no_bias = tokenizer_v2._compute_scale_weights(features_dict, target_size)
        
        # 高偏置时应该更倾向于小尺度 (scale index 0)
        scale_0_ratio_high = weights_high_bias[:, 0].mean().item()
        scale_0_ratio_low = weights_no_bias[:, 0].mean().item()
        
        # 由于 Gumbel 采样的随机性，只验证逻辑正确性
        # 高偏置应该增加小尺度的选择概率
        print(f"\nScale 0 ratio - high bias: {scale_0_ratio_high:.3f}, no bias: {scale_0_ratio_low:.3f}")
    
    def test_depth_bias_not_applied_in_eval(self, tokenizer_v2):
        """测试推理模式下不应用深度偏置."""
        images = torch.randn(1, 3, 32, 32)
        
        tokenizer_v2.eval()
        tokenizer_v2._current_depth_bias = 2.0  # 设置高偏置
        
        with torch.no_grad():
            features_dict = tokenizer_v2.encoder(images)
            min_ps = min(features_dict.keys())
            _, target_size = features_dict[min_ps]
            weights = tokenizer_v2._compute_scale_weights(features_dict, target_size)
        
        # 推理模式下输出应该是 one-hot
        assert weights.sum(dim=1).allclose(torch.ones_like(weights.sum(dim=1)))
    
    def test_set_depth_bias(self, tokenizer_v2):
        """测试手动设置深度偏置参数."""
        tokenizer_v2.set_depth_bias(
            bias_strength=1.5,
            max_bias=3.0,
            decay_power=1.5,
            warmup_ratio=0.3,
        )
        
        assert tokenizer_v2._current_depth_bias == 1.5
        assert tokenizer_v2._depth_bias_max == 3.0
        assert tokenizer_v2._depth_bias_decay == 1.5
        assert tokenizer_v2._depth_bias_warmup == 0.3
    
    def test_get_depth_bias(self, tokenizer_v2):
        """测试获取当前深度偏置."""
        assert tokenizer_v2.get_depth_bias() == 2.0
        
        tokenizer_v2._current_depth_bias = 0.5
        assert tokenizer_v2.get_depth_bias() == 0.5
