# -*- coding: utf-8 -*-
"""
测试 FractalConfig 和 FractalPath 模块
"""

import math

import pytest
import torch

from vit_pytorch import (
    FractalConfig,
    FractalPathEmbedding,
    HierarchicalAttentionBias,
    VectorizedPathEncoder,
    create_fractal_config,
)


class TestFractalConfig:
    """测试 FractalConfig 参数推导."""
    
    def test_basic_config_64_4(self) -> None:
        """测试 64×64 图像, min_patch=4 的配置."""
        config = FractalConfig(64, 4)
        
        assert config.image_size == 64
        assert config.min_patch_size == 4
        assert config.max_depth == 4  # log2(64/4) = 4
        assert config.num_scales == 5  # 0, 1, 2, 3, 4
        assert config.patch_sizes == (4, 8, 16, 32, 64)
        assert config.grid_size == 16  # 64/4
        assert config.num_tokens == 256  # 16^2
    
    def test_basic_config_256_16(self) -> None:
        """测试 256×256 图像, min_patch=16 的配置."""
        config = FractalConfig(256, 16)
        
        assert config.image_size == 256
        assert config.min_patch_size == 16
        assert config.max_depth == 4  # log2(256/16) = 4
        assert config.num_scales == 5
        assert config.patch_sizes == (16, 32, 64, 128, 256)
        assert config.grid_size == 16
        assert config.num_tokens == 256
    
    def test_basic_config_32_4(self) -> None:
        """测试 32×32 图像 (CIFAR) 的配置."""
        config = FractalConfig(32, 4)
        
        assert config.max_depth == 3  # log2(32/4) = 3
        assert config.num_scales == 4
        assert config.patch_sizes == (4, 8, 16, 32)
        assert config.grid_size == 8
        assert config.num_tokens == 64
    
    def test_scale_depth_conversion(self) -> None:
        """测试尺度和深度之间的转换."""
        config = FractalConfig(64, 4)
        
        # scale_idx=0 (最细 4×4) → depth=4
        assert config.scale_to_depth(0) == 4
        # scale_idx=4 (最粗 64×64) → depth=0
        assert config.scale_to_depth(4) == 0
        
        # 反向转换
        assert config.depth_to_scale(4) == 0
        assert config.depth_to_scale(0) == 4
    
    def test_patch_size_at_scale(self) -> None:
        """测试获取指定尺度的 patch 大小."""
        config = FractalConfig(64, 4)
        
        assert config.patch_size_at_scale(0) == 4
        assert config.patch_size_at_scale(2) == 16
        assert config.patch_size_at_scale(4) == 64
    
    def test_grid_size_at_scale(self) -> None:
        """测试获取指定尺度的网格大小."""
        config = FractalConfig(64, 4)
        
        assert config.grid_size_at_scale(0) == 16  # 64/4
        assert config.grid_size_at_scale(2) == 4   # 64/16
        assert config.grid_size_at_scale(4) == 1   # 64/64
    
    def test_invalid_divisibility(self) -> None:
        """测试不可整除时抛出异常."""
        with pytest.raises(ValueError, match="必须能被"):
            FractalConfig(65, 4)
    
    def test_non_2k_grid_size_allowed(self) -> None:
        """测试非 2^k grid_size 现在可以使用 (Pseudo-Hilbert)."""
        # grid_size = 60 / 5 = 12，不是 2 的幂
        # 现在支持，使用 Pseudo-Hilbert
        config = FractalConfig(60, 5)
        assert config.grid_size == 12
        assert config.uses_pseudo_hilbert  # padding_ratio > 4/3
        
        # 96 / 6 = 16 是 2 的幂，使用标准 Hilbert
        config2 = FractalConfig(96, 6)
        assert config2.grid_size == 16
        assert not config2.uses_pseudo_hilbert
    
    def test_convenience_function(self) -> None:
        """测试便捷函数."""
        config = create_fractal_config(64, 4)
        assert isinstance(config, FractalConfig)
        assert config.max_depth == 4
    
    def test_config_repr(self) -> None:
        """测试配置的字符串表示."""
        config = FractalConfig(64, 4)
        repr_str = repr(config)
        assert 'FractalConfig' in repr_str
        assert 'image_size=64' in repr_str


class TestVectorizedPathEncoder:
    """测试向量化路径编码器."""
    
    def test_compute_quadrant_paths_basic(self) -> None:
        """测试基本的四叉树路径计算."""
        # 2×2 网格，max_depth=1
        x = torch.tensor([0, 1, 0, 1])
        y = torch.tensor([0, 0, 1, 1])
        
        paths = VectorizedPathEncoder.compute_quadrant_paths(x, y, max_depth=1)
        
        # 象限: (0,0)→0, (1,0)→1, (0,1)→2, (1,1)→3
        expected = torch.tensor([[0], [1], [2], [3]])
        assert torch.equal(paths, expected)
    
    def test_compute_quadrant_paths_deeper(self) -> None:
        """测试更深的四叉树路径."""
        # 4×4 网格，max_depth=2
        # (0,0): path=[0,0], (3,3): path=[3,3]
        x = torch.tensor([0, 3])
        y = torch.tensor([0, 3])
        
        paths = VectorizedPathEncoder.compute_quadrant_paths(x, y, max_depth=2)
        
        assert paths.shape == (2, 2)
        assert paths[0].tolist() == [0, 0]  # 左上→左上
        assert paths[1].tolist() == [3, 3]  # 右下→右下
    
    def test_compute_common_ancestor_depth(self) -> None:
        """测试共同祖先深度计算."""
        # 路径: [0, 0], [0, 1], [1, 0]
        paths = torch.tensor([
            [0, 0],  # token 0
            [0, 1],  # token 1 - 与 token 0 共享前缀 [0]
            [1, 0],  # token 2 - 与 token 0, 1 无共享前缀
        ])
        
        common_depth = VectorizedPathEncoder.compute_common_ancestor_depth(paths)
        
        # common_depth[i, j] = 共同前缀长度
        assert common_depth[0, 0] == 2  # 自身
        assert common_depth[0, 1] == 1  # [0] 是公共前缀
        assert common_depth[0, 2] == 0  # 无公共前缀
        assert common_depth[1, 1] == 2  # 自身
        assert common_depth[1, 2] == 0  # 无公共前缀


class TestFractalPathEmbedding:
    """测试四叉树路径位置编码."""
    
    @pytest.fixture
    def config(self) -> FractalConfig:
        return FractalConfig(32, 4)  # 简单配置: 8×8 网格
    
    def test_init(self, config: FractalConfig) -> None:
        """测试初始化."""
        emb = FractalPathEmbedding(dim=64, config=config)
        
        assert emb.max_depth == 3
        assert emb.scale_embedding.num_embeddings == 4
        assert emb.quadrant_embedding.num_embeddings == 4 * 3  # 4 象限 × 3 层
    
    def test_forward_shape(self, config: FractalConfig) -> None:
        """测试前向传播输出形状."""
        emb = FractalPathEmbedding(dim=64, config=config)
        
        B, N = 2, 64
        scale_indices = torch.zeros(B, N, dtype=torch.long)  # 全部使用最细尺度
        
        output = emb(scale_indices)
        
        assert output.shape == (B, N, 64)
    
    def test_forward_different_scales(self, config: FractalConfig) -> None:
        """测试不同尺度产生不同编码."""
        emb = FractalPathEmbedding(dim=64, config=config)
        
        B, N = 1, 64
        scale_fine = torch.zeros(B, N, dtype=torch.long)
        scale_coarse = torch.full((B, N), config.num_scales - 1, dtype=torch.long)
        
        out_fine = emb(scale_fine)
        out_coarse = emb(scale_coarse)
        
        # 不同尺度应该产生不同的编码
        assert not torch.allclose(out_fine, out_coarse)


class TestHierarchicalAttentionBias:
    """测试层级注意力偏置."""
    
    @pytest.fixture
    def config(self) -> FractalConfig:
        return FractalConfig(32, 4)
    
    def test_init(self, config: FractalConfig) -> None:
        """测试初始化."""
        bias_module = HierarchicalAttentionBias(config=config, heads=4)
        
        assert bias_module.max_depth == 3
        assert bias_module.ancestor_bias.num_embeddings == 5  # max_depth + 2
    
    def test_forward_shape(self, config: FractalConfig) -> None:
        """测试前向传播输出形状."""
        bias_module = HierarchicalAttentionBias(config=config, heads=4)
        
        bias = bias_module(seq_len=64, batch_size=2)
        
        assert bias.shape == (2, 4, 64, 64)
    
    def test_symmetry(self, config: FractalConfig) -> None:
        """测试偏置矩阵的对称性."""
        bias_module = HierarchicalAttentionBias(config=config, heads=4)
        
        bias = bias_module(seq_len=64, batch_size=1)
        
        # 层级关系是对称的
        assert torch.allclose(bias, bias.transpose(-1, -2))
    
    def test_self_bias_highest(self, config: FractalConfig) -> None:
        """测试对角线 (自身) 偏置最高."""
        bias_module = HierarchicalAttentionBias(config=config, heads=4)
        
        bias = bias_module(seq_len=64, batch_size=1)
        
        # 对角线元素 (共同祖先深度 = max_depth) 应该有一致的值
        diag = torch.diagonal(bias[0, 0])
        assert torch.allclose(diag, diag[0].expand_as(diag))
