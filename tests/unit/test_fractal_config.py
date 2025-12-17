# -*- coding: utf-8 -*-
"""FractalConfig 单元测试."""

import math
import pytest
import sys
sys.path.insert(0, 'src')

from vit_pytorch import (
    FractalConfig,
    create_fractal_config,
    BiasMode,
    AnnealSchedule,
    StreamingFractalTokenizerV2,
)


class TestFractalConfigBasic:
    """基础功能测试."""
    
    def test_default_config(self):
        """测试默认配置创建."""
        config = FractalConfig(64, 4)
        
        assert config.image_size == 64
        assert config.min_patch_size == 4
        assert config.max_depth == 4  # log2(64/4) = 4
        assert config.num_scales == 5
        assert config.patch_sizes == (4, 8, 16, 32, 64)
        assert config.grid_size == 16
        assert config.num_tokens == 256
    
    def test_gumbel_defaults(self):
        """测试 Gumbel-Softmax 默认参数."""
        config = FractalConfig(64, 4)
        
        assert config.gumbel_tau_init == 2.0
        assert config.gumbel_tau_min == 0.5
        assert config.gumbel_tau_max == 5.0
        assert config.gumbel_anneal_schedule == 'cosine'
    
    def test_hilbert_defaults(self):
        """测试 Hilbert Bias 默认参数."""
        config = FractalConfig(64, 4)
        
        assert config.hilbert_bias_mode == 'lca'
        assert config.low_rank_r == 32
    
    def test_tokenizer_defaults(self):
        """测试 Tokenizer 默认参数."""
        config = FractalConfig(64, 4)
        
        assert config.variable_tokens is False
        assert config.use_soft_weights is False


class TestFractalConfigCustomization:
    """自定义配置测试."""
    
    def test_custom_gumbel_params(self):
        """测试自定义 Gumbel 参数."""
        config = FractalConfig(
            64, 4,
            gumbel_tau_init=1.5,
            gumbel_tau_min=0.3,
            gumbel_tau_max=3.0,
            gumbel_anneal_schedule='exponential',
        )
        
        assert config.gumbel_tau_init == 1.5
        assert config.gumbel_tau_min == 0.3
        assert config.gumbel_tau_max == 3.0
        assert config.gumbel_anneal_schedule == 'exponential'
    
    def test_custom_hilbert_params(self):
        """测试自定义 Hilbert 参数."""
        config = FractalConfig(
            64, 4,
            hilbert_bias_mode='low_rank',
            low_rank_r=64,
        )
        
        assert config.hilbert_bias_mode == 'low_rank'
        assert config.low_rank_r == 64
    
    def test_variable_tokens_enabled(self):
        """测试启用 variable_tokens."""
        config = FractalConfig(64, 4, variable_tokens=True)
        
        assert config.variable_tokens is True
    
    def test_create_fractal_config_helper(self):
        """测试便捷函数."""
        config = create_fractal_config(
            64, 4,
            gumbel_tau_init=1.0,
            variable_tokens=True,
        )
        
        assert config.image_size == 64
        assert config.gumbel_tau_init == 1.0
        assert config.variable_tokens is True


class TestFractalConfigValidation:
    """参数验证测试."""
    
    def test_invalid_image_size(self):
        """测试无效图像尺寸."""
        with pytest.raises(ValueError, match="必须为正数"):
            FractalConfig(0, 4)
        
        with pytest.raises(ValueError, match="必须为正数"):
            FractalConfig(-64, 4)
    
    def test_invalid_patch_size(self):
        """测试无效 patch 尺寸."""
        with pytest.raises(ValueError, match="必须为正数"):
            FractalConfig(64, 0)
    
    def test_non_2k_grid_size_allowed(self):
        """测试非 2^k grid_size 现在可以使用 (使用 Pseudo-Hilbert)."""
        # 60 / 5 = 12 (不是 2 的幂) → 现在支持，使用 Pseudo-Hilbert
        config = FractalConfig(60, 5)
        assert config.grid_size == 12
        assert config.uses_pseudo_hilbert  # 因为 padding_ratio > 4/3
    
    def test_non_power_of_two_patch_size_allowed(self):
        """测试非 2 幂 patch 尺寸可以使用."""
        # 96 / 6 = 16 = 2^4，应该允许 (标准 Hilbert)
        config = FractalConfig(96, 6)
        assert config.grid_size == 16
        assert config.num_tokens == 256
        assert not config.uses_pseudo_hilbert
        
        # 192 / 12 = 16 = 2^4，应该允许 (标准 Hilbert)
        config2 = FractalConfig(192, 12)
        assert config2.grid_size == 16
        assert not config2.uses_pseudo_hilbert
    
    def test_not_divisible(self):
        """测试不可整除情况."""
        with pytest.raises(ValueError, match="必须能被"):
            FractalConfig(64, 5)  # 64 不能被 5 整除
    
    def test_invalid_tau_range(self):
        """测试无效温度范围."""
        with pytest.raises(ValueError, match="必须小于"):
            FractalConfig(64, 4, gumbel_tau_min=3.0, gumbel_tau_max=1.0)
    
    def test_tau_init_out_of_range(self):
        """测试初始温度超出范围."""
        with pytest.raises(ValueError, match="必须在"):
            FractalConfig(64, 4, gumbel_tau_init=10.0)  # 超出默认上界 5.0


class TestFractalConfigMethods:
    """辅助方法测试."""
    
    def test_scale_to_depth(self):
        """测试尺度到深度转换."""
        config = FractalConfig(64, 4)  # max_depth = 4
        
        assert config.scale_to_depth(0) == 4  # 最细尺度
        assert config.scale_to_depth(4) == 0  # 最粗尺度
    
    def test_depth_to_scale(self):
        """测试深度到尺度转换."""
        config = FractalConfig(64, 4)
        
        assert config.depth_to_scale(4) == 0
        assert config.depth_to_scale(0) == 4
    
    def test_patch_size_at_scale(self):
        """测试获取尺度对应的 patch 大小."""
        config = FractalConfig(64, 4)
        
        assert config.patch_size_at_scale(0) == 4
        assert config.patch_size_at_scale(4) == 64
    
    def test_grid_size_at_scale(self):
        """测试获取尺度对应的网格大小."""
        config = FractalConfig(64, 4)
        
        assert config.grid_size_at_scale(0) == 16  # 64 / 4
        assert config.grid_size_at_scale(4) == 1   # 64 / 64


class TestTemperatureAnneal:
    """温度退火测试."""
    
    def test_linear_anneal(self):
        """测试线性退火."""
        config = FractalConfig(
            64, 4,
            gumbel_tau_init=2.0,
            gumbel_tau_min=0.5,
            gumbel_tau_max=5.0,
            gumbel_anneal_schedule='linear',
        )
        
        # 开始时
        tau_start = config.compute_temperature(0, 100)
        assert abs(tau_start - 5.0) < 0.01
        
        # 结束时
        tau_end = config.compute_temperature(99, 100)
        assert abs(tau_end - 0.5) < 0.01
        
        # 中间点
        tau_mid = config.compute_temperature(49, 100)
        expected_mid = 5.0 - (5.0 - 0.5) * 0.5
        assert abs(tau_mid - expected_mid) < 0.1
    
    def test_exponential_anneal(self):
        """测试指数退火."""
        config = FractalConfig(
            64, 4,
            gumbel_tau_min=0.5,
            gumbel_tau_max=5.0,
            gumbel_anneal_schedule='exponential',
        )
        
        tau_start = config.compute_temperature(0, 100)
        tau_end = config.compute_temperature(99, 100)
        
        assert abs(tau_start - 5.0) < 0.01
        assert abs(tau_end - 0.5) < 0.1
    
    def test_cosine_anneal(self):
        """测试余弦退火."""
        config = FractalConfig(
            64, 4,
            gumbel_tau_min=0.5,
            gumbel_tau_max=5.0,
            gumbel_anneal_schedule='cosine',
        )
        
        tau_start = config.compute_temperature(0, 100)
        tau_end = config.compute_temperature(99, 100)
        
        # 余弦退火从 tau_max 开始，到 tau_min 结束
        assert abs(tau_start - 5.0) < 0.01
        assert abs(tau_end - 0.5) < 0.1


class TestTokenizerIntegration:
    """与 Tokenizer 集成测试."""
    
    def test_from_config(self):
        """测试 from_config 工厂方法."""
        config = FractalConfig(
            64, 4,
            gumbel_tau_init=1.5,
            variable_tokens=True,
        )
        
        tokenizer = StreamingFractalTokenizerV2.from_config(config)
        
        assert tokenizer.tau_init == 1.5
        assert tokenizer.variable_tokens is True
        assert len(tokenizer.patch_sizes) == config.num_scales
    
    def test_from_config_with_custom_d_model(self):
        """测试 from_config 自定义 d_model."""
        config = FractalConfig(64, 4)
        
        tokenizer = StreamingFractalTokenizerV2.from_config(config, d_model=512)
        
        assert tokenizer.d_model == 512
    
    def test_from_config_forward_pass(self):
        """测试 from_config 创建的 tokenizer 前向传播."""
        import torch
        
        config = FractalConfig(64, 4, variable_tokens=False)
        tokenizer = StreamingFractalTokenizerV2.from_config(config, d_model=128)
        
        images = torch.randn(2, 3, 64, 64)
        output = tokenizer(images)
        
        assert len(output.sequences) == 2
        assert output.sequences[0].tokens.shape[-1] == 128


class TestRepr:
    """字符串表示测试."""
    
    def test_repr_contains_all_sections(self):
        """测试 repr 包含所有配置段."""
        config = FractalConfig(64, 4)
        repr_str = repr(config)
        
        assert "# Geometry" in repr_str
        assert "# Gumbel-Softmax" in repr_str
        assert "# Hilbert Bias" in repr_str
        assert "# Tokenizer" in repr_str
        assert "image_size=64" in repr_str
        assert "tau=" in repr_str
        assert "bias_mode='lca'" in repr_str
