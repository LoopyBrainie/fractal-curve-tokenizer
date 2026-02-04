# -*- coding: utf-8 -*-
"""
L1 Foundation: Config Tests

对应模块: vit_pytorch.config

测试内容:
- AttentionEncoderConfig 初始化参数
- AreaEncoderConfig cutoff_ratio 参数
- HilbertAwareMultiScaleAttention 配置初始化
"""

import pytest
import torch

from vit_pytorch.config import (
    AttentionEncoderConfig,
    AreaEncoderConfig,
    ShapeScaleEncoderConfig,
)
from vit_pytorch.attn_hilbert_bias import (
    HilbertAwareMultiScaleAttention,
    AreaEncoder,
)


class TestAttentionEncoderConfig:
    """AttentionEncoderConfig 测试类."""

    def test_default_init_values(self):
        """默认初始化值与原硬编码值一致."""
        config = AttentionEncoderConfig()

        assert abs(config.level_scale_init - 0.5413) < 0.0001
        assert config.hierarchical_scale_bounds == (0.5, 1.5)
        # I113-11: 初始化从 ln(0.1) 改为 ln(1.0)，配合 √d_k 量纲对齐
        assert abs(config.hilbert_bias_init - 0.0) < 0.0001
        assert abs(config.level_bias_init - 0.0) < 0.0001
        assert config.energy_injection_enabled is True

    def test_custom_init_values(self):
        """自定义初始化值正确应用."""
        config = AttentionEncoderConfig(
            level_scale_init=0.7,
            hierarchical_scale_bounds=(0.3, 1.2),
            hilbert_bias_init=-1.5,
            level_bias_init=-2.0,
            energy_injection_enabled=False,
        )

        assert config.level_scale_init == 0.7
        assert config.hierarchical_scale_bounds == (0.3, 1.2)
        assert config.hilbert_bias_init == -1.5
        assert config.level_bias_init == -2.0
        assert config.energy_injection_enabled is False

    def test_to_dict_includes_init_params(self):
        """to_dict 方法包含初始化参数."""
        config = AttentionEncoderConfig()
        config_dict = config.to_dict()

        assert 'level_scale_init' in config_dict
        assert 'hierarchical_scale_bounds' in config_dict
        assert 'hilbert_bias_init' in config_dict
        assert 'level_bias_init' in config_dict
        assert 'energy_injection_enabled' in config_dict

    def test_nested_config_preserved(self):
        """嵌套配置保持完整."""
        config = AttentionEncoderConfig()

        assert isinstance(config.shape_scale, ShapeScaleEncoderConfig)
        assert isinstance(config.area, AreaEncoderConfig)
        assert hasattr(config.lca, 'embedding_dim')


class TestAreaEncoderConfig:
    """AreaEncoderConfig 测试类."""

    def test_default_cutoff_ratio(self):
        """默认 cutoff_ratio 为 0.8."""
        config = AreaEncoderConfig()
        assert config.cutoff_ratio == 0.8

    def test_custom_cutoff_ratio(self):
        """自定义 cutoff_ratio 正确应用."""
        for ratio in [0.5, 0.6, 0.7, 0.9, 1.0]:
            config = AreaEncoderConfig(cutoff_ratio=ratio)
            assert config.cutoff_ratio == ratio

    def test_boundary_cutoff_ratio(self):
        """边界值验证."""
        config_low = AreaEncoderConfig(cutoff_ratio=0.01)
        config_high = AreaEncoderConfig(cutoff_ratio=1.0)

        assert config_low.cutoff_ratio == 0.01
        assert config_high.cutoff_ratio == 1.0

    def test_to_dict_includes_cutoff_ratio(self):
        """to_dict 方法包含 cutoff_ratio."""
        config = AreaEncoderConfig(cutoff_ratio=0.6)
        config_dict = config.to_dict()

        assert 'cutoff_ratio' in config_dict
        assert config_dict['cutoff_ratio'] == 0.6


class TestHilbertAwareMultiScaleAttentionConfig:
    """HilbertAwareMultiScaleAttention 配置初始化测试类."""

    @pytest.fixture
    def dim(self):
        return 128

    @pytest.fixture
    def heads(self):
        return 4

    @pytest.fixture
    def max_level(self):
        return 3

    def test_level_scale_init_from_config(self, dim, heads, max_level):
        """level_scale_raw 从配置初始化."""
        custom_init = 0.7
        config = AttentionEncoderConfig(level_scale_init=custom_init)

        attn = HilbertAwareMultiScaleAttention(
            dim=dim,
            heads=heads,
            max_level=max_level,
            use_level_scaling=True,
            encoder_config=config,
        )

        init_value = attn._level_scale_raw.weight.mean().item()
        assert abs(init_value - custom_init) < 1e-6

    def test_hierarchical_scale_bounds_from_config(self, dim, heads, max_level):  # max_level fixture defined above
        """hierarchical_depth_scale 从配置初始化."""
        low, high = 0.3, 1.2
        config = AttentionEncoderConfig(hierarchical_scale_bounds=(low, high))

        attn = HilbertAwareMultiScaleAttention(
            dim=dim,
            heads=heads,
            max_level=max_level,
            use_hierarchical_attention=True,
            encoder_config=config,
        )

        scale_values = attn._hierarchical_depth_scale.data
        assert scale_values.min() >= low - 1e-6
        assert scale_values.max() <= high + 1e-6

    def test_bias_init_from_config(self, dim, heads, max_level):
        """偏置缩放从配置初始化."""
        hilbert_init = -1.5
        level_init = -2.0
        config = AttentionEncoderConfig(
            hilbert_bias_init=hilbert_init,
            level_bias_init=level_init,
        )

        attn = HilbertAwareMultiScaleAttention(
            dim=dim,
            heads=heads,
            max_level=max_level,
            encoder_config=config,
        )

        assert abs(attn._hilbert_bias_scale_raw.item() - hilbert_init) < 1e-6
        assert abs(attn._level_bias_scale_raw.item() - level_init) < 1e-6

    def test_default_config_when_none(self, dim, heads, max_level):  # max_level fixture defined above
        """未提供配置时使用默认配置."""
        attn = HilbertAwareMultiScaleAttention(
            dim=dim,
            heads=heads,
            max_level=max_level,
            use_level_scaling=True,
            use_hierarchical_attention=True,
        )

        default_config = AttentionEncoderConfig()
        assert attn.config.level_scale_init == default_config.level_scale_init
        assert attn.config.hierarchical_scale_bounds == default_config.hierarchical_scale_bounds
        assert attn.config.hilbert_bias_init == default_config.hilbert_bias_init
        assert attn.config.level_bias_init == default_config.level_bias_init

    def test_get_config_method(self, dim, heads, max_level):  # max_level fixture defined above
        """get_config 方法返回配置."""
        config = AttentionEncoderConfig(level_scale_init=0.9)
        attn = HilbertAwareMultiScaleAttention(
            dim=dim,
            heads=heads,
            max_level=max_level,
            encoder_config=config,
        )

        retrieved_config = attn.get_config()
        assert retrieved_config.level_scale_init == 0.9


class TestAreaEncoderConfig:
    """AreaEncoder 配置测试类."""

    @pytest.fixture
    def dim(self):
        return 128

    def test_cutoff_ratio_from_config(self, dim):
        """AreaEncoder 从配置读取 cutoff_ratio."""
        custom_ratio = 0.6
        area_config = AreaEncoderConfig(cutoff_ratio=custom_ratio)

        encoding = AreaEncoder(dim=dim, config=area_config)

        assert encoding.cutoff_ratio == custom_ratio

    def test_default_cutoff_ratio(self, dim):
        """默认 cutoff_ratio 为 0.8."""
        encoding = AreaEncoder(dim=dim)

        assert encoding.cutoff_ratio == 0.8

    def test_get_config_includes_cutoff_ratio(self, dim):
        """get_config 方法返回 cutoff_ratio."""
        custom_ratio = 0.7
        area_config = AreaEncoderConfig(cutoff_ratio=custom_ratio)
        encoding = AreaEncoder(dim=dim, config=area_config)

        retrieved_config = encoding.get_config()
        assert retrieved_config.cutoff_ratio == custom_ratio


class TestConfigIntegration:
    """配置集成测试类."""

    def test_full_attention_config(self):
        """完整注意力配置链."""
        area_config = AreaEncoderConfig(
            fourier_levels=4,
            cutoff_ratio=0.75,
        )

        enc_config = AttentionEncoderConfig(
            level_scale_init=0.65,
            hierarchical_scale_bounds=(0.4, 1.3),
            hilbert_bias_init=-2.0,
            level_bias_init=-2.5,
            energy_injection_enabled=True,
            area=area_config,
        )

        assert enc_config.level_scale_init == 0.65
        assert enc_config.area.cutoff_ratio == 0.75

    def test_config_serialization_roundtrip(self):
        """配置序列化-反序列化."""
        original = AttentionEncoderConfig(
            level_scale_init=0.72,
            hilbert_bias_init=-1.8,
        )

        serialized = original.to_dict()
        assert serialized['level_scale_init'] == 0.72
        assert serialized['hilbert_bias_init'] == -1.8


class TestConfigBackwardCompatibility:
    """向后兼容性测试."""

    def test_default_equals_original_hardcoded(self):
        """默认值与原硬编码值等价."""
        config = AttentionEncoderConfig()

        assert config.level_scale_init == 0.5413
        assert config.hierarchical_scale_bounds == (0.5, 1.5)
        # I113-11: 初始化从 ln(0.1) 改为 ln(1.0)
        assert abs(config.hilbert_bias_init - 0.0) < 0.0001
        assert abs(config.level_bias_init - 0.0) < 0.0001

    def test_attention_without_config_uses_defaults(self):
        """不传配置时使用默认行为."""
        attn = HilbertAwareMultiScaleAttention(
            dim=128,
            heads=4,
            max_level=3,
            use_level_scaling=True,
            use_hierarchical_attention=True,
        )

        default_config = AttentionEncoderConfig()

        assert attn.config.level_scale_init == default_config.level_scale_init
        assert attn.config.hierarchical_scale_bounds == default_config.hierarchical_scale_bounds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
