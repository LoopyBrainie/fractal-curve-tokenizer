# -*- coding: utf-8 -*-
"""
L4 Application Tests: FractalConfig

对应模块: vit_pytorch.config

测试内容:
- FractalConfig 参数推导
- Hilbert 策略选择
- 尺度深度转换
"""

import math
import pytest

from vit_pytorch import (
    FractalConfig,
    create_fractal_config,
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

    def test_tokenizer_defaults(self):
        """测试 Tokenizer 默认参数."""
        config = FractalConfig(64, 4)

        assert config.tokenizer_type == 'streaming_v3'


class TestFractalConfigCustomization:
    """自定义配置测试."""

    def test_create_fractal_config_helper(self):
        """测试便捷函数."""
        config = create_fractal_config(64, 4)

        assert config.image_size == 64
        assert config.min_patch_size == 4


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
        config = FractalConfig(60, 5)
        assert config.grid_size == 12
        assert config.uses_pseudo_hilbert

    def test_non_power_of_two_patch_size_allowed(self):
        """测试非 2 幂 patch 尺寸可以使用."""
        config = FractalConfig(96, 6)
        assert config.grid_size == 16
        assert config.num_tokens == 256
        assert not config.uses_pseudo_hilbert

        config2 = FractalConfig(192, 12)
        assert config2.grid_size == 16
        assert not config2.uses_pseudo_hilbert

    def test_not_divisible(self):
        """测试不可整除情况."""
        with pytest.raises(ValueError, match="必须能被"):
            FractalConfig(64, 5)


class TestFractalConfigMethods:
    """辅助方法测试."""

    def test_scale_to_depth(self):
        """测试尺度到深度转换.

        scale_idx=0 (最细) → depth=max_depth
        scale_idx=max_depth (最粗) → depth=0
        """
        config = FractalConfig(64, 4)

        assert config.scale_to_depth(0) == 4
        assert config.scale_to_depth(4) == 0

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

        assert config.grid_size_at_scale(0) == 16
        assert config.grid_size_at_scale(4) == 1


class TestRepr:
    """字符串表示测试."""

    def test_repr_contains_all_sections(self):
        """测试 repr 包含所有配置段."""
        config = FractalConfig(64, 4)
        repr_str = repr(config)

        assert "# Geometry" in repr_str
        assert "# Hilbert Bias" in repr_str or "LCA" in repr_str  # P11-8 simplified
        assert "# Tokenizer" in repr_str
        assert "image_size=64" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
