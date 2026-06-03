# -*- coding: utf-8 -*-
"""
L1 Foundation: Config Tests

对应模块: vit_pytorch.core.config

测试内容:
- HilbertSplitterConfig: 验证, 候选数计算, K 边界, 配额初始化
- FractalConfig: 自动推导参数, 尺度↔深度转换
- 编码器配置: ShapeScaleEncoderConfig, AreaEncoderConfig, LCAEncoderConfig
- 工厂函数: create_splitter_config, create_fractal_config
- 边界条件: 无效参数验证, 退化尺寸
"""

import pytest

from vit_pytorch.core.config import (
    HilbertSplitterConfig,
    SplitterConfig,
    NeighborAwareSplitterConfig,
    SemanticSplitterConfig,
    AttentionConfig,
    TokenizerConfig,
    FractalConfig,
    ShapeScaleEncoderConfig,
    AreaEncoderConfig,
    LCAEncoderConfig,
    AttentionEncoderConfig,
    create_splitter_config,
    create_semantic_splitter_config,
    create_fractal_config,
)
from vit_pytorch.core.outcome import Err, Ok


class TestHilbertSplitterConfigDefaults:
    """默认值测试"""

    def test_default_min_patch_size(self):
        cfg = HilbertSplitterConfig()
        assert cfg.min_patch_size == 4

    def test_default_max_level_limit(self):
        cfg = HilbertSplitterConfig()
        assert cfg.max_level_limit == 8

    def test_default_temperature_anneal(self):
        cfg = HilbertSplitterConfig()
        assert cfg.temperature_anneal == "linear"


class TestHilbertSplitterConfigValidate:
    """validate() 方法测试"""

    def test_valid_default_passes(self):
        """默认配置通过验证"""
        cfg = HilbertSplitterConfig()
        cfg.validate()

    def test_negative_min_patch_size(self):
        """min_patch_size <= 0 抛出异常"""
        cfg = HilbertSplitterConfig(min_patch_size=0)
        with pytest.raises(ValueError, match="min_patch_size"):
            cfg.validate()

    def test_low_max_level_limit(self):
        """max_level_limit < 2 抛出异常"""
        cfg = HilbertSplitterConfig(max_level_limit=1)
        with pytest.raises(ValueError, match="max_level_limit"):
            cfg.validate()

    def test_invalid_coverage_base(self):
        """coverage_base 必须在 (0, 1] 范围内"""
        cfg = HilbertSplitterConfig(coverage_base=1.5)
        with pytest.raises(ValueError, match="coverage_base"):
            cfg.validate()

    def test_invalid_coverage_order(self):
        """coverage_min >= coverage_max_hard 抛出异常"""
        cfg = HilbertSplitterConfig(coverage_min=0.8, coverage_max_hard=0.7)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_invalid_temperature_min(self):
        """temperature_min >= temperature_init 抛出异常"""
        cfg = HilbertSplitterConfig(temperature_init=0.5, temperature_min=1.0)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_invalid_entropy_mode(self):
        """无效的 entropy_mode 抛出异常"""
        cfg = HilbertSplitterConfig(entropy_mode="invalid")
        with pytest.raises(ValueError, match="entropy_mode"):
            cfg.validate()

    def test_invalid_temperature_anneal(self):
        """无效的 temperature_anneal 抛出异常"""
        cfg = HilbertSplitterConfig(temperature_anneal="invalid")
        with pytest.raises(ValueError):
            cfg.validate()


class TestHilbertSplitterConfigCompute:
    """计算方法测试"""

    def test_compute_candidate_count(self):
        """候选数公式: N = (4^(L+1) - 1) / 3"""
        for L in [2, 4, 6, 8]:
            cfg = HilbertSplitterConfig(max_level_limit=L)
            expected = (4 ** (L + 1) - 1) // 3
            assert cfg.compute_candidate_count() == expected

    def test_compute_k_bounds_no_image(self):
        """无图像尺寸时的 K 边界"""
        cfg = HilbertSplitterConfig()
        K_min, K_max = cfg.compute_k_bounds()
        assert K_min >= cfg.K_min_abs
        assert K_max <= cfg.K_max_hard
        assert K_min <= K_max

    def test_compute_k_bounds_with_image(self):
        """有图像尺寸时的 K 边界"""
        cfg = HilbertSplitterConfig()
        K_min, K_max = cfg.compute_k_bounds(image_size=(224, 224))
        assert K_min >= cfg.K_min_abs
        assert K_max <= cfg.K_max_hard

    def test_scale_factor_small_image(self):
        """小图像降低 K 上界"""
        cfg = HilbertSplitterConfig()
        _, K_max_small = cfg.compute_k_bounds(image_size=(32, 32))
        _, K_max_large = cfg.compute_k_bounds(image_size=(512, 512))
        assert K_max_small <= K_max_large

    def test_get_quota_init_tensor_default(self):
        """默认返回全零元组"""
        cfg = HilbertSplitterConfig()
        init = cfg.get_quota_init_tensor(D=4)
        assert init == (0.0, 0.0, 0.0, 0.0)

    def test_get_quota_init_tensor_custom(self):
        """自定义配额初始值"""
        cfg = HilbertSplitterConfig(quota_init_logits=(0.5, -0.3))
        init = cfg.get_quota_init_tensor(D=4)
        assert init == (0.5, -0.3, 0.0, 0.0)

    def test_get_quota_init_tensor_truncated(self):
        """配额初始值过长时截断"""
        cfg = HilbertSplitterConfig(quota_init_logits=(1.0, 2.0, 3.0, 4.0))
        init = cfg.get_quota_init_tensor(D=2)
        assert init == (1.0, 2.0)

    def test_compute_absolute_targets(self):
        """计算绝对目标值"""
        cfg = HilbertSplitterConfig()
        targets = cfg.compute_absolute_targets()
        assert "N_base" in targets
        assert "N_target" in targets
        assert "N_min" in targets
        assert "N_max" in targets
        assert targets["N_base"] > 0
        # N_target 不应超过总候选数 N_base
        assert targets["N_target"] <= targets["N_base"]


class TestHilbertSplitterConfigSerialization:
    """序列化测试"""

    def test_to_dict_contains_keys(self):
        """to_dict() 包含关键参数"""
        d = HilbertSplitterConfig().to_dict()
        assert "min_patch_size" in d
        assert "max_level_limit" in d
        assert "coverage_base" in d
        assert "temperature_init" in d

    def test_to_dict_roundtrip_values(self):
        """to_dict() 值正确"""
        cfg = HilbertSplitterConfig(min_patch_size=8)
        d = cfg.to_dict()
        assert d["min_patch_size"] == 8

    def test_splitter_config_alias(self):
        """SplitterConfig 是 HilbertSplitterConfig 的别名"""
        assert SplitterConfig is HilbertSplitterConfig


class TestFractalConfig:
    """FractalConfig 测试"""

    def test_basic_derivation(self):
        """基本参数推导"""
        cfg = FractalConfig(image_size=64, min_patch_size=4)
        assert cfg.grid_size == 16
        assert cfg.num_tokens == 256
        assert cfg.max_level >= 0

    def test_power_of_two_no_pseudo(self):
        """2^k 尺寸不使用 Pseudo-Hilbert"""
        cfg = FractalConfig(image_size=64, min_patch_size=4)
        assert cfg.uses_pseudo_hilbert is False

    def test_non_divisible_raises(self):
        """不能整除时抛出异常"""
        with pytest.raises(ValueError):
            FractalConfig(image_size=63, min_patch_size=4)

    def test_negative_size_raises(self):
        """负数尺寸抛出异常"""
        with pytest.raises(ValueError):
            FractalConfig(image_size=-64, min_patch_size=4)

    def test_scale_to_depth(self):
        """scale_to_depth 公式: d = max_level - scale_idx"""
        cfg = FractalConfig(image_size=64, min_patch_size=4)
        L = cfg.max_level
        assert cfg.scale_to_depth(0) == L
        assert cfg.scale_to_depth(L) == 0

    def test_depth_to_scale(self):
        """depth_to_scale 是 scale_to_depth 的逆运算"""
        cfg = FractalConfig(image_size=64, min_patch_size=4)
        for scale in range(cfg.num_scales):
            depth = cfg.scale_to_depth(scale)
            assert cfg.depth_to_scale(depth) == scale

    def test_patch_size_at_scale(self):
        """patch_size_at_scale 返回正确值"""
        cfg = FractalConfig(image_size=64, min_patch_size=4)
        assert cfg.patch_size_at_scale(0) == 4
        assert cfg.patch_size_at_scale(1) == 8

    def test_grid_size_at_scale(self):
        """grid_size_at_scale 返回正确值"""
        cfg = FractalConfig(image_size=64, min_patch_size=4)
        assert cfg.grid_size_at_scale(0) == 16

    def test_num_scales(self):
        """num_scales = max_level + 1"""
        cfg = FractalConfig(image_size=64, min_patch_size=4)
        assert cfg.num_scales == cfg.max_level + 1

    def test_repr(self):
        """__repr__ 包含关键信息"""
        cfg = FractalConfig(image_size=64, min_patch_size=4)
        r = repr(cfg)
        assert "image_size=64" in r
        assert "FractalConfig" in r


class TestFractalConfigEdgeCases:
    """FractalConfig 边界条件"""

    def test_image_size_equals_patch_size(self):
        """image_size == min_patch_size → grid_size=1"""
        cfg = FractalConfig(image_size=4, min_patch_size=4)
        assert cfg.grid_size == 1
        assert cfg.num_tokens == 1
        assert cfg.max_level == 0

    def test_large_image(self):
        """大图像推导正确"""
        cfg = FractalConfig(image_size=1024, min_patch_size=4)
        assert cfg.grid_size == 256
        assert cfg.num_tokens == 65536


class TestEncoderConfigs:
    """编码器配置测试"""

    def test_shape_scale_defaults(self):
        cfg = ShapeScaleEncoderConfig()
        assert cfg.enabled is True
        assert cfg.hidden_dim == 64

    def test_area_defaults(self):
        cfg = AreaEncoderConfig()
        assert cfg.fourier_levels == 4
        assert cfg.freq_base == 2.0

    def test_lca_defaults(self):
        cfg = LCAEncoderConfig()
        assert cfg.embedding_dim == 128

    def test_attention_encoder_composition(self):
        cfg = AttentionEncoderConfig()
        assert cfg.shape_scale.enabled is True
        assert cfg.area.fourier_levels == 4
        assert cfg.lca.embedding_dim == 128

    def test_to_dict(self):
        for cfg_class in [ShapeScaleEncoderConfig, AreaEncoderConfig,
                          LCAEncoderConfig, AttentionEncoderConfig]:
            d = cfg_class().to_dict()
            assert isinstance(d, dict)


class TestAttentionConfig:
    """AttentionConfig 测试"""

    def test_defaults(self):
        cfg = AttentionConfig()
        assert cfg.dim == 256
        assert cfg.heads == 8
        assert cfg.lca_bias is True

    def test_head_dim_default(self):
        cfg = AttentionConfig()
        assert cfg.head_dim is None


class TestTokenizerConfig:
    """TokenizerConfig 测试"""

    def test_default_splitter_config(self):
        cfg = TokenizerConfig()
        assert cfg.splitter_config is not None
        assert isinstance(cfg.splitter_config, HilbertSplitterConfig)

    def test_custom_splitter_config(self):
        sc = HilbertSplitterConfig(min_patch_size=8)
        cfg = TokenizerConfig(splitter_config=sc)
        assert cfg.splitter_config is not None
        assert cfg.splitter_config.min_patch_size == 8


class TestNeighborAwareSplitterConfig:
    """NeighborAwareSplitterConfig 测试"""

    def test_validate_pass(self):
        cfg = NeighborAwareSplitterConfig()
        cfg.validate()

    def test_compute_candidate_count(self):
        cfg = NeighborAwareSplitterConfig(max_level_limit=4)
        expected = (4 ** (4 + 1) - 1) // 3
        assert cfg.compute_candidate_count() == expected


class TestSemanticSplitterConfig:
    """SemanticSplitterConfig 测试"""

    def test_default_max_level(self):
        cfg = SemanticSplitterConfig()
        assert cfg.max_level_limit is None

    def test_validate_pass(self):
        cfg = SemanticSplitterConfig()
        cfg.validate()

    def test_invalid_split_threshold(self):
        cfg = SemanticSplitterConfig(split_threshold=1.5)
        with pytest.raises(ValueError):
            cfg.validate()


class TestFactoryFunctions:
    """工厂函数测试"""

    def test_create_splitter_config_default(self):
        cfg = create_splitter_config()
        assert isinstance(cfg, HilbertSplitterConfig)
        assert cfg.min_patch_size == 4

    def test_create_splitter_config_override(self):
        cfg = create_splitter_config(
            min_patch_size=8,
            max_level_limit=6,
            coverage_base=0.3,
            entropy_mode="target",
            temperature_anneal="exponential",
        )
        assert cfg.min_patch_size == 8
        assert cfg.max_level_limit == 6
        assert cfg.coverage_base == 0.3
        assert cfg.entropy_mode == "target"
        assert cfg.temperature_anneal == "exponential"

    def test_create_fractal_config(self):
        cfg = create_fractal_config(image_size=128, min_patch_size=8)
        assert cfg.image_size == 128
        assert cfg.min_patch_size == 8

    def test_create_semantic_splitter_config_default(self):
        cfg = create_semantic_splitter_config()
        assert isinstance(cfg, SemanticSplitterConfig)
        cfg.validate()


class TestHilbertSplitterConfigTryValidate:
    """Phase 1 (AEH): try_validate() returns Outcome[None, ConfigError]."""

    def test_try_validate_ok_for_valid_config(self):
        cfg = HilbertSplitterConfig()
        result = cfg.try_validate()
        assert isinstance(result, Ok)
        assert result.value is None

    def test_try_validate_err_for_invalid_min_patch(self):
        cfg = HilbertSplitterConfig(min_patch_size=0)
        result = cfg.try_validate()
        assert isinstance(result, Err)
        assert result.error.kind == "min_patch_size"
        assert isinstance(result.error, ValueError)  # ConfigError ⊂ ValueError

    def test_validate_still_raises_for_backcompat(self):
        """Existing pytest.raises(ValueError) sites must continue to work."""
        cfg = HilbertSplitterConfig(min_patch_size=0)
        with pytest.raises(ValueError, match="min_patch_size"):
            cfg.validate()  # OLD API: still raises


class TestNeighborAwareSplitterConfigTryValidate:
    """Phase 1 (AEH): try_validate() returns Outcome[None, ConfigError]."""

    def test_try_validate_ok_for_valid_config(self):
        cfg = NeighborAwareSplitterConfig()
        result = cfg.try_validate()
        assert isinstance(result, Ok)

    def test_try_validate_err_for_invalid_min_patch(self):
        cfg = NeighborAwareSplitterConfig(min_patch_size=-1)
        result = cfg.try_validate()
        assert isinstance(result, Err)
        assert result.error.kind == "min_patch_size"

    def test_validate_still_raises(self):
        cfg = NeighborAwareSplitterConfig(min_patch_size=-1)
        with pytest.raises(ValueError, match="min_patch_size"):
            cfg.validate()


class TestSemanticSplitterConfigTryValidate:
    """Phase 1 (AEH): try_validate() returns Outcome[None, ConfigError]."""

    def test_try_validate_ok_for_valid_config(self):
        cfg = SemanticSplitterConfig()
        result = cfg.try_validate()
        assert isinstance(result, Ok)

    def test_try_validate_err_for_invalid_split_threshold(self):
        cfg = SemanticSplitterConfig(split_threshold=1.5)
        result = cfg.try_validate()
        assert isinstance(result, Err)
        assert result.error.kind == "split_threshold"

    def test_validate_still_raises(self):
        cfg = SemanticSplitterConfig(split_threshold=1.5)
        with pytest.raises(ValueError, match="split_threshold"):
            cfg.validate()
