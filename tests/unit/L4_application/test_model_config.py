# -*- coding: utf-8 -*-
"""
L4 Application Tests: FractalConfig

对应模块: vit_pytorch.config

测试内容:
- FractalConfig 参数推导
- Hilbert 策略选择
- 尺度深度转换
- LogitsClamp (从 test_i147_refactoring.py 迁移)
"""

import math
import pytest
import torch

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
        assert config.max_level == 4  # log2(64/4) = 4
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


# =============================================================================
# 迁移自 test_i147_refactoring.py (I147 重构测试)
# =============================================================================

class TestLogitsClamp:
    """LogitsClamp 钳制层的验证 (I147)"""

    def test_clamp_boundaries(self):
        """验证钳制边界"""
        from vit_pytorch.models.fractal_vit import LogitsClamp
        from vit_pytorch.core.constants import LOGIT_CLAMP_BOUND

        clamp = LogitsClamp()
        assert clamp.bound == LOGIT_CLAMP_BOUND

        # 测试边界钳制
        x = torch.tensor([-100.0, -10.0, 0.0, 10.0, 100.0])
        y = clamp(x)

        assert y[0].item() == -LOGIT_CLAMP_BOUND
        assert y[1].item() == -10.0
        assert y[2].item() == 0.0
        assert y[3].item() == 10.0
        assert y[4].item() == LOGIT_CLAMP_BOUND

    def test_gradient_flow(self):
        """验证梯度流动"""
        from vit_pytorch.models.fractal_vit import LogitsClamp

        clamp = LogitsClamp()
        x = torch.tensor([-5.0, 0.0, 5.0], requires_grad=True)
        y = clamp(x)
        loss = y.sum()
        loss.backward()

        # 梯度应该能流动
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_no_gradient_modification(self):
        """验证钳制不修改有效梯度"""
        from vit_pytorch.models.fractal_vit import LogitsClamp

        clamp = LogitsClamp(bound=10.0)

        # 在边界内的值，梯度应该不变
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = clamp(x)
        loss = y.sum()
        loss.backward()

        assert torch.allclose(x.grad, torch.ones(3))

    def test_clamp_extreme_values(self):
        """验证极端值的钳制"""
        from vit_pytorch.models.fractal_vit import LogitsClamp
        from vit_pytorch.core.constants import LOGIT_CLAMP_BOUND

        clamp = LogitsClamp()

        # 极端正值
        x_pos = torch.tensor([1e10, 1e5, 1000.0])
        y_pos = clamp(x_pos)
        assert torch.all(y_pos <= LOGIT_CLAMP_BOUND)

        # 极端负值
        x_neg = torch.tensor([-1e10, -1e5, -1000.0])
        y_neg = clamp(x_neg)
        assert torch.all(y_neg >= -LOGIT_CLAMP_BOUND)

    def test_extra_repr(self):
        """验证字符串表示"""
        from vit_pytorch.models.fractal_vit import LogitsClamp

        clamp = LogitsClamp(bound=10.0)
        repr_str = repr(clamp)
        assert "bound=10.0" in repr_str


class TestLogitsClampIntegration:
    """LogitsClamp 与模型的集成测试 (I147)"""

    def test_model_output_clamped(self):
        """验证模型输出被钳制"""
        from vit_pytorch import FractalCurveViT
        from vit_pytorch.core.constants import LOGIT_CLAMP_BOUND

        model = FractalCurveViT(
            image_size=64,
            num_classes=10,
            dim=64,
            num_layers=2,
            heads=4,
        )
        model.eval()

        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = model(x)

        # 输出 logits 应该被钳制
        logits = out.logits
        max_val = logits.abs().max().item()

        # 考虑数值精度，允许微小误差
        assert max_val <= LOGIT_CLAMP_BOUND + 1e-5, \
            f"Logits should be clamped: max={max_val}"

    def test_gradients_exist(self):
        """验证模型参数有梯度"""
        from vit_pytorch import FractalCurveViT

        model = FractalCurveViT(
            image_size=64,
            num_classes=10,
            dim=64,
            num_layers=2,
            heads=4,
        )
        model.train()

        x = torch.randn(2, 3, 64, 64)
        y = model(x)

        # 计算损失并反向传播
        loss = y.logits.sum()
        loss.backward()

        # 关键参数应该有梯度（MLP Head, Transformer Layers）
        key_params_have_grad = False
        for name, param in model.named_parameters():
            # 跳过 splitter.quota_logits（可能默认冻结）
            if 'quota_logits' in name:
                continue
            if param.requires_grad:
                if param.grad is not None:
                    key_params_have_grad = True
                # 验证没有 NaN/Inf 梯度
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any(), f"{name} has NaN grad"
                    assert not torch.isinf(param.grad).any(), f"{name} has Inf grad"

        assert key_params_have_grad, "Key model parameters should have gradients"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
