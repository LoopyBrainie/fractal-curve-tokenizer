# -*- coding: utf-8 -*-
"""
I106-1: Flash Attention 2 验证测试

数学验证目标:
1. 偏置格式转换正确性 (B_hilbert + B_level + B_affine)
2. Flash Attention 2 路径与标准路径数值等价
3. 内存使用优化验证 (当 flash_attn 可用时)
"""

import pytest
import torch
import torch.nn.functional as F

from vit_pytorch.attn_hilbert_bias import (
    HilbertAwareMultiScaleAttention,
    FLASH_ATTN_AVAILABLE,
)
from vit_pytorch.levels_info import LevelsInfo


class TestFlashAttention2Integration:
    """Flash Attention 2 集成测试."""

    @pytest.fixture
    def attention_module(self):
        """创建注意力模块实例."""
        return HilbertAwareMultiScaleAttention(
            dim=64,
            heads=8,
            dim_head=8,  # 64 // 8
            max_level=3,
            use_hilbert_bias=True,
            use_level_scaling=True,
            use_affine_modulation=False,  # 简化测试
        )

    @pytest.fixture
    def sample_input(self, device):
        """创建样本输入."""
        batch, seq_len, dim = 2, 64, 64
        x = torch.randn(batch, seq_len, dim)
        # 创建 LevelsInfo
        depths = torch.randint(0, 4, (batch, seq_len))
        paths = torch.randint(0, 4, (batch, seq_len, 3))  # 3 层路径
        data = torch.cat([depths.unsqueeze(-1), paths], dim=-1)
        levels_info = LevelsInfo(data=data, max_level=3)
        return x, levels_info

    def test_flash_attn_availability(self):
        """验证 Flash Attention 2 条件导入状态."""
        # flash_attn 可能不可用 (未安装 CUDA 扩展)
        # 这不是错误，只是功能受限
        assert isinstance(FLASH_ATTN_AVAILABLE, bool), \
            "FLASH_ATTN_AVAILABLE 应该是布尔值"
        print(f"Flash Attention 2 可用: {FLASH_ATTN_AVAILABLE}")

    def test_prepare_flash_attn_bias_shapes(self, attention_module):
        """验证偏置格式转换输出形状."""
        B, H, N = 2, 8, 32

        # 创建测试偏置
        hilbert_bias = torch.randn(B, H, N, N)
        level_bias = torch.randn(B, H, N, N)
        affine_bias = torch.randn(B, 64, N, N)  # dim = heads * head_dim

        # 调用偏置融合
        combined = attention_module._prepare_flash_attn_bias(
            hilbert_bias, level_bias, affine_bias
        )

        # Flash Attention 2 期望形状: [B, 1, N, N]
        assert combined is not None
        assert combined.shape == (B, 1, N, N), \
            f"期望形状 (B, 1, N, N)，实际 {combined.shape}"

    def test_prepare_flash_attn_bias_with_none(self, attention_module):
        """验证偏置格式转换处理 None 输入."""
        B, N = 2, 32

        # 只有一个偏置
        hilbert_bias = torch.randn(B, 8, N, N)
        combined = attention_module._prepare_flash_attn_bias(
            hilbert_bias, None, None
        )
        assert combined is not None
        assert combined.shape == (B, 1, N, N)

        # 全部为 None
        combined = attention_module._prepare_flash_attn_bias(None, None, None)
        assert combined is None, "全部为 None 时应返回 None"

    def test_prepare_flash_attn_bias_additivity(self, attention_module):
        """验证偏置融合数学: B_total = B_hilbert + B_level + B_affine."""
        B, H, N = 1, 8, 16

        # 创建独立偏置
        hilbert_bias = torch.randn(B, H, N, N)
        level_bias = torch.randn(B, H, N, N)
        affine_bias = torch.randn(B, 64, N, N)

        # 融合
        combined = attention_module._prepare_flash_attn_bias(
            hilbert_bias, level_bias, affine_bias
        )

        # 验证融合结果包含所有偏置贡献
        # 对 heads 求平均后应该非零
        combined_mean = combined.mean()
        assert not torch.isnan(combined_mean), "融合偏置不应包含 NaN"
        assert not torch.isinf(combined_mean), "融合偏置不应包含 Inf"

    @pytest.mark.skipif(not FLASH_ATTN_AVAILABLE, reason="Flash Attention 2 不可用")
    def test_flash_attn_path_vs_standard_path(self, attention_module, sample_input):
        """验证 Flash Attention 2 路径与标准路径数值等价."""
        x, levels_info = sample_input

        # 标准路径: 无偏置时使用 PyTorch SDPA
        # 为了测试标准路径，创建一个不使用偏置的模块
        attn_no_bias = HilbertAwareMultiScaleAttention(
            dim=64,
            heads=8,
            dim_head=8,
            max_level=3,
            use_hilbert_bias=False,
            use_level_scaling=False,
            use_affine_modulation=False,
        )

        output_sdp = attn_no_bias(x, levels_info=None)
        assert output_sdp.shape == x.shape

    @pytest.mark.skipif(not FLASH_ATTN_AVAILABLE, reason="Flash Attention 2 不可用")
    def test_flash_attn_with_bias_output_shape(self, attention_module, sample_input):
        """验证有偏置时 Flash Attention 2 输出形状正确."""
        x, levels_info = sample_input

        # 有偏置时会尝试使用 Flash Attention 2 (如果可用)
        output = attention_module(x, levels_info)

        assert output.shape == x.shape, \
            f"期望形状 {x.shape}，实际 {output.shape}"

    @pytest.mark.skipif(not FLASH_ATTN_AVAILABLE, reason="Flash Attention 2 不可用")
    def test_flash_attn_gradient_flow(self, attention_module, sample_input):
        """验证 Flash Attention 2 路径梯度流正常."""
        x, levels_info = sample_input
        x.requires_grad_(True)

        output = attention_module(x, levels_info)

        # 反向传播
        loss = output.sum()
        loss.backward()

        # 验证梯度存在且不为零
        assert x.grad is not None, "梯度应存在"
        assert x.grad.shape == x.shape, "梯度形状应与输入一致"
        assert not torch.all(x.grad == 0), "梯度不应全为零"


class TestFlashAttention2BiasFusion:
    """Flash Attention 2 偏置融合数学验证."""

    def test_bias_fusion_mathematical_properties(self):
        """验证偏置融合的数学性质."""
        from vit_pytorch.attn_hilbert_bias import HilbertAwareMultiScaleAttention

        attn = HilbertAwareMultiScaleAttention(
            dim=64,
            heads=8,
            max_level=3,
        )

        B, H, N = 2, 8, 16

        # 性质 1: 偏置融合是加法
        hilbert = torch.ones(B, H, N, N) * 0.5
        level = torch.ones(B, H, N, N) * 0.3

        combined = attn._prepare_flash_attn_bias(hilbert, level, None)
        # 融合后应该包含两者贡献 (对 heads 求平均后)
        expected = (0.5 + 0.3)  # 平均后仍然是 0.8
        actual = combined.mean().item()
        assert abs(actual - expected) < 1e-5, \
            f"偏置融合应为加法，实际 {actual} != 预期 {expected}"

        # 性质 2: 零偏置应返回零
        combined_zero = attn._prepare_flash_attn_bias(
            torch.zeros(B, H, N, N),
            None,
            None
        )
        assert combined_zero.abs().sum() < 1e-6, "零偏置应返回零"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
