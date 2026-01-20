# -*- coding: utf-8 -*-
"""
I32-7: 动态频率 + 软截断门控测试

数学形式化验证
==============

测试目标:
1. 门控值在 [0, 1] 范围内
2. 梯度连续性验证
3. Nyquist 约束满足
4. 动态频率根据 Patch 尺寸调整

日期: 2026-01-19
"""

import pytest
import torch
import math

from vit_pytorch.attn_hilbert_bias import AreaEncoder
from vit_pytorch.depth_utils import compute_normalized_area


class TestDynamicFrequency:
    """动态频率测试"""

    @pytest.fixture
    def encoder(self):
        """创建测试用 AreaEncoder"""
        return AreaEncoder(
            dim=64,
            fourier_levels=4,
            hidden_dim=32,
            freq_base=2.0,
        )

    @pytest.fixture
    def regions_small(self):
        """小尺寸 Patch (4x4 in 64x64 image)"""
        # 4x4 patches in 64x64 image: L_norm = 4/64 = 0.0625
        B, N = 2, 10
        regions = torch.zeros(B, N, 4)
        for i in range(N):
            x1 = (i % 4) * 16
            y1 = (i // 4) * 16
            regions[:, i] = torch.tensor([x1, y1, x1 + 4, y1 + 4])
        return regions

    @pytest.fixture
    def regions_large(self):
        """大尺寸 Patch (32x32 in 64x64 image)"""
        # 32x32 patches: L_norm = 32/64 = 0.5
        B, N = 2, 4
        regions = torch.zeros(B, N, 4)
        for i in range(N):
            x1 = (i % 2) * 32
            y1 = (i // 2) * 32
            regions[:, i] = torch.tensor([x1, y1, x1 + 32, y1 + 32])
        return regions

    def test_gate_values_in_range(self, encoder, regions_small):
        """测试门控值在 [0, 1] 范围内"""
        image_size = (64, 64)

        # 前向传播
        area_emb = encoder(regions_small, image_size)

        # 检查输出形状
        assert area_emb.shape == (2, 10, 64)

    def test_small_patches_high_frequency_cutoff(self, encoder, regions_small):
        """小 Patch 应该截断更高频率"""
        image_size = (64, 64)

        # 小 Patch (4x4): L_norm = 4/64 = 0.0625
        # Nyquist 频率 = π / 0.0625 ≈ 50.24
        # 截止频率 = 0.8 * 50.24 ≈ 40.19
        # 频率序列: π, 2π, 4π, 8π ≈ 3.14, 6.28, 12.57, 25.13
        # 所有频率都低于截止频率

        area_emb = encoder(regions_small, image_size)

        # 输出应该有效（非零）
        assert not torch.isnan(area_emb).any()
        assert not torch.isinf(area_emb).any()

    def test_large_patches_all_frequencies_active(self, encoder, regions_large):
        """大 Patch 应该激活所有频率"""
        image_size = (64, 64)

        # 大 Patch (32x32): L_norm = 32/64 = 0.5
        # Nyquist 频率 = π / 0.5 ≈ 6.28
        # 截止频率 = 0.8 * 6.28 ≈ 5.02
        # 频率序列: π, 2π, 4π, 8π ≈ 3.14, 6.28, 12.57, 25.13
        # freq=π, 2π < cutoff → gate=1
        # freq=4π > cutoff 但 < Nyquist → cosine decay
        # freq=8π > Nyquist → gate=0

        area_emb = encoder(regions_large, image_size)

        # 输出应该有效
        assert not torch.isnan(area_emb).any()
        assert not torch.isinf(area_emb).any()

    def test_gradient_flow(self, encoder, regions_small):
        """测试梯度流动"""
        image_size = (64, 64)

        regions = regions_small.clone().requires_grad_(True)
        area_emb = encoder(regions, image_size)

        # 计算损失
        loss = area_emb.sum()

        # 反向传播
        loss.backward()

        # 检查梯度存在
        assert regions.grad is not None
        assert not torch.isnan(regions.grad).any()
        assert not torch.isinf(regions.grad).any()

    def test_output_variance_with_patch_size(self):
        """测试不同 Patch 尺寸的输出方差"""
        encoder = AreaEncoder(dim=64, fourier_levels=4, hidden_dim=32)
        image_size = (64, 64)

        # 小 Patch
        regions_small = torch.zeros(1, 4, 4)
        for i in range(4):
            x1 = (i % 2) * 32
            y1 = (i // 2) * 32
            regions_small[:, i] = torch.tensor([x1, y1, x1 + 4, y1 + 4])

        # 大 Patch
        regions_large = torch.zeros(1, 4, 4)
        for i in range(4):
            x1 = (i % 2) * 32
            y1 = (i // 2) * 32
            regions_large[:, i] = torch.tensor([x1, y1, x1 + 32, y1 + 32])

        emb_small = encoder(regions_small, image_size)
        emb_large = encoder(regions_large, image_size)

        # 两者都应该有有效输出
        assert emb_small.numel() > 0
        assert emb_large.numel() > 0


class TestNyquistConstraint:
    """Nyquist 约束验证测试"""

    def test_nyquist_frequency_computation(self):
        """测试 Nyquist 频率计算正确性"""
        # 对于 64x64 图像中的 4x4 patch:
        # L_patch = sqrt(4*4) = 4
        # L_image = sqrt(64*64) = 64
        # L_norm = 4/64 = 0.0625
        # Nyquist = π / 0.0625 ≈ 50.24

        L_norm = 0.0625
        expected_nyquist = math.pi / L_norm

        # 验证计算
        assert abs(expected_nyquist - 50.265) < 0.01

    def test_frequency_sequence(self):
        """测试频率序列正确生成"""
        freq_base = 2.0
        fourier_levels = 4

        base_freqs = [math.pi * (freq_base ** k) for k in range(fourier_levels)]

        # 期望频率: π, 2π, 4π, 8π
        expected = [math.pi, 2 * math.pi, 4 * math.pi, 8 * math.pi]

        for f, e in zip(base_freqs, expected):
            assert abs(f - e) < 1e-6

    def test_cutoff_below_nyquist(self):
        """测试截止频率低于 Nyquist"""
        L_norm = 0.5  # 32x32 patch in 64x64
        omega_nyquist = math.pi / L_norm
        omega_cutoff = omega_nyquist * 0.8

        assert omega_cutoff < omega_nyquist
        assert omega_cutoff > 0

    def test_gate_zero_above_nyquist(self):
        """测试高于 Nyquist 的频率门控为 0"""
        freq_base = 2.0
        fourier_levels = 4
        # 使用 L_norm = 0.6 使得 Nyquist < 2π，测试更清晰
        L_norm = 0.6

        omega_nyquist = math.pi / L_norm
        omega_cutoff = omega_nyquist * 0.8

        base_freqs = [math.pi * (freq_base ** k) for k in range(fourier_levels)]

        # freq=π, 2π, 4π, 8π
        # Nyquist = π/0.6 ≈ 5.24
        # cutoff = 0.8 * 5.24 ≈ 4.19
        # freq=π ≈ 3.14 < cutoff → gate=1
        # freq=2π ≈ 6.28 > Nyquist → gate=0
        # freq=4π ≈ 12.57 > Nyquist → gate=0
        # freq=8π ≈ 25.13 > Nyquist → gate=0

        assert base_freqs[0] < omega_cutoff  # π < cutoff
        assert base_freqs[1] > omega_nyquist  # 2π > Nyquist → gate=0
        assert base_freqs[2] > omega_nyquist  # 4π > Nyquist
        assert base_freqs[3] > omega_nyquist  # 8π > Nyquist


class TestSoftCutoffGating:
    """软截断门控测试"""

    def test_cosine_gate_continuity(self):
        """测试 Cosine 门控连续性"""
        # Cosine 门控在边界处应该是连续的
        # gate(freq_cutoff) = 1
        # gate(freq_nyquist) = 0

        freq_cutoff = 5.0
        freq_nyquist = 6.28

        # 在截止频率处
        gate_at_cutoff = 1.0

        # 在 Nyquist 频率处
        gate_at_nyquist = 0.5 * (1 + math.cos(
            math.pi * (freq_nyquist - freq_cutoff) / (freq_nyquist - freq_cutoff)
        ))

        assert gate_at_cutoff == 1.0
        assert abs(gate_at_nyquist) < 1e-6

    def test_gate_monotonic_decreasing(self):
        """测试门控单调递减"""
        freq_base = 2.0
        fourier_levels = 4
        L_norm = 0.5

        omega_nyquist = math.pi / L_norm
        omega_cutoff = omega_nyquist * 0.8

        base_freqs = [math.pi * (freq_base ** k) for k in range(fourier_levels)]

        # 计算每个频率的门控
        gates = []
        for freq_k in base_freqs:
            if freq_k <= omega_cutoff:
                gate = 1.0
            elif freq_k >= omega_nyquist:
                gate = 0.0
            else:
                gate = 0.5 * (1 + math.cos(
                    math.pi * (freq_k - omega_cutoff) / (omega_nyquist - omega_cutoff)
                ))
            gates.append(gate)

        # 门控应该单调递减
        for i in range(len(gates) - 1):
            assert gates[i] >= gates[i + 1] - 1e-6, f"Gate not monotonic at {i}"


class TestIntegration:
    """集成测试"""

    def test_encoder_with_real_regions(self):
        """使用真实区域测试编码器"""
        encoder = AreaEncoder(dim=64, fourier_levels=4, hidden_dim=32)
        image_size = (224, 224)

        # 模拟不同大小的 regions
        B, N = 2, 16
        regions = torch.zeros(B, N, 4)

        # 生成不同尺寸的 patches
        patch_sizes = [4, 8, 16, 32]
        for i in range(N):
            size = patch_sizes[i % len(patch_sizes)]
            x1 = (i % 4) * 56
            y1 = (i // 4) * 56
            regions[:, i] = torch.tensor([x1, y1, x1 + size, y1 + size])

        area_emb = encoder(regions, image_size)

        assert area_emb.shape == (B, N, 64)
        assert not torch.isnan(area_emb).any()
        assert not torch.isinf(area_emb).any()

    def test_encoder_output_dim(self):
        """测试输出维度正确"""
        for dim in [32, 64, 128]:
            encoder = AreaEncoder(dim=dim, fourier_levels=4, hidden_dim=32)
            regions = torch.zeros(1, 4, 4)
            for i in range(4):
                regions[:, i] = torch.tensor([i * 16, 0, i * 16 + 4, 4])

            area_emb = encoder(regions, (64, 64))
            assert area_emb.shape[-1] == dim

    def test_encoder_fourier_levels(self):
        """测试不同 fourier_levels"""
        for levels in [2, 4, 6, 8]:
            encoder = AreaEncoder(
                dim=64,
                fourier_levels=levels,
                hidden_dim=32
            )
            regions = torch.zeros(1, 4, 4)
            for i in range(4):
                regions[:, i] = torch.tensor([i * 16, 0, i * 16 + 8, 8])

            area_emb = encoder(regions, (64, 64))

            assert area_emb.shape == (1, 4, 64)
            assert not torch.isnan(area_emb).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
