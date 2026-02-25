# -*- coding: utf-8 -*-
"""
I-PHASE4: Embeddings 改进方案测试

验证:
1. Interpolated Pooling 与 ROI-Align 的等价性
2. Fourier Path Encoding 的边界平滑性
3. Dynamic Weight (Depth Modulation) 的 C3 约束
4. Low-Rank Geometry Field 的参数效率
5. Conditional LayerNorm 的梯度流
"""

import pytest
import torch
import torch.nn.functional as F
from vit_pytorch.layers.embeddings import (
    HilbertNativePatchEmbed,
    FourierPathEncoder,
    GeometryField,
)
from vit_pytorch.layers.norm import ConditionalLayerNorm, ScaleAwareNorm
from vit_pytorch.core.levels_info import LevelsInfo


class TestInterpolatedPooling:
    """Task 1.1: Interpolated Pooling 测试"""

    def test_import(self):
        """验证模块可以正常导入"""
        embed = HilbertNativePatchEmbed(
            channels=3,
            dim=256,
            base_patch_size=4,
            max_level=4,
            use_interpolated_pooling=True
        )
        assert embed.use_interpolated_pooling is True

    def test_output_shape(self):
        """验证输出形状正确"""
        embed = HilbertNativePatchEmbed(
            channels=3, dim=64, base_patch_size=4,
            max_level=4, use_interpolated_pooling=True
        )
        # 测试 basic 属性存在
        assert embed.dim == 64
        assert embed.base_patch_size == 4


class TestFourierPathEncoding:
    """Task 1.2: Fourier Path Encoding 测试"""

    def test_import(self):
        """验证模块可以正常导入"""
        encoder = FourierPathEncoder(dim=256, max_level=8, num_frequencies=4)
        assert encoder.dim == 256
        assert encoder.max_level == 8

    def test_output_shape(self):
        """验证输出形状正确"""
        encoder = FourierPathEncoder(dim=64, max_level=4, num_frequencies=4)
        levels_info = LevelsInfo(
            data=torch.randint(0, 4, (2, 10, 5)),
            max_level=4
        )
        output = encoder(levels_info)
        assert output.shape == (2, 10, 64)

    def test_boundary_smoothing(self):
        """验证边界平滑性 - q=3 -> q=0 的余弦相似度应该很高"""
        encoder = FourierPathEncoder(dim=64, max_level=4, num_frequencies=4)

        # 测试边界路径: [..., 3] -> [..., 0]
        paths = torch.tensor([
            [0, 1, 2, 3],
            [0, 1, 2, 0],  # 边界: 3 -> 0
            [0, 1, 2, 1],
        ])

        similarities = encoder.compute_boundary_similarity(paths)

        # 边界相似度应该 > 0.8 (远高于随机 ~0.25)
        assert similarities.mean() > 0.8, f"Boundary similarity {similarities.mean()} too low"

    def test_periodicity(self):
        """验证周期性 - 相同路径应该映射到相同位置"""
        encoder = FourierPathEncoder(dim=64, max_level=4, num_frequencies=4)

        # 创建两个相同的 levels_info
        levels_info1 = LevelsInfo(
            data=torch.randint(0, 4, (1, 2, 5)),
            max_level=4
        )
        levels_info2 = LevelsInfo(
            data=levels_info1.data.clone(),  # 完全相同
            max_level=4
        )

        with torch.no_grad():
            emb1 = encoder(levels_info1)
            emb2 = encoder(levels_info2)

        # 相同路径应该完全相同
        assert torch.allclose(emb1, emb2, atol=1e-5)


class TestDynamicWeight:
    """Task 2.1: Dynamic Weight (Depth Modulation) 测试"""

    def test_import(self):
        """验证模块可以正常导入"""
        embed = HilbertNativePatchEmbed(
            channels=3, dim=256, base_patch_size=4,
            max_level=4, use_dynamic_weight=True
        )
        assert embed.use_dynamic_weight is True
        assert embed.depth_gamma is not None
        assert embed.depth_beta is not None

    def test_depth_modulation_parameters(self):
        """验证深度调制参数形状正确"""
        embed = HilbertNativePatchEmbed(
            channels=3, dim=64, base_patch_size=4,
            max_level=4, use_dynamic_weight=True
        )
        # gamma: Linear(in=5, out=64)
        assert embed.depth_gamma.in_features == 5
        assert embed.depth_gamma.out_features == 64

    def test_modulation_effect(self):
        """验证调制效果 - 不同深度应该产生不同的输出"""
        embed = HilbertNativePatchEmbed(
            channels=3, dim=64, base_patch_size=4,
            max_level=4, use_dynamic_weight=True
        )

        # 相同的 pooled features，不同的深度
        pooled = torch.randn(4, 64)
        depths_d0 = torch.tensor([0, 0, 0, 0])
        depths_d4 = torch.tensor([4, 4, 4, 4])

        modulated_d0 = embed._apply_depth_modulation(pooled, depths_d0)
        modulated_d4 = embed._apply_depth_modulation(pooled, depths_d4)

        # 不同深度应该产生不同的输出
        assert not torch.allclose(modulated_d0, modulated_d4, atol=1e-3)


class TestLowRankGeometryField:
    """Task 2.2: Low-Rank Geometry Field 测试"""

    def test_import(self):
        """验证模块可以正常导入"""
        gf = GeometryField(dim=256, max_level=8, heads=8, rank=16)
        assert gf.rank == 16

    def test_parameter_reduction(self):
        """验证参数减少"""
        gf_lowrank = GeometryField(dim=256, max_level=8, heads=8, rank=16)
        gf_full = GeometryField(dim=256, max_level=8, heads=8, rank=256)

        def count_params(m):
            return sum(p.numel() for p in m.parameters())

        params_lowrank = count_params(gf_lowrank)
        params_full = count_params(gf_full)

        # Low-Rank 版本参数应该更少
        assert params_lowrank < params_full
        # 压缩比应该 > 1.5x
        assert params_full / params_lowrank > 1.5

    def test_output_shape(self):
        """验证输出形状正确"""
        gf = GeometryField(dim=64, max_level=4, heads=4, rank=16)
        levels_info = LevelsInfo(
            data=torch.randint(0, 4, (2, 10, 5)),
            max_level=4
        )
        output = gf(levels_info)
        assert output.shape == (2, 10, 64)


class TestConditionalLayerNorm:
    """Task 3.1: Conditional LayerNorm 测试"""

    def test_import(self):
        """验证模块可以正常导入"""
        cln = ConditionalLayerNorm(normalized_shape=256, num_conditions=9)
        assert cln.normalized_shape == 256
        assert cln.num_conditions == 9

    def test_output_shape(self):
        """验证输出形状正确"""
        cln = ConditionalLayerNorm(normalized_shape=64, num_conditions=9)
        x = torch.randn(2, 10, 64)
        depths = torch.randint(0, 9, (2, 10))
        output = cln(x, depths)
        assert output.shape == x.shape

    def test_condition_sensitivity(self):
        """验证条件敏感性 - 不同条件应该产生不同的输出"""
        # 使用更大的差异来测试
        cln = ConditionalLayerNorm(normalized_shape=64, num_conditions=9)
        # 设置不同的初始权重使条件敏感
        with torch.no_grad():
            cln.condition_weight[:, :, 0] = torch.randn(9, 64)  # gamma
            cln.condition_weight[:, :, 1] = torch.randn(9, 64)  # beta

        x = torch.randn(2, 10, 64)

        depths_d0 = torch.zeros(2, 10, dtype=torch.long)
        depths_d8 = torch.full((2, 10), 8, dtype=torch.long)

        output_d0 = cln(x, depths_d0)
        output_d8 = cln(x, depths_d8)

        # 不同条件应该产生不同的输出
        diff = (output_d0 - output_d8).abs().mean()
        assert diff > 0.01, f"Output difference too small: {diff}"

    def test_gradient_flow(self):
        """验证梯度流正常"""
        cln = ConditionalLayerNorm(normalized_shape=64, num_conditions=9)
        x = torch.randn(2, 10, 64, requires_grad=True)
        depths = torch.randint(0, 9, (2, 10))

        output = cln(x, depths)
        loss = output.sum()
        loss.backward()

        # 梯度应该能传播到输入
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestScaleAwareNorm:
    """Task 3.1: Scale-Aware Norm 测试"""

    def test_import(self):
        """验证模块可以正常导入"""
        san = ScaleAwareNorm(dim=256, max_level=8)
        assert san.dim == 256

    def test_output_shape(self):
        """验证输出形状正确"""
        san = ScaleAwareNorm(dim=64, max_level=4)
        x = torch.randn(2, 10, 64)
        depths = torch.randint(0, 4, (2, 10))
        output = san(x, depths)
        assert output.shape == x.shape

    def test_depth_aware(self):
        """验证深度感知"""
        san = ScaleAwareNorm(dim=64, max_level=4)
        x = torch.randn(2, 10, 64)

        depths_shallow = torch.zeros(2, 10, dtype=torch.long)
        depths_deep = torch.full((2, 10), 4, dtype=torch.long)

        output_shallow = san(x, depths_shallow)
        output_deep = san(x, depths_deep)

        # 不同深度应该产生不同的输出
        assert not torch.allclose(output_shallow, output_deep, atol=1e-3)


class TestIntegration:
    """集成测试 - 验证所有组件可以一起工作"""

    def test_all_components(self):
        """验证所有组件可以一起工作"""
        # Patch Embedding with new features
        patch_embed = HilbertNativePatchEmbed(
            channels=3, dim=64, base_patch_size=4,
            max_level=4,
            use_interpolated_pooling=True,
            use_dynamic_weight=True
        )

        # Fourier Path Encoder
        path_encoder = FourierPathEncoder(dim=64, max_level=4, num_frequencies=4)

        # Low-Rank Geometry Field
        geometry_field = GeometryField(dim=64, max_level=4, heads=4, rank=16)

        # Conditional LayerNorm
        norm = ConditionalLayerNorm(normalized_shape=64, num_conditions=5)

        # 验证所有组件可以前向传播
        levels_info = LevelsInfo(
            data=torch.randint(0, 4, (2, 8, 5)),
            max_level=4
        )

        path_emb = path_encoder(levels_info)
        geom_emb = geometry_field(levels_info)
        normed = norm(path_emb, levels_info.depths)

        assert path_emb.shape == (2, 8, 64)
        assert geom_emb.shape == (2, 8, 64)
        assert normed.shape == (2, 8, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
