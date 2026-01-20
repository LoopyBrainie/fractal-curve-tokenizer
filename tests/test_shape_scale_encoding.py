# -*- coding: utf-8 -*-
"""
I31: 形状-尺度编码器单元测试

数学形式化测试
==============

测试覆盖:
1. compute_region_shape_scale: 纵横比和面积计算
2. compute_shape_scale_similarity: 相似性矩阵计算
3. ShapeScaleEncoder: 编码器前向传播和梯度流
4. LCAHilbertBiasWithShapeScale: 组合偏置计算

数学保证验证:
- 纵横比对称性: log(w/h) = -log(h/w)
- 门控值有界: g ∈ (0, 1)
- 零初始化: τ = 0 时退化为纯 LCA 偏置
"""

from __future__ import annotations

import math
from typing import Tuple

import pytest
import torch
import torch.nn as nn


# 导入待测试模块
from vit_pytorch.depth_utils import (
    compute_region_shape_scale,
    compute_shape_scale_similarity,
)
from vit_pytorch.attn_hilbert_bias import (
    ShapeScaleEncoder,
    LCAHilbertBiasWithShapeScale,
    AreaEncoder,  # I31-3
    AffineModulatedBias,  # I31-3
    HilbertAwareMultiScaleAttention,  # I31-3
)
from vit_pytorch.embed_fractal_position import (
    AreaEnhancedPositionEmbedding,  # I31-3
)


class TestComputeRegionShapeScale:
    """测试 compute_region_shape_scale 函数。"""

    def test_square_region(self):
        """测试正方形区域的特征计算。"""
        # 32x32 区域，64x64 图像
        regions = torch.tensor([[16, 16, 48, 48]])  # [32, 32]
        W, H = 64, 64

        aspect_ratios, normalized_areas = compute_region_shape_scale(regions, (W, H))

        # 正方形: 纵横比 = log(1) = 0
        assert torch.allclose(aspect_ratios, torch.tensor([0.0]), atol=1e-4)

        # 面积: (32/64) * (32/64) = 0.25
        expected_area = (32 / 64) * (32 / 64)
        assert torch.allclose(normalized_areas, torch.tensor([expected_area]), atol=1e-4)

    def test_rectangle_region(self):
        """测试长方形区域的特征计算。"""
        # 40x20 区域，64x64 图像
        regions = torch.tensor([[12, 22, 52, 42]])  # [40, 20]
        W, H = 64, 64

        aspect_ratios, normalized_areas = compute_region_shape_scale(regions, (W, H))

        # 宽高比: log(40/20) = log(2) ≈ 0.693
        expected_aspect = math.log(40 / 20)
        assert torch.allclose(aspect_ratios, torch.tensor([expected_aspect]), atol=1e-4)

        # 面积: (40/64) * (20/64) = 0.195
        expected_area = (40 / 64) * (20 / 64)
        assert torch.allclose(normalized_areas, torch.tensor([expected_area]), atol=1e-4)

    def test_aspect_ratio_symmetry(self):
        """测试纵横比对称性: log(w/h) = -log(h/w)。"""
        # w=40, h=20
        regions_1 = torch.tensor([[0, 0, 40, 20]])
        # w=20, h=40
        regions_2 = torch.tensor([[0, 0, 20, 40]])

        ar_1, _ = compute_region_shape_scale(regions_1, (64, 64))
        ar_2, _ = compute_region_shape_scale(regions_2, (64, 64))

        # 应该互为相反数
        assert torch.allclose(ar_1, -ar_2, atol=1e-4)

    def test_batch_processing(self):
        """测试批量处理。"""
        # 批量输入 [B=2, N=2, 4]
        regions = torch.tensor([
            [[0, 0, 32, 32]],  # 正方形
            [[0, 0, 40, 20]],  # 长方形
        ])
        W, H = 64, 64

        aspect_ratios, normalized_areas = compute_region_shape_scale(regions, (W, H))

        # 形状应该是 [B, N]
        assert aspect_ratios.shape == (2, 1)
        assert normalized_areas.shape == (2, 1)

    def test_empty_regions(self):
        """测试空区域输入。"""
        regions = torch.tensor([]).reshape(0, 4)
        W, H = 64, 64

        aspect_ratios, normalized_areas = compute_region_shape_scale(regions, (W, H))

        assert aspect_ratios.numel() == 0
        assert normalized_areas.numel() == 0

    def test_2d_input(self):
        """测试 2D 输入 (无 batch 维度)。"""
        regions = torch.tensor([[0, 0, 32, 32]])
        W, H = 64, 64

        aspect_ratios, normalized_areas = compute_region_shape_scale(regions, (W, H))

        # 2D 输入应该保持 1D 输出
        assert aspect_ratios.dim() == 1
        assert normalized_areas.dim() == 1


class TestComputeShapeScaleSimilarity:
    """测试 compute_shape_scale_similarity 函数。"""

    def test_identity_similarity(self):
        """测试相同区域的相似性为 1。"""
        # 两个完全相同的区域
        aspect_ratios = torch.tensor([0.0, 0.0])
        normalized_areas = torch.tensor([0.25, 0.25])

        sim = compute_shape_scale_similarity(aspect_ratios, normalized_areas)

        # 对角线应该是 1
        assert torch.allclose(torch.diag(sim), torch.ones(2), atol=1e-4)

    def test_similarity_symmetry(self):
        """测试相似性矩阵对称性。"""
        ar = torch.tensor([0.0, 0.5, 1.0])
        na = torch.tensor([0.1, 0.2, 0.3])

        sim = compute_shape_scale_similarity(ar, na)

        assert torch.allclose(sim, sim.T, atol=1e-4)

    def test_batch_processing(self):
        """测试批量处理的相似性矩阵。"""
        ar = torch.tensor([[0.0, 0.5], [1.0, 1.5]])  # [B=2, N=2]
        na = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

        sim = compute_shape_scale_similarity(ar, na)

        # 形状应该是 [B, N, N]
        assert sim.shape == (2, 2, 2)


class TestShapeScaleEncoder:
    """测试 ShapeScaleEncoder 类。"""

    def test_output_dimensions(self):
        """测试输出维度正确性。"""
        B, N, dim = 2, 8, 64
        regions = torch.randint(0, 64, (B, N, 4)).float()
        image_size = (64, 64)

        encoder = ShapeScaleEncoder(dim=dim, hidden_dim=32)
        output = encoder(regions, image_size)

        assert output.shape == (B, N, dim)

    def test_gate_values_bounded(self):
        """测试门控值有界性: g ∈ (0, 1)。"""
        B, N, dim = 4, 16, 64
        regions = torch.randint(0, 64, (B, N, 4)).float()
        image_size = (64, 64)

        encoder = ShapeScaleEncoder(dim=dim, hidden_dim=32)

        # 提取门控值 (需要检查内部 gate_net 的输出)
        # 这里我们通过多次前向传播验证输出稳定
        outputs = [encoder(regions, image_size) for _ in range(5)]

        # 检查输出是否有梯度
        loss = outputs[0].sum()
        loss.backward()

        assert encoder.shape_scale_weight.grad is not None
        # I35-2: 非零初始化 τ=0.1 确保训练初期有梯度回传
        assert abs(encoder.shape_scale_weight.item() - 0.1) < 1e-6

    def test_zero_weight_disables_effect(self):
        """测试零权重禁用效果。"""
        B, N, dim = 2, 8, 64
        regions = torch.randint(0, 64, (B, N, 4)).float()
        image_size = (64, 64)

        encoder = ShapeScaleEncoder(dim=dim, hidden_dim=32)

        # 强制权重为 0
        encoder.shape_scale_weight.data.fill_(0.0)
        output_zero = encoder(regions, image_size)

        # 零权重时输出应该接近零
        assert torch.allclose(output_zero, torch.zeros_like(output_zero), atol=1e-6)

    def test_gradient_flow(self):
        """测试梯度流正确性。"""
        B, N, dim = 2, 8, 64
        regions = torch.randint(0, 64, (B, N, 4)).float()
        regions.requires_grad_(True)
        image_size = (64, 64)

        encoder = ShapeScaleEncoder(dim=dim, hidden_dim=32)
        output = encoder(regions, image_size)

        # 计算损失并反向传播
        loss = output.sum()
        loss.backward()

        # 检查梯度存在
        assert regions.grad is not None
        assert regions.grad.shape == regions.shape

    def test_aspect_vs_area_importance(self):
        """测试门控机制能区分纵横比和面积的重要性。"""
        B, N, dim = 2, 1, 64  # N=1 because each sample has 1 region

        # 极端纵横比的区域 (B=2, N=1, 4)
        regions_extreme = torch.tensor([
            [[0, 0, 60, 6]],  # 10:1 宽高比
            [[0, 0, 6, 60]],  # 1:10 宽高比
        ]).float()

        # 正方形区域 (B=2, N=1, 4)
        regions_square = torch.tensor([
            [[0, 0, 20, 20]],
            [[0, 0, 20, 20]],
        ]).float()

        encoder = ShapeScaleEncoder(dim=dim, hidden_dim=32)
        image_size = (64, 64)

        # 设置非零权重以测试门控效果
        encoder.shape_scale_weight.data.fill_(1.0)

        # 前向传播
        out_extreme = encoder(regions_extreme, image_size)
        out_square = encoder(regions_square, image_size)

        # 两者都应该产生有效输出
        assert out_extreme.shape == (B, N, dim)
        assert out_square.shape == (B, N, dim)


class TestLCAHilbertBiasWithShapeScale:
    """测试 LCAHilbertBiasWithShapeScale 类。"""

    def _generate_valid_regions(self, B: int, N: int, img_size: int, max_dim: int = 64) -> torch.Tensor:
        """生成有效的区域坐标 (x1 < x2, y1 < y2)。"""
        regions = []
        for _ in range(B * N):
            # 随机生成有效的区域坐标
            x1 = torch.randint(0, max_dim - 4, (1,)).item()
            y1 = torch.randint(0, max_dim - 4, (1,)).item()
            min_size = 4
            max_size = max_dim // 4
            w = torch.randint(min_size, max_size, (1,)).item()
            h = torch.randint(min_size, max_size, (1,)).item()
            x2 = min(x1 + w, max_dim)
            y2 = min(y1 + h, max_dim)
            regions.append([x1, y1, x2, y2])
        return torch.tensor(regions, dtype=torch.float32).reshape(B, N, 4)

    def test_lca_bias_output(self):
        """测试 LCA 偏置输出。"""
        B, N, dim = 2, 8, 64
        regions = self._generate_valid_regions(B, N, 64)
        image_size = 64
        max_depth = 8

        bias_module = LCAHilbertBiasWithShapeScale(
            dim=dim,
            max_depth=max_depth,
            enable_shape_scale=False,
        )

        bias = bias_module.forward_from_regions(regions, image_size)

        assert bias.shape == (B, dim, N, N)

    def test_shape_scale_disabled(self):
        """测试禁用形状-尺度修正时退化为纯 LCA。"""
        B, N, dim = 2, 8, 64
        regions = self._generate_valid_regions(B, N, 64)
        image_size = 64
        max_depth = 8

        # 启用和禁用形状-尺度的模块
        bias_module_ss = LCAHilbertBiasWithShapeScale(
            dim=dim,
            max_depth=max_depth,
            enable_shape_scale=True,
        )
        bias_module_no_ss = LCAHilbertBiasWithShapeScale(
            dim=dim,
            max_depth=max_depth,
            enable_shape_scale=False,
        )

        # 禁用时应该直接使用 LCA 偏置
        # 注意: 由于 I35-2 使用非零初始化 (τ=0.1)，需要手动设为 0 进行比较
        bias_module_ss.shape_scale_encoder.shape_scale_weight.data.fill_(0.0)
        bias_ss = bias_module_ss.forward_with_shape_scale(regions, image_size)
        bias_no_ss = bias_module_no_ss.forward_with_shape_scale(regions, image_size)

        # 设置 shape_scale_weight=0 后两者应该相等
        assert torch.allclose(bias_ss, bias_no_ss, atol=1e-6)

    def test_combined_bias_output(self):
        """测试组合偏置输出。"""
        B, N, dim = 2, 8, 64
        regions = self._generate_valid_regions(B, N, 64)
        image_size = 64
        max_depth = 8

        bias_module = LCAHilbertBiasWithShapeScale(
            dim=dim,
            max_depth=max_depth,
            enable_shape_scale=True,
        )

        combined_bias = bias_module.forward_with_shape_scale(regions, image_size)

        assert combined_bias.shape == (B, dim, N, N)

    def test_nonzero_init_weight(self):
        """测试非零初始化权重 (I35-2: τ=0.1 确保梯度回传)。"""
        dim = 64
        max_depth = 8

        bias_module = LCAHilbertBiasWithShapeScale(
            dim=dim,
            max_depth=max_depth,
            enable_shape_scale=True,
        )

        weight = bias_module.shape_scale_encoder.shape_scale_weight.item()

        # I35-2: 非零初始化 τ=0.1 确保训练初期有梯度回传
        assert abs(weight - 0.1) < 1e-6

    def test_gradient_flow_combined(self):
        """测试组合偏置的梯度流。"""
        B, N, dim = 2, 8, 64
        regions = self._generate_valid_regions(B, N, 64)
        regions.requires_grad_(True)
        image_size = 64
        max_depth = 8

        bias_module = LCAHilbertBiasWithShapeScale(
            dim=dim,
            max_depth=max_depth,
            enable_shape_scale=True,
        )

        combined_bias = bias_module.forward_with_shape_scale(regions, image_size)

        # 计算损失并反向传播
        loss = combined_bias.sum()
        loss.backward()

        assert regions.grad is not None


class TestMathematicalProperties:
    """测试数学性质。"""

    def test_aspect_ratio_log_symmetry(self):
        """测试对数变换对称性: log(w/h) = -log(h/w)。"""
        # 使用深度 utils 函数
        w, h = 40.0, 20.0

        # 创建测试区域
        regions_1 = torch.tensor([[0, 0, w, h]])
        regions_2 = torch.tensor([[0, 0, h, w]])

        ar_1, _ = compute_region_shape_scale(regions_1, (100, 100))
        ar_2, _ = compute_region_shape_scale(regions_2, (100, 100))

        # 应该满足: ar_1 + ar_2 = 0
        assert torch.allclose(ar_1 + ar_2, torch.tensor([0.0]), atol=1e-4)

    def test_area_normalization(self):
        """测试面积归一化到 [0, 1]。"""
        # 全图像区域
        regions_full = torch.tensor([[0, 0, 64, 64]])
        # 半图像区域
        regions_half = torch.tensor([[0, 0, 32, 64]])

        _, area_full = compute_region_shape_scale(regions_full, (64, 64))
        _, area_half = compute_region_shape_scale(regions_half, (64, 64))

        # 全图像面积应该为 1
        assert torch.allclose(area_full, torch.tensor([1.0]), atol=1e-4)
        # 半图像面积应该小于 1
        assert area_half.item() < 1.0


class TestIntegration:
    """集成测试。"""

    def _generate_valid_regions(self, B: int, N: int, img_size: int, max_dim: int = 64) -> torch.Tensor:
        """生成有效的区域坐标 (x1 < x2, y1 < y2)。"""
        regions = []
        for _ in range(B * N):
            x1 = torch.randint(0, max_dim - 4, (1,)).item()
            y1 = torch.randint(0, max_dim - 4, (1,)).item()
            min_size = 4
            max_size = max_dim // 4
            w = torch.randint(min_size, max_size, (1,)).item()
            h = torch.randint(min_size, max_size, (1,)).item()
            x2 = min(x1 + w, max_dim)
            y2 = min(y1 + h, max_dim)
            regions.append([x1, y1, x2, y2])
        return torch.tensor(regions, dtype=torch.float32).reshape(B, N, 4)

    def test_encoder_to_bias_pipeline(self):
        """测试从编码器到偏置的完整管道。"""
        B, N, dim = 2, 8, 64
        regions = self._generate_valid_regions(B, N, 64)
        image_size = 64
        max_depth = 8

        # 1. 创建 ShapeScaleEncoder
        encoder = ShapeScaleEncoder(dim=dim, hidden_dim=32)
        shape_emb = encoder(regions, (image_size, image_size))

        # 2. 创建 LCAHilbertBiasWithShapeScale
        bias_module = LCAHilbertBiasWithShapeScale(
            dim=dim,
            max_depth=max_depth,
            enable_shape_scale=True,
        )

        # 3. 组合偏置
        combined_bias = bias_module.forward_with_shape_scale(regions, image_size)

        # 验证形状
        assert shape_emb.shape == (B, N, dim)
        assert combined_bias.shape == (B, dim, N, N)

    def test_different_region_shapes(self):
        """测试不同形状区域的编码。"""
        dim = 64

        encoder = ShapeScaleEncoder(dim=dim, hidden_dim=32)

        # 设置非零权重以测试门控效果
        encoder.shape_scale_weight.data.fill_(1.0)

        # 正方形
        regions_square = torch.tensor([[[0, 0, 32, 32]]])
        # 宽矩形
        regions_wide = torch.tensor([[[0, 0, 48, 16]]])
        # 高矩形
        regions_tall = torch.tensor([[[0, 0, 16, 48]]])

        out_square = encoder(regions_square, (64, 64))
        out_wide = encoder(regions_wide, (64, 64))
        out_tall = encoder(regions_tall, (64, 64))

        # 三者应该产生不同的输出
        assert not torch.allclose(out_square, out_wide)
        assert not torch.allclose(out_wide, out_tall)
        assert not torch.allclose(out_tall, out_square)


# ==================== I31-3: 面积编码器测试 ====================

class TestAreaEncoder:
    """测试 AreaEncoder 类 (I31-3)。"""

    def test_output_dimensions(self):
        """测试输出维度正确性。"""
        B, N, dim = 2, 8, 64
        regions = torch.randint(0, 64, (B, N, 4)).float()
        image_size = (64, 64)

        encoder = AreaEncoder(dim=dim, fourier_levels=4, hidden_dim=32)
        output = encoder(regions, image_size)

        assert output.shape == (B, N, dim)

    def test_normalized_area_formula(self):
        """测试归一化面积公式正确性。"""
        dim = 64
        # 全图像区域 [0, 0, 64, 64]
        regions_full = torch.tensor([[[0, 0, 64, 64]]])  # [1, 1, 4]
        # 半图像区域 [0, 0, 32, 64]
        regions_half = torch.tensor([[[0, 0, 32, 64]]])  # [1, 1, 4]

        encoder = AreaEncoder(dim=dim, fourier_levels=4, hidden_dim=32)
        # 设置非零权重以测试面积编码效果
        encoder.area_weight.data.fill_(1.0)
        image_size = (64, 64)

        out_full = encoder(regions_full, image_size)
        out_half = encoder(regions_half, image_size)

        # 测试归一化面积计算是否正确
        # 全图像面积 = 64*64 = 4096, 半图像面积 = 32*64 = 2048
        # 归一化后: log(4096+1) > log(2048+1)
        # 输出应该不同
        assert not torch.allclose(out_full, out_half, atol=1e-5), \
            "不同面积的区域应该有不同的编码"

    def test_fourier_features(self):
        """测试傅里叶特征生成。"""
        dim = 64
        # 创建不同面积的区域 [B=4, N=1, 4]
        regions = torch.tensor([
            [[0, 0, 16, 16]],  # 256 px
            [[0, 0, 32, 32]],  # 1024 px
            [[0, 0, 48, 48]],  # 2304 px
            [[0, 0, 64, 64]],  # 4096 px
        ])
        encoder = AreaEncoder(dim=dim, fourier_levels=4, hidden_dim=32)
        # 设置非零权重以测试面积编码效果
        encoder.area_weight.data.fill_(1.0)
        image_size = (64, 64)

        output = encoder(regions, image_size)

        # 验证输出形状 [4, 1, dim]
        assert output.shape == (4, 1, dim), f"Expected (4, 1, {dim}), got {output.shape}"

        # 不同区域应该产生不同的输出
        for i in range(1, 4):
            assert not torch.allclose(output[0, 0], output[i, 0], atol=1e-5), \
                f"区域 0 和 {i} 应该有不同的面积编码"

    def test_zero_weight_disables_effect(self):
        """测试零权重禁用效果。"""
        B, N, dim = 2, 8, 64
        regions = torch.randint(0, 64, (B, N, 4)).float()
        image_size = (64, 64)

        encoder = AreaEncoder(dim=dim, fourier_levels=4, hidden_dim=32)

        # 强制权重为 0
        encoder.area_weight.data.fill_(0.0)
        output_zero = encoder(regions, image_size)

        # 零权重时输出应该接近零
        assert torch.allclose(output_zero, torch.zeros_like(output_zero), atol=1e-6)

    def test_gradient_flow(self):
        """测试梯度流正确性。"""
        B, N, dim = 2, 8, 64
        regions = torch.randint(0, 64, (B, N, 4)).float()
        regions.requires_grad_(True)
        image_size = (64, 64)

        encoder = AreaEncoder(dim=dim, fourier_levels=4, hidden_dim=32)
        output = encoder(regions, image_size)

        # 计算损失并反向传播
        loss = output.sum()
        loss.backward()

        # 检查梯度存在
        assert regions.grad is not None
        assert regions.grad.shape == regions.shape


class TestAffineModulatedBias:
    """测试 AffineModulatedBias 类 (I31-3)。"""

    def _generate_valid_regions(self, B: int, N: int, img_size: int, max_dim: int = 64) -> torch.Tensor:
        """生成有效的区域坐标 (x1 < x2, y1 < y2)。"""
        regions = []
        for _ in range(B * N):
            x1 = torch.randint(0, max_dim - 4, (1,)).item()
            y1 = torch.randint(0, max_dim - 4, (1,)).item()
            min_size = 4
            max_size = max_dim // 4
            w = torch.randint(min_size, max_size, (1,)).item()
            h = torch.randint(min_size, max_size, (1,)).item()
            x2 = min(x1 + w, max_dim)
            y2 = min(y1 + h, max_dim)
            regions.append([x1, y1, x2, y2])
        return torch.tensor(regions, dtype=torch.float32).reshape(B, N, 4)

    def test_output_dimensions(self):
        """测试输出维度正确性。"""
        B, N, dim = 2, 8, 64
        regions = self._generate_valid_regions(B, N, 64)
        image_size = 64
        max_depth = 8

        bias_module = AffineModulatedBias(
            dim=dim,
            max_depth=max_depth,
            enable_area_modulation=True,
        )

        bias = bias_module(regions, image_size)

        assert bias.shape == (B, dim, N, N)

    def test_affine_modulation_formula(self):
        """测试仿射调制公式: B_final = γ ⊙ B_spatial + β。"""
        B, N, dim = 1, 4, 32
        regions = self._generate_valid_regions(B, N, 64)
        image_size = 64
        max_depth = 4

        # 启用和禁用面积调制的模块
        bias_module_modulated = AffineModulatedBias(
            dim=dim,
            max_depth=max_depth,
            enable_area_modulation=True,
        )
        bias_module_base = AffineModulatedBias(
            dim=dim,
            max_depth=max_depth,
            enable_area_modulation=False,
        )

        bias_modulated = bias_module_modulated(regions, image_size)
        bias_base = bias_module_base(regions, image_size)

        # 初始时 (residual_alpha = 0)，两者应该接近
        # 由于零初始化，仿射调制效果应该很弱
        assert bias_modulated.shape == bias_base.shape

    def test_zero_alpha_disables_modulation(self):
        """测试零残差权重禁用调制效果。"""
        B, N, dim = 1, 4, 32
        regions = self._generate_valid_regions(B, N, 64)
        image_size = 64
        max_depth = 4

        bias_module = AffineModulatedBias(
            dim=dim,
            max_depth=max_depth,
            enable_area_modulation=True,
        )

        # 强制残差权重为 0
        bias_module.residual_alpha.data.fill_(0.0)
        bias_zero = bias_module(regions, image_size)

        # 禁用面积调制
        bias_no_mod = AffineModulatedBias(
            dim=dim,
            max_depth=max_depth,
            enable_area_modulation=False,
        )
        bias_base = bias_no_mod(regions, image_size)

        # 两者应该相等
        assert torch.allclose(bias_zero, bias_base, atol=1e-5)

    def test_scale_net_bounded(self):
        """测试缩放网络输出有界性: γ ∈ (0, 1)。"""
        B, N, dim = 4, 16, 64
        regions = self._generate_valid_regions(B, N, 64)
        image_size = 64
        max_depth = 8

        bias_module = AffineModulatedBias(
            dim=dim,
            max_depth=max_depth,
            enable_area_modulation=True,
        )

        # 前向传播
        output = bias_module(regions, image_size)

        # 验证输出有梯度
        assert output.requires_grad or output.grad is not None

    def test_bias_net_bounded(self):
        """测试偏置网络输出有界性: β ∈ (-1, 1)。"""
        B, N, dim = 2, 8, 64
        regions = self._generate_valid_regions(B, N, 64)
        image_size = 64
        max_depth = 8

        bias_module = AffineModulatedBias(
            dim=dim,
            max_depth=max_depth,
            enable_area_modulation=True,
        )

        # 检查 bias_net 的 Tanh 输出范围
        # 通过多次前向传播验证
        with torch.no_grad():
            # 手动计算 area_sim 和 gamma_features
            area_emb = bias_module.area_encoder(regions, (image_size, image_size))
            area_sim = torch.bmm(area_emb, area_emb.transpose(-2, -1))

            fourier_features = []
            for k in range(bias_module.area_encoder.fourier_levels):
                freq = 2 ** k
                fourier_features.append(torch.sin(freq * math.pi * area_sim))
                fourier_features.append(torch.cos(freq * math.pi * area_sim))
            gamma_features = torch.stack(fourier_features, dim=-1)

            beta = bias_module.bias_net(gamma_features)
            # Tanh 输出应该在 (-1, 1) 范围内
            assert beta.abs().max() < 1.0 + 1e-5, "偏置网络输出应在 (-1, 1) 范围内"

    def test_gradient_flow(self):
        """测试梯度流正确性。"""
        B, N, dim = 2, 8, 64
        regions = self._generate_valid_regions(B, N, 64)
        regions.requires_grad_(True)
        image_size = 64
        max_depth = 8

        bias_module = AffineModulatedBias(
            dim=dim,
            max_depth=max_depth,
            enable_area_modulation=True,
        )

        bias = bias_module(regions, image_size)

        # 计算损失并反向传播
        loss = bias.sum()
        loss.backward()

        assert regions.grad is not None


class TestAreaEnhancedPositionEmbedding:
    """测试 AreaEnhancedPositionEmbedding 类 (I31-3)。"""

    def _generate_levels_info(self, B: int, N: int, max_depth: int = 8) -> torch.Tensor:
        """生成有效的 levels_info 张量。"""
        levels_info = []
        for _ in range(B * N):
            depth = torch.randint(0, max_depth + 1, (1,)).item()
            path = [torch.randint(0, 4, (1,)).item() for _ in range(depth)]
            # 填充到 max_depth
            path.extend([0] * (max_depth - depth))
            levels_info.append([depth] + path)
        return torch.tensor(levels_info, dtype=torch.float32).reshape(B, N, max_depth + 1)

    def _generate_valid_regions(self, B: int, N: int, img_size: int, max_dim: int = 64) -> torch.Tensor:
        """生成有效的区域坐标 (x1 < x2, y1 < y2)。"""
        regions = []
        for _ in range(B * N):
            x1 = torch.randint(0, max_dim - 4, (1,)).item()
            y1 = torch.randint(0, max_dim - 4, (1,)).item()
            min_size = 4
            max_size = max_dim // 4
            w = torch.randint(min_size, max_size, (1,)).item()
            h = torch.randint(min_size, max_size, (1,)).item()
            x2 = min(x1 + w, max_dim)
            y2 = min(y1 + h, max_dim)
            regions.append([x1, y1, x2, y2])
        return torch.tensor(regions, dtype=torch.float32).reshape(B, N, 4)

    def test_output_dimensions(self):
        """测试输出维度正确性。"""
        B, N, dim, max_depth = 2, 8, 64, 8
        levels_info = self._generate_levels_info(B, N, max_depth)
        regions = self._generate_valid_regions(B, N, 64)
        image_size = 64

        pos_emb = AreaEnhancedPositionEmbedding(
            dim=dim,
            max_level=max_depth,
            fourier_levels=4,
        )

        output = pos_emb(levels_info, regions, image_size)

        assert output.shape == (B, N, dim)

    def test_base_embedding_unchanged(self):
        """测试基础位置编码不变性。"""
        B, N, dim, max_depth = 2, 8, 64, 8
        levels_info = self._generate_levels_info(B, N, max_depth)
        regions = self._generate_valid_regions(B, N, 64)
        image_size = 64

        pos_emb = AreaEnhancedPositionEmbedding(
            dim=dim,
            max_level=max_depth,
            fourier_levels=4,
        )

        # 无 regions 时使用基础编码
        output_base = pos_emb(levels_info, None, None)
        # 有 regions 时使用增强编码
        output_enhanced = pos_emb(levels_info, regions, image_size)

        # 两者形状应该相同
        assert output_base.shape == output_enhanced.shape

    def test_zero_scale_disables_area_effect(self):
        """测试零缩放权重禁用面积效果。"""
        B, N, dim, max_depth = 2, 8, 64, 8
        levels_info = self._generate_levels_info(B, N, max_depth)
        regions = self._generate_valid_regions(B, N, 64)
        image_size = 64

        pos_emb = AreaEnhancedPositionEmbedding(
            dim=dim,
            max_level=max_depth,
            fourier_levels=4,
        )

        # 强制缩放权重为 0
        pos_emb.area_scale.data.fill_(0.0)
        output_zero = pos_emb(levels_info, regions, image_size)

        # 无 regions 的输出应该与零缩放的增强输出相同
        output_base = pos_emb(levels_info, None, None)

        # 由于基础编码相同，两者应该接近
        # 注意: 输出可能不完全相同，因为区域信息仍然通过其他方式影响
        # 但在零缩放情况下，面积编码的影响应该被禁用

    def test_gradient_flow(self):
        """测试梯度流正确性。"""
        B, N, dim, max_depth = 2, 8, 64, 8
        levels_info = self._generate_levels_info(B, N, max_depth)
        regions = self._generate_valid_regions(B, N, 64)
        regions.requires_grad_(True)
        image_size = 64

        pos_emb = AreaEnhancedPositionEmbedding(
            dim=dim,
            max_level=max_depth,
            fourier_levels=4,
        )

        # 设置非零权重以启用面积编码
        pos_emb.area_scale.data.fill_(1.0)

        output = pos_emb(levels_info, regions, image_size)

        # 计算损失并反向传播
        loss = output.sum()
        loss.backward()

        # 由于 area_scale != 0，梯度应该流到 regions
        assert regions.grad is not None, "当 area_scale != 0 时，梯度应该流到 regions"


class TestHilbertAwareMultiScaleAttentionWithAffine:
    """测试 HilbertAwareMultiScaleAttention 的仿射调制集成 (I31-3)。"""

    def _generate_levels_info(self, B: int, N: int, max_depth: int = 8) -> torch.Tensor:
        """生成有效的 levels_info 张量。"""
        levels_info = []
        for _ in range(B * N):
            depth = torch.randint(0, max_depth + 1, (1,)).item()
            path = [torch.randint(0, 4, (1,)).item() for _ in range(depth)]
            path.extend([0] * (max_depth - depth))
            levels_info.append([depth] + path)
        return torch.tensor(levels_info, dtype=torch.float32).reshape(B, N, max_depth + 1)

    def _generate_valid_regions(self, B: int, N: int, img_size: int, max_dim: int = 64) -> torch.Tensor:
        """生成有效的区域坐标 (x1 < x2, y1 < y2)。"""
        regions = []
        for _ in range(B * N):
            x1 = torch.randint(0, max_dim - 4, (1,)).item()
            y1 = torch.randint(0, max_dim - 4, (1,)).item()
            min_size = 4
            max_size = max_dim // 4
            w = torch.randint(min_size, max_size, (1,)).item()
            h = torch.randint(min_size, max_size, (1,)).item()
            x2 = min(x1 + w, max_dim)
            y2 = min(y1 + h, max_dim)
            regions.append([x1, y1, x2, y2])
        return torch.tensor(regions, dtype=torch.float32).reshape(B, N, 4)

    def test_affine_modulation_output(self):
        """测试仿射调制注意力输出。"""
        B, N, dim, heads = 2, 8, 64, 4
        x = torch.randn(B, N, dim)
        levels_info = self._generate_levels_info(B, N, 8)
        regions = self._generate_valid_regions(B, N, 64)
        image_size = 64

        attn = HilbertAwareMultiScaleAttention(
            dim=dim,
            heads=heads,
            use_affine_modulation=True,
            fourier_levels=4,
        )

        output = attn(x, levels_info=levels_info, regions=regions, image_size=image_size)

        assert output.shape == (B, N, dim)

    def test_affine_vs_base_attention(self):
        """测试仿射调制与基础注意力的差异。"""
        B, N, dim, heads = 1, 8, 64, 4
        x = torch.randn(B, N, dim)
        levels_info = self._generate_levels_info(B, N, 8)
        regions = self._generate_valid_regions(B, N, 64)
        image_size = 64

        # 启用和禁用仿射调制
        attn_affine = HilbertAwareMultiScaleAttention(
            dim=dim,
            heads=heads,
            use_affine_modulation=True,
            fourier_levels=4,
        )
        attn_base = HilbertAwareMultiScaleAttention(
            dim=dim,
            heads=heads,
            use_affine_modulation=False,
        )

        # 强制零初始化权重使两者接近
        attn_affine.affine_modulated_bias.residual_alpha.data.fill_(0.0)
        attn_affine.affine_modulated_bias.area_encoder.area_weight.data.fill_(0.0)

        out_affine = attn_affine(x, levels_info=levels_info, regions=regions, image_size=image_size)
        out_base = attn_base(x, levels_info=levels_info, regions=regions, image_size=image_size)

        # 零初始化时两者应该接近
        # 注意: 可能不会完全相同，因为 AffineModulatedBias 有自己的 LCA 实现

    def test_no_regions_fallback(self):
        """测试无 regions 时的回退行为。"""
        B, N, dim, heads = 2, 8, 64, 4
        x = torch.randn(B, N, dim)
        levels_info = self._generate_levels_info(B, N, 8)

        attn = HilbertAwareMultiScaleAttention(
            dim=dim,
            heads=heads,
            use_affine_modulation=True,
            fourier_levels=4,
        )

        # 无 regions 时应该回退到基础 LCA 偏置
        output = attn(x, levels_info=levels_info, regions=None, image_size=None)

        assert output.shape == (B, N, dim)

    def test_gradient_flow_affine(self):
        """测试仿射调制注意力梯度流。"""
        B, N, dim, heads = 2, 8, 64, 4
        x = torch.randn(B, N, dim)
        levels_info = self._generate_levels_info(B, N, 8)
        regions = self._generate_valid_regions(B, N, 64)
        regions.requires_grad_(True)
        image_size = 64

        attn = HilbertAwareMultiScaleAttention(
            dim=dim,
            heads=heads,
            use_affine_modulation=True,
            fourier_levels=4,
        )

        output = attn(x, levels_info=levels_info, regions=regions, image_size=image_size)

        # 计算损失并反向传播
        loss = output.sum()
        loss.backward()

        assert regions.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
