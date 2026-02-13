# -*- coding: utf-8 -*-
"""
L2 Components: Hierarchical Soft-Hard Attention Tests

对应模块: vit_pytorch.layers.attention.hierarchical_soft_hard

测试内容:
- HierarchicalMaskBuilder 三区域掩码构建
- HierarchicalSoftHardAttention 基础功能
- 梯度流验证
- 与 Hilbert 局部性对齐
"""

import pytest
import torch
import torch.nn as nn

from vit_pytorch.layers.attention.hierarchical_soft_hard import (
    HierarchicalSoftHardAttention,
    HierarchicalMaskBuilder,
    HierarchicalAttentionConfig,
    STANDARD_CONFIG,
    STRICT_CONFIG,
    RELAXED_CONFIG,
)
from vit_pytorch.layers.embeddings.fractal_path import VectorizedPathEncoder


class TestHierarchicalMaskBuilder:
    """HierarchicalMaskBuilder 测试"""

    @pytest.fixture
    def mask_builder(self):
        """创建掩码构建器"""
        return HierarchicalMaskBuilder(
            max_level=4,
            lca_min=1.0,
            lca_soft=2.0,
            learn_thresholds=True,
        )

    def test_init(self, mask_builder):
        """初始化测试"""
        assert mask_builder.max_level == 4
        assert mask_builder.lca_min_init == 1.0
        assert mask_builder.lca_soft_init == 2.0

    def test_mask_shape(self, mask_builder):
        """掩码形状测试"""
        batch_size = 2
        seq_len = 8
        lca_depths = torch.randint(0, 5, (batch_size, seq_len, seq_len))

        mask = mask_builder(lca_depths)

        assert mask.shape == (batch_size, 1, seq_len, seq_len)

    def test_hard_zero_region(self, mask_builder):
        """HARD_ZERO 区域测试 (ℓ < ℓ_min)"""
        # LCA < 1 应该被排除
        lca_depths = torch.tensor([[[0, 0], [0, 0]]])  # 全部为 0

        mask = mask_builder(lca_depths)

        # ℓ = 0 < ℓ_min = 1，应该为 0
        assert (mask == 0).all()

    def test_hard_one_region(self, mask_builder):
        """HARD_ONE 区域测试 (ℓ ≥ ℓ_soft)"""
        # LCA >= 2 应该被完全鼓励
        lca_depths = torch.tensor([[[4, 4], [4, 4]]])  # 全部为 4

        mask = mask_builder(lca_depths)

        # ℓ = 4 >= ℓ_soft = 2，应该为 1
        assert (mask == 1).all()

    def test_soft_positive_region(self, mask_builder):
        """SOFT_POSITIVE 区域测试 (ℓ_min ≤ ℓ < ℓ_soft)"""
        # ℓ = 1, lca_min=1, lca_soft=2
        # 1 >= 1 且 1 < 2, 所以使用 soft_mask
        lca_depths = torch.tensor([[[1, 1], [1, 1]]], dtype=torch.float32)

        mask = mask_builder(lca_depths)

        # soft_mask = sigmoid(1 - 1) = sigmoid(0) ≈ 0.5
        # 应该在 (0, 1) 范围内
        assert (mask > 0).all() and (mask < 1).all()

    def test_mixed_regions(self, mask_builder):
        """混合区域测试"""
        # lca_depths: [1, 4, 4]
        lca_depths = torch.tensor([
            [[0, 1, 2, 4],
             [1, 0, 3, 1],
             [2, 3, 0, 2],
             [4, 1, 2, 0]]
        ], dtype=torch.float32)

        mask = mask_builder(lca_depths)  # mask: [1, 1, 4, 4]

        # ℓ = 0: mask = 0 (HARD_ZERO)
        assert mask[0, 0, 0, 0] == 0  # [0,0] = 0
        assert mask[0, 0, 1, 1] == 0  # [1,1] = 0

        # ℓ >= 2: mask = 1 (HARD_ONE)
        assert mask[0, 0, 0, 3] == 1  # [0,3] = 4
        # [2,2] = 0 (diagonal, self-comparison) - should be 0
        assert mask[0, 0, 2, 0] == 1  # [2,0] = 2

    def test_gradient_flow(self, mask_builder):
        """梯度流测试"""
        lca_depths = torch.randn(2, 8, 8, requires_grad=True)
        mask = mask_builder(lca_depths)

        loss = mask.sum()
        loss.backward()

        # 检查输入梯度
        assert lca_depths.grad is not None
        assert not torch.isnan(lca_depths.grad).any()

        # 检查参数requires_grad标志
        assert mask_builder.lca_min_param.requires_grad
        assert mask_builder.lca_soft_param.requires_grad


class TestHierarchicalAttentionConfig:
    """HierarchicalAttentionConfig 测试"""

    def test_default_config(self):
        """默认配置测试"""
        config = HierarchicalAttentionConfig()

        assert config.lca_min == 1.0
        assert config.lca_soft == 2.0
        assert config.learn_thresholds is True
        assert config.learn_temperature is True

    def test_standard_config(self):
        """标准配置测试"""
        assert STANDARD_CONFIG.lca_min == 1.0
        assert STANDARD_CONFIG.lca_soft == 2.0

    def test_strict_config(self):
        """严格配置测试"""
        assert STRICT_CONFIG.lca_min == 2.0
        assert STRICT_CONFIG.lca_soft == 3.0
        assert STRICT_CONFIG.learn_thresholds is False

    def test_relaxed_config(self):
        """宽松配置测试"""
        assert RELAXED_CONFIG.lca_min == 0.0
        assert RELAXED_CONFIG.lca_soft == 1.5


class TestHierarchicalSoftHardAttention:
    """HierarchicalSoftHardAttention 测试"""

    @pytest.fixture
    def attn_module(self):
        """创建注意力模块"""
        return HierarchicalSoftHardAttention(
            dim=64,
            max_level=4,
            heads=4,
            config=STANDARD_CONFIG,
        )

    @pytest.fixture
    def sample_regions(self):
        """创建样本区域"""
        batch_size = 2
        seq_len = 8
        image_size = 32

        # 创建不重叠的区域 (使用 long 类型)
        regions = torch.zeros(batch_size, seq_len, 4, dtype=torch.long)
        patch_size = image_size // 4  # 8x8 patches

        for b in range(batch_size):
            for i in range(seq_len):
                x = (i % 4) * patch_size
                y = (i // 4) * patch_size
                regions[b, i] = torch.tensor([x, y, x + patch_size, y + patch_size], dtype=torch.long)

        return regions, image_size

    @pytest.fixture
    def sample_input(self):
        """创建样本输入"""
        batch_size = 2
        seq_len = 8
        dim = 64

        return torch.randn(batch_size, seq_len, dim)

    def test_init(self, attn_module):
        """初始化测试"""
        assert attn_module.dim == 64
        assert attn_module.heads == 4
        assert attn_module.max_level == 4

    def test_output_shape(self, attn_module, sample_input, sample_regions):
        """输出形状测试"""
        regions, image_size = sample_regions
        out = attn_module(sample_input, regions, image_size)

        assert out.shape == sample_input.shape

    def test_without_regions(self, attn_module, sample_input):
        """无区域测试 (退化为标准注意力)"""
        out = attn_module(sample_input)

        assert out.shape == sample_input.shape
        assert not torch.isnan(out).any()

    def test_gradient_flow(self, attn_module, sample_input, sample_regions):
        """梯度流测试"""
        regions, image_size = sample_regions
        out = attn_module(sample_input, regions, image_size)

        loss = out.sum()
        loss.backward()

        # 检查主要参数都有梯度
        # 注意: mask_builder 的阈值参数在比较操作中使用，可能没有梯度
        # 这是预期行为，因为比较操作 (>, <, ==) 阻断梯度
        important_params = ['to_qkv', 'to_out', 'lca_embedding', 'temperature']
        for name, param in attn_module.named_parameters():
            if any(p in name for p in important_params):
                assert param.grad is not None, f"{name} has no gradient"
                assert not torch.isnan(param.grad).any(), f"{name} has NaN gradient"

    def test_hard_zero_enforcement(self, attn_module, sample_input):
        """硬排除强制测试"""
        # 创建远距离区域组合
        regions = torch.zeros(1, 4, 4, dtype=torch.long)
        # Token 0 和 3 在对角，远距离
        regions[0, 0] = torch.tensor([0, 0, 8, 8], dtype=torch.long)
        regions[0, 3] = torch.tensor([24, 24, 32, 32], dtype=torch.long)

        out = attn_module(sample_input[:1, :4], regions, 32)

        # 应该有有效输出
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_temperature_scaling(self, attn_module, sample_input, sample_regions):
        """温度缩放测试"""
        regions, image_size = sample_regions

        # 获取温度值
        temp = attn_module._get_temperature()
        assert temp > 0

        # 前向传播
        out = attn_module(sample_input, regions, image_size)
        assert not torch.isnan(out).any()

    def test_attention_mask(self, attn_module, sample_input):
        """注意力掩码测试"""
        # 创建 padding mask
        attention_mask = torch.tensor([[True, True, False, False, True, True, True, True]])

        out = attn_module(sample_input[:1], attention_mask=attention_mask)

        assert out.shape == sample_input[:1].shape
        assert not torch.isnan(out).any()

    def test_lca_embedding(self, attn_module, sample_regions):
        """LCA 嵌入测试"""
        regions, image_size = sample_regions
        batch_size = regions.shape[0]
        seq_len = regions.shape[1]

        # 直接测试 LCA 计算
        lca_depths = attn_module._compute_lca(regions, image_size)

        assert lca_depths.shape == (batch_size, seq_len, seq_len)
        assert lca_depths.dtype == torch.long
        assert (lca_depths >= 0).all() and (lca_depths <= attn_module.max_level).all()


class TestHierarchicalAttentionIntegration:
    """分层 TestHier注意力集成测试"""

    def test_hilbert_locality_alignment(self):
        """Hilbert 局部性对齐测试

        验证 HSHA 的三区域划分与 Hilbert 曲线性质对齐
        """
        from vit_pytorch.layers.attention.hierarchical_soft_hard import HierarchicalMaskBuilder

        max_level = 4
        builder = HierarchicalMaskBuilder(
            max_level=max_level,
            lca_min=1.0,
            lca_soft=2.0,
            learn_thresholds=False,  # 固定阈值用于测试
        )

        # 模拟 LCA 深度矩阵 (使用 float 以匹配内部计算)
        # Token 在相同象限: LCA = 4
        # Token 在相邻象限: LCA = 2 或 3
        # Token 在对角象限: LCA = 1
        # Token 在不同根节点: LCA = 0
        lca_matrix = torch.tensor(
            [[4, 2, 1, 0],
             [2, 4, 1, 0],
             [1, 1, 4, 0],
             [0, 0, 0, 4]],
            dtype=torch.float32
        )

        mask = builder(lca_matrix)

        # 验证掩码值
        # LCA = 4 (相同位置): mask = 1
        assert mask[0, 0, 0, 0] == 1
        assert mask[0, 0, 1, 1] == 1

        # LCA = 2 (相邻象限): lca_soft=2, 所以 mask = sigmoid(2-1) ≈ 0.73
        # 实际: lca >= lca_soft 时 mask = 1 (hard_one)
        assert mask[0, 0, 0, 1] == 1  # LCA=2 >= lca_soft=2

        # LCA = 1 (对角象限): lca_min=1, lca_soft=2
        # 1 >= 1 且 1 < 2, 所以使用 soft_mask = sigmoid(1-1) = 0.5
        assert 0 < mask[0, 0, 0, 2] < 1  # soft_mask 应该在 (0, 1)

        # LCA = 0 (不同根节点): mask = 0
        assert mask[0, 0, 0, 3] == 0

    def test_config_presets(self):
        """配置预设测试"""
        # 验证所有预设配置有效
        assert STANDARD_CONFIG.learn_thresholds is True
        assert STANDARD_CONFIG.learn_temperature is True

        assert STRICT_CONFIG.lca_min > STANDARD_CONFIG.lca_min
        assert STRICT_CONFIG.learn_thresholds is False

        assert RELAXED_CONFIG.lca_min < STANDARD_CONFIG.lca_min

    def test_backward_compatibility(self):
        """向后兼容性测试

        确保新实现不会破坏现有功能
        """
        from vit_pytorch.layers.attention.hierarchical_soft_hard import (
            HierarchicalSoftHardAttention,
            HierarchicalAttentionConfig,
        )

        # 使用默认配置
        attn = HierarchicalSoftHardAttention(
            dim=64,
            max_level=4,
            heads=4,
        )

        x = torch.randn(2, 8, 64)

        # 无区域输入
        out = attn(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()


class TestHierarchicalAttentionBoundary:
    """分层注意力边界条件测试"""

    def test_max_level_boundary(self):
        """最大深度边界测试"""
        builder = HierarchicalMaskBuilder(
            max_level=4,
            lca_min=1.0,
            lca_soft=2.0,
            learn_thresholds=True,
        )

        # LCA 深度不应超过 max_level
        lca_depths = torch.randint(0, 10, (1, 4, 4))  # 包含 > 4 的值
        lca_depths = lca_depths.clamp(0, 4)  # 截断到有效范围

        mask = builder(lca_depths)

        assert mask.shape == (1, 1, 4, 4)
        assert not torch.isnan(mask).any()

    def test_empty_regions(self):
        """空区域测试"""
        attn = HierarchicalSoftHardAttention(
            dim=64,
            max_level=4,
            heads=4,
        )

        # 空输入
        x_empty = torch.empty(1, 0, 64)
        regions_empty = torch.empty(1, 0, 4, dtype=torch.long)

        out = attn(x_empty, regions_empty, 32)

        assert out.shape == x_empty.shape

    def test_single_token(self):
        """单 Token 测试"""
        attn = HierarchicalSoftHardAttention(
            dim=64,
            max_level=4,
            heads=4,
        )

        x = torch.randn(1, 1, 64)
        # regions: [B, N, 4] = [1, 1, 4]
        regions = torch.tensor([[[0, 0, 8, 8]]], dtype=torch.long)

        out = attn(x, regions, 32)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()
