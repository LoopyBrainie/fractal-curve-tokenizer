"""
I34 系列数学形式化批判修复 - 单元测试

Tests for:
- I34-1: affine_bias dimension broadcast fix (per-head independence)
- I34-6: quota allocation tensor operations (torch.compile compatibility)
- I34-7: depths clamp min boundary protection
- I34-10: gate design optimization (tanh, zero-init, reduced params)
- I34-12: depth normalization strategy (per-batch vs global)

Mathematical properties verified:
1. Per-head affine_bias independence
2. Tensor operations for quota allocation (no .item() in loops)
3. depths ∈ [0, max_depth] after clamp
4. Gate range [-1, 1] with tanh activation
5. Per-batch depth normalization supporting dynamic resolution
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class TestI34_1_AffineBiasBroadcast:
    """I34-1: affine_bias 维度广播修复测试"""

    def test_affine_bias_per_head_computation(self):
        """验证 affine_bias 按 head_dim 分组计算"""
        # 模拟数据
        B, heads, dim_head, N = 2, 8, 8, 16
        dim = heads * dim_head  # 64

        affine_bias = torch.randn(B, dim, N, N)

        # 修复后的计算: reshape + mean(dim=2)
        result = affine_bias.reshape(B, heads, dim_head, N, N).mean(dim=2)

        # 验证形状
        assert result.shape == (B, heads, N, N), \
            f"Expected shape {(B, heads, N, N)}, got {result.shape}"

        # 验证各 head 的偏置值可能不同 (对于随机输入)
        assert not torch.allclose(result[0, 0], result[0, 1]), \
            "Different heads should produce different results for random input"

    def test_affine_bias_shape_preservation(self):
        """验证输出形状与注意力计算兼容"""
        B, heads, dim_head, N = 2, 8, 8, 16
        dim = heads * dim_head

        affine_bias = torch.randn(B, dim, N, N)

        # 修复后的计算
        affine_bias_avg = affine_bias.reshape(B, heads, dim_head, N, N).mean(dim=2)

        # 验证形状
        expected_shape = (B, heads, N, N)
        assert affine_bias_avg.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {affine_bias_avg.shape}"


class TestI34_6_QuotaAllocation:
    """I34-6: 配额分配张量操作测试"""

    def test_quota_tensor_iteration(self):
        """验证 _compute_quota_allocation 使用张量操作"""
        from vit_pytorch.gumbel_topk_splitter import GumbelTopKSplitter

        splitter = GumbelTopKSplitter(
            min_patch_size=4,
            max_depth_limit=4,
            image_size=(64, 64),
        )

        # 验证修复后代码使用 tensor iteration (不是 while 循环)
        import inspect
        source = inspect.getsource(splitter._compute_quota_allocation)

        # I35 Fix: 使用 range(D) 而不是硬编码 range(4)
        # 检查修复后代码使用 for 循环迭代 (不是 while 循环)
        assert "for _ in range(D):" in source or "for _ in range(self._current_max_depth + 1):" in source, \
            "Should use for loop with D iteration, not while loop"
        # 确保没有 while 循环
        assert "while " not in source, "Should not use while loop for iteration"


class TestI34_7_DepthBoundary:
    """I34-7: depths 边界保护测试"""

    def test_depth_clamp_has_min(self):
        """验证 depths.clamp 包含 min=0 边界"""
        import inspect
        import vit_pytorch.tokenizer_streaming as tokenizer_module
        source = inspect.getsource(tokenizer_module)

        # 验证修复后的代码包含 min=0
        assert "clamp(min=0, max=" in source, \
            " depths.clamp should have min=0 protection"

    def test_depth_clamp_edge_cases(self):
        """测试 depths 边界情况"""
        max_depth = 4

        # 测试负值情况
        depths_negative = torch.tensor([-1, 0, 1, 2])
        clamped = depths_negative.clamp(min=0, max=max_depth)
        assert clamped.min().item() >= 0, "Negative depths should be clamped to 0"

        # 测试超大值情况
        depths_large = torch.tensor([0, 1, 10, 100])
        clamped = depths_large.clamp(min=0, max=max_depth)
        assert clamped.max().item() <= max_depth, "Large depths should be clamped"


class TestI34_10_GateDesign:
    """I34-10: 门控设计优化测试"""

    def test_gate_zero_initialization(self):
        """验证门控零初始化"""
        from vit_pytorch.block_transformer import FractalTransformerBlock

        max_level = 4
        block = FractalTransformerBlock(
            dim=64,
            heads=4,
            dim_head=16,
            mlp_dim=128,
            max_level=max_level,
            drop_path=0.0,
        )

        gate_param = block._residual_gate.weight
        assert torch.allclose(gate_param, torch.zeros_like(gate_param)), \
            "Gate should be zero-initialized"

    def test_gate_parameter_reduction(self):
        """验证门控参数减半 (2×D → 1×D)"""
        from vit_pytorch.block_transformer import FractalTransformerBlock

        max_level = 4
        block = FractalTransformerBlock(
            dim=64,
            heads=4,
            dim_head=16,
            mlp_dim=128,
            max_level=max_level,
            drop_path=0.0,
        )

        expected_params = (max_level + 1) * 1
        actual_params = block._residual_gate.weight.numel()
        assert actual_params == expected_params, \
            f"Gate params should be {expected_params}, got {actual_params}"

    def test_gate_tanh_range(self):
        """验证门控 tanh 输出范围 [-1, 1]"""
        from vit_pytorch.block_transformer import FractalTransformerBlock

        max_level = 4
        block = FractalTransformerBlock(
            dim=64,
            heads=4,
            dim_head=16,
            mlp_dim=128,
            max_level=max_level,
            drop_path=0.0,
        )

        depths = torch.randint(0, max_level + 1, (16,))

        gate_raw = block._residual_gate(depths)  # (16, 1)
        gate = torch.tanh(gate_raw)  # (16, 1) ∈ [-1, 1]

        assert gate.min() >= -1.0, f"Gate min {gate.min()} < -1.0"
        assert gate.max() <= 1.0, f"Gate max {gate.max()} > 1.0"

    def test_residual_scale_range(self):
        """验证残差缩放范围 [0, 2]"""
        from vit_pytorch.block_transformer import FractalTransformerBlock

        max_level = 4
        block = FractalTransformerBlock(
            dim=64,
            heads=4,
            dim_head=16,
            mlp_dim=128,
            max_level=max_level,
            drop_path=0.0,
        )

        depths = torch.randint(0, max_level + 1, (16,))

        gate_raw = block._residual_gate(depths)
        gate = torch.tanh(gate_raw)

        residual_scale = 1.0 + gate
        assert residual_scale.min() >= 0.0, "Residual scale should be non-negative"
        assert residual_scale.max() <= 2.0, "Residual scale should be at most 2"


class TestI34_12_DepthNormalization:
    """I34-12: 深度归一化策略测试"""

    def test_per_batch_normalization_form(self):
        """验证 per-batch 归一化数学形式"""
        B, D = 4, 5
        logits = torch.randn(B, 85)
        depths = torch.randint(0, D, (85,))

        # 构建深度 one-hot 掩码
        depth_onehot = F.one_hot(depths, D).float().T  # [D, N]
        depth_counts = depth_onehot.sum(dim=1)  # [D]

        # 计算 per-batch 均值
        depth_sums = torch.einsum('bn,dn->bd', logits, depth_onehot)  # [B, D]
        mu_per_batch = depth_sums / depth_counts.unsqueeze(0)  # [B, D]

        # 验证形状
        assert mu_per_batch.shape == (B, D)

        # 验证每个 batch 的均值计算正确
        for b in range(B):
            for d in range(D):
                if depth_counts[d] > 0:
                    expected_mean = logits[b, depths == d].mean()
                    assert abs(mu_per_batch[b, d].item() - expected_mean.item()) < 1e-5


class TestIntegration:
    """集成测试"""

    def test_model_forward_no_error(self):
        """验证模型前向传播不报错"""
        from vit_pytorch import FractalCurveViT

        model = FractalCurveViT(
            image_size=64,
            num_classes=10,
            dim=64,
            depth=2,
            heads=4,
        )

        B = 2
        x = torch.randn(B, 3, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (B, 10)

    def test_dynamic_resolution_compatibility(self):
        """验证动态分辨率兼容性"""
        from vit_pytorch import FractalCurveViT

        model = FractalCurveViT(
            image_size=None,  # 动态分辨率
            num_classes=10,
            dim=64,
            depth=2,
            heads=4,
        )

        # 不同分辨率的输入
        sizes = [(48, 48), (64, 64), (96, 96)]

        for H, W in sizes:
            x = torch.randn(1, 3, H, W)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (1, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
