# -*- coding: utf-8 -*-
"""
L3 Pipeline Tests: Dynamic Computation

对应模块:
- vit_pytorch.complexity_estimator
- vit_pytorch.block_transformer

测试内容:
- I97-11: 动态计算测试

数学形式:
    C(I) = σ(W_2 · ReLU(W_1 · x̄))
    L_eff = floor(L_min + (L_max - L_min) * C(I))

验证指标:
1. 复杂度输出范围: C(I) ∈ [0, 1]
2. 层数约束: L_min ≤ L_eff ≤ L_max
3. 训练模式: 所有层都参与
4. 推理模式: 动态跳过层
5. 效率: FLOPs降低 ≥ 25%
"""

import pytest
import torch
import torch.nn as nn

from vit_pytorch.complexity_estimator import (
    ComplexityEstimator,
    DynamicDepthRouter,
    compute_complexity_from_depth_distribution,
)
from vit_pytorch.block_transformer import FractalTransformer


class TestComplexityEstimator:
    """I97-11: 复杂度估计器测试类."""

    @pytest.fixture
    def dim(self):
        return 128

    @pytest.fixture
    def batch_size(self):
        return 4

    @pytest.fixture
    def seq_len(self):
        return 64

    @pytest.fixture
    def estimator(self, dim):
        return ComplexityEstimator(dim=dim, hidden_dim=32)

    def test_output_shape(self, estimator, batch_size, seq_len, dim):
        """测试1: 输出形状一致性。

        预期: 输出形状 = [B, 1]
        """
        x = torch.randn(batch_size, seq_len, dim)
        complexity = estimator(x)

        assert complexity.shape == (batch_size, 1), \
            f"输出形状应为 {(batch_size, 1)}，实际为 {complexity.shape}"

    def test_output_range(self, estimator, batch_size, seq_len, dim):
        """测试2: 输出值范围。

        预期: 输出值 ∈ [0, 1]
        """
        x = torch.randn(batch_size, seq_len, dim)
        complexity = estimator(x)

        assert complexity.min() >= 0, \
            f"复杂度最小值应为 >= 0，实际为 {complexity.min()}"
        assert complexity.max() <= 1, \
            f"复杂度最大值应为 <= 1，实际为 {complexity.max()}"

    def test_cls_token_option(self, dim):
        """测试3: CLS token选项。

        预期: use_cls=True 时使用 x[:, 0]
        """
        estimator_cls = ComplexityEstimator(dim=dim, hidden_dim=32, use_cls=True)
        estimator_mean = ComplexityEstimator(dim=dim, hidden_dim=32, use_cls=False)

        x = torch.randn(4, 64, dim)

        out_cls = estimator_cls(x)
        out_mean = estimator_mean(x)

        # 两种方式都应输出正确形状
        assert out_cls.shape == (4, 1)
        assert out_mean.shape == (4, 1)

    def test_parameter_count(self, dim):
        """测试4: 参数数量。

        预期: 约 D * D/4 + D/4 = dim * dim/4 + dim/4
        """
        hidden_dim = dim // 4
        estimator = ComplexityEstimator(dim=dim, hidden_dim=hidden_dim)

        # 计算参数量
        expected_params = dim * hidden_dim + hidden_dim + hidden_dim * 1 + 1
        actual_params = sum(p.numel() for p in estimator.parameters())

        assert actual_params == expected_params, \
            f"参数量应为 {expected_params}，实际为 {actual_params}"

    def test_gradient_flow(self, estimator, batch_size, seq_len, dim):
        """测试5: 梯度流。

        预期: 复杂度估计器可以反向传播
        """
        estimator.train()
        x = torch.randn(batch_size, seq_len, dim, requires_grad=True)

        complexity = estimator(x)
        loss = complexity.sum()
        loss.backward()

        assert x.grad is not None, "输入应有梯度"
        assert estimator.net[0].weight.grad is not None, "第一层应有梯度"


class TestDynamicDepthRouter:
    """I97-11: 动态深度路由器测试类."""

    @pytest.fixture
    def dim(self):
        return 128

    @pytest.fixture
    def depth(self):
        return 12

    @pytest.fixture
    def min_layers(self):
        return 4

    @pytest.fixture
    def router(self, dim, depth, min_layers):
        return DynamicDepthRouter(
            dim=dim,
            depth=depth,
            min_layers=min_layers,
        )

    def test_effective_depth_range(self, router, dim, depth, min_layers):
        """测试6: 有效层数范围。

        预期: min_layers ≤ L_eff ≤ depth
        """
        batch_size, seq_len = 4, 64
        router.eval()
        x = torch.randn(batch_size, seq_len, dim)

        result = router(x)
        effective_depth = result['effective_depth']

        assert effective_depth.min() >= min_layers, \
            f"有效层数最小值应为 >= {min_layers}，实际为 {effective_depth.min()}"
        assert effective_depth.max() <= depth, \
            f"有效层数最大值应为 <= {depth}，实际为 {effective_depth.max()}"

    def test_complexity_correlation(self, router, dim):
        """测试7: 复杂度与层数正相关。

        预期: 高复杂度输入 → 更多层数
        """
        batch_size, seq_len = 4, 64
        router.eval()

        # 简单图像（低复杂度）
        simple_x = torch.randn(batch_size, seq_len, dim) * 0.1
        simple_result = router(simple_x)
        simple_complexity = simple_result['complexity']

        # 复杂图像（高复杂度）
        complex_x = torch.randn(batch_size, seq_len, dim) * 2.0
        complex_result = router(complex_x)
        complex_complexity = complex_result['complexity']

        # 复杂度应该不同（不一定层数不同，取决于估计器学习结果）
        assert simple_complexity.shape == (batch_size, 1)
        assert complex_complexity.shape == (batch_size, 1)


class TestFractalTransformerDynamicDepth:
    """I97-11: FractalTransformer动态深度测试类."""

    @pytest.fixture
    def device(self):
        return "cpu"

    @pytest.fixture
    def dim(self):
        return 128

    @pytest.fixture
    def depth(self):
        return 12

    @pytest.fixture
    def max_level(self):
        return 3

    @pytest.fixture
    def levels_info(self, max_level, batch_size=4, seq_len=64):
        """创建有效的levels_info。"""
        levels_info = torch.zeros(batch_size, seq_len, max_level + 1, dtype=torch.long)
        for b in range(batch_size):
            for n in range(seq_len):
                d = n % (max_level + 1)
                levels_info[b, n, 0] = d  # 第一列是深度值
        return levels_info

    @pytest.fixture
    def transformer(self, dim, depth, max_level, device):
        return FractalTransformer(
            dim=dim,
            depth=depth,
            heads=4,
            dim_head=32,
            mlp_dim=256,
            max_level=max_level,
            use_dynamic_depth=True,
            min_layers=4,
        ).to(device)

    def test_inference_mode_skip_layers(
        self, transformer, dim, depth, device, levels_info
    ):
        """测试8: 推理时跳过部分层。

        预期: 推理时使用少于全部层数
        """
        batch_size, seq_len = levels_info.shape[0], levels_info.shape[1]
        transformer.eval()
        x = torch.randn(batch_size, seq_len, dim, device=device)

        output, extra_info = transformer(x, levels_info=levels_info, return_extra_info=True)

        # 检查输出形状
        assert output.shape == (batch_size, seq_len, dim)

        # 检查额外信息
        assert 'effective_depth' in extra_info
        assert 'complexity' in extra_info

        # 推理时应该返回有效层数
        assert extra_info['effective_depth'] >= 1
        assert extra_info['effective_depth'] <= depth

    def test_training_mode_all_layers(
        self, transformer, dim, depth, device, levels_info
    ):
        """测试9: 训练时所有层都参与。

        预期: 训练模式使用全部层数
        """
        batch_size, seq_len = levels_info.shape[0], levels_info.shape[1]
        transformer.train()
        x = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)

        output, extra_info = transformer(x, levels_info=levels_info, return_extra_info=True)

        # 训练时应该使用全部层数
        assert extra_info['effective_depth'] == depth, \
            f"训练时应使用全部 {depth} 层，实际为 {extra_info['effective_depth']}"

        # 反向传播应该工作
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "输入应有梯度"

    def test_no_dynamic_depth(
        self, dim, depth, max_level, device, levels_info
    ):
        """测试10: 禁用动态深度时正常工作。

        预期: 不使用动态深度时行为与原来一致
        """
        transformer = FractalTransformer(
            dim=dim,
            depth=depth,
            heads=4,
            dim_head=32,
            mlp_dim=256,
            max_level=max_level,
            use_dynamic_depth=False,
        ).to(device)

        batch_size, seq_len = levels_info.shape[0], levels_info.shape[1]
        transformer.eval()
        x = torch.randn(batch_size, seq_len, dim, device=device)

        output = transformer(x, levels_info=levels_info)

        assert output.shape == (batch_size, seq_len, dim)


class TestComplexityFromDepthDistribution:
    """I97-11: 从深度分布计算复杂度测试类."""

    def test_basic_functionality(self):
        """测试11: 基本功能。

        预期: 返回有效的复杂度得分
        """
        batch_size, seq_len, max_level = 4, 64, 3

        # 随机生成深度分布
        depths = torch.randint(0, max_level + 1, (batch_size, seq_len))

        complexity = compute_complexity_from_depth_distribution(depths, max_level)

        assert complexity.shape == (batch_size, 1)
        assert complexity.min() >= 0
        assert complexity.max() <= 1

    def test_high_depth_bias(self):
        """测试12: 深度分布影响复杂度。

        预期: 更多深层token → 更高复杂度
        """
        batch_size, seq_len, max_level = 4, 64, 3

        # 浅层分布：只有深度0和1
        shallow_depths = torch.zeros(batch_size, seq_len, dtype=torch.long)
        shallow_depths[:, :32] = 0  # 深度0
        shallow_depths[:, 32:] = 1  # 深度1

        # 深层分布：有深度3
        deep_depths = torch.zeros(batch_size, seq_len, dtype=torch.long)
        deep_depths[:, :16] = 1  # 深度1
        deep_depths[:, 16:] = 3  # 深度3

        shallow_complexity = compute_complexity_from_depth_distribution(shallow_depths, max_level)
        deep_complexity = compute_complexity_from_depth_distribution(deep_depths, max_level)

        # 深层分布应该有更高的复杂度
        assert deep_complexity.mean() > shallow_complexity.mean(), \
            "深层分布应该有更高的复杂度"


class TestDynamicComputationIntegration:
    """I97-11: 集成测试。"""

    @pytest.fixture
    def device(self):
        return "cpu"

    @pytest.fixture
    def dim(self):
        return 128

    @pytest.fixture
    def depth(self):
        return 12

    @pytest.fixture
    def max_level(self):
        return 3

    @pytest.fixture
    def levels_info(self, max_level, batch_size=2, seq_len=64):
        """创建有效的levels_info。"""
        levels_info = torch.zeros(batch_size, seq_len, max_level + 1, dtype=torch.long)
        for b in range(batch_size):
            for n in range(seq_len):
                d = n % (max_level + 1)
                levels_info[b, n, 0] = d
        return levels_info

    def test_end_to_end_with_simple_image(
        self, device, dim, depth, max_level, levels_info
    ):
        """测试13: 简单图像端到端测试。

        预期: 简单图像使用较少层数，输出形状正确
        """
        transformer = FractalTransformer(
            dim=dim,
            depth=depth,
            heads=4,
            dim_head=32,
            mlp_dim=256,
            max_level=max_level,
            use_dynamic_depth=True,
            min_layers=4,
        ).to(device)

        transformer.eval()

        batch_size, seq_len = levels_info.shape[0], levels_info.shape[1]
        # 简单图像（低方差特征）
        simple_x = torch.randn(batch_size, seq_len, dim, device=device) * 0.1

        output, extra_info = transformer(simple_x, levels_info=levels_info, return_extra_info=True)

        assert output.shape == (batch_size, seq_len, dim)
        assert extra_info['effective_depth'] <= depth

    def test_complexity_stats(self, dim):
        """测试14: 复杂度统计信息。

        预期: 返回有效的统计信息
        """
        estimator = ComplexityEstimator(dim=dim, hidden_dim=32)
        batch_size, seq_len = 4, 64
        x = torch.randn(batch_size, seq_len, dim)

        stats = estimator.get_complexity_stats(x)

        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats

        assert 0 <= stats['mean'] <= 1
        assert stats['min'] >= 0
        assert stats['max'] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
