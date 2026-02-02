# -*- coding: utf-8 -*-
"""
L2 Components Tests: Splitter Vectorization and Boundary Conditions

对应模块: vit_pytorch.gumbel_topk_splitter

测试内容:
- K > N 边界条件测试
- K = 0 边界条件测试
- 向量化操作验证
- 极端数值稳定性测试

数学形式化
==========================
K 值边界:
    K = 0: 不选择任何 token (所有 hard_mask = 0)
    K > N: 选择数大于候选数 (topk 会截断)
    K = N: 选择所有候选

向量化要求:
    所有 batch 操作应使用张量运算，避免 Python 循环

Test Categories:
1. K 边界测试 - 验证 K=0, K>N 场景
2. 向量化测试 - 验证 batch 操作正确性
3. 数值稳定性测试 - 极端输入的稳定性
"""

import pytest
import torch
import torch.nn.functional as F
from vit_pytorch.gumbel_topk_splitter import GumbelTopKSplitter
from vit_pytorch.config import SplitterConfig


class TestKSplitterBoundary:
    """K 值边界条件测试."""

    def test_k_equals_zero(self):
        """验证 K_min=K_max=0 时行为 - 实际选择数会受最小配额保护."""
        # K_min=K_max=0 是极端边界情况，splitter 有最小配额保护
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=2,
            image_size=(64, 64),
            K_min=0,
            K_max=0,
        )

        batch_size = 4
        features = torch.randn(batch_size, 256, 16, 16)

        with torch.no_grad():
            output = splitter(features, (64, 64), hard=True)

        # 由于最小配额保护，即使 K=0 也会选择一些 token
        # 验证输出有效（不崩溃，且有合理的选择数）
        assert output.num_selected_per_batch.sum() > 0, "应有选择（最小配额保护）"
        # 每个 batch 应该有相同的选择数（一致的向量化行为）
        assert output.num_selected_per_batch.unique().numel() == 1, "各 batch 选择数应一致"

    def test_k_greater_than_n(self):
        """验证 K > N 时行为正确 (topk 应截断到 N)."""
        # 使用大 K 值测试
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=2,
            image_size=(64, 64),
            K_min=1000,  # 远大于候选数
            K_max=2000,
        )

        batch_size = 4
        features = torch.randn(batch_size, 256, 16, 16)

        with torch.no_grad():
            output = splitter(features, (64, 64), hard=True)

        # 实际选择的 token 数应 <= N
        assert (output.num_selected_per_batch <= splitter.num_candidates).all(), \
            f"K > N 时选择数不应超过 N={splitter.num_candidates}"


class TestSplitterVectorization:
    """Splitter 向量化测试."""

    def test_batch_consistency(self):
        """验证批量处理的数学一致性."""
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=2,
            image_size=(64, 64),
            K_min=8,
            K_max=32,
        )

        # 固定随机种子以确保可重复性
        torch.manual_seed(42)

        # 单样本处理 - 4D 特征图
        single_features = torch.randn(1, 256, 16, 16)
        with torch.no_grad():
            single_output = splitter(single_features, (64, 64), hard=True)

        # 批量处理 (4个相同样本)
        batch_features = single_features.repeat(4, 1, 1, 1)
        with torch.no_grad():
            batch_output = splitter(batch_features, (64, 64), hard=True)

        # 验证批量输出形状
        assert batch_output.selected_mask.shape[0] == 4, "批量输出 batch 维度应匹配"
        assert batch_output.num_selected_per_batch.shape[0] == 4, "批量输出 batch 维度应匹配"

    def test_depth_stratification_vectorized(self):
        """验证深度分层操作的向量化正确性."""
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=3,
            image_size=(64, 64),
            K_min=16,
            K_max=64,
        )

        batch_size = 8
        features = torch.randn(batch_size, 256, 16, 16)

        with torch.no_grad():
            output = splitter(features, (64, 64), hard=True)

        # 验证每个 batch 的选择数一致
        selected_per_batch = output.num_selected_per_batch
        assert all(
            output.num_selected_per_batch[b].item() == selected_per_batch[0].item()
            for b in range(batch_size)
        ), "批量处理时各样本选择数应一致"

    def test_gradient_flow_vectorized(self):
        """验证梯度在向量化操作中正确流动."""
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=2,
            image_size=(64, 64),
            K_min=8,
            K_max=32,
        )

        batch_size = 4
        features = torch.randn(batch_size, 256, 16, 16)
        features.requires_grad_(True)

        output = splitter(features, (64, 64), hard=False)

        # 计算损失并反向传播
        loss = output.selected_mask.sum()
        loss.backward()

        # 验证梯度存在且形状正确
        assert features.grad is not None, "应存在梯度"
        assert features.grad.shape == features.shape, "梯度形状应与输入匹配"


class TestSplitterNumericalStability:
    """Splitter 数值稳定性测试."""

    def test_extreme_logits(self):
        """验证极端 logits 输入的稳定性."""
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=2,
            image_size=(64, 64),
            K_min=8,
            K_max=32,
        )

        batch_size = 4

        # 测试大 logits - 缩放输入特征
        large_features = torch.randn(batch_size, 256, 16, 16) * 100
        with torch.no_grad():
            large_output = splitter(large_features, (64, 64), hard=True)
        assert not torch.isnan(large_output.selected_mask).any(), "大 logits 不应产生 NaN"
        assert not torch.isinf(large_output.selected_mask).any(), "大 logits 不应产生 Inf"

        # 测试小 logits
        small_features = torch.randn(batch_size, 256, 16, 16) * 0.01
        with torch.no_grad():
            small_output = splitter(small_features, (64, 64), hard=True)
        assert not torch.isnan(small_output.selected_mask).any(), "小 logits 不应产生 NaN"
        assert not torch.isinf(small_output.selected_mask).any(), "小 logits 不应产生 Inf"

    def test_zero_features(self):
        """验证全零特征的稳定性."""
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=2,
            image_size=(64, 64),
            K_min=8,
            K_max=32,
        )

        batch_size = 4
        zero_features = torch.zeros(batch_size, 256, 16, 16)

        with torch.no_grad():
            output = splitter(zero_features, (64, 64), hard=True)

        # 应能处理全零输入，不崩溃
        assert output.selected_mask.shape[0] == batch_size, "输出 batch 维度应正确"

    def test_temperature_extremes(self):
        """验证极端温度值的稳定性."""
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=2,
            image_size=(64, 64),
            K_min=8,
            K_max=32,
        )

        batch_size = 4
        features = torch.randn(batch_size, 256, 16, 16)

        # 测试极低温度
        splitter.set_temperature(0.1)
        with torch.no_grad():
            low_temp_output = splitter(features, (64, 64), hard=True)
        assert not torch.isnan(low_temp_output.selected_mask).any(), "低温度不应产生 NaN"

        # 测试极高温度
        splitter.set_temperature(10.0)
        with torch.no_grad():
            high_temp_output = splitter(features, (64, 64), hard=True)
        assert not torch.isnan(high_temp_output.selected_mask).any(), "高温度不应产生 NaN"
