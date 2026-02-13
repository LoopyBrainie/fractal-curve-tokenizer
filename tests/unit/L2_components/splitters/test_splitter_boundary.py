# -*- coding: utf-8 -*-
"""
L2 Components Tests: Splitter Boundary Conditions

对应模块: vit_pytorch.gumbel_topk_splitter

测试内容:
- I24-4: max_depth=1 边界条件测试

数学形式化
==========================
单层四叉树:
    N_{depth=1} = 4^1 = 5 个候选区域
    (1 depth-0 根区域 + 4 depth-1 子区域)

区域分布 (64x64 图像):
    - depth=0: 1 个候选 (整个图像 64x64)
    - depth=1: 4 个候选 (2x2 grid, 每个 32x32)

配额分配问题:
    - depth=0 有 1 个候选
    - depth=1 有 4 个候选
    - 比例 1:4 可能导致配额不平衡

Test Categories:
1. 警告测试 - 验证边界警告产生
2. 功能测试 - 验证输出形状正确
3. 梯度测试 - 验证梯度流正常
4. 配置测试 - 验证 SplitterConfig.validate()
"""

import pytest
import torch
from vit_pytorch.layers.splitters.gumbel_topk import GumbelTopKSplitter
from vit_pytorch.core.config import SplitterConfig


class TestMaxDepthBoundary:
    """I24-4: max_depth=1 边界条件测试."""

    def test_max_depth_1_warning(self):
        """验证 max_depth=1 产生警告但正常工作."""
        with pytest.warns(UserWarning, match="max_depth.*2"):
            splitter = GumbelTopKSplitter(
                feature_dim=256,
                min_patch_size=4,
                max_level_limit=1,
                image_size=(64, 64),
            )

    def test_max_depth_0_warning(self):
        """验证 max_depth=0 产生警告但边界情况."""
        with pytest.warns(UserWarning, match="max_depth.*2"):
            splitter = GumbelTopKSplitter(
                feature_dim=256,
                min_patch_size=4,
                max_level_limit=0,
                image_size=(64, 64),
            )

    def test_max_depth_1_output_shape(self):
        """验证 max_depth=1 输出形状正确 (5 个候选)."""
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=1,
            image_size=(64, 64),
        )
        features = torch.randn(1, 256, 16, 16)
        result = splitter(features, (64, 64), hard=True)

        # max_depth=1: 1 (depth-0) + 4 (depth-1) = 5 个候选
        assert result.selected_mask.shape == (1, 5), \
            f"Expected (1, 5), got {result.selected_mask.shape}"

    def test_max_depth_1_output_batch(self):
        """验证 max_depth=1 支持批量处理."""
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=1,
            image_size=(64, 64),
        )
        features = torch.randn(4, 256, 16, 16)
        result = splitter(features, (64, 64), hard=True)

        # Batch size = 4
        assert result.selected_mask.shape == (4, 5), \
            f"Expected (4, 5), got {result.selected_mask.shape}"
        assert result.num_selected_per_batch.shape == (4,)

    def test_max_depth_1_gradient_flow(self):
        """验证 max_depth=1 梯度流正常."""
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=1,
            image_size=(64, 64),
        )
        features = torch.randn(1, 256, 16, 16, requires_grad=True)

        result = splitter(features, (64, 64), hard=False)
        loss = result.selected_mask.sum()
        loss.backward()

        assert features.grad is not None
        assert not torch.isnan(features.grad).any()
        assert not torch.isinf(features.grad).any()

    def test_max_depth_1_soft_mode(self):
        """验证 max_depth=1 软模式输出有效概率分布."""
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=1,
            image_size=(64, 64),
        )
        features = torch.randn(1, 256, 16, 16)
        result = splitter(features, (64, 64), hard=False)

        # probs 是 sigmoid 输出 (不是 softmax)
        assert result.probs.shape == (1, 5)
        assert torch.all(result.probs >= 0)
        assert torch.all(result.probs <= 1)
        # 验证概率在有效范围内
        assert result.probs.min().item() >= 0
        assert result.probs.max().item() <= 1

    def test_max_depth_1_regions_count(self):
        """验证 max_depth=1 区域计数正确."""
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=1,
            image_size=(64, 64),
        )
        features = torch.randn(1, 256, 16, 16)
        result = splitter(features, (64, 64), hard=True)

        # 5 个候选区域
        assert splitter.num_candidates == 5, \
            f"Expected 5 candidates, got {splitter.num_candidates}"

        # depth 分布: 1 个 depth-0, 4 个 depth-1
        assert (splitter.candidate_depths == 0).sum() == 1
        assert (splitter.candidate_depths == 1).sum() == 4

    def test_max_depth_1_num_selected_range(self):
        """验证 max_depth=1 选中数量在合理范围内."""
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=1,
            image_size=(64, 64),
        )

        for _ in range(10):
            features = torch.randn(1, 256, 16, 16)
            result = splitter(features, (64, 64), hard=True)
            num_selected = result.num_selected_per_batch[0].item()

            # 最多选 5 个候选
            assert 1 <= num_selected <= 5, \
                f"num_selected={num_selected} out of range [1, 5]"


class TestSplitterConfigValidation:
    """SplitterConfig 验证测试."""

    def test_config_validate_max_depth_1(self):
        """验证 SplitterConfig.validate() 对 max_depth=1 抛出异常."""
        config = SplitterConfig(max_level_limit=1)

        with pytest.raises(ValueError, match="max_level_limit.*2"):
            config.validate()

    def test_config_validate_max_depth_2(self):
        """验证 SplitterConfig.validate() 对 max_depth=2 通过."""
        config = SplitterConfig(max_level_limit=2)

        # 不应该抛出异常
        config.validate()

    def test_config_validate_max_depth_8(self):
        """验证 SplitterConfig.validate() 对 max_depth=8 通过."""
        config = SplitterConfig(max_level_limit=8)

        # 不应该抛出异常
        config.validate()


class TestMaxDepthBoundaryComparison:
    """max_depth 边界对比测试."""

    def test_max_depth_1_vs_2_regions(self):
        """对比 max_depth=1 和 max_depth=2 的区域数量."""
        splitter_1 = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=1,
            image_size=(64, 64),
        )
        splitter_2 = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=2,
            image_size=(64, 64),
        )

        # max_depth=1: 5 个候选 (1 + 4)
        assert splitter_1.num_candidates == 5

        # max_depth=2: 21 个候选 (1 + 4 + 16)
        assert splitter_2.num_candidates == 21

    def test_max_depth_1_no_crash(self):
        """验证 max_depth=1 不会导致崩溃."""
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=1,
            image_size=(64, 64),
        )

        # 多次前向传播不应崩溃
        for _ in range(5):
            features = torch.randn(2, 256, 16, 16)
            result = splitter(features, (64, 64), hard=True)
            assert result.selected_mask.numel() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
