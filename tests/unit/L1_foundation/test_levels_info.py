# -*- coding: utf-8 -*-
"""
L1 Foundation: LevelsInfo Tests

对应模块: vit_pytorch.levels_info

数学形式化
==========

LevelsInfo dataclass 契约验证:

C1: depth ∈ [-1, D]
    - -1 表示 padding sentinel
    - 0 到 D 是有效深度

C2: path ∈ [0, 3] for valid tokens
    - 四象限编码
    - 仅对 depth >= 0 的 token 有效

C3: path length = depth
    - path 的有效长度等于 depth
    - padding 位置用 0 填充
"""

from __future__ import annotations

import pytest
import torch

from vit_pytorch import LevelsInfo


class TestLevelsInfoContract:
    """LevelsInfo 契约验证 (C1, C2, C3)."""

    def test_c1_depth_range_valid(self):
        """C1: depth ∈ [-1, D] (有效情况)."""
        depths = torch.tensor([[0, 1, 2, -1]], dtype=torch.long)
        paths = torch.zeros(1, 4, 3, dtype=torch.long)
        levels = LevelsInfo.from_arrays(depths, paths, max_depth=3)
        assert levels.depths.equal(depths)

    def test_c1_depth_violation_negative(self):
        """C1 违反: depth < -1."""
        depths = torch.tensor([[-2]], dtype=torch.long)
        paths = torch.zeros(1, 1, 3, dtype=torch.long)
        with pytest.raises(AssertionError):
            LevelsInfo.from_arrays(depths, paths, max_depth=3)

    def test_c1_depth_violation_exceeds_max(self):
        """C1 违反: depth > max_depth."""
        depths = torch.tensor([[5]], dtype=torch.long)
        paths = torch.zeros(1, 1, 3, dtype=torch.long)
        with pytest.raises(AssertionError):
            LevelsInfo.from_arrays(depths, paths, max_depth=3)

    def test_c2_path_range_valid(self):
        """C2: path ∈ [0, 3] (有效情况)."""
        depths = torch.tensor([[0, 1, 2]], dtype=torch.long)
        paths = torch.tensor([[[0, 0, 0], [1, 2, 0], [3, 1, 0]]], dtype=torch.long)
        levels = LevelsInfo.from_arrays(depths, paths, max_depth=3)
        assert (levels.paths >= 0).all() and (levels.paths <= 3).all()

    def test_c2_path_violation(self):
        """C2 违反: path > 3."""
        depths = torch.tensor([[1]], dtype=torch.long)
        paths = torch.tensor([[[5, 0, 0]]], dtype=torch.long)
        with pytest.raises(AssertionError):
            LevelsInfo.from_arrays(depths, paths, max_depth=3)

    def test_c2_path_negative_violation(self):
        """C2 违反: path < 0."""
        depths = torch.tensor([[1]], dtype=torch.long)
        paths = torch.tensor([[[-1, 0, 0]]], dtype=torch.long)
        with pytest.raises(AssertionError):
            LevelsInfo.from_arrays(depths, paths, max_depth=3)

    def test_padding_sentinel_allowed(self):
        """Padding sentinel (-1) 应该被允许."""
        depths = torch.tensor([[-1, 0, 1]], dtype=torch.long)
        paths = torch.zeros(1, 3, 3, dtype=torch.long)
        levels = LevelsInfo.from_arrays(depths, paths, max_depth=3)
        assert (levels.depths == -1).any()

    def test_c3_path_length_consistency(self):
        """C3: 路径长度应与深度一致."""
        depths = torch.tensor([[2, 1, 0]], dtype=torch.long)
        paths = torch.tensor([[[0, 1, 2], [3, 0, 0], [0, 0, 0]]], dtype=torch.long)
        levels = LevelsInfo.from_arrays(depths, paths, max_depth=3)
        assert levels.paths.shape == (1, 3, 3)


class TestLevelsInfoFactoryMethods:
    """工厂方法测试."""

    def test_from_arrays_basic(self):
        """基础 from_arrays 测试."""
        depths = torch.tensor([[1, 2, 3]], dtype=torch.long)
        paths = torch.zeros(1, 3, 4, dtype=torch.long)
        levels = LevelsInfo.from_arrays(depths, paths, max_depth=4)

        assert levels.data.shape == (1, 3, 5)
        assert levels.max_depth == 4
        assert levels.batch_size == 1
        assert levels.num_tokens == 3
        assert levels.paths.shape == (1, 3, 4)

    def test_from_arrays_batch(self):
        """Batch 处理测试."""
        depths = torch.tensor([[1, 2], [3, 0]], dtype=torch.long)
        paths = torch.zeros(2, 2, 4, dtype=torch.long)
        levels = LevelsInfo.from_arrays(depths, paths, max_depth=4)

        assert levels.batch_size == 2
        assert levels.num_tokens == 2

    def test_from_tokenizer_output(self):
        """TokenizerOutput 集成测试."""
        from vit_pytorch import TokenizerOutput, TokenSequence

        tokens = torch.randn(2, 10, 64)
        levels = torch.zeros(2, 10, 5, dtype=torch.long)
        levels[:, :, 0] = torch.randint(0, 5, (2, 10))

        sequences = [TokenSequence(tokens=t) for t in tokens]
        for i, seq in enumerate(sequences):
            seq.metadata["levels"] = levels[i]

        output = TokenizerOutput(sequences=sequences)
        levels_info = output.get_levels_info(max_depth=4)

        assert isinstance(levels_info, LevelsInfo)
        assert levels_info.batch_size == 2
        assert levels_info.max_depth == 4

    def test_random_levels_info(self):
        """随机 LevelsInfo 生成."""
        levels = LevelsInfo.random(B=4, N=16, max_depth=6, device='cpu')

        assert levels.batch_size == 4
        assert levels.num_tokens == 16
        assert levels.max_depth == 6
        assert levels.depths.shape == (4, 16)
        assert levels.paths.shape == (4, 16, 6)


class TestLevelsInfoProperties:
    """属性访问器测试."""

    def test_depths_property(self):
        """depths 属性测试."""
        depths = torch.tensor([[1, 2, 3]], dtype=torch.long)
        paths = torch.zeros(1, 3, 4, dtype=torch.long)
        levels = LevelsInfo.from_arrays(depths, paths, max_depth=4)

        assert levels.depths.equal(depths)
        assert levels.depths.shape == (1, 3)

    def test_paths_property(self):
        """paths 属性测试."""
        depths = torch.tensor([[1, 2]], dtype=torch.long)
        paths = torch.randint(0, 4, (1, 2, 5), dtype=torch.long)
        levels = LevelsInfo.from_arrays(depths, paths, max_depth=5)

        assert levels.paths.shape == (1, 2, 5)

    def test_shape_property(self):
        """shape 属性测试."""
        depths = torch.tensor([[1, 2]], dtype=torch.long)
        paths = torch.zeros(1, 2, 4, dtype=torch.long)
        levels = LevelsInfo.from_arrays(depths, paths, max_depth=4)

        assert levels.shape == (1, 2, 5)

    def test_max_level_property(self):
        """max_level 属性测试."""
        depths = torch.zeros(1, 4, dtype=torch.long)
        paths = torch.zeros(1, 4, 6, dtype=torch.long)
        levels = LevelsInfo.from_arrays(depths, paths, max_depth=6)

        assert levels.max_level == 6

    def test_len_and_num_tokens(self):
        """len() 和 num_tokens 测试."""
        depths = torch.zeros(3, 8, dtype=torch.long)
        paths = torch.zeros(3, 8, 5, dtype=torch.long)
        levels = LevelsInfo.from_arrays(depths, paths, max_depth=5)

        assert len(levels) == 8
        assert levels.num_tokens == 8


class TestLevelsInfoDeviceMigration:
    """设备迁移测试."""

    def test_to_cpu(self):
        """迁移到 CPU."""
        depths = torch.tensor([[1, 2]], dtype=torch.long)
        paths = torch.zeros(1, 2, 4, dtype=torch.long)
        levels = LevelsInfo.from_arrays(depths, paths, max_depth=4)

        levels_cpu = levels.cpu()
        assert levels_cpu.data.device.type == 'cpu'

    def test_to_device(self):
        """迁移到指定设备."""
        depths = torch.tensor([[1, 2]], dtype=torch.long)
        paths = torch.zeros(1, 2, 4, dtype=torch.long)
        levels = LevelsInfo.from_arrays(depths, paths, max_depth=4)

        levels_gpu = levels.to(torch.device('cpu'))
        assert levels_gpu.data.device.type == 'cpu'


class TestLevelsInfoHilbertTools:
    """Hilbert Curve 工具测试."""

    def test_get_hilbert_indices_basic(self):
        """Hilbert 索引计算 (基本功能)."""
        depths = torch.tensor([[1, 2]], dtype=torch.long)
        paths = torch.zeros(1, 2, 4, dtype=torch.long)
        paths[0, 0, 0] = 1
        paths[0, 1, :2] = torch.tensor([1, 2])

        levels = LevelsInfo.from_arrays(depths, paths, max_depth=4)
        hilbert_indices = levels.get_hilbert_indices()

        assert hilbert_indices.shape == (1, 2)
        assert (hilbert_indices >= 0).all()

    def test_get_lca_matrix_basic(self):
        """LCA 矩阵计算 (基本功能)."""
        depths = torch.tensor([[1, 1, 2]], dtype=torch.long)
        paths = torch.zeros(1, 3, 4, dtype=torch.long)
        paths[0, 0, 0] = 1
        paths[0, 1, 0] = 1
        paths[0, 2, :2] = torch.tensor([1, 2])

        levels = LevelsInfo.from_arrays(depths, paths, max_depth=4)
        lca_matrix = levels.get_lca_matrix()

        assert lca_matrix.shape == (1, 3, 3)
        assert (lca_matrix >= 0).all()
        assert lca_matrix[0, 0, 1] == lca_matrix[0, 1, 0]


class TestLevelsInfoEdgeCases:
    """边界情况测试."""

    def test_single_token(self):
        """单 token 情况."""
        depths = torch.tensor([[2]], dtype=torch.long)
        paths = torch.zeros(1, 1, 4, dtype=torch.long)
        paths[0, 0, :2] = torch.tensor([0, 1])

        levels = LevelsInfo.from_arrays(depths, paths, max_depth=4)
        assert levels.num_tokens == 1
        assert levels.depths[0, 0] == 2

    def test_max_depth(self):
        """最大深度情况."""
        depths = torch.tensor([[8]], dtype=torch.long)
        paths = torch.randint(0, 4, (1, 1, 8), dtype=torch.long)

        levels = LevelsInfo.from_arrays(depths, paths, max_depth=8)
        assert levels.max_depth == 8
        assert levels.paths.shape == (1, 1, 8)

    def test_empty_paths_padding(self):
        """路径 padding 测试."""
        depths = torch.tensor([[0, 8]], dtype=torch.long)
        paths = torch.zeros(1, 2, 8, dtype=torch.long)

        levels = LevelsInfo.from_arrays(depths, paths, max_depth=8)
        assert levels.paths[0, 0].sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
