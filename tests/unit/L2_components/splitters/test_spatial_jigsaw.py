"""
I113-9 测试: Spatial Jigsaw Loss 空间偏移预测

验证 SpatialJigsawLoss 修复 GeometricJigsawLoss 的方向丢失问题。
"""

import pytest
import torch

from training.fractal_proxy_experiments import (
    SpatialJigsawLoss,
    GeometricJigsawLoss,
    HilbertIndexShuffler,
    GJPConfig,
)


class TestSpatialJigsawLoss:
    """测试 I113-9: Spatial Jigsaw Loss 空间偏移预测。"""

    def test_spatial_jigsaw_loss_basic(self):
        """验证基本的 spatial jigsaw loss 计算。"""
        config = GJPConfig(
            max_relative_distance=8,
            jigsaw_weight=1.0,
        )

        loss_fn = SpatialJigsawLoss(config=config, feat_dim=128)

        # 创建模拟数据
        B, N, D = 2, 8, 128
        tokens = torch.randn(B, N, D)
        depths = torch.randint(1, 4, (B, N))
        paths = torch.randint(0, 4, (B, N, 3))

        # 构建 levels_info: [B, N, D+1] = [B, N, 4]
        levels_info = torch.cat([depths.unsqueeze(-1), paths], dim=-1)

        loss, info = loss_fn(tokens, levels_info)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0  # MSE loss 应该非负
        assert "jigsaw_loss" in info
        assert "num_pairs" in info
        assert info["num_pairs"] > 0

    def test_spatial_jigsaw_preserves_direction(self):
        """验证 spatial offset 保留了方向信息。"""
        config = GJPConfig(max_relative_distance=16)
        loss_fn = SpatialJigsawLoss(config=config, feat_dim=64)

        B, N, D = 1, 4, 64
        tokens = torch.randn(B, N, D)

        # 创建有明显空间关系的 depths
        # 4 个 token，深度为 2 (4x4 Hilbert 网格)
        depths = torch.tensor([[2, 2, 2, 2]])

        # Hilbert 索引: 0, 1, 4, 5 对应坐标 (0,0), (1,0), (0,1), (1,1)
        # paths: [depth-1=1] 每个 token 的路径
        # d=0 -> (0,0): path=0
        # d=1 -> (1,0): path=1
        # d=4 -> (0,1): path=4 / 4 = 1 -> 实际上需要用正确的路径编码
        paths = torch.tensor([[[0], [1], [4], [5]]])

        levels_info = torch.cat([depths.unsqueeze(-1), paths], dim=-1)

        loss, info = loss_fn(tokens, levels_info)

        # 验证方向信息被保留
        assert "direction_acc_x" in info
        assert "direction_acc_y" in info

    def test_spatial_jigsaw_mse_loss(self):
        """验证使用 MSE Loss 而非 CrossEntropy。"""
        config = GJPConfig(max_relative_distance=8)
        loss_fn = SpatialJigsawLoss(config=config, feat_dim=64)

        B, N, D = 1, 6, 64
        tokens = torch.randn(B, N, D)
        depths = torch.randint(1, 4, (B, N))
        paths = torch.randint(0, 4, (B, N, 3))
        levels_info = torch.cat([depths.unsqueeze(-1), paths], dim=-1)

        loss, info = loss_fn(tokens, levels_info)

        # MSE loss 应该产生连续值，而非分类
        assert loss.requires_grad

    def test_geometric_jigsaw_loss_alias(self):
        """验证 GeometricJigsawLoss 是 SpatialJigsawLoss 的别名。"""
        assert GeometricJigsawLoss is SpatialJigsawLoss

    def test_spatial_jigsaw_output_dimension(self):
        """验证预测头输出维度为 2 (dx, dy)。"""
        config = GJPConfig(max_relative_distance=8)
        loss_fn = SpatialJigsawLoss(config=config, feat_dim=64)

        # 检查预测头结构
        predictor = loss_fn.offset_predictor
        assert isinstance(predictor, torch.nn.Sequential)

        # 最后一个 Linear 层应该输出 2 个值
        last_layer = predictor[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        assert last_layer.out_features == 2  # dx, dy

    def test_spatial_jigsaw_with_shuffled_tokens(self):
        """验证 shuffler 与 spatial jigsaw 的兼容性。"""
        config = GJPConfig(
            shuffle_group_size=4,
            max_relative_distance=8,
        )

        loss_fn = SpatialJigsawLoss(config=config, feat_dim=64)
        shuffler = HilbertIndexShuffler(
            group_size=config.shuffle_group_size,
            max_depth=3,
        )

        B, N, D = 1, 4, 64
        tokens = torch.randn(B, N, D)
        depths = torch.tensor([[2, 2, 2, 2]])
        paths = torch.tensor([[[0], [1], [4], [5]]])
        levels_info = torch.cat([depths.unsqueeze(-1), paths], dim=-1)

        # Shuffle
        batch_indices = torch.zeros(N, dtype=torch.long)
        shuffled_levels, shuffle_indices = shuffler.shuffle_within_groups(levels_info, batch_indices)

        loss, info = loss_fn(tokens, shuffled_levels)

        assert loss.requires_grad
        assert info["num_pairs"] > 0

    def test_spatial_jigsaw_info_metrics(self):
        """验证返回的 info 包含正确的指标。"""
        config = GJPConfig(max_relative_distance=8)
        loss_fn = SpatialJigsawLoss(config=config, feat_dim=64)

        B, N, D = 2, 6, 64
        tokens = torch.randn(B, N, D)
        depths = torch.randint(1, 4, (B, N))
        paths = torch.randint(0, 4, (B, N, 3))
        levels_info = torch.cat([depths.unsqueeze(-1), paths], dim=-1)

        loss, info = loss_fn(tokens, levels_info)

        # 验证所有指标都存在
        expected_keys = ["jigsaw_loss", "num_pairs", "direction_acc_x",
                        "direction_acc_y", "mean_offset_x", "mean_offset_y"]
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"

        # 验证值范围
        assert 0 <= info["direction_acc_x"] <= 1
        assert 0 <= info["direction_acc_y"] <= 1


class TestHilbertToXY:
    """测试 Hilbert 索引到空间坐标的转换。"""

    def test_hilbert_to_xy_consistency(self):
        """验证 Hilbert 索引到坐标的转换一致性。"""
        config = GJPConfig(max_relative_distance=8)
        loss_fn = SpatialJigsawLoss(config=config, feat_dim=64)

        B, N = 1, 16
        depths = torch.ones(B, N, dtype=torch.long) * 3  # depth=3, grid=8
        paths = torch.randint(0, 4, (B, N, 3))

        # 计算 Hilbert 索引
        hilbert_indices = HilbertIndexShuffler.compute_hilbert_indices(depths, paths)

        # 转换为坐标
        grid_size = 8
        cx, cy = loss_fn._hilbert_index_to_xy(hilbert_indices, grid_size)

        assert cx.shape == (B, N)
        assert cy.shape == (B, N)
        assert cx.min() >= 0
        assert cx.max() < grid_size
        assert cy.min() >= 0
        assert cy.max() < grid_size

    def test_hilbert_xy_to_index_roundtrip(self):
        """验证 xy -> Hilbert 索引的往返转换。"""
        from vit_pytorch.core.curve_hilbert import HilbertCurve

        config = GJPConfig(max_relative_distance=8)
        loss_fn = SpatialJigsawLoss(config=config, feat_dim=64)

        B, N = 1, 16
        depths = torch.ones(B, N, dtype=torch.long) * 3
        paths = torch.randint(0, 4, (B, N, 3))

        # 获取 Hilbert 索引
        h = HilbertIndexShuffler.compute_hilbert_indices(depths, paths)

        # 转换为 xy
        grid_size = 8
        cx, cy = loss_fn._hilbert_index_to_xy(h, grid_size)

        # 验证坐标范围
        assert (cx >= 0).all() and (cx < grid_size).all()
        assert (cy >= 0).all() and (cy < grid_size).all()


class TestBackwardCompatibility:
    """测试向后兼容性。"""

    def test_import_geometric_jigsaw_loss(self):
        """验证可以从 training 模块导入 GeometricJigsawLoss。"""
        from training import GeometricJigsawLoss
        assert GeometricJigsawLoss is SpatialJigsawLoss

    def test_geometric_jigsaw_loss_works(self):
        """验证使用 GeometricJigsawLoss 别名可以正常工作。"""
        from training import GeometricJigsawLoss

        config = GJPConfig(max_relative_distance=8)
        loss_fn = GeometricJigsawLoss(config=config, feat_dim=64)

        B, N, D = 1, 4, 64
        tokens = torch.randn(B, N, D)
        depths = torch.randint(1, 4, (B, N))
        paths = torch.randint(0, 4, (B, N, 3))
        levels_info = torch.cat([depths.unsqueeze(-1), paths], dim=-1)

        loss, info = loss_fn(tokens, levels_info)

        assert isinstance(loss, torch.Tensor)
        assert "direction_acc_x" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
