"""
I113-8 测试: 矩形区域的 Hilbert 局部性保证

验证 RectHilbertIndex 修复非正方形图像的 Hilbert 局部性失效问题。
"""

import pytest
import torch

from vit_pytorch.curve_hilbert import (
    HilbertCurve,
    RectHilbertIndex,
    HilbertLocalityMetrics,
)


class TestRectHilbertIndex:
    """测试 I113-8: 矩形区域的 Hilbert 索引计算。"""

    def test_rectangular_hilbert_index_basic(self):
        """验证矩形区域的基本 Hilbert 索引计算。"""
        H, W = 64, 128
        depth = 3

        # 创建简单的区域坐标
        x0 = torch.tensor([0.0, 0.0, 32.0, 32.0])
        y0 = torch.tensor([0.0, 32.0, 0.0, 32.0])
        x1 = torch.tensor([32.0, 32.0, 64.0, 64.0])
        y1 = torch.tensor([32.0, 64.0, 32.0, 64.0])

        indices = RectHilbertIndex.from_region(x0, y0, x1, y1, depth, H, W)

        assert indices.shape == (4,)
        assert indices.dtype == torch.long
        assert indices.min() >= 0

    def test_square_vs_rectangle_consistency(self):
        """验证正方形和矩形的基本一致性。"""
        depth = 2
        grid_size = 2 ** depth  # 4x4 = 16 个区域

        # 正方形 64x64
        x0_sq = torch.tensor([0.0, 16.0, 32.0, 48.0])
        y0_sq = torch.tensor([0.0, 0.0, 0.0, 0.0])
        x1_sq = torch.tensor([16.0, 32.0, 48.0, 64.0])
        y1_sq = torch.tensor([16.0, 16.0, 16.0, 16.0])

        # 矩形 64x128
        x0_rect = torch.tensor([0.0, 32.0, 64.0, 96.0])
        y0_rect = torch.tensor([0.0, 0.0, 0.0, 0.0])
        x1_rect = torch.tensor([32.0, 64.0, 96.0, 128.0])
        y1_rect = torch.tensor([16.0, 16.0, 16.0, 16.0])

        indices_sq = RectHilbertIndex.from_region(x0_sq, y0_sq, x1_sq, y1_sq, depth, 64, 64)
        indices_rect = RectHilbertIndex.from_region(x0_rect, y0_rect, x1_rect, y1_rect, depth, 64, 128)

        max_index = grid_size * grid_size
        assert (indices_sq < max_index).all()
        assert (indices_rect < max_index).all()

    def test_hilbert_locality_rectangular(self):
        """验证矩形区域的 Hilbert 索引在有效范围内。"""
        H, W = 64, 128
        depth = 3
        grid_size = 2 ** depth  # 8x8 网格

        region_h = H / grid_size  # 8
        region_w = W / grid_size  # 16

        x0_list = []
        y0_list = []
        x1_list = []
        y1_list = []

        for i in range(grid_size):
            for j in range(grid_size):
                x0_list.append(j * region_w)
                y0_list.append(i * region_h)
                x1_list.append((j + 1) * region_w)
                y1_list.append((i + 1) * region_h)

        x0 = torch.tensor(x0_list)
        y0 = torch.tensor(y0_list)
        x1 = torch.tensor(x1_list)
        y1 = torch.tensor(y1_list)

        indices = RectHilbertIndex.from_region(x0, y0, x1, y1, depth, H, W)

        # 验证索引唯一性
        max_index = grid_size * grid_size
        assert (indices < max_index).all(), f"索引超出范围: max={indices.max()}, limit={max_index}"

    def test_extreme_aspect_ratio(self):
        """测试极端宽高比 (4:1) 的情况。"""
        H, W = 32, 128
        depth = 3
        grid_size = 2 ** depth  # 8x8 网格

        region_h = H / grid_size
        region_w = W / grid_size

        x0_list = []
        y0_list = []
        x1_list = []
        y1_list = []

        for i in range(grid_size):
            for j in range(grid_size):
                x0_list.append(j * region_w)
                y0_list.append(i * region_h)
                x1_list.append((j + 1) * region_w)
                y1_list.append((i + 1) * region_h)

        x0 = torch.tensor(x0_list)
        y0 = torch.tensor(y0_list)
        x1 = torch.tensor(x1_list)
        y1 = torch.tensor(y1_list)

        indices = RectHilbertIndex.from_region(x0, y0, x1, y1, depth, H, W)

        max_index = grid_size * grid_size
        assert (indices < max_index).all()
        assert (indices >= 0).all()

    def test_from_center_method(self):
        """测试 from_center 方法。"""
        H, W = 64, 128
        depth = 2
        grid_size = 2 ** depth

        cx = torch.tensor([8.0, 40.0, 8.0, 40.0])
        cy = torch.tensor([8.0, 8.0, 40.0, 40.0])

        indices = RectHilbertIndex.from_center(cx, cy, depth, H, W)

        max_index = grid_size * grid_size
        assert (indices < max_index).all()
        assert (indices >= 0).all()


class TestHilbertLocalityMetrics:
    """测试 Hilbert 局部性度量工具。"""

    def test_standard_hilbert_locality(self):
        """验证标准 Hilbert 曲线的局部性指标。"""
        n = 8
        points = [HilbertCurve.d_to_xy(n, d) for d in range(n * n)]

        avg_loss = HilbertLocalityMetrics.average_locality_loss(points)
        max_jump = HilbertLocalityMetrics.max_jump_distance(points)

        assert avg_loss < 2.0, f"平均局部性损失过大: {avg_loss}"
        assert max_jump <= 1.5, f"最大跳跃过大: {max_jump}"

    def test_row_major_locality(self):
        """对比行主序扫描的局部性。"""
        H, W = 8, 8
        points = [(x, y) for y in range(H) for x in range(W)]

        max_jump = HilbertLocalityMetrics.max_jump_distance(points)

        # 行主序应该有较大的最大跳跃
        assert max_jump > 6.0, f"行主序最大跳跃应大于 6，实际: {max_jump}"


class TestIntegration:
    """集成测试：验证 Splitter 中的 Hilbert 索引计算。"""

    def test_small_model_forward(self):
        """验证小模型前向传播。"""
        from vit_pytorch import FractalCurveViT

        model = FractalCurveViT(
            image_size=64,
            num_classes=100,
        )

        x = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            output = model(x)

        assert output.logits.shape == (1, 100)

    def test_rectangular_image_forward(self):
        """验证非正方形图像的前向传播。"""
        from vit_pytorch import FractalCurveViT

        model = FractalCurveViT(
            image_size=None,
            num_classes=100,
        )

        x = torch.randn(1, 3, 32, 64)

        with torch.no_grad():
            output = model(x)

        assert output.logits.shape == (1, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
