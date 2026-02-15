# -*- coding: utf-8 -*-
"""
FastBitwiseHilbert 单元测试

验证:
1. 双射性质: d ↔ (x,y) 互为逆运算
2. 与 HilbertCurve 的一致性 (2^k × 2^k)
3. 边界情况处理
4. dtype → 位宽推导
5. fast_lca 数学正确性
"""

import pytest
import torch

from vit_pytorch.core.fast_bitwise_hilbert import FastBitwiseHilbert, fast_lca
from vit_pytorch.core.curve_hilbert import HilbertCurve


class TestFastBitwiseHilbert:
    """FastBitwiseHilbert 核心测试"""

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    @pytest.mark.parametrize("H,W", [(4, 4), (8, 8), (16, 16)])
    def test_bijection_2_power(self, dtype, H, W):
        """验证双射性质: d ↔ (x,y) 互为逆运算 (2^k × 2^k)"""
        x = torch.randint(0, W, (100,), dtype=dtype)
        y = torch.randint(0, H, (100,), dtype=dtype)

        d = FastBitwiseHilbert.xy_to_d(x, y, H, W)
        x_rec, y_rec = FastBitwiseHilbert.d_to_xy(d, H, W)

        assert torch.all(x == x_rec), f"x mismatch: {x[:5]} vs {x_rec[:5]}"
        assert torch.all(y == y_rec), f"y mismatch: {y[:5]} vs {y_rec[:5]}"

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    @pytest.mark.parametrize("H,W", [(7, 13), (100, 100), (3, 7), (50, 200)])
    def test_bijection_rectangle(self, dtype, H, W):
        """验证双射性质: 任意矩形"""
        x = torch.randint(0, W, (50,), dtype=dtype)
        y = torch.randint(0, H, (50,), dtype=dtype)

        d = FastBitwiseHilbert.xy_to_d(x, y, H, W)
        x_rec, y_rec = FastBitwiseHilbert.d_to_xy(d, H, W)

        assert torch.all(x == x_rec)
        assert torch.all(y == y_rec)

    @pytest.mark.parametrize("n", [2, 4, 8, 16])
    def test_consistency_with_hilbert_curve(self, n):
        """验证与 HilbertCurve 的一致性 (2^k × 2^k)"""
        # 生成所有坐标
        x = torch.arange(n, dtype=torch.int64)
        y = torch.arange(n, dtype=torch.int64)
        xv, yv = torch.meshgrid(x, y, indexing="ij")
        xv = xv.flatten()
        yv = yv.flatten()

        # HilbertCurve 结果 (作为参考)
        d_ref = torch.tensor(
            [HilbertCurve.xy_to_d(n, xi.item(), yi.item()) for xi, yi in zip(xv, yv)]
        )

        # FastBitwiseHilbert 结果
        d_new = FastBitwiseHilbert.xy_to_d(xv, yv, n, n)

        # 验证完全一致
        assert torch.all(
            d_ref == d_new
        ), f"Mismatch at n={n}: {d_ref[:10]} vs {d_new[:10]}"

    def test_edge_case_1x1(self):
        """边界测试: 1×1"""
        x = torch.tensor([0], dtype=torch.int64)
        y = torch.tensor([0], dtype=torch.int64)
        d = FastBitwiseHilbert.xy_to_d(x, y, 1, 1)
        assert d.numel() == 1
        assert d[0] == 0

    def test_edge_case_1xN(self):
        """边界测试: 1×N"""
        W = 100
        x = torch.arange(W, dtype=torch.int64)
        y = torch.zeros(W, dtype=torch.int64)
        d = FastBitwiseHilbert.xy_to_d(x, y, 1, W)
        x_rec, y_rec = FastBitwiseHilbert.d_to_xy(d, 1, W)
        assert torch.all(x == x_rec)
        assert torch.all(y == y_rec)

    def test_edge_case_Nx1(self):
        """边界测试: N×1"""
        H = 100
        x = torch.zeros(H, dtype=torch.int64)
        y = torch.arange(H, dtype=torch.int64)
        d = FastBitwiseHilbert.xy_to_d(x, y, H, 1)
        x_rec, y_rec = FastBitwiseHilbert.d_to_xy(d, H, 1)
        assert torch.all(x == x_rec)
        assert torch.all(y == y_rec)

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_dtype_bit_width(self, dtype):
        """验证 dtype → 位宽推导"""
        B = FastBitwiseHilbert.infer_bit_width(dtype)
        if dtype == torch.int32:
            assert B == 16
        elif dtype == torch.int64:
            assert B == 32


class TestFastLCA:
    """fast_lca 函数测试"""

    def test_fast_lca_same(self):
        """测试相同索引的 LCA"""
        assert fast_lca(7, 7) == 7
        assert fast_lca(0, 0) == 0
        assert fast_lca(100, 100) == 100

    def test_fast_lca_zero_xor(self):
        """测试 XOR 为 0 的情况"""
        # 相同索引应该返回自身
        for i in range(10):
            assert fast_lca(i, i) == i

    def test_fast_lca_tensor(self):
        """测试张量版本的 LCA"""
        idx1 = torch.tensor([5, 7, 10])
        idx2 = torch.tensor([13, 7, 12])
        result = fast_lca(idx1, idx2)
        # 验证返回值是有效的（非负且在合理范围内）
        assert result.dtype == torch.long
        assert torch.all(result >= 0)


class TestNormalizedMapping:
    """归一化定点映射测试"""

    def test_normalize_preserves_order(self):
        """验证归一化保持坐标顺序"""
        x = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
        y = torch.zeros(6, dtype=torch.int64)

        xv, yv = FastBitwiseHilbert.normalize_coords(x, y, 1, 6, 16)

        # 验证单调性
        assert torch.all(xv[:-1] <= xv[1:])

    def test_normalize_symmetry(self):
        """验证 x,y 对称性"""
        # 相同尺寸时 x 和 y 的映射应该对称
        x = torch.tensor([10, 20, 30], dtype=torch.int64)
        y = torch.tensor([10, 20, 30], dtype=torch.int64)

        xv, yv = FastBitwiseHilbert.normalize_coords(x, y, 100, 100, 16)

        assert torch.all(xv == yv)

    def test_normalize_extreme_values(self):
        """验证极端值的映射"""
        x = torch.tensor([0, 99], dtype=torch.int64)
        y = torch.tensor([0, 99], dtype=torch.int64)

        xv_min, yv_min = FastBitwiseHilbert.normalize_coords(x[:1], y[:1], 100, 100, 16)
        xv_max, yv_max = FastBitwiseHilbert.normalize_coords(x[1:], y[1:], 100, 100, 16)

        # 最小值应该映射到 0 附近
        assert xv_min[0] == 0
        assert yv_min[0] == 0
        # 最大值应该映射到接近最大值（由于整数除法略有偏差）
        assert xv_max[0] >= 64000  # 接近 2^16 - 1 = 65535
        assert yv_max[0] >= 64000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
