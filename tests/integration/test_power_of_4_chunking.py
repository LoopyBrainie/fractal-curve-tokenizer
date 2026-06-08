"""
Power-of-4 Alignment Chunking integration tests (B.13).

Covers:
- test_chunk_size_is_4_to_the_k: 每个 chunk 大小 = 4^k
- test_chunk_indices_form_connected_subtree: chunk 索引映射到 2D 后是 2^k × 2^k 连通块
- test_no_chunk_crosses_quadtree_boundary: 验证 chunk 不跨越四叉树边界
- test_divisibility_error: L 不能被 4^k 整除时抛错
"""

import pytest
import torch

from vit_pytorch.core.curve_hilbert import HilbertCurve
from vit_pytorch.modules.power_of_4_chunking import power_of_4_chunk, verify_chunk_connectedness


class TestPowerOf4Chunking:
    """Power-of-4 Alignment Chunking 集成测试"""

    def test_chunk_size_is_4_to_the_k(self):
        """每个 chunk 大小严格 = 4^k."""
        L = 64  # 4^3
        seq = torch.arange(L, dtype=torch.float32)
        chunks = power_of_4_chunk(seq, k=3)
        assert len(chunks) == 1
        assert chunks[0].shape == (64,)

        # 4^2 = 16 → 4 个 chunk
        chunks2 = power_of_4_chunk(seq, k=2)
        assert len(chunks2) == 4
        for c in chunks2:
            assert c.shape == (16,)

        # 4^1 = 4 → 16 个 chunk
        chunks1 = power_of_4_chunk(seq, k=1)
        assert len(chunks1) == 16
        for c in chunks1:
            assert c.shape == (4,)

    def test_chunk_indices_form_connected_subtree(self):
        """chunk 的 Hilbert 索引映射到 2D 后构成 2^k × 2^k 连通子块."""
        L = 64  # 4^3 in 8x8 Hilbert grid
        n = 8   # 2^3
        seq = torch.arange(L, dtype=torch.float32)

        for k in (1, 2, 3):
            chunks = power_of_4_chunk(seq, k=k)
            chunk_size = 4 ** k
            for i, c in enumerate(chunks):
                indices = torch.arange(i * chunk_size, (i + 1) * chunk_size, dtype=torch.long)
                assert verify_chunk_connectedness(indices, n=n, k=k), (
                    f"Chunk {i} at k={k} is not a connected 2^k × 2^k sub-block"
                )

    def test_no_chunk_crosses_quadtree_boundary(self):
        """验证 chunk 不跨越四叉树边界（top-1 level 边界）."""
        # 在 16x16 网格上 (n=16, max_depth=4), 4^2=16 对应深度 d=2 的子树
        # 每个 chunk 应当完全位于某个 4x4 子网格内
        n = 16
        L = n * n  # 256
        seq = torch.arange(L, dtype=torch.float32)
        k = 2
        chunk_size = 4 ** k
        chunks = power_of_4_chunk(seq, k=k)
        for i, _ in enumerate(chunks):
            indices = torch.arange(i * chunk_size, (i + 1) * chunk_size, dtype=torch.long)
            x_coords, y_coords = HilbertCurve.d_to_xy_batch(n, indices)
            # 2^k = 4: chunk 占据 4x4 子网格
            # 验证 max-min < 4 (within 4x4 cell)
            assert x_coords.max().item() - x_coords.min().item() < 4
            assert y_coords.max().item() - y_coords.min().item() < 4

    def test_divisibility_error(self):
        """L 不能被 4^k 整除时抛 ValueError."""
        seq = torch.arange(20, dtype=torch.float32)  # not divisible by 16
        with pytest.raises(ValueError, match="not divisible"):
            power_of_4_chunk(seq, k=2)

    def test_k_zero_returns_singletons(self):
        """k=0 → chunk_size=1, 每个 chunk 单 token."""
        seq = torch.arange(8, dtype=torch.float32)
        chunks = power_of_4_chunk(seq, k=0)
        assert len(chunks) == 8
        for c in chunks:
            assert c.shape == (1,)

    def test_preserves_values(self):
        """chunk 切分不改变值（仅 re-arrange 顺序）."""
        seq = torch.arange(32, dtype=torch.float32)
        chunks = power_of_4_chunk(seq, k=2)  # 4 chunks of 8
        reconstructed = torch.cat(chunks, dim=0)
        assert torch.equal(reconstructed, seq)

    def test_multidim_sequence(self):
        """支持高维张量 (L, D)。"""
        L, D = 64, 4
        seq = torch.arange(L * D, dtype=torch.float32).reshape(L, D)
        chunks = power_of_4_chunk(seq, k=2)  # 64/16=4 chunks of 16x4
        assert len(chunks) == 4
        for c in chunks:
            assert c.shape == (16, D)
