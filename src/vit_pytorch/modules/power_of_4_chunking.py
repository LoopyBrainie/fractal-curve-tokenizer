"""
Power-of-4 Alignment Chunking (v1.3 — Hilbert sub-manifold batching).

数学形式化
============

输入: 序列 seq ∈ R^{L × D} (L = token 数)
目标: 沿 Hilbert 曲线将序列切分为大小为 4^k 的连续子序列，每个子序列的
       2D 坐标（通过 d → (x, y) 映射得到）必须构成一个连通的四叉树子块。

切分规则:
    chunk_size = 4^k     (k >= 0)
    n_chunks   = L / 4^k
    每个 chunk 对应 Hilbert 索引范围 [c·i, c·(i+1))，映射回 2D 后
    应位于同一 2^k × 2^k 网格子区域（连通性验证通过 HilbertCurve.d_to_xy）。

Raises:
    ValueError: 当 L 不能被 4^k 整除时

Reference:
    docs/superpowers/lemmas/2026-06-08-power-of-4-chunking-proof.md
    v1.3 spec — B.13 Power-of-4 Alignment Chunking
"""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor

from ..core.curve_hilbert import HilbertCurve


def power_of_4_chunk(seq: Tensor, k: int) -> List[Tensor]:
    """将序列按 4^k 大小切分，每个 chunk 的 Hilbert 索引构成连通 2D 区域.

    Args:
        seq: 形状 [L, ...] 的张量
        k:  切分粒度, chunk_size = 4^k

    Returns:
        List[Tensor]: n_chunks 个张量，每个形状 [4^k, ...]

    Raises:
        ValueError: 当 k < 0, L 不可被 4^k 整除，或 L == 0 时
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    L = seq.shape[0]
    if L == 0:
        raise ValueError("seq must be non-empty")
    chunk_size = 4 ** k
    if L % chunk_size != 0:
        raise ValueError(
            f"Sequence length L={L} is not divisible by 4^k={chunk_size} (k={k})"
        )

    # 序列按 Hilbert 顺序排列：第 i 个 token 对应 Hilbert 索引 i
    # 4^k 个连续 Hilbert 索引映射回 2D 后正好占据一个 2^k × 2^k 网格子块
    # （Hilbert 曲线的自相似性保证）
    return [seq[i * chunk_size : (i + 1) * chunk_size] for i in range(L // chunk_size)]


def verify_chunk_connectedness(
    chunk_indices: Tensor,
    n: int,
    k: int,
) -> bool:
    """验证 chunk 的 Hilbert 索引映射到 2D 后是否构成 2^k × 2^k 连通子块.

    Args:
        chunk_indices: [4^k] 范围的整数张量
        n: Hilbert 网格阶数 (n × n)
        k: 对应 chunk 粒度

    Returns:
        True 当 2D 坐标都在 [a, a+2^k) × [b, b+2^k) 的某个 2^k 子网格中
    """
    chunk_size = 4 ** k
    if chunk_indices.numel() != chunk_size:
        return False
    if n & (n - 1) != 0 or k < 0 or (1 << k) > n:
        return False

    d_long = chunk_indices.long()
    x_coords, y_coords = HilbertCurve.d_to_xy_batch(n, d_long)

    side = 1 << k
    x_min, x_max = x_coords.min().item(), x_coords.max().item()
    y_min, y_max = y_coords.min().item(), y_coords.max().item()
    return (x_max - x_min < side) and (y_max - y_min < side)
