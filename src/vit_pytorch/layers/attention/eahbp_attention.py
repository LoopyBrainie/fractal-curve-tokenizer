"""v1.3 STANDARD: EAHBP Attention (alpha-c-rev4).

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.2
EAHBP = Efficient Attention with Hilbert Block Pooling.

Design:
  - Block size b = 16 (EAHBP_BLOCK_SIZE)
  - Global pool size g = 8 (EAHBP_GLOBAL_POOL)
  - Per-head attention: (b² + g²) ops per query, vs full N²
  - For N=64: 256 + 64 = 320 ops/query, vs 4096 full = ~7.8% (theoretical 92% reduction)
  - Design target: 62.5% reduction (conservative, accounts for overhead)

3-gate system (§9.2):
  G1: precision gain >= EAHBP_G1_PRECISION_GAIN (0.2%) — 1D gate
  G2: throughput gain >= EAHBP_G2_THROUGHPUT_GAIN (1.4×) AND GME >= EAHBP_G2_GME_FLOOR (60%) — 2D gate
  G3: GME < EAHBP_G3_GME_FLOOR (40%) — rollback trigger
  Uncertainty band [40%, 60%): Paced Window takes over (Phase 2 §9.4)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from vit_pytorch.core.constants import (
    EAHBP_BLOCK_SIZE,
    EAHBP_G1_PRECISION_GAIN,
    EAHBP_G2_GME_FLOOR,
    EAHBP_G2_THROUGHPUT_GAIN,
    EAHBP_G3_GME_FLOOR,
    EAHBP_GLOBAL_POOL,
)


@dataclass
class EAHBPAttentionConfig:
    """Configuration for EAHBPAttention."""
    block_size: int = EAHBP_BLOCK_SIZE
    global_pool: int = EAHBP_GLOBAL_POOL
    dim: int = 256
    heads: int = 4
    dim_head: int = 64
    g1_precision_gain: float = EAHBP_G1_PRECISION_GAIN
    g2_throughput_gain: float = EAHBP_G2_THROUGHPUT_GAIN
    g2_gme_floor: float = EAHBP_G2_GME_FLOOR
    g3_gme_floor: float = EAHBP_G3_GME_FLOOR


class EAHBPAttention(nn.Module):
    """v1.3 STANDARD: EAHBP Attention (alpha-c-rev4).

    Replaces full N×N attention with block-local (b²) + global pool (g²):
        b = EAHBP_BLOCK_SIZE = 16
        g = EAHBP_GLOBAL_POOL = 8

    Theoretical FLOPs per head per token: (b² + g²) vs full N²
    For N=64: 320 vs 4096 = 7.8% = 92% reduction
    Design target: 62.5% reduction (more conservative)

    Gate signals are computed on the side and exposed via get_gate_signals():
        - GME (gradient magnitude estimate): ||grad|| / ||param||
        - throughput: tokens/sec
    """

    def __init__(
        self,
        config: Optional[EAHBPAttentionConfig] = None,
        base_attention: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = EAHBPAttentionConfig()
        self._config = config
        # Optional base attention (for QKV/RoPE reuse)
        self.base = base_attention
        # QKV projection (independent of base for skeleton)
        inner_dim = config.dim_head * config.heads
        self.to_q = nn.Linear(config.dim, inner_dim, bias=False)
        self.to_k = nn.Linear(config.dim, inner_dim, bias=False)
        self.to_v = nn.Linear(config.dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, config.dim)
        # Gate signal state (set externally by trainer)
        self._last_gme: float = 0.0
        self._last_throughput: float = 0.0
        self._last_precision_gain: float = 0.0

    def _block_local_attention(
        self, q: Tensor, k: Tensor, v: Tensor, b: int,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute block-local attention for each b×b block.

        Args:
            q, k, v: [B, H, N, d] where N is divisible by b
            b: block size
            attention_mask: [B, 1, 1, N] bool mask (True=valid, False=masked).
                Reshaped to [B, 1, n_blocks, b] to apply to per-block key positions.
        Returns:
            out: [B, H, N, d] (concatenation of block-local outputs)
        """
        B, H, N, d = q.shape
        n_blocks = N // b
        # Reshape to [B, H, n_blocks, b, d]
        q_b = q.view(B, H, n_blocks, b, d)
        k_b = k.view(B, H, n_blocks, b, d)
        v_b = v.view(B, H, n_blocks, b, d)
        # Block-local attention: [B, H, n_blocks, b, d]
        # Use scaled dot product within each block
        scale = 1.0 / (d ** 0.5)
        # q: bhNbd (N=n_blocks index, b=query-in-block index)
        # k: bhNbd (same n_blocks, b=key-in-block index — same block)
        # For per-block, q and k share the block index. Reshape to use distinct
        # subscripts: treat q as [B, H, N, b, d] and k as [B, H, b, N, d] via swap
        # then contract over d → [B, H, N, b_q, b_k]
        q_e = q_b.reshape(B, H, n_blocks * b, d)  # [B, H, n_blocks*b, d]
        k_e = k_b.reshape(B, H, n_blocks * b, d)  # [B, H, n_blocks*b, d]
        # Reshape back to per-block and use broadcast
        q_b2 = q_e.view(B, H, n_blocks, b, d)
        k_b2 = k_e.view(B, H, n_blocks, b, d)
        # attn[i, n, q, k] = q_b2[n, q, :] · k_b2[n, k, :]
        attn = torch.einsum("bhnqd,bhnkd->bhnqk", q_b2, k_b2) * scale

        # 🚀 Hot path 修复：注入 block-local mask
        # 形状: attention_mask [B, 1, 1, N] -> [B, 1, n_blocks, b] 应用到 key 轴
        if attention_mask is not None:
            mask_block = attention_mask.view(B, 1, n_blocks, b)  # per-block
            attn = attn.masked_fill(~mask_block.unsqueeze(-2), -1e4)

        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnqk,bhnkd->bhnqd", attn, v_b)
        return out.reshape(B, H, N, d)

    def _global_pool_attention(
        self, q: Tensor, k: Tensor, v: Tensor, g: int,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute attention with k, v reduced to g global tokens via mean pool.

        Args:
            attention_mask: [B, 1, 1, N] bool mask. Aggregated to [B, 1, 1, g]
                via OR (any-position-valid → global token valid).
        Returns:
            out: [B, H, N, d] (each query attends to g global tokens)
        """
        B, H, N, d = q.shape
        # Pool k, v down to g tokens via mean
        n_per_pool = N // g
        k_p = k.view(B, H, g, n_per_pool, d).mean(dim=3)
        v_p = v.view(B, H, g, n_per_pool, d).mean(dim=3)
        scale = 1.0 / (d ** 0.5)
        attn = torch.einsum("bhnd,bhgd->bhng", q, k_p) * scale

        # 🚀 Hot path 修复：注入 global pool mask
        # 形状: attention_mask [B, 1, 1, N] -> reshape & any() -> [B, 1, 1, g]
        if attention_mask is not None:
            mask_g = attention_mask.view(B, 1, 1, g, n_per_pool).any(dim=-1)
            attn = attn.masked_fill(~mask_g, -1e4)

        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhng,bhgd->bhnd", attn, v_p)
        return out

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass: split into block-local + global pool attention.

        Args:
            x: [B, N, D] input tokens
            attention_mask: [B, 1, 1, N] bool mask (True=valid, False=masked).
                Best-effort 注入到 block-local 和 global pool 两条路径。
                契约: 与 ManifoldNativeAttention 保持一致（与 FractalTransformerBlock 对齐）。
        Returns:
            out: [B, N, D] attention output
        """
        B, N, D = x.shape
        b = self._config.block_size
        g = self._config.global_pool
        # Project to Q, K, V
        q = self.to_q(x).view(B, N, self._config.heads, self._config.dim_head).transpose(1, 2)
        k = self.to_k(x).view(B, N, self._config.heads, self._config.dim_head).transpose(1, 2)
        v = self.to_v(x).view(B, N, self._config.heads, self._config.dim_head).transpose(1, 2)
        # Block-local attention (only if N divisible by b)
        if N % b == 0:
            local_out = self._block_local_attention(q, k, v, b, attention_mask=attention_mask)
        else:
            local_out = torch.zeros_like(q)
        # Global pool attention (only if N divisible by g)
        if N % g == 0:
            global_out = self._global_pool_attention(q, k, v, g, attention_mask=attention_mask)
        else:
            global_out = torch.zeros_like(q)
        # Combine: out = local + global (residual-style sum)
        combined = local_out + global_out
        # Reshape back to [B, N, D]
        combined = combined.transpose(1, 2).reshape(B, N, D)
        return self.to_out(combined)

    def estimate_flops(self, N: int) -> Tuple[int, int]:
        """Estimate FLOPs: EAHBP vs full attention.

        Returns:
            (eahbp_flops, full_flops)
        """
        b, g, d = self._config.block_size, self._config.global_pool, self._config.dim_head
        n_blocks = N // b
        n_pools = N // g
        # Block-local: per block, q·k.T is b×b·d, then attn·v is b×b·d
        # Total per block: 2 * b * b * d
        # n_blocks blocks per head, H heads, B batches
        local_flops = n_blocks * 2 * b * b * d
        # Global: q·k_p.T is N×g×d, attn·v_p is N×g×d
        # Total: 2 * N * g * d
        global_flops = 2 * N * g * d
        eahbp_flops = local_flops + global_flops
        # Full attention: 2 * N * N * d
        full_flops = 2 * N * N * d
        return eahbp_flops, full_flops

    def get_gate_signals(self) -> Dict[str, float]:
        """Return current gate signal values (set by trainer).

        Returns:
            {
                'gme': gradient magnitude estimate (set externally),
                'throughput': tokens/sec (set externally),
                'precision_gain': top-1 precision gain (set externally),
            }
        """
        return {
            "gme": self._last_gme,
            "throughput": self._last_throughput,
            "precision_gain": self._last_precision_gain,
        }

    def set_gate_signals(
        self,
        gme: Optional[float] = None,
        throughput: Optional[float] = None,
        precision_gain: Optional[float] = None,
    ) -> None:
        """Update gate signal values (called by trainer each step)."""
        if gme is not None:
            self._last_gme = gme
        if throughput is not None:
            self._last_throughput = throughput
        if precision_gain is not None:
            self._last_precision_gain = precision_gain

    def check_g1(self) -> bool:
        """G1: precision gain >= EAHBP_G1_PRECISION_GAIN.

        Returns True if gate passes.
        """
        return self._last_precision_gain >= self._config.g1_precision_gain

    def check_g2(self) -> bool:
        """G2: throughput gain >= EAHBP_G2_THROUGHPUT_GAIN AND GME >= EAHBP_G2_GME_FLOOR.

        Returns True if gate passes.
        """
        return (
            self._last_throughput >= self._config.g2_throughput_gain
            and self._last_gme >= self._config.g2_gme_floor
        )

    def check_g3_rollback(self) -> bool:
        """G3: GME < EAHBP_G3_GME_FLOOR triggers rollback.

        Returns True if rollback is required.
        """
        return self._last_gme < self._config.g3_gme_floor
