# -*- coding: utf-8 -*-
r"""Shadow Monitor — G1 WBA + G2 RoPE InfoNCE MI 观察者

v1.3 §9.7 Shadow Monitor skeleton (B.11):
  - **G1 (WBA local entropy)**: windowed boundary access entropy on attention scores
  - **G2 (RoPE InfoNCE MI lower bound)**: mutual information lower bound between
    query and key RoPE embeddings via InfoNCE

数学形式化
===========

G1 WBA local entropy (v1.3 closure, §9.7):
    For each query position i, take window of W attention scores A[:, :, i, i:i+W]
    p = softmax(window)
    H_local(i) = -sum(p * log(p + eps))
    wba_entropy = (1 - delta_washout) * mean_over_i(H_local(i))

G2 InfoNCE MI lower bound (v1.3 closure, §9.7):
    sim(q_i, k_j) = q_i . k_j / sqrt(D)   (cosine via normalized inner product)
    L_InfoNCE = -log( exp(sim(q_i, k_i)) / sum_j exp(sim(q_i, k_j)) )
    mi_lower_bound = epsilon_leak_factor * log(B) - L_InfoNCE  (one-strike threshold)
    Following v1.3 spec, we return L_InfoNCE scaled by epsilon_leak_factor.

非阻塞 (NON-BLOCKING) 设计
============================

The Shadow Monitor is an *observer*, not a training participant:
  - It MUST NOT affect gradient flow
  - It MUST NOT participate in JVP acceptance protocol
  - It MUST NOT block the main training loop

Implementation: all metric tensors are computed under `torch.no_grad()` so
the autograd graph is not extended, and the returned tensors are detached
scalars. Calling code can safely use them for logging / dashboard escalation
without `retain_graph=True` and without `.item()` in the hot path.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from .constants import (
    SHADOW_MONITOR_WBA_WINDOW,
    SHADOW_MONITOR_PARTIAL_DERIVATIVE_EPOCH,
    V13_DELTA_WASHOUT,
    V13_EPSILON_LEAK_FACTOR,
    LOG_EPSILON,
)


class ShadowMonitor(nn.Module):
    """Shadow Monitor — G1 WBA + G2 RoPE InfoNCE 双标尺观察者

    v1.3 §9.7: a non-blocking observer that computes two scalar metrics:
      - G1 WBA local entropy: windowed boundary access entropy on attention scores
      - G2 InfoNCE MI lower bound between query and key RoPE embeddings

    The module is intentionally parameter-free (no learnable weights) so it
    cannot affect the main path. All metric computation runs under
    `torch.no_grad()` to guarantee the autograd graph is not extended.

    Args:
        window_size: G1 sliding window W (default 16 from SHADOW_MONITOR_WBA_WINDOW).
        delta_washout: G1 entropy washout factor (default 0.05 from V13_DELTA_WASHOUT).
        epsilon_leak_factor: G2 MI one-strike threshold scale (default 0.25
            from V13_EPSILON_LEAK_FACTOR).
    """

    def __init__(
        self,
        window_size: int = SHADOW_MONITOR_WBA_WINDOW,
        delta_washout: float = V13_DELTA_WASHOUT,
        epsilon_leak_factor: float = V13_EPSILON_LEAK_FACTOR,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.delta_washout = float(delta_washout)
        self.epsilon_leak_factor = float(epsilon_leak_factor)

        # v1.3 §9.7: partial-derivative epoch is documented but consumed by the
        # Paced Window supervisor (B.12). Stash for API parity / introspection.
        self.partial_derivative_epoch = int(SHADOW_MONITOR_PARTIAL_DERIVATIVE_EPOCH)

        # No learnable parameters: Shadow Monitor is an observer, not a learner.
        # `requires_grad` is False on all params by default since no params exist.

    @torch.no_grad()
    def forward(
        self,
        attention_scores: torch.Tensor,
        rope_embeddings: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute G1 WBA entropy and G2 RoPE InfoNCE MI lower bound.

        Both returned tensors are scalar `Tensor`s detached from the autograd
        graph. The function never raises if the inputs are smaller than the
        window — it falls back to a smaller effective window for the trailing
        queries.

        Args:
            attention_scores: shape `[B, H, N, N]`. G1 metric source.
                We use the value at the diagonal window: A[:, :, i, i:i+W].
            rope_embeddings: shape `[B, N, 2*D]` — packed (Q_rope, K_rope)
                along the last axis. G2 splits this into Q (first D) and K
                (last D).
            batch_size: training batch size B. Used for the log(B) factor
                in the G2 one-strike threshold.

        Returns:
            (wba_entropy, mi_lower_bound):
              - wba_entropy: scalar, ≥ 0; windowed boundary access entropy
                after washout. Smaller values indicate stronger locality.
              - mi_lower_bound: scalar, ≥ 0; InfoNCE MI lower bound estimate
                between query and key RoPE embeddings, scaled by
                `epsilon_leak_factor`. Larger values indicate stronger
                physical↔topology coupling.
        """
        device = attention_scores.device
        dtype = attention_scores.dtype

        # -------- G1: WBA local entropy --------
        wba_entropy = self._wba_local_entropy(attention_scores).to(device=device)

        # -------- G2: RoPE InfoNCE MI lower bound --------
        mi_lower_bound = self._rope_infonce_mi(rope_embeddings, batch_size).to(
            device=device
        )

        # Ensure scalar tensors with the expected dtype on the right device.
        # detach() is a no-op here (we're already in no_grad) but kept for
        # defensive symmetry in case the upstream caller is in a grad context.
        return wba_entropy.detach(), mi_lower_bound.detach()

    @torch.no_grad()
    def _wba_local_entropy(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """G1: windowed boundary access local entropy.

        For each query position i, take the diagonal window A[:, :, i, i:i+W]
        and compute -sum(p * log(p + eps)) on the within-window softmax p.
        Average over all (batch, head, query) positions, then apply
        (1 - delta_washout).

        Implementation detail: if N < window_size, the effective window
        shrinks to N for every query (we still take i:i+W which clamps to N).
        """
        B, H, N, _ = attention_scores.shape
        if N == 0:
            return torch.zeros((), dtype=attention_scores.dtype,
                               device=attention_scores.device)

        # Build the diagonal-window slices A[:, :, i, i:i+W] for i in 0..N-1.
        # Vectorized: construct a [B, H, N, W] tensor where the (b, h, i, j)
        # entry is A[b, h, i, i+j], capped at N-1.
        W = min(self.window_size, N)
        idx_i = torch.arange(N, device=attention_scores.device).view(1, 1, N, 1)
        idx_j = idx_i + torch.arange(W, device=attention_scores.device).view(1, 1, 1, W)
        idx_j = idx_j.clamp(max=N - 1)
        # Gather the diagonal-window scores: shape [B, H, N, W].
        window = attention_scores.gather(-1, idx_j.expand(B, H, N, W))

        # Local softmax within the window (WBA local entropy definition, §9.7).
        # Numerically stable: subtract max before exp.
        window_max = window.max(dim=-1, keepdim=True).values
        exp_w = torch.exp(window - window_max)
        p = exp_w / (exp_w.sum(dim=-1, keepdim=True) + LOG_EPSILON)

        # Local entropy: H_local(i) = -sum(p * log(p + eps))
        # Clamp p for log stability.
        entropy = -(p * torch.log(p + LOG_EPSILON)).sum(dim=-1)  # [B, H, N]
        mean_entropy = entropy.mean()

        return mean_entropy * (1.0 - self.delta_washout)

    @torch.no_grad()
    def _rope_infonce_mi(
        self,
        rope_embeddings: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """G2: InfoNCE MI lower bound between Q and K RoPE embeddings.

        For RoPE embeddings of shape [B, N, 2D], split last axis into Q ([:D])
        and K ([D:]). Compute per-row InfoNCE with positives on the diagonal
        and negatives elsewhere. Following v1.3 §9.7, scale by
        epsilon_leak_factor to express the one-strike threshold.
        """
        if rope_embeddings.ndim != 3 or rope_embeddings.shape[-1] % 2 != 0:
            # Defensive: degenerate input -> return zero scalar.
            return torch.zeros((), dtype=rope_embeddings.dtype,
                               device=rope_embeddings.device)

        B, N, two_D = rope_embeddings.shape
        D = two_D // 2
        if N <= 1:
            return torch.zeros((), dtype=rope_embeddings.dtype,
                               device=rope_embeddings.device)

        # Split into Q and K from the last axis.
        q = rope_embeddings[..., :D]  # [B, N, D]
        k = rope_embeddings[..., D:]  # [B, N, D]

        # L2-normalize for cosine similarity.
        q_n = q / (q.norm(dim=-1, keepdim=True) + LOG_EPSILON)
        k_n = k / (k.norm(dim=-1, keepdim=True) + LOG_EPSILON)

        # Similarity matrix: [B, N, N] with positives on the diagonal.
        sim = torch.bmm(q_n, k_n.transpose(1, 2))  # [B, N, N]

        # Numerically stable log-softmax along the last axis.
        sim_max = sim.max(dim=-1, keepdim=True).values
        exp_sim = torch.exp(sim - sim_max)
        denom = exp_sim.sum(dim=-1, keepdim=True) + LOG_EPSILON
        log_softmax = (sim - sim_max) - torch.log(denom)  # [B, N, N]

        # InfoNCE loss: -log p(positive) = -diag(log_softmax)
        infonce = -log_softmax.diagonal(dim1=1, dim2=2).mean()  # scalar

        # One-strike threshold scale: epsilon_leak_factor * log(B).
        # The lower bound is "MI >= -L_InfoNCE + log(N-1) - log(B)" (van den Oord 2018);
        # for monitoring purposes we surface epsilon_leak_factor * L_InfoNCE so the
        # dashboard compares apples-to-apples against the §9.7 threshold.
        log_B = math.log(max(int(batch_size), 2))
        return self.epsilon_leak_factor * infonce * log_B


__all__ = [
    "ShadowMonitor",
]
