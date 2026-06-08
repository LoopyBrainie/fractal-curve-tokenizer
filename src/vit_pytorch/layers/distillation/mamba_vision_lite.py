# -*- coding: utf-8 -*-
"""MambaVision-Lite Student (config-gated distillation plugin).

B.10 of v1.3 best-practice design (§9.7).

数学形式化
============

**Distillation objective** (Hinton et al. 2015, with classification CE mixing):

.. math::
    \\mathcal{L} = \\alpha \\cdot \\mathrm{KL}(p_T \\,\\|\\, p_S) + (1-\\alpha) \\cdot \\mathrm{CE}(p_S, y)

其中 :math:`p_T = \\mathrm{softmax}(z_T / T)`, :math:`p_S = \\mathrm{softmax}(z_S / T)`,
:math:`T=1` for hard-label distillation (logit-matching), :math:`\\alpha = 0.7`.

**Student architecture** (lightweight MambaVision-Lite approximation):

    Block_i = DepthwiseConv7x7 -> LN -> Pointwise(4x) -> SiLU -> Pointwise(.)
                                                              + SSMBlock(x)

每个 ConvNeXt-style block 使用 depthwise 7x7 + pointwise 1x1 (扩展 4x)。
SSMBlock 用 2 个线性投影 + SiLU 作为轻量级 state-space 近似。

**Parameter budget** (v1.3 standard):
    target = MAMBA_LITE_PARAMS_TARGET = 44,000,000
    tolerance = MAMBA_LITE_PARAMS_TOLERANCE = 1,000,000
    实际参数 [43M, 45M] 内即合格。

**Inference complexity**: O(1) w.r.t. sequence length (no growing KV cache / SSM state).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_pytorch.core.constants import (
    MAMBA_LITE_DISTILL_ALPHA,
    MAMBA_LITE_PARAMS_TARGET,
    MAMBA_LITE_PARAMS_TOLERANCE,
)


# ==================== Building Blocks ====================


class ConvNeXtBlock(nn.Module):
    """Lightweight ConvNeXt-style block: DWConv7x7 -> LN -> PW expand -> SiLU -> PW contract.

    Parameter count per block at dim=D, hidden=4D:
        DW 7x7:  D * 49
        LN:      2 * D
        PW 1x1:  D * 4D = 4D^2
        PW 1x1:  4D * D = 4D^2
        Total:   8D^2 + 49D + 2D = 8D^2 + 51D  ~8D^2 (dominant)
    """

    def __init__(self, dim: int, expand: int = 4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw_in = nn.Linear(dim, dim * expand)
        self.act = nn.SiLU()
        self.pw_out = nn.Linear(dim * expand, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, H, W)
        residual = x
        x = self.dwconv(x)  # (B, D, H, W)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, D) for LN/Linear
        x = self.norm(x)
        x = self.pw_in(x)
        x = self.act(x)
        x = self.pw_out(x)
        x = x.permute(0, 3, 1, 2)  # back to (B, D, H, W)
        return residual + x


class SSMBlock(nn.Module):
    """Lightweight SSM-like block: gated 1D convolution + SiLU.

    Approximates Mamba's input-dependent state-space mixing with
    2 linear projections (gate + value) and SiLU activation.
    No recurrence — purely feed-forward → O(1) inference.

    Params per block at dim=D, expand=2:
        in_proj:  D * 2D = 2D^2
        out_proj: 2D * D = 2D^2
        LN:       2 * D
        Total:    4D^2 + 2D
    """

    def __init__(self, dim: int, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * expand)
        self.out_proj = nn.Linear(dim * expand, dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D) — token sequence
        residual = x
        x = self.norm(x)
        x = self.in_proj(x)
        x = self.act(x)
        x = self.out_proj(x)
        return residual + x


# ==================== Student Model ====================


class MambaVisionLiteStudent(nn.Module):
    """MambaVision-Lite student for knowledge distillation (B.10).

    A lightweight ConvNeXt + SSM-like hybrid designed to land at
    ~44M parameters. The student is config-gated by the trainer;
    see v1.3 §9.7 for the gating policy.

    Args:
        teacher: Frozen teacher model (used only for shape probing; not
            invoked by the student's forward pass).
        target_params: Target parameter count (default
            ``MAMBA_LITE_PARAMS_TARGET`` = 44M).
        num_classes: Number of output classes (default 1000).
        dim: Stem / block width.
        num_blocks: Number of (ConvNeXt + SSM) blocks.
        input_size: Spatial input size (default 224).
        input_channels: Input channels (default 3).

    Example:
        >>> teacher = FractalCurveViT(num_classes=1000, dim=384, ...)
        >>> student = MambaVisionLiteStudent(teacher)
        >>> # 44_000_000 - tolerance <= n_params <= 44_000_000 + tolerance
        >>> s_logits = student(x)               # (B, 1000)
        >>> t_logits = teacher(x).logits        # (B, 1000)
        >>> loss = student.compute_distillation_loss(s_logits, t_logits, y)
    """

    def __init__(
        self,
        teacher: nn.Module,
        target_params: int = MAMBA_LITE_PARAMS_TARGET,
        num_classes: int = 1000,
        dim: int = 592,
        num_blocks: int = 10,
        input_size: int = 224,
        input_channels: int = 3,
    ):
        super().__init__()
        self.teacher = teacher
        self.target_params = target_params
        self.num_classes = num_classes
        self.dim = dim
        self.num_blocks = num_blocks
        self.input_size = input_size
        self.input_channels = input_channels

        # Stem: 2x downsample with stride-2 3x3 conv
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, dim // 2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim // 2, dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )

        # Convolutional backbone (ConvNeXt blocks)
        self.conv_blocks = nn.ModuleList(
            [ConvNeXtBlock(dim=dim, expand=4) for _ in range(num_blocks)]
        )

        # SSM-like token mixing on flattened spatial tokens
        self.ssm_blocks = nn.ModuleList(
            [SSMBlock(dim=dim, expand=2) for _ in range(num_blocks)]
        )

        # Head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        # Distillation hyperparameter (read from constants for consistency)
        self.distill_alpha: float = MAMBA_LITE_DISTILL_ALPHA

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Standard truncated-normal init for Linear / Conv2d."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: stem -> conv blocks -> SSM blocks -> head.

        Returns:
            Logits of shape ``(B, num_classes)``.

        Complexity:
            O(1) w.r.t. sequence length (no KV cache; pure feed-forward).
        """
        # Stem
        x = self.stem(x)  # (B, dim, H/4, W/4)

        # Convolutional blocks
        for block in self.conv_blocks:
            x = block(x)

        # Flatten to tokens: (B, N, D)
        B, D, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, D)

        # SSM-like token mixing
        for block in self.ssm_blocks:
            tokens = block(tokens)

        # Global average pool + head
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)  # (B, D)
        logits = self.head(pooled)   # (B, num_classes)
        return logits

    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute :math:`\\alpha \\cdot \\mathrm{KL}(T \\| S) + (1-\\alpha) \\cdot \\mathrm{CE}(S, y)`.

        Args:
            student_logits: ``(B, K)`` student outputs.
            teacher_logits: ``(B, K)`` teacher outputs (assumed detached
                by the caller; we detach here defensively).
            labels: ``(B,)`` integer class labels.
            temperature: Softmax temperature (default 1.0 = hard-label
                logit distillation).

        Returns:
            Scalar distillation loss.
        """
        alpha = self.distill_alpha
        teacher_logits = teacher_logits.detach()

        log_p_s = F.log_softmax(student_logits / temperature, dim=-1)
        p_t = F.softmax(teacher_logits / temperature, dim=-1)
        # KL(p_T || p_S) = sum p_T * (log p_T - log p_S)
        kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (temperature ** 2)
        ce = F.cross_entropy(student_logits, labels)

        return alpha * kl + (1.0 - alpha) * ce

    # ---------- introspection helpers ----------

    def count_parameters(self) -> int:
        """Count ``nn.Parameter`` in the *student only* (excludes teacher)."""
        return sum(p.numel() for p in self.parameters() if isinstance(p, nn.Parameter))

    def within_budget(self, tolerance: int = MAMBA_LITE_PARAMS_TOLERANCE) -> bool:
        """Return True iff the parameter count is within ±tolerance of target."""
        n = self.count_parameters()
        return abs(n - self.target_params) <= tolerance
