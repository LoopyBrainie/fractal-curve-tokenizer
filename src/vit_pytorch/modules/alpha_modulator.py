# -*- coding: utf-8 -*-
"""
AlphaModulator (R7) - 深度相关的 Logit 缩放调制器

数学形式化
==========

Alpha 调制器根据当前 Hilbert 归一化索引 H / H_max 调整分类 logits 的尺度。

    alpha_bounded = 2 * tanh(alpha_raw / 2)    ∈ (-2, 2)
    rho = 1 + alpha_bounded * (H / H_max - 0.5)
    rho = clamp(rho, min=eps, max=2.0)         防御性钳制
    M = rho * logit_scale

钩子位置: 在分类头前:
    pooled = ...                                 # [B, D]
    logits = mlp_head(pooled) * M.unsqueeze(-1)  # [B, num_classes]

H / H_max 来自 splitter 的归一化 Hilbert 索引。当 alpha_raw=0 时，alpha_bounded=0，
rho=1，M=logit_scale，因此 logits = mlp_head(pooled) * logit_scale（与 R7 之前一致）。
H 默认约定: H ∈ [0, H_max]，H_max = self.max_level（H 是最大深度归一化后的相对量）。

可学习参数: alpha_raw (1,) 初始化为 0
- alpha=0 → 恒等映射 (M = logit_scale)
- alpha>0 → 浅层放大，深层缩小
- alpha<0 → 浅层缩小，深层放大

注: 此模块为 R7 的 A 设计 (UNCONDITIONAL SHIP)，默认启用，无 kill-switch。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AlphaModulator(nn.Module):
    """深度相关的 logit 缩放调制器 (R7 - 方案 A)

    Args:
        logit_scale: 基础 logit 缩放因子 (H 是最大深度归一化后的相对量)
        eps: 防御性下界，确保 rho > 0
    """

    def __init__(self, logit_scale: float, eps: float = 1e-6) -> None:
        super().__init__()
        self.logit_scale = float(logit_scale)
        self.eps = float(eps)
        # R7: 1 个可学习参数，初始化为 0 → alpha_bounded=0 → rho=1 (恒等)
        self.alpha_raw = nn.Parameter(torch.zeros(1))

    def forward(self, H: torch.Tensor, H_max: float) -> torch.Tensor:
        """计算调制因子 M = rho * logit_scale

        Args:
            H: 当前归一化 Hilbert 索引 [B] 或标量张量
            H_max: 最大 Hilbert 归一化索引 (来自 model.max_level)

        Returns:
            M: [B] 调制因子，用于 mlp_head(pooled) * M.unsqueeze(-1)
        """
        # R7: alpha_bounded = 2 * tanh(alpha_raw / 2) ∈ (-2, 2)
        alpha_bounded = 2.0 * torch.tanh(self.alpha_raw / 2.0)

        # R7: rho = 1 + alpha_bounded * (H / H_max - 0.5)
        H_safe = H.detach() if H.requires_grad else H  # STE-safe: 不追溯到 splitter
        ratio = H_safe / max(float(H_max), 1.0)
        rho = 1.0 + alpha_bounded * (ratio - 0.5)

        # R7 (Q4): 防御性 clamp [eps, 2.0]
        rho = rho.clamp(min=self.eps, max=2.0)

        # M = rho * logit_scale
        M = rho * self.logit_scale
        return M

    def extra_repr(self) -> str:
        return f"logit_scale={self.logit_scale}, eps={self.eps}, alpha_raw={self.alpha_raw.item():.6f}"
