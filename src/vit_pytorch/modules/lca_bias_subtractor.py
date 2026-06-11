# -*- coding: utf-8 -*-
"""
LCABiasSubtractor (R7) - 可学习的 LCA 偏置减法器

数学形式化
==========

LCA 偏置 (来自 LCA Hilbert Bias) 在不同深度的 token 对之间提供拓扑先验。
R7 的 Beta-C 设计允许模型**学习性地减少**部分 LCA 偏置，使注意力分布更灵活。

    lambda_bounded = 2 * tanh(lambda_raw / 2)    ∈ (-2, 2)
    B_sub = bias_table[depths[i], depths[j]]      ∈ R (可学习)
    attn_mask = attn_mask + (B_LCA - lambda_bounded * B_sub)  (when enabled)

    # 静默 (kill-switch):
    attn_mask = attn_mask                          (when disabled)

参数:
    bias_table: [max_depth+1, max_depth+1] = [65, 65] 共享 (per Q3)
        总参数量: 65 * 65 = 4225
    lambda_raw: (1,) 标量参数，初始化为 0

钩子位置: 在 attention 层 (ManifoldNativeAttention) 计算 attention score 时。

kill-switch:
    self.enabled 默认为 True (T14-gated, default-ON)。
    外部可通过设置 self.enabled = False 关闭。
    在 forward() 中根据 self.enabled 分支:
        - enabled=True: 应用 bias 减法
        - enabled=False: passthrough attn_mask (无任何修改)

bit-exact 保证:
    - lambda_raw=0 → lambda_bounded=0 → 减法项为 0 → 退化为原始 attn_mask
    - bias_table=0 → 同上
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LCABiasSubtractor(nn.Module):
    """R7 方案 Beta-C: 可学习 LCA 偏置减法器 (T14-gated, default-ON)

    Args:
        max_depth: 最大四叉树深度 (默认 64, 对应 bias_table 索引 [0, max_depth])
        enabled: 是否启用偏置减法 (T14-gated, default ON with kill-switch)
    """

    def __init__(self, max_depth: int = 64, enabled: bool = True) -> None:
        super().__init__()
        # R7: bias_table 大小为 (max_depth+1, max_depth+1) = (65, 65)
        # 索引 0..max_depth 都合法 (depth=0 是根节点)
        table_size = max_depth + 1
        self.max_depth = max_depth
        # 4225 个共享参数，初始化为 0 (per Q3 shared across heads)
        self.bias_table = nn.Parameter(torch.zeros(table_size, table_size))
        # 1 个标量参数，初始化为 0
        self.lambda_raw = nn.Parameter(torch.zeros(1))
        # kill-switch (T14-gated)
        self.enabled = bool(enabled)

    def forward(
        self,
        attn_mask: torch.Tensor,
        depths: torch.Tensor,
        B_LCA: torch.Tensor,
    ) -> torch.Tensor:
        """应用可学习的 LCA 偏置减法

        Args:
            attn_mask: 原始 attention mask [..., N, N]
            depths: 深度索引 [B, N] (int/long)
            B_LCA: LCA 偏置矩阵 [B, N, N] (从 levels_info.get_lca_matrix() 获取)

        Returns:
            修改后的 attn_mask:
                - enabled: attn_mask + (B_LCA - lambda_bounded * B_sub)
                - disabled: attn_mask (passthrough)
        """
        # kill-switch: passthrough
        if not self.enabled:
            return attn_mask

        # R7: lambda_bounded = 2 * tanh(lambda_raw / 2) ∈ (-2, 2)
        lambda_bounded = 2.0 * torch.tanh(self.lambda_raw / 2.0)

        # R7: B_sub = bias_table[depths[i], depths[j]]
        # 边界 clamp: depths 可能在 [0, max_depth] 之外 (padding -1, max+overhead)
        depths_clamped = depths.clamp(min=0, max=self.max_depth)
        # 索引 [B, N, N] = bias_table[depths_i, depths_j]
        # depths_clamped: [B, N] → [B, N, 1] 和 [B, 1, N]
        B_sub = self.bias_table[depths_clamped.unsqueeze(2), depths_clamped.unsqueeze(1)]

        # R7: attn_mask = attn_mask + (B_LCA - lambda_bounded * B_sub)
        return attn_mask + (B_LCA - lambda_bounded * B_sub)

    def extra_repr(self) -> str:
        return (
            f"max_depth={self.max_depth}, enabled={self.enabled}, "
            f"lambda_raw={self.lambda_raw.item():.6f}, "
            f"bias_table.shape={tuple(self.bias_table.shape)}"
        )
