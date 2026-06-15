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
            attn_mask: 原始 attention mask。
                当前唯一调用方 (FractalCurveViT._create_attention_mask) 输出
                bool [B, 1, 1, S] (True=valid, S=N+1 含 CLS, CLS 在 index 0)。
                也接受已对齐的 float [B, 1, S, S] (兼容未来 caller)。
            depths: 深度索引 [B, N] (int/long)
            B_LCA: LCA 偏置矩阵 [B, N, N] (从 levels_info.get_lca_matrix() 获取)

        Returns:
            修改后的 attn_mask (float [B, 1, S, S], 与 ManifoldNativeAttention
            hot path 的 attn 形状严格匹配):
                - enabled=True + bool input: 门控偏置 (valid×valid = LCA bias,
                  either invalid = -1e4 硬屏蔽, CLS row/col = 0)
                - enabled=True + float input: bias_pre 左/上 pad 后直接相加
                - enabled=False: passthrough (原 attn_mask 原样返回)
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

        # pre-CLS 偏置: [B, N, N]
        bias_pre = B_LCA - lambda_bounded * B_sub

        # R7-I170.3-MASK 形状归一化:
        # 上游 _create_attention_mask 输出 bool [B, 1, 1, S] (True=valid, S 含
        # CLS, CLS 在 index 0)。若直接 attn_mask + bias_pre, PyTorch right-
        # aligned broadcast 会把 3D bias_pre 左侧 pad 为 [1, B, S, S], 与 4D
        # [B, 1, 1, S] 对齐后产生 [B, B, S, S] — 第二个 B 是 batch self-cross,
        # 侵占 head 轴, 触发 ManifoldNativeAttention 严格校验 ValueError。
        # 正确做法: 用 valid_2d 门控 LCA bias 本身 (而非引入 +1.0 软奖励),
        # invalid 端点写入 -1e4 (与 ManifoldNativeAttention bool 路径的
        # masked_fill 值一致, 见 manifold_attention.py:375, 379)。
        # 注意: bias_pre 已是 [B, S, S] (levels_info 由 _apply_position_and_cls
        # 预拼接了 CLS, get_lca_matrix 自然产生 [B, S, S]; CLS row/col 因
        # depth=0 被 LCA 计算中的 valid_mask 排除, 值为 0), 无需再 pad。
        if (attn_mask.dim() == 4
                and attn_mask.dtype == torch.bool
                and attn_mask.shape[1] == 1
                and attn_mask.shape[2] == 1):
            # valid: [B, S] float (1.0=valid, 0.0=invalid, S=attn.shape[-1])
            valid = attn_mask.squeeze(1).squeeze(1).float()
            # valid_2d: [B, S, S] float (1.0 iff 两端均 valid)
            valid_2d = valid.unsqueeze(-1) * valid.unsqueeze(-2)
            # 门控偏置: valid pair → bias (无 +1.0), either invalid → -1e4
            return (valid_2d * bias_pre
                    + (1.0 - valid_2d) * (-1e4)).unsqueeze(1)  # [B, 1, S, S]

        # 兼容路径: 浮点 attn_mask 已对齐到 [B, 1, S, S] (未来 caller 直接构造)。
        # 此处不重做形状校验, 假设 caller 严格遵守契约。
        return attn_mask + bias_pre

    def extra_repr(self) -> str:
        return (
            f"max_depth={self.max_depth}, enabled={self.enabled}, "
            f"lambda_raw={self.lambda_raw.item():.6f}, "
            f"bias_table.shape={tuple(self.bias_table.shape)}"
        )
