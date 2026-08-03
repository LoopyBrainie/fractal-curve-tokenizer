# -*- coding: utf-8 -*-
"""
形状稳定器 (Shape Stabilizer)

将动态 Token 数量 K 映射到固定桶中，解决 torch.compile 的形状动态性问题。

数学形式
========
    B = B_low ∪ B_high
      = {128, 256, 512} ∪ {1024, 2048, 4096, 8192}
      = {128, 256, 512, 1024, 2048, 4096, 8192}

    K ∈ [8, 8192] → K_bucket ∈ B

形状空间压缩效果
    |S|: 8185 → 7 (从 O(N) 降至 O(1))

参考: IMPROVEMENT_PLAN.md 中的 "Bucketing Strategy" 方案
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


class ShapeStabilizer:
    """
    形状稳定器：将动态 K 映射到固定桶中（支持非线性桶）

    设计原理：
    - 低频区（小形状，Level 0-2）：步长小，减少计算浪费
    - 高频区（大形状，Level 3+）：步长指数增长，适应深度增加
    """

    def __init__(
        self,
        low_frequency_buckets: Tuple[int, ...] = (64, 128, 256, 512),
        high_frequency_buckets: Tuple[int, ...] = (1024, 2048),
    ):
        """
        初始化非线性桶集合。

        Args:
            low_frequency_buckets: 低频区桶集合（小形状）
            high_frequency_buckets: 高频区桶集合（大形状）
        """
        self.low_freq = torch.tensor(low_frequency_buckets, dtype=torch.long)
        self.high_freq = torch.tensor(high_frequency_buckets, dtype=torch.long)
        self.buckets = torch.cat([self.low_freq, self.high_freq])  # [7]
        self.max_tokens = int(self.buckets[-1])

    def get_nearest_bucket(self, k: int) -> int:
        """
        获取最小的 b ∈ B，使得 b >= k

        Args:
            k: 原始 token 数量

        Returns:
            最近的桶大小
        """
        for b in self.buckets.tolist():
            if b >= k:
                return b
        return int(self.buckets[-1])

    def get_nearest_bucket_tensor(self, k: torch.Tensor) -> torch.Tensor:
        """
        Tensor 版本：用于 torch.compile 兼容的图内计算。

        关键改进：
        1. 使用 torch.where 避免 .item() 调用
        2. 处理 k > max_tokens 的边界情况（全 False 时回退到最大桶）

        Args:
            k: 原始 token 数量 [B] 或 scalar

        Returns:
            最近的桶大小 [...]
        """
        buckets = self.buckets.to(k.device)  # [m]
        k_expanded = k.unsqueeze(-1)  # [..., 1]
        valid_mask = buckets >= k_expanded  # [..., m]

        # 使用 argmax 找到第一个 True 的位置
        indices = valid_mask.long().argmax(dim=-1)  # [...]

        # 额外保护：如果全为 False，argmax 返回 0，强制修正为最后一个桶
        has_valid = valid_mask.any(dim=-1)  # [...] - 是否有有效桶
        final_indices = torch.where(
            has_valid,
            indices,
            torch.full_like(indices, len(self.buckets) - 1),  # 回退到最大桶
        )

        return buckets[final_indices]  # [...]

    def pad_to_bucket(
        self,
        tokens: torch.Tensor,
        levels: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将动态形状 padding 到最近的桶。

        数学变换：
            tokens ∈ R^{B × K × D} → tokens' ∈ R^{B × K_bucket × D}
            levels ∈ Z^{B × K}    → levels' ∈ Z^{B × K_bucket}
            lengths ∈ Z^B          → lengths（不变）

        Args:
            tokens: [B, K, D] 输入 tokens
            levels: [B, K] 对应深度
            lengths: [B] 有效长度

        Returns:
            (tokens_padded, levels_padded, lengths)
        """
        B, K, D = tokens.shape

        # === Python 标量层早返：保留 is 身份，避免任何 tensor 分配 ===
        # K 是 Python int（tokens.shape[1]，零同步）
        # lengths.max().item() 是本次调用唯一的 sync 点；
        # 早返命中时整个 tensor 路径完全跳过。
        # 在典型工作流中 lengths 是 CPU int64 tensor（B 小），同步代价可忽略。
        k_bucket = self.get_nearest_bucket(int(lengths.max()))  # Python int

        if k_bucket == K:
            # 既不需要 padding 也不需要 truncation：直接返回原引用
            return tokens, levels, lengths

        # === 回落：原 tensor 路径（torch.compile 完全兼容） ===
        # Tensor 版本：K_bucket 基于 lengths.max() 计算
        # 避免 .item()，保持图编译兼容
        actual_max_len = lengths.max()  # Tensor
        k_bucket_tensor = self.get_nearest_bucket_tensor(actual_max_len)  # Tensor

        # 计算需要 padding 的长度（保持为 Tensor）
        pad_len = k_bucket_tensor - K  # Tensor

        # torch.cat 方式：完全图兼容
        # 只有当 pad_len > 0 时才创建 padding tensor
        # 使用 max(pad_len, 0) 确保 size 不为负
        safe_pad_len = torch.clamp(pad_len, min=0)
        zeros_tokens = torch.zeros((B, safe_pad_len, D), dtype=tokens.dtype, device=tokens.device)
        neg_one_levels = torch.full((B, safe_pad_len), -1, dtype=levels.dtype, device=levels.device)

        # 拼接 padding（如果 pad_len <= 0，safe_pad_len = 0，cat 结果与原相同）
        tokens_padded = torch.cat([tokens, zeros_tokens], dim=1)
        levels_padded = torch.cat([levels, neg_one_levels], dim=1)

        # Truncation：如果 K > k_bucket，沿 dim=1 截断
        tokens_padded = tokens_padded[:, :k_bucket_tensor, :]
        levels_padded = levels_padded[:, :k_bucket_tensor]

        return tokens_padded, levels_padded, lengths

    def _pad_attention_mask(
        self,
        mask: torch.Tensor,
        lengths: torch.Tensor,
        k_bucket: int,
    ) -> torch.Tensor:
        """
        Padding Attention Mask，确保 Padding 区域被完全屏蔽。

        关键改进：使用布尔类型的 attn_mask 替代 -inf
        原因：在 AMP（混合精度）下，-inf 可能导致数值溢出
              而 SDPA 对布尔 mask 的优化更好

        Args:
            mask: [B, 1, K, K] 或 None
            lengths: [B] 有效长度
            k_bucket: 目标桶大小

        Returns:
            mask_padded: [B, 1, K_bucket, K_bucket] 布尔类型
        """
        K = mask.shape[-1] if mask is not None else lengths.shape[0]
        pad_len = k_bucket - K

        if mask is not None:
            # 扩展原 mask 并拼接 padding mask
            mask_padded = F.pad(mask, (0, pad_len, 0, pad_len), value=False)  # [B, 1, K_bucket, K_bucket]

            # 创建 4D padding mask：[B, 1, K_bucket, K_bucket]
            # position [b, 0, j, k] is True (masked) if j >= lengths[b] or k >= lengths[b]
            j_indices = torch.arange(k_bucket, device=lengths.device).unsqueeze(0).unsqueeze(0)  # [1, 1, K_bucket]
            k_indices = torch.arange(k_bucket, device=lengths.device).unsqueeze(0).unsqueeze(2)  # [1, K_bucket, 1]
            lengths_3d = lengths.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]

            padding_mask_4d = (j_indices >= lengths_3d) | (k_indices >= lengths_3d)  # [B, K_bucket, K_bucket] broadcast
            padding_mask_4d = padding_mask_4d.unsqueeze(1)  # [B, 1, K_bucket, K_bucket]

            # 结合 padding mask（padding 位置设为 True）
            combined_mask = mask_padded | padding_mask_4d
            return combined_mask
        else:
            # 无原 mask 时，直接返回 padding mask
            j_indices = torch.arange(k_bucket, device=lengths.device).unsqueeze(0).unsqueeze(0)
            k_indices = torch.arange(k_bucket, device=lengths.device).unsqueeze(0).unsqueeze(2)
            lengths_3d = lengths.unsqueeze(1).unsqueeze(2)

            padding_mask_4d = (j_indices >= lengths_3d) | (k_indices >= lengths_3d)
            return padding_mask_4d.unsqueeze(1)

    def _pad_coordinates(
        self,
        coordinates: torch.Tensor,
        levels: torch.Tensor,
        lengths: torch.Tensor,
        k_bucket: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对 Hilbert 坐标和深度进行 Padding。

        关键：Padding 坐标设为原点 (0, 0)，深度设为 -1（无效标记）
        这样 RoPE 在计算频率偏置时不会产生 NaN。

        Args:
            coordinates: [B, K, 2] Hilbert 坐标 (x, y)
            levels: [B, K] 对应深度
            lengths: [B] 有效长度
            k_bucket: 目标桶大小

        Returns:
            (coords_padded, levels_padded)
        """
        K = coordinates.shape[1]
        pad_len = k_bucket - K
        B = coordinates.shape[0]
        device = coordinates.device

        if pad_len > 0:
            # 坐标 Padding：使用原点 (0, 0)
            coord_padding = torch.zeros((B, pad_len, 2), device=device)
            coords_padded = torch.cat([coordinates, coord_padding], dim=1)

            # 深度 Padding：使用 -1 表示无效
            depth_padding = torch.full((B, pad_len), -1, device=device, dtype=levels.dtype)
            levels_padded = torch.cat([levels, depth_padding], dim=1)
        else:
            coords_padded, levels_padded = coordinates, levels

        return coords_padded, levels_padded
