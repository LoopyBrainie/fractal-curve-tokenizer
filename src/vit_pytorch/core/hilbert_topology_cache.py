# -*- coding: utf-8 -*-
"""
HilbertTopologyCache: Hilbert 拓扑缓存 - 惰性预计算

特性:
- 使用 persistent=False 防止缓存保存到权重文件
- 单尺寸缓存（简化逻辑，适用于 ViT 场景）
- 自动设备迁移

数学形式化
============

coord_to_idx[y, x] = d
    将 2D 坐标 (x, y) 映射到 Hilbert 索引 d

idx_to_coord[d] = (x, y)
    将 Hilbert 索引 d 映射回 2D 坐标 (x, y)

这两个表互为逆运算，保证双射性。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from vit_pytorch.core.curve_hilbert import PseudoHilbertCurve


class HilbertTopologyCache(nn.Module):
    """
    Hilbert 拓扑缓存 - 惰性预计算

    使用 persistent=False 防止这些缓存被保存到权重文件。
    """

    def __init__(self):
        super().__init__()
        # 缓存元数据
        self.register_buffer(
            "_cached_h", torch.tensor(-1), persistent=False
        )
        self.register_buffer(
            "_cached_w", torch.tensor(-1), persistent=False
        )
        # 缓存数据
        self.register_buffer(
            "_coord_to_idx", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_idx_to_coord", torch.empty(0, 2, dtype=torch.long), persistent=False
        )

    def get(
        self, H: int, W: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """
        获取 coord_to_idx 和 idx_to_coord

        Args:
            H: 高度
            W: 宽度
            device: 目标设备

        Returns:
            (coord_to_idx, idx_to_coord)
            - coord_to_idx: Tensor of shape [H, W]
            - idx_to_coord: Tensor of shape [H*W, 2]
        """
        # 检查缓存命中
        if self._cached_h.item() == H and self._cached_w.item() == W:
            # 确保设备一致
            if self._coord_to_idx.device == device:
                return self._coord_to_idx, self._idx_to_coord
            else:
                # 设备迁移
                self._coord_to_idx = self._coord_to_idx.to(device)
                self._idx_to_coord = self._idx_to_coord.to(device)
                return self._coord_to_idx, self._idx_to_coord

        # 缓存未命中：生成并缓存
        coord_to_idx, idx_to_coord = self._generate_tables(H, W, device)

        self._cached_h.fill_(H)
        self._cached_w.fill_(W)
        self._coord_to_idx = coord_to_idx
        self._idx_to_coord = idx_to_coord

        return self._coord_to_idx, self._idx_to_coord

    def _generate_tables(
        self, H: int, W: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """生成 Hilbert 映射表"""
        # 使用向量化方法
        coord_to_idx = PseudoHilbertCurve.build_coord_to_idx_tensor(H, W, device)
        idx_to_coord = PseudoHilbertCurve.scan_tensor(H, W, device)
        return coord_to_idx, idx_to_coord

    def xy_to_d(self, x: Tensor, y: Tensor, H: int, W: int) -> Tensor:
        """
        批量坐标 → Hilbert 索引

        Args:
            x: [B] x 坐标
            y: [B] y 坐标
            H: 高度
            W: 宽度

        Returns:
            d: [B] Hilbert 索引
        """
        coord_to_idx, _ = self.get(H, W, x.device)
        return coord_to_idx[y, x]

    def d_to_xy(self, d: Tensor, H: int, W: int) -> Tuple[Tensor, Tensor]:
        """
        批量 Hilbert 索引 → 坐标

        Args:
            d: [B] Hilbert 索引
            H: 高度
            W: 宽度

        Returns:
            (x, y): [B] 坐标
        """
        _, idx_to_coord = self.get(H, W, d.device)
        d_clamped = torch.clamp(d, 0, H * W - 1)
        coords = idx_to_coord[d_clamped]
        return coords[:, 0], coords[:, 1]

    def get_knn_indices(
        self, max_level: int, K: int = 32
    ) -> Tuple[Tensor, Tensor]:
        """
        生成 K 近邻索引 (Gather-Scatter 范式)

        数学形式:
            N(i) = { j | Hilbert距离(i,j) < threshold }
            使用 -1 作为 padding

        Args:
            max_level: 最大 Hilbert 层级
            K: 每节点邻居数

        Returns:
            neighbor_indices: [N, K] 邻居索引 (padding=-1)
            mask: [N, K] 有效掩码 (True=有效)
        """
        # 计算候选区域数量
        N = sum(4 ** d for d in range(max_level + 1))

        # 获取 Hilbert 索引
        hilbert_indices = self._compute_hilbert_indices(max_level)

        # 预分配结果
        neighbor_indices = torch.full((N, K), -1, dtype=torch.long, device=hilbert_indices.device)
        mask = torch.zeros(N, K, dtype=torch.bool, device=hilbert_indices.device)

        # 简化实现: 使用排序找到 K 近邻
        # 由于 Hilbert 曲线的局部性，相邻索引在空间上也相邻
        half_window = K * 2  # 搜索窗口大小

        for i in range(N):
            # 找到窗口内的候选邻居
            left = max(0, i - half_window)
            right = min(N, i + half_window)

            # 创建候选列表（排除自身）
            candidates = list(range(left, i)) + list(range(i + 1, right))

            if len(candidates) == 0:
                continue

            # 计算与候选的距离
            cand_tensor = torch.tensor(candidates, device=hilbert_indices.device)
            dist = (hilbert_indices[i] - hilbert_indices[cand_tensor]).abs()
            dist = torch.min(dist, hilbert_indices.max() - dist)  # 循环处理

            # 取最近的 K 个
            k_actual = min(K, len(candidates))
            _, topk_local = torch.topk(dist, k=k_actual)

            # 获取实际索引
            topk_idx = cand_tensor[topk_local]
            neighbor_indices[i, :k_actual] = topk_idx
            mask[i, :k_actual] = True

        return neighbor_indices, mask

    def _compute_hilbert_indices(self, max_level: int) -> Tensor:
        """计算 Hilbert 索引 (用于 KNN)"""
        # 简化版: 使用行主序 + 深度偏移
        indices = []
        for d in range(max_level + 1):
            n_regions = 4 ** d
            offset = sum(4 ** k for k in range(d))
            indices.extend([i + offset for i in range(n_regions)])

        return torch.tensor(indices, dtype=torch.long, device=self._coord_to_idx.device)
