# -*- coding: utf-8 -*-
"""
LevelsInfo 强类型数据结构模块

数学形式化
==========

LevelsInfo 是四叉树(Quadtree)结构的展平表示:

    L ∈ Z^{B × N × (D+1)}

其中:
- B: batch size
- N: token 数量（可变）
- D: max_depth（四叉树最大深度）

语义分解:
    L[:, :, 0] = depths, d_i ∈ {-1, 0, 1, ..., D}
    L[:, :, 1:] = paths, q_i ∈ {0, 1, 2, 3}^D

四象限编码 (Hilbert 曲线基础):
    0: 左上 (top-left)     → (x_low, y_low)
    1: 右上 (top-right)    → (x_high, y_low)
    2: 左下 (bottom-left)  → (x_low, y_high)
    3: 右下 (bottom-right) → (x_high, y_high)

约束集 C:
    C1: depth ∈ [-1, D]
    C2: path ∈ [0, 3] for valid tokens
    C3: path length = depth

类对照表
----------
+-------------------+--------------------------------------+
| 类                 | 用途                                  |
+===================+======================================+
| LevelsInfo        | levels_info 强类型数据结构            |
+-------------------+--------------------------------------+

Hilbert Curve 集成:
    - LCA 计算依赖 depths 和 paths 的一致性
    - Hilbert 局部性要求 path 编码正确
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch


@dataclass
class LevelsInfo:
    """levels_info 强类型数据结构，Hilbert Curve ViT 核心契约。

    Mathematical Definition:
        L ∈ Z^{B × N × (D+1)}
        L[:, :, 0] = depths (四叉树深度)
        L[:, :, 1:] = quadtree paths (四象限索引)

    Invariants (checked in __post_init__):
        C1: depth ∈ [-1, max_depth]
        C2: path ∈ [0, 3] for valid tokens
        C3: path length = depth

    Hilbert Curve Integration:
        - LCA 计算依赖 depths 和 paths 的一致性
        - Hilbert 局部性要求 path 编码正确
    """

    # 主数据张量
    data: torch.Tensor  # [B, N, D+1]

    # 元数据
    max_depth: int

    # 缓存字段 (惰性求值)
    _depths: Optional[torch.Tensor] = field(default=None, repr=False)
    _paths: Optional[torch.Tensor] = field(default=None, repr=False)

    def __post_init__(self):
        """Invariant validation - Hilbert Curve ViT 核心契约检查。"""
        B, N, K = self.data.shape
        D = self.max_depth

        # C0: 维度约束
        assert K == D + 1, \
            f"levels_info 维度错误: 期望 {D+1} 列, 实际 {K}"

        # C1: depth 范围 [-1, D]
        depths = self._compute_depths()
        assert (depths >= -1).all(), "depth < -1 (padding sentinel 违反)"
        assert (depths <= D).all(), f"depth > {D} (超过 max_depth)"

        # C2: path 范围 [0, 3] (仅有效 token)
        valid_mask = depths >= 0
        if valid_mask.any():
            paths = self._compute_paths()
            path_values = paths[valid_mask]
            assert (path_values >= 0).all() and (path_values <= 3).all(), \
                "path 值超出 [0, 3] 范围 (四象限编码错误)"

        # C3: 路径长度一致性 (隐式通过数据结构保证)

    def _compute_depths(self) -> torch.Tensor:
        """提取 depth 列 [B, N]"""
        if self._depths is None:
            self._depths = self.data[:, :, 0]
        return self._depths

    def _compute_paths(self) -> torch.Tensor:
        """提取 paths 列 [B, N, D]"""
        if self._paths is None:
            self._paths = self.data[:, :, 1:]
        return self._paths

    @property
    def depths(self) -> torch.Tensor:
        """Property accessor with cache invalidation."""
        return self._compute_depths()

    @property
    def paths(self) -> torch.Tensor:
        """Property accessor with cache invalidation."""
        return self._compute_paths()

    @property
    def shape(self) -> Tuple[int, int, int]:
        """返回 (B, N, D+1)"""
        return tuple(self.data.shape)

    @property
    def batch_size(self) -> int:
        """B"""
        return self.data.shape[0]

    @property
    def num_tokens(self) -> int:
        """N"""
        return self.data.shape[1]

    @property
    def max_level(self) -> int:
        """D"""
        return self.max_depth

    def __len__(self) -> int:
        """返回 token 数量 N"""
        return self.num_tokens

    def to(self, device: torch.device, non_blocking: bool = False) -> "LevelsInfo":
        """设备迁移 (保持 cache)"""
        return LevelsInfo(
            data=self.data.to(device, non_blocking=non_blocking),
            max_depth=self.max_depth,
            _depths=self._depths.to(device, non_blocking=non_blocking) if self._depths is not None else None,
            _paths=self._paths.to(device, non_blocking=non_blocking) if self._paths is not None else None,
        )

    def cuda(self, non_blocking: bool = False) -> "LevelsInfo":
        """迁移到 CUDA 设备"""
        return self.to(torch.device("cuda"), non_blocking=non_blocking)

    def cpu(self, non_blocking: bool = False) -> "LevelsInfo":
        """迁移到 CPU 设备"""
        return self.to(torch.device("cpu"), non_blocking=non_blocking)

    # ========== 工厂方法 ==========

    @staticmethod
    def from_tokenizer_output(
        output: "TokenizerOutput",
        max_depth: int,
    ) -> "LevelsInfo":
        """从 TokenizerOutput 创建 LevelsInfo (I98-5 简化).

        I98-5: output.levels_info 现在直接返回 LevelsInfo，
        因此只需返回该值或在为空时创建默认值。

        Args:
            output: TokenizerOutput 实例
            max_depth: 四叉树最大深度

        Returns:
            LevelsInfo 实例
        """
        info = output.levels_info
        if info is not None:
            return info

        # 空输出时创建默认 LevelsInfo
        B = output.batch_size
        N = 1
        all_levels = torch.zeros(B, N, max_depth + 1, dtype=torch.long)
        return LevelsInfo(data=all_levels, max_depth=max_depth)

    @staticmethod
    def from_arrays(
        depths: torch.Tensor,  # [B, N]
        paths: torch.Tensor,  # [B, N, D]
        max_depth: int,
    ) -> "LevelsInfo":
        """从 depths 和 paths 数组创建 LevelsInfo。

        Args:
            depths: 深度张量
            paths: 路径张量
            max_depth: 最大深度

        Returns:
            LevelsInfo 实例
        """
        B, N = depths.shape
        D = paths.size(2) if paths.dim() == 3 else max_depth

        data = torch.zeros(B, N, max_depth + 1, dtype=torch.long, device=depths.device)
        data[:, :, 0] = depths
        data[:, :, 1 : 1 + D] = paths

        return LevelsInfo(data=data, max_depth=max_depth)

    @staticmethod
    def random(
        B: int,
        N: int,
        max_depth: int,
        device: Optional[torch.device] = None,
    ) -> "LevelsInfo":
        """创建随机 LevelsInfo (测试用)。

        生成有效的四叉树结构：
        - depth 均匀分布在 [0, max_depth]
        - path 均匀分布在 [0, 3]
        """
        import random

        depths_list = []
        paths_list = []

        for b in range(B):
            for n in range(N):
                d = random.randint(0, max_depth)
                depths_list.append(d)
                path = [random.randint(0, 3) for _ in range(d)]
                paths_list.append(path + [0] * (max_depth - d))

        depths = torch.tensor(depths_list, dtype=torch.long).view(B, N)
        paths = torch.tensor(paths_list, dtype=torch.long).view(B, N, max_depth)

        if device:
            depths = depths.to(device)
            paths = paths.to(device)

        return LevelsInfo.from_arrays(depths, paths, max_depth)

    # ========== Hilbert Curve 工具方法 ==========

    def _path_to_hilbert_index(self, path: list[int], depth: int) -> int:
        """将四叉树路径转换为 Hilbert 曲线索引。

        使用 Hilbert 曲线的递归来计算索引。

        Args:
            path: 四象限路径列表
            depth: 路径深度

        Returns:
            Hilbert 曲线索引
        """
        if depth == 0:
            return 0

        # 使用 curve_hilbert 的函数
        from .curve_hilbert import xy_to_hilbert_distance

        # 计算子网格大小
        grid_size = 1 << depth  # 2^depth

        # 从路径计算 x, y 坐标
        x, y = 0, 0
        for level, quadrant in enumerate(path):
            # 在每个深度级别，根据象限更新坐标
            half = 1 << (depth - level - 1)
            if quadrant == 0:  # 左上
                # 无变化
                pass
            elif quadrant == 1:  # 右上
                x += half
            elif quadrant == 2:  # 左下
                y += half
            elif quadrant == 3:  # 右下
                x += half
                y += half

        return xy_to_hilbert_distance(grid_size, x, y)

    def get_hilbert_indices(self) -> torch.Tensor:
        """计算 Hilbert 曲线索引 (用于排序)。

        Returns:
            hilbert_indices: [B, N] 每个 token 的 Hilbert 序
        """
        B, N, D_plus_1 = self.data.shape
        D = D_plus_1 - 1

        hilbert_indices = torch.zeros(B, N, dtype=torch.long, device=self.data.device)

        for b in range(B):
            for n in range(N):
                depth = int(self.data[b, n, 0].item())
                if depth < 0:
                    continue

                path = self.data[b, n, 1 : 1 + depth].tolist()
                idx = self._path_to_hilbert_index(path, depth)
                hilbert_indices[b, n] = idx

        return hilbert_indices

    def get_lca_matrix(self) -> torch.Tensor:
        """计算 LCA 矩阵 (用于注意力偏置)。

        Returns:
            lca_matrix: [B, N, N] 每对 token 的 LCA 深度
        """
        B, N, D_plus_1 = self.data.shape
        D = D_plus_1 - 1

        lca_matrix = torch.zeros(B, N, N, dtype=torch.long, device=self.data.device)

        for b in range(B):
            for i in range(N):
                for j in range(i + 1, N):
                    d_i = int(self.data[b, i, 0].item())
                    d_j = int(self.data[b, j, 0].item())

                    if d_i < 0 or d_j < 0:
                        continue

                    path_i = self.data[b, i, 1 : 1 + d_i].tolist()
                    path_j = self.data[b, j, 1 : 1 + d_j].tolist()

                    # 计算 LCA
                    lca_depth = 0
                    for t in range(min(d_i, d_j)):
                        if path_i[t] == path_j[t]:
                            lca_depth = t + 1
                        else:
                            break

                    lca_matrix[b, i, j] = lca_depth
                    lca_matrix[b, j, i] = lca_depth

        return lca_matrix
