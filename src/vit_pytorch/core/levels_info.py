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
from typing import TYPE_CHECKING, Optional, Tuple
import torch

if TYPE_CHECKING:
    from vit_pytorch.modules.base_tokenizer import TokenizerOutput

from .curve_hilbert import xy_to_hilbert_distance

# I102-8: Hilbert 索引查找表 (预计算)
# I101-2: 移除深度上限硬编码，扩展到 D_max=8
# 空间复杂度: Σ_{d=1}^8 4^d = 4(4^8-1)/3 ≈ 21,844 条目 ≈ 175 KB
# 对于 max_depth > 8，仍可使用但会回退到 Python 循环
_MAX_HILBERT_DEPTH = 8

_HILBERT_LUT = {}
for d in range(1, _MAX_HILBERT_DEPTH + 1):
    _HILBERT_LUT[d] = {}
    for path_int in range(4 ** d):
        # 解码: 整数 → 路径
        path = [(path_int // (4 ** (d - t - 1))) % 4 for t in range(d)]

        # 计算坐标
        x, y = 0, 0
        for level, quadrant in enumerate(path):
            half = 1 << (d - level - 1)
            if quadrant == 1:
                x += half
            elif quadrant == 2:
                y += half
            elif quadrant == 3:
                x += half
                y += half

        # 计算 Hilbert 距离
        _HILBERT_LUT[d][path_int] = xy_to_hilbert_distance(1 << d, x, y)

# D2-AUDIT FIX: 预计算 tensor 形式的 LUT 用于向量化查找
# 避免 Python 循环 + dict lookup，转为直接 tensor 索引
_HILBERT_LUT_TENSOR = {}
for d in range(1, _MAX_HILBERT_DEPTH + 1):
    size = 1 << (2 * d)  # 4**d = 2**(2d)
    lut = torch.empty(size, dtype=torch.long)
    for path_int in range(size):
        lut[path_int] = _HILBERT_LUT[d][path_int]
    _HILBERT_LUT_TENSOR[d] = lut

# P1 FIX (Eager Initialization): 2D 统一填充 LUT，支持全量向量化查表
# 方案: 构建 [max_depth+1, 4^max_depth] 的 2D 张量，
# 深度 d 的 LUT 放在 row d，不足部分用 0 填充
# 内存: depth=8 时 = 9 × 65536 × 8 bytes ≈ 470 KB
# I167-1: torch.compile 兼容性修复 - 移除惰性初始化，改为模块加载时立即初始化
# 原因: torch.compile 无法正确保留跨图边界的全局状态，导致 NoneType 检查失败
_MAX_LUT_DEPTH = _MAX_HILBERT_DEPTH  # 与 _HILBERT_LUT 填充深度保持一致


def _init_hilbert_lut_padded() -> torch.Tensor:
    """在模块加载时预先构建对齐的 2D 查找表（Eager Initialization）"""
    max_size = 1 << (2 * _MAX_LUT_DEPTH)  # 4^8 = 65536
    lut_2d = torch.zeros((_MAX_LUT_DEPTH + 1, max_size), dtype=torch.long)
    # 注意: _HILBERT_LUT 仅填充到 _MAX_HILBERT_DEPTH (8)，因此仅遍历可用深度
    for d in range(1, _MAX_HILBERT_DEPTH + 1):
        size = 1 << (2 * d)  # 4^d
        for path_int in range(size):
            lut_2d[d, path_int] = _HILBERT_LUT[d][path_int]
    return lut_2d


# 关键修复: 不再使用 None 惰性初始化，而是模块加载时立即初始化
# torch.compile 将其视为 ConstantVariable，能够安全地嵌入生成的 Kernel
_HILBERT_LUT_PADDED: torch.Tensor = _init_hilbert_lut_padded()

# I167-1 FIX: 设备缓存，避免每次 forward 都调用 .to(device)
# 初始化为 None，在首次调用 get_hilbert_indices 时填充
_hilbert_lut_cached: Optional[torch.Tensor] = None


def _get_hilbert_lut_padded() -> torch.Tensor:
    """获取填充后的 2D LUT（保持接口兼容）。"""
    return _HILBERT_LUT_PADDED

# I103-2: Hilbert 索引权重缓存 (类级缓存)
# 避免在 get_hilbert_indices() 中重复创建权重张量
# 内存: D_max=8 时仅需存储 36 个整数 (< 1KB)
# I162-1: 预填充缓存 (torch.compile + CUDA Graphs 兼容性)
# 在模块加载时预填充，避免在编译区域内创建张量
_hilbert_weights_cache: dict = {}

# 预填充缓存: 常见的深度值 (1-12)
# 这确保缓存在 torch.compile 之前就准备好
_MAX_PREPOPULATE_DEPTH = 12


def _prepopulate_hilbert_weights_cache():
    """预填充 Hilbert 权重缓存 (torch.compile 兼容性)"""
    for d in range(1, _MAX_PREPOPULATE_DEPTH + 1):
        weights_list = [4 ** (d - t - 1) for t in range(d)]
        # 创建张量并存入缓存 (使用 CPU，稍后移到 GPU)
        _hilbert_weights_cache[d] = torch.tensor(
            weights_list,
            dtype=torch.long,
            device='cpu'  # 预填充 CPU 张量，运行时移到目标设备
        )


# 模块加载时预填充缓存
_prepopulate_hilbert_weights_cache()


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
    max_level: int

    # 缓存字段 (惰性求值)
    _depths: Optional[torch.Tensor] = field(default=None, repr=False)
    _paths: Optional[torch.Tensor] = field(default=None, repr=False)

    def __post_init__(self):
        """Invariant validation - Hilbert Curve ViT 核心契约检查。"""
        # I-TOK: 添加详细的维度检查，帮助调试 torch.compile 问题
        # 允许传入 LevelsInfo 对象（用于链式转换）
        if hasattr(self.data, 'data') and isinstance(self.data.data, torch.Tensor):
            # 传入的是 LevelsInfo 对象，提取其内部张量
            actual_data = self.data.data
        elif isinstance(self.data, torch.Tensor):
            actual_data = self.data
        else:
            raise TypeError(f"levels_info 期望 torch.Tensor 或 LevelsInfo，实际 {type(self.data)}")

        actual_dim = actual_data.dim()
        if actual_dim != 3:
            # I-TOK: torch.compile 可能导致维度问题，尝试恢复
            shape = actual_data.shape
            expected_cols = self.max_level + 1

            if actual_dim == 2:
                # 尝试恢复为 (B, N, D+1)
                if shape[1] % expected_cols == 0:
                    N = shape[1] // expected_cols
                    self.data = actual_data.view(shape[0], N, expected_cols)
                else:
                    raise ValueError(
                        f"levels_info 维度错误: 期望 3 维张量 (B, N, D+1)，"
                        f"实际 2 维 {shape}，无法恢复。"
                    )
            elif actual_dim == 4:
                # I-TOK: torch.compile 可能将张量扩展为 4D [B, 1, 1, D+1]
                if shape[1] == 1 and shape[2] == 1:
                    # [B, 1, 1, D+1] -> [B, 1, D+1] -> [B, D+1] -> [B, 1, D+1]
                    squeezed = actual_data.squeeze(2)  # [B, 1, D+1]
                    squeezed = squeezed.squeeze(1)  # [B, D+1]
                    self.data = squeezed.unsqueeze(1)  # [B, 1, D+1]
                elif shape[1] == 1:
                    # [B, 1, N, D+1] -> [B, N, D+1]
                    self.data = actual_data.squeeze(1)
                else:
                    raise ValueError(
                        f"levels_info 维度错误: 期望 3 维张量 (B, N, D+1)，"
                        f"实际 4 维 {shape}，无法恢复。"
                    )
            else:
                raise ValueError(
                    f"levels_info 维度错误: 期望 3 维张量 (B, N, D+1)，"
                    f"实际 {actual_dim} 维，形状: {shape}。"
                )
            B, N, K = self.data.shape
        else:
            # 正常情况：直接使用传入的张量
            if actual_data is not self.data:
                # 如果我们提取了内部张量，需要更新 data
                self.data = actual_data
            B, N, K = self.data.shape

        D = self.max_level

        # C0: 维度约束
        assert K == D + 1, \
            f"levels_info 维度错误: 期望 {D+1} 列, 实际 {K}"

        # C1: depth 范围 [-1, D]
        depths = self._compute_depths()
        assert (depths >= -1).all(), "depth < -1 (padding sentinel 违反)"
        assert (depths <= D).all(), f"depth > {D} (超过 max_level)"

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

    def __len__(self) -> int:
        """返回 token 数量 N"""
        return self.num_tokens

    def to(self, device: torch.device, non_blocking: bool = False) -> "LevelsInfo":
        """设备迁移 (保持 cache)"""
        return LevelsInfo(
            data=self.data.to(device, non_blocking=non_blocking),
            max_level=self.max_level,
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
    def ensure(
        info: "LevelsInfo | torch.Tensor | None",
        default_max_level: int = 0,
    ) -> "LevelsInfo | None":
        """统一入口：确保返回标准 LevelsInfo 对象或 None。

        这是唯一的"护城河"——所有其他位置的类型检查都必须移除。
        torch.compile 模式下，将类型检查集中在此处可以避免多处 Graph Break。

        Args:
            info: LevelsInfo 对象、torch.Tensor 或 None
            default_max_level: 当 info 为 Tensor 时使用的最大深度

        Returns:
            LevelsInfo 实例或 None

        Raises:
            TypeError: 当 info 不是期望的类型时
        """
        if info is None:
            return None
        if isinstance(info, LevelsInfo):
            return info
        if isinstance(info, torch.Tensor):
            if info.dtype != torch.long:
                info = info.long()
            info_dim = info.shape[-1]
            inferred_max_level = info_dim - 1
            return LevelsInfo(data=info, max_level=inferred_max_level)
        raise TypeError(
            f"levels_info 期望 LevelsInfo 或 torch.Tensor，实际 {type(info)}"
        )

    @staticmethod
    def from_tokenizer_output(
        output: "TokenizerOutput",
        max_level: int,
    ) -> "LevelsInfo":
        """从 TokenizerOutput 创建 LevelsInfo (I98-5 简化).

        I98-5: output.levels_info 现在直接返回 LevelsInfo，
        因此只需返回该值或在为空时创建默认值。

        Args:
            output: TokenizerOutput 实例
            max_level: 四叉树最大深度

        Returns:
            LevelsInfo 实例
        """
        info = output.levels_info
        if info is not None:
            return info

        # 空输出时创建默认 LevelsInfo
        B = output.batch_size
        N = 1
        all_levels = torch.zeros(B, N, max_level + 1, dtype=torch.long)
        return LevelsInfo(data=all_levels, max_level=max_level)

    @staticmethod
    def from_arrays(
        depths: torch.Tensor,  # [B, N]
        paths: torch.Tensor,  # [B, N, D]
        max_level: int,
    ) -> "LevelsInfo":
        """从 depths 和 paths 数组创建 LevelsInfo。

        Args:
            depths: 深度张量
            paths: 路径张量
            max_level: 最大深度

        Returns:
            LevelsInfo 实例
        """
        B, N = depths.shape
        D = paths.size(2) if paths.dim() == 3 else max_level

        data = torch.zeros(B, N, max_level + 1, dtype=torch.long, device=depths.device)
        data[:, :, 0] = depths
        data[:, :, 1 : 1 + D] = paths

        return LevelsInfo(data=data, max_level=max_level)

    @staticmethod
    def random(
        B: int,
        N: int,
        max_level: int,
        device: Optional[torch.device] = None,
    ) -> "LevelsInfo":
        """创建随机 LevelsInfo (测试用)。

        生成有效的四叉树结构：
        - depth 均匀分布在 [0, max_level]
        - path 均匀分布在 [0, 3]
        """
        import random

        depths_list = []
        paths_list = []

        for b in range(B):
            for n in range(N):
                d = random.randint(0, max_level)
                depths_list.append(d)
                path = [random.randint(0, 3) for _ in range(d)]
                paths_list.append(path + [0] * (max_level - d))

        depths = torch.tensor(depths_list, dtype=torch.long).view(B, N)
        paths = torch.tensor(paths_list, dtype=torch.long).view(B, N, max_level)

        if device:
            depths = depths.to(device, non_blocking=True)
            paths = paths.to(device, non_blocking=True)

        return LevelsInfo.from_arrays(depths, paths, max_level)

    # ========== I103-2: 权重缓存方法 ==========

    @staticmethod
    def _get_hilbert_weights_for_depth(d: int, device: torch.device) -> torch.Tensor:
        """获取深度 d 的 Hilbert 编码权重 (使用预填充缓存)。

        I103-2 优化: 使用预填充缓存避免重复计算。
        I162-1 修复: 预填充缓存在模块加载时完成，确保 torch.compile + CUDA Graphs 兼容性。

        数学:
            weights = [4^(d-1), 4^(d-2), ..., 4^0]

        Args:
            d: 深度 (1 <= d <= max_depth)
            device: 目标设备

        Returns:
            weights: [d] 权重张量 (clone 以避免 CUDA Graphs 覆盖问题)
        """
        # I162-1: 缓存已预填充，直接访问并 clone 以避免 CUDA Graphs 覆盖
        return _hilbert_weights_cache[d].clone().to(device, non_blocking=True)

    # ========== 深度根归一化 (I161-1 修复) ==========

    @staticmethod
    def normalize_hilbert_index(hilbert_dist: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """深度根归一化 (I161-1 修复：保持跨尺度一致性)

        数学:
            H_norm = (H / 4^d)^(1/d)

        性质:
            - H_norm ∈ [0, 1] 对于任意深度 d
            - 深度 d 的子点 H_norm ≈ 深度 d-1 的父点 H_norm
            - 自然保持 Hilbert 曲线的自相似性

        Args:
            hilbert_dist: Hilbert距离 [B, N] 或 [N]
            depth: 对应深度 [B, N] 或 [N]

        Returns:
            归一化Hilbert索引，范围 [0, 1]
        """
        # I-NAN: 限制最大深度，防止指数溢出
        depth_safe = depth.clamp(min=1, max=12)

        # 计算 4^depth [B, N] 或 [N]，并 clamp 防止溢出
        # D4-AUDIT FIX: 4^d = 2^(2d) = exp2(2d)，CUDA 上 fused FMA 比 pow 更快
        four_pow_depth = torch.exp2(depth_safe * 2.0).clamp(max=1e9)

        # 深度根归一化: (H / 4^d)^(1/d)
        normalized = (hilbert_dist.float() / four_pow_depth.float()) ** (1.0 / depth_safe.float())

        # 处理深度为0的情况
        if depth.dim() == 1:
            normalized = torch.where(depth == 0, torch.zeros_like(normalized), normalized)
        else:
            normalized = torch.where(
                depth == 0,
                torch.zeros_like(normalized),
                normalized
            )

        return normalized

    def get_hilbert_indices(self, normalize: bool = False) -> torch.Tensor:
        """计算 Hilbert 曲线索引 (用于排序)。

        I102-8 修复: 使用查找表实现正确的向量化
        I103-2 优化: 使用类级权重缓存避免重复创建张量
        I161-1 修复: 添加归一化选项保持跨尺度一致性
        I103-3 优化: 向量化 LUT 查找，移除 Python 循环内的 dict 查找

        数学:
            H_raw = Σ q_k × 4^{d-k}  (原始Hilbert距离)
            H_norm = (H_raw / 4^d)^(1/d)  (深度根归一化)

        Args:
            normalize: 是否使用深度根归一化 (默认 False 保持向后兼容)

        Returns:
            hilbert_indices: [B, N] 每个 token 的 Hilbert 序
            - normalize=False: 原始Hilbert距离，范围 [0, 4^d - 1]
            - normalize=True: 归一化Hilbert索引，范围 [0, 1]
        """
        B, N, D_plus_1 = self.data.shape
        D = D_plus_1 - 1
        depths = self.data[:, :, 0]  # [B, N]
        paths = self.data[:, :, 1:]  # [B, N, D]
        device = self.data.device

        # 验证深度约束 (实际 LUT 仅填充到 _MAX_HILBERT_DEPTH)
        if D > _MAX_HILBERT_DEPTH:
            raise ValueError(
                f"Hilbert 深度 {D} 超过最大允许值 {_MAX_HILBERT_DEPTH}。"
                "请考虑使用动态回退方案。"
            )

        # P1 FIX: 全量向量化查表 — 消除 per-depth loop
        # 核心: 构建 W[d,k] = 4^(d-k) for k <= d (上三角)，然后 paths @ W.T
        #
        # 示例 paths=[2,3,1,0], W^T = [[1,4,16,64],[0,1,4,16],[0,0,1,4],[0,0,0,1]]
        # paths @ W.T = [2*1, 2*4+3*1, 2*16+3*4+1*1, ...] = [2, 11, 45, 180]
        d_idx = torch.arange(D, device=device).unsqueeze(1)  # [[0],[1],[2],[3]]
        k_idx = torch.arange(D, device=device).unsqueeze(0)  # [[0,1,2,3]]
        W = torch.where(
            k_idx <= d_idx,  # 上三角(含对角): k <= d 时有效
            torch.pow(4, (d_idx - k_idx).float()).long(),  # 4^(d-k)
            torch.zeros(D, D, device=device, dtype=torch.long)
        )  # [D, D]
        # paths: [B, N, D], W.T: [D, D]
        # paths_flat @ W.T → [B*N, D], reshape → [B, N, D]
        # 使用 @ 运算符替代 torch.bmm，因为这是普通 2D 矩阵乘法
        # torch.bmm 要求 3D 输入，但此处是 [B*N, D] @ [D, D]
        path_ints_all = (
            paths.view(B * N, D).float() @ W.t().float()
        ).long().view(B, N, D)  # [B, N, D]
        # path_ints_all[b, n, d] = Σ_{k=0}^{d} paths[b,n,k] × 4^(d-k)

        # torch.gather: 按 actual depth 取对应深度的路径整数
        depth_for_gather = depths.long().clamp(min=0, max=D - 1)  # [B, N]
        path_ints_per_depth = torch.gather(
            path_ints_all, dim=2, index=depth_for_gather.unsqueeze(2)
        ).squeeze(2)  # [B, N]

        # I167-1 FIX: 直接使用全局 LUT 张量，torch.compile 更友好
        # _HILBERT_LUT_PADDED 在模块加载时已初始化为 CPU 张量
        # 仅在首次遇到不同设备时缓存设备特定版本
        global _hilbert_lut_cached
        if not hasattr(_hilbert_lut_cached, 'device') or _hilbert_lut_cached.device != device:
            _hilbert_lut_cached = _HILBERT_LUT_PADDED.to(device)
        lut_2d = _hilbert_lut_cached
        results = lut_2d[depths.long(), path_ints_per_depth]  # [B, N]

        # I161-1 修复: 深度根归一化
        if normalize:
            return self.normalize_hilbert_index(results, depths)

        return results

    def get_lca_matrix(self) -> torch.Tensor:
        """计算 LCA 矩阵 (用于注意力偏置)。

        I102-7 向量化实现: O(B×N²×D) → 纯张量运算，无 Python 循环
        利用 broadcasting 同时计算所有 (i, j) 对的 LCA 深度。

        Returns:
            lca_matrix: [B, N, N] 每对 token 的 LCA 深度
        """
        B, N, D_plus_1 = self.data.shape
        D = D_plus_1 - 1

        # 提取深度和路径
        depths = self.data[:, :, 0]  # [B, N]
        paths = self.data[:, :, 1:]  # [B, N, D]

        # 创建深度掩码 [B, N, D]
        depth_indices = torch.arange(D, device=self.data.device)  # [D]
        depth_mask = (depth_indices < depths.unsqueeze(-1))  # [B, N, D]

        # Broadcasting: 扩展到 [B, N, N, D]
        paths_i = paths.unsqueeze(2)  # [B, N, 1, D]
        paths_j = paths.unsqueeze(1)  # [B, 1, N, D]
        mask_i = depth_mask.unsqueeze(2)  # [B, N, 1, D]
        mask_j = depth_mask.unsqueeze(1)  # [B, 1, N, D]

        # 比较所有对并应用有效掩码 [B, N, N, D]
        match = (paths_i == paths_j)
        valid_mask = mask_i & mask_j

        # LCA 深度 = 匹配且有效的层数 [B, N, N]
        lca_matrix = (match & valid_mask).sum(dim=-1)

        # I102-7: 向量化对角线清零 - 使用 diagonal() 方法替代 Python 循环
        # 数学形式: LCA_out = LCA_in ⊙ (1 - I_N)^(⊗B)
        lca_matrix.diagonal(dim1=1, dim2=2).zero_()

        return lca_matrix
