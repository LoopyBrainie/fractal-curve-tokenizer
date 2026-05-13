# -*- coding: utf-8 -*-
"""
方向感知子空间隔离 RoPE (C+)

方案 C+ 核心实现：DirectionAwareSubspacedRoPE

数学形式化
===========

核心公式:
    M_l = Rot(ω_base · π/2 · G(GEOM_MAP_TABLE[d_{l-1}, q_l]))  if l <= depth
          I                                                     if l > depth

其中:
    - d_{l-1} ∈ {0,1,2,3} 是第 l-1 层的方向状态
    - q_l ∈ {0,1,2,3} 是第 l 层的象限
    - G 是格雷码，用于局部保序
    - ω_base = base^{-2k/D_s} 是频率展宽因子

完整旋转矩阵:
    R_final = ⊕_{l=1}^{L_max} M_l

频率展宽 (防止表征坍塌):
    Rot_{D_s}(θ_l) = ⊕_{k=0}^{D_s/2-1} Rot(θ_l · ω_k)
    ω_k = base^{-2k/D_s}

与 Cartesian2DRoPE 的互补关系:
    - 物理场 (前 D/4): Cartesian2DRoPE - 全局平移不变性
    - 拓扑场 (后 3D/4): C+ RoPE - 分形树层级与局部拓扑
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple

import torch
import torch.nn as nn

from vit_pytorch.core.levels_info import GEOM_MAP_TABLE, NEXT_DIR_TABLE

if TYPE_CHECKING:
    from vit_pytorch.core.levels_info import LevelsInfo


class Cartesian2DRoPE(nn.Module):
    """
    基于物理坐标的 Cartesian 2D Rotary Position Embedding.

    数学形式化
    ==========
    给定位置 i 的物理坐标 p_i = (x_i, y_i)，
    计算绝对角度 θ_i = atan2(y_i, x_i)

    利用三角恒等式进行高效实现:
    - 存储: 每个位置只需 (cos θ_i, sin θ_i)，O(N) 空间
    - 相对角度: θ_ij = θ_j - θ_i
    - cos(θ_ij) = cos(θ_i)cos(θ_j) + sin(θ_i)sin(θ_j)
    - sin(θ_ij) = sin(θ_j)cos(θ_i) - cos(θ_j)sin(θ_i)

    旋转矩阵作用于每对维度 (2d, 2d+1):
        R(θ) = [[cos(θ), -sin(θ)],
                [sin(θ),  cos(θ)]]

    特性
    ----
    - 相对位置编码，不依赖绝对位置
    - O(N) 空间复杂度（无需存储 N² 角度矩阵）
    - 与 Manifold Bias 正交，可叠加

    参数
    ----
    dim : int
        向量维度 D（必须为偶数）
    theta : float
        基础频率，默认 10000.0
    """

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim
        self.theta = theta

        # 预计算频率（用于高效计算）
        # freqs[i] = theta^(-2i/dim)
        freqs = theta ** (-2 * torch.arange(0, dim // 2, 2).float() / dim)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(
        self,
        coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算每个位置的 (cos θ, sin θ) 用于后续注意力计算。

        参数
        ----
        coords : torch.Tensor
            物理坐标 [B, N, 2]，格式 (x, y)，归一化到 [0, 1)

        返回
        ----
        Tuple[torch.Tensor, torch.Tensor]
            (cos_θ, sin_θ)，每个 [B, N]
        """
        # 计算每个位置的绝对角度 θ_i = atan2(y_i, x_i)
        angles = torch.atan2(coords[..., 1], coords[..., 0])  # [B, N]

        # 预计算 cos 和 sin
        cos_θ = torch.cos(angles)  # [B, N]
        sin_θ = torch.sin(angles)  # [B, N]

        # 存储用于后续应用
        self._last_cos = cos_θ.detach()
        self._last_sin = sin_θ.detach()
        self._last_coords = coords.detach()

        return cos_θ, sin_θ

    def apply_rotation(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos_θ: torch.Tensor,
        sin_θ: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将 2D RoPE 旋转应用到 Q 和 K。

        公式（应用于每对维度）:
            x' = cos(φ) * x_{2d} - sin(φ) * x_{2d+1}
            x'' = sin(φ) * x_{2d} + cos(φ) * x_{2d+1}

        其中 φ = θ * freq，θ 是位置角度，freq 是频率。

        参数
        ----
        q : torch.Tensor
            Query 向量 [B, H, N, d]
        k : torch.Tensor
            Key 向量 [B, H, N, d]
        cos_θ : torch.Tensor
            每个位置的 cos(θ_i), [B, N]
        sin_θ : torch.Tensor
            每个位置的 sin(θ_i), [B, N]

        返回
        ----
        Tuple[torch.Tensor, torch.Tensor]
            旋转后的 (q, k)
        """
        B, H, N, D = q.shape
        dim_pairs = D // 2

        # 计算相位 φ = θ * freq
        # 标准 RoPE: 每对维度 (2i, 2i+1) 应用角度 θ_i = theta^(-2i/D)
        # cos_θ: [B, N] -> [B, 1, N, 1]
        # freqs: [dim_pairs] 频率序列
        cos_θ = cos_θ.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
        sin_θ = sin_θ.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
        # 正确公式: theta^(-2i/D) for i = 0, 1, ..., dim_pairs-1
        freqs = self.theta ** (-2 * torch.arange(dim_pairs, device=q.device, dtype=q.dtype).float() / D)
        freqs = freqs.view(1, 1, 1, dim_pairs)  # [1, 1, 1, dim_pairs]

        cos_phi = cos_θ * freqs
        sin_phi = sin_θ * freqs

        # 重塑 q 和 k 为维度对
        q_pairs = q.reshape(B, H, N, dim_pairs, 2)  # [B, H, N, d//2, 2]
        k_pairs = k.reshape(B, H, N, dim_pairs, 2)

        # 应用旋转到 q
        # q' = cos(φ) * q_{even} - sin(φ) * q_{odd}
        # q'' = sin(φ) * q_{even} + cos(φ) * q_{odd}
        q_rot = torch.empty_like(q)
        q_rot[..., 0::2] = cos_phi * q_pairs[..., 0] - sin_phi * q_pairs[..., 1]
        q_rot[..., 1::2] = sin_phi * q_pairs[..., 0] + cos_phi * q_pairs[..., 1]

        # 应用旋转到 k
        k_rot = torch.empty_like(k)
        k_rot[..., 0::2] = cos_phi * k_pairs[..., 0] - sin_phi * k_pairs[..., 1]
        k_rot[..., 1::2] = sin_phi * k_pairs[..., 0] + cos_phi * k_pairs[..., 1]

        return q_rot, k_rot


class DirectionAwareSubspacedRoPE(nn.Module):
    """
    方向感知子空间隔离 RoPE (C+)

    核心特性:
    - 子空间隔离: 每层独立旋转, l > depth 时为 Identity
    - 方向感知: 通过 GEOM_MAP_TABLE 修正象限映射
    - 频率展宽: 防止表征坍塌

    数学保证:
    - P1 模长守恒: ||R(P)h|| = ||h|| (正交矩阵性质)
    - P2 长距离衰减: LCA 零化实现严格内积不变
    - P3 层级一致性: 每层独立频率, 无频域混叠
    - P4 局部保序: 格雷码 + 方向重映射
    - P5 路径可组合: R(P_child) = R(P_parent) · R(Δq)
    - P6 跨尺度兼容: 子空间隔离提供正交衰减
    - H-Crit 方向感知: 查表注入 Hilbert 等变性
    """

    def __init__(
        self,
        dim: int,
        max_level: int = 8,
        base: float = 10000.0,
        gamma: float = None,
        initial_dir: int = 0,
        macro_ratio: float = 0.5,
        macro_base: float = 1000.0,
    ):
        """初始化 DirectionAwareSubspacedRoPE.

        Args:
            dim: 总维度 D (必须为偶数)
            max_level: 最大层级 L_max
            base: RoPE 频率基准 (默认 10000.0)
            gamma: 保留接口, 默认 None (不使用, 子空间隔离本身提供衰减)
            initial_dir: 初始方向状态 (默认 0=Up)
            macro_ratio: 宏观子空间占比 (默认 0.5, 前 macro_ratio*D 维为宏观)
            macro_base: 宏观频率基准 (默认 1000.0, 比 base 小 10x)
        """
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")

        self.dim = dim
        self.max_level = max_level
        self.base = base
        self.gamma = gamma  # 保留接口, 但不使用
        self.initial_dir = initial_dir
        self.macro_ratio = macro_ratio
        self.macro_base = macro_base

        # 子空间维度: D_s = D / L_max
        self.dim_per_subspace = dim // max_level

        # I-NAN: 统一诊断缓存 (用于 embed_output)
        self._diagnostic_cache: Dict[str, Any] = {}

        if self.dim_per_subspace % 2 != 0:
            raise ValueError(
                f"dim_per_subspace ({self.dim_per_subspace}) must be even. "
                f"Consider using dim divisible by 2 * max_level."
            )

        # 基础角度: 格雷码顺序确保局部保序
        # 象限 0,1,2,3 -> 角度 0, π/2, π, 3π/2 (格雷码: 00,01,11,10)
        # 格雷码: G(0)=0, G(1)=1, G(2)=3, G(3)=2
        self.register_buffer(
            'base_angles',
            torch.tensor([0.0, 0.5, 1.5, 1.0]) * torch.pi,  # [4] = [0, π/2, 3π/2, π]
        )

        # 频率展宽因子 - 分段设计:
        # 前 macro_ratio*D_s/2 维: macro_base (低频, 近线性旋转, 模拟物理场)
        # 后 (1-macro_ratio)*D_s/2 维: base (高频, 快旋转, 捕捉拓扑)
        num_freqs = self.dim_per_subspace // 2
        macro_freqs = int(num_freqs * macro_ratio)
        micro_freqs = num_freqs - macro_freqs

        k_indices = torch.arange(num_freqs, dtype=torch.float32)

        # macro 段: ω_k = macro_base^{-2k/D}
        macro_inv_freq = torch.pow(macro_base, -2.0 * k_indices[:macro_freqs] / dim)
        # micro 段: ω_k = base^{-2k/D}
        micro_inv_freq = torch.pow(base, -2.0 * k_indices[macro_freqs:] / dim)

        # 拼接: [D_s/2]
        inv_freq = torch.cat([macro_inv_freq, micro_inv_freq], dim=0)

        # 注册为 buffer (自动设备迁移)
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # 额外存储 macro/micro 分割点 (用于诊断)
        self._macro_freqs = macro_freqs
        self._micro_freqs = micro_freqs

        # 注册查表表 (作为 buffer)
        self.register_buffer('next_dir_table', NEXT_DIR_TABLE, persistent=False)
        self.register_buffer('geom_map_table', GEOM_MAP_TABLE, persistent=False)

        # P2 FIX: 可学习的 Δθ，与 LCA 深度挂钩
        # 深度越大 → 偏置越大 → 方向区分度越高
        # 初始化为极小值(1e-4)，仅作为打破对称性的扰动，避免训练初期破坏已有拓扑序
        num_freqs = self.dim_per_subspace // 2
        self.delta_theta = nn.Parameter(torch.full((max_level + 1, num_freqs), 1e-4))

    def _compute_dirs(self, paths: torch.Tensor) -> torch.Tensor:
        """计算方向状态轨迹.

        Args:
            paths: [B, N, L_max] 象限路径

        Returns:
            dirs: [B, N, L_max] 方向状态轨迹
        """
        B, N, L = paths.shape
        device = paths.device

        # 初始化方向为 initial_dir (默认 0=Up)
        dirs = torch.zeros(B, N, L, dtype=torch.long, device=device)
        dirs[..., 0] = self.initial_dir

        # 递推计算: d_l = NEXT_DIR_TABLE[d_{l-1}, q_{l-1}]
        # 展开循环 (L_max <= 8, torch.compile 友好)
        for level in range(1, L):
            prev_dirs = dirs[:, :, level - 1]  # [B, N]
            prev_quads = paths[:, :, level - 1]  # [B, N]
            dirs[:, :, level] = self.next_dir_table[prev_dirs, prev_quads]

        return dirs

    def _get_standard_quadrants(
        self,
        dirs: torch.Tensor,
        paths: torch.Tensor,
    ) -> torch.Tensor:
        """获取标准象限 (考虑方向状态).

        Args:
            dirs: [B, N, L_max] 方向状态轨迹
            paths: [B, N, L_max] 象限路径

        Returns:
            std_quads: [B, N, L_max] 标准象限 (物理空间)
        """
        # GEOM_MAP_TABLE[dir, q] -> std_quadrant
        return self.geom_map_table[dirs, paths]

    def _build_rotations(
        self,
        std_quads: torch.Tensor,
        depths: torch.Tensor,
    ) -> tuple:
        """构建块对角旋转矩阵.

        Args:
            std_quads: [B, N, L_max] 标准象限
            depths: [B, N] 有效深度

        Returns:
            tuple: (rotations, theta)
                - rotations: [B, N, L_max, D_s/2, 2, 2] 旋转矩阵
                - theta: [B, N, L_max, D_s/2] 旋转角度
        """
        B, N, L = std_quads.shape
        D_s = self.dim_per_subspace
        num_freqs = D_s // 2
        device = std_quads.device

        # 查表获取基础角度: [B, N, L]
        base_theta = self.base_angles[std_quads]  # std_quads ∈ {0,1,2,3}

        # 频率展宽: theta[..., l, k] = base_theta[..., l] * inv_freq[k]
        # base_theta: [B, N, L] -> [B, N, L, 1]
        # inv_freq: [D_s/2] -> [1, 1, 1, D_s/2]
        theta = base_theta.unsqueeze(-1) * self.inv_freq.view(1, 1, 1, num_freqs)

        # P2 FIX: 应用可学习的 Δθ（与深度挂钩）
        # delta_theta: [max_level+1, num_freqs], 深度越大 → 偏置越大
        # depths: [B, N] -> [B, N, 1, 1] 用于索引 delta_theta
        depths_clamped = depths.clamp(0, self.max_level)  # [B, N]
        delta = self.delta_theta[depths_clamped]  # [B, N, num_freqs]
        theta = theta + delta.unsqueeze(2)  # [B, N, L, num_freqs] + [B, N, 1, num_freqs]

        # 应用深度掩码: l > depth 时角度置零 (等价于 Identity)
        level_indices = torch.arange(L, device=device).view(1, 1, L, 1)
        depth_mask = (level_indices < depths.unsqueeze(-1).unsqueeze(-1)).float()
        theta = theta * depth_mask

        # 构建旋转矩阵: [B, N, L, D_s/2, 2, 2]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # 旋转矩阵结构:
        # [[cos, -sin], [sin, cos]]
        # 注意: 使用 float dtype, std_quads.dtype 是 long
        rotations = torch.zeros(
            B, N, L, num_freqs, 2, 2,
            dtype=torch.float32,
            device=device,
        )
        rotations[..., 0, 0] = cos_theta
        rotations[..., 0, 1] = -sin_theta
        rotations[..., 1, 0] = sin_theta
        rotations[..., 1, 1] = cos_theta

        return rotations, theta

    def forward(
        self,
        x: torch.Tensor,
        levels_info: "LevelsInfo",
    ) -> torch.Tensor:
        """应用方向感知子空间隔离 RoPE.

        Args:
            x: [B, H, N, D_head] 输入张量
            levels_info: LevelsInfo 实例

        Returns:
            x_rot: [B, H, N, D_head] 旋转后张量
        """
        B, H, N, D = x.shape
        L = self.max_level
        D_s = D // L

        if D_s * L != D:
            raise ValueError(
                f"Dimension D={D} must be divisible by 2 * max_level={L}. "
                f"Got D_s={D_s} which gives {D_s * L} != D."
            )

        # 1. 获取 paths 和 depths
        paths = levels_info.paths  # [B, N, L]
        depths = levels_info.depths.clamp(0, L).long()  # [B, N]

        # 2. 计算方向状态轨迹
        dirs = self._compute_dirs(paths)  # [B, N, L]

        # 3. 获取标准象限 (考虑方向状态)
        std_quads = self._get_standard_quadrants(dirs, paths)  # [B, N, L]

        # 4. 构建旋转矩阵
        rotations, theta = self._build_rotations(std_quads, depths)
        # rotations: [B, N, L_info, D_s/2, 2, 2]
        # L_info 由 levels_info 的实际 depth 决定（≤ max_level）
        # 对于 l >= L_info 的层级，depth_mask 已将其角度清零（等价于 Identity）

        # 5. 重新排列 x 为子空间视图: [B, H, N, L, D_s]
        # 使用模型的 max_level (L) 来分量子空间，而非 levels_info 的实际深度
        L_info = rotations.shape[2]
        x_sub = x.view(B, H, N, L, D_s)

        # 6. 对每个子空间应用旋转
        # 输出: [B, H, N, L, D_s]
        x_rot = torch.zeros_like(x_sub)

        for level in range(L_info):
            # 获取当前层的旋转矩阵: [B, N, D_s/2, 2, 2]
            rot_l = rotations[:, :, level]  # [B, N, D_s/2, 2, 2]

            # 对每对维度应用旋转
            for k in range(D_s // 2):
                # x_sub[..., l, 2k:2k+2]: [B, H, N, 2]
                # rot_l[..., k]: [B, N, 2, 2]
                x_pair = x_sub[..., level, 2 * k:2 * k + 2]  # [B, H, N, 2]

                # 转置旋转矩阵以匹配 batch matmul: [B, N, 2, 2] @ [B, H, N, 2]^T
                # -> [B, H, N, 2]
                rot_k = rot_l[:, :, k].transpose(-2, -1)  # [B, N, 2, 2]

                # 广播乘法: [B, N, 2, 2] @ [B, H, N, 2]^T -> [B, H, N, 2]
                x_rot[..., level, 2 * k:2 * k + 2] = torch.einsum(
                    'bnij,bhnj->bhni', rot_k, x_pair
                )

        # 7. 重新排列回原始形状: [B, H, N, D]
        x_rot = x_rot.view(B, H, N, D)

        # I-NAN: 计算诊断指标 (仅在训练模式下)
        if self.training:
            with torch.no_grad():
                self._compute_diagnostic_metrics(
                    x=x,
                    x_rot=x_rot,
                    dirs=dirs,
                    std_quads=std_quads,
                    paths=paths,
                    depths=depths,
                    theta=theta,
                )

        return x_rot

    def _compute_diagnostic_metrics(
        self,
        x: torch.Tensor,
        x_rot: torch.Tensor,
        dirs: torch.Tensor,
        std_quads: torch.Tensor,
        paths: torch.Tensor,
        depths: torch.Tensor,
        theta: torch.Tensor = None,
    ) -> None:
        """计算诊断指标并缓存供 embed_output 使用

        Args:
            x: 原始输入 [B, H, N, D]
            x_rot: 旋转后输出 [B, H, N, D]
            dirs: 方向状态轨迹 [B, N, L]
            std_quads: 标准象限 [B, N, L]
            paths: 原始象限路径 [B, N, L]
            depths: 有效深度 [B, N]
            theta: 旋转角度 [B, N, L, D_s/2] (可选)
        """
        B, N, L = dirs.shape

        # ===== A. 方向状态诊断 (directions/*) =====
        # 方向分布: 0=Up, 1=Right, 2=Down, 3=Left
        dir_counts = torch.zeros(4, device=dirs.device, dtype=torch.float32)
        for d in range(4):
            dir_counts[d] = (dirs == d).sum().float()
        dir_dist = dir_counts / (B * N * L)

        self._diagnostic_cache["directions/dist_u"] = dir_dist[0].item()
        self._diagnostic_cache["directions/dist_r"] = dir_dist[1].item()
        self._diagnostic_cache["directions/dist_d"] = dir_dist[2].item()
        self._diagnostic_cache["directions/dist_l"] = dir_dist[3].item()

        # 方向熵 (信息熵)
        dir_dist_safe = torch.clamp(dir_dist, min=1e-8)
        entropy = -(dir_dist * torch.log(dir_dist_safe)).sum()
        self._diagnostic_cache["directions/entropy"] = entropy.item()

        # 方向状态转换次数
        # d_l != d_{l-1} 的次数
        dir_transitions = (dirs[:, :, 1:] != dirs[:, :, :-1]).sum().float()
        self._diagnostic_cache["directions/state_transitions"] = dir_transitions.item()

        # ===== B. 标准象限映射诊断 (geom/*) =====
        # 被重映射的象限比例: std_quads != paths
        remapped = (std_quads != paths).sum().float()
        self._diagnostic_cache["geom/remapped_ratio"] = (remapped / (B * N * L)).item()

        # 各象限分布
        for q in range(4):
            quad_count = (std_quads == q).sum().float()
            self._diagnostic_cache[f"geom/quad_dist_{q}"] = (quad_count / (B * N * L)).item()

        # 方向大跳转次数 (相邻层级象限差值为2，即对角跳转)
        # 注意: 格雷码相邻差值反映空间局部性
        quad_diff = torch.abs(std_quads[:, :, 1:] - std_quads[:, :, :-1])
        # 在格雷码序列中，相差2表示对角跳转（违反局部性）
        jump_count = ((quad_diff == 2) | (quad_diff == 3)).sum().float()
        # 归一化: 每条路径最多 L-1 次跳转
        self._diagnostic_cache["geom/jump_count"] = (jump_count / (B * N * (L - 1))).item()

        # ===== C. 旋转角度诊断 (rotation/*) =====
        if theta is not None:
            # 有效角度 (非零)
            theta_flat = theta.flatten()
            nonzero_theta = theta_flat[theta_flat != 0]

            self._diagnostic_cache["rotation/angle_mean"] = nonzero_theta.mean().item()
            self._diagnostic_cache["rotation/angle_std"] = nonzero_theta.std().item()

            # 零角度比例
            zero_ratio = (theta == 0).sum().float() / theta.numel()
            self._diagnostic_cache["rotation/zero_angle_count"] = zero_ratio.item()
        else:
            # 如果没有 theta，使用旋转矩阵反推
            self._diagnostic_cache["rotation/angle_mean"] = 0.0
            self._diagnostic_cache["rotation/angle_std"] = 0.0
            self._diagnostic_cache["rotation/zero_angle_count"] = 1.0

        # 深度掩码有效率: 实际使用的子空间比例
        level_indices = torch.arange(L, device=depths.device).view(1, 1, L)
        depth_mask = (level_indices < depths.unsqueeze(-1)).float()  # [B, N, L]
        self._diagnostic_cache["rotation/depth_mask_ratio"] = depth_mask.mean().item()

        # ===== D. 模长守恒验证 (normConservation/*) =====
        # 输入/输出模长
        input_norm = torch.norm(x, dim=-1)  # [B, H, N]
        output_norm = torch.norm(x_rot, dim=-1)  # [B, H, N]

        self._diagnostic_cache["normConservation/input_norm_mean"] = input_norm.mean().item()
        self._diagnostic_cache["normConservation/output_norm_mean"] = output_norm.mean().item()

        # 模长比例 (应接近1.0)
        norm_ratio = output_norm / (input_norm + 1e-6)
        self._diagnostic_cache["normConservation/norm_ratio"] = norm_ratio.mean().item()

    def forward_no_rotate(
        self,
        x: torch.Tensor,
        levels_info: "LevelsInfo",
    ) -> torch.Tensor:
        """返回原始张量 (用于验证).

        Args:
            x: [B, H, N, D_head] 输入张量
            levels_info: LevelsInfo 实例

        Returns:
            x: [B, H, N, D_head] 原始张量
        """
        return x

    def compute_lca_divergence(
        self,
        levels_info: "LevelsInfo",
        sample_size: int = 1024,
    ) -> Dict[str, float]:
        """计算 LCA 分叉衰减指标

        验证拓扑衰减假设：共享前缀越长，旋转内积越小
        预期表现：LCA_depth ↑ → DotProduct ↓ 呈非线性阶梯状

        性能优化：N^2 计算量过大，诊断模式下随机采样 M 对 Token

        Args:
            levels_info: LevelsInfo 实例
            sample_size: 随机采样对数（默认 1024）

        Returns:
            Dict[str, float]: 按 LCA_depth 分桶的平均 DotProduct
        """
        paths = levels_info.paths  # [B, N, L]
        B, N, L = paths.shape

        # 计算所有 token 对的 LCA 深度
        # LCA 深度 = 共同前缀长度 = min(len(path_i), len(path_j)) 直到分叉
        # 使用向量化的方式计算

        # 采样 token 对（同一 batch 内）
        max_samples = min(sample_size, N * N)
        perm = torch.randperm(N * N, device=paths.device)[:max_samples]
        i_idx = perm // N
        j_idx = perm % N

        # 获取采样的 paths: [B, S, L]
        paths_i = paths[:, i_idx, :]  # [B, S, L]
        paths_j = paths[:, j_idx, :]  # [B, S, L]

        # 计算 LCA 深度: 从根开始找到第一个不同的位置
        # 相同为 1，不同为 0，然后 cumsum 找第一个 0
        same = (paths_i == paths_j)  # [B, S, L]
        # 找到第一个不同的位置: cumsum 后第一个 0 的位置
        diff_pos = 1 - same.float()  # [B, S, L]
        lca_depths = diff_pos.cumsum(dim=-1)  # [B, S, L]
        # 第一个 diff 位置 = LCA 深度（因为 0 到 1 的 transition）
        lca_depth = (lca_depths == 0).sum(dim=-1)  # [B, S] = 共同前缀长度

        # 按 LCA_depth 分桶统计
        result = {}
        for d in range(L + 1):
            mask = (lca_depth == d)  # [B, S]
            if mask.sum() > 0:
                result[f"lca_depth_{d}"] = mask.sum().float().item() / mask.numel()
            else:
                result[f"lca_depth_{d}"] = 0.0

        return result

    def check_numerical_stability(self) -> Dict[str, float]:
        """检查 NaN/Inf 和极端值

        用于监控高频子空间在 torch.compile 下的数值稳定性

        Returns:
            Dict[str, float]: {
                "has_nan": bool,
                "has_inf": bool,
                "max_val": float,
                "min_val": float
            }
        """
        result = {
            "has_nan": False,
            "has_inf": False,
            "max_val": float('-inf'),
            "min_val": float('inf'),
        }

        if hasattr(self, 'inv_freq') and self.inv_freq is not None:
            inv_freq = self.inv_freq.detach()
            result["has_nan"] = result["has_nan"] or bool(torch.isnan(inv_freq).any())
            result["has_inf"] = result["has_inf"] or bool(torch.isinf(inv_freq).any())
            result["max_val"] = max(result["max_val"], inv_freq.max().item())
            result["min_val"] = min(result["min_val"], inv_freq.min().item())

        return result

    @property
    @torch._dynamo.disable
    def embed_output(self) -> Dict[str, Any]:
        """DirectionAwareSubspacedRoPE 诊断输出

        命名空间:
            embed/directions/*: 方向状态诊断
            embed/geom/*: 几何映射诊断
            embed/rotation/*: 旋转角度诊断
            embed/normConservation/*: 模长守恒验证

        I-OOM FIX: 使用 Disable & Flush 模式：
        - @torch._dynamo.disable 屏蔽追踪
        - 读取后立即 .cpu().item() 迁移到 CPU
        - 读取后立即清空缓存斩断计算图引用
        """
        output: Dict[str, Any] = {}

        # I-NAN: 从统一诊断缓存合并 forward 中计算的指标
        # I-OOM FIX: 访问后清空缓存，防止累积导致显存泄漏
        if self._diagnostic_cache:
            for k, v in self._diagnostic_cache.items():
                if isinstance(v, torch.Tensor):
                    output[k] = v.detach().cpu().item()
                else:
                    output[k] = v
            self._diagnostic_cache.clear()  # 🌟 立即清空缓存释放计算图

        return output
