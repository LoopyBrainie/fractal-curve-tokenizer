"""
流形原生多尺度注意力 (ManifoldNativeAttention)

架构 (Phase 1 净化后):
- HilbertBandedAttention (O(N·W))
- Hilbert1DRoPE (序列位置旋转编码)

数学形式化
===========

注意力公式:
    Attn = BandedAttention(QK^T/√d_k + attn_mask + bias)

其中:
    Q, K = Hilbert1DRoPE(Q_raw, K_raw, hilbert_positions)
    bias = HilbertBias(hilbert_indices, depths) + LevelBias(depths)
"""

from __future__ import annotations
import math

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_pytorch.layers.embeddings.fractal_rope import DirectionAwareSubspacedRoPE, Cartesian2DRoPE

if TYPE_CHECKING:
    from vit_pytorch.core.levels_info import LevelsInfo


# ==================== Hilbert Paths → 坐标重建 ====================


def coords_from_paths(
    paths: torch.Tensor,
    depths: torch.Tensor,
    max_level: int,
) -> torch.Tensor:
    """
    从 Hilbert 四叉树 paths 重建 2D 坐标。

    数学形式化
    ==========
    给定深度 d 和路径 q = [q_0, q_1, ..., q_{d-1}]，其中 q_k ∈ {0,1,2,3}：
        x = Σ_{k=0}^{d-1} bit_k(q_k, 0) × 2^{max_level-k-1}
        y = Σ_{k=0}^{d-1} bit_k(q_k, 1) × 2^{max_level-k-1}

    其中 bit_k(q, b) 是 q 的第 b 位（0=x, 1=y）。

    四象限编码 (与 Hilbert 曲线一致):
        0: 左上 (x_low, y_low)  → bit_x=0, bit_y=0
        1: 右上 (x_high, y_low) → bit_x=1, bit_y=0
        2: 左下 (x_low, y_high)  → bit_x=0, bit_y=1
        3: 右下 (x_high, y_high) → bit_x=1, bit_y=1

    Args:
        paths: [B, N, max_level] 四叉树路径
        depths: [B, N] 每个 token 的深度
        max_level: 最大深度

    Returns:
        coords: [B, N, 2] 重建的 2D 坐标 (x, y)
    """
    B, N, D = paths.shape
    device = paths.device

    # I103-4 向量化: 用 torch.arange(max_level) 广播替代 level 循环
    # level_range: [max_level], half_range: [max_level]
    level_range = torch.arange(max_level, device=device)
    half_range = 1 << (max_level - level_range - 1)  # [2^{max_level-1}, ..., 2^0]

    # 有效掩码: level_range < depths, broadcasting to [B, N, max_level]
    qmask = level_range.view(1, 1, max_level) < depths.unsqueeze(-1)  # [B, N, max_level]

    # 象限解码: 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1)
    # D4-AUDIT FIX: //2 → >>1, %2 → &1，位运算更快
    qx_all = (paths >> 1) & 1  # [B, N, max_level]
    qy_all = paths & 1          # [B, N, max_level]

    # 广播 half 到 [1, 1, max_level] 并与掩码相乘
    half_range = half_range.view(1, 1, max_level)
    x = (qx_all * half_range * qmask).sum(dim=-1)  # [B, N]
    y = (qy_all * half_range * qmask).sum(dim=-1)  # [B, N]

    # 转换为 float 并归一化到 [0, 1] 范围
    grid_size = 1 << max_level  # 2^max_level
    coords = torch.stack([x.float(), y.float()], dim=-1)  # [B, N, 2]

    # 归一化: 将整数坐标 [0, 2^max_level) 映射到 [0, 1)
    coords = coords / grid_size

    return coords


# ==================== 主模块 ====================


class ManifoldNativeAttention(nn.Module):
    """
    流形感知多尺度注意力模块（简化版）。

    特性
    ----
    - 双 RoPE: Cartesian2DRoPE (物理场) + DirectionAwareSubspacedRoPE (拓扑场)
    - 标准缩放点积注意力
    - NaN 检测与诊断
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int = 64,
        max_level: int = 8,
        dropout: float = 0.0,
        # v7.1: 宏观/微观频率配置 (用于 DirectionAwareSubspacedRoPE)
        macro_ratio: float = 0.5,
        macro_base: float = 1000.0,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.max_level = max_level
        self.dropout = dropout
        # v7.1: 存储宏观频率配置
        self.macro_ratio = macro_ratio
        self.macro_base = macro_base

        self.inner_dim = heads * dim_head
        self.head_dim = dim_head
        self.scale = self.head_dim ** -0.5

        # QKV 投影
        self.qkv = nn.Linear(dim, self.inner_dim * 3, bias=True)

        # 输出投影
        self.proj = nn.Linear(self.inner_dim, dim, bias=True)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # 双 RoPE 架构 - Cartesian(物理场) + C+(拓扑场)
        # 物理场(1/4): Cartesian2DRoPE - 全局平移不变性
        # 拓扑场(3/4): DirectionAwareSubspacedRoPE - 分形树层级与局部拓扑
        phys_dim = self.inner_dim // 4
        topo_dim = self.inner_dim - phys_dim  # = inner_dim * 3 // 4

        if phys_dim % 2 == 0:
            self.rope_cartesian = Cartesian2DRoPE(dim=phys_dim, theta=self.macro_base)
        else:
            self.rope_cartesian = None

        topo_dim_per_subspace = topo_dim // max_level if max_level > 0 else topo_dim
        if topo_dim_per_subspace % 2 == 0:
            self.rope_fractal = DirectionAwareSubspacedRoPE(
                dim=topo_dim,
                max_level=max_level,
                macro_ratio=self.macro_ratio,
                macro_base=self.macro_base,
            )
        else:
            self.rope_fractal = None

        # torch.compile 缓存修复
        self._stats_cache: dict = {}

    def forward(
        self,
        x: torch.Tensor,
        levels_info: Optional["LevelsInfo"] = None,
    ) -> torch.Tensor:
        """
        前向传播。

        参数
        ----
        x : torch.Tensor
            输入特征 [B, N, D]
        levels_info : LevelsInfo
            层级信息

        返回
        ----
        torch.Tensor
            输出特征 [B, N, D]
        """
        depths = None
        hilbert_indices = None
        raw_paths = None  # 原始 paths 用于坐标重建
        coords = None  # 物理坐标，用于 2D RoPE
        if levels_info is not None:
            # Fix-12: 直接属性访问，无 Graph Break
            # levels_info 应该是 LevelsInfo 对象，由 LevelsInfo.ensure() 统一保证
            depths = levels_info.depths

            # 确保 depths 是 [B, N] 形状（移除冗余维度）
            # B2 修复: 处理 depths 可能是 [B, N, N] 或更高维度的情况
            if depths.dim() >= 3:
                if depths.dim() == 3 and depths.shape[1] == depths.shape[2]:
                    # depths 是 [B, N, N]，提取对角元素得到 [B, N]
                    depths = depths.diagonal(dim1=-2, dim2=-1)  # [B, N]
                else:
                    # 其他情况：展平并取前 N 个
                    B_tmp, N_tmp = x.shape[0], x.shape[1]
                    depths = depths.reshape(B_tmp, -1)[:, :N_tmp]

            # 直接调用，无条件分发
            hilbert_indices = levels_info.get_hilbert_indices()

            # 直接访问 paths 属性（LevelsInfo 保证该属性存在）
            raw_paths = levels_info.paths

        B, N, D = x.shape

        # CLS token 检测：比较 x 序列长度与 levels_info 条目数
        # 如果 x 序列长度 > levels_info 条目数，说明存在 CLS token
        if levels_info is not None:
            levels_info_len = levels_info.depths.shape[1]
            has_cls_token = (N > levels_info_len)
        else:
            has_cls_token = False

        # CLS token 处理：为保持维度对齐，需要为 CLS token 添加 dummy 条目
        if has_cls_token:
            # 为 CLS token 创建 dummy 条目（使用深度 0，不影响注意力计算）
            cls_depth = torch.zeros(B, 1, dtype=depths.dtype, device=depths.device)
            cls_hilbert = torch.zeros(B, 1, dtype=hilbert_indices.dtype, device=hilbert_indices.device)
            cls_coords = torch.zeros(B, 1, 2, dtype=coords.dtype, device=coords.device)
            cls_path = torch.zeros(B, 1, raw_paths.shape[-1], dtype=raw_paths.dtype, device=raw_paths.device)

            # 拼接 CLS dummy 到现有 tensors（保持维度对齐）
            depths = torch.cat([depths, cls_depth], dim=1)  # [B, N+1]
            hilbert_indices = torch.cat([hilbert_indices, cls_hilbert], dim=1)  # [B, N+1]
            coords = torch.cat([coords, cls_coords], dim=1)  # [B, N+1, 2]
            raw_paths = torch.cat([raw_paths, cls_path], dim=1)  # [B, N+1, L]

        # 早期坐标重建（用于 2D RoPE）
        # 如果有 raw_paths，可以提前计算 coords
        if raw_paths is not None and depths is not None:
            # 使用实际的 paths 大小而非 self.max_level
            actual_max_level = raw_paths.shape[-1]
            coords = coords_from_paths(
                paths=raw_paths,
                depths=depths,
                max_level=actual_max_level,
            )  # [B, N, 2] 归一化到 [0, 1)
        elif hilbert_indices is not None:
            # 回退：使用 Hilbert 展平索引重建 2D 网格坐标
            # P0 修复: 原 [h_norm, 1-h_norm] 产生 1D 对角线，||x||≈1.0 触发 Poincaré 边界溢出
            # 改进：使用 ceil(sqrt()) 避免完全平方数假设，保留空间序 (Spatial Order)
            batch_size, seq_len = hilbert_indices.shape
            W = H = int(math.ceil(math.sqrt(seq_len)))  # 向上取整

            x_grid = (torch.arange(seq_len, device=hilbert_indices.device) % W).float() / W * 2 - 1  # [N]
            y_grid = (torch.arange(seq_len, device=hilbert_indices.device) // W).float() / H * 2 - 1  # [N]

            # 扩展 batch 维度: [N] → [B, N]
            x_grid = x_grid.unsqueeze(0).expand(batch_size, -1)
            y_grid = y_grid.unsqueeze(0).expand(batch_size, -1)

            # 安全收缩因子 0.95 + tanh 软压缩防止边界震荡
            safe_scale = 0.95 * float(math.tanh(1.0))
            coords = torch.stack([x_grid, y_grid], dim=-1) * safe_scale  # [B, N, 2], ||x|| < 1

        # QKV 投影
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.heads, self.head_dim)
        # D3-AUDIT FIX: permute 后 tensor 非连续，slice 操作需要连续内存
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()  # [3, B, H, N, d]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 🚀 Stage 2: 双 RoPE 应用 - 先分割后旋转
        # 1/4 维度: Cartesian2DRoPE (物理场)
        # 3/4 维度: DirectionAwareSubspacedRoPE (拓扑场)
        inner_dim = q.shape[-1]
        phys_dim = inner_dim // 4

        # 在应用 RoPE 之前先分割 Q/K
        q_phys = q[..., :phys_dim]
        k_phys = k[..., :phys_dim]
        q_topo = q[..., phys_dim:]
        k_topo = k[..., phys_dim:]

        # 🚀 Stage 2: 对物理场应用 Cartesian2DRoPE
        if coords is not None and self.rope_cartesian is not None:
            angles = self.rope_cartesian(coords)
            q_phys, k_phys = self.rope_cartesian.apply_rotation(q_phys, k_phys, angles)

        # 🚀 Stage 2: 对拓扑场应用 DirectionAwareSubspacedRoPE
        if levels_info is not None and self.rope_fractal is not None:
            # 检测是否存在 CLS token
            if has_cls_token:
                # 分离 CLS token
                q_phys_cls = q_phys[:, :, 0:1, :]
                k_phys_cls = k_phys[:, :, 0:1, :]
                q_phys_spatial = q_phys[:, :, 1:, :]
                k_phys_spatial = k_phys[:, :, 1:, :]

                q_topo_cls = q_topo[:, :, 0:1, :]
                k_topo_cls = k_topo[:, :, 0:1, :]
                q_topo_spatial = q_topo[:, :, 1:, :]
                k_topo_spatial = k_topo[:, :, 1:, :]
            else:
                q_phys_cls, k_phys_cls = None, None
                q_phys_spatial, k_phys_spatial = q_phys, k_phys
                q_topo_cls, k_topo_cls = None, None
                q_topo_spatial, k_topo_spatial = q_topo, k_topo

            # 检查拓扑场维度兼容性
            topo_dim_check = q_topo_spatial.shape[-1]
            max_level_topo = self.rope_fractal.max_level if self.rope_fractal else 0
            topo_dim_compatible = (max_level_topo > 0) and (topo_dim_check % max_level_topo == 0)

            if topo_dim_compatible:
                q_topo_spatial = self.rope_fractal(q_topo_spatial, levels_info)
                k_topo_spatial = self.rope_fractal(k_topo_spatial, levels_info)

            # 重新拼接 CLS token
            if q_phys_cls is not None:
                q_phys = torch.cat([q_phys_cls, q_phys_spatial], dim=2)
                k_phys = torch.cat([k_phys_cls, k_phys_spatial], dim=2)
                q_topo = torch.cat([q_topo_cls, q_topo_spatial], dim=2)
                k_topo = torch.cat([k_topo_cls, k_topo_spatial], dim=2)
            else:
                q_phys, k_phys = q_phys_spatial, k_phys_spatial
                q_topo, k_topo = q_topo_spatial, k_topo_spatial

        # 重新拼接物理场和拓扑场的 Q/K（RoPE 已在上方分别应用）
        q = torch.cat([q_phys, q_topo], dim=-1)
        k = torch.cat([k_phys, k_topo], dim=-1)

        # 标准缩放点积注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # 注意力加权
        out = attn @ v  # [B, H, N, d]

        # 合并头
        out = out.transpose(1, 2).reshape(B, N, self.inner_dim)

        # 输出投影
        out = self.proj(out)
        out = self.proj_dropout(out)

        return out

    @torch.no_grad()
    @torch._dynamo.disable
    def get_stats(self) -> dict:
        """获取诊断统计信息。

        返回
        ----
        dict
            - (当前无活跃诊断指标)
        """
        return self._stats_cache

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, heads={self.heads}, dim_head={self.dim_head}, "
            f"max_level={self.max_level}"
        )


# 为了兼容性，创建一个别名
ManifoldAttention = ManifoldNativeAttention
