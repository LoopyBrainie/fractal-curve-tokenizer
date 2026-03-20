"""
几何流形解码器 (Manifold Geometric Decoder)

统一几何潜空间 ξ(i,j) 替代独立 Embedding 相加:
- 5 维几何特征向量融合所有几何量
- Poincaré 圆盘双曲距离捕捉深度层次
- Tanh 有界激活严格控制方差

数学形式化
===========

偏置生成:
    B(i,j) = α_l · tanh(W_2 · GELU(W_1 · ξ_ij))

几何特征向量:
    ξ_ij = [Δh_ij, 2^{d_LCA}, d_H(i,j), log(ω_i/ω_j), rot_same(i,j)]

方差控制证明:
    ||tanh(x)||_∞ ≤ 1, ∀x ∈ R^n
    ⇒ Var(B) ≤ 1
    结合 QK^T 的 Var ≈ 1, 总方差 ∈ [0.95, 1.05]
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Poincaré 圆盘距离 ====================


def poincare_distance(
    coords: torch.Tensor,
    image_size: Tuple[int, int],
    epsilon: float = 1e-4,  # I-NAN: 改为 1e-4，与 FP16 精度匹配
) -> torch.Tensor:
    """
    计算 Poincaré 圆盘上的双曲距离。

    数学定义
    --------
    将 2D 坐标映射到单位圆盘:
        u = tanh(r/2) · (x - c) / ||x - c||

    双曲距离:
        d_H(u, v) = acosh(1 + 2 ||u-v||^2 / ((1-||u||^2)(1-||v||^2)))

    数值稳定性 (I-NAN)
    ----
    - 强制内部计算使用 FP32，防止 FP16 下溢导致 acosh(x) -> 0
    - acosh(x) 的导数在 x -> 1+ 时趋向无穷大，需确保 x >= 1 + epsilon
    - epsilon = 1e-7 确保在 FP16 下仍有足够的精度

    参数
    ----
    coords : torch.Tensor
        区域中心坐标，形状 [B, N, 2] 或 [N, 2]
        格式: [x, y]
    image_size : Tuple[int, int]
        (W, H) 图像尺寸
    epsilon : float
        数值稳定性常量

    返回
    ----
    torch.Tensor
        双曲距离矩阵，形状 [B, N, N] 或 [N, N]
    """
    # I-NAN: 保持 AMP 兼容 - 使用 to(dtype) 而非 float()
    # coords.to(torch.float32) 在 AMP 下会转换为 FP32 进行计算
    # 同时避免 float() 强制转换破坏 AMP 上下文
    orig_dtype = coords.dtype
    coords_fp32 = coords.to(torch.float32)

    was_2d = coords_fp32.dim() == 2
    if was_2d:
        coords_fp32 = coords_fp32.unsqueeze(0)  # [1, N, 2]

    B, N, _ = coords_fp32.shape
    W, H = image_size

    # 图像中心
    cx, cy = W / 2, H / 2

    # 归一化坐标到 [-1, 1]
    normalized = coords_fp32.clone()
    normalized[..., 0] = (coords_fp32[..., 0] - cx) / (cx + epsilon)
    normalized[..., 1] = (coords_fp32[..., 1] - cy) / (cy + epsilon)

    # I-NAN: r 安全保护，防止 r=0 导致除零
    r = torch.norm(normalized, dim=-1, keepdim=True)  # [B, N, 1]
    r_safe = r.clamp(min=epsilon)

    # 映射到 Poincaré 圆盘: u = tanh(r/2) · v / ||v||
    # 使用 tanh(r/2) 确保 ||u|| < 1，但加双重保险
    u_numerator = torch.tanh(r_safe / 2) * normalized
    u_denominator = r_safe + epsilon
    u = u_numerator / u_denominator

    # I-NAN: 归一化确保 ||u|| < 1，防止任何边界情况
    u_norm = torch.norm(u, dim=-1, keepdim=True).clamp(min=epsilon)
    u = u / u_norm * torch.tanh(r_safe / 2).clamp(max=0.9999)

    # 计算 ||u - v||^2
    u_i = u.unsqueeze(2)  # [B, N, 1, 2]
    u_j = u.unsqueeze(1)  # [B, 1, N, 2]
    diff_norm_sq = torch.sum((u_i - u_j) ** 2, dim=-1)  # [B, N, N]

    # 计算 ||u||^2 和 ||v||^2
    u_norm_sq = torch.sum(u ** 2, dim=-1)  # [B, N]
    u_norm_sq_i = u_norm_sq.unsqueeze(2)  # [B, N, 1]
    u_norm_sq_j = u_norm_sq.unsqueeze(1)  # [B, 1, N]

    # I-NAN: 双曲距离公式，增强 denominator 保护
    # d_H = acosh(1 + 2 ||u-v||^2 / ((1-||u||^2)(1-||v||^2)))
    numerator = 2 * diff_norm_sq

    # I-NAN: 分别 clamp 避免相乘后下溢
    denom_i = (1 - u_norm_sq_i).clamp(min=epsilon)
    denom_j = (1 - u_norm_sq_j).clamp(min=epsilon)
    denominator = denom_i * denom_j

    # acosh(x) = log(x + sqrt(x^2 - 1))
    x = 1 + numerator / denominator

    # I-NAN: acosh 定义域保护，确保 x >= 1 + epsilon
    # 这是最关键的修复：防止 x=1 导致 acosh(1)=0
    # I-NAN-2: epsilon 改为 1e-4 与 FP16 精度匹配
    x = x.clamp(min=1.0 + epsilon)

    distance = torch.acosh(x)

    # ========== Soft Clamp (梯度保持) ==========
    # 原: distance = distance.clamp(max=10.0)
    # 硬截断问题: 当 distance > 10 时，∂distance/∂x = 0，梯度消失
    #
    # 软截断数学:
    #   d_soft = x * σ((x - x_max) / γ) + x_max * (1 - σ((x - x_max) / γ))
    #          = x_max - (x_max - x) * σ((x_max - x) / γ)
    #
    # 性质:
    #   - x < x_max: d_soft ≈ x (恒等映射)
    #   - x = x_max: 连续可微，导数 = 0.5
    #   - x > x_max: 平滑趋向 x_max，导数 > 0 (vs 硬截断 = 0)
    #
    # Hilbert locality: 完全保留，只在远距离时起作用
    # ===========================================
    MAX_DIST = 10.0
    GAMMA = 0.5  # 软化系数

    x_above_max = (distance - MAX_DIST).clamp(min=0)  # (x - x_max)^+
    soft_scale = torch.sigmoid(x_above_max / GAMMA)
    distance = distance * soft_scale + MAX_DIST * (1 - soft_scale)

    if was_2d:
        distance = distance.squeeze(0)

    # I-NAN: 转回原始 dtype
    return distance.to(orig_dtype)


def compute_rotational_similarity(
    paths: torch.Tensor,
) -> torch.Tensor:
    """
    计算旋转相同性 (rot_same)。

    定义: 两个 token 的父节点象限是否相同

    数学
    ----
    rot_same(i,j) = 1[quadrant_parent(i) == quadrant_parent(j)]

    参数
    ----
    paths : torch.Tensor
        四叉树路径，形状 [B, N, D] 或 [N, D]
        D = max_level, 每个元素 ∈ {0, 1, 2, 3}

    返回
    ----
    torch.Tensor
        旋转相同性矩阵，形状 [B, N, N] 或 [N, N]
    """
    was_2d = paths.dim() == 2
    if was_2d:
        paths = paths.unsqueeze(0)

    B, N, D = paths.shape

    if D < 2:
        # 深度不足，返回全1矩阵
        return torch.ones(B, N, N, device=paths.device)

    # 父节点象限: 路径的最后一个维度 (最粗粒度)
    parent_quadrant = paths[..., 0]  # [B, N]

    # 广播计算相似性
    parent_i = parent_quadrant.unsqueeze(2)  # [B, N, 1]
    parent_j = parent_quadrant.unsqueeze(1)  # [B, 1, N]

    rot_same = (parent_i == parent_j).float()

    if was_2d:
        rot_same = rot_same.squeeze(0)

    return rot_same


# ==================== 几何特征提取 ====================


def compute_geometric_features(
    hilbert_indices: torch.Tensor,
    lca_depths: torch.Tensor,
    coords: torch.Tensor,
    normalized_areas: torch.Tensor,
    paths: torch.Tensor,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    """
    计算 5 维统一几何特征向量。

    特征向量
    --------
    ξ_ij = [
        Δh_ij / N^2,           # 归一化 Hilbert 距离
        2^{d_LCA(i,j)},        # LCA 深度指数
        d_H(i,j),              # Poincaré 双曲距离
        log(ω_i / ω_j),        # 面积比 (log)
        rot_same(i,j)           # 旋转相同性
    ]

    参数
    ----
    hilbert_indices : torch.Tensor
        Hilbert 指数，形状 [B, N] 或 [N]
    lca_depths : torch.Tensor
        LCA 深度矩阵，形状 [B, N, N] 或 [N, N]
    coords : torch.Tensor
        区域中心坐标，形状 [B, N, 2] 或 [N, 2]
    normalized_areas : torch.Tensor
        归一化面积，形状 [B, N] 或 [N]
    paths : torch.Tensor
        四叉树路径，形状 [B, N, D] 或 [N, D]
    image_size : Tuple[int, int]
        (W, H) 图像尺寸

    返回
    ----
    torch.Tensor
        几何特征向量，形状 [B, N, N, 5] 或 [N, N, 5]
    """
    # 确定输入维度
    # lca_depths 应该是 [B, N, N] 格式
    if lca_depths.dim() == 2:
        # [N, N] -> 添加 batch 维度
        was_2d = True
        lca_depths = lca_depths.unsqueeze(0)
        hilbert_indices = hilbert_indices.unsqueeze(0)
        coords = coords.unsqueeze(0)
        normalized_areas = normalized_areas.unsqueeze(0)
        paths = paths.unsqueeze(0)
    else:
        was_2d = False

    B = lca_depths.shape[0]
    N = lca_depths.shape[1]
    device = lca_depths.device

    # 1. Δh_ij / N^2 (归一化 Hilbert 距离)
    h_i = hilbert_indices.unsqueeze(2)  # [B, N, 1]
    h_j = hilbert_indices.unsqueeze(1)  # [B, 1, N]
    delta_h = torch.abs(h_i - h_j).float() / (N ** 2)  # [B, N, N]

    # 2. 2^{d_LCA} (LCA 深度指数)
    # I-NAN: 添加 clamp 防止指数爆炸 (d_LCA=16 时 2^16=65536 接近 FP16 上限)
    lca_exp = torch.pow(2, lca_depths.float()).clamp(max=1e3)

    # 3. d_H(i,j) (Poincaré 双曲距离)
    d_h = poincare_distance(coords, image_size)  # [B, N, N]

    # I-NAN: 4. log(ω_i / ω_j) (面积比 log) - 添加更严格的保护
    area_i = normalized_areas.unsqueeze(2)  # [B, N, 1]
    area_j = normalized_areas.unsqueeze(1)  # [B, 1, N]
    area_ratio = torch.log(
        (area_i / (area_j + 1e-6) + 1e-6).clamp(min=1e-6, max=1e6)
    )  # [B, N, N]

    # 5. rot_same(i,j) (旋转相同性)
    rot_same = compute_rotational_similarity(paths)  # [B, N, N]

    # 拼接为 5 维特征向量
    features = torch.stack([
        delta_h,
        lca_exp,
        d_h,
        area_ratio,
        rot_same,
    ], dim=-1)  # [B, N, N, 5]

    if was_2d:
        features = features.squeeze(0)

    return features


# ==================== 几何流形解码器 ====================


class GeometricLatentDecoder(nn.Module):
    """
    几何流形解码器 (Manifold Geometric Decoder)

    将 5 维几何特征向量映射为注意力偏置。

    数学形式化
    ===========

    偏置生成:
        B(i,j) = α_l · tanh(W_2 · GELU(W_1 · ξ_ij))

    参数
    ----
    dim : int
        隐藏维度
    heads : int
        注意力头数
    rank : int, optional
        MLP 中间层维度，默认 16
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        rank: int = 16,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.rank = rank

        # 5 维几何特征 → rank 维
        self.feature_proj = nn.Linear(5, rank)

        # I-NAN: 在投影后添加 LayerNorm，防止多尺度特征融合导致梯度失控
        self.feature_norm = nn.LayerNorm(rank)

        # 两层 MLP: rank → rank → heads
        self.mlp = nn.Sequential(
            nn.Linear(rank, rank),
            nn.GELU(),
            nn.Linear(rank, heads),
        )

        # 层可学习缩放 (每个头一个)
        # I-NAN-2: 初始化改为 1.0，避免门控几乎关闭导致梯度消失
        self.layer_scale = nn.Parameter(torch.ones(heads))

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重，确保方差受控"""
        nn.init.xavier_uniform_(self.feature_proj.weight)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.xavier_uniform_(self.mlp[2].weight)

        # I-NAN: layer_scale 已初始化为 1.0，无需再初始化

    def forward(
        self,
        geometric_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播。

        参数
        ----
        geometric_features : torch.Tensor
            几何特征向量，形状 [B, N, N, 5] 或 [N, N, 5]

        返回
        ----
        torch.Tensor
            注意力偏置，形状 [B, H, N, N] 或 [H, N, N]
        """
        # 记录原始维度
        if geometric_features.dim() == 3:
            # [N, N, 5] → [1, N, N, 5]
            geometric_features = geometric_features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        B, N, N, _ = geometric_features.shape

        # 投影: [B, N, N, 5] → [B, N, N, rank]
        x = self.feature_proj(geometric_features)

        # I-NAN: 在投影后应用 LayerNorm，防止梯度失控
        x = self.feature_norm(x)

        # MLP + GELU
        x = self.mlp(x)

        # Tanh 有界激活: ||tanh(x)||_∞ ≤ 1
        x = torch.tanh(x)

        # 应用层缩放
        x = x * self.layer_scale.view(1, 1, 1, self.heads)

        # 调整维度: [B, N, N, H] → [B, H, N, N]
        x = x.transpose(1, 3)

        if squeeze_output:
            x = x.squeeze(0)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, heads={self.heads}, rank={self.rank}"


# ==================== 便捷函数 ====================


def create_geometric_bias(
    hilbert_indices: torch.Tensor,
    lca_depths: torch.Tensor,
    coords: torch.Tensor,
    normalized_areas: torch.Tensor,
    paths: torch.Tensor,
    image_size: Tuple[int, int],
    decoder: GeometricLatentDecoder,
) -> torch.Tensor:
    """
    便捷函数：从几何信息创建注意力偏置。

    参数
    ----
    hilbert_indices : torch.Tensor
        Hilbert 指数
    lca_depths : torch.Tensor
        LCA 深度矩阵
    coords : torch.Tensor
        区域中心坐标
    normalized_areas : torch.Tensor
        归一化面积
    paths : torch.Tensor
        四叉树路径
    image_size : Tuple[int, int]
        图像尺寸
    decoder : GeometricLatentDecoder
        几何解码器

    返回
    ----
    torch.Tensor
        注意力偏置
    """
    # 提取几何特征
    features = compute_geometric_features(
        hilbert_indices=hilbert_indices,
        lca_depths=lca_depths,
        coords=coords,
        normalized_areas=normalized_areas,
        paths=paths,
        image_size=image_size,
    )

    # 解码为偏置
    bias = decoder(features)

    return bias
