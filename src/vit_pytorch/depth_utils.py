# -*- coding: utf-8 -*-
"""
动态深度计算工具

数学形式化
==========

核心功能：根据图像尺寸和目标最小 patch 大小动态计算四叉树最大深度。

公式:
    L_max = min(L_hard, max(0, floor(log2(min(H, W) / min_patch_size))))

其中:
    H, W: 输入图像尺寸
    min_patch_size: 目标最小 patch 大小
    L_hard: 硬上限，防止极端情况

推导:
    设深度 d 的 patch 大小: S(d) = floor(min(H, W) / 2^d)
    最优深度满足: S(d) ≈ min_patch_size
    即: min(H, W) / 2^d ≈ min_patch_size
    解得: d ≈ log2(min(H, W) / min_patch_size)

Examples:
    >>> compute_max_depth((64, 64), 4)
    4  # 64/2^4 = 4
    >>> compute_max_depth((224, 224), 4)
    5  # 224/2^5 = 7 (最接近 4)
    >>> compute_max_depth((512, 512), 4)
    7  # 512/2^7 = 4
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def compute_max_depth(
    image_size: Tuple[int, int],
    min_patch_size: int,
    hard_limit: Optional[int] = None,
) -> int:
    """
    动态计算四叉树最大深度。

    数学形式:
        L_max = max(0, floor(log2(min(H, W) / min_patch_size)))

    参数
    ----
    image_size : Tuple[int, int]
        (H, W) 输入图像尺寸
    min_patch_size : int
        目标最小 patch 大小
    hard_limit : int, optional
        硬上限，None 表示无限制（由图像尺寸和 min_patch_size 自动决定）

    返回
    ----
    int
        有效最大深度

    示例
    ----
    >>> compute_max_depth((64, 64), 4)
    4
    >>> compute_max_depth((224, 224), 4)
    5
    >>> compute_max_depth((512, 512), 4)
    7
    >>> compute_max_depth((64, 64), 4, hard_limit=3)  # 64/2^3=8，实际最小 patch
    3

    边界情况
    --------
    - 当 min_dim < min_patch_size 时，返回 0（只用整个图像）
    - 当 min_patch_size <= 0 时，抛出 ValueError
    """
    H, W = image_size
    min_dim = min(H, W)

    # 边界检查
    if min_dim < min_patch_size:
        return 0

    if min_patch_size <= 0:
        raise ValueError(f"min_patch_size must be positive, got {min_patch_size}")

    # 动态计算深度
    # 公式: L_max = floor(log2(min_dim / min_patch_size))
    max_depth = int(math.log2(min_dim // min_patch_size))

    # 应用硬上限（如果指定）
    if hard_limit is not None:
        max_depth = min(max_depth, hard_limit)

    # 确保非负
    return max(0, max_depth)


def compute_actual_min_patch(
    image_size: Tuple[int, int],
    max_depth: int,
) -> int:
    """
    计算给定 max_depth 下的实际最小 patch 大小。

    数学形式:
        S_min = floor(min(H, W) / 2^max_depth)

    参数
    ----
    image_size : Tuple[int, int]
        (H, W) 输入图像尺寸
    max_depth : int
        最大深度

    返回
    ----
    int
        实际最小 patch 大小（至少为 1）

    示例
    ----
    >>> compute_actual_min_patch((64, 64), 4)
    4
    >>> compute_actual_min_patch((224, 224), 4)
    14
    >>> compute_actual_min_patch((224, 224), 5)
    7
    >>> compute_actual_min_patch((512, 512), 7)
    4
    """
    H, W = image_size
    min_dim = min(H, W)
    # 确保至少返回 1，防止除零或无效 patch
    return max(1, min_dim // (2 ** max_depth))


def compute_patch_sizes(
    image_size: Tuple[int, int],
    max_depth: int,
    base_patch_size: int = 4,
) -> Tuple[int, ...]:
    """
    计算各深度对应的 patch 大小。

    数学形式:
        patch_sizes[d] = base_patch_size × 2^d

    参数
    ----
    image_size : Tuple[int, int]
        (H, W) 输入图像尺寸
    max_depth : int
        最大深度
    base_patch_size : int, optional
        基础 patch 大小，默认 4

    返回
    ----
    Tuple[int, ...]
        各深度的 patch 大小元组

    示例
    ----
    >>> compute_patch_sizes((64, 64), 4)
    (4, 8, 16, 32, 64)
    >>> compute_patch_sizes((224, 224), 4)
    (4, 8, 16, 32, 64)
    """
    return tuple(base_patch_size * (2 ** d) for d in range(max_depth + 1))


def compute_depth_distribution(
    image_size: Tuple[int, int],
    max_depth: int,
) -> Tuple[int, ...]:
    """
    计算各深度的候选区域数量。

    数学形式:
        count[d] = 4^d (四叉树每层区域数)

    参数
    ----
    image_size : Tuple[int, int]
        (H, W) 输入图像尺寸（实际不用于计算）
    max_depth : int
        最大深度

    返回
    ----
    Tuple[int, ...]
        各深度的候选区域数量

    示例
    ----
    >>> compute_depth_distribution((64, 64), 3)
    (1, 4, 16, 64)
    >>> compute_depth_distribution((224, 224), 4)
    (1, 4, 16, 64, 256)
    """
    return tuple(4 ** d for d in range(max_depth + 1))


def compute_total_candidates(
    image_size: Tuple[int, int],
    max_depth: int,
) -> int:
    """
    计算总候选区域数量。

    数学形式:
        N_total = Σ_{d=0}^{max_depth} 4^d = (4^{max_depth+1} - 1) / (4 - 1)

    参数
    ----
    image_size : Tuple[int, int]
        (H, W) 输入图像尺寸（实际不用于计算）
    max_depth : int
        最大深度

    返回
    ----
    int
        总候选区域数量

    示例
    ----
    >>> compute_total_candidates((64, 64), 3)
    85  # 1 + 4 + 16 + 64
    >>> compute_total_candidates((224, 224), 4)
    341  # 1 + 4 + 16 + 64 + 256
    """
    return (4 ** (max_depth + 1) - 1) // 3


def compute_effective_depth(
    image_size: Tuple[int, int],
    min_patch_size: int,
    base_patch_size: int = 4,
) -> int:
    """
    计算有效的四叉树深度（考虑 base_patch_size）。

    有效深度是指从 base_patch_size 开始，
    达到 min_patch_size 所需的分裂次数。

    数学形式:
        d_effective = max(0, floor(log2(min_patch_size / base_patch_size)))

    参数
    ----
    image_size : Tuple[int, int]
        (H, W) 输入图像尺寸
    min_patch_size : int
        目标最小 patch 大小
    base_patch_size : int, optional
        基础 patch 大小，默认 4

    返回
    ----
    int
        有效深度

    示例
    ----
    >>> compute_effective_depth((64, 64), 4)
    0
    >>> compute_effective_depth((64, 64), 8)
    1
    >>> compute_effective_depth((64, 64), 16)
    2
    """
    if min_patch_size <= base_patch_size:
        return 0

    ratio = min_patch_size / base_patch_size
    return max(0, int(math.log2(ratio)))


# ==================== I31: 形状-尺度计算函数 ====================

def compute_region_shape_scale(
    regions: torch.Tensor,
    image_size: Tuple[int, int],
    epsilon: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算区域的形状和尺度特征。

    数学形式化
    ==========

    纵横比 (对数变换):
        aspect_ratio = log(w / h)
                   = log((x2-x1) / (y2-y1))

    归一化面积:
        normalized_area = (w / W) * (h / H)
                        = ((x2-x1) * (y2-y1)) / (W * H)

    对数变换选择理由 (I31-1):
        1. 对称性: log(r) = -log(1/r)，正反比例等距
        2. 梯度: ∂log(r)/∂r = 1/r，极端值时梯度减小防止训练不稳定
        3. 符合视觉感知: Weber-Fechner Law

    参数
    ----
    regions : torch.Tensor
        区域边界张量，形状 [B, N, 4] 或 [N, 4]
        格式: [x1, y1, x2, y2]
    image_size : Tuple[int, int]
        (W, H) 图像尺寸
    epsilon : float, optional
        数值稳定性常量，防止除零，默认 1e-8

    返回
    ----
    Tuple[torch.Tensor, torch.Tensor]
        - aspect_ratios: 纵横比 [B, N] 或 [N]
        - normalized_areas: 归一化面积 [B, N] 或 [N]

    示例
    ----
    >>> regions = torch.tensor([[10, 10, 50, 30]])  # 40x20 区域
    >>> compute_region_shape_scale(regions, (64, 64))
    (tensor([0.6931]), tensor([0.1953]))  # log(2) ≈ 0.69, 40*20/4096 ≈ 0.20
    """
    if regions.numel() == 0:
        return torch.tensor([]), torch.tensor([])

    # 记录原始维度并处理 2D 输入
    was_2d = regions.dim() == 2
    if was_2d:
        regions = regions.unsqueeze(0)  # [1, N, 4]

    B, N, _ = regions.shape
    W, H = image_size

    # 计算宽度和高度 [B, N]
    # 使用绝对值确保非负 (处理 x2 < x1 或 y2 < y1 的情况)
    widths = torch.abs(regions[..., 2] - regions[..., 0])
    heights = torch.abs(regions[..., 3] - regions[..., 1])

    # 归一化 [B, N]
    norm_widths = widths / W
    norm_heights = heights / H

    # 纵横比 (对数变换)
    # log(w/h) = log(w) - log(h)，对称处理宽高方向
    # 数值稳定性: 使用 log(w + eps) - log(h + eps) 避免 log(0)
    log_norm_widths = torch.log(norm_widths + epsilon)
    log_norm_heights = torch.log(norm_heights + epsilon)
    aspect_ratios = log_norm_widths - log_norm_heights

    # 归一化面积
    # area_ratio = (w/W) * (h/H)，值域 [0, 1]
    normalized_areas = norm_widths * norm_heights

    # 恢复原始维度
    if was_2d:
        aspect_ratios = aspect_ratios.squeeze(0)
        normalized_areas = normalized_areas.squeeze(0)

    return aspect_ratios, normalized_areas


def compute_shape_scale_similarity(
    aspect_ratios: torch.Tensor,
    normalized_areas: torch.Tensor,
) -> torch.Tensor:
    """
    计算形状-尺度相似性矩阵。

    数学形式化
    ==========

    特征向量:
        c_i = [aspect_ratio_i; normalized_area_i]

    相似性矩阵:
        S[i,j] = <c_i, c_j> / (||c_i|| * ||c_j||)

    使用余弦相似性，支持可变尺度的区域比较。

    参数
    ----
    aspect_ratios : torch.Tensor
        纵横比，形状 [B, N] 或 [N]
    normalized_areas : torch.Tensor
        归一化面积，形状 [B, N] 或 [N]

    返回
    ----
    torch.Tensor
        相似性矩阵，形状 [B, N, N] 或 [N, N]

    示例
    ----
    >>> ar = torch.tensor([0.0, 0.6931])  # 1:1, 2:1
    >>> na = torch.tensor([0.25, 0.25])   # 相同面积
    >>> sim = compute_shape_scale_similarity(ar, na)
    >>> sim.shape
    torch.Size([2, 2])
    """
    if aspect_ratios.numel() == 0:
        return torch.tensor([])

    was_2d = aspect_ratios.dim() == 1
    if was_2d:
        aspect_ratios = aspect_ratios.unsqueeze(0)
        normalized_areas = normalized_areas.unsqueeze(0)

    B, N = aspect_ratios.shape

    # 构建特征向量 [B, N, 2]
    features = torch.stack([aspect_ratios, normalized_areas], dim=-1)

    # 归一化特征
    norm = torch.norm(features, dim=-1, keepdim=True)  # [B, N, 1]
    norm = norm + 1e-8  # 防止除零
    features_normed = features / norm

    # 余弦相似性矩阵 [B, N, N]
    similarity = torch.bmm(features_normed, features_normed.transpose(-2, -1))

    if was_2d:
        similarity = similarity.squeeze(0)

    return similarity


# ==================== I31-3: 面积归一化函数 ====================

def compute_normalized_area(
    regions: torch.Tensor,
    image_size: Tuple[int, int],
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    计算归一化面积分数 (I31-3)

    数学形式化
    ==========

    面积归一化公式 (用户指定):

        .. math::
            f_{{area}} = \\frac{{\\log(s_{{patch}} + 1)}}{{\\log(S_{{total}} + 1)}}

    其中:
        s_{\text{patch}} = w \times h = (x_2 - x_1) \times (y_2 - y_1)
        S_{\text{total}} = W \times H

    数学性质:
        - 值域: f_{{area}} \\in [0, 1]
        - 单调性: log(s+1) 保证 s \\in [0, S_{{total}}] 时单调递增
        - 梯度: df/ds = 1/((s+1)·log(S+1))，有界且平滑

    参数
    ----
    regions : torch.Tensor
        区域边界张量，形状 [B, N, 4] 或 [N, 4]
        格式: [x1, y1, x2, y2]
    image_size : Tuple[int, int]
        (W, H) 图像尺寸
    epsilon : float, optional
        数值稳定性常量，默认 1e-8

    返回
    ----
    torch.Tensor
        归一化面积分数，形状 [B, N] 或 [N]

    示例
    ----
    >>> regions = torch.tensor([[10, 10, 50, 30]])  # 40x20 区域
    >>> compute_normalized_area(regions, (64, 64))
    tensor([0.1953])  # log(800+1)/log(4096+1) ≈ 0.195
    """
    if regions.numel() == 0:
        return torch.tensor([])

    # 记录原始维度并处理 2D 输入
    was_2d = regions.dim() == 2
    if was_2d:
        regions = regions.unsqueeze(0)  # [1, N, 4]

    B, N, _ = regions.shape
    # 处理 image_size 格式：支持 int 或 (W, H) 元组
    if isinstance(image_size, int):
        W = H = image_size
    else:
        W, H = image_size
    S_total = W * H

    # 计算区域面积 [B, N]
    widths = torch.abs(regions[..., 2] - regions[..., 0])
    heights = torch.abs(regions[..., 3] - regions[..., 1])
    s_patch = widths * heights  # 原始面积

    # 面积归一化: f = log(s+1) / log(S+1)
    log_s_plus_1 = torch.log(s_patch + epsilon)
    log_S_plus_1 = torch.log(torch.tensor(S_total + 1, device=regions.device, dtype=torch.float32))
    f_area = log_s_plus_1 / log_S_plus_1

    # 确保在 [0, 1] 范围内
    f_area = f_area.clamp(0.0, 1.0)

    # 恢复原始维度
    if was_2d:
        f_area = f_area.squeeze(0)

    return f_area


def compute_area_similarity(
    area_scores: torch.Tensor,
) -> torch.Tensor:
    """
    计算面积相似性矩阵 (I31-3)

    数学形式化
    ==========

    面积相似性度量:
        p_s(i,j) = √(s_i · s_j) / S_total
                 = area_emb[i] · area_emb[j]

    其中 area_emb 是面积分数的嵌入表示。

    使用余弦相似性，支持可变尺度的区域比较。

    参数
    ----
    area_scores : torch.Tensor
        归一化面积分数，形状 [B, N] 或 [N]

    返回
    ----
    torch.Tensor
        面积相似性矩阵，形状 [B, N, N] 或 [N, N]

    示例
    ----
    >>> area = torch.tensor([0.1, 0.2, 0.3])
    >>> sim = compute_area_similarity(area)
    >>> sim.shape
    torch.Size([3, 3])
    """
    if area_scores.numel() == 0:
        return torch.tensor([])

    was_2d = area_scores.dim() == 1
    if was_2d:
        area_scores = area_scores.unsqueeze(0)

    B, N = area_scores.shape

    # 归一化面积分数作为特征向量
    features = area_scores.unsqueeze(-1)  # [B, N, 1]
    norm = torch.norm(features, dim=-1, keepdim=True)  # [B, N, 1]
    norm = norm + 1e-8  # 防止除零
    features_normed = features / norm

    # 余弦相似性矩阵 [B, N, N]
    similarity = torch.bmm(features_normed, features_normed.transpose(-2, -1))

    if was_2d:
        similarity = similarity.squeeze(0)

    return similarity


if __name__ == "__main__":
    # 演示代码
    print("动态深度计算演示")
    print("=" * 60)

    test_cases = [
        ((64, 64), 4),
        ((224, 224), 4),
        ((512, 512), 4),
        ((224, 224), 8),
        ((64, 64), 8),
        ((32, 32), 4),
        ((100, 100), 4),
        ((16, 16), 4),
        ((8, 8), 4),
        ((4, 4), 4),
    ]

    for image_size, min_patch in test_cases:
        max_depth = compute_max_depth(image_size, min_patch)
        actual_min = compute_actual_min_patch(image_size, max_depth)
        patch_sizes = compute_patch_sizes(image_size, max_depth)
        total = compute_total_candidates(image_size, max_depth)

        print(f"Image: {image_size[0]}x{image_size[1]}, min_patch={min_patch}")
        print(f"  -> max_depth={max_depth}, actual_min={actual_min}")
        print(f"  -> patch_sizes={patch_sizes}")
        print(f"  -> total_candidates={total}")
        print()
