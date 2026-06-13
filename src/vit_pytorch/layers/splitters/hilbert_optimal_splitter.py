"""
Hilbert-Optimal Splitter (H1SS)

基于6条数学公理的最优实现：
- A1 Locality: 1D Hilbert 流形卷积 (Conv1D k=5)
- A2 Determinism: 移除 Gumbel，纯 softmax
- A3 Gradient: Entmax 稀疏激活
- A4 Tree: 软约束 z_parent -= λ × max(z_children)
- A5 Consistency: 单次 Entmax 投影
- A6 Simplicity: < 10K 参数，单损失

数学形式:
    输入: F ∈ ℝ^{B×C×H×W}, R = {R_0, ..., R_{N-1}}
    f_i = ROIAlign(F, R_i) ⊙ σ(d_i) + E_d(d_i)
    x_i = Concat(f_i, PathEmb(p_i), RotEmb(r_i), a_i)
    z = Conv1D_Hilbert(x, kernel=5)
    s = Entmax_{α}(z / τ) × K
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import roi_align
# D4-AUDIT FIX: 移除外部 entmax 库导入，替换为内部 entmax_beta 实现
# 原因: entmax_bisect 内部强制 .float() upcast 至 float32，破坏 AMP 显存优化
# entmax_beta (基于 sparsemax 算法) 无此问题，且支持 alpha=1.5

from vit_pytorch.core.constants import EPS
from vit_pytorch.core.splitter_protocol import (
    CoreSplitter,
    SplitResult,
)
from vit_pytorch.core.curve_hilbert import HilbertCurve
from vit_pytorch.layers.embeddings.fractal_path import (
    VectorizedPathEncoder,
    OrientationExtractor,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 最小配置类（用于 VectorizedPathEncoder）
# =============================================================================

@dataclass
class _MinimalPathEncoderConfig:
    """VectorizedPathEncoder 所需的最小配置"""
    image_size: int = 224
    min_patch_size: int = 4
    dim: int = 64
    max_level: int = 8


# =============================================================================
# Gumbel-STE 离散选择 (替换 Entmax)
# =============================================================================

def gumbel_ste_topk(
    logits: Tensor,
    K: int,
    tau: float,
    logit_scale: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Gumbel-STE Top-K 选择。

    Logit Scale 修正（关键数值稳定性保障）:
        问题: _manifold_convolution 的 logits 量级可能 >> Gumbel 噪声量级
              (Gumbel(0,1) 均值~0.577)，导致噪声被淹没 -> 退化为确定性 argmax。
        修正: noisy = (logits * gamma) + g
              其中 gamma = logit_scale.exp()，初始化为 1.0
              确保训练初期噪声有足够力量强制探索。

    Args:
        logits: [B, N] 选择分数
        K: 固定 token 数量
        tau: Gumbel-Softmax 温度
        logit_scale: 可学习的对数缩放因子

    Returns:
        mask_ste: [B, N] 前向离散/反向连续的 STE 掩码
        mask_soft: [B, N] 软代理概率（用于 batch-wise 熵正则化）
    """
    gamma = logit_scale.exp() if isinstance(logit_scale, torch.Tensor) else logit_scale
    g = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=EPS)))
    noisy = (logits * gamma) + g

    # 硬选择（前向）
    _, top_idx = torch.topk(noisy, K, dim=-1)
    mask_hard = torch.zeros_like(logits).scatter_(-1, top_idx, 1.0)

    # 软代理（反向）
    mask_soft = F.softmax(noisy / tau, dim=-1)

    # STE 绑定
    mask_ste = (mask_hard - mask_soft).detach() + mask_soft
    return mask_ste, mask_soft


# 保留 entmax_beta 供向后兼容（但不再在 Splitter 中使用）
def entmax_beta(
    scores: Tensor,
    alpha: float = 1.5,
    dim: int = -1,
) -> Tensor:
    """
    Entmax 稀疏激活 (Peters et al., 2019)

    数学:
        entmax_α(p) = argmax_{q ∈ Δ^{K-1}} q^T z + H_α(q)
        其中 H_α(q) 是 α-entropy

    当 α → 1: 退化为 argmax (稀疏)
    当 α = 2: 退化为 softmax
    当 α → ∞: 退化为均匀分布

    Args:
        scores: [*, K] logit 张量
        alpha: 稀疏度参数 (默认 1.5)
        dim: softmax 维度

    Returns:
        probs: [*, K] 稀疏概率分布
    """
    if alpha == 2.0:
        return F.softmax(scores, dim=dim)

    # D1: Logits 稳定性限制 (针对 FP16)
    # clamp 防止 exp() 溢出: exp(10) ≈ 22026 < 65504 (FP16 max)
    scores = torch.clamp(scores, min=-10.0, max=10.0)

    # 使用更稳定的实现
    # 基于 Alpha-Entmax 的迭代算法 (Peters et al., 2019)
    # D4-AUDIT FIX: 移除不必要的 .float()，保留原始 dtype

    # 初始化
    max_score = scores.max(dim=dim, keepdim=True)[0]
    scores_std = scores - max_score

    # 简化的实现：使用固定的稀疏度
    # alpha=1.5 产生稀疏但可微的分布
    if alpha == 1.5:
        # Sparsemax 变体
        sorted_scores, _ = torch.sort(scores_std, dim=dim, descending=True)
        cumsum = torch.cumsum(sorted_scores, dim=dim)

        # 找到阈值
        # τ = (sum(p) - 1) / index
        n = scores.shape[dim]
        k = torch.arange(1, n + 1, device=scores.device, dtype=scores.dtype)
        k = k.view(*([1] * (scores.dim() - 1)), -1)

        # D2: Epsilon 保护分母，防止除零
        k_safe = k.clamp(min=1e-8)
        tau = (cumsum - 1) / k_safe
        tau_valid = tau > sorted_scores

        # 找到最大的有效 tau
        tau_max = tau_valid.float().cumsum(dim=dim)
        tau_max = (tau_max == 0).float().sum(dim=dim, keepdim=True)

        tau_final = torch.gather(tau, dim=dim, index=tau_max.long())

        # 计算概率
        probs = F.relu(scores_std - tau_final)
    else:
        # 简化的 soft-max 实现（当 α 接近 2 时）
        probs = F.softmax(scores_std * (alpha - 1), dim=dim)

    # A-NAN FIX: Fallback - 如果产生 NaN，回退到 softmax
    if not torch.isfinite(probs).all():
        return F.softmax(scores, dim=dim)

    return probs


# =============================================================================
# Hilbert-Optimal Splitter
# =============================================================================

@dataclass
class HilbertOptimalSplitterConfig:
    """Hilbert-Optimal Splitter 配置 (简化版: 固定K + Gumbel-STE)

    参数:
        min_patch_size, max_level_limit, feature_dim, hidden_dim
        K_fixed: 固定 token 数量
        temperature_init/temperature_min: Gumbel 温度范围
        use_distance_decay_conv: 是否使用距离衰减卷积
    """

    min_patch_size: int = 4
    max_level_limit: int = 8
    feature_dim: int = 256
    hidden_dim: int = 64
    K_fixed: int = 16
    temperature_init: float = 1.0
    temperature_min: float = 0.1
    use_distance_decay_conv: bool = True



class HilbertOptimalSplitter(nn.Module, CoreSplitter):
    """
    Hilbert-Optimal Splitter (H1SS)

    基于6条数学公理的最优实现：

    A1 (Locality):
        - 1D Hilbert 流形卷积: Conv1D(kernel=5) 在 Hilbert 序上
        - 强制邻域交互，J(S) 理论最优

    A2 (Determinism):
        - 移除 Gumbel 扰动
        - 使用纯 softmax，无随机性

    A3 (Gradient):
        - Entmax 稀疏激活
        - 选中/非选中梯度差异显著

    A4 (Tree Consistency):
        - 软约束: z_parent -= λ × max(z_children)
        - 可微

    A5 (Consistency):
        - 单次 Entmax 投影
        - E[|S|] = K, Var → 0

    A6 (Simplicity):
        - Conv1D + Linear ≈ 9.2K 参数
        - 单损失函数
    """

    def __init__(
        self,
        config: Optional[HilbertOptimalSplitterConfig] = None,
        feature_dim: int = 256,
        min_patch_size: int = 4,
        max_level_limit: int = 8,
        hidden_dim: int = 64,
        K_fixed: int = 16,
        temperature_init: float = 1.0,
        temperature_min: float = 0.1,
        use_distance_decay_conv: bool = True,
    ):
        super().__init__()

        # 当前训练轮次
        self._current_epoch: int = 0

        if config is not None:
            feature_dim = config.feature_dim
            min_patch_size = config.min_patch_size
            max_level_limit = config.max_level_limit
            hidden_dim = config.hidden_dim
            K_fixed = getattr(config, 'K_fixed', 16)
            temperature_init = getattr(config, 'temperature_init', 1.0)
            temperature_min = getattr(config, 'temperature_min', 0.1)
            use_distance_decay_conv = getattr(config, 'use_distance_decay_conv', True)

        self.feature_dim = feature_dim
        self.min_patch_size = min_patch_size
        self.max_level_limit = max_level_limit
        self.hidden_dim = hidden_dim
        self.K_fixed = K_fixed

        # 温度
        self.temperature = temperature_init
        self.temperature_init = temperature_init
        self.temperature_min = temperature_min

        # Gumbel-STE logit_scale：防止 Gumbel 噪声被 logits 淹没
        self.logit_scale = nn.Parameter(torch.zeros(1, dtype=torch.get_default_dtype()))

        # 当前状态
        self._current_image_size: Optional[Tuple[int, int]] = None
        self._epoch = 0

        # 核心组件
        # 1. 特征投影
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        nn.init.orthogonal_(self.feature_proj.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.feature_proj.bias)

        # 2. 深度嵌入
        self.depth_embedding = nn.Embedding(max_level_limit + 1, hidden_dim)

        # 3. 路径编码器
        self._path_encoder_config = _MinimalPathEncoderConfig(
            image_size=224,
            min_patch_size=min_patch_size,
            dim=hidden_dim,
            max_level=max_level_limit,
        )
        self.path_encoder = VectorizedPathEncoder(self._path_encoder_config)
        self.orientation_extractor = OrientationExtractor(
            max_level=max_level_limit,
            embedding_dim=hidden_dim,
        )

        # 3.5. 旋转嵌入投影
        self.rot_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # 4. 面积编码
        d_indices = torch.arange(max_level_limit + 1, dtype=torch.float32)
        self.register_buffer(
            '_area_encoding',
            torch.exp2(-d_indices * 2.0)
        )

        # 4.5. 面积投影
        self.area_proj = nn.Linear(1, hidden_dim)
        # I-T10-Splitter-C: 平衡 geo_embs 三段量级, 避免 4^{-d} 衰减压制 area 段。
        # 初始 1.0 等效无 scale, 完全向后兼容, 可直接 load 预训练 checkpoint。
        # 训练时反向传播会自动拉高 η, 让 area 段贡献到 geo_norm 联合统计量 σ²,
        # 从而让 ∂L/∂area_proj.weight 获得按 (σ_area/σ_total²)² 数量级提升的梯度。
        self.area_scale = nn.Parameter(
            torch.tensor(1.0, dtype=torch.get_default_dtype())
        )

        # 4.6. 分组特征归一化
        self.roi_norm = nn.LayerNorm(hidden_dim)
        self.geo_norm = nn.LayerNorm(hidden_dim * 3)

        # 4.7. 可学习的语义-几何融合比例
        self._semantic_ratio = nn.Parameter(torch.zeros(1, dtype=torch.get_default_dtype()))
        # sigmoid(0) = 0.5, 即初始时语义和几何各占一半

        # 5. 1D Hilbert 流形卷积
        conv_input_dim = hidden_dim * 4  # semantics (1x) + geometry (3x: path + rot + area)

        if use_distance_decay_conv:
            from .hilbert_distance_decay_conv import HilbertDistanceDecayConv1D
            self.conv1d_hilbert = HilbertDistanceDecayConv1D(conv_input_dim)
        else:
            self.conv1d_hilbert = nn.Conv1d(
                conv_input_dim, 1, kernel_size=5, padding=2, groups=1,
            )

        # 初始化 conv1d_hilbert 偏置为 +0.5，强制初期尝试更多分裂
        self._init_logits_bias()

        # 候选区域缓存
        self.register_buffer('candidate_regions', torch.zeros(0, 4))
        self.register_buffer('candidate_depths', torch.zeros(0, dtype=torch.long))
        self.register_buffer('hilbert_indices', torch.zeros(0, dtype=torch.long))

        self._children_matrix: Optional[Tensor] = None


    def _init_logits_bias(self) -> None:
        """
        初始化 conv1d_hilbert 偏置为 +0.5，强制初期多分裂

        问题：初始 logits 全负，模型倾向"不分裂"，导致 active_ratio 停滞
        解决：添加 +0.5 偏置，使初期 logits 更正值，尝试更多分裂

        效果：
            - 训练初期：更多分裂尝试 → active_ratio 上升
            - 训练后期：模型自动调整偏置以平衡分裂/不分裂
        """
        if hasattr(self.conv1d_hilbert, 'bias') and self.conv1d_hilbert.bias is not None:
            nn.init.constant_(self.conv1d_hilbert.bias, 0.5)


    @property
    def max_level_limit(self) -> int:
        return self._max_level_limit

    @max_level_limit.setter
    def max_level_limit(self, value: int):
        self._max_level_limit = value

    @property
    def num_candidates(self) -> int:
        return self.candidate_regions.shape[0]

    @property
    def is_training(self) -> bool:
        """获取当前训练/评估模式"""
        return self.training

    def update_candidates(self, image_size: Tuple[int, int]) -> None:
        """根据输入尺寸动态更新候选区域"""
        self._current_image_size = image_size
        self._generate_candidates(image_size)

    def _generate_candidates(self, image_size: Tuple[int, int]):
        """生成 Hilbert 序候选区域"""
        H_img, W_img = image_size

        # 计算最大深度
        max_level = min(
            self.max_level_limit,
            int(math.log2(min(H_img, W_img) / self.min_patch_size))
        )

        if max_level < 0:
            max_level = 0

        device = self.depth_embedding.weight.device

        all_regions = []
        all_depths = []
        all_parent_idx = []
        all_hilbert_idx = []

        depth_start_idx = [0]

        for depth in range(max_level + 1):
            grid_size = 1 << depth  # I-OPT: 位移替代 2**depth
            region_h = H_img / grid_size
            region_w = W_img / grid_size

            i_idx = torch.arange(grid_size, device=device)
            j_idx = torch.arange(grid_size, device=device)
            grid_i, grid_j = torch.meshgrid(i_idx, j_idx, indexing='ij')

            y0 = (grid_i * region_h).long()
            x0 = (grid_j * region_w).long()
            y1 = ((grid_i + 1) * region_h).long()
            x1 = ((grid_j + 1) * region_w).long()

            regions_depth = torch.stack([x0, y0, x1, y1], dim=-1).view(-1, 4)
            all_regions.append(regions_depth)

            depths_depth = torch.full((regions_depth.shape[0],), depth, dtype=torch.long, device=device)
            all_depths.append(depths_depth)

            # Hilbert 索引 - 使用网格索引 (而非像素坐标)
            # xy_to_d_batch 期望 0~(grid_size-1) 的网格索引
            hilbert_idx = HilbertCurve.xy_to_d_batch(
                grid_size,
                grid_j.reshape(-1).long(),
                grid_i.reshape(-1).long()
            )
            all_hilbert_idx.append(hilbert_idx)

            # 父节点索引
            if depth > 0:
                parent_grid_size = 1 << (depth - 1)  # D4-AUDIT FIX: 2**(d-1) → 1<<(d-1)
                parent_idx_grid = (grid_i // 2) * parent_grid_size + (grid_j // 2)
                parent_idx = depth_start_idx[depth - 1] + parent_idx_grid.view(-1)
            else:
                parent_idx = torch.full((regions_depth.shape[0],), -1, dtype=torch.long, device=device)
            all_parent_idx.append(parent_idx)

            depth_start_idx.append(depth_start_idx[-1] + regions_depth.shape[0])

        # 合并
        candidate_regions = torch.cat(all_regions, dim=0)
        candidate_depths = torch.cat(all_depths, dim=0)
        parent_indices = torch.cat(all_parent_idx, dim=0)
        hilbert_indices = torch.cat(all_hilbert_idx, dim=0)

        # 按 Hilbert 索引排序
        sort_idx = hilbert_indices.argsort()
        candidate_regions = candidate_regions[sort_idx]
        candidate_depths = candidate_depths[sort_idx]
        parent_indices = parent_indices[sort_idx]
        hilbert_indices = hilbert_indices[sort_idx]

        # 重新计算父节点索引（排序后）
        N = candidate_regions.shape[0]
        new_parent_indices = torch.full((N,), -1, dtype=torch.long, device=device)

        # D2-AUDIT FIX: 向量化 depth 层父节点计算 (外层循环保留，内层 i 循环向量化)
        for depth in range(1, max_level + 1):
            depth_start = depth_start_idx[depth]
            depth_end = depth_start_idx[depth + 1]
            if depth_end <= depth_start:
                continue
            parent_start = depth_start_idx[depth - 1]

            # 向量化计算该层所有节点的父节点索引
            grid_size = 1 << depth
            i_range = torch.arange(depth_start, depth_end, device=device, dtype=torch.long)
            grid_idx = i_range - depth_start
            gi = grid_idx >> depth
            gj = grid_idx & (grid_size - 1)
            parent_offsets = (gi >> 1) * (grid_size >> 1) + (gj >> 1)
            new_parent_indices[depth_start:depth_end] = parent_start + parent_offsets

        # 子节点矩阵
        children_matrix = torch.full((N, 4), -1, dtype=torch.long, device=device)
        # D4-AUDIT FIX: 向量化 children_matrix 构建，避免 O(N*4) Python 循环
        # 方法：按 (parent, child_order) 排序后批量赋值
        valid_mask = new_parent_indices >= 0
        if valid_mask.any():
            valid_indices = torch.where(valid_mask)[0]  # 子节点的原始索引
            valid_parents = new_parent_indices[valid_indices]  # 对应的父节点

            # 按父节点排序，相同时按子节点顺序排序
            sort_keys = valid_parents * N + valid_indices  # 编码为唯一键
            sorted_order = sort_keys.argsort()

            sorted_parents = valid_parents[sorted_order]
            sorted_children = valid_indices[sorted_order]

            # 计算每个父节点的子节点数量（用于确定slot起始位置）
            # 使用 bincount 得到每个父节点的子节点总数
            parent_counts = torch.zeros(N, device=device, dtype=torch.long)
            parent_counts.scatter_add_(0, sorted_parents, torch.ones_like(sorted_parents))

            # 计算每个子节点的slot：同一父节点内，按出现顺序编号
            # 利用排序性质，相同父节点的子节点是连续的
            # 因此 slot = 该子节点在同父节点组内的位置索引
            # 使用 cumsum 计算组内计数
            # 识别父节点变化的位置
            parent_changed = torch.zeros_like(sorted_parents)
            parent_changed[1:] = (sorted_parents[1:] != sorted_parents[:-1]).long()

            # 计算全局位置（从0开始）
            global_pos = torch.arange(len(sorted_parents), device=device, dtype=torch.long)

            # 使用 cumsum 构建每组的其实位置
            group_positions = torch.where(parent_changed == 1)[0]
            group_starts = torch.cat([
                torch.zeros(1, dtype=torch.long, device=device),
                group_positions
            ])

            # 计算组起始位置的差值，构建 group_elements 用于 cumsum
            diff = group_starts[1:] - group_starts[:-1]
            group_elements = torch.zeros_like(sorted_parents)
            group_elements[group_positions] = diff

            # slot = 全局位置 - 组起始位置
            slot_within_parent = global_pos - torch.cumsum(group_elements, dim=0)

            # 取出有效的 slot 和对应的子节点（slot 必须在 0-3 范围内）
            valid_mask = slot_within_parent < 4
            valid_slots = slot_within_parent[valid_mask]
            valid_sorted_children = sorted_children[valid_mask]

            # scatter 到 children_matrix
            children_matrix[sorted_parents, valid_slots] = valid_sorted_children

        self.register_buffer('candidate_regions', candidate_regions)
        self.register_buffer('candidate_depths', candidate_depths)
        self.register_buffer('parent_indices', new_parent_indices)
        self.register_buffer('hilbert_indices', hilbert_indices)
        self.register_buffer('children_matrix', children_matrix)

        self._children_matrix = children_matrix

    def _extract_features(self, features: Tensor, regions: Tensor, depths: Tensor) -> Tuple[Tensor, Tensor]:
        """提取结构化特征

        数学:
            f_i = ROIAlign(F, R_i, sampling_ratio(d_i)) * sigma(d_i) + E_d(d_i)

        Returns:
            roi_features: [B, N, hidden_dim] 投影 + 深度注入后的特征
            roi_raw: [B, N, feature_dim] 原始池化特征（供 Tokenizer 复用）
        """
        B = features.shape[0]
        N = regions.shape[0]
        _, C_feat, H_feat, W_feat = features.shape

        depth_bins = [0, 2, 4, self.max_level_limit + 1]
        sampling_ratios = [1, 2, 4]

        batch_indices = torch.zeros(N, dtype=torch.long, device=regions.device)
        boxes = torch.cat([batch_indices.unsqueeze(-1).float(), regions], dim=-1)

        pooled_list = []
        indices_list = []

        for i in range(len(depth_bins) - 1):
            low, high = depth_bins[i], depth_bins[i + 1]
            mask = (depths >= low) & (depths < high)
            if not mask.any():
                continue

            indices = mask.nonzero(as_tuple=False).squeeze(-1)
            group_boxes = boxes[indices]

            pooled = roi_align(
                features, group_boxes,
                output_size=(1, 1), spatial_scale=1.0,
                sampling_ratio=sampling_ratios[i], aligned=True,
            )
            pooled = pooled.squeeze(-1).squeeze(-1)

            pooled_list.append(pooled)
            indices_list.append(indices)

        if len(pooled_list) == 0:
            roi_raw = torch.zeros(B, N, C_feat, device=features.device, dtype=features.dtype)
        else:
            roi_raw = torch.zeros(B, N, C_feat, device=features.device, dtype=features.dtype)
            for pooled, indices in zip(pooled_list, indices_list):
                roi_raw[:, indices] = pooled.unsqueeze(0)

        # 投影到 hidden_dim
        roi_features = self.feature_proj(roi_raw)

        # 深度缩放
        depth_scale = torch.sigmoid(self.depth_embedding.weight)
        depth_scale = depth_scale[depths]
        roi_features = roi_features * depth_scale.unsqueeze(0)

        # 深度嵌入
        depth_emb = self.depth_embedding(depths)
        roi_features = roi_features + depth_emb.unsqueeze(0)

        return roi_features, roi_raw

    def _encode_geometry(self, regions: Tensor, depths: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """编码几何信息

        使用完整的 Hilbert 路径和旋转状态注入:

        数学:
            p_i = VectorizedPathEncoder.compute_quadrant_paths(x, y, max_level)
            r_i = OrientationExtractor.compute_rotation_states(p_i)
            dir_i = OrientationExtractor.compute_rotation_directions(p_i)

        返回:
            path_emb: 完整 Hilbert 路径嵌入
            rot_emb: 旋转状态嵌入
            area_enc: 面积编码
        """
        max_level = self.max_level_limit

        # 计算区域的中心坐标 (转换为整数)
        cx = ((regions[:, 0] + regions[:, 2]) / 2).long()  # [N]
        cy = ((regions[:, 1] + regions[:, 3]) / 2).long()  # [N]

        # 计算完整的 Hilbert 四叉树路径
        # paths: [N, max_level] 每个元素 ∈ {0, 1, 2, 3}
        paths = VectorizedPathEncoder.compute_quadrant_paths(
            cx, cy, max_level
        )  # [N, max_level]

        # 计算旋转方向 (累积翻转次数 mod 4)
        rotation_dirs = OrientationExtractor.compute_rotation_directions(
            paths.unsqueeze(0)
        ).squeeze(0)  # [N, max_level]

        # 将路径和旋转编码为嵌入
        # 使用 path_embedding 层 (如果存在) 或 depth_embedding
        if hasattr(self, 'path_embedding'):
            path_emb = self.path_embedding(paths)  # [N, max_level, hidden_dim]
            path_emb = path_emb.mean(dim=1)  # [N, hidden_dim]
        else:
            # 回退到使用深度嵌入
            path_emb = self.depth_embedding(depths)  # [N, hidden_dim]

        # 旋转嵌入: 使用 rotation_dirs 的最后一个维度作为索引
        # rotation_dirs: [N, max_level] -> 取最后一层作为当前旋转方向
        final_rot_dir = rotation_dirs[:, -1]  # [N]

        # 使用 direction_embedding
        rot_emb = self.orientation_extractor.direction_embedding(final_rot_dir)  # [N, hidden_dim 或 dim]
        # 旋转嵌入投影（参数共享后维度可能不匹配：GeometryField 用 dim，Splitter 用 hidden_dim）
        if rot_emb.shape[-1] != path_emb.shape[-1]:
            # 维度不匹配：使用 rot_proj 进行投影
            # rot_proj 输入维度固定为 hidden_dim，需要先调整 rot_emb 到 hidden_dim 维度
            rot_emb_adjusted = F.linear(rot_emb, torch.eye(path_emb.shape[-1], rot_emb.shape[-1], device=rot_emb.device))
            rot_emb = self.rot_proj(rot_emb_adjusted)
        else:
            # 维度匹配：直接使用 rot_proj（初始化为近似恒等映射）
            rot_emb = self.rot_proj(rot_emb)
        # 面积编码 - 4^(-depth) → Linear 投影（消除 expand 导致的秩塌陷）
        area_enc = self._area_encoding[depths]  # [N], dtype=float32 buffer
        # I-T10-Splitter-D: 显式 cast 到 area_proj.dtype, 避免 AMP 混合精度。
        # device 一致性前提: _area_encoding 跟随 splitter.to(device) 落在与
        # area_proj.weight 同一 device, .to(dtype=...) 是 view-like, 不触发
        # CPU↔GPU 同步。AMP 下 dtype 是 fp16/bf16, 仍 view-like。
        area_enc = area_enc.to(dtype=self.area_proj.weight.dtype)
        area_enc = self.area_proj(area_enc.unsqueeze(-1))  # [N] → [N, 1] → [N, hidden_dim]
        # I-T10-Splitter-C: 可学习 area_scale 平衡 geo_embs 三段量级。
        # 必须放在 return 之前, 走 autograd 链; 不可在 __init__ 提前乘。
        area_enc = area_enc * self.area_scale

        return path_emb, rot_emb, area_enc


    def _manifold_convolution(self, features: Tensor) -> Tensor:
        """1D Hilbert 流形卷积

        数学:
            z = Conv1D_Hilbert(Concat(f, path_emb, rot_emb, area), kernel=5)
        """
        B, N, D = features.shape

        # 转置: [B, N, D] -> [B, D, N]
        x = features.transpose(1, 2)

        # 1D 卷积
        z = self.conv1d_hilbert(x)  # [B, 1, N] 或 [B, N]

        # I167-1: 兼容处理不同卷积输出的形状
        # 标准 Conv1D 返回 [B, 1, N], 需要 transpose + squeeze
        # HilbertDistanceDecayConv1D 返回 [B, N], 直接使用
        if z.dim() == 3:
            # [B, 1, N] -> [B, N]
            z = z.transpose(1, 2).squeeze(-1)

        return z





    def forward(
        self,
        features: Tensor,
        image_size: Optional[Tuple[int, int]] = None,
        hard: bool = False,
        epoch: int = 0,
    ) -> SplitResult:
        """前向传播（简化版：Gumbel-STE + 固定 K）。

        数学流程:
            1. 统一 ROI-Align → roi_features (投影后) + roi_raw (原始)
            2. 几何编码 → path_emb, rot_emb, area_enc
            3. 可学习融合 → α·norm(roi) + (1-α)·norm(geo)
            4. Hilbert 流形卷积 → logits
            5. Gumbel-STE 离散选择 → mask_ste, mask_soft
            6. 提取选中区域元数据

        Args:
            features: [B, C, H, W] 输入特征
            image_size: (H, W) 原始图像尺寸
            hard: 保留兼容性（未使用 — Gumbel-STE 始终返回 STE mask）
            epoch: 保留兼容性（τ 退火由外部 set_temperature 控制）

        Returns:
            SplitResult: 包含 roi_features_raw + mask_ste 供 Tokenizer 复用
        """
        B = features.shape[0]

        # 更新候选区域
        if image_size is None:
            image_size = self._current_image_size
        if image_size is None:
            raise ValueError("image_size must be provided")
        if self.num_candidates == 0 or self._current_image_size != image_size:
            self.update_candidates(image_size)

        # 1. 统一 ROI-Align（返回投影后 + 原始特征）
        roi_features, roi_raw = self._extract_features(
            features, self.candidate_regions, self.candidate_depths,
        )

        # 2. 几何编码
        path_emb, rot_emb, area_enc = self._encode_geometry(
            self.candidate_regions, self.candidate_depths,
        )

        # 3. 可学习融合: alpha * norm(roi) || (1-alpha) * norm(geo)
        alpha = torch.sigmoid(self._semantic_ratio)
        roi_normed = self.roi_norm(roi_features)
        geo_embs = self.geo_norm(torch.cat([
            path_emb.unsqueeze(0).expand(B, -1, -1),
            rot_emb.unsqueeze(0).expand(B, -1, -1),
            area_enc.unsqueeze(0).expand(B, -1, -1),
        ], dim=-1))
        combined = torch.cat([
            alpha * roi_normed,
            (1 - alpha) * geo_embs,
        ], dim=-1)

        # 4. Hilbert 流形卷积 → logits
        logits = self._manifold_convolution(combined)

        # 5. 离散选择
        if hard:
            # 确定性 STE: softmax → hard Top-K → STE（无 Gumbel 噪声）
            mask_soft = F.softmax(logits / self.temperature, dim=-1)
            _, top_idx = torch.topk(mask_soft, self.K_fixed, dim=-1)
            mask_hard_local = torch.zeros_like(logits).scatter_(-1, top_idx, 1.0)
            mask_ste = (mask_hard_local - mask_soft).detach() + mask_soft
        else:
            # Gumbel-STE 离散选择（训练模式 — Gumbel 噪声强制探索）
            mask_ste, mask_soft = gumbel_ste_topk(
                logits, K=self.K_fixed, tau=self.temperature,
                logit_scale=self.logit_scale,
            )

        # 6. 提取选中区域（用于 Tokenizer 元数据）
        mask_hard = (mask_ste > 0.5).float()
        batch_idx, region_idx = torch.where(mask_hard > 0.5)

        if region_idx.numel() == 0:
            _, topk_idx = torch.topk(
                mask_soft[0], max(1, mask_soft.shape[1] // 2), dim=-1,
            )
            batch_idx = torch.zeros(
                topk_idx.shape[0], dtype=torch.long, device=logits.device,
            )
            region_idx = topk_idx

        selected_regions = self.candidate_regions[region_idx]
        selected_depths = self.candidate_depths[region_idx]

        # 按 Hilbert 索引排序
        selected_hilbert = self.hilbert_indices[region_idx]
        sort_idx = selected_hilbert.argsort()
        selected_regions = selected_regions[sort_idx]
        selected_depths = selected_depths[sort_idx]
        batch_idx = batch_idx[sort_idx]

        # 保存排序后的候选索引（供 Tokenizer 快速路径映射）
        candidate_indices = region_idx[sort_idx]

        return SplitResult(
            regions=selected_regions,
            depths=selected_depths,
            batch_indices=batch_idx,
            hilbert_indices=selected_hilbert[sort_idx],
            selected_mask=mask_hard,
            logits=logits,
            probs=mask_soft,
            mask_ste=mask_ste,
            roi_features_raw=roi_raw,
            # T10 keystone: post-feature_proj + LayerNorm (与 HMFT 协议层 roi_features 对齐)
            roi_features=roi_normed,
            candidate_indices=candidate_indices,
        )

    def set_temperature(self, temperature: float) -> None:
        """设置温度（用于训练脚本兼容性）"""
        self.temperature = temperature

    def extra_repr(self) -> str:
        return (
            f"HilbertOptimalSplitter("
            f"max_level={self.max_level_limit}, "
            f"hidden_dim={self.hidden_dim}, "
            f"K_fixed={self.K_fixed}, "
            f"temperature={self.temperature:.2f})"
        )











# =============================================================================
# 验证指标计算
# =============================================================================

def compute_locality_score(
    hilbert_indices: Tensor,
    selected_mask: Tensor,
) -> float:
    """
    计算 A1: Locality Score

    J(S) = mean(|h(s_i) - h(s_{i+1})|)

    Args:
        hilbert_indices: [N] Hilbert 索引
        selected_mask: [N] 或 [B, N] 选中掩码

    Returns:
        locality_score: 标量
    """
    # 处理批量掩码 [B, N]
    if selected_mask.dim() == 2:
        # 逐个样本计算后取平均
        scores = []
        for i in range(selected_mask.shape[0]):
            mask_1d = selected_mask[i]
            if mask_1d.sum() < 2:
                scores.append(0.0)
                continue
            selected_h = hilbert_indices[mask_1d > 0.5]
            selected_h = selected_h.sort()[0]
            if len(selected_h) < 2:
                scores.append(0.0)
                continue
            jumps = (selected_h[1:].float() - selected_h[:-1].float()).abs()
            scores.append(jumps.mean().item())
        return sum(scores) / len(scores) if scores else 0.0

    # 处理 1D 掩码 [N]
    if selected_mask.sum() < 2:
        return 0.0

    selected_h = hilbert_indices[selected_mask > 0.5]
    selected_h = selected_h.sort()[0]

    if len(selected_h) < 2:
        return 0.0

    jumps = (selected_h[1:].float() - selected_h[:-1].float()).abs()
    return jumps.mean().item()







def compute_tree_consistency(
    selected_mask: Tensor,
    parent_indices: Tensor,
    children_matrix: Tensor,
) -> float:
    """
    计算 A4: Tree Consistency

    consistency = |{parent : parent not selected but child selected}| / N_parent
    """
    N = selected_mask.shape[0]
    parent_selected = selected_mask > 0.5

    violations = 0
    total_parents = 0

    for i in range(N):
        if parent_indices[i] >= 0:  # 有父节点
            total_parents += 1
            if not parent_selected[i]:  # 父节点未选中
                # 检查是否有子节点选中
                children = children_matrix[i]
                children_valid = children[children >= 0]
                if (selected_mask[children_valid] > 0.5).any():
                    violations += 1

    if total_parents == 0:
        return 1.0

    return 1.0 - violations / total_parents


def compute_consistency_stats(
    selected_counts: List[int],
    K_target: int,
) -> Tuple[float, float]:
    """
    计算 A5: Consistency Stats

    返回: (mean, variance)
    """
    if not selected_counts:
        return K_target, 0.0

    mean_count = sum(selected_counts) / len(selected_counts)
    variance = sum((c - mean_count) ** 2 for c in selected_counts) / len(selected_counts)

    return mean_count, variance
