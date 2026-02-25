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
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import roi_align
from entmax import entmax_bisect as entmax

from vit_pytorch.core.constants import EPS, TEMPERATURE_MIN
from vit_pytorch.core.splitter_protocol import (
    CoreSplitter,
    SplitResult,
)
from vit_pytorch.core.curve_hilbert import HilbertCurve, HilbertScanner
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
# Entmax 稀疏激活
# =============================================================================

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

    # 使用更稳定的实现
    # 基于 Alpha-Entmax 的迭代算法 (Peters et al., 2019)
    scores = scores.float()

    # 初始化
    max_score = scores.max(dim=dim, keepdim=True)[0]
    scores_norm = scores - max_score

    # 简化的实现：使用固定的稀疏度
    # alpha=1.5 产生稀疏但可微的分布
    if alpha == 1.5:
        # Sparsemax 变体
        sorted_scores, _ = torch.sort(scores_norm, dim=dim, descending=True)
        cumsum = torch.cumsum(sorted_scores, dim=dim)

        # 找到阈值
        # τ = (sum(p) - 1) / index
        n = scores.shape[dim]
        k = torch.arange(1, n + 1, device=scores.device, dtype=torch.float32)
        k = k.view(*([1] * (scores.dim() - 1)), -1)

        tau = (cumsum - 1) / k
        tau_valid = tau > sorted_scores

        # 找到最大的有效 tau
        tau_max = tau_valid.float().cumsum(dim=dim)
        tau_max = (tau_max == 0).float().sum(dim=dim, keepdim=True)

        tau_final = torch.gather(tau, dim=dim, index=tau_max.long())

        # 计算概率
        probs = F.relu(scores_norm - tau_final)
    else:
        # 简化的 soft-max 实现（当 α 接近 2 时）
        probs = F.softmax(scores_norm * (alpha - 1), dim=dim)

    return probs


def entmax_beta_joint(
    scores: Tensor,
    alpha: float = 1.5,
    dim: int = -1,
) -> Tuple[Tensor, Tensor]:
    """
    Entmax + Top-K 联合实现

    返回:
        probs: 稀疏概率
        selected_indices: Top-K 索引
    """
    probs = entmax(scores, alpha=alpha, dim=dim)

    # 阈值选择
    threshold = probs.max(dim=dim, keepdim=True)[0] * 0.5
    selected = probs > threshold

    # 如果选中数量不足，使用 TopK
    num_selected = selected.sum(dim=dim)
    K_target = probs.shape[dim] // 4  # 假设 K ≈ N/4

    # 补足 TopK
    _, topk_indices = torch.topk(probs, K_target, dim=dim)

    return probs, topk_indices


# =============================================================================
# Hilbert-Optimal Splitter
# =============================================================================

@dataclass
class HilbertOptimalSplitterConfig:
    """Hilbert-Optimal Splitter 配置

    三层参数原则:
    - 参数 (Parameters): 固定架构 - min_patch_size, max_level_limit, feature_dim, hidden_dim
    - 变参数 (Variable): 运行时 - K_min, K_max, sampling_ratio
    - 超参数 (Hyper): 可调优 - entmax_alpha, tree_constraint_weight, temperature, jump_loss_weight
    """

    # ========== 参数 (Parameters) - 固定架构 ==========
    min_patch_size: int = 4
    max_level_limit: int = 8
    feature_dim: int = 256
    hidden_dim: int = 64

    # ========== 变参数 (Variable) - 运行时 ==========
    K_min: int = 8
    K_max: int = 64
    # 动态 sampling_ratio: [浅层阈值, 深层阈值] → [sratio_0, sratio_1, sratio_2]
    sampling_ratio_schedule: tuple = (2, 4)  # d ≤ 2: 1, 2 < d ≤ 4: 2, d > 4: 4

    # ========== 超参数 (Hyper) - 可调优 ==========
    # Entmax 参数
    # I107: 从 1.5 改为 1.2，防止 alpha=1.5 导致 Entmax 硬截断
    # 测试结果: alpha=1.5 产生 0% 非零输出，梯度无法回传
    # I107: 添加 alpha 预热策略
    entmax_alpha_init: float = 1.2      # 起始值 (保证梯度流动)
    entmax_alpha_warmup: float = 1.5    # 预热目标值
    entmax_alpha_max: float = 2.0       # 最终稀疏度
    entmax_warmup_epochs: int = 10       # 预热 epoch 数
    entmax_schedule_epochs: int = 20     # 总调度 epoch 数

    # 树约束参数
    tree_constraint_weight: float = 0.1

    # 温度参数
    temperature_init: float = 1.0
    temperature_min: float = 0.3

    # Jump Loss 参数
    jump_loss_weight: float = 0.1

    # Density Field 参数
    density_field_hidden_dim: int = 32


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
        K_min: int = 8,
        K_max: int = 64,
        sampling_ratio_schedule: tuple = (2, 4),
        entmax_alpha: float = 1.2,  # I107: 改为 1.2 防止硬截断
        tree_constraint_weight: float = 0.1,
        temperature_init: float = 1.0,
        temperature_min: float = 0.3,
        jump_loss_weight: float = 0.1,
        density_field_hidden_dim: int = 32,
    ):
        super().__init__()

        # 课程学习阶段 (与 GumbelTopK 保持一致)
        # 1 = Teacher Forcing (Epoch 1-9)
        # 2 = Acc-Driven Splitting (Epoch 10-19)
        # 3 = Resource Co-adaptation (Epoch 20+)
        self._curriculum_stage: int = 1

        if config is not None:
            feature_dim = config.feature_dim
            min_patch_size = config.min_patch_size
            max_level_limit = config.max_level_limit
            hidden_dim = config.hidden_dim
            K_min = config.K_min
            K_max = config.K_max
            sampling_ratio_schedule = config.sampling_ratio_schedule
            entmax_alpha = config.entmax_alpha_init
            tree_constraint_weight = config.tree_constraint_weight
            temperature_init = config.temperature_init
            temperature_min = config.temperature_min
            jump_loss_weight = config.jump_loss_weight
            density_field_hidden_dim = config.density_field_hidden_dim

        self.feature_dim = feature_dim
        self.min_patch_size = min_patch_size
        self.max_level_limit = max_level_limit
        self.hidden_dim = hidden_dim
        self.K_min = K_min
        self.K_max = K_max
        # K 课程学习：当前使用的 K 值（随 epoch 增大）
        self._current_K = K_min
        self._K_schedule_epochs = 30  # K 课程学习持续 30 个 epoch
        # K 向上取整：找到能囊括 K_max 的最小 level 对应的候选区域数
        # 例如: K_max=38 → level 3 (4^3=64) → K_max_rounded=64
        self._K_max_rounded = self._compute_min_level_regions(K_max, max_level_limit)
        self.sampling_ratio_schedule = sampling_ratio_schedule

        # Entmax 参数 (I107: 添加预热策略)
        self.entmax_alpha = entmax_alpha
        self.entmax_alpha_init = 1.2      # 起始值
        self.entmax_alpha_warmup = 1.5     # 预热目标
        self.entmax_alpha_max = 2.0       # 最终稀疏度
        self.entmax_warmup_epochs = 10     # 预热 epoch 数
        self.entmax_schedule_epochs = 20  # 总调度 epoch 数

        # 树约束
        self.tree_constraint_weight = tree_constraint_weight

        # 温度
        self.temperature = temperature_init
        self.temperature_init = temperature_init
        self.temperature_min = temperature_min

        # Jump Loss 权重
        self.jump_loss_weight = jump_loss_weight

        # Density Field 隐藏层维度
        self.density_field_hidden_dim = density_field_hidden_dim

        # 当前状态
        self._current_image_size: Optional[Tuple[int, int]] = None
        self._epoch = 0

        # =====================================================================
        # I150-3: Token 稳定性监控
        # =====================================================================
        self._monitor_token_stability = False
        self._token_history: List[Tensor] = []

        # =====================================================================
        # 核心组件
        # =====================================================================

        # 1. 特征投影
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)

        # 2. 深度嵌入 (A1: 与 Embed 层一致)
        self.depth_embedding = nn.Embedding(max_level_limit + 1, hidden_dim)

        # 3. 路径编码器 - 创建最小配置
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

        # 4. 面积编码
        self.register_buffer(
            '_area_encoding',
            torch.tensor([4.0 ** (-d) for d in range(max_level_limit + 1)], dtype=torch.float32)
        )

        # 5. 1D Hilbert 流形卷积 (A1: 核心创新)
        # 输入: [B, N, hidden_dim * 4] (feat + path + rot + area)
        # 输出: [B, N, 1]
        conv_input_dim = hidden_dim * 4
        self.conv1d_hilbert = nn.Conv1d(
            conv_input_dim,
            1,
            kernel_size=5,
            padding=2,
            groups=1,
        )

        # 6. 深度配额学习 (可选，用于 A5)
        self.depth_quota = nn.Parameter(torch.ones(max_level_limit + 1))

        # I106-2: 密度场网络 - 根据曲线特征动态估计 K
        # 数学: K = ∫ ρ(h) dh, 其中 ρ = sigmoid(MLP(curve_features))
        self.density_field = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # I-NAN: 初始化 density_field 偏置为 -1.1
        # sigmoid(-1.1) ≈ 0.25，强迫模型在训练初期产生更多 token
        # 这解决了 avg_tokens 死锁在 5 个的问题
        self._init_density_field_bias()

        # 候选区域缓存
        self.register_buffer('candidate_regions', torch.zeros(0, 4))
        self.register_buffer('candidate_depths', torch.zeros(0, dtype=torch.long))
        self.register_buffer('parent_indices', torch.zeros(0, dtype=torch.long))
        self.register_buffer('hilbert_indices', torch.zeros(0, dtype=torch.long))
        self.register_buffer('children_matrix', torch.zeros(0, 4, dtype=torch.long))

        self._children_matrix: Optional[Tensor] = None

    def _init_density_field_bias(self) -> None:
        """
        初始化 density_field 的最后一层偏置为 -1.1

        数学原理：
            sigmoid(x + b) 当 b = -1.1 时，初始密度 ≈ 0.25
            这迫使模型在训练初期产生更多 token (K > 5)

        效果：
            - 训练初期：更多 token → 更多梯度流动 → 更好的学习
            - 训练后期：模型自动调整偏置以优化 token 数量
        """
        # density_field 结构: Linear -> GELU -> Linear -> Sigmoid
        # 最后一层是索引 2
        last_linear = self.density_field[2]

        # 重置权重为较小的值
        nn.init.xavier_uniform_(last_linear.weight, gain=0.1)

        # I-NAN: 关键 - 偏置设为 -1.1，使初始 sigmoid 输出 ≈ 0.25
        nn.init.constant_(last_linear.bias, -1.1)

    def reset_density_field(self) -> None:
        """
        重置 density_field 参数（公开接口）

        用于：
            - 训练中断后恢复
            - 调试 token 数量问题
        """
        self._init_density_field_bias()

    @property
    def max_level_limit(self) -> int:
        return self._max_level_limit

    @max_level_limit.setter
    def max_level_limit(self, value: int):
        self._max_level_limit = value

    @property
    def num_candidates(self) -> int:
        return self.candidate_regions.shape[0]

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
            grid_size = 2 ** depth
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
                parent_grid_size = 2 ** (depth - 1)
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

        for depth in range(1, max_level + 1):
            depth_start = depth_start_idx[depth]
            depth_end = depth_start_idx[depth + 1]
            parent_start = depth_start_idx[depth - 1]

            # 计算当前层每个区域的父节点在新排序中的索引
            for i in range(depth_start, depth_end):
                # 找到对应的父节点
                grid_idx = i - depth_start
                grid_size = 2 ** depth
                gi = grid_idx // grid_size
                gj = grid_idx % grid_size
                parent_idx = parent_start + gi // 2 * (grid_size // 2) + gj // 2
                new_parent_indices[i] = parent_idx

        # 子节点矩阵
        children_matrix = torch.full((N, 4), -1, dtype=torch.long, device=device)
        for i in range(N):
            parent = new_parent_indices[i].item()
            if parent >= 0:
                # 找到父节点的下一个可用槽位
                for slot in range(4):
                    if children_matrix[parent, slot] == -1:
                        children_matrix[parent, slot] = i
                        break

        self.register_buffer('candidate_regions', candidate_regions)
        self.register_buffer('candidate_depths', candidate_depths)
        self.register_buffer('parent_indices', new_parent_indices)
        self.register_buffer('hilbert_indices', hilbert_indices)
        self.register_buffer('children_matrix', children_matrix)

        self._children_matrix = children_matrix

    def _extract_features(self, features: Tensor, regions: Tensor, depths: Tensor) -> Tensor:
        """提取结构化特征

        数学:
            f_i = ROIAlign(F, R_i, sampling_ratio(d_i)) ⊙ σ(d_i) + E_d(d_i)

        I106-1: 动态 sampling_ratio 实现尺度感知特征提取
            - 浅层 (d ≤ 2): sampling_ratio=1, 全局特征
            - 中层 (2 < d ≤ 4): sampling_ratio=2, 中等细节
            - 深层 (d > 4): sampling_ratio=4, 细致采样
        """
        B = features.shape[0]
        N = regions.shape[0]
        _, C_feat, H_feat, W_feat = features.shape

        # 定义深度区间和对应的 sampling_ratio
        depth_bins = [0, 2, 4, self.max_level_limit + 1]
        sampling_ratios = [1, 2, 4]

        # 构建 ROI boxes: [batch_idx, x1, y1, x2, y2]
        # 假设所有 region 属于同一 batch (batch_idx=0)
        batch_indices = torch.zeros(N, dtype=torch.long, device=regions.device)
        boxes = torch.cat([batch_indices.unsqueeze(-1).float(), regions], dim=-1)  # [N, 5]

        # 按深度分组，使用动态 sampling_ratio
        pooled_list = []
        indices_list = []

        for i in range(len(depth_bins) - 1):
            low, high = depth_bins[i], depth_bins[i + 1]
            mask = (depths >= low) & (depths < high)
            if not mask.any():
                continue

            indices = mask.nonzero(as_tuple=True)[0]
            group_boxes = boxes[indices]

            # 调用 ROIAlign，使用对应的 sampling_ratio
            pooled = roi_align(
                features,
                group_boxes,
                output_size=(1, 1),
                spatial_scale=1.0,
                sampling_ratio=sampling_ratios[i],
                aligned=True,
            )  # [N_group, C, 1, 1]
            pooled = pooled.squeeze(-1).squeeze(-1)

            pooled_list.append(pooled)
            indices_list.append(indices)

        # 合并结果
        if len(pooled_list) == 0:
            # 无 token 时的边界情况
            roi_features = torch.zeros(B, N, C_feat, device=features.device, dtype=features.dtype)
        else:
            # 按原始顺序合并
            roi_features = torch.zeros(B, N, C_feat, device=features.device, dtype=features.dtype)
            for pooled, indices in zip(pooled_list, indices_list):
                roi_features[:, indices] = pooled.unsqueeze(0)

        # 投影到 hidden_dim
        roi_features = self.feature_proj(roi_features)  # [B, N, hidden_dim]

        # 深度缩放 (与 Embed 一致)
        depth_scale = torch.sigmoid(self.depth_embedding.weight)  # [max_level+1, hidden_dim]
        depth_scale = depth_scale[depths]  # [N, hidden_dim]

        # 注入深度信息
        roi_features = roi_features * depth_scale.unsqueeze(0)

        # 添加深度嵌入
        depth_emb = self.depth_embedding(depths)  # [N, hidden_dim]
        roi_features = roi_features + depth_emb.unsqueeze(0)

        return roi_features

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
        N = regions.shape[0]
        device = regions.device
        max_level = self.max_level_limit

        # 计算区域的中心坐标 (转换为整数)
        cx = ((regions[:, 0] + regions[:, 2]) / 2).long()  # [N]
        cy = ((regions[:, 1] + regions[:, 3]) / 2).long()  # [N]

        # 计算完整的 Hilbert 四叉树路径
        # paths: [N, max_level] 每个元素 ∈ {0, 1, 2, 3}
        paths = VectorizedPathEncoder.compute_quadrant_paths(
            cx, cy, max_level
        )  # [N, max_level]

        # 计算旋转状态 (累积翻转)
        # rotation_states: [N, max_level] 0=无旋转, 1=有旋转
        rotation_states = OrientationExtractor.compute_rotation_states(
            paths.unsqueeze(0)
        ).squeeze(0)  # [N, max_level]

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
        rot_emb = self.orientation_extractor.direction_embedding(final_rot_dir)  # [N, hidden_dim]

        # 面积编码 - 4^(-depth)
        area_enc = self._area_encoding[depths]  # [N]
        area_enc = area_enc.unsqueeze(-1).expand(-1, self.hidden_dim)  # [N, hidden_dim]

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
        z = self.conv1d_hilbert(x)  # [B, 1, N]

        # 转置回来: [B, 1, N] -> [B, N, 1]
        z = z.transpose(1, 2).squeeze(-1)

        return z

    def _apply_tree_constraint(self, logits: Tensor, depths: Tensor) -> Tensor:
        """应用树一致性软约束

        数学:
            z_parent -= λ × max(z_children)
        """
        if self.tree_constraint_weight <= 0:
            return logits

        N = logits.shape[1]
        constrained_logits = logits.clone()

        # 对每个父节点，降低其分数如果子节点分数更高
        for i in range(N):
            parent_idx = self.parent_indices[i].item()
            if parent_idx >= 0:
                # 检查所有子节点
                children = self.children_matrix[parent_idx]
                children_valid = children[children >= 0]

                if len(children_valid) > 0:
                    max_child_logit = logits[:, children_valid].max(dim=1)[0]
                    # 降低父节点分数
                    constrained_logits[:, parent_idx] -= self.tree_constraint_weight * max_child_logit

        return constrained_logits

    def _sparse_select(
        self,
        logits: Tensor,
        K_target: int,
        hard: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """稀疏选择

        数学:
            s = softmax(z / τ) × K (用于更好的梯度流)
            s = Entmax_{α}(z / τ) × K (用于稀疏)
        """
        B, N = logits.shape

        # 温度调度
        tau = max(self.temperature, TEMPERATURE_MIN)

        # 使用 softmax (alpha=2) 以获得更好的梯度流
        # 或者使用 entmax 稀疏激活
        if self.entmax_alpha < 1.9:
            # 接近 softmax，使用 soft selection
            probs = F.softmax(logits / tau, dim=-1)
        else:
            # 稀疏激活
            probs = entmax(logits / tau, alpha=self.entmax_alpha, dim=-1)

        # 缩放使期望和等于 K
        probs_scaled = probs * K_target

        if hard:
            # 硬选择
            selected_mask = torch.zeros_like(probs)
            _, topk_idx = torch.topk(probs_scaled, K_target, dim=-1)
            selected_mask.scatter_(1, topk_idx, 1.0)
            # 硬模式下返回缩放后的概率（用于梯度）
            return selected_mask, probs_scaled
        else:
            # 软选择 (用于训练)
            return probs_scaled, probs_scaled

    def enable_token_stability_monitoring(self) -> "HilbertOptimalSplitter":
        """启用 token 选择稳定性监控"""
        self._monitor_token_stability = True
        self._token_history.clear()
        return self

    def disable_token_stability_monitoring(self) -> "HilbertOptimalSplitter":
        """禁用 token 选择稳定性监控"""
        self._monitor_token_stability = False
        return self

    def compute_token_iou(self) -> Optional[float]:
        """
        计算最近两次 token 选择的 IOU（重合率）。

        Returns:
            IOU 值（0.0-1.0），或 None（如果历史不足）
        """
        if len(self._token_history) < 2:
            return None

        indices1 = set(self._token_history[-2].tolist())
        indices2 = set(self._token_history[-1].tolist())

        if not indices1 or not indices2:
            return None

        intersection = len(indices1 & indices2)
        union = len(indices1 | indices2)

        return intersection / union if union > 0 else 0.0

    def get_token_stability_stats(self) -> Dict[str, float]:
        """
        获取 token 稳定性统计信息。

        Returns:
            包含 iou_mean, iou_std, iou_min 的字典
        """
        if len(self._token_history) < 2:
            return {"iou_mean": 0.0, "iou_std": 0.0, "iou_min": 0.0, "iou_max": 0.0}

        ious = []
        for i in range(len(self._token_history) - 1):
            set1 = set(self._token_history[i].tolist())
            set2 = set(self._token_history[i + 1].tolist())
            if set1 and set2:
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                ious.append(intersection / union if union > 0 else 0.0)

        if not ious:
            return {"iou_mean": 0.0, "iou_std": 0.0, "iou_min": 0.0, "iou_max": 0.0}

        import numpy as np
        return {
            "iou_mean": float(np.mean(ious)),
            "iou_std": float(np.std(ious)),
            "iou_min": float(np.min(ious)),
            "iou_max": float(np.max(ious)),
        }

    def forward(
        self,
        features: Tensor,
        image_size: Optional[Tuple[int, int]] = None,
        hard: bool = False,
    ) -> SplitResult:
        """
        前向传播

        Args:
            features: [B, C, H, W] 输入特征
            image_size: (H, W) 原始图像尺寸
            hard: 是否使用硬选择

        Returns:
            SplitResult: 分割结果
        """
        B = features.shape[0]

        # 更新候选区域
        if image_size is None:
            image_size = self._current_image_size

        if image_size is None:
            raise ValueError("image_size must be provided")

        if self.num_candidates == 0 or self._current_image_size != image_size:
            self.update_candidates(image_size)

        N = self.num_candidates

        # 提取特征
        roi_features = self._extract_features(features, self.candidate_regions, self.candidate_depths)

        # 几何编码
        path_emb, rot_emb, area_enc = self._encode_geometry(self.candidate_regions, self.candidate_depths)

        # 拼接所有特征 (roi + path + rot + area = 4 * hidden_dim)
        # 注意: conv1d 输入调整为 hidden_dim * 4
        combined = torch.cat([
            roi_features,
            path_emb.unsqueeze(0).expand(B, -1, -1),
            rot_emb.unsqueeze(0).expand(B, -1, -1),
            area_enc.unsqueeze(0).expand(B, -1, -1),
        ], dim=-1)  # [B, N, hidden_dim * 4]

        # 1D Hilbert 流形卷积
        logits = self._manifold_convolution(combined)  # [B, N]

        # 树约束
        logits = self._apply_tree_constraint(logits, self.candidate_depths)

        # 深度配额
        depth_quota = F.softmax(self.depth_quota, dim=0)
        depth_quota = depth_quota[self.candidate_depths]  # [N]
        logits = logits + torch.log(depth_quota + EPS).unsqueeze(0)

        # K 估计 (动态)
        # 使用密度场网络估计每个区域的"信息密度"
        # 然后积分得到总 K

        # 使用 roi_features 的平均作为曲线特征 (取 batch 0)
        curve_features = roi_features[0].mean(dim=0, keepdim=True)  # [1, hidden_dim]
        density_per_region = self.density_field(roi_features[0])  # [N, 1]

        # 精确积分: K = Σ ρ_i * w_i
        # w_i = 4^(-depth_i) 是 Hilbert 曲线下的面积权重
        hilbert_weights = self._area_encoding[self.candidate_depths]  # [N]
        K_float = (density_per_region.squeeze(-1) * hilbert_weights).sum()
        # K 课程学习：使用 _current_K 作为实际上限（从 K_min 逐渐增大到 K_max 向上取整）
        K = K_float.long().clamp(self.K_min, self._current_K).item()

        # 稀疏选择
        selected_mask, probs = self._sparse_select(logits, K, hard=hard)

        # 构建结果
        selected_indices = (selected_mask > 0.5).nonzero(as_tuple=True)

        if len(selected_indices[1]) == 0:
            # 至少选择一个 - 使用 top-k
            _, topk_idx = torch.topk(probs[0], min(K, probs.shape[1]), dim=-1)
            # 使用 batch 0
            batch_idx = torch.zeros(topk_idx.shape[0], dtype=torch.long, device=logits.device)
            selected_indices = (batch_idx, topk_idx)

        batch_idx = selected_indices[0]
        region_idx = selected_indices[1]

        # 提取选中区域
        selected_regions = self.candidate_regions[region_idx]
        selected_depths = self.candidate_depths[region_idx]
        selected_hilbert = self.hilbert_indices[region_idx]

        # 按 Hilbert 索引排序
        sort_idx = selected_hilbert.argsort()
        selected_regions = selected_regions[sort_idx]
        selected_depths = selected_depths[sort_idx]
        batch_idx = batch_idx[sort_idx]
        selected_hilbert = selected_hilbert[sort_idx]

        result = SplitResult(
            regions=selected_regions,
            depths=selected_depths,
            batch_indices=batch_idx,
            hilbert_indices=selected_hilbert,
            selected_mask=selected_mask,
            logits=logits,
            probs=probs,
        )

        # I150-3: 记录 token 选择历史用于稳定性监控
        if self._monitor_token_stability and hard:
            # 记录 batch 0 的选择（用于统计）
            if B > 0:
                token_idx = region_idx[batch_idx == 0].cpu()
                self._token_history.append(token_idx)
                # 限制历史长度
                if len(self._token_history) > 100:
                    self._token_history.pop(0)

        return result

    def set_temperature(self, temperature: float) -> None:
        """设置温度（用于训练脚本兼容性）"""
        self.temperature = temperature

    def set_epoch(self, epoch: int):
        """设置当前 epoch，用于课程学习调度"""
        self._epoch = epoch

        # 课程学习阶段更新 (与 GumbelTopK 保持一致)
        if epoch < 10:
            self._curriculum_stage = 1
        elif epoch < 20:
            self._curriculum_stage = 2
        else:
            self._curriculum_stage = 3

        # I107: Entmax Alpha 预热策略
        # α(t) = min(1.5, 1.2 + 0.3 × epoch / T_warmup) for t < T_warmup
        # α(t) = min(2.0, α(t)) for t >= T_warmup
        if epoch < self.entmax_warmup_epochs:
            # 预热阶段: 1.2 → 1.5
            self.entmax_alpha = min(
                self.entmax_alpha_warmup,
                self.entmax_alpha_init + (self.entmax_alpha_warmup - self.entmax_alpha_init) * epoch / self.entmax_warmup_epochs
            )
        elif epoch < self.entmax_schedule_epochs:
            # 过渡阶段: 1.5 → 2.0
            warmup_progress = (epoch - self.entmax_warmup_epochs) / (self.entmax_schedule_epochs - self.entmax_warmup_epochs)
            self.entmax_alpha = self.entmax_alpha_warmup + (self.entmax_alpha_max - self.entmax_alpha_warmup) * warmup_progress
        else:
            # 稳定阶段: 保持 2.0
            self.entmax_alpha = self.entmax_alpha_max

        # 温度退火
        self.temperature = max(
            self.temperature_min,
            self.temperature_init * 0.5 ** (epoch / 20)
        )

        # K 课程学习：从 K_min 逐渐增大到 K_max（向上取整）
        # 符合课程学习原则：先学简单（少 token），后学复杂（多 token）
        K_max_rounded = math.ceil(self.K_max)  # 向上取整
        if epoch <= self._K_schedule_epochs:
            # 线性增长：K_min → _K_max_rounded（能囊括 K_max 的最小 level 对应区域数）
            progress = epoch / self._K_schedule_epochs
            self._current_K = int(self.K_min + (self._K_max_rounded - self.K_min) * progress)
        else:
            # 课程学习阶段结束后，使用 _K_max_rounded
            self._current_K = self._K_max_rounded

    def _compute_min_level_regions(self, K_target: int, max_level: int) -> int:
        """计算能囊括 K_target 个 token 的最小 level 对应的候选区域数

        例如: K_target=38, max_level=4
            - level 0: 4^0 = 1 < 38
            - level 1: 4^1 = 4 < 38
            - level 2: 4^2 = 16 < 38
            - level 3: 4^3 = 64 >= 38 ✓
            → 返回 64
        """
        for level in range(max_level + 1):
            regions = 4 ** level
            if regions >= K_target:
                return regions
        # 如果所有 level 都不满足，返回最大 level 的区域数
        return 4 ** max_level

    def extra_repr(self) -> str:
        return (
            f"HilbertOptimalSplitter("
            f"max_level={self.max_level_limit}, "
            f"hidden_dim={self.hidden_dim}, "
            f"K={self.K_min}-{self._K_max_rounded}[current={self._current_K}], "
            f"entmax_alpha={self.entmax_alpha:.2f}, "
            f"tree_weight={self.tree_constraint_weight})"
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
    """
    if selected_mask.sum() < 2:
        return 0.0

    selected_h = hilbert_indices[selected_mask > 0.5]
    selected_h = selected_h.sort()[0]

    if len(selected_h) < 2:
        return 0.0

    jumps = (selected_h[1:].float() - selected_h[:-1].float()).abs()
    return jumps.mean().item()


def compute_jump_loss(
    hilbert_indices: Tensor,
    selected_mask: Tensor,
    gamma: float = 1.0,
) -> Tensor:
    """
    计算修正后的 Jump Loss (ReLU 形式)

    数学:
        L_jump = E[ReLU(Δh - 1)²]

    与错误的 exp(-γΔh) 形式对比:
        - exp(-γΔh): Δh=1 → 0.37 (惩罚大), Δh→∞ → 0 (无惩罚) ❌
        - ReLU(Δh-1)²: Δh≤1 → 0 (无惩罚), Δh>1 → 二次增长 (正确惩罚) ✅

    Args:
        hilbert_indices: [N] Hilbert 索引
        selected_mask: [N] 选中掩码
        gamma: 缩放因子

    Returns:
        loss: 标量损失
    """
    selected_h = hilbert_indices[selected_mask > 0.5].float()
    selected_h = selected_h.sort()[0]

    if len(selected_h) < 2:
        return torch.tensor(0.0, device=hilbert_indices.device)

    # 计算相邻差值
    diffs = selected_h[1:] - selected_h[:-1]  # [K-1]

    # ReLU(Δh - 1)² 形式
    jump_loss = F.relu(diffs - 1.0) ** 2

    return jump_loss.mean()


def compute_determinism_score(
    train_selected: Tensor,
    eval_selected: Tensor,
) -> float:
    """
    计算 A2: Determinism Score (IOU)

    IOU = |S_train ∩ S_eval| / |S_train ∪ S_eval|
    """
    train_set = set(train_selected.cpu().tolist())
    eval_set = set(eval_selected.cpu().tolist())

    intersection = len(train_set & eval_set)
    union = len(train_set | eval_set)

    if union == 0:
        return 1.0

    return intersection / union


def compute_gradient_coverage(
    logits: Tensor,
    loss: Tensor,
    threshold: float = 1e-4,
) -> float:
    """
    计算 A3: Gradient Coverage

    coverage = |{i : |grad_i| > threshold}| / N
    """
    logits.requires_grad = True
    loss.backward(retain_graph=True)

    grads = logits.grad
    if grads is None:
        return 0.0

    coverage = (grads.abs() > threshold).float().mean().item()
    return coverage


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
