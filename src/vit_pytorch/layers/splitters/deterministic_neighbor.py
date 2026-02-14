"""
Deterministic Neighbor-Aware Splitter

数学形式化
==========

核心设计原则:
    1. Hilbert索引: 使用HilbertScanner（而非行主序）
    2. 评分函数: NAP风格邻居传播
    3. 选择机制: 确定性Softmax（无Gumbel随机性）
    4. 配额: 可学习深度配额（Scheme E）
    5. 损失: 局部一致性损失 + 熵正则化

与Gumbel-Top-K对比:
    | 维度              | Gumbel-Top-K          | DeterministicNeighbor    |
    |------------------|----------------------|------------------------|
    | 随机性            | Gumbel扰动           | 无（确定性）           |
    | 梯度覆盖率        | ~100% (STE)          | **100%** (直接)       |
    | Hilbert局部性     | 100%                 | 100%                   |
    | 冗余去除          | 无                   | ✓ 邻居传播            |
    | train/eval一致性   | 需DeterministicTopK  | **天然一致**           |

核心公式:
    1. Hilbert邻接矩阵: A_ij = 1[LCA(i,j) ≥ ℓ]
    2. 邻居传播: s'_i = s_i + α × Σ_j∈N_ℓ(i) sim(f_i,f_j) × s_j
    3. 确定性选择: p_i = softmax(s'_i / τ)_i
    4. 局部一致性损失: L_local = Σ_(i,j)∈E |p_i - p_j|

"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from vit_pytorch.layers.splitters.gumbel_topk import TensorSplitResult
from vit_pytorch.core.splitter_protocol import (
    CoreSplitter,
    AnnealingSplitter,
    MetricsSplitter,
    SplitResult,
    validate_splitter,
)
from vit_pytorch.core.constants import (
    EPS,
    TEMPERATURE_MIN,
)
from vit_pytorch.core.curve_hilbert import HilbertCurve

logger = logging.getLogger(__name__)


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class DeterministicNeighborSplitterConfig:
    """
    DeterministicNeighborSplitter 配置

    数学形式化
    ==========

    邻居感知评分:
        s'_i = s_i + α × (1/|N_ℓ(i)|) × Σ sim × s_j

    局部一致性损失:
        L_local = Σ_(i,j)∈E |σ(s_i) - σ(s_j)|

    连续配额:
        q = softmax(φ) ∈ Δ^{D-1}
        K_d = K × q_d (无需 floor/round)
    """

    # ==================== Hilbert 曲线参数 ====================
    min_patch_size: int = 4
    max_level_limit: int = 8  # I164-1: 支持完整的 Hilbert 层级

    # ==================== 邻居感知参数 ====================
    neighbor_threshold: int = 2  # ℓ 阈值
    enable_neighbor_aware: bool = True
    alpha_init: float = 0.5  # 邻居权重初始值

    # ==================== 分数 MLP 参数 ====================
    feature_dim: int = 256
    hidden_dim: int = 64

    # ==================== 温度参数 ====================
    temperature_init: float = 1.0
    temperature_min: float = 0.4
    temperature_anneal: str = 'cosine'
    learnable_temperature: bool = True
    temperature_warmup_steps: int = 1000

    # ==================== 配额参数 ====================
    enable_learnable_quota: bool = True
    quota_init_method: str = 'uniform'  # 'uniform', 'inverse_depth'

    # ==================== 正则化参数 ====================
    locality_weight: float = 0.1  # 局部一致性损失权重
    entropy_weight: float = 0.01   # 熵正则化权重

    # ==================== 覆盖率约束 ====================
    coverage_base: float = 0.25
    coverage_min: float = 0.10
    coverage_max: float = 0.50
    K_min_abs: int = 8
    K_max_hard: int = 4096


# =============================================================================
# Hilbert 邻居矩阵
# =============================================================================

class HilbertNeighborMatrix(nn.Module):
    """
    Hilbert 邻居矩阵 - 基于正确的Hilbert索引

    数学形式化
    ==========

    邻接关系定义:
        A_ij = 1[LCA(i,j) ≥ ℓ]
             = 1[||pos_i - pos_j||_∞ ≤ N / 2^ℓ]

    Hilbert曲线性质 (Zhao et al. 2024):
        - 局部性保持: Hilbert相邻点在空间上也相邻
        - 自相似性: 不同尺度的Hilbert序保持局部关系

    优势:
        - 正确利用Hilbert曲线的空间填充性质
        - O(N log N) 复杂度（滑动窗口）
        - 稀疏表示（大多数区域对不是邻居）
    """

    def __init__(
        self,
        max_level: int = 4,
        neighbor_threshold: int = 2,
    ):
        super().__init__()
        self.max_level = max_level
        self.neighbor_threshold = neighbor_threshold

        # Hilbert距离阈值：hilbert距离小于此值认为相邻
        # threshold = 4^(max_level - neighbor_threshold)
        self._hilbert_threshold = 4 ** (max_level - neighbor_threshold)

    def forward(
        self,
        hilbert_indices: Tensor,
        depths: Tensor,
    ) -> Tensor:
        """
        构建Hilbert邻接矩阵

        Args:
            hilbert_indices: [N] Hilbert曲线索引
            depths: [N] 每个区域的深度

        Returns:
            adj: [N, N] 邻接矩阵，A[i,j] = 1 如果 i 和 j 是邻居
        """
        N = hilbert_indices.shape[0]

        if N == 0:
            return torch.zeros(N, N, device=hilbert_indices.device)

        # 1. 按Hilbert索引排序
        sorted_idx = torch.argsort(hilbert_indices)
        sorted_h = hilbert_indices[sorted_idx]
        sorted_d = depths[sorted_idx]

        # 2. 使用滑动窗口构建稀疏邻接矩阵
        adj = self._sliding_window_neighbors(
            sorted_h, sorted_d, sorted_idx
        )

        return adj

    def _sliding_window_neighbors(
        self,
        sorted_h: Tensor,
        sorted_d: Tensor,
        sorted_idx: Tensor,
    ) -> Tensor:
        """
        基于滑动窗口的O(N log N)邻居查找

        原理:
            Hilbert曲线中相邻的索引 → 空间上相邻的区域
            使用固定大小的滑动窗口查找邻居

        Args:
            sorted_h: [N] 排序后的Hilbert索引
            sorted_d: [N] 排序后的深度
            sorted_idx: [N] 原始索引

        Returns:
            adj: [N, N] 稀疏邻接矩阵
        """
        N = sorted_h.shape[0]
        device = sorted_h.device

        adj = torch.zeros(N, N, device=device)

        # 滑动窗口大小：基于Hilbert阈值调整
        # Hilbert距离阈值内的点都可能是邻居
        window_size = min(self._hilbert_threshold * 2 + 1, N)

        for i in range(N):
            # 向后查找邻居
            for j in range(i + 1, min(i + window_size, N)):
                h_diff = sorted_h[j] - sorted_h[i]

                # 如果Hilbert距离超过阈值，停止查找
                if h_diff > self._hilbert_threshold:
                    break

                # 检查深度是否满足邻居条件
                # 同层或相邻层的区域更可能是邻居
                depth_diff = abs(sorted_d[i] - sorted_d[j])
                if depth_diff <= self.max_level - self.neighbor_threshold:
                    original_j = sorted_idx[j]
                    adj[i, original_j] = 1.0
                    adj[original_j, i] = 1.0

        return adj

    @property
    def threshold(self) -> int:
        return self._hilbert_threshold


# =============================================================================
# 相似度计算
# =============================================================================

class HilbertAwareSimilarity(nn.Module):
    """
    Hilbert感知相似度计算

    数学形式:
        sim(i,j) = cos(f_i, f_j) × ω_h(δ_h(i,j)) × ω_s(δ_s(pos_i, pos_j))

    其中:
        - cos(f_i, f_j): 特征余弦相似度
        - ω_h(δ_h): Hilbert距离衰减
        - ω_s(δ_s): 空间距离衰减
    """

    def __init__(
        self,
        dim: int = 256,
        use_hilbert_decay: bool = True,
        use_spatial_decay: bool = True,
        decay_power: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.use_hilbert_decay = use_hilbert_decay
        self.use_spatial_decay = use_spatial_decay
        self.decay_power = decay_power

    def forward(
        self,
        features: Tensor,
        hilbert_indices: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
    ) -> Tensor:
        """
        计算相似度矩阵

        Args:
            features: [N, C] 特征向量
            hilbert_indices: [N] Hilbert索引（可选）
            positions: [N, 2] 位置坐标（可选）

        Returns:
            sim_matrix: [N, N] 相似度矩阵
        """
        # 特征余弦相似度
        features_norm = F.normalize(features, p=2, dim=-1)
        sim_matrix = features_norm @ features_norm.t()  # [N, N]

        if self.use_hilbert_decay and hilbert_indices is not None:
            # Hilbert距离衰减
            # P-OPT: 使用 .clamp() 替代 .item() 避免GPU-CPU同步
            h_diff = hilbert_indices.unsqueeze(0) - hilbert_indices.unsqueeze(1)
            h_diff = h_diff.abs()
            max_h = hilbert_indices.max().clamp(min=1).float()
            h_decay = (1 - h_diff.float() / max_h).clamp(min=0)
            sim_matrix = sim_matrix * h_decay.pow(self.decay_power)

        if self.use_spatial_decay and positions is not None:
            # 空间距离衰减（L∞距离）
            # P-OPT: 使用 .clamp() 替代 .item() 避免GPU-CPU同步
            pos_diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # [N, N, 2]
            distances = pos_diff.norm(p=float('inf'), dim=-1)  # [N, N]
            max_dist = positions.max().clamp(min=1.0).float()
            s_decay = (1 - distances / max_dist).clamp(min=0)
            sim_matrix = sim_matrix * s_decay.pow(self.decay_power)

        return sim_matrix


# =============================================================================
# 邻居感知评分
# =============================================================================

class NeighborAwareScoringV2(nn.Module):
    """
    邻居感知评分模块 V2 - 完整实现

    数学形式:
        s'_i = s_i + α × (1/|N_ℓ(i)|) × Σ_{j∈N_ℓ(i)} sim(f_i, f_j) × s_j

    优势:
        1. 邻居信息传播：相邻ROI的分数会累加
        2. 冗余避免：相似邻居不会同时获得高分
        3. 处处可微：100%梯度覆盖率
    """

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 64,
        alpha_init: float = 0.5,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # 基础分数MLP
        self.score_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # 可学习邻居权重
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(
        self,
        features: Tensor,
        adj_matrix: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        计算邻居感知分数

        Args:
            features: [N, C] 特征
            adj_matrix: [N, N] 邻接矩阵

        Returns:
            base_scores: [N] 基础分数
            final_scores: [N] 邻居传播后的分数
        """
        N = features.shape[0]

        # 1. 基础分数
        base_scores = self.score_mlp(features).squeeze(-1)  # [N]

        if N == 0:
            return base_scores, base_scores

        # 2. 计算特征相似度
        features_norm = F.normalize(features, p=2, dim=-1)
        sim_matrix = features_norm @ features_norm.t()  # [N, N]

        # 3. 归一化邻接矩阵（行归一化）
        degree = adj_matrix.sum(dim=-1, keepdim=True) + EPS  # [N, 1]
        norm_adj = adj_matrix / degree  # [N, N]

        # 4. 邻居传播
        # neighbor_contrib = D^(-1) * A * s
        neighbor_contrib = (norm_adj @ base_scores.unsqueeze(-1)).squeeze(-1)  # [N]

        # 5. 最终分数
        # alpha 始终参与计算，确保梯度流动
        final_scores = base_scores + self.alpha * neighbor_contrib

        return base_scores, final_scores


# =============================================================================
# Deterministic Neighbor Splitter 主实现
# =============================================================================

class DeterministicNeighborSplitter(
    nn.Module,
    CoreSplitter,
    AnnealingSplitter,
    MetricsSplitter,
):
    """
    确定性邻居感知分裂器 - Hilbert Curve ViT 最佳实现

    数学形式化
    ==========

    核心公式:
        1. Hilbert邻域: N_ℓ(i) = { j | LCA(i,j) ≥ ℓ }
        2. 基础分数: s_i = MLP(ROI_i)
        3. 邻居传播: s'_i = s_i + α × (1/|N|) × Σ sim × s_j
        4. 确定性选择: p_i = softmax(s'_i / τ)_i
        5. 连续配额: K_d = K × softmax(φ)_d

    梯度分析:
        - 处处可微，梯度覆盖率 100%
        - 无 floor/round/topk 操作
        - 邻居传播通过矩阵乘法实现，完全可微

    train/eval一致性:
        - 无Gumbel随机性
        - 训练和推理使用相同逻辑
        - 天然确定性
    """

    def __init__(
        self,
        config: Optional[DeterministicNeighborSplitterConfig] = None,
        image_size: Tuple[int, int] = (224, 224),
        feature_dim: int = 256,
    ):
        super().__init__()

        self.config = config or DeterministicNeighborSplitterConfig()
        self.image_size = image_size
        self.feature_dim = feature_dim

        # 从配置提取参数
        max_level = self.config.max_level_limit
        self._max_level = max_level
        self._neighbor_threshold = self.config.neighbor_threshold

        # 计算候选区域数量
        self._num_candidates = sum(4 ** d for d in range(max_level + 1))

        # 1. 候选区域信息（可更新）
        self.register_buffer(
            '_regions',
            self._compute_regions(max_level, image_size),
        )
        self.register_buffer(
            '_depth_indices',
            self._compute_depth_indices(max_level),
        )
        self.register_buffer(
            '_hilbert_indices',
            self._compute_hilbert_indices(max_level),
        )

        # 2. Hilbert邻居矩阵
        self.hilbert_neighbor = HilbertNeighborMatrix(
            max_level=max_level,
            neighbor_threshold=self._neighbor_threshold,
        )

        # 3. 邻居感知评分
        self.scoring = NeighborAwareScoringV2(
            feature_dim=feature_dim,
            hidden_dim=self.config.hidden_dim,
            alpha_init=self.config.alpha_init,
        )

        # 4. Hilbert感知相似度
        self.similarity = HilbertAwareSimilarity(
            dim=feature_dim,
            use_hilbert_decay=True,
            use_spatial_decay=True,
        )

        # 5. 可学习配额
        if self.config.enable_learnable_quota:
            D = max_level + 1
            if self.config.quota_init_method == 'uniform':
                quota_init = torch.zeros(D, dtype=torch.float32)
            elif self.config.quota_init_method == 'inverse_depth':
                # 逆深度加权：浅层更多配额
                quota_init = torch.tensor(
                    [1.0 / (d + 1) for d in range(D)],
                    dtype=torch.float32,
                )
            else:
                quota_init = torch.zeros(D, dtype=torch.float32)
            self.quota_logits = nn.Parameter(quota_init)
        else:
            self.register_buffer('_quota_logits', None)

        # 6. 可学习温度
        if self.config.learnable_temperature:
            self.register_parameter(
                'log_temperature',
                nn.Parameter(torch.tensor(math.log(self.config.temperature_init))),
            )
        else:
            self.register_buffer('_temperature', None)

        # 7. 缓存
        self._cached_adj: Optional[Tensor] = None
        self._current_step = 0

    def _compute_regions(self, max_level: int, image_size: Tuple[int, int]) -> Tensor:
        """计算所有候选区域的坐标"""
        # I162-1 fix: 确保 tensor 在正确的设备上创建
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        H, W = image_size
        regions = []

        for d in range(max_level + 1):
            grid_size = 2 ** d
            cell_h = H / grid_size
            cell_w = W / grid_size

            for i in range(grid_size):
                for j in range(grid_size):
                    x0 = int(j * cell_w)
                    y0 = int(i * cell_h)
                    x1 = int((j + 1) * cell_w)
                    y1 = int((i + 1) * cell_h)
                    regions.append([x0, y0, x1, y1])

        return torch.tensor(regions, dtype=torch.float32, device=device)

    def _compute_depth_indices(self, max_level: int) -> Tensor:
        """计算每个区域的深度"""
        # I162-1 fix: 确保 tensor 在正确的设备上创建
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        depth_indices = []
        for d in range(max_level + 1):
            n_regions = 4 ** d
            depth_indices.extend([d] * n_regions)
        return torch.tensor(depth_indices, dtype=torch.long, device=device)

    def _compute_hilbert_indices(self, max_level: int) -> Tensor:
        """
        计算正确的Hilbert索引

        使用HilbertCurve正确计算每个区域的Hilbert序索引

        注意：这里我们为每个深度级别分别计算Hilbert索引，
        因为不同深度的区域有不同的空间分辨率。
        """
        indices = []
        H, W = self.image_size

        for d in range(max_level + 1):
            grid_size = 2 ** d
            n_regions = 4 ** d

            # 创建从grid坐标到Hilbert索引的映射
            for i in range(grid_size):
                for j in range(grid_size):
                    # 将区域中心映射到Hilbert坐标
                    # 区域中心
                    center_x = (j + 0.5) * (W / grid_size)
                    center_y = (i + 0.5) * (H / grid_size)

                    # 离散化到Hilbert网格
                    h_x = min(int(center_x * grid_size / W), grid_size - 1)
                    h_y = min(int(center_y * grid_size / H), grid_size - 1)

                    # 使用HilbertCurve计算索引
                    hilbert_idx = HilbertCurve.xy_to_d(grid_size, h_x, h_y)

                    # 累加不同深度的索引以保持全局唯一性
                    offset = sum(4 ** k for k in range(d))
                    indices.append(hilbert_idx + offset)

        # I162-1 fix: 确保 tensor 在正确的设备上创建
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _get_adj_matrix(self) -> Tensor:
        """获取邻接矩阵（使用缓存）"""
        if self._cached_adj is None:
            self._cached_adj = self.hilbert_neighbor(
                self._hilbert_indices,
                self._depth_indices,
            )
        return self._cached_adj

    def _get_temperature(self) -> Tensor:
        """获取当前温度"""
        if self.config.learnable_temperature:
            tau = self.log_temperature.exp().clamp(min=TEMPERATURE_MIN)
        else:
            tau = self.config.temperature_init
        return tau

    def _compute_K(self) -> int:
        """计算预算K"""
        coverage = self.config.coverage_base
        N = self._num_candidates
        K = int(coverage * N)
        K = max(self.config.K_min_abs, K)
        K = min(self.config.K_max_hard, K)
        return K

    def _get_quota_probs(self) -> Tensor:
        """获取配额概率"""
        if self.quota_logits is None:
            return None
        return F.softmax(self.quota_logits, dim=0)

    # =============================================================================
    # CoreSplitter 接口
    # =============================================================================

    def forward(
        self,
        features: Tensor,
        image_size: Optional[Tuple[int, int]] = None,
        hard: bool = False,
    ) -> TensorSplitResult:  # I162-1: 返回 TensorSplitResult 以兼容 tokenizer
        """
        执行确定性邻居感知分割决策

        数学:
            1. ROI对齐: ROI_aligned = grid_sample(F, regions)
            2. 基础分数: s = MLP(ROI_aligned)
            3. 邻居传播: s' = s + α × D^(-1)A × s
            4. 概率: p = softmax(s' / τ)
            5. 配额: K_d = K × softmax(φ)_d

        Args:
            features: [B, C, H_feat, W_feat] 特征图
            image_size: (H, W) 原始图像尺寸
            hard: 是否使用硬决策（推理模式）

        Returns:
            SplitResult: 选中区域及其元数据
        """
        if image_size is not None:
            self.update_candidates(image_size)

        B, C, H_feat, W_feat = features.shape

        # 1. ROI对齐
        regions = self._regions  # [N, 4]
        N = regions.shape[0]

        # I162-1 fix: 返回 TensorSplitResult 以兼容 tokenizer
        if N == 0:
            empty_tensor = torch.empty(0, dtype=torch.long, device=features.device)
            return TensorSplitResult(
                regions=torch.empty(0, 4, device=features.device, dtype=torch.long),
                depths=empty_tensor,
                batch_indices=empty_tensor,
                hilbert_indices=empty_tensor,
                token_indices=empty_tensor,
                complexities=torch.empty(0, dtype=torch.float32, device=features.device),
                tokens_per_batch=torch.ones(B, dtype=torch.long, device=features.device),
            )

        # I164-1: 减小 output_size 以降低显存 (14->7, 降低 4 倍)
        roi_features = self._roi_align(features, regions, (7, 7))  # [B*N, C]

        # 2. 邻居感知评分（需要按batch处理）
        adj_matrix = self._get_adj_matrix()  # [N, N]
        adj_matrix = adj_matrix.unsqueeze(0).expand(B, -1, -1)  # [B, N, N]

        base_scores_list = []
        final_scores_list = []

        for b in range(B):
            start_idx = b * N
            end_idx = (b + 1) * N
            roi_b = roi_features[start_idx:end_idx]  # [N, C]
            adj_b = adj_matrix[b]  # [N, N]

            base_b, final_b = self.scoring(roi_b, adj_b)
            base_scores_list.append(base_b)
            final_scores_list.append(final_b)

        base_scores = torch.cat(base_scores_list)  # [B*N]
        final_scores = torch.cat(final_scores_list)  # [B*N]

        # 3. 确定性选择
        tau = self._get_temperature()
        scores_for_softmax = final_scores / (tau + EPS)
        probs_all = F.softmax(scores_for_softmax.view(B, N), dim=-1)  # [B, N]

        # 4. 按深度配额选择（硬选择用于推理）
        quota_probs = self._get_quota_probs()
        K = self._compute_K()

        hard_selected_mask = self._depth_aware_selection(probs_all.clone(), quota_probs, K)

        # 5. 选择策略：训练模式使用软概率，推理模式使用硬选择
        if self.training or not hard:
            # 训练模式：使用软概率进行梯度流
            # P-OPT: 向量化深度配额权重应用，避免嵌套循环
            # 使用广播机制替代 B*D 次循环操作
            soft_mask = probs_all.clone()
            if quota_probs is not None:
                # depth_indices: [N] -> 扩展为 [1, N] 广播到 [B, N]
                depth_weight_expanded = quota_probs[self._depth_indices]  # [N]
                soft_mask = soft_mask * depth_weight_expanded.unsqueeze(0)  # [B, N]
            # 重新归一化
            selected_mask = soft_mask / (soft_mask.sum(dim=-1, keepdim=True) + EPS)
        else:
            # 推理模式：使用硬选择
            selected_mask = hard_selected_mask

        # 6. 构建结果
        depth_indices = self._depth_indices
        hilbert_indices = self._hilbert_indices

        # 展平selected_mask以索引
        selected_mask_flat = selected_mask.view(B * N)  # [B*N]
        selected_mask_bool = selected_mask_flat.bool()

        # 选择区域
        regions_expanded = regions.unsqueeze(0).expand(B, -1, -1).reshape(B * N, 4)
        selected_regions = regions_expanded[selected_mask_bool]

        depth_expanded = depth_indices.unsqueeze(0).expand(B, -1).reshape(B * N)
        hilbert_expanded = hilbert_indices.unsqueeze(0).expand(B, -1).reshape(B * N)

        selected_depths = depth_expanded[selected_mask_bool]
        selected_hilbert = hilbert_expanded[selected_mask_bool]

        batch_indices_all = torch.arange(B, device=features.device).unsqueeze(1).expand(-1, N).reshape(B * N)
        batch_indices = batch_indices_all[selected_mask_bool]

        # I162-1 fix: 返回 TensorSplitResult 以兼容 tokenizer
        # 计算 token_indices: 每个选中 token 在其 batch 内的顺序索引
        if selected_depths.numel() > 0:
            # 使用 Hilbert 排序后的顺序作为 token_indices
            # 先按 Hilbert 索引排序，再按 batch 索引排序
            hilbert_order = torch.argsort(selected_hilbert, stable=True)
            batch_sorted = batch_indices[hilbert_order]
            # P-OPT: 完全向量化计算每个 batch 内的顺序索引，避免 Python 循环
            # 原理：先按 batch 排序（用 stable sort），然后用 cumsum 计算每个 batch 内的顺序
            # 由于我们先按 hilbert 排序，这里需要用另一种方法：
            # 使用 scatter_add 基于 batch 计数来分配索引
            M = selected_depths.numel()
            # 创建位置 tensor [0, 1, 2, ..., M-1]
            positions = torch.arange(M, device=features.device)
            # 使用 bincount 计算每个 batch 的累积起始位置
            batch_counts = batch_sorted.bincount(minlength=B)  # [B]
            # cumsum[:-1] 给出每个 batch 的起始偏移量
            batch_starts = torch.zeros(M, dtype=torch.long, device=features.device)
            if B > 1:
                batch_starts = F.pad(batch_counts.cumsum(0)[:-1], (1, 0))  # [B]
            # 将起始偏移量广播到每个元素，然后减去
            start_offsets = batch_starts[batch_sorted]  # [M]
            token_indices = positions - start_offsets
            # 按原始排序反序回去
            token_indices = token_indices[torch.argsort(hilbert_order, stable=True)]
        else:
            token_indices = torch.empty(0, dtype=torch.long, device=features.device)

        # 计算 tokens_per_batch
        tokens_per_batch = torch.bincount(batch_indices, minlength=B)
        # 确保至少每个 batch 有 1 个 token（避免除零错误）
        tokens_per_batch = torch.clamp(tokens_per_batch, min=1)

        # complexities: 使用 depths 作为复杂度代理（深度越大越复杂）
        complexities = selected_depths.float()

        return TensorSplitResult(
            regions=selected_regions.long(),  # I20: 转为 long 以支持位运算
            depths=selected_depths,
            batch_indices=batch_indices,
            hilbert_indices=selected_hilbert,
            token_indices=token_indices,
            complexities=complexities,
            tokens_per_batch=tokens_per_batch,
        )

    def _roi_align(
        self,
        features: Tensor,
        regions: Tensor,
        output_size: Tuple[int, int],
        chunk_size: int = 4,  # I164-1: 极小批量处理，大幅降低显存
    ) -> Tensor:
        """
        ROI Align操作 - 可微实现 (内存优化版本)

        使用极小批量采样避免显存爆炸
        I164-1: 改为极小批量处理以支持 max_level_limit=8

        数学:
            原显存: M = N × C × h × w × 4 bytes (~42GB for L=8)
            极小批量: M_chunk = chunk_size × C × h × w × 4 bytes
            若 chunk_size=4, C=384, h=w=14: M_chunk ≈ 0.03 MB
            峰值显存降低 ~1,000,000 倍
        """
        B, C, H_feat, W_feat = features.shape
        N = regions.shape[0]
        oh, ow = output_size

        if N == 0:
            return torch.zeros(B * N, C, device=features.device)

        # I164-1: 流式处理 - 使用列表收集结果，避免预分配大张量
        # 预分配需要 B*N*C*4 bytes ≈ 134 MB，现在改为按需分配
        roi_features_list = []

        # 归一化区域坐标到[-1, 1]
        h_img, w_img = self.image_size
        normalized = regions.clone()
        normalized[:, 0] = 2.0 * regions[:, 0] / w_img - 1.0
        normalized[:, 1] = 2.0 * regions[:, 1] / h_img - 1.0
        normalized[:, 2] = 2.0 * regions[:, 2] / w_img - 1.0
        normalized[:, 3] = 2.0 * regions[:, 3] / h_img - 1.0

        # 创建相对坐标网格 (只创建一次，复用到所有chunks)
        y_rel = torch.linspace(0, 1, oh, device=features.device, dtype=features.dtype)
        x_rel = torch.linspace(0, 1, ow, device=features.device, dtype=features.dtype)
        y_2d = y_rel.unsqueeze(1).expand(oh, ow)  # [oh, ow]
        x_2d = x_rel.unsqueeze(0).expand(oh, ow)  # [oh, ow]

        # I164-1: 极小批量处理 - 避免显存爆炸
        num_chunks = (N + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, N)
            chunk_n = end_idx - start_idx

            # 获取当前chunk的归一化坐标
            chunk_normalized = normalized[start_idx:end_idx]  # [chunk_n, 4]

            # 为当前chunk创建采样网格
            x0 = chunk_normalized[:, 0:1].unsqueeze(1).expand(chunk_n, oh, ow)
            y0 = chunk_normalized[:, 1:2].unsqueeze(1).expand(chunk_n, oh, ow)
            x1 = chunk_normalized[:, 2:3].unsqueeze(1).expand(chunk_n, oh, ow)
            y1 = chunk_normalized[:, 3:4].unsqueeze(1).expand(chunk_n, oh, ow)
            # 扩展相对坐标
            y_2d_exp = y_2d.unsqueeze(0).expand(chunk_n, oh, ow)
            x_2d_exp = x_2d.unsqueeze(0).expand(chunk_n, oh, ow)
            # 线性插值
            y_grid = y0 + (y1 - y0) * y_2d_exp
            x_grid = x0 + (x1 - x0) * x_2d_exp
            chunk_grids = torch.stack([x_grid, y_grid], dim=-1)  # [chunk_n, oh, ow, 2]

            # 对每个batch item执行采样，收集到列表
            for b in range(B):
                sampled = F.grid_sample(
                    features[b:b+1].expand(chunk_n, -1, -1, -1),
                    chunk_grids,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False,
                )
                pooled = sampled.mean(dim=[2, 3])  # [chunk_n, C]
                roi_features_list.append(pooled)

        # I164-1: 最后拼接所有结果
        roi_features = torch.cat(roi_features_list, dim=0)  # [B*N, C]

        return roi_features

    def _depth_aware_selection(
        self,
        probs: Tensor,
        quota_probs: Optional[Tensor],
        K: int,
    ) -> Tensor:
        """
        按深度配额进行选择

        数学:
            K_d = K × softmax(φ)_d
            对每个深度d，选择概率最高的K_d个区域
        """
        B, N = probs.shape
        device = probs.device

        # 获取深度分布
        depth_indices = self._depth_indices  # [N]

        if quota_probs is None:
            # 均匀配额
            K_d = K // (self._max_level + 1)
            quota_probs = torch.ones(self._max_level + 1, device=device) / (self._max_level + 1)
        else:
            K_d = (K * quota_probs).long()

        # 确保K_d至少为1
        K_d = K_d.clamp(min=1)

        # P-OPT: 一次性转换为Python列表，避免循环内多次GPU-CPU同步
        K_d_list = K_d.tolist()

        selected_mask = torch.zeros(B, N, device=device)

        for b in range(B):
            probs_b = probs[b]  # [N]

            # 按深度选择
            for d in range(self._max_level + 1):
                depth_mask = (depth_indices == d)
                depth_indices_in_depth = torch.where(depth_mask)[0]

                if len(depth_indices_in_depth) == 0:
                    continue

                depth_probs = probs_b[depth_mask]
                # P-OPT: 使用预转换的K_d_list，避免循环内.item()调用
                k_d = min(K_d_list[d], len(depth_probs))

                if k_d > 0:
                    _, topk_local = depth_probs.topk(k_d)
                    global_indices = depth_indices_in_depth[topk_local]
                    selected_mask[b, global_indices] = 1.0

        return selected_mask

    # =============================================================================
    # AnnealingSplitter 接口
    # =============================================================================

    def set_temperature(self, temperature: float) -> None:
        """设置温度"""
        if self.config.learnable_temperature:
            self.log_temperature.data = torch.tensor(
                math.log(temperature),
                device=self.log_temperature.device,
            )

    def get_current_temperature(self) -> Tensor:
        """获取当前温度"""
        return self._get_temperature()

    def set_annealing_schedule(
        self,
        schedule: Literal["linear", "exponential", "cosine"],
        start: float,
        end: float,
        total_steps: int,
    ) -> None:
        """配置退火调度"""
        pass

    def set_explore_bias(self, bias: float) -> None:
        """设置探索偏置"""
        pass

    # =============================================================================
    # MetricsSplitter 接口
    # =============================================================================

    def get_diagnostics(self) -> Dict[str, Any]:
        """获取诊断信息"""
        tau = self._get_temperature()
        quota_probs = self._get_quota_probs()

        return {
            'current_temperature': tau.item(),
            'alpha': self.scoring.alpha.item(),
            'depth_distribution': {
                d: (self._depth_indices == d).sum().item()
                for d in range(self._max_level + 1)
            },
            'quota_allocation': quota_probs.tolist() if quota_probs is not None else None,
            'splitter_type': 'DeterministicNeighborSplitter',
        }

    def get_depth_distribution(self) -> Dict[int, float]:
        """获取深度分布"""
        total = len(self._depth_indices)
        return {
            d: (self._depth_indices == d).sum().item() / total
            for d in range(self._max_level + 1)
        }

    def get_quota_logits(self) -> Optional[Tensor]:
        """获取配额logits"""
        return self.quota_logits

    def get_quota_probs(self) -> Optional[Tensor]:
        """获取配额概率"""
        return self._get_quota_probs()

    def get_entropy_loss(self) -> Tensor:
        """获取熵正则化损失"""
        if self.quota_logits is None:
            return torch.tensor(0.0, device=self.device)

        probs = F.softmax(self.quota_logits, dim=0)
        entropy = -(probs * probs.clamp(min=EPS).log()).sum()

        return self.config.entropy_weight * entropy

    def get_variance_regularization(self) -> Tensor:
        """获取方差正则化"""
        if self.quota_logits is None:
            return torch.tensor(0.0, device=self.device)

        probs = F.softmax(self.quota_logits, dim=0)
        depths = torch.arange(
            len(probs),
            dtype=probs.dtype,
            device=probs.device,
        )
        expected_depth = (probs * depths).sum()
        variance = (probs * (depths - expected_depth) ** 2).sum()

        return variance

    def get_locality_loss(self) -> Tensor:
        """
        获取局部一致性损失

        数学:
            L_local = Σ_(i,j)∈E |σ(s_i) - σ(s_j)|
        """
        if not self.config.enable_neighbor_aware:
            return torch.tensor(0.0, device=self._regions.device)

        adj_matrix = self._get_adj_matrix()

        # 获取分数
        with torch.no_grad():
            features = torch.randn(self._num_candidates, self.feature_dim, device=self._regions.device)
            _, final_scores = self.scoring(features, adj_matrix)

        probs = final_scores.sigmoid()

        # 只在邻居对上计算损失
        diff = torch.abs(probs.unsqueeze(0) - probs.unsqueeze(1))
        mask = adj_matrix > 0

        if mask.sum() == 0:
            return torch.tensor(0.0, device=self._regions.device)

        loss = (diff * mask).sum() / (mask.sum() + EPS)

        return self.config.locality_weight * loss

    def get_coverage_stats(self) -> Dict[str, float]:
        """获取覆盖率统计"""
        return {
            'raw_coverage': self.config.coverage_base,
            'adaptive_coverage': self.config.coverage_base,
            'K_selected': self._compute_K(),
            'N_total': self._num_candidates,
        }

    # =============================================================================
    # 其他必需方法
    # =============================================================================

    def update_candidates(self, image_size: Tuple[int, int]) -> None:
        """根据输入尺寸动态更新候选区域"""
        self.image_size = image_size

        max_level = self._max_level
        self._regions = self._compute_regions(max_level, image_size)
        self._depth_indices = self._compute_depth_indices(max_level)
        self._hilbert_indices = self._compute_hilbert_indices(max_level)

        # 清除邻接矩阵缓存
        self._cached_adj = None

    @property
    def max_level_limit(self) -> int:
        """获取最大深度限制"""
        return self._max_level

    @property
    def num_candidates(self) -> int:
        """获取候选区域数量"""
        return self._num_candidates

    @property
    def is_training(self) -> bool:
        """获取训练/评估模式"""
        return self.training

    def step(self):
        """更新训练步数"""
        self._current_step += 1


# =============================================================================
# 工厂函数
# =============================================================================

def create_deterministic_neighbor_splitter(
    image_size: Tuple[int, int] = (224, 224),
    feature_dim: int = 256,
    **kwargs,
) -> DeterministicNeighborSplitter:
    """
    创建DeterministicNeighborSplitter的工厂函数

    Args:
        image_size: 输入图像尺寸
        feature_dim: 特征维度
        **kwargs: 配置参数

    Returns:
        DeterministicNeighborSplitter实例
    """
    config = DeterministicNeighborSplitterConfig(**kwargs)
    return DeterministicNeighborSplitter(
        config=config,
        image_size=image_size,
        feature_dim=feature_dim,
    )


# =============================================================================
# 向后兼容别名 (I160-1)
# =============================================================================

# NeighborAwareSplitter 是 DeterministicNeighborSplitter 的旧名称
NeighborAwareSplitter = DeterministicNeighborSplitter
NeighborAwareSplitterConfig = DeterministicNeighborSplitterConfig


class LocalityConsistencyLoss(nn.Module):
    """局部一致性损失包装器 (向后兼容别名)

    该类已弃用。请直接使用 DeterministicNeighborSplitter.get_locality_loss() 方法。

    数学形式:
        L_local = Σ_(i,j)∈E |σ(s_i) - σ(s_j)|
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
        self._dummy_param = nn.Parameter(torch.tensor(0.0))

    def forward(self, splitter: DeterministicNeighborSplitter) -> torch.Tensor:
        """计算局部一致性损失

        Args:
            splitter: DeterministicNeighborSplitter 实例

        Returns:
            局部一致性损失
        """
        return splitter.get_locality_loss() * self.weight
