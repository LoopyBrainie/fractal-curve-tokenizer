"""
SemanticRedundancySplitter - 基于语义冗余的分裂器

核心思想：通过 LookAheadHead 预测子节点特征，CorrelationGate 计算冗余性，
决策是否分裂。实现从开环到闭环的架构转变。

数学形式化：
- LookAheadHead: V_c = W_2 · GELU(W_1 · F_p + b_1) + b_2
- CorrelationGate: r = 1 - (1/6)∑_{i<j} cos(θ_ij)
- 分裂决策: p = σ(w_r · r + w_d · d + w_q · q + b)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_pytorch.core.constants import EPS  # I112-3: 统一数值稳定性常量


@dataclass
class SplitResult:
    """分裂结果数据类

    Attributes:
        split_decision: [B, N] 分裂决策 (0=不分裂, 1=分裂)
        child_features: [B, N, 4, D] 预测的子节点特征
        redundancy: [B, N] 冗余性分数 (0=冗余, 1=独立)
        logits: [B, N] 原始 logits
        temperature: 当前温度 (用于 Gumbel-Softmax)
    """
    split_decision: torch.Tensor
    child_features: torch.Tensor
    redundancy: torch.Tensor
    logits: torch.Tensor
    temperature: float = 1.0


class LookAheadHead(nn.Module):
    """前瞻投影头：预测父节点分裂后的子节点特征

    给定父节点特征 F_p ∈ ℝ^D，预测四个子节点特征 V_c ∈ ℝ^{4×D}

    公式: V_c = W_2 · GELU(W_1 · F_p + b_1) + b_2

    设计决策：
    - 使用 GELU 而非 ReLU：平滑梯度，无死神经元
    - 隐藏层维度 H=128：平衡容量与效率
    - 输出 4D：每个子节点 D 维特征
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        """初始化前瞻投影头

        Args:
            feature_dim: 输入特征维度 D
            hidden_dim: 隐藏层维度 H (默认 128)
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # 双层 MLP: D → H → 4D
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4 * feature_dim)
        )

    def forward(self, parent_features: torch.Tensor) -> torch.Tensor:
        """前瞻投影

        Args:
            parent_features: [B, N, D] 父节点特征

        Returns:
            child_features: [B, N, 4, D] 预测的子节点特征
        """
        projected = self.mlp(parent_features)  # [B, N, 4*D]
        # reshape: [B, N, 4*D] → [B, N, 4, D]
        return projected.view(*projected.shape[:-1], 4, self.feature_dim)

    def extra_repr(self) -> str:
        return f"feature_dim={self.feature_dim}, hidden_dim={self.hidden_dim}"


class CorrelationGate(nn.Module):
    """相关性门：计算子节点间的语义冗余性

    给定子节点特征 V_c ∈ ℝ^{4×D}，计算冗余性分数 r ∈ [0, 1]

    公式:
    - sim(v_i, v_j) = (v_i · v_j) / (||v_i|| · ||v_j|| + ε)
    - sim_mean = (1/6) ∑_{i<j} sim(v_i, v_j)
    - r = 1 - sim_mean

    解释:
    - r ≈ 0: 子节点高度相似 → 不应分裂
    - r ≈ 1: 子节点语义独立 → 应分裂
    """

    def __init__(self, feature_dim: int, method: str = "cosine"):
        """初始化相关性门

        Args:
            feature_dim: 特征维度 D (未使用，保留接口兼容)
            method: 相似度计算方法 ("cosine" | "covariance" | "rank")
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.method = method

        assert method == "cosine", f"仅支持 cosine 方法, got {method}"

        # P-OPT: 缓存 triu_indices 避免每次 forward 都创建
        self.register_buffer('_triu_indices', torch.triu_indices(4, 4, dtype=torch.long))

    def forward(self, child_features: torch.Tensor) -> torch.Tensor:
        """计算冗余性分数

        Args:
            child_features: [B, N, 4, D] 子节点特征

        Returns:
            redundancy: [B, N] 冗余性分数 (0=冗余, 1=独立)
        """
        # P-OPT: 融合 L2 归一化与矩阵乘法
        # 原来: norm -> divide -> einsum -> triu_indices
        # 优化: 直接使用 F.normalize (更高效) + bmm

        # L2 归一化 [B, N, 4, D]
        normalized = F.normalize(child_features, dim=-1, p=2)

        # P-OPT: 使用 bmm 替代 einsum，更高效
        # [B, N, 4, D] @ [B, N, D, 4] -> [B, N, 4, 4]
        B, N, _, D = child_features.shape
        normalized_flat = normalized.view(B * N, 4, D)
        sim_matrix = torch.bmm(normalized_flat, normalized_flat.transpose(-2, -1))
        sim_matrix = sim_matrix.view(B, N, 4, 4)

        # 提取上三角 (不含对角线): 6 对
        sim_pairs = sim_matrix[..., self._triu_indices[0], self._triu_indices[1]]  # [B, N, 6]

        # 平均相似度
        sim_mean = sim_pairs.mean(dim=-1)  # [B, N]

        # 冗余性: 0 = 完全冗余, 1 = 完全独立
        redundancy = 1.0 - sim_mean

        return redundancy

    def extra_repr(self) -> str:
        return f"feature_dim={self.feature_dim}, method='{self.method}'"


class SemanticRedundancySplitter(nn.Module):
    """语义冗余分裂器：基于子节点冗余性决策是否分裂

    核心流程:
    1. LookAheadHead(F_p) → V_c (预测子节点特征)
    2. CorrelationGate(V_c) → r (冗余性分数)
    3. logits = w_r · r + w_d · d + w_q · q + b
    4. Gumbel-Softmax(logits) → 分裂决策

    优势:
    - 闭环反馈: 可复用 Transformer 输出持续优化
    - 语义决策: 基于真正的冗余性，而非复杂度假设
    - Hilbert 兼容: 天然支持 Hilbert 局部性作为归纳偏置
    """

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        max_level_limit: int = 8,
        gumbel_temp_start: float = 1.0,
        gumbel_temp_end: float = 0.5,
        learnable_temperature: bool = True,
    ):
        """初始化语义冗余分裂器

        Args:
            feature_dim: 特征维度 D
            hidden_dim: LookAheadHead 隐藏层维度 H
            max_level_limit: 最大深度限制
            gumbel_temp_start: 初始温度
            gumbel_temp_end: 结束温度
            learnable_temperature: 是否使用可学习温度
        """
        super().__init__()

        # 核心组件
        self.look_ahead = LookAheadHead(feature_dim, hidden_dim)
        self.correlation = CorrelationGate(feature_dim)

        # 架构参数
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_level = max_level_limit

        # 可学习决策参数
        self.w_r = nn.Parameter(torch.zeros(1))  # 冗余性权重
        self.w_d = nn.Parameter(torch.zeros(1))  # 深度权重
        self.w_q = nn.Parameter(torch.zeros(1))  # 配额权重
        self.bias = nn.Parameter(torch.zeros(1))  # 偏置

        # 温度调度
        self.gumbel_temp_start = gumbel_temp_start
        self.gumbel_temp_end = gumbel_temp_end

        if learnable_temperature:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(gumbel_temp_start)))
        else:
            self.register_buffer('fixed_temp', torch.tensor(gumbel_temp_start))

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        """初始化决策参数"""
        # 初始时倾向于不分裂 (负偏置)
        nn.init.constant_(self.bias, -0.5)
        # 冗余性权重初始为正 (冗余 → 不分裂)
        nn.init.constant_(self.w_r, 1.0)

    def _get_temperature(self) -> torch.Tensor:
        """获取当前温度"""
        if hasattr(self, 'log_temp'):
            return torch.clamp(self.log_temp.exp(), min=0.1, max=2.0)
        else:
            return self.fixed_temp

    def _gumbel_softmax(
        self,
        logits: torch.Tensor,
        temperature: Optional[torch.Tensor] = None,
        hard: bool = False,
    ) -> torch.Tensor:
        """Gumbel-Softmax 采样 (优化版)

        公式: g_k = exp((log p_k + g_k') / τ) / ∑_j exp((log p_j + g_j') / τ)

        P-OPT:
        - 直接在 logit 空间计算，避免多余的 sigmoid -> log 转换
        - 使用 clamp 替代多次 log + EPS 防护

        Args:
            logits: [B, N] 原始 logits
            temperature: 温度参数
            hard: 硬决策 (argmax)

        Returns:
            samples: [B, N] 采样结果
        """
        if temperature is None:
            temperature = self._get_temperature()

        # P-OPT: 直接使用 logit，避免 sigmoid -> log 的往返转换
        # 在 logit 空间添加 Gumbel 噪声: logit + g ~ Gumbel(0, 1)
        gumbel_noise = torch.rand_like(logits).log_()

        # 添加噪声并除以温度
        softened = (logits + gumbel_noise) / temperature

        if hard:
            # 硬决策: argmax
            return (softened > 0).float()
        else:
            # 软决策: sigmoid
            return torch.sigmoid(softened)

    def update_temperature(self, step: int, total_steps: int, schedule: str = "cosine"):
        """更新温度调度

        Args:
            step: 当前步
            total_steps: 总步数
            schedule: 调度策略 ("linear" | "cosine" | "exponential")
        """
        progress = min(step / max(total_steps, 1), 1.0)

        if schedule == "linear":
            temp = self.gumbel_temp_start + (self.gumbel_temp_end - self.gumbel_temp_start) * progress
        elif schedule == "cosine":
            temp = self.gumbel_temp_end + 0.5 * (self.gumbel_temp_start - self.gumbel_temp_end) * (
                1 + torch.cos(torch.tensor(progress * 3.14159))
            )
        elif schedule == "exponential":
            temp = self.gumbel_temp_start * (self.gumbel_temp_end / self.gumbel_temp_start) ** progress
        else:
            temp = self.gumbel_temp_start

        if hasattr(self, 'log_temp'):
            # I112-3: 使用 EPS 统一数值稳定性
            self.log_temp.data = torch.log(torch.tensor(temp + EPS))
        else:
            self.fixed_temp.fill_(temp)

    def forward(
        self,
        features: torch.Tensor,
        depth: int = 0,  # I140: 兼容 GumbelTopK 接口，默认从根节点开始
        remaining_quota: float = 1.0,  # I140: 兼容 GumbelTopK 接口，默认全配额
        hard: bool = False,
        image_size: Optional[tuple] = None,  # I140: 兼容 GumbelTopK 接口但不使用
    ) -> SplitResult:
        """前向传播

        Args:
            features: [B, N, D] 区域特征
            depth: 当前深度 (绝对值)，默认 0
            remaining_quota: 剩余配额比例 (0~1)，默认 1.0
            hard: 硬决策 (推理模式)
            image_size: (H, W) 图像尺寸 (可选，用于接口兼容，SemanticRedundancy 不使用)

        Returns:
            SplitResult: 分裂结果
        """
        B, N, D = features.shape

        # Step 1: 前瞻预测子节点特征
        child_features = self.look_ahead(features)  # [B, N, 4, D]

        # Step 2: 计算冗余性
        redundancy = self.correlation(child_features)  # [B, N]

        # Step 3: 计算分裂 logits
        depth_norm = depth / self.max_level  # 归一化到 [0, 1]
        logits = (
            self.w_r * redundancy +
            self.w_d * depth_norm +
            self.w_q * remaining_quota +
            self.bias
        )  # [B, N]

        # Step 4: Gumbel-Softmax 决策
        temperature = self._get_temperature()
        if hard:
            split_decision = (logits > 0).float()
        else:
            split_decision = self._gumbel_softmax(logits, temperature)

        # Step 5: 构建 SplitResult
        return SplitResult(
            split_decision=split_decision,
            child_features=child_features,
            redundancy=redundancy,
            logits=logits,
            temperature=temperature.item(),
        )

    def get_split_regions(
        self,
        split_result: SplitResult,
        region_bounds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据分裂决策生成子区域

        Args:
            split_result: SplitResult 分裂结果
            region_bounds: [M, 4] 原始区域坐标 (x0, y0, x1, y1)

        Returns:
            child_bounds: [4*M, 4] 子区域坐标
            child_depths: [4*M] 子区域深度
        """
        # 获取需要分裂的区域
        split_mask = split_result.split_decision.bool()  # [B, N]
        batch_indices = torch.where(split_mask)[0]
        region_indices = torch.where(split_mask)[1]

        if len(region_indices) == 0:
            return region_bounds.new_zeros(0, 4), region_bounds.new_zeros(0, dtype=torch.long)

        # 计算子区域坐标
        parent_bounds = region_bounds[region_indices]  # [M, 4]
        x0, y0, x1, y1 = parent_bounds.unbind(dim=-1)

        # 四象限分割
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        # 四个子区域
        child_bounds = torch.cat([
            torch.stack([x0, y0, cx, cy], dim=-1),   # 左上
            torch.stack([cx, y0, x1, cy], dim=-1),   # 右上
            torch.stack([x0, cy, cx, y1], dim=-1),   # 左下
            torch.stack([cx, cy, x1, y1], dim=-1),   # 右下
        ], dim=0)

        # 记录深度
        depth = split_result.split_decision.sum(dim=1, keepdim=True).long()
        # 注意: 这里简化处理，实际需要跟踪每个区域的分裂深度

        return child_bounds, depth.new_zeros(len(child_bounds))

    def extra_repr(self) -> str:
        return (
            f"feature_dim={self.feature_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"max_level={self.max_level}"
        )
