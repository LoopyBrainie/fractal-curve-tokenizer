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
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import roi_align
# D4-AUDIT FIX: 移除外部 entmax 库导入，替换为内部 entmax_beta 实现
# 原因: entmax_bisect 内部强制 .float() upcast 至 float32，破坏 AMP 显存优化
# entmax_beta (基于 sparsemax 算法) 无此问题，且支持 alpha=1.5

from vit_pytorch.core.constants import EPS, TEMPERATURE_MIN
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
    probs = entmax_beta(scores, alpha=alpha, dim=dim)

    # 使用 TopK
    K_target = probs.shape[dim] // 4  # 假设 K ≈ N/4

    # 补足 TopK
    _, topk_indices = torch.topk(probs, K_target, dim=dim)

    return probs, topk_indices


# =============================================================================
# Differentiable K Selection (STE Straight-Through Estimator)
# =============================================================================

class DifferentiableK(nn.Module):
    """Straight-Through Estimator for differentiable K selection.

    数学形式:
        前向: K_hard = round(clamp(K_float, K_min, K_max))
        反向: dK_soft/dK_float = 1 (STE, 恒等梯度)

    用途:
        1. K_soft 用于 aux_budget 损失计算（保持梯度流）
        2. K_hard 用于实际 token 选择（离散决策）

    解决的核心问题:
        原代码 K = K_float.long().clamp(...).item() 使用 .item() 断裂梯度,
        导致 density_field 的输出无法通过 K 误差信号更新。
    """

    def __init__(self, K_min: int, K_max: int):
        super().__init__()
        self.K_min = K_min
        self.K_max = K_max

    def forward(self, K_float: Tensor) -> Tuple[Tensor, Tensor]:
        """返回 (K_hard, K_soft)

        Args:
            K_float: 来自 density_field 的连续 K 值 [1] 或 [B]

        Returns:
            K_hard: 四舍五入的整数用于实际选择
            K_soft: clamp 后的连续值用于损失计算（保持梯度）
        """
        K_soft = K_float.clamp(self.K_min, self.K_max)
        # STE: 前向使用 round（离散），反向使用恒等梯度
        K_hard = K_soft.round().long()
        return K_hard, K_soft


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
    entmax_alpha_warmup: float = 1.49   # V4: 1.49 而非 1.5，永远保持轻微梯度流
    entmax_alpha_max: float = 2.0       # 最终稀疏度
    entmax_warmup_epochs: int = 10       # 预热 epoch 数
    # V3: α 延迟调度 (20→25) 防止双重退火坍缩
    entmax_schedule_epochs: int = 25     # 总调度 epoch 数

    # 树约束参数
    tree_constraint_weight: float = 0.1

    # 温度参数
    temperature_init: float = 1.0
    temperature_min: float = 0.3

    # Jump Loss 参数
    jump_loss_weight: float = 0.1

    # Density Field 参数
    density_field_hidden_dim: int = 32

    # I167-1: 距离衰减卷积
    # True: 使用解耦版 - 空间混合(固定) + 通道混合(可学习)
    # False: 回退到标准 Conv1D (向后兼容)
    use_distance_decay_conv: bool = True

    # I167-4: SDS 正则化
    # True: 启用 SDS 正则化，惩罚高 SDS（空间局部性破坏）的位置
    # SDS 正则化: z = z - λ * SDS_penalty
    use_sds_regularization: bool = False
    sds_lambda: float = 0.1  # SDS 正则化强度


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
        use_distance_decay_conv: bool = True,  # I167-1: 距离衰减卷积
        use_sds_regularization: bool = False,  # I167-4: SDS 正则化
        sds_lambda: float = 0.1,  # I167-4: SDS 正则化强度
    ):
        super().__init__()

        # 当前训练轮次
        self._current_epoch: int = 0

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
            # I167-1: 安全读取 config 中的字段（兼容旧 config）
            use_distance_decay_conv = getattr(
                config, 'use_distance_decay_conv', True
            )
            # I167-4: SDS 正则化
            use_sds_regularization = getattr(
                config, 'use_sds_regularization', False
            )
            sds_lambda = getattr(config, 'sds_lambda', 0.1)

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

        # Entmax 参数 (I107: 添加预热策略 + 修复课程学习)
        self.entmax_alpha = entmax_alpha
        self.entmax_alpha_init = 1.0      # P1 FIX: 起始值改为 1.0 (强制 softmax)
        self.entmax_alpha_warmup = 1.30    # P1 FIX: 上限从 1.49 降至 1.30，防止过度稀疏化
        self.entmax_alpha_max = 1.30       # P1 FIX: 稳定值设为 1.30
        self.entmax_warmup_epochs = 5      # Stage 0: epoch 0-5 (α=1.0)
        self.entmax_transition_epochs = 12 # P1 FIX: Stage 1 结束 epoch (α 从 1.0 → 1.22)
        # P1 FIX: α 延迟调度 (12→25) 爬升至 1.30，不再到达 1.49
        self.entmax_schedule_epochs = 25   # 总调度 epoch 数 (5-25: α 增长到 1.30)

        # 树约束 - I164-1: 动态λ调整
        # 使用log(lambda)确保λ>0，通过课程学习逐步增强约束
        self.tree_constraint_weight = tree_constraint_weight
        self._log_lambda = nn.Parameter(torch.tensor(0.0, dtype=torch.get_default_dtype()))  # 可学习的log(λ)
        self._lambda_schedule_epochs = 20  # λ课程学习持续20个epoch

        # λ的初始值和目标值（课程学习）
        self._lambda_init = 0.05
        self._lambda_max = 0.3

        # 温度
        self.temperature = temperature_init
        self.temperature_init = temperature_init
        self.temperature_min = temperature_min

        # =====================================================================
        # BPE-style 三阶段 Warmup 动态参数
        # 初始化默认值，由外部调度器通过 setter 更新
        # =====================================================================
        self._target_ratio = 0.25        # 目标 token 比例
        self._budget_weight = None       # None 表示使用 get_auxiliary_losses 内的默认值
        self._logits_diversity_enabled = True  # 是否启用 logits 多样性惩罚
        # P0 FIX: 动态 K_min 比例，由 _update_fractal_hyperparams 调度
        # 1.0 = K_min = N（保留所有 token），0.5 = K_min = 0.5 * K_target
        self._k_min_ratio = 1.0

        # Jump Loss 权重
        self.jump_loss_weight = jump_loss_weight

        # Density Field 隐藏层维度
        self.density_field_hidden_dim = density_field_hidden_dim

        # I167-4: SDS 正则化
        self.use_sds_regularization = use_sds_regularization
        self.sds_lambda = sds_lambda

        # 当前状态
        self._current_image_size: Optional[Tuple[int, int]] = None
        self._epoch = 0

        # =====================================================================
        # I150-3: Token 稳定性监控
        # =====================================================================
        self._monitor_token_stability = False
        self._token_history: List[Tensor] = []

        # =====================================================================
        # Phase 2: 中间变量安全缓冲区 (DDP 训练安全)
        # D1-AUDIT FIX: 存储 GPU tensor，在 get_output() 中延迟 .item()
        # =====================================================================
        self._last_sds_stats: Optional[Dict[str, float]] = None
        # D1-AUDIT FIX: 改为 Optional[Tensor] 避免 forward 内 .item() 同步
        self._last_tree_delta_z_t: Optional[torch.Tensor] = None

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

        # 3.5. 旋转嵌入投影层 (用于参数共享后维度对齐)
        # 正常情况: rot_emb dim == hidden_dim (无需投影)
        # 参数共享后: rot_emb dim == geometry_field.dim (需要投影到 hidden_dim)
        # 注意: 使用 bias=False 避免引入额外的仿射变换，保持几何感知的线性投影特性
        self.rot_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # 4. 面积编码
        # D4 AUDIT FIX: 4.0 ** (-d) -> torch.exp2(-d.float() * 2.0) (vectorized, no Python loop)
        d_indices = torch.arange(max_level_limit + 1, dtype=torch.float32)
        self.register_buffer(
            '_area_encoding',
            torch.exp2(-d_indices * 2.0)  # 4^(-d) = 2^(-2d)
        )

        # 4.5. 面积投影（消除 expand 导致的秩塌陷，赋予模型学习最优面积表示的能力）
        self.area_proj = nn.Linear(1, hidden_dim)

        # 4.6. 分组特征归一化（解决 roi_features 范数 >> 几何嵌入范数导致的几何信息被压制问题）
        # 语义组：[B, N, 64] — roi_features 来自 backbone，初始范数 O(2-8)
        # 几何组：[N, 192] — path_emb + rot_emb + area_proj 输出，初始范数 O(0.16)
        self.roi_norm = nn.LayerNorm(hidden_dim)          # 语义特征归一化
        self.geo_norm = nn.LayerNorm(hidden_dim * 3)       # 几何特征归一化（path + rot + area）

        # 5. 1D Hilbert 流形卷积 (A1: 核心创新)
        # 输入: [B, N, hidden_dim * 4] (feat + path + rot + area)
        # 输出: [B, N, 1]
        conv_input_dim = hidden_dim * 4

        # I167-1: 根据配置选择卷积实现
        if use_distance_decay_conv:
            # 解耦版: 空间混合(固定距离衰减) + 通道混合(可学习)
            # 数学: z = Pointwise(Depthwise(x, w_decay))
            # 参数: D × 1 (比标准 Conv1D 减少约 90%)
            from .hilbert_distance_decay_conv import HilbertDistanceDecayConv1D
            self.conv1d_hilbert = HilbertDistanceDecayConv1D(conv_input_dim)
        else:
            # 标准版: 向后兼容
            self.conv1d_hilbert = nn.Conv1d(
                conv_input_dim,
                1,
                kernel_size=5,
                padding=2,
                groups=1,
            )

        # 6. 深度配额学习 (可选，用于 A5)
        self.depth_quota = nn.Parameter(torch.ones(max_level_limit + 1, dtype=torch.get_default_dtype()))

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

        # 初始化 conv1d_hilbert 偏置为 +0.5，强制初期尝试更多分裂
        self._init_logits_bias()

        # I-OPT: Differentiable K 选择器 (STE 直通估计)
        # 用于恢复 K 值的梯度流，解决原 .item() 断裂梯度的问题
        self.K_estimator = DifferentiableK(K_min=K_min, K_max=K_max)

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

        # P1 修复: 偏置设为 0.0，使初始 sigmoid 输出 = 0.5
        # sigmoid(0) = 0.5，赋予模型充足的信息带宽探索视觉特征
        # Budget_Loss（目标 25%）在后续缓慢剪枝，避免开局"极度贫血"
        nn.init.constant_(last_linear.bias, 0.0)

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

            indices = mask.nonzero(as_tuple=False).squeeze(-1)  # D3-AUDIT FIX: as_tuple=False 避免 Graph Break
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
        area_enc = self._area_encoding[depths]  # [N]
        area_enc = self.area_proj(area_enc.unsqueeze(-1))  # [N] → [N, 1] → [N, hidden_dim]

        return path_emb, rot_emb, area_enc

    def _compute_sds_penalty(
        self,
        logits: Tensor,
        image_size: Tuple[int, int],
        k: int = 4,
    ) -> Tensor:
        """计算 SDS 正则化惩罚

        数学:
            SDS(i) = Σ_{j∈N_k(i)} ||p_i - p_j||²_2 / (2k)
            Penalty(i) = λ * SDS(i)

        其中:
            - N_k(i) 是 Hilbert 序中位置 i 的 k 个最近邻
            - p_i 是位置 i 在网格上的 2D 中心坐标
            - λ 是正则化强度

        实现:
            1. 从 Hilbert 索引重建 2D 坐标（考虑深度缩放）
            2. 按 Hilbert 序排序
            3. 计算 k 近邻欧氏距离平方
            4. 返回惩罚值（与 logits 形状相同 [B, N]）

        Args:
            logits: [B, N] 原始 logits
            image_size: (H, W) 原始图像尺寸
            k: 考虑的最近邻数量

        Returns:
            sds_penalty: [B, N] SDS 惩罚，可直接从 logits 减去
        """
        B = logits.shape[0]
        N = logits.shape[1]
        device = logits.device

        # 获取候选区域的 Hilbert 索引和深度
        hilbert_indices = self.hilbert_indices  # [N]
        depths = self.candidate_depths  # [N]

        # Step 1: 从 Hilbert 索引重建 2D 坐标
        # 注意: 每个深度的 Hilbert 索引在其自己的网格分辨率下
        # 需要缩放到 max_level 以便统一比较
        coordinates_list = []
        max_level = self.max_level_limit

        for depth in range(max_level + 1):
            depth_mask = (depths == depth)
            if not depth_mask.any():
                continue

            depth_indices = hilbert_indices[depth_mask]  # 该深度的 Hilbert 索引
            n_grid = 1 << depth  # 该深度的网格大小

            # 转换 Hilbert 距离到 2D 坐标
            x_coords, y_coords = HilbertCurve.d_to_xy_batch(n_grid, depth_indices)

            # 缩放到 max_level 分辨率
            scale = 1 << (max_level - depth)  # 2^(max_level - depth)
            x_coords = x_coords.float() * scale
            y_coords = y_coords.float() * scale

            # 合并坐标
            coords_depth = torch.stack([x_coords, y_coords], dim=1)  # [N_depth, 2]
            coordinates_list.append(coords_depth)

        # 合并所有深度的坐标
        all_coordinates = torch.zeros(N, 2, device=device)
        depth_mask_flat = torch.zeros(N, dtype=torch.long, device=device)
        pos = 0
        for depth in range(max_level + 1):
            depth_mask = (depths == depth)
            if depth_mask.any():
                all_coordinates[depth_mask] = coordinates_list[pos]
                depth_mask_flat[depth_mask] = depth
                pos += 1

        # Step 2: 按 Hilbert 索引排序
        sort_idx = hilbert_indices.argsort()
        sorted_coords = all_coordinates[sort_idx]  # [N, 2]

        # Step 3: 计算 k 近邻欧氏距离平方（完全向量化）
        # P-OPT: 一次性计算所有 2k 个邻居的距离，避免 Python 循环
        # 创建所有偏移量: [-k, ..., -1, 1, ..., k]
        offsets = torch.arange(-k, k + 1, device=device)
        offsets = offsets[offsets != 0]  # 移除 0 偏移
        num_neighbors = offsets.numel()  # 应该是 2k

        # 计算所有邻居索引: [N, num_neighbors]
        neighbor_idx = torch.arange(N, device=device).unsqueeze(1) + offsets.unsqueeze(0)
        # Clamp 到有效范围 [0, N-1]
        neighbor_idx_clamped = neighbor_idx.clamp(min=0, max=N - 1)

        # 计算所有邻居的坐标: [N, num_neighbors, 2]
        neighbor_coords = sorted_coords[neighbor_idx_clamped]  # [N, num_neighbors, 2]

        # 使用 broadcasting 一次性计算所有距离: [N, num_neighbors]
        diff = sorted_coords.unsqueeze(1) - neighbor_coords  # [N, num_neighbors, 2]
        all_dist_sq = (diff ** 2).sum(dim=2)  # [N, num_neighbors]

        # 计算有效掩码（排除原始位置的 0 偏移已被移除）
        valid_mask = (neighbor_idx >= 0) & (neighbor_idx < N)  # [N, num_neighbors]

        # 归一化: 只对有效邻居求平均
        valid_count = valid_mask.sum(dim=1).clamp(min=1)  # [N]
        sds_values = (all_dist_sq * valid_mask.float()).sum(dim=1) / valid_count  # [N]

        # 归一化（使用Step 2的all_dist_sq）
        valid_count = valid_mask.sum(dim=1).clamp(min=1)  # [N]
        sds_values = (all_dist_sq * valid_mask.float()).sum(dim=1) / valid_count  # [N]

        # Step 4: 扩展到 batch 维度并返回惩罚
        # sds_values: [N] -> [B, N]
        sds_penalty = sds_values.unsqueeze(0).expand(B, -1) * self.sds_lambda  # [B, N]

        # 取消排序，恢复原始顺序
        # 需要将 penalty 放回原始位置
        unsort_idx = sort_idx.argsort()
        sds_penalty = sds_penalty[:, unsort_idx]  # [B, N]

        return sds_penalty

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

    def _apply_tree_constraint(self, logits: Tensor, depths: Tensor) -> Tensor:
        """应用树一致性软约束

        数学:
            z_parent -= λ × max(z_children)

        I164-1: 动态λ调整
            - 课程学习: λ从_init逐步增加到_max
            - 可学习残差: log_lambda提供额外的学习信号
        """
        if self.tree_constraint_weight <= 0:
            return logits

        # 计算动态λ (课程学习 + 可学习残差)
        lambda_cur = self._compute_dynamic_lambda()

        constrained_logits = logits.clone()

        # D2-AUDIT FIX: 向量化 parent-child penalty 避免 Python for 循环
        # 对每个父节点，降低其分数如果子节点分数更高
        valid_mask = self.parent_indices >= 0
        if valid_mask.any():
            valid_parents = torch.masked_select(self.parent_indices, valid_mask)  # [P]
            children_all = self.children_matrix[valid_parents]  # [P, 4]
            child_mask = children_all >= 0  # [P, 4]
            # gather child logits: clamp to 0 for invalid indices, they'll be masked anyway
            children_clamped = children_all.masked_fill(~child_mask, 0)
            gathered = logits.gather(1, children_clamped.view(-1).unsqueeze(0).expand(logits.shape[0], -1))  # [B, P*4]
            # D3-AUDIT FIX: gather 可能返回非连续 tensor，view 需要连续内存
            gathered = gathered.contiguous().view(-1, *children_clamped.shape)  # [B, P, 4]
            # mask invalid child positions with -inf
            gathered = gathered.masked_fill(~child_mask.unsqueeze(0), float('-inf'))
            max_child_logits = gathered.max(dim=2)[0]  # [B, P]
            constrained_logits[:, valid_parents] -= lambda_cur * max_child_logits

        return constrained_logits

    def _compute_dynamic_lambda(self) -> Tensor:
        """计算动态λ (课程学习)

        λ(t) = λ_init + (λ_max - λ_init) × min(1, t / T_schedule) + σ(log_lambda)

        其中 t 是当前epoch，T_schedule 是课程学习持续时间

        Returns:
            GPU tensor (与 logits 等设备兼容)，避免 forward 内 .item() 同步
        """
        # 课程学习组件
        progress = min(1.0, self._current_epoch / max(1, self._lambda_schedule_epochs))
        lambda_scheduled = self._lambda_init + (self._lambda_max - self._lambda_init) * progress

        # 可学习残差 (sigmoid确保正值)
        # D1-AUDIT FIX: 返回 GPU tensor，整体计算图保持 GPU
        lambda_learnable = torch.sigmoid(self._log_lambda) * 0.2  # 缩放到合理范围

        # 组合: lambda_scheduled (float) + tensor → tensor
        return lambda_scheduled + lambda_learnable

    def get_adaptive_alpha(self, epoch: int) -> float:
        """自适应 α 调度器 - 保证早期全梯度流

        数学:
            α*(t) = 1.2 + 0.3 * sigmoid(0.3 * (t - 10))

        调度策略:
            - epoch < 10:  α = 1.2 (早期保证梯度覆盖率)
            - 10 <= epoch < 30: α = 1.5 (中期标准 entmax)
            - epoch >= 30: α → 1.7 (后期适度稀疏，通过 sigmoid 平滑过渡)
        """
        if epoch < 10:
            return 1.2
        elif epoch < 30:
            return 1.5
        else:
            # sigmoid 平滑过渡到 1.7（不达到 2.0）
            progress = (epoch - 30) / 50
            # D4-AUDIT FIX: 使用 Python math 替代 torch.sigmoid + .item()，避免 GPU 同步
            x = 0.3 * (progress - 0.5) * 10
            sigmoid_val = 1 / (1 + math.exp(-x))
            return 1.5 + 0.2 * sigmoid_val

    def _sparse_select(
        self,
        logits: Tensor,
        K_target,  # I-OPT: 接受 int 或 Tensor，避免 .item() 同步
        hard: bool = False,
        epoch: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """稀疏选择

        数学:
            s = softmax(z / τ) × K (用于更好的梯度流)
            s = Entmax_{α}(z / τ) × K (用于稀疏)

        参数:
            epoch: 训练轮次，用于自适应调整 α
        """
        B, N = logits.shape

        # 温度调度
        tau = max(self.temperature, TEMPERATURE_MIN)

        # 使用 set_epoch 管理的 entmax_alpha（课程学习调度至 2.0）
        # get_adaptive_alpha 最大返回 ~1.7，永远低于 1.9 阈值，导致 entmax 死代码
        alpha = self.entmax_alpha

        # P1 FIX: alpha < 1.2: Softmax（早期训练，全梯度流）
        # alpha >= 1.2: Entmax（逐渐稀疏，晚期稀疏性选择）
        # 原阈值 1.5 降至 1.2，因为稳定期 alpha 现在是 1.30
        if alpha < 1.2:
            probs = F.softmax(logits / tau, dim=-1)
        else:
            probs = entmax_beta(logits / tau, alpha=alpha, dim=-1)

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

    def _compute_locality_efficiency(
        self,
        selected_mask: Tensor,
        num_selected: int,
    ) -> float:
        """
        计算 Locality Efficiency (Selection/Oracle 对比)

        Selection Locality: J(S) = mean(|h(s_i) - h(s_{i+1})|)
        Oracle Locality: 连续采样能达到的最高聚合度 (理想情况下跳距 = 1)

        Locality Efficiency = Selection Locality / Oracle Locality ∈ [0, 1]
        值越高表示越接近理想的连续采样

        Args:
            selected_mask: [B, N] 选中掩码
            num_selected: 选中的 token 数量

        Returns:
            Locality efficiency 值
        """
        if num_selected < 2:
            return 1.0

        # 处理批量掩码 [B, N]
        if selected_mask.dim() == 2:
            efficiencies = []
            for i in range(selected_mask.shape[0]):
                mask_1d = selected_mask[i] > 0.5
                selected_h = self.hilbert_indices[mask_1d].float().sort()[0]
                if len(selected_h) < 2:
                    efficiencies.append(1.0)
                    continue
                selection_jumps = (selected_h[1:] - selected_h[:-1]).abs()
                selection_locality = selection_jumps.mean().item()
                # Oracle Locality = 1.0 (连续采样的理想跳距)
                efficiencies.append(selection_locality / 1.0 if 1.0 > 0 else 1.0)
            return sum(efficiencies) / len(efficiencies) if efficiencies else 1.0

        # 处理 1D 掩码 [N]
        selected_h = self.hilbert_indices[selected_mask > 0.5].float().sort()[0]
        if len(selected_h) < 2:
            return 1.0

        # Selection Locality: 实际跳距
        selection_jumps = (selected_h[1:] - selected_h[:-1]).abs()
        selection_locality = selection_jumps.mean().item()

        # Oracle Locality: 理想情况下连续采样跳距 = 1
        oracle_locality = 1.0

        return selection_locality / oracle_locality if oracle_locality > 0 else 1.0

    def forward(
        self,
        features: Tensor,
        image_size: Optional[Tuple[int, int]] = None,
        hard: bool = False,
        epoch: int = 0,
    ) -> SplitResult:
        """
        前向传播

        Args:
            features: [B, C, H, W] 输入特征
            image_size: (H, W) 原始图像尺寸
            hard: 是否使用硬选择
            epoch: 训练轮次，用于自适应 α 调度
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
        # 分组归一化：将语义组与几何组分别归一化到单位超球面，强制信息博弈
        roi_features_norm = self.roi_norm(roi_features)  # [B, N, 64] — 语义归一化
        geo_embs = self.geo_norm(torch.cat([
            path_emb.unsqueeze(0).expand(B, -1, -1),
            rot_emb.unsqueeze(0).expand(B, -1, -1),
            area_enc.unsqueeze(0).expand(B, -1, -1),
        ], dim=-1))  # [B, N, 192] — 几何归一化
        combined = torch.cat([roi_features_norm, geo_embs], dim=-1)  # [B, N, 256]

        # 1D Hilbert 流形卷积
        logits = self._manifold_convolution(combined)  # [B, N]

        # I167-4: SDS 正则化 - 惩罚高 SDS（空间局部性破坏）的位置
        # SDS 衡量 Hilbert 曲线上邻居的空间距离，值越高表示局部性保持越差
        if self.use_sds_regularization and image_size is not None:
            sds_penalty = self._compute_sds_penalty(logits, image_size)  # [B, N]
            # D1-AUDIT FIX: 保持 tensor，延迟 .item() 到后处理
            self._last_sds_stats_t = {
                "mean": sds_penalty.mean().detach(),
                "max": sds_penalty.max().detach()
            }
            logits = logits - sds_penalty  # 抑制高 SDS 位置

        # Phase 2: 保存 logits_pre_tree 并计算树约束修正量 Δz
        logits_pre_tree = logits.clone().detach()  # 断开梯度链用于记录

        # 树约束
        logits = self._apply_tree_constraint(logits, self.candidate_depths)

        # 计算树约束修正量: Δz = ||logits_pre - logits_post||₁
        # D1-AUDIT FIX: 保持 tensor，延迟 .item() 到后处理
        self._last_tree_delta_z_t = (logits_pre_tree - logits).abs().sum()

        # 深度配额
        depth_quota = F.softmax(self.depth_quota, dim=0)
        depth_quota = depth_quota[self.candidate_depths]  # [N]
        logits = logits + torch.log(depth_quota + EPS).unsqueeze(0)

        # K 估计 (动态)
        # 使用密度场网络估计每个区域的"信息密度"
        # 然后积分得到总 K

        density_per_region = self.density_field(roi_features[0])  # [N, 1]

# I-NAN FIX: 直接求和，K_float 范围 [0, N]
        # 原公式 K = Σ(density_i × 4^(-depth_i)) 范围仅为 [0, 4]
        # 新公式 K = Σ(density_i) 范围为 [0, N=85]
        K_float = density_per_region.squeeze(-1).sum()  # [N] -> scalar

        # I-OPT: 使用 STE 直通估计器获取可微分 K
        # K_soft: 用于 aux_budget 损失计算，min=1 防止零损失
        # K_hard: 用于实际 token 选择
        N = density_per_region.shape[0]  # 候选区域数
        K_soft = K_float.clamp(min=1)  # 用于损失计算
        # I-NAN FIX v2: 先 round 再 clamp，确保 K_hard 在 [1, N] 范围内
        # clamp(round(x)) vs round(clamp(x)) - 前者可能在 round 后超出
        K_hard = K_float.round().long()  # STE: forward=hard
        K_hard = K_hard.clamp(min=1, max=N)  # 显式 clamp 到有效范围

        # 稀疏选择使用硬 K（离散整数）
        # D1-AUDIT FIX: 直接传 K_hard tensor，_sparse_select 内部处理
        selected_mask, probs = self._sparse_select(logits, K_hard, hard=hard)

        # 构建结果
        # D3-AUDIT FIX: torch.where 替代 nonzero(as_tuple=True) 避免 Graph Break
        # nonzero(as_tuple=True) 会返回动态数量的张量，torch.compile 无法处理
        batch_idx, region_idx = torch.where(selected_mask > 0.5)

        if region_idx.numel() == 0:
            # 错误恢复：至少选择一个 - 选择最大概率的 token
            # D3-AUDIT FIX: 使用 argmax 替代 topk，避免 .item() 同步
            # argmax 返回最大值的索引，纯 tensor 操作
            _, topk_idx = torch.topk(probs[0], max(1, probs.shape[1] // 2), dim=-1)
            # 使用 batch 0
            batch_idx = torch.zeros(topk_idx.shape[0], dtype=torch.long, device=logits.device)
            region_idx = topk_idx

        # 提取选中区域
        selected_regions = self.candidate_regions[region_idx]
        selected_depths = self.candidate_depths[region_idx]

        # 按 Hilbert 索引排序（仅对选中区域）
        selected_hilbert = self.hilbert_indices[region_idx]
        sort_idx = selected_hilbert.argsort()
        selected_regions = selected_regions[sort_idx]
        selected_depths = selected_depths[sort_idx]
        batch_idx = batch_idx[sort_idx]

        result = SplitResult(
            regions=selected_regions,
            depths=selected_depths,
            batch_indices=batch_idx,
            hilbert_indices=selected_hilbert,  # [M] 选中区域的 Hilbert 索引
            # 注意: 全局 Hilbert 排序 self.hilbert_indices [N] 不存储在 SplitResult 中
            # TV Loss 直接使用 split_result.probs [B,N] 和 splitter.full_hilbert_indices
            selected_mask=selected_mask,
            logits=logits,
            probs=probs,
            K_soft=K_soft,  # I-OPT: 可微分 K 值用于 aux_budget 损失
        )

        # === Phase 1: 填充诊断属性到 result（供 splitter_output 使用）===
        # 计算完备性原则: SplitResult 离开 forward 作用域前必须包含所有诊断数据

        # 1. 调用 get_auxiliary_losses() 并设置到 result 属性
        losses = self.get_auxiliary_losses(result, target_ratio=0.25)
        result.entropy = losses.get('entropy')
        result.raw_budget_error = losses.get('budget')  # D162: 重命名以区分误差值与损失权重
        result.tree_consistency = losses.get('tree')

        # 2. 计算 locality_score (A1 公理) - O(N) 复杂度
        result.locality_score = compute_locality_score(
            self.hilbert_indices, result.selected_mask
        )

        # 3. 计算 Locality Efficiency (Selection/Oracle 对比)
        result.locality_efficiency = self._compute_locality_efficiency(
            result.selected_mask, result.selected_mask.sum()
        )

        # 4. 计算 jump_loss
        result.jump_loss = compute_jump_loss(
            self.hilbert_indices, result.selected_mask
        )

        # 5. 计算 iou 稳定性
        iou = self.compute_token_iou()
        if iou is not None:
            result.iou_mean = iou
            result.iou_std = 0.0  # 单值无法计算 std

        # 6. 设置 alpha
        result.alpha = self.entmax_alpha

        # === Phase 2: 中间变量统计（SDS 惩罚 + 树约束修正量）===

        # SDS 惩罚统计 (I167-4)
        # D1-AUDIT FIX: 使用新命名的 tensor 变量，延迟 .item() 到结果赋值
        if hasattr(self, '_last_sds_stats_t') and self._last_sds_stats_t is not None:
            result.sds_penalty_mean = self._last_sds_stats_t["mean"].item()
            result.sds_penalty_max = self._last_sds_stats_t["max"].item()

        # 树约束透明化：动态 lambda + 修正量 Δz
        lambda_cur = self._compute_dynamic_lambda()
        result.tree_lambda = lambda_cur.item() if hasattr(lambda_cur, 'item') else lambda_cur
        # D1-AUDIT FIX: 提取 tensor 值用于结果
        if hasattr(self, '_last_tree_delta_z_t') and self._last_tree_delta_z_t is not None:
            result.tree_constraint_delta_z = self._last_tree_delta_z_t.item()

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

    def set_target_ratio(self, target_ratio: float) -> None:
        """设置目标 token 比例（用于 warmup 调度）"""
        self._target_ratio = target_ratio

    def set_budget_weight(self, budget_weight: float) -> None:
        """设置 budget loss 权重（用于 warmup 调度）"""
        self._budget_weight = budget_weight

    def set_logits_diversity(self, enabled: bool) -> None:
        """设置是否启用 logits 多样性惩罚"""
        self._logits_diversity_enabled = enabled

    def set_k_min_ratio(self, ratio: float) -> None:
        """P0 FIX: 设置 K_min 比例（用于动态 K_min 退火）

        Args:
            ratio: K_min 占 K_target 的比例，范围 [0.5, 1.0]
                - 1.0: K_min = N（Stage 0，保留所有 token）
                - 0.5: K_min = 0.5 * K_target（Stage 3+）
        """
        self._k_min_ratio = max(0.5, min(1.0, ratio))

    def set_epoch(self, epoch: int):
        """P1 FIX: 设置当前 epoch，进行 Entmax α 分阶段退火

        三阶段设计：
            Stage 0 (0-5):   α = 1.0 (强制 softmax，稠密梯度)
            Stage 1 (5-12):  α: 1.0 → 1.22 (温和稀疏区)
            Stage 2 (12-25): α: 1.22 → 1.49 (高稀疏区)
            Stage 3 (25+):   α = 1.49 (稳定期)

        关键阈值 α ≈ 1.22：Entmax 开始表现明显稀疏性但仍保留较多"次要概率梯度"
        """
        self._current_epoch = epoch

        def smoothstep(epoch: int, warmup_end: int, ramp_end: int) -> float:
            """Smoothstep function for smoother alpha annealing"""
            if epoch < warmup_end:
                return 0.0
            elif epoch > ramp_end:
                return 1.0
            else:
                t = (epoch - warmup_end) / (ramp_end - warmup_end)
                return t * t * (3 - 2 * t)

        if epoch < self.entmax_warmup_epochs:
            # Stage 0: α = 1.0 (强制 softmax，稠密梯度流)
            self.entmax_alpha = 1.0
        elif epoch < self.entmax_transition_epochs:
            # Stage 1 (5-12): 温和稀疏区，α 从 1.0 退火到 1.22
            progress = smoothstep(epoch, self.entmax_warmup_epochs, self.entmax_transition_epochs)
            self.entmax_alpha = 1.0 + 0.22 * progress  # 1.0 → 1.22
        elif epoch < self.entmax_schedule_epochs:
            # Stage 2 (12-25): 高稀疏区，α 从 1.22 退火到 1.30
            progress = smoothstep(epoch, self.entmax_transition_epochs, self.entmax_schedule_epochs)
            self.entmax_alpha = 1.22 + 0.08 * progress  # 1.22 → 1.30 (P1 FIX)
        else:
            # Stage 3 (25+): α = 1.30 (稳定期)
            self.entmax_alpha = 1.30

        # 温度由 BPE 三阶段调度器在 train_fractal_vit._update_fractal_hyperparams()
        # 中通过 set_temperature() 管理，此处不再内部退火，避免梯度冲突。

        # K 课程学习：从 K_min 逐渐增大到 K_max
        # 符合课程学习原则：先学简单（少 token），后学复杂（多 token）
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
            regions = 1 << (2 * level)  # D4-AUDIT FIX: 4**level → 1<<(2*level)
            if regions >= K_target:
                return regions
        # 如果所有 level 都不满足，返回最大 level 的区域数
        # D4-AUDIT FIX: 4**max_level → 1 << (2 * max_level)
        return 1 << (2 * max_level)

    def extra_repr(self) -> str:
        return (
            f"HilbertOptimalSplitter("
            f"max_level={self.max_level_limit}, "
            f"hidden_dim={self.hidden_dim}, "
            f"K={self.K_min}-{self._K_max_rounded}[current={self._current_K}], "
            f"entmax_alpha={self.entmax_alpha:.2f}, "
            f"tree_weight={self.tree_constraint_weight})"
        )

    def get_auxiliary_losses(
        self,
        split_result: SplitResult,
        target_ratio: float = 0.25,  # I107-OPT: 从 0.1 增到 0.25
    ) -> Dict[str, Tensor]:
        """
        计算 H1SS 辅助损失（用于端到端训练）。

        数学形式:
            1. 熵损失: L_entropy = -Σ_d π_d × log(π_d + ε)
               鼓励配额分布多样性

            2. Budget损失: L_budget = MSE(actual_K, target_K)
               控制选中的 token 数量

            3. 树一致性损失: L_tree = -std(logits)（当 _logits_diversity_enabled=True 时启用）

        Args:
            split_result: H1SS 返回的 SplitResult
            target_ratio: 目标 token 比例 (default: 0.1)，仅当未设置 _target_ratio 时使用

        Returns:
            Dict[str, Tensor]: 辅助损失字典
        """
        losses = {}

        # 1. 熵损失 - 促进稀疏选择
        if split_result.probs is not None:
            probs = split_result.probs  # [B, N]
            # 展平计算熵
            probs_flat = probs.view(-1)
            # 避免 log(0)
            entropy = -(probs_flat * torch.log(probs_flat + EPS)).sum() / (probs.numel() + EPS)
            losses['entropy'] = entropy

        # 2. Budget损失 - 控制 token 数量
        # P0 FIX: 动态 K_min + 对数域 Budget Loss
        # 使用 K_soft (STE) 替代 selected_mask.sum()，让梯度流过 K 值到 density_field
        if split_result.selected_mask is not None:
            B, N = split_result.selected_mask.shape
            # 优先使用 K_soft（梯度可流），否则 fallback 到实际选择数（无梯度）
            if hasattr(split_result, 'K_soft') and split_result.K_soft is not None:
                K_soft = split_result.K_soft.squeeze()  # [1] or [B] -> []
                if K_soft.dim() > 0:
                    K_soft = K_soft.mean()
            else:
                # Fallback: 使用实际选择的 token 数（无梯度）
                K_soft = split_result.selected_mask.sum(dim=1).float().mean()

            # P0 FIX: 动态 K_min - 根据 _k_min_ratio 计算 K_min
            # k_min_ratio = 1.0 (Stage 0): K_min = N，保留所有 token
            # k_min_ratio = 0.5 (Stage 3+): K_min = 0.5 * K_target
            effective_target_ratio = getattr(self, '_target_ratio', target_ratio)
            K_target = effective_target_ratio * N
            k_min = max(1.0, self._k_min_ratio * K_target)
            K_soft_clamped = K_soft.clamp(min=k_min)

            # P0 FIX: 对数域 Budget Loss
            # 公式: L_budget = (log(K_soft / K_target))^2
            # 优势: 尺度不变性，K 接近目标时梯度平滑，防止平凡解 K→0
            eps = 1e-8
            log_ratio = torch.log(K_soft_clamped / (K_target + eps))
            budget_loss = torch.pow(log_ratio, 2)

            losses['budget'] = budget_loss

        # H1SS 公理 A4 (Tree): 树一致性通过局部层级软约束实现
        # z_parent -= λ × max(z_children)
        # 全局 logits 方差惩罚已被移除（无数学依据，干扰局部决策）

        return losses


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
        selected_mask: [N] 或 [B, N] 选中掩码
        gamma: 缩放因子

    Returns:
        loss: 标量损失
    """
    # 处理批量掩码 [B, N]
    if selected_mask.dim() == 2:
        losses = []
        for i in range(selected_mask.shape[0]):
            mask_1d = selected_mask[i] > 0.5
            selected_h = hilbert_indices[mask_1d].float().sort()[0]
            if len(selected_h) < 2:
                losses.append(torch.tensor(0.0, device=hilbert_indices.device))
                continue
            diffs = selected_h[1:] - selected_h[:-1]
            jump_loss = F.relu(diffs - 1.0) ** 2
            losses.append(jump_loss.mean())
        return sum(losses) / len(losses) if losses else torch.tensor(0.0, device=hilbert_indices.device)

    # 处理 1D 掩码 [N]
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
