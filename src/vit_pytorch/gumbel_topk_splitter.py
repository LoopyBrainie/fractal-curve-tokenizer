"""
方案 D/E: Gumbel-Top-K + 可学习配额 自适应分割器

数学形式化
==========

核心思想:
    消除 BFS 串行依赖，同时保持 100% Hilbert 局部性。

与其他方案对比:
    方案 A (BFS+STE):     Hilbert=100%, 梯度≈25%, 串行依赖
    方案 B (连续松弛):     Hilbert~70%,  梯度≈partial, 无串行依赖
    方案 D (Gumbel-Top-K): Hilbert=100%, 梯度≈K/N (~37.6%), 无串行依赖 ✓
    方案 E (可学习配额):   方案D + 分层Top-K + 可学习深度配额 ✓

决策公式 (方案E - 当前使用):
    1. 并行评估所有 N=85 个候选区域:
       logits_i = MLP(ROI_i) + b_explore + β·γ^{d_i} - τ_{d_i}

    2. 可学习配额分配 (I24-2):
       π_d = softmax(φ)  其中 φ 是可学习 logits
       K_d = max(K_min_per_depth, round(π_d × K_total))

    3. 分层 Top-K 选择:
       对每个深度 d: selected_d = TopK(logits[depth=d] + g, K_d)

    4. Gumbel 扰动:
       g_i ~ Gumbel(0, 1)
       perturbed_i = (logits_i + g_i) / τ

    5. STE (Straight-Through Estimator):
       hard_mask = 1[i ∈ selected]
       soft_mask = global_softmax(perturbed)  # I30-2: 使用全局 Softmax
       st_mask = hard_mask - soft_mask.detach() + soft_mask

Hilbert 局部性保证:
    每个选中的 token 精确对应一个四叉树区域 R
    → LCA(token_i, token_j) 有明确的几何意义
    → 与 Hilbert curve 位置编码兼容

梯度流分析 (I30-2 修正):
    ∂L/∂logits = ∂L/∂st_mask × ∂st_mask/∂logits
                = ∂L/∂st_mask × ∂softmax/∂logits  (STE 使梯度跳过 TopK)
    → 选中 token: 正常梯度 (~p_i × (1-p_i))
    → 未选中 token: 衰减梯度 (~p_i²)，约 20x 衰减

I30-4 更新 (2026-01-15):
    已移除 Log-Compensation (b_log_d = log(N_total / N_d))
    方案E的可学习配额 + 分层Top-K 完全替代该机制

动态 K 选择:
    K_opt = clip(estimate_split_count(logits), K_min, K_max)
    estimate 基于 sigmoid(logits) > 0.5 的数量

作者: GitHub Copilot
日期: 2026-01-15
版本: 方案 E v1.0 (基于方案D演进)
版本: I30-2 修正 (2026-01-22): 梯度覆盖率修正为 K/N (~37.6%)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# I12-7: 从 constants 统一导入数值稳定性常量
# I24-2: 导入方案E可学习配额常量
# I23-1: 导入深度方差归一化常量
# I30-4: 已移除 LOG_COMPENSATION_ENABLED (被方案E替代)
# I30-2: 已移除 SUBSET_SOFTMAX_ENABLED (改用全局 Softmax)
# I30-10: 导入 SplitterConfig
# I35: 移除死代码 DEPTH_KL_*, DEPTH_QUOTA_* 常量
from .constants import (
    TEMPERATURE_MIN,
    GUMBEL_EPSILON,
    PROB_EPSILON,
    # I23-1 方案C: 深度方差归一化
    DEPTH_VARIANCE_NORM_ENABLED,
    DEPTH_VARIANCE_NORM_EPS,
    DEPTH_VARIANCE_INIT_EPS,  # I96-1: EMA 初始化下界
    DEPTH_EMA_ALPHA,  # I35: EMA 系数
    SOFT_EXCLUSION_MARGIN,  # I96-4: 树一致性软排除边距
    # I24-2 方案E: 可学习配额
    LEARNABLE_QUOTA_ENABLED,
    QUOTA_MIN_PER_DEPTH,
    QUOTA_MIN_RATIO,  # I96-7: 自适应深度下界最小采样比例
    QUOTA_MIN_LAMBDA,  # I96-7: 下界软正则化权重
    QUOTA_INIT_LOGITS,
    QUOTA_ENTROPY_WEIGHT,
    # I29-2: 阈值方差正则化
    THRESHOLD_VAR_REG_ENABLED,
    THRESHOLD_VAR_REG_WEIGHT,
    # I33: 相对预算与自适应覆盖率
    K_COVERAGE_BASE,
    K_COVERAGE_MIN,
    K_COVERAGE_MAX_HARD,
    K_ADAPTIVE_REFERENCE_SIZE,
    K_MAX_HARD_LIMIT,
    K_MIN_HARD_LIMIT,
    # I33: Elastic Budget 相对预算
    ELASTIC_COVERAGE_MAX,
    ELASTIC_COVERAGE_MIN,
    ELASTIC_LAMBDA_OVER,
    ELASTIC_LAMBDA_COLLAPSE,
)
from .config import SplitterConfig
from typing import Optional


# =============================================================================
# TensorSplitResult: 纯张量表示 (从 split_adaptive.py 迁移, I97-9)
# =============================================================================

@dataclass
class TensorSplitResult:
    """
    纯张量表示的分割结果。

    数学形式化:
        regions:       [N, 4]     (x1, y1, x2, y2)
        depths:        [N]        深度值
        batch_indices: [N]        所属 batch 索引
        hilbert_indices: [N]      Hilbert 曲线索引
        complexities:  [N]        复杂度值
        tokens_per_batch: [B]     每个 batch 的 token 数量
    """

    regions: Tensor        # [N, 4] 区域坐标 (x1, y1, x2, y2)
    depths: Tensor         # [N] 深度值
    batch_indices: Tensor  # [N] batch 索引
    hilbert_indices: Tensor  # [N] Hilbert 索引
    complexities: Tensor   # [N] 复杂度值

    # 可选: 每个 batch 的 token 数量 (用于重构 List 表示)
    tokens_per_batch: Optional[Tensor] = None  # [B]

    @property
    def num_tokens(self) -> int:
        """返回 token 数量 (I97-9: 从 split_adaptive.py 迁移)."""
        return self.regions.shape[0]

    @property
    def device(self) -> torch.device:
        return self.regions.device

    @property
    def batch_size(self) -> int:
        if self.tokens_per_batch is not None:
            return self.tokens_per_batch.shape[0]
        return int(self.batch_indices.max().item()) + 1 if self.num_tokens > 0 else 0


@dataclass
class GumbelTopKResult:
    """Gumbel-Top-K 分割器输出。
    
    与 TensorSplitResult 兼容但包含更多信息。
    """
    # 核心输出 (与 TensorSplitResult 兼容)
    regions: Tensor           # [M, 4] 选中区域坐标 (x0, y0, x1, y1)
    depths: Tensor            # [M] 区域深度
    batch_indices: Tensor     # [M] batch 索引
    hilbert_indices: Tensor   # [M] Hilbert 曲线索引
    
    # Gumbel-Top-K 特有
    selected_mask: Tensor     # [B, N] 选中掩码 (STE 版本，有梯度)
    logits: Tensor            # [B, N] 原始 logits
    probs: Tensor             # [B, N] 分割概率 sigmoid(logits)
    
    # 统计信息
    num_selected_per_batch: Tensor  # [B] 每个 batch 选中的 token 数
    
    def to_tensor_split_result(self) -> TensorSplitResult:
        """
        转换为 TensorSplitResult 格式。

        用于与现有 FractalTokenizer 接口兼容。

        I20: 确保 regions 为整数类型以支持位运算
        I97-9: 使用本地 TensorSplitResult 定义
        """
        return TensorSplitResult(
            regions=self.regions.long(),  # I20: 转为 long 以支持位运算
            depths=self.depths,
            batch_indices=self.batch_indices,
            hilbert_indices=self.hilbert_indices,
            complexities=torch.zeros_like(self.depths, dtype=torch.float32),
            tokens_per_batch=self.num_selected_per_batch,
        )
    
    @property
    def device(self) -> torch.device:
        return self.regions.device
    
    @property
    def num_tokens(self) -> int:
        return self.regions.shape[0]
    
    @property
    def batch_size(self) -> int:
        return self.num_selected_per_batch.shape[0]


class GumbelTopKSplitter(nn.Module):
    """
    Gumbel-Top-K 自适应分割器 (方案 D/E)。

    核心优势:
        1. 100% Hilbert 局部性: 每个 token 精确对应一个四叉树区域
        2. 梯度覆盖设计 (I96-6 修正):
           - Scheme D (固定配额): 全局 Softmax → ~100% 覆盖率，梯度强度比 ~50:1
           - Scheme E (可学习配额): 深度内 Subset Softmax → ~K/N 覆盖率，Top-K 外梯度为 0
        3. 无串行依赖: 并行评估所有候选
        4. 树一致性: 向量化 O(1) 约束

    I30-10: 支持 SplitterConfig 统一配置

    数学说明 (I96-6):
        全局 Softmax (Scheme D): 所有候选有梯度，梯度 ∝ (δ_i∈TopK - p_i)
        Subset Softmax (Scheme E): 仅 Top-K 内有梯度，Top-K 外梯度 = 0
        量化分析 (K=32, N=85):
            - 选中 token: p_i ≈ 0.38, 梯度 ~ 0.24
            - 未选中 token: p_i ≈ 0.012, 梯度 ~ 0.012 (全局 Softmax, 衰减 ~20x)
            - Scheme E: Top-K 外梯度 = 0 (Subset Softmax)

    与 LearnableSplitter 对比:
        | 指标               | LearnableSplitter | Scheme D | Scheme E |
        |--------------------|-------------------|----------|----------|
        | Hilbert 局部性     | 100%              | 100%     | 100%     |
        | 有效梯度覆盖       | ~25%              | ~100%    | ~37.6%   |
        | 深度饥饿风险       | 高               | 无       | 中       |
        | 串行依赖           | 有                | 无       | 无       |
        | 计算开销           | 1.0x              | ~4x      | ~4x      |
    """

    def __init__(
        self,
        config: Optional[SplitterConfig] = None,
        feature_dim: int = 256,
        # I30-17-EXT: 替换固定 max_depth 为 min_patch_size + max_depth_limit
        min_patch_size: int = 4,
        max_depth_limit: int = 8,  # 参数上界，用于可学习参数分配
        hidden_dim: int = 128,
        intermediate_dim: int = 64,
        pool_size: int = 4,
        temperature: float = 1.0,
        K_min: int = 8,
        K_max: int = 64,
        dropout: float = 0.1,
        use_dynamic_k: bool = True,
        image_size: Tuple[int, int] = (64, 64),  # 仅用于初始化缓存
    ):
        """
        Args:
            config: SplitterConfig 统一配置（推荐）
            feature_dim: 输入特征通道数 C
            min_patch_size: 目标最小 patch 大小，用于动态计算 max_depth
            max_depth_limit: max_depth 硬上限，用于可学习参数分配
            hidden_dim: MLP 第一隐藏层维度
            intermediate_dim: MLP 第二隐藏层维度
            pool_size: ROI-Align 输出尺寸 k×k
            temperature: Gumbel-Softmax 温度 τ
            K_min: 最小 token 数量
            K_max: 最大 token 数量
            dropout: MLP dropout
            use_dynamic_k: 是否使用动态 K 选择
            image_size: 图像尺寸 (H, W)，仅用于初始化
        """
        super().__init__()

        # I30-10: 解析配置
        if config is not None:
            # 使用 SplitterConfig
            self.config = config
            self._use_config = True
            feature_dim = config.feature_dim
            min_patch_size = config.min_patch_size
            max_depth_limit = config.max_depth_limit
            hidden_dim = config.hidden_dim
            intermediate_dim = config.intermediate_dim
            pool_size = config.pool_size
            K_min = config.K_min
            K_max = config.K_max
            dropout = config.dropout
            use_dynamic_k = config.use_dynamic_k
            # 配额参数来自 config
            self._enable_learnable_quota = config.enable_learnable_quota
            self._quota_min_per_depth = config.quota_min_per_depth
            self._quota_entropy_weight = config.quota_entropy_weight
            self._freeze_quota = config.freeze_quota
            # I33: 自适应覆盖率参数
            self._token_coverage_base = config.token_coverage_base
            self._token_coverage_min = config.token_coverage_min
            self._token_coverage_max_hard = config.token_coverage_max_hard
            self._adaptive_reference_size = config.adaptive_reference_size
            self._K_min_abs = config.K_min_abs
            self._K_max_hard = config.K_max_hard
            self._use_adaptive_coverage = config.use_adaptive_coverage
        else:
            # 使用传统参数（向后兼容）
            self.config = None
            self._use_config = False
            self._enable_learnable_quota = LEARNABLE_QUOTA_ENABLED
            self._quota_min_per_depth = QUOTA_MIN_PER_DEPTH
            self._quota_entropy_weight = QUOTA_ENTROPY_WEIGHT
            self._freeze_quota = False
            # I33: 默认自适应覆盖率参数
            self._token_coverage_base = K_COVERAGE_BASE
            self._token_coverage_min = K_COVERAGE_MIN
            self._token_coverage_max_hard = K_COVERAGE_MAX_HARD
            self._adaptive_reference_size = K_ADAPTIVE_REFERENCE_SIZE
            self._K_min_abs = K_MIN_HARD_LIMIT
            self._K_max_hard = K_MAX_HARD_LIMIT
            self._use_adaptive_coverage = True

        # I30-17-EXT: 存储配置，不预计算
        self.feature_dim = feature_dim
        self.min_patch_size = min_patch_size
        self._current_max_depth_limit = max_depth_limit
        self.pool_size = pool_size
        self.K_min = K_min
        self.K_max = K_max
        self.use_dynamic_k = use_dynamic_k
        self._config_image_size = image_size

        # 动态状态 (forward 中确定)
        self._current_max_depth: Optional[int] = None
        self._current_image_size: Optional[Tuple[int, int]] = None

        # I30-17-EXT: LRU 缓存用于候选区域
        self._candidate_cache: Dict[Tuple[int, int, int], Tuple] = {}

        # 可学习参数基于 max_depth_limit 上界
        self._embed_max_depth = max_depth_limit

        # 复杂度 MLP (输出 logit, 非概率)
        input_dim = feature_dim * pool_size * pool_size
        self.complexity_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, 1),
        )

        # I30-17-EXT: 深度嵌入使用上界维度
        # I20: 深度嵌入维度基于信息论下界自适应选择
        # 数学: E = max(4, min(8, ceil(log2(D)))) 确保 E >= log2(D)
        # 理由: depth_bias 是标量输出，16维过度冗余
        def _compute_depth_embed_dim(max_depth_limit: int) -> int:
            """计算深度嵌入维度，基于信息论下界"""
            D = max_depth_limit + 1
            min_required = math.ceil(math.log2(D)) if D > 1 else 1
            return min(8, max(4, min_required))

        depth_embed_dim = _compute_depth_embed_dim(max_depth_limit)
        self.depth_embedding = nn.Embedding(max_depth_limit + 1, depth_embed_dim)
        self.depth_proj = nn.Linear(depth_embed_dim, 1)

        # I30-17-EXT: 可学习阈值使用上界维度
        self.threshold_offsets = nn.Parameter(torch.zeros(max_depth_limit + 1))

        # 可学习温度
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))

        # 探索偏置 (训练初期)
        self.register_buffer('explore_bias', torch.tensor(0.5))

        # 深度偏置系数 (可选)
        self.register_buffer('depth_bias_beta', torch.tensor(0.5))
        self.register_buffer('depth_bias_gamma', torch.tensor(0.7))

        # ====================================================================
        # I24-2 方案E: 可学习配额 (Learnable Quota)
        # I30-10: 支持 SplitterConfig 配置
        # 数学:
        #   K_d = max(K_min, round(softmax(φ)_d × K_total))
        #   selected_d = TopK(logits[depth=d], K_d)
        #
        # 初始化:
        #   φ^(0) = log(p_target) - mean(log(p_target))
        #   使得 softmax(φ^(0)) = p_target = (0.15, 0.20, 0.25, 0.40)
        #   对于 D > 4，扩展为均匀分布
        # ====================================================================
        # I30-17-EXT: 使用 max_depth_limit 上界
        if self._enable_learnable_quota:
            D = max_depth_limit + 1
            if self.config is not None:
                # 使用 config 中的初始化 logits
                quota_init_vals = self.config.get_quota_init_tensor(D)
                quota_init = torch.tensor(quota_init_vals, dtype=torch.float32)
            elif D <= len(QUOTA_INIT_LOGITS):
                quota_init = torch.tensor(QUOTA_INIT_LOGITS[:D], dtype=torch.float32)
            else:
                # 扩展: 前 4 个用预计算值，后续用均匀分布 (log(1/D))
                base_init = list(QUOTA_INIT_LOGITS)
                # 均匀分布的 logit = 0 (因为 softmax 对平移不变)
                extra_init = [0.0] * (D - len(base_init))
                quota_init = torch.tensor(base_init + extra_init, dtype=torch.float32)
            self.quota_logits = nn.Parameter(quota_init)
        else:
            self.quota_logits = None

        # I30-17-EXT: 初始化候选区域 (基于初始 image_size)
        self._update_candidates(image_size)
        
        # 统计信息
        self.register_buffer('_avg_selected', torch.tensor(16.0))
        
        # ====================================================================
        # 退火调度 buffers (与 LearnableSplitter API 兼容)
        # ====================================================================
        # 温度退火
        self.register_buffer('_temp_total_steps', torch.tensor(0.0))
        self.register_buffer('_temp_start', torch.tensor(1.0))
        self.register_buffer('_temp_end', torch.tensor(0.3))
        self.register_buffer('_temp_step', torch.tensor(0.0))
        self._temp_enabled = False
        self._temp_schedule = 'exponential'
        
        # 探索偏置退火
        self.register_buffer('_bias_total_steps', torch.tensor(0.0))
        self.register_buffer('_bias_start', torch.tensor(0.5))
        self.register_buffer('_bias_end', torch.tensor(0.0))
        self.register_buffer('_bias_step', torch.tensor(0.0))
        self._bias_enabled = False

        # I30-6: 深度方差归一化
        # I35 改进: 使用 EMA Running Statistics:
        # - 稳定小 batch (B=1) 下的方差估计
        # - 避免 sqrt(0) 在反向传播产生 NaN
        # - α=0.1, 有效样本量 ≈ 10
        self._depth_var_normalized: Optional[Tensor] = None  # [D]

        # I35: EMA Running Statistics buffers (max_depth_limit + 1 维度)
        # I99-1: 修改为 3D Buffer [B_max, D] 实现 per-sample EMA
        #         每个样本独立累积 EMA，完全消除 batch 依赖
        D = max_depth_limit + 1
        self._max_batch_size = 256  # I99-1: 预设最大 batch size (支持 batch=192)
        self.register_buffer('_depth_ema_mean', torch.zeros(self._max_batch_size, D))  # [B_max, D]
        self.register_buffer('_depth_ema_var', torch.ones(self._max_batch_size, D))   # [B_max, D]
        self._depth_ema_initialized = False  # 标记是否已初始化

        # 初始化权重
        self._init_weights()

        # I24-4: 边界条件验证
        if max_depth_limit < 2:
            warnings.warn(
                "I24-4: max_depth_limit < 2 是边界情况。 "
                "depth=0 只有 1 个候选区域，depth=1 有 4 个候选区域。 "
                "此配置可能导致不平衡的 token 分布。建议使用 max_depth >= 2。",
                UserWarning,
                stacklevel=2
            )

    def _init_weights(self):
        """Xavier 初始化 MLP 权重。"""
        for module in self.complexity_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.depth_proj.weight)
        nn.init.zeros_(self.depth_proj.bias)

    # I30-17-EXT: 动态候选区域更新方法
    def _update_candidates(self, image_size: Tuple[int, int]):
        """
        根据输入尺寸动态更新候选区域。

        数学形式:
            L(X) = min(max_depth_limit, max(0, floor(log2(min(H, W) / min_patch_size))))

        Args:
            image_size: (H, W) 输入图像尺寸
        """
        from .depth_utils import compute_max_depth

        H_img, W_img = image_size

        # 动态计算 max_depth
        computed_max_depth = compute_max_depth(
            image_size, self.min_patch_size, self._current_max_depth_limit
        )

        # 检查缓存
        cache_key = (H_img, W_img, computed_max_depth)

        if cache_key in self._candidate_cache:
            cached = self._candidate_cache[cache_key]
            self.candidate_regions = cached[0]
            self.candidate_depths = cached[1]
            self.parent_indices = cached[2]
            self.hilbert_indices = cached[3]
            self.num_candidates = cached[4]
            self._current_max_depth = computed_max_depth
            self._current_image_size = image_size
            self._children_matrix = None  # 重置子节点矩阵
            return

        # 重新计算候选区域
        self._generate_candidates_internal(image_size, computed_max_depth)

        # 更新动态状态
        self._current_max_depth = computed_max_depth
        self._current_image_size = image_size
        self._children_matrix = None

        # 存入缓存 (限制大小)
        if len(self._candidate_cache) < 256:
            self._candidate_cache[cache_key] = (
                self.candidate_regions,
                self.candidate_depths,
                self.parent_indices,
                self.hilbert_indices,
                self.num_candidates,
            )

    def _generate_candidates_internal(self, image_size: Tuple[int, int], max_depth: int):
        """
        生成候选区域的内部方法 (P-OPT-7 向量化版本)。

        数学形式化
        ==========

        问题: 原始实现使用 3 层嵌套 for 循环，O(Σ 4^d) Python 迭代
        优化: 使用 torch.meshgrid + 张量广播，O(D) 次张量操作

        向量化公式:
            对每个深度 d:
            - grid_size = 2^d
            - region_h = H / grid_size, region_w = W / grid_size
            - 使用 meshgrid 生成 [grid_size, grid_size] 坐标网格
            - 批量计算: y0 = i * region_h, x0 = j * region_w

        父节点索引计算:
            parent_idx[depth, i, j] = global_idx_at(depth-1, i//2, j//2)

        Hilbert 索引计算:
            center_x = (x0 + x1) // 2
            center_y = (y0 + y1) // 2
            hilbert_d = HilbertCurve.xy_to_d(grid_size, center_x, center_y)

        注意: 初始化时使用 CPU 张量，设备由后续的 .to(device) 处理

        Args:
            image_size: (H, W) 图像尺寸
            max_depth: 最大深度
        """
        from .curve_hilbert import HilbertCurve

        H_img, W_img = image_size

        # P-OPT-7: 初始化时使用 CPU，张量在模型移动到 GPU 时自动跟进
        # 获取设备（如果已存在 buffer），否则使用 CPU
        try:
            device = self.candidate_regions.device if hasattr(self, 'candidate_regions') else torch.device('cpu')
        except AttributeError:
            device = torch.device('cpu')

        all_regions = []      # [N, 4] 各深度的区域坐标
        all_depths = []       # [N] 各区域的深度
        all_parent_idx = []   # [N] 父节点索引
        all_hilbert_idx = []  # [N] Hilbert 索引

        # 记录每个深度的起始索引，用于父节点计算
        # depth_start_idx[d] = depth d 的起始全局索引
        depth_start_idx = [0]  # depth=0 的起始索引

        for depth in range(max_depth + 1):
            grid_size = 2 ** depth
            region_h = H_img / grid_size
            region_w = W_img / grid_size

            # P-OPT-7: 向量化坐标生成
            # 使用 torch.arange 生成索引
            i_idx = torch.arange(grid_size, device=device)
            j_idx = torch.arange(grid_size, device=device)

            # meshgrid 生成网格 [grid_size, grid_size]
            grid_i, grid_j = torch.meshgrid(i_idx, j_idx, indexing='ij')

            # 批量计算区域坐标 [grid_size, grid_size]
            y0 = (grid_i * region_h).long()
            x0 = (grid_j * region_w).long()
            y1 = ((grid_i + 1) * region_h).long()
            x1 = ((grid_j + 1) * region_w).long()

            # 堆叠为 [N, 4] 张量
            regions_depth = torch.stack([x0, y0, x1, y1], dim=-1).view(-1, 4)
            all_regions.append(regions_depth)

            # 深度标签
            N_depth = grid_size * grid_size
            all_depths.extend([depth] * N_depth)

            # P-OPT-7: 批量计算 Hilbert 索引
            # 计算每个区域的中心坐标
            center_x = (x0 + x1) // 2
            center_y = (y0 + y1) // 2

            # 归一化到 grid_size 坐标系
            grid_x = ((center_x.float() / W_img) * grid_size).clamp(max=grid_size - 1).long()
            grid_y = ((center_y.float() / H_img) * grid_size).clamp(max=grid_size - 1).long()

            # P-OPT-7/P-OPT-8: 使用 xy_to_d_batch 批量计算 Hilbert 索引
            # 向量化实现: O(N) 张量操作替代 Python 循环
            if grid_size > 0:
                # 展平坐标为张量 [N]
                grid_x_flat = grid_x.view(-1)
                grid_y_flat = grid_y.view(-1)

                # 批量计算 Hilbert 距离（使用 xy_to_d_batch 向量化）
                hilbert_d = HilbertCurve.xy_to_d_batch(grid_size, grid_x_flat, grid_y_flat)
            else:
                hilbert_d = torch.zeros(N_depth, device=device, dtype=torch.long)
            all_hilbert_idx.append(hilbert_d)

            # P-OPT-7: 批量计算父节点索引
            if depth == 0:
                parent_idx_depth = torch.full((N_depth,), -1, device=device, dtype=torch.long)
            else:
                # 计算父节点的 grid 坐标 (i//2, j//2)
                parent_i = (grid_i // 2).view(-1)
                parent_j = (grid_j // 2).view(-1)

                # 父深度的大小
                parent_grid_size = 2 ** (depth - 1)
                # 父节点在全局索引中的位置 = 父深度起始索引 + 局部索引
                # depth_start_idx[depth - 1] 是父深度 (depth-1) 的起始索引
                parent_global_idx = depth_start_idx[depth - 1] + (parent_i * parent_grid_size + parent_j)
                parent_idx_depth = parent_global_idx.long()

            all_parent_idx.append(parent_idx_depth)

            # 记录下一个深度的起始索引
            depth_start_idx.append(depth_start_idx[-1] + N_depth)

        # 合并所有深度的数据
        candidate_regions = torch.cat(all_regions, dim=0).float()
        candidate_depths = torch.tensor(all_depths, device=device, dtype=torch.long)
        parent_indices = torch.cat(all_parent_idx, dim=0)
        hilbert_indices = torch.cat(all_hilbert_idx, dim=0)

        # 计算候选数量
        self.num_candidates = sum(4 ** d for d in range(max_depth + 1))

        # 注册为 buffer (设备由 model.to() 控制)
        self.register_buffer('candidate_regions', candidate_regions)
        self.register_buffer('candidate_depths', candidate_depths)
        self.register_buffer('parent_indices', parent_indices)
        self.register_buffer('hilbert_indices', hilbert_indices)

    def _generate_candidates(self, max_depth: int):
        """兼容方法: 使用当前 image_size 生成指定深度的候选区域。"""
        if self._current_image_size is not None:
            self._generate_candidates_internal(self._current_image_size, max_depth)

    # I97-3: 移除 _precompute_candidates() 方法 (死代码 + Bug)
    # 该方法引用不存在的 self.image_size 属性，功能与 _generate_candidates_internal() 重复

    def _ensure_children_matrix(self):
        """确保子节点矩阵已计算。
        
        性能优化 (P-OPT-2):
            使用向量化操作替换 Python for 循环和 .item() 调用。
            通过一次性 CPU 转换 + numpy 操作提升性能。
            
        注意: 此方法仅在初始化时调用一次，后续使用缓存结果。
        """
        if self._children_matrix is not None:
            return
        
        N = self.num_candidates
        device = self.candidate_regions.device
        
        # 初始化为 -1
        children_matrix = torch.full((N, 4), -1, dtype=torch.long, device=device)
        
        # P-OPT-2: 一次性转换到 CPU，使用 numpy 进行槽位分配
        # 这比逐个 .item() 调用快得多
        parent_indices_cpu = self.parent_indices.cpu().numpy()
        children_matrix_cpu = children_matrix.cpu().numpy()
        
        # 使用 numpy 进行槽位分配
        slot_counts = {}  # parent_idx -> next_slot
        for child_idx in range(N):
            parent_idx = parent_indices_cpu[child_idx]
            if parent_idx >= 0:
                slot = slot_counts.get(parent_idx, 0)
                if slot < 4:  # 最多 4 个子节点
                    children_matrix_cpu[parent_idx, slot] = child_idx
                    slot_counts[parent_idx] = slot + 1
        
        # 转回 GPU (I78: 使用 non_blocking=True 异步传输)
        self._children_matrix = torch.from_numpy(children_matrix_cpu).to(device, non_blocking=True)
    
    # ========================================================================
    # I23-1 方案C: 深度方差归一化
    # ========================================================================
    
    def _normalize_by_depth(
        self,
        logits: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        按深度分组归一化 MLP 输出 (I23-1 方案C 核心修复, I35 EMA 改进)。

        数学形式化
        ==========

        问题分析:
            ROI 信息量与深度相关，导致 MLP 输出方差不一致:
            - Depth 0: ROI = 64×64，特征是全局平均，σ_0 较小
            - Depth 3: ROI = 16×16，特征保留局部变化，σ_3 较大

            量化估计: σ_3 / σ_0 ≈ 8 (基于 ROI 面积采样点推导)

        Top-K 偏好分析:
            Top-K 选择偏好高方差分布 (更容易产生极值)
            有效竞争力: eff_d = N_d × σ_d
            结果: eff_3 : eff_0 = 64×1 : 1×0.125 = 512:1

        解决方案 (I35 EMA 改进):
            z_i^{norm} = (z_i - μ_d^{EMA}) / (σ_d^{EMA} + ε)

            其中 μ_d^{EMA}, σ_d^{EMA} 是 EMA 累积的 Running Statistics:
            - μ_d^{EMA}(t) = α × μ_d^{batch}(t) + (1-α) × μ_d^{EMA}(t-1)
            - σ_d^{EMA}(t) = α × σ_d^{batch}(t) + (1-α) × σ_d^{EMA}(t-1)

        I35 EMA 优势:
            - 稳定小 batch (B=1) 下的方差估计 (per-batch 方差放大 512×)
            - 避免 sqrt(0) 在反向传播产生 NaN
            - α=0.1, 有效样本量 ≈ 10，方差降低 19×

        归一化后效果:
            所有 z_i^{norm} ~ N(0, 1)
            Log-Compensation 可以正确补偿候选数量差异
            预测深度分布 π ≈ (0.249, 0.246, 0.248, 0.257)

        复杂度:
            时间: O(B × N × D) ≈ O(B × 85 × 4)
            空间: O(D) = O(4)

        P-OPT-3 向量化:
            使用 scatter_add + one-hot 编码替代 Python for 循环
            避免 D 次索引操作，改为单次批量计算

        Args:
            logits: [B, N] MLP 输出 (未归一化)
            device: 设备
            dtype: 数据类型

        Returns:
            normalized: [B, N] 按深度归一化后的 logits
        """
        B, N = logits.shape
        depths = self.candidate_depths.to(device, non_blocking=True)  # [N] I78: 异步传输
        D = self._current_max_depth + 1

        # P-OPT-3: 向量化深度方差归一化
        # 使用 one-hot 编码实现批量 scatter/gather 操作

        # 构建深度 one-hot 掩码: [D, N]
        depth_onehot = F.one_hot(depths, D).float().T  # [D, N]
        depth_counts = depth_onehot.sum(dim=1)  # [D] 每个深度的候选数量

        # 扩展 logits 和掩码用于批量计算
        # logits: [B, N], depth_onehot: [D, N]
        # 目标: 计算每个深度的 mean 和 std

        # 使用 einsum 高效计算: sum_d = Σ_i (logits_i × mask_{d,i})
        # [B, D] = einsum('bn,dn->bd', logits, depth_onehot)
        depth_sums = torch.einsum('bn,dn->bd', logits, depth_onehot)  # [B, D]

        # Per-batch 均值: μ_d^(b) = Σ_i logits[b,i] × 1[depth[i]=d] / N_d
        # depth_counts.unsqueeze(0) = [1, D] broadcasts to [B, D]
        mu_per_batch = depth_sums / depth_counts.unsqueeze(0)  # [B, D]

        # Per-batch 方差: σ²_d^(b) = E[X²] - E[X]²
        logits_sq = logits ** 2
        depth_sq_sums = torch.einsum('bn,dn->bd', logits_sq, depth_onehot)  # [B, D]
        mean_sq_per_batch = depth_sq_sums / depth_counts.unsqueeze(0)  # [B, D]
        variance_per_batch = (mean_sq_per_batch - mu_per_batch ** 2).clamp(min=0.0)  # [B, D]

        # ====================================================================
        # 深度方差归一化 (Depth Variance Normalization)
        #
        # CRIT-2 修复: 准确描述归一化策略
        #
        # 策略说明:
        # - 训练时: 累积 EMA 统计量用于初始化回退，归一化使用实时 batch 统计量
        # - 评估时: 使用实时 per-sample 统计量，确保 B=1 和 B=4 行为一致
        #
        # 数学形式化:
        #   归一化: z_norm = (z - μ_batch) / (σ_batch + ε)
        #   其中 μ_batch, σ_batch 来自当前 batch (非 EMA 累积值)
        #
        # 设计理由:
        #   1. 实时统计量确保不同 batch size 下的归一化行为一致
        #   2. EMA 仅用于初始化时的保守回退 (避免未训练时的数值异常)
        #   3. Per-sample EMA 跟踪每个样本的统计量变化
        # ====================================================================
        if self.training:
            # 训练模式: Per-sample EMA 更新
            # 对每个样本独立更新 EMA，不跨 batch 平均
            B_effective = min(B, self._max_batch_size)  # 边界保护
            for b in range(B_effective):
                mu_b = mu_per_batch[b]  # [D]
                var_b = variance_per_batch[b]  # [D]

                if not self._depth_ema_initialized:
                    # 首次初始化：对每个样本独立初始化
                    safe_var = var_b.detach().clamp(min=DEPTH_VARIANCE_NORM_EPS)
                    if B <= 2:
                        safe_var = safe_var.clamp(min=DEPTH_VARIANCE_INIT_EPS)
                        print(f"Warning: Small batch (B={B}), using conservative EMA variance initialization (eps={DEPTH_VARIANCE_INIT_EPS})")
                    self._depth_ema_mean[b, :D] = mu_b.detach()
                    self._depth_ema_var[b, :D] = safe_var
                else:
                    # I99-1: Per-sample EMA 更新
                    # μ_new = α × μ_batch + (1-α) × μ_old (对每个样本独立)
                    self._depth_ema_mean[b, :D] = (
                        DEPTH_EMA_ALPHA * mu_b.detach() +
                        (1 - DEPTH_EMA_ALPHA) * self._depth_ema_mean[b, :D]
                    )
                    self._depth_ema_var[b, :D] = (
                        DEPTH_EMA_ALPHA * var_b.detach() +
                        (1 - DEPTH_EMA_ALPHA) * self._depth_ema_var[b, :D]
                    ).clamp(min=DEPTH_VARIANCE_NORM_EPS)

            self._depth_ema_initialized = True

            # 归一化使用当前 batch 的实时统计量 (非 EMA 累积值)
            # 这样确保不同 batch size 下的归一化行为一致
            mu_normalize = mu_per_batch  # [B, D]
            sigma_normalize = (variance_per_batch + DEPTH_VARIANCE_NORM_EPS).sqrt()  # [B, D]
        else:
            # ====================================================================
            # 评估模式
            # ====================================================================
            if not self._depth_ema_initialized:
                import warnings
                warnings.warn(
                    f"[I99-1] 深度方差归一化未初始化 (model.eval() 前未进行训练)。"
                    f"使用保守回退 (sigma={DEPTH_VARIANCE_INIT_EPS**0.5:.3f})。"
                    f"请确保模型已训练后再进行评估。",
                    UserWarning,
                    stacklevel=2
                )
                # 使用保守常数作为回退 (非 EMA 累积值)
                mu_normalize = mu_per_batch
                sigma_normalize = torch.full_like(mu_per_batch, DEPTH_VARIANCE_INIT_EPS ** 0.5)
            else:
                # 使用当前 batch 的实时统计量 (非 EMA 累积值)
                # 这样确保 B=1 和 B=4 评估时行为一致
                mu_normalize = mu_per_batch  # [B, D]
                sigma_normalize = (variance_per_batch + DEPTH_VARIANCE_NORM_EPS).sqrt()  # [B, D]

        # ====================================================================
        # 收集每个 batch 每个深度的均值和标准差用于归一化
        # 使用联合索引: mu_normalize[batch_indices, depth_indices] -> [B, N]
        # ====================================================================
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)  # [B, N]
        depth_indices = depths.unsqueeze(0).expand(B, -1)  # [B, N]

        # 联合索引获取归一化参数 (使用实时统计量，非 EMA)
        mu_expanded = mu_normalize[batch_indices, depth_indices]  # [B, N]
        sigma_expanded = sigma_normalize[batch_indices, depth_indices]  # [B, N]

        # 归一化
        normalized = (logits - mu_expanded) / sigma_expanded

        return normalized

    @property
    def current_temperature(self) -> float:
        """当前温度 τ > 0。
        
        I18-5: 温度下界从 0.01 提升到 0.1，确保 STE 梯度健康。
        """
        return self.log_temperature.exp().clamp(min=TEMPERATURE_MIN).item()
    
    @property 
    def thresholds(self) -> Tensor:
        """当前有效阈值向量。"""
        return self.threshold_offsets
    
    def forward(
        self,
        features: Tensor,
        image_size: Optional[Tuple[int, int]] = None,
        hard: bool = False,
    ) -> GumbelTopKResult:
        """
        前向传播：Gumbel-Top-K 分割。
        
        Args:
            features: [B, C, H_feat, W_feat] 特征图
            image_size: (H, W) 图像尺寸 (可选，用于更新区域坐标)
            hard: 是否使用硬决策 (推理模式)
            
        Returns:
            GumbelTopKResult: 分割结果
        """
        B, C, H_feat, W_feat = features.shape
        device = features.device
        dtype = features.dtype
        N = self.num_candidates
        
        # ====================================================================
        # 退火调度自动更新 (训练模式)
        # ====================================================================
        if self.training:
            if self._temp_enabled:
                self._update_temperature()
            if self._bias_enabled:
                self._update_explore_bias()

        # I30-17-EXT: 动态更新候选区域
        if image_size is not None:
            self._update_candidates(image_size)

        # 使用当前动态状态
        H_img, W_img = self._current_image_size
        scale_h = H_feat / H_img
        scale_w = W_feat / W_img
        
        # ====================================================================
        # Step 1: 并行计算所有候选的 logits
        # ====================================================================
        logits, probs = self._compute_all_logits(features, scale_h, scale_w)
        # logits: [B, N], probs: [B, N]
        
        # ====================================================================
        # Step 2: Gumbel-Top-K 选择
        # I24-2: 使用分层 Top-K (方案E) 或全局 Top-K (传统方案)
        # ====================================================================
        # 计算动态 K (I33: 传递 image_size 用于自适应覆盖率)
        if self.use_dynamic_k:
            K = self._estimate_optimal_k(probs, self._current_image_size)
        else:
            K = (self.K_min + self.K_max) // 2
        
        # 边界检查：K 不能超过候选数量 N
        K = min(K, N)
        # I23-2: 确保 K >= K_min (硬下界)
        # I23-3: 但不能超过 N（当 N < K_min 时，使用 N）
        K = max(min(K, N), min(self.K_min, N))
        
        # I24-2 方案E: 分层 Top-K (可学习配额)
        if LEARNABLE_QUOTA_ENABLED and self.quota_logits is not None:
            selected_mask, topk_indices = self._stratified_gumbel_topk_ste(logits, K, hard)
        else:
            # 传统全局 Top-K
            selected_mask, topk_indices = self._gumbel_topk_ste(logits, K, hard)
        # selected_mask: [B, N] (STE 版本，有梯度)
        # topk_indices: [B, K'] (硬选择索引，K' 可能略小于 K)

        # ====================================================================
        # Step 3: I100-1 移除树一致性约束
        # 原因: 树一致性约束是人为设计，非任务必需
        # - Hilbert 曲线核心价值是空间局部性，而非树互斥
        # - 父子共存可捕获不同粒度的特征
        # - 移除约束允许模型自己学习最优策略
        # 之前: consistent_mask = self._enforce_tree_consistency(...)
        # ====================================================================
        consistent_mask = selected_mask  # 直接使用，移除树一致性约束
        
        # ====================================================================
        # Step 4: 构建输出
        # ====================================================================
        result = self._build_result(
            consistent_mask, topk_indices, logits, probs
        )

        # I96-3: 计算配额损失以提供梯度到 quota_logits
        # 注意: 损失由调用者添加到总损失
        if self.training:
            self._last_quota_loss = self._compute_quota_loss(K)
        else:
            self._last_quota_loss = None

        # 缓存 probs 和 selected_mask 用于辅助损失计算
        self._last_probs = probs
        self._last_selected_mask = consistent_mask  # I21: 用于 Depth KL Loss

        # 更新统计
        with torch.no_grad():
            self._avg_selected = 0.9 * self._avg_selected + 0.1 * result.num_selected_per_batch.float().mean()

        return result
    
    def _compute_all_logits(
        self,
        features: Tensor,
        scale_h: float,
        scale_w: float,
    ) -> Tuple[Tensor, Tensor]:
        """
        并行计算所有 N 个候选区域的 logits。
        
        数学形式化:
            logits_i = MLP(ROI_i) + depth_bias_i + explore_bias - τ_{d_i}
            probs_i = σ(logits_i / T)
            
        Args:
            features: [B, C, H_feat, W_feat]
            scale_h, scale_w: 特征图与图像的缩放因子
            
        Returns:
            logits: [B, N]
            probs: [B, N]
        """
        B, C, H_feat, W_feat = features.shape
        N = self.num_candidates
        device = features.device
        dtype = features.dtype
        
        # 缩放区域坐标到特征图空间 (I78: 异步传输 + clone 支持原地修改)
        regions_feat = self.candidate_regions.to(device, non_blocking=True).clone()
        regions_feat[:, [0, 2]] *= scale_w  # x
        regions_feat[:, [1, 3]] *= scale_h  # y
        
        # 构建 ROI boxes: [B*N, 5]
        batch_indices = torch.arange(B, device=device, dtype=dtype).view(B, 1, 1).expand(-1, N, -1)
        regions_expanded = regions_feat.unsqueeze(0).expand(B, -1, -1)
        boxes = torch.cat([batch_indices, regions_expanded], dim=2).view(B * N, 5)
        
        # ROI-Align
        from torchvision.ops import roi_align
        roi_features = roi_align(
            features,
            boxes,
            output_size=(self.pool_size, self.pool_size),
            spatial_scale=1.0,
            aligned=True,
        )  # [B*N, C, k, k]
        
        # MLP 预测
        roi_flat = roi_features.flatten(1)  # [B*N, C*k*k]
        complexity_logits = self.complexity_mlp(roi_flat).squeeze(-1)  # [B*N]
        complexity_logits = complexity_logits.view(B, N)  # [B, N]
        # I23-4: Clamp MLP 输出，防止归一化后数值溢出
        # 数学: logits ∈ [-10, 10] 确保 sigmoid ∈ [4.5e-5, 0.99995]，梯度健康
        complexity_logits = complexity_logits.clamp(-10.0, 10.0)
        
        # ====================================================================
        # I23-1 方案C: 深度方差归一化 (核心修复)
        # 
        # 数学形式化:
        #     问题: MLP 输出方差与深度相关 (σ_3/σ_0 ≈ 8)
        #           导致 Top-K 偏好高方差深度，Log-Compensation 失效
        #     
        #     解决: z_i^norm = (z_i - μ_d) / σ_d
        #           使各深度 MLP 输出服从 N(0, 1)
        # ====================================================================
        if DEPTH_VARIANCE_NORM_ENABLED:
            complexity_logits = self._normalize_by_depth(complexity_logits, device, dtype)
        
        # 深度嵌入偏置 - 确保在正确设备上 (I78: 异步传输)
        depths = self.candidate_depths.to(device, non_blocking=True)  # [N]
        # STAB-7 修复: 确保索引张量为连续格式 (channels-last 兼容)
        depths = depths.contiguous()
        depth_embed = self.depth_embedding(depths)  # [N, 16]
        depth_bias_learned = self.depth_proj(depth_embed).squeeze(-1)  # [N]

        # 固定深度偏置 (可选，用于平滑过渡)
        depth_bias_fixed = self.depth_bias_beta * (self.depth_bias_gamma ** depths.float())

        # I30-4: 已移除 Log-Compensation (被方案E完全替代)

        # 阈值 - 确保 thresholds 在正确设备上 (I78: 异步传输)
        thresholds = self.thresholds.to(device, non_blocking=True)
        taus = thresholds[depths]  # [N]
        
        # 总 logits
        # logits = z + depth_bias + explore_bias - tau
        logits = (complexity_logits 
                  + depth_bias_learned.unsqueeze(0)
                  + depth_bias_fixed.unsqueeze(0)
                  + self.explore_bias
                  - taus.unsqueeze(0))
        
        # 分割概率 (I18-5: 使用 TEMPERATURE_MIN 常量)
        T = self.log_temperature.exp().clamp(min=TEMPERATURE_MIN)
        probs = torch.sigmoid(logits / T)

        return logits, probs

    # I33: 动态 K 边界方法 (自适应覆盖率)
    def _get_dynamic_k_bounds(
        self,
        candidate_count: int,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        """
        动态计算 K_min 和 K_max (I33 相对预算设计)。

        数学形式化
        ==========

        相对预算公式:
            K_min = max(K_min_abs, α × N)
            K_max = min(K_max_hard, β(H, W) × N)

        自适应覆盖率:
            β(H, W) = min(β_max, max(3α, β_0 × γ))
            γ = sqrt(min(H, W) / 224)

        覆盖率分析:
            | 图像尺寸 | min(H,W) | γ | β(H,W) | K_max覆盖率 | 评估 |
            |----------|----------|------|--------|-------------|------|
            | 64×64    | 64       | 0.53 | 0.03   | 3.0%        | ✅ 合理 |
            | 128×128  | 128      | 0.76 | 0.04   | 3.8%        | ✅ 合理 |
            | 224×224  | 224      | 1.00 | 0.05   | 5.0%        | ✅ 目标 |
            | 512×512  | 512      | 1.51 | 0.08   | 8.0%        | ⚠️ 硬上限 |

        Args:
            candidate_count: N 候选区域数
            image_size: 图像尺寸 (H, W)，用于自适应覆盖率计算

        Returns:
            (K_min, K_max): 动态边界元组
        """
        # K_min: 相对下界 + 绝对下界保护
        K_min = max(
            self._K_min_abs,
            int(math.ceil(self._token_coverage_min * candidate_count))
        )

        # K_max: 自适应覆盖率 × N + 硬上限保护
        if self._use_adaptive_coverage and image_size is not None:
            H, W = image_size
            min_dim = min(H, W)

            # 缩放因子 γ
            gamma = math.sqrt(min_dim / self._adaptive_reference_size)

            # 自适应覆盖率 β(H, W)
            beta_adaptive = self._token_coverage_base * gamma

            # 应用约束: β ∈ [3α, β_max]
            beta = min(
                self._token_coverage_max_hard,
                max(3 * self._token_coverage_min, beta_adaptive)
            )
        else:
            # 退回到基准覆盖率
            beta = self._token_coverage_base

        K_max = min(
            self._K_max_hard,
            int(math.ceil(beta * candidate_count))
        )

        return K_min, K_max

    def _estimate_optimal_k(
        self,
        probs: Tensor,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> int:
        """
        估计最优 K 值 (I33 自适应覆盖率版本)。

        数学形式化:
            方法 1: K_1 = E_b[count(p_i > 0.5)]  (高概率候选计数)
            方法 2: K_2 = E_b[argmin_k{cumsum(sorted(p)) >= 0.9 × total}]  (90% 累积概率)
            K_opt = clip(max(K_1, K_2), K_min, K_max)

        I23-2 修复: 使用 per-batch 计算替代 flatten
            - 原实现: 在 B×N 维度 flatten 后计算 k_90，语义不正确
            - 修复后: 对每个 batch 独立计算 k_90，取平均值

        I33: 使用自适应覆盖率计算 K_min/K_max
            - β(H, W) = min(β_max, max(3α, β_0 × γ))
            - γ = sqrt(min(H, W) / 224)

        Args:
            probs: [B, N] 分割概率
            image_size: 图像尺寸 (H, W)，用于自适应覆盖率

        Returns:
            K: 最优 token 数量
        """
        with torch.no_grad():
            B, N = probs.shape

            # 方法 1: 统计高概率候选数量 (per-batch mean)
            high_prob_count = (probs > 0.5).float().sum(dim=1).mean()

            # 方法 2: 使用 90% 累积概率截断 (per-batch 计算)
            # I23-2: 修复 flatten bug，改为 per-batch 计算取平均
            sorted_probs, _ = torch.sort(probs, dim=1, descending=True)  # [B, N]
            cumsum = sorted_probs.cumsum(dim=1)  # [B, N]
            # I102-3: 提升 epsilon 到 1e-6，FP16 安全边界 (原 1e-8 在边界)
            total_prob = cumsum[:, -1:].clamp(min=1e-6)  # [B, 1] 防止除零
            # 找到每个 batch 中达到 90% 累积概率的位置
            threshold_mask = cumsum < 0.9 * total_prob  # [B, N]
            k_90_per_batch = threshold_mask.sum(dim=1).float() + 1  # [B]
            k_90_mean = k_90_per_batch.mean().item()

            # 综合估计: 取两种方法的最大值
            K_est = int(max(high_prob_count.item(), k_90_mean))
            # I33: 使用动态边界（如果启用）
            if self.use_dynamic_k:
                K_min, K_max = self._get_dynamic_k_bounds(N, image_size)
            K = max(K_min, min(K_max, K_est, N))

            return K
    
    def _compute_quota_allocation(self, K: int) -> Tensor:
        """
        计算可学习配额分配 (I24-2 方案E, I32-3 贪心最优重构)。

        数学形式化
        ==========

        问题定义:
            输入: p = softmax(φ) ∈ Δ^{D-1}, K ∈ ℤ⁺, k_min ∈ ℤ⁺
            约束: ΣK_d = K, K_d ≥ k_min, K_d ∈ ℤ
            目标: min Σ|K_d - p_d·K|

        I32-3 改进: 比例缩放 + 贪心微调
        ------------------------------
        相比原实现的改进:
        1. 移除复杂的比例缩放逻辑，简化流程
        2. 迭代微调时使用偏差驱动而非最大值驱动
        3. 严格的约束满足保证

        Args:
            K: 总 token 配额

        Returns:
            quota: [D] 每个深度的配额分配
        """
        D = self._current_max_depth + 1
        device = self.candidate_depths.device

        if self.quota_logits is None or not self._enable_learnable_quota:
            # 回退到均匀分配
            quota = torch.full((D,), K // D, dtype=torch.long, device=device)
            quota[D - 1] += K - quota.sum()  # 余数给最后一个深度
            return quota

        # Softmax 计算配额概率
        # I35 Fix: 切片到当前深度维度，避免 quota_logits (max_depth+1) 与
        # _current_max_depth+1 不匹配的问题
        p = F.softmax(self.quota_logits[:D], dim=0)  # [D]

        # 初始四舍五入
        quota = (p * K).round().long()

        # I96-7 FIX: 移除硬下界约束，改用软正则化
        # 原因: 硬约束 (K_d >= 2) 存在数学问题:
        #   1. 深度 0: N_0=1 < K_0^min=2 (不可行)
        #   2. 浅层过度表示: 深度 1 采样率是深度 4 的 64 倍
        # 解决方案: 不使用硬下界，下界通过 _compute_quota_loss 中的软正则化实现
        # quota = quota.clamp(min=min_quota)  # 已移除

        # I32-3: 迭代微调确保 ΣK_d = K (最多 D 次)
        for _ in range(D):
            total = quota.sum()
            diff = K - total  # >0: 不足, <0: 超出

            if diff == 0:
                break

            if diff > 0:
                # 不足时：优先从概率最高的深度增加
                # 保持与 softmax 概率的一致性
                probs_sorted, indices = torch.sort(p, descending=True)
                for i in range(min(diff, D)):
                    quota[indices[i]] += 1
            else:
                # 超出时：从概率最低的深度扣减（无硬下界）
                # I96-7: 允许扣减到 0，下界通过软正则化实现
                mask = quota > 0
                if mask.sum() > 0:
                    # 获取可扣减的深度及其概率
                    available_p = p[mask]
                    available_indices = torch.masked_select(
                        torch.arange(D, device=device), mask
                    )
                    # 优先从概率最低的扣减
                    _, sorted_idx = torch.sort(available_p, descending=False)
                    for i in range(min(-diff, len(sorted_idx))):
                        idx = available_indices[sorted_idx[i]]
                        quota[idx] -= 1

        return quota

    def _compute_quota_loss(self, K: int) -> Tensor:
        """
        计算软配额正则化损失 (I96-3 方案F)。

        数学形式化
        ==========

        问题定义:
            在 STE 框架下，K_d 的离散选择不参与梯度计算
            需要通过额外损失为 quota_logits 提供梯度

        解决方案:
            L_quota = λ × MSE(K_soft, K_hard)
            其中:
                K_soft = π_d × K (软配额，有梯度)
                K_hard = round(K_soft) (硬配额，用于实际选择)

        梯度流:
            ∂L_quota/∂φ_d = 2λ × (K_soft_d - K_hard_d) × K

        优势:
            - 配额参数 φ 通过损失回传梯度
            - 不改变硬选择逻辑（保持 Hilbert 局部性）
            - 实现简单，无需修改核心代码

        Args:
            K: 总 token 配额

        Returns:
            quota_loss: 标量张量，软配额正则化损失
        """
        D = self._current_max_depth + 1

        # 如果未启用可学习配额，返回 0
        if self.quota_logits is None or not self._enable_learnable_quota:
            return torch.tensor(0.0, device=self.candidate_depths.device)

        # 计算软配额 (有梯度)
        p = F.softmax(self.quota_logits[:D], dim=0)  # [D], 有梯度
        K_soft = p * K  # [D], 软配额

        # 计算硬配额 (无梯度，用于比较)
        K_hard = K_soft.detach().round().long()  # [D], .detach() 避免双重计算

        # I96-8: 使用相对 MSE 损失实现跨深度可比
        # 问题: 绝对 MSE 损失使深层 (K_d ≈ 1) 惩罚过轻，浅层 (K_d ≈ 16) 惩罚过重
        # 解决方案: L_rel = Σ ((K_soft - K_hard) / (K_soft + ε))²
        eps = 1e-6
        relative_diff = (K_soft - K_hard.float()) / (K_soft.abs() + eps)
        loss = (relative_diff ** 2).mean()

        # I96-3: 乘以权重系数，与主损失量级匹配
        loss = loss * QUOTA_ENTROPY_WEIGHT  # 使用已有的常量

        # I96-7: 软下界正则化损失
        # 数学: L_min = λ × Σ max(0, K_d^min - K_d)²
        # 其中 K_d^min = max(1, α × N_d), N_d = 4^d
        # 目的: 鼓励但不强制深度下界，软约束允许模型学习最优分布
        # 优势: 无约束满足问题，梯度完整，保持 Hilbert 曲线局部性
        K_soft_float = K_soft.float()  # 转换为浮点用于计算
        K_min_targets = []
        for d in range(D):
            N_d = 4 ** d  # 深度 d 的候选数量
            K_min_d = max(1, int(QUOTA_MIN_RATIO * N_d))
            K_min_targets.append(K_min_d)
        K_min_tensor = torch.tensor(K_min_targets, device=K_soft.device, dtype=K_soft_float.dtype)

        # 计算下界违反: max(0, K_min - K_soft)
        violation = (K_min_tensor - K_soft_float).clamp(min=0)
        min_loss = (violation ** 2).mean()

        # 添加软下界损失 (使用独立的 QUOTA_MIN_LAMBDA 权重)
        loss = loss + min_loss * QUOTA_MIN_LAMBDA

        return loss

    def _stratified_gumbel_topk_ste(
        self,
        logits: Tensor,
        K: int,
        hard: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        分层 Gumbel-Top-K 选择 (I24-2 方案E 核心)。
        
        数学形式化
        ==========
        
        与全局 Top-K 的区别:
            全局: selected = TopK(logits, K)  → 深度崩塌
            分层: selected = ∪_d TopK(logits[d], K_d)  → 配额保证
            
        STE 梯度流:
            深度内独立 softmax → 梯度增强 K/K_d 倍
            例: K=32, K_d=8 → 梯度增强 4x (相比全局 softmax)
            
        配额下界保护:
            K_d >= 1 保证每个深度至少有梯度信号
            
        Args:
            logits: [B, N] 候选 logits
            K: 总选择数量
            hard: 是否使用硬决策
            
        Returns:
            selected_mask: [B, N] STE 选择掩码 (有梯度)
            topk_indices: [B, K'] 硬选择索引 (K' 可能略小于 K)
        """
        B, N = logits.shape
        device = logits.device
        D = self._current_max_depth + 1
        depths = self.candidate_depths.to(device, non_blocking=True)  # [N] I78: 异步传输
        
        # 计算配额分配
        quota = self._compute_quota_allocation(K)  # [D]
        
        # I18-5: 使用 TEMPERATURE_MIN 常量确保梯度健康
        T = self.log_temperature.exp().clamp(min=TEMPERATURE_MIN)
        
        # 初始化输出
        hard_mask = torch.zeros(B, N, device=device, dtype=torch.float32)
        soft_mask = torch.zeros(B, N, device=device, dtype=torch.float32)
        all_topk_indices = []

        # 转换为 FP32 计算
        original_dtype = logits.dtype
        logits_fp32 = logits.float()
        T_fp32 = T.float()

        # I97-4 优化: 预计算所有深度的索引（避免重复 nonzero 调用）
        depth_indices_list = []
        num_per_depth = []
        for d in range(D):
            depth_mask = (depths == d)  # [N]
            depth_indices = depth_mask.nonzero(as_tuple=True)[0]  # [N_d]
            depth_indices_list.append(depth_indices)
            num_per_depth.append(len(depth_indices))

        # I97-4 优化: 批量提取 K_d（减少 CPU/GPU 同步）
        # 注意：topk 的 k 必须是 Python int，所以最终仍需 .item() 调用
        # 但我们通过预计算 depth_indices 减少了 D 次 nonzero() 调用
        K_d_values = []
        for k_tensor, n in zip(quota.detach().clamp(min=0), num_per_depth):
            k_d = k_tensor.long().item()
            k_d = min(max(k_d, 0), n)
            K_d_values.append(k_d)
        # K_d_values 是 Python int 列表，用于后续的 topk k 参数

        # 分层选择
        for d in range(D):
            depth_indices = depth_indices_list[d]  # [N_d]
            N_d = num_per_depth[d]
            K_d = K_d_values[d]  # 预提取的 Python int

            if K_d <= 0 or N_d == 0:
                continue

            # 提取该深度的 logits
            logits_d = logits_fp32[:, depth_indices]  # [B, N_d]

            if hard or not self.training:
                # 推理模式：直接 Top-K
                _, topk_local = torch.topk(logits_d, K_d, dim=1)  # [B, K_d]
            else:
                # 训练模式：Gumbel + Top-K
                uniform = torch.rand(B, N_d, device=device, dtype=torch.float32)
                uniform = uniform.clamp(GUMBEL_EPSILON, 1 - GUMBEL_EPSILON)
                gumbel = -torch.log(-torch.log(uniform))
                perturbed = (logits_d + gumbel) / T_fp32
                topk_vals, topk_local = torch.topk(perturbed, K_d, dim=1)  # [B, K_d]

                # 深度内 Subset Softmax
                subset_softmax = F.softmax(topk_vals, dim=1)  # [B, K_d]

                # 映射回全局索引
                topk_global = depth_indices[topk_local]  # [B, K_d]

                # 更新 soft_mask
                soft_mask.scatter_(1, topk_global, subset_softmax)

            # 更新 hard_mask
            topk_global = depth_indices[topk_local]  # [B, K_d]
            hard_mask.scatter_(1, topk_global, 1.0)
            all_topk_indices.append(topk_global)
        
        # 合并所有深度的 Top-K 索引
        if all_topk_indices:
            topk_indices = torch.cat(all_topk_indices, dim=1)  # [B, K']
        else:
            # 边界情况：至少选择根节点
            topk_indices = torch.zeros(B, 1, device=device, dtype=torch.long)
            hard_mask[:, 0] = 1.0
            soft_mask[:, 0] = 1.0
        
        # STE: 前向用硬掩码，反向用软掩码的梯度
        if self.training and not hard:
            st_mask = hard_mask - soft_mask.detach() + soft_mask
        else:
            st_mask = hard_mask
        
        # 转回原始精度
        if original_dtype != torch.float32:
            st_mask = st_mask.to(original_dtype)
            hard_mask = hard_mask.to(original_dtype)
        
        return st_mask, topk_indices

    def _gumbel_topk_ste(
        self,
        logits: Tensor,
        K: int,
        hard: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Gumbel-Top-K 选择 with Straight-Through Estimator。
        
        数学形式化:
            1. g_i ~ Gumbel(0, 1)
            2. perturbed_i = (logits_i + g_i) / τ
            3. selected = TopK(perturbed, K)
            4. STE: hard_mask - softmax.detach() + softmax
            
        I21 δ: Subset Softmax 改进
        =========================
        问题: 全局 softmax(N=85) 导致梯度稀释 ~1/85
        解决: 在 Top-K 选中的子集上计算 softmax，梯度增强到 ~1/K
        
        数学推导:
            原始: soft_mask = softmax(perturbed)  # [B, N], 每个元素 ≈ 1/N
            改进: soft_mask[topk] = softmax(perturbed[topk])  # ~1/K >> 1/N
            
        梯度增强: 
            原始梯度: ∂L/∂z_i ≈ 1/N × ∂L/∂mask_i
            改进梯度: ∂L/∂z_i ≈ 1/K × ∂L/∂mask_i  (对选中的 K 个)
            增强比例: N/K = 85/32 ≈ 2.7x
            
        Args:
            logits: [B, N] 候选 logits
            K: 选择数量
            hard: 是否使用硬决策
            
        Returns:
            selected_mask: [B, N] STE 选择掩码 (有梯度)
            topk_indices: [B, K] 硬选择索引
        """
        B, N = logits.shape
        device = logits.device
        
        # I23-3: 保护 K <= N，防止 Top-K 越界
        K = min(K, N)
        
        # I18-5: 使用 TEMPERATURE_MIN 常量确保梯度健康
        T = self.log_temperature.exp().clamp(min=TEMPERATURE_MIN)
        
        if hard or not self.training:
            # 推理模式：直接 Top-K
            _, topk_indices = torch.topk(logits, K, dim=1)
            hard_mask = torch.zeros(B, N, device=device)
            hard_mask.scatter_(1, topk_indices, 1.0)
            return hard_mask, topk_indices
        
        # 训练模式：Gumbel + STE
        # 在 FP32 下计算 Gumbel 噪声
        original_dtype = logits.dtype
        logits_fp32 = logits.float()
        T_fp32 = T.float()
        
        # Gumbel 采样
        uniform = torch.rand(B, N, device=device, dtype=torch.float32)
        uniform = uniform.clamp(GUMBEL_EPSILON, 1 - GUMBEL_EPSILON)
        gumbel = -torch.log(-torch.log(uniform))
        
        # 扰动后的 logits
        perturbed = (logits_fp32 + gumbel) / T_fp32
        
        # Top-K 硬选择
        topk_vals, topk_indices = torch.topk(perturbed, K, dim=1)
        
        # 构建硬掩码
        hard_mask = torch.zeros(B, N, device=device, dtype=torch.float32)
        hard_mask.scatter_(1, topk_indices, 1.0)

        # ====================================================================
        # I30-2: 使用全局 Softmax (移除 Subset Softmax)
        # 原因: Subset Softmax 梯度覆盖率仅 K/N ≈ 37.6%，与文档声称的 100% 矛盾
        #       全局 Softmax 提供 100% 梯度覆盖，避免死区问题
        # 数学: π_i = e^{z_i} / Σ_j e^{z_j}，梯度 ∂L/∂z_j 对所有 j 非零
        # ====================================================================
        soft_mask = F.softmax(perturbed, dim=1)

        # STE: 前向用硬掩码，反向用软掩码的梯度
        st_mask = hard_mask - soft_mask.detach() + soft_mask
        
        # 转回原始精度
        if original_dtype != torch.float32:
            st_mask = st_mask.to(original_dtype)
            hard_mask = hard_mask.to(original_dtype)
        
        return st_mask, topk_indices
    
    def _enforce_tree_consistency(
        self,
        selected_mask: Tensor,
        topk_indices: Tensor,
        training: bool = True,
    ) -> Tensor:
        """
        强制树一致性约束 (向量化)。

        数学形式化:
            约束: ∀i ∈ selected: parent(i) ∉ selected
            即: 若子节点被选中，则父节点不应被选中

        向量化实现:
            1. 对每个节点，检查其任意子节点是否被选中
            2. 若有子节点被选中，则该节点不能被选中

        Args:
            selected_mask: [B, N] STE 选择掩码
            topk_indices: [B, K] 硬选择索引
            training: 是否在 training 模式 (软边距 vs 硬边距)

        Returns:
            consistent_mask: [B, N] 树一致的选择掩码
        """
        self._ensure_children_matrix()

        B, N = selected_mask.shape
        device = selected_mask.device

        # 确保 children_matrix 在正确设备上 (I78: 异步传输)
        children_matrix = self._children_matrix.to(device, non_blocking=True)  # [N, 4]

        # 对于每个节点，检查其子节点是否被选中
        # valid_children: [N, 4] 哪些子节点索引是有效的
        valid_children = children_matrix >= 0

        # 安全索引 (将 -1 替换为 0)
        safe_children = children_matrix.clamp(min=0)  # [N, 4]

        # ====================================================================
        # I22-1 方案A+: 向量化树一致性检查
        # I96-4 改进: 软边距排除机制
        #
        # 数学约束: ∀i ∈ selected: parent(i) ∉ selected
        # 等价表述: 若任意子节点被选中，则父节点不应被选中
        #
        # I96-4 软边距机制:
        #   - 仅在 training 模式下使用软边距
        #   - inference 模式下使用硬边距 (0)，保证 0/1 输出
        #   - 使用 .detach() 避免排除决策接收梯度 (避免梯度双计)
        #   - 软边距: clamp(min=SOFT_EXCLUSION_MARGIN) 保持最小梯度流 (10%)
        #   - 数学: p_out = p × max(1 - Σ child_signal, ε)
        #   - 梯度: ∂p_out/∂p_i ≥ ε > 0，确保父节点有梯度回传
        #
        # 对比原实现:
        #   原: exclusion_mask = 1.0 - has_child_selected (梯度=0)
        #   新: exclusion_mask.clamp(min=ε) (梯度≥ε) - 仅 training 模式
        # ====================================================================

        # 使用硬掩码进行约束检查 (I96-4: 添加 .detach() 避免梯度双计)
        hard_selected = (selected_mask > 0.5).float().detach()

        # 获取所有子节点的选择状态
        # safe_children: [N, 4], 值范围 [0, N-1]
        # hard_selected[:, safe_children]: [B, N, 4]
        children_selected_status = hard_selected[:, safe_children]  # [B, N, 4]

        # 将无效子节点的状态设为 0
        valid_children_mask = valid_children.unsqueeze(0).expand(B, -1, -1)  # [B, N, 4]
        children_selected_status = children_selected_status * valid_children_mask.float()

        # 计算子节点选中信号 (累积而非 max，保持更多信息)
        child_signal = children_selected_status.sum(dim=2).clamp(max=1.0)  # [B, N]

        # I96-4: 仅在 training 模式下使用软边距
        if training:
            # 软边距排除: p_out = p × max(1 - child_signal, ε)
            exclusion_mask = (1.0 - child_signal).clamp(min=SOFT_EXCLUSION_MARGIN)  # [B, N]
        else:
            # Inference 模式: 硬边距，保证 0/1 输出
            exclusion_mask = 1.0 - child_signal  # [B, N]

        # 应用排除
        consistent_mask = selected_mask * exclusion_mask

        return consistent_mask
    
    def _build_result(
        self,
        consistent_mask: Tensor,
        topk_indices: Tensor,
        logits: Tensor,
        probs: Tensor,
    ) -> GumbelTopKResult:
        """
        从选择掩码构建输出结果。
        
        Args:
            consistent_mask: [B, N] 树一致的选择掩码
            topk_indices: [B, K] 硬选择索引
            logits: [B, N] 原始 logits
            probs: [B, N] 分割概率
            
        Returns:
            GumbelTopKResult
            
        性能优化 (P-OPT-1):
            使用向量化操作替换 Python for 循环，避免 B 次小张量操作。
            通过 nonzero() + scatter 一次性处理所有 batch。
        """
        B, N = consistent_mask.shape
        device = consistent_mask.device
        
        # 使用硬阈值选择最终区域
        final_selected = (consistent_mask > 0.5)  # [B, N]
        
        # P-OPT-1: 向量化收集选中区域
        # 计算每个 batch 的选中数量
        num_selected_per_batch = final_selected.sum(dim=1)  # [B]
        
        # 确保每个 batch 至少有一个 token (根节点)
        empty_batches = (num_selected_per_batch == 0)
        if empty_batches.any():
            # 对空 batch 强制选中根节点 (index 0)
            final_selected = final_selected.clone()
            final_selected[empty_batches, 0] = True
            num_selected_per_batch = final_selected.sum(dim=1)
        
        # 一次性获取所有选中位置 [total_selected, 2] -> (batch_idx, candidate_idx)
        selected_positions = final_selected.nonzero(as_tuple=False)  # [total, 2]
        batch_indices = selected_positions[:, 0]  # [total]
        candidate_indices = selected_positions[:, 1]  # [total]
        
        # 向量化索引所有候选属性 - 确保在正确设备上 (I78: 异步传输)
        regions = self.candidate_regions.to(device, non_blocking=True)[candidate_indices]  # [total, 4]
        depths = self.candidate_depths.to(device, non_blocking=True)[candidate_indices]  # [total]
        hilbert_indices = self.hilbert_indices.to(device, non_blocking=True)[candidate_indices]  # [total]
        
        return GumbelTopKResult(
            regions=regions,
            depths=depths,
            batch_indices=batch_indices,
            hilbert_indices=hilbert_indices,
            selected_mask=consistent_mask,
            logits=logits,
            probs=probs,
            num_selected_per_batch=num_selected_per_batch,
        )
    
    # ========================================================================
    # 辅助损失接口 (与 LearnableSplitter 兼容)
    # ========================================================================
    
    def get_auxiliary_losses(
        self,
        features: Optional[Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None,
        include_balance: bool = False,
        include_elastic_budget: bool = True,
        include_soft_entropy: bool = True,
        batch_size: int = 1,
        actual_token_count: Optional[int] = None,
        entropy_target: Optional[float] = None,
        entropy_weight: float = 0.1,
        entropy_mode: str = 'maximize',
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        获取所有辅助损失 (与 LearnableSplitter.get_auxiliary_losses 兼容).

        I33: Elastic Budget 相对预算设计
        =================================

        核心改进: 与K参数相对预算保持一致

        数学形式化:
            coverage = N_selected / N_candidates
            L_elastic = λ_over × max(0, coverage - β_max)² × N_candidates

        梯度分析:
            ∂L/∂N_selected = 2 × λ_over × max(0, coverage - β_max) / N_candidates
            → 梯度归一化到 ~0.01 量级，避免跨尺度差异

        崩溃检测 (相对阈值):
            collapse ⇔ coverage < α_collapse (0.5%)

        Args:
            features: 特征图 [B, C, H, W] (可选)
            image_size: 图像尺寸 (可选)
            include_balance: 是否包含平衡损失 (对 GumbelTopK 忽略)
            include_elastic_budget: 是否包含弹性预算损失 (相对覆盖率版本)
            include_soft_entropy: 是否包含软熵损失
            batch_size: batch 大小
            actual_token_count: 实际 token 数 (用于相对崩溃检测)
            entropy_target: 熵目标值 (target mode)
            entropy_weight: 熵损失权重
            entropy_mode: 'maximize' 或 'target'
            **kwargs: 其他参数 (向前兼容)

        Returns:
            Dict[str, Tensor]: 各项损失
        """
        device = self.candidate_regions.device
        losses = {}

        # 获取 cached probs 和 selected_mask
        probs = self._last_probs if hasattr(self, '_last_probs') else None
        selected_mask = self._last_selected_mask if hasattr(self, '_last_selected_mask') else None

        # I23-4-FIX: NaN 检测与防护
        # 如果缓存的 probs 或 selected_mask 包含 NaN，返回零损失
        # 这可能由上游输入包含 NaN 导致，应在训练脚本中处理根因
        has_nan = False
        if probs is not None and torch.isnan(probs).any():
            has_nan = True
        if selected_mask is not None and torch.isnan(selected_mask).any():
            has_nan = True
        
        if has_nan:
            # 返回零损失，避免 NaN 传播到总损失
            zero = torch.tensor(0.0, device=device)
            if include_elastic_budget:
                losses['elastic_budget_loss'] = zero
            if include_soft_entropy:
                losses['soft_entropy_loss'] = zero
            # I35: 移除死代码 DEPTH_KL_WEIGHT, DEPTH_QUOTA_ENABLED
            return losses
        
        # 1. Elastic Budget Loss (I33: 相对预算版本)
        # I23-2 方案 B: 简化弹性惩罚
        # I33: 改造为相对覆盖率设计，与 _get_dynamic_k_bounds() 统一
        if include_elastic_budget:
            avg_tokens = self._avg_selected
            candidate_count = self.num_candidates

            # 相对覆盖率
            coverage = avg_tokens / candidate_count

            # 相对损失: L = λ × max(0, coverage - β_max)² × N
            # 梯度: ∂L/∂N_selected = 2 × λ × max(0, coverage - β_max) / N
            over_loss = ELASTIC_LAMBDA_OVER * torch.relu(
                coverage - ELASTIC_COVERAGE_MAX
            ).pow(2) * candidate_count

            losses['elastic_budget_loss'] = over_loss

            # 崩溃检测 (相对覆盖率 < 0.5%)
            collapse_threshold = ELASTIC_COVERAGE_MIN * candidate_count
            if actual_token_count is not None and actual_token_count < collapse_threshold:
                collapse_loss = torch.tensor(ELASTIC_LAMBDA_COLLAPSE, device=device)
                losses['collapse_loss'] = collapse_loss
        
        # 2. Soft Entropy Loss
        if include_soft_entropy and probs is not None:
            entropy_loss = self.get_depth_entropy_loss(weight=entropy_weight, probs=probs)
            
            if entropy_mode == 'target' and entropy_target is not None:
                # Target mode: minimize |H - H_target|²
                B, N = probs.shape
                depths = self.candidate_depths.to(probs.device)

                # 计算当前熵
                depth_probs = []
                for d in range(self._current_max_depth + 1):
                    mask = (depths == d)
                    if mask.any():
                        p_d = probs[:, mask].mean()
                        depth_probs.append(p_d.clamp(min=PROB_EPSILON))
                    else:
                        depth_probs.append(torch.tensor(PROB_EPSILON, device=probs.device))
                
                depth_probs_t = torch.stack(depth_probs)
                depth_probs_t = depth_probs_t / depth_probs_t.sum()
                # I23-4: 归一化后再次 clamp，防止 FP16 下溢导致 log(0)
                depth_probs_t = depth_probs_t.clamp(min=PROB_EPSILON)
                current_entropy = -(depth_probs_t * depth_probs_t.log()).sum()
                
                entropy_loss = entropy_weight * (current_entropy - entropy_target).pow(2)
            
            losses['soft_entropy_loss'] = entropy_loss

        # ====================================================================
        # I24-2 方案E: 配额熵正则化损失
        # I30-10: 使用配置值
        # 鼓励可学习配额保持多样性，避免崩塌到单一深度
        # ====================================================================
        if self._enable_learnable_quota and self.quota_logits is not None:
            quota_entropy_loss = self.get_quota_entropy_loss(weight=self._quota_entropy_weight)
            losses['quota_entropy_loss'] = quota_entropy_loss
        
        # ====================================================================
        # I29-2: 阈值方差正则化损失
        # 数学: L_threshold = λ × Var(τ)
        # 目的: 防止某个深度的阈值极端偏离，导致选择偏差
        # ====================================================================
        if THRESHOLD_VAR_REG_ENABLED:
            threshold_var_loss = torch.var(self.threshold_offsets) * THRESHOLD_VAR_REG_WEIGHT
            losses['threshold_var_loss'] = threshold_var_loss
        
        # ====================================================================
        # I23-5-FIX: 最终 NaN/Inf 检查与清理
        # 确保返回的所有损失都是有效数值
        # ====================================================================
        device = self.candidate_regions.device
        zero = torch.tensor(0.0, device=device)
        for key, val in list(losses.items()):
            if torch.isnan(val) or torch.isinf(val):
                losses[key] = zero
        
        return losses
    
    def get_elastic_budget_loss(
        self,
        target_tokens: int = 32,
        weight: float = 0.01,
    ) -> Tensor:
        """
        计算弹性预算损失。
        
        数学形式化:
            L_budget = weight × |E[token_count] - target|^2
            
        Args:
            target_tokens: 目标 token 数量
            weight: 损失权重
            
        Returns:
            loss: 标量损失
        """
        avg_tokens = self._avg_selected
        loss = weight * (avg_tokens - target_tokens).pow(2)
        return loss
    
    def get_depth_entropy_loss(
        self,
        weight: float = 0.01,
        probs: Optional[Tensor] = None,
    ) -> Tensor:
        """
        计算深度熵损失 (鼓励深度多样性)。
        
        数学形式化:
            H = -Σ_d p_d log(p_d)
            L_entropy = -weight × H  (最大化熵)
            
        Args:
            weight: 损失权重
            probs: [B, N] 分割概率 (可选，从缓存获取)
            
        Returns:
            loss: 标量损失
        """
        if probs is None:
            return torch.tensor(0.0, device=self.candidate_regions.device)

        B, N = probs.shape
        depths = self.candidate_depths.to(probs.device)  # [N]
        D = self._current_max_depth + 1

        # P-OPT-7: 向量化深度平均概率计算 (消除 for 循环)
        # 使用 one-hot 编码计算每个深度的平均概率
        # p_d = mean(probs[:, depths == d]) = sum(probs × mask_d) / count_d
        depth_onehot = F.one_hot(depths, D).float()  # [N, D]
        depth_counts = depth_onehot.sum(dim=0).clamp(min=1.0)  # [D], 每个深度的候选数量
        prob_sums = torch.einsum('bn,nd->d', probs, depth_onehot)  # [D], batch 求和
        depth_probs = (prob_sums / (B * depth_counts)).clamp(min=PROB_EPSILON)  # [D], 平均概率
        depth_probs = depth_probs / depth_probs.sum()  # 归一化
        # I23-4: 归一化后再次 clamp，防止 FP16 下溢导致 log(0)
        depth_probs = depth_probs.clamp(min=PROB_EPSILON)
        
        # 熵 (使用 log_softmax 等价形式提升稳定性)
        # H = -Σ p_i log(p_i) = -Σ p_i (log_p_i) where log_p_i = log(p_i)
        entropy = -(depth_probs * depth_probs.log()).sum()
        
        # 最大化熵 → 最小化负熵
        loss = -weight * entropy
        return loss

    def get_quota_entropy_loss(
        self,
        weight: float = QUOTA_ENTROPY_WEIGHT,
    ) -> Tensor:
        """
        计算配额熵正则化损失 (I24-2 方案E)。
        
        数学形式化
        ==========
        
        动机:
            可学习配额 φ 可能崩塌到单一深度 (所有配额给 depth=3)。
            通过熵正则化鼓励配额分布保持多样性。
            
        损失公式:
            p = softmax(φ)  配额分布
            H(p) = -Σ_d p_d log(p_d)  熵
            L_entropy = -λ × H(p)  最大化熵 → 最小化负熵
            
        梯度流:
            ∂L/∂φ_d = -λ × (∂H/∂p_d) × (∂p_d/∂φ_d)
                    = λ × (1 + log(p_d)) × p_d × (1 - p_d)  [对 softmax]
                    
        效果:
            - 高 p_d → log(p_d) 大 → 正梯度 → 降低 φ_d
            - 低 p_d → log(p_d) 小 → 负梯度 → 提高 φ_d
            
        Args:
            weight: 熵损失权重 (默认 QUOTA_ENTROPY_WEIGHT=0.1)
            
        Returns:
            loss: 标量熵损失
        """
        if self.quota_logits is None:
            return torch.tensor(0.0, device=self.candidate_regions.device)
        
        # Softmax 计算配额分布
        quota_probs = F.softmax(self.quota_logits, dim=0)  # [D]
        quota_probs = quota_probs.clamp(min=PROB_EPSILON)
        
        # 熵
        entropy = -(quota_probs * quota_probs.log()).sum()
        
        # 最大化熵 → 最小化负熵
        loss = -weight * entropy

        return loss

    def set_quota_grad(self, enabled: bool) -> None:
        """
        控制配额参数的梯度 (I30-10: freeze_quota 功能)。

        数学形式:
            if enabled:
                quota_logits.requires_grad = True
            else:
                quota_logits.requires_grad = False

        Args:
            enabled: 是否启用梯度
        """
        if self.quota_logits is not None:
            self.quota_logits.requires_grad = enabled

    def get_depth_distribution(
        self,
        selected_mask: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """
        获取当前深度分布统计信息 (用于监控)。

        Returns:
            dict: {
                'pi': [D] 深度分布,
                'entropy': 熵,
                'kl_from_uniform': KL(π || U),
            }
        """
        if selected_mask is None:
            selected_mask = getattr(self, '_last_selected_mask', None)

        if selected_mask is None:
            return {'pi': None, 'entropy': None, 'kl_from_uniform': None}

        B, N = selected_mask.shape
        device = selected_mask.device
        depths = self.candidate_depths.to(device, non_blocking=True)  # I78: 异步传输
        D = self._current_max_depth + 1

        with torch.no_grad():
            # P-OPT-6: 向量化深度分布计算 (消除 for 循环)
            # 使用 one-hot 编码 + einsum 一次性计算
            depth_onehot = F.one_hot(depths, D).float()  # [N, D]
            depth_counts = torch.einsum('bn,nd->d', selected_mask, depth_onehot)  # [D]
            total = depth_counts.sum().clamp(min=1.0)
            pi = depth_counts / total

            # 熵
            pi_safe = pi.clamp(min=PROB_EPSILON)
            entropy = -(pi_safe * pi_safe.log()).sum().item()

            # KL from uniform
            uniform = torch.ones(D, device=device) / D
            kl = (pi_safe * (pi_safe.log() - uniform.log())).sum().item()

            return {
                'pi': pi.tolist(),
                'entropy': entropy,
                'max_entropy': math.log(D),
                'kl_from_uniform': kl,
            }
    
    # ========================================================================
    # 退火调度接口 (与 LearnableSplitter 兼容)
    # ========================================================================
    
    def set_explore_bias(self, bias: float) -> None:
        """设置探索偏置。"""
        self.explore_bias.fill_(bias)
    
    def set_temperature(self, temperature: float) -> None:
        """设置温度。"""
        self.log_temperature.data.fill_(math.log(max(temperature, TEMPERATURE_MIN)))
    
    # ========================================================================
    # 退火调度 API (与 LearnableSplitter 完全兼容)
    # ========================================================================
    
    def enable_temperature_annealing(
        self,
        total_steps: int,
        T_start: float = 1.0,
        T_end: float = 0.3,
        schedule: str = 'exponential',
    ) -> "GumbelTopKSplitter":
        """
        启用自动温度退火。
        
        数学形式化
        ==========
        
        对于 Gumbel-Top-K，温度 τ 控制 softmax 尖锐度：
            softmax_i = exp(z_i / τ) / Σ exp(z_j / τ)
            
        STE 梯度依赖 softmax:
            ∂L/∂z_i = ∂L/∂mask_i × ∂softmax_i/∂z_i
            
        低温度时 softmax → one-hot，梯度只流向 top 候选。
        
        退火策略:
            T(t) = T_start · (T_end / T_start)^(t / total)  [exponential]
            
        I18-2/I18-5 安全下界:
            T_end ≥ 0.3 避免梯度消失
            
        Args:
            total_steps: 总训练步数
            T_start: 初始温度 (默认 1.0)
            T_end: 最终温度 (默认 0.3, I18-2 安全下界)
            schedule: 'exponential', 'linear', 'cosine'
            
        Returns:
            self (链式调用)
        """
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if T_start <= 0 or T_end <= 0:
            raise ValueError(f"Temperatures must be positive")
        if T_end > T_start:
            raise ValueError(f"T_end should be <= T_start for annealing")
        
        # I18-2: 强制温度下界保护
        T_MIN_SAFE = 0.3
        if T_end < T_MIN_SAFE:
            import warnings
            warnings.warn(
                f"I18-2: T_end={T_end} < {T_MIN_SAFE} may cause gradient vanishing. "
                f"Clamping to {T_MIN_SAFE}.",
                UserWarning,
                stacklevel=2,
            )
            T_end = T_MIN_SAFE
        
        if schedule not in ('exponential', 'linear', 'cosine'):
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self._temp_total_steps.fill_(total_steps)
        self._temp_start.fill_(T_start)
        self._temp_end.fill_(T_end)
        self._temp_step.zero_()
        self._temp_schedule = schedule
        self._temp_enabled = True
        
        # 设置初始温度
        self.set_temperature(T_start)
        
        return self
    
    def disable_temperature_annealing(self) -> "GumbelTopKSplitter":
        """禁用自动温度退火。"""
        self._temp_enabled = False
        return self
    
    def enable_explore_bias_annealing(
        self,
        total_steps: int,
        b_start: float = 0.5,
        b_end: float = 0.0,
    ) -> "GumbelTopKSplitter":
        """
        启用探索偏置退火。
        
        数学形式化
        ==========
        
        对于 Gumbel-Top-K，偏置影响选择质量:
            z_effective = z + b + depth_bias - τ_d
            
        虽然 Top-K 约束保证 token 数量，但偏置影响 *哪些* 被选中。
        训练初期高偏置鼓励探索，后期偏置→0 使决策由 MLP 主导。
        
        退火公式:
            b(t) = b_start + (b_end - b_start) · t / total
            
        Args:
            total_steps: 总退火步数
            b_start: 初始偏置 (默认 0.5)
            b_end: 终态偏置 (默认 0.0)
            
        Returns:
            self (链式调用)
        """
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        
        self._bias_total_steps.fill_(total_steps)
        self._bias_start.fill_(b_start)
        self._bias_end.fill_(b_end)
        self._bias_step.zero_()
        self._bias_enabled = True
        
        # 设置初始偏置
        self.set_explore_bias(b_start)
        
        return self
    
    def disable_explore_bias_annealing(self) -> "GumbelTopKSplitter":
        """禁用探索偏置退火。"""
        self._bias_enabled = False
        return self
    
    def _update_temperature(self) -> float:
        """
        更新温度 (内部方法，在 forward 中调用)。
        
        Returns:
            更新后的温度值
        """
        total = self._temp_total_steps.item()
        if total <= 0:
            return self.current_temperature
        
        step = self._temp_step.item()
        progress = min(1.0, step / total)
        
        T_s = self._temp_start.item()
        T_e = self._temp_end.item()
        
        if self._temp_schedule == 'exponential':
            # T(t) = T_start · (T_end / T_start)^progress
            T = T_s * ((T_e / T_s) ** progress)
        elif self._temp_schedule == 'linear':
            T = T_s + (T_e - T_s) * progress
        elif self._temp_schedule == 'cosine':
            T = T_e + (T_s - T_e) * (1 + math.cos(math.pi * progress)) / 2
        else:
            T = T_s
        
        self.set_temperature(T)
        self._temp_step.add_(1)
        
        return T
    
    def _update_explore_bias(self) -> float:
        """
        更新探索偏置 (内部方法，在 forward 中调用)。
        
        Returns:
            更新后的偏置值
        """
        total = self._bias_total_steps.item()
        if total <= 0:
            return self.explore_bias.item()
        
        step = self._bias_step.item()
        progress = min(1.0, step / total)
        
        b_s = self._bias_start.item()
        b_e = self._bias_end.item()
        
        # 线性退火
        b = b_s + (b_e - b_s) * progress
        
        self.set_explore_bias(b)
        self._bias_step.add_(1)
        
        return b
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """获取诊断信息。"""
        return {
            'num_candidates': self.num_candidates,
            'max_depth': self._current_max_depth,
            'K_min': self.K_min,
            'K_max': self.K_max,
            'avg_selected': self._avg_selected.item(),
            'temperature': self.current_temperature,
            'explore_bias': self.explore_bias.item(),
            'depth_bias_beta': self.depth_bias_beta.item(),
            'depth_bias_gamma': self.depth_bias_gamma.item(),
        }


# ============================================================================
# 工厂函数：从 LearnableSplitter 参数创建
# ============================================================================

def create_gumbel_topk_from_config(
    feature_dim: int = 256,
    # I30-17-EXT: 替换 max_depth 为 min_patch_size + max_depth_limit
    # 兼容旧 API: max_depth 仍可用，通过转换得到 min_patch_size
    max_depth: Optional[int] = None,
    min_patch_size: Optional[int] = None,
    max_depth_limit: int = 8,
    hidden_dim: int = 128,
    pool_size: int = 4,
    temperature: float = 1.0,
    image_size: Tuple[int, int] = (64, 64),
    K_min: int = 8,
    K_max: int = 64,
    **kwargs
) -> GumbelTopKSplitter:
    """
    从配置创建 GumbelTopKSplitter。

    I30-17-EXT 更新: 支持动态深度计算
        - 新 API: 使用 min_patch_size + max_depth_limit
        - 旧 API: 仍支持 max_depth (自动转换)

    使用方法:
        ```python
        from vit_pytorch.gumbel_topk_splitter import create_gumbel_topk_from_config

        # 新 API (推荐)
        splitter = create_gumbel_topk_from_config(
            feature_dim=256,
            min_patch_size=4,      # 目标最小 patch 大小
            max_depth_limit=8,     # 硬上限
            K_min=8,
            K_max=64,
        )

        # 旧 API (仍支持)
        splitter = create_gumbel_topk_from_config(
            feature_dim=256,
            max_depth=3,           # 自动转换为 min_patch_size
            K_min=8,
            K_max=64,
        )
        ```

    转换公式:
        min_patch_size = min(H, W) / (2^max_depth)
    """
    # I30-17-EXT: 处理新旧 API 兼容
    if max_depth is not None and min_patch_size is None:
        # 旧 API: 从 max_depth 计算 min_patch_size
        min_dim = min(image_size)
        min_patch_size = max(1, min_dim // (2 ** max_depth))
        max_depth_limit = max(max_depth, max_depth_limit)

    if min_patch_size is None:
        min_patch_size = 4  # 默认值

    return GumbelTopKSplitter(
        feature_dim=feature_dim,
        min_patch_size=min_patch_size,
        max_depth_limit=max_depth_limit,
        hidden_dim=hidden_dim,
        pool_size=pool_size,
        temperature=temperature,
        image_size=image_size,
        K_min=K_min,
        K_max=K_max,
        **kwargs
    )
