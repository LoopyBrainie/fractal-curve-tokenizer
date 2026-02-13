# -*- coding: utf-8 -*-
"""
Fractal ViT 统一配置模块 (I98-3: 协议驱动配置化)

组件配置原则:
1. 组件自治: 每个组件接收完整配置，独立运行
2. 精选暴露: 只暴露高影响参数 (impact > 0.1)
3. 可组合: 支持 YAML 配置 + CLI 参数覆盖
4. 协议驱动: 使用 Protocol 确保类型安全

数学形式化
==========
参数影响度度量:
    impact(p, D) = ||∂L/D||₂ × Var_D(p)

暴露决策公式:
    E(p) = EXPOSE     if impact(p) > 0.5
         = OPTIONAL   if 0.1 < impact(p) ≤ 0.5
         = HIDDEN     if impact(p) ≤ 0.1

日期: 2026-01-17
更新: 2026-01-22 (I98-3 协议驱动配置)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple
import math

# 导入 I33 相对预算常量
from .constants import (
    K_COVERAGE_BASE,
    K_COVERAGE_MIN,
    K_COVERAGE_MAX_HARD,
    K_MIN_HARD_LIMIT,
    K_MAX_HARD_LIMIT,
    K_ADAPTIVE_REFERENCE_SIZE,
    LEARNABLE_QUOTA_ENABLED,
    QUOTA_INIT_LOGITS,
    QUOTA_ENTROPY_WEIGHT,
    HILBERT_BIAS_SCALE,
    LEVEL_BIAS_SCALE,
    SPLITTER_TEMP_START,
    SPLITTER_TEMP_END,
)


# ==================== 类型别名 ====================

# 温度退火调度类型
# I122-7: 移除 'cosine' (无理论依据)，新增 'inverse_time'
AnnealSchedule = Literal['linear', 'exponential', 'inverse_time']
# Tokenizer 类型 (I145: 移除废弃的 streaming_v1/streaming_v2)
TokenizerType = Literal['streaming_v3']


# ==================== Splitter 配置 ====================

@dataclass
class HilbertSplitterConfig:
    """
    GumbelTopKSplitter 配置 (I111-4: 数学最优重构)

    数学形式化
    ==========

    Hilbert 曲线参数:
        N_candidates = Σ(4^d), d=0..L = (4^(L+1) - 1) / 3
        L_max = ⌈log₂(min(H,W) / P_min)⌉

    相对预算公式 (I33):
        K_min = max(K_min_abs, α × N)
        K_max = min(K_max_hard, β × γ(H,W) × N)
        γ(H,W) = √(min(H,W) / 224)

    熵正则化 (自适应):
        H_target = log(D) × (1 - 1/√D)

    版本历史:
        - v2.0 (2026-02-02): 重构配置层，反映 Hilbert 曲线数学结构
                          移除简化参数，使用相对预算公式
    """

    # ==================== Hilbert 曲线参数 ====================
    # 最小 patch 尺寸 (像素)
    min_patch_size: int = 4

    # 最大分区深度 (由图像尺寸动态计算，默认 8)
    max_level_limit: int = 8

    # ==================== 覆盖率约束 (相对预算) ====================
    # 基准覆盖率 (224×224 图像的目标采样率)
    # I145 修复: 注释更新以反映实际常量值 K_COVERAGE_BASE = 0.25
    coverage_base: float = K_COVERAGE_BASE

    # 覆盖率范围 [α, β] (用于计算 K 边界)
    coverage_min: float = K_COVERAGE_MIN
    # I145 修复: K_COVERAGE_MAX_HARD = 0.50，注释应为 0.50
    coverage_max_hard: float = K_COVERAGE_MAX_HARD

    # K 绝对边界 (硬限制)
    K_min_abs: int = K_MIN_HARD_LIMIT  # 8
    K_max_hard: int = K_MAX_HARD_LIMIT  # 4096

    # 覆盖率自适应参考尺寸
    adaptive_reference_size: int = K_ADAPTIVE_REFERENCE_SIZE  # 224

    # ==================== 架构参数 ====================
    feature_dim: int = 256
    hidden_dim: int = 64
    intermediate_dim: int = 64
    pool_size: int = 4
    use_dynamic_k: bool = True

    # ==================== 正则化参数 ====================
    # I120-2: dropout 必须为 0.0 以确保 Tokenizer 确定性
    # Tokenizer 是预处理器，不应引入随机性。Transformer 负责正则化。
    dropout: float = 0.0

    # Elastic Budget 目标导向损失参数 (I109-4)
    elastic_lambda_target: float = 0.1
    elastic_lambda_boundary: float = 0.1

    # ==================== 配额参数 (方案 E) ====================
    enable_learnable_quota: bool = LEARNABLE_QUOTA_ENABLED  # True
    quota_init_logits: Optional[Tuple[float, ...]] = None
    quota_entropy_weight: float = QUOTA_ENTROPY_WEIGHT  # 0.1
    quota_min_ratio: float = 0.02  # I96-7: 最小采样比例
    quota_min_lambda: float = 0.1  # 下界软正则化权重

    # I120-3: 选中率均衡配额 (解决深度分布单一化)
    # 目标: 各深度选中率均衡 P(选中|d) = C，避免深度 0 选中率是深度 4 的 256 倍
    enable_rate_balanced_quota: bool = True

    # I120-3: 分层自适应配额 (推荐方案)
    # 目标: 根据图像内容动态调整各深度配额，保留嵌套结构语义
    # 注意: 需要 features 输入，启用后会覆盖其他配额模式
    enable_hierarchical_quota: bool = False

    # ==================== 熵正则化 (I111-3) ====================
    # 'adaptive': H_target = log(D) × (1 - 1/√D) (推荐)
    # 'target': H_target = 固定值
    # 'disabled': 不使用熵正则化
    entropy_mode: str = 'adaptive'
    entropy_weight_base: float = 0.1  # 基础权重 (动态调整)
    entropy_target: Optional[float] = None  # 固定目标 (target 模式)

    # ==================== LookAheadHead 相关参数 (I113-2: 相关性分裂) ====================
    # L1: 相对参数 - 归一化的控制参数
    lookahead_dim: int = 128  # LookAheadHead 输出维度
    target_ratio: float = 0.5  # 目标分裂率 τ_target ∈ [0, 1]
    max_ratio: float = 0.5  # 最大分裂率上限 (替代硬编码 64) I113-2 修正
    gamma: float = 1.0  # Lagrangian 预算约束系数
    lambda_div: float = 0.1  # Diversity Loss 权重

    # ==================== 温度调度 (I111-1, I122-7) ====================
    # I122-7: 默认使用 'linear' 调度 (恒定变化率，行为可预测)
    temperature_init: float = SPLITTER_TEMP_START  # 1.0
    temperature_min: float = SPLITTER_TEMP_END  # 0.4
    temperature_anneal: str = 'linear'  # I122-7: 移除 'cosine'，默认 'linear'
    learnable_temperature: bool = True
    temperature_warmup_steps: int = 1000

    # ==================== 冻结控制 ====================
    freeze_quota: bool = False

    # ==================== I120-2: DeterministicTopK 配置 ====================
    # I130-2: Hilbert 最佳实现 - 默认启用确定性 Top-K 选择
    # 理由: 消除 train/eval 输出差异，恢复 Hilbert 曲线确定性保证
    # 预期效果: train/eval max_diff: 4.48 → <0.05 (89× 改善)
    use_deterministic_topk: bool = True

    # 确定性 Top-K 的温度参数（控制 softmax 的"锐度"）
    # 较小的值 → 更接近硬选择 (Top-K)
    # 较大的值 → 更软的分布 (Softmax)
    deterministic_temperature: float = 0.5

    # 确定性 Top-K 的 STE 混合系数
    # α = 0: 完全硬选择
    # α = 1: 完全软分布
    deterministic_ste_alpha: float = 0.5

    # ==================== I113-12: 静态 K 模式配置 ====================
    # 启用静态 K 模式以支持 torch.compile 的 cudagraphs 优化
    # 启用后，token 数量将 padding 到 static_k_max
    static_k_mode: bool = False
    static_k_max: int = 64  # Padding 目标 K 值

    # ==================== 验证与工具方法 ====================

    def get_quota_init_tensor(self, D: int) -> Tuple[float, ...]:
        """获取适合给定深度 D 的初始化 logits"""
        if self.quota_init_logits is not None:
            init = list(self.quota_init_logits)
            if len(init) < D:
                init.extend([0.0] * (D - len(init)))
            elif len(init) > D:
                init = init[:D]
            return tuple(init)
        else:
            return tuple([0.0] * D)

    def compute_candidate_count(self) -> int:
        """计算四叉树候选节点总数 (I111-4)

        数学公式: N = (4^(L+1) - 1) / 3
        """
        L = self.max_level_limit
        return (4 ** (L + 1) - 1) // 3

    def compute_k_bounds(
        self,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        """计算 K 边界 (I111-5: 完整相对预算公式)

        数学公式:
            N = (4^(L+1) - 1) / 3
            γ = √(min(H,W) / 224)  (尺度因子)
            K_min = max(K_min_abs, coverage_min × N)
            K_max = min(K_max_hard, coverage_max_hard × γ × N)

        Args:
            image_size: 图像尺寸 (H, W)，用于尺度因子计算

        Returns:
            (K_min, K_max): 动态边界元组
        """
        # 计算候选节点总数
        N = self.compute_candidate_count()

        # 计算尺度因子
        if image_size is not None:
            H, W = image_size
            min_dim = min(H, W)
            scale_factor = math.sqrt(min_dim / self.adaptive_reference_size)
        else:
            scale_factor = 1.0

        # 计算 K_min
        K_min = max(
            self.K_min_abs,
            int(math.ceil(self.coverage_min * N))
        )

        # 计算 K_max
        K_max = min(
            self.K_max_hard,
            int(math.ceil(self.coverage_max_hard * scale_factor * N))
        )

        return K_min, K_max

    def validate(self) -> None:
        """验证配置参数的有效性 (I111-4: 数学一致性验证)"""

        # Hilbert 曲线参数验证
        if self.min_patch_size <= 0:
            raise ValueError(f"min_patch_size 必须为正数, got {self.min_patch_size}")
        if self.max_level_limit < 2:
            raise ValueError(
                f"max_level_limit >= 2 是推荐配置, got {self.max_level_limit}"
            )

        # 覆盖率参数验证
        if not 0 < self.coverage_base <= 1.0:
            raise ValueError(f"coverage_base 必须在 (0, 1] 范围内, got {self.coverage_base}")
        if not 0 < self.coverage_min < self.coverage_max_hard <= 1.0:
            raise ValueError(
                f"coverage_min ({self.coverage_min}) < coverage_max_hard ({self.coverage_max_hard}) "
                f"且都在 (0, 1] 范围内"
            )

        # K 边界验证 (I111-4: 覆盖率参数和 K 边界独立设置，不做交叉验证)
        # 用户可以自由配置覆盖率参数和 K 边界，它们在实际使用时通过 compute_k_bounds() 协调

        # 温度参数验证
        if not 0 < self.temperature_min <= self.temperature_init:
            raise ValueError(
                f"temperature_min ({self.temperature_min}) 必须 < "
                f"temperature_init ({self.temperature_init})"
            )
        if self.temperature_min < 0.3:
            raise ValueError(
                f"temperature_min ({self.temperature_min}) 必须 >= 0.3 "
                "以避免梯度消失问题"
            )
        if self.temperature_anneal not in ('linear', 'exponential', 'inverse_time'):
            raise ValueError(
                f"temperature_anneal 必须是 'linear', 'exponential' 或 'inverse_time', "
                f"got {self.temperature_anneal}"
            )

        # 熵模式验证
        if self.entropy_mode not in ('adaptive', 'target', 'disabled'):
            raise ValueError(
                f"entropy_mode 必须是 'adaptive', 'target', 或 'disabled', "
                f"got {self.entropy_mode}"
            )

    def to_dict(self) -> dict:
        """转换为字典 (用于序列化)"""
        return {
            'min_patch_size': self.min_patch_size,
            'max_level_limit': self.max_level_limit,
            'coverage_base': self.coverage_base,
            'coverage_min': self.coverage_min,
            'coverage_max_hard': self.coverage_max_hard,
            'K_min_abs': self.K_min_abs,
            'K_max_hard': self.K_max_hard,
            'adaptive_reference_size': self.adaptive_reference_size,
            'feature_dim': self.feature_dim,
            'hidden_dim': self.hidden_dim,
            'intermediate_dim': self.intermediate_dim,
            'pool_size': self.pool_size,
            'use_dynamic_k': self.use_dynamic_k,
            'dropout': self.dropout,
            'elastic_lambda_target': self.elastic_lambda_target,
            'elastic_lambda_boundary': self.elastic_lambda_boundary,
            'enable_learnable_quota': self.enable_learnable_quota,
            'quota_init_logits': self.quota_init_logits,
            'quota_entropy_weight': self.quota_entropy_weight,
            'quota_min_ratio': self.quota_min_ratio,
            'quota_min_lambda': self.quota_min_lambda,
            'entropy_mode': self.entropy_mode,
            'entropy_weight_base': self.entropy_weight_base,
            'entropy_target': self.entropy_target,
            # LookAheadHead 参数 (I113-2)
            'lookahead_dim': self.lookahead_dim,
            'target_ratio': self.target_ratio,
            'max_ratio': self.max_ratio,  # I113-2: 最大分裂率上限
            'gamma': self.gamma,
            'lambda_div': self.lambda_div,
            # 温度参数
            'temperature_init': self.temperature_init,
            'temperature_min': self.temperature_min,
            'temperature_anneal': self.temperature_anneal,
            'learnable_temperature': self.learnable_temperature,
            'temperature_warmup_steps': self.temperature_warmup_steps,
            'freeze_quota': self.freeze_quota,
            # I120-2: DeterministicTopK 配置
            'use_deterministic_topk': self.use_deterministic_topk,
            'deterministic_temperature': self.deterministic_temperature,
            'deterministic_ste_alpha': self.deterministic_ste_alpha,
        }

    # ==================== L2 绝对值计算方法 (I113-2) ====================

    def compute_n_base(self) -> int:
        """计算四叉树候选节点总数 (L2 绝对值)

        数学公式: N_base = (4^(L+1) - 1) / 3

        Returns:
            N_base: 基础候选节点数 (绝对值)
        """
        return self.compute_candidate_count()

    def compute_n_target(self, image_size: Optional[Tuple[int, int]] = None) -> int:
        """计算目标 Token 数 (L2 绝对值)

        数学公式:
            N_base = (4^(L+1) - 1) / 3
            N_target = N_base × τ_target
            N_max = N_base × max_ratio  (替代硬编码 64)

        Args:
            image_size: 图像尺寸 (H, W)，可选，用于日志记录

        Returns:
            N_target: 目标 Token 数 (绝对值，范围 [8, N_base × max_ratio])
        """
        N_base = self.compute_n_base()
        N_target = int(N_base * self.target_ratio)

        # 边界限制: 下界=8，上界=N_base × max_ratio
        N_max = int(N_base * self.max_ratio)
        N_target = max(8, min(N_max, N_target))

        return N_target

    def compute_absolute_targets(self, image_size: Optional[Tuple[int, int]] = None) -> dict:
        """计算所有绝对目标值 (L2)

        Returns:
            dict: 包含 N_base, N_target, N_min, N_max
        """
        N_base = self.compute_n_base()
        N_target = self.compute_n_target(image_size)

        # 获取动态 K 边界作为绝对边界
        K_min, K_max = self.compute_k_bounds(image_size)

        # N_max 使用相对比例 max_ratio，避免硬编码 64
        N_max = min(int(N_base * self.max_ratio), K_max)

        return {
            'N_base': N_base,        # 绝对: 基础候选数
            'N_target': N_target,    # 绝对: 目标Token数
            'N_min': max(8, K_min),  # 绝对: 最小Token数
            'N_max': N_max,          # 绝对: 最大Token数 (L1: max_ratio × N_base)
        }


# 向后兼容别名 (I111-4: 过渡期使用)
SplitterConfig = HilbertSplitterConfig


# ==================== I130-3/I160-1: Neighbor-Aware Splitter 配置 ====================

# I130-3: NeighborAwareSplitter 的配置类
# I160-1: 重命名为 DeterministicNeighborSplitter，使用确定性选择
# 注意：原配置类已合并到 deterministic_neighbor.py，此处保留完整实现以避免循环导入

@dataclass
class NeighborAwareSplitterConfig:
    """
    NeighborAwareSplitter 配置 (I130-3: 解决收敛极慢问题)

    数学形式化
    ==========

    核心创新:
    1. 梯度覆盖率 100%: 使用软 softmax 配额，无需 floor() 操作
    2. 邻居感知评分: s_i' = s_i + α × Σ_{j∈N(i)} sim(f_i, f_j) × s_j
    3. 可微 ROI Align: 使用 grid_sample 替代整数索引

    与 GumbelTopKSplitter 对比:

    | 方面                | GumbelTopKSplitter    | NeighborAwareSplitter |
    |---------------------|----------------------|---------------------|
    | 梯度覆盖率          | 0% (floor)           | 100% (softmax)      |
    | 邻居关系            | 无                   | 显式建模 (LCA)     |
    | 局部一致性          | 不保证               | LocalityConsistencyLoss |
    | ROI Align          | 整数索引             | grid_sample 可微    |

    版本历史:
    - v1.0 (2026-02-12): 初始实现，借鉴 NAP 论文 (Li & Xu 2025)
    - v2.0 (2026-02-12): I160-1 重命名，使用 DeterministicNeighborSplitter
    """

    # ==================== Hilbert 曲线参数 ====================
    min_patch_size: int = 4
    max_level_limit: int = 8

    # ==================== 覆盖率约束 ====================
    coverage_base: float = K_COVERAGE_BASE
    coverage_min: float = K_COVERAGE_MIN
    coverage_max_hard: float = K_COVERAGE_MAX_HARD
    K_min_abs: int = K_MIN_HARD_LIMIT
    K_max_hard: int = K_MAX_HARD_LIMIT
    adaptive_reference_size: int = K_ADAPTIVE_REFERENCE_SIZE

    # ==================== 架构参数 ====================
    feature_dim: int = 256
    hidden_dim: int = 64
    pool_size: int = 4

    # ==================== 邻居感知参数 ====================
    neighbor_threshold: int = 2
    alpha_init: float = 0.5
    learnable_alpha: bool = True

    # ==================== 温度参数 ====================
    temperature_init: float = SPLITTER_TEMP_START
    temperature_min: float = SPLITTER_TEMP_END
    learnable_temperature: bool = True

    # ==================== 配额参数 ====================
    enable_learnable_quota: bool = LEARNABLE_QUOTA_ENABLED
    quota_init_logits: Optional[Tuple[float, ...]] = None
    quota_entropy_weight: float = QUOTA_ENTROPY_WEIGHT

    # ==================== 局部一致性损失 ====================
    locality_weight: float = 0.1

    def compute_candidate_count(self) -> int:
        """计算四叉树候选节点总数"""
        L = self.max_level_limit
        return (4 ** (L + 1) - 1) // 3

    def compute_k_bounds(
        self,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        """计算 K 边界"""
        N = self.compute_candidate_count()

        if image_size is not None:
            H, W = image_size
            min_dim = min(H, W)
            scale_factor = math.sqrt(min_dim / self.adaptive_reference_size)
        else:
            scale_factor = 1.0

        K_min = max(
            self.K_min_abs,
            int(math.ceil(self.coverage_min * N))
        )

        K_max = min(
            self.K_max_hard,
            int(math.ceil(self.coverage_max_hard * scale_factor * N))
        )

        return K_min, K_max

    def validate(self) -> None:
        """验证配置参数的有效性"""
        if self.min_patch_size <= 0:
            raise ValueError(f"min_patch_size 必须为正数, got {self.min_patch_size}")
        if self.max_level_limit < 2:
            raise ValueError(f"max_level_limit >= 2 是推荐配置, got {self.max_level_limit}")
        if not 0 < self.coverage_base <= 1.0:
            raise ValueError(f"coverage_base 必须在 (0, 1] 范围内, got {self.coverage_base}")
        if not 0 < self.coverage_min < self.coverage_max_hard <= 1.0:
            raise ValueError(f"coverage_min ({self.coverage_min}) < coverage_max_hard")
        if not 0 < self.temperature_min <= self.temperature_init:
            raise ValueError(f"temperature_min ({self.temperature_min}) 必须 < temperature_init")

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'min_patch_size': self.min_patch_size,
            'max_level_limit': self.max_level_limit,
            'coverage_base': self.coverage_base,
            'coverage_min': self.coverage_min,
            'coverage_max_hard': self.coverage_max_hard,
            'K_min_abs': self.K_min_abs,
            'K_max_hard': self.K_max_hard,
            'adaptive_reference_size': self.adaptive_reference_size,
            'feature_dim': self.feature_dim,
            'hidden_dim': self.hidden_dim,
            'pool_size': self.pool_size,
            'neighbor_threshold': self.neighbor_threshold,
            'alpha_init': self.alpha_init,
            'learnable_alpha': self.learnable_alpha,
            'temperature_init': self.temperature_init,
            'temperature_min': self.temperature_min,
            'learnable_temperature': self.learnable_temperature,
            'enable_learnable_quota': self.enable_learnable_quota,
            'quota_init_logits': self.quota_init_logits,
            'quota_entropy_weight': self.quota_entropy_weight,
            'locality_weight': self.locality_weight,
        }# ==================== Attention 配置 ====================

@dataclass
class AttentionConfig:
    """
    HilbertAwareAttention 配置

    I122-2 简化: 移除 lca_temperature，由 hilbert_bias_scale 统一缩放

    暴露参数:
    - hilbert_bias_scale: impact=0.1 → 可选暴露
    """
    # 注意力维度
    dim: int = 256
    heads: int = 8
    head_dim: Optional[int] = None  # None = dim // heads

    # Hilbert 偏置参数
    lca_bias: bool = True
    max_level: int = 8
    lca_embedding_dim: int = 128
    lca_fp16: bool = False  # I104-3: 使用 FP16 存储 LCA embedding

    # 偏置缩放
    hilbert_bias_scale: float = HILBERT_BIAS_SCALE
    level_bias_scale: float = LEVEL_BIAS_SCALE

    # Dropout
    attention_dropout: float = 0.0
    dropout: float = 0.0

    # 存储选项
    store_attn_weights: bool = False


# ==================== Tokenizer 配置 ====================

@dataclass
class TokenizerConfig:
    """
    StreamingFractalTokenizerV3 配置
    """
    # 图像参数
    image_size: Tuple[int, int] = (64, 64)
    channels: int = 3

    # 嵌入参数
    patch_size: int = 4
    embed_dim: int = 256

    # 深度参数
    max_level: int = 8

    # Hilbert 排序
    use_hilbert_order: bool = True

    # Dropout
    embed_dropout: float = 0.0

    # 分裂器参数
    splitter_config: Optional[SplitterConfig] = None

    def __post_init__(self):
        if self.splitter_config is None:
            self.splitter_config = SplitterConfig()


# ==================== Transformer 配置 ====================

@dataclass
class TransformerConfig:
    """
    FractalTransformer 配置
    """
    dim: int = 256
    num_layers: int = 8
    heads: int = 8
    head_dim: Optional[int] = None

    # FFN 参数
    mlp_ratio: float = 4.0
    use_swiglu: bool = True

    # Dropout
    attention_dropout: float = 0.0
    dropout: float = 0.0
    drop_path: float = 0.0

    # 注意力配置
    attention_config: Optional[AttentionConfig] = None

    def __post_init__(self):
        if self.attention_config is None:
            self.attention_config = AttentionConfig(
                dim=self.dim,
                heads=self.heads,
                head_dim=self.head_dim,
            )


# ==================== 模型主配置 ====================

@dataclass
class FractalViTConfig:
    """
    Fractal ViT 统一配置

    组合所有子组件配置，支持 YAML 序列化 + CLI 覆盖
    """
    # 任务参数
    num_classes: int = 1000

    # Tokenizer 配置
    tokenizer_config: TokenizerConfig = field(default_factory=TokenizerConfig)

    # Transformer 配置
    transformer_config: TransformerConfig = field(default_factory=TransformerConfig)

    # 分类头
    use_head: bool = True
    head_hidden_dim: Optional[int] = None

    # 预训练配置
    pretrained_dim: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: str) -> "FractalViTConfig":
        """从 YAML 文件加载配置"""
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """保存配置到 YAML 文件"""
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.as_dict(), f, indent=2)

    def as_dict(self) -> dict:
        """转换为字典（用于序列化）"""
        return {
            'num_classes': self.num_classes,
            'tokenizer_config': {
                'image_size': self.tokenizer_config.image_size,
                'channels': self.tokenizer_config.channels,
                'patch_size': self.tokenizer_config.patch_size,
                'embed_dim': self.tokenizer_config.embed_dim,
                'max_level': self.tokenizer_config.max_level,
                'use_hilbert_order': self.tokenizer_config.use_hilbert_order,
                'splitter_config': {
                    'enable_learnable_quota': self.tokenizer_config.splitter_config.enable_learnable_quota,
                    'quota_init_logits': self.tokenizer_config.splitter_config.quota_init_logits,
                    'quota_min_ratio': self.tokenizer_config.splitter_config.quota_min_ratio,
                    'quota_min_lambda': self.tokenizer_config.splitter_config.quota_min_lambda,
                    'K_min_abs': self.tokenizer_config.splitter_config.K_min_abs,
                    'K_max_hard': self.tokenizer_config.splitter_config.K_max_hard,
                    'coverage_base': self.tokenizer_config.splitter_config.coverage_base,
                    'coverage_min': self.tokenizer_config.splitter_config.coverage_min,
                    'coverage_max_hard': self.tokenizer_config.splitter_config.coverage_max_hard,
                    'target_ratio': self.tokenizer_config.splitter_config.target_ratio,
                    'max_ratio': self.tokenizer_config.splitter_config.max_ratio,
                },
            },
            'transformer_config': {
                'dim': self.transformer_config.dim,
                'num_layers': self.transformer_config.num_layers,
                'heads': self.transformer_config.heads,
                'mlp_ratio': self.transformer_config.mlp_ratio,
                'use_swiglu': self.transformer_config.use_swiglu,
                'dropout': self.transformer_config.dropout,
                'drop_path': self.transformer_config.drop_path,
            },
        }


# ==================== I98-3: 编码器配置类 ====================

@dataclass
class ShapeScaleEncoderConfig:
    """形状-尺度编码器配置 (I98-3, I31-P2: 添加 enabled 标志).

    影响度驱动的常量暴露:
    - enabled: 启用/禁用编码器，impact=1.0 → 必须暴露
    - hidden_dim: 隐藏层维度，impact=0.5 → 必须暴露
    - weight_init: 权重初始化值，impact=0.3 → 可选暴露

    数学形式化
    ==========
    形状特征: r = log(w/h) ∈ (-∞, ∞)
    尺度特征: s = (w/W)(h/H) ∈ [0, 1]

    组合编码: [r_norm, s_log] → MLP → dim
    其中 r_norm = tanh(r / (1 + |r|)) ∈ (-1, 1)
          s_log = log(s + ε) ∈ (-∞, 0]
    """
    enabled: bool = True  # I31-P2: 启用/禁用开关
    hidden_dim: int = 64
    output_dim: int = 256
    weight_init: float = 0.1  # I35-2: 非零初始化确保梯度

    def to_dict(self) -> dict:
        return {
            'enabled': self.enabled,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'weight_init': self.weight_init,
        }


@dataclass
class AreaEncoderConfig:
    """面积编码器配置 (I98-3).

    影响度驱动的常量暴露:
    - fourier_levels: Fourier 级别数，impact=0.24 → 必须暴露
    - hidden_dim: 隐藏层维度，impact=0.3 → 可选暴露
    - freq_base: 频率基数，impact=0.12 → 可选暴露
    - cutoff_ratio: Nyquist裁剪比例，impact=0.15 → 可选暴露

    数学形式化
    ==========
    傅里叶特征 (I32-7 动态频率 + 软截断门控):
        f_k = π · b^k · g_k(freq_k, L_norm)

    其中:
        b = freq_base (频率基数)
        g_k = CosineGate(freq_k, L_norm) (软截断门控)
        cutoff_ratio = omega_cutoff / omega_nyquist (Nyquist频率裁剪比例)
    """
    fourier_levels: int = 4
    freq_base: float = 2.0
    hidden_dim: int = 32
    output_dim: int = 256

    # I98-3: Nyquist频率裁剪配置
    # f_cutoff = cutoff_ratio * f_nyquist，其中 cutoff_ratio ∈ (0, 1]
    cutoff_ratio: float = 0.8

    def to_dict(self) -> dict:
        return {
            'fourier_levels': self.fourier_levels,
            'freq_base': self.freq_base,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'cutoff_ratio': self.cutoff_ratio,
        }


@dataclass
class LCAEncoderConfig:
    """LCA 编码器配置 (I98-3).

    影响度驱动的常量暴露:
    - embedding_dim: 嵌入维度，impact=0.5 → 必须暴露
    - scale_init_factor: 初始化缩放因子，impact=0.15 → 可选暴露

    数学形式化
    ==========
    LCA 嵌入初始化:
        scale_d = scale_factor × (1 + log(d + 1))

    其中 d 为四叉树深度，scale_factor 控制整体缩放。
    """
    embedding_dim: int = 128
    scale_init_factor: float = 0.1  # 替代硬编码 0.1
    temperature: Optional[float] = None  # None = 自动

    def to_dict(self) -> dict:
        return {
            'embedding_dim': self.embedding_dim,
            'scale_init_factor': self.scale_init_factor,
            'temperature': self.temperature,
        }


@dataclass
class AttentionEncoderConfig:
    """注意力编码器统一配置 (I98-3).

    组合所有编码器配置，支持协议驱动:
    - ShapeScaleEncoder: 形状-尺度编码
    - AreaEncoder: 面积编码
    - LCAEncoder: LCA 嵌入

    同时包含偏置缩放参数 (从 constants.py 独立):
    - hilbert_bias_scale: Hilbert 偏置缩放
    - level_bias_scale: Level 偏置缩放

    设计原则
    =========
    1. 正交性: 每个编码器配置完全独立
    2. 可组合: 支持灵活组合不同编码器
    3. 可禁用: fourier_levels=0 禁用 AreaEncoder
    """
    shape_scale: ShapeScaleEncoderConfig = field(
        default_factory=ShapeScaleEncoderConfig
    )
    area: AreaEncoderConfig = field(
        default_factory=AreaEncoderConfig
    )
    lca: LCAEncoderConfig = field(
        default_factory=LCAEncoderConfig
    )

    # 偏置缩放 (从 constants.py 独立，I98-3)
    hilbert_bias_scale: float = 0.1
    level_bias_scale: float = 0.05

    # I98-3: 初始化参数配置 (用于 HilbertAwareMultiScaleAttention)
    # 能量基准初始化: raw* = softplus^{-1}(1.0) ≈ 0.5413
    level_scale_init: float = 0.5413
    # I113-11: 偏置缩放初始化
    # 修复前: raw = log(scale)，对应 scale=0.1 和 scale=0.05
    # 修复后: 初始 scale=1.0，配合 √d_k 量纲对齐后有效 scale ≈ √d_k ≈ 5.66
    hilbert_bias_init: float = 0.0  # ln(1.0)
    level_bias_init: float = 0.0    # ln(1.0)
    # 可禁用能量注入（用于消融实验）
    energy_injection_enabled: bool = True

    def to_dict(self) -> dict:
        return {
            'shape_scale': self.shape_scale.to_dict(),
            'area': self.area.to_dict(),
            'lca': self.lca.to_dict(),
            'hilbert_bias_scale': self.hilbert_bias_scale,
            'level_bias_scale': self.level_bias_scale,
            'level_scale_init': self.level_scale_init,
            'hilbert_bias_init': self.hilbert_bias_init,
            'level_bias_init': self.level_bias_init,
            'energy_injection_enabled': self.energy_injection_enabled,
        }


# ==================== 工厂函数 (I111-4) ====================

def create_splitter_config(
    # 核心参数
    min_patch_size: Optional[int] = None,
    max_level_limit: Optional[int] = None,
    coverage_base: Optional[float] = None,
    coverage_min: Optional[float] = None,
    coverage_max_hard: Optional[float] = None,
    K_min_abs: Optional[int] = None,
    K_max_hard: Optional[int] = None,
    adaptive_reference_size: Optional[int] = None,
    # 架构参数
    feature_dim: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    intermediate_dim: Optional[int] = None,
    pool_size: Optional[int] = None,
    use_dynamic_k: Optional[bool] = None,
    dropout: Optional[float] = None,
    # 正则化参数
    elastic_lambda_target: Optional[float] = None,
    elastic_lambda_boundary: Optional[float] = None,
    enable_learnable_quota: Optional[bool] = None,
    quota_init_logits: Optional[Tuple[float, ...]] = None,
    quota_entropy_weight: Optional[float] = None,
    quota_min_ratio: Optional[float] = None,
    quota_min_lambda: Optional[float] = None,
    # 熵正则化
    entropy_mode: Optional[str] = None,
    entropy_weight_base: Optional[float] = None,
    entropy_target: Optional[float] = None,
    # LookAheadHead 参数 (I113-2)
    lookahead_dim: Optional[int] = None,
    target_ratio: Optional[float] = None,
    max_ratio: Optional[float] = None,  # I113-2: 最大分裂率上限
    gamma: Optional[float] = None,
    lambda_div: Optional[float] = None,
    # 温度调度
    temperature_init: Optional[float] = None,
    temperature_min: Optional[float] = None,
    temperature_anneal: Optional[str] = None,
    learnable_temperature: Optional[bool] = None,
    temperature_warmup_steps: Optional[int] = None,
    # 冻结控制
    freeze_quota: Optional[bool] = None,
    **kwargs,
) -> HilbertSplitterConfig:
    """
    工厂函数: 创建 HilbertSplitterConfig（用于 CLI 参数解析）

    数学保证
    =========
    相对预算公式 (I33):
        K_min = max(K_min_abs, α × N)
        K_max = min(K_max_hard, β × γ(H,W) × N)
        γ(H,W) = √(min(H,W) / 224)

    版本历史:
        - v2.0 (2026-02-02): 使用 HilbertSplitterConfig，重构参数命名
    """
    config = HilbertSplitterConfig()

    # Hilbert 曲线参数
    if min_patch_size is not None:
        config.min_patch_size = min_patch_size
    if max_level_limit is not None:
        config.max_level_limit = max_level_limit

    # 覆盖率约束
    if coverage_base is not None:
        config.coverage_base = coverage_base
    if coverage_min is not None:
        config.coverage_min = coverage_min
    if coverage_max_hard is not None:
        config.coverage_max_hard = coverage_max_hard
    if K_min_abs is not None:
        config.K_min_abs = K_min_abs
    if K_max_hard is not None:
        config.K_max_hard = K_max_hard
    if adaptive_reference_size is not None:
        config.adaptive_reference_size = adaptive_reference_size

    # 架构参数
    if feature_dim is not None:
        config.feature_dim = feature_dim
    if hidden_dim is not None:
        config.hidden_dim = hidden_dim
    if intermediate_dim is not None:
        config.intermediate_dim = intermediate_dim
    if pool_size is not None:
        config.pool_size = pool_size
    if use_dynamic_k is not None:
        config.use_dynamic_k = use_dynamic_k
    if dropout is not None:
        config.dropout = dropout

    # 正则化参数
    if elastic_lambda_target is not None:
        config.elastic_lambda_target = elastic_lambda_target
    if elastic_lambda_boundary is not None:
        config.elastic_lambda_boundary = elastic_lambda_boundary
    if enable_learnable_quota is not None:
        config.enable_learnable_quota = enable_learnable_quota
    if quota_init_logits is not None:
        config.quota_init_logits = quota_init_logits
    if quota_entropy_weight is not None:
        config.quota_entropy_weight = quota_entropy_weight
    if quota_min_ratio is not None:
        config.quota_min_ratio = quota_min_ratio
    if quota_min_lambda is not None:
        config.quota_min_lambda = quota_min_lambda

    # 熵正则化
    if entropy_mode is not None:
        config.entropy_mode = entropy_mode
    if entropy_weight_base is not None:
        config.entropy_weight_base = entropy_weight_base
    if entropy_target is not None:
        config.entropy_target = entropy_target

    # LookAheadHead 参数 (I113-2)
    if lookahead_dim is not None:
        config.lookahead_dim = lookahead_dim
    if target_ratio is not None:
        config.target_ratio = target_ratio
    if max_ratio is not None:
        config.max_ratio = max_ratio  # I113-2: 最大分裂率上限
    if gamma is not None:
        config.gamma = gamma
    if lambda_div is not None:
        config.lambda_div = lambda_div

    # 温度调度
    if temperature_init is not None:
        config.temperature_init = temperature_init
    if temperature_min is not None:
        config.temperature_min = temperature_min
    if temperature_anneal is not None:
        config.temperature_anneal = temperature_anneal
    if learnable_temperature is not None:
        config.learnable_temperature = learnable_temperature
    if temperature_warmup_steps is not None:
        config.temperature_warmup_steps = temperature_warmup_steps

    # 冻结控制
    if freeze_quota is not None:
        config.freeze_quota = freeze_quota

    # 应用额外参数
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


# ==================== I110-5: 语义冗余分裂器配置 ====================

@dataclass
class SemanticSplitterConfig:
    """语义冗余分裂器配置 (I110-5)

    数学设计原则
    ============
    理论最大深度（像素级上界）:
        L_theory = floor(log2(max(H, W)))  # min_patch_size=1 时的最大深度

    上界保护:
        max_level_limit ≤ L_theory

    示例（min_patch_size=4）:
        - 64x64 图像: L_theory = 6, 有效范围 [2, 6]
        - 224x224 图像: L_theory = 8, 有效范围 [2, 8]
        - 512x512 图像: L_theory = 9, 有效范围 [2, 9]
        - 1024x1024 图像: L_theory = 10, 有效范围 [2, 10]

    损失权重 (impact=0.1):
    - diversity_weight: 0.1 (标准)
    - reconstruction_weight: 0.1 (标准)

    决策参数 (impact=0.5):
    - split_threshold: 0.5 (标准)
    - use_gumbel_softmax: True

    温度调度 (impact=0.8):
    - gumbel_temp_start: 1.0
    - gumbel_temp_end: 0.5
    - learnable_temperature: True
    """

    # ========== 核心架构参数 ==========
    feature_dim: int = 256
    hidden_dim: int = 128

    # 深度参数：基于图像像素数动态计算，不使用硬编码
    image_size: Optional[Tuple[int, int]] = None  # (H, W) 用于计算理论上界
    min_patch_size: int = 4  # 最小 patch 大小
    max_level_limit: Optional[int] = None  # None = 自动计算到像素级上界

    # ========== 损失权重 ==========
    diversity_weight: float = 0.1
    reconstruction_weight: float = 0.1

    # ========== 决策参数 ==========
    split_threshold: float = 0.5
    use_gumbel_softmax: bool = True

    # ========== 温度调度 ==========
    gumbel_temp_start: float = 1.0
    gumbel_temp_end: float = 0.5
    learnable_temperature: bool = True

    def _compute_theoretical_max_level(self) -> int:
        """计算理论最大深度（基于图像像素数）

        公式: L_theory = floor(log2(max(H, W)))
        含义: min_patch_size=1 时的最大分裂深度

        I-OPT: 使用 bit_length() 替代 math.log2，避免浮点精度问题
        """
        if self.image_size is None:
            return 8
        H, W = self.image_size
        max_dim = max(H, W)
        # bit_length() 返回二进制表示的位数，对于 2^k 返回 k+1
        # 因此 bit_length() - 1 = floor(log2(n))
        return max_dim.bit_length() - 1

    def _compute_effective_max_level(self) -> int:
        """计算有效最大深度（应用上界保护）"""
        L_theory = self._compute_theoretical_max_level()
        if self.max_level_limit is None:
            return L_theory
        return min(self.max_level_limit, L_theory)

    def get_max_level(self) -> int:
        """获取有效最大深度（供外部使用）"""
        return self._compute_effective_max_level()

    def validate(self) -> None:
        """验证配置参数的有效性"""
        if self.feature_dim <= 0:
            raise ValueError(f"feature_dim 必须为正数, got {self.feature_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim 必须为正数, got {self.hidden_dim}")
        if self.min_patch_size <= 0:
            raise ValueError(f"min_patch_size 必须为正数, got {self.min_patch_size}")

        L_theory = self._compute_theoretical_max_level()
        if self.max_level_limit is not None:
            if self.max_level_limit < 2:
                raise ValueError(f"max_level_limit >= 2 是推荐配置, got {self.max_level_limit}")
            if self.max_level_limit > L_theory:
                raise ValueError(
                    f"max_level_limit ({self.max_level_limit}) 超过理论最大值 ({L_theory}), "
                    f"将自动截断"
                )

        if not 0 < self.diversity_weight <= 1:
            raise ValueError(f"diversity_weight 必须在 (0, 1] 范围内, got {self.diversity_weight}")
        if not 0 < self.reconstruction_weight <= 1:
            raise ValueError(f"reconstruction_weight 必须在 (0, 1] 范围内")
        if not 0 < self.split_threshold < 1:
            raise ValueError(f"split_threshold 必须在 (0, 1) 范围内, got {self.split_threshold}")
        if self.gumbel_temp_end >= self.gumbel_temp_start:
            raise ValueError(
                f"gumbel_temp_end ({self.gumbel_temp_end}) 必须 < gumbel_temp_start ({self.gumbel_temp_start})"
            )

    def to_dict(self) -> dict:
        """转换为字典（用于序列化）"""
        return {
            'feature_dim': self.feature_dim,
            'hidden_dim': self.hidden_dim,
            'image_size': self.image_size,
            'min_patch_size': self.min_patch_size,
            'max_level_limit': self.max_level_limit,
            'diversity_weight': self.diversity_weight,
            'reconstruction_weight': self.reconstruction_weight,
            'split_threshold': self.split_threshold,
            'use_gumbel_softmax': self.use_gumbel_softmax,
            'gumbel_temp_start': self.gumbel_temp_start,
            'gumbel_temp_end': self.gumbel_temp_end,
            'learnable_temperature': self.learnable_temperature,
            '_effective_max_level': self.get_max_level(),
        }


def create_semantic_splitter_config(
    feature_dim: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    image_size: Optional[Tuple[int, int]] = None,
    min_patch_size: Optional[int] = None,
    max_level_limit: Optional[int] = None,
    diversity_weight: Optional[float] = None,
    reconstruction_weight: Optional[float] = None,
    split_threshold: Optional[float] = None,
    use_gumbel_softmax: Optional[bool] = None,
    gumbel_temp_start: Optional[float] = None,
    gumbel_temp_end: Optional[float] = None,
    learnable_temperature: Optional[bool] = None,
    **kwargs,
) -> SemanticSplitterConfig:
    """工厂函数: 创建 SemanticSplitterConfig（用于 CLI 参数解析）"""
    config = SemanticSplitterConfig()

    if feature_dim is not None:
        config.feature_dim = feature_dim
    if hidden_dim is not None:
        config.hidden_dim = hidden_dim
    if image_size is not None:
        config.image_size = image_size
    if min_patch_size is not None:
        config.min_patch_size = min_patch_size
    if max_level_limit is not None:
        config.max_level_limit = max_level_limit
    if diversity_weight is not None:
        config.diversity_weight = diversity_weight
    if reconstruction_weight is not None:
        config.reconstruction_weight = reconstruction_weight
    if split_threshold is not None:
        config.split_threshold = split_threshold
    if use_gumbel_softmax is not None:
        config.use_gumbel_softmax = use_gumbel_softmax
    if gumbel_temp_start is not None:
        config.gumbel_temp_start = gumbel_temp_start
    if gumbel_temp_end is not None:
        config.gumbel_temp_end = gumbel_temp_end
    if learnable_temperature is not None:
        config.learnable_temperature = learnable_temperature

    # 应用额外参数
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


# ==================== I97-5: 合并 config_fractal.py 功能 ====================


@dataclass
class FractalConfig:
    """Fractal ViT 统一配置 (原 config_fractal.py).

    从 (image_size, min_patch_size) 自动推导所有几何参数。

    数学形式化
    ==========
    推导链:
        max_level = ⌈log₂(image_size / min_patch_size)⌉
        num_scales = max_level + 1
        patch_sizes = (min_patch_size × 2^i)_{i=0}^{max_level}
        grid_size = image_size / min_patch_size
        num_tokens = grid_size²

    Hilbert 策略 (自动选择):
        - grid_size = 2^k: 使用标准 Hilbert 曲线
        - padding_ratio ≥ 4/3: 使用 Pseudo-Hilbert
    """

    # ========== 基础参数 (必需) ==========
    image_size: int
    min_patch_size: int = 4

    # ========== Tokenizer 配置 ==========
    tokenizer_type: TokenizerType = 'streaming_v3'

    # ========== 推导参数 (自动计算) ==========
    max_level: int = field(init=False)
    num_scales: int = field(init=False)
    patch_sizes: Tuple[int, ...] = field(init=False)
    grid_size: int = field(init=False)
    num_tokens: int = field(init=False)
    uses_pseudo_hilbert: bool = field(init=False)

    # 混合策略阈值: ρ* = 4/3
    PADDING_RATIO_THRESHOLD: float = 4 / 3

    def __post_init__(self) -> None:
        """从基础参数推导所有配置."""
        # 验证约束
        if self.image_size <= 0 or self.min_patch_size <= 0:
            raise ValueError(
                f"image_size ({self.image_size}) 和 min_patch_size ({self.min_patch_size}) 必须为正数"
            )

        if self.image_size % self.min_patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) 必须能被 min_patch_size ({self.min_patch_size}) 整除"
            )

        # 计算 grid_size
        ratio = self.image_size // self.min_patch_size
        if ratio <= 0:
            raise ValueError(f"grid_size = {ratio} 必须为正数")

        # 计算推导参数
        if ratio > 1:
            max_level = math.ceil(math.log2(ratio))
        else:
            max_level = 0

        object.__setattr__(self, 'max_level', max_level)
        object.__setattr__(self, 'num_scales', max_level + 1)
        object.__setattr__(self, 'patch_sizes', tuple(
            self.min_patch_size * (2 ** i) for i in range(max_level + 1)
        ))
        object.__setattr__(self, 'grid_size', ratio)
        object.__setattr__(self, 'num_tokens', ratio * ratio)

        # 确定 Hilbert 策略
        is_power_of_2 = ratio > 0 and (ratio & (ratio - 1) == 0)
        if is_power_of_2:
            uses_pseudo = False
        else:
            n = 1
            while n < ratio:
                n *= 2
            padding_ratio = (n * n) / (ratio * ratio)
            uses_pseudo = padding_ratio >= self.PADDING_RATIO_THRESHOLD

        object.__setattr__(self, 'uses_pseudo_hilbert', uses_pseudo)

    def scale_to_depth(self, scale_idx: int) -> int:
        """将尺度索引转换为四叉树深度."""
        return self.max_level - scale_idx

    def depth_to_scale(self, depth: int) -> int:
        """将四叉树深度转换为尺度索引."""
        return self.max_level - depth

    def patch_size_at_scale(self, scale_idx: int) -> int:
        """获取指定尺度的 patch 大小."""
        return self.patch_sizes[scale_idx]

    def grid_size_at_scale(self, scale_idx: int) -> int:
        """获取指定尺度的网格边长."""
        return self.image_size // self.patch_sizes[scale_idx]

    def __repr__(self) -> str:
        hilbert_strategy = "Pseudo-Hilbert" if self.uses_pseudo_hilbert else "Standard Hilbert"
        is_power_of_2 = self.grid_size > 0 and (self.grid_size & (self.grid_size - 1) == 0)
        grid_note = "" if is_power_of_2 else f" (非 2^k, 使用 {hilbert_strategy})"

        return (
            f"FractalConfig(\n"
            f"  # Geometry\n"
            f"  image_size={self.image_size}, min_patch_size={self.min_patch_size}\n"
            f"  max_level={self.max_level}, num_scales={self.num_scales}\n"
            f"  patch_sizes={self.patch_sizes}\n"
            f"  grid_size={self.grid_size}{grid_note}, num_tokens={self.num_tokens}\n"
            f"  # Hilbert Strategy\n"
            f"  uses_pseudo_hilbert={self.uses_pseudo_hilbert}\n"
            f"  # Tokenizer\n"
            f"  tokenizer_type='streaming_v3' (Variable Depth, 推荐)\n"
            f"  # Hilbert Bias: LCA mode\n"
            f")"
        )


def create_fractal_config(
    image_size: int,
    min_patch_size: int = 4,
    **kwargs,
) -> FractalConfig:
    """便捷函数：创建 FractalConfig."""
    return FractalConfig(image_size, min_patch_size, **kwargs)
