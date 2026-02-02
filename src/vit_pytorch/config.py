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

# 导入默认常量（作为配置默认值）
from .constants import (
    LEARNABLE_QUOTA_ENABLED,
    QUOTA_INIT_LOGITS,
    QUOTA_ENTROPY_WEIGHT,
    HILBERT_BIAS_SCALE,
    LEVEL_BIAS_SCALE,
    SPLITTER_TEMP_START,
    SPLITTER_TEMP_END,
    # I33: 相对预算常量
    K_COVERAGE_BASE,
    K_COVERAGE_MIN,
    K_COVERAGE_MAX_HARD,
    K_ADAPTIVE_REFERENCE_SIZE,
    K_MAX_HARD_LIMIT,
    K_MIN_HARD_LIMIT,
    # I109-4: Elastic Budget 常量
    ELASTIC_LAMBDA_TARGET,
    ELASTIC_LAMBDA_BOUNDARY,
)


# ==================== 类型别名 ====================

# 温度退火调度类型
AnnealSchedule = Literal['linear', 'exponential', 'cosine']
# Tokenizer 类型 (streaming_v2 已移除)
TokenizerType = Literal['streaming_v1', 'streaming_v3']


# ==================== Splitter 配置 ====================

@dataclass
class SplitterConfig:
    """
    GumbelTopKSplitter 配置

    暴露参数 (基于数学形式化分析):
    - enable_learnable_quota: impact=1.0 (离散) → 必须暴露
    - quota_init_logits: impact=0.01×0.1=0.001 → 建议暴露
    - quota_min_per_depth: impact=0.1×0.05=0.005 → 可选暴露

    I33: 相对预算设计
    ================
    绝对预算 K=64 的问题:
    - 64×64: 覆盖率 = 64/256 = 25% (过高)
    - 224×224: 覆盖率 = 64/4096 = 1.6% (不足)
    - 512×512: 覆盖率 = 64/16384 = 0.4% (严重不足)

    相对预算公式:
    - K_min = max(K_min_abs, α × N)
    - K_max = min(K_max_hard, β × N)
    - β(H, W) = β_0 × sqrt(min(H, W) / 224) 实现尺度不变性
    """
    # 核心架构参数
    feature_dim: int = 256
    min_patch_size: int = 4
    max_level_limit: int = 8
    hidden_dim: int = 64
    intermediate_dim: int = 64
    pool_size: int = 4

    # Top-K 参数 (I33: 已废弃绝对值，保留兼容)
    K_min: int = 8  # 废弃，使用 K_min_abs
    K_max: int = 64  # 废弃，使用相对预算
    use_dynamic_k: bool = True

    # I33: 相对预算参数 (替代绝对 K_min/K_max)
    # 基准覆盖率 (224×224 目标 ~12%)
    token_coverage_base: float = K_COVERAGE_BASE
    # 最小覆盖率 (防止欠采样)
    token_coverage_min: float = K_COVERAGE_MIN
    # 最大覆盖率上限 (参与自适应计算，约束 β(H,W))
    token_coverage_max: float = K_COVERAGE_MAX_HARD
    # 自适应参考尺寸
    adaptive_reference_size: int = K_ADAPTIVE_REFERENCE_SIZE
    # 绝对下界保护
    K_min_abs: int = K_MIN_HARD_LIMIT
    # 显存硬上限
    K_max_hard: int = K_MAX_HARD_LIMIT
    # 是否启用自适应覆盖率
    use_adaptive_coverage: bool = True

    # 正则化参数
    dropout: float = 0.1

    # I109-4: Elastic Budget 目标导向损失参数
    elastic_lambda_target: float = ELASTIC_LAMBDA_TARGET
    elastic_lambda_boundary: float = ELASTIC_LAMBDA_BOUNDARY

    # I30-10: 配额参数暴露
    enable_learnable_quota: bool = LEARNABLE_QUOTA_ENABLED
    quota_init_logits: Optional[Tuple[float, ...]] = None  # None = 使用默认
    quota_min_per_depth: int = 2  # DEPRECATED: I96-7, 使用 quota_min_ratio 替代
    quota_entropy_weight: float = QUOTA_ENTROPY_WEIGHT

    # I30-10: 冻结控制
    freeze_quota: bool = False

    def get_quota_init_tensor(self, D: int) -> Tuple[float, ...]:
        """获取适合给定深度 D 的初始化 logits"""
        if self.quota_init_logits is not None:
            # 使用用户提供的初始化
            init = list(self.quota_init_logits)
            if len(init) < D:
                # 扩展为均匀分布
                init.extend([0.0] * (D - len(init)))
            elif len(init) > D:
                # 截断
                init = init[:D]
            return tuple(init)
        else:
            # 使用默认常量
            if D <= len(QUOTA_INIT_LOGITS):
                return QUOTA_INIT_LOGITS[:D]
            else:
                # 扩展为均匀分布
                base = list(QUOTA_INIT_LOGITS)
                base.extend([0.0] * (D - len(base)))
                return tuple(base)

    # I24-4: 边界条件验证
    def validate(self) -> None:
        """验证配置参数的有效性"""
        if self.max_level_limit < 2:
            raise ValueError(
                "I24-4: max_level_limit >= 2 是推荐配置。"
                f"当前 max_level_limit={self.max_level_limit} 是边界情况，"
                "支持但可能导致不平衡的 token 分布。"
            )


# ==================== Attention 配置 ====================

@dataclass
class AttentionConfig:
    """
    HilbertAwareAttention 配置

    暴露参数:
    - lca_temperature: impact>0.5 → 必须暴露
    - learnable_temperature: impact=1.0 → 必须暴露
    - bias_scale: impact=0.1 → 可选暴露
    """
    # 注意力维度
    dim: int = 256
    heads: int = 8
    head_dim: Optional[int] = None  # None = dim // heads

    # Hilbert 偏置参数
    lca_bias: bool = True
    max_level: int = 8
    lca_embedding_dim: int = 128
    lca_temperature: Optional[float] = None  # None = 自动
    learnable_temperature: bool = True
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
                    'quota_min_per_depth': self.tokenizer_config.splitter_config.quota_min_per_depth,
                    'K_min': self.tokenizer_config.splitter_config.K_min,
                    'K_max': self.tokenizer_config.splitter_config.K_max,
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
    # 深度缩放范围 (hierarchical_depth_scale 初始化边界)
    hierarchical_scale_bounds: tuple[float, float] = (0.5, 1.5)
    # 偏置缩放初始化: raw = log(scale)，对应 scale=0.1 和 scale=0.05
    hilbert_bias_init: float = -2.302585  # ln(0.1)
    level_bias_init: float = -2.995732    # ln(0.05)
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
            'hierarchical_scale_bounds': self.hierarchical_scale_bounds,
            'hilbert_bias_init': self.hilbert_bias_init,
            'level_bias_init': self.level_bias_init,
            'energy_injection_enabled': self.energy_injection_enabled,
        }


# ==================== 工厂函数 ====================

def create_splitter_config(
    enable_learnable_quota: Optional[bool] = None,
    quota_init_logits: Optional[Tuple[float, ...]] = None,
    quota_min_per_depth: Optional[int] = None,
    K_min: Optional[int] = None,
    K_max: Optional[int] = None,
    freeze_quota: Optional[bool] = None,
    # I33: 相对预算参数
    token_coverage_base: Optional[float] = None,
    token_coverage_min: Optional[float] = None,
    token_coverage_max: Optional[float] = None,  # I109-3: 参与自适应计算
    adaptive_reference_size: Optional[int] = None,
    K_min_abs: Optional[int] = None,
    K_max_hard: Optional[int] = None,
    use_adaptive_coverage: Optional[bool] = None,
    **kwargs,
) -> SplitterConfig:
    """
    工厂函数: 创建 SplitterConfig（用于 CLI 参数解析）

    数学保证:
        配置参数影响度经过消融实验验证

    I33: 相对预算设计
    ================
    相对预算公式:
        K_min = max(K_min_abs, α × N)
        K_max = min(K_max_hard, β × N)
        β(H, W) = β_0 × sqrt(min(H, W) / 224)
    """
    config = SplitterConfig()

    if enable_learnable_quota is not None:
        config.enable_learnable_quota = enable_learnable_quota
    if quota_init_logits is not None:
        config.quota_init_logits = quota_init_logits
    if quota_min_per_depth is not None:
        config.quota_min_per_depth = quota_min_per_depth
    if K_min is not None:
        config.K_min = K_min
    if K_max is not None:
        config.K_max = K_max
    if freeze_quota is not None:
        config.freeze_quota = freeze_quota

    # I33: 相对预算参数
    if token_coverage_base is not None:
        config.token_coverage_base = token_coverage_base
    if token_coverage_min is not None:
        config.token_coverage_min = token_coverage_min
    if token_coverage_max is not None:
        config.token_coverage_max = token_coverage_max  # I109-3
    if adaptive_reference_size is not None:
        config.adaptive_reference_size = adaptive_reference_size
    if K_min_abs is not None:
        config.K_min_abs = K_min_abs
    if K_max_hard is not None:
        config.K_max_hard = K_max_hard
    if use_adaptive_coverage is not None:
        config.use_adaptive_coverage = use_adaptive_coverage

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

        if self.tokenizer_type == 'streaming_v3':
            tokenizer_info = f"  tokenizer_type='{self.tokenizer_type}' (Variable Depth, 推荐)\n"
        else:
            tokenizer_info = f"  tokenizer_type='{self.tokenizer_type}' (单尺度)\n"

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
            f"{tokenizer_info}"
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
