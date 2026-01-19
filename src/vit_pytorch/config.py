# -*- coding: utf-8 -*-
"""
Fractal ViT 统一配置模块

组件配置原则:
1. 组件自治: 每个组件接收完整配置，独立运行
2. 精选暴露: 只暴露高影响参数 (impact > 0.1)
3. 可组合: 支持 YAML 配置 + CLI 参数覆盖

数学形式化
==========
参数影响度度量:
    impact(p, D) = ||∂L/D||₂ × Var_D(p)

暴露决策公式:
    E(p) = EXPOSE     if impact(p) > 0.5
         = OPTIONAL   if 0.1 < impact(p) ≤ 0.5
         = HIDDEN     if impact(p) ≤ 0.1

日期: 2026-01-17
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

# 导入默认常量（作为配置默认值）
from .constants import (
    LEARNABLE_QUOTA_ENABLED,
    QUOTA_MIN_PER_DEPTH,
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
)


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
    max_depth_limit: int = 8
    hidden_dim: int = 64
    intermediate_dim: int = 64
    pool_size: int = 4

    # Top-K 参数 (I33: 已废弃绝对值，保留兼容)
    K_min: int = 8  # 废弃，使用 K_min_abs
    K_max: int = 64  # 废弃，使用相对预算
    use_dynamic_k: bool = True

    # I33: 相对预算参数 (替代绝对 K_min/K_max)
    # 基准覆盖率 (224×224 目标 5%)
    token_coverage_base: float = K_COVERAGE_BASE
    # 最小覆盖率 (防止欠采样)
    token_coverage_min: float = K_COVERAGE_MIN
    # 最大覆盖率硬上限 (防止 OOM)
    token_coverage_max_hard: float = K_COVERAGE_MAX_HARD
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

    # I30-10: 配额参数暴露
    enable_learnable_quota: bool = LEARNABLE_QUOTA_ENABLED
    quota_init_logits: Optional[Tuple[float, ...]] = None  # None = 使用默认
    quota_min_per_depth: int = QUOTA_MIN_PER_DEPTH
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
    max_depth: int = 8
    lca_embedding_dim: int = 128
    lca_temperature: Optional[float] = None  # None = 自动
    learnable_temperature: bool = True

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
    max_depth: int = 8

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
    depth: int = 8
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
                'max_depth': self.tokenizer_config.max_depth,
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
                'depth': self.transformer_config.depth,
                'heads': self.transformer_config.heads,
                'mlp_ratio': self.transformer_config.mlp_ratio,
                'use_swiglu': self.transformer_config.use_swiglu,
                'dropout': self.transformer_config.dropout,
                'drop_path': self.transformer_config.drop_path,
            },
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
    token_coverage_max_hard: Optional[float] = None,
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
    if token_coverage_max_hard is not None:
        config.token_coverage_max_hard = token_coverage_max_hard
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
