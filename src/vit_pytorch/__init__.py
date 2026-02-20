# -*- coding: utf-8 -*-
"""
Fractal Curve ViT - 分形曲线视觉 Transformer

模块层级 (重构后)
---------
Layer 4 (应用层):
    models/         FractalCurveViT

Layer 3 (管道层):
    modules/        StreamingFractalTokenizer, FractalTransformer

Layer 2 (组件层):
    layers/attention/       HilbertAwareMultiScaleAttention
    layers/splitters/       GumbelTopKSplitter, DeterministicNeighborSplitter
    layers/ffn/             SwiGLUFFN
    layers/embeddings/      FractalPositionEmbedding, FractalPathEmbedding

Layer 1 (基础层):
    core/          HilbertCurve, LevelsInfo, Constants, Utils
"""

# === L4: Models ===
from .models import (
    FractalCurveViT,
    TrainingStats,
    # V2 models (I162-1)
    DualPathFractalViT,
    FractalCurveViTV2,
    TrainingStatsV2,
)

# === L3: Modules ===
from .modules import (
    StreamingFractalTokenizer,
    StreamingFractalTokenizerV3,  # 向后兼容
    BaseTokenizer,
    BaseTokenProcessor,
    TokenSequence,
    TokenizerOutput,
    FractalTransformer,
    FractalTransformerBlock,
)

# === L2: Layers ===
from .layers import (
    HilbertAwareMultiScaleAttention,
    HilbertBiasBase,
    LCAHilbertBias,
    GumbelTopKSplitter,
    TensorSplitResult,
    GumbelTopKResult,
    DeterministicNeighborSplitter,
    DeterministicNeighborSplitterConfig,
    SemanticRedundancySplitter,
    SwiGLUFFN,
    AdaptiveFractalFeedForward,
    FFNType,
    FractalPathEmbedding,
    HierarchicalAttentionBias,
    VectorizedPathEncoder,
    FractalPositionEmbedding,
    MultiScalePatchEncoder,
    HilbertNativePatchEmbed,
)

# === L2/L3 additional imports (from modules) ===
from .modules import (
    CoreSplitter,
    AnnealingSplitter,
    MetricsSplitter,
    SplitResult,
    SemanticRedundancyLoss,
)

# === L1: Core ===
from .core import (
    HilbertCurve,
    PseudoHilbertCurve,
    HilbertIndexer,
    HilbertPathCache,
    HilbertScanner,
    get_quadrant_order,
    hilbert_distance_to_xy,
    xy_to_hilbert_distance,
    # I162-1: Pattern Encoder
    HilbertPatternEncoder,
    HilbertPatternEncoderLight,
    create_hilbert_pattern_encoder,
    # I162-1: Pattern Plugin
    HilbertPatternPlugin,
    create_hilbert_pattern_plugin,
    FractalConfig,
    SplitterConfig,
    create_fractal_config,
    AnnealSchedule,
    TokenizerType,
    SemanticSplitterConfig,
    create_semantic_splitter_config,
    AttentionConfig,
    TokenizerConfig,
    TransformerConfig,
    FractalViTConfig,
    NeighborAwareSplitterConfig,
    EPS,
    TEMPERATURE_MIN,
    LevelsInfo,
    pair,
    exists,
    default,
    sanitize_tensor,
    create_attention_mask,
    compute_max_depth,
    compute_actual_min_patch,
    compute_patch_sizes,
    compute_depth_distribution,
    compute_total_candidates,
    compute_region_shape_scale,
    compute_shape_scale_similarity,
    compute_normalized_area,
    compute_area_similarity,
    # Splitter Protocol (L1 - Protocol interfaces)
    CoreSplitter,
    AnnealingSplitter,
    MetricsSplitter,
    SplitResult,
    validate_splitter,
)

# === 向后兼容别名 ===
StreamingFractalTokenizerV3 = StreamingFractalTokenizer

__all__ = [
    # === L4: Models ===
    "FractalCurveViT",
    "TrainingStats",
    # V2 models (I162-1)
    "DualPathFractalViT",
    "FractalCurveViTV2",
    "TrainingStatsV2",
    # === L3: Modules ===
    "StreamingFractalTokenizer",
    "StreamingFractalTokenizerV3",  # 向后兼容
    "BaseTokenizer",
    "BaseTokenProcessor",
    "TokenSequence",
    "TokenizerOutput",
    "FractalTransformer",
    "FractalTransformerBlock",
    # === L2: Layers ===
    "HilbertAwareMultiScaleAttention",
    "HilbertBiasBase",
    "LCAHilbertBias",
    "GumbelTopKSplitter",
    "TensorSplitResult",
    "GumbelTopKResult",
    "DeterministicNeighborSplitter",
    "DeterministicNeighborSplitterConfig",
    "SemanticRedundancySplitter",
    "SemanticRedundancyLoss",
    "SwiGLUFFN",
    "AdaptiveFractalFeedForward",
    "FFNType",
    "FractalPathEmbedding",
    "HierarchicalAttentionBias",
    "VectorizedPathEncoder",
    "FractalPositionEmbedding",
    "MultiScalePatchEncoder",
    "HilbertNativePatchEmbed",
    "CoreSplitter",
    "AnnealingSplitter",
    "MetricsSplitter",
    "SplitResult",
    # === L1: Core ===
    "HilbertCurve",
    "PseudoHilbertCurve",
    "HilbertIndexer",
    "HilbertPathCache",
    "HilbertScanner",
    "get_quadrant_order",
    "hilbert_distance_to_xy",
    "xy_to_hilbert_distance",
    # I162-1: Pattern Encoder
    "HilbertPatternEncoder",
    "HilbertPatternEncoderLight",
    "create_hilbert_pattern_encoder",
    # I162-1: Pattern Plugin
    "HilbertPatternPlugin",
    "create_hilbert_pattern_plugin",
    "FractalConfig",
    "SplitterConfig",
    "create_fractal_config",
    "AnnealSchedule",
    "TokenizerType",
    "SemanticSplitterConfig",
    "create_semantic_splitter_config",
    "AttentionConfig",
    "TokenizerConfig",
    "TransformerConfig",
    "FractalViTConfig",
    "NeighborAwareSplitterConfig",
    "EPS",
    "TEMPERATURE_MIN",
    "LevelsInfo",
    "pair",
    "exists",
    "default",
    "sanitize_tensor",
    "create_attention_mask",
    "compute_max_depth",
    "compute_actual_min_patch",
    "compute_patch_sizes",
    "compute_depth_distribution",
    "compute_total_candidates",
    "compute_region_shape_scale",
    "compute_shape_scale_similarity",
    "compute_normalized_area",
    "compute_area_similarity",
    # Splitter Protocol (L1 - Protocol interfaces)
    "CoreSplitter",
    "AnnealingSplitter",
    "MetricsSplitter",
    "SplitResult",
    "validate_splitter",
]
