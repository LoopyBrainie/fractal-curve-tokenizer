# -*- coding: utf-8 -*-
"""
Fractal Curve ViT - 分形曲线视觉 Transformer

项目目标: 验证 Hilbert 曲线分形 tokenization 对 ViT 的可行性

数学形式化
============

核心流程:
    I → T(I) → E_pos → Transformer → Pool → MLP → ŷ
    
其中:
    I ∈ R^{B × C × H × W}     输入图像
    T: I → (T, L)             Tokenization
    T ∈ R^{B × N × D}         Token 嵌入
    L ∈ Z^{B × N}             层级信息

模块层级
---------
Layer 4 (应用层):
    fractal_vit.py          FractalCurveViT

Layer 3 (管道层):
    streaming_tokenizer.py  StreamingFractalTokenizer, V2
    transformer.py          FractalTransformer

Layer 2 (组件层):
    attention.py            HilbertAwareMultiScaleAttention
    feedforward.py          SwiGLUFFN, AdaptiveFractalFeedForward
    positional.py           FractalPositionEmbedding

Layer 1 (基础层):
    hilbert.py              HilbertCurve, PseudoHilbertCurve
    tokenization.py         BaseTokenizer, TokenizerOutput
    constants.py            超参数默认值
    utils.py                工具函数
"""

# === 推荐组件 ===
from .attention import (
    HilbertAwareMultiScaleAttention,
    LCAHilbertBias,
    LowRankHilbertBias,
    HierarchicalHilbertBias,
)
from .feedforward import AdaptiveFractalFeedForward, FFNType, SwiGLUFFN
from .fractal_config import (
    FractalConfig,
    create_fractal_config,
    BiasMode,
    AnnealSchedule,
    TokenizerType,
)
from .fractal_path import (
    FractalPathEmbedding,
    HierarchicalAttentionBias,
    VectorizedPathEncoder,
)
from .fractal_vit import FractalCurveViT
from .hilbert import (
    HilbertCurve,
    PseudoHilbertCurve,
    get_quadrant_order,
    hilbert_distance_to_xy,
    xy_to_hilbert_distance,
)
from .positional import FractalPositionEmbedding
from .streaming_tokenizer import (
    CrossScaleAttention,
    HilbertIndexer,
    HilbertPathCache,
    MultiScalePatchEncoder,
    StreamingFractalTokenizer,
    StreamingFractalTokenizerV2,
    StreamingFractalTokenizerV3,
)
from .tokenization import BaseTokenizer, BaseTokenProcessor, TokenSequence, TokenizerOutput
from .transformer import FractalTransformer, FractalTransformerBlock
from .utils import extract_depths, normalize_levels_info


__all__ = [
    # === 主要模型 ===
    "StreamingFractalTokenizer",
    "StreamingFractalTokenizerV2",  # 已废弃，请使用 V3
    "StreamingFractalTokenizerV3",  # 推荐
    "CrossScaleAttention",
    "FractalCurveViT",
    # === 配置与路径编码 ===
    "FractalConfig",
    "create_fractal_config",
    "BiasMode",
    "AnnealSchedule",
    "TokenizerType",
    "FractalPathEmbedding",
    "HierarchicalAttentionBias",
    "VectorizedPathEncoder",
    # === 核心组件 ===
    "BaseTokenProcessor",
    "BaseTokenizer",
    "TokenSequence",
    "TokenizerOutput",
    "HilbertCurve",
    "PseudoHilbertCurve",
    "HilbertIndexer",
    "HilbertPathCache",
    "MultiScalePatchEncoder",
    # === Transformer 组件 ===
    "FractalTransformer",
    "FractalTransformerBlock",
    "HilbertAwareMultiScaleAttention",
    "LCAHilbertBias",
    "LowRankHilbertBias",
    "HierarchicalHilbertBias",
    "AdaptiveFractalFeedForward",
    "SwiGLUFFN",
    "FFNType",
    # === 位置编码 ===
    "FractalPositionEmbedding",
    # === 工具函数 ===
    "extract_depths",
    "normalize_levels_info",
    "get_quadrant_order",
    "hilbert_distance_to_xy",
    "xy_to_hilbert_distance",
    # === 向后兼容别名 (Backward Compatibility) ===
    "EnhancedFractalTransformer",
    "EnhancedFractalTransformerBlock",
    "AdvancedFractalPositionEmbedding",
    "NextGenerationFractalViT",
]

# === 向后兼容别名 (Backward Compatibility) ===
# 旧类名映射到新类名，保证 API 兼容性
# 计划于 v2.0 版本移除
EnhancedFractalTransformer = FractalTransformer
EnhancedFractalTransformerBlock = FractalTransformerBlock
AdvancedFractalPositionEmbedding = FractalPositionEmbedding
NextGenerationFractalViT = FractalCurveViT
