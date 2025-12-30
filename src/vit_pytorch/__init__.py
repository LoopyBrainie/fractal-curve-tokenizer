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
    model_fractal_vit.py    FractalCurveViT

Layer 3 (管道层):
    streaming_tokenizer.py  StreamingFractalTokenizerV3
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
from .attn_hilbert_bias import (
    HilbertAwareMultiScaleAttention,
    HilbertBiasBase,
    LCAHilbertBias,
)
from .ffn_swiglu import AdaptiveFractalFeedForward, FFNType, SwiGLUFFN
from .config_fractal import (
    FractalConfig,
    create_fractal_config,
    AnnealSchedule,
    TokenizerType,
)
from .embed_fractal_path import (
    FractalPathEmbedding,
    HierarchicalAttentionBias,
    VectorizedPathEncoder,
)
from .model_fractal_vit import FractalCurveViT
from .curve_hilbert import (
    HilbertCurve,
    PseudoHilbertCurve,
    get_quadrant_order,
    hilbert_distance_to_xy,
    xy_to_hilbert_distance,
)
from .embed_fractal_position import FractalPositionEmbedding
from .tokenizer_streaming import StreamingFractalTokenizerV3
from .curve_hilbert_indexer import (
    HilbertIndexer,
    HilbertPathCache,
)
from .embed_multiscale_patch import MultiScalePatchEncoder
from .base_tokenizer import BaseTokenizer, BaseTokenProcessor, TokenSequence, TokenizerOutput
from .block_transformer import FractalTransformer, FractalTransformerBlock
from .utils import extract_depths, normalize_levels_info
from .split_adaptive import (
    AdaptiveSplitConfig,
    TemperatureScheduler,
    LearnableSplitter,
    SplitScheme,
    TensorSplitResult,  # P9-1: 纯张量分割结果
)


__all__ = [
    # === 主要模型 ===
    "StreamingFractalTokenizerV3",  # Variable Depth Tokenizer
    "FractalCurveViT",
    # === 配置与路径编码 ===
    "FractalConfig",
    "create_fractal_config",
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
    "HilbertBiasBase",
    "LCAHilbertBias",
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
    # === 分割器 (P7/P8/P9) ===
    "AdaptiveSplitConfig",
    "TemperatureScheduler",
    "LearnableSplitter",
    "SplitScheme",
    "TensorSplitResult",  # P9-1: 完全向量化分割结果
]
