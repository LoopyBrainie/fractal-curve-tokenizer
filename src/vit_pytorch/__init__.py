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
    fractal_vit.py          NextGenerationFractalViT, SimpleFractalViT

Layer 3 (管道层):
    streaming_tokenizer.py  StreamingFractalTokenizer, V2
    transformer.py          EnhancedFractalTransformer

Layer 2 (组件层):
    attention.py            HilbertAwareMultiScaleAttention
    feedforward.py          SwiGLUFFN, AdaptiveFractalFeedForward
    positional.py           AdvancedFractalPositionEmbedding

Layer 1 (基础层):
    hilbert.py              HilbertCurve (H: d ↔ (x,y))
    tokenization.py         BaseTokenizer, TokenizerOutput
    constants.py            超参数默认值
    utils.py                工具函数

废弃模块 (_deprecated/):
    fractal_curve_tokenizer.py  BFS + REINFORCE (v1.0 移除)
    token_processor.py          功能已集成 (v1.0 移除)
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
)
from .fractal_path import (
    FractalPathEmbedding,
    HierarchicalAttentionBias,
    VectorizedPathEncoder,
)
from .fractal_vit import (
    NextGenerationFractalViT,
    SimpleFractalViT,
)
from .hilbert import (
    HilbertCurve,
    PseudoHilbertCurve,
    get_quadrant_order,
    hilbert_distance_to_xy,
    xy_to_hilbert_distance,
)
from .positional import AdvancedFractalPositionEmbedding
from .streaming_tokenizer import (
    HilbertIndexer,
    HilbertPathCache,
    MultiScalePatchEncoder,
    StreamingFractalTokenizer,
    StreamingFractalTokenizerV2,
)
from .tokenization import BaseTokenizer, BaseTokenProcessor, TokenSequence, TokenizerOutput
from .transformer import EnhancedFractalTransformer, EnhancedFractalTransformerBlock
from .utils import extract_depths, normalize_levels_info


# === 废弃模块的延迟导入 ===
def __getattr__(name: str):
    """延迟导入废弃模块，并发出警告。"""
    import warnings
    
    deprecated_map = {
        "FractalHilbertTokenizer": "fractal_curve_tokenizer",
        "EnhancedFractalTokenProcessor": "token_processor",
        "LearnableSplitDecision": "fractal_curve_tokenizer",
        "MiniCNN": "fractal_curve_tokenizer",
        "PatchInfo": "fractal_curve_tokenizer",
    }
    
    if name in deprecated_map:
        warnings.warn(
            f"'{name}' 已废弃，将在 v1.0 移除。"
            f"请使用 StreamingFractalTokenizer 替代。"
            f"详见 docs/MATHEMATICAL_FORMALIZATION.md",
            DeprecationWarning,
            stacklevel=2,
        )
        module_name = deprecated_map[name]
        from importlib import import_module
        module = import_module(f"vit_pytorch._deprecated.{module_name}")
        return getattr(module, name)
    
    raise AttributeError(f"module 'vit_pytorch' has no attribute '{name}'")


__all__ = [
    # === 推荐使用 (v0.5.0+) ===
    "StreamingFractalTokenizer",
    "StreamingFractalTokenizerV2",
    "NextGenerationFractalViT",
    "SimpleFractalViT",
    # === 配置与路径编码 (v0.6.0+) ===
    "FractalConfig",
    "create_fractal_config",
    "BiasMode",
    "AnnealSchedule",
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
    "EnhancedFractalTransformer",
    "EnhancedFractalTransformerBlock",
    "HilbertAwareMultiScaleAttention",
    "LCAHilbertBias",
    "LowRankHilbertBias",
    "HierarchicalHilbertBias",
    "AdaptiveFractalFeedForward",
    "SwiGLUFFN",
    "FFNType",
    # === 位置编码 ===
    "AdvancedFractalPositionEmbedding",
    # === 工具函数 ===
    "extract_depths",
    "normalize_levels_info",
    "get_quadrant_order",
    "hilbert_distance_to_xy",
    "xy_to_hilbert_distance",
    # === 废弃 (v1.0 移除，通过 __getattr__ 延迟导入) ===
    # 这些仍然可以通过 from vit_pytorch import FractalHilbertTokenizer 访问
    # 但会发出 DeprecationWarning
]
