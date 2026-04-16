"""L4 Application Layer - Main model

This module re-exports the main FractalCurveViT model and dual-path variants.
"""

from .fractal_vit import (
    FractalCurveViT,
    TrainingStats,
    DualPathFractalViT,  # 向后兼容别名 (I162-1)
    FractalCurveViTV2,    # V2 版本
    TrainingStatsV2,      # 向后兼容别名 (I162-1)
)

# Pattern Encoder 类 - 从 core.pattern_encoder 导入
from vit_pytorch.core.pattern_encoder import (
    HilbertPatternEncoder,
    HilbertPatternEncoderLight,
    create_hilbert_pattern_encoder,
)

__all__ = [
    # Original
    "FractalCurveViT",
    "TrainingStats",
    # V2 (Dual-Path)
    "DualPathFractalViT",
    "FractalCurveViTV2",
    "TrainingStatsV2",
    # Pattern Encoder
    "HilbertPatternEncoder",
    "HilbertPatternEncoderLight",
    "create_hilbert_pattern_encoder",
]
