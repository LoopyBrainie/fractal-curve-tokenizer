"""L4 Application Layer - Main model

This module re-exports the main FractalCurveViT model and dual-path variants.
"""

from .fractal_vit import FractalCurveViT, TrainingStats
from .dual_path_fractal_vit import (
    DualPathFractalViT,
    FractalCurveViTV2,
    TrainingStatsV2,
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
