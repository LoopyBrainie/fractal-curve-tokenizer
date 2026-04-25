"""Embedding modules - Position and patch embeddings

This module exports various embedding components for the fractal VI model.
"""
from .fractal_path import (
    VectorizedPathEncoder,
    OrientationExtractor,  # Scheme C: 旋转感知提取器
)
from .fractal_position import (
    GeometryField,  # Scheme C: 几何流形场
)
from .hilbert_patch import HilbertNativePatchEmbed

__all__ = [
    "VectorizedPathEncoder",
    "OrientationExtractor",  # Scheme C
    "GeometryField",  # Scheme C
    "HilbertNativePatchEmbed",
]
