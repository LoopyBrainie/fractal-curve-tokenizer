"""Embedding modules - Position and patch embeddings

This module exports various embedding components for the fractal VI model.
"""
from .fractal_path import (
    FractalPathEmbedding,
    VectorizedPathEncoder,
    OrientationExtractor,  # Scheme C: 旋转感知提取器
)
from .fractal_position import (
    FractalPositionEmbedding,
    GeometryField,  # Scheme C: 几何流形场
    MultiLayerGeometryField,  # Scheme C: 多层几何流形场
)
from .hilbert_patch import HilbertNativePatchEmbed

__all__ = [
    "FractalPathEmbedding",
    "VectorizedPathEncoder",
    "OrientationExtractor",  # Scheme C
    "FractalPositionEmbedding",
    "GeometryField",  # Scheme C
    "MultiLayerGeometryField",  # Scheme C
    "HilbertNativePatchEmbed",
]
