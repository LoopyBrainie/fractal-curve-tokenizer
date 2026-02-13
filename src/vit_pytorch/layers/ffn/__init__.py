"""Feed-forward network modules

This module exports FFN components for the fractal VI model.
"""
from .swiglu import (
    SwiGLUFFN,
    AdaptiveFractalFeedForward,
    FFNType,
)

__all__ = [
    "SwiGLUFFN",
    "AdaptiveFractalFeedForward",
    "FFNType",
]
