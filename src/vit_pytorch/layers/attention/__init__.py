"""Attention modules - Hilbert-aware multi-scale attention

This module exports attention components for the fractal ViT model.
"""
from .hilbert_bias import (
    HilbertAwareMultiScaleAttention,
    HilbertBiasBase,
    LCAHilbertBias,
)
from .hierarchical_soft_hard import (
    HierarchicalSoftHardAttention,
    HierarchicalMaskBuilder,
    HierarchicalAttentionConfig,
)

__all__ = [
    "HilbertAwareMultiScaleAttention",
    "HilbertBiasBase",
    "LCAHilbertBias",
    "HierarchicalSoftHardAttention",
    "HierarchicalMaskBuilder",
    "HierarchicalAttentionConfig",
]
