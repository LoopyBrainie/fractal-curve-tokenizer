"""Attention modules - Hilbert-aware multi-scale attention

This module exports attention components for the fractal ViT model.
"""
from .manifold_attention import (
    ManifoldNativeAttention,
    ManifoldAttention,
)
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
from .manifold_decoder import (
    GeometricLatentDecoder,
    compute_geometric_features,
    poincare_distance,
    create_geometric_bias,
)
from .hilbert_banded import (
    HilbertBandedAttention,
    HilbertBandedAttentionFused,
    compute_hilbert_bandwidth,
    create_hilbert_band_mask,
    compute_attention_complexity,
)
from .fractal_residuals import (
    ScaleAwareResidual,
    ParentTokenLookup,
    FractalTransformerBlockV2,
    create_fractal_residual_block,
)

__all__ = [
    # Manifold-Native (new implementation)
    "ManifoldNativeAttention",
    "ManifoldAttention",
    # Legacy (for backward compatibility)
    "HilbertAwareMultiScaleAttention",
    "HilbertBiasBase",
    "LCAHilbertBias",
    "HierarchicalSoftHardAttention",
    "HierarchicalMaskBuilder",
    "HierarchicalAttentionConfig",
    "GeometricLatentDecoder",
    "compute_geometric_features",
    "poincare_distance",
    "create_geometric_bias",
    "HilbertBandedAttention",
    "HilbertBandedAttentionFused",
    "compute_hilbert_bandwidth",
    "create_hilbert_band_mask",
    "compute_attention_complexity",
    "ScaleAwareResidual",
    "ParentTokenLookup",
    "FractalTransformerBlockV2",
    "create_fractal_residual_block",
]
