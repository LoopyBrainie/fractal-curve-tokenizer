"""Attention modules - Manifold-Native multi-scale attention."""

from .manifold_attention import ManifoldNativeAttention, ManifoldAttention
from .eahbp_attention import EAHBPAttention, EAHBPAttentionConfig

__all__ = [
    "ManifoldNativeAttention",
    "ManifoldAttention",
    # v1.3 STANDARD: EAHBP (alpha-c-rev4)
    "EAHBPAttention",
    "EAHBPAttentionConfig",
]
