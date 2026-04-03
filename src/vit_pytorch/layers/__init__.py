"""L2 Component Layer - Attention, Splitters, FFN, Embeddings

This module re-exports component layer functionality from subdirectories.
"""
from .attention import (
    ManifoldNativeAttention,
)
from .splitters import (
    HilbertOptimalSplitter,
    HilbertOptimalSplitterConfig,
    HilbertOrderedEntmaxSplitter,
    HilbertLocalComplexity,
    entmax_1_5,
    entmax,
)
from .ffn import (
    SwiGLUFFN,
    AdaptiveFractalFeedForward,
    FFNType,
)
from .embeddings import (
    FractalPathEmbedding,
    HierarchicalAttentionBias,
    VectorizedPathEncoder,
    FractalPositionEmbedding,
    MultiScalePatchEncoder,
    HilbertNativePatchEmbed,
)
# Note: CoreSplitter, AnnealingSplitter, MetricsSplitter, SplitResult
# are imported from vit_pytorch.modules.base_splitter in the modules layer

__all__ = [
    # Attention
    "ManifoldNativeAttention",
    # Splitters
    "HilbertOptimalSplitter",
    "HilbertOptimalSplitterConfig",
    "HilbertOrderedEntmaxSplitter",
    "HilbertLocalComplexity",
    "entmax_1_5",
    "entmax",
    # FFN
    "SwiGLUFFN",
    "AdaptiveFractalFeedForward",
    "FFNType",
    # Embeddings
    "FractalPathEmbedding",
    "HierarchicalAttentionBias",
    "VectorizedPathEncoder",
    "FractalPositionEmbedding",
    "MultiScalePatchEncoder",
    "HilbertNativePatchEmbed",
]
