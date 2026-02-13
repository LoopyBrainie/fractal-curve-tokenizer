"""L2 Component Layer - Attention, Splitters, FFN, Embeddings

This module re-exports component layer functionality from subdirectories.
"""
from .attention import (
    HilbertAwareMultiScaleAttention,
    HilbertBiasBase,
    LCAHilbertBias,
)
from .splitters import (
    GumbelTopKSplitter,
    TensorSplitResult,
    GumbelTopKResult,
    DepthMonitor,  # I111-6: 深度分布监控
    NeighborAwareSplitter,
    NeighborAwareSplitterConfig,
    LocalityConsistencyLoss,
    SemanticRedundancySplitter,
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
    "HilbertAwareMultiScaleAttention",
    "HilbertBiasBase",
    "LCAHilbertBias",
    # Splitters
    "GumbelTopKSplitter",
    "TensorSplitResult",
    "GumbelTopKResult",
    "DepthMonitor",  # I111-6: 深度分布监控
    "NeighborAwareSplitter",
    "NeighborAwareSplitterConfig",
    "LocalityConsistencyLoss",
    "SemanticRedundancySplitter",
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
