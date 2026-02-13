"""Splitter modules - Adaptive token splitting strategies

This module exports various splitter components for token selection.
"""
from .gumbel_topk import (
    GumbelTopKSplitter,
    TensorSplitResult,
    GumbelTopKResult,
    create_gumbel_topk_from_config,
)
from .deterministic_neighbor import (
    DeterministicNeighborSplitter,
    DeterministicNeighborSplitterConfig,
    HilbertNeighborMatrix,
    HilbertAwareSimilarity,
    create_deterministic_neighbor_splitter,
    # 向后兼容别名 (I160-1)
    NeighborAwareSplitter,
    NeighborAwareSplitterConfig,
    LocalityConsistencyLoss,
)
from .semantic_redundancy import SemanticRedundancySplitter

__all__ = [
    "GumbelTopKSplitter",
    "TensorSplitResult",
    "GumbelTopKResult",
    "create_gumbel_topk_from_config",
    "DeterministicNeighborSplitter",
    "DeterministicNeighborSplitterConfig",
    "HilbertNeighborMatrix",
    "HilbertAwareSimilarity",
    "create_deterministic_neighbor_splitter",
    # 向后兼容别名 (I160-1)
    "NeighborAwareSplitter",
    "NeighborAwareSplitterConfig",
    "LocalityConsistencyLoss",
    "SemanticRedundancySplitter",
]
