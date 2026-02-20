"""Splitter modules - Adaptive token splitting strategies

This module exports various splitter components for token selection.
"""
from .gumbel_topk import (
    GumbelTopKSplitter,
    TensorSplitResult,
    GumbelTopKResult,
    create_gumbel_topk_from_config,
    DepthMonitor,  # I111-6: 深度分布监控
)
from .deterministic_neighbor import (
    DeterministicNeighborSplitter,
    DeterministicNeighborSplitterConfig,
    HilbertNeighborMatrix,
    HilbertAwareSimilarity,
    create_deterministic_neighbor_splitter,
)
from .semantic_redundancy import SemanticRedundancySplitter

__all__ = [
    "GumbelTopKSplitter",
    "TensorSplitResult",
    "GumbelTopKResult",
    "create_gumbel_topk_from_config",
    "DepthMonitor",  # I111-6: 深度分布监控
    "DeterministicNeighborSplitter",
    "DeterministicNeighborSplitterConfig",
    "HilbertNeighborMatrix",
    "HilbertAwareSimilarity",
    "create_deterministic_neighbor_splitter",
    "SemanticRedundancySplitter",
]
