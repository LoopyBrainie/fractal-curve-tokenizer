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
from .hilbert_optimal_splitter import (
    HilbertOptimalSplitter,
    HilbertOptimalSplitterConfig,
    compute_locality_score,
    compute_determinism_score,
    compute_gradient_coverage,
    compute_tree_consistency,
    compute_consistency_stats,
)

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
    "HilbertOptimalSplitter",
    "HilbertOptimalSplitterConfig",
    "compute_locality_score",
    "compute_determinism_score",
    "compute_gradient_coverage",
    "compute_tree_consistency",
    "compute_consistency_stats",
]
