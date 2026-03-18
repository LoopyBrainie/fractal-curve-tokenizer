"""Splitter modules - Adaptive token splitting strategies

This module exports various splitter components for token selection.

Splitter Status (as of I164):
    - DEFAULT: HilbertOptimalSplitter (H1SS) - 推荐使用
    - COMPONENT: HilbertOrderedEntmaxSplitter (H-Entmax) - 基础组件
    - DEPRECATED: GumbelTopKSplitter, DeterministicNeighborSplitter, SemanticRedundancySplitter
"""
from .gumbel_topk import (
    GumbelTopKSplitter,  # DEPRECATED
    TensorSplitResult,
    GumbelTopKResult,
    create_gumbel_topk_from_config,
    DepthMonitor,  # I111-6: 深度分布监控
)
from .deterministic_neighbor import (
    DeterministicNeighborSplitter,  # DEPRECATED
    DeterministicNeighborSplitterConfig,
    HilbertNeighborMatrix,
    HilbertAwareSimilarity,
    create_deterministic_neighbor_splitter,
)
from .semantic_redundancy import (
    SemanticRedundancySplitter,  # DEPRECATED
    LookAheadHead,  # 迁移到H1SS，保留导出
)
from .hilbert_optimal_splitter import (
    HilbertOptimalSplitter,  # DEFAULT - 推荐使用
    HilbertOptimalSplitterConfig,
    compute_locality_score,
    compute_determinism_score,
    compute_gradient_coverage as hilbert_gradient_coverage,
    compute_tree_consistency,
    compute_consistency_stats,
)
from .hilbert_entmax import (
    HilbertOrderedEntmaxSplitter,  # 基础组件
    HilbertLocalComplexity,
    entmax_1_5,
    entmax,
    compute_gradient_coverage,
)

__all__ = [
    # 推荐使用
    "HilbertOptimalSplitter",
    "HilbertOptimalSplitterConfig",
    # 基础组件
    "HilbertOrderedEntmaxSplitter",
    "HilbertLocalComplexity",
    "entmax_1_5",
    "entmax",
    # 工具函数
    "compute_locality_score",
    "compute_determinism_score",
    "compute_gradient_coverage",
    "compute_tree_consistency",
    "compute_consistency_stats",
    # DEPRECATED (保留向后兼容)
    "GumbelTopKSplitter",
    "TensorSplitResult",
    "GumbelTopKResult",
    "create_gumbel_topk_from_config",
    "DepthMonitor",
    "DeterministicNeighborSplitter",
    "DeterministicNeighborSplitterConfig",
    "HilbertNeighborMatrix",
    "HilbertAwareSimilarity",
    "create_deterministic_neighbor_splitter",
    "SemanticRedundancySplitter",
    "LookAheadHead",  # 迁移到H1SS
]
