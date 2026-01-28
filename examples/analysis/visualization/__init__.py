# -*- coding: utf-8 -*-
"""
Visualization module exports.

New expert-level visualizations:
- feature_manifold.py: t-SNE/UMAP cross-depth feature analysis
- hilbert_attention.py: Hilbert-space attention distribution
- efficiency_analysis.py: FLOPs heatmap and Pareto frontier
"""

from .architecture_diagram import plot_architecture_comparison, plot_model_components
from .architecture_design import (
    plot_token_count_range,
    plot_position_encoding_comparison,
    plot_multi_scale_representation,
)
from .attention_patterns import (
    plot_attention_heatmap,
    plot_depth_attention_matrix,
    plot_head_comparison,
)
from .depth_encoding import (
    plot_depth_embedding_similarity,
    plot_depth_hierarchy,
    plot_pca_projection,
    plot_depth_clustermap,
)
from .feature_manifold import (
    plot_cross_depth_feature_manifold,
    plot_depth_feature_statistics,
)
from .hilbert_attention import (
    plot_hilbert_attention_map,
    compare_attention_patterns,
)
from .hilbert_splitter import (
    visualize_quadtree_split,
    plot_hilbert_curve,
    compare_orderings,
    visualize_splitter_decision,
    visualize_mixed_depth_regions,
)
from .metrics_dashboard import (
    plot_metrics_dashboard,
    plot_training_curves,
    plot_performance_comparison,
)
from .spatial_locality import (
    plot_hilbert_vs_raster,
    plot_locality_preservation,
    animate_hilbert_curve,
)
from .tokenization_comparison import (
    plot_tokenization_comparison,
    plot_depth_distribution,
)
from .efficiency_analysis import (
    plot_adaptive_flops_heatmap,
    plot_pareto_frontier,
    plot_computation_comparison,
)

__all__ = [
    # Architecture
    "plot_architecture_comparison",
    "plot_model_components",
    "plot_token_count_range",
    "plot_position_encoding_comparison",
    "plot_multi_scale_representation",
    # Attention
    "plot_attention_heatmap",
    "plot_depth_attention_matrix",
    "plot_head_comparison",
    # Depth Encoding
    "plot_depth_embedding_similarity",
    "plot_depth_hierarchy",
    "plot_pca_projection",
    "plot_depth_clustermap",
    # Feature Manifold (NEW)
    "plot_cross_depth_feature_manifold",
    "plot_depth_feature_statistics",
    # Hilbert Attention (NEW)
    "plot_hilbert_attention_map",
    "compare_attention_patterns",
    # Hilbert Splitter
    "visualize_quadtree_split",
    "plot_hilbert_curve",
    "compare_orderings",
    "visualize_splitter_decision",
    "visualize_mixed_depth_regions",
    # Metrics
    "plot_metrics_dashboard",
    "plot_training_curves",
    "plot_performance_comparison",
    # Spatial Locality
    "plot_hilbert_vs_raster",
    "plot_locality_preservation",
    "animate_hilbert_curve",
    # Tokenization
    "plot_tokenization_comparison",
    "plot_depth_distribution",
    # Efficiency Analysis (NEW)
    "plot_adaptive_flops_heatmap",
    "plot_pareto_frontier",
    "plot_computation_comparison",
]
