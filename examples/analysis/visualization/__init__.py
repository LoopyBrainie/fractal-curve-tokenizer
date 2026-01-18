# -*- coding: utf-8 -*-
"""
Visualization module exports.
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

__all__ = [
    "plot_architecture_comparison",
    "plot_model_components",
    "plot_token_count_range",
    "plot_position_encoding_comparison",
    "plot_multi_scale_representation",
    "plot_attention_heatmap",
    "plot_depth_attention_matrix",
    "plot_head_comparison",
    "plot_depth_embedding_similarity",
    "plot_depth_hierarchy",
    "plot_pca_projection",
    "plot_depth_clustermap",
    "visualize_quadtree_split",
    "plot_hilbert_curve",
    "compare_orderings",
    "visualize_splitter_decision",
    "visualize_mixed_depth_regions",
    "plot_metrics_dashboard",
    "plot_training_curves",
    "plot_performance_comparison",
    "plot_hilbert_vs_raster",
    "plot_locality_preservation",
    "animate_hilbert_curve",
    "plot_tokenization_comparison",
    "plot_depth_distribution",
]
