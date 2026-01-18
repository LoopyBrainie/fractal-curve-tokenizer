# -*- coding: utf-8 -*-
"""
Analysis utilities and visualization entry points.
"""

from .visualization import *
from .visualization.hilbert_splitter import demo_hilbert_splitter

__all__ = [
    # visualization exports
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
    "demo_hilbert_splitter",
]
