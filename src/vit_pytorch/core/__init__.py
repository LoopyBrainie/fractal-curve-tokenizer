"""L1 Foundation Layer - Core operators and utilities

This module exports core functionality including Hilbert curves, configurations, constants, and utilities.
"""
from .curve_hilbert import (
    HilbertCurve,
    PseudoHilbertCurve,
    HilbertScanner,
    get_quadrant_order,
    hilbert_distance_to_xy,
    xy_to_hilbert_distance,
)
from .hilbert_indexer import HilbertIndexer, HilbertPathCache
# I162-1: Pattern Encoder
from .pattern_encoder import (
    HilbertPatternEncoder,
    HilbertPatternEncoderLight,
    create_hilbert_pattern_encoder,
)
from .config import (
    FractalConfig,
    SplitterConfig,
    create_fractal_config,
    AnnealSchedule,
    TokenizerType,
    SemanticSplitterConfig,
    create_semantic_splitter_config,
    AttentionConfig,
    TokenizerConfig,
    TransformerConfig,
    FractalViTConfig,
    NeighborAwareSplitterConfig,
)
from .constants import EPS, TEMPERATURE_MIN
from .levels_info import LevelsInfo
from .utils import pair, exists, default, sanitize_tensor, create_attention_mask
from .depth_utils import (
    compute_max_depth,
    compute_actual_min_patch,
    compute_patch_sizes,
    compute_depth_distribution,
    compute_total_candidates,
    compute_region_shape_scale,
    compute_shape_scale_similarity,
    compute_normalized_area,
    compute_area_similarity,
)
from .splitter_protocol import (
    CoreSplitter,
    AnnealingSplitter,
    MetricsSplitter,
    SplitResult,
    validate_splitter,
)

__all__ = [
    # Hilbert Curve
    "HilbertCurve",
    "PseudoHilbertCurve",
    "HilbertScanner",
    "get_quadrant_order",
    "hilbert_distance_to_xy",
    "xy_to_hilbert_distance",
    # Indexer
    "HilbertIndexer",
    "HilbertPathCache",
    # I162-1: Pattern Encoder
    "HilbertPatternEncoder",
    "HilbertPatternEncoderLight",
    "create_hilbert_pattern_encoder",
    # Config
    "FractalConfig",
    "SplitterConfig",
    "create_fractal_config",
    "AnnealSchedule",
    "TokenizerType",
    "SemanticSplitterConfig",
    "create_semantic_splitter_config",
    "AttentionConfig",
    "TokenizerConfig",
    "TransformerConfig",
    "FractalViTConfig",
    "NeighborAwareSplitterConfig",
    # Constants
    "EPS",
    "TEMPERATURE_MIN",
    # Levels Info
    "LevelsInfo",
    # Utils
    "pair",
    "exists",
    "default",
    "sanitize_tensor",
    "create_attention_mask",
    # Depth utils
    "compute_max_depth",
    "compute_actual_min_patch",
    "compute_patch_sizes",
    "compute_depth_distribution",
    "compute_total_candidates",
    "compute_region_shape_scale",
    "compute_shape_scale_similarity",
    "compute_normalized_area",
    "compute_area_similarity",
    # Splitter Protocol (L1 - moved from modules to fix hierarchical import)
    "CoreSplitter",
    "AnnealingSplitter",
    "MetricsSplitter",
    "SplitResult",
    "validate_splitter",
]
