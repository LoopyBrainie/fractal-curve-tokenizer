"""Splitter modules - 只保留 H1SS 和 H-Entmax

H1SS (Hilbert-Optimal Splitter):
    - 基于6条数学公理的最优实现
    - Entmax 稀疏激活
    - Conv1D Hilbert 流形卷积
    - 支持辅助损失接口

H-Entmax (Hilbert-Ordered Entmax Splitter):
    - α-Entmax (α=1.5) 稀疏激活
    - 100% 梯度覆盖率
    - Hilbert 邻域复杂度提取
"""

from .hilbert_optimal_splitter import (
    HilbertOptimalSplitter,
    HilbertOptimalSplitterConfig,
)

from .hilbert_entmax import (
    HilbertOrderedEntmaxSplitter,
    HilbertLocalComplexity,
    entmax_1_5,
    entmax,
)

from .polar_voronoi_splitter import (
    PolarVoronoiSplitter,
    PolarVoronoiSplitterConfig,
    compute_sigma,
    compute_soft_switch,
    compute_wba_64,
)

__all__ = [
    # H1SS
    "HilbertOptimalSplitter",
    "HilbertOptimalSplitterConfig",
    # H-Entmax
    "HilbertOrderedEntmaxSplitter",
    "HilbertLocalComplexity",
    "entmax_1_5",
    "entmax",
    # Polar Voronoi (v1.3 §9.7, B.8)
    "PolarVoronoiSplitter",
    "PolarVoronoiSplitterConfig",
    "compute_sigma",
    "compute_soft_switch",
    "compute_wba_64",
]
