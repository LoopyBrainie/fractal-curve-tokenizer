"""L3_critique - Theoretical critique and analysis tests for Hilbert Curve ViT

This module contains theoretical critique and analysis tests for:
1. Hilbert sequence scanning (Phase 1)
2. Quota allocation gradient quality (Phase 2)
3. Information density estimation (Phase 3)

批判分析维度:
- 数学严谨性: 是否有严格证明？
- 计算效率: 时间/空间复杂度
- 梯度质量: STE 近似偏差
- 数值稳定性: 边界情况处理
- 可学习性: 参数是否能收敛
"""

from .test_hilbert_locality import (
    LocalityMetrics,
    compute_locality_metrics,
    TestHilbertLocalityCritique,
    TestHilbertGradientQuality,
)

__all__ = [
    'LocalityMetrics',
    'compute_locality_metrics',
    'TestHilbertLocalityCritique',
    'TestHilbertGradientQuality',
]
