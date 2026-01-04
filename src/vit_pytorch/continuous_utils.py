"""
I10-19 连续松弛工具函数

数学形式化:
    实际可达深度 d* = ⌊log2(H / s_min)⌋
    
    其中:
        H: 图像尺寸
        s_min: 最小区域尺寸
        
    理论依据:
        区域在depth d的尺寸 = H / 2^d
        终止条件: size < 2 * s_min
        → H / 2^d < 2 * s_min
        → 2^d > H / (2 * s_min)
        → d > log2(H / s_min) - 1
        → d_max = ⌊log2(H / s_min)⌋
"""

import math
import warnings
from typing import Tuple


def compute_optimal_parallel_depth(
    image_size: int,
    min_region_size: int = 8,
) -> int:
    """
    计算最优并行评估深度 (无浪费候选数)
    
    数学形式化:
        d* = ⌊log2(H / s_min)⌋
        
    Args:
        image_size: 图像尺寸 (假设正方形)
        min_region_size: 最小区域尺寸 (LearnableSplitter配置)
        
    Returns:
        optimal_depth: 最优并行深度
        
    Examples:
        >>> compute_optimal_parallel_depth(32, 8)  # CIFAR-10
        2
        >>> compute_optimal_parallel_depth(64, 8)  # Tiny-ImageNet
        3
        >>> compute_optimal_parallel_depth(224, 8) # ImageNet
        5
    """
    if image_size < min_region_size:
        return 0
    
    d_star = math.floor(math.log2(image_size / min_region_size))
    return max(0, d_star)


def compute_num_candidates(max_depth_parallel: int) -> int:
    """
    计算候选区域数量
    
    数学形式化:
        N = Σ_{d=0}^{D} 4^d = (4^{D+1} - 1) / 3
        
    Args:
        max_depth_parallel: 并行评估深度
        
    Returns:
        num_candidates: 候选区域总数
    """
    return (4 ** (max_depth_parallel + 1) - 1) // 3


def validate_continuous_config(
    image_size: int,
    min_region_size: int,
    max_depth_parallel: int,
    warn: bool = True,
) -> Tuple[int, float]:
    """
    验证连续松弛配置并计算浪费率
    
    返回:
        (optimal_depth, waste_ratio)
    """
    optimal = compute_optimal_parallel_depth(image_size, min_region_size)
    
    if max_depth_parallel > optimal:
        reachable = compute_num_candidates(optimal)
        configured = compute_num_candidates(max_depth_parallel)
        waste_ratio = (configured - reachable) / configured
        
        if warn:
            warnings.warn(
                f"连续松弛配置次优:\n"
                f"  image_size={image_size}, min_region_size={min_region_size}\n"
                f"  实际可达深度: {optimal}\n"
                f"  配置深度: {max_depth_parallel}\n"
                f"  候选数: {configured} (实际可达: {reachable})\n"
                f"  浪费率: {waste_ratio:.1%}\n"
                f"  建议: 使用 max_depth_parallel={optimal}"
            )
        
        return optimal, waste_ratio
    
    return optimal, 0.0


def get_continuous_config_stats(
    image_size: int,
    min_region_size: int,
    max_depth_parallel: int,
) -> dict:
    """获取连续松弛配置的详细统计"""
    optimal = compute_optimal_parallel_depth(image_size, min_region_size)
    configured_candidates = compute_num_candidates(max_depth_parallel)
    optimal_candidates = compute_num_candidates(optimal)
    
    return {
        'image_size': image_size,
        'min_region_size': min_region_size,
        'max_depth_parallel': max_depth_parallel,
        'optimal_depth': optimal,
        'configured_candidates': configured_candidates,
        'optimal_candidates': optimal_candidates,
        'reachable_candidates': optimal_candidates,
        'waste_ratio': (configured_candidates - optimal_candidates) / configured_candidates
                      if configured_candidates > 0 else 0.0,
        'compute_overhead': configured_candidates / optimal_candidates
                          if optimal_candidates > 0 else float('inf'),
    }
