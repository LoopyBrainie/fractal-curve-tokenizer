# -*- coding: utf-8 -*-
"""
Hilbert 索引与路径缓存

数学形式化
============

Hilbert 路径缓存:
    对于网格 G_{h×w}，使用混合策略选择扫描方式:
    
    1. 标准 Hilbert (h = w = 2^k):
       H: [0, n²) → [0,n)×[0,n)，最优局部性 ~√2
    
    2. Hilbert + Padding (padding_ratio < 4/3):
       扩展到 n = 2^⌈log₂(max(h,w))⌉，过滤有效点
    
    3. Pseudo-Hilbert (padding_ratio ≥ 4/3):
       递归区域细分，保持良好局部性 ~1.5

缓存内容:
    1. hilbert_to_raster[d] = y * w + x
    2. quadtree_paths[d, ℓ] = qₗ, 四叉树第ℓ层象限索引

复杂度:
    - 预计算: O(h × w) 一次性
    - 查询: O(1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple

import torch
import torch._dynamo

from .curve_hilbert import PseudoHilbertCurve, _next_power_of_2


def _dynamo_safe_lru_cache(maxsize: int = 128):
    """LRU 缓存装饰器，兼容 torch.compile."""
    def decorator(func):
        cached = lru_cache(maxsize=maxsize)(func)
        return torch._dynamo.disable(cached)
    return decorator


@dataclass(frozen=True)
class _HilbertCacheKey:
    """Hilbert 缓存键 (可哈希)"""
    grid_h: int
    grid_w: int
    max_depth: int


class HilbertPathCache:
    """统一的 Hilbert 路径预计算缓存.
    
    缓存两种映射:
    1. Hilbert 索引 → 光栅索引 (用于特征重排序)
    2. Hilbert 索引 → 四叉树路径 (用于 levels_info)
    
    使用类级别缓存，所有实例共享。
    支持设备感知缓存，避免重复的 .to(device) 调用，
    从而支持 CUDA graphs 优化。
    """

    # I108-4: 补充内存界计算
    # 类级别缓存 (CPU 版本，作为源)
    _cache: Dict[_HilbertCacheKey, Tuple[torch.Tensor, torch.Tensor]] = {}
    # 设备感知缓存: (key, device_str) -> (tensor, tensor)
    _device_cache: Dict[Tuple[_HilbertCacheKey, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    # 内存上界计算:
    # - _max_cache_size=64: maxsize × N × (4 + 8×max_depth) bytes
    #   典型 (64×64, max_depth=8): 64 × 4096 × 68 ≈ 17.7 MB
    #   极端 (128×128, max_depth=8): 64 × 16384 × 68 ≈ 71 MB
    # - _max_device_cache_size=256: 4 × _max_cache_size
    _max_cache_size: int = 64
    _max_device_cache_size: int = 256
    
    @classmethod
    def get_or_compute(
        cls,
        grid_h: int,
        grid_w: int,
        max_depth: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取或计算 Hilbert 缓存.
        
        Args:
            grid_h: 网格高度
            grid_w: 网格宽度
            max_depth: 四叉树最大深度 (用于路径计算)
            device: 目标设备 (可选，指定后返回该设备上的张量)
            
        Returns:
            hilbert_to_raster: [N] Hilbert 索引到光栅索引的映射
            quadtree_paths: [N, max_depth] 四叉树路径
        """
        key = _HilbertCacheKey(grid_h, grid_w, max_depth)
        
        # 如果指定了设备，尝试从设备缓存获取
        if device is not None:
            device_str = str(device)
            device_key = (key, device_str)
            
            if device_key in cls._device_cache:
                return cls._device_cache[device_key]
        
        # 确保 CPU 缓存存在
        if key not in cls._cache:
            # 缓存淘汰 (简单 FIFO)
            if len(cls._cache) >= cls._max_cache_size:
                oldest_key = next(iter(cls._cache))
                del cls._cache[oldest_key]
                # 清理相关设备缓存
                cls._device_cache = {
                    k: v for k, v in cls._device_cache.items() 
                    if k[0] != oldest_key
                }
            
            # 计算并缓存 (CPU 版本)
            cls._cache[key] = cls._compute(grid_h, grid_w, max_depth)
        
        # 如果不需要特定设备，返回 CPU 版本
        if device is None:
            return cls._cache[key]
        
        # 创建设备版本并缓存
        device_str = str(device)
        device_key = (key, device_str)
        
        if device_key not in cls._device_cache:
            # 设备缓存淘汰
            if len(cls._device_cache) >= cls._max_device_cache_size:
                oldest_device_key = next(iter(cls._device_cache))
                del cls._device_cache[oldest_device_key]
            
            cpu_h2r, cpu_paths = cls._cache[key]
            cls._device_cache[device_key] = (
                cpu_h2r.to(device, non_blocking=True),
                cpu_paths.to(device, non_blocking=True),
            )
        
        return cls._device_cache[device_key]
    
    @classmethod
    def _compute(
        cls,
        grid_h: int,
        grid_w: int,
        max_depth: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 Hilbert/Pseudo-Hilbert 映射和四叉树路径."""
        # 使用 PseudoHilbertCurve 获取扫描序列
        scan_points = PseudoHilbertCurve.scan(grid_h, grid_w)
        
        # 转换为光栅索引
        hilbert_to_raster = []
        x_coords = []
        y_coords = []
        
        for x, y in scan_points:
            raster_idx = y * grid_w + x
            hilbert_to_raster.append(raster_idx)
            x_coords.append(x)
            y_coords.append(y)
        
        num_tokens = len(hilbert_to_raster)
        hilbert_to_raster = torch.tensor(hilbert_to_raster, dtype=torch.long)
        
        # 计算四叉树路径 (向量化)
        x_tensor = torch.tensor(x_coords, dtype=torch.long)
        y_tensor = torch.tensor(y_coords, dtype=torch.long)
        
        n = _next_power_of_2(max(grid_h, grid_w))
        # I109-7: 使用 bit_length() 替代 int(math.log2(...)) 避免浮点精度问题
        actual_max_depth = max(1, max(n, 2).bit_length() - 1)
        path_depth = min(max_depth, actual_max_depth)
        
        quadtree_paths = torch.zeros(num_tokens, max_depth, dtype=torch.long)
        
        for depth in range(path_depth):
            shift = actual_max_depth - depth - 1
            if shift >= 0:
                qx = (x_tensor >> shift) & 1
                qy = (y_tensor >> shift) & 1
                quadtree_paths[:, depth] = qx + 2 * qy
        
        return hilbert_to_raster, quadtree_paths
    
    @classmethod
    def clear_cache(cls) -> None:
        """清空缓存 (用于测试或内存管理)"""
        cls._cache.clear()
        cls._device_cache.clear()


class HilbertIndexer:
    """预计算 Hilbert/Pseudo-Hilbert 曲线索引，用于特征重排序.
    
    对于给定的网格大小，生成从光栅顺序到 Hilbert 顺序的映射。
    支持设备感知缓存，避免重复 .to(device) 调用。
    
    支持任意尺寸网格:
    - 2^k × 2^k: 使用标准 Hilbert 曲线
    - 其他尺寸: 使用混合策略 (Hilbert+Padding 或 Pseudo-Hilbert)
    """
    
    # 设备感知缓存
    _device_cache: Dict[Tuple[int, str], torch.Tensor] = {}
    _rect_device_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}
    _max_cache_size: int = 128
    
    @staticmethod
    @_dynamo_safe_lru_cache(maxsize=256)
    def get_hilbert_order(grid_size: int) -> torch.Tensor:
        """获取 grid_size × grid_size 正方形网格的 Hilbert 遍历顺序.
        
        Args:
            grid_size: 网格边长 (支持任意正整数)
            
        Returns:
            indices: [grid_size²] 长的索引张量，将光栅顺序映射到 Hilbert 顺序
        """
        if grid_size <= 0:
            return torch.tensor([], dtype=torch.long)
        
        scan_points = PseudoHilbertCurve.scan(grid_size, grid_size)
        positions = [y * grid_size + x for x, y in scan_points]
        
        return torch.tensor(positions, dtype=torch.long)
    
    @classmethod
    def get_hilbert_order_on_device(cls, grid_size: int, device: torch.device) -> torch.Tensor:
        """获取指定设备上的 Hilbert 顺序索引 (支持 CUDA graphs)."""
        device_str = str(device)
        cache_key = (grid_size, device_str)
        
        if cache_key not in cls._device_cache:
            if len(cls._device_cache) >= cls._max_cache_size:
                oldest_key = next(iter(cls._device_cache))
                del cls._device_cache[oldest_key]
            
            cpu_tensor = cls.get_hilbert_order(grid_size)
            cls._device_cache[cache_key] = cpu_tensor.to(device, non_blocking=True)
        
        return cls._device_cache[cache_key]
    
    @staticmethod
    @_dynamo_safe_lru_cache(maxsize=256)
    def get_hilbert_order_rect(grid_h: int, grid_w: int) -> torch.Tensor:
        """获取 H × W 矩形网格的 Hilbert 遍历顺序."""
        if grid_h <= 0 or grid_w <= 0:
            return torch.tensor([], dtype=torch.long)
        
        scan_points = PseudoHilbertCurve.scan(grid_h, grid_w)
        positions = [y * grid_w + x for x, y in scan_points]
        
        return torch.tensor(positions, dtype=torch.long)
    
    @classmethod
    def get_hilbert_order_rect_on_device(
        cls, grid_h: int, grid_w: int, device: torch.device
    ) -> torch.Tensor:
        """获取指定设备上的矩形网格 Hilbert 顺序索引."""
        device_str = str(device)
        cache_key = (grid_h, grid_w, device_str)
        
        if cache_key not in cls._rect_device_cache:
            if len(cls._rect_device_cache) >= cls._max_cache_size:
                oldest_key = next(iter(cls._rect_device_cache))
                del cls._rect_device_cache[oldest_key]
            
            cpu_tensor = cls.get_hilbert_order_rect(grid_h, grid_w)
            cls._rect_device_cache[cache_key] = cpu_tensor.to(device, non_blocking=True)
        
        return cls._rect_device_cache[cache_key]
    
    @classmethod
    def clear_device_cache(cls) -> None:
        """清空设备缓存"""
        cls._device_cache.clear()
        cls._rect_device_cache.clear()
    
    @staticmethod
    def reorder_to_hilbert(
        features: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """将特征从光栅顺序重排为 Hilbert 顺序.
        
        Args:
            features: [B, D, H, W] 的特征图
            grid_h: 网格高度
            grid_w: 网格宽度
            
        Returns:
            reordered: [B, H*W, D] 按 Hilbert 顺序排列的特征
        """
        B, D, H, W = features.shape
        device = features.device
        
        flat = features.flatten(2)  # [B, D, H*W]
        grid_size = max(H, W)
        
        if grid_size <= 1:
            return flat.transpose(1, 2)
        
        hilbert_to_raster, _ = HilbertPathCache.get_or_compute(
            grid_h=H, grid_w=W, max_depth=8, device=device
        )
        
        reordered = flat.index_select(2, hilbert_to_raster)
        return reordered.transpose(1, 2)
