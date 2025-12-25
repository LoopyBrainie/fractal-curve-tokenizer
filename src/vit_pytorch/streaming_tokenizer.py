# -*- coding: utf-8 -*-
"""
Streaming Fractal Tokenizer - 统一架构实现

数学形式化
============

Tokenization 过程:
    T: R^{B × C × H × W} → (T ∈ R^{B × N × D}, L ∈ Z^{B × N})

模式 1 (可变 Token 数量, variable_tokens=True, 默认):
    N ∈ [N_min, N_max] 根据图像内容自适应
    Patch = Token: 分割决策直接产生 token
    
    数学形式:
        1. 尺度决策: s_{ij} = argmax_k π_{ij}^{(k)}
        2. 四叉树一致性: s_{i'j'} = s_{ij}, ∀(i',j') ∈ Block(i,j,s)
        3. Token 提取: T_k = PatchEmbed_{s_k}(P_k)
        4. Hilbert 排序: T = HilbertSort(T, paths)

模式 2 (固定 Token 数量, variable_tokens=False):
    N = (H/p_min) × (W/p_min) 是固定的 token 数量
    所有尺度特征加权融合后输出

核心组件
--------
1. 多尺度卷积金字塔 (MultiScalePatchEncoder):
   F_s = Conv_s(I), s ∈ {1, ..., S}
   每个尺度: kernel_size = stride = patch_size_s

2. 尺度选择 (V2, Gumbel-Softmax):
   π_{ij} = softmax(ScoreNet(F_{ij}) / τ)
   训练时: π̂_k = exp((log π_k + g_k) / τ) / Σ_l exp(...)
   推理时: s* = argmax_s π_s

3. Hilbert 重排序 (HilbertIndexer):
   T = Hilbert-Reorder(F_{s*})
   将 2D 特征图按 Hilbert 顺序展平为 1D 序列

类对照表
----------
+---------------------------+-------------------------------------------+
| 类                         | 数学定义                                    |
+===========================+===========================================+
| HilbertIndexer            | H: Grid_{h×w} → Seq_{n}                  |
| MultiScalePatchEncoder    | ConvPyramid: I → {F_s}_{s=1}^S            |
| StreamingFractalTokenizer | T_v1: I → (T, L), 固定尺度               |
| StreamingFractalTokenizerV3| T_v3: I → (T, L), Variable Depth (推荐)  |
+---------------------------+-------------------------------------------+

与原架构对比
----------
- 原架构: Image → BFS-Split → TokenProcessor → Transformer
- 新架构: Image → ConvPyramid → Hilbert-Reorder → Transformer

优势: 消除 Python 循环瓶颈，全 GPU 执行，端到端可微分
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F

from .hilbert import HilbertCurve, PseudoHilbertCurve, _is_power_of_2, _next_power_of_2
from .tokenization import BaseTokenizer, TokenizerOutput, TokenSequence
from .fractal_config import FractalConfig


# 创建兼容 torch.compile 的缓存装饰器
def _dynamo_safe_lru_cache(maxsize: int = 128):
    """LRU 缓存装饰器，兼容 torch.compile."""
    def decorator(func):
        cached = lru_cache(maxsize=maxsize)(func)
        return torch._dynamo.disable(cached)
    return decorator


# ==============================================================================
# HilbertPathCache: 统一的 Hilbert/Pseudo-Hilbert 路径预计算缓存
# ==============================================================================
#
# 数学形式化:
#   对于网格 G_{h×w}，使用混合策略选择扫描方式:
#
#   1. 标准 Hilbert (h = w = 2^k):
#      H: [0, n²) → [0,n)×[0,n)，最优局部性 ~√2
#
#   2. Hilbert + Padding (padding_ratio < 4/3):
#      扩展到 n = 2^⌈log₂(max(h,w))⌉，过滤有效点
#
#   3. Pseudo-Hilbert (padding_ratio ≥ 4/3):
#      递归区域细分，保持良好局部性 ~1.5
#
#   缓存内容:
#   1. hilbert_to_raster[d] = y * w + x
#   2. quadtree_paths[d, ℓ] = qₗ, 四叉树第ℓ层象限索引
#
#   复杂度:
#   - 预计算: O(h × w) 一次性
#   - 查询: O(1)
# ==============================================================================


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
    
    # 类级别缓存 (CPU 版本，作为源)
    _cache: Dict[_HilbertCacheKey, Tuple[torch.Tensor, torch.Tensor]] = {}
    # 设备感知缓存: (key, device_str) -> (tensor, tensor)
    _device_cache: Dict[Tuple[_HilbertCacheKey, str], Tuple[torch.Tensor, torch.Tensor]] = {}
    _max_cache_size: int = 64
    _max_device_cache_size: int = 256  # 设备缓存可以更大
    
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
        """计算 Hilbert/Pseudo-Hilbert 映射和四叉树路径.
        
        使用 PseudoHilbertCurve 的混合策略:
        - padding_ratio < 4/3: 标准 Hilbert + 过滤
        - padding_ratio ≥ 4/3: Pseudo-Hilbert 递归
        
        这是一次性的预计算，后续通过缓存获取。
        """
        # 使用 PseudoHilbertCurve 获取扫描序列
        # 该函数内部自动选择最优策略
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
        
        # 计算实际需要的深度
        # 对于非 2^k 情况，使用扩展后的虚拟深度
        n = _next_power_of_2(max(grid_h, grid_w))
        actual_max_depth = max(1, int(math.log2(max(n, 2))))
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
    @_dynamo_safe_lru_cache(maxsize=64)
    def get_hilbert_order(grid_size: int) -> torch.Tensor:
        """获取 grid_size × grid_size 正方形网格的 Hilbert 遍历顺序.
        
        Args:
            grid_size: 网格边长 (支持任意正整数)
            
        Returns:
            indices: [grid_size²] 长的索引张量，将光栅顺序映射到 Hilbert 顺序
        """
        if grid_size <= 0:
            return torch.tensor([], dtype=torch.long)
        
        # 使用 PseudoHilbertCurve 获取扫描序列
        scan_points = PseudoHilbertCurve.scan(grid_size, grid_size)
        
        # 转换为光栅索引
        positions = [y * grid_size + x for x, y in scan_points]
        
        return torch.tensor(positions, dtype=torch.long)
    
    @classmethod
    def get_hilbert_order_on_device(cls, grid_size: int, device: torch.device) -> torch.Tensor:
        """获取指定设备上的 Hilbert 顺序索引 (支持 CUDA graphs).
        
        Args:
            grid_size: 网格边长
            device: 目标设备
            
        Returns:
            indices: 在指定设备上的索引张量
        """
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
    @_dynamo_safe_lru_cache(maxsize=64)
    def get_hilbert_order_rect(grid_h: int, grid_w: int) -> torch.Tensor:
        """获取 H × W 矩形网格的 Hilbert 遍历顺序.
        
        Args:
            grid_h: 网格高度
            grid_w: 网格宽度
            
        Returns:
            indices: [H × W] 长的索引张量
        """
        if grid_h <= 0 or grid_w <= 0:
            return torch.tensor([], dtype=torch.long)
        
        # 使用 PseudoHilbertCurve 获取扫描序列
        scan_points = PseudoHilbertCurve.scan(grid_h, grid_w)
        
        # 转换为光栅索引
        positions = [y * grid_w + x for x, y in scan_points]
        
        return torch.tensor(positions, dtype=torch.long)
    
    @classmethod
    def get_hilbert_order_rect_on_device(cls, grid_h: int, grid_w: int, device: torch.device) -> torch.Tensor:
        """获取指定设备上的矩形网格 Hilbert 顺序索引 (支持 CUDA graphs).
        
        Args:
            grid_h: 网格高度
            grid_w: 网格宽度
            device: 目标设备
            
        Returns:
            indices: 在指定设备上的索引张量
        """
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
        
        使用设备感知缓存，支持 CUDA graphs。
        
        Args:
            features: [B, D, H, W] 的特征图
            grid_h: 网格高度
            grid_w: 网格宽度
            
        Returns:
            reordered: [B, H*W, D] 按 Hilbert 顺序排列的特征
        """
        B, D, H, W = features.shape
        device = features.device
        
        # Flatten 到 [B, D, H*W]
        flat = features.flatten(2)  # [B, D, H*W]
        
        grid_size = max(H, W)
        
        if grid_size <= 1:
            return flat.transpose(1, 2)  # [B, H*W, D]
        
        # 使用设备感知缓存获取 Hilbert 索引
        hilbert_to_raster, _ = HilbertPathCache.get_or_compute(
            grid_h=H, grid_w=W, max_depth=8, device=device
        )
        
        # 使用索引重排
        # flat: [B, D, H*W], indices: [H*W]
        reordered = flat.index_select(2, hilbert_to_raster)  # [B, D, H*W]
        
        return reordered.transpose(1, 2)  # [B, H*W, D]


class MultiScalePatchEncoder(nn.Module):
    """多尺度 Patch 编码器.
    
    使用不同大小的卷积核提取多尺度特征，替代 BFS 递归分割。
    每个尺度独立处理，最后融合。
    
    Args:
        channels: 输入图像通道数
        d_model: 输出嵌入维度
        patch_sizes: 支持的 patch 大小列表
    """
    
    def __init__(
        self,
        channels: int = 3,
        d_model: int = 256,
        patch_sizes: Tuple[int, ...] = (4, 8, 16),
    ) -> None:
        super().__init__()
        self.channels = channels
        self.d_model = d_model
        self.patch_sizes = patch_sizes
        self.num_scales = len(patch_sizes)
        
        # 每个尺度的编码器
        # 使用 Conv2d: kernel_size = stride = patch_size
        self.encoders = nn.ModuleDict({
            f"scale_{ps}": nn.Sequential(
                nn.Conv2d(channels, d_model // 2, kernel_size=ps, stride=ps),
                nn.BatchNorm2d(d_model // 2),
                nn.GELU(),
                nn.Conv2d(d_model // 2, d_model, kernel_size=1),
                nn.BatchNorm2d(d_model),
            )
            for ps in patch_sizes
        })
        
        # 尺度级别嵌入
        self.scale_embedding = nn.Embedding(self.num_scales, d_model)
        
    def forward(
        self,
        images: torch.Tensor,
    ) -> Dict[int, Tuple[torch.Tensor, Tuple[int, int]]]:
        """提取多尺度特征.
        
        Args:
            images: [B, C, H, W] 输入图像
            
        Returns:
            features_dict: {patch_size: (features, (grid_h, grid_w))}
                - features: [B, D, grid_h, grid_w]
        """
        B, C, H, W = images.shape
        features_dict = {}
        
        for scale_idx, ps in enumerate(self.patch_sizes):
            # 检查图像是否足够大
            if H >= ps and W >= ps:
                feat = self.encoders[f"scale_{ps}"](images)  # [B, D, H/ps, W/ps]
                grid_h, grid_w = feat.shape[2], feat.shape[3]
                
                # 添加尺度嵌入
                scale_emb = self.scale_embedding(
                    torch.tensor([scale_idx], device=images.device)
                )  # [1, D]
                feat = feat + scale_emb.view(1, -1, 1, 1)
                
                features_dict[ps] = (feat, (grid_h, grid_w))
        
        return features_dict


class StreamingFractalTokenizer(BaseTokenizer):
    """流式统一 Tokenizer - 单次前向完成 tokenization.
    
    核心创新：
    1. 多尺度卷积金字塔替代 BFS 循环 (GPU 友好)
    2. 固定尺度组合，可选区域自适应 (Phase 2)
    3. Hilbert 顺序重排保持空间局部性
    4. 单次前向完成 patch → embedding
    
    与原 FractalHilbertTokenizer 的接口保持兼容：
    - 输出 TokenizerOutput 格式
    - levels_info 包含深度和路径信息
    
    Args:
        image_size: 输入图像尺寸
        channels: 输入通道数
        d_model: 输出嵌入维度
        patch_sizes: 支持的 patch 大小元组
        primary_scale: 主要使用的尺度索引 (Phase 1 简化版)
        use_hilbert_order: 是否按 Hilbert 顺序排列输出
        max_level: 最大层级（用于 levels_info 兼容）
    """
    
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        channels: int = 3,
        d_model: int = 256,
        patch_sizes: Tuple[int, ...] = (4, 8, 16),
        primary_scale: Optional[int] = None,
        use_hilbert_order: bool = True,
        max_level: int = 50,
    ) -> None:
        super().__init__()
        
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        
        self.image_size = image_size
        self.channels = channels
        self.d_model = d_model
        self.patch_sizes = patch_sizes
        self.primary_scale = primary_scale if primary_scale is not None else 0
        self.use_hilbert_order = use_hilbert_order
        self.max_level = max_level
        
        # 多尺度编码器
        self.encoder = MultiScalePatchEncoder(
            channels=channels,
            d_model=d_model,
            patch_sizes=patch_sizes,
        )
        
        # 特征融合层 (替代 TokenProcessor 的功能)
        self.feature_fusion = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # 用于生成 levels_info 的层级映射
        # patch_size → level 的对应关系
        self._setup_level_mapping()
        
    def _setup_level_mapping(self) -> None:
        """建立 patch_size 到 level 的映射关系."""
        # 较大的 patch 对应较浅的层级
        # 例如: patch_size=16 → level=2, patch_size=8 → level=3, patch_size=4 → level=4
        max_ps = max(self.patch_sizes)
        self.scale_to_level = {}
        for ps in self.patch_sizes:
            # level = log2(max_ps / ps) + 1
            level = int(math.log2(max_ps / ps)) + 1
            self.scale_to_level[ps] = level
    
    def _create_levels_info(
        self,
        batch_size: int,
        num_tokens: int,
        scale_level: int,
        grid_h: int,
        grid_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """创建与原 tokenizer 兼容的 levels_info.
        
        使用 HilbertPathCache 统一缓存，避免重复计算。
        
        数学形式化:
            levels_info[b, i, 0] = scale_level (深度)
            levels_info[b, i, 1:] = [q₁, q₂, ..., qₗ] (四叉树路径)
            
            其中 qₗ = bit(x, d-ℓ) + 2 × bit(y, d-ℓ)
            (x, y) = H(i) 是第 i 个 token 的 Hilbert 坐标
        
        Args:
            batch_size: 批次大小
            num_tokens: token 数量
            scale_level: 当前尺度对应的层级
            grid_h: 网格高度
            grid_w: 网格宽度
            device: 设备
            
        Returns:
            levels_info: [B, num_tokens, info_len]
        """
        # 信息长度：depth + 最多 max_level 个路径节点
        info_len = min(self.max_level + 1, 16)
        max_depth = info_len - 1
        
        levels_info = torch.zeros(
            batch_size, num_tokens, info_len,
            dtype=torch.long, device=device
        )
        
        # 设置深度（第0列）
        levels_info[:, :, 0] = scale_level
        
        if max_depth <= 0:
            return levels_info
        
        actual_tokens = min(num_tokens, grid_h * grid_w)
        
        if self.use_hilbert_order:
            # 使用设备感知缓存获取四叉树路径
            _, quadtree_paths = HilbertPathCache.get_or_compute(
                grid_h=grid_h, grid_w=grid_w, max_depth=max_depth, device=device
            )
            # quadtree_paths: [N, max_depth]，已在正确设备上
            paths = quadtree_paths[:actual_tokens]
            
            # 填充路径 (广播到 batch)
            path_len = min(paths.shape[1], max_depth)
            levels_info[:, :actual_tokens, 1:path_len+1] = paths[:, :path_len].unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # 光栅顺序 (完全向量化，无缓存依赖)
            grid_size = max(grid_h, grid_w)
            n = 1
            while n < grid_size:
                n *= 2
            
            actual_max_depth = max(1, int(math.log2(max(n, 2))))
            path_depth = min(max_depth, actual_max_depth)
            
            indices = torch.arange(actual_tokens, device=device)
            y_coords = indices // grid_w
            x_coords = indices % grid_w
            
            for depth in range(path_depth):
                shift = actual_max_depth - depth - 1
                if shift >= 0:
                    qx = (x_coords >> shift) & 1
                    qy = (y_coords >> shift) & 1
                    quadrant = qx + 2 * qy
                    levels_info[:, :actual_tokens, depth + 1] = quadrant.unsqueeze(0).expand(batch_size, -1)
        
        return levels_info
    
    def tokenize(self, images: torch.Tensor) -> TokenizerOutput:
        """将图像转换为 token 序列.
        
        Args:
            images: [B, C, H, W] 输入图像
            
        Returns:
            TokenizerOutput 包含每个图像的 token 序列
        """
        if images.dim() != 4:
            raise ValueError(
                f"StreamingFractalTokenizer.tokenize expects 4D input [B, C, H, W], "
                f"got {images.dim()}D tensor with shape {tuple(images.shape)}."
            )
        
        B, C, H, W = images.shape
        device = images.device
        
        # 1. 多尺度特征提取
        features_dict = self.encoder(images)
        
        if not features_dict:
            raise ValueError(
                f"No valid scales for image size ({H}, {W}). "
                f"Minimum patch size is {min(self.patch_sizes)}."
            )
        
        # 2. Phase 1 简化版：只使用主要尺度
        primary_ps = self.patch_sizes[self.primary_scale]
        if primary_ps not in features_dict:
            # 回退到最小可用尺度
            primary_ps = min(features_dict.keys())
        
        features, (grid_h, grid_w) = features_dict[primary_ps]  # [B, D, H', W']
        scale_level = self.scale_to_level[primary_ps]
        
        # 3. Hilbert 顺序重排
        if self.use_hilbert_order:
            tokens = HilbertIndexer.reorder_to_hilbert(features, grid_h, grid_w)
        else:
            # 光栅顺序
            tokens = features.flatten(2).transpose(1, 2)  # [B, H'*W', D]
        
        num_tokens = tokens.shape[1]
        
        # 4. 特征融合
        tokens = self.feature_fusion(tokens)  # [B, num_tokens, D]
        
        # 5. 创建 levels_info
        levels_info = self._create_levels_info(
            batch_size=B,
            num_tokens=num_tokens,
            scale_level=scale_level,
            grid_h=grid_h,
            grid_w=grid_w,
            device=device,
        )
        
        # 6. 构建输出
        sequences = []
        for b in range(B):
            seq = TokenSequence(
                tokens=tokens[b],  # [num_tokens, D]
                metadata={"levels": levels_info[b]},  # [num_tokens, info_len]
            )
            sequences.append(seq)
        
        return TokenizerOutput(sequences)
    
    def forward(self, images: torch.Tensor) -> TokenizerOutput:
        """前向传播，等价于 tokenize."""
        return self.tokenize(images)


class StreamingFractalTokenizerV3(BaseTokenizer):
    """Variable Depth Tokenizer with Adaptive Quadtree Splitting.
    
    数学形式化
    ==========
    
    核心架构变更 (2025-12-25 重构):
    
    旧架构 (Cross-Scale Attention):
        F_s = MultiScaleConv(I)           # 多尺度特征
        α_{i,s} = softmax(Q_i · K_{i,s})  # 学习尺度权重
        Token_i = Σ_s α_{i,s} · V_{i,s}   # 加权融合
        问题: α 必然崩塌到单尺度 (信息论必然性)
    
    新架构 (Variable Depth Tokens):
        Regions = AdaptiveQuadtreeSplit(I)  # 内容自适应分割
        F = SharedConv(I)                    # 共享特征提取
        Token_i = Pool(F[R_i]) * σ_d + E_d  # 区域池化 + 深度编码
        优势: 深度由内容决定，非学习崩塌
    
    满足的数学约束:
    1. 维度一致性: 所有 region → 相同 dim
    2. Hilbert 路径一致性: 四叉树路径 = Hilbert 索引前缀
    3. 尺度等变性: depth_scale 编码尺度信息
    4. LCA 兼容性: 与现有 LCA bias 无缝工作
    
    Args:
        image_size: 输入图像尺寸
        channels: 图像通道数
        d_model: 输出嵌入维度
        base_patch_size: 最细粒度 patch 大小
        max_depth: 最大四叉树深度
        use_hilbert_order: 是否使用 Hilbert 曲线排序
        split_scheme: 分割方案 ('balanced_greedy' 或 'fixed_budget_dp')
        target_tokens: 目标 token 数量 (仅 fixed_budget_dp)
        complexity_alpha: 复杂度函数中方差权重
        enforce_balance: 是否强制 2:1 平衡约束
    """
    
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        channels: int = 3,
        d_model: int = 256,
        base_patch_size: int = 4,
        max_depth: int = 4,
        use_hilbert_order: bool = True,
        split_scheme: str = 'balanced_greedy',
        target_tokens: Optional[int] = None,
        complexity_alpha: float = 0.5,
        enforce_balance: bool = True,
    ) -> None:
        super().__init__()
        
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        
        self.image_size = image_size
        self.channels = channels
        self.d_model = d_model
        self.base_patch_size = base_patch_size
        self.max_depth = max_depth
        self.use_hilbert_order = use_hilbert_order
        self.split_scheme = split_scheme
        
        # Hilbert-Native Patch Embedding
        from .patch_embed import HilbertNativePatchEmbed
        self.patch_embed = HilbertNativePatchEmbed(
            channels=channels,
            dim=d_model,
            base_patch_size=base_patch_size,
            max_depth=max_depth,
            conv_layers=2,
            use_batch_norm=True,
        )
        
        # Adaptive Quadtree Splitter
        from .adaptive_split import (
            AdaptiveSplitConfig,
            BalancedGreedySplitter,
            FixedBudgetDPSplitter,
            SplitScheme,
        )
        
        if split_scheme == 'fixed_budget_dp' or split_scheme == SplitScheme.FIXED_BUDGET_DP:
            # Scheme C: Fixed budget DP
            split_config = AdaptiveSplitConfig.scheme_c(
                token_budget=target_tokens if target_tokens else 64,
                max_depth=max_depth,
                alpha=complexity_alpha,
            )
            self.splitter = FixedBudgetDPSplitter(split_config)
        else:
            # Scheme B: Balanced greedy
            split_config = AdaptiveSplitConfig.scheme_b(
                max_depth=max_depth,
                alpha=complexity_alpha,
                enforce_balance=enforce_balance,
                target_tokens=target_tokens,
            )
            self.splitter = BalancedGreedySplitter(split_config)
        
        # 统计信息
        self._last_split_stats: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_config(
        cls,
        config: "FractalConfig",
        channels: int = 3,
        d_model: int = 256,
    ) -> "StreamingFractalTokenizerV3":
        """从 FractalConfig 创建 Tokenizer 实例."""
        return cls(
            image_size=config.image_size,
            channels=channels,
            d_model=d_model,
            base_patch_size=config.patch_sizes[0] if config.patch_sizes else 4,
            max_depth=config.max_depth,
            use_hilbert_order=True,
        )
    
    def tokenize(self, images: torch.Tensor) -> TokenizerOutput:
        """Variable Depth tokenization.
        
        流程:
            1. AdaptiveQuadtreeSplit → 内容自适应分割
            2. HilbertNativePatchEmbed → 区域池化 + 深度编码
            3. HilbertSort → Hilbert 顺序排列
        
        Args:
            images: [B, C, H, W] 输入图像
            
        Returns:
            TokenizerOutput 包含 token 序列和 levels_info
        """
        if images.dim() != 4:
            raise ValueError(
                f"StreamingFractalTokenizerV3.tokenize expects 4D input [B, C, H, W], "
                f"got {images.dim()}D tensor."
            )
        
        B, C, H, W = images.shape
        device = images.device
        
        # 1. Adaptive Quadtree Splitting
        split_results = self.splitter.split_batch(images)
        
        # 收集统计信息
        self._last_split_stats = {
            'num_tokens': [sr.num_tokens for sr in split_results],
            'depth_distributions': [sr.depth_distribution for sr in split_results],
        }
        
        # 2. Hilbert-Native Patch Embedding
        tokens, levels_info = self.patch_embed(images, split_results)
        # tokens: [B, N_max, d_model]
        # levels_info: [B, N_max, max_depth+1]
        
        # 3. 构建输出
        sequences = []
        for b in range(B):
            num_tokens = split_results[b].num_tokens
            seq = TokenSequence(
                tokens=tokens[b, :num_tokens],  # 截断 padding
                metadata={
                    "levels": levels_info[b, :num_tokens],
                    "split_stats": {
                        "num_tokens": num_tokens,
                        "depth_distribution": split_results[b].depth_distribution,
                    },
                },
            )
            sequences.append(seq)
        
        return TokenizerOutput(sequences)
    
    def forward(self, images: torch.Tensor) -> TokenizerOutput:
        """前向传播，等价于 tokenize."""
        return self.tokenize(images)
    
    @torch.no_grad()
    def get_split_stats(self) -> Optional[Dict[str, Any]]:
        """获取最近一次分割的统计信息.
        
        Returns:
            Dict 包含:
            - num_tokens: List[int] 每个图像的 token 数量
            - depth_distributions: List[Dict[int, int]] 每个图像的深度分布
        """
        return self._last_split_stats
    
    def get_entropy_loss(self) -> Optional[torch.Tensor]:
        """获取熵正则化损失.
        
        注意: Variable Depth 架构不需要熵正则化
        (深度由内容决定，非学习权重)
        
        Returns:
            None (保持接口兼容)
        """
        return None
    
    def get_scale_entropy(self) -> Optional[float]:
        """获取尺度分布熵值.
        
        对于 Variable Depth，计算深度分布的熵。
        """
        if self._last_split_stats is None:
            return None
        
        import math
        
        # 合并所有图像的深度分布
        total_dist: Dict[int, int] = {}
        for dist in self._last_split_stats['depth_distributions']:
            for d, count in dist.items():
                total_dist[d] = total_dist.get(d, 0) + count
        
        total = sum(total_dist.values())
        if total == 0:
            return None
        
        # 计算熵
        entropy = 0.0
        for count in total_dist.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log(p)
        
        return entropy
    
    @torch.no_grad()
    def compute_scale_distribution(
        self, 
        images: torch.Tensor,
    ) -> Dict[str, Any]:
        """计算深度分布统计信息.
        
        Variable Depth 架构中，"scale" 对应四叉树深度：
        - depth=0: 最粗尺度 (base_patch_size * 2^max_depth)
        - depth=max_depth: 最细尺度 (base_patch_size)
        
        Args:
            images: [B, C, H, W] 输入图像
            
        Returns:
            Dict 包含:
            - scale_ratios: Dict[int, float] 每个深度的 token 比例 (key 为 patch_size)
            - entropy: float 深度分布熵值
            - max_entropy: float 最大可能熵值
            - dominant_scale: int 主导 patch 尺寸
            - depth_distribution: Dict[int, int] 原始深度计数
        """
        # 执行分割以获取统计信息
        _ = self.tokenize(images)
        
        if self._last_split_stats is None:
            return {
                'scale_ratios': {},
                'entropy': 0.0,
                'max_entropy': 0.0,
                'dominant_scale': self.base_patch_size,
                'depth_distribution': {},
            }
        
        import math
        
        # 合并所有图像的深度分布
        total_dist: Dict[int, int] = {}
        for dist in self._last_split_stats['depth_distributions']:
            for d, count in dist.items():
                total_dist[d] = total_dist.get(d, 0) + count
        
        total_tokens = sum(total_dist.values())
        if total_tokens == 0:
            return {
                'scale_ratios': {},
                'entropy': 0.0,
                'max_entropy': 0.0,
                'dominant_scale': self.base_patch_size,
                'depth_distribution': {},
            }
        
        # 将深度转换为 patch_size: depth=d → ps=base_patch_size * 2^(max_depth - d)
        scale_ratios: Dict[int, float] = {}
        for depth, count in total_dist.items():
            ps = self.base_patch_size * (2 ** (self.max_depth - depth))
            scale_ratios[ps] = count / total_tokens
        
        # 计算熵
        entropy = 0.0
        for count in total_dist.values():
            p = count / total_tokens
            if p > 0:
                entropy -= p * math.log(p)
        
        # 最大熵 (均匀分布)
        num_depths = self.max_depth + 1
        max_entropy = math.log(num_depths) if num_depths > 1 else 0.0
        
        # 主导尺度 (token 数最多的深度对应的 patch_size)
        dominant_depth = max(total_dist.keys(), key=lambda d: total_dist[d])
        dominant_scale = self.base_patch_size * (2 ** (self.max_depth - dominant_depth))
        
        return {
            'scale_ratios': scale_ratios,
            'entropy': entropy,
            'max_entropy': max_entropy,
            'dominant_scale': dominant_scale,
            'depth_distribution': total_dist,
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练状态统计信息."""
        stats = {
            'tokenizer_version': 'v3_variable_depth',
            'architecture': 'adaptive_quadtree_split + hilbert_native_embed',
            'split_scheme': self.split_scheme,
            'max_depth': self.max_depth,
        }
        
        if self._last_split_stats:
            avg_tokens = sum(self._last_split_stats['num_tokens']) / len(self._last_split_stats['num_tokens'])
            stats['avg_tokens_per_image'] = avg_tokens
            stats['depth_entropy'] = self.get_scale_entropy()
        
        return stats
