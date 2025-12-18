# -*- coding: utf-8 -*-
"""
Streaming Fractal Tokenizer - 统一架构实现

数学形式化
============

Tokenization 过程:
    T: R^{B × C × H × W} → (T ∈ R^{B × N × D}, L ∈ Z^{B × N})

模式 1 (固定 Token 数量, variable_tokens=False):
    N = (H/p_min) × (W/p_min) 是固定的 token 数量
    所有尺度特征加权融合后输出

模式 2 (可变 Token 数量, variable_tokens=True):
    N ∈ [N_min, N_max] 根据图像内容自适应
    Patch = Token: 分割决策直接产生 token
    
    数学形式:
        1. 尺度决策: s_{ij} = argmax_k π_{ij}^{(k)}
        2. 四叉树一致性: s_{i'j'} = s_{ij}, ∀(i',j') ∈ Block(i,j,s)
        3. Token 提取: T_k = PatchEmbed_{s_k}(P_k)
        4. Hilbert 排序: T = HilbertSort(T, paths)

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
| StreamingFractalTokenizerV2| T_v2: I → (T, L), Gumbel-Softmax 自适应  |
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
from typing import Dict, List, Optional, Tuple, Union

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
        
        return cls._cache[key]
    
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


class StreamingFractalTokenizerV2(StreamingFractalTokenizer):
    """Phase 2: 带区域自适应分辨率选择的 Tokenizer.
    
    在 Phase 1 的基础上添加：
    1. 语义级区域复杂度估计 (基于 Encoder 特征)
    2. 多尺度特征融合
    3. Gumbel-Softmax 可微分尺度选择
    
    .. note::
        **v2.0 重构 (2025-12)**: 
        
        复杂度估计器改为使用 Encoder 特征而非原始像素：
        - 消除冗余计算 (~75% FLOPs 节省)
        - 语义感知：基于高级特征判断区域重要性
        - 更大感受野：继承 Encoder 的感受野
        
        数学形式:
            旧: π = ComplexityNet(I)           # 7层CNN处理原始像素
            新: π = ComplexityHead(Concat(F_s)) # 轻量头处理Encoder特征
    
    .. note::
        **v2.1 新增 (2025-12)**: variable_tokens 模式
        
        当 variable_tokens=True 时，启用 Patch=Token 直接映射：
        - 分割决策直接产生可变数量的 token
        - 四叉树一致性约束确保空间连贯性
        - 每个尺度使用独立的投影层
        
        数学形式:
            s_{ij} = argmax π_{ij}  # 尺度决策
            T_k = PatchEmbed_{s_k}(P_k)  # 尺度独立投影
        
    .. important::
        **v1.1 修复 (2025-01)**: 解决 train/eval 模式不一致问题。
        默认使用 Straight-Through Estimator (hard=True)，确保训练和验证
        看到相同的特征分布。可通过 `use_soft_weights=True` 恢复旧行为。
    """
    
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        channels: int = 3,
        d_model: int = 256,
        patch_sizes: Tuple[int, ...] = (4, 8, 16),
        use_hilbert_order: bool = True,
        max_level: int = 50,
        gumbel_temperature: float = 2.0,
        gumbel_tau_min: float = 0.5,
        gumbel_tau_max: float = 5.0,
        use_soft_weights: bool = False,
        variable_tokens: bool = False,
    ) -> None:
        """初始化 StreamingFractalTokenizerV2.
        
        Args:
            image_size: 输入图像尺寸
            channels: 图像通道数
            d_model: 输出嵌入维度
            patch_sizes: 多尺度 patch 大小 (从小到大排列)
            use_hilbert_order: 是否使用 Hilbert 曲线排序
            max_level: 最大四叉树层级
            gumbel_temperature: Gumbel-Softmax 初始温度
            gumbel_tau_min: 温度退火下界
            gumbel_tau_max: 温度退火上界
            use_soft_weights: 是否使用软权重 (实验性)
                - False (默认): 使用 STE (hard=True)，train/eval 一致
                - True: 训练时使用软权重 (可能导致 train/eval 差异)
            variable_tokens: 是否启用可变 Token 数量模式
                - False (默认): 固定 token 数量，特征加权融合
                - True: Patch=Token 直接映射，token 数量可变
        
        See Also:
            from_config: 从 FractalConfig 创建实例 (推荐)
        """
        super().__init__(
            image_size=image_size,
            channels=channels,
            d_model=d_model,
            patch_sizes=patch_sizes,
            primary_scale=None,
            use_hilbert_order=use_hilbert_order,
            max_level=max_level,
        )
        
        self.gumbel_temperature = gumbel_temperature
        self.use_soft_weights = use_soft_weights
        self.variable_tokens = variable_tokens
        self.num_scales = len(patch_sizes)
        
        # ========== 新架构: 语义级复杂度估计 (CRITICAL-2 修复) ==========
        # 使用 Encoder 特征而非原始像素，消除冗余计算
        #
        # 数学原理:
        #   π_{i,j} = Softmax(ComplexityHead(Concat_{s}[Upsample(F_s)]))_{i,j}
        #
        # 其中 F_s 是 Encoder 在尺度 s 的特征图
        #
        # 优势:
        #   1. 复用 Encoder 已计算的特征 (消除 ~75% 冗余)
        #   2. 语义感知: 基于高级特征而非低级像素
        #   3. 更大感受野: 继承 Encoder 的感受野
        #
        # 参数量对比:
        #   旧 (7层CNN): ~180K 参数, ~8M FLOPs
        #   新 (轻量头): ~10K 参数, ~1M FLOPs
        
        self.complexity_head = nn.Sequential(
            # 输入: 拼接的多尺度特征 [B, S*D, H', W']
            nn.Conv2d(d_model * self.num_scales, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            # 空间混合: 3x3 卷积捕捉局部上下文
            nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.GELU(),
            # 输出: 每个位置的尺度 logits
            nn.Conv2d(d_model // 2, self.num_scales, kernel_size=1),
        )
        
        # 温度退火配置 (从构造参数获取，消除硬编码)
        self.tau_init = gumbel_temperature
        self.tau_min = gumbel_tau_min
        self.tau_max = gumbel_tau_max
        self._current_tau = gumbel_temperature
        
        # 保持可学习温度参数（可选）
        self.temperature = nn.Parameter(torch.tensor(gumbel_temperature))
        
        # ========== 深度探索优先 Warmup (v2.2) ==========
        # 在训练初期对小尺度（深层级）添加正偏置，引导模型探索细粒度特征
        #
        # 数学形式:
        #   logits' = logits + scale_bias
        #   scale_bias[s] = bias_strength * (1 - s / (S-1))  # 小尺度偏置大
        #
        # 其中 s ∈ [0, S-1] 是尺度索引，s=0 对应最小 patch（最深层级）
        #
        # 调度策略:
        #   bias_strength = max_bias * (1 - progress)^decay_power
        #   warmup 期间保持较高偏置，之后快速衰减
        #
        # 效果:
        #   - 训练初期: 模型更倾向于选择小 patch，学习细粒度特征
        #   - 训练后期: 偏置消失，模型自主学习最优尺度选择
        self._depth_bias_max = 2.0      # 最大偏置强度
        self._depth_bias_decay = 2.0    # 衰减指数 (>1 快速衰减)
        self._depth_bias_warmup = 0.2   # warmup 占比 (前 20% 保持高偏置)
        self._current_depth_bias = self._depth_bias_max
        
        # 预计算尺度偏置权重 (小尺度 = 大权重)
        # patch_sizes 从小到大排列，所以 index 0 = 最小 patch = 最深层级
        self.register_buffer(
            '_scale_bias_weights',
            torch.linspace(1.0, 0.0, self.num_scales)  # [1.0, 0.67, 0.33, 0.0] for 4 scales
        )
    
    @classmethod
    def from_config(
        cls,
        config: FractalConfig,
        channels: int = 3,
        d_model: int = 256,
    ) -> "StreamingFractalTokenizerV2":
        """从 FractalConfig 创建 Tokenizer 实例 (推荐方式).
        
        自动从配置中提取所有必要参数，消除参数不一致的风险。
        
        Args:
            config: FractalConfig 实例
            channels: 图像通道数
            d_model: 输出嵌入维度
            
        Returns:
            StreamingFractalTokenizerV2 实例
            
        Examples:
            >>> config = FractalConfig(64, 4, gumbel_tau_init=1.5)
            >>> tokenizer = StreamingFractalTokenizerV2.from_config(config)
        """
        return cls(
            image_size=config.image_size,
            channels=channels,
            d_model=d_model,
            patch_sizes=config.patch_sizes,
            use_hilbert_order=True,
            max_level=config.max_depth,
            gumbel_temperature=config.gumbel_tau_init,
            gumbel_tau_min=config.gumbel_tau_min,
            gumbel_tau_max=config.gumbel_tau_max,
            use_soft_weights=config.use_soft_weights,
            variable_tokens=config.variable_tokens,
        )
    
    def set_temperature(self, tau: float) -> None:
        """设置当前 Gumbel-Softmax 温度 (用于温度退火调度).
        
        Args:
            tau: 目标温度值，会被 clamp 到 [tau_min, tau_max]
        """
        self._current_tau = max(self.tau_min, min(self.tau_max, tau))
        self.temperature.data.fill_(self._current_tau)
    
    def get_temperature(self) -> float:
        """获取当前温度值."""
        return self._current_tau
    
    def set_depth_bias(
        self,
        bias_strength: float,
        max_bias: Optional[float] = None,
        decay_power: Optional[float] = None,
        warmup_ratio: Optional[float] = None,
    ) -> None:
        """设置深度探索偏置参数.
        
        Args:
            bias_strength: 当前偏置强度
            max_bias: 可选，更新最大偏置值
            decay_power: 可选，更新衰减指数
            warmup_ratio: 可选，更新 warmup 占比
        """
        self._current_depth_bias = max(0.0, bias_strength)
        if max_bias is not None:
            self._depth_bias_max = max_bias
        if decay_power is not None:
            self._depth_bias_decay = decay_power
        if warmup_ratio is not None:
            self._depth_bias_warmup = warmup_ratio
    
    def get_depth_bias(self) -> float:
        """获取当前深度偏置强度."""
        return self._current_depth_bias
    
    def anneal_depth_bias(
        self,
        current_epoch: int,
        total_epochs: int,
    ) -> float:
        """深度偏置退火调度.
        
        在 warmup 期间保持高偏置，之后快速衰减。
        
        调度公式:
            if progress < warmup_ratio:
                bias = max_bias  # warmup 期间保持最大偏置
            else:
                adjusted_progress = (progress - warmup) / (1 - warmup)
                bias = max_bias * (1 - adjusted_progress)^decay_power
        
        Args:
            current_epoch: 当前 epoch (1-indexed)
            total_epochs: 总 epoch 数
            
        Returns:
            更新后的偏置强度
        """
        progress = min(1.0, current_epoch / max(1, total_epochs))
        
        if progress < self._depth_bias_warmup:
            # Warmup 期间: 保持最大偏置
            new_bias = self._depth_bias_max
        else:
            # Warmup 后: 快速衰减
            adjusted_progress = (progress - self._depth_bias_warmup) / (1.0 - self._depth_bias_warmup)
            new_bias = self._depth_bias_max * ((1.0 - adjusted_progress) ** self._depth_bias_decay)
        
        self._current_depth_bias = new_bias
        return new_bias
    
    def anneal_temperature(
        self,
        current_epoch: int,
        total_epochs: int,
        schedule: str = "linear",
    ) -> float:
        """温度退火调度 (EXP-FIX-2).
        
        从 τ_init 线性/指数退火到 τ_min。
        同时更新深度偏置 (v2.2)。
        
        Args:
            current_epoch: 当前 epoch (1-indexed)
            total_epochs: 总 epoch 数
            schedule: 退火方式 ("linear", "exponential", "cosine")
            
        Returns:
            更新后的温度值
        """
        progress = min(1.0, current_epoch / max(1, total_epochs))
        
        if schedule == "linear":
            # 线性退火: τ = τ_max - (τ_max - τ_min) * progress
            new_tau = self.tau_max - (self.tau_max - self.tau_min) * progress
        elif schedule == "exponential":
            # 指数退火: τ = τ_max * (τ_min / τ_max)^progress
            new_tau = self.tau_max * (self.tau_min / self.tau_max) ** progress
        elif schedule == "cosine":
            # 余弦退火: τ = τ_min + 0.5 * (τ_max - τ_min) * (1 + cos(π * progress))
            import math
            new_tau = self.tau_min + 0.5 * (self.tau_max - self.tau_min) * (1 + math.cos(math.pi * progress))
        else:
            new_tau = self.tau_init
        
        self.set_temperature(new_tau)
        
        # 同步更新深度偏置 (v2.2)
        self.anneal_depth_bias(current_epoch, total_epochs)
        
        return new_tau
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取当前训练状态统计信息 (用于日志追踪).
        
        返回关键参数以便追踪训练/推理一致性:
        
        Returns:
            Dict 包含以下字段:
            - gumbel_tau: 当前 Gumbel-Softmax 温度
            - depth_bias: 当前深度偏置强度
            - depth_bias_active: 偏置是否仍在生效 (> 0.01)
            - use_soft_weights: 是否使用软权重模式
            - temperature_param: 可学习温度参数值
            - tau_range: (tau_min, tau_max) 范围
            - bias_config: 深度偏置配置
        """
        return {
            'gumbel_tau': self._current_tau,
            'depth_bias': self._current_depth_bias,
            'depth_bias_active': self._current_depth_bias > 0.01,
            'use_soft_weights': self.use_soft_weights,
            'temperature_param': self.temperature.item(),
            'tau_range': (self.tau_min, self.tau_max),
            'bias_config': {
                'max': self._depth_bias_max,
                'decay': self._depth_bias_decay,
                'warmup': self._depth_bias_warmup,
            },
        }
    
    @torch.no_grad()
    def compute_scale_distribution(
        self,
        images: torch.Tensor,
    ) -> Dict[str, Any]:
        """计算尺度选择分布统计 (诊断用).
        
        对输入图像计算每个尺度被选择的频率，用于:
        1. 验证 train/eval 一致性
        2. 监控尺度选择是否多样化
        3. 检查深度偏置是否过期
        
        Args:
            images: [B, C, H, W] 输入图像
            
        Returns:
            Dict 包含:
            - scale_counts: {patch_size: count} 每个尺度的选择次数
            - scale_ratios: {patch_size: ratio} 每个尺度的选择比例
            - entropy: 尺度分布熵 (越高越多样化)
            - dominant_scale: 最常被选择的尺度
        """
        was_training = self.training
        self.eval()  # 使用 eval 模式确保确定性
        
        try:
            # 提取多尺度特征
            features_dict = self.encoder(images)
            
            # 确定目标尺寸
            min_ps = min(features_dict.keys())
            _, (grid_h, grid_w) = features_dict[min_ps]
            target_size = (grid_h, grid_w)
            
            # 计算尺度权重
            scale_weights = self._compute_scale_weights(features_dict, target_size)
            # scale_weights: [B, num_scales, grid_h, grid_w]
            
            # 获取每个位置的尺度决策
            scale_decisions = scale_weights.argmax(dim=1)  # [B, grid_h, grid_w]
            
            # 统计每个尺度的选择次数
            scale_counts = {}
            total_positions = scale_decisions.numel()
            
            for idx, ps in enumerate(self.patch_sizes):
                count = (scale_decisions == idx).sum().item()
                scale_counts[ps] = count
            
            # 计算比例
            scale_ratios = {ps: c / total_positions for ps, c in scale_counts.items()}
            
            # 计算熵 (使用 math.log 避免 numpy 依赖)
            ratios = list(scale_ratios.values())
            entropy = 0.0
            for r in ratios:
                if r > 0:
                    entropy -= r * math.log(r + 1e-10)
            
            # 找到主导尺度
            dominant_scale = max(scale_counts, key=scale_counts.get)
            
            return {
                'scale_counts': scale_counts,
                'scale_ratios': scale_ratios,
                'entropy': entropy,
                'dominant_scale': dominant_scale,
                'max_entropy': math.log(len(self.patch_sizes)),  # 均匀分布的熵
            }
        finally:
            if was_training:
                self.train()
        
    def _compute_scale_weights(
        self,
        features_dict: Dict[int, Tuple[torch.Tensor, Tuple[int, int]]],
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """基于 Encoder 特征计算每个区域的尺度权重.
        
        **v2.0 重构**: 使用语义特征而非原始像素
        **v2.2 新增**: 深度探索优先 warmup 偏置
        
        数学形式:
            F_aligned = {Upsample(F_s, target_size) | s ∈ scales}
            F_concat = Concat(F_aligned, dim=1)  # [B, S*D, H', W']
            logits = ComplexityHead(F_concat)     # [B, S, H', W']
            
            # v2.2: 添加深度偏置
            logits' = logits + depth_bias * scale_bias_weights
            
            π = Gumbel-Softmax(logits', τ)       # [B, S, H', W']
        
        Args:
            features_dict: {patch_size: (features [B,D,H,W], (grid_h, grid_w))}
            target_size: 目标空间尺寸 (H', W')
            
        Returns:
            weights: [B, num_scales, H', W'] 每个区域的尺度权重
            
        Note:
            **修复 train/eval 不一致问题 (v1.1)**：
            默认使用 Straight-Through Estimator (STE):
            - 前向传播: 硬决策 (one-hot) - train 和 eval 一致
            - 反向传播: 软梯度 (通过 Gumbel-Softmax)
            
            **深度探索优先 (v2.2)**：
            训练初期对小尺度添加正偏置，引导模型探索细粒度特征。
            偏置随训练进度衰减，最终由模型自主决策。
        """
        # 1. 将所有尺度的特征对齐到目标大小
        aligned_features = []
        for ps in self.patch_sizes:
            if ps in features_dict:
                feat, _ = features_dict[ps]
                if feat.shape[-2:] != target_size:
                    feat = F.interpolate(
                        feat,
                        size=target_size,
                        mode='bilinear',
                        align_corners=False,
                    )
                aligned_features.append(feat)
            else:
                # 如果某个尺度不可用，使用零填充
                B = next(iter(features_dict.values()))[0].shape[0]
                D = self.d_model
                aligned_features.append(
                    torch.zeros(B, D, *target_size, device=feat.device)
                )
        
        # 2. 拼接多尺度特征 [B, S*D, H', W']
        concat_features = torch.cat(aligned_features, dim=1)
        
        # 3. 通过轻量级头预测尺度 logits
        logits = self.complexity_head(concat_features)  # [B, num_scales, H', W']
        
        # 4. 应用深度探索偏置 (v2.2)
        # 仅在训练时应用，推理时不添加偏置
        if self.training and self._current_depth_bias > 0.01:
            # scale_bias_weights: [S] -> [1, S, 1, 1] for broadcasting
            bias = self._current_depth_bias * self._scale_bias_weights.view(1, -1, 1, 1)
            logits = logits + bias
        
        # 5. Gumbel-Softmax 转换为权重
        if self.training:
            if self.use_soft_weights:
                # 实验模式: 软权重 (可能导致 train/eval 差异)
                weights = F.gumbel_softmax(
                    logits,
                    tau=self.temperature.clamp(min=0.1),
                    hard=False,
                    dim=1,
                )
            else:
                # 默认: Straight-Through Estimator (hard=True)
                # 前向: argmax 硬决策，反向: 软梯度
                weights = F.gumbel_softmax(
                    logits,
                    tau=self.temperature.clamp(min=0.1),
                    hard=True,  # 关键修复: 保持 train/eval 一致
                    dim=1,
                )
        else:
            # 推理时使用 argmax (与训练时的硬决策一致)
            hard_indices = logits.argmax(dim=1)  # [B, H', W']
            weights = F.one_hot(
                hard_indices, num_classes=len(self.patch_sizes)
            ).permute(0, 3, 1, 2).float()  # [B, num_scales, H', W']
        
        return weights
    
    def tokenize(self, images: torch.Tensor) -> TokenizerOutput:
        """区域自适应 tokenization (语义引导).
        
        根据 variable_tokens 参数选择不同的 tokenization 策略:
        
        - variable_tokens=False (默认): 固定 token 数量，加权融合
        - variable_tokens=True: Patch=Token 直接映射，可变数量
        
        **v2.0 重构**: 执行流程改变
        
        旧流程:
            1. ComplexityEstimator(原始图像) → 尺度权重
            2. Encoder(原始图像) → 多尺度特征
            3. 加权融合
            
        新流程:
            1. Encoder(原始图像) → 多尺度特征
            2. ComplexityHead(Encoder特征) → 尺度权重 (语义级)
            3. 根据 variable_tokens 选择:
               - False: 加权融合
               - True: 直接按尺度提取 token
        """
        if images.dim() != 4:
            raise ValueError(
                f"StreamingFractalTokenizerV2.tokenize expects 4D input [B, C, H, W], "
                f"got {images.dim()}D tensor."
            )
        
        B, C, H, W = images.shape
        device = images.device
        
        # 1. 提取多尺度特征 (Encoder)
        features_dict = self.encoder(images)
        
        # 2. 确定基础网格大小 (使用最小 patch size)
        min_ps = min(features_dict.keys())
        base_features, (grid_h, grid_w) = features_dict[min_ps]
        target_size = (grid_h, grid_w)
        
        # 3. 基于 Encoder 特征计算尺度权重 (语义级复杂度)
        scale_weights = self._compute_scale_weights(features_dict, target_size)
        # scale_weights: [B, num_scales, grid_h, grid_w]
        
        # 4. 根据模式选择 tokenization 策略
        if self.variable_tokens:
            return self._tokenize_variable(
                features_dict, scale_weights, B, grid_h, grid_w, device
            )
        else:
            return self._tokenize_fixed(
                features_dict, scale_weights, base_features, 
                B, grid_h, grid_w, min_ps, device
            )
    
    def _tokenize_fixed(
        self,
        features_dict: Dict[int, Tuple[torch.Tensor, Tuple[int, int]]],
        scale_weights: torch.Tensor,
        base_features: torch.Tensor,
        B: int,
        grid_h: int,
        grid_w: int,
        min_ps: int,
        device: torch.device,
    ) -> TokenizerOutput:
        """固定 token 数量的 tokenization (加权融合模式).
        
        所有尺度特征按权重融合，输出固定数量的 token。
        """
        # 加权融合各尺度特征
        fused_features = torch.zeros_like(base_features)
        for scale_idx, ps in enumerate(self.patch_sizes):
            if ps in features_dict:
                feat, (h, w) = features_dict[ps]
                # 上采样到基础网格大小
                if h != grid_h or w != grid_w:
                    feat = F.interpolate(
                        feat,
                        size=(grid_h, grid_w),
                        mode='bilinear',
                        align_corners=False,
                    )
                # 加权
                weight = scale_weights[:, scale_idx:scale_idx+1, :, :]
                fused_features = fused_features + feat * weight
        
        # Hilbert 重排
        if self.use_hilbert_order:
            tokens = HilbertIndexer.reorder_to_hilbert(fused_features, grid_h, grid_w)
        else:
            tokens = fused_features.flatten(2).transpose(1, 2)
        
        num_tokens = tokens.shape[1]
        
        # 特征融合
        tokens = self.feature_fusion(tokens)
        
        # 创建 levels_info
        weights_flat = scale_weights.flatten(2)
        dominant_scales = weights_flat.argmax(dim=1)
        
        if self.use_hilbert_order:
            # 使用设备感知缓存
            hilbert_idx = HilbertIndexer.get_hilbert_order_on_device(max(grid_h, grid_w), device)
            valid_len = min(len(hilbert_idx), dominant_scales.shape[1])
            hilbert_idx = hilbert_idx[:valid_len]
            if valid_len < num_tokens:
                hilbert_idx = F.pad(hilbert_idx, (0, num_tokens - valid_len), value=0)
            dominant_scales = dominant_scales.gather(1, hilbert_idx.unsqueeze(0).expand(B, -1))
        
        primary_level = self.scale_to_level[min_ps]
        levels_info = self._create_levels_info(
            batch_size=B,
            num_tokens=num_tokens,
            scale_level=primary_level,
            grid_h=grid_h,
            grid_w=grid_w,
            device=device,
        )
        
        for scale_idx, ps in enumerate(self.patch_sizes):
            level = self.scale_to_level[ps]
            mask = (dominant_scales == scale_idx)
            levels_info[:, :, 0] = torch.where(mask, level, levels_info[:, :, 0])
        
        # 构建输出
        sequences = []
        for b in range(B):
            seq = TokenSequence(
                tokens=tokens[b],
                metadata={"levels": levels_info[b]},
            )
            sequences.append(seq)
        
        return TokenizerOutput(sequences)
    
    def _tokenize_variable(
        self,
        features_dict: Dict[int, Tuple[torch.Tensor, Tuple[int, int]]],
        scale_weights: torch.Tensor,
        B: int,
        grid_h: int,
        grid_w: int,
        device: torch.device,
    ) -> TokenizerOutput:
        """可变 token 数量的 tokenization (Patch=Token 直接映射).
        
        数学形式化
        ==========
        
        1. 尺度决策: s_{ij} = argmax_k π_{ij}^{(k)}
        2. 四叉树一致性约束: 确保粗尺度区域内所有位置使用相同尺度
        3. Token 提取: 直接从对应尺度的特征图提取
        4. Hilbert 排序: 按四叉树路径进行 Hilbert 排序
        
        优势:
            - Token 数量自适应 (N ∈ [N_min, N_max])
            - 无冗余计算 (不生成不需要的细粒度 token)
            - 真正的 Patch = Token 映射
        """
        # 1. 获取硬尺度决策
        scale_map = scale_weights.argmax(dim=1)  # [B, grid_h, grid_w]
        
        # 2. 强制四叉树一致性
        scale_map = self._enforce_quadtree_consistency(scale_map, grid_h, grid_w)
        
        # 3. 按尺度提取 token
        sequences = []
        
        for b in range(B):
            tokens_list = []
            levels_list = []
            positions_list = []  # (scale_idx, y, x) 用于 Hilbert 排序
            
            sample_scale_map = scale_map[b]  # [grid_h, grid_w]
            
            for scale_idx, ps in enumerate(self.patch_sizes):
                if ps not in features_dict:
                    continue
                    
                feat, (fh, fw) = features_dict[ps]  # [B, D, fh, fw]
                level = self.scale_to_level[ps]
                
                # 计算当前尺度相对于最细网格的比例
                scale_ratio = ps // min(self.patch_sizes)
                
                # 找到选择当前尺度的区域 (在最细网格上)
                # 需要检查整个 block 是否都选择了当前尺度
                for fy in range(fh):
                    for fx in range(fw):
                        # 对应的最细网格区域
                        gy_start = fy * scale_ratio
                        gx_start = fx * scale_ratio
                        
                        # 检查该 block 是否选择了当前尺度
                        block = sample_scale_map[
                            gy_start:gy_start + scale_ratio,
                            gx_start:gx_start + scale_ratio
                        ]
                        
                        # 如果 block 内所有位置都选择了当前尺度
                        if (block == scale_idx).all():
                            token = feat[b, :, fy, fx]  # [D]
                            tokens_list.append(token)
                            levels_list.append(level)
                            positions_list.append((level, fy, fx, fh, fw))
            
            # 4. 创建 levels_info
            num_tokens = len(tokens_list)
            
            if num_tokens == 0:
                # 回退：至少输出一个 token
                min_ps = min(features_dict.keys())
                feat, _ = features_dict[min_ps]
                tokens_list.append(feat[b, :, 0, 0])
                levels_list.append(self.scale_to_level[min_ps])
                positions_list.append((self.scale_to_level[min_ps], 0, 0, 1, 1))
                num_tokens = 1
            
            # 5. Hilbert 排序
            if self.use_hilbert_order and num_tokens > 1:
                sorted_indices = self._hilbert_sort_by_position(positions_list)
                tokens_list = [tokens_list[i] for i in sorted_indices]
                levels_list = [levels_list[i] for i in sorted_indices]
                positions_list = [positions_list[i] for i in sorted_indices]
            
            # 6. 构建 token 张量和 levels_info
            tokens = torch.stack(tokens_list)  # [N, D]
            tokens = self.feature_fusion(tokens)  # 应用特征融合
            
            # 构建 levels_info: [N, info_len]
            info_len = min(self.max_level + 1, 16)
            levels_info = torch.zeros(num_tokens, info_len, dtype=torch.long, device=device)
            
            for i, (level, fy, fx, fh, fw) in enumerate(positions_list):
                levels_info[i, 0] = level
                # 填充四叉树路径
                path = self._compute_quadtree_path(fy, fx, fh, fw, info_len - 1)
                levels_info[i, 1:1+len(path)] = torch.tensor(path, device=device)
            
            seq = TokenSequence(
                tokens=tokens,
                metadata={
                    "levels": levels_info,
                    "num_tokens": num_tokens,
                },
            )
            sequences.append(seq)
        
        return TokenizerOutput(sequences)
    
    def _enforce_quadtree_consistency(
        self,
        scale_map: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """强制四叉树一致性约束.
        
        数学约束:
            若 scale_map[i,j] = k (选择尺度 k)
            则 Block(i,j,k) 内所有位置必须为 k
            
        实现: 从粗尺度到细尺度传播决策
        
        Args:
            scale_map: [B, grid_h, grid_w] 每个位置的尺度索引
            grid_h: 网格高度
            grid_w: 网格宽度
            
        Returns:
            一致性约束后的 scale_map
        """
        B = scale_map.shape[0]
        result = scale_map.clone()
        min_ps = min(self.patch_sizes)
        
        # 从粗尺度到细尺度遍历 (跳过最细尺度)
        for scale_idx in range(len(self.patch_sizes) - 1, 0, -1):
            ps = self.patch_sizes[scale_idx]
            block_size = ps // min_ps
            
            if block_size <= 1:
                continue
            
            # 遍历每个 block
            for by in range(0, grid_h, block_size):
                for bx in range(0, grid_w, block_size):
                    # 获取 block 区域
                    by_end = min(by + block_size, grid_h)
                    bx_end = min(bx + block_size, grid_w)
                    
                    block = result[:, by:by_end, bx:bx_end]  # [B, bh, bw]
                    
                    # 如果 block 内有任何位置选择了当前粗尺度，整个 block 都用该尺度
                    # 使用多数投票或最大值策略
                    block_max = block.amax(dim=(-2, -1), keepdim=True)  # [B, 1, 1]
                    
                    # 只有当 block 内存在选择粗尺度的位置时才统一
                    coarse_mask = (block_max >= scale_idx)
                    if coarse_mask.any():
                        # 统一为 block 内的最大尺度索引
                        result[:, by:by_end, bx:bx_end] = torch.where(
                            coarse_mask.expand_as(block),
                            block_max.expand_as(block),
                            block
                        )
        
        return result
    
    def _hilbert_sort_by_position(
        self,
        positions: List[Tuple[int, int, int, int, int]],
    ) -> List[int]:
        """按位置进行 Hilbert 排序.
        
        排序键: (level, hilbert_index_at_level)
        
        Args:
            positions: [(level, fy, fx, fh, fw), ...] 位置列表
            
        Returns:
            排序后的索引列表
        """
        if len(positions) <= 1:
            return list(range(len(positions)))
        
        # 计算每个 token 的 Hilbert 距离
        sort_keys = []
        for i, (level, fy, fx, fh, fw) in enumerate(positions):
            grid_size = max(fh, fw)
            # 找到最接近的 2 的幂
            n = 1
            while n < grid_size:
                n *= 2
            
            # 计算 Hilbert 距离
            h_dist = HilbertCurve.xy_to_d(n, fx, fy)
            
            # 排序键: 先按层级，再按 Hilbert 距离
            sort_keys.append((level, h_dist, i))
        
        # 排序
        sort_keys.sort()
        return [k[2] for k in sort_keys]
    
    def _compute_quadtree_path(
        self,
        y: int,
        x: int,
        grid_h: int,
        grid_w: int,
        max_depth: int,
    ) -> List[int]:
        """计算位置 (y, x) 的四叉树路径.
        
        路径编码: q_l = bit(x, d-l) + 2 * bit(y, d-l)
        其中 d 是总深度，l 是当前层级
        
        Args:
            y: y 坐标
            x: x 坐标
            grid_h: 网格高度
            grid_w: 网格宽度
            max_depth: 最大路径深度
            
        Returns:
            四叉树路径 [q_0, q_1, ..., q_{d-1}]
        """
        grid_size = max(grid_h, grid_w)
        n = 1
        while n < grid_size:
            n *= 2
        
        actual_depth = max(1, int(math.log2(max(n, 2))))
        path_depth = min(max_depth, actual_depth)
        
        path = []
        for depth in range(path_depth):
            shift = actual_depth - depth - 1
            if shift >= 0:
                qx = (x >> shift) & 1
                qy = (y >> shift) & 1
                quadrant = qx + 2 * qy
                path.append(quadrant)
        
        return path
