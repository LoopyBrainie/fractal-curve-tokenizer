# -*- coding: utf-8 -*-
"""
Streaming Fractal Tokenizer - 统一架构实现

数学形式化
============

Tokenization 过程:
    T: R^{B × C × H × W} → (T ∈ R^{B × N × D}, L ∈ Z^{B × N})

其中 N = (H/p) × (W/p) 是固定的 token 数量。

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
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hilbert import HilbertCurve
from .tokenization import BaseTokenizer, TokenizerOutput, TokenSequence


class HilbertIndexer:
    """预计算 Hilbert 曲线索引，用于特征重排序.
    
    对于给定的网格大小，生成从光栅顺序到 Hilbert 顺序的映射。
    使用 LRU 缓存避免重复计算。
    """
    
    @staticmethod
    @lru_cache(maxsize=64)
    def get_hilbert_order(grid_size: int) -> torch.Tensor:
        """获取 grid_size x grid_size 网格的 Hilbert 遍历顺序.
        
        Args:
            grid_size: 网格边长 (必须是 2 的幂)
            
        Returns:
            indices: [grid_size^2] 长的索引张量，将光栅顺序映射到 Hilbert 顺序
        """
        if grid_size <= 0:
            return torch.tensor([], dtype=torch.long)
        
        # 找到最接近的 2 的幂
        n = 1
        while n < grid_size:
            n *= 2
        
        total_cells = grid_size * grid_size
        
        # 生成 Hilbert 距离到坐标的映射
        positions = []
        for d in range(n * n):
            x, y = HilbertCurve.d_to_xy(n, d)
            if x < grid_size and y < grid_size:
                # 光栅顺序索引: y * width + x
                raster_idx = y * grid_size + x
                positions.append(raster_idx)
        
        # 返回的是按 Hilbert 顺序排列的光栅索引
        return torch.tensor(positions[:total_cells], dtype=torch.long)
    
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
        
        # Flatten 到 [B, D, H*W]
        flat = features.flatten(2)  # [B, D, H*W]
        
        # 获取 Hilbert 顺序索引
        # 注意：对于非正方形，我们使用较大边作为基准
        grid_size = max(H, W)
        
        if grid_size <= 1:
            return flat.transpose(1, 2)  # [B, H*W, D]
        
        # 生成完整的 Hilbert 顺序
        hilbert_indices = HilbertIndexer.get_hilbert_order(grid_size).to(device)
        
        # 过滤出有效的索引（对于非正方形网格）
        if H != W:
            # 对于非正方形，我们需要重新映射
            valid_indices = []
            for d in range(grid_size * grid_size):
                x, y = HilbertCurve.d_to_xy(grid_size, d)
                if x < W and y < H:
                    raster_idx = y * W + x
                    valid_indices.append(raster_idx)
            hilbert_indices = torch.tensor(valid_indices, dtype=torch.long, device=device)
        
        # 使用索引重排
        # flat: [B, D, H*W], indices: [H*W]
        reordered = flat.index_select(2, hilbert_indices)  # [B, D, H*W]
        
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
        
        Args:
            batch_size: 批次大小
            num_tokens: token 数量
            scale_level: 当前尺度对应的层级
            grid_h: 网格高度
            grid_w: 网格宽度
            device: 设备
            
        Returns:
            levels_info: [B, num_tokens, info_len] 包含深度和路径信息
        """
        # 信息长度：depth + 最多 max_level 个路径节点
        info_len = min(self.max_level + 1, 16)  # 限制最大长度
        
        levels_info = torch.zeros(
            batch_size, num_tokens, info_len,
            dtype=torch.long, device=device
        )
        
        # 设置深度（第0列）
        levels_info[:, :, 0] = scale_level
        
        # 生成简化的路径信息
        # 对于网格中的每个位置，计算其 Hilbert 路径
        if self.use_hilbert_order:
            grid_size = max(grid_h, grid_w)
            n = 1
            while n < grid_size:
                n *= 2
            
            for token_idx in range(num_tokens):
                if token_idx < grid_h * grid_w:
                    # 从 token 索引恢复网格坐标
                    # 如果使用 Hilbert 顺序，token_idx 对应 Hilbert 距离
                    if n >= 2:
                        x, y = HilbertCurve.d_to_xy(n, token_idx % (n * n))
                        # 简化路径：使用象限索引
                        quadrant = (1 if x >= n // 2 else 0) + (2 if y >= n // 2 else 0)
                        if info_len > 1:
                            levels_info[:, token_idx, 1] = quadrant
        
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
    1. 区域复杂度估计器
    2. 多尺度特征融合
    3. 可变长度输出（需要 padding）
    
    .. note::
        此版本保留自适应分辨率特性，但使用 Gumbel-Softmax 替代 REINFORCE。
    """
    
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        channels: int = 3,
        d_model: int = 256,
        patch_sizes: Tuple[int, ...] = (4, 8, 16),
        use_hilbert_order: bool = True,
        max_level: int = 50,
        gumbel_temperature: float = 1.0,
    ) -> None:
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
        
        # 区域复杂度估计器
        self.complexity_estimator = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, len(patch_sizes), kernel_size=1),
        )
        
        # 可学习温度参数
        self.temperature = nn.Parameter(torch.tensor(gumbel_temperature))
        
    def _compute_scale_weights(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """计算每个区域的尺度权重.
        
        Args:
            images: [B, C, H, W]
            
        Returns:
            weights: [B, num_scales, H', W'] 每个区域的尺度权重
        """
        logits = self.complexity_estimator(images)  # [B, num_scales, H', W']
        
        # Gumbel-Softmax
        if self.training:
            weights = F.gumbel_softmax(
                logits,
                tau=self.temperature.clamp(min=0.1),
                hard=False,
                dim=1,
            )
        else:
            # 推理时使用硬决策
            hard_indices = logits.argmax(dim=1)  # [B, H', W']
            weights = F.one_hot(
                hard_indices, num_classes=len(self.patch_sizes)
            ).permute(0, 3, 1, 2).float()  # [B, num_scales, H', W']
        
        return weights
    
    def tokenize(self, images: torch.Tensor) -> TokenizerOutput:
        """区域自适应 tokenization.
        
        根据每个区域的复杂度，选择合适的分辨率尺度。
        
        .. warning::
            Phase 2 实现，可能产生可变长度序列。
        """
        if images.dim() != 4:
            raise ValueError(
                f"StreamingFractalTokenizerV2.tokenize expects 4D input [B, C, H, W], "
                f"got {images.dim()}D tensor."
            )
        
        B, C, H, W = images.shape
        device = images.device
        
        # 1. 计算区域尺度权重
        scale_weights = self._compute_scale_weights(images)  # [B, num_scales, H', W']
        
        # 2. 提取多尺度特征
        features_dict = self.encoder(images)
        
        # 3. 加权融合 (简化版：使用最小尺度的网格)
        min_ps = min(features_dict.keys())
        base_features, (grid_h, grid_w) = features_dict[min_ps]
        
        # 将权重插值到基础网格大小
        weights_resized = F.interpolate(
            scale_weights,
            size=(grid_h, grid_w),
            mode='bilinear',
            align_corners=False,
        )  # [B, num_scales, grid_h, grid_w]
        
        # 加权求和各尺度特征
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
                weight = weights_resized[:, scale_idx:scale_idx+1, :, :]  # [B, 1, H', W']
                fused_features = fused_features + feat * weight
        
        # 4. Hilbert 重排
        if self.use_hilbert_order:
            tokens = HilbertIndexer.reorder_to_hilbert(fused_features, grid_h, grid_w)
        else:
            tokens = fused_features.flatten(2).transpose(1, 2)
        
        num_tokens = tokens.shape[1]
        
        # 5. 特征融合
        tokens = self.feature_fusion(tokens)
        
        # 6. 创建 levels_info (使用主要尺度)
        # 计算每个 token 位置的主要尺度
        weights_flat = weights_resized.flatten(2)  # [B, num_scales, H'*W']
        dominant_scales = weights_flat.argmax(dim=1)  # [B, H'*W']
        
        # 如果使用 Hilbert 顺序，需要重排 dominant_scales
        if self.use_hilbert_order:
            hilbert_idx = HilbertIndexer.get_hilbert_order(max(grid_h, grid_w)).to(device)
            valid_len = min(len(hilbert_idx), dominant_scales.shape[1])
            hilbert_idx = hilbert_idx[:valid_len]
            if valid_len < num_tokens:
                hilbert_idx = F.pad(hilbert_idx, (0, num_tokens - valid_len), value=0)
            dominant_scales = dominant_scales.gather(1, hilbert_idx.unsqueeze(0).expand(B, -1))
        
        # 创建 levels_info
        info_len = min(self.max_level + 1, 16)
        levels_info = torch.zeros(B, num_tokens, info_len, dtype=torch.long, device=device)
        
        for scale_idx, ps in enumerate(self.patch_sizes):
            level = self.scale_to_level[ps]
            mask = (dominant_scales == scale_idx)
            levels_info[:, :, 0] = torch.where(mask, level, levels_info[:, :, 0])
        
        # 7. 构建输出
        sequences = []
        for b in range(B):
            seq = TokenSequence(
                tokens=tokens[b],
                metadata={"levels": levels_info[b]},
            )
            sequences.append(seq)
        
        return TokenizerOutput(sequences)
