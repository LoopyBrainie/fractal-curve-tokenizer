# -*- coding: utf-8 -*-
"""
Streaming Fractal Tokenizer V3

数学形式化
============

Tokenization 过程:
    T: R^{B × C × H × W} → (T ∈ R^{B × N × D}, L ∈ Z^{B × N})

Variable Depth Tokenization:
    N ∈ [N_min, N_max] 根据图像内容自适应
    Regions = AdaptiveQuadtreeSplit(I)
    Token_i = Pool(F[R_i]) * σ_d + E_d

复杂度分析
----------
StreamingFractalTokenizerV3:
    时间: O(C · H · W) + O(N_max · log N_max)
          ├─ 特征提取 (Conv):      O(C · H · W)         — 共享卷积
          ├─ 自适应分割 (Greedy):  O(N_max · log N_max) — 优先队列
          └─ 自适应分割 (DP):      O(4^D_max · D_max)   — 动态规划
    空间: O(C · H · W) + O(N_max · D)
          ├─ 特征图:  O(C · H · W)
          └─ Token:   O(N_max · D)

其中: C=channels, H×W=image_size, N_max=max_tokens, D=d_model, D_max=max_depth
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .base_tokenizer import BaseTokenizer, TokenizerOutput, TokenSequence
from .config_fractal import FractalConfig


class StreamingFractalTokenizerV3(BaseTokenizer):
    """Variable Depth Tokenizer with Adaptive Quadtree Splitting.
    
    数学形式化
    ==========
    
    新架构 (Variable Depth Tokens):
        Regions = AdaptiveQuadtreeSplit(I)  # 内容自适应分割
        F = SharedConv(I)                    # 共享特征提取
        Token_i = Pool(F[R_i]) * σ_d + E_d  # 区域池化 + 深度编码
    
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
        depth_scale_range: (P6-1) 深度缩放范围 (σ_min, σ_max)，默认 (0.5, 2.0)
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
        depth_scale_range: Optional[Tuple[float, float]] = (0.5, 2.0),
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
        
        # Hilbert-Native Patch Embedding (P6-1: 支持可学习深度缩放)
        from .embed_hilbert_patch import HilbertNativePatchEmbed
        self.patch_embed = HilbertNativePatchEmbed(
            channels=channels,
            dim=d_model,
            base_patch_size=base_patch_size,
            max_depth=max_depth,
            conv_layers=2,
            use_batch_norm=True,
            depth_scale_range=depth_scale_range,
        )
        
        # Adaptive Quadtree Splitter
        from .split_adaptive import (
            AdaptiveSplitConfig,
            BalancedGreedySplitter,
            FixedBudgetDPSplitter,
            SplitScheme,
        )
        
        if split_scheme == 'fixed_budget_dp' or split_scheme == SplitScheme.FIXED_BUDGET_DP:
            split_config = AdaptiveSplitConfig.scheme_c(
                token_budget=target_tokens if target_tokens else 64,
                max_depth=max_depth,
                alpha=complexity_alpha,
            )
            self.splitter = FixedBudgetDPSplitter(split_config)
        else:
            split_config = AdaptiveSplitConfig.scheme_b(
                max_depth=max_depth,
                alpha=complexity_alpha,
                enforce_balance=enforce_balance,
                target_tokens=target_tokens,
            )
            self.splitter = BalancedGreedySplitter(split_config)
        
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
        """Variable Depth tokenization."""
        if images.dim() != 4:
            raise ValueError(
                f"StreamingFractalTokenizerV3.tokenize expects 4D input [B, C, H, W], "
                f"got {images.dim()}D tensor."
            )
        
        B, C, H, W = images.shape
        
        # 1. Adaptive Quadtree Splitting
        split_results = self.splitter.split_batch(images)
        
        self._last_split_stats = {
            'num_tokens': [sr.num_tokens for sr in split_results],
            'depth_distributions': [sr.depth_distribution for sr in split_results],
        }
        
        # 2. Hilbert-Native Patch Embedding
        tokens, levels_info = self.patch_embed(images, split_results)
        
        # 3. 构建输出
        sequences = []
        for b in range(B):
            num_tokens = split_results[b].num_tokens
            seq = TokenSequence(
                tokens=tokens[b, :num_tokens],
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
        """获取最近一次分割的统计信息."""
        return self._last_split_stats
    
    def get_entropy_loss(self) -> Optional[torch.Tensor]:
        """获取熵正则化损失 (Variable Depth 架构不需要)."""
        return None
    
    def get_scale_entropy(self) -> Optional[float]:
        """获取尺度分布熵值."""
        if self._last_split_stats is None:
            return None
        
        total_dist: Dict[int, int] = {}
        for dist in self._last_split_stats['depth_distributions']:
            for d, count in dist.items():
                total_dist[d] = total_dist.get(d, 0) + count
        
        total = sum(total_dist.values())
        if total == 0:
            return None
        
        entropy = 0.0
        for count in total_dist.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log(p)
        
        return entropy
    
    @torch.no_grad()
    def compute_scale_distribution(self, images: torch.Tensor) -> Dict[str, Any]:
        """计算深度分布统计信息."""
        _ = self.tokenize(images)
        
        if self._last_split_stats is None:
            return {
                'scale_ratios': {},
                'entropy': 0.0,
                'max_entropy': 0.0,
                'dominant_scale': self.base_patch_size,
                'depth_distribution': {},
            }
        
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
        
        scale_ratios: Dict[int, float] = {}
        for depth, count in total_dist.items():
            ps = self.base_patch_size * (2 ** (self.max_depth - depth))
            scale_ratios[ps] = count / total_tokens
        
        entropy = 0.0
        for count in total_dist.values():
            p = count / total_tokens
            if p > 0:
                entropy -= p * math.log(p)
        
        num_depths = self.max_depth + 1
        max_entropy = math.log(num_depths) if num_depths > 1 else 0.0
        
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
