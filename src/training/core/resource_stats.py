# -*- coding: utf-8 -*-
"""
模型资源统计接口

定义模型应暴露的统计信息，用于训练器的资源感知约束。

设计原则
========
1. 只读接口：训练器读取但不修改
2. 解耦设计：模型不知道统计如何被使用
3. 可扩展性：新增统计不影响现有组件

数学形式化
==========

Token 统计:
    N_total: 总 token 数
    N_d: 深度 d 的 token 数
    depth_distribution: [N_0, N_1, ..., N_D]
    
深度加权 Token 数:
    N_weighted = Σ_d N_d · w(d)
    
    其中 w(d) = exp(α·d) 惩罚深层 token

FLOPS 计算:
    FLOPS_tokenizer = N_regions · C · M_complexity
    FLOPS_attention = N_tokens · N_tokens · D
    FLOPS_ffn = N_tokens · D · 4D
    FLOPS_total = FLOPS_tokenizer + L · (FLOPS_attention + FLOPS_ffn)
    
深度熵:
    p_d = N_d / N_total
    H = -Σ_d p_d log p_d
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class ModelResourceStats:
    """
    模型资源使用统计（只读数据类）
    
    由模型在前向传播后计算，训练器通过此接口监控资源使用。
    
    属性
    ----
    avg_tokens_per_image : float
        每张图像的平均 token 数
    token_depth_distribution : List[float]
        各深度的 token 数分布 [N_0, N_1, ..., N_D]
    total_flops : float
        总 FLOPS (浮点运算次数)
    flops_breakdown : Dict[str, float]
        各模块的 FLOPS 分解
    avg_depth : float
        平均 token 深度
    depth_entropy : float
        深度分布的 Shannon 熵
    """
    
    # Token 统计
    avg_tokens_per_image: float = 0.0
    token_depth_distribution: List[float] = field(default_factory=list)
    
    # FLOPS 统计
    total_flops: float = 0.0
    flops_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # 深度统计
    avg_depth: float = 0.0
    depth_entropy: float = 0.0
    
    # 额外统计（可选）
    memory_allocated: Optional[float] = None  # MB
    num_parameters: Optional[int] = None
    
    @property
    def num_depth_levels(self) -> int:
        """深度层级数"""
        return len(self.token_depth_distribution)
    
    def get_weighted_token_count(self, alpha: float = 0.1) -> float:
        """
        计算深度加权 token 数
        
        数学公式:
            N_weighted = Σ_d N_d · exp(α·d)
            
        参数
        ----
        alpha : float, optional (default=0.1)
            深度惩罚系数
            - α = 0: 无权重 (N_weighted = N_total)
            - α > 0: 惩罚深层 token
            
        返回
        ----
        weighted_count : float
            加权 token 数
            
        示例
        ----
        >>> stats = ModelResourceStats(
        ...     token_depth_distribution=[50, 30, 20]  # 深度 0,1,2
        ... )
        >>> stats.get_weighted_token_count(alpha=0.1)
        104.9  # 50*1.0 + 30*1.105 + 20*1.221
        """
        if not self.token_depth_distribution:
            return 0.0
        
        weights = np.exp(alpha * np.arange(len(self.token_depth_distribution)))
        weighted_sum = np.sum(np.array(self.token_depth_distribution) * weights)
        
        return float(weighted_sum)
    
    def get_flops_per_token(self) -> float:
        """
        计算每个 token 的平均 FLOPS
        
        返回
        ----
        flops_per_token : float
            平均每个 token 消耗的 FLOPS
        """
        if self.avg_tokens_per_image == 0:
            return 0.0
        return self.total_flops / self.avg_tokens_per_image
    
    def to_dict(self) -> dict:
        """转换为字典（用于日志记录）"""
        return {
            "avg_tokens": self.avg_tokens_per_image,
            "total_flops": self.total_flops,
            "avg_depth": self.avg_depth,
            "depth_entropy": self.depth_entropy,
            "weighted_tokens_01": self.get_weighted_token_count(0.1),
            "flops_per_token": self.get_flops_per_token(),
            **self.flops_breakdown,
        }
    
    @classmethod
    def from_model_output(
        cls,
        token_output,  # TokenizerOutput
        model_config: dict,
        batch_size: int = 1,
    ) -> ModelResourceStats:
        """
        从模型输出计算统计信息
        
        参数
        ----
        token_output : TokenizerOutput
            Tokenizer 的输出
        model_config : dict
            模型配置 (dim, depth, heads, etc.)
        batch_size : int
            批次大小
            
        返回
        ----
        stats : ModelResourceStats
            资源统计对象
        """
        # Token 统计
        if hasattr(token_output, 'actual_token_count'):
            avg_tokens = token_output.actual_token_count.float().mean().item()
        else:
            avg_tokens = token_output.tokens.size(1) if hasattr(token_output, 'tokens') else 0.0
        
        # 深度分布
        if hasattr(token_output, 'depths'):
            depths = token_output.depths.cpu().numpy()
            max_depth = model_config.get('max_level', 4)
            depth_dist = np.bincount(depths.flatten(), minlength=max_depth+1).tolist()
            
            # 深度熵
            depth_probs = np.array(depth_dist) / (np.sum(depth_dist) + 1e-10)
            depth_entropy = -np.sum(depth_probs * np.log(depth_probs + 1e-10))
            
            # 平均深度
            avg_depth = np.mean(depths)
        else:
            depth_dist = []
            depth_entropy = 0.0
            avg_depth = 0.0
        
        # FLOPS 计算
        flops_breakdown = cls._compute_flops_breakdown(
            avg_tokens=avg_tokens,
            model_config=model_config,
            batch_size=batch_size,
        )
        total_flops = sum(flops_breakdown.values())
        
        return cls(
            avg_tokens_per_image=avg_tokens,
            token_depth_distribution=depth_dist,
            total_flops=total_flops,
            flops_breakdown=flops_breakdown,
            avg_depth=avg_depth,
            depth_entropy=depth_entropy,
        )
    
    @staticmethod
    def _compute_flops_breakdown(
        avg_tokens: float,
        model_config: dict,
        batch_size: int,
    ) -> Dict[str, float]:
        """
        计算各模块 FLOPS
        
        数学公式
        --------
        Tokenizer (ROI-Align + MLP):
            FLOPS = N_regions · C · (k² + M_mlp)
            
        Attention (每层):
            Q, K, V projection: 3 · N · D · D = 3ND²
            Attention weights: N · N · D
            Output projection: N · D · D = ND²
            Total per layer: 4ND² + N²D
            
        FFN (每层):
            Up projection: N · D · 4D = 4ND²
            Down projection: N · 4D · D = 4ND²
            Total per layer: 8ND²
            
        Total:
            FLOPS = FLOPS_tokenizer + L · (4ND² + N²D + 8ND²)
                  = FLOPS_tokenizer + L · (12ND² + N²D)
        """
        D = model_config.get('dim', 384)
        L = model_config.get('depth', 10)
        C = model_config.get('channels', 512)  # 特征图通道数
        
        N = avg_tokens * batch_size
        
        # Tokenizer FLOPS (近似)
        # ROI-Align: N_regions · C · k²
        # MLP: N_regions · C · M_mlp
        flops_tokenizer = N * C * (49 + 256)  # k=7, M_mlp≈256
        
        # Attention FLOPS (per layer)
        flops_attn_per_layer = 4 * N * D * D + N * N * D
        flops_attn_total = L * flops_attn_per_layer
        
        # FFN FLOPS (per layer)
        flops_ffn_per_layer = 8 * N * D * D
        flops_ffn_total = L * flops_ffn_per_layer
        
        # Classification head
        flops_head = N * D * model_config.get('num_classes', 200)
        
        return {
            'tokenizer': flops_tokenizer,
            'attention': flops_attn_total,
            'ffn': flops_ffn_total,
            'head': flops_head,
        }


def compute_resource_stats_batch(
    model,
    batch_input: torch.Tensor,
    return_detailed: bool = False,
) -> ModelResourceStats:
    """
    计算一个 batch 的资源统计
    
    工具函数，用于在训练循环中快速获取统计信息。
    
    参数
    ----
    model : nn.Module
        模型实例（应有 get_tokenizer_output 方法）
    batch_input : Tensor
        输入张量 [B, C, H, W]
    return_detailed : bool, optional
        是否返回详细的每样本统计
        
    返回
    ----
    stats : ModelResourceStats
        批次平均资源统计
        
    示例
    ----
    >>> model = FractalViT(...)
    >>> images = torch.randn(32, 3, 224, 224)
    >>> stats = compute_resource_stats_batch(model, images)
    >>> print(f"Avg tokens: {stats.avg_tokens_per_image:.1f}")
    >>> print(f"Total FLOPS: {stats.total_flops / 1e9:.2f} G")
    """
    with torch.no_grad():
        # 获取 tokenizer 输出
        if hasattr(model, 'get_tokenizer_output'):
            token_output = model.get_tokenizer_output(batch_input)
        else:
            # 如果模型没有专门接口，通过 forward 获取
            _ = model(batch_input)
            if hasattr(model.tokenizer, 'last_output'):
                token_output = model.tokenizer.last_output
            else:
                raise ValueError("Model must provide tokenizer output interface")
    
    # 构建配置字典
    model_config = {
        'dim': getattr(model, 'dim', 384),
        'depth': getattr(model, 'depth', 10),
        'max_level': getattr(model, 'max_level', 4),
        'num_classes': getattr(model, 'num_classes', 200),
    }
    
    stats = ModelResourceStats.from_model_output(
        token_output,
        model_config,
        batch_size=batch_input.size(0),
    )
    
    return stats
