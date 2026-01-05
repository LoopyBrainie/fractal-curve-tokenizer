"""资源感知预算调度模块

数学形式化
==========

**Token-level FLOPS 估算**:
对于 Transformer，主要计算量来自注意力机制:
    FLOPS_attn = 4 · N · D²     (Q, K, V 投影 + 输出投影)
    FLOPS_attn_score = 2 · N² · D  (注意力分数计算)
    FLOPS_ffn = 8 · N · D²      (两层 FFN)

总计每层:
    FLOPS_layer = 4ND² + 2N²D + 8ND² = 12ND² + 2N²D

对于 L 层 Transformer:
    FLOPS_total = L · (12ND² + 2N²D)

**深度加权 Token 预算**:
深层 token 产生更高计算成本 (O(N²))，应该被惩罚:
    w(d) = e^{α·d}
    N_weighted = Σ_i w(depth_i)

**FLOPS 预算损失**:
    L_FLOPS = λ · ReLU((FLOPS_actual / FLOPS_budget) - 1)²
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class FLOPSConfig:
    """FLOPS 计算配置
    
    Attributes:
        embed_dim: 嵌入维度 D
        num_layers: Transformer 层数 L
        num_heads: 注意力头数
        mlp_ratio: FFN 扩展比例
    """
    embed_dim: int = 256
    num_layers: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0


def compute_transformer_flops(
    num_tokens: int,
    config: FLOPSConfig,
) -> int:
    """计算 Transformer 的 FLOPS
    
    数学形式:
        FLOPS = L · (12ND² + 2N²D)
        
    其中:
        L: 层数
        N: token 数量
        D: 嵌入维度
    
    Args:
        num_tokens: token 数量 N
        config: 模型配置
        
    Returns:
        FLOPS 估算值
    """
    N = num_tokens
    D = config.embed_dim
    L = config.num_layers
    
    # 每层 FLOPS
    # Attention: QKV 投影 (3 * N * D²) + 注意力分数 (2 * N² * D) + 输出投影 (N * D²)
    flops_attn = 4 * N * D * D + 2 * N * N * D
    
    # FFN: 两层线性变换 (2 * N * D * mlp_ratio * D * 2)
    flops_ffn = int(8 * N * D * D * config.mlp_ratio / 4)  # 简化: 8ND²
    
    flops_per_layer = flops_attn + flops_ffn
    
    return L * flops_per_layer


def compute_batch_flops(
    token_counts: Tensor,
    config: FLOPSConfig,
) -> Tensor:
    """计算 batch 中每个样本的 FLOPS
    
    Args:
        token_counts: [B] 每个样本的 token 数量
        config: 模型配置
        
    Returns:
        [B] 每个样本的 FLOPS
    """
    N = token_counts.float()
    D = config.embed_dim
    L = config.num_layers
    
    # FLOPS = L · (12ND² + 2N²D)
    flops = L * (12 * N * D * D + 2 * N * N * D)
    
    return flops


class FLOPSBudgetLoss(nn.Module):
    """FLOPS 预算损失
    
    数学形式:
        L = λ · ReLU((FLOPS_actual / FLOPS_budget) - 1)²
    
    当 FLOPS 超过预算时，产生二次惩罚。
    
    Args:
        budget_flops: FLOPS 预算
        lambda_weight: 损失权重
        config: FLOPS 计算配置
    """
    
    def __init__(
        self,
        budget_flops: float,
        lambda_weight: float = 0.01,
        config: Optional[FLOPSConfig] = None,
    ) -> None:
        super().__init__()
        
        self.budget_flops = budget_flops
        self.lambda_weight = lambda_weight
        self.config = config or FLOPSConfig()
    
    def forward(self, token_counts: Tensor) -> Tensor:
        """
        Args:
            token_counts: [B] 每个样本的 token 数量
            
        Returns:
            标量损失
        """
        # 计算实际 FLOPS
        actual_flops = compute_batch_flops(token_counts, self.config)
        mean_flops = actual_flops.mean()
        
        # 超出预算的比例
        excess_ratio = mean_flops / self.budget_flops - 1.0
        
        # ReLU + 二次惩罚
        loss = self.lambda_weight * torch.relu(excess_ratio) ** 2
        
        return loss


class DepthWeightedBudgetLoss(nn.Module):
    """深度加权 Token 预算损失
    
    数学形式:
        w(d) = e^{α·d}          # 深度权重
        N_weighted = Σ_i w(d_i) # 加权 token 数
        L = λ · ReLU((N_weighted / N_budget) - 1)²
    
    目的: 惩罚深层 token，鼓励浅层分割。
    
    Args:
        budget_tokens: token 预算 (加权后)
        alpha: 深度权重指数
        lambda_weight: 损失权重
        max_depth: 最大深度
    """
    
    def __init__(
        self,
        budget_tokens: float,
        alpha: float = 0.5,
        lambda_weight: float = 0.01,
        max_depth: int = 4,
    ) -> None:
        super().__init__()
        
        self.budget_tokens = budget_tokens
        self.alpha = alpha
        self.lambda_weight = lambda_weight
        self.max_depth = max_depth
        
        # 预计算深度权重
        depths = torch.arange(max_depth + 1, dtype=torch.float32)
        weights = torch.exp(alpha * depths)
        self.register_buffer("depth_weights", weights)
    
    def forward(
        self,
        depths: Tensor,
        token_counts: Tensor,
    ) -> Tensor:
        """
        Args:
            depths: [total_tokens] 每个 token 的深度
            token_counts: [B] 每个样本的 token 数量
            
        Returns:
            标量损失
        """
        # 获取深度权重
        depths_clamped = depths.clamp(0, self.max_depth)
        weights = self.depth_weights[depths_clamped]  # [total_tokens]
        
        # 加权 token 数
        weighted_count = weights.sum() / token_counts.size(0)  # 平均每样本
        
        # 超出预算比例
        excess_ratio = weighted_count / self.budget_tokens - 1.0
        
        # ReLU + 二次惩罚
        loss = self.lambda_weight * torch.relu(excess_ratio) ** 2
        
        return loss


class BudgetScheduler:
    """动态预算调度器
    
    数学形式:
        Budget(t) = Budget_max · (1 + η · cos(π · t / T))
    
    训练过程:
        - 初期: 预算宽松 (探索)
        - 后期: 预算收紧 (效率)
    
    Args:
        budget_max: 最大预算
        budget_min: 最小预算
        total_epochs: 总训练轮数
        schedule: 调度策略 ('cosine' | 'linear')
    """
    
    def __init__(
        self,
        budget_max: float,
        budget_min: float,
        total_epochs: int,
        schedule: str = "cosine",
    ) -> None:
        self.budget_max = budget_max
        self.budget_min = budget_min
        self.total_epochs = total_epochs
        self.schedule = schedule
        
        self._current_epoch = 0
    
    def step(self, epoch: int) -> float:
        """获取当前 epoch 的预算
        
        Args:
            epoch: 当前 epoch
            
        Returns:
            当前预算值
        """
        self._current_epoch = epoch
        progress = min(epoch / max(self.total_epochs - 1, 1), 1.0)
        
        if self.schedule == "cosine":
            # Cosine annealing: 从 max 到 min
            import math
            budget = self.budget_min + 0.5 * (self.budget_max - self.budget_min) * (
                1 + math.cos(math.pi * progress)
            )
        else:  # linear
            budget = self.budget_max - (self.budget_max - self.budget_min) * progress
        
        return budget
    
    @property
    def current_budget(self) -> float:
        return self.step(self._current_epoch)


def estimate_baseline_flops(
    image_size: int = 224,
    patch_size: int = 16,
    config: Optional[FLOPSConfig] = None,
) -> int:
    """估算标准 ViT 的基准 FLOPS
    
    用于设置预算参考值。
    
    Args:
        image_size: 输入图像尺寸
        patch_size: patch 尺寸
        config: 模型配置
        
    Returns:
        基准 FLOPS
    """
    config = config or FLOPSConfig()
    
    # 标准 patch 数量
    num_patches = (image_size // patch_size) ** 2
    num_tokens = num_patches + 1  # + CLS token
    
    return compute_transformer_flops(num_tokens, config)


# 导出
__all__ = [
    "FLOPSConfig",
    "compute_transformer_flops",
    "compute_batch_flops",
    "FLOPSBudgetLoss",
    "DepthWeightedBudgetLoss",
    "BudgetScheduler",
    "estimate_baseline_flops",
]
