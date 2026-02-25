# -*- coding: utf-8 -*-
"""
Conditional Layer Normalization for Scale-Aware Token Processing

I-PHASE4: 为分形 ViT 添加深度/面积感知的归一化

数学形式化
==========

标准 LayerNorm:
    y = γ · (x - μ) / σ + β
    其中 γ, β 是全局可学习的固定参数

Conditional LayerNorm:
    y = γ(d) · (x - μ) / σ + β(d)
    其中 γ(d), β(d) 由 token 深度 d 动态生成

优势:
    1. 浅层 token (大面积) 和深层 token (小面积) 可以有不同的归一化策略
    2. 深层 token 梯度流得以改善
    3. 模型可以学习不同尺度下的最优激活分布
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalLayerNorm(nn.Module):
    """条件层归一化 - 根据深度/面积条件动态调整归一化参数

    数学形式:
        γ_d = W_γ @ one_hot(d)  (或投影)
        β_d = W_β @ one_hot(d)

        y = γ_d · (x - μ) / σ + β_d

    Args:
        normalized_shape: 归一化的维度 (通常为 dim)
        num_conditions: 条件数量 (通常为 max_level + 1)
        condition_type: 条件类型 - 'depth' 或 'area'
        eps: 数值稳定性常数
    """

    def __init__(
        self,
        normalized_shape: int,
        num_conditions: int = 9,
        condition_type: str = 'depth',
        eps: float = 1e-5,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.num_conditions = num_conditions
        self.condition_type = condition_type
        self.eps = eps

        # 条件嵌入表: 每个深度/面积级别对应一组 γ, β
        # 参数量: num_conditions * normalized_shape * 2
        self.condition_weight = nn.Parameter(
            torch.zeros(num_conditions, normalized_shape, 2)  # [C, D, 2] - last dim: [gamma, beta]
        )

        # 全局可学习的缩放和偏移 (可选，用于微调)
        self.gamma_scale = nn.Parameter(torch.ones(1))
        self.beta_scale = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        """初始化条件权重

        策略:
            - 默认使用标准 LayerNorm 初始化
            - 第一个条件 (d=0) 使用较小的缩放，避免初始过度归一化
        """
        with torch.no_grad():
            # γ 初始化为 1，β 初始化为 0
            self.condition_weight[:, :, 0].fill_(1.0)  # gamma
            self.condition_weight[:, :, 1].fill_(0.0)  # beta

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """条件层归一化前向传播

        Args:
            x: [B, N, D] 输入张量
            condition: [B, N] 条件索引 (通常是深度)

        Returns:
            output: [B, N, D] 归一化后的输出
        """
        B, N, D = x.shape

        # 计算标准 LayerNorm 的均值和方差
        mean = x.mean(dim=-1, keepdim=True)  # [B, N, 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # [B, N, 1]
        std = (var + self.eps).sqrt()  # [B, N, 1]

        # 标准化
        x_normalized = (x - mean) / std  # [B, N, D]

        # 获取条件参数
        # condition: [B, N] -> [B, N, 1]
        cond_clamped = condition.clamp(0, self.num_conditions - 1).long()

        # 查找条件权重: [B, N, D, 2]
        cond_weight = self.condition_weight[cond_clamped]  # [B, N, D, 2]

        # 分离 γ 和 β
        gamma = cond_weight[..., 0]  # [B, N, D]
        beta = cond_weight[..., 1]    # [B, N, D]

        # 应用全局缩放
        gamma = gamma * self.gamma_scale
        beta = beta + self.beta_scale

        # 应用条件归一化
        output = x_normalized * gamma + beta

        return output


class AdaptiveLayerNorm(nn.Module):
    """自适应层归一化 - 基于输入特征动态生成归一化参数

    与 Conditional LN 的区别:
        - Conditional: 参数由离散条件 (深度) 决定
        - Adaptive: 参数由输入特征的 MLP 生成

    数学形式:
        γ, β = MLP(x.mean(dim=-2))  # 基于序列级统计量

    适用于:
        - 序列长度变化
        - 条件无法提前预知
    """

    def __init__(
        self,
        normalized_shape: int,
        hidden_dim: int = 128,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        # 自适应参数生成网络
        self.adaptor = nn.Sequential(
            nn.Linear(normalized_shape, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, normalized_shape * 2),  # 输出 gamma 和 beta
        )

        # 全局默认参数
        self.register_buffer('default_gamma', torch.ones(1, 1, normalized_shape))
        self.register_buffer('default_beta', torch.zeros(1, 1, normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """自适应层归一化

        Args:
            x: [B, N, D] 输入张量

        Returns:
            output: [B, N, D] 归一化后的输出
        """
        B, N, D = x.shape

        # 计算标准统计量
        mean = x.mean(dim=-1, keepdim=True)  # [B, N, 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # [B, N, 1]
        std = (var + self.eps).sqrt()  # [B, N, 1]

        # 标准化
        x_normalized = (x - mean) / std

        # 生成自适应参数
        # 使用序列级统计量作为条件
        seq_stats = x.mean(dim=1, keepdim=True)  # [B, 1, D]
        adapt_params = self.adaptor(seq_stats)  # [B, 1, D*2]

        # 分离 gamma 和 beta
        gamma = adapt_params[..., :D] + self.default_gamma  # [B, 1, D]
        beta = adapt_params[..., D:] + self.default_beta    # [B, 1, D]

        # 广播到序列维度
        gamma = gamma.expand(B, N, D)
        beta = beta.expand(B, N, D)

        # 应用
        output = x_normalized * gamma + beta

        return output


class ScaleAwareNorm(nn.Module):
    """尺度感知归一化 - 结合面积信息的综合归一化

    融合了:
        1. 标准 LayerNorm 的统计归一化
        2. 深度/面积条件的参数调整

    适用于分形 ViT 的多尺度 token 处理
    """

    def __init__(
        self,
        dim: int,
        max_level: int = 8,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.max_level = max_level
        self.eps = eps

        # 1. 标准 LayerNorm (用于基础统计归一化)
        self.ln = nn.LayerNorm(dim, eps=eps)

        # 2. 深度感知调制 (用于尺度调整)
        # 根据深度调整激活分布
        self.depth_gamma = nn.Embedding(max_level + 1, dim)
        self.depth_beta = nn.Embedding(max_level + 1, dim)

        # 3. 面积编码 (指数衰减)
        self.register_buffer(
            'area_weights',
            torch.tensor([4.0 ** (-d) for d in range(max_level + 1)])
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.ones_(self.depth_gamma.weight)
        nn.init.zeros_(self.depth_beta.weight)

    def forward(
        self,
        x: torch.Tensor,
        depths: torch.Tensor,
    ) -> torch.Tensor:
        """尺度感知归一化

        Args:
            x: [B, N, D] 输入张量
            depths: [B, N] token 深度

        Returns:
            output: [B, N, D] 归一化后的输出
        """
        # 1. 基础 LayerNorm
        x_normed = self.ln(x)

        # 2. 获取深度调制参数
        depths_clamped = depths.clamp(0, self.max_level)
        gamma = self.depth_gamma(depths_clamped)  # [B, N, D]
        beta = self.depth_beta(depths_clamped)    # [B, N, D]

        # 3. 应用面积缩放 (浅层 token 需要更强的归一化)
        area_scale = self.area_weights[depths_clamped].unsqueeze(-1)  # [B, N, 1]

        # 组合: base_normed * depth_gamma + depth_beta
        # 面积权重用于调整不同尺度 token 的激活强度
        output = x_normed * (gamma * area_scale) + beta

        return output
