# -*- coding: utf-8 -*-
"""
资源感知损失函数

针对 Hilbert Curve ViT 的资源约束损失函数，通过惩罚超预算的计算资源
和不合理的深度分布，引导模型在满足性能目标的同时保持资源效率。

数学形式化
==========

总体损失:
    L_total = L_task + λ_resource · L_resource
    
其中:
    L_task: 任务损失 (交叉熵、Focal Loss 等)
    L_resource: 资源惩罚项
    λ_resource: 资源损失权重 (典型值 0.01-0.1)

资源损失分解:
    L_resource = λ_flops · L_flops + λ_token · L_token + λ_entropy · L_entropy

1. FLOPS 约束损失
------------------
仅惩罚超预算的情况，使用 ReLU 软约束：

    L_flops = ReLU(FLOPS_actual / FLOPS_budget - 1)²

数学性质:
    - FLOPS < budget: L_flops = 0 (不惩罚)
    - FLOPS > budget: L_flops > 0 (二次增长)
    - 梯度连续性: dL/dFLOPS 在 FLOPS=budget 处光滑

计算验证:
    budget = 5e9, actual = 4e9 → L = 0
    budget = 5e9, actual = 6e9 → L = (1.2-1)² = 0.04
    budget = 5e9, actual = 7e9 → L = (1.4-1)² = 0.16

2. Token 数量约束损失
---------------------
使用深度加权 token 数量，惩罚深层 token：

    N_weighted = Σ_d N_d · exp(α·d)
    L_token = ReLU(N_weighted - N_budget)²

深度权重 (α=0.1):
    depth=0: w=1.00  (基准)
    depth=1: w=1.11  (+11%)
    depth=2: w=1.22  (+22%)
    depth=3: w=1.35  (+35%)
    depth=4: w=1.49  (+49%)

设计原理:
    - 深层 token (小 patch) 计算成本更高
    - 鼓励模型优先使用浅层 token
    - 权重呈指数增长，强制约束深层使用

计算验证:
    分布 [50, 30, 20] (depth 0,1,2), α=0.1:
        N_weighted = 50*1.0 + 30*1.105 + 20*1.221 = 107.6
    分布 [20, 30, 50] (depth 0,1,2), α=0.1:
        N_weighted = 20*1.0 + 30*1.105 + 50*1.221 = 114.6
    后者惩罚更大 → 鼓励浅层分布

3. 深度熵正则损失
------------------
防止深度坍缩（所有 token 集中在单一深度）：

    H_actual = -Σ_d p_d · log(p_d)

I111-2 自适应熵目标:
    H_target = log(D) × (1 - 1/√D)

    | D  | H_target/H_max | 含义                    |
    |----|----------------|------------------------|
    | 2  | 29%            | 强烈偏向浅层            |
    | 4  | 50%            | 中等深度利用            |
    | 8  | 65%            | 允许更均匀的深度分布    |

    L_entropy = (H_actual - H_target)²

数学性质:
    - H ∈ [0, log(D)]
    - H = 0: 完全坍缩（所有 token 同一深度）
    - H = log(D): 均匀分布（最大熵）

计算验证 (D=5):
    均匀分布 [0.2, 0.2, 0.2, 0.2, 0.2]:
        H = 1.609 (最大)
    自适应目标 (D=5): H_target = 1.609 × (1 - 1/√5) = 0.89
    坍缩分布 [1.0, 0, 0, 0, 0]:
        H = 0 (最小)

设计原则
========
1. 无侵入性: 通过损失函数间接约束，不修改模型架构
2. 可微分性: 所有组件支持反向传播
3. 可解释性: 每个损失项有明确的物理含义
4. 鲁棒性: 使用 ReLU 和二次函数避免梯度爆炸

示例
====
>>> from core.resource_stats import ModelResourceStats
>>> 
>>> # 创建资源损失
>>> resource_loss = ResourceAwareLoss(
...     flops_budget=5e9,
...     token_budget=128,
...     depth_weight_alpha=0.1,
...     lambda_flops=0.1,
...     lambda_token=0.01,
...     lambda_entropy=0.05,
... )
>>> 
>>> # 模拟资源统计
>>> stats = ModelResourceStats(
...     total_flops=6e9,  # 超预算 20%
...     token_depth_distribution=[50, 30, 20, 10, 5],
...     depth_entropy=1.2,
... )
>>> 
>>> # 计算损失
>>> loss = resource_loss(stats)
>>> print(f"Resource loss: {loss.item():.4f}")
Resource loss: 0.0240

组件
====
- ResourceAwareLoss: 主损失类
- compute_depth_entropy: 深度熵计算
- get_weighted_token_count: 深度加权 token 数
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from training.core.resource_stats import ModelResourceStats
from vit_pytorch.constants import EPS  # I112-3: 统一数值稳定性常量


def compute_depth_entropy(
    depth_distribution: torch.Tensor,
    epsilon: float = EPS,  # I112-3: 使用统一 EPS (1e-6)
) -> torch.Tensor:
    """
    计算深度分布的 Shannon 熵
    
    数学公式:
        H = -Σ_d p_d · log(p_d)
    
    其中 p_d = N_d / N_total 是深度 d 的概率。
    
    参数
    ----
    depth_distribution : Tensor [D+1]
        各深度的 token 数量 [N_0, N_1, ..., N_D]
    epsilon : float, optional
        数值稳定性参数，防止 log(0)
        
    返回
    ----
    entropy : Tensor (scalar)
        深度分布的熵
        
    示例
    ----
    >>> dist = torch.tensor([50.0, 30.0, 20.0])
    >>> H = compute_depth_entropy(dist)
    >>> print(f"Entropy: {H:.3f}")
    Entropy: 1.030
    
    >>> # 均匀分布应该有最大熵
    >>> uniform = torch.ones(5) * 20.0
    >>> H_uniform = compute_depth_entropy(uniform)
    >>> H_max_theoretical = np.log(5)
    >>> assert abs(H_uniform - H_max_theoretical) < 0.01
    """
    # 归一化为概率分布
    total = depth_distribution.sum() + epsilon
    probs = depth_distribution / total
    
    # Shannon 熵: H = -Σ p·log(p)
    # 注意: 当 p=0 时，p·log(p) → 0 (极限)
    log_probs = torch.log(probs + epsilon)
    entropy = -(probs * log_probs).sum()
    
    return entropy


def get_weighted_token_count(
    depth_distribution: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    计算深度加权 token 数量
    
    数学公式:
        N_weighted = Σ_d N_d · exp(α·d)
    
    权重函数 w(d) = exp(α·d) 呈指数增长，惩罚深层 token。
    
    参数
    ----
    depth_distribution : Tensor [D+1]
        各深度的 token 数量 [N_0, N_1, ..., N_D]
    alpha : float, optional (default=0.1)
        深度惩罚系数
        - α=0: 无权重 (N_weighted = N_total)
        - α>0: 惩罚深层 token
        - 典型值: 0.05-0.2
        
    返回
    ----
    weighted_count : Tensor (scalar)
        加权 token 数量
        
    计算验证
    --------
    >>> # 浅层分布
    >>> shallow = torch.tensor([50.0, 30.0, 20.0, 0.0, 0.0])
    >>> w_shallow = get_weighted_token_count(shallow, alpha=0.1)
    >>> # 深层分布
    >>> deep = torch.tensor([20.0, 30.0, 50.0, 0.0, 0.0])
    >>> w_deep = get_weighted_token_count(deep, alpha=0.1)
    >>> # 深层分布应该有更高的加权数量
    >>> assert w_deep > w_shallow
    
    >>> # α=0 时应该等于总数
    >>> dist = torch.tensor([50.0, 30.0, 20.0])
    >>> w_zero = get_weighted_token_count(dist, alpha=0.0)
    >>> assert abs(w_zero - 100.0) < 1e-5
    """
    num_depths = len(depth_distribution)
    depths = torch.arange(num_depths, device=depth_distribution.device, dtype=depth_distribution.dtype)
    
    # 权重: w(d) = exp(α·d)
    weights = torch.exp(alpha * depths)
    
    # 加权和
    weighted_count = (depth_distribution * weights).sum()
    
    return weighted_count


class ResourceAwareLoss(nn.Module):
    """
    资源感知损失函数
    
    通过多个损失项的加权组合，引导模型在满足任务性能的同时
    遵守计算资源约束和深度分布约束。
    
    数学形式:
        L_resource = λ_flops·L_flops + λ_token·L_token + λ_entropy·L_entropy
    
    参数
    ----
    flops_budget : float, optional (default=5e9)
        FLOPS 预算（浮点运算次数）
        - RTX 4070 Laptop: ~5 GFLOPS 合理
        - RTX 4090: ~10 GFLOPS
        
    token_budget : int, optional (default=128)
        Token 数量预算（深度加权后）
        - 标准 ViT-B/16 (224x224): 196 tokens
        - Fractal ViT (adaptive): 64-256 tokens
        
    depth_weight_alpha : float, optional (default=0.1)
        深度权重系数
        - 0.05: 温和惩罚
        - 0.1: 标准惩罚 (推荐)
        - 0.2: 强惩罚
        
    lambda_flops : float, optional (default=0.1)
        FLOPS 损失权重
        
    lambda_token : float, optional (default=0.01)
        Token 数量损失权重
        
    lambda_entropy : float, optional (default=0.05)
        深度熵损失权重
        
    target_entropy_ratio : float, optional (default=0.8)
        目标熵与最大熵的比率
        - 1.0: 要求均匀分布
        - 0.8: 允许一定程度的偏斜
        - 0.6: 更宽松的约束
        
    epsilon : float, optional (default=1e-10)
        数值稳定性参数
        
    属性
    ----
    flops_budget : float
        FLOPS 预算
    token_budget : int
        Token 数量预算
    alpha : float
        深度权重系数
    lambda_flops, lambda_token, lambda_entropy : float
        各损失项权重
        
    示例
    ----
    >>> # 创建资源损失
    >>> loss_fn = ResourceAwareLoss(
    ...     flops_budget=5e9,
    ...     token_budget=128,
    ...     depth_weight_alpha=0.1,
    ... )
    >>> 
    >>> # 前向传播
    >>> stats = ModelResourceStats(
    ...     total_flops=6e9,
    ...     token_depth_distribution=[50, 30, 20, 10, 5],
    ...     depth_entropy=1.2,
    ... )
    >>> loss = loss_fn(stats)
    >>> 
    >>> # 反向传播 (如果 stats 来自模型输出)
    >>> loss.backward()
    """
    
    def __init__(
        self,
        flops_budget: float = 5e9,
        token_budget: int = 128,
        depth_weight_alpha: float = 0.1,
        lambda_flops: float = 0.1,
        lambda_token: float = 0.01,
        lambda_entropy: float = 0.05,
        target_entropy_ratio: float = 0.8,
        epsilon: float = EPS,  # I112-3: 使用统一 EPS (1e-6)
    ):
        super().__init__()

        self.flops_budget = flops_budget
        self.token_budget = token_budget
        self.alpha = depth_weight_alpha
        self.lambda_flops = lambda_flops
        self.lambda_token = lambda_token
        self.lambda_entropy = lambda_entropy
        self.target_entropy_ratio = target_entropy_ratio
        self.epsilon = epsilon

        # I-OPT: 缓存设备避免每次 forward 检查 CUDA
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 预注册标量张量避免重复创建
        self._zero_tensor = torch.tensor(0.0, device=self._device, dtype=torch.float32)

        # 验证参数合法性
        assert flops_budget > 0, "FLOPS budget must be positive"
        assert token_budget > 0, "Token budget must be positive"
        assert 0 <= depth_weight_alpha <= 1, "Alpha must be in [0, 1]"
        assert 0 <= target_entropy_ratio <= 1, "Target entropy ratio must be in [0, 1]"
        
    def forward(
        self,
        stats: ModelResourceStats,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        计算资源感知损失

        参数
        ----
        stats : ModelResourceStats
            模型资源统计信息
        return_components : bool, optional
            是否返回各损失分量 (用于调试)

        返回
        ----
        loss : Tensor (scalar)
            总资源损失

        或 (loss, components) : (Tensor, dict)
            如果 return_components=True
        """
        # I-OPT: 使用缓存的设备避免重复检查
        device = self._device

        # 1. FLOPS 约束损失
        # P-OPT: 使用温和的 L¹ 惩罚避免 L² 平方放大效应
        # 原始 L²: ratio=2.0 → L=1.0 (放大 100%)
        # 新 L¹: ratio=2.0 → L=1.0 (线性，梯度稳定)
        flops_ratio = stats.total_flops / self.flops_budget
        flops_ratio_tensor = torch.tensor(flops_ratio, device=device, dtype=torch.float32)
        L_flops = F.relu(flops_ratio_tensor - 1.0)  # 线性惩罚

        # 2. Token 数量约束损失
        if stats.token_depth_distribution:
            depth_dist_tensor = torch.tensor(
                stats.token_depth_distribution,
                device=device,
                dtype=torch.float32
            )
            weighted_tokens = get_weighted_token_count(depth_dist_tensor, self.alpha)
        else:
            # I-OPT: 如果没有深度分布，使用平均 token 数
            weighted_tokens = torch.tensor(
                stats.avg_tokens_per_image,
                device=device,
                dtype=torch.float32
            )

        L_token = F.relu(weighted_tokens - self.token_budget)  # 线性惩罚

        # 3. 深度熵正则损失
        H_actual = None  # 初始化
        if stats.token_depth_distribution and len(stats.token_depth_distribution) > 1:
            # I-OPT: 复用已创建的 depth_dist_tensor
            depth_dist_tensor = torch.tensor(
                stats.token_depth_distribution,
                device=device,
                dtype=torch.float32
            )
            H_actual = compute_depth_entropy(depth_dist_tensor, self.epsilon)

            # I111-2: 自适应熵目标
            # H_target = log(D) × (1 - 1/√D)
            # 深度越深，允许的分布越均匀
            num_depths = len(stats.token_depth_distribution)
            adaptive_ratio = 1.0 - 1.0 / np.sqrt(num_depths)
            H_target = np.log(num_depths) * adaptive_ratio
            H_target_tensor = torch.tensor(H_target, device=device, dtype=torch.float32)

            L_entropy = (H_actual - H_target_tensor) ** 2
        else:
            # I-OPT: 如果深度信息不可用，跳过熵损失，使用预注册的零张量
            L_entropy = self._zero_tensor

        # 加权总损失
        total_loss = (
            self.lambda_flops * L_flops +
            self.lambda_token * L_token +
            self.lambda_entropy * L_entropy
        )

        if return_components:
            components = {
                "L_flops": L_flops.item(),
                "L_token": L_token.item(),
                "L_entropy": L_entropy.item(),
                "flops_ratio": flops_ratio,
                "weighted_tokens": weighted_tokens.item() if isinstance(weighted_tokens, torch.Tensor) else weighted_tokens,
                "depth_entropy": H_actual.item() if isinstance(H_actual, torch.Tensor) else 0.0,
            }
            return total_loss, components

        return total_loss
    
    def get_diagnostics(self, stats: ModelResourceStats) -> dict:
        """
        获取诊断信息（不计算梯度）
        
        参数
        ----
        stats : ModelResourceStats
            资源统计
            
        返回
        ----
        diagnostics : dict
            诊断信息，包括各损失分量和统计指标
        """
        with torch.no_grad():
            _, components = self.forward(stats, return_components=True)
        
        # 添加额外诊断信息
        diagnostics = {
            **components,
            "flops_budget": self.flops_budget,
            "token_budget": self.token_budget,
            "flops_usage_percent": components["flops_ratio"] * 100,
            "token_usage_percent": (components["weighted_tokens"] / self.token_budget) * 100,
        }
        
        # 深度分布统计
        if stats.token_depth_distribution:
            total_tokens = sum(stats.token_depth_distribution)
            depth_percentages = [
                (count / total_tokens) * 100
                for count in stats.token_depth_distribution
            ]
            diagnostics["depth_distribution_percent"] = depth_percentages
        
        return diagnostics
    
    def __repr__(self) -> str:
        return (
            f"ResourceAwareLoss(\n"
            f"  flops_budget={self.flops_budget:.2e},\n"
            f"  token_budget={self.token_budget},\n"
            f"  alpha={self.alpha},\n"
            f"  λ_flops={self.lambda_flops},\n"
            f"  λ_token={self.lambda_token},\n"
            f"  λ_entropy={self.lambda_entropy}\n"
            f")"
        )


# ============================================================================
# 工具函数
# ============================================================================

def validate_resource_loss():
    """
    计算验证：验证资源损失函数的正确性
    
    测试场景:
        1. FLOPS 约束测试
        2. Token 约束测试
        3. 深度熵测试
        4. 深度权重测试
    """
    print("=" * 60)
    print("资源损失函数计算验证")
    print("=" * 60)
    
    # 测试 1: FLOPS 约束
    print("\n[测试 1] FLOPS 约束")
    print("-" * 60)
    
    loss_fn = ResourceAwareLoss(flops_budget=5e9, lambda_flops=1.0, lambda_token=0.0, lambda_entropy=0.0)
    
    test_cases = [
        (4e9, "低于预算"),
        (5e9, "恰好预算"),
        (6e9, "超出 20%"),
        (7.5e9, "超出 50%"),
    ]
    
    for flops, desc in test_cases:
        stats = ModelResourceStats(total_flops=flops)
        loss, comp = loss_fn(stats, return_components=True)
        print(f"  {desc}: FLOPS={flops/1e9:.1f}G, L_flops={comp['L_flops']:.4f}")
    
    # 测试 2: Token 约束
    print("\n[测试 2] Token 数量约束")
    print("-" * 60)
    
    loss_fn = ResourceAwareLoss(
        token_budget=100,
        depth_weight_alpha=0.1,
        lambda_flops=0.0,
        lambda_token=1.0,
        lambda_entropy=0.0
    )
    
    distributions = [
        ([50, 30, 20], "浅层为主"),
        ([20, 30, 50], "深层为主"),
        ([33, 33, 34], "均匀分布"),
    ]
    
    for dist, desc in distributions:
        stats = ModelResourceStats(token_depth_distribution=dist)
        loss, comp = loss_fn(stats, return_components=True)
        print(f"  {desc}: dist={dist}, weighted={comp['weighted_tokens']:.1f}, L_token={comp['L_token']:.4f}")
    
    # 测试 3: 深度熵
    print("\n[测试 3] 深度熵正则")
    print("-" * 60)
    
    loss_fn = ResourceAwareLoss(
        lambda_flops=0.0,
        lambda_token=0.0,
        lambda_entropy=1.0,
        target_entropy_ratio=1.0
    )
    
    distributions = [
        ([100, 0, 0, 0, 0], "完全坍缩"),
        ([20, 20, 20, 20, 20], "完全均匀"),
        ([40, 30, 20, 10, 0], "渐变分布"),
    ]
    
    for dist, desc in distributions:
        stats = ModelResourceStats(token_depth_distribution=dist)
        loss, comp = loss_fn(stats, return_components=True)
        H_max = np.log(len(dist))
        print(f"  {desc}: H={comp['depth_entropy']:.3f} (max={H_max:.3f}), L_entropy={comp['L_entropy']:.4f}")
    
    # 测试 4: 深度权重验证
    print("\n[测试 4] 深度权重验证")
    print("-" * 60)
    
    alphas = [0.0, 0.05, 0.1, 0.2]
    dist = torch.tensor([20.0, 30.0, 50.0])  # 深层为主
    
    print(f"  分布: {dist.tolist()}")
    for alpha in alphas:
        weighted = get_weighted_token_count(dist, alpha=alpha)
        print(f"  α={alpha:.2f}: N_weighted={weighted:.1f} (无权重=100)")
    
    print("\n" + "=" * 60)
    print("验证完成！")
    print("=" * 60)


if __name__ == "__main__":
    # 运行计算验证
    validate_resource_loss()
