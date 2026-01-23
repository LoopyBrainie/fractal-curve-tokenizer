# -*- coding: utf-8 -*-
"""
I97-11: 动态计算 - 复杂度估计器

数学形式化
==========

复杂度估计器:
    C(I) = σ(W_2 · ReLU(W_1 · x̄))

其中:
- x̄ = mean(x) ∈ R^D: Token特征的平均池化
- W_1 ∈ R^{D × D/4}: 第一层权重
- W_2 ∈ R^{D/4 × 1}: 第二层权重
- σ: Sigmoid激活函数，输出范围 [0, 1]

复杂度-层数映射:
    L_eff = floor(L_min + (L_max - L_min) * C(I))

优势:
1. 非线性复杂度映射
2. 端到端可训练
3. 与Hilbert局部性兼容

复杂度分析
----------
时间: O(D²) - 两个矩阵乘
空间: O(D²) - 存储参数

Author: Claude Code
Date: 2026-01-22
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexityEstimator(nn.Module):
    """I97-11: 可学习复杂度估计器。

    根据输入图像的token特征估计其复杂度，用于动态计算决策。

    数学形式:
        C(I) = σ(W_2 · ReLU(W_1 · x̄))

    其中 x̄ = mean(x) 是token特征的平均池化。

    Args:
        dim: 输入特征维度
        hidden_dim: 隐藏层维度，默认为 dim // 4
        use_cls: 是否使用CLS token（False则使用mean pooling）
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        use_cls: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or (dim // 4)
        self.use_cls = use_cls

        # 复杂度估计MLP: 2层
        self.net = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """估计图像复杂度。

        Args:
            x: Token特征，形状为 [B, N, D]

        Returns:
            复杂度得分，形状为 [B, 1]，值域 [0, 1]
        """
        if self.use_cls:
            # 使用CLS token作为全局表示
            pooled = x[:, 0]  # [B, D]
        else:
            # 使用平均池化
            pooled = x.mean(dim=1)  # [B, D]

        # 通过MLP估计复杂度
        complexity = self.net(pooled)  # [B, 1]

        return complexity

    def get_complexity_stats(self, x: torch.Tensor) -> dict:
        """获取复杂度统计信息（用于调试和分析）。

        Args:
            x: Token特征 [B, N, D]

        Returns:
            包含统计信息的字典
        """
        complexity = self.forward(x)

        return {
            'mean': complexity.mean().item(),
            'std': complexity.std().item(),
            'min': complexity.min().item(),
            'max': complexity.max().item(),
            'median': complexity.median().item(),
        }


class DynamicDepthRouter(nn.Module):
    """I97-11: 动态深度路由器。

    根据复杂度估计结果决定使用多少层Transformer。

    数学形式:
        L_eff = Quantize(L_min + (L_max - L_min) * C(I))
               = floor(L_min + (L_max - L_min) * C(I))

    其中 Quantize 表示离散化（向下取整）以获得确定性的延迟。

    Args:
        dim: 特征维度（传递给ComplexityEstimator）
        depth: Transformer总层数
        min_layers: 最小层数（默认为 depth // 2）
        complexity_estimator: 已训练的复杂度估计器
        hidden_dim: 复杂度估计器隐藏层维度
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        min_layers: int | None = None,
        complexity_estimator: ComplexityEstimator | None = None,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.min_layers = min_layers or (depth // 2)

        # 复杂度估计器
        if complexity_estimator is not None:
            self.complexity_estimator = complexity_estimator
        else:
            self.complexity_estimator = ComplexityEstimator(
                dim=dim,
                hidden_dim=hidden_dim,
                use_cls=False,
            )

    def compute_effective_depth(
        self,
        complexity: torch.Tensor,
    ) -> torch.Tensor:
        """根据复杂度计算有效层数。

        数学形式:
            L_target = L_min + (L_max - L_min) * C(I)
            L_eff = floor(L_target)

        Args:
            complexity: 复杂度得分 [B, 1]

        Returns:
            有效层数 [B]
        """
        # 线性映射到 [min_layers, depth]
        target_layers = self.min_layers + \
            (self.depth - self.min_layers) * complexity  # [B, 1]

        # 离散化（向下取整以节省更多计算）
        effective_depth = target_layers.floor().long().clamp(
            min=self.min_layers,
            max=self.depth,
        )  # [B, 1]

        return effective_depth.squeeze(-1)  # [B]

    def forward(
        self,
        x: torch.Tensor,
        return_extra_info: bool = False,
    ) -> dict[str, torch.Tensor]:
        """前向传播，计算有效层数。

        Args:
            x: Token特征 [B, N, D]
            return_extra_info: 是否返回额外信息

        Returns:
            包含以下键的字典:
            - effective_depth: 有效层数 [B]
            - complexity: 复杂度得分 [B, 1]
        """
        # 计算复杂度
        complexity = self.complexity_estimator(x)  # [B, 1]

        # 计算有效层数
        effective_depth = self.compute_effective_depth(complexity)  # [B]

        result = {
            'effective_depth': effective_depth,
            'complexity': complexity,
        }

        return result

    def get_layer_selection_stats(
        self,
        effective_depth: torch.Tensor,
    ) -> dict:
        """获取层数选择统计信息。

        Args:
            effective_depth: 有效层数 [B]

        Returns:
            统计信息字典
        """
        depth_counts = effective_depth.bincount(
            minlength=self.depth + 1
        )  # [depth + 1]

        return {
            'mean': effective_depth.float().mean().item(),
            'std': effective_depth.float().std().item(),
            'distribution': depth_counts.cpu().tolist(),
        }


def compute_complexity_from_depth_distribution(
    depths: torch.Tensor,
    max_level: int,
) -> torch.Tensor:
    """从深度分布计算复杂度（无需额外参数）。

    数学形式:
        C(I) = Σ_d w_d * p_d

    其中 p_d 是深度为 d 的token比例，w_d 是深度权重。

    Args:
        depths: 深度值 [B, N]
        max_level: 最大深度

    Returns:
        复杂度得分 [B, 1]
    """
    B, N = depths.shape

    # 深度权重：深度越大，权重越高（细节越丰富）
    # 权重数量 = max_level + 1 (深度 0 到 max_level)
    depth_weights = torch.arange(
        1, max_level + 2, dtype=torch.float32, device=depths.device
    )  # [max_level + 1]
    depth_weights = depth_weights / depth_weights.sum()  # 归一化

    # 计算深度分布 - 使用 num_classes=max_level+2 确保与 depths 中的值兼容
    depth_distribution = F.one_hot(depths, num_classes=max_level + 2).float()
    depth_distribution = depth_distribution.mean(dim=1)  # [B, max_level + 2]

    # 只使用前 max_level + 1 列进行计算
    depth_distribution = depth_distribution[:, :max_level + 1]  # [B, max_level + 1]

    # 计算复杂度
    complexity = depth_distribution @ depth_weights  # [B]
    complexity = complexity.unsqueeze(-1)  # [B, 1]

    return complexity


# Alias for backwards compatibility
ComplexityRouter = DynamicDepthRouter


if __name__ == "__main__":
    # 简单测试
    print("=" * 50)
    print("ComplexityEstimator 测试")
    print("=" * 50)

    # 创建估计器
    estimator = ComplexityEstimator(dim=128, hidden_dim=32)

    # 创建测试输入
    x = torch.randn(4, 64, 128)  # B=4, N=64, D=128

    # 前向传播
    complexity = estimator(x)

    print(f"输入形状: {x.shape}")
    print(f"复杂度输出形状: {complexity.shape}")
    print(f"复杂度统计: {estimator.get_complexity_stats(x)}")

    # 测试DynamicDepthRouter
    print("\n" + "=" * 50)
    print("DynamicDepthRouter 测试")
    print("=" * 50)

    router = DynamicDepthRouter(dim=128, depth=12, min_layers=4)
    result = router(x)

    print(f"有效层数: {result['effective_depth']}")
    print(f"层数选择统计: {router.get_layer_selection_stats(result['effective_depth'])}")

    print("\n✓ 所有测试通过")
