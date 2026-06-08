"""
R12 Auxiliary Loss (v1.3 §9.7) — Axiom A4 enforcement.

数学形式化
============

R12 由两项组成，用于在训练阶段对 Axiom A4（树形一致性 / 深度均衡）施加软约束。

L_tree (parent–child hinge²):
    对每一对 (parent, child) 在四叉树中的关系：
        z = max(0, parent_score - child_score - margin)
    L_tree = mean(z²)               # squared hinge
    含义: 当父节点分数显著高于子节点时（子节点本应被选中），产生二次惩罚。

L_skew (KL to balanced prior):
    depth_distribution ∈ Δ^{max_depth+1}
    Q_balanced(d) = 4^{-d} / Σ_l 4^{-l}    # 几何先验
    L_skew = KL(depth_distribution || Q_balanced) + eps
    含义: 鼓励实际深度分布与四叉树的几何先验对齐，避免单深度集中。

Total:
    L_R12 = λ_tree · L_tree + λ_skew · L_skew

References:
    v1.3 spec §9.7 — R12 Auxiliary Loss for Axiom A4 enforcement
    src/vit_pytorch/core/constants.py — R12_LAMBDA_TREE, R12_LAMBDA_SKEW
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .constants import EPS, R12_LAMBDA_SKEW, R12_LAMBDA_TREE


@dataclass(frozen=True)
class R12AuxLossConfig:
    """R12 Auxiliary Loss 配置.

    Attributes:
        lambda_tree: L_tree 系数（默认取自 constants.R12_LAMBDA_TREE = 0.10）
        lambda_skew: L_skew 系数（默认取自 constants.R12_LAMBDA_SKEW = 0.10）
        margin: 父-子 hinge 边界 (parent_score - child_score - margin)
    """

    lambda_tree: float = R12_LAMBDA_TREE
    lambda_skew: float = R12_LAMBDA_SKEW
    margin: float = 0.0


class R12AuxLoss(nn.Module):
    """R12 Auxiliary Loss: L_tree + L_skew.

    用法:
        loss_module = R12AuxLoss(R12AuxLossConfig())
        total = loss_module(parent_logits, child_logits, depth_distribution)
    """

    def __init__(self, config: R12AuxLossConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = R12AuxLossConfig()
        self.config = config

    def compute_l_tree(
        self,
        parent_logits: torch.Tensor,
        child_logits: torch.Tensor,
    ) -> torch.Tensor:
        """计算 L_tree: 父-子 squared hinge.

        对每对 (parent, child) 节点:
            z = max(0, parent_score - child_score - margin)
        L_tree = mean(z²)

        Args:
            parent_logits: [..., 1] 或 [...] 父节点分数（与 child 同形状即可逐对相减）
            child_logits:  [..., 1] 或 [...] 子节点分数

        Returns:
            标量损失（mean over all elements）
        """
        margin = self.config.margin
        z = torch.relu(parent_logits - child_logits - margin)
        return z.pow(2).mean()

    def compute_l_skew(self, depth_distribution: torch.Tensor) -> torch.Tensor:
        """计算 L_skew: depth_distribution 到几何先验 Q_balanced 的 KL 散度.

        数学形式化:
            Q_balanced(d) = 4^{-d} / Σ_l 4^{-l}
            L_skew = KL(P || Q) + eps = Σ_d P(d) · log(P(d) / Q(d)) + eps

        Args:
            depth_distribution: 形状 [max_depth + 1] 的概率分布（和为 1）

        Returns:
            标量损失 (>= eps)
        """
        if depth_distribution.dim() != 1:
            depth_distribution = depth_distribution.flatten()

        d = depth_distribution.float()
        max_depth = d.shape[0] - 1
        depths = torch.arange(max_depth + 1, device=d.device, dtype=d.dtype)
        # Q_balanced(d) ∝ 4^{-d}
        unnormalized = torch.pow(torch.tensor(4.0, device=d.device, dtype=d.dtype), -depths)
        q = unnormalized / unnormalized.sum()

        # KL(P || Q) = Σ p·log(p/q); 加 eps 防止 log(0)
        p_safe = torch.clamp(d, min=EPS)
        q_safe = torch.clamp(q, min=EPS)
        kl = (p_safe * (torch.log(p_safe) - torch.log(q_safe))).sum()
        return kl + EPS

    def forward(
        self,
        parent_logits: torch.Tensor,
        child_logits: torch.Tensor,
        depth_distribution: torch.Tensor,
    ) -> torch.Tensor:
        """计算 R12 总损失.

        Args:
            parent_logits: 父节点 logits 张量
            child_logits:  子节点 logits 张量
            depth_distribution: [max_depth + 1] 概率分布

        Returns:
            标量张量 L_R12 = λ_tree · L_tree + λ_skew · L_skew
        """
        l_tree = self.compute_l_tree(parent_logits, child_logits)
        l_skew = self.compute_l_skew(depth_distribution)
        return self.config.lambda_tree * l_tree + self.config.lambda_skew * l_skew
