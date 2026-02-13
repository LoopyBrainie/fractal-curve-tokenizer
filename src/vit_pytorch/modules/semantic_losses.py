"""
语义冗余损失函数

核心损失函数：
1. Diversity Loss: 多样性损失（正交性约束）
2. Reconstruction Loss: 重构一致性损失
3. SemanticRedundancyLoss: 总损失

数学形式化（I112-1 修复版）:

多样性损失（归一化版）:
    v̂_i = v_i / (||v_i||₂ + ε)    # L2 归一化
    Ŝ = v̂ · v̂ᵀ                   # 余弦相似度矩阵
    L_div = ||Ŝ - I||²_F            # Frobenius 范数平方

    设计原理:
    - 使用余弦相似度确保损失与特征范数解耦
    - 与 CorrelationGate 保持一致性
    - 损失范围 [0, 4]

重构一致性损失:
    L_rec = ||F_p - AvgPool(V_c)||^2

总损失:
    L_total = L_cls + λ_d · L_div + λ_r · L_rec
"""

from typing import Dict, Optional

import torch

from vit_pytorch.core.constants import EPS  # I112-3: 统一数值稳定性常量
import torch.nn as nn
import torch.nn.functional as F


class DiversityLoss(nn.Module):
    """多样性损失：鼓励子节点特征正交（语义独立）

    数学形式化（I112-1 修复版）:
        L_div = ||Ŝ - I||_F²

    其中:
        v̂_i = v_i / (||v_i||₂ + ε)  # L2 归一化
        Ŝ = v̂ · v̂ᵀ                   # 余弦相似度矩阵
        L_div = ||Ŝ - I||²_F           # Frobenius 范数平方

    设计原理:
        1. 使用余弦相似度而非原始点积，确保损失与特征范数解耦
        2. 与 CorrelationGate 的归一化策略保持一致
        3. 损失范围 [0, 4]，4 个子节点完全平行时达到最大值

    效果:
        - 当四个子节点特征正交时，Ŝ = I，L_div = 0
        - 当子节点特征相似时，L_div 增加（最大 4）
        - 损失值独立于特征 L2 范数
    """

    def __init__(self, reduction: str = "mean", epsilon: float = EPS):  # I112-3: EPS = 1e-6
        """初始化多样性损失

        Args:
            reduction: 归约方式 ("mean" | "sum" | "none")
            epsilon: 防止除零的小常数
        """
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, child_features: torch.Tensor) -> torch.Tensor:
        """计算多样性损失

        Args:
            child_features: [B, N, 4, D] 子节点特征

        Returns:
            loss: 多样性损失 (范围 [0, 4])
        """
        # 重塑为 [B*N, 4, D]
        B, N, num_children, D = child_features.shape
        child_flat = child_features.view(-1, num_children, D)

        # I112-1: L2 归一化 - 确保损失与特征范数解耦
        # I112-3: 使用 EPS 统一数值稳定性 (F.normalize 的 eps 参数)
        normalized = F.normalize(child_flat, p=2, dim=-1, eps=EPS)

        # 计算余弦相似度矩阵: [B*N, 4, 4]
        similarity = torch.bmm(normalized, normalized.transpose(1, 2))

        # 单位矩阵
        identity = torch.eye(num_children, device=child_features.device)

        # Frobenius 范数的平方
        diff = similarity - identity
        loss = (diff ** 2).sum(dim=[1, 2])  # [B*N]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss.view(B, N)


class ReconstructionLoss(nn.Module):
    """重构一致性损失：确保子节点能重构父节点

    公式: L_rec = ||F_p - AvgPool(V_c)||^2

    效果:
    - 确保分裂不丢失信息
    - 子节点特征的均值应接近父节点特征
    """

    def __init__(self, reduction: str = "mean"):
        """初始化重构损失

        Args:
            reduction: 归约方式 ("mean" | "sum" | "none")
        """
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

    def forward(
        self,
        parent_features: torch.Tensor,
        child_features: torch.Tensor,
    ) -> torch.Tensor:
        """计算重构一致性损失

        Args:
            parent_features: [B, N, D] 父节点特征
            child_features: [B, N, 4, D] 子节点特征

        Returns:
            loss: 重构损失
        """
        # 重塑为 [B*N, 4, D] 和 [B*N, D]
        B, N, D = parent_features.shape
        parent_flat = parent_features.view(-1, D)
        child_flat = child_features.view(-1, 4, D)

        # 平均池化子节点
        pooled = child_flat.mean(dim=1)  # [B*N, D]

        # L2 损失
        loss = ((pooled - parent_flat) ** 2).sum(dim=-1)  # [B*N]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss.view(B, N)


class SemanticRedundancyLoss(nn.Module):
    """语义冗余总损失

    总损失:
        L_total = L_cls + λ_d · L_div + λ_r · L_rec

    设计决策:
    - λ_d = λ_r = 0.1: 与主分类损失量级匹配
    - 仅对分裂的区域计算辅助损失
    """

    def __init__(
        self,
        diversity_weight: float = 0.1,
        reconstruction_weight: float = 0.1,
        reduction: str = "mean",
    ):
        """初始化语义冗余损失

        Args:
            diversity_weight: 多样性损失权重
            reconstruction_weight: 重构损失权重
            reduction: 归约方式
        """
        super().__init__()
        self.diversity_weight = diversity_weight
        self.reconstruction_weight = reconstruction_weight
        self.reduction = reduction

        self.diversity_loss = DiversityLoss(reduction=reduction)
        self.reconstruction_loss = ReconstructionLoss(reduction=reduction)

    def forward(
        self,
        parent_features: torch.Tensor,
        child_features: torch.Tensor,
        split_decisions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """计算语义冗余损失

        Args:
            parent_features: [B, N, D] 父节点特征
            child_features: [B, N, 4, D] 子节点特征
            split_decisions: [B, N] 分裂决策 (可选，不提供则计算所有)

        Returns:
            losses: {
                "loss": 总损失,
                "diversity_loss": 多样性损失,
                "reconstruction_loss": 重构损失
            }
        """
        # 计算多样性损失
        diversity = self.diversity_loss(child_features)  # [B, N] 或标量

        # 计算重构损失
        reconstruction = self.reconstruction_loss(parent_features, child_features)

        # 如果提供了 split_decisions，只对分裂区域计算损失
        if split_decisions is not None and self.reduction == "none":
            # 展平为 [B*N]
            diversity_flat = diversity.view(-1)
            reconstruction_flat = reconstruction.view(-1)
            split_flat = split_decisions.view(-1)

            # 只对分裂的区域计算损失
            # I112-3: 使用 EPS 统一数值稳定性
            num_splits = split_flat.sum()
            diversity = (diversity_flat * split_flat).sum() / (num_splits + EPS)
            reconstruction = (reconstruction_flat * split_flat).sum() / (num_splits + EPS)

        # 总损失
        total = (
            self.diversity_weight * diversity +
            self.reconstruction_weight * reconstruction
        )

        return {
            "loss": total,
            "diversity_loss": diversity,
            "reconstruction_loss": reconstruction,
        }

    def extra_repr(self) -> str:
        return (
            f"diversity_weight={self.diversity_weight}, "
            f"reconstruction_weight={self.reconstruction_weight}"
        )
