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
    r"""
    Diversity loss: Encourages child node features to be orthogonal (semantically independent).

    Mathematically (I112-1 fixed version):
        L_div = ||Ŝ - I||_F²

    Where:
        v̂_i = v_i / (||v_i||₂ + ε)  # L2 normalized
        Ŝ = v̂ · v̂ᵀ                   # Cosine similarity matrix
        L_div = ||Ŝ - I||²_F           # Squared Frobenius norm

    Design principles:
        1. Uses cosine similarity instead of raw dot product to decouple loss from feature norms
        2. Consistent with CorrelationGate's normalization strategy
        3. Loss range [0, 4], maximum when 4 children are parallel

    Effects:
        - When four child features are orthogonal, Ŝ = I, L_div = 0
        - When child features are similar, L_div increases (max 4)
        - Loss value is independent of feature L2 norm
    """

    def __init__(self, reduction: str = "mean", epsilon: float = EPS):  # I112-3: EPS = 1e-6
        r"""
        Args:
            reduction (str): Reduction method. Options: ``"mean"``, ``"sum"``, ``"none"``.
                Default: ``"mean"``
            epsilon (float): Small constant to prevent division by zero. Default: ``1e-6``
        """
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, child_features: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the diversity loss.

        Args:
            child_features (Tensor): Child node features of shape :math:`(B, N, 4, D)`

        Returns:
            Tensor: Diversity loss with range [0, 4]. Shape depends on ``reduction``:
                - ``"mean"``: scalar
                - ``"sum"``: scalar
                - ``"none"``: :math:`(B, N)`

        Examples::

            >>> child_features = torch.randn(2, 8, 4, 64)  # [B, N, num_children, D]
            >>> loss_fn = DiversityLoss(reduction="mean")
            >>> loss = loss_fn(child_features)
            >>> loss.item()
            1.234
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
    r"""
    Reconstruction consistency loss: Ensures child nodes can reconstruct the parent node.

    Formula: :math:`L_{rec} = ||F_p - AvgPool(V_c)||^2`

    Effects:
        - Ensures splitting does not lose information
        - Mean of child features should be close to parent features
    """

    def __init__(self, reduction: str = "mean"):
        r"""
        Args:
            reduction (str): Reduction method. Options: ``"mean"``, ``"sum"``, ``"none"``.
                Default: ``"mean"``
        """
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

    def forward(
        self,
        parent_features: torch.Tensor,
        child_features: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Computes the reconstruction consistency loss.

        Args:
            parent_features (Tensor): Parent node features of shape :math:`(B, N, D)`
            child_features (Tensor): Child node features of shape :math:`(B, N, 4, D)`

        Returns:
            Tensor: Reconstruction loss. Shape depends on ``reduction``:
                - ``"mean"``: scalar
                - ``"sum"``: scalar
                - ``"none"``: :math:`(B, N)`

        Examples::

            >>> parent_features = torch.randn(2, 8, 64)  # [B, N, D]
            >>> child_features = torch.randn(2, 8, 4, 64)  # [B, N, num_children, D]
            >>> loss_fn = ReconstructionLoss(reduction="mean")
            >>> loss = loss_fn(parent_features, child_features)
            >>> loss.item()
            0.567
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
    r"""
    Total semantic redundancy loss.

    Total loss:
        :math:`L_{total} = L_{cls} + λ_d · L_{div} + λ_r · L_{rec}`

    Design decisions:
        - λ_d = λ_r = 0.1: Matches the magnitude of main classification loss
        - Only computes auxiliary loss for split regions
    """

    def __init__(
        self,
        diversity_weight: float = 0.1,
        reconstruction_weight: float = 0.1,
        reduction: str = "mean",
    ):
        r"""
        Args:
            diversity_weight (float): Weight for diversity loss. Default: ``0.1``
            reconstruction_weight (float): Weight for reconstruction loss. Default: ``0.1``
            reduction (str): Reduction method. Options: ``"mean"``, ``"sum"``, ``"none"``.
                Default: ``"mean"``
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
        r"""
        Computes the semantic redundancy loss.

        Args:
            parent_features (Tensor): Parent node features of shape :math:`(B, N, D)`
            child_features (Tensor): Child node features of shape :math:`(B, N, 4, D)`
            split_decisions (Tensor, optional): Split decisions of shape :math:`(B, N)`.
                If provided, only computes loss for split regions. Default: ``None``

        Returns:
            Dict[str, Tensor]: Dictionary containing:
                - ``"loss"``: Total weighted loss (scalar)
                - ``"diversity_loss"``: Diversity loss component
                - ``"reconstruction_loss"``: Reconstruction loss component

        Examples::

            >>> parent_features = torch.randn(2, 8, 64)
            >>> child_features = torch.randn(2, 8, 4, 64)
            >>> split_decisions = torch.randint(0, 2, (2, 8)).bool()
            >>> loss_fn = SemanticRedundancyLoss(diversity_weight=0.1, reconstruction_weight=0.1)
            >>> losses = loss_fn(parent_features, child_features, split_decisions)
            >>> losses["loss"].item()
            0.234
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
