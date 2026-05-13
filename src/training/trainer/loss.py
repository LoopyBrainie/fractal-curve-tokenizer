"""Loss Computation Module

Provides loss computation with Mixup/Cutmix support.
Following the three-layer parameter principle, this module only handles
Layer 3 (hyperparameters) for loss computation.

Updates:
- Phase 3: Simplified to UnifiedLoss (CE + batch-wise entropy)
"""

from __future__ import annotations

from typing import Tuple, Optional, Dict
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class MixupCutmixLoss:
    r"""
    Loss computation with Mixup/Cutmix augmentation support.

    Mathematical forms:

    - Mixup:
        :math:`\tilde{x} = \lambda \cdot x_i + (1 - \lambda) \cdot x_j`
        :math:`\tilde{y} = \lambda \cdot y_i + (1 - \lambda) \cdot y_j`
        where :math:`\lambda \sim Beta(\alpha, \alpha)`

    - CutMix:
        :math:`\tilde{x} = Mix(x_i, x_j, region)`
        :math:`\tilde{y} = \lambda \cdot y_i + (1 - \lambda) \cdot y_j`
        where :math:`\lambda = 1 - (region\_area / total\_area)`

    See `mixup <https://arxiv.org/abs/1710.09412>`_ and
    `CutMix <https://arxiv.org/abs/1905.04899>`_ for details.
    """

    def __init__(
        self,
        num_classes: int,
        label_smoothing: float = 0.0,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 0.5,
    ):
        r"""
        Args:
            num_classes (int): Number of classes for one-hot encoding
            label_smoothing (float): Label smoothing factor. Default: ``0.0``
            mixup_alpha (float): Mixup Beta distribution alpha parameter. Default: ``0.8``
            cutmix_alpha (float): CutMix Beta distribution alpha parameter. Default: ``1.0``
            mixup_prob (float): Probability of applying mixup/cutmix. Default: ``0.5``
        """
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob

    def apply_mixup(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Mixup augmentation

        Args:
            images: [B, C, H, W] input images
            labels: [B] class labels

        Returns:
            mixed_images: [B, C, H, W] mixed images
            mixed_labels: [B, num_classes] mixed one-hot labels
        """
        batch_size = images.size(0)
        device = images.device

        # Sample lambda from Beta distribution
        # D1-AUDIT FIX: 添加 non_blocking=True 避免 forward pass 阻塞
        lam = torch.distributions.Beta(
            self.mixup_alpha, self.mixup_alpha
        ).sample((batch_size,)).to(device, non_blocking=True)

        # Random permutation
        index = torch.randperm(batch_size, device=device)

        # Mix images
        lam = lam.view(batch_size, 1, 1, 1)
        mixed_images = lam * images + (1 - lam) * images[index]

        # Mix labels - 使用正确的形状 [B, 1] 而不是 [B, 1, 1]
        lam_for_labels = lam.squeeze(-1).squeeze(-1)  # [B, 1, 1] -> [B, 1]
        labels_one_hot = F.one_hot(labels, self.num_classes).float()
        mixed_labels = lam_for_labels * labels_one_hot + (1 - lam_for_labels) * labels_one_hot[index]

        return mixed_images, mixed_labels

    def apply_cutmix(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply CutMix augmentation

        Args:
            images: [B, C, H, W] input images
            labels: [B] class labels

        Returns:
            mixed_images: [B, C, H, W] mixed images
            mixed_labels: [B, num_classes] mixed one-hot labels
        """
        batch_size = images.size(0)
        device = images.device
        _, _, H, W = images.shape

        # Sample lambda
        # D1-AUDIT FIX: 添加 non_blocking=True 避免 forward pass 阻塞
        lam = torch.distributions.Beta(
            self.cutmix_alpha, self.cutmix_alpha
        ).sample((batch_size,)).to(device, non_blocking=True)

        # Random permutation
        index = torch.randperm(batch_size, device=device)

        # Generate random box
        cut_rat = torch.sqrt(1.0 - lam)
        cut_w = (W * cut_rat).to(torch.int64)
        cut_h = (H * cut_rat).to(torch.int64)

        # Random center
        cx = torch.randint(0, W, (batch_size,), device=device)
        cy = torch.randint(0, H, (batch_size,), device=device)

        # Bounding box
        x1 = torch.clamp(cx - cut_w // 2, 0, W)
        x2 = torch.clamp(cx + cut_w // 2, 0, W)
        y1 = torch.clamp(cy - cut_h // 2, 0, H)
        y2 = torch.clamp(cy + cut_h // 2, 0, H)

        # Apply CutMix
        mixed_images = images.clone()
        for i in range(batch_size):
            mixed_images[i, :, y1[i]:y2[i], x1[i]:x2[i]] = images[index[i,], :, y1[i]:y2[i], x1[i]:x2[i]]

        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H)).float()

        # Mix labels
        labels_one_hot = F.one_hot(labels, self.num_classes).float()
        mixed_labels = lam.view(-1, 1) * labels_one_hot + (1 - lam.view(-1, 1)) * labels_one_hot[index]

        return mixed_images, mixed_labels

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        apply_aug: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Mixup or CutMix if enabled

        Args:
            images: [B, C, H, W] input images
            labels: [B] class labels
            apply_aug: Whether to apply augmentation

        Returns:
            processed_images: [B, C, H, W] (possibly mixed) images
            processed_labels: [B, num_classes] (possibly mixed) labels
        """
        if not apply_aug or (self.mixup_alpha == 0 and self.cutmix_alpha == 0):
            # No augmentation, return original
            labels_one_hot = F.one_hot(labels, self.num_classes).float()
            if self.label_smoothing > 0:
                labels_one_hot = labels_one_hot * (1 - self.label_smoothing) + \
                                 self.label_smoothing / self.num_classes
            return images, labels_one_hot

        # Decide whether to apply augmentation
        if random.random() > self.mixup_prob:
            return self.apply_mixup(images, labels)

        # Random choice between Mixup and CutMix
        if random.random() > 0.5 and self.cutmix_alpha > 0:
            return self.apply_cutmix(images, labels)
        else:
            return self.apply_mixup(images, labels)


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
    aux_losses: Optional[Dict[str, Tensor]] = None,
    aux_weight: float = 0.08,  # I107-OPT: 从 0.02 增到 0.08，增加预算损失梯度影响
    budget_weight_override: Optional[float] = None,
) -> Tuple[torch.Tensor, dict]:
    """Compute cross-entropy loss with optional auxiliary losses

    Args:
        logits: [B, C] or [B, num_classes] model outputs
        targets: [B, C] (one-hot) or [B] (class indices)
        reduction: Loss reduction method
        aux_losses: Optional dict of auxiliary losses from splitter
        aux_weight: Base weight for auxiliary losses (default: 0.08)
        budget_weight_override: Optional budget-specific weight from GradBalancer.
            When provided, used instead of dynamic_w for budget loss.
            Expected to be schedule_weight * w_adaptive.

    Returns:
        loss: Scalar loss tensor (main + weighted aux)
        components: Dict of loss components (cross_entropy + individual aux losses)
    """

    # Handle one-hot targets (from Mixup/Cutmix)
    if targets.dim() == 2:
        # One-hot targets - use direct computation for mixed labels
        log_probs = F.log_softmax(logits, dim=-1)
        # Compute: -sum(y * log_p) for mixed labels
        loss = -torch.sum(targets * log_probs, dim=-1)
        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
    else:
        # Standard targets
        loss = F.cross_entropy(logits, targets, reduction=reduction)

    # Track components - return GPU tensors, caller handles .item()
    components = {"cross_entropy": loss if loss.dim() == 0 else loss.mean()}

    # Integrate auxiliary losses with dynamic weighting
    if aux_losses is not None:
        ce_mag = loss.abs().detach().mean() if loss.dim() == 0 else loss.abs().mean()

        for name, aux_loss in aux_losses.items():
            if aux_loss is not None:
                aux_mag = aux_loss.abs().detach().mean()

                # I-OPT: 动态权重 - 保持 aux loss 在 CE 的 2-20% 范围
                if aux_mag > 1e-8:
                    ratio = (ce_mag / aux_mag).clamp(0.02, 1.0)
                    dynamic_w = aux_weight * ratio
                else:
                    dynamic_w = aux_weight

                # P0 FIX: GradBalancer 闭环 - 对 budget loss 应用 budget_weight_override
                if name == "budget" and budget_weight_override is not None:
                    final_weight = budget_weight_override
                else:
                    final_weight = dynamic_w

                # I-OPT: 不再 .detach()，让梯度流过 aux_loss（特别是 budget loss）
                loss = loss + final_weight * aux_loss
                components[f"aux_{name}"] = aux_loss
                components[f"aux_weight_{name}"] = final_weight

    return loss, components


def compute_label_smoothing_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    smoothing: float = 0.1,
) -> torch.Tensor:
    r"""
    Compute cross entropy with label smoothing.

    Formula:
        :math:`y'_c = (1 - \epsilon) \cdot y_c + \epsilon / C`
        :math:`CE_{smoothed} = -\sum_c y'_c \cdot \log(p_c)`

    Args:
        logits (Tensor): Model outputs of shape :math:`(B, C)`
        labels (Tensor): Class labels of shape :math:`(B,)`
        num_classes (int): Number of classes
        smoothing (float): Smoothing factor :math:`\epsilon`. Default: ``0.1``

    Returns:
        Tensor: Scalar loss

    Examples::

        >>> logits = torch.randn(32, 10)
        >>> labels = torch.randint(0, 10, (32,))
        >>> loss = compute_label_smoothing_loss(logits, labels, num_classes=10, smoothing=0.1)
        >>> loss.item()
        2.123
    """
    log_probs = F.log_softmax(logits, dim=-1)

    # Create smoothed labels
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / num_classes)
        true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)

    return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class AuxiliaryLossTracker:
    """Track auxiliary losses during training

    Useful for Fractal ViT which may have multiple loss components.
    """

    def __init__(self):
        self.losses: dict = {}
        self.counts: dict = {}

    def add(self, name: str, value: float) -> None:
        """Add a loss component"""
        if name not in self.losses:
            self.losses[name] = 0.0
            self.counts[name] = 0
        self.losses[name] += value
        self.counts[name] += 1

    def get_components(self) -> dict:
        """Get averaged loss components"""
        return {
            name: self.losses[name] / max(self.counts[name], 1)
            for name in self.losses
        }

    def reset(self) -> None:
        """Reset for new epoch"""
        self.losses.clear()
        self.counts.clear()

class UnifiedLoss:
    """Unified loss: Cross-Entropy + optional batch-wise entropy regularization.

    数学形式:
        L = -Σ y_i log(ŷ_i) - λ · H(p_bar)
        其中 p_bar = mean_batch(probs), H = -Σ p_bar log(p_bar)

    为什么用 Batch-wise 熵而非逐样本熵？
        逐样本熵: H_i = -Σ p_i log p_i, 最小化 → 每个样本的 probs 趋于均匀
        → 模糊单个样本的 Top-K 选择边界，降低 token 质量的区分度

        Batch-wise 熵: H_batch = -Σ p_bar log p_bar, p_bar = mean_i(probs_i)
        → 强制跨样本多样性: 每个候选区域在整个 batch 中有机会被选中
        → 不强制单样本内部均匀，Top-K 边界保持清晰
        → 有效防止"只选图像中心"的局部最优解

    Args:
        num_classes: 分类类别数
        entropy_weight: 熵正则化权重 (default: 0.01)
        label_smoothing: 标签平滑因子 (default: 0.0)
    """

    def __init__(
        self,
        num_classes: int,
        entropy_weight: float = 0.01,
        label_smoothing: float = 0.0,
    ):
        self.num_classes = num_classes
        self.entropy_weight = entropy_weight
        self.label_smoothing = label_smoothing

    def __call__(
        self,
        logits,
        targets,
        probs=None,
    ):
        """计算统一损失。

        Args:
            logits: [B, num_classes] 模型输出
            targets: [B] 标签 (class indices)
            probs: [B, N] splitter 输出选择概率 (mask_soft)，可选

        Returns:
            loss: 标量损失
            components: dict of loss components (用于日志)
        """
        ce = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
        components = {'ce_loss': ce.detach()}

        if probs is not None:
            # Batch-wise 熵最大化: H(mean_batch(probs))
            from vit_pytorch.core.constants import EPS
            p_bar = probs.mean(dim=0).clamp(min=EPS)  # [N]
            entropy = -(p_bar * torch.log(p_bar)).sum()  # 标量
            loss = ce - self.entropy_weight * entropy  # 负号: 最大化熵 → 促进多样性
            components['entropy_loss'] = entropy.detach()
            components['total_loss'] = loss.detach()
            return loss, components

        components['total_loss'] = ce.detach()
        return ce, components

__all__ = [
    "MixupCutmixLoss",
    "compute_loss",
    "compute_label_smoothing_loss",
    "AuxiliaryLossTracker",
    "UnifiedLoss",
]
