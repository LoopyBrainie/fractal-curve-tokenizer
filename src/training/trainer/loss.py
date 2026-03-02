"""Loss Computation Module

Provides loss computation with Mixup/Cutmix support.
Following the three-layer parameter principle, this module only handles
Layer 3 (hyperparameters) for loss computation.
"""

from __future__ import annotations

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class MixupCutmixLoss:
    """Loss computation with Mixup/Cutmix augmentation support

    Mathematical forms:
    - Mixup:
        x̃ = λ · x_i + (1 - λ) · x_j
        ỹ = λ · y_i + (1 - λ) · y_j
        where λ ~ Beta(α, α)

    - CutMix:
        x̃ = Mix(x_i, x_j, region)
        ỹ = λ · y_i + (1 - λ) · y_j
        where λ = 1 - (region_area / total_area)

    Args:
        num_classes: Number of classes
        label_smoothing: Label smoothing factor
        mixup_alpha: Mixup alpha parameter
        cutmix_alpha: CutMix alpha parameter
        mixup_prob: Probability of applying mixup/cutmix
    """

    def __init__(
        self,
        num_classes: int,
        label_smoothing: float = 0.0,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 0.5,
    ):
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
        lam = torch.distributions.Beta(
            self.mixup_alpha, self.mixup_alpha
        ).sample((batch_size,)).to(device)

        # Random permutation
        index = torch.randperm(batch_size, device=device)

        # Mix images
        lam = lam.view(batch_size, 1, 1, 1)
        mixed_images = lam * images + (1 - lam) * images[index]

        # Mix labels
        labels_one_hot = F.one_hot(labels, self.num_classes).float()
        mixed_labels = lam.squeeze(-1) * labels_one_hot + (1 - lam.squeeze(-1)) * labels_one_hot[index]

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
        lam = torch.distributions.Beta(
            self.cutmix_alpha, self.cutmix_alpha
        ).sample((batch_size,)).to(device)

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
) -> Tuple[torch.Tensor, dict]:
    """Compute cross-entropy loss with optional components tracking

    Args:
        logits: [B, C] or [B, num_classes] model outputs
        targets: [B, C] (one-hot) or [B] (class indices)
        reduction: Loss reduction method

    Returns:
        loss: Scalar loss tensor
        components: Dict of loss components
    """
    # DEBUG: 添加临时调试信息
    print(f"[DEBUG] logits.shape: {logits.shape}, targets.shape: {targets.shape}, targets.dim(): {targets.dim()}")

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

    # Track components
    components = {"cross_entropy": loss.item() if loss.dim() == 0 else loss.mean().item()}

    return loss, components


def compute_label_smoothing_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """Compute cross entropy with label smoothing

    Formula:
        y'_c = (1 - ε) * y_c + ε / C
        CE_smoothed = -Σ_c y'_c * log(p_c)

    Args:
        logits: [B, C] model outputs
        labels: [B] class labels
        num_classes: Number of classes
        smoothing: Smoothing factor ε

    Returns:
        loss: Scalar loss
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


__all__ = [
    "MixupCutmixLoss",
    "compute_loss",
    "compute_label_smoothing_loss",
    "AuxiliaryLossTracker",
]
