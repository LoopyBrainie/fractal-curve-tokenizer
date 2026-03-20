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


class FractalViTLoss(nn.Module):
    """整合 Mixup/Cutmix + H1SS 可微辅助损失

    数学形式:
        L_total = L_CE + λ_budget * L_budget + λ_tv * L_tv

        其中:
        - L_budget = MSE(Σ_i p_i, K)        # 连续配额损失
        - L_tv = E[|p_{i+1} - p_i|]        # Hilbert 全变分连续性损失

    Hilbert 全变分损失的物理意义:
        - 相邻 Hilbert 位置的概率突变被惩罚
        - 促进"同选或同不选"的局部一致性
        - A1 Locality 公理的端到端实现
    """

    def __init__(
        self,
        num_classes: int,
        expected_k: float = 32.0,
        budget_weight: float = 0.05,
        tv_weight: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            num_classes: 分类类别数
            expected_k: 期望的 Token 数量（预算）
            budget_weight: 配额损失权重
            tv_weight: 全变分损失权重
            label_smoothing: 标签平滑因子
        """
        super().__init__()
        self.num_classes = num_classes
        self.expected_k = expected_k
        self.budget_weight = budget_weight
        self.tv_weight = tv_weight
        self.label_smoothing = label_smoothing

    def _compute_budget_loss(self, probs: torch.Tensor, expected_k: float) -> torch.Tensor:
        """连续配额损失 (Continuous Budget Loss)

        数学: L_budget = MSE(Σ_i p_i, K)

        目标: Entmax 输出的概率总和等于期望的 Token 数量 K
        这使得 E[|S|] = K，实现 A5 Consistency 公理
        """
        # probs: [B, N] - 所有候选的分割概率
        current_k = probs.sum(dim=-1)  # [B] - 每个 batch 的实际 token 数量
        budget_loss = F.mse_loss(current_k, torch.full_like(current_k, expected_k))
        return budget_loss

    def _compute_tv_loss(
        self,
        probs: torch.Tensor,
        full_hilbert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Hilbert 全变分连续性损失 (1D Total Variation Loss)

        数学: L_tv = E[|p_{i+1} - p_i|]  在 Hilbert 1D 序列上

        目标: 惩罚在 Hilbert 1D 序列上突变的分裂概率
        相邻区域（同属一个空间局部流形）应同选或同不选

        实现要点:
        - 直接在全量 probs [B, N] 上按全局 Hilbert 顺序计算 TV
        - full_hilbert_indices [N] 来自 splitter.hilbert_indices
        - 这确保了选中区域与未选中区域边界之间也有梯度约束
        """
        if probs is None or full_hilbert_indices is None:
            return torch.tensor(0.0, device=probs.device if probs is not None else 'cpu')

        # 获取全局 Hilbert 排序索引
        sort_idx = torch.argsort(full_hilbert_indices)  # [N]

        # 对所有 Batch 按 Hilbert 顺序重排 probs
        # probs: [B, N] -> sorted_probs: [B, N]
        # 利用广播效应沿 B 维度同时排序
        sorted_probs = probs[:, sort_idx]  # [B, N]

        # 计算相邻概率的绝对差
        # sorted_probs[:, 1:] - sorted_probs[:, :-1] -> [B, N-1]
        tv_diffs = torch.abs(sorted_probs[:, 1:] - sorted_probs[:, :-1])  # [B, N-1]

        # 返回均值
        return tv_diffs.mean()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        splitter_outputs,
        expected_k: float,
        full_hilbert_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """前向传播

        Args:
            logits: [B, C] 模型输出的分类 logits
            targets: [B, C] (one-hot, Mixup/Cutmix) 或 [B] (class indices)
            splitter_outputs: SplitResult 对象，包含 probs
            expected_k: 期望的 Token 数量
            full_hilbert_indices: [N] 全局 Hilbert 排序索引（来自 splitter.hilbert_indices）

        Returns:
            total_loss: 组合损失
            components: 各损失分量的字典
        """
        # 1. 交叉熵损失
        if targets.dim() == 2:
            # One-hot targets (来自 Mixup/Cutmix)
            log_probs = F.log_softmax(logits, dim=-1)
            ce_loss = -torch.sum(targets * log_probs, dim=-1).mean()
        else:
            # Standard targets
            ce_loss = F.cross_entropy(
                logits, targets,
                label_smoothing=self.label_smoothing if self.label_smoothing > 0 else 0.0,
            )

        # 2. 连续配额损失 (Continuous Budget Loss)
        budget_loss = self._compute_budget_loss(splitter_outputs.probs, expected_k)

        # 3. Hilbert 全变分连续性损失（使用全量 probs 和全局 Hilbert 排序）
        if full_hilbert_indices is not None:
            tv_loss = self._compute_tv_loss(splitter_outputs.probs, full_hilbert_indices)
        else:
            tv_loss = torch.tensor(0.0, device=logits.device)

        # 4. 组合损失
        total_loss = ce_loss + self.budget_weight * budget_loss + self.tv_weight * tv_loss

        # 5. 构建分量字典
        components = {
            "cross_entropy": ce_loss.detach(),
            "budget_loss": budget_loss.detach(),
            "tv_loss": tv_loss.detach(),
            "total": total_loss.detach(),
        }

        return total_loss, components


__all__ = [
    "MixupCutmixLoss",
    "FractalViTLoss",
    "compute_loss",
    "compute_label_smoothing_loss",
    "AuxiliaryLossTracker",
]
