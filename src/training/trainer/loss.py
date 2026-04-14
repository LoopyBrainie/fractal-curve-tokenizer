"""Loss Computation Module

Provides loss computation with Mixup/Cutmix support.
Following the three-layer parameter principle, this module only handles
Layer 3 (hyperparameters) for loss computation.

Updates:
- I150-3: Added FractalViTLoss for multi-task learning with differentiable probabilities
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
    aux_losses: Optional[Dict[str, Tensor]] = None,
    aux_weight: float = 0.08,  # I107-OPT: 从 0.02 增到 0.08，增加预算损失梯度影响
) -> Tuple[torch.Tensor, dict]:
    """Compute cross-entropy loss with optional auxiliary losses

    Args:
        logits: [B, C] or [B, num_classes] model outputs
        targets: [B, C] (one-hot) or [B] (class indices)
        reduction: Loss reduction method
        aux_losses: Optional dict of auxiliary losses from splitter
        aux_weight: Base weight for auxiliary losses (default: 0.02)

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

    # Track components
    components = {"cross_entropy": loss.item() if loss.dim() == 0 else loss.mean().item()}

    # Integrate auxiliary losses with dynamic weighting
    if aux_losses is not None:
        ce_mag = loss.abs().detach().mean() if loss.dim() == 0 else loss.abs().mean()

        for name, aux_loss in aux_losses.items():
            if aux_loss is not None:
                aux_mag = aux_loss.abs().detach().mean()

                # I-OPT: 动态权重 - 保持 aux loss 在 CE 的 2-20% 范围
                # dynamic_weight = aux_weight * (ce_mag / aux_mag).clamp(0.02, 1.0)
                if aux_mag > 1e-8:
                    ratio = (ce_mag / aux_mag).clamp(0.02, 1.0)
                    dynamic_w = aux_weight * ratio
                else:
                    dynamic_w = aux_weight

                # I-OPT: 不再 .detach()，让梯度流过 aux_loss（特别是 budget loss）
                loss = loss + dynamic_w * aux_loss
                components[f"aux_{name}"] = aux_loss.item()
                components[f"aux_weight_{name}"] = dynamic_w

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


class FractalViTLoss(nn.Module):
    """Fractal ViT Multi-Task Loss Function

    Mathematical form:
        L_total = L_CE + λ_depth * L_depth
                + λ_budget * L_budget
                + λ_manifold * L_manifold
                + λ_residual * L_residual
                + λ_entropy * L_entropy

    Key design decisions:
    1. Fixed weight scheduler (not nn.Parameter) to avoid weight collapse
    2. Soft probabilities (not hard statistics) for gradient flow
    3. detach() on parent features to prevent representation collapse
    4. No nan_to_num - let NaN propagate for outer-layer catching

    Mathematical forms:
    - Weight schedule: λ(t) = λ_max * min(1, t / t_warmup) * γ^epoch
    - Depth loss: KL(actual_probs || target_distribution)
    - Budget loss: E[N] = sum(P(split_i)), L = max(0, E[N] - K_max) + max(0, K_min - E[N])
    - Residual loss: ||parent.detach() - child||^2
    """

    def __init__(
        self,
        num_classes: int,
        label_smoothing: float = 0.0,
        # Fixed weights (adjusted via scheduler)
        depth_loss_weight: float = 0.1,
        budget_loss_weight: float = 0.05,
        manifold_loss_weight: float = 0.01,
        residual_loss_weight: float = 0.02,
        entropy_loss_weight: float = 0.01,
        # Budget constraints
        min_tokens: int = 16,
        max_tokens: int = 256,
        # Numerical protection
        eps: float = 1e-6,
        bias_clamp: float = 5.0,
        # Mixup/Cutmix
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 0.5,
        # Scheduler parameters
        warmup_epochs: int = 5,
        decay_rate: float = 0.9,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.eps = eps
        self.bias_clamp = bias_clamp

        # Fixed weights (not nn.Parameter)
        self.depth_loss_weight = depth_loss_weight
        self.budget_loss_weight = budget_loss_weight
        self.manifold_loss_weight = manifold_loss_weight
        self.residual_loss_weight = residual_loss_weight
        self.entropy_loss_weight = entropy_loss_weight

        # P2: Learnable entropy target (instead of fixed 1.0)
        self.entropy_target = nn.Parameter(torch.tensor(1.0))

        # Budget parameters
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

        # Scheduler parameters
        self.warmup_epochs = warmup_epochs
        self.decay_rate = decay_rate

        # Mixup/Cutmix
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob

        # MixupCutmixLoss instance
        self.mixup_cutmix = MixupCutmixLoss(
            num_classes=num_classes,
            label_smoothing=label_smoothing,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            mixup_prob=mixup_prob,
        )

    def get_scheduled_weights(self, epoch: int) -> Dict[str, float]:
        """Dynamic weight scheduler with auxiliary loss decay

        Mathematical form:
            λ(t) = λ_max * min(1, t / t_warmup) * γ^epoch

        Key insight (I165-1): After epoch 5, aux_budget has converged.
        entropy/tree losses become "noise" that drives Splitter暴走.
        → Exponential decay迫使 Splitter 中后期听从 CE 梯度信号
        """
        warmup_factor = min(1.0, epoch / max(1, self.warmup_epochs))
        decay_factor = self.decay_rate ** max(0, epoch - self.warmup_epochs)
        factor = warmup_factor * decay_factor

        # 辅助损失指数衰减：Epoch > 5 时，每 5 epochs 减半
        # Epoch 0-5:  full weight（迫使 Splitter 快速学会几何划分）
        # Epoch 5-10: decay to 50%
        # Epoch 10-15: decay to 25%
        aux_decay = 0.5 ** max(0, (epoch - 5) / 5)

        return {
            'depth': self.depth_loss_weight * factor,
            'budget': self.budget_loss_weight * factor,  # 保持稳定（已收敛）
            'manifold': self.manifold_loss_weight * factor,
            'residual': self.residual_loss_weight * factor,
            'entropy': self.entropy_loss_weight * aux_decay * factor,  # 衰减
        }

    def compute_ce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss (with label smoothing)"""
        log_probs = F.log_softmax(logits, dim=-1)

        if targets.dim() == 2:
            # Mixed labels (from Mixup/Cutmix)
            loss = -torch.sum(targets * log_probs, dim=-1).mean()
        else:
            # Standard labels
            if self.label_smoothing > 0:
                n_classes = logits.size(-1)
                with torch.no_grad():
                    smooth_labels = torch.zeros_like(log_probs)
                    smooth_labels.fill_(self.label_smoothing / n_classes)
                    smooth_labels.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)
                loss = -(smooth_labels * log_probs).sum(-1).mean()
            else:
                loss = F.cross_entropy(logits, targets)

        return loss

    def compute_depth_loss(
        self,
        depth_probs: torch.Tensor,
        target_distribution: Optional[torch.Tensor] = None,
        epoch: int = 0,
    ) -> torch.Tensor:
        """Depth distribution regularization loss (differentiable version)

        Input: Entmax scaled depth probability distribution from HilbertOptimalSplitter
        Gradient can backpropagate to Splitter

        Mathematical form:
            L_depth = KL(actual || target)
            where actual = depth_probs is a continuous distribution

        Curriculum Learning:
            - t < t_warmup: Uniform distribution (encourage exploration)
            - t >= t_warmup: Geometric distribution (prefer shallow depths)
        """
        if depth_probs.dim() == 1:
            depth_probs = depth_probs.unsqueeze(0)

        actual = depth_probs
        max_depth = actual.size(-1)

        if target_distribution is None:
            # Curriculum learning target distribution
            if epoch < self.warmup_epochs:
                # Warmup: Uniform distribution - encourage exploration of all depths
                target = torch.ones_like(actual) / max_depth
            else:
                # Decay: Geometric distribution - prefer shallow depths
                # 修复：从高值开始衰减，避免 p=0 导致 one-hot 突变
                # alpha 从 0.9 衰减到 0.3，保留深层表示能力
                alpha = 0.9 - 0.6 * min((epoch - self.warmup_epochs) / 50, 1.0)
                alpha = max(alpha, 0.3)  # 保留最小概率给深层
                depths = torch.arange(max_depth, device=actual.device, dtype=actual.dtype)
                target = (1 - alpha) * (alpha ** depths)
                target = target / target.sum()
                target = target.unsqueeze(0).expand_as(actual)
        else:
            target = target_distribution

        loss = F.kl_div(
            (actual + self.eps).log(),
            target,
            reduction='batchmean',
            log_target=(target + self.eps).log()
        )

        return loss.clamp(max=10.0)

    def compute_raw_budget_error(
        self,
        split_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Token budget consistency loss (differentiable version)

        Input: Entmax scaled split probabilities
        Expected token count = sum(probs) is differentiable!

        Mathematical form (softplus version):
            E[N] = sum_i P(split_i)
            L_budget = softplus(E[N] - K_max) + softplus(K_min - E[N])

        Key improvements:
        - Softplus replaces ReLU for smooth gradient
        - Added clamp(max=20.0) for NaN stability
        - D162: 重命名为 compute_raw_budget_error 以区分损失值与损失权重
        """
        expected_tokens = split_probs.sum(dim=-1).mean()

        over = expected_tokens - self.max_tokens
        under = self.min_tokens - expected_tokens

        # 严重超标时平方惩罚（梯度线性增长），轻微超标时 Softplus 平滑
        loss = torch.where(
            over > 0,
            over ** 2 * 0.5,         # 平方惩罚（强拉回）
            F.softplus(over)         # 轻微超标时平滑
        ) + F.softplus(under)

        return loss.clamp(max=20.0)

    def compute_manifold_loss(
        self,
        attention_bias: Optional[torch.Tensor] = None,
        poincare_distances: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Manifold geometric consistency loss

        Mathematical form:
            L_manifold = L_bias + L_poincare + L_locality

        Components:
            - L_bias: Soft clamping loss for attention bias
              L_bias = ||tanh(b / C) * C - b||_1

            - L_poincare: Poincaré distance variance
              L_poincare = Var(d^2) - encourage moderate dispersion

            - L_locality: Hilbert locality preservation (placeholder)
              L_locality = ||f[i] - f[i+1]||^2 for adjacent Hilbert positions

        Note:
            - When tensors are None, returns 0 (backward compatible)
            - Uses clamp(max=10.0) for NaN stability
        """
        # Determine device
        if attention_bias is not None:
            device = attention_bias.device
        elif poincare_distances is not None:
            device = poincare_distances.device
        else:
            device = 'cpu' if not torch.cuda.is_available() else 'cuda'

        loss = torch.tensor(0.0, device=device)

        # 1. Bias regularization (L2 + hard threshold)
        # 修复 tanh 梯度掩蔽问题：|bias|>>C 时 tanh 饱和导致梯度消失
        if attention_bias is not None:
            # L2 weight decay style: 恒定梯度，永不消失
            loss = loss + 0.01 * (attention_bias ** 2).mean()
            # Hard threshold: 仅惩罚超过阈值的偏差
            loss = loss + F.relu(attention_bias.abs() - self.bias_clamp).mean()

        # 2. Poincaré distance variance (encourage moderate dispersion)
        if poincare_distances is not None:
            # Compute variance of squared distances
            dist_mean = poincare_distances.mean()
            dist_var = ((poincare_distances - dist_mean) ** 2).mean()

            # Encourage variance in [0.1, 1.0] range
            # Too small: all points at same distance (collapsed)
            # Too large: points too spread out (unstable)
            loss = loss + F.relu(0.1 - dist_var) + F.relu(dist_var - 1.0)

            # P2 ENHANCEMENT: Add mean distance regularization
            # Encourage moderate mean distance (not too close, not too far)
            target_mean = 0.5  # Target mean distance
            loss = loss + F.mse_loss(dist_mean, torch.tensor(target_mean, device=device))

        # 3. (Future) Hilbert locality preservation
        # This would require additional inputs (Hilbert indices, token features)
        # placeholder for future extension

        # Clamp for numerical stability (align with other losses)
        return loss.clamp(max=10.0)

    def compute_residual_loss(
        self,
        parent_features: Optional[torch.Tensor] = None,
        child_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fractal residual consistency loss (corrected)

        Key fix: use detach() to prevent representation collapse
        修复: MSE 改为负余弦相似度，解决高维空间维度灾难问题

        Mathematical form:
            L_residual = -cosine_similarity(parent.detach(), child)
            = - (p / ||p||) · (c / ||c||)
        """
        if parent_features is None or child_features is None:
            device = parent_features.device if parent_features is not None else 'cpu'
            device = child_features.device if child_features is not None else device
            return torch.tensor(0.0, device=device)

        # 负余弦相似度（归一化到 [-1, 1]，解决 MSE 维度灾难）
        p = F.normalize(parent_features.detach(), dim=-1)
        c = F.normalize(child_features, dim=-1)
        loss = - (p * c).sum(dim=-1).mean()   # Negative Cosine Similarity

        return loss.clamp(max=2.0)

    def compute_entropy_loss(
        self,
        splitter_probs: torch.Tensor,
        target_entropy: Optional[float] = None,
    ) -> torch.Tensor:
        """Splitter entropy regularization loss

        Mathematical form:
            L_entropy = (H(P) - H_target)^2
            H(P) = -sum(p * log(p))

        P2: 使用可学习的目标 entropy_target（默认为 1.0）
        """
        entropy = -(splitter_probs * (splitter_probs + self.eps).log()).sum(-1).mean()

        # P2: 使用可学习目标，默认为 self.entropy_target
        if target_entropy is None:
            target = self.entropy_target
        else:
            target = torch.tensor(target_entropy, device=splitter_probs.device)

        loss = F.mse_loss(entropy, target)

        return loss

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        # Auxiliary information (differentiable versions)
        depth_probs: Optional[torch.Tensor] = None,
        split_probs: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        poincare_distances: Optional[torch.Tensor] = None,
        parent_features: Optional[torch.Tensor] = None,
        child_features: Optional[torch.Tensor] = None,
        splitter_probs: Optional[torch.Tensor] = None,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Complete forward computation (corrected)

        Key fixes:
        - Use Soft Probabilities (depth_probs, split_probs)
        - Don't use nan_to_num, let NaN propagate
        - P2: Add closed-loop weight adaptation
        """
        scheduled_weights = self.get_scheduled_weights(epoch)

        # Main loss
        ce_loss = self.compute_ce_loss(logits, targets)
        total_loss = ce_loss
        loss_dict = {'ce_loss': ce_loss.item()}

        # P2: Closed-loop weight adaptation
        # 如果 CE 突然变大，减少辅助权重以防止辅助损失主导
        if hasattr(self, '_last_ce_loss') and self._last_ce_loss > 0:
            ce_ratio = ce_loss.item() / (self._last_ce_loss + 1e-8)
            # 如果 CE 变大超过 1.2x，减少辅助权重
            if ce_ratio > 1.2:
                factor = 1.0 / (1.0 + 0.3 * (ce_ratio - 1.0))
                scheduled_weights = {k: v * factor for k, v in scheduled_weights.items()}
        self._last_ce_loss = ce_loss.item()

        # Depth distribution loss (using probability distribution with curriculum learning)
        if depth_probs is not None:
            depth_loss = self.compute_depth_loss(depth_probs, epoch=epoch)
            total_loss = total_loss + scheduled_weights['depth'] * depth_loss
            loss_dict['depth_loss'] = depth_loss.item()

        # Budget loss (using expected value of split probabilities)
        if split_probs is not None:
            budget_loss = self.compute_raw_budget_error(split_probs)  # D162: 重命名
            total_loss = total_loss + scheduled_weights['budget'] * budget_loss
            loss_dict['raw_budget_error'] = budget_loss.item()  # D162: 重命名 key

        # Manifold loss
        manifold_loss = self.compute_manifold_loss(attention_bias, poincare_distances)
        total_loss = total_loss + scheduled_weights['manifold'] * manifold_loss
        loss_dict['manifold_loss'] = manifold_loss.item()

        # Residual loss (using detach)
        residual_loss = self.compute_residual_loss(parent_features, child_features)
        total_loss = total_loss + scheduled_weights['residual'] * residual_loss
        loss_dict['residual_loss'] = residual_loss.item()

        # Entropy loss
        if splitter_probs is not None:
            entropy_loss = self.compute_entropy_loss(splitter_probs)
            total_loss = total_loss + scheduled_weights['entropy'] * entropy_loss
            loss_dict['entropy_loss'] = entropy_loss.item()

        loss_dict['total_loss'] = total_loss.item()

        # Don't use nan_to_num! Let NaN propagate for outer-layer catching
        return total_loss, loss_dict


__all__ = [
    "MixupCutmixLoss",
    "compute_loss",
    "compute_label_smoothing_loss",
    "AuxiliaryLossTracker",
    "FractalViTLoss",
]
