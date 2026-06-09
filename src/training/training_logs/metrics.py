r"""Metrics Computation Module

Provides metric computation utilities for model evaluation.
"""

from __future__ import annotations

from typing import List, Optional, Dict
import torch


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    top_k: int = 1,
) -> float:
    r"""
    Compute top-k accuracy.

    Args:
        predictions (Tensor): Prediction logits of shape :math:`(N, C)` where N is batch size
            and C is number of classes
        targets (Tensor): Ground truth labels of shape :math:`(N,)`
        top_k (int): Compute top-k accuracy. Default: ``1``

    Returns:
        float: Accuracy percentage in range [0, 1]

    Examples::

        >>> predictions = torch.randn(32, 10)
        >>> targets = torch.randint(0, 10, (32,))
        >>> accuracy = compute_accuracy(predictions, targets, top_k=1)
        >>> accuracy
        0.3125
    """
    batch_size = targets.size(0)
    num_classes = predictions.size(-1)

    if top_k == 1:
        pred = predictions.argmax(dim=-1)
        correct = (pred == targets).sum().item()
    else:
        _, top_k_pred = predictions.topk(min(top_k, num_classes), dim=-1)
        correct = (top_k_pred == targets.unsqueeze(-1)).any(dim=-1).sum().item()

    return correct / batch_size


def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    r"""
    Compute confusion matrix.

    Args:
        predictions (Tensor): Predicted class indices of shape :math:`(N,)`
        targets (Tensor): Ground truth class indices of shape :math:`(N,)`
        num_classes (int): Number of classes

    Returns:
        Tensor: Confusion matrix of shape :math:`(num\_classes, num\_classes)` where
            element [i, j] is the count of samples with true label i predicted as j

    Examples::

        >>> predictions = torch.tensor([0, 1, 2, 0, 1])
        >>> targets = torch.tensor([0, 1, 1, 0, 2])
        >>> compute_confusion_matrix(predictions, targets, num_classes=3)
        tensor([[2, 0, 0],
                [0, 1, 1],
                [0, 1, 0]])
    """
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for t, p in zip(targets, predictions):
        confusion[t.item(), p.item()] += 1

    return confusion


def compute_per_class_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> Dict[int, float]:
    r"""
    Compute per-class accuracy.

    Args:
        predictions (Tensor): Predicted class indices of shape :math:`(N,)`
        targets (Tensor): Ground truth class indices of shape :math:`(N,)`
        num_classes (int): Number of classes

    Returns:
        Dict[int, float]: Dictionary mapping class index to accuracy in range [0, 1]

    Examples::

        >>> predictions = torch.tensor([0, 1, 2, 0, 1])
        >>> targets = torch.tensor([0, 1, 1, 0, 2])
        >>> compute_per_class_accuracy(predictions, targets, num_classes=3)
        {0: 1.0, 1: 0.5, 2: 0.0}
    """
    per_class_correct = torch.zeros(num_classes)
    per_class_total = torch.zeros(num_classes)

    for t, p in zip(targets, predictions):
        t_item = t.item()
        p_item = p.item()
        per_class_total[t_item] += 1
        if t_item == p_item:
            per_class_correct[t_item] += 1

    per_class_acc = {}
    for c in range(num_classes):
        if per_class_total[c] > 0:
            per_class_acc[c] = (per_class_correct[c] / per_class_total[c]).item()

    return per_class_acc


def compute_ece(
    confidences: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 15,
) -> float:
    r"""
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between confidence and accuracy across bins:

    .. math::
        ECE = \sum_{b=1}^{B} \frac{|B_b|}{N} |acc(B_b) - conf(B_b)|

    See `On Calibration of Modern Neural Networks <https://arxiv.org/abs/1706.04599>`_ for details.

    Args:
        confidences (Tensor): Prediction confidences (max probability) of shape :math:`(N,)`
            with values in range [0, 1]
        targets (Tensor): Ground truth correctness (bool) of shape :math:`(N,)`
        num_bins (int): Number of bins for ECE computation. Default: ``15``

    Returns:
        float: ECE score in range [0, 1]

    Examples::

        >>> confidences = torch.tensor([0.9, 0.3, 0.8, 0.6])
        >>> targets = torch.tensor([True, False, True, False])
        >>> compute_ece(confidences, targets, num_bins=10)
        0.125
    """
    if confidences.numel() == 0:
        return 0.0

    # Create bins
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    total_samples = confidences.size(0)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_count = in_bin.sum().item()

        if bin_count > 0:
            # Compute accuracy and confidence in this bin
            bin_correct = targets[in_bin].sum().item()
            bin_confidence = confidences[in_bin].mean().item()

            accuracy = bin_correct / bin_count
            avg_confidence = bin_confidence

            # Add to ECE
            ece += (bin_count / total_samples) * abs(accuracy - avg_confidence)

    return ece



class MetricsComputer:
    """Class for computing metrics during evaluation

    Accumulates predictions and targets, then computes metrics.
    """

    def __init__(self, num_classes: int = 200):
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        """Reset accumulators"""
        self.all_predictions: List[torch.Tensor] = []
        self.all_targets: List[torch.Tensor] = []
        self.all_probs: List[torch.Tensor] = []

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
    ) -> None:
        """Update with batch predictions"""
        self.all_predictions.append(predictions.cpu())
        self.all_targets.append(targets.cpu())
        if probs is not None:
            self.all_probs.append(probs.cpu())

    def compute(self) -> Dict[str, float]:
        """No-op stub (PR0: compute_all_metrics deleted as transitively dead).

        MetricsComputer is kept for Q5 backward compat only; collect/update are
        still functional but compute() returns an empty dict since all underlying
        metrics functions (compute_nll / compute_brier_score / compute_all_metrics)
        are removed in PR0 as 0-callers / transitively-dead.
        """
        if not self.all_predictions:
            return {}
        return {}


__all__ = [
    "compute_accuracy",
    "compute_confusion_matrix",
    "compute_per_class_accuracy",
    "compute_ece",
    "MetricsComputer",
]
