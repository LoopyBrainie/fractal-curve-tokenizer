"""Metrics Computation Module

Provides metric computation utilities.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict
import torch
import numpy as np


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    top_k: int = 1,
) -> float:
    """Compute top-k accuracy

    Args:
        predictions: [N, C] prediction logits
        targets: [N] ground truth labels
        top_k: Compute top-k accuracy

    Returns:
        Accuracy percentage
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
    """Compute confusion matrix

    Args:
        predictions: [N] predicted class indices
        targets: [N] ground truth class indices
        num_classes: Number of classes

    Returns:
        [num_classes, num_classes] confusion matrix
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
    """Compute per-class accuracy

    Args:
        predictions: [N] predicted class indices
        targets: [N] ground truth class indices
        num_classes: Number of classes

    Returns:
        Dictionary mapping class index to accuracy
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
    """Compute Expected Calibration Error

    Args:
        confidences: [N] prediction confidences [0, 1]
        targets: [N] ground truth correctness (bool)
        num_bins: Number of bins for ECE

    Returns:
        ECE score
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


def compute_nll(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Compute Negative Log Likelihood

    Args:
        log_probs: [N, C] log probabilities
        targets: [N] ground truth class indices

    Returns:
        NLL score
    """
    nll = torch.nn.functional.nll_loss(log_probs, targets)
    return nll.item()


def compute_brier_score(
    probs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Compute Brier Score

    BS = (1/N) * Σ ||p(y) - o(y)||²

    where p(y) is predicted probability and o(y) is one-hot ground truth.

    Args:
        probs: [N, C] predicted probabilities
        targets: [N] ground truth class indices

    Returns:
        Brier score
    """
    batch_size = probs.size(0)
    num_classes = probs.size(-1)

    # Create one-hot targets
    targets_one_hot = torch.zeros_like(probs)
    targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

    # Compute squared difference
    brier = ((probs - targets_one_hot) ** 2).sum(dim=-1).mean()

    return brier.item()


def compute_all_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    num_classes: int = 200,
) -> Dict[str, float]:
    """Compute all available metrics

    Args:
        predictions: [N] predicted class indices
        targets: [N] ground truth class indices
        probs: [N, C] predicted probabilities (optional)
        num_classes: Number of classes

    Returns:
        Dictionary of metric names to values
    """
    metrics = {}

    # Top-1 accuracy
    metrics["accuracy"] = compute_accuracy(predictions, targets, top_k=1)

    # Top-5 accuracy (if applicable)
    if num_classes > 1:
        metrics["top5_accuracy"] = compute_accuracy(predictions, targets, top_k=5)

    # Per-class accuracy
    per_class_acc = compute_per_class_accuracy(predictions, targets, num_classes)
    metrics["avg_class_accuracy"] = sum(per_class_acc.values()) / max(len(per_class_acc), 1)

    # Confusion matrix (just return as list for logging)
    confusion = compute_confusion_matrix(predictions, targets, num_classes)
    metrics["confusion"] = confusion.tolist()

    # ECE (if probs provided)
    if probs is not None:
        confidences, _ = probs.max(dim=-1)
        correctness = (predictions == targets)
        metrics["ece"] = compute_ece(confidences, correctness)

        # NLL
        log_probs = torch.log(probs + 1e-10)
        metrics["nll"] = compute_nll(log_probs, targets)

        # Brier score
        metrics["brier_score"] = compute_brier_score(probs, targets)

    return metrics


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
        """Compute all metrics"""
        if not self.all_predictions:
            return {}

        predictions = torch.cat(self.all_predictions)
        targets = torch.cat(self.all_targets)
        probs = torch.cat(self.all_probs) if self.all_probs else None

        return compute_all_metrics(predictions, targets, probs, self.num_classes)


__all__ = [
    "compute_accuracy",
    "compute_confusion_matrix",
    "compute_per_class_accuracy",
    "compute_ece",
    "compute_nll",
    "compute_brier_score",
    "compute_all_metrics",
    "MetricsComputer",
]
