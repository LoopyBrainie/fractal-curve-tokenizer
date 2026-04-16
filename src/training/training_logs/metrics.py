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


def compute_nll(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    r"""
    Compute Negative Log Likelihood (NLL).

    Args:
        log_probs (Tensor): Log probabilities of shape :math:`(N, C)`
        targets (Tensor): Ground truth class indices of shape :math:`(N,)`

    Returns:
        float: NLL score

    Examples::

        >>> log_probs = torch.randn(32, 10).log_softmax(dim=-1)
        >>> targets = torch.randint(0, 10, (32,))
        >>> compute_nll(log_probs, targets)
        2.345
    """
    nll = torch.nn.functional.nll_loss(log_probs, targets)
    return nll.item()


def compute_brier_score(
    probs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    r"""
    Compute Brier Score.

    Measures the mean squared difference between predicted probabilities and one-hot ground truth:

    .. math::
        BS = \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} (p_c(y_i) - o_c(y_i))^2

    where :math:`p_c(y_i)` is the predicted probability and :math:`o_c(y_i)` is the one-hot ground truth.

    See `Stochastic Gradient Estimation Using Single Sample Partially Descent Neural Networks <https://arxiv.org/abs/1807.01118>`_ for details.

    Args:
        probs (Tensor): Predicted probabilities of shape :math:`(N, C)`
        targets (Tensor): Ground truth class indices of shape :math:`(N,)`

    Returns:
        float: Brier score in range [0, 2]

    Examples::

        >>> probs = torch.softmax(torch.randn(32, 10), dim=-1)
        >>> targets = torch.randint(0, 10, (32,))
        >>> compute_brier_score(probs, targets)
        1.456
    """
    probs.size(0)
    probs.size(-1)

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
    r"""
    Compute all available metrics.

    Computes accuracy (top-1 and top-5), per-class accuracy, confusion matrix,
    and optionally ECE, NLL, and Brier score if probabilities are provided.

    Args:
        predictions (Tensor): Predicted class indices of shape :math:`(N,)`
        targets (Tensor): Ground truth class indices of shape :math:`(N,)`
        probs (Tensor, optional): Predicted probabilities of shape :math:`(N, C)`.
            If provided, computes calibration metrics. Default: ``None``
        num_classes (int): Number of classes. Default: ``200``

    Returns:
        Dict[str, float]: Dictionary of metric names to values:
            - ``"accuracy"``: Top-1 accuracy
            - ``"top5_accuracy"``: Top-5 accuracy (if num_classes > 1)
            - ``"avg_class_accuracy"``: Average per-class accuracy
            - ``"confusion"``: Confusion matrix as list
            - ``"ece"``: Expected Calibration Error (if probs provided)
            - ``"nll"``: Negative Log Likelihood (if probs provided)
            - ``"brier_score"``: Brier Score (if probs provided)

    Examples::

        >>> predictions = torch.randint(0, 10, (32,))
        >>> targets = torch.randint(0, 10, (32,))
        >>> probs = torch.softmax(torch.randn(32, 10), dim=-1)
        >>> metrics = compute_all_metrics(predictions, targets, probs, num_classes=10)
        >>> list(metrics.keys())
        ['accuracy', 'top5_accuracy', 'avg_class_accuracy', 'confusion', 'ece', 'nll', 'brier_score']
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
