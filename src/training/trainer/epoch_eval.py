"""Evaluation Module

Implements independent evaluate function with metrics calculation.
Following the three-layer parameter principle, this module
handles Layer 3 (hyperparameters) for evaluation.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from ..config import Config
from .state import EvaluationMetrics
from .loss import compute_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: Optional[Config] = None,
    num_classes: int = 200,
    compute_ece: bool = True,
    ece_bins: int = 15,
) -> EvaluationMetrics:
    """Evaluate model on validation/test set

    This function is COMPLETELY INDEPENDENT from model architecture.
    It only receives model through the generic nn.Module interface.

    Args:
        model: Neural network model
        dataloader: Validation/test data loader
        device: Device to evaluate on
        config: Optional training config
        num_classes: Number of classes
        compute_ece: Whether to compute Expected Calibration Error
        ece_bins: Number of bins for ECE calculation

    Returns:
        EvaluationMetrics with validation statistics
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_top5_correct = 0
    total_samples = 0
    num_batches = 0

    # For ECE calculation
    all_confidences: List[float] = []
    all_correct: List[bool] = []

    # For per-class accuracy
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)

    total_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=total_batches, desc="Evaluating", leave=False):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(device, non_blocking=True)
                labels = batch[1].to(device, non_blocking=True)
            else:
                images = batch.to(device, non_blocking=True)
                labels = None

            # Forward pass with AMP
            use_amp = config.amp.enabled if config else False
            with autocast(enabled=use_amp):
                outputs = model(images)

                # Handle TrainingStats from Fractal ViT
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

            # Compute loss
            if labels is not None:
                loss = nn.functional.cross_entropy(logits, labels)
                total_loss += loss.item()

            # Get predictions
            if logits is not None:
                batch_size = images.size(0)

                # Top-1 accuracy
                pred = logits.argmax(dim=-1)

                if labels is not None:
                    correct = (pred == labels).sum().item()
                    total_correct += correct

                    # Per-class accuracy
                    for c in range(num_classes):
                        mask = labels == c
                        if mask.any():
                            class_correct[c] += ((pred == labels) & mask).sum().item()
                            class_total[c] += mask.sum().item()

                    # Top-5 accuracy
                    if num_classes > 1:
                        _, top5_pred = logits.topk(min(5, num_classes), dim=-1)
                        top5_correct = (top5_pred == labels.unsqueeze(-1)).any(dim=-1).sum().item()
                        total_top5_correct += top5_correct

                    # Collect for ECE
                    probs = torch.softmax(logits, dim=-1)
                    confidences, _ = probs.max(dim=-1)
                    all_confidences.extend(confidences.cpu().tolist())
                    all_correct.extend((pred == labels).cpu().tolist())

                total_samples += batch_size

            num_batches += 1

            # Batch 进度日志 (每 10 个 batch 打印一次)
            if batch_idx > 0 and batch_idx % 10 == 0 and labels is not None:
                batch_acc = correct / batch_size if batch_size > 0 else 0.0
                print(f"  Eval batch {batch_idx}/{total_batches} | Loss: {loss.item():.4f} | Acc: {batch_acc:.2%}")

    # Compute final metrics
    avg_loss = total_loss / max(num_batches, 1)
    accuracy = total_correct / max(total_samples, 1) if total_samples > 0 else 0.0
    top5_accuracy = total_top5_correct / max(total_samples, 1) if total_samples > 0 else None

    # Compute ECE
    ece = None
    if compute_ece and all_confidences:
        ece = compute_ece_score(
            all_confidences,
            all_correct,
            num_bins=ece_bins,
        )

    # Per-class accuracy
    per_class_acc = {}
    for c in range(num_classes):
        if class_total[c] > 0:
            per_class_acc[c] = (class_correct[c] / class_total[c]).item()

    return EvaluationMetrics(
        loss=avg_loss,
        accuracy=accuracy,
        top5_accuracy=top5_accuracy,
        ece=ece,
        per_class_accuracy=per_class_acc if per_class_acc else None,
        num_samples=total_samples,
    )


def compute_ece_score(
    confidences: List[float],
    correct: List[bool],
    num_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error

    ECE = Σ_{b=1}^{n} (|B_b| / N) * |acc(B_b) - conf(B_b)|

    where B_b is the set of samples whose confidence falls in bin b.

    Args:
        confidences: List of prediction confidences [0, 1]
        correct: List of boolean correctness
        num_bins: Number of bins

    Returns:
        ECE score
    """
    if not confidences or not correct:
        return 0.0

    confidences = torch.tensor(confidences)
    correct = torch.tensor(correct, dtype=torch.float)

    # Create bins
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    total_samples = len(confidences)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_count = in_bin.sum().item()

        if bin_count > 0:
            # Compute accuracy and confidence in this bin
            bin_correct = correct[in_bin].sum().item()
            bin_confidence = confidences[in_bin].mean().item()

            accuracy = bin_correct / bin_count
            avg_confidence = bin_confidence

            # Add to ECE
            ece += (bin_count / total_samples) * abs(accuracy - avg_confidence)

    return ece


def evaluate_simple(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 200,
) -> Dict[str, float]:
    """Simplified evaluation (minimal version)

    For quick testing without full config.

    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device
        num_classes: Number of classes

    Returns:
        Dictionary of metrics
    """
    model.eval()

    total_correct = 0
    total_top5_correct = 0
    total_samples = 0

    total_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=total_batches, desc="Evaluating", leave=False):
            images = batch[0].to(device)
            labels = batch[1].to(device)

            outputs = model(images)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits

            pred = outputs.argmax(dim=-1)
            total_correct += (pred == labels).sum().item()

            if num_classes > 1:
                _, top5_pred = outputs.topk(min(5, num_classes), dim=-1)
                top5_correct = (top5_pred == labels.unsqueeze(-1)).any(dim=-1).sum().item()
                total_top5_correct += top5_correct

            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    top5_accuracy = total_top5_correct / total_samples if total_samples > 0 else 0.0

    return {
        "accuracy": accuracy,
        "top5_accuracy": top5_accuracy,
    }


__all__ = [
    "evaluate",
    "evaluate_simple",
    "compute_ece_score",
]
