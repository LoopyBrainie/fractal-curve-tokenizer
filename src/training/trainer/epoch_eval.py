"""Evaluation Module

Implements independent evaluate function with metrics calculation.
Following the three-layer parameter principle, this module
handles Layer 3 (hyperparameters) for evaluation.
"""

from __future__ import annotations

from typing import Optional, Dict, List, Union
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader

from ..config import Config
from .state import EvaluationMetrics
from ..training_logs.metrics import compute_confusion_matrix


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

    # I-OPT: 初始化为 GPU tensor，直接累加避免循环内 .item() 同步
    # 注意: 评估模式下使用 torch.no_grad()，所以 tensor 累积是安全的
    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0, device=device)
    total_top5_correct = torch.tensor(0, device=device)
    total_samples = 0
    num_batches = 0

    # For ECE calculation - 保持 GPU tensor，最后统一转换
    all_confidences: List[torch.Tensor] = []
    all_correct: List[torch.Tensor] = []

    # For confusion matrix - 保持 GPU tensor
    all_predictions: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    # For per-class accuracy - 使用 bincount 向量化的 GPU tensor
    # I-OPT: 使用 torch.long 避免 scatter_add dtype 不匹配
    class_correct = torch.zeros(num_classes, device=device, dtype=torch.long)
    class_total = torch.zeros(num_classes, device=device, dtype=torch.long)

    total_batches = len(dataloader)

    # P1 诊断: 验证样本数量
    total_batches * dataloader.batch_size
    actual_samples = len(dataloader.dataset)
    print(f"  [P0诊断] Validation: {actual_samples} samples, {total_batches} batches (batch_size={dataloader.batch_size})")

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
            with autocast('cuda', enabled=use_amp):
                outputs = model(images)

                # Handle TrainingStats from Fractal ViT
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

            # Compute loss
            # I-OPT: 直接累加 tensor，不在循环内 .item()
            if labels is not None:
                loss = nn.functional.cross_entropy(logits, labels)
                total_loss = total_loss + loss.detach()

            # Get predictions
            if logits is not None:
                batch_size = images.size(0)

                # Top-1 accuracy
                pred = logits.argmax(dim=-1)

                if labels is not None:
                    correct = (pred == labels).sum()  # I-OPT: tensor, no .item()
                    total_correct = total_correct + correct.detach()

                    # I-OPT: 向量化 per-class accuracy，使用 scatter_add 替代循环
                    # 原实现: for c in range(num_classes): .item() x2 per class (200 syncs/batch)
                    # 新实现: 使用 scatter_add 向量化，延迟 .item() 到最后
                    # 每个类的正确预测数
                    correct_mask = (pred == labels).long()  # [B]
                    class_correct.scatter_add_(0, labels.long(), correct_mask)
                    # 每个类的总样本数
                    class_total.scatter_add_(0, labels.long(), torch.ones_like(labels).long())

                    # Top-5 accuracy - I-OPT: 直接累加 tensor
                    if num_classes > 1:
                        _, top5_pred = logits.topk(min(5, num_classes), dim=-1)
                        top5_correct = (top5_pred == labels.unsqueeze(-1)).any(dim=-1).sum()
                        total_top5_correct = total_top5_correct + top5_correct.detach()

                    # I-OPT: ECE 数据保持 GPU tensor，最后统一转换
                    probs = torch.softmax(logits, dim=-1)
                    confidences, _ = probs.max(dim=-1)
                    all_confidences.append(confidences.detach())  # Keep tensor
                    all_correct.append((pred == labels).detach())  # Keep tensor

                    # I-OPT: Confusion matrix 保持 GPU tensor
                    all_predictions.append(pred.detach())
                    all_targets.append(labels.detach())

                total_samples += batch_size

            num_batches += 1

            # I-OPT: Batch 进度日志 - 延迟 .item() 到日志输出时
            # 只在每 10 个 batch 打印，不影响性能
            if batch_idx > 0 and batch_idx % 10 == 0 and labels is not None:
                loss_val = loss.detach().item()
                batch_acc = correct.detach().item() / batch_size
                print(f"  Eval batch {batch_idx}/{total_batches} | Loss: {loss_val:.4f} | Acc: {batch_acc:.2%}")

    # Compute final metrics
    # I-OPT: 直接 .item() 转换，tensor 已在 GPU
    avg_loss = (total_loss / max(num_batches, 1)).item()
    accuracy = (total_correct / max(total_samples, 1)).item()
    top5_accuracy = (total_top5_correct / max(total_samples, 1)).item()

    # Compute ECE
    # I-OPT: all_confidences/all_correct 现在是 List[Tensor]，需要转换
    ece = None
    if compute_ece and all_confidences:
        # 批量转换所有 tensor 到 CPU 列表（单次同步）
        all_conf_flat = torch.cat(all_confidences).cpu().tolist()
        all_correct_flat = torch.cat(all_correct).cpu().tolist()
        # D3-AUDIT FIX: 传入 device 避免 compute_ece_score 内部创建 CPU tensor
        ece = compute_ece_score(
            all_conf_flat,
            all_correct_flat,
            num_bins=ece_bins,
            device=device,
        )

    # Per-class accuracy - I-OPT: class_correct/class_total 现在是 GPU tensor
    per_class_acc = {}
    for c in range(num_classes):
        if class_total[c] > 0:
            per_class_acc[c] = (class_correct[c] / class_total[c]).item()

    # Compute confusion matrix - I-OPT: all_predictions/targets 已是 GPU tensor
    confusion_matrix = None
    if all_predictions and all_targets:
        all_pred = torch.cat(all_predictions)
        all_target = torch.cat(all_targets)
        confusion_matrix = compute_confusion_matrix(all_pred, all_target, num_classes)

    return EvaluationMetrics(
        loss=avg_loss,
        accuracy=accuracy,
        top5_accuracy=top5_accuracy,
        ece=ece,
        per_class_accuracy=per_class_acc if per_class_acc else None,
        confusion_matrix=confusion_matrix,
        num_samples=total_samples,
    )


def compute_ece_score(
    confidences: List[float],
    correct: List[bool],
    num_bins: int = 15,
    device: Optional[torch.device] = None,
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

    # D3-AUDIT FIX: 指定 device 避免隐式 CPU 创建
    # 若 device 为 None，则默认 CPU（对 ECE 计算无性能影响）
    _device = device or torch.device('cpu')
    confidences = torch.tensor(confidences, device=_device)
    correct = torch.tensor(correct, dtype=torch.float, device=_device)

    # D1-AUDIT FIX: 向量化 ECE 计算，避免 45 次 .item() 同步
    # 原来: 3 .item() × 15 bins = 45 次同步
    # 现在: 仅 2 次同步（bin_boundaries 创建时可能有一次）
    total_samples = len(confidences)
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=_device)

    # 为每个样本分配 bin 索引
    bin_indices = torch.searchsorted(bin_boundaries[1:-1], confidences)
    bin_indices = bin_indices.clamp(0, num_bins - 1)

    # 使用 scatter_add 一次性计算每个 bin 的统计量
    bin_counts = torch.zeros(num_bins, device=_device)
    bin_counts.scatter_add_(0, bin_indices, torch.ones_like(bin_indices).float())

    bin_correct_sum = torch.zeros(num_bins, device=_device)
    bin_correct_sum.scatter_add_(0, bin_indices, correct)

    bin_confidence_sum = torch.zeros(num_bins, device=_device)
    bin_confidence_sum.scatter_add_(0, bin_indices, confidences)

    # 计算每个 bin 的 ECE 并求和（仅在非空 bin 上计算）
    ece = 0.0
    nonzero_mask = bin_counts > 0
    if nonzero_mask.any():
        nonzero_bins = nonzero_mask.nonzero(as_tuple=True)[0]
        for b in nonzero_bins:
            count = bin_counts[b].item()
            accuracy = (bin_correct_sum[b] / count).item()
            avg_confidence = (bin_confidence_sum[b] / count).item()
            ece += (count / total_samples) * abs(accuracy - avg_confidence)

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
            images = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)

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
