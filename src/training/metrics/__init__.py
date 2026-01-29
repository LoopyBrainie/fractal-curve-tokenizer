"""分类评估指标模块

数学形式化
==========

**Top-K Accuracy**:
    Acc@K = (1/N) · Σ_i 𝟙[y_i ∈ top_K(ŷ_i)]

**Per-Class Accuracy**:
    Acc_c = TP_c / n_c
    
其中 n_c 是类别 c 的样本数。

**Mean Class Accuracy (MCA)**:
    MCA = (1/C) · Σ_c Acc_c

与 Top-1 的区别:
    - Top-1: 对所有样本平等计数
    - MCA: 对所有类别平等计数

**Balanced Accuracy**:
    BA = MCA (在平衡数据集上等价于 Top-1)

**示例**:
    类别分布: {A: 90, B: 10}
    预测: A 全对, B 全错
    Top-1 = 90/100 = 90%
    MCA = (100% + 0%) / 2 = 50%
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import Tensor


@dataclass
class ClassificationMetricsState:
    """分类指标的累积状态"""
    
    # 每个类别的统计
    class_correct: Tensor  # [num_classes]
    class_total: Tensor    # [num_classes]
    
    # 全局统计
    total_correct: int = 0
    total_samples: int = 0
    
    # Top-5 统计
    top5_correct: int = 0
    
    # 混淆矩阵 (可选)
    confusion_matrix: Optional[Tensor] = None


@dataclass
class ClassificationMetricsResult:
    """分类指标计算结果"""
    
    top1_accuracy: float
    top5_accuracy: float
    mean_class_accuracy: float  # MCA
    per_class_accuracy: Dict[int, float]
    
    # 额外统计
    num_samples: int
    num_classes: int
    
    # 类别分布信息
    head_accuracy: float = 0.0  # 头部类别准确率 (样本数 > median)
    tail_accuracy: float = 0.0  # 尾部类别准确率 (样本数 < median)
    
    def __repr__(self) -> str:
        return (
            f"ClassificationMetricsResult(\n"
            f"  Top-1: {self.top1_accuracy:.2%}\n"
            f"  Top-5: {self.top5_accuracy:.2%}\n"
            f"  MCA:   {self.mean_class_accuracy:.2%}\n"
            f"  Head:  {self.head_accuracy:.2%}\n"
            f"  Tail:  {self.tail_accuracy:.2%}\n"
            f"  Samples: {self.num_samples}, Classes: {self.num_classes}\n"
            f")"
        )


class ClassificationMetrics:
    """分类评估指标收集器
    
    特点:
    - 增量更新，支持分批计算
    - 同时计算 Top-1, Top-5, MCA, Per-Class
    - 区分头部/尾部类别性能
    
    Args:
        num_classes: 类别数量
        class_counts: [num_classes] 每个类别的样本数 (用于头部/尾部划分)
        
    Example:
        >>> metrics = ClassificationMetrics(num_classes=200)
        >>> for batch in dataloader:
        ...     logits, targets = model(batch), batch.labels
        ...     metrics.update(logits, targets)
        >>> result = metrics.compute()
        >>> print(result.mean_class_accuracy)
    """
    
    def __init__(
        self,
        num_classes: int,
        class_counts: Optional[Tensor] = None,
    ) -> None:
        self.num_classes = num_classes
        self.class_counts = class_counts
        
        # 计算头部/尾部类别划分阈值
        if class_counts is not None:
            self.count_median = float(class_counts.float().median())
            self.head_classes = set((class_counts > self.count_median).nonzero().squeeze(-1).tolist())
            self.tail_classes = set((class_counts <= self.count_median).nonzero().squeeze(-1).tolist())
        else:
            self.count_median = None
            self.head_classes = set()
            self.tail_classes = set()
        
        self.reset()
    
    def reset(self) -> None:
        """重置所有累积状态"""
        self._class_correct = torch.zeros(self.num_classes, dtype=torch.long)
        self._class_total = torch.zeros(self.num_classes, dtype=torch.long)
        self._total_correct = 0
        self._total_samples = 0
        self._top5_correct = 0
    
    @torch.no_grad()
    def update(self, logits: Tensor, targets: Tensor) -> None:
        """更新指标状态
        
        Args:
            logits: [B, C] 模型输出 (未归一化)
            targets: [B] 真实标签
        """
        device = logits.device
        batch_size = logits.size(0)
        
        # 确保统计张量在正确设备上
        if self._class_correct.device != device:
            self._class_correct = self._class_correct.to(device)
            self._class_total = self._class_total.to(device)
        
        # Top-1 预测
        _, pred = logits.max(dim=-1)  # [B]
        correct = pred.eq(targets)    # [B]
        
        # 全局统计
        self._total_correct += correct.sum().item()
        self._total_samples += batch_size
        
        # Top-5 预测
        if self.num_classes >= 5:
            _, top5_pred = logits.topk(5, dim=-1)  # [B, 5]
            top5_correct = top5_pred.eq(targets.unsqueeze(-1)).any(dim=-1)
            self._top5_correct += top5_correct.sum().item()
        else:
            self._top5_correct = self._total_correct

        # Per-class 统计 (FIX: 向量化使用 bincount，避免 Python 循环)
        # _class_total[c] = count of samples with target class c
        # _class_correct[c] = count of correctly predicted samples with target class c
        targets_long = targets.long()
        pred_long = pred.long()
        # 使用 bincount 批量更新，避免循环
        class_total_batch = torch.bincount(targets_long, minlength=self.num_classes)
        # 修复: 只统计正确预测的类别索引，避免错误预测被计入类别 0
        # 正确预测的类别索引
        correct_class_indices = targets_long[correct]  # 只取正确预测对应的类别
        class_correct_batch = torch.bincount(correct_class_indices, minlength=self.num_classes)
        # 累积到全局统计
        self._class_total += class_total_batch
        self._class_correct += class_correct_batch
    
    def compute(self) -> ClassificationMetricsResult:
        """计算最终指标"""
        # Top-1
        top1 = self._total_correct / max(self._total_samples, 1)
        
        # Top-5
        top5 = self._top5_correct / max(self._total_samples, 1)
        
        # Per-class accuracy
        per_class_acc = {}
        valid_classes = 0
        mca_sum = 0.0
        
        head_correct = 0
        head_total = 0
        tail_correct = 0
        tail_total = 0
        
        for c in range(self.num_classes):
            total = self._class_total[c].item()
            correct = self._class_correct[c].item()
            
            if total > 0:
                acc = correct / total
                per_class_acc[c] = acc
                mca_sum += acc
                valid_classes += 1
                
                # 头部/尾部统计
                if c in self.head_classes:
                    head_correct += correct
                    head_total += total
                elif c in self.tail_classes:
                    tail_correct += correct
                    tail_total += total
            else:
                per_class_acc[c] = 0.0
        
        # MCA
        mca = mca_sum / max(valid_classes, 1)
        
        # 头部/尾部准确率
        head_acc = head_correct / max(head_total, 1) if head_total > 0 else 0.0
        tail_acc = tail_correct / max(tail_total, 1) if tail_total > 0 else 0.0
        
        return ClassificationMetricsResult(
            top1_accuracy=top1,
            top5_accuracy=top5,
            mean_class_accuracy=mca,
            per_class_accuracy=per_class_acc,
            num_samples=self._total_samples,
            num_classes=self.num_classes,
            head_accuracy=head_acc,
            tail_accuracy=tail_acc,
        )


class ResourceMetrics:
    """资源使用指标收集器
    
    跟踪:
    - Token 数量统计
    - 深度分布
    - (可选) FLOPS 估算
    
    Args:
        max_depth: 最大深度
    """
    
    def __init__(self, max_depth: int = 4) -> None:
        self.max_depth = max_depth
        self.reset()
    
    def reset(self) -> None:
        """重置状态"""
        self._token_counts: List[int] = []
        self._depth_counts = torch.zeros(self.max_depth + 1, dtype=torch.long)
        self._total_samples = 0
    
    @torch.no_grad()
    def update(
        self,
        token_counts: Tensor,
        depths: Optional[Tensor] = None,
    ) -> None:
        """更新资源指标
        
        Args:
            token_counts: [B] 每个样本的 token 数量
            depths: [total_tokens] 每个 token 的深度
        """
        # Token 数量
        for count in token_counts.tolist():
            self._token_counts.append(count)

        self._total_samples += token_counts.size(0)

        # 深度分布 (I108-3: 向量化 bincount 替代 Python 循环)
        if depths is not None:
            depths = depths.clamp(0, self.max_depth)
            # bincount: counts[d] = |{i: depth_i = d}|
            depth_counts_batch = torch.bincount(depths.long(), minlength=self.max_depth + 1)
            self._depth_counts += depth_counts_batch
    
    def compute(self) -> Dict[str, float]:
        """计算资源指标"""
        token_counts = torch.tensor(self._token_counts, dtype=torch.float)
        
        results = {
            "avg_tokens": token_counts.mean().item() if len(token_counts) > 0 else 0,
            "min_tokens": token_counts.min().item() if len(token_counts) > 0 else 0,
            "max_tokens": token_counts.max().item() if len(token_counts) > 0 else 0,
            "std_tokens": token_counts.std().item() if len(token_counts) > 1 else 0,
        }
        
        # 深度分布 (I108-3: 向量化计算)
        total_tokens = self._depth_counts.sum().item()
        if total_tokens > 0:
            # 向量化: ratios = counts / total
            depth_ratios = (self._depth_counts.float() / total_tokens).tolist()
            for d, ratio in enumerate(depth_ratios):
                results[f"depth_{d}_ratio"] = ratio
        
        return results
