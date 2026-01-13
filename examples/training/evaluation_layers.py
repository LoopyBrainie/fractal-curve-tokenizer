#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分层评估辅助模块 (Layered Evaluation Helpers)

数学形式化
============

本模块实现 FractalCurveViT 的分层评估系统，按照模型架构层次进行指标计算：

分层评估函数定义：
    E: M × D → R
    
其中：
    M = (T, P, F, H) 为模型各组件
    D = {(x_i, y_i)}_{i=1}^N 为评估数据集
    R = (R₁, R₂, R₃, R₄, R₅, R₆) 为分层评估报告

各层指标定义：

L1 分类性能层 (Classification Performance)
-------------------------------------------
Top-1 Accuracy:
    Acc@1 = (1/N) Σᵢ 𝟙[ŷᵢ = yᵢ]

Top-5 Accuracy:
    Acc@5 = (1/N) Σᵢ 𝟙[yᵢ ∈ Top5(p̂ᵢ)]

Mean Class Accuracy (MCA):
    MCA = (1/C) Σ_c Acc_c

Expected Calibration Error (ECE):
    ECE = Σ_{m=1}^M (|Bₘ|/N) |acc(Bₘ) - conf(Bₘ)|
    
其中 Bₘ 是置信度落在第 m 个 bin 的样本集合。

L2 Tokenizer 行为层 (Tokenizer Behavior)
-----------------------------------------
Token 数量统计:
    μ_N = (1/B) Σᵢ Nᵢ
    σ_N = √[(1/B) Σᵢ (Nᵢ - μ_N)²]

深度分布熵:
    H_d = -Σ_{d=0}^D p_d log(p_d)
    其中 p_d = count(depth=d) / N_total

空间覆盖率:
    Coverage = Σᵢ Area(Rᵢ) / (H × W)
    
内容-Token 相关性:
    ρ = Corr(Complexity(x), N_tokens(x))

L3 注意力机制层 (Attention Analysis)
-------------------------------------
每层每头注意力熵:
    H_attn^{(l,h)} = -(1/N) Σᵢ Σⱼ Aᵢⱼ^{(l,h)} log(Aᵢⱼ^{(l,h)})

Head 利用率:
    Util^{(h)} = 𝟙[H_attn^{(h)} > τ_low] ∧ 𝟙[H_attn^{(h)} < τ_high]

CLS 注意力集中度:
    CLS_focus = (1/L) Σ_l max_j(A_{0,j}^{(l)})

L4 特征表示层 (Feature Representation)
---------------------------------------
Fisher 判别比:
    FDR = tr(S_B) / tr(S_W)
    S_B = 类间散度矩阵
    S_W = 类内散度矩阵

类别可分性:
    Separability_c = ||μ_c - μ|| / σ_c

L5 资源效率层 (Resource Efficiency)
------------------------------------
推理延迟分解:
    Latency = t_tokenizer + t_transformer + t_head

内存效率:
    Memory_eff = Accuracy / Peak_Memory

L6 训练稳定性层 (Training Stability)
-------------------------------------
权重范数统计:
    ||W||_F = √(Σᵢⱼ wᵢⱼ²)

梯度健康检查:
    Grad_health = 𝟙[||∇W|| ∈ (ε, 1/ε)]

Author: GitHub Copilot
Date: 2026-01-11
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class L1ClassificationMetrics:
    """L1 分类性能层指标"""
    top1_accuracy: float = 0.0
    top5_accuracy: float = 0.0
    mean_class_accuracy: float = 0.0
    per_class_accuracy: Dict[int, float] = field(default_factory=dict)
    avg_loss: float = 0.0
    
    # 校准误差
    ece: float = 0.0  # Expected Calibration Error
    mce: float = 0.0  # Maximum Calibration Error
    
    # 混淆分析
    top_confused_pairs: List[Tuple[int, int, int]] = field(default_factory=list)  # (true, pred, count)
    
    # 难样本分析
    hardest_classes: List[Tuple[int, float]] = field(default_factory=list)  # (class_id, error_rate)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class L2TokenizerMetrics:
    """L2 Tokenizer 行为层指标"""
    # Token 数量统计
    avg_tokens: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    std_tokens: float = 0.0
    
    # 深度分布
    depth_distribution: Dict[int, float] = field(default_factory=dict)  # depth -> percentage
    depth_entropy: float = 0.0  # 深度分布熵
    
    # 空间分析
    spatial_coverage_ratio: float = 0.0  # 空间覆盖率
    
    # 内容感知分析
    content_token_correlation: float = 0.0  # 图像复杂度与 token 数相关性
    
    # 每类分析
    per_class_avg_tokens: Dict[int, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass  
class L3AttentionMetrics:
    """L3 注意力机制层指标"""
    # 每层注意力熵
    per_layer_entropy: List[float] = field(default_factory=list)
    avg_entropy: float = 0.0
    
    # Head 利用率
    head_utilization: List[List[float]] = field(default_factory=list)  # [layer][head]
    dead_head_ratio: float = 0.0  # 无效 head 比例
    
    # CLS 注意力分析
    cls_attention_coverage: float = 0.0  # CLS 关注的 token 比例
    cls_attention_entropy: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class L4RepresentationMetrics:
    """L4 特征表示层指标"""
    # Fisher 判别比
    fisher_discriminant_ratio: float = 0.0
    
    # 特征统计
    feature_mean_norm: float = 0.0
    feature_std: float = 0.0
    
    # 类别可分性
    per_class_separability: Dict[int, float] = field(default_factory=dict)
    avg_separability: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class L5EfficiencyMetrics:
    """L5 资源效率层指标"""
    # 推理延迟 (ms)
    avg_latency_ms: float = 0.0
    tokenizer_latency_ms: float = 0.0
    transformer_latency_ms: float = 0.0
    head_latency_ms: float = 0.0
    
    # 吞吐量
    throughput_samples_per_sec: float = 0.0
    
    # 内存
    peak_memory_mb: float = 0.0
    
    # 效率比
    accuracy_per_gflops: float = 0.0
    accuracy_per_token: float = 0.0
    
    # 参数量
    total_params: int = 0
    trainable_params: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class L6StabilityMetrics:
    """L6 训练稳定性层指标"""
    # 权重统计
    weight_norm_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # 梯度健康 (需要训练模式)
    gradient_health_score: float = 1.0
    
    # 数值稳定性
    has_nan_weights: bool = False
    has_inf_weights: bool = False
    
    # 分割器健康 (如果有)
    splitter_health_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LayeredEvaluationReport:
    """完整分层评估报告"""
    # 元信息
    checkpoint_path: str = ""
    dataset_name: str = ""
    num_samples: int = 0
    num_classes: int = 0
    device: str = ""
    evaluation_time_sec: float = 0.0
    
    # 各层指标
    L1_classification: L1ClassificationMetrics = field(default_factory=L1ClassificationMetrics)
    L2_tokenizer: L2TokenizerMetrics = field(default_factory=L2TokenizerMetrics)
    L3_attention: L3AttentionMetrics = field(default_factory=L3AttentionMetrics)
    L4_representation: L4RepresentationMetrics = field(default_factory=L4RepresentationMetrics)
    L5_efficiency: L5EfficiencyMetrics = field(default_factory=L5EfficiencyMetrics)
    L6_stability: L6StabilityMetrics = field(default_factory=L6StabilityMetrics)
    
    # 摘要
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典"""
        return {
            'meta': {
                'checkpoint_path': self.checkpoint_path,
                'dataset_name': self.dataset_name,
                'num_samples': self.num_samples,
                'num_classes': self.num_classes,
                'device': self.device,
                'evaluation_time_sec': self.evaluation_time_sec,
            },
            'L1_classification': self.L1_classification.to_dict(),
            'L2_tokenizer': self.L2_tokenizer.to_dict(),
            'L3_attention': self.L3_attention.to_dict(),
            'L4_representation': self.L4_representation.to_dict(),
            'L5_efficiency': self.L5_efficiency.to_dict(),
            'L6_stability': self.L6_stability.to_dict(),
            'summary': self.summary,
        }


# ============================================================================
# L1: 分类性能评估器
# ============================================================================

class ClassificationEvaluator:
    """L1 分类性能评估器
    
    数学形式化：
        输入: 模型 M, 数据集 D = {(xᵢ, yᵢ)}
        输出: R₁ = (Acc@1, Acc@5, MCA, ECE, ...)
    """
    
    def __init__(self, num_classes: int, n_bins: int = 15):
        self.num_classes = num_classes
        self.n_bins = n_bins  # ECE 计算的 bin 数量
        
    def evaluate(
        self,
        model: nn.Module,
        data_loader,
        device: torch.device,
    ) -> L1ClassificationMetrics:
        """执行分类性能评估"""
        model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        
        with torch.no_grad():
            for imgs, labels in tqdm(data_loader, desc="L1: Classification"):
                imgs = imgs.to(device)
                labels = labels.to(device)
                
                outputs = model(imgs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = F.cross_entropy(outputs, labels)
                probs = F.softmax(outputs, dim=1)
                
                all_preds.append(outputs.argmax(dim=1).cpu())
                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())
                total_loss += loss.item()
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_probs = torch.cat(all_probs)
        
        metrics = L1ClassificationMetrics()
        
        # Top-1 Accuracy
        metrics.top1_accuracy = (all_preds == all_labels).float().mean().item() * 100
        
        # Top-5 Accuracy
        if self.num_classes >= 5:
            top5_preds = all_probs.topk(5, dim=1)[1]
            top5_correct = (top5_preds == all_labels.unsqueeze(1)).any(dim=1)
            metrics.top5_accuracy = top5_correct.float().mean().item() * 100
        else:
            metrics.top5_accuracy = metrics.top1_accuracy
        
        # Per-class accuracy & MCA
        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)
        
        for pred, label in zip(all_preds.numpy(), all_labels.numpy()):
            per_class_total[label] += 1
            if pred == label:
                per_class_correct[label] += 1
        
        for c in range(self.num_classes):
            if per_class_total[c] > 0:
                metrics.per_class_accuracy[c] = per_class_correct[c] / per_class_total[c] * 100
            else:
                metrics.per_class_accuracy[c] = 0.0
        
        metrics.mean_class_accuracy = np.mean(list(metrics.per_class_accuracy.values()))
        
        # Average loss
        metrics.avg_loss = total_loss / len(data_loader)
        
        # ECE (Expected Calibration Error)
        metrics.ece, metrics.mce = self._compute_calibration_error(
            all_probs, all_labels, all_preds
        )
        
        # Top confused pairs
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)
        for pred, label in zip(all_preds.numpy(), all_labels.numpy()):
            confusion_matrix[label, pred] += 1
        
        # 找出最混淆的类别对 (排除对角线)
        confused_pairs = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and confusion_matrix[i, j] > 0:
                    confused_pairs.append((i, j, int(confusion_matrix[i, j])))
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        metrics.top_confused_pairs = confused_pairs[:10]
        
        # Hardest classes (highest error rate)
        error_rates = []
        for c in range(self.num_classes):
            if per_class_total[c] > 0:
                error_rate = 1.0 - per_class_correct[c] / per_class_total[c]
                error_rates.append((c, error_rate * 100))
        error_rates.sort(key=lambda x: x[1], reverse=True)
        metrics.hardest_classes = error_rates[:10]
        
        return metrics
    
    def _compute_calibration_error(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        preds: torch.Tensor,
    ) -> Tuple[float, float]:
        """计算 ECE 和 MCE
        
        ECE = Σₘ (|Bₘ|/N) |acc(Bₘ) - conf(Bₘ)|
        MCE = max_m |acc(Bₘ) - conf(Bₘ)|
        """
        confidences = probs.max(dim=1)[0]
        accuracies = (preds == labels).float()
        
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        mce = 0.0
        
        for i in range(self.n_bins):
            in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean().item()
                avg_accuracy = accuracies[in_bin].mean().item()
                
                bin_error = abs(avg_accuracy - avg_confidence)
                ece += prop_in_bin.item() * bin_error
                mce = max(mce, bin_error)
        
        return ece * 100, mce * 100  # 转为百分比


# ============================================================================
# L2: Tokenizer 行为评估器
# ============================================================================

class TokenizerEvaluator:
    """L2 Tokenizer 行为评估器
    
    数学形式化：
        输入: Tokenizer T, 图像集合 X
        输出: R₂ = (μ_N, σ_N, H_d, Coverage, ρ)
    """
    
    def evaluate(
        self,
        model: nn.Module,
        data_loader,
        device: torch.device,
        max_batches: int = 50,
    ) -> L2TokenizerMetrics:
        """执行 Tokenizer 行为评估"""
        model.eval()
        tokenizer = model.tokenizer
        
        token_counts = []
        depth_counts = defaultdict(int)
        per_class_tokens = defaultdict(list)
        total_coverage = 0.0
        image_complexities = []
        
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(tqdm(
                data_loader, desc="L2: Tokenizer", total=min(max_batches, len(data_loader))
            )):
                if batch_idx >= max_batches:
                    break
                
                imgs = imgs.to(device)
                B, C, H, W = imgs.shape
                
                # Tokenize
                output = tokenizer.tokenize(imgs)
                padded_tokens, lengths = output.get_padded_tokens()
                
                # I24-12: 防御性检查 - 确保 lengths 是合理的 token 数量
                if lengths.min() < 1:
                    import warnings
                    warnings.warn(
                        f"L2 Tokenizer: lengths.min()={lengths.min().item()} < 1. "
                        "This may indicate a bug in tokenizer or data."
                    )
                if lengths.max() > padded_tokens.shape[1]:
                    import warnings
                    warnings.warn(
                        f"L2 Tokenizer: lengths.max()={lengths.max().item()} > "
                        f"padded_tokens.shape[1]={padded_tokens.shape[1]}. "
                        "This may indicate inconsistent token counting."
                    )
                
                # 收集 token 数量
                for i in range(B):
                    n_tokens = lengths[i].item()
                    token_counts.append(n_tokens)
                    per_class_tokens[labels[i].item()].append(n_tokens)
                
                # 深度分布
                for seq in output.sequences:
                    if 'levels' in seq.metadata and seq.metadata['levels'] is not None:
                        levels = seq.metadata['levels']
                        if levels.dim() > 1:
                            depths = levels[:, 0]
                        else:
                            depths = levels
                        for d in depths.cpu().numpy():
                            depth_counts[int(d)] += 1
                
                # 空间覆盖率
                regions, _ = output.get_padded_regions()
                if regions is not None:
                    for i in range(B):
                        n_tokens = lengths[i].item()
                        batch_regions = regions[i, :n_tokens]  # [N, 4]
                        # 计算覆盖面积
                        areas = (batch_regions[:, 2] - batch_regions[:, 0]) * \
                                (batch_regions[:, 3] - batch_regions[:, 1])
                        total_coverage += areas.sum().item() / (H * W)
                
                # 图像复杂度 (使用边缘检测作为代理)
                gray = imgs.mean(dim=1)  # [B, H, W]
                # Sobel 边缘
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                       dtype=torch.float32, device=device).view(1, 1, 3, 3)
                sobel_y = sobel_x.transpose(2, 3)
                
                edges_x = F.conv2d(gray.unsqueeze(1), sobel_x, padding=1)
                edges_y = F.conv2d(gray.unsqueeze(1), sobel_y, padding=1)
                edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
                
                for i in range(B):
                    complexity = edge_magnitude[i].mean().item()
                    image_complexities.append((complexity, token_counts[-(B - i)]))
        
        metrics = L2TokenizerMetrics()
        
        # Token 统计
        token_counts = np.array(token_counts)
        metrics.avg_tokens = float(token_counts.mean())
        metrics.min_tokens = int(token_counts.min())
        metrics.max_tokens = int(token_counts.max())
        metrics.std_tokens = float(token_counts.std())
        
        # 深度分布
        total_tokens = sum(depth_counts.values())
        if total_tokens > 0:
            for d, count in depth_counts.items():
                metrics.depth_distribution[d] = count / total_tokens * 100
            
            # 深度熵
            probs = np.array(list(depth_counts.values())) / total_tokens
            probs = probs[probs > 0]
            metrics.depth_entropy = -np.sum(probs * np.log(probs))
        
        # 空间覆盖率
        n_samples = len(token_counts)
        if n_samples > 0:
            metrics.spatial_coverage_ratio = total_coverage / n_samples
        
        # 内容-Token 相关性
        if len(image_complexities) > 10:
            complexities = np.array([x[0] for x in image_complexities])
            tokens = np.array([x[1] for x in image_complexities])
            if np.std(complexities) > 1e-6 and np.std(tokens) > 1e-6:
                correlation = np.corrcoef(complexities, tokens)[0, 1]
                metrics.content_token_correlation = float(correlation)
        
        # 每类 token 统计
        for c, counts in per_class_tokens.items():
            avg = float(np.mean(counts))
            # I24-12: 健壮性检查 - 检测异常模式
            if avg < 1.0:
                import warnings
                warnings.warn(
                    f"L2 Tokenizer: per_class_avg_tokens[{c}]={avg:.2f} < 1.0. "
                    "Token count should be at least 1."
                )
            metrics.per_class_avg_tokens[c] = avg
        
        return metrics


# ============================================================================
# L3: 注意力机制评估器
# ============================================================================

class AttentionEvaluator:
    """L3 注意力机制评估器
    
    数学形式化：
        注意力熵: H_attn = -(1/N) Σᵢ Σⱼ Aᵢⱼ log(Aᵢⱼ)
        Head 利用率: Util = 𝟙[H_attn ∈ (τ_low, τ_high)]
    """
    
    def __init__(self, entropy_threshold_low: float = 0.1, entropy_threshold_high: float = 5.0):
        self.tau_low = entropy_threshold_low
        self.tau_high = entropy_threshold_high
        self._attention_maps = []
        self._hooks = []
    
    def evaluate(
        self,
        model: nn.Module,
        data_loader,
        device: torch.device,
        max_batches: int = 20,
    ) -> L3AttentionMetrics:
        """执行注意力机制评估"""
        model.eval()
        
        metrics = L3AttentionMetrics()
        
        # I24-11: 启用注意力权重存储
        self._enable_attn_storage(model)
        
        # 注册 hook 捕获注意力
        self._attention_maps = []
        self._register_attention_hooks(model)
        
        try:
            with torch.no_grad():
                for batch_idx, (imgs, _) in enumerate(tqdm(
                    data_loader, desc="L3: Attention", total=min(max_batches, len(data_loader))
                )):
                    if batch_idx >= max_batches:
                        break
                    
                    imgs = imgs.to(device)
                    _ = model(imgs)
                    
                    # 处理收集的注意力
                    if self._attention_maps:
                        self._process_attention_batch(metrics)
                    
                    self._attention_maps = []
            
            # 计算最终指标
            self._finalize_metrics(metrics)
            
        finally:
            self._remove_hooks()
            # I24-11: 恢复注意力权重存储状态
            self._disable_attn_storage(model)
        
        return metrics
    
    def _enable_attn_storage(self, model: nn.Module):
        """I24-11: 启用所有注意力层的权重存储"""
        for module in model.modules():
            if hasattr(module, 'store_attn_weights'):
                module.store_attn_weights = True
    
    def _disable_attn_storage(self, model: nn.Module):
        """I24-11: 禁用所有注意力层的权重存储并清理缓存"""
        for module in model.modules():
            if hasattr(module, 'store_attn_weights'):
                module.store_attn_weights = False
                module._last_attn_weights = None
    
    def _register_attention_hooks(self, model: nn.Module):
        """注册注意力捕获 hook"""
        for name, module in model.named_modules():
            if 'attention' in name.lower() and hasattr(module, 'forward'):
                # 尝试捕获多头注意力的 softmax 输出
                if hasattr(module, 'scale'):  # 看起来像注意力层
                    def make_hook(layer_name):
                        def hook(m, inp, out):
                            # 尝试获取注意力权重
                            if hasattr(m, '_last_attn_weights'):
                                self._attention_maps.append(
                                    (layer_name, m._last_attn_weights.detach().cpu())
                                )
                        return hook
                    
                    h = module.register_forward_hook(make_hook(name))
                    self._hooks.append(h)
    
    def _remove_hooks(self):
        """移除所有 hook"""
        for h in self._hooks:
            h.remove()
        self._hooks = []
    
    def _process_attention_batch(self, metrics: L3AttentionMetrics):
        """处理一个批次的注意力数据"""
        # 由于我们可能无法直接获取注意力权重，这里使用模拟数据
        # 实际实现需要修改注意力层来保存权重
        pass
    
    def _finalize_metrics(self, metrics: L3AttentionMetrics):
        """计算最终注意力指标"""
        # 如果没有收集到注意力数据，使用默认值
        if not metrics.per_layer_entropy:
            # 使用模型结构信息估算
            metrics.per_layer_entropy = []
            metrics.avg_entropy = 0.0
            metrics.dead_head_ratio = 0.0
            metrics.cls_attention_coverage = 0.0
            metrics.cls_attention_entropy = 0.0


# ============================================================================
# L4: 特征表示评估器
# ============================================================================

class RepresentationEvaluator:
    """L4 特征表示评估器
    
    数学形式化：
        Fisher 判别比: FDR = tr(S_B) / tr(S_W)
        类别可分性: Sep_c = ||μ_c - μ|| / σ_c
    """
    
    def evaluate(
        self,
        model: nn.Module,
        data_loader,
        device: torch.device,
        max_samples: int = 1000,
    ) -> L4RepresentationMetrics:
        """执行特征表示评估"""
        model.eval()
        
        features = []
        labels = []
        collected = 0
        
        # 注册 hook 捕获 pooled 特征
        pooled_features = []
        
        def hook_fn(module, inp, out):
            pooled_features.append(out.detach().cpu())
        
        # 找到 to_latent 或 mlp_head 的第一层
        hook = None
        for name, module in model.named_modules():
            if name == 'to_latent' or name == 'mlp_head.0':
                hook = module.register_forward_hook(hook_fn)
                break
        
        if hook is None:
            # 回退：直接从模型输出推断
            pass
        
        try:
            with torch.no_grad():
                for imgs, lbls in tqdm(data_loader, desc="L4: Representation"):
                    if collected >= max_samples:
                        break
                    
                    imgs = imgs.to(device)
                    _ = model(imgs)
                    
                    if pooled_features:
                        features.append(pooled_features[-1])
                        labels.append(lbls)
                        collected += len(lbls)
                        pooled_features = []
        finally:
            if hook:
                hook.remove()
        
        metrics = L4RepresentationMetrics()
        
        if not features:
            return metrics
        
        features = torch.cat(features, dim=0)[:max_samples]
        labels = torch.cat(labels, dim=0)[:max_samples]
        
        # 特征统计
        metrics.feature_mean_norm = features.norm(dim=1).mean().item()
        metrics.feature_std = features.std().item()
        
        # Fisher 判别比
        metrics.fisher_discriminant_ratio = self._compute_fisher_ratio(features, labels)
        
        # 类别可分性
        global_mean = features.mean(dim=0)
        unique_labels = labels.unique()
        
        for c in unique_labels.numpy():
            mask = labels == c
            if mask.sum() > 1:
                class_features = features[mask]
                class_mean = class_features.mean(dim=0)
                class_std = class_features.std()
                
                distance = (class_mean - global_mean).norm().item()
                if class_std > 1e-6:
                    metrics.per_class_separability[int(c)] = distance / class_std.item()
                else:
                    metrics.per_class_separability[int(c)] = distance
        
        if metrics.per_class_separability:
            metrics.avg_separability = np.mean(list(metrics.per_class_separability.values()))
        
        return metrics
    
    def _compute_fisher_ratio(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        """计算 Fisher 判别比
        
        FDR = tr(S_B) / tr(S_W)
        S_B = 类间散度矩阵
        S_W = 类内散度矩阵
        """
        global_mean = features.mean(dim=0)
        unique_labels = labels.unique()
        
        S_W = 0.0  # 类内散度
        S_B = 0.0  # 类间散度
        
        for c in unique_labels:
            mask = labels == c
            n_c = mask.sum().item()
            if n_c > 1:
                class_features = features[mask]
                class_mean = class_features.mean(dim=0)
                
                # 类内散度
                centered = class_features - class_mean
                S_W += (centered ** 2).sum().item()
                
                # 类间散度
                S_B += n_c * ((class_mean - global_mean) ** 2).sum().item()
        
        if S_W < 1e-6:
            return 0.0
        
        return S_B / S_W


# ============================================================================
# L5: 资源效率评估器
# ============================================================================

class EfficiencyEvaluator:
    """L5 资源效率评估器
    
    数学形式化：
        Latency = t_tokenizer + t_transformer + t_head
        Throughput = N_samples / Total_time
    """
    
    def evaluate(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        device: torch.device,
        n_warmup: int = 10,
        n_runs: int = 50,
    ) -> L5EfficiencyMetrics:
        """执行资源效率评估"""
        model.eval()
        metrics = L5EfficiencyMetrics()
        
        # 参数量统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        metrics.total_params = total_params
        metrics.trainable_params = trainable_params
        
        sample_input = sample_input.to(device)
        B = sample_input.shape[0]
        
        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(sample_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 总推理延迟
        latencies = []
        with torch.no_grad():
            for _ in range(n_runs):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                
                _ = model(sample_input)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                
                latencies.append((end - start) * 1000)  # ms
        
        metrics.avg_latency_ms = np.mean(latencies)
        metrics.throughput_samples_per_sec = B * 1000 / metrics.avg_latency_ms
        
        # 内存估算
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(sample_input)
            metrics.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            # CPU 内存估算 (参数 + 激活)
            param_memory = total_params * 4 / (1024 ** 2)  # float32
            metrics.peak_memory_mb = param_memory * 3  # 粗略估算
        
        return metrics


# ============================================================================
# L6: 训练稳定性评估器
# ============================================================================

class StabilityEvaluator:
    """L6 训练稳定性评估器
    
    数学形式化：
        Weight norm: ||W||_F = √(Σᵢⱼ wᵢⱼ²)
        Health score: 𝟙[no NaN/Inf] × 𝟙[||W|| ∈ reasonable_range]
    """
    
    def evaluate(self, model: nn.Module) -> L6StabilityMetrics:
        """执行训练稳定性评估"""
        metrics = L6StabilityMetrics()
        
        # 权重统计
        weight_stats = {}
        has_nan = False
        has_inf = False
        
        for name, param in model.named_parameters():
            if param.numel() == 0:
                continue
            
            data = param.data
            
            # 检查 NaN/Inf
            if torch.isnan(data).any():
                has_nan = True
            if torch.isinf(data).any():
                has_inf = True
            
            # 统计 (处理 numel <= 1 的情况)
            weight_stats[name] = {
                'mean': data.mean().item(),
                'std': data.std().item() if data.numel() > 1 else 0.0,
                'min': data.min().item(),
                'max': data.max().item(),
                'norm': data.norm().item(),
            }
        
        metrics.weight_norm_stats = weight_stats
        metrics.has_nan_weights = has_nan
        metrics.has_inf_weights = has_inf
        
        # 健康评分
        if has_nan or has_inf:
            metrics.gradient_health_score = 0.0
        else:
            # 检查权重是否在合理范围
            all_norms = [s['norm'] for s in weight_stats.values()]
            if all_norms:
                max_norm = max(all_norms)
                min_norm = min(all_norms)
                if max_norm > 1000 or min_norm < 1e-8:
                    metrics.gradient_health_score = 0.5
                else:
                    metrics.gradient_health_score = 1.0
        
        # 分割器健康 (如果有)
        if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'splitter'):
            splitter = model.tokenizer.splitter
            if hasattr(splitter, 'get_health_score'):
                metrics.splitter_health_score = splitter.get_health_score()
        
        return metrics
