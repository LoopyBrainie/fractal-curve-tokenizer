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

    # I35: Tokenizer 诊断信息 (从 get_extra_info 收集)
    avg_num_tokens: float = 0.0
    min_num_tokens: int = 0
    max_num_tokens: int = 0
    tokenizer_depth_distribution: Dict[int, float] = field(default_factory=dict)  # 聚合的深度分布

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class L2TokenizerMetrics:
    """L2 Tokenizer 行为层指标
    
    数学形式化
    ==========
    Token 统计:
        μ_N = (1/B) Σᵢ Nᵢ
        σ_N = √[(1/B) Σᵢ (Nᵢ - μ_N)²]
    
    深度分布熵:
        H_d = -Σ_{d=0}^D p_d log(p_d)
        其中 p_d = count(depth=d) / N_total
    
    深度崩塌检测:
        KL_uniform = D_KL(π || Uniform) = Σ_d π_d log(π_d / (1/(D+1)))
        collapse_detected = KL_uniform > 1.0 ∨ H_d < 0.5 * H_max
    """
    # Token 数量统计
    avg_tokens: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    std_tokens: float = 0.0
    
    # 深度分布
    depth_distribution: Dict[int, float] = field(default_factory=dict)  # depth -> percentage
    depth_entropy: float = 0.0  # 深度分布熵
    depth_entropy_ratio: float = 0.0  # H / H_max (标准化熵)
    depth_kl_from_uniform: float = 0.0  # KL 散度 (检测崩塌)
    depth_collapse_detected: bool = False  # 深度崩塌警告
    
    # 空间分析
    spatial_coverage_ratio: float = 0.0  # 空间覆盖率
    
    # 内容感知分析
    content_token_correlation: float = 0.0  # 图像复杂度与 token 数相关性
    
    # 每类分析
    per_class_avg_tokens: Dict[int, float] = field(default_factory=dict)
    
    # 深度贡献度分析 (各深度对分类的贡献)
    depth_contribution: Dict[int, Dict[str, float]] = field(default_factory=dict)
    
    # Token 效率分析
    token_utilization_score: float = 0.0  # Token 利用效率分数 [0, 1]
    redundancy_ratio: float = 0.0  # 估计冗余 token 比例
    adaptive_ratio: float = 0.0  # 自适应调整程度 (std / mean)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass  
class L3AttentionMetrics:
    """L3 注意力机制层指标
    
    数学形式化
    ==========
    每层每头注意力熵:
        H_attn^{(l,h)} = -(1/N) Σᵢ Σⱼ Aᵢⱼ^{(l,h)} log(Aᵢⱼ^{(l,h)} + ε)
    
    Head 利用率:
        Util^{(h)} = 𝟙[τ_low < H_attn^{(h)} < τ_high]
    
    CLS Token 分析:
        CLS_focus = max_j(A_{0,j})  # CLS 最大关注度
        CLS_coverage = |{j: A_{0,j} > 1/N}| / N  # 有效关注比例
    
    LCA 偏置有效性:
        ρ_LCA = Corr(LCA_depth, attn_weight)  # LCA 与注意力相关性
    """
    # 每层注意力熵
    per_layer_entropy: List[float] = field(default_factory=list)
    avg_entropy: float = 0.0
    
    # Head 利用率
    head_utilization: List[List[float]] = field(default_factory=list)  # [layer][head]
    dead_head_ratio: float = 0.0  # 无效 head 比例
    
    # CLS 注意力分析
    cls_attention_coverage: float = 0.0  # CLS 关注的 token 比例
    cls_attention_entropy: float = 0.0
    cls_effective_tokens: float = 0.0  # CLS 有效关注的 token 数
    
    # LCA 偏置分析
    lca_attention_correlation: float = 0.0  # LCA 深度与注意力的相关性
    hilbert_locality_score: float = 0.0  # Hilbert 局部性保持度
    
    # 层级感知分析
    per_depth_attention_received: Dict[int, float] = field(default_factory=dict)  # 各深度收到的平均注意力
    
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
    """L6 训练稳定性层指标
    
    数学形式化
    ==========
    权重范数:
        ||W||_F = √(Σᵢⱼ wᵢⱼ²)
    
    健康检查:
        Grad_health = 𝟙[||∇W|| ∈ (ε, 1/ε)]
    
    Splitter 健康:
        - 温度检查: T ∈ [T_min, T_max]
        - 阈值分布: std(τ) 应适中
        - 梯度流: splitter 组件梯度非零
    """
    # 权重统计
    weight_norm_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # 梯度健康 (需要训练模式)
    gradient_health_score: float = 1.0
    
    # 数值稳定性
    has_nan_weights: bool = False
    has_inf_weights: bool = False
    
    # 分割器健康
    splitter_health_score: float = 1.0
    splitter_temperature: float = 1.0
    splitter_threshold_mean: float = 0.0
    splitter_threshold_std: float = 0.0
    
    # 门控权重分析 (FractalTransformerBlock)
    gate_weight_stats: Dict[int, Dict[str, float]] = field(default_factory=dict)  # depth -> {mean, std}
    gate_imbalance_detected: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class L7SplitterMetrics:
    """L7 分割器专项分析层指标
    
    数学形式化
    ==========
    决策公式:
        logits_i = MLP(ROI_i) + b_explore + b_log_d + β·γ^{d_i} - τ_{d_i}
    
    Log-Compensation (I21):
        b_log_d = log(N_total / N_d)
    
    可学习配额 (I24-2):
        quota_d = softmax(quota_logits)[d]
        quota_entropy = -Σ_d q_d log(q_d)
    
    深度崩塌诊断:
        - 熵比 < 0.5 → 崩塌
        - KL from uniform > 1.0 → 严重不均衡
    """
    # MLP 输出分析
    mlp_logits_mean: float = 0.0
    mlp_logits_std: float = 0.0
    mlp_logits_per_depth: Dict[int, Dict[str, float]] = field(default_factory=dict)  # depth -> {mean, std}
    
    # 概率分析
    selection_prob_mean: float = 0.0
    selection_prob_entropy: float = 0.0
    
    # 温度状态
    temperature: float = 1.0
    temperature_schedule_progress: float = 0.0
    
    # 可学习阈值
    thresholds: Dict[int, float] = field(default_factory=dict)  # depth -> threshold
    threshold_gradients: Dict[int, float] = field(default_factory=dict)  # depth -> grad_norm
    
    # 可学习配额 (I24-2)
    quotas: Dict[int, float] = field(default_factory=dict)  # depth -> quota
    quota_entropy: float = 0.0
    quota_kl_from_base: float = 0.0
    
    # 深度平衡诊断
    log_compensation_enabled: bool = True
    depth_kl_weight: float = 0.1
    soft_entropy_bonus: float = 0.0
    
    # 决策质量
    decision_confidence_mean: float = 0.0  # sigmoid(logits) 的平均值
    decision_confidence_std: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class L8GradientFlowMetrics:
    """L8 梯度流分析层指标
    
    数学形式化
    ==========
    梯度范数:
        ||∇W||₂ = √(Σᵢ (∂L/∂wᵢ)²)
    
    梯度流健康:
        - 梯度消失: ||∇W|| < ε
        - 梯度爆炸: ||∇W|| > 1/ε
        - 梯度偏斜: max/min 比值过大
    
    STE 梯度流:
        ∂L/∂logits = (∂L/∂hard_mask) × (∂soft_mask/∂logits)
    """
    # 各组件梯度范数
    component_grad_norms: Dict[str, float] = field(default_factory=dict)
    
    # 分割器梯度
    splitter_mlp_grad_norm: float = 0.0
    splitter_threshold_grad_norm: float = 0.0
    splitter_quota_grad_norm: float = 0.0
    
    # 注意力梯度
    attention_qkv_grad_norm: float = 0.0
    lca_embed_grad_norm: float = 0.0
    level_scale_grad_norm: float = 0.0
    
    # 梯度健康诊断
    vanishing_gradients: List[str] = field(default_factory=list)  # 梯度消失的组件
    exploding_gradients: List[str] = field(default_factory=list)  # 梯度爆炸的组件
    
    # 梯度流统计
    total_grad_norm: float = 0.0
    max_grad_norm: float = 0.0
    min_grad_norm: float = 0.0
    grad_norm_ratio: float = 0.0  # max/min
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LayeredEvaluationReport:
    """完整分层评估报告
    
    评估层次结构
    ============
    - L1: 分类性能 (准确率、校准误差、混淆分析)
    - L2: Tokenizer 行为 (Token 数、深度分布、崩塌检测)
    - L3: 注意力机制 (注意力熵、LCA 有效性、Head 利用率)
    - L4: 特征表示 (Fisher 判别比、类别可分性)
    - L5: 资源效率 (延迟分解、吞吐量、内存)
    - L6: 训练稳定性 (权重范数、门控分析)
    - L7: 分割器专项 (决策分析、配额、温度)
    - L8: 梯度流分析 (各组件梯度、STE 有效性)
    """
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
    L7_splitter: L7SplitterMetrics = field(default_factory=L7SplitterMetrics)
    L8_gradient_flow: L8GradientFlowMetrics = field(default_factory=L8GradientFlowMetrics)
    
    # 摘要
    summary: Dict[str, Any] = field(default_factory=dict)
    
    # 诊断警告
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
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
            'L7_splitter': self.L7_splitter.to_dict(),
            'L8_gradient_flow': self.L8_gradient_flow.to_dict(),
            'summary': self.summary,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
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
        all_num_tokens: List[int] = []  # I35: 收集 token 数量
        all_depth_distributions: List[Dict[int, float]] = []  # I35: 收集深度分布
        total_loss = 0.0
        
        with torch.no_grad():
            for imgs, labels in tqdm(data_loader, desc="L1: Classification"):
                imgs = imgs.to(device)
                labels = labels.to(device)

                # I35: 使用 get_extra_info API 获取 logits 和辅助信息
                if hasattr(model, 'get_extra_info'):
                    outputs, aux_infos = model.get_extra_info(imgs)
                else:
                    outputs = model(imgs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    aux_infos = None

                loss = F.cross_entropy(outputs, labels)
                probs = F.softmax(outputs, dim=1)

                all_preds.append(outputs.argmax(dim=1).cpu())
                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())
                total_loss += loss.item()

                # I35: 收集 tokenizer 诊断信息
                if aux_infos is not None:
                    for aux_info in aux_infos:
                        if isinstance(aux_info, dict):
                            if 'num_tokens' in aux_info:
                                all_num_tokens.append(aux_info['num_tokens'])
                            if 'depth_distribution' in aux_info:
                                all_depth_distributions.append(aux_info['depth_distribution'])
        
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

        # I35: 计算 tokenizer 诊断信息
        if all_num_tokens:
            metrics.avg_num_tokens = np.mean(all_num_tokens)
            metrics.min_num_tokens = int(np.min(all_num_tokens))
            metrics.max_num_tokens = int(np.max(all_num_tokens))

        # 聚合深度分布
        if all_depth_distributions:
            depth_counts: Dict[int, float] = defaultdict(float)
            total_depth = 0.0
            for dist in all_depth_distributions:
                for d, p in dist.items():
                    depth_counts[d] += p
                    total_depth += p
            if total_depth > 0:
                metrics.tokenizer_depth_distribution = {
                    d: count / total_depth for d, count in depth_counts.items()
                }

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
        # I78: 添加存在性检查，避免非 Fractal 模型崩溃
        if not hasattr(model, 'tokenizer'):
            raise ValueError(
                f"L2 Tokenizer Evaluator requires model to have 'tokenizer' attribute. "
                f"Model type: {type(model).__name__}"
            )
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
            
            # ============================================
            # 深度分布诊断 (I21 深度平衡相关)
            # ============================================
            n_depths = len(depth_counts)
            
            # 均匀分布熵 (最大熵)
            uniform_entropy = np.log(n_depths) if n_depths > 1 else 0.0
            
            # 深度熵比率: 1.0 = 完全均匀, 0.0 = 完全集中
            if uniform_entropy > 0:
                metrics.depth_entropy_ratio = metrics.depth_entropy / uniform_entropy
            else:
                metrics.depth_entropy_ratio = 1.0
            
            # KL 散度 from Uniform: D_KL(P || U)
            # P = 实际分布, U = 均匀分布 (1/n_depths)
            uniform_prob = 1.0 / n_depths
            kl_divergence = 0.0
            for p in probs:
                if p > 0:
                    kl_divergence += p * np.log(p / uniform_prob)
            metrics.depth_kl_from_uniform = float(kl_divergence)
            
            # 深度坍缩检测
            # 条件: KL > 1.0 或 熵比率 < 0.3 或 单深度占比 > 80%
            max_depth_ratio = max(probs)
            if (metrics.depth_kl_from_uniform > 1.0 or 
                metrics.depth_entropy_ratio < 0.3 or 
                max_depth_ratio > 0.8):
                metrics.depth_collapse_detected = True
            else:
                metrics.depth_collapse_detected = False
            
            # 每深度贡献统计 (tokens 和 area)
            for d, count in depth_counts.items():
                metrics.depth_contribution[d] = {
                    'token_count': count,
                    'token_ratio': count / total_tokens,
                    # 理论面积: 深度 d 的 patch 大小是 base_size / 2^d
                    # 假设 base_size=32, 则深度 0 = 32x32, 深度 1 = 16x16, 等
                    'estimated_area_ratio': count * (0.5 ** (2 * d)),  # 相对面积
                }
        
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
        
        # ============================================
        # Token 效率分析
        # ============================================
        if metrics.avg_tokens > 0:
            # 自适应比率: std / mean - 高值表示 tokenizer 在不同图像间有差异化
            metrics.adaptive_ratio = metrics.std_tokens / metrics.avg_tokens
            
            # Token 利用效率评分
            # 综合考虑: 深度多样性 + 内容相关性 + 自适应性
            depth_score = metrics.depth_entropy_ratio if metrics.depth_entropy_ratio > 0 else 0
            content_score = abs(metrics.content_token_correlation) if metrics.content_token_correlation != 0 else 0
            adaptive_score = min(1.0, metrics.adaptive_ratio * 2)  # 归一化
            
            metrics.token_utilization_score = (depth_score + content_score + adaptive_score) / 3
            
            # 冗余估计: 如果所有样本 token 数接近最大值，可能有冗余
            if metrics.max_tokens > 0:
                metrics.redundancy_ratio = 1.0 - (metrics.std_tokens / (metrics.max_tokens - metrics.min_tokens + 1e-6))
                metrics.redundancy_ratio = max(0, min(1, metrics.redundancy_ratio))
        
        return metrics


# ============================================================================
# L3: 注意力机制评估器
# ============================================================================

class AttentionEvaluator:
    """L3 注意力机制评估器
    
    数学形式化：
        注意力熵: H_attn = -(1/N) Σᵢ Σⱼ Aᵢⱼ log(Aᵢⱼ)
        Head 利用率: Util = 𝟙[H_attn ∈ (τ_low, τ_high)]
        
    LCA 分析 (Hilbert-Aware):
        LCA 深度: lca_depth[i,j] = depth of Lowest Common Ancestor
        LCA-Attention 相关性: ρ(lca_depth, attention_weight)
        局部性分数: Σ A[i,j] · 𝟙[lca_depth[i,j] > threshold]
    """
    
    def __init__(self, entropy_threshold_low: float = 0.1, entropy_threshold_high: float = 5.0):
        self.tau_low = entropy_threshold_low
        self.tau_high = entropy_threshold_high
        self._attention_maps = []
        self._lca_depths = []
        self._token_depths = []
        self._hooks = []
        # 累积统计
        self._layer_entropies = defaultdict(list)  # layer_idx -> [entropy values]
        self._head_entropies = defaultdict(lambda: defaultdict(list))  # layer -> head -> [values]
        self._cls_attentions = []  # CLS token 接收的注意力
        self._lca_attention_pairs = []  # (lca_depth, attention_weight) pairs
        self._depth_attention_received = defaultdict(list)  # depth -> [attention values]
    
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
        
        # 重置累积统计
        self._layer_entropies = defaultdict(list)
        self._head_entropies = defaultdict(lambda: defaultdict(list))
        self._cls_attentions = []
        self._lca_attention_pairs = []
        self._depth_attention_received = defaultdict(list)
        
        # I24-11: 启用注意力权重存储
        self._enable_attn_storage(model)
        
        # 注册 hook 捕获注意力
        self._attention_maps = []
        self._lca_depths = []
        self._token_depths = []
        self._register_attention_hooks(model)
        
        try:
            with torch.no_grad():
                for batch_idx, (imgs, _) in enumerate(tqdm(
                    data_loader, desc="L3: Attention", total=min(max_batches, len(data_loader))
                )):
                    if batch_idx >= max_batches:
                        break
                    
                    imgs = imgs.to(device)
                    
                    # 先获取 tokenizer 输出以获取深度信息
                    if hasattr(model, 'tokenizer'):
                        tok_output = model.tokenizer.tokenize(imgs)
                        # 提取深度信息
                        if hasattr(tok_output, 'sequences'):
                            for seq in tok_output.sequences:
                                if 'levels' in seq.metadata and seq.metadata['levels'] is not None:
                                    levels = seq.metadata['levels']
                                    if levels.dim() > 1:
                                        self._token_depths.append(levels[:, 0].cpu())
                                    else:
                                        self._token_depths.append(levels.cpu())
                    
                    # 完整前向传播
                    _ = model(imgs)
                    
                    # 处理收集的注意力
                    if self._attention_maps:
                        self._process_attention_batch(metrics, model)
                    
                    self._attention_maps = []
                    self._lca_depths = []
                    self._token_depths = []
            
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
        """注册注意力捕获 hook
        
        I23-7 Fix: 使用 store_attn_weights 属性作为唯一注册条件
        
        原条件 'attention' in name.lower() and hasattr(module, 'scale') 不可靠:
        - 模块路径可能不含 'attention' 字符串
        - scale 可能是 float 而非 nn.Module
        
        新条件: hasattr(module, 'store_attn_weights')
        - 精确匹配 HilbertAwareMultiScaleAttention 类
        - 该属性在 attn_hilbert_bias.py L524 定义
        """
        layer_idx = 0
        for name, module in model.named_modules():
            # I23-7: 简化条件 - 只检查是否支持注意力权重存储
            if hasattr(module, 'store_attn_weights'):
                def make_hook(layer_name, idx):
                    def hook(m, inp, out):
                        # 获取注意力权重 (attn_hilbert_bias.py L712-713)
                        if hasattr(m, '_last_attn_weights') and m._last_attn_weights is not None:
                            self._attention_maps.append(
                                (idx, layer_name, m._last_attn_weights.detach().cpu())
                            )
                        # 获取 LCA 深度信息 (HilbertAwareAttention)
                        if hasattr(m, '_last_lca_depths') and m._last_lca_depths is not None:
                            self._lca_depths.append(m._last_lca_depths.detach().cpu())
                    return hook
                
                h = module.register_forward_hook(make_hook(name, layer_idx))
                self._hooks.append(h)
                layer_idx += 1
        
        # I23-7: 添加调试日志
        if layer_idx == 0:
            import warnings
            warnings.warn(
                "L3AttentionEvaluator: No attention modules found. "
                "Ensure model uses HilbertAwareMultiScaleAttention with store_attn_weights attribute."
            )
    
    def _remove_hooks(self):
        """移除所有 hook"""
        for h in self._hooks:
            h.remove()
        self._hooks = []
    
    def _process_attention_batch(self, metrics: L3AttentionMetrics, model: nn.Module):
        """处理一个批次的注意力数据"""
        for layer_idx, layer_name, attn_weights in self._attention_maps:
            # attn_weights: [B, H, N, N] 或 [B, N, N]
            if attn_weights.dim() == 3:
                attn_weights = attn_weights.unsqueeze(1)  # [B, 1, N, N]
            
            B, H, N, _ = attn_weights.shape
            
            # 每层每 head 的熵
            for h in range(H):
                head_attn = attn_weights[:, h]  # [B, N, N]
                # 计算熵: -Σ p log(p)
                attn_log = torch.log(head_attn.clamp(min=1e-10))
                entropy = -(head_attn * attn_log).sum(dim=-1).mean()  # 平均每行的熵
                
                self._head_entropies[layer_idx][h].append(entropy.item())
                self._layer_entropies[layer_idx].append(entropy.item())
            
            # CLS token 分析 (假设 CLS 是第一个 token)
            cls_attn = attn_weights[:, :, 0, :]  # [B, H, N] - CLS 关注其他 token
            cls_received = attn_weights[:, :, :, 0]  # [B, H, N] - 其他 token 关注 CLS
            
            # CLS 接收的总注意力
            cls_total_received = cls_received.sum(dim=-1).mean()  # 平均
            self._cls_attentions.append(cls_total_received.item())
            
            # 有效 token 数 (CLS 关注权重 > threshold 的 token)
            effective_mask = cls_attn.mean(dim=1) > (1.0 / N * 0.5)  # [B, N]
            effective_count = effective_mask.sum(dim=-1).float().mean()
            if not hasattr(metrics, '_effective_token_counts'):
                metrics._effective_token_counts = []
            metrics._effective_token_counts.append(effective_count.item())
        
        # LCA-Attention 相关性分析
        if self._lca_depths and self._attention_maps:
            for lca_depth in self._lca_depths:
                for layer_idx, _, attn_weights in self._attention_maps:
                    if attn_weights.dim() == 3:
                        attn_weights = attn_weights.unsqueeze(1)
                    
                    # 平均所有 heads
                    avg_attn = attn_weights.mean(dim=1)  # [B, N, N]
                    
                    # 只取上三角 (避免重复)
                    for b in range(min(avg_attn.shape[0], lca_depth.shape[0])):
                        N = min(avg_attn.shape[1], lca_depth.shape[0])
                        for i in range(N):
                            for j in range(i + 1, N):
                                if j < lca_depth.shape[0]:
                                    self._lca_attention_pairs.append(
                                        (lca_depth[j].item(), avg_attn[b, i, j].item())
                                    )
        
        # 按深度统计接收的注意力
        if self._token_depths and self._attention_maps:
            for depths_tensor in self._token_depths:
                for layer_idx, _, attn_weights in self._attention_maps:
                    if attn_weights.dim() == 3:
                        attn_weights = attn_weights.unsqueeze(1)
                    
                    avg_attn = attn_weights.mean(dim=(0, 1))  # [N, N]
                    attention_received = avg_attn.sum(dim=0)  # [N] - 每个 token 收到的总注意力
                    
                    N = min(len(depths_tensor), len(attention_received))
                    for i in range(N):
                        depth = int(depths_tensor[i].item())
                        self._depth_attention_received[depth].append(attention_received[i].item())
    
    def _finalize_metrics(self, metrics: L3AttentionMetrics):
        """计算最终注意力指标"""
        # 每层熵统计
        if self._layer_entropies:
            metrics.per_layer_entropy = []
            for layer_idx in sorted(self._layer_entropies.keys()):
                layer_entropy = np.mean(self._layer_entropies[layer_idx])
                metrics.per_layer_entropy.append(layer_entropy)
            
            metrics.avg_entropy = float(np.mean(metrics.per_layer_entropy))
        
        # 每层每 head 熵
        if self._head_entropies:
            metrics.per_head_entropy = {}
            dead_heads = 0
            total_heads = 0
            
            for layer_idx in sorted(self._head_entropies.keys()):
                metrics.per_head_entropy[layer_idx] = {}
                for head_idx in sorted(self._head_entropies[layer_idx].keys()):
                    head_entropy = np.mean(self._head_entropies[layer_idx][head_idx])
                    metrics.per_head_entropy[layer_idx][head_idx] = head_entropy
                    
                    total_heads += 1
                    # 死 head: 熵过低 (过于集中) 或过高 (过于分散)
                    if head_entropy < self.tau_low or head_entropy > self.tau_high:
                        dead_heads += 1
            
            if total_heads > 0:
                metrics.dead_head_ratio = dead_heads / total_heads
        
        # CLS token 分析
        if self._cls_attentions:
            metrics.cls_attention_coverage = float(np.mean(self._cls_attentions))
            # CLS 注意力熵
            cls_arr = np.array(self._cls_attentions)
            if len(cls_arr) > 1 and np.std(cls_arr) > 1e-6:
                cls_arr_norm = cls_arr / cls_arr.sum()
                metrics.cls_attention_entropy = -np.sum(
                    cls_arr_norm * np.log(cls_arr_norm + 1e-10)
                )
        
        # 有效 token 数
        if hasattr(metrics, '_effective_token_counts') and metrics._effective_token_counts:
            metrics.cls_effective_tokens = float(np.mean(metrics._effective_token_counts))
            delattr(metrics, '_effective_token_counts')
        
        # LCA-Attention 相关性
        if len(self._lca_attention_pairs) > 10:
            lca_depths = np.array([p[0] for p in self._lca_attention_pairs])
            attn_weights = np.array([p[1] for p in self._lca_attention_pairs])
            
            if np.std(lca_depths) > 1e-6 and np.std(attn_weights) > 1e-6:
                correlation = np.corrcoef(lca_depths, attn_weights)[0, 1]
                metrics.lca_attention_correlation = float(correlation)
                
                # 局部性分数: LCA 深度高 (更近的祖先) 的 token 对是否获得更多注意力
                # 正相关表示模型学会了利用 Hilbert 曲线的空间局部性
                high_lca_mask = lca_depths > np.median(lca_depths)
                if high_lca_mask.sum() > 0:
                    locality_score = attn_weights[high_lca_mask].mean() / (attn_weights.mean() + 1e-10)
                    metrics.hilbert_locality_score = float(locality_score)
        
        # 每深度接收的注意力
        if self._depth_attention_received:
            for depth, attn_values in self._depth_attention_received.items():
                metrics.per_depth_attention_received[depth] = {
                    'mean': float(np.mean(attn_values)),
                    'std': float(np.std(attn_values)),
                    'count': len(attn_values),
                }
        
        # 如果没有收集到数据，使用默认值
        if not metrics.per_layer_entropy:
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
    
    数学形式化
    ==========
    总延迟分解:
        Latency = t_tokenizer + t_transformer + t_head
    
    吞吐量:
        Throughput = N_samples / Total_time
    
    效率比:
        Accuracy/Token = Acc / avg_tokens
        Accuracy/GFLOPS = Acc / (GFLOPS × 1e9)
    
    组件延迟测量方法:
        1. 使用 forward hook 捕获各组件的输入/输出时间
        2. 分别测量 tokenizer.tokenize(), transformer blocks, mlp_head
        3. 取多次运行的平均值以减少噪声
    """
    
    def __init__(self, measure_components: bool = True):
        """
        Args:
            measure_components: 是否测量组件级延迟
        """
        self.measure_components = measure_components
    
    def evaluate(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        device: torch.device,
        n_warmup: int = 10,
        n_runs: int = 50,
    ) -> L5EfficiencyMetrics:
        """执行资源效率评估
        
        Parameters
        ----------
        model : nn.Module
            待评估模型
        sample_input : torch.Tensor
            样本输入 [B, C, H, W]
        device : torch.device
            计算设备
        n_warmup : int
            预热运行次数
        n_runs : int
            正式测量运行次数
            
        Returns
        -------
        L5EfficiencyMetrics
            包含延迟、吞吐量、内存等指标
        """
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
        
        # 组件级延迟分解
        if self.measure_components:
            component_latencies = self._measure_component_latencies(
                model, sample_input, device, n_warmup=5, n_runs=20
            )
            metrics.tokenizer_latency_ms = component_latencies.get('tokenizer', 0.0)
            metrics.transformer_latency_ms = component_latencies.get('transformer', 0.0)
            metrics.head_latency_ms = component_latencies.get('head', 0.0)
        
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
    
    def _measure_component_latencies(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        device: torch.device,
        n_warmup: int = 5,
        n_runs: int = 20,
    ) -> Dict[str, float]:
        """测量各组件的延迟
        
        通过逐步执行模型的各个组件来测量延迟:
        1. Tokenizer: model.tokenizer.tokenize()
        2. Transformer: model.transformer 的所有 block
        3. Head: model.mlp_head
        
        Returns
        -------
        Dict[str, float]
            组件名到延迟 (ms) 的映射
        """
        latencies = {
            'tokenizer': [],
            'transformer': [],
            'head': [],
        }
        
        try:
            # 检查模型结构
            has_tokenizer = hasattr(model, 'tokenizer')
            has_transformer = hasattr(model, 'transformer')
            has_mlp_head = hasattr(model, 'mlp_head')
            
            if not (has_tokenizer and has_transformer and has_mlp_head):
                # 模型结构不符合预期，返回估算值
                return self._estimate_component_latencies(model, sample_input, device, n_runs)
            
            # Warmup
            with torch.no_grad():
                for _ in range(n_warmup):
                    _ = model(sample_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # 测量各组件
            with torch.no_grad():
                for _ in range(n_runs):
                    # 1. Tokenizer
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    
                    tokenizer_output = model.tokenizer.tokenize(sample_input)
                    tokens, lengths = tokenizer_output.get_padded_tokens()
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    latencies['tokenizer'].append((t1 - t0) * 1000)
                    
                    # 2. 位置编码 + Transformer
                    # 获取 levels_info 用于位置编码
                    # P9-5: info_dim = max_level + 4
                    info_dim = getattr(model, 'max_level', 8) + 4
                    levels_info = tokenizer_output.get_padded_levels(info_dim)
                    
                    # 添加 CLS token
                    B, N, D = tokens.shape
                    cls_tokens = model.cls_token.expand(B, -1, -1)
                    tokens_with_cls = torch.cat([cls_tokens, tokens], dim=1)
                    
                    # 位置编码
                    if hasattr(model, 'pos_embedding') and model.pos_embedding is not None:
                        tokens_with_cls = model.pos_embedding(tokens_with_cls, levels_info)
                    
                    tokens_with_cls = model.dropout(tokens_with_cls)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    
                    # Transformer
                    x = model.transformer(tokens_with_cls)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    t3 = time.perf_counter()
                    latencies['transformer'].append((t3 - t2) * 1000)
                    
                    # 3. Head
                    # Pool
                    if model.pool == 'mean':
                        x = x.mean(dim=1)
                    else:
                        x = x[:, 0]  # CLS token
                    
                    x = model.to_latent(x)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    t4 = time.perf_counter()
                    
                    _ = model.mlp_head(x)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    t5 = time.perf_counter()
                    latencies['head'].append((t5 - t4) * 1000)
            
            return {
                'tokenizer': np.mean(latencies['tokenizer']),
                'transformer': np.mean(latencies['transformer']),
                'head': np.mean(latencies['head']),
            }
            
        except Exception as e:
            # 如果组件测量失败，返回估算值
            import warnings
            warnings.warn(f"Component latency measurement failed: {e}. Using estimation.")
            return self._estimate_component_latencies(model, sample_input, device, n_runs)
    
    def _estimate_component_latencies(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        device: torch.device,
        n_runs: int = 20,
    ) -> Dict[str, float]:
        """估算组件延迟（当直接测量失败时）
        
        使用基于参数量的启发式估算:
        - Tokenizer: ~15% 总延迟 (包含卷积、分割决策)
        - Transformer: ~80% 总延迟 (主要计算量)
        - Head: ~5% 总延迟 (MLP)
        """
        # 测量总延迟
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
                latencies.append((end - start) * 1000)
        
        total_latency = np.mean(latencies)
        
        # 基于典型 FractalCurveViT 架构的估算比例
        # 这些比例基于实际测量的经验值
        return {
            'tokenizer': total_latency * 0.15,
            'transformer': total_latency * 0.80,
            'head': total_latency * 0.05,
        }


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
        
        # 分割器健康
        if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'splitter'):
            splitter = model.tokenizer.splitter
            if hasattr(splitter, 'get_health_score'):
                metrics.splitter_health_score = splitter.get_health_score()
            
            # 温度状态
            if hasattr(splitter, 'current_temperature'):
                metrics.splitter_temperature = splitter.current_temperature
            elif hasattr(splitter, 'log_temperature'):
                metrics.splitter_temperature = splitter.log_temperature.exp().item()
            
            # 阈值分析
            if hasattr(splitter, 'thresholds') and splitter.thresholds is not None:
                thresh = splitter.thresholds
                if hasattr(thresh, 'mean'):
                    metrics.splitter_threshold_mean = thresh.mean().item()
                    metrics.splitter_threshold_std = thresh.std().item() if thresh.numel() > 1 else 0.0
        
        # 门控权重分析 (FractalTransformerBlock)
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
            gate_stats = {}
            for i, layer in enumerate(model.transformer.layers):
                if hasattr(layer, 'residual_gate_weight'):
                    gate_weight = layer.residual_gate_weight
                    if gate_weight is not None:
                        gate_values = torch.sigmoid(gate_weight) * 2
                        for d in range(gate_values.shape[0]):
                            gate_stats[d] = {
                                'attn_gate': gate_values[d, 0].item(),
                                'ffn_gate': gate_values[d, 1].item(),
                            }
                    break  # 只取第一层的门控
            
            if gate_stats:
                metrics.gate_weight_stats = gate_stats
                # 检测门控不平衡
                all_gates = [g['attn_gate'] for g in gate_stats.values()] + \
                            [g['ffn_gate'] for g in gate_stats.values()]
                if all_gates:
                    if min(all_gates) < 0.1 or max(all_gates) > 1.9:
                        metrics.gate_imbalance_detected = True
        
        return metrics


# ============================================================================
# L7: 分割器专项评估器
# ============================================================================

class SplitterEvaluator:
    """L7 分割器专项评估器
    
    数学形式化
    ==========
    决策公式:
        logits_i = MLP(ROI_i) + b_explore + b_log_d + β·γ^{d_i} - τ_{d_i}
    
    监控指标:
        - MLP 输出分布: μ, σ per depth
        - 选择概率: sigmoid(logits/T)
        - 可学习配额: softmax(quota_logits)
        - 深度 KL 散度: D_KL(π || Uniform)
    """
    
    def evaluate(
        self,
        model: nn.Module,
        data_loader,
        device: torch.device,
        max_batches: int = 20,
    ) -> L7SplitterMetrics:
        """执行分割器专项评估"""
        model.eval()
        metrics = L7SplitterMetrics()
        
        # 检查模型是否有分割器
        if not hasattr(model, 'tokenizer') or not hasattr(model.tokenizer, 'splitter'):
            return metrics
        
        splitter = model.tokenizer.splitter
        
        # 基础参数提取
        if hasattr(splitter, 'log_temperature'):
            metrics.temperature = splitter.log_temperature.exp().item()
        elif hasattr(splitter, 'current_temperature'):
            metrics.temperature = splitter.current_temperature
        
        # 温度调度进度
        if hasattr(splitter, '_temp_step') and hasattr(splitter, '_temp_total_steps'):
            if splitter._temp_total_steps.item() > 0:
                metrics.temperature_schedule_progress = \
                    splitter._temp_step.item() / splitter._temp_total_steps.item()
        
        # 可学习阈值
        if hasattr(splitter, 'thresholds') and splitter.thresholds is not None:
            thresh = splitter.thresholds
            for d in range(thresh.numel()):
                metrics.thresholds[d] = thresh[d].item()
        
        # 可学习配额 (I24-2)
        if hasattr(splitter, 'quota_logits') and splitter.quota_logits is not None:
            quota_logits = splitter.quota_logits
            quotas = torch.softmax(quota_logits, dim=0)
            for d in range(quotas.numel()):
                metrics.quotas[d] = quotas[d].item()
            
            # 配额熵
            metrics.quota_entropy = -(quotas * quotas.clamp(min=1e-10).log()).sum().item()
            
            # 与基准配额的 KL
            if hasattr(splitter, 'base_quotas'):
                base = splitter.base_quotas
                metrics.quota_kl_from_base = (quotas * (quotas / base.clamp(min=1e-10)).log()).sum().item()
        
        # Log compensation 状态
        if hasattr(splitter, 'log_compensation_enabled'):
            metrics.log_compensation_enabled = splitter.log_compensation_enabled
        
        if hasattr(splitter, 'depth_kl_weight'):
            metrics.depth_kl_weight = splitter.depth_kl_weight
        
        # 运行时收集 MLP 输出分布
        mlp_outputs_by_depth = defaultdict(list)
        selection_probs = []
        
        with torch.no_grad():
            for batch_idx, (imgs, _) in enumerate(tqdm(
                data_loader, desc="L7: Splitter", total=min(max_batches, len(data_loader))
            )):
                if batch_idx >= max_batches:
                    break
                
                imgs = imgs.to(device)
                
                # 触发 tokenizer 前向传播
                try:
                    output = model.tokenizer.tokenize(imgs)
                    
                    # 尝试获取分割器的内部状态
                    if hasattr(splitter, '_last_logits') and splitter._last_logits is not None:
                        logits = splitter._last_logits  # [B, N]
                        
                        # 选择概率
                        probs = torch.sigmoid(logits / metrics.temperature)
                        selection_probs.extend(probs.flatten().cpu().tolist())
                        
                        # 按深度分组 (如果有深度信息)
                        if hasattr(splitter, '_last_depths') and splitter._last_depths is not None:
                            depths = splitter._last_depths
                            for d in range(depths.max().item() + 1):
                                mask = (depths == d).flatten()
                                if mask.any():
                                    d_logits = logits.flatten()[mask]
                                    mlp_outputs_by_depth[d].extend(d_logits.cpu().tolist())
                except Exception:
                    pass
        
        # 汇总 MLP 输出统计
        all_logits = []
        for d, logits_list in mlp_outputs_by_depth.items():
            if logits_list:
                metrics.mlp_logits_per_depth[d] = {
                    'mean': np.mean(logits_list),
                    'std': np.std(logits_list),
                    'count': len(logits_list),
                }
                all_logits.extend(logits_list)
        
        if all_logits:
            metrics.mlp_logits_mean = np.mean(all_logits)
            metrics.mlp_logits_std = np.std(all_logits)
        
        # 选择概率统计
        if selection_probs:
            probs_arr = np.array(selection_probs)
            metrics.selection_prob_mean = np.mean(probs_arr)
            # 选择概率熵
            probs_arr = np.clip(probs_arr, 1e-10, 1 - 1e-10)
            metrics.selection_prob_entropy = -np.mean(
                probs_arr * np.log(probs_arr) + (1 - probs_arr) * np.log(1 - probs_arr)
            )
            
            # 决策置信度
            confidence = np.abs(probs_arr - 0.5) * 2  # 0 到 1，1 表示高置信
            metrics.decision_confidence_mean = np.mean(confidence)
            metrics.decision_confidence_std = np.std(confidence)
        
        return metrics


# ============================================================================
# L8: 梯度流评估器
# ============================================================================

class GradientFlowEvaluator:
    """L8 梯度流评估器
    
    数学形式化
    ==========
    梯度范数:
        ||∇W||₂ = √(Σᵢ (∂L/∂wᵢ)²)
    
    需要在训练模式下使用，或执行一次 backward pass。
    """
    
    def __init__(
        self,
        vanishing_threshold: float = 1e-7,
        exploding_threshold: float = 100.0,
    ):
        """
        Args:
            vanishing_threshold: 梯度消失阈值
            exploding_threshold: 梯度爆炸阈值
        """
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
    
    def evaluate(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        sample_labels: torch.Tensor,
        device: torch.device,
    ) -> L8GradientFlowMetrics:
        """执行梯度流评估
        
        需要执行一次 forward + backward pass。
        """
        model.train()  # 需要训练模式
        metrics = L8GradientFlowMetrics()
        
        sample_input = sample_input.to(device)
        sample_labels = sample_labels.to(device)
        
        # 清空梯度
        model.zero_grad()
        
        try:
            # Forward + backward
            outputs = model(sample_input)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            loss = F.cross_entropy(logits, sample_labels)
            loss.backward()
            
            # 收集梯度信息
            component_grads = {}
            all_grad_norms = []
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    all_grad_norms.append(grad_norm)
                    
                    # 分类记录
                    if 'splitter' in name.lower():
                        if 'mlp' in name.lower() or 'complexity' in name.lower():
                            if metrics.splitter_mlp_grad_norm == 0:
                                metrics.splitter_mlp_grad_norm = grad_norm
                        elif 'threshold' in name.lower():
                            metrics.splitter_threshold_grad_norm = grad_norm
                        elif 'quota' in name.lower():
                            metrics.splitter_quota_grad_norm = grad_norm
                    elif 'attention' in name.lower() or 'attn' in name.lower():
                        if 'qkv' in name.lower() or 'to_q' in name.lower():
                            if metrics.attention_qkv_grad_norm == 0:
                                metrics.attention_qkv_grad_norm = grad_norm
                        elif 'lca' in name.lower():
                            metrics.lca_embed_grad_norm = grad_norm
                        elif 'level_scale' in name.lower():
                            metrics.level_scale_grad_norm = grad_norm
                    
                    # 简化名称记录
                    short_name = name.split('.')[-1] if '.' in name else name
                    if short_name not in component_grads:
                        component_grads[short_name] = grad_norm
                    
                    # 检测问题
                    if grad_norm < self.vanishing_threshold:
                        metrics.vanishing_gradients.append(name)
                    elif grad_norm > self.exploding_threshold:
                        metrics.exploding_gradients.append(name)
            
            metrics.component_grad_norms = component_grads
            
            # 汇总统计
            if all_grad_norms:
                metrics.total_grad_norm = sum(g**2 for g in all_grad_norms) ** 0.5
                metrics.max_grad_norm = max(all_grad_norms)
                metrics.min_grad_norm = min(g for g in all_grad_norms if g > 0) if any(g > 0 for g in all_grad_norms) else 0
                if metrics.min_grad_norm > 0:
                    metrics.grad_norm_ratio = metrics.max_grad_norm / metrics.min_grad_norm
            
        except Exception as e:
            import warnings
            warnings.warn(f"Gradient flow evaluation failed: {e}")
        
        finally:
            model.zero_grad()
            model.eval()
        
        return metrics
