# -*- coding: utf-8 -*-
r"""Splitter 输入特征 SVD/有效秩分析模块

用于检测训练早期 Splitter 输入特征的坍塌问题。通过 SVD 分解和有效秩分析，
识别由于过度强烈的 LayerNorm 或不当初始化导致的特征同质化。

使用方式:
    from vit_pytorch.core.splitter_feature_analyzer import SplitterFeatureAnalyzer

    analyzer = SplitterFeatureAnalyzer(enabled=True)
    # 在 Splitter 的 ROI-Align 后调用
    result = analyzer.analyze(roi_features)  # [B*N, C, k, k]

    # 获取分析结果
    print(f"有效秩: {result.effective_rank}")
    print(f"第一奇异值占比: {result.svd_top1_ratio}")
    print(f"是否坍塌: {result.is_collapsed}")
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import math


@dataclass
class SVDAnalysisResult:
    """SVD 分析结果"""
    # 奇异值统计
    singular_values: torch.Tensor  # [D] 奇异值向量
    explained_variance_ratio: torch.Tensor  # [D] 归一化方差占比

    # 有效秩指标
    effective_rank: float  # 基于熵的有效秩
    normalized_entropy: float  # 归一化熵 [0, 1]

    # 能量占比
    svd_top1_ratio: float  # 第一奇异值占比
    svd_top5_ratio: float  # 前5奇异值占比
    svd_top10_ratio: float  # 前10奇异值占比
    energy_99_percent_dims: int  # 达到99%能量所需的维度数

    # 坍塌检测
    is_collapsed: bool  # 是否坍塌
    collapse_severity: str  # 坍塌严重程度: "none", "mild", "moderate", "severe"

    # 原始数据信息
    feature_dim: int  # 特征维度
    num_samples: int  # 样本数量

    # 可能原因（放在最后，因为有默认值）
    collapse_reasons: list = field(default_factory=list)  # 可能原因


class SplitterFeatureAnalyzer:
    """Splitter 输入特征分析器

    对 ROI-Align 后的特征进行 SVD 分解，计算有效秩和能量占比，
    用于检测特征坍塌问题。
    """

    # 坍塌检测阈值
    COLLAPSE_THRESHOLD_TOP1 = 0.95  # 第一奇异值占比 > 95% 视为严重
    COLLAPSE_THRESHOLD_TOP5 = 0.90  # 前5占比 > 90% 视为严重
    COLLAPSE_THRESHOLD_EFFECTIVE_RANK = 10  # 有效秩 < 10 视为坍塌

    def __init__(
        self,
        enabled: bool = True,
        sample_interval: int = 10,
        record_history: bool = True,
    ):
        """
        Args:
            enabled: 是否启用分析
            sample_interval: 每隔多少步采样一次（避免计算开销过大）
            record_history: 是否记录历史统计
        """
        self.enabled = enabled
        self.sample_interval = sample_interval
        self.record_history = record_history

        # 历史记录
        self.effective_rank_history: list[float] = []
        self.top1_ratio_history: list[float] = []
        self.collapse_warning_history: list[bool] = []
        self.step_counter = 0

        # 统计信息
        self.total_analyzes = 0

    def should_analyze(self) -> bool:
        """判断当前是否应该进行分析"""
        if not self.enabled:
            return False
        self.step_counter += 1
        return self.step_counter % self.sample_interval == 0

    @torch._dynamo.disable  # 🌟 I-OOM FIX: 禁用 Dynamo 追踪，防止图断裂导致 backward 泄漏
    def analyze(self, features: torch.Tensor) -> Optional[SVDAnalysisResult]:
        """
        分析输入特征的 SVD 分解结果

        Args:
            features: [B*N, C, k, k] ROI-Align 后的特征

        Returns:
            SVDAnalysisResult 或 None（如果禁用）
        """
        if not self.enabled:
            return None

        # 展平空间维度，只保留通道维度
        # [B*N, C, k, k] -> [B*N, C]
        if features.dim() == 4:
            features = features.mean(dim=[2, 3])  # 全局平均池化

        # 转置使每列是一个样本 [C, B*N]
        features_T = features.T.contiguous()

        # 奇异值分解
        try:
            U, S, Vt = torch.svd(features_T, compute_uv=False)
        except RuntimeError:
            # SVD 失败时返回 None
            return None

        D = S.shape[0]
        num_samples = features_T.shape[1]

        # 计算解释方差比
        variance = S ** 2
        total_variance = variance.sum()
        explained_variance_ratio = variance / (total_variance + 1e-8)

        # 有效秩计算 (基于归一化熵)
        effective_rank, normalized_entropy = self._compute_effective_rank(S)

        # 能量占比
        cumulative_energy = torch.cumsum(explained_variance_ratio, dim=0)

        top1_ratio = explained_variance_ratio[0].item()
        top5_ratio = cumulative_energy[min(4, D-1)].item()
        top10_ratio = cumulative_energy[min(9, D-1)].item()

        # 计算达到99%能量需要的维度数
        energy_99_dims = torch.searchsorted(cumulative_energy, 0.99).item() + 1
        energy_99_dims = min(energy_99_dims, D)

        # 坍塌检测
        is_collapsed, severity, reasons = self._detect_collapse(
            top1_ratio, effective_rank, energy_99_dims, D
        )

        result = SVDAnalysisResult(
            singular_values=S,
            explained_variance_ratio=explained_variance_ratio,
            effective_rank=effective_rank,
            normalized_entropy=normalized_entropy,
            svd_top1_ratio=top1_ratio,
            svd_top5_ratio=top5_ratio,
            svd_top10_ratio=top10_ratio,
            energy_99_percent_dims=energy_99_dims,
            is_collapsed=is_collapsed,
            collapse_severity=severity,
            collapse_reasons=reasons,
            feature_dim=D,
            num_samples=num_samples,
        )

        # 记录历史
        if self.record_history:
            self.effective_rank_history.append(effective_rank)
            self.top1_ratio_history.append(top1_ratio)
            self.collapse_warning_history.append(is_collapsed)

        self.total_analyzes += 1

        return result

    def _compute_effective_rank(self, singular_values: torch.Tensor) -> tuple[float, float]:
        """
        计算有效秩

        有效秩定义: ER = -∑ p_i * log(p_i)
        其中 p_i = σ_i / ∑σ_j

        Args:
            singular_values: 奇异值向量

        Returns:
            (effective_rank, normalized_entropy)
        """
        # 归一化奇异值
        S = singular_values + 1e-10  # 避免除零
        p = S / S.sum()

        # 计算熵
        entropy = -(p * torch.log(p)).sum()

        # 归一化熵 (除以最大熵 log(D))
        D = singular_values.shape[0]
        max_entropy = math.log(D)
        normalized_entropy = entropy.item() / max_entropy

        return entropy.item(), normalized_entropy

    def _detect_collapse(
        self,
        top1_ratio: float,
        effective_rank: float,
        energy_99_dims: int,
        feature_dim: int,
    ) -> tuple[bool, str, list[str]]:
        """
        检测特征是否坍塌

        Args:
            top1_ratio: 第一奇异值占比
            effective_rank: 有效秩
            energy_99_dims: 达到99%能量需要的维度数
            feature_dim: 总特征维度

        Returns:
            (is_collapsed, severity, reasons)
        """
        reasons = []
        severity = "none"

        # 检查各项指标
        if top1_ratio > self.COLLAPSE_THRESHOLD_TOP1:
            reasons.append(f"第一奇异值占比过高 ({top1_ratio:.2%} > {self.COLLAPSE_THRESHOLD_TOP1:.0%})")

        if effective_rank < self.COLLAPSE_THRESHOLD_EFFECTIVE_RANK:
            reasons.append(f"有效秩过低 ({effective_rank:.1f} < {self.COLLAPSE_THRESHOLD_EFFECTIVE_RANK})")

        # 严重程度判断
        if len(reasons) >= 2:
            severity = "severe"
        elif top1_ratio > 0.90 or effective_rank < 15:
            severity = "moderate"
        elif top1_ratio > 0.80 or effective_rank < 20:
            severity = "mild"

        is_collapsed = severity in ["moderate", "severe"]

        # 额外建议
        if reasons:
            if top1_ratio > 0.95:
                reasons.append("可能原因: LayerNorm 强度过高或初始化不当")
            if effective_rank < 10:
                reasons.append("建议: 检查 Patch Embedding 输出、降低 LayerNorm 强度、调整初始化")

        return is_collapsed, severity, reasons

    def get_diagnostics_summary(self) -> Dict[str, Any]:
        """获取诊断摘要"""
        if not self.effective_rank_history:
            return {"status": "no_data"}

        return {
            "status": "analyzing",
            "total_analyzes": self.total_analyzes,
            "avg_effective_rank": sum(self.effective_rank_history) / len(self.effective_rank_history),
            "min_effective_rank": min(self.effective_rank_history),
            "max_effective_rank": max(self.effective_rank_history),
            "avg_top1_ratio": sum(self.top1_ratio_history) / len(self.top1_ratio_history),
            "collapse_warnings": sum(self.collapse_warning_history),
            "recent_trend": "improving" if len(self.effective_rank_history) >= 3 and \
                self.effective_rank_history[-1] > self.effective_rank_history[-3] else "stable/worsening",
        }

    def reset(self):
        """重置历史记录"""
        self.effective_rank_history.clear()
        self.top1_ratio_history.clear()
        self.collapse_warning_history.clear()
        self.step_counter = 0
        self.total_analyzes = 0


def analyze_feature_health(
    features: torch.Tensor,
    enabled: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    便捷函数：快速分析特征健康度

    Args:
        features: [B*N, C, k, k] 或 [B*N, C] 特征
        enabled: 是否启用

    Returns:
        包含关键指标的字典，或 None
    """
    if not enabled:
        return None

    analyzer = SplitterFeatureAnalyzer(enabled=True)
    result = analyzer.analyze(features)

    if result is None:
        return None

    return {
        "effective_rank": result.effective_rank,
        "svd_top1_ratio": result.svd_top1_ratio,
        "svd_top5_ratio": result.svd_top5_ratio,
        "energy_99_percent_dims": result.energy_99_percent_dims,
        "is_collapsed": result.is_collapsed,
        "collapse_severity": result.collapse_severity,
    }
