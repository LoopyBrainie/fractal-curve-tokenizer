#!/usr/bin/env python3
"""
尺度偏置分析 (Scale Bias Analysis)

形式化分析为什么当前模型倾向于使用最小尺度 (最深层 split)

数学背景:
=========

复杂度函数:
    C(R) = α · C_var(R) + (1-α) · C_grad(R)
    
    其中:
        C_var(R) = Var(R) / (Var(R) + σ₀²)
        C_grad(R) = G(R) / (G(R) + g₀²)

Split 决策规则:
    SPLIT(R, d) = True  iff  C(R) ≥ τ_d  AND  d < d_max  AND  |R| ≥ 2·s_min
    
    其中:
        τ_d = τ₀ · γ^d

问题分析:
=========

观察: 模型倾向于最小尺度 (深层 split)

假设检验:
1. H1: 阈值衰减过快 (γ 过小)，导致深层几乎总是满足 C(R) ≥ τ_d
2. H2: 复杂度归一化参数 (σ₀², g₀²) 设置不当，导致 C(R) 总是过高
3. H3: 自然图像在多尺度下的复杂度分布特性导致偏置
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class ThresholdAnalysis:
    """阈值衰减分析结果"""
    depth: int
    threshold: float
    threshold_ratio: float  # 相对于 τ₀ 的比例


@dataclass
class ComplexityAtDepth:
    """某深度的复杂度统计"""
    depth: int
    region_size: int
    mean_complexity: float
    std_complexity: float
    percentile_10: float
    percentile_50: float
    percentile_90: float
    split_probability: float  # P(C(R) ≥ τ_d)


def analyze_threshold_decay(
    tau_0: float = 0.15,
    gamma: float = 0.85,
    max_depth: int = 4
) -> List[ThresholdAnalysis]:
    """分析阈值在各深度的衰减
    
    数学形式:
        τ_d = τ₀ · γ^d
        
    关键问题: 当 γ < 1 时，τ_d 随深度指数衰减
    """
    results = []
    for d in range(max_depth + 1):
        threshold = tau_0 * (gamma ** d)
        ratio = gamma ** d
        results.append(ThresholdAnalysis(
            depth=d,
            threshold=threshold,
            threshold_ratio=ratio
        ))
    return results


def compute_region_complexity(
    image: torch.Tensor,
    x1: int, y1: int, x2: int, y2: int,
    alpha: float = 0.5,
    sigma_0_sq: float = 0.01,
    g_0_sq: float = 0.08
) -> float:
    """计算区域复杂度
    
    C(R) = α · C_var(R) + (1-α) · C_grad(R)
    """
    # 提取区域
    region = image[:, y1:y2, x1:x2]
    
    # 转灰度
    if region.shape[0] == 3:
        gray = 0.299 * region[0] + 0.587 * region[1] + 0.114 * region[2]
    else:
        gray = region[0]
    
    # 计算方差
    var = gray.var().item()
    c_var = var / (var + sigma_0_sq)
    
    # 计算梯度能量
    if gray.shape[0] > 1 and gray.shape[1] > 1:
        dx = gray[:-1, 1:] - gray[:-1, :-1]
        dy = gray[1:, :-1] - gray[:-1, :-1]
        grad_energy = (dx.pow(2) + dy.pow(2)).mean().item()
    else:
        grad_energy = 0.0
    c_grad = grad_energy / (grad_energy + g_0_sq)
    
    return alpha * c_var + (1 - alpha) * c_grad


def sample_complexity_at_depth(
    image: torch.Tensor,
    depth: int,
    image_size: int,
    alpha: float = 0.5,
    sigma_0_sq: float = 0.01,
    g_0_sq: float = 0.08,
    num_samples: int = 100
) -> List[float]:
    """在给定深度采样区域复杂度
    
    深度 d 对应的区域大小: size = image_size / 2^d
    """
    region_size = image_size // (2 ** depth)
    if region_size < 4:
        return []
    
    complexities = []
    for _ in range(num_samples):
        # 随机采样区域位置
        max_x = image_size - region_size
        max_y = image_size - region_size
        if max_x <= 0 or max_y <= 0:
            continue
            
        x1 = np.random.randint(0, max_x + 1)
        y1 = np.random.randint(0, max_y + 1)
        x2 = x1 + region_size
        y2 = y1 + region_size
        
        c = compute_region_complexity(
            image, x1, y1, x2, y2,
            alpha, sigma_0_sq, g_0_sq
        )
        complexities.append(c)
    
    return complexities


def analyze_complexity_vs_depth(
    images: List[torch.Tensor],
    tau_0: float = 0.15,
    gamma: float = 0.85,
    alpha: float = 0.5,
    sigma_0_sq: float = 0.01,
    g_0_sq: float = 0.08,
    max_depth: int = 4,
    samples_per_depth: int = 200
) -> List[ComplexityAtDepth]:
    """分析复杂度随深度的变化
    
    关键洞察:
        - 如果 P(C(R) ≥ τ_d) 随深度增加，则模型倾向深层 split
        - 这是因为 τ_d 衰减速度快于 C(R) 的减小速度
    """
    image_size = images[0].shape[-1]
    results = []
    
    for d in range(max_depth + 1):
        all_complexities = []
        for img in images:
            complexities = sample_complexity_at_depth(
                img, d, image_size,
                alpha, sigma_0_sq, g_0_sq,
                num_samples=samples_per_depth // len(images)
            )
            all_complexities.extend(complexities)
        
        if not all_complexities:
            continue
        
        c_array = np.array(all_complexities)
        threshold = tau_0 * (gamma ** d)
        region_size = image_size // (2 ** d)
        
        results.append(ComplexityAtDepth(
            depth=d,
            region_size=region_size,
            mean_complexity=float(np.mean(c_array)),
            std_complexity=float(np.std(c_array)),
            percentile_10=float(np.percentile(c_array, 10)),
            percentile_50=float(np.percentile(c_array, 50)),
            percentile_90=float(np.percentile(c_array, 90)),
            split_probability=float(np.mean(c_array >= threshold))
        ))
    
    return results


def diagnose_scale_bias(
    complexity_results: List[ComplexityAtDepth],
    threshold_results: List[ThresholdAnalysis]
) -> Dict[str, any]:
    """诊断尺度偏置的根本原因
    
    诊断逻辑:
        1. 如果 split_probability 随深度单调增加 → 阈值衰减过快
        2. 如果 mean_complexity > τ₀ 在根节点 → 归一化参数过小
        3. 如果 std_complexity 过小 → 复杂度区分度不足
    """
    diagnosis = {
        "bias_detected": False,
        "primary_cause": None,
        "recommendations": []
    }
    
    # 检查 split probability 趋势
    split_probs = [r.split_probability for r in complexity_results]
    if len(split_probs) >= 2:
        # 深层 split 概率是否显著高于浅层?
        shallow_prob = np.mean(split_probs[:2])
        deep_prob = np.mean(split_probs[-2:]) if len(split_probs) >= 3 else split_probs[-1]
        
        if deep_prob > shallow_prob + 0.2:
            diagnosis["bias_detected"] = True
            diagnosis["primary_cause"] = "threshold_decay_too_fast"
            diagnosis["recommendations"].append(
                f"增加 γ 值: 当前 γ 使深层阈值衰减至 {threshold_results[-1].threshold_ratio:.1%}，"
                f"建议增加到 γ ≈ 0.92 使衰减更平缓"
            )
    
    # 检查根节点复杂度
    if complexity_results:
        root_complexity = complexity_results[0]
        if root_complexity.mean_complexity > threshold_results[0].threshold:
            diagnosis["bias_detected"] = True
            if diagnosis["primary_cause"] is None:
                diagnosis["primary_cause"] = "normalization_params_too_small"
            diagnosis["recommendations"].append(
                f"增加归一化参数: 当前 mean C(root) = {root_complexity.mean_complexity:.3f} > τ₀ = {threshold_results[0].threshold:.3f}，"
                f"建议增加 σ₀² 和 g₀²"
            )
    
    # 检查复杂度区分度
    if complexity_results:
        avg_std = np.mean([r.std_complexity for r in complexity_results])
        if avg_std < 0.05:
            diagnosis["recommendations"].append(
                f"复杂度区分度不足: σ(C) = {avg_std:.3f}，可能需要调整 α 参数"
            )
    
    return diagnosis


def run_analysis(
    dataset_name: str = "cifar10",
    num_samples: int = 100,
    tau_0: float = 0.15,
    gamma: float = 0.85,
    alpha: float = 0.5,
    sigma_0_sq: float = 0.01,
    g_0_sq: float = 0.08,
    max_depth: int = 4
):
    """运行完整分析"""
    print("\n" + "="*70)
    print("尺度偏置形式化分析 (Scale Bias Formal Analysis)")
    print("="*70)
    
    # 1. 阈值衰减分析
    print("\n【1】阈值衰减分析")
    print("-"*50)
    print(f"参数: τ₀ = {tau_0}, γ = {gamma}")
    print()
    print(f"{'深度 d':<10} {'区域大小':<15} {'阈值 τ_d':<15} {'衰减比例':<15}")
    print("-"*50)
    
    threshold_results = analyze_threshold_decay(tau_0, gamma, max_depth)
    for r in threshold_results:
        region_size = 64 // (2 ** r.depth) if r.depth <= 4 else "N/A"
        print(f"{r.depth:<10} {str(region_size)+'×'+str(region_size):<15} {r.threshold:<15.4f} {r.threshold_ratio:<15.1%}")
    
    # 关键数学观察
    print()
    print("📐 数学观察:")
    print(f"   τ_max_depth / τ_0 = γ^{max_depth} = {gamma}^{max_depth} = {gamma**max_depth:.3f}")
    print(f"   → 最深层阈值仅为根节点的 {gamma**max_depth:.1%}")
    print()
    
    # 2. 加载数据集
    print("\n【2】数据集复杂度分布分析")
    print("-"*50)
    
    data_root = PROJECT_ROOT / "data"
    if dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = datasets.CIFAR10(data_root, train=True, download=True, transform=transform)
        image_size = 32
    elif dataset_name == "tiny-imagenet":
        # 简化处理
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        try:
            dataset = datasets.ImageFolder(data_root / "tiny-imagenet-200" / "train", transform=transform)
            image_size = 64
        except:
            print("Tiny-ImageNet 未找到，使用 CIFAR-10")
            dataset = datasets.CIFAR10(data_root, train=True, download=True, transform=transform)
            image_size = 32
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # 采样图像
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    images = [dataset[i][0] for i in indices]
    
    print(f"数据集: {dataset_name}")
    print(f"图像大小: {image_size}×{image_size}")
    print(f"样本数: {len(images)}")
    print()
    
    # 3. 复杂度 vs 深度分析
    complexity_results = analyze_complexity_vs_depth(
        images, tau_0, gamma, alpha, sigma_0_sq, g_0_sq, max_depth
    )
    
    print(f"{'深度':<6} {'区域':<10} {'E[C]':<10} {'σ[C]':<10} {'P10':<10} {'P50':<10} {'P90':<10} {'P(split)':<10}")
    print("-"*76)
    
    for r in complexity_results:
        print(f"{r.depth:<6} {r.region_size}×{r.region_size:<6} "
              f"{r.mean_complexity:<10.3f} {r.std_complexity:<10.3f} "
              f"{r.percentile_10:<10.3f} {r.percentile_50:<10.3f} {r.percentile_90:<10.3f} "
              f"{r.split_probability:<10.1%}")
    
    # 4. 数学解释
    print("\n【3】数学解释: 为什么深层 Split 概率更高?")
    print("-"*50)
    print()
    print("定理 (尺度偏置定理):")
    print()
    print("  设 C_d 为深度 d 区域的复杂度，τ_d = τ₀ · γ^d 为阈值")
    print("  Split 概率为: P_d = P(C_d ≥ τ_d)")
    print()
    print("  观察: 当 d 增加时:")
    print("  - τ_d 以指数速率 γ^d 衰减")
    print("  - C_d 的变化取决于图像统计特性")
    print()
    print("  对于自然图像:")
    print("  - 小区域方差 Var(R) 通常较小 → C_var 减小")
    print("  - 但衰减速度慢于 τ_d 的指数衰减")
    print()
    print("  证明尺度偏置的充分条件:")
    print("  若 E[C_d] / τ_d 随 d 增加 → 深层 split 概率更高")
    print()
    
    # 验证充分条件
    if len(complexity_results) >= 2:
        print("  验证:")
        for i, (c, t) in enumerate(zip(complexity_results, threshold_results)):
            ratio = c.mean_complexity / t.threshold
            print(f"    d={i}: E[C]/τ = {c.mean_complexity:.3f} / {t.threshold:.3f} = {ratio:.2f}")
    
    # 5. 诊断与建议
    print("\n【4】诊断结果")
    print("-"*50)
    
    diagnosis = diagnose_scale_bias(complexity_results, threshold_results)
    
    if diagnosis["bias_detected"]:
        print(f"⚠️  检测到尺度偏置!")
        print(f"   主要原因: {diagnosis['primary_cause']}")
        print()
        print("📋 改进建议:")
        for i, rec in enumerate(diagnosis["recommendations"], 1):
            print(f"   {i}. {rec}")
    else:
        print("✅ 未检测到明显尺度偏置")
    
    # 6. 参数调整建议
    print("\n【5】参数调整数学推导")
    print("-"*50)
    
    # 计算理想 γ 使得各深度 split 概率相等
    if len(complexity_results) >= 2:
        # 设目标 P_d = P_0 对所有 d
        # 需要 τ_d 与 E[C_d] 同比例衰减
        
        complexities = [r.mean_complexity for r in complexity_results]
        
        # 估计复杂度衰减率
        if complexities[0] > 0 and complexities[-1] > 0:
            complexity_decay = (complexities[-1] / complexities[0]) ** (1 / (len(complexities) - 1))
            
            print(f"当前设置:")
            print(f"  γ (阈值衰减) = {gamma}")
            print(f"  估计复杂度衰减率 = {complexity_decay:.3f}")
            print()
            print(f"分析:")
            print(f"  阈值衰减: τ_d ∝ {gamma}^d")
            print(f"  复杂度衰减: E[C_d] ∝ {complexity_decay:.3f}^d")
            print()
            
            if gamma < complexity_decay:
                print(f"  ⚠️ γ < 复杂度衰减率 → 深层 split 概率增加")
                print(f"  建议: 增加 γ 至 ≈ {min(0.95, complexity_decay + 0.05):.2f}")
            else:
                print(f"  ✅ γ ≥ 复杂度衰减率 → 尺度相对平衡")
    
    return diagnosis, complexity_results, threshold_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="尺度偏置分析")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "tiny-imagenet"])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--tau0", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.85)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--sigma0-sq", type=float, default=0.01)
    parser.add_argument("--g0-sq", type=float, default=0.08)
    parser.add_argument("--max-depth", type=int, default=4)
    
    args = parser.parse_args()
    
    run_analysis(
        dataset_name=args.dataset,
        num_samples=args.samples,
        tau_0=args.tau0,
        gamma=args.gamma,
        alpha=args.alpha,
        sigma_0_sq=args.sigma0_sq,
        g_0_sq=args.g0_sq,
        max_depth=args.max_depth
    )
