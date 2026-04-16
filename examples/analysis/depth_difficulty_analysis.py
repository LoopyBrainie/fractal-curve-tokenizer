# -*- coding: utf-8 -*-
"""
深度分布与任务难度关系评估
===========================

对训练好的模型(pth)进行评估，分析深度分布与任务难度的关系。

假设:
- H1: 简单图像 → 深层Token占主导
- H2: 复杂图像 → 浅层Token更活跃

使用方法:
    python examples/analysis/depth_difficulty_analysis.py --checkpoint path/to/best.pth
"""
import sys
# Windows multiprocessing 修复
import multiprocessing
if sys.platform == 'win32':
    multiprocessing.freeze_support()

sys.path.insert(0, 'src')

import argparse
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from training.layered_evaluator import LayeredEvaluator


def compute_image_complexity(images: torch.Tensor) -> np.ndarray:
    """使用边缘检测作为复杂度代理"""
    device = images.device
    gray = images.mean(dim=1).unsqueeze(1)  # [B, 1, H, W]

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)

    edges_x = F.conv2d(gray, sobel_x, padding=1)
    edges_y = F.conv2d(gray, sobel_y, padding=1)
    edge_magnitude = torch.sqrt(edges_x ** 2 + edges_y ** 2)

    return edge_magnitude.mean(dim=(2, 3)).squeeze().cpu().numpy()


def compute_image_entropy(images: torch.Tensor) -> np.ndarray:
    """计算图像信息熵"""
    gray = images.mean(dim=1)
    B = gray.size(0)

    entropies = []
    for i in range(B):
        img = gray[i]
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img_norm = (img - img_min) / (img_max - img_min)
        else:
            img_norm = img - img_min

        hist = torch.histc(img_norm.flatten(), bins=256, min=0.0, max=1.0)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        entropy = -(probs * torch.log(probs)).sum().item()
        entropies.append(entropy)

    return np.array(entropies)


def compute_prediction_entropy(logits: torch.Tensor) -> np.ndarray:
    """计算预测熵"""
    probs = F.softmax(logits, dim=-1).clamp(min=1e-10)
    entropy = -(probs * torch.log(probs)).sum(dim=-1)
    return entropy.cpu().numpy()


def run_depth_difficulty_analysis(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 1000,
) -> Dict[str, Any]:
    """运行深度-难度分析

    Returns:
        包含分析结果的字典
    """
    model.eval()

    complexity_list = []
    entropy_list = []
    pred_entropy_list = []
    num_tokens_list = []
    depth_dist_list = []
    correct_list = []
    max_depth = 8

    sample_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="分析深度-难度关系"):
            if sample_count >= max_samples:
                break

            images = batch[0].to(device)
            labels = batch[1].to(device)
            B = images.size(0)

            # Forward
            output = model(images)

            # 计算复杂度
            complexity = compute_image_complexity(images)
            if complexity.ndim == 0:
                complexity = np.array([complexity])
            complexity_list.extend(complexity.tolist())

            # 计算图像熵
            entropy = compute_image_entropy(images)
            entropy_list.extend(entropy.tolist())

            # 预测熵
            pred_entropy = compute_prediction_entropy(output.logits)
            pred_entropy_list.extend(pred_entropy.tolist())

            # Token数量
            num_tokens = output.num_tokens
            if isinstance(num_tokens, torch.Tensor):
                tokens = num_tokens.cpu().tolist()
            else:
                tokens = list(num_tokens)
            num_tokens_list.extend(tokens)

            # 深度分布 - 从output获取
            depth_dist = output.depth_distribution
            if depth_dist is None or len(depth_dist) == 0:
                # 尝试从splitter获取
                if hasattr(model, 'splitter') and hasattr(model.splitter, 'quota_logits'):
                    quota = torch.softmax(model.splitter.quota_logits, dim=0).cpu().numpy()
                    depth_arr = np.zeros(max_depth)
                    for d in range(min(len(quota), max_depth)):
                        depth_arr[d] = quota[d]
                else:
                    depth_arr = np.zeros(max_depth)
            else:
                depth_arr = np.zeros(max_depth)
                for d, p in depth_dist.items():
                    if d < max_depth:
                        depth_arr[d] = p

            for _ in range(B):
                depth_dist_list.append(depth_arr.copy())

            # 预测正确性
            preds = output.logits.argmax(dim=-1)
            correct = (preds == labels).cpu().numpy()
            correct_list.extend(correct.tolist())

            sample_count += B

    # 转换为数组
    complexity_arr = np.array(complexity_list)
    entropy_arr = np.array(entropy_list)
    pred_entropy_arr = np.array(pred_entropy_list)
    token_count_arr = np.array(num_tokens_list)
    depth_dist_arr = np.array(depth_dist_list)
    correct_arr = np.array(correct_list)

    # 计算浅层/深层比例
    p_shallow = depth_dist_arr[:, :2].sum(axis=1) if depth_dist_arr.shape[1] >= 2 else depth_dist_arr[:, 0]
    p_deep = depth_dist_arr[:, 2:].sum(axis=1) if depth_dist_arr.shape[1] > 2 else np.zeros(len(depth_dist_arr))

    # 计算相关性
    results = {
        "num_samples": len(complexity_arr),
        "complexity_stats": {
            "mean": float(complexity_arr.mean()),
            "std": float(complexity_arr.std()),
            "min": float(complexity_arr.min()),
            "max": float(complexity_arr.max()),
        },
        "entropy_stats": {
            "mean": float(entropy_arr.mean()),
            "std": float(entropy_arr.std()),
            "min": float(entropy_arr.min()),
            "max": float(entropy_arr.max()),
        },
        "pred_entropy_stats": {
            "mean": float(pred_entropy_arr.mean()),
            "std": float(pred_entropy_arr.std()),
        },
        "depth_distribution_stats": {},
        "correlations": {},
        "group_analysis": {},
    }

    # 深度分布统计
    for d in range(min(max_depth, depth_dist_arr.shape[1])):
        results["depth_distribution_stats"][f"depth_{d}"] = float(depth_dist_arr[:, d].mean())

    results["depth_distribution_stats"]["shallow_ratio"] = float(p_shallow.mean())
    results["depth_distribution_stats"]["deep_ratio"] = float(p_deep.mean())

    # 计算相关性
    def safe_corr(x, y):
        if np.std(x) < 1e-6 or np.std(y) < 1e-6:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    results["correlations"]["complexity_vs_tokens"] = safe_corr(complexity_arr, token_count_arr)
    results["correlations"]["complexity_vs_shallow"] = safe_corr(complexity_arr, p_shallow)
    results["correlations"]["complexity_vs_deep"] = safe_corr(complexity_arr, p_deep)
    results["correlations"]["entropy_vs_shallow"] = safe_corr(entropy_arr, p_shallow)
    results["correlations"]["entropy_vs_deep"] = safe_corr(entropy_arr, p_deep)
    results["correlations"]["pred_entropy_vs_shallow"] = safe_corr(pred_entropy_arr, p_shallow)
    results["correlations"]["pred_entropy_vs_deep"] = safe_corr(pred_entropy_arr, p_deep)
    results["correlations"]["complexity_vs_pred_entropy"] = safe_corr(complexity_arr, pred_entropy_arr)

    # 分组分析
    # 按图像复杂度分组
    complexity_median = np.median(complexity_arr)
    high_complexity = complexity_arr > complexity_median
    low_complexity = ~high_complexity

    results["group_analysis"]["by_complexity"] = {
        "high_complexity_samples": int(high_complexity.sum()),
        "low_complexity_samples": int(low_complexity.sum()),
        "high_complexity_shallow_ratio": float(p_shallow[high_complexity].mean()),
        "low_complexity_shallow_ratio": float(p_shallow[low_complexity].mean()),
        "high_complexity_deep_ratio": float(p_deep[high_complexity].mean()),
        "low_complexity_deep_ratio": float(p_deep[low_complexity].mean()),
        "high_complexity_accuracy": float(correct_arr[high_complexity].mean() * 100),
        "low_complexity_accuracy": float(correct_arr[low_complexity].mean() * 100),
    }

    # 按预测熵分组（任务难度）
    pred_entropy_median = np.median(pred_entropy_arr)
    hard_samples = pred_entropy_arr > pred_entropy_median
    easy_samples = ~hard_samples

    results["group_analysis"]["by_difficulty"] = {
        "hard_samples": int(hard_samples.sum()),
        "easy_samples": int(easy_samples.sum()),
        "hard_samples_shallow_ratio": float(p_shallow[hard_samples].mean()),
        "easy_samples_shallow_ratio": float(p_shallow[easy_samples].mean()),
        "hard_samples_deep_ratio": float(p_deep[hard_samples].mean()),
        "easy_samples_deep_ratio": float(p_deep[easy_samples].mean()),
    }

    return results


def run_evaluation(
    checkpoint_path: str,
    dataset_name: str = "cub200",
    batch_size: int = 32,
    max_samples: int = 1000,
    output_dir: str = None,
) -> Dict[str, Any]:
    """运行完整评估

    Args:
        checkpoint_path: 模型checkpoint路径
        dataset_name: 数据集名称
        batch_size: 批次大小
        max_samples: 最大样本数
        output_dir: 输出目录

    Returns:
        分析结果字典
    """
    print("=" * 60)
    print("深度分布与任务难度关系评估")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Max samples: {max_samples}")
    print("=" * 60)

    # 创评估器
    evaluator = LayeredEvaluator(
        checkpoint_path=checkpoint_path,
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=2,
    )

    # 加载模型
    model = evaluator._load_model()
    device = evaluator.device
    model = model.to(device)
    model.eval()

    # 加载数据
    _, _, test_loader = evaluator._load_data()

    return run_depth_difficulty_analysis_with_model(
        model=model,
        dataloader=test_loader,
        device=device,
        max_samples=max_samples,
        output_dir=output_dir,
    )


def run_depth_difficulty_analysis_with_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 1000,
    output_dir: str = None,
) -> Dict[str, Any]:
    """使用已有模型运行分析（供评估流程调用）

    Args:
        model: 已加载的模型
        dataloader: 数据加载器
        device: 设备
        max_samples: 最大样本数
        output_dir: 输出目录

    Returns:
        分析结果字典
    """
    # 运行分析
    results = run_depth_difficulty_analysis(
        model=model,
        dataloader=dataloader,
        device=device,
        max_samples=max_samples,
    )

    # 保存结果
    output_dir = Path(output_dir) if output_dir else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "depth_difficulty_analysis.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 分析结果已保存: {output_path}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("分析结果摘要")
    print("=" * 60)

    print(f"\n样本数: {results['num_samples']}")

    print("\n深度分布:")
    for key, value in results['depth_distribution_stats'].items():
        print(f"  {key}: {value:.4f}")

    print("\n相关性分析:")
    for key, value in results['correlations'].items():
        sign = "+" if value > 0 else ""
        print(f"  {key}: {sign}{value:.4f}")

    print("\n分组分析 (按图像复杂度):")
    ga = results['group_analysis']['by_complexity']
    print(f"  高复杂度样本: {ga['high_complexity_samples']}, 准确率: {ga['high_complexity_accuracy']:.1f}%")
    print(f"  低复杂度样本: {ga['low_complexity_samples']}, 准确率: {ga['low_complexity_accuracy']:.1f}%")

    # 假设验证结论
    print("\n" + "=" * 60)
    print("假设验证结论")
    print("=" * 60)

    corr_shallow = results['correlations']['complexity_vs_shallow']
    corr_deep = results['correlations']['complexity_vs_deep']

    if corr_shallow > 0.1:
        print("✓ H2 支持: 复杂度越高，浅层token越多")
    elif corr_shallow < -0.1:
        print("✗ H2 不支持: 复杂度越高，浅层token越少")
    else:
        print("○ H2 中性: 复杂度与浅层token比例无明显相关性")

    if corr_deep < -0.1:
        print("✓ H1 支持: 复杂度越高，深层token越少")
    elif corr_deep > 0.1:
        print("✗ H1 不支持: 复杂度越高，深层token越多")
    else:
        print("○ H1 中性: 复杂度与深层token比例无明显相关性")

    return results


def main():
    parser = argparse.ArgumentParser(description="I133-2 深度分布与任务难度关系评估")
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                        help="模型checkpoint路径")
    parser.add_argument("--dataset", "-d", type=str, default="cub200",
                        choices=["cifar10", "cifar100", "cub200", "tiny-imagenet"],
                        help="数据集名称")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--max-samples", "-n", type=int, default=1000,
                        help="最大样本数")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="输出目录")

    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
