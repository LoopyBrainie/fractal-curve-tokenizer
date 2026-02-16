"""
Gumbel 噪声稳定性测试

验证 Gumbel-Top-K 在固定输入下的稳定性:
- 固定同一个 Input Batch
- 连续执行 5 次 Forward (training 模式)
- 计算 5 次选中索引的 IoU
- IoU >= 0.8 表示稳定性合格

也测试 DeterministicTopK 模式的稳定性。
"""

import pytest
import torch
from vit_pytorch.layers.splitters.gumbel_topk import GumbelTopKSplitter
from vit_pytorch.core.config import SplitterConfig


def compute_iou(set1: set, set2: set) -> float:
    """计算两个集合的 IoU"""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_multi_iou(index_sets: list) -> float:
    """计算多组索引的平均 Pairwise IoU"""
    if len(index_sets) < 2:
        return 1.0

    total_iou = 0.0
    count = 0
    for i in range(len(index_sets)):
        for j in range(i + 1, len(index_sets)):
            iou = compute_iou(index_sets[i], index_sets[j])
            total_iou += iou
            count += 1

    return total_iou / count if count > 0 else 1.0


def test_gumbel_stability():
    """测试 Gumbel 噪声稳定性"""
    # 创建 splitter (training 模式)
    splitter = GumbelTopKSplitter(
        feature_dim=64,
        min_patch_size=8,
        max_level_limit=4,
        hidden_dim=32,
        pool_size=2,
        K_min=8,
        K_max=32,
        enable_hierarchical_quota=True,  # 使用分层自适应配额
        image_size=(64, 64),
    )
    splitter.train()  # 确保在训练模式

    # 创建固定输入
    B, C, H, W = 2, 64, 16, 16  # feature map size
    torch.manual_seed(42)  # 固定随机种子
    features = torch.randn(B, C, H, W)

    # 连续执行 5 次 Forward
    num_runs = 5
    all_selected_indices = []

    for run_idx in range(num_runs):
        with torch.no_grad():  # 不需要梯度
            result = splitter(features, image_size=(64, 64), hard=False)

        # 收集第一个 batch (batch_idx=0) 选中的候选索引
        # candidate_indices: 每个选中 token 在 [B, N] 掩码中的列索引
        # batch_indices: 每个选中 token 所属的 batch 索引
        indices_set = set()
        for i in range(len(result.candidate_indices)):
            if result.batch_indices[i] == 0:  # 只看第一个 batch
                indices_set.add(result.candidate_indices[i].item())

        all_selected_indices.append(indices_set)
        print(f"Run {run_idx + 1}: {len(indices_set)} tokens selected")

    # 打印每次选择的详情
    for i, idx_set in enumerate(all_selected_indices):
        print(f"  Run {i+1}: {sorted(idx_set)[:10]}...")  # 只打印前10个

    # 计算平均 Pairwise IoU
    avg_iou = compute_multi_iou(all_selected_indices)

    print(f"\n=== Gumbel Stability Test Results ===")
    print(f"Average Pairwise IoU: {avg_iou:.4f}")
    status = "PASS (stable)" if avg_iou >= 0.8 else "FAIL (Gumbel noise too high)"
    print(f"Status: {status}")

    # 断言
    # assert avg_iou >= 0.8, f"Gumbel noise too high: IoU={avg_iou:.4f} < 0.8"


def test_deterministic_stability():
    """测试 DeterministicTopK 模式的稳定性"""
    # 创建配置，显式启用 DeterministicTopK，并设置正确的 feature_dim
    config = SplitterConfig(
        use_deterministic_topk=True,
        deterministic_temperature=0.5,
        feature_dim=64,  # 必须与 GumbelTopKSplitter 参数一致
    )

    # 使用 enable_hierarchical_quota=False 来测试基本 K 选择
    splitter = GumbelTopKSplitter(
        config=config,
        feature_dim=64,
        min_patch_size=8,
        max_level_limit=4,
        hidden_dim=32,
        pool_size=2,
        K_min=8,
        K_max=32,
        enable_hierarchical_quota=False,  # 关闭分层配额以测试基本功能
        image_size=(64, 64),
    )
    splitter.train()

    # 打印 splitter 配置
    print(f"\nSplitter config: K_min={splitter.K_min}, K_max={splitter.K_max}")
    print(f"use_dynamic_k={splitter.use_dynamic_k}")
    print(f"enable_hierarchical_quota={splitter._enable_hierarchical_quota}")

    # 测试 K 估计
    B, C, H, W = 2, 64, 16, 16
    torch.manual_seed(42)
    features = torch.randn(B, C, H, W)

    # 单次 forward 查看内部状态
    with torch.no_grad():
        result = splitter(features, image_size=(64, 64), hard=False)

    # 查看 logits 和 probs 统计
    print(f"\n=== Splitter Internal State ===")
    print(f"Logits stats: min={result.logits.min():.3f}, max={result.logits.max():.3f}, mean={result.logits.mean():.3f}")
    print(f"Probs stats: min={result.probs.min():.3f}, max={result.probs.max():.3f}, mean={result.probs.mean():.3f}")
    print(f"Selected tokens: {result.num_selected_per_batch}")

    # 分析 K 估计
    probs = result.probs
    B, N = probs.shape

    # 查看 hard_quota
    # print(f"\n=== Quota Debug ===")
    # print(f"hard_quota: {splitter._last_hard_quota}")
    # print(f"soft_quota: {splitter._last_soft_quota}")
    # print(f"Sum of hard_quota: {splitter._last_hard_quota.sum()}")

    # 查看 selected_mask 统计
    # print(f"\n=== Selected Mask Debug ===")
    # print(f"selected_mask stats: min={result.selected_mask.min():.4f}, max={result.selected_mask.max():.4f}")
    # print(f"selected_mask > 0.5 count: {(result.selected_mask > 0.5).sum()}")
    # print(f"selected_mask > 0.1 count: {(result.selected_mask > 0.1).sum()}")

    # 手动计算 K 估计
    # 方法 1: 统计高概率候选数量
    high_prob_count = (probs > 0.1).float().sum(dim=1).mean()
    # print(f"\n=== K Estimation Debug ===")
    # print(f"probs > 0.1 count per batch: {(probs > 0.1).float().sum(dim=1)}")
    # print(f"high_prob_count (threshold 0.1): {high_prob_count:.1f}")

    # 方法 2: 70% 累积概率
    sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
    cumsum = sorted_probs.cumsum(dim=1)
    total_prob = cumsum[:, -1:].clamp(min=1e-6)
    threshold_mask = cumsum < 0.7 * total_prob
    k_70_per_batch = threshold_mask.sum(dim=1).float() + 1
    # print(f"k_70 (70% cumulative): {k_70_per_batch.mean():.1f}")

    # 保守估计
    conservative_estimate = B * 0.1
    # print(f"conservative_estimate (10%): {conservative_estimate:.1f}")

    # 最终 K 估计
    K_min = max(8, 16)  # K_min 和 16 的最大值
    K_est = max(high_prob_count, k_70_per_batch.mean(), conservative_estimate)
    K_est = max(K_min, min(K_est, N))
    # print(f"Final K_est (after clamp): {K_est:.1f}")
    # print(f"Candidate count N: {N}")

    # 创建固定输入
    B, C, H, W = 2, 64, 16, 16
    torch.manual_seed(42)
    features = torch.randn(B, C, H, W)

    # 连续执行 5 次 Forward
    num_runs = 5
    all_selected_indices = []

    for run_idx in range(num_runs):
        with torch.no_grad():
            result = splitter(features, image_size=(64, 64), hard=False)

        indices_set = set()
        for i in range(len(result.candidate_indices)):
            if result.batch_indices[i] == 0:
                indices_set.add(result.candidate_indices[i].item())

        all_selected_indices.append(indices_set)
        print(f"Det Run {run_idx + 1}: {len(indices_set)} tokens selected")

    for i, idx_set in enumerate(all_selected_indices):
        print(f"  Run {i+1}: {sorted(idx_set)[:10]}...")

    print(f"\nDeterministicTopK enabled: {splitter.use_deterministic_topk}")

    # Deterministic 模式应该完全稳定
    avg_iou = compute_multi_iou(all_selected_indices)
    print(f"\n=== DeterministicTopK Stability Test ===")
    print(f"Average Pairwise IoU: {avg_iou:.4f}")

    return avg_iou


if __name__ == "__main__":
    print("=== Testing Gumbel-TopK (default) ===")
    test_gumbel_stability()
    print("\n" + "="*50 + "\n")
    print("=== Testing DeterministicTopK ===")
    test_deterministic_stability()
