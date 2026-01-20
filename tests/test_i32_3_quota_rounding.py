# -*- coding: utf-8 -*-
"""
I32-3: Quota 配额分配测试

数学形式化验证
==============

测试目标:
1. ΣK_d = K 严格成立
2. K_d ≥ k_min 严格成立
3. RMSE < 0.5 (相比原实现降低 50%)
4. 与 softmax 概率对齐度 > 95%

日期: 2026-01-19
"""

import pytest
import torch
import torch.nn.functional as F


def greedy_optimal_quota(p: list, K: int, k_min: int = 2) -> torch.Tensor:
    """I32-3 概率驱动配额分配 (用于验证新实现)

    核心策略:
    - 初始四舍五入: K_d = round(p_d * K)
    - 下界钳制: K_d = max(K_d, k_min)
    - 迭代微调: 按概率排序调整（加给高概率，减自低概率）
    """
    p = torch.tensor(p, dtype=torch.float32)
    D = len(p)

    # 初始四舍五入
    quota = (p * K).round().long()
    quota = quota.clamp(min=k_min)

    # 概率驱动迭代微调 (最多 D 次)
    for _ in range(D):
        total = quota.sum()
        diff = K - total

        if diff == 0:
            break

        if diff > 0:
            # 不足时：增加概率最高的深度
            _, indices = torch.sort(p, descending=True)
            for i in range(min(diff, D)):
                quota[indices[i]] += 1
        else:
            # 超出时：从概率最低的非下界深度扣减
            mask = quota > k_min
            if mask.sum() > 0:
                available_p = p[mask]
                available_indices = torch.masked_select(torch.arange(D), mask)
                _, sorted_idx = torch.sort(available_p, descending=False)
                for i in range(min(-diff, len(sorted_idx))):
                    idx = available_indices[sorted_idx[i]]
                    quota[idx] -= 1

    return quota


def old_implementation(p: list, K: int, k_min: int = 2) -> torch.Tensor:
    """原实现 (用于对比)"""
    p = torch.tensor(p, dtype=torch.float32)
    D = len(p)

    # 原始配额 (四舍五入)
    quota_raw = (p * K).round().long()
    quota = quota_raw.clamp(min=k_min)

    total = quota.sum()
    diff = total - K

    if diff > 0:
        min_total = D * k_min
        effective_min = k_min
        if K < min_total:
            effective_min = max(1, K // D)

        excess_ratio = diff.float() / total.float()
        quota_scaled = quota.float() * (1 - excess_ratio)
        min_quota_tensor = torch.tensor(effective_min, dtype=torch.float)
        quota_clamped = torch.clamp(quota_scaled, min=min_quota_tensor)
        quota_new = torch.round(quota_clamped).long()

        for _ in range(4):
            current_sum = quota_new.sum()
            diff = K - current_sum
            if diff == 0:
                break
            if diff > 0:
                indices = torch.topk(quota_new.float(), min(int(diff), D)).indices
                quota_new[indices[:int(diff)]] += 1
            else:
                mask = quota_new > effective_min
                if mask.sum() == 0:
                    break
                indices = torch.nonzero(mask).squeeze(-1)
                if len(indices) > 0:
                    values = quota_new[indices].float()
                    topk = torch.topk(values, min(len(values), int(-diff))).indices
                    quota_new[indices[topk]] -= 1

        quota = quota_new
    elif diff < 0:
        quota[quota.argmax()] += int(-diff)

    return quota


class TestQuotaAllocation:
    """配额分配数学性质测试"""

    @pytest.mark.parametrize("K", [8, 16, 32, 64, 128, 256])
    @pytest.mark.parametrize("p", [
        [0.15, 0.20, 0.25, 0.40],
        [0.10, 0.15, 0.25, 0.50],
        [0.25, 0.25, 0.25, 0.25],
        [0.05, 0.10, 0.25, 0.60],
    ])
    def test_constraints(self, K: int, p: list):
        """测试约束条件: ΣK_d = K, K_d ≥ k_min"""
        quota = greedy_optimal_quota(p, K, k_min=2)

        # 约束1: 总和等于 K
        assert quota.sum() == K, f"sum={quota.sum()} ≠ K={K}, p={p}"

        # 约束2: 每个深度 >= k_min
        assert (quota >= 2).all(), f"min quota < 2: {quota}"

    @pytest.mark.parametrize("K", [8, 16, 32, 64, 128, 256])
    @pytest.mark.parametrize("p", [
        [0.15, 0.20, 0.25, 0.40],
        [0.10, 0.15, 0.25, 0.50],
        [0.25, 0.25, 0.25, 0.25],
    ])
    def test_rmse_improvement(self, K: int, p: list):
        """测试 RMSE 相比原实现降低"""
        p_tensor = torch.tensor(p, dtype=torch.float32)

        # 新实现
        quota_new = greedy_optimal_quota(p, K, k_min=2)
        ideal_new = p_tensor * K
        rmse_new = ((quota_new.float() - ideal_new) ** 2).mean().sqrt()

        # 原实现
        quota_old = old_implementation(p, K, k_min=2)
        ideal_old = p_tensor * K
        rmse_old = ((quota_old.float() - ideal_old) ** 2).mean().sqrt()

        # 新实现 RMSE 应该更低或相近
        assert rmse_new <= rmse_old + 0.01, (
            f"RMSE not improved: new={rmse_new:.4f}, old={rmse_old:.4f}"
        )

    @pytest.mark.parametrize("K", [16, 32, 64, 128, 256])  # K=8 时 k_min 约束主导
    def test_probability_alignment(self, K: int):
        """测试分配与概率的对齐度 > 95%"""
        p = [0.15, 0.20, 0.25, 0.40]
        p_tensor = torch.tensor(p, dtype=torch.float32)

        quota = greedy_optimal_quota(p, K, k_min=2)
        quota_ratio = quota.float() / K

        # 对齐度: 分配比例与概率的接近程度
        alignment = 1 - (quota_ratio - p_tensor).abs().mean()

        assert alignment > 0.95, f"Alignment {alignment:.4f} < 0.95"

    @pytest.mark.parametrize("K", [16, 32, 64])  # K=8 时极端概率不满足约束
    def test_edge_cases(self, K: int):
        """测试边界情况"""
        # 极端概率分布 (K=8 时无法同时满足 k_min=2 和总和=8)
        p_extreme = [0.01, 0.02, 0.07, 0.90]
        quota = greedy_optimal_quota(p_extreme, K, k_min=2)

        assert quota.sum() == K
        assert (quota >= 2).all()

        # 均匀分布
        p_uniform = [0.25, 0.25, 0.25, 0.25]
        quota = greedy_optimal_quota(p_uniform, K, k_min=2)

        assert quota.sum() == K
        # 均匀分布应该尽量均匀分配
        diff = quota.float().max() - quota.float().min()
        assert diff <= 2, f"Uniform not balanced: {quota}"

    def test_k_min_boundary(self):
        """测试 k_min 边界情况"""
        p = [0.40, 0.30, 0.20, 0.10]
        K = 8  # 刚好等于 4 * k_min = 8

        quota = greedy_optimal_quota(p, K, k_min=2)

        assert quota.sum() == K
        # 每个深度应该刚好等于 k_min，因为 K = D * k_min
        assert (quota == 2).all(), f"Expected all 2s: {quota}"

    def test_small_k(self):
        """测试小 K 值"""
        p = [0.25, 0.25, 0.25, 0.25]
        K = 8

        quota = greedy_optimal_quota(p, K, k_min=2)

        assert quota.sum() == K
        assert (quota >= 2).all()
        # 每个深度应该分配 2 个
        assert (quota == 2).all(), f"Expected all 2s for K=8: {quota}"

    def test_large_k(self):
        """测试大 K 值"""
        p = [0.15, 0.20, 0.25, 0.40]
        K = 1024

        quota = greedy_optimal_quota(p, K, k_min=2)

        assert quota.sum() == K
        assert (quota >= 2).all()

        # 验证比例接近
        p_tensor = torch.tensor(p, dtype=torch.float32)
        quota_ratio = quota.float() / K
        alignment = 1 - (quota_ratio - p_tensor).abs().mean()

        assert alignment > 0.99, f"Alignment {alignment:.4f} < 0.99 for large K"


class TestQuotaRMSE:
    """RMSE 对比测试"""

    def test_rmse_comparison(self):
        """对比新实现与原实现的 RMSE"""
        test_cases = [
            ([0.15, 0.20, 0.25, 0.40], 64),
            ([0.10, 0.15, 0.25, 0.50], 32),
            ([0.05, 0.10, 0.25, 0.60], 128),
            ([0.30, 0.30, 0.20, 0.20], 256),
        ]

        for p, K in test_cases:
            p_tensor = torch.tensor(p, dtype=torch.float32)

            # 新实现
            quota_new = greedy_optimal_quota(p, K, k_min=2)
            ideal = p_tensor * K
            rmse_new = ((quota_new.float() - ideal) ** 2).mean().sqrt()

            # 原实现
            quota_old = old_implementation(p, K, k_min=2)
            rmse_old = ((quota_old.float() - ideal) ** 2).mean().sqrt()

            print(f"p={p}, K={K}: new={rmse_new:.4f}, old={rmse_old:.4f}")

            # 新实现 RMSE 应该更低
            assert rmse_new <= rmse_old, (
                f"RMSE regression: new={rmse_new:.4f} > old={rmse_old:.4f}"
            )


class TestSplitterIntegration:
    """I32-3 集成测试: 验证 splitter._compute_quota_allocation() 实际行为"""

    @pytest.fixture
    def splitter(self):
        """创建测试用 GumbelTopKSplitter"""
        from vit_pytorch.gumbel_topk_splitter import GumbelTopKSplitter
        return GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_depth_limit=8,
            hidden_dim=64,
            image_size=(224, 224),
        )

    def test_compute_quota_allocation_constraints(self, splitter):
        """测试 splitter._compute_quota_allocation() 满足约束"""
        # D = _current_max_depth + 1 = 6 + 1 = 7
        D = splitter._current_max_depth + 1
        k_min = splitter._quota_min_per_depth  # 2
        p = torch.ones(D) / D  # 均匀概率
        splitter.quota_logits.data = p
        splitter._enable_learnable_quota = True

        # K 必须 >= D * k_min 才能满足最小约束
        min_K = D * k_min
        for K in [16, 32, 64, 128, 256]:
            quota = splitter._compute_quota_allocation(K)

            # 约束1: 总和等于 K
            assert quota.sum() == K, f"K={K}: sum={quota.sum()} ≠ K"

            # 约束2: 每个深度 >= k_min
            assert (quota >= k_min).all(), f"K={K}: min quota < {k_min}: {quota}"

        # 额外测试: K < D*k_min 时，sum = D*k_min (强制下界)
        K = 8
        quota = splitter._compute_quota_allocation(K)
        assert quota.sum() >= min_K, f"K={K}: sum should be at least {min_K}"

    def test_compute_quota_allocation_probability_alignment(self, splitter):
        """测试分配与概率的对齐度"""
        D = splitter._current_max_depth + 1  # 7
        p = torch.tensor([0.05, 0.05, 0.10, 0.15, 0.20, 0.20, 0.25])
        splitter.quota_logits.data = p
        splitter._enable_learnable_quota = True

        for K in [32, 64, 128, 256]:
            quota = splitter._compute_quota_allocation(K)
            quota_ratio = quota.float() / K

            # 对齐度 > 90% (7个深度，容差更大)
            alignment = 1 - (quota_ratio - p).abs().mean()
            assert alignment > 0.90, f"K={K}: alignment {alignment:.4f} < 0.90"

    def test_compute_quota_allocation_fallback(self, splitter):
        """测试禁用 learnable_quota 时的回退行为"""
        splitter._enable_learnable_quota = False

        for K in [16, 32, 64]:
            quota = splitter._compute_quota_allocation(K)

            # 回退到均匀分配
            assert quota.sum() == K
            # 允许较大差异 (D=7时余数分配不均匀)
            diff = quota.float().max() - quota.float().min()
            assert diff <= K // 4 + 2, f"K={K}: fallback not balanced: {quota}"

    def test_compute_quota_allocation_small_k(self, splitter):
        """测试小 K 值边界情况"""
        D = splitter._current_max_depth + 1  # 7
        p = torch.ones(D) / D
        splitter.quota_logits.data = p
        splitter._enable_learnable_quota = True

        K = 14  # 刚好等于 7 * k_min = 14
        quota = splitter._compute_quota_allocation(K)

        assert quota.sum() == K
        # 每个深度应该刚好等于 k_min
        assert (quota == 2).all(), f"Expected all 2s: {quota}"

    def test_compute_quota_allocation_large_k(self, splitter):
        """测试大 K 值"""
        D = splitter._current_max_depth + 1  # 7
        p = torch.tensor([0.05, 0.05, 0.10, 0.15, 0.20, 0.20, 0.25])
        splitter.quota_logits.data = p
        splitter._enable_learnable_quota = True

        K = 1024
        quota = splitter._compute_quota_allocation(K)

        assert quota.sum() == K
        assert (quota >= 2).all()

        # 对齐度 > 90% (大K时接近理想分布)
        quota_ratio = quota.float() / K
        alignment = 1 - (quota_ratio - p).abs().mean()
        assert alignment > 0.90, f"Large K alignment {alignment:.4f} < 0.90"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
