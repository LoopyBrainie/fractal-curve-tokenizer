# -*- coding: utf-8 -*-
"""
L2 Components: Splitter Base Tests

对应模块: vit_pytorch.gumbel_topk_splitter

测试内容:
- 配额分配数学性质
- 配额与概率对齐度
- 边界情况验证
"""

import pytest
import torch
import torch.nn.functional as F

from vit_pytorch.gumbel_topk_splitter import GumbelTopKSplitter


def greedy_optimal_quota(p: list, K: int, k_min: int = 2) -> torch.Tensor:
    """概率驱动配额分配 (用于验证)"""
    p = torch.tensor(p, dtype=torch.float32)
    D = len(p)

    quota = (p * K).round().long()
    quota = quota.clamp(min=k_min)

    for _ in range(D):
        total = quota.sum()
        diff = K - total

        if diff == 0:
            break

        if diff > 0:
            _, indices = torch.sort(p, descending=True)
            for i in range(min(diff, D)):
                quota[indices[i]] += 1
        else:
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

        assert quota.sum() == K, f"sum={quota.sum()} ≠ K={K}, p={p}"
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

        quota_new = greedy_optimal_quota(p, K, k_min=2)
        ideal_new = p_tensor * K
        rmse_new = ((quota_new.float() - ideal_new) ** 2).mean().sqrt()

        quota_old = old_implementation(p, K, k_min=2)
        ideal_old = p_tensor * K
        rmse_old = ((quota_old.float() - ideal_old) ** 2).mean().sqrt()

        assert rmse_new <= rmse_old + 0.01

    @pytest.mark.parametrize("K", [16, 32, 64, 128, 256])
    def test_probability_alignment(self, K: int):
        """测试分配与概率的对齐度 > 95%"""
        p = [0.15, 0.20, 0.25, 0.40]
        p_tensor = torch.tensor(p, dtype=torch.float32)

        quota = greedy_optimal_quota(p, K, k_min=2)
        quota_ratio = quota.float() / K

        alignment = 1 - (quota_ratio - p_tensor).abs().mean()

        assert alignment > 0.95, f"Alignment {alignment:.4f} < 0.95"

    @pytest.mark.parametrize("K", [16, 32, 64])
    def test_edge_cases(self, K: int):
        """测试边界情况"""
        p_extreme = [0.01, 0.02, 0.07, 0.90]
        quota = greedy_optimal_quota(p_extreme, K, k_min=2)

        assert quota.sum() == K
        assert (quota >= 2).all()

        p_uniform = [0.25, 0.25, 0.25, 0.25]
        quota = greedy_optimal_quota(p_uniform, K, k_min=2)

        assert quota.sum() == K
        diff = quota.float().max() - quota.float().min()
        assert diff <= 2, f"Uniform not balanced: {quota}"

    def test_k_min_boundary(self):
        """测试 k_min 边界情况"""
        p = [0.40, 0.30, 0.20, 0.10]
        K = 8

        quota = greedy_optimal_quota(p, K, k_min=2)

        assert quota.sum() == K
        assert (quota == 2).all(), f"Expected all 2s: {quota}"

    def test_small_k(self):
        """测试小 K 值"""
        p = [0.25, 0.25, 0.25, 0.25]
        K = 8

        quota = greedy_optimal_quota(p, K, k_min=2)

        assert quota.sum() == K
        assert (quota >= 2).all()
        assert (quota == 2).all(), f"Expected all 2s for K=8: {quota}"

    def test_large_k(self):
        """测试大 K 值"""
        p = [0.15, 0.20, 0.25, 0.40]
        K = 1024

        quota = greedy_optimal_quota(p, K, k_min=2)

        assert quota.sum() == K
        assert (quota >= 2).all()

        p_tensor = torch.tensor(p, dtype=torch.float32)
        quota_ratio = quota.float() / K
        alignment = 1 - (quota_ratio - p_tensor).abs().mean()

        assert alignment > 0.99


class TestSplitterIntegration:
    """集成测试: 验证 splitter._compute_quota_allocation() 实际行为"""

    @pytest.fixture
    def splitter(self):
        """创建测试用 GumbelTopKSplitter"""
        return GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=8,
            hidden_dim=64,
            image_size=(224, 224),
        )

    def test_compute_quota_allocation_constraints(self, splitter):
        """测试 splitter._compute_quota_allocation() 满足约束"""
        D = splitter._current_max_depth + 1
        k_min = splitter._quota_min_per_depth
        p = torch.ones(D) / D
        splitter.quota_logits.data = p
        splitter._enable_learnable_quota = True

        min_K = D * k_min
        for K in [16, 32, 64, 128, 256]:
            quota = splitter._compute_quota_allocation(K)

            assert quota.sum() == K, f"K={K}: sum={quota.sum()} ≠ K"
            assert (quota >= k_min).all(), f"K={K}: min quota < {k_min}: {quota}"

        K = 8
        quota = splitter._compute_quota_allocation(K)
        assert quota.sum() == K

    def test_compute_quota_allocation_probability_alignment(self, splitter):
        """测试分配与概率的对齐度"""
        D = splitter._current_max_depth + 1
        p = torch.tensor([0.05, 0.05, 0.10, 0.15, 0.20, 0.20, 0.25])
        splitter.quota_logits.data = p
        splitter._enable_learnable_quota = True

        for K in [32, 64, 128, 256]:
            quota = splitter._compute_quota_allocation(K)
            quota_ratio = quota.float() / K

            alignment = 1 - (quota_ratio - p).abs().mean()
            assert alignment > 0.90

    def test_compute_quota_allocation_fallback(self, splitter):
        """测试禁用 learnable_quota 时的回退行为"""
        splitter._enable_learnable_quota = False

        for K in [16, 32, 64]:
            quota = splitter._compute_quota_allocation(K)

            assert quota.sum() == K
            diff = quota.float().max() - quota.float().min()
            assert diff <= K // 4 + 2

    def test_compute_quota_allocation_small_k(self, splitter):
        """测试小 K 值边界情况"""
        D = splitter._current_max_depth + 1
        p = torch.ones(D) / D
        splitter.quota_logits.data = p
        splitter._enable_learnable_quota = True

        K = 14
        quota = splitter._compute_quota_allocation(K)

        assert quota.sum() == K
        assert (quota == 2).all()

    def test_compute_quota_allocation_large_k(self, splitter):
        """测试大 K 值"""
        D = splitter._current_max_depth + 1
        p = torch.tensor([0.05, 0.05, 0.10, 0.15, 0.20, 0.20, 0.25])
        splitter.quota_logits.data = p
        splitter._enable_learnable_quota = True

        K = 1024
        quota = splitter._compute_quota_allocation(K)

        assert quota.sum() == K
        assert (quota >= 2).all()

        quota_ratio = quota.float() / K
        alignment = 1 - (quota_ratio - p).abs().mean()
        assert alignment > 0.90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
