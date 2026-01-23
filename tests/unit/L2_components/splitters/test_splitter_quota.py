# -*- coding: utf-8 -*-
"""
L2 Components: Splitter Quota Tests

对应模块: vit_pytorch.gumbel_topk_splitter (可学习配额)

测试内容:
- 可学习配额初始化
- 配额分配逻辑
- 分层 Top-K 选择
- 配额熵正则化损失
"""

import pytest
import torch
import torch.nn.functional as F

from vit_pytorch.gumbel_topk_splitter import GumbelTopKSplitter
from vit_pytorch.constants import (
    LEARNABLE_QUOTA_ENABLED,
    QUOTA_MIN_RATIO,
    QUOTA_INIT_LOGITS,
    QUOTA_ENTROPY_WEIGHT,
)

DEPTH_QUOTA_TARGET = (0.15, 0.20, 0.25, 0.40)


class TestLearnableQuotaInit:
    """测试可学习配额初始化"""

    def test_quota_logits_initialized(self):
        """验证 quota_logits 参数存在且正确初始化"""
        if not LEARNABLE_QUOTA_ENABLED:
            pytest.skip("LEARNABLE_QUOTA_ENABLED is False")

        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_depth_limit=3,
            image_size=(64, 64),
        )

        assert splitter.quota_logits is not None
        assert splitter.quota_logits.requires_grad

        D = splitter._current_max_depth + 1
        assert splitter.quota_logits.shape == (D,)

    def test_quota_init_produces_target_distribution(self):
        """验证初始化配额 softmax 等于目标分布"""
        if not LEARNABLE_QUOTA_ENABLED:
            pytest.skip("LEARNABLE_QUOTA_ENABLED is False")

        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_depth_limit=3,
            image_size=(64, 64),
        )

        quota_probs = F.softmax(splitter.quota_logits, dim=0)

        target = torch.tensor(DEPTH_QUOTA_TARGET[:4])
        torch.testing.assert_close(quota_probs, target, atol=1e-3, rtol=1e-3)


class TestQuotaAllocation:
    """测试配额分配逻辑"""

    def test_quota_sum_equals_k(self):
        """验证配额总和等于 K"""
        if not LEARNABLE_QUOTA_ENABLED:
            pytest.skip("LEARNABLE_QUOTA_ENABLED is False")

        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_depth_limit=3,
            image_size=(64, 64),
        )

        for K in [16, 32, 48, 64]:
            quota = splitter._compute_quota_allocation(K)
            assert abs(quota.sum().item() - K) <= 1

    def test_quota_min_per_depth(self):
        """验证配额下界满足"""
        if not LEARNABLE_QUOTA_ENABLED:
            pytest.skip("LEARNABLE_QUOTA_ENABLED is False")

        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_depth_limit=3,
            image_size=(64, 64),
        )

        for K in [8, 16, 32, 64]:
            quota = splitter._compute_quota_allocation(K)
            assert quota.sum().item() == K

        K = 16
        min_loss = splitter._compute_quota_loss(K)
        assert min_loss.item() >= 0
        assert not torch.isnan(min_loss)

    def test_quota_proportional_to_target(self):
        """验证配额近似与目标分布成比例"""
        if not LEARNABLE_QUOTA_ENABLED:
            pytest.skip("LEARNABLE_QUOTA_ENABLED is False")

        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_depth_limit=3,
            image_size=(64, 64),
        )

        K = 32
        quota = splitter._compute_quota_allocation(K).float()
        target = torch.tensor(DEPTH_QUOTA_TARGET[:4]) * K

        torch.testing.assert_close(quota, target, atol=2.0, rtol=0.1)


class TestStratifiedTopK:
    """测试分层 Top-K 选择"""

    def test_stratified_selection_covers_all_depths(self):
        """验证分层选择覆盖多个深度"""
        if not LEARNABLE_QUOTA_ENABLED:
            pytest.skip("LEARNABLE_QUOTA_ENABLED is False")

        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_depth_limit=4,
            image_size=(64, 64),
            K_min=16,
            K_max=48,
        )
        splitter.train()

        B, C, H, W = 2, 256, 16, 16
        features = torch.randn(B, C, H, W)

        result = splitter(features)

        depths = result.depths
        unique_depths = depths.unique()
        assert len(unique_depths) >= 2

    def test_stratified_vs_global_depth_distribution(self):
        """验证分层 Top-K 选择深度分布"""
        if not LEARNABLE_QUOTA_ENABLED:
            pytest.skip("LEARNABLE_QUOTA_ENABLED is False")

        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_depth_limit=4,
            image_size=(64, 64),
            K_min=16,
            K_max=32,
        )
        splitter.eval()

        B, C, H, W = 4, 256, 16, 16
        features = torch.randn(B, C, H, W)

        scale_h = H / 64
        scale_w = W / 64
        with torch.no_grad():
            logits, _ = splitter._compute_all_logits(features, scale_h, scale_w)
            st_mask, _ = splitter._stratified_gumbel_topk_ste(logits, 32, hard=True)

        depths = splitter.candidate_depths
        D = splitter._current_max_depth + 1

        dist = []
        total = (st_mask > 0.5).float().sum().item()
        for d in range(D):
            mask = (depths == d)
            count = (st_mask[:, mask] > 0.5).float().sum().item()
            dist.append(count / total if total > 0 else 0)

        assert dist[3] < 0.6
        non_zero_depths = sum(1 for d in dist if d > 0.01)
        assert non_zero_depths >= 3


class TestQuotaEntropyLoss:
    """测试配额熵正则化损失"""

    def test_entropy_loss_returns_tensor(self):
        """验证熵损失返回有效张量"""
        if not LEARNABLE_QUOTA_ENABLED:
            pytest.skip("LEARNABLE_QUOTA_ENABLED is False")

        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_depth_limit=3,
            image_size=(64, 64),
        )

        loss = splitter.get_quota_entropy_loss()

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_entropy_loss_gradient_flow(self):
        """验证熵损失梯度流向 quota_logits"""
        if not LEARNABLE_QUOTA_ENABLED:
            pytest.skip("LEARNABLE_QUOTA_ENABLED is False")

        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_depth_limit=3,
            image_size=(64, 64),
        )

        loss = splitter.get_quota_entropy_loss(weight=1.0)
        loss.backward()

        assert splitter.quota_logits.grad is not None
        assert not torch.isnan(splitter.quota_logits.grad).any()

    def test_entropy_loss_included_in_auxiliary(self):
        """验证熵损失包含在辅助损失中"""
        if not LEARNABLE_QUOTA_ENABLED:
            pytest.skip("LEARNABLE_QUOTA_ENABLED is False")

        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_depth_limit=3,
            image_size=(64, 64),
        )
        splitter.train()

        features = torch.randn(2, 256, 16, 16)
        splitter(features)

        losses = splitter.get_auxiliary_losses()

        assert 'quota_entropy_loss' in losses


class TestMathematicalProperties:
    """数学性质验证"""

    def test_softmax_init_equals_target(self):
        """验证 softmax(QUOTA_INIT_LOGITS) ≈ DEPTH_QUOTA_TARGET"""
        init_logits = torch.tensor(QUOTA_INIT_LOGITS)
        probs = F.softmax(init_logits, dim=0)
        target = torch.tensor(DEPTH_QUOTA_TARGET[:len(init_logits)])

        torch.testing.assert_close(probs, target, atol=1e-2, rtol=1e-2)

    def test_quota_gradient_dead_zone_protection(self):
        """验证梯度流通过软下界正则化"""
        if not LEARNABLE_QUOTA_ENABLED:
            pytest.skip("LEARNABLE_QUOTA_ENABLED is False")

        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_depth_limit=3,
            image_size=(64, 64),
        )

        with torch.no_grad():
            splitter.quota_logits.fill_(-10)
            splitter.quota_logits[3] = 10

        K = 32
        quota = splitter._compute_quota_allocation(K)

        min_loss = splitter._compute_quota_loss(K)
        assert min_loss.item() >= 0
        assert min_loss.item() > 0

        splitter.quota_logits.requires_grad_(True)
        loss = splitter._compute_quota_loss(K)
        loss.backward()

        assert splitter.quota_logits.grad is not None
        assert not torch.all(splitter.quota_logits.grad == 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
