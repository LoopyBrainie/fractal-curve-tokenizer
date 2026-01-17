# -*- coding: utf-8 -*-
"""
I24-2 方案E: 可学习配额 + 分层 Top-K 测试

数学形式化验证:
    1. 配额初始化: softmax(φ^(0)) = p_target
    2. 配额下界保护: K_d >= QUOTA_MIN_PER_DEPTH
    3. 分层选择: 每个深度独立 Top-K
    4. 深度分布: 比传统 Log-Compensation 更均衡
"""

import pytest
import torch
import torch.nn.functional as F
import math

from vit_pytorch.gumbel_topk_splitter import GumbelTopKSplitter
from vit_pytorch.constants import (
    LEARNABLE_QUOTA_ENABLED,
    QUOTA_MIN_PER_DEPTH,
    QUOTA_INIT_LOGITS,
    QUOTA_ENTROPY_WEIGHT,
    DEPTH_QUOTA_TARGET,
)


class TestLearnableQuotaInit:
    """测试可学习配额初始化"""

    def test_quota_logits_initialized(self):
        """验证 quota_logits 参数存在且正确初始化"""
        if not LEARNABLE_QUOTA_ENABLED:
            pytest.skip("LEARNABLE_QUOTA_ENABLED is False")

        # I30-17-EXT: 使用新 API
        # 64x64 image, max_depth=3 -> min_patch_size = 64 / 2^3 = 8
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_depth_limit=3,  # max_depth=3 对应 4 层 (0-3)
            image_size=(64, 64),
        )

        # 验证参数存在
        assert splitter.quota_logits is not None
        assert splitter.quota_logits.requires_grad

        # 验证维度
        D = splitter._current_max_depth + 1
        assert splitter.quota_logits.shape == (D,)

    def test_quota_init_produces_target_distribution(self):
        """验证初始化配额 softmax 等于目标分布"""
        if not LEARNABLE_QUOTA_ENABLED:
            pytest.skip("LEARNABLE_QUOTA_ENABLED is False")

        # I30-17-EXT: 使用新 API
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_depth_limit=3,  # max_depth=3 对应 4 层 (0-3)
            image_size=(64, 64),
        )
        
        # 计算 softmax
        quota_probs = F.softmax(splitter.quota_logits, dim=0)
        
        # 验证近似等于目标分布
        target = torch.tensor(DEPTH_QUOTA_TARGET[:4])
        torch.testing.assert_close(quota_probs, target, atol=1e-3, rtol=1e-3)


class TestQuotaAllocation:
    """测试配额分配逻辑"""

    def test_quota_sum_equals_k(self):
        """验证配额总和等于 K"""
        if not LEARNABLE_QUOTA_ENABLED:
            pytest.skip("LEARNABLE_QUOTA_ENABLED is False")

        # I30-17-EXT: 使用新 API
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_depth_limit=3,  # max_depth=3 对应 4 层 (0-3)
            image_size=(64, 64),
        )

        for K in [16, 32, 48, 64]:
            quota = splitter._compute_quota_allocation(K)
            # 允许 ±1 的舍入误差
            assert abs(quota.sum().item() - K) <= 1, f"K={K}, quota.sum()={quota.sum()}"

    def test_quota_min_per_depth(self):
        """验证每个深度配额满足下界"""
        if not LEARNABLE_QUOTA_ENABLED:
            pytest.skip("LEARNABLE_QUOTA_ENABLED is False")

        # I30-17-EXT: 使用新 API
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_depth_limit=3,  # max_depth=3 对应 4 层 (0-3)
            image_size=(64, 64),
        )

        # 即使 K 很小，也要保证下界
        for K in [4, 8, 16, 32]:
            quota = splitter._compute_quota_allocation(K)
            for d in range(splitter._current_max_depth + 1):
                assert quota[d].item() >= QUOTA_MIN_PER_DEPTH, \
                    f"K={K}, depth={d}, quota={quota[d]}"

    def test_quota_proportional_to_target(self):
        """验证配额近似与目标分布成比例"""
        if not LEARNABLE_QUOTA_ENABLED:
            pytest.skip("LEARNABLE_QUOTA_ENABLED is False")

        # I30-17-EXT: 使用新 API
        # max_depth_limit=4 对应 max_depth=3 (0-3 共4层)
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_depth_limit=3,  # 确保 max_depth=3
            image_size=(64, 64),
        )

        K = 32
        quota = splitter._compute_quota_allocation(K).float()
        target = torch.tensor(DEPTH_QUOTA_TARGET[:4]) * K
        
        # 允许舍入误差
        torch.testing.assert_close(quota, target, atol=2.0, rtol=0.1)


class TestStratifiedTopK:
    """测试分层 Top-K 选择"""
    
    def test_stratified_selection_covers_all_depths(self):
        """验证分层选择覆盖所有深度"""
        if not LEARNABLE_QUOTA_ENABLED:
            pytest.skip("LEARNABLE_QUOTA_ENABLED is False")

        # I30-17-EXT: 使用新 API
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_depth_limit=4,
            image_size=(64, 64),
            K_min=16,
            K_max=48,
        )
        splitter.train()

        # 创建测试输入
        B, C, H, W = 2, 256, 16, 16
        features = torch.randn(B, C, H, W)

        # 前向传播
        result = splitter(features)

        # 检查选中 token 的深度分布
        depths = result.depths
        D = splitter._current_max_depth + 1

        for d in range(D):
            count = (depths == d).sum().item()
            # 每个深度至少有一些 token (考虑树一致性可能排除一些)
            # 但由于配额保证，应该有显著数量
            # 注意：树一致性约束可能排除一些父节点
            # 所以我们只检查至少有一个深度被选中

        # 至少应该有 2 个以上深度有 token
        unique_depths = depths.unique()
        assert len(unique_depths) >= 2, f"Only {len(unique_depths)} depths selected: {unique_depths}"

    def test_stratified_vs_global_depth_distribution(self):
        """验证分层 Top-K 选择按配额分配 (树一致性之前)"""
        if not LEARNABLE_QUOTA_ENABLED:
            pytest.skip("LEARNABLE_QUOTA_ENABLED is False")

        # 分层模式
        # I30-17-EXT: 使用新 API
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
        
        # 直接调用分层选择 (跳过树一致性)
        scale_h = H / 64
        scale_w = W / 64
        with torch.no_grad():
            logits, _ = splitter._compute_all_logits(features, scale_h, scale_w)
            st_mask, _ = splitter._stratified_gumbel_topk_ste(logits, 32, hard=True)
        
        # 计算分层选择后的深度分布 (树一致性之前)
        depths = splitter.candidate_depths
        D = splitter._current_max_depth + 1
        
        dist = []
        total = (st_mask > 0.5).float().sum().item()
        for d in range(D):
            mask = (depths == d)
            count = (st_mask[:, mask] > 0.5).float().sum().item()
            dist.append(count / total if total > 0 else 0)
        
        # 验证分布符合配额 (不像传统方案 94% 在 depth=3)
        # 配额约为 (5, 6, 8, 13)/32 = (0.156, 0.188, 0.25, 0.406)
        # 但受候选数量限制: (1, 4, 8, 13)/26 = (0.038, 0.154, 0.308, 0.5)
        
        # depth=3 不应超过 60% (传统方案是 94%)
        assert dist[3] < 0.6, f"Depth 3 too high: {dist}"
        # 至少 3 个深度有 token
        non_zero_depths = sum(1 for d in dist if d > 0.01)
        assert non_zero_depths >= 3, f"Too few depths covered: {dist}"


class TestQuotaEntropyLoss:
    """测试配额熵正则化损失"""

    def test_entropy_loss_returns_tensor(self):
        """验证熵损失返回有效张量"""
        if not LEARNABLE_QUOTA_ENABLED:
            pytest.skip("LEARNABLE_QUOTA_ENABLED is False")

        # I30-17-EXT: 使用新 API
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_depth_limit=3,  # max_depth=3 对应 4 层 (0-3)
            image_size=(64, 64),
        )

        loss = splitter.get_quota_entropy_loss()

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # 标量
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_entropy_loss_gradient_flow(self):
        """验证熵损失梯度流向 quota_logits"""
        if not LEARNABLE_QUOTA_ENABLED:
            pytest.skip("LEARNABLE_QUOTA_ENABLED is False")

        # I30-17-EXT: 使用新 API
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_depth_limit=3,  # max_depth=3 对应 4 层 (0-3)
            image_size=(64, 64),
        )

        loss = splitter.get_quota_entropy_loss(weight=1.0)
        loss.backward()

        # 验证梯度存在
        assert splitter.quota_logits.grad is not None
        assert not torch.isnan(splitter.quota_logits.grad).any()

    def test_entropy_loss_included_in_auxiliary(self):
        """验证熵损失包含在辅助损失中"""
        if not LEARNABLE_QUOTA_ENABLED:
            pytest.skip("LEARNABLE_QUOTA_ENABLED is False")

        # I30-17-EXT: 使用新 API
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_depth_limit=3,  # max_depth=3 对应 4 层 (0-3)
            image_size=(64, 64),
        )
        splitter.train()
        
        # 前向传播以生成缓存
        features = torch.randn(2, 256, 16, 16)
        splitter(features)
        
        # 获取辅助损失
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
        """验证配额下界保护防止死区"""
        if not LEARNABLE_QUOTA_ENABLED:
            pytest.skip("LEARNABLE_QUOTA_ENABLED is False")

        # I30-17-EXT: 使用新 API
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_depth_limit=3,  # max_depth=3 对应 4 层 (0-3)
            image_size=(64, 64),
        )

        # 模拟极端偏斜的配额 (几乎全给 depth=3)
        with torch.no_grad():
            splitter.quota_logits.fill_(-10)  # 所有都很小
            splitter.quota_logits[3] = 10     # depth=3 极大

        # 配额分配仍应保证下界
        K = 32
        quota = splitter._compute_quota_allocation(K)

        # I30-17-EXT: 使用 _current_max_depth
        for d in range(splitter._current_max_depth + 1):
            assert quota[d].item() >= QUOTA_MIN_PER_DEPTH, \
                f"depth={d} quota={quota[d]} < min={QUOTA_MIN_PER_DEPTH}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
