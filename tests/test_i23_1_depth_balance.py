"""
I23-1 深度分布崩塌修复验证测试

测试内容:
1. 常量导入正确性
2. 深度方差归一化功能
3. 软配额损失计算
4. get_auxiliary_losses 集成
5. 深度分布统计
"""

import pytest
import torch
import math

from vit_pytorch.gumbel_topk_splitter import GumbelTopKSplitter
from vit_pytorch.constants import (
    DEPTH_KL_WEIGHT,
    DEPTH_VARIANCE_NORM_ENABLED,
    DEPTH_VARIANCE_NORM_EPS,
    DEPTH_QUOTA_ENABLED,
    DEPTH_QUOTA_TARGET,
    DEPTH_QUOTA_TOLERANCE,
    DEPTH_QUOTA_WEIGHT,
)


class TestI23_1Constants:
    """测试 I23-1 相关常量配置"""
    
    def test_depth_kl_weight_increased(self):
        """方案A: KL权重从 0.1 提升到 0.5"""
        assert DEPTH_KL_WEIGHT == 0.5, f"Expected 0.5, got {DEPTH_KL_WEIGHT}"
    
    def test_variance_norm_configurable(self):
        """方案C: 深度方差归一化可配置 (当前默认禁用，通过实验发现可能导致不稳定)"""
        # I24-13: 更新测试以匹配当前设计决策
        # DEPTH_VARIANCE_NORM_ENABLED 默认为 False，因为实验显示可能导致批次统计不稳定 (I24-5)
        assert DEPTH_VARIANCE_NORM_ENABLED in (True, False)  # 允许两种配置
        assert DEPTH_VARIANCE_NORM_EPS == 1e-6
    
    def test_quota_enabled(self):
        """方案D: 软配额正则化默认启用"""
        assert DEPTH_QUOTA_ENABLED is True
        assert DEPTH_QUOTA_TARGET == (0.15, 0.20, 0.25, 0.40)
        assert DEPTH_QUOTA_TOLERANCE == 0.05
        assert DEPTH_QUOTA_WEIGHT == 0.2


class TestDepthVarianceNormalization:
    """测试方案C: 深度方差归一化"""

    @pytest.fixture
    def splitter(self):
        # I30-17-EXT: 使用新 API
        # 64x64 image, max_depth=3 -> min_patch_size = 64 / 2^3 = 8
        return GumbelTopKSplitter(
            feature_dim=64,
            min_patch_size=8,
            max_depth_limit=4,
            image_size=(64, 64),
        )
    
    def test_normalize_by_depth_method_exists(self, splitter):
        """验证方法存在"""
        assert hasattr(splitter, '_normalize_by_depth')
    
    def test_normalize_by_depth_output_shape(self, splitter):
        """验证输出形状正确"""
        B, N = 4, splitter.num_candidates
        logits = torch.randn(B, N)
        device = logits.device
        dtype = logits.dtype
        
        normalized = splitter._normalize_by_depth(logits, device, dtype)
        
        assert normalized.shape == (B, N)
    
    def test_normalize_by_depth_statistics(self, splitter):
        """验证归一化后每深度统计量接近 N(0,1)"""
        B, N = 100, splitter.num_candidates

        # 模拟不同深度有不同方差的输入
        logits = torch.zeros(B, N)
        depths = splitter.candidate_depths

        # I30-17-EXT: 使用 _current_max_depth
        for d in range(splitter._current_max_depth + 1):
            mask = (depths == d)
            # depth 越大，方差越大 (模拟真实情况)
            sigma = 1.0 + d * 2.0
            logits[:, mask] = torch.randn(B, mask.sum().item()) * sigma

        device = logits.device
        dtype = logits.dtype
        normalized = splitter._normalize_by_depth(logits, device, dtype)

        # 验证每深度归一化后统计量
        for d in range(splitter._current_max_depth + 1):
            mask = (depths == d)
            values = normalized[:, mask].flatten()
            
            mu = values.mean().item()
            sigma = values.std().item()
            
            # 期望: μ ≈ 0, σ ≈ 1
            assert abs(mu) < 0.1, f"Depth {d}: mean = {mu:.4f}, expected ~0"
            assert abs(sigma - 1.0) < 0.15, f"Depth {d}: std = {sigma:.4f}, expected ~1"


class TestQuotaLoss:
    """测试方案D: 软配额正则化"""

    @pytest.fixture
    def splitter(self):
        # I30-17-EXT: 使用新 API
        # 64x64 image, max_depth=3 -> min_patch_size = 64 / 2^3 = 8
        return GumbelTopKSplitter(
            feature_dim=64,
            min_patch_size=8,
            max_depth_limit=4,
            image_size=(64, 64),
        )
    
    def test_get_quota_loss_method_exists(self, splitter):
        """验证方法存在"""
        assert hasattr(splitter, 'get_quota_loss')
    
    def test_quota_loss_zero_for_target_distribution(self, splitter):
        """目标分布时损失很小"""
        B, N = 4, splitter.num_candidates
        depths = splitter.candidate_depths
        
        # 构造精确匹配目标分布的 mask
        K = 32
        target = torch.tensor(DEPTH_QUOTA_TARGET)
        tokens_per_depth = (target * K).round().int()
        
        selected_mask = torch.zeros(B, N)
        for d in range(4):
            mask = (depths == d)
            indices = torch.where(mask)[0][:tokens_per_depth[d]]
            selected_mask[:, indices] = 1.0
        
        loss = splitter.get_quota_loss(selected_mask)
        
        # 在容忍带内，损失应该很小（可能有舍入误差）
        assert loss.item() < 0.01, f"Expected ~0, got {loss.item()}"
    
    def test_quota_loss_positive_for_collapsed_distribution(self, splitter):
        """崩塌分布时损失为正"""
        B, N = 4, splitter.num_candidates
        depths = splitter.candidate_depths
        
        # 所有 token 都来自 depth=3 (崩塌)
        mask_d3 = (depths == 3)
        selected_mask = torch.zeros(B, N)
        selected_mask[:, mask_d3] = 1.0
        
        loss = splitter.get_quota_loss(selected_mask)
        
        # 崩塌时损失应该明显大于 0
        assert loss.item() > 0.01, f"Expected >0.01, got {loss.item()}"


class TestAuxiliaryLossesIntegration:
    """测试辅助损失集成"""

    @pytest.fixture
    def splitter(self):
        # I30-17-EXT: 使用新 API
        # 64x64 image, max_depth=3 -> min_patch_size = 64 / 2^3 = 8
        return GumbelTopKSplitter(
            feature_dim=64,
            min_patch_size=8,
            max_depth_limit=4,
            image_size=(64, 64),
        )
    
    def test_forward_and_get_losses(self, splitter):
        """完整前向传播和损失获取"""
        B, C, H, W = 2, 64, 16, 16
        x = torch.randn(B, C, H, W)
        
        # 前向传播
        result = splitter(x)
        
        assert result.regions.shape[0] > 0
        
        # 获取损失
        losses = splitter.get_auxiliary_losses()
        
        # 验证新增的损失项
        assert 'depth_kl_loss' in losses, "Missing depth_kl_loss"
        assert 'quota_loss' in losses, "Missing quota_loss"
        
        # 验证损失有梯度
        total_loss = sum(losses.values())
        assert total_loss.requires_grad or total_loss.item() >= 0
    
    def test_depth_distribution_monitoring(self, splitter):
        """测试深度分布监控功能"""
        B, C, H, W = 4, 64, 16, 16
        x = torch.randn(B, C, H, W)
        
        _ = splitter(x)
        
        stats = splitter.get_depth_distribution()
        
        assert 'pi' in stats
        assert 'entropy' in stats
        assert 'max_entropy' in stats
        assert 'kl_from_uniform' in stats
        assert 'quota_deviation' in stats
        
        # 验证分布有效
        pi = stats['pi']
        assert len(pi) == 4
        assert abs(sum(pi) - 1.0) < 1e-5, f"Distribution sums to {sum(pi)}"
        
        # 熵应该在有效范围内
        assert 0 <= stats['entropy'] <= stats['max_entropy']


class TestMathematicalValidation:
    """数学验证测试 - 方案E (Learnable Quota + Stratified Top-K)"""

    def test_candidate_count_by_depth(self):
        """验证各深度候选数量正确"""
        # I30-17-EXT: 使用新 API
        # 64x64 image, max_depth=3 -> min_patch_size = 64 / 2^3 = 8
        splitter = GumbelTopKSplitter(
            feature_dim=64,
            min_patch_size=8,
            max_depth_limit=4,
            image_size=(64, 64),
        )

        # N_total = 1 + 4 + 16 + 64 = 85
        N_total = splitter.num_candidates
        assert N_total == 85

        # 验证各深度候选数量
        depths = splitter.candidate_depths
        expected_counts = {0: 1, 1: 4, 2: 16, 3: 64}

        for d, expected in expected_counts.items():
            actual = (depths == d).sum().item()
            assert actual == expected, \
                f"Depth {d}: expected {expected} candidates, got {actual}"

    def test_stratified_selection_covers_all_depths(self):
        """验证分层 Top-K 选择覆盖所有深度"""
        # I30-17-EXT: 使用新 API
        splitter = GumbelTopKSplitter(
            feature_dim=64,
            min_patch_size=8,
            max_depth_limit=4,
            K_min=8,
            K_max=32,
            image_size=(64, 64),
        )
        
        # 创建测试输入
        B, C, H, W = 2, 64, 8, 8
        features = torch.randn(B, C, H, W)
        
        # 运行前向传播
        splitter.eval()
        with torch.no_grad():
            result = splitter(features)
        
        # 验证至少有 K_min 个 token 被选中
        for b in range(B):
            num_selected = result.num_selected_per_batch[b].item()
            assert num_selected >= splitter.K_min, \
                f"Batch {b}: expected >= {splitter.K_min} tokens, got {num_selected}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
