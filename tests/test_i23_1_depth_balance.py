"""
I23-1 深度分布崩塌修复验证测试 (I35 优化后)

测试内容:
1. 常量导入正确性 (DEPTH_KL_*, DEPTH_QUOTA_* 已移除)
2. 深度方差归一化功能 (EMA 改进)
3. Scheme E 配额熵正则化
4. get_auxiliary_losses 集成
5. 深度分布统计

I35 优化总结:
- DEPTH_KL_WEIGHT, DEPTH_QUOTA_* 常量已完全移除
- DEPTH_EMA_ALPHA = 0.1 (新增 EMA 系数)
- 保留: QUOTA_ENTROPY_WEIGHT=0.1 (Scheme E核心机制)
- 深度方差归一化使用 EMA Running Statistics
"""

import pytest
import torch
import math

from vit_pytorch.gumbel_topk_splitter import GumbelTopKSplitter
from vit_pytorch.constants import (
    DEPTH_VARIANCE_NORM_ENABLED,
    DEPTH_VARIANCE_NORM_EPS,
    DEPTH_EMA_ALPHA,
    QUOTA_ENTROPY_WEIGHT,
    QUOTA_MIN_PER_DEPTH,
)


class TestI23_1Constants:
    """测试 I23-1 相关常量配置 (I35 优化后)"""

    def test_depth_kl_constants_removed(self):
        """I35: DEPTH_KL_WEIGHT 常量已移除"""
        with pytest.raises(ImportError):
            from vit_pytorch.constants import DEPTH_KL_WEIGHT

    def test_depth_quota_constants_removed(self):
        """I35: DEPTH_QUOTA_* 常量已移除"""
        with pytest.raises(ImportError):
            from vit_pytorch.constants import DEPTH_QUOTA_ENABLED
        with pytest.raises(ImportError):
            from vit_pytorch.constants import DEPTH_QUOTA_TARGET
        with pytest.raises(ImportError):
            from vit_pytorch.constants import DEPTH_QUOTA_TOLERANCE
        with pytest.raises(ImportError):
            from vit_pytorch.constants import DEPTH_QUOTA_WEIGHT

    def test_ema_alpha_configurable(self):
        """I35: EMA 系数可配置"""
        assert DEPTH_EMA_ALPHA == 0.1

    def test_variance_norm_configurable(self):
        """方案C: 深度方差归一化可配置"""
        assert DEPTH_VARIANCE_NORM_ENABLED in (True, False)  # 允许两种配置
        assert DEPTH_VARIANCE_NORM_EPS == 1e-6

    def test_scheme_e_core_preserved(self):
        """A16: 保留 Scheme E 核心机制"""
        # 配额熵正则化：防止配额logits崩溃到单一深度
        assert QUOTA_ENTROPY_WEIGHT == 0.1
        # 最小配额：硬保证每深度至少2个token
        assert QUOTA_MIN_PER_DEPTH == 2


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
            count_per_depth = mask.sum().item()

            # I35 Fix: 跳过每深度只有1个候选的深度 (per-batch 方差无意义)
            # 数学: 当 N_d = 1 时，μ = x_1, σ = 0，归一化后值恒为 0
            if count_per_depth < 2:
                continue

            values = normalized[:, mask].flatten()
            mu = values.mean().item()
            sigma = values.std().item()

            # I35: 使用 EMA 统计量，std 可能略有偏差
            # 期望: μ ≈ 0, σ ≈ 1 (放宽容忍度以适应 EMA)
            assert abs(mu) < 0.15, f"Depth {d}: mean = {mu:.4f}, expected ~0"
            assert abs(sigma - 1.0) < 0.25, f"Depth {d}: std = {sigma:.4f}, expected ~1"


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

    def test_quota_loss_method_removed(self, splitter):
        """I35: get_quota_loss 方法已移除 (死代码清理)"""
        assert not hasattr(splitter, 'get_quota_loss'), "方法应已移除"


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
        """A16: 完整前向传播和损失获取 (KL和软配额已禁用)"""
        B, C, H, W = 2, 64, 16, 16
        x = torch.randn(B, C, H, W)

        # 前向传播
        result = splitter(x)

        assert result.regions.shape[0] > 0

        # 获取损失
        losses = splitter.get_auxiliary_losses()

        # A16: depth_kl_loss 和 quota_loss 应不在损失字典中 (已禁用)
        assert 'depth_kl_loss' not in losses, "depth_kl_loss 应被禁用"
        assert 'quota_loss' not in losses, "quota_loss 应被禁用"

        # A16: 保留的损失项 (Scheme E 核心机制)
        assert 'quota_entropy_loss' in losses, "Missing quota_entropy_loss (Scheme E核心)"
        assert 'soft_entropy_loss' in losses, "Missing soft_entropy_loss"

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
        # I35: 移除 quota_deviation (DEPTH_QUOTA_TARGET 已移除)

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
        # A1: 设置 use_dynamic_k=False 以使用静态 K 边界
        splitter = GumbelTopKSplitter(
            feature_dim=64,
            min_patch_size=8,
            max_depth_limit=4,
            K_min=8,
            K_max=32,
            use_dynamic_k=False,  # 使用静态边界而非动态计算
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
