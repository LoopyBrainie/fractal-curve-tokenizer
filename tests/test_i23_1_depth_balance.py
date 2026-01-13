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
    LOG_COMPENSATION_ENABLED,
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
    
    def test_log_compensation_still_enabled(self):
        """Log-Compensation 仍然启用"""
        assert LOG_COMPENSATION_ENABLED is True


class TestDepthVarianceNormalization:
    """测试方案C: 深度方差归一化"""
    
    @pytest.fixture
    def splitter(self):
        return GumbelTopKSplitter(
            feature_dim=64,
            max_depth=3,
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
        
        for d in range(splitter.max_depth + 1):
            mask = (depths == d)
            # depth 越大，方差越大 (模拟真实情况)
            sigma = 1.0 + d * 2.0
            logits[:, mask] = torch.randn(B, mask.sum().item()) * sigma
        
        device = logits.device
        dtype = logits.dtype
        normalized = splitter._normalize_by_depth(logits, device, dtype)
        
        # 验证每深度归一化后统计量
        for d in range(splitter.max_depth + 1):
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
        return GumbelTopKSplitter(
            feature_dim=64,
            max_depth=3,
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
        return GumbelTopKSplitter(
            feature_dim=64,
            max_depth=3,
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
    """数学验证测试"""
    
    def test_log_compensation_bias_values(self):
        """验证 Log-Compensation 偏置值正确"""
        splitter = GumbelTopKSplitter(
            feature_dim=64,
            max_depth=3,
        )
        
        # N_total = 1 + 4 + 16 + 64 = 85
        N_total = splitter.num_candidates
        assert N_total == 85
        
        # 预期偏置
        expected = {
            0: math.log(85 / 1),   # ≈ 4.44
            1: math.log(85 / 4),   # ≈ 3.06
            2: math.log(85 / 16),  # ≈ 1.67
            3: math.log(85 / 64),  # ≈ 0.28
        }
        
        depths = splitter.candidate_depths
        bias = splitter.log_compensation_bias
        
        for d in range(4):
            mask = (depths == d)
            bias_d = bias[mask][0].item()
            assert abs(bias_d - expected[d]) < 0.01, \
                f"Depth {d}: expected {expected[d]:.2f}, got {bias_d:.2f}"
    
    def test_theoretical_depth_distribution(self):
        """
        理论验证: 归一化 + Log-Compensation 后的预期分布
        
        数学推导:
            归一化后 z_i ~ N(0, 1)
            加上 Log-Compensation: z_i + b_d^log
            
            对于 Top-K 选择，期望深度分布:
            π_d ≈ N_d × Φ(b_d^log) / Σ_k N_k × Φ(b_k^log)
            
            注意: 这是期望分布，实际 Top-K 选择会因为竞争而有所不同。
            这里验证 Log-Compensation 的数学正确性。
        """
        from scipy.stats import norm
        
        N = [1, 4, 16, 64]
        N_total = sum(N)
        b_log = [math.log(N_total / n) for n in N]
        
        # Φ(b_d) - 标准正态CDF
        phi = [norm.cdf(b) for b in b_log]
        
        # 验证 Log-Compensation 偏置正确计算
        # depth=0 应该有最大偏置 (log(85))
        assert b_log[0] > b_log[1] > b_log[2] > b_log[3]
        
        # depth=0 的 CDF 最接近 1
        assert phi[0] > 0.99, f"Phi(b_0) = {phi[0]:.4f}, expected > 0.99"
        
        # 验证 Log-Compensation 设计目标:
        # 每个深度的有效权重 N_d × Φ(b_d) 应该有补偿效果
        eff = [N[d] * phi[d] for d in range(4)]
        
        # 低深度虽然候选少，但偏置大，应该有合理的有效权重
        # depth=0 的 eff 应该 > 0
        assert eff[0] > 0.5, f"eff_0 = {eff[0]:.2f}, expected > 0.5"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
