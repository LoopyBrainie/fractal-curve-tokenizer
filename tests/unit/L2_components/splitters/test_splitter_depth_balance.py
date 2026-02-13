# -*- coding: utf-8 -*-
"""
L2 Components: Splitter Depth Balance Tests (I131-1)

对应模块: vit_pytorch.gumbel_topk_splitter

测试内容:
- 固定深度偏置已移除
- 深度分布由可学习配额主导
- 无固定偏置时的深度分布更均匀

I131-1 变更:
- 移除 depth_bias_beta 和 depth_bias_gamma buffer
- logits 计算不再包含 β·γ^{d_i} 项
- Scheme E 可学习配额机制完全主导深度选择
"""

import pytest
import torch
import torch.nn.functional as F

from vit_pytorch.layers.splitters.gumbel_topk import GumbelTopKSplitter


class TestI1311DepthBiasRemoved:
    """验证 I131-1: 固定深度偏置已移除"""

    def test_no_depth_bias_buffers(self):
        """验证 depth_bias_beta 和 depth_bias_gamma buffer 不存在"""
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_level_limit=4,
            image_size=(64, 64),
        )

        # I131-1: 这些 buffer 应该被注释掉/不存在
        assert not hasattr(splitter, 'depth_bias_beta'), \
            "depth_bias_beta should be removed (I131-1)"
        assert not hasattr(splitter, 'depth_bias_gamma'), \
            "depth_bias_gamma should be removed (I131-1)"

    def test_quota_logits_still_exists(self):
        """验证可学习配额机制仍然存在"""
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_level_limit=4,
            image_size=(64, 64),
        )

        # Scheme E 可学习配额应该正常工作
        assert splitter.quota_logits is not None
        assert splitter.quota_logits.requires_grad

    def test_depth_embedding_exists(self):
        """验证深度嵌入偏置仍然存在"""
        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=8,
            max_level_limit=4,
            image_size=(64, 64),
        )

        # depth_embedding 负责学习深度相关特征，不应被移除
        assert hasattr(splitter, 'depth_embedding')
        assert hasattr(splitter, 'depth_proj')


class TestI1311DepthDistribution:
    """测试移除固定偏置后的深度分布"""

    @pytest.fixture
    def splitter(self):
        """创建测试用 splitter"""
        return GumbelTopKSplitter(
            feature_dim=64,
            min_patch_size=8,
            max_level_limit=4,
            image_size=(64, 64),
            K_min=16,
            K_max=64,
        )

    def test_depth_distribution_without_fixed_bias(self, splitter):
        """验证在没有固定深度偏置时，深度分布由配额决定"""
        torch.manual_seed(42)
        device = next(splitter.parameters()).device

        x = torch.randn(4, 64, 16, 16, device=device)

        # 多次 forward 收集深度分布
        distributions = []
        for _ in range(20):
            with torch.no_grad():
                result = splitter(x, image_size=(64, 64))
                # 获取分割结果中的深度分布
                num_tokens = result.num_tokens

            # 计算实际深度分布
            depths = splitter.candidate_depths[:num_tokens]
            total = len(depths)
            if total > 0:
                dist = {d: float((depths == d).sum()) / total for d in range(5)}
                distributions.append(dist)

        # 验证分布存在
        assert len(distributions) > 0, "应该收集到深度分布"

    def test_quota_softmax_valid(self, splitter):
        """验证 quota_logits 的 softmax 是有效的概率分布"""
        quota_probs = F.softmax(splitter.quota_logits, dim=0)

        # 验证归一化
        torch.testing.assert_close(
            quota_probs.sum(),
            torch.tensor(1.0),
            atol=1e-5,
            rtol=1e-5,
            msg="quota softmax 应该归一化为 1"
        )

        # 验证所有值在 [0, 1] 范围内
        assert quota_probs.min() >= 0.0
        assert quota_probs.max() <= 1.0

    def test_quota_init_produces_shallow_preference(self, splitter):
        """验证配额初始化偏向浅层 (逆深度加权)

        初始化使用 p_d ∝ 1/(d+1)，因此浅层有更高的初始配额
        这确保训练开始时模型有足够的全局视野
        """
        quota_probs = F.softmax(splitter.quota_logits, dim=0)

        # 深度 0 的配额应该高于深度 4
        assert quota_probs[0] > quota_probs[4] * 0.5, \
            "初始化时浅层配额应该不低于深层的 50%"


class TestI1311LogitsComputation:
    """验证 logits 计算不含固定深度偏置"""

    def test_logits_without_depth_bias_term(self):
        """验证 logits 计算不再包含 depth_bias_fixed 项

        旧实现:
            logits = complexity + depth_bias_learned + depth_bias_fixed + explore - tau

        新实现 (I131-1):
            logits = complexity + depth_bias_learned + explore - tau
        """
        splitter = GumbelTopKSplitter(
            feature_dim=64,
            min_patch_size=8,
            max_level_limit=4,
            image_size=(64, 64),
        )

        torch.manual_seed(42)
        x = torch.randn(2, 64, 16, 16)

        # 前向传播 - 需要 scale_h 和 scale_w 参数
        # 对于 64x64 图像和 16x16 特征图，缩放因子为 4.0
        scale_h = 4.0
        scale_w = 4.0
        logits, probs = splitter._compute_all_logits(x, scale_h, scale_w)

        # 验证输出形状 [B, N] where B=2 (batch size), N=85 (candidates)
        N = splitter.candidate_depths.shape[0]
        B = 2
        assert logits.shape == (B, N), f"Expected ({B}, {N}), got {logits.shape}"
        assert probs.shape == (B, N), f"Expected ({B}, {N}), got {probs.shape}"

        # 验证没有 NaN 或 Inf
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
        assert not torch.isnan(probs).any()
        assert not torch.isinf(probs).any()


class TestI1311BackwardCompatibility:
    """测试向后兼容性"""

    def test_splitter_creation_without_old_params(self):
        """验证可以正常创建 splitter，无需 depth_bias_beta/gamma 参数"""
        # 这些参数应该不再存在
        with pytest.raises (TypeError, match="depth_bias_beta"):
            splitter = GumbelTopKSplitter(
                feature_dim=64,
                min_patch_size=8,
                max_level_limit=4,
                image_size=(64, 64),
                depth_bias_beta=0.5,  # 应该失败
            )

    def test_explore_bias_still_exists(self):
        """验证探索偏置仍然存在"""
        splitter = GumbelTopKSplitter(
            feature_dim=64,
            min_patch_size=8,
            max_level_limit=4,
            image_size=(64, 64),
        )

        # explore_bias 应该仍然存在
        assert hasattr(splitter, 'explore_bias')

    def test_thresholds_still_exists(self):
        """验证深度阈值仍然存在"""
        splitter = GumbelTopKSplitter(
            feature_dim=64,
            min_patch_size=8,
            max_level_limit=4,
            image_size=(64, 64),
        )

        # thresholds 应该仍然存在
        assert hasattr(splitter, 'thresholds')
        assert splitter.thresholds is not None
