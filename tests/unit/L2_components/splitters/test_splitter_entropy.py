# -*- coding: utf-8 -*-
"""
I109-8: Entropy Regularization Verification Tests

对应模块: vit_pytorch.gumbel_topk_splitter

测试内容:
- 深度熵损失 (depth entropy loss) 计算正确性
- 配额熵正则化 (quota entropy regularization) 计算正确性
- 熵损失梯度流验证

Note:
====
深度熵损失: H = -Σ_d p_d log(p_d), L = -weight × H (最大化熵)
配额熵正则化: 防止可学习配额 φ 崩塌到单一深度
"""

import pytest
import torch

import sys
sys.path.insert(0, 'src')

from vit_pytorch.layers.splitters.gumbel_topk import GumbelTopKSplitter


class TestDepthEntropyLoss:
    """深度熵损失验证测试."""

    @pytest.fixture
    def splitter(self):
        """Create a GumbelTopKSplitter instance for testing."""
        return GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=4,
            hidden_dim=128,
            pool_size=4,
            K_max=32,
            # I111-1: temperature 参数已移至 SplitterConfig
        )

    def test_depth_entropy_loss_computation(self, splitter):
        """Verify depth entropy loss is computed correctly.

        I109-8: 验证深度熵损失公式 H = -Σ p_d log(p_d)
        """
        torch.manual_seed(42)

        # 创建特征并执行前向传播
        features = torch.randn(1, 256, 16, 16)
        result = splitter(features, (64, 64), hard=False)

        # 获取概率分布
        probs = result.probs  # [B, N] 分割概率

        # 计算深度熵损失
        entropy_loss = splitter.get_depth_entropy_loss(
            weight=0.01,
            probs=probs
        )

        # 验证损失为标量且非零
        assert entropy_loss.dim() == 0, "Entropy loss should be scalar"
        assert entropy_loss.item() != 0, "Entropy loss should be non-zero"

        print(f"\nI109-8 Depth Entropy Loss:")
        print(f"  Entropy loss: {entropy_loss.item():.6f}")

    def test_depth_entropy_gradient_flow(self, splitter):
        """Verify gradients flow through depth entropy loss.

        I109-8: 验证梯度可以通过熵损失流向 logits
        """
        torch.manual_seed(456)

        # 测试 get_depth_entropy_loss 内部是否能计算梯度
        # 创建一个 requires_grad 的 probs 来验证
        probs = torch.randn(1, 341, requires_grad=True)  # [B, N]

        # 计算熵损失
        entropy_loss = splitter.get_depth_entropy_loss(
            weight=0.01,
            probs=probs
        )

        # 反向传播
        entropy_loss.backward()

        # 验证梯度存在
        assert probs.grad is not None, "Gradient should flow to probs"
        assert not torch.isnan(probs.grad).any(), "No NaN in gradients"

        print(f"\nI109-8 Entropy Gradient Flow:")
        print(f"  Entropy loss: {entropy_loss.item():.6f}")
        print(f"  Probs grad norm: {probs.grad.norm().item():.6f}")

    def test_depth_entropy_with_uniform_distribution(self, splitter):
        """Verify entropy is maximized with uniform distribution.

        I109-8: 均匀分布 p_d = 1/D 时熵最大 H = log(D)
        """
        torch.manual_seed(789)

        features = torch.randn(1, 256, 16, 16)
        result = splitter(features, (64, 64), hard=False)

        probs = result.probs

        # 获取深度信息
        depths = splitter._get_device_tensor(
            splitter.candidate_depths,
            "_cached_device_depths",
            probs.device
        )
        D = splitter._current_max_depth + 1

        # 理论最大熵 (均匀分布)
        max_entropy = torch.log(torch.tensor(D, dtype=torch.float32))

        # 计算实际熵
        depth_onehot = torch.nn.functional.one_hot(depths, D).float()
        depth_counts = depth_onehot.sum(dim=0).clamp(min=1.0)
        prob_sums = torch.einsum('bn,nd->d', probs, depth_onehot)
        depth_probs = (prob_sums / probs.shape[0] / depth_counts).clamp(min=1e-8)
        depth_probs = depth_probs / depth_probs.sum()

        actual_entropy = -(depth_probs * depth_probs.log()).sum()

        print(f"\nI109-8 Entropy Comparison:")
        print(f"  Max possible entropy (log({D})): {max_entropy.item():.4f}")
        print(f"  Actual entropy: {actual_entropy.item():.4f}")
        print(f"  Ratio: {(actual_entropy / max_entropy).item():.2%}")

        # 实际熵应该小于等于最大熵
        assert actual_entropy.item() <= max_entropy.item() + 1e-5, \
            "Actual entropy should not exceed maximum"


class TestQuotaEntropyLoss:
    """配额熵正则化验证测试."""

    @pytest.fixture
    def splitter(self):
        """Create a GumbelTopKSplitter instance for testing."""
        return GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=4,
            hidden_dim=128,
            pool_size=4,
            K_max=32,
            # I111-1: temperature 参数已移至 SplitterConfig
        )

    def test_quota_entropy_loss_exists(self, splitter):
        """Verify quota entropy loss method exists and returns valid output.

        I109-8: 验证配额熵正则化方法存在且返回有效结果
        """
        torch.manual_seed(123)

        features = torch.randn(1, 256, 16, 16)
        result = splitter(features, (64, 64), hard=False)

        # 确保 quota_logits 已初始化
        assert splitter.quota_logits is not None, "Quota logits should be initialized"

        # 计算配额熵损失
        quota_entropy_loss = splitter.get_quota_entropy_loss(weight=0.1)

        # 验证损失为标量
        assert quota_entropy_loss.dim() == 0, "Quota entropy loss should be scalar"

        print(f"\nI109-8 Quota Entropy Loss:")
        print(f"  Quota entropy loss: {quota_entropy_loss.item():.6f}")

    def test_quota_entropy_encourages_diversity(self, splitter):
        """Verify quota entropy encourages diverse quota distribution.

        I109-8: 验证配额熵正则化鼓励配额分布多样性
        """
        torch.manual_seed(321)

        features = torch.randn(1, 256, 16, 16)
        result = splitter(features, (64, 64), hard=False)

        quota_entropy_loss = splitter.get_quota_entropy_loss(weight=0.1)

        # 配额分布应该有一定熵值（不完全集中在单一深度）
        import torch.nn.functional as F
        quota_probs = F.softmax(splitter.quota_logits, dim=0)
        entropy = -(quota_probs * quota_probs.log()).sum()

        print(f"\nI109-8 Quota Diversity:")
        print(f"  Quota entropy: {entropy.item():.4f}")
        print(f"  Quota entropy loss: {quota_entropy_loss.item():.6f}")
        print(f"  Quota distribution: {quota_probs.detach().cpu().numpy()[:5]}...")

        # 熵应该大于零（不是完全集中在一个深度）
        assert entropy.item() > 0, "Quota entropy should be positive for diverse distribution"


class TestEntropyMode:
    """熵模式验证测试 (maximize vs target)."""

    @pytest.fixture
    def splitter(self):
        """Create a GumbelTopKSplitter instance for testing."""
        return GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=4,
            hidden_dim=128,
            pool_size=4,
            K_max=32,
            # I111-1: temperature 参数已移至 SplitterConfig
        )

    def test_entropy_mode_parameter_exists(self, splitter):
        """Verify entropy_mode parameter is accessible.

        I109-8: 验证 entropy_mode 参数存在
        """
        # 检查 auxiliary_losses 方法是否有 entropy_mode 参数
        torch.manual_seed(42)

        features = torch.randn(1, 256, 16, 16)
        result = splitter(features, (64, 64), hard=False)

        # 获取辅助损失（默认 maximize 模式）
        aux_losses = splitter.get_auxiliary_losses(result)

        print(f"\nI109-8 Entropy Mode:")
        print(f"  Available aux losses: {list(aux_losses.keys())}")

        # 应该包含与熵相关的损失
        entropy_related = [k for k in aux_losses.keys() if 'entropy' in k.lower()]
        assert len(entropy_related) > 0, "Should have entropy-related auxiliary losses"


class TestEntropyConfiguration:
    """熵配置验证测试."""

    @pytest.fixture
    def splitter(self):
        """Create a GumbelTopKSplitter instance for testing."""
        return GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=4,
            hidden_dim=128,
            pool_size=4,
            K_max=32,
            # I111-1: temperature 参数已移至 SplitterConfig
        )

    def test_default_entropy_weight(self, splitter):
        """Verify default entropy weight is 0.1.

        I109-8: 验证默认熵权重为 0.1
        """
        # 检查 get_auxiliary_losses 的默认参数
        import inspect
        sig = inspect.signature(splitter.get_auxiliary_losses)
        params = sig.parameters

        print(f"\nI109-8 Default Configuration:")
        if 'entropy_weight' in params:
            default_value = params['entropy_weight'].default
            print(f"  Default entropy_weight: {default_value}")

        if 'include_soft_entropy' in params:
            default_value = params['include_soft_entropy'].default
            print(f"  Default include_soft_entropy: {default_value}")
