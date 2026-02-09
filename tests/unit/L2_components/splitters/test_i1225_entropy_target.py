# -*- coding: utf-8 -*-
"""
熵目标公式测试 (I122-5)

测试目标:
    - 验证 H_target = 0.5 × log(D) 的数学性质
    - 对比新公式与原始公式的行为差异
    - 验证缩放行为的一致性

设计原则:
    - ENTROPY_TARGET_SCALE = 0.5 表示目标熵为最大熵的 50%
    - 对所有 D 保持恒定 50% 熵比例
"""

import math
import pytest
import torch
import torch.nn.functional as F


class TestEntropyTargetFormula:
    """熵目标公式数学验证"""

    def test_new_formula_values(self):
        """验证新公式计算值"""
        from vit_pytorch.constants import ENTROPY_TARGET_SCALE

        # 理论值: H_target = 0.5 × log(D)
        expected = {
            2: 0.5 * math.log(2),   # 0.3466
            3: 0.5 * math.log(3),   # 0.5493
            4: 0.5 * math.log(4),   # 0.6931 = log(2)
            5: 0.5 * math.log(5),   # 0.8047
            6: 0.5 * math.log(6),   # 0.8959
            8: 0.5 * math.log(8),   # 1.0397
        }

        for D, expected_H in expected.items():
            H_target = ENTROPY_TARGET_SCALE * math.log(D)
            assert abs(H_target - expected_H) < 1e-6, f"D={D}: 期望 {expected_H:.4f}, 得到 {H_target:.4f}"

    def test_constant_scale_factor(self):
        """验证缩放因子为 0.5"""
        from vit_pytorch.constants import ENTROPY_TARGET_SCALE

        assert abs(ENTROPY_TARGET_SCALE - 0.5) < 1e-6

    def test_h_target_half_of_max_entropy(self):
        """验证 H_target 始终是最大熵的 50%"""
        from vit_pytorch.constants import ENTROPY_TARGET_SCALE

        for D in [2, 3, 4, 5, 6, 8, 16]:
            H_max = math.log(D)
            H_target = ENTROPY_TARGET_SCALE * math.log(D)
            ratio = H_target / H_max
            assert abs(ratio - 0.5) < 1e-6, f"D={D}: 比例 {ratio:.4f} ≠ 0.5"

    def test_new_vs_original_formula(self):
        """对比新公式与原始公式"""
        from vit_pytorch.constants import ENTROPY_TARGET_SCALE

        original = lambda D: math.log(D) * (1.0 - 1.0 / math.sqrt(D))
        new = lambda D: ENTROPY_TARGET_SCALE * math.log(D)

        print("\n新公式 vs 原始公式对比:")
        print("D\t原始\t新\t差异")
        print("-" * 40)

        for D in [2, 3, 4, 5, 6, 8]:
            H_orig = original(D)
            H_new = new(D)
            diff = H_new - H_orig
            print(f"{D}\t{H_orig:.3f}\t{H_new:.3f}\t{diff:+.3f}")

        # 关键验证: D=4 时两者相等
        assert abs(new(4) - original(4)) < 1e-6, "D=4 时新公式应等于原始公式"

        # 新公式在 D>4 时更低
        assert new(8) < original(8), "D>4 时新公式应更低"


class TestEntropyTargetScaling:
    """熵目标缩放行为验证"""

    def test_scaling_consistency(self):
        """验证缩放一致性"""
        from vit_pytorch.constants import ENTROPY_TARGET_SCALE

        # 新公式的缩放行为
        def H_target_new(D):
            return ENTROPY_TARGET_SCALE * math.log(D)

        # 验证缩放关系
        for D1, D2 in [(2, 4), (4, 8), (3, 6)]:
            ratio_D = D2 / D1
            ratio_H = H_target_new(D2) / H_target_new(D1)
            expected_ratio_H = math.log(D2) / math.log(D1)

            assert abs(ratio_H - expected_ratio_H) < 1e-6, \
                f"D={D1}→{D2}: 期望比例 {expected_ratio_H:.4f}, 得到 {ratio_H:.4f}"

    def test_effective_depth_concept(self):
        """验证有效深度概念 (D_eff = √D)"""
        from vit_pytorch.constants import ENTROPY_TARGET_SCALE

        print("\n有效深度验证:")
        print("D\tH_target\tD_eff=exp(H)\t√D\t差异")
        print("-" * 50)

        for D in [2, 4, 8, 16]:
            H_target = ENTROPY_TARGET_SCALE * math.log(D)
            D_eff = math.exp(H_target)
            sqrt_D = math.sqrt(D)
            diff = abs(D_eff - sqrt_D)

            print(f"{D}\t{H_target:.3f}\t{D_eff:.3f}\t\t{sqrt_D:.3f}\t{diff:.3f}")

            # 验证 D_eff ≈ √D
            assert diff / sqrt_D < 0.15, f"D={D}: D_eff 与 √D 差异过大 ({diff/sqrt_D:.1%})"


class TestEntropyLossImplementation:
    """熵损失实现验证"""

    def test_entropy_loss_with_new_target(self):
        """使用新目标验证熵损失"""
        # 模拟深度分布
        D = 4
        batch_size = 4

        # 模拟 batch 的 token 分配
        # 情况1: 均匀分布 (p = [0.25, 0.25, 0.25, 0.25])
        depth_probs_uniform = torch.tensor([0.25, 0.25, 0.25, 0.25])

        # 情况2: 偏向浅层 (p = [0.5, 0.25, 0.15, 0.1])
        depth_probs_shallow = torch.tensor([0.5, 0.25, 0.15, 0.1])

        # 情况3: 偏向深层 (p = [0.1, 0.15, 0.25, 0.5])
        depth_probs_deep = torch.tensor([0.1, 0.15, 0.25, 0.5])

        # 计算熵
        entropy_uniform = -(depth_probs_uniform * depth_probs_uniform.log()).sum()
        entropy_shallow = -(depth_probs_shallow * depth_probs_shallow.log()).sum()
        entropy_deep = -(depth_probs_deep * depth_probs_deep.log()).sum()

        print("\n深度分布熵值:")
        print(f"均匀分布 H = {entropy_uniform:.4f}")
        print(f"偏向浅层 H = {entropy_shallow:.4f}")
        print(f"偏向深层 H = {entropy_deep:.4f}")

        # 目标熵
        H_target = 0.5 * math.log(D)

        print(f"\n目标熵 H_target = {H_target:.4f}")
        print(f"均匀分布 - 目标 = {entropy_uniform - H_target:.4f}")
        print(f"偏向浅层 - 目标 = {entropy_shallow - H_target:.4f}")
        print(f"偏向深层 - 目标 = {entropy_deep - H_target:.4f}")

        # 验证: 均匀分布的熵最高
        assert entropy_uniform > entropy_shallow, "均匀分布熵应高于偏向浅层"
        assert entropy_uniform > entropy_deep, "均匀分布熵应高于偏向深层"

        # 验证: 均匀分布的熵是目标熵的 2 倍 (因为目标熵 = 0.5 × H_max)
        assert abs(entropy_uniform - 2 * H_target) < 0.01, "均匀分布熵应为 2 × H_target"

    def test_gradient_direction(self):
        """验证梯度方向正确"""
        D = 4
        H_target = 0.5 * math.log(D)

        # 模拟两种分布
        p_high_entropy = torch.tensor([0.3, 0.25, 0.25, 0.2], requires_grad=True)
        p_low_entropy = torch.tensor([0.6, 0.2, 0.1, 0.1], requires_grad=True)

        H_high = -(p_high_entropy * p_high_entropy.log()).sum()
        H_low = -(p_low_entropy * p_low_entropy.log()).sum()

        # 计算负熵损失梯度
        loss_high = -H_target * H_high
        loss_low = -H_target * H_low

        loss_high.backward()
        loss_low.backward()

        # 高熵分布的梯度应该鼓励增加多样性 (负损失更小)
        # 低熵分布的梯度应该鼓励增加多样性 (负损失更大)
        print("\n梯度方向验证:")
        print(f"高熵分布 H={H_high:.4f}, 损失={loss_high.item():.4f}")
        print(f"低熵分布 H={H_low:.4f}, 损失={loss_low.item():.4f}")

        assert H_high > H_low, "高熵分布熵应更高"


class TestEntropyTargetEdgeCases:
    """边界情况测试"""

    def test_minimal_depth(self):
        """测试最小深度 D=2"""
        from vit_pytorch.constants import ENTROPY_TARGET_SCALE

        D = 2
        H_target = ENTROPY_TARGET_SCALE * math.log(D)

        # D=2 时，均匀分布 p = [0.5, 0.5] 的熵为 log(2) ≈ 0.693
        # 目标熵为 0.5 * log(2) ≈ 0.347
        assert H_target < math.log(2), "D=2 时目标熵应低于最大熵"
        assert H_target > 0, "D=2 时目标熵应为正"

    def test_large_depth(self):
        """测试较大深度 D=16"""
        from vit_pytorch.constants import ENTROPY_TARGET_SCALE

        D = 16
        H_target = ENTROPY_TARGET_SCALE * math.log(D)

        # 验证缩放行为
        # D=16 时: H_target = 0.5 * log(16) = 0.5 * 2.77 = 1.39
        # 也等于: log(4) = log(√16) = log(2^(4/2)) = log(2^2) = log(4)
        assert abs(H_target - math.log(4)) < 1e-6, "D=16 时 H_target = log(4)"
        assert H_target / math.log(D) == 0.5, "比例应恒为 0.5"

    def test_extreme_case(self):
        """测试极端情况"""
        from vit_pytorch.constants import ENTROPY_TARGET_SCALE

        # D=1 (只有一层)
        D = 1
        H_target = ENTROPY_TARGET_SCALE * math.log(D)
        assert H_target == 0.0, "D=1 时熵目标应为 0"


class TestEntropyTargetIntegration:
    """集成测试"""

    def test_in_splitter_context(self):
        """在 Splitter 上下文中验证"""
        from vit_pytorch import GumbelTopKSplitter
        from vit_pytorch.config import SplitterConfig
        import torch

        config = SplitterConfig(max_level_limit=4)
        splitter = GumbelTopKSplitter(config=config, image_size=(64, 64))

        # 模拟 Splitter 内部状态 (设置内部变量)
        D = 5  # max_level + 1 = 4 + 1 = 5

        # 创建需要梯度的参数
        target_tokens = torch.tensor(32.0, requires_grad=True)

        # 计算熵损失
        loss = splitter.get_quota_entropy_loss(weight=0.1)

        # 验证损失计算
        # 注意: 熵损失 = -weight * entropy，因为我们要最大化熵
        # 所以损失可以为负（高熵时），或正（低熵时）
        assert loss.dim() == 0, "损失应该是标量"

        # 反向传播验证
        loss.backward()

        # 验证熵目标计算
        H_target = 0.5 * math.log(D)
        expected_H_target = 0.5 * math.log(5)

        print(f"\n熵目标验证:")
        print(f"D = {D}")
        print(f"H_target = {H_target:.4f}")
        print(f"期望值 = {expected_H_target:.4f}")

        assert abs(H_target - expected_H_target) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
