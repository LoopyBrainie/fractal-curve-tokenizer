"""
I32-4 数学形式化验证测试

Tests for:
- Softplus 逆函数数学正确性
- τ=1.5 时 γ≈1.247 (非 Issue 声称的 0)
- 边界值稳定性

Mathematical verification:
    ∀ τ > 0: softplus(τ + log(1 - exp(-τ))) = τ
"""

import math
import pytest
import torch
import torch.nn.functional as F


class TestI32_4TemperatureInit:
    """I32-4: Gumbel温度初始化数学正确性验证"""

    def test_softplus_inverse_property(self):
        """验证 softplus 逆函数性质 ∀ τ > 0"""
        test_cases = [0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0, 100.0]

        for tau in test_cases:
            # 当前实现的公式
            gamma = tau + math.log(1 - math.exp(-tau))
            # 重建验证
            reconstructed = math.log(1 + math.exp(gamma))

            assert abs(reconstructed - tau) < 1e-10, \
                f"τ={tau}: 期望 {tau}, 得到 {reconstructed}"

    def test_tau_1_5_exact(self):
        """验证 Issue 中的具体案例 τ=1.5

        Issue 声称: τ=1.5 时 γ=0
        实际: τ=1.5 时 γ=1.2475, softplus(γ)=1.5
        """
        tau = 1.5
        gamma = tau + math.log(1 - math.exp(-tau))
        reconstructed = math.log(1 + math.exp(gamma))

        # 精确值验证
        assert abs(gamma - 1.2475) < 0.001, \
            f"γ 应约 1.247, 得到 {gamma}"

        assert abs(reconstructed - 1.5) < 1e-10, \
            f"应重建 τ=1.5, 得到 {reconstructed}"

    def test_extreme_values(self):
        """验证极端值稳定性"""
        # τ → 0+
        tau = 1e-6
        gamma = tau + math.log(1 - math.exp(-tau))
        reconstructed = math.log(1 + math.exp(gamma))
        assert abs(reconstructed - tau) < 1e-10

        # τ = 100
        tau = 100.0
        gamma = tau + math.log(1 - math.exp(-tau))
        reconstructed = math.log(1 + math.exp(gamma))
        assert abs(reconstructed - tau) < 1e-6

    def test_lca_temperature_init_direct(self):
        """直接验证 LCAHilbertBias 温度初始化"""
        from vit_pytorch.attn_hilbert_bias import LCAHilbertBias

        # 创建模块
        bias = LCAHilbertBias(
            max_depth=8,
            heads=8,
            lca_temperature=1.5,
            learnable_temperature=True
        )

        # 验证 γ 值
        gamma = bias._lca_temp_raw[0].item()
        assert abs(gamma - 1.2475) < 0.001, \
            f"γ 应约 1.247, 得到 {gamma}"

        # 验证 τ = softplus(γ)
        tau = bias.lca_temperature[0].item()
        assert abs(tau - 1.5) < 0.01, \
            f"τ 应约 1.5, 得到 {tau}"

    def test_lca_temperature_fixed_mode(self):
        """验证固定温度模式"""
        from vit_pytorch.attn_hilbert_bias import LCAHilbertBias

        # 固定温度模式
        bias = LCAHilbertBias(
            max_depth=8,
            heads=8,
            lca_temperature=1.5,
            learnable_temperature=False
        )

        assert bias._lca_temp_raw is None
        # I34-16: _lca_temp_fixed 现在是 Tensor buffer
        assert bias._lca_temp_fixed is not None
        assert bias._lca_temp_fixed.shape == (8,)
        assert torch.allclose(bias._lca_temp_fixed, torch.tensor(1.5))

        tau = bias.lca_temperature
        # I34-16: lca_temperature 现在返回 (H,) 向量
        assert tau.shape == (8,)
        assert torch.allclose(tau, torch.tensor(1.5))

    def test_lca_temperature_disabled(self):
        """验证温度禁用模式"""
        from vit_pytorch.attn_hilbert_bias import LCAHilbertBias

        # 禁用温度
        bias = LCAHilbertBias(
            max_depth=8,
            heads=8,
            lca_temperature=None,
            learnable_temperature=True
        )

        assert bias._lca_temp_raw is None
        assert bias._lca_temp_fixed is None
        assert bias.lca_temperature is None


class TestI32_4MathematicalProof:
    """I32-4: 数学证明验证"""

    def test_softplus_inverse_derivation(self):
        """验证 softplus 逆函数推导

        定理: γ = τ + log(1 - exp(-τ)) 是 softplus 的逆函数

        证明:
        softplus(γ) = log(1 + exp(γ))
                    = log(1 + exp(τ + log(1 - exp(-τ))))
                    = log(1 + exp(τ) * (1 - exp(-τ)))
                    = log(1 + exp(τ) - 1)
                    = log(exp(τ))
                    = τ
        """
        for tau in [0.5, 1.0, 1.5, 2.0]:
            gamma = tau + math.log(1 - math.exp(-tau))
            softplus_gamma = math.log(1 + math.exp(gamma))
            assert abs(softplus_gamma - tau) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
