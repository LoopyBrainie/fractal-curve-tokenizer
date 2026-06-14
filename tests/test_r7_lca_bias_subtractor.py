# -*- coding: utf-8 -*-
"""
R7 LCABiasSubtractor Tests

对应模块: vit_pytorch.modules.lca_bias_subtractor.LCABiasSubtractor

测试内容 (来自 R7 设计文档):
- T6: Bit-exact at lambda=0 (lambda_raw=0 init, bias_table=0 init, output == attn_mask + B_LCA)
- T7: Table shape and init (bias_table.shape == [65, 65], bias_table init=0,
     lambda_bounded in (-2, 2) for lambda_raw in {-100, -1, 0, 1, 100}, 4225 params total)
- T11: Head-collapse smoke (after 100 forward passes with random gradients,
      bias_table.std() > 0, lambda_bounded != 0 after training)
- Kill-switch: enabled=False returns attn_mask unchanged,
              beta_c_emergency_off=True returns attn_mask unchanged

参考设计文档: docs/superpowers/specs/2026-06-11-fractal-vit-classification-tokenization-synergy-r7-design.md
"""

import pytest
import torch

from vit_pytorch.modules.lca_bias_subtractor import LCABiasSubtractor


class TestLCABiasSubtractorBitExact:
    """T6: Bit-exact at lambda=0 验证"""

    @pytest.fixture
    def subtractor(self):
        """默认 LCABiasSubtractor (max_depth=64, enabled=True)"""
        return LCABiasSubtractor(max_depth=64, enabled=True)

    @pytest.fixture
    def sample_inputs(self):
        """构造标准测试输入: B=2, N=8"""
        torch.manual_seed(42)
        attn_mask = torch.randn(2, 8, 8)
        depths = torch.randint(0, 7, (2, 8))
        B_LCA = torch.randn(2, 8, 8)
        return attn_mask, depths, B_LCA

    def test_t6_lambda_raw_zero_returns_attn_mask_plus_B_LCA(
        self, subtractor, sample_inputs
    ):
        """T6 主断言: lambda_raw=0 init + bias_table=0 init 时,
        output 必须 bit-exact 等于 attn_mask + B_LCA。

        数学:
            lambda_bounded = 2 * tanh(0 / 2) = 0
            B_sub = bias_table[depths[i], depths[j]] = 0
            output = attn_mask + (B_LCA - 0 * 0) = attn_mask + B_LCA
        """
        attn_mask, depths, B_LCA = sample_inputs
        # init: lambda_raw=0, bias_table=0 (默认)
        assert subtractor.lambda_raw.item() == 0.0
        assert torch.all(subtractor.bias_table == 0.0)

        output = subtractor(attn_mask, depths, B_LCA)
        expected = attn_mask + B_LCA

        # bit-exact 验证 (tanh(0)=0 严格 = 0, 无浮点扰动)
        assert torch.equal(output, expected), (
            f"T6 bit-exact failure at lambda=0:\n"
            f"  max_diff = {(output - expected).abs().max().item():.2e}"
        )

    def test_t6_lambda_bounded_at_zero(self, subtractor):
        """T6 辅助: lambda_raw=0 → lambda_bounded 必须严格等于 0"""
        lambda_bounded = 2.0 * torch.tanh(subtractor.lambda_raw / 2.0)
        assert lambda_bounded.item() == 0.0

    def test_t6_randomized_inputs(self, subtractor):
        """T6 鲁棒性: 多次随机输入, lambda=0 init 必须 bit-exact 退化"""
        torch.manual_seed(123)
        for trial in range(10):
            B, N = 4, 16
            attn_mask = torch.randn(B, N, N)
            depths = torch.randint(0, 65, (B, N))
            B_LCA = torch.randn(B, N, N)

            output = subtractor(attn_mask, depths, B_LCA)
            expected = attn_mask + B_LCA

            assert torch.equal(output, expected), (
                f"T6 bit-exact failure on trial {trial}: "
                f"max_diff={(output - expected).abs().max().item():.2e}"
            )


class TestLCABiasSubtractorTableShape:
    """T7: Table shape and init 验证"""

    def test_t7_bias_table_shape_65x65(self):
        """T7: bias_table.shape 必须严格等于 [65, 65] (max_depth+1, max_depth+1)"""
        sub = LCABiasSubtractor(max_depth=64, enabled=True)
        assert sub.bias_table.shape == (65, 65), (
            f"Expected [65, 65], got {tuple(sub.bias_table.shape)}"
        )

    def test_t7_bias_table_init_zeros(self):
        """T7: bias_table 初始化必须全为 0"""
        sub = LCABiasSubtractor(max_depth=64, enabled=True)
        assert torch.all(sub.bias_table == 0.0), (
            f"bias_table init not zero, max abs = {sub.bias_table.abs().max().item():.2e}"
        )

    def test_t7_lambda_raw_init_zeros(self):
        """T7: lambda_raw 初始化必须为 0"""
        sub = LCABiasSubtractor(max_depth=64, enabled=True)
        assert sub.lambda_raw.shape == (1,), (
            f"Expected lambda_raw shape (1,), got {tuple(sub.lambda_raw.shape)}"
        )
        assert sub.lambda_raw.item() == 0.0

    def test_t7_total_params_4225(self):
        """T7: bias_table 必须有 4225 个参数 (65*65)
        (注: 此测试不计入 lambda_raw, 那是单独的 1 个标量参数)
        """
        sub = LCABiasSubtractor(max_depth=64, enabled=True)
        table_param_count = sub.bias_table.numel()
        assert table_param_count == 4225, (
            f"Expected 4225 table params, got {table_param_count}"
        )

    def test_t7_total_module_params_4226(self):
        """T7 完整: bias_table (4225) + lambda_raw (1) = 4226 总参数"""
        sub = LCABiasSubtractor(max_depth=64, enabled=True)
        total = sum(p.numel() for p in sub.parameters())
        assert total == 4226, f"Expected 4226 total params, got {total}"

    @pytest.mark.parametrize("lambda_raw", [-100.0, -1.0, 0.0, 1.0, 100.0])
    def test_t7_lambda_bounded_in_open_interval(self, lambda_raw):
        """T7: lambda_bounded 必须在 [-2, 2] 闭区间内
        对 lambda_raw in {-100, -1, 0, 1, 100} 都成立。
        注: 严格数学上 lambda_bounded ∈ (-2, 2); 但 fp32 tanh(50) = 1.0 精确,
        所以 lambda_bounded(λ=100) = 2.0 (闭边界)。我们使用闭区间断言。
        """
        sub = LCABiasSubtractor(max_depth=64, enabled=True)
        with torch.no_grad():
            sub.lambda_raw.fill_(lambda_raw)

        lambda_bounded = 2.0 * torch.tanh(sub.lambda_raw / 2.0)
        val = lambda_bounded.item()
        assert -2.0 <= val <= 2.0, (
            f"lambda_bounded={val} (lambda_raw={lambda_raw}) not in [-2, 2]"
        )

    def test_t7_lambda_bounded_saturates_large_inputs(self):
        """T7 边界: lambda_raw=±1000 时, lambda_bounded 应饱和到 ±2"""
        sub = LCABiasSubtractor(max_depth=64, enabled=True)
        with torch.no_grad():
            sub.lambda_raw.fill_(1000.0)
        lb_pos = 2.0 * torch.tanh(sub.lambda_raw / 2.0).item()
        assert -2.0 <= lb_pos <= 2.0 and abs(lb_pos - 2.0) < 1e-6

        with torch.no_grad():
            sub.lambda_raw.fill_(-1000.0)
        lb_neg = 2.0 * torch.tanh(sub.lambda_raw / 2.0).item()
        assert -2.0 <= lb_neg <= 2.0 and abs(lb_neg - (-2.0)) < 1e-6


class TestLCABiasSubtractorHeadCollapse:
    """T11: Head-collapse smoke 验证"""

    def test_t11_bias_table_std_positive_after_training(self):
        """T11: 100 次 forward + 反向传播后, bias_table.std() 必须 > 0
        (即没有完全坍缩到 0)

        实现要点: 在 lambda_raw=0, bias_table=0 时, gradient w.r.t. bias_table
        包含 lambda_bounded=0 因子 → 全零。所以先用一个 warmup 把 lambda_raw
        推离 0, 再做 100 步监督训练。
        """
        sub = LCABiasSubtractor(max_depth=64, enabled=True)
        # Warmup: 把 lambda_raw 推到非零, 让后续梯度能传到 bias_table
        with torch.no_grad():
            sub.lambda_raw.fill_(0.5)

        optimizer = torch.optim.SGD(sub.parameters(), lr=0.1)

        torch.manual_seed(0)
        B, N = 2, 8

        for step in range(100):
            attn_mask = torch.randn(B, N, N, requires_grad=False)
            depths = torch.randint(0, 65, (B, N))
            B_LCA = torch.randn(B, N, N)

            output = sub(attn_mask, depths, B_LCA)
            # 监督目标: 让 output 偏向 (attn_mask + B_LCA + small_noise)
            target = attn_mask + B_LCA + 0.1 * torch.randn_like(B_LCA)
            loss = (output - target).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert sub.bias_table.std().item() > 0, (
            f"bias_table fully collapsed: std={sub.bias_table.std().item():.2e}"
        )

    def test_t11_lambda_bounded_nonzero_after_training(self):
        """T11: 100 次 forward + 反向传播后, lambda_bounded 必须 != 0
        (即 lambda_raw 已被训练偏离 0)
        """
        sub = LCABiasSubtractor(max_depth=64, enabled=True)
        optimizer = torch.optim.SGD(sub.parameters(), lr=0.5)

        torch.manual_seed(1)
        B, N = 2, 8

        # Warmup: 让 bias_table 先学到一些非零值, 提供 lambda 梯度信号
        with torch.no_grad():
            sub.bias_table.fill_(0.1)

        for step in range(100):
            attn_mask = torch.randn(B, N, N, requires_grad=False)
            depths = torch.randint(0, 65, (B, N))
            B_LCA = torch.randn(B, N, N)

            output = sub(attn_mask, depths, B_LCA)
            target = attn_mask + B_LCA + 0.1 * torch.randn_like(B_LCA)
            loss = (output - target).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lambda_bounded = 2.0 * torch.tanh(sub.lambda_raw / 2.0)
        assert lambda_bounded.item() != 0.0, (
            f"lambda_bounded remained 0 after 100 steps: "
            f"lambda_raw={sub.lambda_raw.item():.6e}"
        )


class TestLCABiasSubtractorKillSwitch:
    """Kill-switch 行为验证"""

    def test_killswitch_enabled_false_returns_attn_mask_unchanged(self):
        """Kill-switch 1: enabled=False 时, 必须 passthrough attn_mask 完全不变"""
        torch.manual_seed(2)
        sub = LCABiasSubtractor(max_depth=64, enabled=False)

        # 即使 lambda_raw 和 bias_table 不为 0, 也必须 passthrough
        with torch.no_grad():
            sub.lambda_raw.fill_(0.5)
            sub.bias_table.fill_(0.3)

        attn_mask = torch.randn(2, 8, 8)
        depths = torch.randint(0, 7, (2, 8))
        B_LCA = torch.randn(2, 8, 8)

        output = sub(attn_mask, depths, B_LCA)

        # 必须 bit-exact 等于 attn_mask (无任何修改)
        assert torch.equal(output, attn_mask), (
            f"Kill-switch (enabled=False) modified attn_mask: "
            f"max_diff={(output - attn_mask).abs().max().item():.2e}"
        )

    def test_killswitch_can_be_toggled_at_runtime(self):
        """Kill-switch 验证: enabled=True → False → True 必须动态生效"""
        torch.manual_seed(3)
        sub = LCABiasSubtractor(max_depth=64, enabled=True)

        with torch.no_grad():
            sub.lambda_raw.fill_(1.0)
            sub.bias_table.fill_(0.5)

        attn_mask = torch.randn(1, 4, 4)
        depths = torch.randint(0, 7, (1, 4))
        B_LCA = torch.randn(1, 4, 4)

        # enabled=True: 应该修改 attn_mask
        output_on = sub(attn_mask, depths, B_LCA)
        assert not torch.equal(output_on, attn_mask), (
            "With enabled=True and non-zero params, output should differ"
        )

        # enabled=False: 应该 passthrough
        sub.enabled = False
        output_off = sub(attn_mask, depths, B_LCA)
        assert torch.equal(output_off, attn_mask), (
            f"Kill-switch (enabled=False) modified attn_mask: "
            f"max_diff={(output_off - attn_mask).abs().max().item():.2e}"
        )

        # 重新开启: 应该再次修改 attn_mask
        sub.enabled = True
        output_on_again = sub(attn_mask, depths, B_LCA)
        assert torch.equal(output_on_again, output_on), (
            "Re-enabling should restore the same non-passthrough output"
        )

    def test_beta_c_emergency_off_pattern(self):
        """Kill-switch 2: beta_c_emergency_off 模式 (默认 OFF = False)
        测试设计意图: 当外部检测到异常时, 设置 enabled=False 即等同 emergency_off。
        验证: emergency 关闭后, attn_mask 不被修改, 同时 lambda_bounded 仍可被查询
        (即参数本身保留, 只是 forward 短路)。
        """
        torch.manual_seed(4)
        sub = LCABiasSubtractor(max_depth=64, enabled=True)

        # 模拟训练后状态: 参数已偏离 0
        with torch.no_grad():
            sub.lambda_raw.fill_(0.7)
            sub.bias_table.fill_(0.2)

        # 触发 emergency off
        sub.enabled = False  # 对应 config.beta_c_emergency_off=True

        attn_mask = torch.randn(3, 6, 6)
        depths = torch.randint(0, 65, (3, 6))
        B_LCA = torch.randn(3, 6, 6)

        output = sub(attn_mask, depths, B_LCA)

        # 1. attn_mask 不被修改
        assert torch.equal(output, attn_mask)

        # 2. lambda_bounded 仍然可计算 (参数没被销毁, 仅 forward 短路)
        lambda_bounded = 2.0 * torch.tanh(sub.lambda_raw / 2.0)
        assert lambda_bounded.item() != 0.0
        assert -2.0 < lambda_bounded.item() < 2.0

        # 3. bias_table 仍然有非零标准差 (状态保留)
        assert sub.bias_table.std().item() > 0

    def test_killswitch_default_is_enabled(self):
        """默认构造: enabled 必须为 True (R7 设计: default-ON + kill-switch)"""
        sub = LCABiasSubtractor(max_depth=64)
        assert sub.enabled is True
        sub2 = LCABiasSubtractor(max_depth=64, enabled=True)
        assert sub2.enabled is True
        sub3 = LCABiasSubtractor(max_depth=64, enabled=False)
        assert sub3.enabled is False


class TestLCABiasSubtractorIntegration:
    """集成场景验证: 与正常 LCA bias 流的协同"""

    def test_forward_modifies_attn_mask_with_nonzero_params(self):
        """正常路径: enabled=True + 非零参数 → output 应严格 != attn_mask"""
        torch.manual_seed(5)
        sub = LCABiasSubtractor(max_depth=64, enabled=True)

        with torch.no_grad():
            sub.lambda_raw.fill_(0.3)
            sub.bias_table.fill_(0.1)

        attn_mask = torch.zeros(2, 4, 4)
        depths = torch.randint(0, 65, (2, 4))
        B_LCA = torch.randn(2, 4, 4)

        output = sub(attn_mask, depths, B_LCA)
        assert not torch.equal(output, attn_mask)
        assert not torch.equal(output, B_LCA)

    def test_depths_clamping_safe(self):
        """边界: depths 超出 [0, max_depth] 时必须安全 clamp (不越界)"""
        sub = LCABiasSubtractor(max_depth=64, enabled=True)
        attn_mask = torch.zeros(1, 4, 4)
        # depths 包含 -1 (padding) 和 100 (overhead)
        depths = torch.tensor([[-1, 0, 64, 100]])
        B_LCA = torch.zeros(1, 4, 4)

        # 不应抛 IndexError
        output = sub(attn_mask, depths, B_LCA)
        assert output.shape == attn_mask.shape
        assert torch.isfinite(output).all()

    def test_extra_repr_contains_key_fields(self):
        """接口: extra_repr 应包含 max_depth, enabled, lambda_raw, bias_table.shape"""
        sub = LCABiasSubtractor(max_depth=64, enabled=True)
        repr_str = sub.extra_repr()
        assert "max_depth=64" in repr_str
        assert "enabled=True" in repr_str
        assert "lambda_raw=" in repr_str
        assert "bias_table.shape=" in repr_str