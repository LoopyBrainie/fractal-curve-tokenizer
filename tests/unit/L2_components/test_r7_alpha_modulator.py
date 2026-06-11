# -*- coding: utf-8 -*-
"""
AlphaModulator (R7) - 单元测试

数学形式化
==========
    alpha_bounded = 2 * tanh(alpha_raw / 2)    ∈ (-2, 2)
    rho = 1 + alpha_bounded * (H / H_max - 0.5)
    rho = clamp(rho, min=eps, max=2.0)         防御性钳制
    M = rho * logit_scale

测试套件 (依据 R7 设计文档 §Testing Strategy):
    T1 Bit-exact at alpha=0        — 验证恒等映射 (alpha_raw=0 init)
    T2 Bound                       — 验证 tanh 界 (-2, 2)
    T3 Gradient flow               — 验证 alpha_raw.grad 非零、有限
    T4 Per-step clamp on rho       — 验证钳制边界不引起梯度死亡
    T5 Reparam identity            — 验证 M = logit_scale @ alpha_raw=0
"""

from __future__ import annotations

import importlib.util
import math
import pathlib

import pytest
import torch

# 直接加载 alpha_modulator.py, 绕过 vit_pytorch/__init__.py 的链式导入
_SRC_PATH = pathlib.Path(__file__).resolve().parents[3] / "src" / "vit_pytorch" / "modules" / "alpha_modulator.py"
_spec = importlib.util.spec_from_file_location("alpha_modulator", _SRC_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load spec for {_SRC_PATH}")
_alpha_modulator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_alpha_modulator)
AlphaModulator = _alpha_modulator.AlphaModulator


# =============================================================================
# T1: Bit-exact at alpha=0
# =============================================================================

class TestT1BitExactAtAlphaZero:
    """T1 — alpha_raw=0 init 时 M == logit_scale (fp32 1e-6)"""

    def test_alpha_raw_initialized_to_zero(self):
        """alpha_raw 初始化应为 0 (确保 bit-exact identity)"""
        logit_scale = 0.125
        mod = AlphaModulator(logit_scale=logit_scale)

        assert mod.alpha_raw.shape == (1,)
        assert mod.alpha_raw.requires_grad
        assert mod.alpha_raw.item() == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize(
        "H_value,H_max",
        [
            (0.0, 1.0),         # 边界: H=0
            (0.25, 1.0),        # 浅层
            (0.5, 1.0),         # 中点 (rho 应为 1.0)
            (0.75, 1.0),        # 深层
            (1.0, 1.0),         # 边界: H=H_max
            (3.0, 6.0),         # 一般浮点
            (0.123, 1.0),       # 不规整值
        ],
    )
    def test_M_equals_logit_scale_for_scalar_H(self, H_value, H_max):
        """标量 H: alpha_raw=0 → M == logit_scale within fp32 1e-6"""
        logit_scale = 0.125
        mod = AlphaModulator(logit_scale=logit_scale)

        H = torch.tensor([H_value], dtype=torch.float32)
        M = mod(H, H_max)

        assert M.shape == (1,)
        expected = torch.full_like(H, logit_scale)
        assert torch.allclose(M, expected, atol=1e-6, rtol=0), (
            f"Bit-exact failed: M={M.item()}, expected={logit_scale}, "
            f"diff={abs(M.item() - logit_scale):.2e}"
        )

    @pytest.mark.parametrize("B", [1, 2, 4, 8])
    def test_M_equals_logit_scale_for_batched_H(self, B):
        """批量 H: alpha_raw=0 → M == logit_scale * ones_like(H) within fp32 1e-6"""
        logit_scale = 0.25
        mod = AlphaModulator(logit_scale=logit_scale)

        # 随机 H ∈ [0, H_max]
        H_max = 6.0
        H = torch.rand(B, dtype=torch.float32) * H_max

        M = mod(H, H_max)

        assert M.shape == (B,)
        expected = torch.full_like(H, logit_scale)
        assert torch.allclose(M, expected, atol=1e-6, rtol=0), (
            f"Batched bit-exact failed: max diff = {(M - expected).abs().max():.2e}"
        )

    def test_different_H_max_values(self):
        """不同 H_max 输入都应给出 M == logit_scale (alpha_raw=0)"""
        logit_scale = 0.1
        mod = AlphaModulator(logit_scale=logit_scale)
        H = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.float32)

        for H_max in [1.0, 4.0, 8.0, 16.0]:
            M = mod(H, H_max)
            expected = torch.full_like(H, logit_scale)
            assert torch.allclose(M, expected, atol=1e-6, rtol=0), (
                f"H_max={H_max}: M={M.tolist()}, expected={logit_scale}"
            )


# =============================================================================
# T2: Bound — alpha_bounded ∈ (-2, 2)
# =============================================================================

class TestT2Bound:
    """T2 — alpha_bounded = 2 * tanh(alpha_raw / 2) ∈ (-2, 2) for alpha_raw ∈ {-100,-1,0,1,100}"""

    @pytest.mark.parametrize("alpha_raw_value", [-100.0, -1.0, 0.0, 1.0, 100.0])
    def test_alpha_bounded_in_open_interval(self, alpha_raw_value):
        """alpha_bounded ∈ (-2, 2) 且对每个 alpha_raw 值都成立 (闭区间端点: tanh 饱和值)"""
        logit_scale = 1.0
        mod = AlphaModulator(logit_scale=logit_scale)

        with torch.no_grad():
            mod.alpha_raw.fill_(alpha_raw_value)

        alpha_bounded = 2.0 * torch.tanh(mod.alpha_raw / 2.0)

        assert alpha_bounded.shape == (1,)
        assert alpha_bounded.requires_grad  # 来自 alpha_raw 参数

        value = alpha_bounded.item()
        # tanh 严格在 (-1, 1) 内, 乘 2 → (-2, 2) 开区间
        # 但 fp32 下 alpha_raw=±100 时 tanh 饱和 → α_bounded ≈ ±2.0 (在 1e-10 量级误差内)
        # 这里使用 ≤/≥ 允许闭区间端点 (R7 设计允许钳制后达到边界)
        assert -2.0 - 1e-6 <= value <= 2.0 + 1e-6, (
            f"alpha_bounded={value} not in [-2, 2] for alpha_raw={alpha_raw_value}"
        )
        # 但严格 tanh 上不能超出
        if abs(alpha_raw_value) < 50.0:
            assert -2.0 < value < 2.0, (
                f"alpha_bounded={value} not strictly in (-2, 2) for alpha_raw={alpha_raw_value}"
            )

    @pytest.mark.parametrize(
        "alpha_raw_value,expected_sign",
        [
            (-100.0, -1),   # 极负 → 应接近 -2
            (-1.0, -1),
            (0.0, 0),       # 0 → 应为 0
            (1.0, 1),
            (100.0, 1),     # 极正 → 应接近 +2
        ],
    )
    def test_alpha_bounded_sign_and_magnitude(self, alpha_raw_value, expected_sign):
        """alpha_bounded 的符号与量级应符合 tanh 行为"""
        mod = AlphaModulator(logit_scale=1.0)

        with torch.no_grad():
            mod.alpha_raw.fill_(alpha_raw_value)

        alpha_bounded = 2.0 * torch.tanh(mod.alpha_raw / 2.0).item()

        if expected_sign == 0:
            assert alpha_bounded == pytest.approx(0.0, abs=1e-6)
        elif expected_sign > 0:
            # fp32 下极值 alpha_raw=100 时 tanh 饱和到 +2.0
            assert 0.0 < alpha_bounded <= 2.0
        else:
            assert -2.0 <= alpha_bounded < 0.0

    def test_alpha_bounded_at_extreme_negative_saturates_near_minus_two(self):
        """alpha_raw=-100: alpha_bounded ≈ -2 (tanh 饱和)"""
        mod = AlphaModulator(logit_scale=1.0)

        with torch.no_grad():
            mod.alpha_raw.fill_(-100.0)

        alpha_bounded = (2.0 * torch.tanh(mod.alpha_raw / 2.0)).item()

        # tanh(-50) ≈ -1.0 + 2e-43 → 2 * tanh ≈ -2 (fp32 下饱和到 -2)
        assert alpha_bounded == pytest.approx(-2.0, abs=1e-6)
        # 严格上仍 ≤ 2 (由 tanh 性质)
        assert alpha_bounded >= -2.0

    def test_alpha_bounded_at_extreme_positive_saturates_near_two(self):
        """alpha_raw=+100: alpha_bounded ≈ +2 (tanh 饱和)"""
        mod = AlphaModulator(logit_scale=1.0)

        with torch.no_grad():
            mod.alpha_raw.fill_(100.0)

        alpha_bounded = (2.0 * torch.tanh(mod.alpha_raw / 2.0)).item()

        assert alpha_bounded == pytest.approx(2.0, abs=1e-6)
        assert alpha_bounded <= 2.0

    def test_M_bounded_under_logit_scale(self):
        """M 的范围在 logit_scale 维度上同理有界: |M| < 2 * |logit_scale|"""
        logit_scale = 0.5
        mod = AlphaModulator(logit_scale=logit_scale)

        H = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
        H_max = 1.0

        with torch.no_grad():
            mod.alpha_raw.fill_(100.0)  # 极正 → 应让 |rho| 接近 2

        M = mod(H, H_max)

        # M = rho * logit_scale, 其中 rho ∈ (0, 2] → M ∈ (0, 2 * logit_scale]
        assert (M >= 0.0).all()
        assert (M <= 2.0 * logit_scale + 1e-6).all()


# =============================================================================
# T3: Gradient flow — loss.backward() updates alpha_raw.grad
# =============================================================================

class TestT3GradientFlow:
    """T3 — loss.backward() 应更新 alpha_raw.grad, 且 grad 非零、有限"""

    def test_gradient_updates_alpha_raw(self):
        """基本梯度流: 标量 loss 应回传至 alpha_raw"""
        logit_scale = 0.125
        mod = AlphaModulator(logit_scale=logit_scale)

        H = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float32)
        H_max = 1.0

        M = mod(H, H_max)
        # 构造标量 loss 使梯度有定义
        loss = M.sum()
        loss.backward()

        assert mod.alpha_raw.grad is not None, "alpha_raw.grad should not be None"
        assert mod.alpha_raw.grad.shape == (1,)
        assert torch.isfinite(mod.alpha_raw.grad).all(), (
            f"Non-finite gradient: {mod.alpha_raw.grad}"
        )

    def test_gradient_is_non_zero(self):
        """alpha_raw ≠ 0 时, grad 应非零 (确认参数被影响)"""
        logit_scale = 0.125
        mod = AlphaModulator(logit_scale=logit_scale)

        # 注入非零 alpha_raw
        with torch.no_grad():
            mod.alpha_raw.fill_(0.5)

        # H 不对称 (全部 > H_max/2) → grad 非零
        H = torch.tensor([0.7, 0.8, 0.9], dtype=torch.float32)
        H_max = 1.0

        M = mod(H, H_max)
        loss = M.sum()
        loss.backward()

        assert mod.alpha_raw.grad is not None
        assert mod.alpha_raw.grad.abs().item() > 1e-9, (
            f"Gradient magnitude too small: {mod.alpha_raw.grad.item():.2e}"
        )

    def test_gradient_zero_at_alpha_raw_zero_with_homogeneous_H(self):
        """alpha_raw=0 且 H 全等于中点 0.5: rho=1, ∂M/∂alpha_raw = 0"""
        logit_scale = 0.125
        mod = AlphaModulator(logit_scale=logit_scale)

        H_max = 1.0
        # H 全部为 H_max/2 → ratio=0.5 → rho=1 (无 alpha_raw 依赖)
        H = torch.full((4,), H_max / 2.0, dtype=torch.float32)

        M = mod(H, H_max)
        loss = M.sum()
        loss.backward()

        assert mod.alpha_raw.grad is not None
        assert mod.alpha_raw.grad.item() == pytest.approx(0.0, abs=1e-9), (
            f"Expected zero grad at H=H_max/2, got {mod.alpha_raw.grad.item():.2e}"
        )

    def test_gradient_via_logits_chain(self):
        """完整链路: pooled → logits = pooled * M → loss"""
        logit_scale = 0.125
        mod = AlphaModulator(logit_scale=logit_scale)

        # 模拟 pooled 张量 (来自上游, requires_grad)
        pooled = torch.randn(3, 10, dtype=torch.float32, requires_grad=True)
        H = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float32)
        H_max = 1.0

        M = mod(H, H_max)  # [B]
        # logits = pooled * M.unsqueeze(-1) → [B, num_classes]
        logits = pooled * M.unsqueeze(-1)

        # 伪 labels + CE 损失
        labels = torch.tensor([0, 1, 2], dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()

        assert mod.alpha_raw.grad is not None
        assert torch.isfinite(mod.alpha_raw.grad).all()
        assert mod.alpha_raw.grad.abs().item() > 1e-9, (
            f"Chain-through grad too small: {mod.alpha_raw.grad.item():.2e}"
        )
        # 上游梯度也应存在
        assert pooled.grad is not None
        assert torch.isfinite(pooled.grad).all()

    def test_gradient_does_not_propagate_to_H(self):
        """R7: H 在 forward 中被 detach → 不应回传梯度至 H"""
        logit_scale = 0.125
        mod = AlphaModulator(logit_scale=logit_scale)

        H = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float32, requires_grad=True)
        H_max = 1.0

        M = mod(H, H_max)
        loss = M.sum()
        loss.backward()

        # H 被 detach, 所以 H.grad 应为 None
        assert H.grad is None, "H should be STE-safe (detached); grad should not propagate"


# =============================================================================
# T4: Per-step clamp on rho — 不引起梯度死亡
# =============================================================================

class TestT4PerStepClamp:
    """T4 — 验证钳制边界不引起梯度死亡"""

    def test_rho_in_safe_range_keeps_gradient_flowing(self):
        """rho 在 (EPS, 2.0) 内时, 梯度正常回传"""
        logit_scale = 0.125
        mod = AlphaModulator(logit_scale=logit_scale)

        # alpha_bounded=1.5, H=0.0 → rho = 1 + 1.5*(-0.5) = 0.25 ∈ (EPS, 2.0)
        with torch.no_grad():
            # alpha_raw ≈ 2 * atanh(1.5/2) = 2 * atanh(0.75)
            alpha_raw_val = 2.0 * math.atanh(0.75)
            mod.alpha_raw.fill_(alpha_raw_val)

        # 使用单一 H 值避免对称抵消
        H = torch.tensor([0.9, 1.0], dtype=torch.float32)
        H_max = 1.0

        M = mod(H, H_max)
        loss = M.sum()
        loss.backward()

        assert mod.alpha_raw.grad is not None
        assert torch.isfinite(mod.alpha_raw.grad).all()
        assert mod.alpha_raw.grad.abs().item() > 1e-9, (
            f"Gradient died at safe rho: {mod.alpha_raw.grad.item():.2e}"
        )

    @pytest.mark.parametrize("alpha_raw_value", [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0])
    def test_clamp_does_not_cause_gradient_death(self, alpha_raw_value):
        """即使 alpha_raw 极值, 钳制不应引起 grad 死亡 (tanh 已保证 bounded)

        注: tanh 在大输入下饱和 → 梯度按 tanh' 缩小, 这是 tanh 平滑性的预期行为,
        不是 "死亡"。我们仅验证梯度是有限 (finite) 且非 NaN。
        钳制边界本身: rho 在 (EPS, 2.0] 内 → clamp 无效 → 梯度来自 tanh 链。
        """
        logit_scale = 0.125
        mod = AlphaModulator(logit_scale=logit_scale)

        with torch.no_grad():
            mod.alpha_raw.fill_(alpha_raw_value)

        # 使用全为 H_max 的 H → ratio=1.0, (ratio-0.5)=0.5, 偏置固定
        H = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        H_max = 1.0

        M = mod(H, H_max)
        loss = M.sum()
        loss.backward()

        assert mod.alpha_raw.grad is not None
        # 关键: 梯度应有限 (无 NaN/Inf)
        assert torch.isfinite(mod.alpha_raw.grad).all(), (
            f"Non-finite grad: alpha_raw={alpha_raw_value}, "
            f"grad={mod.alpha_raw.grad.item():.2e}"
        )
        # 在 tanh 非饱和区 (|alpha_raw| < ~5), 梯度应有非平凡量级
        if abs(alpha_raw_value) <= 2.0:
            assert mod.alpha_raw.grad.abs().item() > 1e-9, (
                f"Gradient died in tanh non-saturated region: alpha_raw={alpha_raw_value}, "
                f"grad={mod.alpha_raw.grad.item():.2e}"
            )

    def test_clamp_keeps_rho_in_eps_to_2(self):
        """无论 alpha_raw/H 怎么取, rho ∈ (EPS, 2.0] (钳制后)"""
        logit_scale = 0.125
        mod = AlphaModulator(logit_scale=logit_scale)

        with torch.no_grad():
            mod.alpha_raw.fill_(1000.0)  # 极值, 让 tanh 饱和

        # 故意构造可能触发下界的 H 值
        H = torch.tensor([0.0], dtype=torch.float32)
        H_max = 1.0

        M = mod(H, H_max)
        rho = (M / logit_scale).item()

        # rho 应在钳制范围内
        assert rho >= mod.eps - 1e-9, f"rho={rho} below eps={mod.eps}"
        assert rho <= 2.0 + 1e-6, f"rho={rho} above 2.0"

    def test_gradient_sign_is_consistent(self):
        """梯度符号应反映: ∂M/∂alpha_raw ∝ (ratio - 0.5) * sech²(α/2)

        对于 H > H_max/2 (深层), ratio - 0.5 > 0 → 梯度为正 (与 alpha_raw 符号无关).
        对于 H < H_max/2 (浅层), ratio - 0.5 < 0 → 梯度为负.
        """
        logit_scale = 0.125
        mod_deep = AlphaModulator(logit_scale=logit_scale)
        mod_shallow = AlphaModulator(logit_scale=logit_scale)

        with torch.no_grad():
            mod_deep.alpha_raw.fill_(0.5)
            mod_shallow.alpha_raw.fill_(0.5)

        H_max = 1.0
        H_deep = torch.tensor([0.8, 0.9], dtype=torch.float32)        # ratio - 0.5 > 0
        H_shallow = torch.tensor([0.1, 0.2], dtype=torch.float32)     # ratio - 0.5 < 0

        M_deep = mod_deep(H_deep, H_max).sum()
        M_shallow = mod_shallow(H_shallow, H_max).sum()

        M_deep.backward()
        M_shallow.backward()

        # 深层: 梯度为正
        assert mod_deep.alpha_raw.grad.item() > 0, (
            f"Expected positive grad at H>H_max/2, got {mod_deep.alpha_raw.grad.item():.2e}"
        )
        # 浅层: 梯度为负
        assert mod_shallow.alpha_raw.grad.item() < 0, (
            f"Expected negative grad at H<H_max/2, got {mod_shallow.alpha_raw.grad.item():.2e}"
        )


# =============================================================================
# T5: Reparam identity — alpha_raw=0 → M = logit_scale
# =============================================================================

class TestT5ReparamIdentity:
    """T5 — M = ρ * logit_scale 是 1 个有效参数 (reparam 验证)"""

    def test_single_learnable_parameter(self):
        """AlphaModulator 应只有 1 个可学习参数: alpha_raw"""
        mod = AlphaModulator(logit_scale=0.125)

        params = list(mod.parameters())
        assert len(params) == 1, (
            f"Expected exactly 1 learnable param (alpha_raw), got {len(params)}"
        )
        assert params[0] is mod.alpha_raw
        assert params[0].shape == (1,)

    def test_reparam_identity_at_alpha_zero_scalar(self):
        """alpha_raw=0 → M == logit_scale (标量 H)"""
        logit_scale = 0.125
        mod = AlphaModulator(logit_scale=logit_scale)

        H = torch.tensor([0.7], dtype=torch.float32)
        H_max = 1.0

        M = mod(H, H_max)

        assert M.shape == (1,)
        # M 应精确等于 logit_scale (bit-exact)
        assert M.item() == pytest.approx(logit_scale, abs=1e-6)

    def test_reparam_identity_at_alpha_zero_batched(self):
        """alpha_raw=0 → M == logit_scale * ones_like(H) (批量 H)"""
        logit_scale = 0.125
        mod = AlphaModulator(logit_scale=logit_scale)

        # 多种 H_max 下
        for H_max_val in [1.0, 4.0, 8.0]:
            H = torch.linspace(0.0, H_max_val, 10, dtype=torch.float32)
            M = mod(H, H_max_val)

            expected = torch.full_like(H, logit_scale)
            assert torch.allclose(M, expected, atol=1e-6, rtol=0), (
                f"H_max={H_max_val}: M={M.tolist()[:3]}..., expected={logit_scale}"
            )

    def test_no_extra_state_in_module(self):
        """模块不应有非参数的额外状态 (除 logit_scale/eps 标量外)"""
        mod = AlphaModulator(logit_scale=0.125)

        # logit_scale/eps 是配置, 不是 Parameter
        assert not isinstance(mod.logit_scale, torch.nn.Parameter)
        assert not isinstance(mod.eps, torch.nn.Parameter)

        # 缓冲区应为空 (无 running stats 等)
        assert len(mod._buffers) == 0

    def test_M_equals_logit_scale_for_various_H(self):
        """α=0 时, M 应 = logit_scale (与 H/H_max 输入无关) — reparam 恒等性"""
        logit_scale = 0.3
        mod = AlphaModulator(logit_scale=logit_scale)

        # 边界值 + 中间值 + 不规整值
        test_cases = [
            (0.0, 1.0),
            (0.001, 1.0),
            (0.5, 1.0),
            (0.999, 1.0),
            (1.0, 1.0),
            (1.5, 3.0),
            (3.14, 6.28),
            (0.123456789, 1.0),
        ]

        for H_val, H_max in test_cases:
            H = torch.tensor([H_val], dtype=torch.float32)
            M = mod(H, H_max)

            assert M.item() == pytest.approx(logit_scale, abs=1e-6), (
                f"H={H_val}, H_max={H_max}: M={M.item()}, expected={logit_scale}"
            )

    def test_extra_repr_includes_alpha_raw(self):
        """extra_repr 应展示 alpha_raw 当前值 (便于调试)"""
        mod = AlphaModulator(logit_scale=0.125)
        with torch.no_grad():
            mod.alpha_raw.fill_(0.5)

        repr_str = mod.extra_repr()
        assert "logit_scale=0.125" in repr_str
        assert "alpha_raw=0.500000" in repr_str