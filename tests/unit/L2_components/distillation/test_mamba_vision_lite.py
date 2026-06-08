# -*- coding: utf-8 -*-
"""
L2 Components: MambaVision-Lite Student Tests (B.10, v1.3 §9.7)

对应模块: vit_pytorch.layers.distillation.mamba_vision_lite

测试内容:
- 参数预算: nn.Parameter 总数 ∈ [target - tolerance, target + tolerance]
- 蒸馏损失: α·KL + (1-α)·CE 输出 scalar tensor
- O(1) 推理: 不同 spatial size 下前向均产生 (B, num_classes) 输出
  且无任何 growing cache (无 KV-cache / SSM-state 累积)
"""

import pytest
import torch
import torch.nn as nn

from vit_pytorch.core.constants import (
    MAMBA_LITE_DISTILL_ALPHA,
    MAMBA_LITE_PARAMS_TARGET,
    MAMBA_LITE_PARAMS_TOLERANCE,
)
from vit_pytorch.layers.distillation import MambaVisionLiteStudent


class TestMambaVisionLiteStudent:
    """MambaVision-Lite student 蒸馏插件验证 (B.10)"""

    @pytest.fixture
    def teacher(self):
        """Dummy teacher — student does not invoke it."""
        return nn.Linear(10, 1000)

    @pytest.fixture
    def student(self, teacher):
        """Create student with default config (target ≈ 44M)."""
        return MambaVisionLiteStudent(teacher)

    def test_mamba_lite_param_count(self, student):
        """参数预算: 实际 nn.Parameter 数 ∈ [43M, 45M] (= target ± tolerance).

        v1.3 standard: MAMBA_LITE_PARAMS_TARGET = 44_000_000,
        MAMBA_LITE_PARAMS_TOLERANCE = 1_000_000.
        """
        n_params = student.count_parameters()
        lower = MAMBA_LITE_PARAMS_TARGET - MAMBA_LITE_PARAMS_TOLERANCE
        upper = MAMBA_LITE_PARAMS_TARGET + MAMBA_LITE_PARAMS_TOLERANCE

        assert lower <= n_params <= upper, (
            f"Student params {n_params:,} not in budget "
            f"[{lower:,}, {upper:,}] "
            f"(target={MAMBA_LITE_PARAMS_TARGET:,}, "
            f"tolerance={MAMBA_LITE_PARAMS_TOLERANCE:,})"
        )

    def test_mamba_lite_distillation_loss_runs(self, student):
        """蒸馏损失: L = α·KL(T‖S) + (1-α)·CE(S, y)  →  scalar tensor.

        验证:
        - loss 是 scalar (0-dim tensor)
        - α 等于 MAMBA_LITE_DISTILL_ALPHA (0.7)
        - loss 可反向传播 (gradient 不为 None)
        """
        B, K = 4, 1000
        student_logits = torch.randn(B, K, requires_grad=True)
        teacher_logits = torch.randn(B, K)
        labels = torch.randint(0, K, (B,))

        loss = student.compute_distillation_loss(
            student_logits, teacher_logits, labels
        )

        # scalar check
        assert loss.dim() == 0, f"loss must be scalar, got shape {loss.shape}"
        assert torch.isfinite(loss), f"loss must be finite, got {loss.item()}"

        # alpha wiring
        assert student.distill_alpha == MAMBA_LITE_DISTILL_ALPHA == 0.7

        # gradient flows back
        loss.backward()
        assert student_logits.grad is not None
        assert torch.isfinite(student_logits.grad).all()

    def test_mamba_lite_inference_state_is_o1(self, student):
        """O(1) 推理: 不同 spatial size 下前向均稳定, 无 growing cache.

        验证:
        - 不同 spatial size 均得到 (B, num_classes) 输出
        - student 不持有任何会随序列增长的缓存 (KV cache / SSM state)
        """
        student.eval()
        num_classes = student.num_classes

        # Verify NO cache attributes exist (no KV cache, no SSM state, no buffers)
        forbidden_buffer_keywords = ("cache", "kv", "state", "memory")
        for name, _buf in student.named_buffers():
            assert not any(kw in name.lower() for kw in forbidden_buffer_keywords), (
                f"Student must not hold growing-cache buffer: {name!r}"
            )

        # Forward at multiple spatial sizes
        with torch.no_grad():
            for spatial in (16, 32, 64, 128):
                x = torch.randn(1, 3, spatial, spatial)
                y = student(x)
                assert y.shape == (1, num_classes), (
                    f"spatial={spatial}: expected (1, {num_classes}), "
                    f"got {tuple(y.shape)}"
                )
                # Output should not depend on prior calls — O(1) state
                y2 = student(x)
                assert torch.allclose(y, y2), (
                    f"spatial={spatial}: outputs differ across calls — "
                    f"state is not O(1)"
                )
