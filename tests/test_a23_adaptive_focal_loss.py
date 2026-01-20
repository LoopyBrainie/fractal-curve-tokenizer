"""A23: Adaptive Focal Loss 单元测试.

测试 AdaptiveFocalLossWrapper 的各项功能:
- 不同自适应模式的正确性
- γ 值的范围约束
- 与标准 FocalLoss 的一致性
- 滑动窗口统计
"""

import math
from typing import Tuple

import pytest
import torch
import torch.nn as nn

from examples.training.losses.adaptive_focal_loss import (
    AdaptiveFocalLossWrapper,
    create_adaptive_focal_loss,
)
from examples.training.losses.focal_loss import FocalLoss


class TestAdaptiveFocalLossWrapper:
    """AdaptiveFocalLossWrapper 测试类."""

    def _create_test_data(
        self, batch_size: int = 32, num_classes: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建测试数据."""
        logits = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        return logits, targets

    def test_fixed_mode_basic(self):
        """固定模式测试 (退化为标准 FocalLoss)."""
        logits, targets = self._create_test_data()

        # fixed 模式应该与标准 FocalLoss 结果一致
        loss_fn = AdaptiveFocalLossWrapper(
            base_gamma=2.0,
            adaptive_mode="fixed",
        )
        focal_loss = FocalLoss(gamma=2.0)

        adaptive_result = loss_fn(logits, targets)
        standard_result = focal_loss(logits, targets)

        assert torch.allclose(adaptive_result, standard_result, atol=1e-5), (
            f"Fixed mode should match standard FocalLoss: "
            f"{adaptive_result.item()} vs {standard_result.item()}"
        )

    def test_difficulty_mode_gamma_adaptation(self):
        """难度模式 γ 自适应测试."""
        logits, targets = self._create_test_data(num_classes=10)

        loss_fn = AdaptiveFocalLossWrapper(
            base_gamma=2.0,
            adaptive_mode="difficulty",
            gamma_min=1.0,
            gamma_max=5.0,
        )

        # 初始 γ 应该接近 base_gamma
        initial_gamma = loss_fn.get_current_gamma()
        assert abs(initial_gamma - 2.0) < 0.5, "Initial gamma should be close to base_gamma"

        # 调用后 γ 应该更新
        loss = loss_fn(logits, targets, epoch=1, total_epochs=10)
        updated_gamma = loss_fn.get_current_gamma()

        assert 1.0 <= updated_gamma <= 5.0, f"Gamma should be in range [1.0, 5.0], got {updated_gamma}"
        assert loss.item() > 0, "Loss should be positive"

    def test_annealing_mode_gamma_annealing(self):
        """退火模式 γ 随训练进度变化."""
        logits, targets = self._create_test_data()

        loss_fn = AdaptiveFocalLossWrapper(
            base_gamma=2.0,
            adaptive_mode="annealing",
            gamma_min=1.0,
            gamma_max=5.0,
        )

        # 训练早期: γ 应该接近 gamma_min
        loss_early = loss_fn(logits, targets, epoch=1, total_epochs=10)
        gamma_early = loss_fn.get_current_gamma()

        # 训练晚期: γ 应该接近 gamma_max
        loss_late = loss_fn(logits, targets, epoch=10, total_epochs=10)
        gamma_late = loss_fn.get_current_gamma()

        assert gamma_early <= gamma_late, (
            f"Gamma should increase with training progress: "
            f"early={gamma_early}, late={gamma_late}"
        )
        assert 1.0 <= gamma_early <= gamma_late <= 5.0, (
            f"Gamma should be in range [1.0, 5.0]: "
            f"early={gamma_early}, late={gamma_late}"
        )

    def test_combined_mode(self):
        """综合模式测试."""
        logits, targets = self._create_test_data(num_classes=100)

        loss_fn = AdaptiveFocalLossWrapper(
            base_gamma=2.0,
            adaptive_mode="combined",
            gamma_min=1.0,
            gamma_max=5.0,
        )

        # 调用多次模拟训练
        for epoch in range(1, 6):
            loss = loss_fn(logits, targets, epoch=epoch, total_epochs=5)
            gamma = loss_fn.get_current_gamma()

            assert 1.0 <= gamma <= 5.0, f"Gamma should be in range [1.0, 5.0], got {gamma}"
            assert loss.item() > 0, "Loss should be positive"

    def test_gamma_range_constraints(self):
        """γ 范围约束测试."""
        logits, targets = self._create_test_data()

        # 测试边界值
        loss_fn = AdaptiveFocalLossWrapper(
            base_gamma=100.0,  # 非常大的 base_gamma
            adaptive_mode="difficulty",
            gamma_min=1.0,
            gamma_max=5.0,
        )

        loss = loss_fn(logits, targets)
        gamma = loss_fn.get_current_gamma()

        assert gamma <= 5.0, f"Gamma should be clamped to max: {gamma}"
        assert gamma >= 1.0, f"Gamma should be clamped to min: {gamma}"

    def test_gradient_flow(self):
        """梯度流测试."""
        logits, targets = self._create_test_data()
        logits.requires_grad = True  # 确保需要梯度

        loss_fn = AdaptiveFocalLossWrapper(
            base_gamma=2.0,
            adaptive_mode="combined",
        )

        loss = loss_fn(logits, targets, epoch=3, total_epochs=5)
        loss.backward()

        # 检查 logits 有梯度
        assert logits.grad is not None, "Gradients should flow to logits"
        assert torch.isfinite(logits.grad).all(), "Gradients should be finite"

    def test_average_difficulty_tracking(self):
        """平均难度跟踪测试."""
        logits, targets = self._create_test_data()

        loss_fn = AdaptiveFocalLossWrapper(
            base_gamma=2.0,
            adaptive_mode="difficulty",
            difficulty_window=10,
        )

        # 多次调用后检查平均难度
        for _ in range(5):
            loss_fn(logits, targets)

        avg_difficulty = loss_fn.get_average_difficulty()

        # 熵应该在 [0, log(C)] 范围内
        max_entropy = math.log(10)  # num_classes = 10
        assert 0 <= avg_difficulty <= max_entropy, (
            f"Difficulty should be in range [0, {max_entropy}]: {avg_difficulty}"
        )

    def test_different_reduction_modes(self):
        """不同约简模式测试."""
        logits, targets = self._create_test_data()

        for reduction in ["mean", "sum", "none"]:
            loss_fn = AdaptiveFocalLossWrapper(
                base_gamma=2.0,
                adaptive_mode="fixed",
                reduction=reduction,
            )

            loss = loss_fn(logits, targets)

            if reduction == "none":
                assert loss.shape == logits.shape[:1], f"None reduction should return per-sample loss"
            else:
                assert loss.shape == torch.Size([]), f"{reduction} reduction should return scalar"

    def test_label_smoothing(self):
        """标签平滑测试."""
        logits, targets = self._create_test_data()

        loss_fn = AdaptiveFocalLossWrapper(
            base_gamma=2.0,
            adaptive_mode="fixed",
            label_smoothing=0.1,
        )

        loss = loss_fn(logits, targets)
        assert loss.item() > 0, "Loss should be positive with label smoothing"

    def test_alpha_parameter(self):
        """Alpha 参数测试."""
        logits, targets = self._create_test_data(num_classes=5)

        # 测试统一 alpha
        loss_fn_uniform = AdaptiveFocalLossWrapper(
            base_gamma=2.0,
            adaptive_mode="fixed",
            alpha=0.5,
        )
        loss_uniform = loss_fn_uniform(logits, targets)

        # 测试类别特定 alpha
        loss_fn_class = AdaptiveFocalLossWrapper(
            base_gamma=2.0,
            adaptive_mode="fixed",
            alpha=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
        loss_class = loss_fn_class(logits, targets)

        assert loss_uniform.item() > 0, "Uniform alpha loss should be positive"
        assert loss_class.item() > 0, "Class-specific alpha loss should be positive"

    def test_factory_function(self):
        """工厂函数测试."""
        loss_fn = create_adaptive_focal_loss(
            num_classes=10,
            base_gamma=2.0,
            adaptive_mode="combined",
            gamma_min=1.0,
            gamma_max=5.0,
            alpha=0.25,
        )

        assert isinstance(loss_fn, AdaptiveFocalLossWrapper), "Factory should return AdaptiveFocalLossWrapper"

        logits, targets = self._create_test_data()
        loss = loss_fn(logits, targets, epoch=3, total_epochs=5)

        assert loss.item() > 0, "Factory-created loss should compute valid loss"

    def test_extra_repr(self):
        """额外信息测试."""
        loss_fn = AdaptiveFocalLossWrapper(
            base_gamma=2.0,
            adaptive_mode="combined",
            gamma_min=1.0,
            gamma_max=5.0,
        )

        repr_str = repr(loss_fn)

        assert "base_gamma=2.0" in repr_str, "Repr should include base_gamma"
        assert "combined" in repr_str, "Repr should include adaptive_mode"
        assert "gamma_range=[1.0, 5.0]" in repr_str, "Repr should include gamma range"

    def test_device_handling(self):
        """设备处理测试."""
        loss_fn = AdaptiveFocalLossWrapper(
            base_gamma=2.0,
            adaptive_mode="difficulty",
        )

        # 测试 CPU
        logits_cpu, targets_cpu = self._create_test_data()
        loss_cpu = loss_fn(logits_cpu, targets_cpu)
        assert loss_cpu.device == torch.device("cpu"), "CPU loss should be on CPU"

        # 测试 GPU (如果可用)
        if torch.cuda.is_available():
            loss_fn_gpu = AdaptiveFocalLossWrapper(
                base_gamma=2.0,
                adaptive_mode="difficulty",
            ).cuda()

            logits_gpu = logits_cpu.cuda()
            targets_gpu = targets_cpu.cuda()

            loss_gpu = loss_fn_gpu(logits_gpu, targets_gpu)
            assert loss_gpu.device == torch.device("cuda"), "GPU loss should be on CUDA"

    def test_invalid_adaptive_mode(self):
        """无效自适应模式测试."""
        with pytest.raises(ValueError, match="adaptive_mode must be one of"):
            AdaptiveFocalLossWrapper(
                base_gamma=2.0,
                adaptive_mode="invalid_mode",
            )

    def test_invalid_gamma_range(self):
        """无效 γ 范围测试."""
        with pytest.raises(ValueError, match="gamma_min must be >= 0"):
            AdaptiveFocalLossWrapper(
                base_gamma=2.0,
                gamma_min=-1.0,
            )

        with pytest.raises(ValueError, match="gamma_max must be >= gamma_min"):
            AdaptiveFocalLossWrapper(
                base_gamma=2.0,
                gamma_min=5.0,
                gamma_max=1.0,
            )


class TestAdaptiveFocalLossIntegration:
    """Adaptive Focal Loss 集成测试."""

    def _create_test_data(
        self, batch_size: int = 32, num_classes: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建测试数据."""
        logits = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        return logits, targets

    def test_training_loop_simulation(self):
        """模拟训练循环测试."""
        batch_size, num_classes = 32, 100
        num_epochs = 5

        # 创建模型和损失函数
        model = nn.Linear(128, num_classes)
        loss_fn = AdaptiveFocalLossWrapper(
            base_gamma=2.0,
            adaptive_mode="combined",
            gamma_min=1.0,
            gamma_max=5.0,
        )

        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(1, num_epochs + 1):
            # 模拟训练数据
            x = torch.randn(batch_size, 128)
            y = torch.randint(0, num_classes, (batch_size,))

            # 前向传播
            logits = model(x)
            loss = loss_fn(logits, y, epoch=epoch, total_epochs=num_epochs)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 验证
            assert loss.item() > 0, f"Epoch {epoch}: Loss should be positive"
            gamma = loss_fn.get_current_gamma()
            assert 1.0 <= gamma <= 5.0, f"Epoch {epoch}: Gamma should be in range"

    def test_comparison_with_fixed_gamma(self):
        """与固定 γ 对比测试."""
        logits, targets = self._create_test_data()

        # 自适应模式
        adaptive_loss = AdaptiveFocalLossWrapper(
            base_gamma=2.0,
            adaptive_mode="difficulty",
        )

        # 固定模式
        fixed_loss = FocalLoss(gamma=2.0)

        loss_adaptive = adaptive_loss(logits, targets)
        loss_fixed = fixed_loss(logits, targets)

        # 两者都应该是有效的损失值
        assert torch.isfinite(loss_adaptive), "Adaptive loss should be finite"
        assert torch.isfinite(loss_fixed), "Fixed loss should be finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
