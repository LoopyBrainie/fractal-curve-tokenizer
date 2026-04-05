"""Unified Monitor - 整合所有监控组件

提供统一的监控接口，整合:
- GradientMonitor: 梯度范数监控
- LossMonitor: 损失组件监控
- NumericalDefender: 数值稳定性保护
- ActivationStatsCollector: 激活统计
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import torch.nn as nn

from ..metrics.collector import MetricsCollector

if TYPE_CHECKING:
    from ..trainer.state import TrainingStats


class UnifiedMonitor:
    """统一监控器

    整合所有监控组件到单一接口，简化训练循环中的监控代码。

    设计原则:
    - 所有子监控器共享同一个 MetricsCollector
    - 提供 pre_forward/post_forward/post_backward 生命周期钩子
    - 自动从 TrainingStats 提取并记录指标

    使用示例:
        collector = MetricsCollector(MetricRegistry.create_default())
        unified = UnifiedMonitor(model, collector, config)

        # 训练循环
        unified.pre_forward()
        outputs = model(images)
        unified.post_forward(outputs)
        loss.backward()
        should_proceed = unified.post_backward()
        if not should_proceed:
            optimizer.zero_grad()
            continue
    """

    def __init__(
        self,
        model: nn.Module,
        collector: MetricsCollector,
        config: Optional[dict] = None,
    ):
        """初始化统一监控器

        Args:
            model: 要监控的模型
            collector: MetricsCollector 实例
            config: 配置字典，包含以下键:
                - detect_anomaly: bool, 是否启用异常检测
                - skip_on_nan: bool, 是否在 NaN 时跳过优化器步骤
                - record_layer_norms: bool, 是否记录每层梯度范数
                - track_key_components: bool, 是否追踪关键组件梯度
        """
        self.model = model
        self.collector = collector
        self.config = config or {}

        # 延迟导入避免循环依赖
        from .gradient_monitor import GradientMonitor
        from .loss_monitor import LossMonitor
        from .numerical_defense import NumericalDefender, ActivationStatsCollector

        # 初始化子监控器
        self.gradient_monitor = GradientMonitor(
            model=model,
            record_layer_norms=self.config.get("record_layer_norms", True),
            hooks_enabled=self.config.get("hooks_enabled", True),
            track_key_components=self.config.get("track_key_components", True),
            collector=collector,
        )

        self.loss_monitor = LossMonitor(
            force_track_all=self.config.get("force_track_all", True),
            collector=collector,
        )

        self.defender = NumericalDefender(
            model=model,
            detect_anomaly=self.config.get("detect_anomaly", False),
            skip_on_nan=self.config.get("skip_on_nan", True),
            collector=collector,
        )

        self.activation_collector = ActivationStatsCollector(
            model=model,
            collector=collector,
        )

    def pre_forward(self) -> None:
        """前向传播前调用

        清除激活统计，准备捕获新的前向传播数据。
        """
        self.activation_collector.clear()

    def post_forward(self, outputs: "TrainingStats") -> None:
        """前向传播后调用

        1. 记录 auxiliary_outputs（layer-packaged → trainer-unpacked 架构）
        2. 记录 TrainingStats 直接字段中不在 auxiliary_outputs 里的指标

        Args:
            outputs: 模型前向传播返回的 TrainingStats
        """
        # 1. layer-packaged: 展平各层诊断包裹，自动解开任意嵌套结构
        self._record_auxiliary_outputs(outputs)

        # 2. TrainingStats 直接字段（不含 auxiliary_outputs 中已有的冗余字段）
        self._record_training_stats(outputs)

    def post_backward(self) -> bool:
        """反向传播后调用

        计算梯度范数并检查数值稳定性。

        Returns:
            True 表示继续优化器步骤，False 表示跳过
        """
        # 计算梯度范数
        self.gradient_monitor.compute_grad_norms()
        self.gradient_monitor.compute_component_grad_norms()

        # 数值防御检查
        should_proceed = self.defender.post_backward()

        return should_proceed

    def _record_auxiliary_outputs(self, stats: "TrainingStats") -> None:
        """自动解开并记录所有层的包裹（layer-packaged → trainer-unpacked 架构）

        训练器不知道也不需要知道各层返回了什么字段。
        只需调用 flatten_layer_outputs() 即可自动处理任意层输出结构。

        Args:
            stats: TrainingStats 实例
        """
        if not hasattr(stats, 'auxiliary_outputs') or not stats.auxiliary_outputs:
            return

        from vit_pytorch.core.layer_output import flatten_layer_outputs

        flat_metrics = flatten_layer_outputs(stats.auxiliary_outputs, prefix="train")

        for key, value in flat_metrics.items():
            self.collector.record(key, value)

    def _record_training_stats(self, stats: "TrainingStats") -> None:
        """从 TrainingStats 提取并记录所有指标

        Args:
            stats: TrainingStats 实例
        """
        # Token 指标
        if stats.num_tokens is not None:
            if isinstance(stats.num_tokens, (int, float)):
                self.collector.record("num_tokens", float(stats.num_tokens))
            elif hasattr(stats.num_tokens, 'item'):
                self.collector.record("num_tokens", stats.num_tokens.item())

        # 深度指标
        if stats.depth_used is not None:
            if isinstance(stats.depth_used, (int, float)):
                self.collector.record("depth_used", float(stats.depth_used))
            elif hasattr(stats.depth_used, 'item'):
                self.collector.record("depth_used", stats.depth_used.item())

        # 分类 logits
        if hasattr(stats, "logits") and stats.logits is not None:
            logits = stats.logits
            self.collector.record("logits_mean", float(logits.mean().item()))
            self.collector.record("logits_std", float(logits.std().item()))

        # Splitter 统计 → 已迁移至 auxiliary_outputs (train/splitter/logits_mean, active_ratio)

        # 流形偏置统计
        if stats.manifold_bias_max is not None:
            self.collector.record("manifold_bias_max", float(stats.manifold_bias_max))
        if stats.manifold_bias_mean is not None:
            self.collector.record("manifold_bias_mean", float(stats.manifold_bias_mean))
        if hasattr(stats, 'manifold_bias_std') and stats.manifold_bias_std is not None:
            self.collector.record("manifold_bias_std", float(stats.manifold_bias_std))

        # Poincare 距离统计
        if stats.poincare_dist_mean is not None:
            self.collector.record("poincare_dist_mean", float(stats.poincare_dist_mean))
        if hasattr(stats, 'poincare_dist_std') and stats.poincare_dist_std is not None:
            self.collector.record("poincare_dist_std", float(stats.poincare_dist_std))

        # 辅助损失
        if hasattr(stats, 'auxiliary_losses') and stats.auxiliary_losses:
            for name, value in stats.auxiliary_losses.items():
                if hasattr(value, 'item'):
                    value = value.item()
                self.collector.record(f"loss_{name}", float(value))

        # 额外的梯度相关指标
        if hasattr(stats, 'backbone_grad_norm') and stats.backbone_grad_norm is not None:
            self.collector.record("backbone_grad_norm", float(stats.backbone_grad_norm))
        if hasattr(stats, 'splitter_grad_norm') and stats.splitter_grad_norm is not None:
            self.collector.record("splitter_grad_norm", float(stats.splitter_grad_norm))

    def reset(self) -> None:
        """重置所有监控器状态"""
        self.gradient_monitor.reset()
        self.loss_monitor.reset()
        self.defender.reset()
        self.activation_collector.clear()
