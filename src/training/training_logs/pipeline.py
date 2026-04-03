"""Metrics Pipeline - 连接 MetricsCollector 和 EpochLogger

提供统一的指标记录接口，将 MetricsCollector 的数据转换为 EpochLogger 格式。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..metrics.collector import MetricsCollector
    from .epoch_logger import EpochLogger


class MetricsPipeline:
    """将 MetricsCollector 的数据转换为 EpochLogger 格式

    设计原则:
    - 延迟转换：只在 log_epoch 时才聚合和转换
    - 自动重置：log_epoch 后自动重置 collector

    使用示例:
        collector = MetricsCollector(MetricRegistry.create_default())
        logger = EpochLogger(output_dir)
        pipeline = MetricsPipeline(collector, logger)

        # 每个 epoch
        pipeline.log_epoch(epoch, eval_metrics)
    """

    def __init__(
        self,
        collector: "MetricsCollector",
        logger: "EpochLogger",
    ):
        """初始化 MetricsPipeline

        Args:
            collector: MetricsCollector 实例
            logger: EpochLogger 实例
        """
        self.collector = collector
        self.logger = logger

    def log_epoch(
        self,
        epoch: int,
        eval_metrics: Optional[dict] = None,
        extra_train_metrics: Optional[dict] = None,
    ) -> None:
        """将当前 epoch 的指标记录到 logger

        从 collector 获取汇总指标，转换为兼容格式后传递给 logger。

        Args:
            epoch: 当前 epoch 编号
            eval_metrics: 评估指标字典（可选）
            extra_train_metrics: 额外的训练指标（可选，会覆盖自动计算的指标）
        """
        # 获取汇总指标
        summary = self.collector.get_summary()

        # 构建训练指标字典（兼容格式）
        train_metrics = {
            # 核心指标
            "loss": summary.get("loss", 0.0),
            "grad_norm": summary.get("grad_norm", 0.0),
            "learning_rate": summary.get("learning_rate", 0.0),
            # Token 指标
            "num_tokens": summary.get("num_tokens", 0.0),
            "avg_tokens": summary.get("num_tokens_mean", 0.0),
            # 深度指标
            "depth_used": summary.get("depth_used", 0.0),
            # Splitter 指标
            "splitter_logits_mean": summary.get("splitter_logits_mean", 0.0),
            "splitter_logits_std": summary.get("splitter_logits_std", 0.0),
            "active_ratio": summary.get("active_ratio", 0.0),
            "mean_abs_logits": summary.get("mean_abs_logits", 0.0),
            # 流形指标
            "manifold_bias_max": summary.get("manifold_bias_max", 0.0),
            "manifold_bias_mean": summary.get("manifold_bias_mean", 0.0),
            "poincare_dist_mean": summary.get("poincare_dist_mean", 0.0),
            # 梯度指标
            "backbone_grad_norm": summary.get("backbone_grad_norm", 0.0),
            "splitter_grad_norm": summary.get("splitter_grad_norm", 0.0),
            "entmax_grad_norm": summary.get("entmax_grad_norm", 0.0),
            "manifold_decoder_grad_norm": summary.get("manifold_decoder_grad_norm", 0.0),
            "backbone_vs_splitter_grad_ratio": summary.get("backbone_vs_splitter_grad_ratio", 0.0),
            # 数值稳定性指标
            "nan_count": summary.get("nan_count", 0.0),
            "inf_count": summary.get("inf_count", 0.0),
            "issue_count": summary.get("issue_count", 0.0),
        }

        # 应用额外的训练指标（会覆盖自动计算的）
        if extra_train_metrics:
            train_metrics.update(extra_train_metrics)

        # 记录到 logger
        self.logger.log(epoch, train_metrics, eval_metrics)

        # 重置 collector
        self.collector.reset()
