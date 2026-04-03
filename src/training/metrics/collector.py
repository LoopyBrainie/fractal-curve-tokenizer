"""统一指标收集管道"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .record import MetricRecord, MetricSource
from .registry import MetricRegistry, MetricSpec


@dataclass
class SimpleAggregator:
    """简单聚合器 - 维护 running mean, latest, max

    用于在收集过程中实时聚合指标值。
    """
    values: list[float] = field(default_factory=list)

    def update(self, value: float) -> None:
        """添加新值"""
        self.values.append(value)

    def mean(self) -> float:
        """计算均值"""
        return sum(self.values) / len(self.values) if self.values else 0.0

    def latest(self) -> float:
        """获取最新值"""
        return self.values[-1] if self.values else 0.0

    def max(self) -> float:
        """获取最大值"""
        return max(self.values) if self.values else 0.0

    def min(self) -> float:
        """获取最小值"""
        return min(self.values) if self.values else 0.0

    def std(self) -> float:
        """计算标准差"""
        if len(self.values) < 2:
            return 0.0
        mean = self.mean()
        variance = sum((v - mean) ** 2 for v in self.values) / len(self.values)
        return variance ** 0.5

    def reset(self) -> None:
        """清空所有历史值"""
        self.values.clear()


class MetricsCollector:
    """统一指标收集管道

    接收来自各监控组件的指标，记录并聚合。

    设计原则:
    - 线程不安全（单线程训练循环使用）
    - 所有指标通过同一接口记录
    - 自动创建聚合器维护统计
    - 支持派生指标血缘追踪

    使用示例:
        registry = MetricRegistry.create_default()
        collector = MetricsCollector(registry)

        # 记录原始指标
        collector.record("loss", 1.234)
        collector.record("grad_norm", 0.567)

        # 记录派生指标（指定父指标血缘）
        collector.record_derived(
            "loss_ratio",
            0.1,
            parents=["loss", "grad_norm"]
        )

        # 获取聚合摘要
        summary = collector.get_summary()
        print(summary)
        # {'loss': 1.234, 'loss_mean': 1.234, 'loss_max': 1.234,
        #  'grad_norm': 0.567, 'grad_norm_mean': 0.567, ...}
    """

    def __init__(self, registry: Optional[MetricRegistry] = None):
        """初始化收集器

        Args:
            registry: 可选的指标注册中心。如果为 None，则创建空注册表。
        """
        self.registry = registry or MetricRegistry()
        self._records: dict[str, list[MetricRecord]] = defaultdict(list)
        self._aggregators: dict[str, SimpleAggregator] = {}
        self._step = 0
        self._epoch = 0

    def record(self, name: str, value: float, **metadata) -> None:
        """记录原始指标值

        自动创建 MetricRecord 并更新聚合器。

        Args:
            name: 指标名称
            value: 指标值
            metadata: 额外元数据
        """
        spec = self.registry.get(name)
        source = spec.source if spec else MetricSource.COMPUTED

        record = MetricRecord(
            name=name,
            value=value,
            source=source,
            lineage=(),
            step=self._step,
            epoch=self._epoch,
            metadata=metadata,
        )
        self._records[name].append(record)

        # 确保聚合器存在并更新
        if name not in self._aggregators:
            self._aggregators[name] = SimpleAggregator()
        self._aggregators[name].update(value)

    def record_derived(
        self,
        name: str,
        value: float,
        parents: list[str],
        **metadata
    ) -> None:
        """记录派生指标及其血缘

        用于记录由其他指标计算得出的衍生指标。

        Args:
            name: 派生指标名称
            value: 派生指标值
            parents: 父指标名称列表
            metadata: 额外元数据
        """
        record = MetricRecord(
            name=name,
            value=value,
            source=MetricSource.COMPUTED,
            lineage=tuple(parents),
            step=self._step,
            epoch=self._epoch,
            metadata=metadata,
        )
        self._records[name].append(record)

        if name not in self._aggregators:
            self._aggregators[name] = SimpleAggregator()
        self._aggregators[name].update(value)

    def set_step(self, step: int, epoch: int) -> None:
        """设置当前步和轮次

        Args:
            step: 当前训练步数
            epoch: 当前训练轮次
        """
        self._step = step
        self._epoch = epoch

    def get_summary(self) -> dict[str, float]:
        """获取 epoch 汇总指标

        根据注册时定义的聚合类型，为每个指标生成对应的汇总值。

        Returns:
            汇总指标字典，格式为 {metric_name: latest_value} 或 {metric_name_aggType: agg_value}
        """
        summary = {}
        for name, agg in self._aggregators.items():
            spec = self.registry.get(name)
            if spec:
                for agg_type in spec.aggregate_types:
                    if agg_type == "latest":
                        key = name
                    else:
                        key = f"{name}_{agg_type}"
                    summary[key] = getattr(agg, agg_type)()
            else:
                # 未注册的指标只提供 latest 值
                summary[name] = agg.latest()
        return summary

    def get_record_history(self, name: str) -> list[MetricRecord]:
        """获取指标的历史记录

        Args:
            name: 指标名称

        Returns:
            MetricRecord 列表
        """
        return self._records.get(name, [])

    def get_aggregator(self, name: str) -> Optional[SimpleAggregator]:
        """获取指标的聚合器

        Args:
            name: 指标名称

        Returns:
            SimpleAggregator 或 None（如果不存在）
        """
        return self._aggregators.get(name)

    def has_metric(self, name: str) -> bool:
        """检查指标是否已记录

        Args:
            name: 指标名称

        Returns:
            True 如果指标已记录
        """
        return name in self._aggregators

    def reset(self) -> None:
        """重置，准备下一个 epoch

        清空所有历史记录和聚合器，步数清零。
        """
        self._records.clear()
        self._aggregators.clear()
        self._step = 0
