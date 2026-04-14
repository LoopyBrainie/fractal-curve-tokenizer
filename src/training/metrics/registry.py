"""指标注册中心 - 所有指标的单一来源"""
from dataclasses import dataclass
from typing import Optional

from .record import MetricSource


@dataclass
class MetricSpec:
    """指标规格定义

    描述每个指标的元数据，包括来源、单位、描述和聚合方式。

    Attributes:
        name: 指标名称（唯一标识）
        source: 指标来源 (MetricSource 枚举)
        unit: 单位字符串，如 "", "tokens", "%", "px"
        description: 指标描述，说明其含义
        aggregate_types: 支持的聚合类型元组，如 ("mean", "latest", "max")
    """
    name: str
    source: MetricSource
    unit: str = ""
    description: str = ""
    aggregate_types: tuple[str, ...] = ("mean", "latest", "max")


class MetricRegistry:
    """指标注册中心 - 所有指标的单一来源

    提供统一的指标注册、查询和管理接口。

    使用示例:
        registry = MetricRegistry()
        registry.register(
            "loss",
            MetricSource.TRAINING_LOOP,
            unit="",
            description="总损失值",
            aggregate_types=("mean", "latest", "max")
        )

        spec = registry.get("loss")
        if spec:
            print(f"Metric: {spec.name}, Source: {spec.source.value}")
    """

    def __init__(self):
        self._metrics: dict[str, MetricSpec] = {}

    def register(
        self,
        name: str,
        source: MetricSource,
        unit: str = "",
        description: str = "",
        aggregate_types: tuple[str, ...] = ("mean", "latest", "max"),
    ) -> None:
        """注册新指标

        Args:
            name: 指标名称（必须唯一）
            source: 指标来源
            unit: 单位字符串
            description: 指标描述
            aggregate_types: 支持的聚合类型

        Raises:
            ValueError: 如果指标名称已存在
        """
        if name in self._metrics:
            raise ValueError(f"Metric '{name}' already registered")
        self._metrics[name] = MetricSpec(
            name=name,
            source=source,
            unit=unit,
            description=description,
            aggregate_types=aggregate_types,
        )

    def get(self, name: str) -> Optional[MetricSpec]:
        """获取指标规格

        Args:
            name: 指标名称

        Returns:
            MetricSpec 或 None（如果不存在）
        """
        return self._metrics.get(name)

    def by_source(self, source: MetricSource) -> list[str]:
        """获取指定来源的所有指标名称

        Args:
            source: MetricSource 枚举值

        Returns:
            指标名称列表
        """
        return [n for n, s in self._metrics.items() if s.source == source]

    def all_metrics(self) -> dict[str, MetricSpec]:
        """获取所有已注册的指标

        Returns:
            指标名称到 MetricSpec 的字典副本
        """
        return self._metrics.copy()

    @classmethod
    def create_default(cls) -> "MetricRegistry":
        """创建包含所有标准指标的默认注册表

        预注册训练系统中常用的标准指标。

        Returns:
            配置好的 MetricRegistry 实例
        """
        registry = cls()

        # MODEL_OUTPUT 指标 - 来自 TrainingStats 模型输出
        model_output_specs = [
            ("num_tokens", "tokens", "序列token数量"),
            ("depth_used", "level", "使用的最大深度"),
            ("depth_distribution", "dict", "深度概率分布"),
            ("splitter_logits_mean", "", "Splitter logits均值"),
            ("splitter_logits_std", "", "Splitter logits标准差"),
            ("active_ratio", "%", "活跃token比例"),
            ("manifold_bias_max", "", "流形偏置最大值"),
            ("manifold_bias_min", "", "流形偏置最小值"),
            ("manifold_bias_mean", "", "流形偏置均值"),
            ("manifold_bias_std", "", "流形偏置标准差"),
            ("poincare_dist_mean", "", "Poincare距离均值"),
            ("poincare_dist_std", "", "Poincare距离标准差"),
            ("backbone_grad_norm", "", "骨干网络梯度范数"),
            ("splitter_grad_norm", "", "Splitter梯度范数"),
            ("entmax_grad_norm", "", "Entmax梯度范数"),
            ("manifold_decoder_grad_norm", "", "流形解码器梯度范数"),
            ("mean_abs_logits", "", "Splitter logits绝对值均值"),
            ("logits_mean", "", "分类logits均值"),
            ("logits_std", "", "分类logits标准差"),
        ]
        for name, unit, desc in model_output_specs:
            registry.register(name, MetricSource.MODEL_OUTPUT, unit, desc)

        # TRAINING_LOOP 指标 - 在训练循环中计算
        training_loop_specs = [
            ("loss", "", "总损失"),
            ("grad_norm", "", "梯度范数（裁剪前）"),
            ("learning_rate", "", "当前学习率"),
            ("epoch_time", "s", "当前epoch耗时"),
            ("samples_per_second", "", "样本处理速度"),
        ]
        for name, unit, desc in training_loop_specs:
            registry.register(name, MetricSource.TRAINING_LOOP, unit, desc)

        # HOOK_BACKWARD 指标 - 反向传播hook捕获
        hook_specs = [
            ("layer_grad_norm", "", "每层梯度范数"),
            ("backbone_vs_splitter_grad_ratio", "", "骨干与Splitter梯度比例"),
            ("bottleneck_layer_grad", "", "瓶颈层梯度"),
        ]
        for name, unit, desc in hook_specs:
            registry.register(name, MetricSource.HOOK_BACKWARD, unit, desc)

        # CONFIG 指标 - 来自超参数配置
        config_specs = [
            ("image_size", "px", "图像尺寸"),
            ("batch_size", "", "批次大小"),
            ("num_classes", "", "分类类别数"),
            ("dim", "", "模型维度"),
        ]
        for name, unit, desc in config_specs:
            registry.register(name, MetricSource.CONFIG, unit, desc)

        # COMPUTED 指标 - 派生指标（通常动态添加）
        computed_specs = [
            ("nan_count", "", "NaN梯度数量"),
            ("inf_count", "", "Inf梯度数量"),
            ("issue_count", "", "梯度异常总数"),
            ("skipped_steps", "", "跳过的优化器步数"),
        ]
        for name, unit, desc in computed_specs:
            registry.register(name, MetricSource.COMPUTED, unit, desc)

        # 损失组件指标
        loss_component_specs = [
            ("loss_cross_entropy", "", "交叉熵损失"),
            ("loss_raw_budget_error", "", "原始预算误差"),  # D162: 重命名 (原 loss_budget_loss)
            ("loss_density_regularization", "", "密度正则化"),
            ("loss_consistency_loss", "", "一致性损失"),
            ("loss_entropy_loss", "", "熵损失"),
            ("loss_elastic_budget", "", "弹性预算"),
            ("loss_sparsity_penalty", "", "稀疏性惩罚"),
            ("loss_semantic_loss", "", "语义损失"),
            ("loss_diversity_loss", "", "多样性损失"),
            ("loss_reconstruction_loss", "", "重建损失"),
            ("loss_quota_entropy", "", "配额熵"),
            ("loss_jump_loss", "", "跳跃损失"),
            ("loss_tree_constraint", "", "树约束"),
            ("loss_manifold_regularization", "", "流形正则化"),
        ]
        for name, unit, desc in loss_component_specs:
            registry.register(name, MetricSource.MODEL_OUTPUT, unit, desc)

        return registry
