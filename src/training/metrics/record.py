"""指标记录和来源枚举 - 追踪数据血缘"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MetricSource(Enum):
    """指标来源枚举 - 追踪数据血缘

    用于标识每个指标的原始来源，确保数据溯源能力。

    来源分类:
    - MODEL_OUTPUT: 直接来自 TrainingStats 模型输出
    - TRAINING_LOOP: 在 train_one_epoch 训练循环中计算
    - HOOK_FORWARD: Forward hook 捕获的激活统计
    - HOOK_BACKWARD: Backward hook 捕获的梯度统计
    - CONFIG: 来自超参数配置
    - COMPUTED: 由其他指标派生的衍生指标
    """
    MODEL_OUTPUT = "model_output"
    TRAINING_LOOP = "training_loop"
    HOOK_FORWARD = "hook_forward"
    HOOK_BACKWARD = "hook_backward"
    CONFIG = "config"
    COMPUTED = "computed"


@dataclass(frozen=True)
class MetricRecord:
    """不可变的单条指标记录

    包含指标的完整元数据，用于数据血缘追踪和审计。

    Attributes:
        name: 指标名称
        value: 指标值
        source: 指标来源 (MetricSource 枚举)
        lineage: 血缘追踪 - 父指标名称元组，空表示叶子指标
        step: 训练步数
        epoch: 训练轮次
        metadata: 额外元数据字典
    """
    name: str
    value: float
    source: MetricSource
    lineage: tuple[str, ...] = ()
    step: int = 0
    epoch: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_leaf(self) -> bool:
        """判断是否为叶子指标（无父指标）"""
        return len(self.lineage) == 0

    def parent_name(self) -> Optional[str]:
        """获取直接父指标名称"""
        return self.lineage[-1] if self.lineage else None

    def has_lineage(self) -> bool:
        """判断是否有血缘信息（是否为派生指标）"""
        return len(self.lineage) > 0
