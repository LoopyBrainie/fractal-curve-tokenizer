"""标准层输出结构 — 所有 layer 的包裹必须实现此协议

设计原则:
    - 每层自己打包诊断数据 (splitter_output, attn_output, ffn_output)
    - 训练器只负责解开包裹 (flatten + record)
    - 新增 Splitter 变体无需修改训练器
"""
from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class LayerOutputProtocol(Protocol):
    """每层输出的标准协议

    实现此协议的类必须提供 output_dict 属性，
    返回该层所有诊断指标的字典。

    Example:
        class MyLayerOutput:
            @property
            def output_dict(self) -> Dict[str, Any]:
                return {"metric_a": 0.5, "metric_b": 1.2}
    """

    @property
    def output_dict(self) -> Dict[str, Any]:
        """返回该层所有诊断指标的字典"""
        ...


def flatten_layer_outputs(
    auxiliary_outputs: Dict[str, Any],
    prefix: str = "train",
) -> Dict[str, float]:
    """将 auxiliary_outputs 展平为 logger 可用的 (key, value) 格式

    核心原则：训练器不知道也不需要知道各层返回了什么字段。
    此函数自动处理任意嵌套的层输出结构。

    Args:
        auxiliary_outputs: 主模型返回的各层包裹字典
            {"splitter": {...}, "attn_0": {...}, "ffn_0": {...}}
        prefix: 日志前缀，默认 "train"

    Returns:
        展平后的字典，如:
        {
            "train/splitter/entropy": 0.5,
            "train/splitter/alpha": 1.5,
            "train/attn_0/geometric_bias_mean": 0.2
        }

    Example:
        >>> flatten_layer_outputs({
        ...     "splitter": {"entropy": 0.5, "nested": {"alpha": 1.5}},
        ...     "attn": {"geometric_bias_mean": 0.2}
        ... }, prefix="train")
        {
            "train/splitter/entropy": 0.5,
            "train/splitter/nested/alpha": 1.5,
            "train/attn/geometric_bias_mean": 0.2
        }
    """
    result: Dict[str, float] = {}

    def _flatten(data: Dict[str, Any], path: str) -> None:
        for key, value in data.items():
            full_key = f"{path}/{key}" if path else key
            if isinstance(value, dict):
                _flatten(value, full_key)
            elif isinstance(value, (int, float)):
                result[full_key] = float(value)
            # 跳过 None, Tensor, str, list 等其他类型

    for layer_name, layer_output in auxiliary_outputs.items():
        if layer_output is None:
            continue
        if hasattr(layer_output, "output_dict"):
            # LayerOutputProtocol 实现类
            _flatten(layer_output.output_dict, f"{prefix}/{layer_name}")
        elif isinstance(layer_output, dict):
            # 原始字典
            _flatten(layer_output, f"{prefix}/{layer_name}")
        # 跳过 None 或其他类型

    return result
