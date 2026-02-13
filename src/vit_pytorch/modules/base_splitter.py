"""
Splitter 接口协议定义 - 向后兼容层

⚠️ DEPRECATED ⚠️
此模块已弃用。请从 vit_pytorch.core.splitter_protocol 导入:

    from vit_pytorch.core.splitter_protocol import CoreSplitter, SplitResult

此模块保留用于向后兼容，通过重新导出 core.splitter_protocol 实现。
"""

# 向后兼容：重新导出核心协议
from vit_pytorch.core.splitter_protocol import (
    CoreSplitter,
    AnnealingSplitter,
    MetricsSplitter,
    SplitResult,
    validate_splitter,
)

__all__ = [
    "CoreSplitter",
    "AnnealingSplitter",
    "MetricsSplitter",
    "SplitResult",
    "validate_splitter",
]
