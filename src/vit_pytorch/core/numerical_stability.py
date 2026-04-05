# -*- coding: utf-8 -*-
"""NaN/Inf 数值稳定性监控

提供全局 NaN/Inf 修复计数器和 DDP-safe 统计功能。

使用方式:
    from vit_pytorch.core.numerical_stability import increment_nan_fix, get_nan_fix_count

    # 在 NaN/Inf 修复处调用
    increment_nan_fix("safe_softplus")

    # 在日志系统中获取计数
    counts = get_nan_fix_count()
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from typing import Dict


# 全局 NaN/Inf 修复计数器
_nan_fix_counter: Dict[str, int] = {
    "safe_softplus": 0,
    "nan_grad_hooks": 0,
}


def is_main_process() -> bool:
    """检查是否为 rank_0 或非分布式环境

    Returns:
        True 如果是主进程或非分布式环境
    """
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def increment_nan_fix(scope: str, count: int = 1) -> None:
    """递增 NaN/Inf 修复计数器（多 GPU 安全：仅 rank_0 记录）

    Args:
        scope: 修复来源 ("safe_softplus" 或 "nan_grad_hooks")
        count: 递增数量，默认 1
    """
    if is_main_process():
        if scope in _nan_fix_counter:
            _nan_fix_counter[scope] += count
        else:
            _nan_fix_counter[scope] = count


def _sync_counts() -> None:
    """所有进程同步计数（供内部使用）"""
    if not (dist.is_available() and dist.is_initialized()):
        return

    for scope in _nan_fix_counter:
        tensor = torch.tensor([_nan_fix_counter[scope]], dtype=torch.long, device="cpu")
        dist.broadcast(tensor, src=0)
        _nan_fix_counter[scope] = tensor.item()


def get_nan_fix_count() -> Dict[str, int]:
    """获取 NaN/Inf 修复计数器（分布式环境下自动同步）

    Returns:
        计数器字典，键为 "safe_softplus" 和 "nan_grad_hooks"
    """
    if dist.is_available() and dist.is_initialized():
        # 广播 rank_0 的计数到所有进程
        _sync_counts()
    return _nan_fix_counter.copy()
