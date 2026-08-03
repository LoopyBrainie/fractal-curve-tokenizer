"""v1.3 STANDARD: Multi-Block HMFT diagnostics verification.

Per I170.4 plan: 17 个 HMFT 缺口补强,Group 5 — Diagnostics (T14-T15)。

覆盖 HMFT 的"自描述"接口:
  T14: get_diagnostics()["h_probs"] 跨 grid 满足 5 维 + 归一化
  T15: get_entropy_loss() 反向传播到 h_logits (5-bin softmax 熵梯度链)

Path bootstrap 由 ``tests/conftest.py`` 自动处理。
"""
from __future__ import annotations

import pytest
import torch

from vit_pytorch.layers.splitters.multi_block_hmft_splitter import (
    MultiBlockHMFTSplitter,
    MultiBlockHMFTSplitterConfig,
)


# T14 — h_probs 跨 5 grid 满足 5 维 + 归一化
@pytest.mark.parametrize("grid_size", [32, 64, 128])
def test_h_probs_sum_to_one(make_hmft_splitter, grid_size: int):
    """T14: 验证 h_probs 是 5-bin softmax,跨 grid 保持归一化。

    h_probs 由 ``get_diagnostics()`` 从 ``softmax(h_logits)`` 派生,即使
    未调用 forward 也应可访问 (I170 Commit 1 协议变更)。

    h_probs 实际是 Python list (get_diagnostics 序列化形式),因此用 ``len()``
    + ``sum()`` 兼容性检查,不强求 ``.shape`` 或 ``.item()``。
    """
    torch.manual_seed(42)
    splitter = make_hmft_splitter()
    splitter.eval()
    # 构造 forward 触发 _current_epoch 等内部状态
    features = torch.randn(2, 256, grid_size, grid_size)
    _ = splitter(features, image_size=(grid_size, grid_size), hard=True)

    diag = splitter.get_diagnostics()
    assert "h_probs" in diag, "get_diagnostics() 缺少 'h_probs' 字段"
    h_probs = diag["h_probs"]
    # 5-bin softmax 必须有 5 个元素 (兼容 list / Tensor)
    assert len(h_probs) == 5, f"h_probs 应为 5-bin,got {len(h_probs)} 元素"
    # 归一化 (兼容 list / Tensor,sum 是标量)
    h_sum = float(sum(h_probs))
    assert abs(h_sum - 1.0) < 1e-5, f"h_probs 未归一化: sum={h_sum} (grid={grid_size})"
    # 所有概率必须非负
    assert all(float(p) >= 0 for p in h_probs), f"h_probs 含负值 (grid={grid_size})"


# T15 — get_entropy_loss() 反向到 h_logits
def test_entropy_loss_backward_flows(make_hmft_splitter):
    """T15: 验证熵正则化路径完整 — entropy loss 必须在 autograd 图中,反向到 h_logits。

    关键区别于 T4 (mask_ste 不可微回到 h_logits): entropy loss 是直接对
    ``softmax(h_logits)`` 计算的,不经过采样/argmax,所以 h_logits 必须
    接收非零梯度。这是 R7 路由 aux loss 接入 HMFT 时的关键守卫。
    """
    torch.manual_seed(42)
    splitter = make_hmft_splitter()
    splitter.train()
    # 构造 forward 触发完整状态
    features = torch.randn(2, 256, 32, 32)
    _ = splitter(features, image_size=(32, 32), hard=True)

    loss = splitter.get_entropy_loss()
    # 熵 loss 必须在 autograd 图中 (requires_grad=True)
    assert loss.requires_grad, (
        f"entropy_loss 不在 autograd 图: requires_grad={loss.requires_grad}"
    )
    # 反向传播
    loss.backward()
    # h_logits 必须接收非零梯度 (entropy 关于 h_logits 是非平凡函数)
    assert splitter.h_logits.grad is not None, (
        "entropy_loss 反向未触发 h_logits.grad 记录"
    )
    assert splitter.h_logits.grad.abs().sum().item() > 0, (
        "h_logits 熵梯度静默 0 — 熵正则化路径断流"
    )


# 显式 re-export
__all__ = [
    "MultiBlockHMFTSplitter",
    "MultiBlockHMFTSplitterConfig",
    "test_h_probs_sum_to_one",
    "test_entropy_loss_backward_flows",
]
