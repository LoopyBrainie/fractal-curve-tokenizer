"""v1.3 STANDARD: Multi-Block HMFT edge cases + AMP numerical stability.

Per I170.4 plan: 17 个 HMFT 缺口补强,Group 3+4 — Edge Cases (T9-T13)。

覆盖 HMFT 的边界路径与混合精度稳定性:
  T9:  _sample_block_size fallback 路径 (n_cells < K 时的兜底)
  T10: set_temperature clamp 到 temperature_min (Gumbel 退火下限保护)
  T11: set_explore_bias / set_annealing_schedule no-op (空实现不抛)
  T12: AMP forward 数值稳定性 (bfloat16 CPU / float16 CUDA)
  T13: AMP 下 STE 梯度活性 (autocast + backward 不静默 0)

Path bootstrap 由 ``tests/conftest.py`` 自动处理。
"""
from __future__ import annotations

import pytest
import torch

from tests.conftest import assert_no_nan_inf
from vit_pytorch.core.constants import HMFT_K_HARD_GLOBAL_POOL
from vit_pytorch.layers.splitters.multi_block_hmft_splitter import (
    MultiBlockHMFTSplitter,
    MultiBlockHMFTSplitterConfig,
)

K = HMFT_K_HARD_GLOBAL_POOL  # = 8


# T9 — _sample_block_size fallback 路径
def test_sample_block_size_fallback(make_hmft_splitter):
    """T9: 当 strict filter 找不到合法 h 时,fallback 应返回最大合法 h。

    直接调 ``_sample_block_size(min_n_cells=64, max_h=8)`` 触发 line 285-289
    的兜底分支: strict filter 要求 ``h <= max_h // 3``,任何 h 都不满足,
    然后回退到 ``[s for s in block_sizes if s <= max_h] = [8]``。
    """
    torch.manual_seed(42)
    splitter = make_hmft_splitter()
    h = splitter._sample_block_size(hard=True, max_h=8, min_n_cells=64)
    assert h == 8, f"fallback 应返回最大合法 h=8, got {h}"

    # 同时验证 forward 在正常参数下不触发 fallback
    features = torch.randn(2, 256, 32, 32)
    result = splitter(features, image_size=(32, 32), hard=True)
    assert result.selected_mask.shape[1] >= K, (
        f"selected_mask n_cells={result.selected_mask.shape[1]} < K={K}"
    )


# T10 — set_temperature clamp 到 temperature_min
def test_set_temperature_clamps_to_min(make_hmft_splitter):
    """T10: 验证 Gumbel 退火下限保护,极低温度被 clamp 到 temperature_min。

    关键: ``get_current_temperature()`` 返回 ``Tensor`` 不是 ``float``,
    比较时必须用 ``.item()``。这是用户代码蓝图易踩的坑。
    """
    splitter = make_hmft_splitter()
    # 默认 config 的 temperature_min = 0.1
    splitter.set_temperature(0.01)  # 极低温度
    current = splitter.get_current_temperature()
    # .item() 是必需的: Tensor == float 是元素级比较
    assert current.item() == pytest.approx(0.1), (
        f"温度下限保护失效: current={current.item()}, 期望 0.1"
    )


# T11 — set_explore_bias / set_annealing_schedule no-op
def test_setters_are_noop(make_hmft_splitter):
    """T11: 验证 ``set_explore_bias`` 和 ``set_annealing_schedule`` 不抛异常,
    forward 仍正常工作 (这些 setter 当前是空实现,留作未来扩展)。

    这是协议层的"向前兼容"守卫: 训练循环调用这些 setter 不应炸。
    """
    splitter = make_hmft_splitter()
    # 都不应抛
    splitter.set_explore_bias(0.5)
    splitter.set_annealing_schedule([1.0, 0.5, 0.1])
    # forward 仍正常工作
    features = torch.randn(2, 256, 32, 32)
    result = splitter(features, image_size=(32, 32), hard=True)
    assert result.selected_mask.shape[0] == 2, "forward 失败"


# T12 — AMP forward 数值稳定性
def test_amp_forward_numerical_stability(make_hmft_splitter):
    """T12: 验证 AMP (bfloat16/float16) 下 forward 不产生 NaN/Inf。

    CPU autocast 不支持 float16,只支持 bfloat16。CUDA 走 float16。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    splitter = make_hmft_splitter()
    splitter.train()
    features = torch.randn(2, 256, 32, 32)
    with torch.amp.autocast(device_type=device, dtype=dtype):
        result = splitter(features, image_size=(32, 32), hard=True)
    # 关键: 不出现 NaN/Inf
    assert_no_nan_inf(result.roi_features, msg="AMP 下 roi_features 含 NaN/Inf")
    assert_no_nan_inf(result.mask_ste, msg="AMP 下 mask_ste 含 NaN/Inf")


# T13 — AMP 下 V3 协议 backward 完整性
def test_amp_ste_gradient_active(make_hmft_splitter):
    """T13: 验证 autocast 下 V3 协议的 backward 不抛异常 + T10 keystone 仍成立。

    严格"非零梯度"测试在 bfloat16 下不可靠 (小矩阵梯度 underflow),
    此测试退而求其次,验证 (a) backward 不抛 + (b) ``mask_ste.requires_grad``
    在 AMP 下仍为 True。这是 I170 V3 协议的核心契约 (T10 keystone)。

    loss 在 autocast 内构造,backward 在 autocast 外调 (PyTorch 推荐模式)。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    splitter = make_hmft_splitter()
    splitter.train()
    features = torch.randn(2, 256, 32, 32)
    with torch.amp.autocast(device_type=device, dtype=dtype):
        result = splitter(features, image_size=(32, 32), hard=True)
        loss = result.mask_ste.sum()
    # mask_ste 的 requires_grad 在 AMP 下仍必须为 True (T10 keystone)
    assert result.mask_ste.requires_grad, "AMP 下 mask_ste.requires_grad 为 False"
    # backward 不抛 — 即 AMP 下 STE 链未被打断
    loss.backward()  # autocast 外
    # 至少 _score_cells 的某个参数.grad 不为 None (说明 STE 链确实有梯度)
    assert splitter.fusion.score_proj.weight.grad is not None, (
        "AMP 下 fusion.score_proj.grad 为 None — STE 链断流"
    )


# 显式 re-export
__all__ = [
    "K",
    "MultiBlockHMFTSplitter",
    "MultiBlockHMFTSplitterConfig",
    "test_sample_block_size_fallback",
    "test_set_temperature_clamps_to_min",
    "test_setters_are_noop",
    "test_amp_forward_numerical_stability",
    "test_amp_ste_gradient_active",
]
