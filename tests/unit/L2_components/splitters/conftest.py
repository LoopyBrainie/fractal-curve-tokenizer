"""Shared fixtures for splitter unit tests.

v1.3 STANDARD: I170.4 (17 HMFT 缺口补强) 重构基础设施。

将原本散落在 4 个 HMFT 测试文件中的 `_make_splitter()` 模板统一抽取到
本 conftest,任何 splitter 单元测试可通过 fixture 名 `make_hmft_splitter`
获取工厂函数,用 kwargs 覆盖 config 字段。

Usage::

    def test_X(make_hmft_splitter):
        splitter = make_hmft_splitter(hidden_dim=32)
        ...
"""
from __future__ import annotations

from typing import Callable

import pytest

from vit_pytorch.core.constants import HMFT_MIN_IMAGE_SIZE
from vit_pytorch.layers.splitters.multi_block_hmft_splitter import (
    MultiBlockHMFTSplitter,
    MultiBlockHMFTSplitterConfig,
)

# A5 input guard (multi_block_hmft_splitter.py forward): image_size below
# HMFT_MIN_IMAGE_SIZE is rejected with ValueError. The 5-grid EAS design
# table contains sizes 8/16/32/64/128; parametrize subsets are restricted
# to sizes that satisfy the A5 contract.
HMFT_VALID_GRIDS: tuple[int, ...] = tuple(
    sorted(g for g in (8, 16, 32, 64, 128) if g >= HMFT_MIN_IMAGE_SIZE)
)


@pytest.fixture
def make_hmft_splitter() -> Callable[..., MultiBlockHMFTSplitter]:
    """工厂 fixture: 返回 ``MultiBlockHMFTSplitter(MultiBlockHMFTSplitterConfig(**overrides))``。

    默认行为 (无 overrides) 等价于 ``_make_splitter()`` 在 a5 文件中的实现。
    测试可通过 kwargs 覆盖任意 config 字段,例如::

        splitter = make_hmft_splitter(feature_dim_in=64, feature_dim=64)
    """
    def _make(**overrides) -> MultiBlockHMFTSplitter:
        cfg = MultiBlockHMFTSplitterConfig(**overrides)
        return MultiBlockHMFTSplitter(cfg)
    return _make
