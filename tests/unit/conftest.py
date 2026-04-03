# -*- coding: utf-8 -*-
"""
Unit Test Fixtures

模块级 fixtures 用于单元测试.

包含:
- splitter_base: 基础 HilbertOptimalSplitter
- splitter_with_quota: 启用了可学习配额的 splitter
- splitter_for_ema: 用于 EMA 测试的 splitter
- features_2d: 2D 特征张量
- features_4d: 4D 特征张量
"""

import pytest
import torch


@pytest.fixture
def splitter_base():
    """基础 HilbertOptimalSplitter (无特定配置).

    用于测试 splitter 的基础功能.
    """
    from vit_pytorch.layers.splitters.hilbert_optimal_splitter import HilbertOptimalSplitter

    return HilbertOptimalSplitter(
        feature_dim=64,
        min_patch_size=4,
        max_level_limit=3,
        hidden_dim=32,
        K_min=4,
        K_max=16,
    )


@pytest.fixture
def splitter_with_quota():
    """启用了可学习配额的 HilbertOptimalSplitter.

    用于测试 Scheme E 可学习配额机制.
    """
    from vit_pytorch.layers.splitters.hilbert_optimal_splitter import HilbertOptimalSplitter

    return HilbertOptimalSplitter(
        feature_dim=64,
        min_patch_size=8,
        max_level_limit=4,
        hidden_dim=32,
        K_min=8,
        K_max=32,
    )


@pytest.fixture
def splitter_for_ema():
    """用于 EMA 测试的 HilbertOptimalSplitter.

    具有完整 EMA 配置的 splitter.
    """
    from vit_pytorch.layers.splitters.hilbert_optimal_splitter import HilbertOptimalSplitter

    return HilbertOptimalSplitter(
        feature_dim=256,
        min_patch_size=4,
        max_level_limit=8,
        hidden_dim=128,
        K_min=16,
        K_max=64,
    )


@pytest.fixture
def features_2d():
    """2D 特征张量 [B, C]."""
    return torch.randn(2, 256)


@pytest.fixture
def features_4d():
    """4D 特征张量 [B, C, H, W]."""
    return torch.randn(2, 64, 8, 8)


@pytest.fixture
def features_for_splitter(splitter_base):
    """与 splitter 匹配的 4D 特征张量.

    特征维度与 splitter 配置匹配.
    """
    B = 2
    feature_dim = splitter_base.feature_dim
    H, W = 8, 8  # 32 / 4 = 8 (min_patch_size=4)
    return torch.randn(B, feature_dim, H, W)
