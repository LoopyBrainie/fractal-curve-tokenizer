# -*- coding: utf-8 -*-
"""Utilities Tests: General Utils

对应模块: vit_pytorch.utils

测试内容:
- compute_token_features - Token 特征计算
- create_attention_mask - 注意力掩码创建
"""

import sys
from pathlib import Path

import torch

# 添加 tests 目录到 Python 路径
_tests_dir = Path(__file__).parent.parent.parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

# features.py 已移入 utilities 目录
from tests.unit.utilities.features import compute_token_features
from vit_pytorch.core.utils import create_attention_mask


def test_compute_token_features_shapes() -> None:
    """测试 Token 特征计算形状."""
    tokens = torch.randn(3, 16)
    features = compute_token_features(tokens, level=2, patch_size=(8, 8))

    assert features.stats.shape == (3, 2)
    assert features.edge.shape == (3, 1)
    assert features.spatial.shape == (3, 2)
    assert features.level.shape == (3, 1)

    assert torch.allclose(features.level[:, 0], torch.full((3,), 2.0))
    assert torch.all(features.spatial[:, 0] == 8.0)
    assert torch.all(features.spatial[:, 1] == 8.0)


def test_compute_token_features_single_value_edge_is_zero() -> None:
    """测试单值 Token 的边缘特征为零."""
    tokens = torch.randn(4, 1)
    features = compute_token_features(tokens, level=0, patch_size=(4, 4))

    assert torch.all(features.edge == 0)


def test_create_attention_mask_empty_input() -> None:
    """测试空输入的注意力掩码."""
    mask = create_attention_mask([], torch.device("cpu"))
    assert mask.numel() == 0
    assert mask.shape == (0, 0, 0)


def test_create_attention_mask_level_relationships() -> None:
    """测试层级关系的注意力掩码."""
    level_info = torch.tensor([[1], [1], [2], [4]])
    mask = create_attention_mask([level_info], torch.device("cpu"))

    assert mask.shape == (1, 4, 4)
    assert torch.allclose(mask[0, 0, 1], torch.tensor(1.2))
    assert torch.allclose(mask[0, 0, 2], torch.tensor(1.1))
    assert torch.allclose(mask[0, 0, 3], torch.tensor(1.0))
    assert torch.allclose(mask[0, 2, 3], torch.tensor(1.0))
