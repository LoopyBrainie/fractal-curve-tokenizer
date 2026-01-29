# -*- coding: utf-8 -*-
"""
Integration Test Fixtures

集成测试 fixtures 用于测试多组件交互.

包含:
- model_for_training: 用于训练测试的模型
- model_for_inference: 用于推理测试的模型
- sample_batch: 标准样本批次
- training_dataset: 模拟训练数据集
"""

import pytest
import torch


@pytest.fixture
def model_for_training():
    """用于训练测试的 FractalCurveViT 模型.

    小型配置，适合快速测试.
    """
    from vit_pytorch import FractalCurveViT

    return FractalCurveViT(
        image_size=64,
        num_classes=10,
        dim=64,
        num_layers=2,
        heads=4,
        mlp_dim=128,
        pool="cls",
        dropout=0.0,
        drop_path=0.0,
    )


@pytest.fixture
def model_for_inference():
    """用于推理测试的 FractalCurveViT 模型.

    .eval() 模式专用配置.
    """
    from vit_pytorch import FractalCurveViT

    return FractalCurveViT(
        image_size=128,
        num_classes=100,
        dim=128,
        num_layers=4,
        heads=8,
        mlp_dim=256,
        pool="mean",
        dropout=0.0,
        drop_path=0.0,
    )


@pytest.fixture
def sample_batch():
    """标准样本批次 (图像, 标签).

    形状: [B, C, H, W], [B]
    """
    images = torch.randn(2, 3, 64, 64)
    labels = torch.randint(0, 10, (2,))
    return images, labels


@pytest.fixture
def sample_batch_variable_size():
    """可变尺寸样本批次.

    用于测试动态分辨率支持.
    """
    images = torch.randn(2, 3, 128, 128)
    labels = torch.randint(0, 10, (2,))
    return images, labels


@pytest.fixture
def training_dataset(mock_dataset):
    """模拟训练数据集.

    Args:
        mock_dataset: fixtures.conftest 中的 mock_dataset fixture
    """
    return mock_dataset
