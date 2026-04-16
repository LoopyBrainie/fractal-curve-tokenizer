# -*- coding: utf-8 -*-
"""
数据处理工具函数
"""

import sys
from pathlib import Path
from typing import Tuple

import torch

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_test_image(
    size: int = 64,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    加载或创建测试图像

    Args:
        size: 图像尺寸
        device: 设备

    Returns:
        测试图像 [3, size, size]
    """
    # 创建合成测试图像 (渐变 + 纹理)
    x = torch.linspace(0, 1, size)
    y = torch.linspace(0, 1, size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # 组合多种图案
    img = torch.zeros(3, size, size)

    # 渐变背景
    img[0] = 0.5 + 0.3 * xx + 0.2 * yy
    img[1] = 0.4 + 0.4 * torch.sin(xx * 4 * torch.pi)
    img[2] = 0.6 + 0.3 * torch.cos(yy * 4 * torch.pi)

    # 添加一些纹理
    img[0] += 0.1 * torch.randn_like(img[0])
    img[1] += 0.1 * torch.randn_like(img[1])
    img[2] += 0.1 * torch.randn_like(img[2])

    # 裁剪到 [0, 1]
    img = img.clamp(0, 1)

    return img.to(device)


def preprocess_image(
    image: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> torch.Tensor:
    """
    预处理图像 (ImageNet 标准化)

    Args:
        image: 输入图像 [C, H, W] 或 [B, C, H, W]
        mean: RGB 均值
        std: RGB 标准差

    Returns:
        标准化后的图像
    """
    mean = torch.tensor(mean, device=image.device).view(-1, 1, 1)
    std = torch.tensor(std, device=image.device).view(-1, 1, 1)

    if image.dim() == 3:
        image = (image - mean) / std
    else:
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
        image = (image - mean) / std

    return image


def create_synthetic_image(
    size: int = 64,
    pattern: str = 'mixed',
    device: str = 'cpu'
) -> torch.Tensor:
    """
    创建合成测试图像

    Args:
        size: 图像尺寸
        pattern: 图案类型 ('mixed', 'gradient', 'checkerboard', 'noise', 'edges')
        device: 设备

    Returns:
        合成图像 [3, size, size]
    """
    x = torch.linspace(0, 1, size)
    y = torch.linspace(0, 1, size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    img = torch.zeros(3, size, size)

    if pattern == 'mixed':
        # 综合图案
        img[0] = 0.5 + 0.3 * xx + 0.2 * yy
        img[1] = 0.4 + 0.4 * torch.sin(xx * 4 * torch.pi)
        img[2] = 0.6 + 0.3 * torch.cos(yy * 4 * torch.pi)
        # 添加边缘
        img[0] += 0.2 * torch.sin(xx * 16 * torch.pi) * torch.cos(yy * 16 * torch.pi)

    elif pattern == 'gradient':
        img[0] = xx
        img[1] = yy
        img[2] = 1 - xx - yy

    elif pattern == 'checkerboard':
        checker_size = size // 8
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    img[:, i*checker_size:(i+1)*checker_size, j*checker_size:(j+1)*checker_size] = 1.0

    elif pattern == 'noise':
        img = torch.rand(3, size, size)

    elif pattern == 'edges':
        # 创建边缘图案
        for i in range(size):
            for j in range(size):
                dist_center = torch.sqrt(((i - size/2)/size)**2 + ((j - size/2)/size)**2)
                if 0.2 < dist_center < 0.25:
                    img[:, i, j] = 1.0
                if 0.4 < dist_center < 0.45:
                    img[:, i, j] = 1.0

    # 添加随机噪声
    img += 0.05 * torch.randn_like(img)
    img = img.clamp(0, 1)

    return img.to(device)


def get_test_batch(
    batch_size: int = 4,
    size: int = 64,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    获取测试批次

    Args:
        batch_size: 批次大小
        size: 图像尺寸
        device: 设备

    Returns:
        测试图像批次 [batch_size, 3, size, size]
    """
    patterns = ['mixed', 'gradient', 'checkerboard', 'noise']
    images = []

    for i in range(batch_size):
        pattern = patterns[i % len(patterns)]
        img = create_synthetic_image(size, pattern=pattern)
        images.append(img)

    return torch.stack(images).to(device)


def create_depth_map(
    regions: list,
    depth: int,
    image_size: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    创建深度图

    Args:
        regions: 区域列表 [(x1, y1, x2, y2), ...]
        depth: 最大深度
        image_size: 图像尺寸
        device: 设备

    Returns:
        深度图 [image_size, image_size]
    """
    depth_map = torch.zeros(image_size, image_size, device=device)

    for i, region in enumerate(regions):
        if len(region) == 4:
            x1, y1, x2, y2 = region
            depth_map[y1:y2, x1:x2] = depth

    return depth_map


def compute_region_complexity(
    image: torch.Tensor,
    region: tuple,
    alpha: float = 0.5
) -> float:
    """
    计算区域复杂度

    Args:
        image: 输入图像 [C, H, W]
        region: 区域 (x1, y1, x2, y2)
        alpha: 方差权重

    Returns:
        复杂度分数
    """
    x1, y1, x2, y2 = region
    region_img = image[:, y1:y2, x1:x2]

    # 转灰度
    if region_img.shape[0] == 3:
        gray = 0.299 * region_img[0] + 0.587 * region_img[1] + 0.114 * region_img[2]
    else:
        gray = region_img[0]

    # 方差
    var = gray.var().item()

    # 梯度能量
    if gray.shape[0] > 1 and gray.shape[1] > 1:
        dx = gray[:-1, 1:] - gray[:-1, :-1]
        dy = gray[1:, :-1] - gray[:-1, :-1]
        grad_energy = (dx.pow(2) + dy.pow(2)).mean().item()
    else:
        grad_energy = 0.0

    # 归一化
    sigma_0_sq = 0.01
    g_0_sq = 0.08

    c_var = var / (var + sigma_0_sq)
    c_grad = grad_energy / (grad_energy + g_0_sq)

    return alpha * c_var + (1 - alpha) * c_grad
