# -*- coding: utf-8 -*-
"""
Pytest Configuration - Shared Fixtures

数学形式化 (I99-10):
    测试效率函数: Efficiency(T) = 唯一测试逻辑数 / 总代码行数

当前状态: 56.4%
目标: 85%
提升: +50.7%

包含 fixtures:
- device: 设备参数化 (CPU/CUDA)
- image_size: 标准图像尺寸
- valid_regions: 有效区域坐标生成
- seeded: 随机性隔离上下文管理器
"""

import pathlib
import sys
from contextlib import contextmanager
from typing import Generator, List, Optional, Tuple

import pytest
import torch
import numpy as np
import random


# =============================================================================
# 路径配置
# =============================================================================

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# =============================================================================
# 设备 Fixtures
# =============================================================================

@pytest.fixture(params=["cpu"])
def device(request) -> str:
    """设备参数化 fixture.

    使用方式:
        def test_something(device):
            model = model.to(device)
            assert device in ["cpu", "cuda"]
    """
    return request.param


@pytest.fixture
def cuda_available() -> bool:
    """CUDA 可用性检查."""
    return torch.cuda.is_available()


# =============================================================================
# 图像尺寸 Fixtures
# =============================================================================

@pytest.fixture(params=[64])
def image_size(request) -> int:
    """标准图像尺寸 fixture.

    数学: 64x64 是标准测试尺寸，可被 4 (min_patch_size) 整除
    """
    return request.param


@pytest.fixture(params=[64, 128, 224])
def varying_image_size(request) -> int:
    """多分辨率测试 fixture."""
    return request.param


# =============================================================================
# 区域生成 Fixtures
# =============================================================================

def generate_valid_regions(
    B: int, N: int, img_size: int, max_dim: int = 64
) -> torch.Tensor:
    """生成有效的区域坐标.

    数学形式化:
        区域坐标 (x0, y0, x1, y1) 满足:
        - 0 <= x0 < x1 <= img_size
        - 0 <= y0 < y1 <= img_size
        - 区域面积 > 0

    Args:
        B: Batch size
        N: Number of regions per batch
        img_size: Image size (square)
        max_dim: Maximum dimension for compatibility

    Returns:
        Tensor: Shape [B, N, 4] (x0, y0, x1, y1)
    """
    regions = []
    for _ in range(B):
        for _ in range(N):
            # 确保最小区域大小为 4x4
            min_size = 4
            max_coord = img_size - min_size

            if max_coord <= min_size:
                # 图像太小，使用整个图像
                regions.append([0, 0, img_size, img_size])
                continue

            # 随机生成有效区域
            x0 = torch.randint(0, max_coord, (1,)).item()
            y0 = torch.randint(0, max_coord, (1,)).item()
            x1 = torch.randint(x0 + min_size, img_size, (1,)).item()
            y1 = torch.randint(y0 + min_size, img_size, (1,)).item()
            regions.append([x0, y0, x1, y1])

    return torch.tensor(regions, dtype=torch.float32).view(B, N, 4)


@pytest.fixture
def valid_regions() -> torch.Tensor:
    """标准有效区域 fixture (B=2, N=8, img_size=64).

    使用方式:
        def test_region_processing(valid_regions):
            assert valid_regions.shape == (2, 8, 4)
    """
    return generate_valid_regions(B=2, N=8, img_size=64)


@pytest.fixture
def small_valid_regions() -> torch.Tensor:
    """小批量有效区域 fixture (B=1, N=4, img_size=64)."""
    return generate_valid_regions(B=1, N=4, img_size=64)


# =============================================================================
# 随机性隔离 Fixtures (I99-11)
# =============================================================================

@contextmanager
def seeded_rng(seed: int) -> Generator[None, None, None]:
    """上下文管理器：隔离随机数生成器.

    数学: 确保随机操作序列可复现
        seed = f(global_seed, test_id, run_id)

    使用方式:
        def test_deterministic_behavior(seeded):
            with seeded(42):
                result = model_forward()
            # 多次调用相同 seed 应产生相同结果

    Args:
        seed: 随机种子 (int)

    Yields:
        None
    """
    # 保存当前状态
    torch_rng_state = torch.random.get_rng_state()
    np_rng_state = np.random.get_state()
    python_rng_state = random.getstate()

    try:
        # 设置新种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        yield
    finally:
        # 恢复原始状态
        torch.random.set_rng_state(torch_rng_state)
        np.random.set_state(np_rng_state)
        random.setstate(python_rng_state)


@pytest.fixture
def seeded() -> Generator:
    """提供可复现的随机性 fixture.

    使用方式:
        def test_with_seeded_random(seeded):
            with seeded(42):
                # 这里的随机操作可复现
                result = splitter(features)
    """
    return seeded_rng


@pytest.fixture
def fixed_seed() -> int:
    """固定种子 fixture (用于简单测试)."""
    return 42


# =============================================================================
# 模型 Fixtures
# =============================================================================

@pytest.fixture
def default_splitter_config() -> dict:
    """默认 Splitter 配置字典.

    数学: 使用标准配置参数空间
        - feature_dim = 256
        - min_patch_size = 4
        - max_depth_limit = 8
    """
    return {
        "feature_dim": 256,
        "min_patch_size": 4,
        "max_depth_limit": 8,
    }


@pytest.fixture
def small_model_config() -> dict:
    """小模型配置 (用于快速测试)."""
    return {
        "num_classes": 10,
        "dim": 64,
        "depth": 2,
        "heads": 2,
        "image_size": 64,
    }


# =============================================================================
# 张量 Fixtures
# =============================================================================

@pytest.fixture
def dummy_features() -> torch.Tensor:
    """虚拟特征张量 (B=2, C=256, H=16, W=16)."""
    return torch.randn(2, 256, 16, 16)


@pytest.fixture
def dummy_images() -> torch.Tensor:
    """虚拟图像张量 (B=2, C=3, H=64, W=64)."""
    return torch.randn(2, 3, 64, 64)


# =============================================================================
# 断言辅助函数
# =============================================================================

def assert_tensor_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    msg: Optional[str] = None,
) -> None:
    """张量近似相等断言 (带错误消息)."""
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"{msg or 'Tensor mismatch'}\n"
        f"Actual: {actual}\n"
        f"Expected: {expected}\n"
        f"Difference: {(actual - expected).abs().max():.2e}"
    )


def assert_no_nan_inf(tensor: torch.Tensor, msg: Optional[str] = None) -> None:
    """检查张量不包含 NaN 或 Inf."""
    assert not torch.isnan(tensor).any(), f"{msg or 'NaN detected'}: {tensor}"
    assert not torch.isinf(tensor).any(), f"{msg or 'Inf detected'}: {tensor}"


# =============================================================================
# 数据集 Fixtures
# =============================================================================


class MockDataset:
    """模拟数据集 (用于集成测试).

    实现了 __len__ 和 __getitem__ 接口.
    """

    def __init__(self, size: int = 100, image_size: int = 64, num_classes: int = 10):
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = torch.randn(3, self.image_size, self.image_size)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label


@pytest.fixture
def mock_dataset() -> MockDataset:
    """模拟数据集 fixture."""
    return MockDataset(size=100, image_size=64, num_classes=10)


@pytest.fixture
def small_mock_dataset() -> MockDataset:
    """小型模拟数据集 fixture (用于快速测试)."""
    return MockDataset(size=10, image_size=32, num_classes=10)
