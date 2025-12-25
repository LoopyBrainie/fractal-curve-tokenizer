# -*- coding: utf-8 -*-
"""System integration tests for FractalCurveViT.

测试整个系统的端到端功能：
1. 模型初始化与设备移动
2. 不同分辨率的处理
3. 批处理能力
4. GPU 支持（如果可用）
"""

import torch
import pytest
from vit_pytorch import FractalCurveViT


def _device_parametrization():
    """获取可用设备列表."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@torch.no_grad()
@pytest.mark.parametrize("device", _device_parametrization())
def test_next_gen_fractal_vit_forward_pass(device: str) -> None:
    """测试 FractalCurveViT 的基础前向传播."""
    batch, channels, height, width = 2, 3, 64, 64  # 使用 2 的幂次
    images = torch.randn(batch, channels, height, width, device=device)

    model = FractalCurveViT(
        image_size=(height, width),
        num_classes=10,
        dim=192,
        depth=2,
        heads=4,
        mlp_dim=384,
        min_patch_size=(4, 4),  # 使用 2 的幂次
        max_level=3,
        num_scales=2,
    ).to(device)
    model.eval()

    logits = model(images)
    assert logits.shape == (batch, 10)
    assert logits.device.type == device


@torch.no_grad()
@pytest.mark.parametrize("device", _device_parametrization())
def test_next_gen_fractal_vit_handles_varied_sizes(device: str) -> None:
    """测试 FractalCurveViT 处理不同图像尺寸."""
    sizes = [(32, 32), (64, 64), (128, 128)]  # 使用 2 的幂次

    for height, width in sizes:
        model = FractalCurveViT(
            image_size=(height, width),
            num_classes=5,
            dim=128,
            depth=2,
            heads=2,
            mlp_dim=256,
            min_patch_size=(4, 4),  # 使用 2 的幂次
            max_level=3,
            num_scales=2,
        ).to(device)
        model.eval()

        images = torch.randn(2, 3, height, width, device=device)
        logits = model(images)
        assert logits.shape == (2, 5)
        assert logits.device.type == device


@torch.no_grad()
@pytest.mark.parametrize("device", _device_parametrization())
def test_streaming_tokenizer_device_consistency(device: str) -> None:
    """测试流式 tokenizer 的设备一致性."""
    from vit_pytorch import StreamingFractalTokenizer
    
    tokenizer = StreamingFractalTokenizer(
        image_size=32,
        channels=3,
        d_model=64,
        patch_sizes=(4, 8),
    ).to(device)
    
    images = torch.randn(2, 3, 32, 32, device=device)
    output = tokenizer.tokenize(images)
    
    for seq in output:
        assert seq.tokens.device.type == device
        assert seq.get_levels().device.type == device


@torch.no_grad()
@pytest.mark.parametrize("device", _device_parametrization())
def test_streaming_v3_tokenizer_device_consistency(device: str) -> None:
    """测试流式 V3 tokenizer 的设备一致性."""
    from vit_pytorch import StreamingFractalTokenizerV3
    
    tokenizer = StreamingFractalTokenizerV3(
        image_size=32,
        channels=3,
        d_model=64,
        base_patch_size=4,
        max_depth=3,
    ).to(device)
    
    images = torch.randn(2, 3, 32, 32, device=device)
    output = tokenizer.tokenize(images)
    
    for seq in output:
        assert seq.tokens.device.type == device


@torch.no_grad()
def test_batch_size_independence() -> None:
    """测试不同批次大小的独立性."""
    model = FractalCurveViT(
        image_size=32,
        num_classes=10,
        dim=64,
        depth=2,
        heads=2,
        mlp_dim=128,
        num_scales=2,
    )
    model.eval()
    
    # 同一张图像，不同批次大小
    single_image = torch.randn(1, 3, 32, 32)
    batch_images = single_image.repeat(4, 1, 1, 1)
    
    output_single = model(single_image)
    output_batch = model(batch_images)
    
    # 批次中的每个输出应该相同（在数值精度范围内）
    for i in range(4):
        assert torch.allclose(output_single[0], output_batch[i], atol=1e-5)


@torch.no_grad()
def test_deterministic_eval_mode() -> None:
    """测试 eval 模式下的确定性."""
    model = FractalCurveViT(
        image_size=32,
        num_classes=10,
        dim=64,
        depth=2,
        heads=2,
        mlp_dim=128,
        num_scales=2,
    )
    model.eval()
    
    images = torch.randn(2, 3, 32, 32)
    
    output1 = model(images)
    output2 = model(images)
    
    assert torch.equal(output1, output2)
