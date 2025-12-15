import torch
import pytest

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from vit_pytorch._deprecated.fractal_curve_tokenizer import FractalHilbertTokenizer
from vit_pytorch.fractal_vit import SimpleFractalViT


def _device_parametrization():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@torch.no_grad()
@pytest.mark.parametrize("device", _device_parametrization())
def test_simple_fractal_vit_forward_pass(device: str) -> None:
    batch, channels, height, width = 2, 3, 64, 36
    images = torch.randn(batch, channels, height, width, device=device)

    model = SimpleFractalViT(
        image_size=(height, width),
        num_classes=10,
        dim=192,
        depth=2,
        heads=4,
        mlp_dim=384,
        min_patch_size=(16, 9),
        max_level=3,
    ).to(device)

    logits = model(images)
    assert logits.shape == (batch, 10)
    assert logits.device.type == device

    tokenizer_output = model.fractal_tokenizer.tokenize(images[:1])
    sequence = tokenizer_output.sequences[0]
    assert sequence.tokens.device.type == device
    assert sequence.tokens.shape[0] > 0
    assert sequence.metadata["levels"].device.type == device
    assert sequence.metadata["levels"].shape[0] == sequence.tokens.shape[0]


@torch.no_grad()
@pytest.mark.parametrize("device", _device_parametrization())
def test_simple_fractal_vit_handles_varied_sizes(device: str) -> None:
    sizes = [(32, 18), (64, 36), (128, 72)]

    for height, width in sizes:
        model = SimpleFractalViT(
            image_size=(height, width),
            num_classes=5,
            dim=128,
            depth=2,
            heads=2,
            mlp_dim=256,
            min_patch_size=(16, 9),
            max_level=3,
        ).to(device)

        images = torch.randn(1, 3, height, width, device=device)
        logits = model(images)
        assert logits.shape == (1, 5)
        assert logits.device.type == device

        tokenizer = FractalHilbertTokenizer(min_patch_size=(16, 9), max_level=3).to(device)
        output = tokenizer.tokenize(images)
        sequence = output.sequences[0]
        assert sequence.tokens.device.type == device
        assert sequence.tokens.shape[0] > 0
