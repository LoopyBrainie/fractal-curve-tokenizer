import pytest
import torch

from vit_pytorch.fractal_curve_tokenizer import FractalHilbertTokenizer


@pytest.mark.parametrize(
    "batch,channels,height,width,min_patch,max_level",
    [
        (1, 3, 32, 32, (4, 4), 3),
        (2, 3, 48, 96, (4, 4), None),
        (1, 1, 40, 12, (2, 2), 5),
    ],
)
def test_tokenize_shapes_and_levels(
    batch: int,
    channels: int,
    height: int,
    width: int,
    min_patch: tuple[int, int],
    max_level: int | None,
) -> None:
    tokenizer = FractalHilbertTokenizer(min_patch_size=min_patch, max_level=max_level)
    images = torch.randn(batch, channels, height, width)

    output = tokenizer.tokenize(images)
    assert len(output.sequences) == batch

    expected_dim = channels * min_patch[0] * min_patch[1]

    for sequence in output:
        tokens = sequence.tokens
        levels = sequence.metadata["levels"]

        assert tokens.ndim == 2
        assert levels.ndim == 2
        assert tokens.shape[1] == expected_dim
        assert tokens.shape[0] == levels.shape[0]

        # 深度信息应不超过动态深度限制
        estimated_max = tokenizer._estimate_max_possible_level(height, width)
        dynamic_cap = max_level if max_level is not None else max(estimated_max + 5, 12)
        if levels.numel() > 0:
            assert levels[:, 0].max().item() <= dynamic_cap


def test_tokenize_handles_extreme_aspect_ratio() -> None:
    tokenizer = FractalHilbertTokenizer(min_patch_size=(4, 4), max_level=None)
    images = torch.randn(1, 3, 24, 96)

    output = tokenizer.tokenize(images)
    sequence = output.sequences[0]

    assert sequence.tokens.shape[0] > 0
    assert sequence.metadata["levels"].shape[0] == sequence.tokens.shape[0]


@torch.no_grad()
def test_tokenize_respects_input_device() -> None:
    tokenizer = FractalHilbertTokenizer(min_patch_size=(4, 4))
    images = torch.randn(1, 3, 32, 32, device="cpu")

    output = tokenizer.tokenize(images)
    assert output.sequences[0].tokens.device == images.device
