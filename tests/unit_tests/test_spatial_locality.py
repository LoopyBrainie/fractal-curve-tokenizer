import pytest
import torch

from vit_pytorch.fractal_curve_tokenizer import FractalHilbertTokenizer


def _gradient_image(h: int, w: int) -> torch.Tensor:
    y = torch.linspace(0, 1, h).view(1, 1, h, 1)
    x = torch.linspace(0, 1, w).view(1, 1, 1, w)
    base = torch.cat([y.expand(-1, 1, -1, w), x.expand(-1, 1, h, -1)], dim=1)
    radial = torch.sqrt(
        (torch.arange(h).float().view(1, 1, h, 1) - h / 2) ** 2
        + (torch.arange(w).float().view(1, 1, 1, w) - w / 2) ** 2
    )
    radial = radial / radial.max().clamp(min=1)
    return torch.cat([base, radial], dim=1)


def test_token_count_scales_with_image_area() -> None:
    tokenizer = FractalHilbertTokenizer(min_patch_size=(4, 4), max_level=3)
    small = tokenizer.tokenize(_gradient_image(16, 16)).sequences[0].tokens.shape[0]
    large = tokenizer.tokenize(_gradient_image(32, 32)).sequences[0].tokens.shape[0]

    assert large >= small


@pytest.mark.parametrize(
    "height,width",
    [(1, 32), (32, 1), (3, 5), (5, 3), (9, 17)],
)
def test_tokenizer_handles_edge_shapes(height: int, width: int) -> None:
    tokenizer = FractalHilbertTokenizer(min_patch_size=(2, 2), max_level=4)
    output = tokenizer.tokenize(_gradient_image(height, width))
    sequence = output.sequences[0]

    assert sequence.tokens.shape[0] > 0
    assert sequence.metadata["levels"].shape[0] == sequence.tokens.shape[0]


def test_tokenization_is_deterministic_for_same_input() -> None:
    tokenizer = FractalHilbertTokenizer(min_patch_size=(4, 4), max_level=3)
    image = _gradient_image(24, 18)

    first = tokenizer.tokenize(image)
    second = tokenizer.tokenize(image)

    first_seq = first.sequences[0]
    second_seq = second.sequences[0]

    assert torch.allclose(first_seq.tokens, second_seq.tokens)
    assert torch.equal(first_seq.metadata["levels"], second_seq.metadata["levels"])
