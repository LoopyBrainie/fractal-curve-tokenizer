import pytest
import torch

from vit_pytorch.fractal_curve_tokenizer import FractalHilbertTokenizer


def test_adaptive_split_returns_quadrants_when_possible() -> None:
    tokenizer = FractalHilbertTokenizer(min_patch_size=(4, 4))
    patch = torch.randn(3, 18, 18)

    sub_patches = tokenizer._adaptive_split(patch, 18, 18, True, True)

    assert len(sub_patches) == 4
    for sub in sub_patches:
        assert sub.ndim == 3
        assert sub.shape[1] > 0 and sub.shape[2] > 0


def test_adaptive_split_respects_single_axis_split() -> None:
    tokenizer = FractalHilbertTokenizer(min_patch_size=(4, 4))
    patch = torch.randn(3, 8, 32)

    sub_patches = tokenizer._adaptive_split(patch, 8, 32, False, True)

    assert len(sub_patches) == 2
    widths = {sub.shape[2] for sub in sub_patches}
    assert sum(widths) == 32


def test_high_threshold_disables_recursive_splitting() -> None:
    tokenizer = FractalHilbertTokenizer(
        min_patch_size=(4, 4),
        max_level=5,
        learnable_split=True,
        adaptive_threshold=1.1,  # 阈值大于1，强制停止
    )

    images = torch.randn(1, 3, 32, 32)
    output = tokenizer.tokenize(images)
    tokens = output.sequences[0].tokens

    assert tokens.shape[0] == 1


@pytest.mark.parametrize("learnable", [True, False])
def test_default_split_produces_tokens(learnable: bool) -> None:
    tokenizer = FractalHilbertTokenizer(
        min_patch_size=(4, 4),
        max_level=3,
        learnable_split=learnable,
    )

    images = torch.randn(1, 3, 64, 64)
    output = tokenizer.tokenize(images)
    tokens = output.sequences[0].tokens

    assert tokens.shape[0] > 1
