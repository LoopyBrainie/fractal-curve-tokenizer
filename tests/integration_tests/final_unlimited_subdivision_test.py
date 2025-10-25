import pytest
import torch

from vit_pytorch.fractal_curve_tokenizer import FractalHilbertTokenizer


@pytest.mark.parametrize(
    "min_patch,threshold,height,width",
    [
        ((1, 1), 0.1, 32, 32),
        ((2, 2), 0.5, 48, 24),
        ((4, 4), 0.8, 64, 64),
    ],
)
def test_unlimited_subdivision_generates_tokens(
    min_patch: tuple[int, int],
    threshold: float,
    height: int,
    width: int,
) -> None:
    tokenizer = FractalHilbertTokenizer(
        min_patch_size=min_patch,
        max_level=None,
        learnable_split=True,
        adaptive_threshold=threshold,
    )

    image = torch.randn(1, 3, height, width)
    output = tokenizer.tokenize(image)
    sequence = output.sequences[0]

    assert sequence.tokens.shape[0] > 0
    assert sequence.metadata["levels"].shape[0] == sequence.tokens.shape[0]

    levels = sequence.metadata["levels"]
    if levels.numel() > 0:
        estimated_max = tokenizer._estimate_max_possible_level(height, width)
        dynamic_cap = max(estimated_max + 5, 12)
        assert levels[:, 0].max().item() <= dynamic_cap + 5


def test_feature_extraction_returns_valid_features() -> None:
    tokenizer = FractalHilbertTokenizer(min_patch_size=(1, 1), learnable_split=True)
    patch = torch.randn(3, 8, 8)

    features = tokenizer._extract_enhanced_patch_features(patch, level=2, h=8, w=8)

    assert features.shape == (1, 6)
    assert torch.isfinite(features).all()

    if tokenizer.split_decision is not None:
        prob = tokenizer.split_decision(features)
        assert prob.shape == (1,)
        assert 0.0 <= prob.item() <= 1.0
