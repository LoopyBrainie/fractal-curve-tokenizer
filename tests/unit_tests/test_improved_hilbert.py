import torch

from vit_pytorch.fractal_curve_tokenizer import FractalHilbertTokenizer


def test_generate_true_hilbert_curve_is_continuous() -> None:
    tokenizer = FractalHilbertTokenizer()
    points = tokenizer.generate_true_hilbert_curve(3)

    max_jump = 0
    for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
        jump = abs(x1 - x2) + abs(y1 - y2)
        max_jump = max(max_jump, jump)

    assert max_jump == 1


def test_enhanced_hilbert_order_consistency() -> None:
    tokenizer = FractalHilbertTokenizer(max_level=6)
    result_a = tokenizer.get_enhanced_hilbert_order(level=3, h=24, w=12)
    result_b = tokenizer.get_enhanced_hilbert_order(level=3, h=24, w=12)

    assert result_a == result_b
    assert sorted(result_a) == [0, 1, 2, 3]


def test_tokenizer_integration_produces_levels() -> None:
    tokenizer = FractalHilbertTokenizer(min_patch_size=(4, 4), max_level=3)
    image = torch.randn(1, 3, 64, 48)

    output = tokenizer.tokenize(image)
    sequence = output.sequences[0]

    assert sequence.tokens.shape[0] == sequence.metadata["levels"].shape[0]
    assert sequence.metadata["levels"].shape[1] >= 1
