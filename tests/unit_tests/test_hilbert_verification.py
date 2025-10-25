import pytest

from vit_pytorch.fractal_curve_tokenizer import FractalHilbertTokenizer


@pytest.mark.parametrize("order", [2, 3, 4])
def test_generate_true_hilbert_curve_has_expected_length(order: int) -> None:
    tokenizer = FractalHilbertTokenizer()
    points = tokenizer.generate_true_hilbert_curve(order)

    expected_len = 4 ** order
    assert len(points) == expected_len
    assert len(set(points)) == expected_len


def test_hilbert_distance_matches_curve_sequence() -> None:
    tokenizer = FractalHilbertTokenizer()
    order = 3
    points = tokenizer.generate_true_hilbert_curve(order)
    grid_size = 1 << order

    distances = [tokenizer.hilbert_distance(x, y, grid_size) for x, y in points]
    assert distances == list(range(len(points)))
