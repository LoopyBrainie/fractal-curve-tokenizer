import pytest

from vit_pytorch.fractal_curve_tokenizer import FractalHilbertTokenizer


@pytest.mark.parametrize(
    "level,height,width,expected",
    [
        (2, 16, 16, {0, 1, 2, 3}),
        (2, 16, 48, [2, 0, 1, 3]),  # 宽矩形使用水平连续顺序
        (2, 48, 16, [0, 1, 3, 2]),  # 高矩形使用垂直连续顺序
    ],
)
def test_get_enhanced_hilbert_order(level: int, height: int, width: int, expected) -> None:
    tokenizer = FractalHilbertTokenizer()
    order = tokenizer.get_enhanced_hilbert_order(level, height, width)

    assert sorted(order) == [0, 1, 2, 3]

    if isinstance(expected, list):
        assert order == expected
    else:
        assert set(order) == expected


def test_adaptive_hilbert_mapping_matches_level_zero_order() -> None:
    tokenizer = FractalHilbertTokenizer()

    mapping = tokenizer.adaptive_hilbert_mapping([0, 1, 2, 3], h=20, w=12)
    order = tokenizer.get_enhanced_hilbert_order(level=0, h=20, w=12)

    assert mapping == order


def test_generate_true_hilbert_curve_has_unique_points() -> None:
    tokenizer = FractalHilbertTokenizer()
    points = tokenizer.generate_true_hilbert_curve(3)

    assert len(points) == 4 ** 3
    assert len(set(points)) == len(points)
