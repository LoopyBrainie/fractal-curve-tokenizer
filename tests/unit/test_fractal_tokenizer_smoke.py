import torch

from vit_pytorch.fractal_curve_tokenizer import FractalHilbertTokenizer
from vit_pytorch.tokenization import TokenizerOutput


def test_tokenizer_returns_tokens_and_levels():
    tokenizer = FractalHilbertTokenizer(min_patch_size=(4, 4), max_level=3)
    dummy = torch.randn(2, 3, 32, 32)

    output = tokenizer.tokenize(dummy)
    assert isinstance(output, TokenizerOutput)

    tokens = output.tokens_list()
    levels = output.levels_list()

    assert len(tokens) == 2
    assert len(levels) == 2
    assert tokens[0].dim() == 2
    assert levels[0].dim() == 2
