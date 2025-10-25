import math

import pytest
import torch

from vit_pytorch.fractal_curve_tokenizer import FractalHilbertTokenizer


@pytest.mark.parametrize(
    "height,width",
    [
        (32, 32),
        (64, 32),
        (48, 96),
        (45, 30),
        (128, 72),
    ],
)
def test_fractal_tokenizer_handles_varied_resolutions(height: int, width: int) -> None:
    tokenizer = FractalHilbertTokenizer(min_patch_size=(4, 4), max_level=None)
    images = torch.randn(2, 3, height, width)

    output = tokenizer.tokenize(images)
    assert len(output.sequences) == images.shape[0]

    expected_token_dim = 3 * tokenizer.min_patch_size[0] * tokenizer.min_patch_size[1]

    estimated_max_level = tokenizer._estimate_max_possible_level(height, width)
    dynamic_depth_cap = (
        tokenizer.max_level
        if tokenizer.max_level is not None
        else max(estimated_max_level + 5, 12)
    )
    min_patch_dim = max(1, min(tokenizer.min_patch_size))
    longest_edge = max(height, width)
    ratio = longest_edge / min_patch_dim if min_patch_dim > 0 else 1
    additional_levels = int(math.ceil(math.log2(ratio))) if ratio > 0 else 0
    max_info_len = max(dynamic_depth_cap + additional_levels + 4, 16)

    for sequence in output:
        tokens = sequence.tokens
        levels = sequence.metadata["levels"]

        assert tokens.shape[0] == levels.shape[0] > 0
        assert tokens.shape[1] == expected_token_dim
        assert levels.shape[1] == max_info_len

        # 深度信息应在 [0, dynamic_depth_cap] 范围内
        assert levels[:, 0].min().item() >= 0
        assert levels[:, 0].max().item() <= dynamic_depth_cap

        # 路径编码中非零元素数量不应超过 max_info_len
        non_zero_counts = (levels != 0).sum(dim=1)
        assert torch.all(non_zero_counts <= max_info_len)
