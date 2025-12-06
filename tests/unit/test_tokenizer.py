import pytest
import torch
from vit_pytorch.fractal_curve_tokenizer import FractalHilbertTokenizer

class TestFractalHilbertTokenizer:
    
    @pytest.fixture
    def tokenizer(self):
        return FractalHilbertTokenizer(min_patch_size=(4, 4), max_level=5)

    @pytest.mark.parametrize(
        "batch,channels,height,width,min_patch,max_level",
        [
            (1, 3, 32, 32, (4, 4), 3),
            (2, 3, 48, 96, (4, 4), None),
            (1, 1, 40, 12, (2, 2), 5),
        ],
    )
    def test_tokenize_shapes_and_levels(
        self,
        batch: int,
        channels: int,
        height: int,
        width: int,
        min_patch: tuple[int, int],
        max_level: int | None,
    ) -> None:
        tokenizer = FractalHilbertTokenizer(min_patch_size=min_patch, max_level=max_level, channels=channels)
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

    def test_tokenize_handles_extreme_aspect_ratio(self) -> None:
        tokenizer = FractalHilbertTokenizer(min_patch_size=(4, 4), max_level=None)
        images = torch.randn(1, 3, 24, 96)

        output = tokenizer.tokenize(images)
        sequence = output.sequences[0]

        assert sequence.tokens.shape[0] > 0
        assert sequence.metadata["levels"].shape[0] == sequence.tokens.shape[0]

    @torch.no_grad()
    def test_tokenize_respects_input_device(self) -> None:
        # Skip if CUDA not available, or just test CPU
        device = "cpu"
        tokenizer = FractalHilbertTokenizer(min_patch_size=(4, 4))
        images = torch.randn(1, 3, 32, 32, device=device)

        output = tokenizer.tokenize(images)
        assert output.sequences[0].tokens.device == images.device

    def test_adaptive_split_returns_quadrants_when_possible(self, tokenizer) -> None:
        patch = torch.randn(3, 18, 18)
        sub_patches = tokenizer._adaptive_split(patch, 18, 18, True, True)

        assert len(sub_patches) == 4
        for sub in sub_patches:
            assert sub.ndim == 3
            assert sub.shape[1] > 0 and sub.shape[2] > 0

    def test_adaptive_split_respects_single_axis_split(self, tokenizer) -> None:
        patch = torch.randn(3, 8, 32)
        sub_patches = tokenizer._adaptive_split(patch, 8, 32, False, True)

        assert len(sub_patches) == 2
        widths = {sub.shape[2] for sub in sub_patches}
        assert sum(widths) == 32

    def test_high_threshold_disables_recursive_splitting(self) -> None:
        # Test the learnable split logic with high threshold (should stop early)
        # Note: learnable_split=True requires split_decision module, which is initialized in __init__
        tokenizer = FractalHilbertTokenizer(
            min_patch_size=(4, 4),
            max_level=5,
            learnable_split=True,
            adaptive_threshold=1.1,  # Threshold > 1 means probability (0-1) will always be < threshold
        )
        
        # Mock split_decision to return 0.5 (which is < 1.1, so should_stop = True)
        # Wait, logic is: should_stop = split_prob < self.adaptive_threshold
        # If threshold is 1.1, and prob is 0.5, then 0.5 < 1.1 is True -> Stop.
        # Correct.
        
        images = torch.randn(1, 3, 32, 32)
        output = tokenizer.tokenize(images)
        tokens = output.sequences[0].tokens

        # Should be 1 token (the whole image)
        assert tokens.shape[0] == 1

    @pytest.mark.parametrize("learnable", [True, False])
    def test_default_split_produces_tokens(self, learnable: bool) -> None:
        tokenizer = FractalHilbertTokenizer(
            min_patch_size=(4, 4),
            max_level=3,
            learnable_split=learnable,
            adaptive_threshold=0.0001 # Low threshold to encourage splitting
        )

        images = torch.randn(1, 3, 64, 64)
        output = tokenizer.tokenize(images)
        tokens = output.sequences[0].tokens

        assert tokens.shape[0] > 1
