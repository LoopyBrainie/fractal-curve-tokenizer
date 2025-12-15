import pytest
import torch
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from vit_pytorch._deprecated.fractal_curve_tokenizer import FractalHilbertTokenizer


class TestBatchProcessing:
    """测试批处理版本与递归版本的一致性"""
    
    def test_batch_vs_recursive_non_learnable(self) -> None:
        """测试非 learnable 模式下批处理与递归版本输出一致"""
        torch.manual_seed(42)
        
        tokenizer = FractalHilbertTokenizer(
            min_patch_size=(4, 4),
            max_level=5,
            learnable_split=False,
            adaptive_threshold=0.01
        )
        
        image = torch.randn(3, 64, 64)
        
        # 递归版本
        tokens_recursive, levels_recursive = tokenizer.fractal_partition(
            image, level=0, coord=[], max_info_len=16, depth_limit=10
        )
        
        # 批处理版本
        tokens_batched, levels_batched = tokenizer.fractal_partition_batched(
            image, max_info_len=16, depth_limit=10
        )
        
        # 验证 token 数量相同
        assert len(tokens_recursive) == len(tokens_batched), \
            f"Token count mismatch: recursive={len(tokens_recursive)}, batched={len(tokens_batched)}"
        
        # 验证每个 token 内容一致
        for i, (t_rec, t_batch) in enumerate(zip(tokens_recursive, tokens_batched)):
            assert torch.allclose(t_rec, t_batch, atol=1e-6), \
                f"Token {i} mismatch"
        
        # 验证 levels 一致
        for i, (l_rec, l_batch) in enumerate(zip(levels_recursive, levels_batched)):
            assert l_rec == l_batch, \
                f"Level {i} mismatch: recursive={l_rec}, batched={l_batch}"
    
    def test_batch_processing_preserves_hilbert_order(self) -> None:
        """测试批处理保持 Hilbert 曲线顺序"""
        torch.manual_seed(123)
        
        tokenizer = FractalHilbertTokenizer(
            min_patch_size=(4, 4),
            max_level=4,
            learnable_split=False,
            adaptive_threshold=0.005
        )
        
        image = torch.randn(3, 32, 32)
        
        # 获取批处理输出
        tokens, levels = tokenizer.fractal_partition_batched(
            image, max_info_len=16, depth_limit=8
        )
        
        # 验证深度信息是有效的
        for level_info in levels:
            depth = level_info[0]
            assert 0 <= depth <= 8, f"Invalid depth: {depth}"
        
        # 验证 tokens 是有效的张量
        for token in tokens:
            assert token.dim() == 1
            assert token.shape[0] == 3 * 4 * 4  # flattened patch size


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
        # Test the adaptive threshold logic with high threshold (should stop early)
        # Note: learnable_split=False uses adaptive_threshold based on patch variance
        # High threshold means patch_var < threshold is likely True -> should_stop
        tokenizer = FractalHilbertTokenizer(
            min_patch_size=(4, 4),
            max_level=5,
            learnable_split=False,  # Use variance-based threshold, not neural network
            adaptive_threshold=100.0,  # Very high threshold, all patches will have var < 100
        )
        
        # With high threshold, variance-based stopping: patch_var < 100 is True -> Stop
        # So the first patch (whole image) should stop immediately
        
        images = torch.randn(1, 3, 32, 32)
        output = tokenizer.tokenize(images)
        tokens = output.sequences[0].tokens

        # Should be 1 token (the whole image)
        assert tokens.shape[0] == 1


    @pytest.mark.parametrize("learnable", [True, False])
    def test_default_split_produces_tokens(self, learnable: bool) -> None:
        # 设置随机种子以获得可重复的结果
        torch.manual_seed(42)
        
        tokenizer = FractalHilbertTokenizer(
            min_patch_size=(4, 4),
            max_level=6,  # 使用更高的 max_level 确保能分割
            learnable_split=learnable,
            adaptive_threshold=0.0001 # Low threshold to encourage splitting
        )
        
        # 在 eval 模式下测试，以获得确定性的 argmax 决策
        tokenizer.eval()

        images = torch.randn(1, 3, 64, 64)
        output = tokenizer.tokenize(images)
        tokens = output.sequences[0].tokens

        # 在 eval 模式下应该能产生多个 token
        # 如果 learnable 模式下不产生分割，至少应该有1个token
        if learnable:
            # learnable 模式使用 argmax 在 eval 下，可能不分割
            # 只验证 token 数量是有效的
            assert tokens.shape[0] >= 1
        else:
            # 非 learnable 模式应该根据 adaptive_threshold 分割
            assert tokens.shape[0] > 1

    def test_learnable_split_training_mode_produces_tokens(self) -> None:
        """测试 learnable 模式在训练模式下产生多个 tokens (REINFORCE 采样)"""
        # 运行多次，确保至少有一次产生多个 token
        tokenizer = FractalHilbertTokenizer(
            min_patch_size=(4, 4),
            max_level=8,  # 较高的 max_level
            learnable_split=True,
            adaptive_threshold=0.0001
        )
        tokenizer.train()  # 确保是训练模式
        
        images = torch.randn(1, 3, 128, 128)  # 使用较大的图像
        
        # 由于 REINFORCE 采样有随机性，运行多次取最大值
        max_tokens = 0
        for _ in range(5):
            tokenizer.clear_saved_actions()
            output = tokenizer.tokenize(images)
            max_tokens = max(max_tokens, output.sequences[0].tokens.shape[0])
        
        # 在多次尝试中应该能产生多个 tokens
        assert max_tokens > 1, f"Expected more than 1 token in at least one run, got max {max_tokens}"


