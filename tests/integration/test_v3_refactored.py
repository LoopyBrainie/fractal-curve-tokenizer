"""Test script for refactored V3 tokenizer with Variable Depth (I98-1 架构).

I98-1 架构变更:
- Splitter 从 Tokenizer 内部移到外部作为独立组件
- Tokenizer 现在需要外部传入 split_result
"""

import torch
from vit_pytorch.tokenizer_streaming import StreamingFractalTokenizerV3
from vit_pytorch.gumbel_topk_splitter import GumbelTopKSplitter
from vit_pytorch.config import SplitterConfig


def _create_pipeline():
    """创建 I98-1 pipeline 组件."""
    tokenizer = StreamingFractalTokenizerV3(
        image_size=64,
        d_model=128,
        base_patch_size=4,
        max_depth=3,
    )

    splitter_config = SplitterConfig(
        feature_dim=128,
        min_patch_size=4,
        max_depth_limit=3,
        hidden_dim=64,
        intermediate_dim=64,
        pool_size=4,
        K_min=8,
        K_max=32,
    )
    splitter = GumbelTopKSplitter(
        config=splitter_config,
        image_size=(64, 64),
    )
    return tokenizer, splitter


def _run_pipeline(tokenizer, splitter, x):
    """运行 I98-1 pipeline."""
    features = tokenizer.shared_conv(x)
    split_result = splitter(
        features,
        image_size=(x.shape[2], x.shape[3]),
        hard=not splitter.training,
    )
    out = tokenizer.tokenize(x, split_result)
    return out


def test_v3_basic():
    """Test basic functionality of refactored V3 (I98-1 Pipeline 架构)."""
    print("=" * 60)
    print("Test 1: Basic V3 Tokenizer Functionality (I98-1)")
    print("=" * 60)

    tokenizer, splitter = _create_pipeline()

    print(f"Tokenizer created: {type(tokenizer).__name__}")
    print(f"Patch embed type: {type(tokenizer.patch_embed).__name__}")
    print(f"Splitter type: {type(splitter).__name__}")

    # I98-1: 使用完整 pipeline
    x = torch.randn(2, 3, 64, 64)
    out = _run_pipeline(tokenizer, splitter, x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output sequences: {len(out.sequences)}")
    print(f"Tokens shape (seq 0): {out.sequences[0].tokens.shape}")
    print(f"Tokens shape (seq 1): {out.sequences[1].tokens.shape}")

    # Check statistics
    stats = tokenizer.get_split_stats()
    print(f"\nSplit Statistics:")
    print(f"  Num tokens: {stats['num_tokens']}")
    print(f"  Depth distributions: {stats['depth_distributions']}")

    # Check training stats
    print(f"\nTraining Stats:")
    for k, v in tokenizer.get_training_stats().items():
        print(f"  {k}: {v}")

    # Check entropy loss (should be None for Variable Depth)
    print(f"\nEntropy loss: {tokenizer.get_entropy_loss()}")
    print(f"Scale entropy: {tokenizer.get_scale_entropy()}")

    print("\n Test 1 PASSED")


def test_v3_levels_info():
    """Test levels_info compatibility."""
    print("\n" + "=" * 60)
    print("Test 2: Levels Info Compatibility")
    print("=" * 60)

    tokenizer, splitter = _create_pipeline()

    x = torch.randn(1, 3, 64, 64)
    out = _run_pipeline(tokenizer, splitter, x)

    # Check levels_info in metadata
    seq = out.sequences[0]
    levels = seq.metadata.get("levels")

    print(f"Levels info shape: {levels.shape if levels is not None else None}")
    print(f"Levels info dtype: {levels.dtype if levels is not None else None}")

    if levels is not None:
        print(f"First token levels: {levels[0][:8]}")  # Show first 8 values
        print(f"Unique depths in level[0]: {torch.unique(levels[:, 0]).tolist()}")

    print("\n Test 2 PASSED")


def test_v3_fixed_budget():
    """Test learnable splitter with temperature control."""
    print("\n" + "=" * 60)
    print("Test 3: LearnableSplitter with Temperature")
    print("=" * 60)

    tokenizer, splitter = _create_pipeline()

    # I98-1: 温度控制现在在 splitter 上
    splitter.set_temperature(0.5)

    x = torch.randn(2, 3, 64, 64)
    out = _run_pipeline(tokenizer, splitter, x)

    print(f"Output tokens per image: {[s.tokens.shape[0] for s in out.sequences]}")
    print("\n Test 3 PASSED")
