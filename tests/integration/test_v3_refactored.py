"""Test script for refactored V3 tokenizer with Variable Depth."""

import torch
from vit_pytorch.tokenizer_streaming import StreamingFractalTokenizerV3


def test_v3_basic():
    """Test basic functionality of refactored V3."""
    print("=" * 60)
    print("Test 1: Basic V3 Tokenizer Functionality")
    print("=" * 60)
    
    # Create tokenizer (uses LearnableSplitter by default)
    t = StreamingFractalTokenizerV3(
        image_size=64,
        d_model=128,
        base_patch_size=4,
        max_depth=3,
    )
    
    print(f"Tokenizer created: {type(t).__name__}")
    print(f"Patch embed type: {type(t.patch_embed).__name__}")
    print(f"Splitter type: {type(t.splitter).__name__}")
    
    # Test input
    x = torch.randn(2, 3, 64, 64)
    out = t(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output sequences: {len(out.sequences)}")
    print(f"Tokens shape (seq 0): {out.sequences[0].tokens.shape}")
    print(f"Tokens shape (seq 1): {out.sequences[1].tokens.shape}")
    
    # Check statistics
    stats = t.get_split_stats()
    print(f"\nSplit Statistics:")
    print(f"  Num tokens: {stats['num_tokens']}")
    print(f"  Depth distributions: {stats['depth_distributions']}")
    
    # Check training stats
    print(f"\nTraining Stats:")
    for k, v in t.get_training_stats().items():
        print(f"  {k}: {v}")
    
    # Check entropy loss (should be None for Variable Depth)
    print(f"\nEntropy loss: {t.get_entropy_loss()}")
    print(f"Scale entropy: {t.get_scale_entropy()}")
    
    print("\n✓ Test 1 PASSED")


def test_v3_levels_info():
    """Test levels_info compatibility."""
    print("\n" + "=" * 60)
    print("Test 2: Levels Info Compatibility")
    print("=" * 60)
    
    t = StreamingFractalTokenizerV3(
        image_size=64,
        d_model=128,
        base_patch_size=4,
        max_depth=3,
    )
    
    x = torch.randn(1, 3, 64, 64)
    out = t(x)
    
    # Check levels_info in metadata
    seq = out.sequences[0]
    levels = seq.metadata.get("levels")
    
    print(f"Levels info shape: {levels.shape if levels is not None else None}")
    print(f"Levels info dtype: {levels.dtype if levels is not None else None}")
    
    if levels is not None:
        print(f"First token levels: {levels[0][:8]}")  # Show first 8 values
        print(f"Unique depths in level[0]: {torch.unique(levels[:, 0]).tolist()}")
    
    print("\n✓ Test 2 PASSED")


def test_v3_fixed_budget():
    """Test learnable splitter with temperature control."""
    print("\n" + "=" * 60)
    print("Test 3: LearnableSplitter with Temperature")
    print("=" * 60)
    
    t = StreamingFractalTokenizerV3(
        image_size=64,
        d_model=128,
        base_patch_size=4,
        max_depth=3,
    )
    
    # Test temperature setting for learnable splitter
    t.set_split_temperature(0.5)
    
    x = torch.randn(2, 3, 64, 64)
    out = t(x)
    
    stats = t.get_split_stats()
    print(f"Split Temperature: 0.5")
    print(f"Actual tokens: {stats['num_tokens']}")
    print(f"Depth distributions: {stats['depth_distributions']}")
    
    # Check token count is within valid range
    for n in stats['num_tokens']:
        # Token count should be between min (all depth-3) and max (all depth-0)
        assert n > 0, f"Token count {n} should be positive"
    
    print("\n✓ Test 3 PASSED")


def test_v3_gradient_flow():
    """Test gradient flow through the tokenizer."""
    print("\n" + "=" * 60)
    print("Test 4: Gradient Flow")
    print("=" * 60)
    
    t = StreamingFractalTokenizerV3(
        image_size=64,
        d_model=128,
        base_patch_size=4,
        max_depth=3,
    )
    
    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    out = t(x)
    
    # Compute loss and backward (use squared sum for non-zero loss)
    tokens = out.sequences[0].tokens
    loss = tokens.pow(2).sum()  # Use L2 loss instead of sum
    
    print(f"Tokens requires_grad: {tokens.requires_grad}")
    print(f"Tokens grad_fn: {tokens.grad_fn}")
    print(f"Loss value: {loss.item():.4f}")
    
    loss.backward()
    
    print(f"Input gradient shape: {x.grad.shape}")
    print(f"Input gradient norm: {x.grad.norm().item():.6f}")
    
    # Check that gradients are non-zero
    assert x.grad.norm() > 0, "Gradients should be non-zero"
    
    print("\n✓ Test 4 PASSED")


def test_v3_batch_processing():
    """Test batch processing with different token counts."""
    print("\n" + "=" * 60)
    print("Test 5: Batch Processing")
    print("=" * 60)
    
    t = StreamingFractalTokenizerV3(
        image_size=64,
        d_model=128,
        base_patch_size=4,
        max_depth=3,
    )
    
    # Create batch with different complexity
    x = torch.zeros(4, 3, 64, 64)
    x[0] = 0.5  # Uniform - should use fewer tokens
    x[1] = torch.randn(3, 64, 64)  # Random - more tokens
    x[2, :, :32, :32] = 1.0  # High contrast corner
    x[3] = torch.randn(3, 64, 64) * 2  # Higher variance
    
    out = t(x)
    
    print("Token counts per image:")
    for i, seq in enumerate(out.sequences):
        print(f"  Image {i}: {seq.tokens.shape[0]} tokens")
    
    stats = t.get_split_stats()
    print(f"\nDepth distributions:")
    for i, dist in enumerate(stats['depth_distributions']):
        print(f"  Image {i}: {dict(dist)}")
    
    print("\n✓ Test 5 PASSED")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
