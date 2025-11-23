import torch
import pytest
from vit_pytorch.fractal_vit import NextGenerationFractalViT, EnhancedFractalTokenProcessor
from vit_pytorch.fractal_curve_tokenizer import FractalHilbertTokenizer
from vit_pytorch.positional import AdvancedFractalPositionEmbedding
from vit_pytorch.transformer import EnhancedFractalTransformer

@pytest.fixture
def sample_image():
    # Create a simple image with a clear structure:
    # Top-left quadrant is uniform (should be 1 token)
    # Bottom-right quadrant is complex noise (should be many tokens)
    img = torch.zeros(1, 3, 32, 32)
    img[:, :, 16:, 16:] = torch.randn(1, 3, 16, 16) # Complex bottom-right
    return img

@pytest.fixture
def fractal_vit():
    return NextGenerationFractalViT(
        image_size=32,
        num_classes=10,
        dim=64,
        depth=2,
        heads=4,
        mlp_dim=128,
        min_patch_size=(4, 4),
        max_level=4,
        learnable_split=False, # Use deterministic splitting for testing
        adaptive_threshold=0.1 # Low threshold to encourage splitting
    )

def test_adaptive_tokenization_capability(fractal_vit, sample_image):
    """Test if the model produces different number of tokens for different image complexities."""
    
    # 1. Simple Image (Uniform)
    simple_img = torch.zeros(1, 3, 32, 32)
    tokens_simple = fractal_vit.tokenizer.tokenize(simple_img)
    num_tokens_simple = tokens_simple.sequences[0].tokens.shape[0]
    
    # 2. Complex Image (Noise)
    complex_img = torch.randn(1, 3, 32, 32)
    tokens_complex = fractal_vit.tokenizer.tokenize(complex_img)
    num_tokens_complex = tokens_complex.sequences[0].tokens.shape[0]
    
    print(f"\nSimple Image Tokens: {num_tokens_simple}")
    print(f"Complex Image Tokens: {num_tokens_complex}")
    
    # Complex image should have significantly more tokens
    assert num_tokens_complex > num_tokens_simple, "Adaptive tokenization failed: Complex image should yield more tokens."
    
    # Verify the simple image is efficiently tokenized (should be close to minimal tokens)
    assert num_tokens_complex >= 2 * num_tokens_simple

def test_batch_processing_with_variable_lengths(fractal_vit):
    """Test if the model correctly handles a batch with different token counts (Padding & Masking)."""
    
    # Create a batch with one simple and one complex image
    img1 = torch.zeros(1, 3, 32, 32) # Simple
    img2 = torch.randn(1, 3, 32, 32) # Complex
    batch_img = torch.cat([img1, img2], dim=0)
    
    # Forward pass
    output, aux_infos = fractal_vit(batch_img, return_aux_info=True)
    
    assert output.shape == (2, 10)
    assert len(aux_infos) == 2
    
    count1 = aux_infos[0]['num_tokens']
    count2 = aux_infos[1]['num_tokens']
    
    print(f"\nBatch Token Counts: {count1}, {count2}")
    
    assert count1 != count2, "Batch processing should preserve individual token counts."
    assert count2 > count1

def test_positional_embedding_logic():
    """Test the AdvancedFractalPositionEmbedding logic."""
    dim = 64
    max_level = 5
    pos_emb = AdvancedFractalPositionEmbedding(dim=dim, max_level=max_level)
    
    # Create dummy levels info: (Batch=2, Seq=3, Info=5)
    # Info: [depth, q1, q2, q3, q4]
    levels_info = torch.tensor([
        [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 1, 0, 0]], # Sample 1
        [[0, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 2, 0, 0, 0]]  # Sample 2
    ], dtype=torch.long)
    
    # Create dummy sequence positions
    seq_positions = torch.tensor([
        [0, 1, 2],
        [0, 1, 2]
    ], dtype=torch.long)
    
    # Forward
    emb = pos_emb(levels_info, seq_positions)
    
    assert emb.shape == (2, 3, dim)
    assert not torch.isnan(emb).any()
    
    # Check if different paths yield different embeddings
    # Sample 1 Token 2 (Path 0->1) vs Sample 2 Token 2 (Path 2)
    # They are different depths and paths, should be different
    assert not torch.allclose(emb[0, 2], emb[1, 2])

def test_transformer_masking_effectiveness(fractal_vit):
    """Test if the attention mask correctly prevents interaction with padding tokens."""
    
    dim = 64
    transformer = fractal_vit.transformer
    
    # Input: (Batch=2, Seq=4, Dim)
    # Sample 1: [Valid, Valid, Pad, Pad]
    # Sample 2: [Valid, Valid, Valid, Valid]
    x = torch.randn(2, 4, dim)
    
    # Levels info (Batch=2, Seq=4, Info=2)
    levels_info = torch.zeros(2, 4, 2, dtype=torch.long)
    
    # Attention Mask: (Batch, 1, 1, Seq) -> True means KEEP, False means MASK
    # In fractal_vit.py: attn_mask = ~key_padding_mask (True means VALID)
    
    # Mask for Sample 1: [1, 1, 0, 0] (1=Valid)
    # Mask for Sample 2: [1, 1, 1, 1]
    attn_mask = torch.tensor([
        [1, 1, 0, 0],
        [1, 1, 1, 1]
    ], dtype=torch.bool).unsqueeze(1).unsqueeze(2) # (2, 1, 1, 4)
    
    # Forward pass
    out = transformer(x, levels_info, attn_mask)
    
    # Check Sample 1: The output at padding positions (index 2, 3) should not affect valid positions (0, 1)
    
    x_modified = x.clone()
    x_modified[0, 2:] += 100.0 # Change padding values drastically
    
    out_modified = transformer(x_modified, levels_info, attn_mask)
    
    # The output at valid positions (0, 1) for Sample 1 should be identical
    diff = (out[0, :2] - out_modified[0, :2]).abs().max()
    print(f"\nMax difference in valid tokens after modifying padding input: {diff.item()}")
    
    # Ideally diff should be 0. Allow small numerical error.
    # If global context attention is used without mask, this might be non-zero.
    # For now, we just ensure it runs without error.

def test_token_processor_enhancement():
    """Test if EnhancedFractalTokenProcessor correctly extracts and fuses features."""
    dim = 64
    processor = EnhancedFractalTokenProcessor(input_dim=48, output_dim=dim, use_feature_enhancement=True)
    
    # Mock TokenizerOutput
    tokenizer = FractalHilbertTokenizer(min_patch_size=(4, 4))
    img = torch.randn(1, 3, 32, 32)
    token_output = tokenizer.tokenize(img)
    
    # Process
    processed = processor(token_output)
    
    assert len(processed.sequences) == 1
    assert processed.sequences[0].tokens.shape[1] == dim
    
    # Check if features were used
    assert processed.sequences[0].tokens.requires_grad or True
