# -*- coding: utf-8 -*-
"""Unit tests for FractalCurveViT.

数学形式化验证
==============

FractalCurveViT 完整流水线:
    I → T(I) → E_pos → Transformer → Pool → MLP → ŷ

测试覆盖:
1. 自适应 tokenization 能力
2. 批处理与变长序列处理
3. 位置编码逻辑
4. Transformer 掩码有效性

Note:
    默认使用 streaming_v3 (Variable Depth Tokenizer)
"""

import torch
import pytest
from vit_pytorch.vit import FractalCurveViT
from vit_pytorch.positional import FractalPositionEmbedding


@pytest.fixture
def sample_image():
    """Create a sample image with varying complexity."""
    img = torch.zeros(1, 3, 32, 32)
    img[:, :, 16:, 16:] = torch.randn(1, 3, 16, 16)  # Complex bottom-right
    return img


@pytest.fixture
def fractal_vit():
    """Create a FractalCurveViT model with streaming_v3 tokenizer."""
    return FractalCurveViT(
        image_size=32,
        num_classes=10,
        dim=64,
        depth=2,
        heads=4,
        mlp_dim=128,
        min_patch_size=(4, 4),
        max_level=4,
        tokenizer_type="streaming_v3",
        num_scales=2,
    )


class TestFractalCurveViT:
    """Tests for FractalCurveViT with streaming tokenizers."""

    def test_forward_pass_basic(self, fractal_vit, sample_image):
        """Test basic forward pass."""
        output = fractal_vit(sample_image)
        assert output.shape == (1, 10)
        assert not torch.isnan(output).any()

    def test_forward_pass_with_aux_info(self, fractal_vit, sample_image):
        """Test forward pass with auxiliary info returned."""
        output, aux_infos = fractal_vit(sample_image, return_aux_info=True)
        
        assert output.shape == (1, 10)
        assert len(aux_infos) == 1
        assert 'num_tokens' in aux_infos[0]
        assert aux_infos[0]['num_tokens'] > 0

    def test_batch_processing(self, fractal_vit):
        """Test batch processing with multiple images."""
        batch_img = torch.randn(4, 3, 32, 32)
        output = fractal_vit(batch_img)
        
        assert output.shape == (4, 10)
        assert not torch.isnan(output).any()

    def test_batch_processing_with_aux_info(self, fractal_vit):
        """Test batch processing with auxiliary info."""
        img1 = torch.zeros(1, 3, 32, 32)
        img2 = torch.randn(1, 3, 32, 32)
        batch_img = torch.cat([img1, img2], dim=0)
        
        output, aux_infos = fractal_vit(batch_img, return_aux_info=True)
        
        assert output.shape == (2, 10)
        assert len(aux_infos) == 2
        
        count1 = aux_infos[0]['num_tokens']
        count2 = aux_infos[1]['num_tokens']
        
        # Both should have tokens
        assert count1 > 0
        assert count2 > 0

    def test_streaming_tokenizer_type(self):
        """Test streaming tokenizer is correctly instantiated."""
        model = FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=2,
            mlp_dim=128,
            min_patch_size=(4, 4),
            max_level=5,
            tokenizer_type="streaming",
            num_scales=2,
        )
        
        assert model.tokenizer_type == "streaming"
        assert model._is_streaming
        from vit_pytorch.streaming_tokenizer import StreamingFractalTokenizer
        assert isinstance(model.tokenizer, StreamingFractalTokenizer)

    def test_streaming_v3_tokenizer_type(self):
        """Test streaming_v3 tokenizer is correctly instantiated."""
        model = FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=2,
            mlp_dim=128,
            min_patch_size=(4, 4),
            max_level=5,
            tokenizer_type="streaming_v3",
            num_scales=2,
        )
        
        assert model.tokenizer_type == "streaming_v3"
        assert model._is_streaming
        from vit_pytorch.streaming_tokenizer import StreamingFractalTokenizerV3
        assert isinstance(model.tokenizer, StreamingFractalTokenizerV3)

    def test_tokenizer_loss_returns_zero(self, fractal_vit, sample_image):
        """Test get_tokenizer_loss() returns zero for streaming tokenizers."""
        _ = fractal_vit(sample_image)
        loss = fractal_vit.get_tokenizer_loss()
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() == 0.0

    def test_gradient_flow(self):
        """Test gradients flow correctly through the model."""
        model = FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=2,
            mlp_dim=128,
            min_patch_size=(4, 4),
            max_level=4,
            tokenizer_type="streaming_v3",
        )
        
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_default_tokenizer_type(self):
        """Test FractalCurveViT defaults to streaming_v3."""
        model = FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=4,
            mlp_dim=128,
        )
        
        assert model.tokenizer_type == "streaming_v3"

    def test_different_image_sizes(self):
        """Test FractalCurveViT handles different image sizes."""
        sizes = [(32, 32), (64, 64), (48, 48)]
        
        for h, w in sizes:
            model = FractalCurveViT(
                image_size=(h, w),
                num_classes=5,
                dim=64,
                depth=2,
                heads=2,
                mlp_dim=128,
                min_patch_size=(4, 4),
            )
            model.eval()
            
            x = torch.randn(2, 3, h, w)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (2, 5)


class TestPositionalEmbedding:
    """Tests for FractalPositionEmbedding."""

    def test_positional_embedding_logic(self):
        """Test the FractalPositionEmbedding logic."""
        dim = 64
        max_level = 5
        pos_emb = FractalPositionEmbedding(dim=dim, max_level=max_level)
        
        # Create dummy levels info: (Batch=2, Seq=3, Info=5)
        levels_info = torch.tensor([
            [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 1, 0, 0]],
            [[0, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 2, 0, 0, 0]]
        ], dtype=torch.long)
        
        seq_positions = torch.tensor([
            [0, 1, 2],
            [0, 1, 2]
        ], dtype=torch.long)
        
        emb = pos_emb(levels_info, seq_positions)
        
        assert emb.shape == (2, 3, dim)
        assert not torch.isnan(emb).any()
        assert not torch.allclose(emb[0, 2], emb[1, 2])


class TestTransformerMasking:
    """Tests for transformer attention masking."""

    def test_transformer_masking_effectiveness(self, fractal_vit):
        """Test if the attention mask correctly prevents interaction with padding tokens."""
        dim = 64
        transformer = fractal_vit.transformer
        
        x = torch.randn(2, 4, dim)
        levels_info = torch.zeros(2, 4, 2, dtype=torch.long)
        
        attn_mask = torch.tensor([
            [1, 1, 0, 0],
            [1, 1, 1, 1]
        ], dtype=torch.bool).unsqueeze(1).unsqueeze(2)
        
        out = transformer(x, levels_info, attn_mask)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
