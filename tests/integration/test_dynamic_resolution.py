# -*- coding: utf-8 -*-
"""
Dynamic Resolution Tests (I99-9 Supplement)

Verification: model accepts variable input sizes (image_size=None).

Mathematical Formalization
==========================
For variable image size I ∈ R^{B × C × H × W} where H, W are dynamic:
    f(I; θ) → ŷ  ∀ (H, W) ∈ N⁺²

This verifies the model maintains:
1. Input dimension compatibility: B × C × H × W → B × N(H,W) × D
2. Output consistency: B × num_classes for all (H, W)
3. No hidden assumptions about fixed input size
"""

import pytest
import torch
from vit_pytorch import FractalCurveViT


class TestDynamicResolution:
    """Dynamic resolution tests."""

    def test_none_image_size_model_creation(self):
        """Verify model can be created with image_size=None."""
        model = FractalCurveViT(
            image_size=None,  # Dynamic resolution
            num_classes=1000,
            dim=256,
            num_layers=4,
            heads=4,
        )

        # Verify model has expected attributes
        assert hasattr(model, 'tokenizer')
        assert hasattr(model, 'transformer')

    def test_varying_image_sizes(self):
        """Verify model accepts different image sizes."""
        model = FractalCurveViT(
            image_size=None,
            num_classes=100,
            dim=128,
            num_layers=2,
            heads=2,
        )

        # Test various image sizes
        test_sizes = [(64, 64), (128, 128), (224, 224), (256, 256)]

        for H, W in test_sizes:
            images = torch.randn(2, 3, H, W)
            with torch.no_grad():
                result = model(images)
            logits = result.logits if hasattr(result, 'logits') else result
            assert logits.shape == (2, 100), f"Failed for size {H}x{W}"
            assert not torch.isnan(logits).any(), f"NaN detected for size {H}x{W}"

    def test_varying_batch_sizes(self):
        """Verify model handles different batch sizes correctly."""
        model = FractalCurveViT(
            image_size=None,
            num_classes=50,
            dim=128,
            num_layers=2,
            heads=2,
        )

        for B in [1, 4, 8, 16]:
            images = torch.randn(B, 3, 128, 128)  # Create fresh tensor for each batch size
            with torch.no_grad():
                result = model(images)
            logits = result.logits if hasattr(result, 'logits') else result
            assert logits.shape == (B, 50), f"Failed for batch size {B}"

    def test_aspect_ratios(self):
        """Verify model handles various aspect ratios."""
        model = FractalCurveViT(
            image_size=None,
            num_classes=10,
            dim=64,
            num_layers=2,
            heads=2,
        )

        # Test non-square aspect ratios
        aspect_ratios = [
            (64, 128),   # Portrait
            (128, 64),   # Landscape
            (100, 200),  # 1:2
            (200, 100),  # 2:1
        ]

        for H, W in aspect_ratios:
            images = torch.randn(2, 3, H, W)
            with torch.no_grad():
                result = model(images)
            logits = result.logits if hasattr(result, 'logits') else result
            assert logits.shape == (2, 10)

    def test_gradient_flow_with_dynamic_size(self):
        """Verify gradients flow correctly with dynamic input sizes."""
        model = FractalCurveViT(
            image_size=None,
            num_classes=10,
            dim=64,
            num_layers=2,
            heads=2,
        )

        # Enable training mode
        model.train()

        images = torch.randn(4, 3, 128, 128, requires_grad=True)

        result = model(images)
        logits = result.logits if hasattr(result, 'logits') else result
        loss = logits.sum()
        loss.backward()

        # Verify gradients exist and are non-zero
        assert images.grad is not None
        assert images.grad.numel() > 0
        assert not torch.allclose(images.grad, torch.zeros_like(images.grad))


class TestTokenizerDynamicResolution:
    """Tokenizer dynamic resolution tests."""

    def test_tokenizer_dynamic_size(self):
        """Verify tokenizer handles dynamic input sizes."""
        from vit_pytorch import StreamingFractalTokenizerV3
        from vit_pytorch.layers.splitters.gumbel_topk import GumbelTopKSplitter

        # Note: tokenizer requires image_size at init, but supports variable input
        tokenizer = StreamingFractalTokenizerV3(
            image_size=(64, 64),  # Base size for initialization
            base_patch_size=4,
            max_level=8,
        )

        splitter = GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=8,
        )

        # Test with different sizes - tokenizer should adapt to input
        for size in [(64, 64), (128, 128)]:
            images = torch.randn(1, 3, *size)

            features = tokenizer.shared_conv(images)
            split_result = splitter(features, images.shape[2:], hard=True)
            output = tokenizer.tokenize(images, split_result)

            assert output.tokens.shape[0] == 1  # Batch size preserved
            assert output.tokens.shape[2] == 256  # Feature dim preserved


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
