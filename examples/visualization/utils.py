import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.vit_pytorch.model_fractal_vit import FractalCurveViT
from src.vit_pytorch.tokenizer_streaming import StreamingFractalTokenizerV3

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_synthetic_image(size=256):
    """
    Creates a synthetic image with varying complexity to demonstrate adaptive splitting.
    Contains:
    - Smooth gradients (low complexity)
    - Sharp edges (high complexity)
    - High frequency texture (high complexity)
    """
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    # Background: Smooth gradient
    img = 0.5 * torch.sin(3 * xx) * torch.cos(3 * yy)
    
    # Feature 1: Sharp Circle
    radius = torch.sqrt(xx**2 + yy**2)
    circle = (radius < 0.5).float()
    img = img * (1 - circle) + circle # Sharp edge
    
    # Feature 2: High frequency noise in a corner
    noise = torch.randn(size, size) * 0.2
    mask = (xx > 0.5) & (yy > 0.5)
    img[mask] += noise[mask]
    
    # Normalize to [0, 1]
    img = (img - img.min()) / (img.max() - img.min())
    
    # Make it 3 channels (B, C, H, W)
    img = img.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    return img

def load_model(device):
    """Loads a FractalCurveViT model with default settings."""
    model = FractalCurveViT(
        image_size=256,
        num_classes=1000,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
        tokenizer_type='streaming_v3'
    ).to(device)
    model.eval()
    return model

def plot_quadtree(ax, tokens, levels, image_size=256):
    """
    Plots the quadtree decomposition based on tokens and their levels.
    Note: This requires the tokenizer to return spatial info or we need to reconstruct it.
    Since the tokenizer returns a sequence, we might need to access the internal 'regions' if available,
    or infer position from the Hilbert index if the sequence is strictly Hilbert ordered.
    
    However, the StreamingFractalTokenizerV3 returns (tokens, levels).
    We might need to hook into the tokenizer to get the actual bounding boxes.
    """
    pass # Implemented in specific scripts
