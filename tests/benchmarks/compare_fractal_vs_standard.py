#!/usr/bin/env python3
"""
Fractal ViT vs Standard ViT Real-time Benchmark
Performs a fair comparison by running both models under identical configurations.
"""

import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import gc
import torch.optim as optim

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vit_pytorch.fractal_vit import NextGenerationFractalViT

# --- 1. Standard ViT Implementation for Comparison ---
class StandardViT(nn.Module):
    """A standard Vision Transformer implementation for baseline comparison."""
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Use PyTorch's native TransformerEncoderLayer for standard behavior
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim, 
                nhead=heads, 
                dim_feedforward=mlp_dim, 
                dropout=dropout, 
                activation='gelu', 
                batch_first=True,
                norm_first=True
            ),
            num_layers=depth
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.channels = channels

    def forward(self, img):
        b, c, h, w = img.shape
        p_h, p_w = self.patch_height, self.patch_width
        
        # Patch partition: (b, c, h, w) -> (b, n, p_dim)
        x = img.reshape(b, c, h // p_h, p_h, w // p_w, p_w)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(b, -1, p_h * p_w * c)
        
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x[:, 0] # CLS token
        return self.mlp_head(x)

# --- 2. Benchmarking Utilities ---

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_peak_memory(device):
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024 # MB
    return 0

def reset_memory(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    gc.collect()

def benchmark_model(model_name, model_factory, device, batch_size=32, image_size=(256, 256), steps=50, warmup=10):
    print(f"\nBenchmarking {model_name}...")
    
    # Instantiate model
    reset_memory(device)
    try:
        model = model_factory().to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to create model {model_name}: {e}")
        return None

    params = count_parameters(model)
    print(f"  Parameters: {params:,}")

    # Input data
    dummy_input = torch.randn(batch_size, 3, *image_size).to(device)
    
    # --- Inference Benchmark ---
    print("  Running Inference Benchmark...")
    inference_times = []
    
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(dummy_input)
        
        if device.type == 'cuda': torch.cuda.synchronize()
        
        # Measurement
        for _ in range(steps):
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(torch.cuda.current_stream())
                _ = model(dummy_input)
                end_event.record(torch.cuda.current_stream())
                torch.cuda.synchronize()
                inference_times.append(start_event.elapsed_time(end_event)) # ms
            else:
                start = time.perf_counter()
                _ = model(dummy_input)
                end = time.perf_counter()
                inference_times.append((end - start) * 1000) # ms

    avg_inference_ms = np.mean(inference_times)
    inference_fps = batch_size / (avg_inference_ms / 1000)
    peak_mem_inference = get_peak_memory(device)
    
    print(f"  Inference: {avg_inference_ms:.2f} ms/batch | {inference_fps:.2f} img/s")

    # --- Training Benchmark (Forward + Backward) ---
    print("  Running Training Benchmark...")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    labels = torch.randint(0, 10, (batch_size,)).to(device)
    
    train_times = []
    reset_memory(device)
    
    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        outputs = model(dummy_input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    if device.type == 'cuda': torch.cuda.synchronize()

    # Measurement
    for _ in range(steps):
        optimizer.zero_grad()
        
        if device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream())
            
            outputs = model(dummy_input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            end_event.record(torch.cuda.current_stream())
            torch.cuda.synchronize()
            train_times.append(start_event.elapsed_time(end_event))
        else:
            start = time.perf_counter()
            
            outputs = model(dummy_input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            end = time.perf_counter()
            train_times.append((end - start) * 1000)

    avg_train_ms = np.mean(train_times)
    train_fps = batch_size / (avg_train_ms / 1000)
    peak_mem_train = get_peak_memory(device)
    
    print(f"  Training: {avg_train_ms:.2f} ms/batch | {train_fps:.2f} img/s")
    
    return {
        "Model": model_name,
        "Params": params,
        "Inference Latency (ms)": avg_inference_ms,
        "Inference Throughput (img/s)": inference_fps,
        "Training Latency (ms)": avg_train_ms,
        "Training Throughput (img/s)": train_fps,
        "Peak Memory (MB)": peak_mem_train if device.type == 'cuda' else 0
    }

# --- 3. Main Execution ---

def main():
    # Configuration
    IMAGE_SIZE = (32, 32) # Reduced for CPU benchmarking
    BATCH_SIZE = 4
    DIM = 64
    DEPTH = 2
    HEADS = 4
    MLP_DIM = 128
    PATCH_SIZE = (4, 4)
    NUM_CLASSES = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running benchmarks on {device}...")
    print(f"Config: Image={IMAGE_SIZE}, Batch={BATCH_SIZE}, Dim={DIM}, Depth={DEPTH}")

    results = []

    # 1. Standard ViT
    std_factory = lambda: StandardViT(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES,
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        mlp_dim=MLP_DIM
    )
    res_std = benchmark_model("Standard ViT", std_factory, device, BATCH_SIZE, IMAGE_SIZE)
    if res_std: results.append(res_std)

    # 2. Fractal ViT (Next Gen)
    fractal_factory = lambda: NextGenerationFractalViT(
        image_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        mlp_dim=MLP_DIM,
        min_patch_size=PATCH_SIZE,
        max_level=4, # Limit depth for fair comparison
        learnable_split=True # Enable the key feature
    )
    res_fractal = benchmark_model("Fractal ViT", fractal_factory, device, BATCH_SIZE, IMAGE_SIZE)
    if res_fractal: results.append(res_fractal)

    # --- Visualization ---
    if not results:
        print("No results collected.")
        return

    df = pd.DataFrame(results)
    print("\n📊 Final Results:")
    print(df.to_string(index=False))
    
    # Plotting
    metrics = ["Params", "Inference Throughput (img/s)", "Training Throughput (img/s)"]
    if device.type == 'cuda':
        metrics.append("Peak Memory (MB)")
        
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1: axes = [axes]
    
    colors = ['#4ECDC4', '#FF6B6B'] # Standard, Fractal
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(df["Model"], df[metric], color=colors)
        ax.set_title(metric)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.1f}' if height > 10 else f'{height:.2f}',
                    ha='center', va='bottom')

    plt.tight_layout()
    output_path = Path("workspace") / "benchmark_comparison_realtime.png"
    plt.savefig(output_path)
    print(f"\n📈 Chart saved to {output_path}")

if __name__ == "__main__":
    main()
