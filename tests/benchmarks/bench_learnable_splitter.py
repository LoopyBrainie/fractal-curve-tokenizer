# -*- coding: utf-8 -*-
"""
LearnableSplitter 性能基准测试

数学形式化
----------
测量以下优化的效果:

P-PERF-1: Hilbert LUT 预计算
    传统: O(N × log G) Python 递归
    优化: O(1) 张量索引
    
P-PERF-2: 全批次并行 BFS
    传统: O(B × D) Python 循环
    优化: O(D) 批量操作
    
P-PERF-3: 向量化嵌入分配
    传统: O(N) Python 循环
    优化: O(1) 高级索引

使用方法:
    cd D:\myProject\fractal-curve-tokenizer
    $env:PYTHONPATH="src"
    uv run python tests/benchmarks/bench_learnable_splitter.py
"""

import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def benchmark_splitter(
    batch_sizes: List[int] = [1, 4, 16, 32, 64, 128],
    image_size: int = 64,
    feature_dim: int = 256,
    max_depth: int = 4,
    num_warmup: int = 3,
    num_runs: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Dict[int, float]]:
    """
    基准测试 LearnableSplitter 性能。
    
    Returns:
        结果字典: {
            'splitter_forward': {batch_size: time_ms},
            'hilbert_sort': {batch_size: time_ms},
            'total': {batch_size: time_ms},
        }
    """
    from vit_pytorch.split_adaptive import LearnableSplitter
    
    print(f"设备: {device}")
    print(f"图像尺寸: {image_size}×{image_size}")
    print(f"特征维度: {feature_dim}")
    print(f"最大深度: {max_depth}")
    print(f"预热次数: {num_warmup}, 测量次数: {num_runs}")
    print("-" * 60)
    
    # 创建 splitter
    splitter = LearnableSplitter(
        feature_dim=feature_dim,
        max_depth=max_depth,
        hidden_dim=128,
        pool_size=4,
        temperature=1.0,
        use_gumbel=False,  # 推理模式
    ).to(device)
    splitter.eval()
    
    results = {
        'splitter_forward_fast': {},
        'splitter_forward_slow': {},
        'speedup': {},
    }
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # 创建随机特征图
        # 假设 patch_size=4，所以特征图尺寸 = image_size / 4
        feat_size = image_size // 4
        features = torch.randn(
            batch_size, feature_dim, feat_size, feat_size,
            device=device, dtype=torch.float32
        )
        
        # 预热 (快速路径)
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = splitter(features, (image_size, image_size), hard=True, use_fast_path=True)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        # 测量快速路径
        times_fast = []
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            with torch.no_grad():
                _ = splitter(features, (image_size, image_size), hard=True, use_fast_path=True)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            times_fast.append((time.perf_counter() - start) * 1000)  # ms
        
        avg_fast = sum(times_fast) / len(times_fast)
        results['splitter_forward_fast'][batch_size] = avg_fast
        print(f"  快速路径: {avg_fast:.2f} ms")
        
        # 预热 (慢速路径)
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = splitter(features, (image_size, image_size), hard=True, use_fast_path=False)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        # 测量慢速路径
        times_slow = []
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            with torch.no_grad():
                _ = splitter(features, (image_size, image_size), hard=True, use_fast_path=False)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            times_slow.append((time.perf_counter() - start) * 1000)  # ms
        
        avg_slow = sum(times_slow) / len(times_slow)
        results['splitter_forward_slow'][batch_size] = avg_slow
        print(f"  慢速路径: {avg_slow:.2f} ms")
        
        speedup = avg_slow / avg_fast if avg_fast > 0 else 0
        results['speedup'][batch_size] = speedup
        print(f"  加速比: {speedup:.2f}x")
    
    return results


def benchmark_tokenizer(
    batch_sizes: List[int] = [1, 4, 16, 32, 64],
    image_size: int = 64,
    d_model: int = 384,
    max_depth: int = 4,
    num_warmup: int = 3,
    num_runs: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Dict[int, float]]:
    """
    基准测试完整的 StreamingFractalTokenizerV3。
    """
    from vit_pytorch.tokenizer_streaming import StreamingFractalTokenizerV3
    
    print(f"\n{'='*60}")
    print("完整 Tokenizer 基准测试")
    print(f"{'='*60}")
    print(f"设备: {device}")
    print(f"图像尺寸: {image_size}×{image_size}")
    print(f"模型维度: {d_model}")
    print(f"最大深度: {max_depth}")
    print("-" * 60)
    
    tokenizer = StreamingFractalTokenizerV3(
        image_size=image_size,
        channels=3,
        d_model=d_model,
        base_patch_size=4,
        max_depth=max_depth,
    ).to(device)
    tokenizer.eval()
    
    results = {'tokenize': {}}
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # 创建随机图像
        images = torch.randn(
            batch_size, 3, image_size, image_size,
            device=device, dtype=torch.float32
        )
        
        # 预热
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = tokenizer(images)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        # 测量
        times = []
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            with torch.no_grad():
                _ = tokenizer(images)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            times.append((time.perf_counter() - start) * 1000)  # ms
        
        avg_time = sum(times) / len(times)
        results['tokenize'][batch_size] = avg_time
        print(f"  Tokenize: {avg_time:.2f} ms ({avg_time/batch_size:.2f} ms/image)")
    
    return results


def benchmark_hilbert_lut(
    grid_sizes: List[int] = [16, 32, 64, 128, 256],
    num_points: int = 1000,
    num_runs: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Dict[int, float]]:
    """
    基准测试 Hilbert LUT vs Python 递归。
    """
    from vit_pytorch.split_adaptive import HilbertLUT, HilbertCurve
    
    print(f"\n{'='*60}")
    print("Hilbert LUT 基准测试")
    print(f"{'='*60}")
    print(f"设备: {device}")
    print(f"查询点数: {num_points}")
    print("-" * 60)
    
    results = {
        'lut_lookup': {},
        'python_recursive': {},
        'speedup': {},
    }
    
    for grid_size in grid_sizes:
        print(f"\n网格大小: {grid_size}×{grid_size}")
        
        # 生成随机坐标
        cx = torch.rand(num_points, device=device) * grid_size
        cy = torch.rand(num_points, device=device) * grid_size
        
        # 预热 LUT
        _ = HilbertLUT.batch_lookup(cx, cy, grid_size, device)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        # 测量 LUT
        times_lut = []
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = HilbertLUT.batch_lookup(cx, cy, grid_size, device)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            times_lut.append((time.perf_counter() - start) * 1000)  # ms
        
        avg_lut = sum(times_lut) / len(times_lut)
        results['lut_lookup'][grid_size] = avg_lut
        print(f"  LUT 查询: {avg_lut:.4f} ms")
        
        # 测量 Python 递归
        cx_cpu = (cx.cpu().numpy() * grid_size / grid_size).astype(int)
        cy_cpu = (cy.cpu().numpy() * grid_size / grid_size).astype(int)
        
        times_python = []
        for _ in range(min(num_runs, 10)):  # Python 版本较慢，减少次数
            start = time.perf_counter()
            for i in range(num_points):
                _ = HilbertCurve.xy_to_d(grid_size, int(cx_cpu[i]), int(cy_cpu[i]))
            times_python.append((time.perf_counter() - start) * 1000)
        
        avg_python = sum(times_python) / len(times_python)
        results['python_recursive'][grid_size] = avg_python
        print(f"  Python 递归: {avg_python:.4f} ms")
        
        speedup = avg_python / avg_lut if avg_lut > 0 else 0
        results['speedup'][grid_size] = speedup
        print(f"  加速比: {speedup:.1f}x")
    
    return results


def main():
    """运行所有基准测试。"""
    print("=" * 60)
    print("LearnableSplitter 性能基准测试")
    print("=" * 60)
    
    # 检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
    
    # 1. Hilbert LUT 基准测试
    hilbert_results = benchmark_hilbert_lut(device=device)
    
    # 2. Splitter 基准测试
    splitter_results = benchmark_splitter(device=device)
    
    # 3. 完整 Tokenizer 基准测试
    tokenizer_results = benchmark_tokenizer(device=device)
    
    # 汇总
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    
    print("\nHilbert LUT 加速比:")
    for grid_size, speedup in hilbert_results['speedup'].items():
        print(f"  {grid_size}×{grid_size}: {speedup:.1f}x")
    
    print("\nSplitter 加速比 (快速路径 vs 慢速路径):")
    for batch_size, speedup in splitter_results['speedup'].items():
        print(f"  Batch {batch_size}: {speedup:.2f}x")
    
    print("\nTokenizer 吞吐量:")
    for batch_size, time_ms in tokenizer_results['tokenize'].items():
        throughput = batch_size / (time_ms / 1000)  # images/s
        print(f"  Batch {batch_size}: {throughput:.1f} images/s")


if __name__ == "__main__":
    main()
