#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P9 训练性能基准测试

目的:
    验证 P9-1~P9-6 优化后的训练速度提升
    目标: 从 24s/iter 降至 < 5s/iter

数学形式化:
    T_iter = T_forward + T_backward + T_optimizer
    
    P9 优化前:
        T_forward ≈ 12s (Splitter: O(D×B) Python loops + GPU kernels)
        T_backward ≈ 10s (梯度计算)
        T_optimizer ≈ 2s (参数更新)
    
    P9 优化后:
        T_forward ≈ 2s (完全向量化 BFS)
        T_backward ≈ 2s (简化计算图)
        T_optimizer ≈ 1s (参数更新)

运行方法:
    uv run python tests/benchmarks/bench_p9_training.py --device cuda

Author: GitHub Copilot
Date: 2025-12-28
"""

import argparse
import gc
import sys
import time
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# 添加 src 目录
sys.path.insert(0, 'src')


def get_device(device_str: str) -> torch.device:
    """获取设备."""
    if device_str == 'cuda':
        if not torch.cuda.is_available():
            print("CUDA 不可用，回退到 CPU")
            return torch.device('cpu')
        return torch.device('cuda')
    return torch.device('cpu')


def create_model(device: torch.device, batch_size: int = 8) -> nn.Module:
    """创建 FractalCurveViT 模型."""
    from vit_pytorch import FractalCurveViT
    
    model = FractalCurveViT(
        image_size=224,
        num_classes=200,
        dim=384,
        depth=6,
        heads=6,
        mlp_dim=1536,
        pool='cls',
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        min_patch_size=(4, 4),
        max_level=4,
        ffn_type='swiglu_level',
        tokenizer_type='streaming_v3',
        lca_temperature=1.5,
        learnable_temperature=True,
    ).to(device)
    
    return model


def create_inputs(device: torch.device, batch_size: int = 8) -> torch.Tensor:
    """创建测试输入."""
    return torch.randn(batch_size, 3, 224, 224, device=device)


def warmup_gpu(model: nn.Module, device: torch.device, batch_size: int = 4):
    """GPU 预热."""
    print("GPU 预热中...")
    
    with torch.no_grad():
        for _ in range(5):
            x = create_inputs(device, batch_size)
            _ = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print("预热完成")


def benchmark_forward_only(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 8,
    n_iters: int = 20,
) -> Dict[str, float]:
    """仅前向传播基准测试."""
    model.eval()
    
    times = []
    
    with torch.no_grad():
        for i in range(n_iters):
            x = create_inputs(device, batch_size)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(x)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
            if (i + 1) % 5 == 0:
                print(f"  前向 {i+1}/{n_iters}: {elapsed:.4f}s")
    
    return {
        'mode': 'forward_only',
        'batch_size': batch_size,
        'n_iters': n_iters,
        'mean_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'total_time': sum(times),
        'throughput_img_per_sec': batch_size * n_iters / sum(times),
    }


def benchmark_training_step(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 8,
    n_iters: int = 20,
    use_amp: bool = True,
) -> Dict[str, float]:
    """完整训练步骤基准测试 (forward + backward + optimizer)."""
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler(enabled=use_amp and device.type == 'cuda')
    
    times = []
    forward_times = []
    backward_times = []
    optimizer_times = []
    
    for i in range(n_iters):
        x = create_inputs(device, batch_size)
        labels = torch.randint(0, 200, (batch_size,), device=device)
        
        optimizer.zero_grad(set_to_none=True)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Forward
        start_total = time.perf_counter()
        start_forward = time.perf_counter()
        
        with autocast(enabled=use_amp and device.type == 'cuda'):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, labels)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        forward_time = time.perf_counter() - start_forward
        
        # Backward
        start_backward = time.perf_counter()
        scaler.scale(loss).backward()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        backward_time = time.perf_counter() - start_backward
        
        # Optimizer step
        start_optimizer = time.perf_counter()
        scaler.step(optimizer)
        scaler.update()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        optimizer_time = time.perf_counter() - start_optimizer
        
        total_time = time.perf_counter() - start_total
        
        times.append(total_time)
        forward_times.append(forward_time)
        backward_times.append(backward_time)
        optimizer_times.append(optimizer_time)
        
        if (i + 1) % 5 == 0:
            print(f"  训练 {i+1}/{n_iters}: total={total_time:.3f}s "
                  f"(fwd={forward_time:.3f}s, bwd={backward_time:.3f}s, opt={optimizer_time:.3f}s)")
    
    return {
        'mode': 'training_step',
        'batch_size': batch_size,
        'n_iters': n_iters,
        'use_amp': use_amp,
        'mean_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'mean_forward_time': sum(forward_times) / len(forward_times),
        'mean_backward_time': sum(backward_times) / len(backward_times),
        'mean_optimizer_time': sum(optimizer_times) / len(optimizer_times),
        'total_time': sum(times),
        'throughput_img_per_sec': batch_size * n_iters / sum(times),
    }


def benchmark_tokenizer_only(
    device: torch.device,
    batch_size: int = 8,
    n_iters: int = 50,
) -> Dict[str, float]:
    """仅 Tokenizer 基准测试 (隔离分割器性能)."""
    from vit_pytorch.tokenizer_streaming import StreamingFractalTokenizerV3
    
    tokenizer = StreamingFractalTokenizerV3(
        image_size=224,
        channels=3,
        d_model=384,
        base_patch_size=4,
        max_depth=4,
        use_hilbert_order=True,
        gamma=0.85,
        learnable_temperature=1.0,
        use_gumbel=True,
    ).to(device)
    
    tokenizer.eval()
    
    # 预热
    with torch.no_grad():
        for _ in range(5):
            x = create_inputs(device, batch_size)
            _ = tokenizer(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 基准测试
    times = []
    token_counts = []
    
    with torch.no_grad():
        for i in range(n_iters):
            x = create_inputs(device, batch_size)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            output = tokenizer(x)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
            # 统计 token 数量
            for seq in output:
                token_counts.append(seq.tokens.shape[0])
            
            if (i + 1) % 10 == 0:
                print(f"  Tokenizer {i+1}/{n_iters}: {elapsed*1000:.2f}ms")
    
    return {
        'mode': 'tokenizer_only',
        'batch_size': batch_size,
        'n_iters': n_iters,
        'mean_time_ms': sum(times) / len(times) * 1000,
        'min_time_ms': min(times) * 1000,
        'max_time_ms': max(times) * 1000,
        'mean_tokens_per_image': sum(token_counts) / len(token_counts),
        'total_time': sum(times),
    }


def benchmark_splitter_only(
    device: torch.device,
    batch_size: int = 8,
    n_iters: int = 50,
) -> Dict[str, float]:
    """仅 LearnableSplitter 基准测试 (隔离 P9-1 优化效果)."""
    from vit_pytorch.split_adaptive import LearnableSplitter
    
    splitter = LearnableSplitter(
        feature_dim=384,
        max_depth=4,
        hidden_dim=128,
        intermediate_dim=64,
        pool_size=4,
        temperature=1.0,
        use_gumbel=True,
        min_region_size=7,
    ).to(device)
    
    splitter.eval()
    
    # 模拟特征图
    H_feat, W_feat = 56, 56
    features = torch.randn(batch_size, 384, H_feat, W_feat, device=device)
    image_size = (224, 224)
    
    # 预热
    with torch.no_grad():
        for _ in range(5):
            _ = splitter(features, image_size, hard=True)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 基准测试
    times = []
    token_counts = []
    
    with torch.no_grad():
        for i in range(n_iters):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            result = splitter(features, image_size, hard=True)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            token_counts.append(result.num_tokens)
            
            if (i + 1) % 10 == 0:
                print(f"  Splitter {i+1}/{n_iters}: {elapsed*1000:.2f}ms, tokens={result.num_tokens}")
    
    return {
        'mode': 'splitter_only',
        'batch_size': batch_size,
        'n_iters': n_iters,
        'mean_time_ms': sum(times) / len(times) * 1000,
        'min_time_ms': min(times) * 1000,
        'max_time_ms': max(times) * 1000,
        'mean_tokens': sum(token_counts) / len(token_counts),
        'total_time': sum(times),
    }


def print_results(results: List[Dict[str, Any]]):
    """打印基准测试结果."""
    print("\n" + "=" * 70)
    print("📊 P9 训练性能基准测试结果")
    print("=" * 70)
    
    for r in results:
        mode = r['mode']
        print(f"\n### {mode.upper().replace('_', ' ')}")
        print("-" * 40)
        
        if mode == 'training_step':
            print(f"  Batch size:     {r['batch_size']}")
            print(f"  AMP 启用:       {r['use_amp']}")
            print(f"  迭代次数:       {r['n_iters']}")
            print(f"  平均时间:       {r['mean_time']:.3f}s/iter")
            print(f"  最小时间:       {r['min_time']:.3f}s/iter")
            print(f"  最大时间:       {r['max_time']:.3f}s/iter")
            print(f"  - 前向传播:     {r['mean_forward_time']:.3f}s ({r['mean_forward_time']/r['mean_time']*100:.1f}%)")
            print(f"  - 反向传播:     {r['mean_backward_time']:.3f}s ({r['mean_backward_time']/r['mean_time']*100:.1f}%)")
            print(f"  - 优化器:       {r['mean_optimizer_time']:.3f}s ({r['mean_optimizer_time']/r['mean_time']*100:.1f}%)")
            print(f"  吞吐量:         {r['throughput_img_per_sec']:.1f} img/sec")
            
            # P9 目标对比
            target_time = 5.0
            baseline_time = 24.0
            current_time = r['mean_time']
            
            if current_time < target_time:
                print(f"\n  ✅ 目标达成: {current_time:.2f}s < {target_time}s")
                speedup = baseline_time / current_time
                print(f"  ✅ 相比基线 ({baseline_time}s) 加速: {speedup:.1f}x")
            else:
                improvement = (baseline_time - current_time) / baseline_time * 100
                print(f"\n  ⚠️ 当前: {current_time:.2f}s, 目标: {target_time}s")
                print(f"  ⚠️ 相比基线 ({baseline_time}s) 改善: {improvement:.1f}%")
                
        elif mode == 'forward_only':
            print(f"  Batch size:     {r['batch_size']}")
            print(f"  迭代次数:       {r['n_iters']}")
            print(f"  平均时间:       {r['mean_time']*1000:.1f}ms/batch")
            print(f"  吞吐量:         {r['throughput_img_per_sec']:.1f} img/sec")
            
        elif mode in ('tokenizer_only', 'splitter_only'):
            print(f"  Batch size:     {r['batch_size']}")
            print(f"  迭代次数:       {r['n_iters']}")
            print(f"  平均时间:       {r['mean_time_ms']:.2f}ms")
            print(f"  最小时间:       {r['min_time_ms']:.2f}ms")
            print(f"  最大时间:       {r['max_time_ms']:.2f}ms")
            if 'mean_tokens' in r:
                print(f"  平均 tokens:    {r['mean_tokens']:.1f}")
            if 'mean_tokens_per_image' in r:
                print(f"  平均 tokens/img: {r['mean_tokens_per_image']:.1f}")
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='P9 训练性能基准测试')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='设备类型')
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小')
    parser.add_argument('--n-iters', type=int, default=20, help='迭代次数')
    parser.add_argument('--no-amp', action='store_true', help='禁用混合精度')
    parser.add_argument('--skip-training', action='store_true', help='跳过训练测试')
    parser.add_argument('--only-splitter', action='store_true', help='仅测试 Splitter')
    
    args = parser.parse_args()
    
    device = get_device(args.device)
    print(f"\n设备: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"迭代次数: {args.n_iters}")
    print(f"AMP: {'禁用' if args.no_amp else '启用'}")
    
    results = []
    
    # Splitter 基准测试
    print(f"\n{'='*60}")
    print("1. LearnableSplitter 基准测试 (P9-1 向量化 BFS)")
    print("="*60)
    r = benchmark_splitter_only(device, args.batch_size, n_iters=50)
    results.append(r)
    
    if args.only_splitter:
        print_results(results)
        return
    
    # Tokenizer 基准测试
    print(f"\n{'='*60}")
    print("2. StreamingFractalTokenizerV3 基准测试")
    print("="*60)
    r = benchmark_tokenizer_only(device, args.batch_size, n_iters=50)
    results.append(r)
    
    # 创建完整模型
    print(f"\n{'='*60}")
    print("3. FractalCurveViT 完整模型")
    print("="*60)
    model = create_model(device, args.batch_size)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    warmup_gpu(model, device, args.batch_size)
    
    # 前向传播基准测试
    print(f"\n{'='*60}")
    print("4. 前向传播基准测试 (inference)")
    print("="*60)
    r = benchmark_forward_only(model, device, args.batch_size, args.n_iters)
    results.append(r)
    
    # 训练步骤基准测试
    if not args.skip_training:
        print(f"\n{'='*60}")
        print("5. 完整训练步骤基准测试 (forward + backward + optimizer)")
        print("="*60)
        r = benchmark_training_step(
            model, device, args.batch_size, args.n_iters,
            use_amp=not args.no_amp
        )
        results.append(r)
    
    print_results(results)


if __name__ == '__main__':
    main()
