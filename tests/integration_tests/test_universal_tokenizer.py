#!/usr/bin/env python3
"""
测试通用图像分形tokenizer的功能
支持任意尺寸图像的分形分割和Hilbert遍历
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from vit_pytorch.fractal_curve_tokenizer import FractalHilbertTokenizer

def test_various_image_sizes():
    """测试不同尺寸的图像"""
    print("=== 测试不同尺寸图像 ===")
    
    # 测试多种图像尺寸
    test_sizes = [
        (32, 32),    # 方形，2的幂
        (48, 32),    # 长方形
        (31, 29),    # 奇数尺寸
        (64, 48),    # 常见比例
        (100, 75),   # 任意尺寸
        (28, 28),    # MNIST尺寸
        (224, 224),  # ImageNet尺寸
        (227, 227),  # AlexNet尺寸
    ]
    
    tokenizer = FractalHilbertTokenizer(
        min_patch_size=(8, 8), 
        max_level=3,
        learnable_split=False
    )
    
    for h, w in test_sizes:
        print(f"\n测试图像尺寸: {h} x {w}")
        
        # 创建测试图像 (3通道)
        image = torch.randn(1, 3, h, w)
        
        try:
            # 分形分割
            output = tokenizer.tokenize(image)
            sequences = output.sequences

            if len(sequences) > 0 and sequences[0].tokens.shape[0] > 0:
                sequence = sequences[0]
                tokens = sequence.tokens
                levels = sequence.get_levels()

                if levels is None:
                    print("  - 未找到层级信息")
                    continue

                print(f"  - 成功分割为 {tokens.shape[0]} 个tokens")
                print(f"  - Token维度: {tokens[0].shape}")
                print(f"  - 层级分布: {levels[:5].tolist()}")  # 显示前5个层级

                # 检查token尺寸一致性
                token_dims = [token.shape[0] for token in tokens]
                if len(set(token_dims)) == 1:
                    print(f"  ✓ 所有tokens维度一致: {token_dims[0]}")
                else:
                    print(f"  ✗ Token维度不一致: {set(token_dims)}")
            else:
                print("  ✗ 没有生成tokens")
                
        except Exception as e:
            print(f"  ✗ 分割失败: {e}")

def test_adaptive_hilbert():
    """测试自适应Hilbert遍历"""
    print("\n=== 测试自适应Hilbert遍历 ===")
    
    tokenizer = FractalHilbertTokenizer(
        min_patch_size=(4, 4), 
        max_level=2,
        learnable_split=False
    )
    
    # 测试不同层级的Hilbert顺序
    test_cases = [
        (0, 16, 16, "方形顶层"),
        (1, 8, 12, "长方形第1层"),
        (2, 6, 4, "高长方形第2层"),
        (0, 32, 8, "很宽的图像"),
        (1, 8, 32, "很高的图像"),
    ]
    
    for level, h, w, desc in test_cases:
        order = tokenizer.get_enhanced_hilbert_order(level, h, w)
        adaptive_order = tokenizer.adaptive_hilbert_mapping([0, 1, 2, 3], h, w)
        
        print(f"{desc} (Level {level}, {h}x{w}):")
        print(f"  - 基础顺序: {order}")
        print(f"  - 自适应顺序: {adaptive_order}")

def test_extreme_cases():
    """测试极端情况"""
    print("\n=== 测试极端情况 ===")
    
    tokenizer = FractalHilbertTokenizer(
        min_patch_size=(2, 2), 
        max_level=4,
        learnable_split=False
    )
    
    # 极端尺寸测试
    extreme_cases = [
        (1, 1, "1x1像素"),
        (2, 1, "2x1条状"),
        (1, 8, "1x8细条"),
        (3, 3, "3x3小方形"),
        (5, 7, "5x7小长方形"),
    ]
    
    for h, w, desc in extreme_cases:
        print(f"\n测试 {desc} ({h}x{w}):")
        
        try:
            image = torch.randn(1, 3, h, w)
            output = tokenizer.tokenize(image)
            sequences = output.sequences

            if len(sequences) > 0 and sequences[0].tokens.shape[0] > 0:
                tokens = sequences[0].tokens
                print(f"  ✓ 生成 {tokens.shape[0]} 个tokens")
                print(f"  ✓ Token形状: {tokens[0].shape if tokens.shape[0] > 0 else 'None'}")
            else:
                print("  - 无tokens生成")
                
        except Exception as e:
            print(f"  ✗ 错误: {e}")

def test_learnable_split():
    """测试可学习分割决策"""
    print("\n=== 测试可学习分割决策 ===")
    
    tokenizer_learnable = FractalHilbertTokenizer(
        min_patch_size=(8, 8), 
        max_level=3,
        learnable_split=True
    )
    
    tokenizer_fixed = FractalHilbertTokenizer(
        min_patch_size=(8, 8), 
        max_level=3,
        learnable_split=False
    )
    
    # 测试图像
    image = torch.randn(2, 3, 64, 64)  # 批量测试
    
    # 可学习分割
    output_learnable = tokenizer_learnable.tokenize(image)
    
    # 固定分割  
    output_fixed = tokenizer_fixed.tokenize(image)

    print(f"可学习分割:")
    for idx, sequence in enumerate(output_learnable.sequences):
        print(f"  图像{idx}: {sequence.tokens.shape[0]} tokens")
        
    print(f"固定分割:")
    for idx, sequence in enumerate(output_fixed.sequences):
        print(f"  图像{idx}: {sequence.tokens.shape[0]} tokens")

def visualize_tokenization(h=32, w=32):
    """可视化分形分割过程"""
    print(f"\n=== 可视化分形分割 ({h}x{w}) ===")
    
    tokenizer = FractalHilbertTokenizer(
        min_patch_size=(4, 4), 
        max_level=3,
        learnable_split=False
    )
    
    # 创建渐变测试图像便于可视化
    image = torch.zeros(1, 3, h, w)
    y, x = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing='ij')
    image[0, 0] = y
    image[0, 1] = x
    image[0, 2] = (y + x) / 2
    
    output = tokenizer.tokenize(image)
    sequences = output.sequences

    if len(sequences) > 0 and sequences[0].tokens.shape[0] > 0:
        tokens = sequences[0].tokens
        levels = sequences[0].get_levels()

        if levels is None:
            print("未找到层级信息")
            return

        print(f"总共生成 {tokens.shape[0]} 个tokens")

        # 统计每层的token数量
        level_counts = {}
        for level_info in levels:
            depth = level_info[0].item()
            level_counts[depth] = level_counts.get(depth, 0) + 1

        print("各层token分布:")
        for depth in sorted(level_counts.keys()):
            print(f"  层级 {depth}: {level_counts[depth]} 个tokens")

if __name__ == "__main__":
    print("开始测试通用分形图像tokenizer...")
    
    test_various_image_sizes()
    test_adaptive_hilbert()
    test_extreme_cases()
    test_learnable_split()
    visualize_tokenization(48, 32)
    
    print("\n✓ 测试完成！")
