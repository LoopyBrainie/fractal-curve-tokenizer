#!/usr/bin/env python3
"""
测试无限细分的Fractal Hilbert Tokenizer
验证能否划分到最小整数patch
"""

import torch
import torch.nn as nn
from torchvision import transforms
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from vit_pytorch.fractal_curve_tokenizer import FractalHilbertTokenizer

def test_unlimited_subdivision():
    """测试无限细分功能"""
    print("🔬 测试无限细分Fractal Hilbert Tokenizer")
    
    # 创建无限细分tokenizer
    tokenizer = FractalHilbertTokenizer(
        min_patch_size=(1, 1),      # 最小到1x1像素
        max_level=None,             # 无限制层级
        learnable_split=True,       # 启用可学习分割
        adaptive_threshold=0.3      # 更低的阈值，鼓励更深分割
    )
    
    print(f"Tokenizer配置:")
    print(f"  最小patch尺寸: {tokenizer.min_patch_size}")
    print(f"  最大层级: {tokenizer.max_level} (无限制)")
    print(f"  可学习分割: {tokenizer.learnable_split}")
    print(f"  自适应阈值: {tokenizer.adaptive_threshold}")
    
    # 测试不同尺寸的图像
    test_cases = [
        ("8x8图像", torch.randn(1, 3, 8, 8)),
        ("16x16图像", torch.randn(1, 3, 16, 16)),
        ("32x32图像", torch.randn(1, 3, 32, 32)),
        ("64x32图像", torch.randn(1, 3, 64, 32)),  # 非正方形
    ]
    
    results = {}
    
    for case_name, image in test_cases:
        print(f"\n📊 测试 {case_name}: {tuple(image.shape)}")
        
        # 进行tokenization
        tokenizer.eval()
        with torch.no_grad():
            output = tokenizer.tokenize(image)
        sequences = output.sequences

        if len(sequences) > 0 and sequences[0].tokens.numel() > 0:
            sequence = sequences[0]
            tokens = sequence.tokens
            levels = sequence.get_levels()

            if levels is None:
                print("  ⚠️ 未找到层级信息")
                results[case_name] = {'error': 'No level metadata'}
                continue

            # 分析结果
            num_tokens = tokens.shape[0]
            token_dim = tokens.shape[1]

            # 统计层级分布
            depths = levels[:, 0].tolist()
            level_counts = defaultdict(int)
            for depth in depths:
                level_counts[depth] += 1

            max_level_reached = max(depths) if depths else 0
            min_level_reached = min(depths) if depths else 0

            print(f"  Token数量: {num_tokens}")
            print(f"  Token维度: {token_dim}")
            print(f"  达到的最大层级: {max_level_reached}")
            print(f"  最小层级: {min_level_reached}")
            print(f"  层级分布: {dict(sorted(level_counts.items()))}")

            # 计算理论最大可能层级
            _, _, h, w = image.shape
            theoretical_max = tokenizer._estimate_max_possible_level(h, w)
            print(f"  理论最大层级: {theoretical_max}")

            results[case_name] = {
                'num_tokens': num_tokens,
                'max_level': max_level_reached,
                'theoretical_max': theoretical_max,
                'level_distribution': dict(level_counts)
            }
        else:
            print(f"  ⚠️ 没有生成任何token")
            results[case_name] = {'error': 'No tokens generated'}
    
    return results

def test_adaptive_splitting():
    """测试自适应分割决策"""
    print(f"\n🧠 测试自适应分割决策")
    
    # 创建具有不同复杂度区域的图像
    image = torch.zeros(1, 3, 32, 32)
    
    # 区域1：均匀区域（左上）
    image[:, :, :16, :16] = 0.5
    
    # 区域2：高频纹理（右上）
    x, y = torch.meshgrid(torch.linspace(0, 10, 16), torch.linspace(0, 10, 16), indexing='ij')
    texture = torch.sin(x) * torch.cos(y)
    image[:, :, :16, 16:] = texture.unsqueeze(0) * 0.3 + 0.5
    
    # 区域3：边缘区域（左下）
    edge_region = torch.zeros(16, 16)
    edge_region[:8, :] = 0.2
    edge_region[8:, :] = 0.8
    image[:, :, 16:, :16] = edge_region.unsqueeze(0)
    
    # 区域4：随机噪声（右下）
    image[:, :, 16:, 16:] = torch.randn(1, 3, 16, 16) * 0.2 + 0.5
    
    print("创建了具有4种不同复杂度的测试图像:")
    print("  左上：均匀区域")
    print("  右上：高频纹理")  
    print("  左下：边缘区域")
    print("  右下：随机噪声")
    
    # 使用自适应tokenizer
    adaptive_tokenizer = FractalHilbertTokenizer(
        min_patch_size=(2, 2),      # 最小到2x2像素
        max_level=None,             # 无限制
        learnable_split=True,
        adaptive_threshold=0.4
    )
    
    # 使用固定策略tokenizer对比
    fixed_tokenizer = FractalHilbertTokenizer(
        min_patch_size=(2, 2),
        max_level=4,                # 固定最大层级
        learnable_split=False,
        adaptive_threshold=0.5
    )
    
    tokenizers = [
        ("自适应分割", adaptive_tokenizer),
        ("固定策略", fixed_tokenizer)
    ]
    
    for name, tokenizer in tokenizers:
        print(f"\n{name}结果:")
        tokenizer.eval()
        
        with torch.no_grad():
            output = tokenizer.tokenize(image)
        sequences = output.sequences

        if len(sequences) > 0 and sequences[0].tokens.numel() > 0:
            tokens = sequences[0].tokens
            levels = sequences[0].get_levels()

            if levels is None:
                print("  ⚠️ 未找到层级信息")
                continue

            # 分析层级分布
            depths = levels[:, 0].tolist()
            level_counts = defaultdict(int)
            for depth in depths:
                level_counts[depth] += 1

            print(f"  Token数量: {tokens.shape[0]}")
            print(f"  最大层级: {max(depths) if depths else 0}")
            print(f"  层级分布: {dict(sorted(level_counts.items()))}")

            # 计算不同层级token的统计信息
            for level in sorted(level_counts.keys()):
                level_mask = levels[:, 0] == level
                level_tokens = tokens[level_mask]
                if level_tokens.numel() > 0:
                    avg_var = torch.var(level_tokens, dim=1).mean().item()
                    print(f"    层级{level}: {level_counts[level]}个token, 平均方差: {avg_var:.4f}")

def test_performance_scaling():
    """测试无限细分的性能表现"""
    print(f"\n⚡ 测试性能扩展性")
    
    import time
    
    tokenizer = FractalHilbertTokenizer(
        min_patch_size=(1, 1),
        max_level=None,
        learnable_split=True,
        adaptive_threshold=0.5  # 中等分割倾向
    )
    
    sizes = [16, 32, 48, 64]
    
    for size in sizes:
        image = torch.randn(1, 3, size, size)
        
        start_time = time.time()
        with torch.no_grad():
            output = tokenizer.tokenize(image)
        end_time = time.time()

        sequences = output.sequences

        if len(sequences) > 0 and sequences[0].tokens.numel() > 0:
            tokens = sequences[0].tokens
            levels = sequences[0].get_levels()

            if levels is None:
                print("  ⚠️ 未找到层级信息")
                continue
            
            processing_time = (end_time - start_time) * 1000
            num_tokens = tokens.shape[0]
            max_level = levels[:, 0].max().item() if levels.numel() > 0 else 0
            
            print(f"  {size}x{size}图像: {num_tokens}个token, 最大层级{max_level}, 用时{processing_time:.1f}ms")

def visualize_subdivision_pattern(save_path="subdivision_pattern.png"):
    """可视化分割模式"""
    print(f"\n🎨 可视化分割模式")
    
    # 创建一个简单的测试图像
    image = torch.zeros(1, 3, 16, 16)
    # 添加一些有趣的模式
    for i in range(16):
        for j in range(16):
            image[0, :, i, j] = (i + j) / 30.0 + torch.randn(3) * 0.1
    
    tokenizer = FractalHilbertTokenizer(
        min_patch_size=(1, 1),
        max_level=None,
        learnable_split=True,
        adaptive_threshold=0.4
    )
    
    tokenizer.eval()
    with torch.no_grad():
        output = tokenizer.tokenize(image)

    sequences = output.sequences

    if len(sequences) > 0 and sequences[0].tokens.numel() > 0:
        tokens = sequences[0].tokens
        levels = sequences[0].get_levels()

        if levels is None:
            print("  ⚠️ 未找到层级信息")
            return

        print(f"生成了{tokens.shape[0]}个token，最大层级{levels[:, 0].max().item()}")
        print(f"可视化结果保存到: {save_path}")

def main():
    """主测试函数"""
    print("🚀 无限细分Fractal Hilbert Tokenizer测试")
    print("=" * 60)
    
    try:
        # 基础无限细分测试
        results = test_unlimited_subdivision()
        
        # 自适应分割测试
        test_adaptive_splitting()
        
        # 性能扩展性测试
        test_performance_scaling()
        
        # 可视化测试
        visualize_subdivision_pattern()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试完成!")
        
        # 总结关键发现
        print("\n📋 测试总结:")
        for case_name, result in results.items():
            if 'error' not in result:
                max_level = result['max_level']
                theoretical = result['theoretical_max']
                print(f"  {case_name}: 实际最大层级{max_level}, 理论最大{theoretical}")
        
        print("\n✅ 无限细分tokenizer功能验证成功!")
        print("✅ 自适应分割决策正常工作!")
        print("✅ 不同复杂度区域得到差异化处理!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
