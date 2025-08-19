#!/usr/bin/env python3
"""
测试改进的Hilbert曲线递归算法
"""

import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from vit_pytorch.fractal_curve_tokenizer import FractalHilbertTokenizer

def test_hilbert_curve_generation():
    """测试Hilbert曲线生成算法"""
    print("🔧 测试Hilbert曲线生成算法...")
    
    tokenizer = FractalHilbertTokenizer(min_patch_size=(4, 4), max_level=3)
    
    # 测试不同阶数的Hilbert曲线
    for n in range(1, 4):
        print(f"\n📐 测试 {n} 阶Hilbert曲线:")
        curve_points = tokenizer.generate_true_hilbert_curve(n)
        print(f"   生成 {len(curve_points)} 个点")
        print(f"   前10个点: {curve_points[:10]}")
        
        # 验证曲线连续性
        if len(curve_points) > 1:
            max_distance = 0
            for i in range(1, len(curve_points)):
                x1, y1 = curve_points[i-1]
                x2, y2 = curve_points[i]
                distance = abs(x2-x1) + abs(y2-y1)  # 曼哈顿距离
                max_distance = max(max_distance, distance)
            print(f"   最大跳跃距离: {max_distance} (应该 <= 1)")

def test_hilbert_distance():
    """测试Hilbert距离计算"""
    print("\n🧮 测试Hilbert距离计算...")
    
    tokenizer = FractalHilbertTokenizer()
    
    # 测试2x2网格的Hilbert距离
    coords = [(0,0), (0,1), (1,0), (1,1)]
    distances = []
    
    for x, y in coords:
        dist = tokenizer.hilbert_distance(x, y, 2)
        distances.append((x, y, dist))
        print(f"   坐标 ({x},{y}) -> Hilbert距离: {dist}")
    
    # 按距离排序，应该符合Hilbert曲线顺序
    distances.sort(key=lambda x: x[2])
    hilbert_order = [coord[:2] for coord in distances]
    print(f"   Hilbert顺序: {hilbert_order}")

def test_adaptive_hilbert_mapping():
    """测试自适应Hilbert映射"""
    print("\n🎯 测试自适应Hilbert映射...")
    
    tokenizer = FractalHilbertTokenizer()
    
    # 测试不同尺寸的映射
    test_sizes = [(2,2), (3,2), (2,3), (4,4), (5,3)]
    
    for h, w in test_sizes:
        patch_indices = [0, 1, 2, 3]  # [左上, 右上, 左下, 右下]
        mapped_order = tokenizer.adaptive_hilbert_mapping(patch_indices, h, w)
        print(f"   尺寸 {h}x{w}: {patch_indices} -> {mapped_order}")

def test_enhanced_hilbert_order():
    """测试增强的Hilbert顺序生成"""
    print("\n⚡ 测试增强Hilbert顺序生成...")
    
    tokenizer = FractalHilbertTokenizer(max_level=4)
    
    # 测试不同层级和尺寸
    test_cases = [
        (0, 32, 32),   # 第0层，正方形
        (1, 16, 32),   # 第1层，宽矩形
        (1, 32, 16),   # 第1层，高矩形
        (2, 8, 8),     # 第2层，正方形
        (3, 4, 12),    # 第3层，很宽的矩形
    ]
    
    for level, h, w in test_cases:
        order = tokenizer.get_enhanced_hilbert_order(level, h, w)
        aspect_ratio = w / h
        print(f"   层级{level}, {h}x{w} (比例{aspect_ratio:.2f}): {order}")

def visualize_hilbert_curves():
    """可视化Hilbert曲线"""
    print("\n📊 生成Hilbert曲线可视化...")
    
    tokenizer = FractalHilbertTokenizer()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, n in enumerate([1, 2, 3]):
        ax = axes[i]
        curve_points = tokenizer.generate_true_hilbert_curve(n)
        
        if len(curve_points) > 1:
            # 绘制曲线路径
            x_coords = [p[0] for p in curve_points]
            y_coords = [p[1] for p in curve_points]
            
            ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7, label='Hilbert路径')
            ax.scatter(x_coords, y_coords, c=range(len(x_coords)), 
                      cmap='viridis', s=50, zorder=5)
            
            # 标记起始点
            ax.scatter(x_coords[0], y_coords[0], c='red', s=100, 
                      marker='o', label='起始点', zorder=10)
            ax.scatter(x_coords[-1], y_coords[-1], c='orange', s=100, 
                      marker='s', label='结束点', zorder=10)
            
            ax.set_title(f'{n}阶 Hilbert曲线 ({2**n}x{2**n})')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('hilbert_curves_visualization.png', dpi=300, bbox_inches='tight')
    print("   可视化已保存为: hilbert_curves_visualization.png")

def benchmark_hilbert_algorithms():
    """基准测试Hilbert算法性能"""
    print("\n⏱️  基准测试Hilbert算法性能...")
    
    import time
    tokenizer = FractalHilbertTokenizer(max_level=5)
    
    # 测试不同方法的性能
    test_cases = [(h, w) for h in [8, 16, 32, 64] for w in [8, 16, 32, 64]]
    
    methods = [
        ('递归算法', lambda h, w: tokenizer.get_hilbert_order_for_level(0, h, w)),
        ('自适应映射', lambda h, w: tokenizer.adaptive_hilbert_mapping([0,1,2,3], h, w)),
        ('增强算法', lambda h, w: tokenizer.get_enhanced_hilbert_order(0, h, w))
    ]
    
    for method_name, method_func in methods:
        start_time = time.time()
        
        for h, w in test_cases:
            result = method_func(h, w)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / len(test_cases) * 1000  # ms
        print(f"   {method_name}: 平均 {avg_time:.3f} ms/调用")

def test_fractal_tokenizer_integration():
    """测试改进的Hilbert算法与分形tokenizer的集成"""
    print("\n🧩 测试与FractalTokenizer的集成...")
    
    tokenizer = FractalHilbertTokenizer(
        min_patch_size=(8, 8), 
        max_level=3, 
        learnable_split=False
    )
    
    # 创建测试图像
    test_image = torch.randn(1, 3, 64, 64)
    
    try:
        # 进行分形tokenization
        tokens_list, levels_list = tokenizer(test_image)
        
        if len(tokens_list) > 0 and tokens_list[0].numel() > 0:
            tokens = tokens_list[0]
            levels = levels_list[0]
            
            print(f"   ✅ 生成 {tokens.shape[0]} 个tokens")
            print(f"   ✅ Token维度: {tokens.shape}")
            print(f"   ✅ Levels信息: {levels.shape}")
            print(f"   ✅ 层级分布: {levels[:, 0].tolist()[:10]}...")  # 显示前10个的层级
            
            # 分析层级分布
            level_counts = {}
            for level in levels[:, 0].tolist():
                level_counts[level] = level_counts.get(level, 0) + 1
            print(f"   📊 层级统计: {level_counts}")
            
        else:
            print("   ❌ 没有生成tokens")
            
    except Exception as e:
        print(f"   ❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    print("🚀 开始测试改进的Hilbert曲线递归算法\n")
    
    test_hilbert_curve_generation()
    test_hilbert_distance()
    test_adaptive_hilbert_mapping()
    test_enhanced_hilbert_order()
    
    try:
        visualize_hilbert_curves()
    except Exception as e:
        print(f"   可视化跳过: {e}")
    
    benchmark_hilbert_algorithms()
    test_fractal_tokenizer_integration()
    
    print("\n✅ 所有测试完成!")

if __name__ == "__main__":
    main()
