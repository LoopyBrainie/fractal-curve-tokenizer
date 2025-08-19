#!/usr/bin/env python3
"""验证Hilbert曲线实现的正确性"""

import numpy as np
from vit_pytorch.fractal_curve_tokenizer import FractalHilbertTokenizer

def visualize_hilbert_order(h, w, order):
    """可视化Hilbert遍历顺序"""
    grid = np.full((h, w), -1)
    for visit_idx, linear_idx in enumerate(order):
        r, c = divmod(linear_idx, w)
        if 0 <= r < h and 0 <= c < w:
            grid[r, c] = visit_idx
    
    print(f"\n{h}x{w} Hilbert遍历顺序可视化:")
    print("遍历顺序矩阵 (数字表示访问顺序):")
    for row in grid:
        print("  [" + ", ".join(f"{x:2d}" if x >= 0 else " ?" for x in row) + "]")
    
    # 验证连续性 - 检查相邻访问点在空间上是否邻近
    print(f"\n连续性检查:")
    连续点对 = 0
    总点对 = len(order) - 1
    
    for i in range(len(order) - 1):
        curr_r, curr_c = divmod(order[i], w)
        next_r, next_c = divmod(order[i+1], w)
        
        # 检查是否为相邻点（曼哈顿距离为1）
        if abs(curr_r - next_r) + abs(curr_c - next_c) == 1:
            连续点对 += 1
    
    连续率 = 连续点对 / 总点对 * 100 if 总点对 > 0 else 0
    print(f"  相邻连续点对: {连续点对}/{总点对} ({连续率:.1f}%)")
    
    return grid

def test_standard_hilbert_curves():
    """测试标准尺寸的Hilbert曲线"""
    print("=== Hilbert曲线验证 ===")
    
    tokenizer = FractalHilbertTokenizer()
    
    # 测试经典尺寸
    test_sizes = [(2, 2), (4, 4), (3, 3), (4, 2), (2, 4)]
    
    for h, w in test_sizes:
        order = tokenizer.hilbert_order(h, w)
        print(f"\n测试尺寸: {h}x{w}")
        print(f"遍历顺序: {order}")
        
        if len(order) == h * w:
            visualize_hilbert_order(h, w, order)
        else:
            print(f"❌ 错误：期望长度 {h*w}，实际长度 {len(order)}")

def compare_with_reference_hilbert():
    """与标准Hilbert曲线对比"""
    print("\n=== 与标准Hilbert曲线对比 ===")
    
    # 标准2x2 Hilbert曲线（从左下开始）
    standard_2x2 = [2, 0, 1, 3]  # 左下→左上→右上→右下
    
    tokenizer = FractalHilbertTokenizer()
    our_2x2 = tokenizer.hilbert_order(2, 2)
    
    print(f"标准2x2 Hilbert: {standard_2x2}")
    print(f"我们的2x2实现:   {our_2x2}")
    print(f"是否匹配: {'✅' if our_2x2 == standard_2x2 else '❌'}")
    
    # 可视化标准Hilbert曲线
    print(f"\n标准2x2 Hilbert曲线路径:")
    visualize_hilbert_order(2, 2, standard_2x2)

def test_fractal_context():
    """在分形上下文中测试"""
    print("\n=== 分形递归上下文测试 ===")
    
    tokenizer = FractalHilbertTokenizer(
        min_patch_size=(16, 9),
        max_level=2,
        learnable_split=False
    )
    
    # 创建测试图像
    import torch
    test_img = torch.randn(1, 3, 64, 36)
    
    print(f"输入图像: {test_img.shape}")
    
    # 在四分法中的实际使用
    print(f"\n四分法中的Hilbert顺序:")
    print(f"- 分割后产生4个象限: 左上(0), 右上(1), 左下(2), 右下(3)")
    print(f"- Hilbert遍历顺序: [2, 0, 1, 3] = 左下→左上→右上→右下")
    
    # 执行tokenization
    tokens, levels = tokenizer(test_img)
    print(f"生成tokens: {tokens[0].shape}")
    print(f"层级信息: {levels[0].shape}")

if __name__ == "__main__":
    test_standard_hilbert_curves()
    compare_with_reference_hilbert()
    test_fractal_context()
