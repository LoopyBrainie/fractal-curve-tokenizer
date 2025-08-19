import torch
import torch.nn as nn
from vit_pytorch.fractal_curve_tokenizer import FractalHilbertTokenizer
import numpy as np

def test_hilbert_order_effectiveness():
    """测试 get_hilbert_order_for_level 方法的实际效果"""
    
    print("=" * 100)
    print("get_hilbert_order_for_level 方法实际效果测试")
    print("=" * 100)
    
    # 创建tokenizer实例
    tokenizer = FractalHilbertTokenizer(min_patch_size=(4, 4), max_level=4, learnable_split=False)
    
    def create_pattern_image(h, w, pattern_type="gradient"):
        """创建具有特定模式的测试图像"""
        image = torch.zeros(1, 1, h, w)
        
        if pattern_type == "gradient":
            # 梯度模式 - 测试空间连续性
            for i in range(h):
                for j in range(w):
                    image[0, 0, i, j] = (i + j) / (h + w)
                    
        elif pattern_type == "checkerboard":
            # 棋盘模式 - 测试高频细节处理
            for i in range(h):
                for j in range(w):
                    image[0, 0, i, j] = (i + j) % 2
                    
        elif pattern_type == "position":
            # 位置编码 - 测试空间定位
            for i in range(h):
                for j in range(w):
                    image[0, 0, i, j] = i * w + j
                    
        return image
    
    # 测试不同图像模式和尺寸
    test_configs = [
        (16, 16, "gradient", "正方形梯度"),
        (32, 16, "gradient", "宽矩形梯度"),
        (16, 32, "gradient", "高矩形梯度"),
        (16, 16, "checkerboard", "正方形棋盘"),
        (24, 12, "position", "宽矩形位置编码"),
        (12, 24, "position", "高矩形位置编码"),
    ]
    
    print("\n1. 不同图像模式的tokenization效果")
    print("-" * 70)
    
    for h, w, pattern, desc in test_configs:
        print(f"\n{desc} ({h}x{w}, {pattern}):")
        
        # 创建测试图像
        test_image = create_pattern_image(h, w, pattern)
        
        try:
            # 获取tokens
            tokens_per_image, levels_per_image = tokenizer(test_image)
            
            if len(tokens_per_image) > 0 and len(tokens_per_image[0]) > 0:
                tokens = tokens_per_image[0]
                levels = levels_per_image[0]
                
                print(f"  Token数量: {tokens.shape[0]}")
                print(f"  Token维度: {tokens.shape[1]}")
                
                # 分析层级分布
                depth_counts = {}
                path_analysis = {}
                
                for level_info in levels:
                    depth = level_info[0].item()
                    depth_counts[depth] = depth_counts.get(depth, 0) + 1
                    
                    # 分析路径模式
                    path = tuple(level_info[1:depth+1].tolist())
                    if len(path) > 0:
                        path_key = f"depth_{depth}_len_{len(path)}"
                        if path_key not in path_analysis:
                            path_analysis[path_key] = 0
                        path_analysis[path_key] += 1
                
                print(f"  层级分布: {dict(sorted(depth_counts.items()))}")
                
                # 计算空间相关性（相邻token的相似性）
                spatial_correlation = 0.0
                if len(tokens) > 1:
                    correlations = []
                    for i in range(min(10, len(tokens) - 1)):  # 只计算前10对，避免计算量过大
                        corr = torch.cosine_similarity(tokens[i].unsqueeze(0), tokens[i+1].unsqueeze(0)).item()
                        correlations.append(corr)
                    spatial_correlation = np.mean(correlations)
                
                print(f"  空间相关性: {spatial_correlation:.3f}")
                
            else:
                print("  无tokens生成")
                
        except Exception as e:
            print(f"  处理错误: {e}")
    
    # 2. 测试顺序一致性对token质量的影响
    print("\n\n2. 顺序一致性对token质量的影响")
    print("-" * 70)
    
    def analyze_token_sequence_quality(tokens, levels):
        """分析token序列的质量指标"""
        if len(tokens) == 0:
            return {}
        
        metrics = {}
        
        # 1. 层级一致性
        depths = [level[0].item() for level in levels]
        depth_variance = np.var(depths)
        metrics['depth_variance'] = depth_variance
        
        # 2. 路径复杂度
        path_lengths = []
        for level in levels:
            depth = level[0].item()
            path_lengths.append(depth)
        metrics['avg_path_length'] = np.mean(path_lengths)
        
        # 3. Token间差异
        if len(tokens) > 1:
            pairwise_distances = []
            for i in range(min(20, len(tokens) - 1)):  # 限制计算量
                dist = torch.norm(tokens[i] - tokens[i+1]).item()
                pairwise_distances.append(dist)
            metrics['avg_token_distance'] = np.mean(pairwise_distances)
        else:
            metrics['avg_token_distance'] = 0.0
        
        return metrics
    
    # 测试不同宽高比的影响
    aspect_ratios = [0.5, 1.0, 2.0]
    base_size = 24
    
    for ratio in aspect_ratios:
        if ratio < 1.0:
            h, w = int(base_size / ratio), base_size
        else:
            h, w = base_size, int(base_size * ratio)
        
        print(f"\n宽高比 {ratio:.1f} ({h}x{w}):")
        
        # 使用梯度图像测试
        test_image = create_pattern_image(h, w, "gradient")
        tokens_per_image, levels_per_image = tokenizer(test_image)
        
        if len(tokens_per_image) > 0 and len(tokens_per_image[0]) > 0:
            tokens = tokens_per_image[0]
            levels = levels_per_image[0]
            
            metrics = analyze_token_sequence_quality(tokens, levels)
            
            print(f"  层级方差: {metrics['depth_variance']:.3f}")
            print(f"  平均路径长度: {metrics['avg_path_length']:.2f}")
            print(f"  平均token距离: {metrics['avg_token_distance']:.3f}")
            
            # 分析第1层级使用的顺序
            order_level1 = tokenizer.get_hilbert_order_for_level(1, h, w)
            
            if ratio < 0.67:
                expected_type = "垂直优先"
            elif ratio > 1.5:
                expected_type = "水平优先"  
            else:
                expected_type = "标准"
                
            print(f"  层级1顺序: {order_level1} ({expected_type})")

    # 3. 极端情况测试
    print("\n\n3. 极端情况处理能力")
    print("-" * 70)
    
    extreme_cases = [
        (100, 4, "极宽图"),
        (4, 100, "极高图"),
        (1, 64, "单像素高度"),
        (64, 1, "单像素宽度"),
    ]
    
    for h, w, desc in extreme_cases:
        print(f"\n{desc} ({h}x{w}):")
        
        # 测试是否能正常处理
        try:
            test_image = create_pattern_image(h, w, "position")
            tokens_per_image, levels_per_image = tokenizer(test_image)
            
            if len(tokens_per_image) > 0 and len(tokens_per_image[0]) > 0:
                tokens = tokens_per_image[0]
                levels = levels_per_image[0]
                
                print(f"  成功处理 - Tokens: {tokens.shape[0]}, 维度: {tokens.shape[1]}")
                
                # 检查层级1的顺序选择
                order_level1 = tokenizer.get_hilbert_order_for_level(1, h, w)
                aspect_ratio = w / h if h > 0 else float('inf')
                
                print(f"  宽高比: {aspect_ratio:.2f}")
                print(f"  层级1顺序: {order_level1}")
                
                # 验证顺序的合理性
                if aspect_ratio < 0.67:
                    expected = [2, 3, 0, 1]
                    reasonable = order_level1 == expected
                elif aspect_ratio > 1.5:
                    expected = [2, 0, 1, 3]
                    reasonable = order_level1 == expected
                else:
                    reasonable = True  # 标准情况多种顺序都合理
                
                print(f"  顺序合理性: {'✓' if reasonable else '✗'}")
                
            else:
                print("  无tokens生成")
                
        except Exception as e:
            print(f"  处理失败: {e}")

    print("\n" + "=" * 100)
    print("实际效果测试完成")
    print("=" * 100)

def benchmark_hilbert_order_performance():
    """性能基准测试"""
    print("\n\n4. 性能基准测试")
    print("-" * 70)
    
    import time
    
    tokenizer = FractalHilbertTokenizer(min_patch_size=(4, 4), max_level=5, learnable_split=False)
    
    # 测试不同规模的性能
    test_sizes = [
        (32, 32, 1000),   # 中等尺寸，多次调用
        (64, 64, 100),    # 大尺寸，较少调用
        (128, 128, 10),   # 很大尺寸，少量调用
    ]
    
    for h, w, iterations in test_sizes:
        print(f"\n尺寸 {h}x{w}, 迭代 {iterations} 次:")
        
        # 预热
        for _ in range(5):
            tokenizer.get_hilbert_order_for_level(1, h, w)
        
        # 实际测试
        start_time = time.time()
        for _ in range(iterations):
            for level in range(6):
                tokenizer.get_hilbert_order_for_level(level, h, w)
        end_time = time.time()
        
        total_calls = iterations * 6
        total_time = end_time - start_time
        avg_time = total_time / total_calls * 1000  # 转换为毫秒
        
        print(f"  总调用次数: {total_calls}")
        print(f"  总耗时: {total_time:.3f}秒")
        print(f"  平均耗时: {avg_time:.4f}毫秒/调用")

if __name__ == "__main__":
    test_hilbert_order_effectiveness()
    benchmark_hilbert_order_performance()
