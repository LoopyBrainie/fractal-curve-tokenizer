import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from vit_pytorch.fractal_curve_tokenizer import FractalHilbertTokenizer

def test_adaptive_tokenization():
    """测试自适应tokenization功能"""
    print("=== 自适应Tokenization验证 ===")
    
    # 创建可学习的自适应tokenizer
    adaptive_tokenizer = FractalHilbertTokenizer(
        min_patch_size=(16, 9),
        max_level=3,
        learnable_split=True
    )
    
    # 创建默认tokenizer作为对比（不可学习）
    default_tokenizer = FractalHilbertTokenizer(
        min_patch_size=(16, 9),
        max_level=3,
        learnable_split=False
    )
    
    # 创建测试图像：一半复杂，一半简单
    test_img = torch.randn(1, 3, 64, 36)
    
    # 让左半部分更复杂（高方差）
    test_img[:, :, :, :18] = torch.randn(1, 3, 64, 18) * 2.0  # 高方差
    # 让右半部分更简单（低方差）
    test_img[:, :, :, 18:] = torch.ones(1, 3, 64, 18) * 0.1   # 低方差
    
    print(f"\n测试图像形状: {test_img.shape}")
    print(f"左半部分方差: {torch.var(test_img[:, :, :, :18]).item():.4f}")
    print(f"右半部分方差: {torch.var(test_img[:, :, :, 18:]).item():.4f}")
    
    print("\n--- 可学习自适应Tokenizer ---")
    adaptive_tokens_list, adaptive_levels_list = adaptive_tokenizer(test_img)
    adaptive_tokens, adaptive_levels = adaptive_tokens_list[0], adaptive_levels_list[0]
    print(f"生成token数: {adaptive_tokens.shape[0]}")
    print(f"Token维度: {adaptive_tokens.shape}")
    print(f"层级信息: {adaptive_levels.shape}")
    
    print("\n--- 默认Tokenizer ---")
    default_tokens_list, default_levels_list = default_tokenizer(test_img)
    default_tokens, default_levels = default_tokens_list[0], default_levels_list[0]
    print(f"生成token数: {default_tokens.shape[0]}")
    print(f"Token维度: {default_tokens.shape}")
    print(f"层级信息: {default_levels.shape}")
    
    print(f"\n可学习vs默认token数量比较: {adaptive_tokens.shape[0]} vs {default_tokens.shape[0]}")
    
    # 分析层级分布
    print("\n--- 层级分布分析 ---")
    print("可学习tokenizer的层级分布:")
    for level in range(4):
        count = (adaptive_levels[:, 0] == level).sum().item()
        print(f"  Level {level}: {count} tokens")
    
    print("默认tokenizer的层级分布:")
    for level in range(4):
        count = (default_levels[:, 0] == level).sum().item()
        print(f"  Level {level}: {count} tokens")

def test_learnable_split_training():
    """测试可学习分割决策的基本功能"""
    print("\n=== 可学习分割决策基本测试 ===")
    
    # 创建可学习tokenizer
    tokenizer = FractalHilbertTokenizer(
        min_patch_size=(16, 9),
        max_level=3,
        learnable_split=True
    )
    
    # 创建测试图像
    test_img = torch.randn(2, 3, 64, 36)  # batch of 2 images
    
    if tokenizer.split_decision is not None:
        print("分割决策网络已启用")
        print("网络结构:", tokenizer.split_decision)
        
        # 测试前向传播
        tokens_list, levels_list = tokenizer(test_img)
        
        print(f"图像1 token数: {tokens_list[0].shape[0]}")
        print(f"图像2 token数: {tokens_list[1].shape[0]}")
        
        # 分析层级分布
        print("图像1层级分布:")
        for level in range(4):
            count = (levels_list[0][:, 0] == level).sum().item()
            print(f"  Level {level}: {count} tokens")
            
        print("图像2层级分布:")
        for level in range(4):
            count = (levels_list[1][:, 0] == level).sum().item()
            print(f"  Level {level}: {count} tokens")
    else:
        print("分割决策网络未启用，跳过测试")

def test_different_patch_features():
    """测试不同patch特征对分割决策的影响"""
    print("\n=== Patch特征影响测试 ===")
    
    # 创建可学习tokenizer
    tokenizer = FractalHilbertTokenizer(
        min_patch_size=(8, 8),
        max_level=2,
        learnable_split=True
    )
    
    # 创建不同特征的测试patches
    test_cases = [
        ("高方差patch", torch.randn(1, 3, 32, 32) * 2.0),
        ("低方差patch", torch.ones(1, 3, 32, 32) * 0.1),
        ("梯度丰富patch", torch.zeros(1, 3, 32, 32)),
        ("普通随机patch", torch.randn(1, 3, 32, 32)),
    ]
    
    # 创建梯度丰富的patch（棋盘模式）
    test_cases[2] = ("梯度丰富patch", 
        torch.zeros(1, 3, 32, 32).fill_(0.1) + 
        torch.tensor([[[[(i+j) % 2 for j in range(32)] for i in range(32)] for _ in range(3)]]).float())
    
    for name, img in test_cases:
        print(f"\n--- {name} ---")
        print(f"图像方差: {torch.var(img).item():.4f}")
        
        tokens_list, levels_list = tokenizer(img)
        tokens, levels = tokens_list[0], levels_list[0]
        print(f"生成token数: {tokens.shape[0]}")
        
        # 层级分布
        for level in range(3):
            count = (levels[:, 0] == level).sum().item()
            print(f"  Level {level}: {count} tokens")

if __name__ == "__main__":
    test_adaptive_tokenization()
    test_learnable_split_training()
    test_different_patch_features()
