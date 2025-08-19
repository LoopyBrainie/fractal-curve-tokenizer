#!/usr/bin/env python3
"""完整系统测试：新的可学习分形tokenizer + ViT集成"""

import torch
import torch.nn as nn
from vit_pytorch.fractal_curve_tokenizer import FractalHilbertTokenizer
from vit_pytorch.fractal_vit import SimpleFractalViT

def test_complete_system():
    """测试完整的分形ViT系统"""
    print("=== 完整分形ViT系统测试 ===")
    
    # 创建测试数据
    batch_size = 2
    channels = 3
    height, width = 64, 36
    num_classes = 10
    
    test_images = torch.randn(batch_size, channels, height, width)
    print(f"输入图像形状: {test_images.shape}")
    
    # 测试1: 基础ViT模型（不可学习分割）
    print("\n--- 测试1: 基础分形ViT ---")
    model_basic = SimpleFractalViT(
        image_size=(height, width),
        num_classes=num_classes,
        dim=256,
        depth=2,
        heads=4,
        mlp_dim=512,
        min_patch_size=(16, 9),
        max_level=3
    )
    
    # 设置tokenizer为非可学习模式
    model_basic.fractal_tokenizer = FractalHilbertTokenizer(
        min_patch_size=(16, 9),
        max_level=3,
        learnable_split=False
    )
    
    with torch.no_grad():
        output_basic = model_basic(test_images)
        print(f"基础模型输出形状: {output_basic.shape}")
        print(f"输出值范围: [{output_basic.min().item():.4f}, {output_basic.max().item():.4f}]")
    
    # 测试2: 可学习分割ViT模型
    print("\n--- 测试2: 可学习分形ViT ---")
    model_learnable = SimpleFractalViT(
        image_size=(height, width),
        num_classes=num_classes,
        dim=256,
        depth=2,
        heads=4,
        mlp_dim=512,
        min_patch_size=(16, 9),
        max_level=3
    )
    
    # 设置tokenizer为可学习模式
    model_learnable.fractal_tokenizer = FractalHilbertTokenizer(
        min_patch_size=(16, 9),
        max_level=3,
        learnable_split=True
    )
    
    with torch.no_grad():
        output_learnable = model_learnable(test_images)
        print(f"可学习模型输出形状: {output_learnable.shape}")
        print(f"输出值范围: [{output_learnable.min().item():.4f}, {output_learnable.max().item():.4f}]")
    
    # 测试3: 模型参数统计
    print("\n--- 测试3: 模型参数统计 ---")
    basic_params = sum(p.numel() for p in model_basic.parameters())
    learnable_params = sum(p.numel() for p in model_learnable.parameters())
    
    print(f"基础模型参数数量: {basic_params:,}")
    print(f"可学习模型参数数量: {learnable_params:,}")
    print(f"额外参数: {learnable_params - basic_params:,}")
    
    # 测试4: tokenizer行为对比
    print("\n--- 测试4: Tokenizer行为对比 ---")
    single_image = test_images[0:1]
    
    # 基础tokenizer
    basic_tokens, basic_levels = model_basic.fractal_tokenizer(single_image)
    basic_token_count = basic_tokens[0].shape[0]
    print(f"基础tokenizer token数: {basic_token_count}")
    
    # 可学习tokenizer
    learnable_tokens, learnable_levels = model_learnable.fractal_tokenizer(single_image)
    learnable_token_count = learnable_tokens[0].shape[0]
    print(f"可学习tokenizer token数: {learnable_token_count}")
    
    print(f"Token数量差异: {basic_token_count - learnable_token_count}")
    
    # 测试5: 梯度传播检查
    print("\n--- 测试5: 梯度传播测试 ---")
    
    # 创建损失函数
    criterion = nn.CrossEntropyLoss()
    target = torch.randint(0, num_classes, (batch_size,))
    
    # 测试可学习模型的梯度
    model_learnable.train()
    output = model_learnable(test_images)
    loss = criterion(output, target)
    
    # 检查分割决策网络是否有梯度
    if model_learnable.fractal_tokenizer.split_decision is not None:
        # 清零梯度
        model_learnable.zero_grad()
        loss.backward()
        
        # 检查分割网络的梯度
        split_net = model_learnable.fractal_tokenizer.split_decision
        has_gradient = any(p.grad is not None and p.grad.sum().item() != 0 
                          for p in split_net.parameters())
        print(f"分割决策网络有梯度: {has_gradient}")
        print(f"训练损失: {loss.item():.4f}")
    else:
        print("分割决策网络未启用")

def test_different_image_sizes():
    """测试不同图像尺寸的处理能力"""
    print("\n=== 不同图像尺寸测试 ===")
    
    sizes = [
        (32, 18),   # 小图像
        (64, 36),   # 中等图像  
        (128, 72),  # 大图像
    ]
    
    for h, w in sizes:
        print(f"\n--- 测试尺寸: {h}x{w} ---")
        
        model = SimpleFractalViT(
            image_size=(h, w),
            num_classes=5,
            dim=128,
            depth=2,
            heads=2,
            mlp_dim=256,
            min_patch_size=(16, 9),
            max_level=2
        )
        
        # 设置为可学习分割
        model.fractal_tokenizer = FractalHilbertTokenizer(
            min_patch_size=(16, 9),
            max_level=2,
            learnable_split=True
        )
        
        test_img = torch.randn(1, 3, h, w)
        
        with torch.no_grad():
            output = model(test_img)
            tokens, levels = model.fractal_tokenizer(test_img)
            
        print(f"  输出形状: {output.shape}")
        print(f"  Token数: {tokens[0].shape[0]}")
        print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    test_complete_system()
    test_different_image_sizes()
    print("\n=== 所有测试完成 ===")
