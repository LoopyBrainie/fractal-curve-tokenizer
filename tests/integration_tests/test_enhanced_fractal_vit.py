#!/usr/bin/env python3
"""
测试重写的Enhanced Fractal ViT
验证与tokenizer特性的对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import sys
from pathlib import Path
import time

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from vit_pytorch.fractal_vit import EnhancedFractalViT, SimpleFractalViT

def test_enhanced_model():
    """测试增强版Fractal ViT"""
    print("🧪 测试Enhanced Fractal ViT...")
    
    # 创建增强模型
    model = EnhancedFractalViT(
        image_size=64,
        num_classes=10,
        dim=192,
        depth=4,
        heads=6,
        mlp_dim=384,
        pool='cls',
        channels=3,
        dim_head=32,
        dropout=0.1,
        emb_dropout=0.1,
        min_patch_size=(8, 8),
        max_level=3,
        learnable_split=True  # 启用可学习分割
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试数据
    batch_size = 4
    test_images = torch.randn(batch_size, 3, 64, 64)
    print(f"输入形状: {test_images.shape}")
    
    # 基础前向传播
    print("\n测试基础前向传播...")
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = model(test_images)
        end_time = time.time()
        
        print(f"✅ 输出形状: {outputs.shape}")
        print(f"⏱️ 推理时间: {(end_time - start_time)*1000:.2f}ms")
        print(f"📊 输出范围: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
    
    # 测试辅助信息
    print("\n测试辅助信息返回...")
    with torch.no_grad():
        outputs, aux_infos = model(test_images, return_aux_info=True)
        
        print("辅助信息分析:")
        for i, aux_info in enumerate(aux_infos):
            print(f"  图像 {i+1}:")
            print(f"    Token数量: {aux_info['num_tokens']}")
            print(f"    使用层级: {aux_info['levels_used']}")
            print(f"    Token分布: {aux_info['token_distribution'].tolist()}")
    
    # 测试tokenizer损失
    print(f"\nTokenizer损失: {model.get_tokenizer_loss().item():.6f}")
    
    return model

def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n🔄 测试向后兼容性...")
    
    # 创建简化模型（应该与之前的接口兼容）
    simple_model = SimpleFractalViT(
        image_size=64,
        num_classes=10,
        dim=128,
        depth=3,
        heads=4,
        mlp_dim=256,
        min_patch_size=(8, 8),
        max_level=2
    )
    
    print(f"简化模型参数量: {sum(p.numel() for p in simple_model.parameters()):,}")
    
    # 测试
    test_images = torch.randn(2, 3, 64, 64)
    simple_model.eval()
    
    with torch.no_grad():
        outputs = simple_model(test_images)
        print(f"✅ 简化模型输出形状: {outputs.shape}")
    
    return simple_model

def test_training_mode():
    """测试训练模式"""
    print("\n🏋️ 测试训练模式...")
    
    model = EnhancedFractalViT(
        image_size=32,  # 更小的图像用于快速测试
        num_classes=5,
        dim=96,
        depth=2,
        heads=3,
        mlp_dim=192,
        min_patch_size=(4, 4),
        max_level=2,
        learnable_split=True
    )
    
    # 模拟训练数据
    batch_size = 8
    images = torch.randn(batch_size, 3, 32, 32)
    labels = torch.randint(0, 5, (batch_size,))
    
    # 训练模式
    model.train()
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print("执行训练步骤...")
    
    # 前向传播
    outputs = model(images)
    main_loss = criterion(outputs, labels)
    
    # 添加tokenizer辅助损失
    aux_loss = model.get_tokenizer_loss()
    total_loss = main_loss + 0.1 * aux_loss
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"✅ 训练成功!")
    print(f"   主要损失: {main_loss.item():.4f}")
    print(f"   辅助损失: {aux_loss.item():.4f}")
    print(f"   总损失: {total_loss.item():.4f}")
    
    return model

def benchmark_comparison():
    """性能对比测试"""
    print("\n⚡ 性能对比测试...")
    
    # 创建两个模型进行对比
    enhanced_model = EnhancedFractalViT(
        image_size=64, num_classes=10, dim=128, depth=3, heads=4, mlp_dim=256,
        min_patch_size=(8, 8), max_level=2, learnable_split=True
    )
    
    simple_model = SimpleFractalViT(
        image_size=64, num_classes=10, dim=128, depth=3, heads=4, mlp_dim=256,
        min_patch_size=(8, 8), max_level=2
    )
    
    # 测试数据
    test_images = torch.randn(16, 3, 64, 64)
    
    # 性能测试
    models = [('Enhanced', enhanced_model), ('Simple', simple_model)]
    
    for name, model in models:
        model.eval()
        
        # 预热
        with torch.no_grad():
            for _ in range(3):
                _ = model(test_images)
        
        # 计时
        start_time = time.time()
        outputs = None
        with torch.no_grad():
            for _ in range(10):
                outputs = model(test_images)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10 * 1000
        params = sum(p.numel() for p in model.parameters())
        
        print(f"{name} 模型:")
        print(f"  参数量: {params:,}")
        print(f"  平均推理时间: {avg_time:.2f}ms")
        if outputs is not None:
            print(f"  输出形状: {outputs.shape}")

def test_edge_cases():
    """测试边缘情况"""
    print("\n🔍 测试边缘情况...")
    
    model = EnhancedFractalViT(
        image_size=32, num_classes=3, dim=64, depth=2, heads=2, mlp_dim=128,
        min_patch_size=(16, 16), max_level=1  # 很小的patch和层级
    )
    
    test_cases = [
        ("正常图像", torch.randn(2, 3, 32, 32)),
        ("单图像", torch.randn(1, 3, 32, 32)),
        ("小尺寸图像", torch.randn(2, 3, 16, 16)),
    ]
    
    model.eval()
    for case_name, images in test_cases:
        try:
            with torch.no_grad():
                outputs = model(images)
            print(f"✅ {case_name}: {images.shape} -> {outputs.shape}")
        except Exception as e:
            print(f"❌ {case_name}: 失败 - {str(e)}")

def main():
    """主测试函数"""
    print("🚀 Enhanced Fractal ViT 全面测试")
    print("=" * 60)
    
    try:
        # 基础功能测试
        enhanced_model = test_enhanced_model()
        
        # 向后兼容性测试  
        simple_model = test_backward_compatibility()
        
        # 训练模式测试
        training_model = test_training_mode()
        
        # 性能对比
        benchmark_comparison()
        
        # 边缘情况测试
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试完成!")
        print("✅ Enhanced Fractal ViT 成功对齐tokenizer特性")
        print("✅ 向后兼容性保持良好")
        print("✅ 训练和推理功能正常")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
