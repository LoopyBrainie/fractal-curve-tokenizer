import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from vit_pytorch.fractal_curve_tokenizer import FractalHilbertTokenizer
from vit_pytorch.fractal_vit import SimpleFractalViT

def benchmark_fractal_tokenizer():
    """基准测试分形tokenizer"""
    print("=== 分形Tokenizer基准测试 ===")
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建tokenizer
    tokenizer = FractalHilbertTokenizer(
        min_patch_size=(16, 9),
        max_level=3
    ).to(device)
    
    # 测试不同尺寸的输入
    test_sizes = [
        (1, 3, 32, 18),    # 小图像
        (1, 3, 64, 36),    # 中等图像
        (1, 3, 128, 72),   # 大图像
        (2, 3, 64, 36),    # Batch处理
    ]
    
    tokenizer.eval()
    with torch.no_grad():
        for i, size in enumerate(test_sizes):
            print(f"\n测试 {i+1}: 输入尺寸 {size}")
            
            # 创建随机输入
            x = torch.randn(*size).to(device)
            
            # 记录时间
            start_time = time.time()
            
            # tokenization
            tokens, levels = tokenizer(x)
            
            end_time = time.time()
            
            print(f"  输出tokens形状: {tokens.shape}")
            print(f"  输出levels形状: {levels.shape}")
            print(f"  处理时间: {end_time - start_time:.4f} 秒")
            print(f"  生成token数: {tokens.shape[0]}")
            
    print("\n=== Tokenizer基准测试完成 ===")

def benchmark_fractal_vit():
    """基准测试分形ViT模型"""
    print("\n=== 分形ViT模型基准测试 ===")
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = SimpleFractalViT(
        image_size=(224, 224),
        num_classes=1000,
        dim=512,
        depth=4,  # 减少层数以加快测试
        heads=8,
        mlp_dim=1024,
        min_patch_size=(16, 9),
        max_level=3
    ).to(device)
    
    # 测试不同尺寸的输入
    test_sizes = [
        (1, 3, 32, 18),    # 小图像
        (1, 3, 64, 36),    # 中等图像
        (1, 3, 128, 72),   # 大图像
    ]
    
    model.eval()
    with torch.no_grad():
        for i, size in enumerate(test_sizes):
            print(f"\n测试 {i+1}: 输入尺寸 {size}")
            
            # 创建随机输入
            x = torch.randn(*size).to(device)
            
            # 记录时间
            start_time = time.time()
            
            # 前向传播
            try:
                output = model(x)
                end_time = time.time()
                
                print(f"  输出形状: {output.shape}")
                print(f"  处理时间: {end_time - start_time:.4f} 秒")
                
                # 检查输出是否合理
                assert output.shape == (size[0], 1000), f"输出形状错误: {output.shape}"
            except Exception as e:
                print(f"  错误: {e}")
                end_time = time.time()
                print(f"  处理时间: {end_time - start_time:.4f} 秒")
            
    print("\n=== ViT基准测试完成 ===")

def compare_tokenization_methods():
    """比较不同tokenization方法"""
    print("\n=== Tokenization方法比较 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建分形tokenizer
    fractal_tokenizer = FractalHilbertTokenizer(
        min_patch_size=(16, 9),
        max_level=3
    ).to(device)
    
    # 测试图像
    test_images = [
        torch.randn(1, 3, 32, 18),
        torch.randn(1, 3, 64, 36),
        torch.randn(1, 3, 128, 72),
    ]
    
    for i, img in enumerate(test_images):
        x = img.to(device)
        print(f"\n图像 {i+1}: {x.shape}")
        
        # 分形tokenization
        start_time = time.time()
        tokens, levels = fractal_tokenizer(x)
        end_time = time.time()
        
        print(f"  分形tokenization:")
        print(f"    输出tokens形状: {tokens.shape}")
        print(f"    输出levels形状: {levels.shape}")
        print(f"    处理时间: {end_time - start_time:.4f} 秒")
        print(f"    生成token数: {tokens.shape[0]}")
        
        # 显示一些层级信息
        if levels.shape[0] > 0:
            print(f"    前3个token的层级向量:")
            for j in range(min(3, len(levels))):
                print(f"      Token {j}: {levels[j]}")

def test_model_training():
    """测试模型训练"""
    print("\n=== 模型训练测试 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = SimpleFractalViT(
        image_size=(128, 72),
        num_classes=10,
        dim=256,
        depth=2,
        heads=4,
        mlp_dim=512,
        min_patch_size=(16, 9),
        max_level=2
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 创建一些训练数据
    train_data = torch.randn(4, 3, 64, 36).to(device)
    train_labels = torch.randint(0, 10, (4,)).to(device)
    
    # 训练几个epoch
    model.train()
    for epoch in range(3):
        optimizer.zero_grad()
        
        try:
            # 前向传播
            outputs = model(train_data)
            loss = criterion(outputs, train_labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        except Exception as e:
            print(f"Epoch {epoch+1}, 错误: {e}")
    
    print("训练测试完成!")

if __name__ == "__main__":
    benchmark_fractal_tokenizer()
    benchmark_fractal_vit()
    compare_tokenization_methods()
    test_model_training()
