#!/usr/bin/env python3
"""
标准ViT模型训练脚本 - 用于与Fractal ViT对比
严格控制变量以确保公平比较
"""

import os
import sys
import time
import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入标准ViT（从vit_pytorch库）
from vit_pytorch import ViT

def create_standard_vit(config):
    """创建标准ViT模型，参数量与Fractal ViT尽可能接近"""
    
    # 根据图像大小和patch大小计算序列长度
    image_size = config['image_size']
    patch_size = 8  # 使用与Fractal ViT类似的patch大小
    seq_len = (image_size // patch_size) ** 2
    
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=10,
        dim=config['dim'],           # 与Fractal ViT相同的维度
        depth=config['depth'],       # 与Fractal ViT相同的深度
        heads=config['heads'],       # 与Fractal ViT相同的头数
        mlp_dim=config['mlp_dim'],   # 与Fractal ViT相同的MLP维度
        dropout=0.1,
        emb_dropout=0.1
    )
    
    return model

def main():
    print("🎯 标准ViT训练 - 对比基准测试")
    print("严格控制变量以确保公平比较")
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 控制变量配置 - 与Enhanced Fractal ViT完全相同
    config = {
        'image_size': 64,
        'batch_size': 24,        # 相同batch size
        'learning_rate': 8e-4,   # 相同学习率
        'num_epochs': 3,         # 相同训练轮数
        'subset_size': 3000,     # 相同数据量
        
        # 模型参数 - 尽量匹配Fractal ViT
        'dim': 256,              # 相同维度
        'depth': 5,              # 相同深度
        'heads': 8,              # 相同头数
        'mlp_dim': 512,          # 相同MLP维度
    }
    
    print("标准ViT配置:", config)
    
    # 数据预处理 - 与Fractal ViT完全相同
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    
    # 加载相同的CIFAR-10数据集子集
    full_dataset = datasets.CIFAR10(
        root="workspace",
        train=True,
        download=False,
        transform=transform
    )
    
    train_subset = Subset(full_dataset, range(config['subset_size']))
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
    
    # 验证集 - 相同配置
    val_dataset = datasets.CIFAR10(
        root="workspace",
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
    )
    val_subset = Subset(val_dataset, range(800))  # 相同验证集大小
    val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"训练样本: {len(train_subset)}, 验证样本: {len(val_subset)}")
    
    # 创建标准ViT模型
    model = create_standard_vit(config).to(device)
    
    print(f"标准ViT参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器 - 与Fractal ViT相同
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    
    # 学习率调度器 - 与Fractal ViT相同
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    # CIFAR-10类别名称
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 训练性能记录
    training_metrics = {
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': [],
        'epoch_times': []
    }
    
    # 训练循环
    print(f"\n开始训练标准ViT {config['num_epochs']} 个epoch...")
    best_acc = 0.0
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 15 == 0:
                print(f"  Batch {batch_idx:2d}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc: {100.*correct/total:.1f}%")
        
        epoch_time = time.time() - start_time
        train_acc = 100. * correct / total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # 更新学习率
        scheduler.step()
        
        # 记录指标
        training_metrics['train_losses'].append(train_loss/len(train_loader))
        training_metrics['train_accs'].append(train_acc)
        training_metrics['val_losses'].append(val_loss/len(val_loader))
        training_metrics['val_accs'].append(val_acc)
        training_metrics['epoch_times'].append(epoch_time)
        
        print(f"\nEpoch {epoch} 结果:")
        print(f"  训练 - Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.1f}%")
        print(f"  验证 - Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.1f}%")
        print(f"  耗时: {epoch_time:.1f}s")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # 保存最佳模型
            save_path = Path("workspace") / f"standard_vit_best_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'config': config,
                'training_metrics': training_metrics
            }, save_path)
            print(f"  🎉 新的最佳准确率: {best_acc:.1f}% - 模型已保存: {save_path}")
    
    print(f"\n✅ 标准ViT训练完成! 最佳验证准确率: {best_acc:.1f}%")
    
    # 性能分析
    print(f"\n📊 标准ViT性能分析:")
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  最终训练准确率: {training_metrics['train_accs'][-1]:.1f}%")
    print(f"  最佳验证准确率: {best_acc:.1f}%")
    print(f"  平均每epoch时间: {sum(training_metrics['epoch_times'])/len(training_metrics['epoch_times']):.1f}s")
    
    # 推理速度测试
    print(f"\n⚡ 推理速度测试:")
    model.eval()
    test_batch = next(iter(val_loader))[0].to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(5):
            _ = model(test_batch)
    
    # 计时
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    with torch.no_grad():
        for _ in range(20):
            _ = model(test_batch)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / 20 * 1000
    print(f"  平均推理时间: {avg_inference_time:.2f}ms")
    print(f"  推理吞吐量: {len(test_batch) / (avg_inference_time/1000):.1f} images/s")
    
    # 最终测试：随机预测几个样本
    print(f"\n📸 预测示例:")
    model.eval()
    with torch.no_grad():
        test_inputs, test_targets = next(iter(val_loader))
        test_inputs, test_targets = test_inputs[:8].to(device), test_targets[:8]
        test_outputs = model(test_inputs)
        _, test_predicted = test_outputs.max(1)
        
        correct_count = 0
        for i in range(8):
            true_class = class_names[test_targets[i]]
            pred_class = class_names[test_predicted[i]]
            confidence = torch.softmax(test_outputs[i], 0).max().item()
            status = "✅" if test_predicted[i] == test_targets[i] else "❌"
            if test_predicted[i] == test_targets[i]:
                correct_count += 1
            print(f"  {status} 真实: {true_class:10} | 预测: {pred_class:10} | 置信度: {confidence:.2f}")
        
        sample_acc = correct_count / 8 * 100
        print(f"  样本准确率: {sample_acc:.1f}%")
    
    # 保存详细的训练指标用于对比
    metrics_path = Path("workspace") / f"standard_vit_metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json"
    import json
    
    # 转换为JSON可序列化的格式
    json_metrics = {
        'model_type': 'Standard ViT',
        'config': config,
        'training_metrics': training_metrics,
        'final_results': {
            'best_val_acc': best_acc,
            'final_train_acc': training_metrics['train_accs'][-1],
            'total_params': sum(p.numel() for p in model.parameters()),
            'avg_epoch_time': sum(training_metrics['epoch_times'])/len(training_metrics['epoch_times']),
            'avg_inference_time_ms': avg_inference_time,
            'sample_accuracy': sample_acc
        }
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    
    print(f"\n💾 训练指标已保存: {metrics_path}")
    print("\n🎯 标准ViT基准测试完成，可用于与Fractal ViT对比!")

if __name__ == "__main__":
    main()
