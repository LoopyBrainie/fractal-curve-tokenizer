#!/usr/bin/env python3
"""
使用Enhanced Fractal ViT训练CIFAR-10
展示新增强特性的使用
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

from vit_pytorch.fractal_vit import EnhancedFractalViT

def main():
    print("🚀 Enhanced Fractal ViT 训练演示")
    print("展示与改进tokenizer特性的对齐")
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 增强配置
    config = {
        'image_size': 64,
        'batch_size': 24,
        'learning_rate': 8e-4,
        'num_epochs': 3,
        'subset_size': 3000,  # 快速演示用
        'learnable_split': True,  # 启用可学习分割
        'aux_loss_weight': 0.1,   # 辅助损失权重
    }
    
    print("增强配置:", config)
    
    # 数据预处理 
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    
    # 加载CIFAR-10数据集的子集
    full_dataset = datasets.CIFAR10(
        root="workspace",
        train=True,
        download=False,
        transform=transform
    )
    
    train_subset = Subset(full_dataset, range(config['subset_size']))
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
    
    # 验证集
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
    val_subset = Subset(val_dataset, range(800))
    val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"训练样本: {len(train_subset)}, 验证样本: {len(val_subset)}")
    
    # 创建增强版Fractal ViT模型
    model = EnhancedFractalViT(
        image_size=config['image_size'],
        num_classes=10,
        dim=256,                 # 较大维度
        depth=5,                 # 更深网络
        heads=8,
        mlp_dim=512,
        pool='cls',
        channels=3,
        dim_head=32,
        dropout=0.1,
        emb_dropout=0.1,
        min_patch_size=(8, 8),
        max_level=3,             # 更深的分形层级
        learnable_split=config['learnable_split']
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    # CIFAR-10类别名称
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 训练循环
    print(f"\n开始训练 {config['num_epochs']} 个epoch...")
    best_acc = 0.0
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_aux_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播，获取辅助信息
            outputs, aux_infos = model(inputs, return_aux_info=True)
            
            # 主要损失
            main_loss = criterion(outputs, targets)
            
            # 辅助损失（tokenizer相关）
            aux_loss = model.get_tokenizer_loss()
            
            # 总损失
            total_loss = main_loss + config['aux_loss_weight'] * aux_loss
            
            total_loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += main_loss.item()
            train_aux_loss += aux_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 15 == 0:
                # 显示tokenization统计信息
                avg_tokens = sum(info['num_tokens'] for info in aux_infos) / len(aux_infos)
                levels_used = set()
                for info in aux_infos:
                    levels_used.update(info['levels_used'])
                
                print(f"  Batch {batch_idx:2d}/{len(train_loader)} | "
                      f"Loss: {main_loss.item():.4f} | Aux: {aux_loss.item():.4f} | "
                      f"Acc: {100.*correct/total:.1f}% | "
                      f"Tokens: {avg_tokens:.1f} | Levels: {sorted(levels_used)}")
        
        epoch_time = time.time() - start_time
        train_acc = 100. * correct / total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_token_stats = {'total_tokens': 0, 'level_counts': [0]*4}
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, aux_infos = model(inputs, return_aux_info=True)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # 收集tokenization统计
                for info in aux_infos:
                    val_token_stats['total_tokens'] += info['num_tokens']
                    for level in info['levels_used']:
                        if level < len(val_token_stats['level_counts']):
                            val_token_stats['level_counts'][level] += 1
        
        val_acc = 100. * val_correct / val_total
        avg_val_tokens = val_token_stats['total_tokens'] / val_total
        
        # 更新学习率
        scheduler.step()
        
        print(f"\nEpoch {epoch} 结果:")
        print(f"  训练 - Loss: {train_loss/len(train_loader):.4f}, "
              f"Aux Loss: {train_aux_loss/len(train_loader):.4f}, Acc: {train_acc:.1f}%")
        print(f"  验证 - Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.1f}%")
        print(f"  Tokenization - 平均tokens: {avg_val_tokens:.1f}, "
              f"层级分布: {val_token_stats['level_counts']}")
        print(f"  耗时: {epoch_time:.1f}s")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # 保存最佳模型
            save_path = Path("workspace") / f"enhanced_fractal_vit_best_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'config': config
            }, save_path)
            print(f"  🎉 新的最佳准确率: {best_acc:.1f}% - 模型已保存: {save_path}")
    
    print(f"\n✅ 训练完成! 最佳验证准确率: {best_acc:.1f}%")
    
    # 最终分析
    print(f"\n🔍 最终模型分析:")
    model.eval()
    with torch.no_grad():
        # 分析一个batch的tokenization行为
        test_inputs, test_targets = next(iter(val_loader))
        test_inputs = test_inputs[:8].to(device)  # 取8个样本
        test_targets = test_targets[:8]
        
        outputs, aux_infos = model(test_inputs, return_aux_info=True)
        _, predicted = outputs.max(1)
        
        print("样本分析:")
        for i in range(len(aux_infos)):
            true_class = class_names[test_targets[i]]
            pred_class = class_names[predicted[i]]
            info = aux_infos[i]
            
            status = "✅" if predicted[i] == test_targets[i] else "❌"
            print(f"  {status} {true_class:10} → {pred_class:10} | "
                  f"Tokens: {info['num_tokens']:2d} | "
                  f"Levels: {info['levels_used']} | "
                  f"分布: {[int(x) for x in info['token_distribution']]}")

if __name__ == "__main__":
    main()
