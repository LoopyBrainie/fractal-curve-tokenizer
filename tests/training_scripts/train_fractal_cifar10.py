#!/usr/bin/env python3
"""
使用Fractal ViT训练CIFAR-10分类模型
"""

import os
import sys
import time
import datetime
import tarfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from vit_pytorch.fractal_vit import SimpleFractalViT

def prepare_cifar10_data():
    """准备CIFAR-10数据集"""
    data_dir = project_root / "workspace"
    tar_path = data_dir / "cifar-10-python.tar.gz"
    extract_dir = data_dir / "cifar-10-batches-py"
    
    # 如果数据集还没解压，自动解压
    if not extract_dir.exists():
        if tar_path.exists():
            print("正在解压CIFAR-10数据集...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=data_dir)
            print("数据集已解压完成")
        else:
            raise FileNotFoundError(f"未找到 {tar_path}，请检查数据集文件")
    
    return str(data_dir)

def get_data_loaders(data_root, batch_size=32, num_workers=2):
    """获取数据加载器"""
    
    # 数据增强和预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整为适合ViT的尺寸
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                           std=[0.2023, 0.1994, 0.2010])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                           std=[0.2023, 0.1994, 0.2010])
    ])
    
    # 加载训练集和测试集
    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False,
        transform=train_transform
    )
    
    val_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=False,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def create_model(num_classes=10):
    """创建Fractal ViT模型"""
    model = SimpleFractalViT(
        image_size=224,
        num_classes=num_classes,
        dim=384,              # 减小维度以适应较小的训练数据
        depth=6,
        heads=6,
        mlp_dim=768,
        pool='cls',
        channels=3,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.1,
        min_patch_size=(16, 16),  # 调整patch大小
        max_level=3              # 限制分形层级
    )
    return model

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 打印进度
        if batch_idx % 50 == 0:
            print(f'Epoch {epoch:2d} | Batch {batch_idx:3d}/{len(train_loader)} | '
                  f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def save_checkpoint(model, optimizer, epoch, loss, acc, save_dir):
    """保存检查点"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fractal_vit_cifar10_e{epoch:02d}_acc{acc:.1f}_{timestamp}.pth"
    filepath = save_dir / filename
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': acc,
    }, filepath)
    
    print(f"模型已保存: {filepath}")
    return filepath

def main():
    """主训练函数"""
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 超参数
    config = {
        'batch_size': 16,        # 减小batch size以适应分形tokenizer
        'learning_rate': 1e-4,   # 较小的学习率
        'num_epochs': 20,
        'num_workers': 2,
        'save_dir': 'workspace/checkpoints'
    }
    
    print("配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 准备数据
    print("\n准备数据集...")
    data_root = prepare_cifar10_data()
    train_loader, val_loader = get_data_loaders(
        data_root, 
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    
    # 创建模型
    print("\n创建Fractal ViT模型...")
    model = create_model()
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config['learning_rate'], 
                           weight_decay=0.01)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs']
    )
    
    # 训练循环
    print(f"\n开始训练 {config['num_epochs']} 个epoch...")
    best_acc = 0.0
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 训练
        start_time = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch} 结果:")
        print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  耗时: {epoch_time:.1f}s")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc, 
                config['save_dir']
            )
            print(f"  🎉 新的最佳准确率: {best_acc:.2f}%")
    
    print(f"\n训练完成! 最佳验证准确率: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
