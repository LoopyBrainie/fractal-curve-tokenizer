#!/usr/bin/env python3
"""
完整的Fractal ViT vs 标准ViT训练比较程序
数据集: CIFAR-10
设备: CUDA加速

控制变量实验设计:
- 相同的数据预处理和增强
- 相同的优化器、学习率调度
- 相同的训练轮次和批次大小
- 相同的模型维度和深度参数
- 只有tokenizer不同：Fractal vs Standard
"""

import os
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wandb

# 导入我们的模型
from vit_pytorch.vit import ViT
from vit_pytorch.fractal_vit import EnhancedFractalViT


def set_seed(seed=42):
    """设置随机种子以确保实验的可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ExperimentConfig:
    """实验配置类，确保控制变量的一致性"""
    
    def __init__(self):
        # 数据集配置
        self.dataset_name = "CIFAR-10"
        self.num_classes = 10
        self.image_size = 32
        self.channels = 3
        
        # 训练配置
        self.batch_size = 128
        self.num_epochs = 100
        self.learning_rate = 3e-4
        self.weight_decay = 0.1
        self.warmup_epochs = 10
        
        # 模型配置（控制变量 - 两个模型使用相同参数）
        self.dim = 384              # 嵌入维度
        self.depth = 12             # Transformer层数
        self.heads = 6              # 注意力头数
        self.mlp_dim = 1536         # MLP隐藏层维度
        self.dim_head = 64          # 每个注意力头的维度
        self.dropout = 0.1
        self.emb_dropout = 0.1
        
        # 标准ViT配置
        self.standard_patch_size = 4    # CIFAR-10适合的patch size
        
        # Fractal ViT配置
        self.fractal_min_patch_size = (4, 4)
        self.fractal_max_level = 3
        self.fractal_learnable_split = True
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = True  # 混合精度训练
        
        # 保存配置
        self.save_dir = Path("./experiments") / f"cifar10_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.results_dir = self.save_dir / "results"
        
        # 创建目录
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)


def get_data_loaders(config):
    """获取CIFAR-10数据加载器，使用数据增强"""
    
    # 数据增强配置
    train_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 加载数据集
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, test_loader


def create_models(config):
    """创建标准ViT和Fractal ViT模型"""
    
    # 标准ViT模型
    standard_vit = ViT(
        image_size=config.image_size,
        patch_size=config.standard_patch_size,
        num_classes=config.num_classes,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        mlp_dim=config.mlp_dim,
        channels=config.channels,
        dim_head=config.dim_head,
        dropout=config.dropout,
        emb_dropout=config.emb_dropout,
        pool='cls'
    )
    
    # Fractal ViT模型
    fractal_vit = EnhancedFractalViT(
        image_size=config.image_size,
        num_classes=config.num_classes,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        mlp_dim=config.mlp_dim,
        channels=config.channels,
        dim_head=config.dim_head,
        dropout=config.dropout,
        emb_dropout=config.emb_dropout,
        min_patch_size=config.fractal_min_patch_size,
        max_level=config.fractal_max_level,
        learnable_split=config.fractal_learnable_split,
        pool='cls'
    )
    
    return standard_vit, fractal_vit


def create_optimizer_and_scheduler(model, config, total_steps):
    """创建优化器和学习率调度器"""
    
    # AdamW优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Cosine学习率调度器（带warmup）
    def lr_lambda(step):
        if step < config.warmup_epochs * (total_steps // config.num_epochs):
            # Warmup阶段
            return step / (config.warmup_epochs * (total_steps // config.num_epochs))
        else:
            # Cosine衰减阶段
            progress = (step - config.warmup_epochs * (total_steps // config.num_epochs)) / \
                      (total_steps - config.warmup_epochs * (total_steps // config.num_epochs))
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


def train_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, device, config):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if config.use_amp:
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100. * correct / total:.2f}%',
            'LR': f'{scheduler.get_last_lr()[0]:.6f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def test_epoch(model, test_loader, criterion, device):
    """测试一个epoch"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train_model(model, model_name, train_loader, test_loader, config):
    """训练单个模型的完整流程"""
    print(f"\n{'='*60}")
    print(f"开始训练 {model_name}")
    print(f"{'='*60}")
    
    model = model.to(config.device)
    
    # 创建优化器和调度器
    total_steps = len(train_loader) * config.num_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, total_steps)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 混合精度训练
    scaler = GradScaler() if config.use_amp else None
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'lr': []
    }
    
    best_test_acc = 0
    best_epoch = 0
    
    # 训练循环
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print("-" * 40)
        
        start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion, config.device, config
        )
        
        # 测试
        test_loss, test_acc = test_epoch(model, test_loader, criterion, config.device)
        
        epoch_time = time.time() - start_time
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(scheduler.get_last_lr()[0])
        
        # 打印结果
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"Time: {epoch_time:.2f}s")
        
        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss
            }, config.checkpoint_dir / f"best_{model_name.lower().replace(' ', '_')}.pth")
        
        # 定期保存检查点
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history
            }, config.checkpoint_dir / f"checkpoint_{model_name.lower().replace(' ', '_')}_epoch_{epoch+1}.pth")
    
    print(f"\n{model_name} 训练完成!")
    print(f"最佳测试准确率: {best_test_acc:.2f}% (Epoch {best_epoch+1})")
    
    return history, best_test_acc, best_epoch


def plot_training_curves(standard_history, fractal_history, config):
    """绘制训练曲线对比图"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Fractal ViT vs Standard ViT Training Comparison', fontsize=16)
    
    epochs = range(1, config.num_epochs + 1)
    
    # 训练损失
    axes[0, 0].plot(epochs, standard_history['train_loss'], label='Standard ViT', color='blue', linewidth=2)
    axes[0, 0].plot(epochs, fractal_history['train_loss'], label='Fractal ViT', color='red', linewidth=2)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 训练准确率
    axes[0, 1].plot(epochs, standard_history['train_acc'], label='Standard ViT', color='blue', linewidth=2)
    axes[0, 1].plot(epochs, fractal_history['train_acc'], label='Fractal ViT', color='red', linewidth=2)
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 测试损失
    axes[1, 0].plot(epochs, standard_history['test_loss'], label='Standard ViT', color='blue', linewidth=2)
    axes[1, 0].plot(epochs, fractal_history['test_loss'], label='Fractal ViT', color='red', linewidth=2)
    axes[1, 0].set_title('Test Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 测试准确率
    axes[1, 1].plot(epochs, standard_history['test_acc'], label='Standard ViT', color='blue', linewidth=2)
    axes[1, 1].plot(epochs, fractal_history['test_acc'], label='Fractal ViT', color='red', linewidth=2)
    axes[1, 1].set_title('Test Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.results_dir / "training_curves_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(config.results_dir / "training_curves_comparison.pdf", bbox_inches='tight')
    plt.show()


def generate_experiment_report(config, standard_history, fractal_history, 
                              standard_best_acc, fractal_best_acc, 
                              standard_best_epoch, fractal_best_epoch,
                              total_time):
    """生成实验报告"""
    
    report = {
        'experiment_info': {
            'dataset': config.dataset_name,
            'total_training_time': f"{total_time:.2f} seconds",
            'device': str(config.device),
            'timestamp': datetime.now().isoformat()
        },
        'model_config': {
            'dim': config.dim,
            'depth': config.depth,
            'heads': config.heads,
            'mlp_dim': config.mlp_dim,
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs,
            'learning_rate': config.learning_rate
        },
        'standard_vit': {
            'patch_size': config.standard_patch_size,
            'best_test_accuracy': float(standard_best_acc),
            'best_epoch': int(standard_best_epoch + 1),
            'final_train_accuracy': float(standard_history['train_acc'][-1]),
            'final_test_accuracy': float(standard_history['test_acc'][-1])
        },
        'fractal_vit': {
            'min_patch_size': config.fractal_min_patch_size,
            'max_level': config.fractal_max_level,
            'learnable_split': config.fractal_learnable_split,
            'best_test_accuracy': float(fractal_best_acc),
            'best_epoch': int(fractal_best_epoch + 1),
            'final_train_accuracy': float(fractal_history['train_acc'][-1]),
            'final_test_accuracy': float(fractal_history['test_acc'][-1])
        },
        'comparison': {
            'accuracy_improvement': float(fractal_best_acc - standard_best_acc),
            'relative_improvement': float((fractal_best_acc - standard_best_acc) / standard_best_acc * 100),
            'fractal_is_better': fractal_best_acc > standard_best_acc
        }
    }
    
    # 保存JSON报告
    with open(config.results_dir / "experiment_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 生成Markdown报告
    md_report = f"""# Fractal ViT vs Standard ViT Comparison Report

## Experiment Information
- **Dataset**: {config.dataset_name}
- **Total Training Time**: {total_time:.2f} seconds
- **Device**: {config.device}
- **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Configuration
- **Embedding Dimension**: {config.dim}
- **Transformer Depth**: {config.depth}
- **Attention Heads**: {config.heads}
- **MLP Dimension**: {config.mlp_dim}
- **Batch Size**: {config.batch_size}
- **Training Epochs**: {config.num_epochs}
- **Learning Rate**: {config.learning_rate}

## Results Summary

### Standard ViT
- **Patch Size**: {config.standard_patch_size}x{config.standard_patch_size}
- **Best Test Accuracy**: {standard_best_acc:.2f}% (Epoch {standard_best_epoch + 1})
- **Final Train Accuracy**: {standard_history['train_acc'][-1]:.2f}%
- **Final Test Accuracy**: {standard_history['test_acc'][-1]:.2f}%

### Fractal ViT
- **Min Patch Size**: {config.fractal_min_patch_size}
- **Max Level**: {config.fractal_max_level}
- **Learnable Split**: {config.fractal_learnable_split}
- **Best Test Accuracy**: {fractal_best_acc:.2f}% (Epoch {fractal_best_epoch + 1})
- **Final Train Accuracy**: {fractal_history['train_acc'][-1]:.2f}%
- **Final Test Accuracy**: {fractal_history['test_acc'][-1]:.2f}%

## Comparison Analysis
- **Accuracy Improvement**: {fractal_best_acc - standard_best_acc:+.2f}%
- **Relative Improvement**: {(fractal_best_acc - standard_best_acc) / standard_best_acc * 100:+.2f}%
- **Winner**: {'Fractal ViT' if fractal_best_acc > standard_best_acc else 'Standard ViT'}

## Key Findings
{'✅ Fractal ViT outperformed Standard ViT' if fractal_best_acc > standard_best_acc else '❌ Standard ViT outperformed Fractal ViT'}

## Files Generated
- `training_curves_comparison.png/pdf`: Training curves visualization
- `experiment_report.json`: Detailed numerical results
- `best_standard_vit.pth`: Best Standard ViT checkpoint
- `best_fractal_vit.pth`: Best Fractal ViT checkpoint
"""
    
    with open(config.results_dir / "experiment_report.md", 'w', encoding='utf-8') as f:
        f.write(md_report)
    
    return report


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Fractal ViT vs Standard ViT Comparison on CIFAR-10')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建实验配置
    config = ExperimentConfig()
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    
    print("🚀 Fractal ViT vs Standard ViT 控制变量对比实验")
    print(f"📊 数据集: {config.dataset_name}")
    print(f"🔧 设备: {config.device}")
    print(f"📁 保存路径: {config.save_dir}")
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("⚠️  警告: CUDA不可用，使用CPU训练，速度会较慢")
    else:
        print(f"✅ CUDA可用: {torch.cuda.get_device_name()}")
    
    # 准备数据
    print("\n📥 加载CIFAR-10数据集...")
    train_loader, test_loader = get_data_loaders(config)
    print(f"✅ 训练集大小: {len(train_loader.dataset)}")
    print(f"✅ 测试集大小: {len(test_loader.dataset)}")
    
    # 创建模型
    print("\n🏗️  创建模型...")
    standard_vit, fractal_vit = create_models(config)
    
    standard_params = count_parameters(standard_vit)
    fractal_params = count_parameters(fractal_vit)
    
    print(f"📊 Standard ViT 参数数量: {standard_params:,}")
    print(f"📊 Fractal ViT 参数数量: {fractal_params:,}")
    print(f"📊 参数数量差异: {fractal_params - standard_params:+,}")
    
    # 保存配置
    config_dict = {
        'seed': args.seed,
        'device': str(config.device),
        'dataset': config.dataset_name,
        'batch_size': config.batch_size,
        'num_epochs': config.num_epochs,
        'learning_rate': config.learning_rate,
        'standard_vit_params': standard_params,
        'fractal_vit_params': fractal_params,
        'model_config': {
            'dim': config.dim,
            'depth': config.depth,
            'heads': config.heads,
            'mlp_dim': config.mlp_dim
        }
    }
    
    with open(config.save_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # 开始训练
    print("\n🎯 开始对比训练...")
    start_time = time.time()
    
    # 训练Standard ViT
    standard_history, standard_best_acc, standard_best_epoch = train_model(
        standard_vit, "Standard ViT", train_loader, test_loader, config
    )
    
    # 训练Fractal ViT
    fractal_history, fractal_best_acc, fractal_best_epoch = train_model(
        fractal_vit, "Fractal ViT", train_loader, test_loader, config
    )
    
    total_time = time.time() - start_time
    
    # 生成结果
    print("\n📊 生成实验结果...")
    
    # 绘制对比图
    plot_training_curves(standard_history, fractal_history, config)
    
    # 生成报告
    report = generate_experiment_report(
        config, standard_history, fractal_history,
        standard_best_acc, fractal_best_acc,
        standard_best_epoch, fractal_best_epoch,
        total_time
    )
    
    # 打印总结
    print("\n" + "="*60)
    print("🎉 实验完成!")
    print("="*60)
    print(f"⏱️  总训练时间: {total_time:.2f} 秒")
    print(f"📊 Standard ViT 最佳准确率: {standard_best_acc:.2f}%")
    print(f"📊 Fractal ViT 最佳准确率: {fractal_best_acc:.2f}%")
    print(f"📈 准确率提升: {fractal_best_acc - standard_best_acc:+.2f}%")
    print(f"📁 结果保存在: {config.save_dir}")
    
    if fractal_best_acc > standard_best_acc:
        print("🏆 Fractal ViT 获胜!")
    else:
        print("🏆 Standard ViT 获胜!")
    
    print("\n生成的文件:")
    print(f"  📄 experiment_report.md - 详细实验报告")
    print(f"  📄 experiment_report.json - 数值结果")
    print(f"  📊 training_curves_comparison.png - 训练曲线对比图")
    print(f"  💾 best_standard_vit.pth - 最佳Standard ViT模型")
    print(f"  💾 best_fractal_vit.pth - 最佳Fractal ViT模型")


if __name__ == "__main__":
    main()
