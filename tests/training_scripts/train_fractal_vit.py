#!/usr/bin/env python3
"""
优化版 Fractal ViT 训练脚本
适用于快速测试、原型开发和实验记录

特性:
- 支持多种数据集 (CIFAR-10/100, MNIST)
- 自动实验日志记录
- 优化的模型参数配置
- 早停机制和学习率调度
- 完整的文件管理和目录结构

用法:
    python train_fractal_vit.py                    # 默认配置
    python train_fractal_vit.py --epochs 20       # 自定义轮次
    python train_fractal_vit.py --quick-test       # 快速测试模式
    python train_fractal_vit.py --dataset cifar100 # 使用CIFAR-100数据集
    python train_fractal_vit.py --early-stopping   # 启用早停机制
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.adamw import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

import numpy as np
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from vit_pytorch.fractal_vit import EnhancedFractalViT
    print("✅ 成功导入 Fractal ViT 模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保项目路径正确")
    sys.exit(1)


def set_seed(seed: int = 42) -> None:
    """设置随机种子确保结果可重现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_experiment_dir() -> Path:
    """创建实验目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"fractal_vit_simple_{timestamp}"
    
    # 创建experiments目录结构
    exp_dir = Path("../../experiments") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "visualizations").mkdir(exist_ok=True)
    
    print(f"📁 实验目录创建: {exp_dir}")
    return exp_dir


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """获取数据集信息"""
    dataset_configs = {
        'cifar10': {
            'num_classes': 10,
            'image_size': 32,
            'channels': 3,
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2023, 0.1994, 0.2010),
            'dataset_class': CIFAR10
        },
        'cifar100': {
            'num_classes': 100,
            'image_size': 32,
            'channels': 3,
            'mean': (0.5071, 0.4867, 0.4408),
            'std': (0.2675, 0.2565, 0.2761),
            'dataset_class': CIFAR100
        },
        'mnist': {
            'num_classes': 10,
            'image_size': 28,
            'channels': 1,
            'mean': (0.1307,),
            'std': (0.3081,),
            'dataset_class': MNIST
        }
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    return dataset_configs[dataset_name]


def create_data_loaders(dataset_name: str = 'cifar10', batch_size: int = 64, 
                       subset_size: Optional[int] = None, val_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器"""
    print(f"📥 准备{dataset_name.upper()}数据集...")
    
    dataset_info = get_dataset_info(dataset_name)
    
    # 数据变换
    if dataset_name == 'mnist':
        transform_train = transforms.Compose([
            transforms.Resize(32),  # 统一到32x32
            transforms.ToTensor(),
            transforms.Normalize(dataset_info['mean'], dataset_info['std'])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(dataset_info['mean'], dataset_info['std'])
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(dataset_info['mean'], dataset_info['std'])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(dataset_info['mean'], dataset_info['std'])
        ])
    
    # 数据集
    dataset_class = dataset_info['dataset_class']
    train_dataset = dataset_class(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = dataset_class(root='./data', train=False, download=True, transform=transform_test)
    
    # 创建训练/验证分割
    train_size = len(train_dataset)
    indices = list(range(train_size))
    np.random.shuffle(indices)
    
    val_size = int(val_split * train_size)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # 如果使用子集
    if subset_size:
        train_indices = train_indices[:subset_size]
        val_indices = val_indices[:max(1, subset_size // 10)]
        print(f"📊 使用数据子集 - 训练: {len(train_indices)}, 验证: {len(val_indices)}")
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=2, pin_memory=True
    )
    
    val_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=SubsetRandomSampler(val_indices),
        num_workers=2, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    print(f"✅ 数据加载完成 - 训练: {len(train_indices)}, 验证: {len(val_indices)}, 测试: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def create_model(config: Dict[str, Any]) -> nn.Module:
    """创建Fractal ViT模型"""
    print("🏗️ 创建 Fractal ViT 模型...")
    
    model = EnhancedFractalViT(
        image_size=config['image_size'],
        num_classes=config['num_classes'],
        dim=config['dim'],
        depth=config['depth'],
        heads=config['heads'],
        mlp_dim=config['mlp_dim'],
        channels=config['channels'],
        dropout=config['dropout'],
        min_patch_size=tuple(config['min_patch_size']),
        max_level=config['max_level'],
        learnable_split=config['learnable_split']
    )
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 总参数数量: {total_params:,}")
    print(f"📊 可训练参数: {trainable_params:,}")
    
    return model


def save_experiment_config(exp_dir: Path, config: Dict[str, Any], args: argparse.Namespace) -> None:
    """保存实验配置"""
    config_dict = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'script_version': 'simple_optimized',
            'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        },
        'model_config': config,
        'training_args': vars(args)
    }
    
    config_path = exp_dir / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"💾 实验配置已保存: {config_path}")


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def restore_best_weights_fn(self, model: nn.Module) -> None:
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer, 
                device: torch.device, epoch: int) -> Tuple[float, float]:
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} 训练")
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        output = model(data)
        
        # 处理可能的辅助信息返回
        if isinstance(output, tuple):
            output, _ = output
        
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # 更新进度条
        current_acc = 100. * correct / total
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{current_acc:.1f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    return total_loss / len(train_loader), 100. * correct / total


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device, 
            desc: str = "评估") -> Tuple[float, float]:
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=desc, leave=False)
        for data, target in progress_bar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            
            if isinstance(output, tuple):
                output, _ = output
            
            loss = F.cross_entropy(output, target)
            total_loss += loss.item()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # 更新进度条
            current_acc = 100. * correct / total
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.1f}%'
            })
    
    return total_loss / len(data_loader), 100. * correct / total


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='优化版 Fractal ViT 训练脚本')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮次')
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=5e-4, help='初始学习率')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='权重衰减')
    
    # 数据参数
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100', 'mnist'], help='数据集选择')
    parser.add_argument('--val-split', type=float, default=0.1, help='验证集比例')
    
    # 模型参数
    parser.add_argument('--dim', type=int, default=192, help='模型维度')
    parser.add_argument('--depth', type=int, default=8, help='Transformer深度')
    parser.add_argument('--heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--max-level', type=int, default=3, help='分形最大层级')
    
    # 训练策略
    parser.add_argument('--early-stopping', action='store_true', help='启用早停机制')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--quick-test', action='store_true', help='快速测试模式')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num-workers', type=int, default=2, help='数据加载worker数')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建实验目录
    exp_dir = create_experiment_dir()
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"🔧 GPU: {torch.cuda.get_device_name()}")
        print(f"🔧 CUDA版本: {torch.version.cuda}")
    
    # 快速测试模式
    if args.quick_test:
        args.epochs = 5
        subset_size = 1000
        print("⚡ 快速测试模式已启用")
    else:
        subset_size = None
    
    # 获取数据集信息
    dataset_info = get_dataset_info(args.dataset)
    
    # 模型配置（基于训练结果优化）
    config = {
        'image_size': dataset_info['image_size'],
        'num_classes': dataset_info['num_classes'],
        'channels': dataset_info['channels'],
        'dim': args.dim,
        'depth': args.depth,
        'heads': args.heads,
        'mlp_dim': args.dim * 2,  # 通常是dim的2倍
        'dropout': args.dropout,
        'min_patch_size': [4, 4],
        'max_level': args.max_level,
        'learnable_split': True
    }
    
    # 保存实验配置
    save_experiment_config(exp_dir, config, args)
    
    print("📋 训练配置:")
    print(f"  - 数据集: {args.dataset.upper()}")
    print(f"  - 轮次: {args.epochs}")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 学习率: {args.lr}")
    print(f"  - 模型维度: {config['dim']}")
    print(f"  - Transformer深度: {config['depth']}")
    print(f"  - 最小patch: {config['min_patch_size']}")
    print(f"  - 最大层级: {config['max_level']}")
    print(f"  - 早停机制: {'启用' if args.early_stopping else '禁用'}")
    
    # 创建数据和模型
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size, 
        subset_size=subset_size,
        val_split=args.val_split
    )
    
    model = create_model(config).to(device)
    
    # 优化器和调度器（基于分析结果优化）
    optimizer = AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 使用余弦退火调度器，带重启
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=args.epochs // 4,  # 1/4 epochs作为一个周期
        T_mult=2,
        eta_min=args.lr * 0.01
    )
    
    # 早停机制
    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)
        print(f"📊 早停机制: 耐心值={args.patience}")
    
    # 训练历史
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0
    best_epoch = 0
    
    print("\n🚀 开始训练...")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print("-" * 30)
            
            # 训练
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch+1)
            
            # 验证
            val_loss, val_acc = evaluate(model, val_loader, device, "验证")
            
            # 更新调度器
            scheduler.step()
            
            # 记录历史
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                
                # 保存到实验目录
                exp_checkpoint_path = exp_dir / "checkpoints" / "best_model.pth"
                torch.save(model.state_dict(), exp_checkpoint_path)
                
                # 同时保存到workspace目录
                workspace_dir = Path("../../workspace/models/fractal_vit")
                workspace_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                workspace_checkpoint_path = workspace_dir / f"fractal_vit_simple_best_{timestamp}.pth"
                torch.save(model.state_dict(), workspace_checkpoint_path)
                
                print(f"💾 保存新的最佳模型")
                print(f"   实验目录: {exp_checkpoint_path}")
                print(f"   工作目录: {workspace_checkpoint_path}")
            
            # 打印结果
            print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
            print(f"学习率: {optimizer.param_groups[0]['lr']:.2e}")
            
            # 早停检查
            if early_stopping is not None:
                if early_stopping(val_loss, model):
                    print(f"\n🛑 早停触发! 验证损失在 {args.patience} 个epoch内没有改善")
                    early_stopping.restore_best_weights_fn(model)
                    break
            
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    
    training_time = time.time() - start_time
    
    # 最终测试
    print("\n📊 最终测试...")
    # 加载最佳模型
    exp_checkpoint_path = exp_dir / "checkpoints" / "best_model.pth"
    if exp_checkpoint_path.exists():
        model.load_state_dict(torch.load(exp_checkpoint_path))
        print(f"📁 加载最佳模型: {exp_checkpoint_path}")
    
    test_loss, test_acc = evaluate(model, test_loader, device, "测试")
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_acc': test_acc,
        'training_time': training_time
    }
    
    history_path = exp_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # 结果总结
    print("\n" + "=" * 60)
    print("🎉 训练完成!")
    print("=" * 60)
    print(f"⏱️ 训练时间: {training_time:.2f} 秒")
    print(f"🏆 最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"📊 最终测试准确率: {test_acc:.2f}%")
    print(f"📁 实验目录: {exp_dir}")
    
    # 保存训练曲线
    save_training_plots(exp_dir, history, args.epochs)


def save_training_plots(exp_dir: Path, history: Dict[str, Any], total_epochs: int) -> None:
    """保存训练曲线图"""
    if total_epochs <= 1:
        return
        
    try:
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history['train_losses']) + 1)
        
        # 损失曲线
        ax1.plot(epochs, history['train_losses'], 'b-', label='训练损失', linewidth=2)
        ax1.plot(epochs, history['val_losses'], 'r-', label='验证损失', linewidth=2)
        ax1.set_title('训练损失曲线', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(epochs, history['train_accs'], 'b-', label='训练准确率', linewidth=2)
        ax2.plot(epochs, history['val_accs'], 'r-', label='验证准确率', linewidth=2)
        ax2.axhline(y=history['best_val_acc'], color='g', linestyle='--', alpha=0.7, label=f'最佳验证准确率: {history["best_val_acc"]:.2f}%')
        ax2.set_title('准确率曲线', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 损失差异
        loss_diff = np.array(history['val_losses']) - np.array(history['train_losses'])
        ax3.plot(epochs, loss_diff, 'purple', linewidth=2)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('过拟合指标 (验证损失 - 训练损失)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss Difference')
        ax3.grid(True, alpha=0.3)
        
        # 准确率差异
        acc_diff = np.array(history['train_accs']) - np.array(history['val_accs'])
        ax4.plot(epochs, acc_diff, 'orange', linewidth=2)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('准确率差异 (训练准确率 - 验证准确率)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy Difference (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存到实验目录
        plot_path = exp_dir / "visualizations" / "training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        # 同时保存到workspace目录
        workspace_vis_dir = Path("../../workspace/visualizations")
        workspace_vis_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workspace_plot_path = workspace_vis_dir / f"fractal_vit_simple_curves_{timestamp}.png"
        plt.savefig(workspace_plot_path, dpi=150, bbox_inches='tight')
        
        print(f"📈 训练曲线已保存:")
        print(f"   实验目录: {plot_path}")
        print(f"   工作目录: {workspace_plot_path}")
        
        plt.close()
        
    except ImportError:
        print("⚠️ matplotlib 未安装，跳过绘制训练曲线")


if __name__ == "__main__":
    main()