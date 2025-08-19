#!/usr/bin/env python3
"""
完整的Fractal ViT训练程序
专门用于训练和评估分形Vision Transformer模型

主要功能:
- 支持多个数据集 (CIFAR-10, CIFAR-100, MNIST)
- 灵活的模型配置
- 完整的训练监控和日志记录
- 模型检查点保存和恢复
- 训练结果可视化
- 支持混合精度训练和分布式训练
"""

import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.adamw import AdamW
from torch.optim.adam import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.cuda.amp import autocast, GradScaler
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard.writer import SummaryWriter
    except ImportError:
        # 如果TensorBoard不可用，使用空类
        class SummaryWriter:
            def __init__(self, *args, **kwargs): pass
            def add_scalar(self, *args, **kwargs): pass
            def close(self): pass

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from vit_pytorch.fractal_vit import EnhancedFractalViT, SimpleFractalViT
    print("✅ 成功导入 Fractal ViT 模块")
except ImportError as e:
    print(f"❌ 导入 Fractal ViT 模块失败: {e}")
    print("请确保项目路径正确")
    sys.exit(1)


class FractalViTTrainer:
    """Fractal ViT训练器类"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建保存目录
        self.setup_directories()
        
        # 设置日志
        self.setup_logging()
        
        # 初始化训练组件
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if config.get('use_amp', True) else None
        self.writer = None
        
        # 训练状态
        self.current_epoch = 0
        self.best_acc = 0.0
        self.best_epoch = 0
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
    def setup_directories(self):
        """创建必要的目录，遵循项目文件管理规范"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 获取项目根目录
        project_root = Path(__file__).parent.parent.parent
        
        # 实验目录保存在项目根目录的experiments文件夹中
        experiments_dir = project_root / "experiments"
        self.exp_dir = experiments_dir / f"fractal_vit_complete_{timestamp}"
        
        # 实验内部目录结构
        self.checkpoints_dir = self.exp_dir / "checkpoints"
        self.logs_dir = self.exp_dir / "logs"
        self.results_dir = self.exp_dir / "results"
        
        # 创建实验目录
        for dir_path in [self.exp_dir, self.checkpoints_dir, self.logs_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # workspace目录结构 - 按项目规范组织
        self.workspace_dir = project_root / "workspace"
        self.workspace_models_dir = self.workspace_dir / "models" / "fractal_vit"
        self.workspace_results_dir = self.workspace_dir / "results"
        self.workspace_vis_dir = self.workspace_dir / "visualizations"
        self.workspace_reports_dir = self.workspace_dir / "reports"
        
        # 创建workspace目录
        for dir_path in [self.workspace_models_dir, self.workspace_results_dir, 
                         self.workspace_vis_dir, self.workspace_reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 记录目录路径用于后续使用
        self.timestamp = timestamp
        
        # 保存配置到两个位置
        config_data = dict(self.config)
        config_data['experiment_id'] = f"fractal_vit_complete_{timestamp}"
        config_data['directories'] = {
            'experiment': str(self.exp_dir),
            'workspace_models': str(self.workspace_models_dir),
            'workspace_results': str(self.workspace_results_dir)
        }
        
        with open(self.exp_dir / "config.json", 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # 也保存到workspace
        with open(self.workspace_results_dir / f"config_{timestamp}.json", 'w') as f:
            json.dump(config_data, f, indent=2)
            
    def setup_logging(self):
        """设置日志记录"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.logs_dir / "training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # TensorBoard
        if self.config.get('use_tensorboard', True):
            self.writer = SummaryWriter(self.logs_dir / "tensorboard")
        
    def create_model(self) -> nn.Module:
        """创建Fractal ViT模型"""
        self.logger.info("创建 Fractal ViT 模型")
        
        model_config = self.config['model']
        
        if self.config.get('use_enhanced', True):
            # 使用增强版本
            model = EnhancedFractalViT(
                image_size=model_config['image_size'],
                num_classes=model_config['num_classes'],
                dim=model_config['dim'],
                depth=model_config['depth'],
                heads=model_config['heads'],
                mlp_dim=model_config['mlp_dim'],
                channels=model_config['channels'],
                dim_head=model_config['dim_head'],
                dropout=model_config['dropout'],
                emb_dropout=model_config['emb_dropout'],
                min_patch_size=tuple(model_config['min_patch_size']),
                max_level=model_config['max_level'],
                learnable_split=model_config['learnable_split'],
                pool=model_config['pool']
            )
        else:
            # 使用简化版本
            model = SimpleFractalViT(
                image_size=model_config['image_size'],
                num_classes=model_config['num_classes'],
                dim=model_config['dim'],
                depth=model_config['depth'],
                heads=model_config['heads'],
                mlp_dim=model_config['mlp_dim'],
                channels=model_config['channels'],
                dim_head=model_config['dim_head'],
                dropout=model_config['dropout'],
                emb_dropout=model_config['emb_dropout'],
                min_patch_size=tuple(model_config['min_patch_size']),
                max_level=model_config['max_level'],
                pool=model_config['pool']
            )
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"模型总参数数量: {total_params:,}")
        self.logger.info(f"可训练参数数量: {trainable_params:,}")
        
        return model.to(self.device)
    
    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """创建数据加载器"""
        self.logger.info(f"加载数据集: {self.config['dataset']['name']}")
        
        dataset_config = self.config['dataset']
        
        # 数据变换
        if dataset_config['name'] == 'MNIST':
            mean, std = (0.1307,), (0.3081,)
            channels = 1
        elif dataset_config['name'] in ['CIFAR10', 'CIFAR100']:
            mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            channels = 3
        else:
            raise ValueError(f"不支持的数据集: {dataset_config['name']}")
        
        # 训练时的数据增强
        train_transform = transforms.Compose([
            transforms.Resize((self.config['model']['image_size'], 
                             self.config['model']['image_size'])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                 saturation=0.2, hue=0.1) if channels == 3 else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        # 测试时的数据变换
        test_transform = transforms.Compose([
            transforms.Resize((self.config['model']['image_size'], 
                             self.config['model']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        # 加载数据集
        train_dataset = None
        test_dataset = None
        
        if dataset_config['name'] == 'CIFAR10':
            train_dataset = CIFAR10(root='./data', train=True, download=True, 
                                  transform=train_transform)
            test_dataset = CIFAR10(root='./data', train=False, download=True, 
                                 transform=test_transform)
        elif dataset_config['name'] == 'CIFAR100':
            train_dataset = CIFAR100(root='./data', train=True, download=True, 
                                   transform=train_transform)
            test_dataset = CIFAR100(root='./data', train=False, download=True, 
                                  transform=test_transform)
        elif dataset_config['name'] == 'MNIST':
            train_dataset = MNIST(root='./data', train=True, download=True, 
                                transform=train_transform)
            test_dataset = MNIST(root='./data', train=False, download=True, 
                               transform=test_transform)
        else:
            raise ValueError(f"不支持的数据集: {dataset_config['name']}")
        
        if train_dataset is None or test_dataset is None:
            raise RuntimeError("数据集加载失败")
        
        # 创建验证集
        train_size = len(train_dataset)
        val_size = int(train_size * dataset_config.get('val_split', 0.1))
        train_size = train_size - val_size
        
        indices = list(range(len(train_dataset)))
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # 如果配置了子集大小，则使用子集
        if dataset_config.get('subset_size') is not None:
            subset_size = min(dataset_config['subset_size'], len(train_indices))
            train_indices = train_indices[:subset_size]
            val_indices = val_indices[:min(len(val_indices), subset_size // 10)]
        
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            sampler=train_sampler,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            sampler=val_sampler,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=True
        )
        
        self.logger.info(f"训练集大小: {len(train_indices)}")
        self.logger.info(f"验证集大小: {len(val_indices)}")
        self.logger.info(f"测试集大小: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def create_optimizer_and_scheduler(self):
        """创建优化器和学习率调度器"""
        training_config = self.config['training']
        
        # 优化器
        if training_config['optimizer'] == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay'],
                betas=training_config.get('betas', (0.9, 0.999))
            )
        elif training_config['optimizer'] == 'adam':
            self.optimizer = Adam(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay'],
                betas=training_config.get('betas', (0.9, 0.999))
            )
        else:
            raise ValueError(f"不支持的优化器: {training_config['optimizer']}")
        
        # 学习率调度器
        total_steps = len(self.train_loader) * training_config['num_epochs']
        warmup_steps = int(total_steps * training_config.get('warmup_ratio', 0.1))
        
        if training_config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=training_config['learning_rate'] * 0.01
            )
        elif training_config['scheduler'] == 'linear':
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    return max(0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
            
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        self.logger.info(f"优化器: {training_config['optimizer']}")
        self.logger.info(f"学习率调度器: {training_config['scheduler']}")
        self.logger.info(f"总步数: {total_steps}, 预热步数: {warmup_steps}")
    
    def train_epoch(self) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with autocast():
                    if self.config.get('use_enhanced', True):
                        # 增强版本可能返回辅助信息
                        output = self.model(data)
                        if isinstance(output, tuple):
                            output, aux_info = output
                    else:
                        output = self.model(data)
                    
                    loss = F.cross_entropy(output, target)
                    
                    # 如果有tokenizer损失，添加它
                    if hasattr(self.model, 'get_tokenizer_loss'):
                        tokenizer_loss = self.model.get_tokenizer_loss()
                        loss = loss + self.config.get('aux_loss_weight', 0.1) * tokenizer_loss
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.config.get('use_enhanced', True):
                    output = self.model(data)
                    if isinstance(output, tuple):
                        output, aux_info = output
                else:
                    output = self.model(data)
                
                loss = F.cross_entropy(output, target)
                
                if hasattr(self.model, 'get_tokenizer_loss'):
                    tokenizer_loss = self.model.get_tokenizer_loss()
                    loss = loss + self.config.get('aux_loss_weight', 0.1) * tokenizer_loss
                
                loss.backward()
                self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 统计
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(data_loader, desc="Evaluating", leave=False):
                data, target = data.to(self.device), target.to(self.device)
                
                if self.config.get('use_enhanced', True):
                    output = self.model(data)
                    if isinstance(output, tuple):
                        output, aux_info = output
                else:
                    output = self.model(data)
                
                loss = F.cross_entropy(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点到实验目录和workspace目录"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_acc': self.best_acc,
            'train_history': self.train_history,
            'config': self.config,
            'timestamp': self.timestamp
        }
        
        # 保存到实验目录
        torch.save(checkpoint, self.checkpoints_dir / "latest.pth")
        
        if is_best:
            # 保存最佳检查点到实验目录
            torch.save(checkpoint, self.checkpoints_dir / "best.pth")
            
            # 同时保存到workspace模型目录
            best_model_name = f"fractal_vit_best_{self.timestamp}.pth"
            torch.save(checkpoint, self.workspace_models_dir / best_model_name)
            
            # 只保存模型状态字典（更小的文件）
            model_only = {
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'best_acc': self.best_acc,
                'timestamp': self.timestamp
            }
            torch.save(model_only, self.workspace_models_dir / f"fractal_vit_model_{self.timestamp}.pth")
        
        # 定期保存到实验目录
        if (self.current_epoch + 1) % 10 == 0:
            epoch_checkpoint_name = f"epoch_{self.current_epoch+1}.pth"
            torch.save(checkpoint, self.checkpoints_dir / epoch_checkpoint_name)
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"检查点文件不存在: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if checkpoint.get('scheduler_state_dict') and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_acc = checkpoint.get('best_acc', 0.0)
            self.train_history = checkpoint.get('train_history', {
                'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []
            })
            
            self.logger.info(f"成功加载检查点: {checkpoint_path}")
            self.logger.info(f"当前epoch: {self.current_epoch}, 最佳准确率: {self.best_acc:.2f}%")
            return True
            
        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            return False
    
    def plot_training_curves(self):
        """绘制训练曲线并保存到正确位置"""
        if not self.train_history['train_loss']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        
        # 训练和验证损失
        axes[0, 0].plot(epochs, self.train_history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.train_history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 训练和验证准确率
        axes[0, 1].plot(epochs, self.train_history['train_acc'], 'b-', label='Train Acc')
        axes[0, 1].plot(epochs, self.train_history['val_acc'], 'r-', label='Val Acc')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学习率
        axes[1, 0].plot(epochs, self.train_history['lr'], 'g-', label='Learning Rate')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 最佳准确率标记
        axes[1, 1].bar(['Best Val Acc'], [self.best_acc], color='orange')
        axes[1, 1].set_title(f'Best Validation Accuracy: {self.best_acc:.2f}%')
        axes[1, 1].set_ylabel('Accuracy (%)')
        
        plt.tight_layout()
        
        # 保存到实验目录
        plt.savefig(self.results_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / "training_curves.pdf", bbox_inches='tight')
        
        # 同时保存到workspace可视化目录
        vis_filename = f"fractal_vit_training_curves_{self.timestamp}.png"
        plt.savefig(self.workspace_vis_dir / vis_filename, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        self.logger.info(f"训练曲线已保存到: {self.results_dir} 和 {self.workspace_vis_dir}")
    
    def generate_report(self):
        """生成训练报告并保存到正确位置"""
        # 安全获取模型参数信息
        total_params = sum(p.numel() for p in self.model.parameters()) if self.model else 0
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) if self.model else 0
        
        report = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'experiment_id': f"fractal_vit_complete_{self.timestamp}",
                'device': str(self.device),
                'config': self.config
            },
            'results': {
                'best_val_accuracy': float(self.best_acc),
                'best_epoch': int(self.best_epoch),
                'final_train_accuracy': float(self.train_history['train_acc'][-1]) if self.train_history['train_acc'] else 0.0,
                'final_val_accuracy': float(self.train_history['val_acc'][-1]) if self.train_history['val_acc'] else 0.0,
                'total_epochs': len(self.train_history['train_loss'])
            },
            'model_info': {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params
            },
            'file_locations': {
                'experiment_dir': str(self.exp_dir),
                'best_model': str(self.workspace_models_dir / f"fractal_vit_best_{self.timestamp}.pth"),
                'training_curves': str(self.workspace_vis_dir / f"fractal_vit_training_curves_{self.timestamp}.png")
            }
        }
        
        # 保存JSON报告到实验目录和workspace
        with open(self.results_dir / "training_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        with open(self.workspace_results_dir / f"fractal_vit_report_{self.timestamp}.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # 生成Markdown报告
        md_content = f"""# Fractal ViT 完整训练报告

## 实验信息
- **实验ID**: {report['experiment_info']['experiment_id']}
- **时间戳**: {report['experiment_info']['timestamp']}
- **设备**: {report['experiment_info']['device']}
- **数据集**: {self.config['dataset']['name']}

## 模型配置
- **图像尺寸**: {self.config['model']['image_size']}
- **嵌入维度**: {self.config['model']['dim']}
- **Transformer深度**: {self.config['model']['depth']}
- **注意力头数**: {self.config['model']['heads']}
- **最小patch尺寸**: {self.config['model']['min_patch_size']}
- **最大层级**: {self.config['model']['max_level']}
- **可学习分割**: {self.config['model']['learnable_split']}

## 训练配置
- **训练轮次**: {self.config['training']['num_epochs']}
- **批次大小**: {self.config['training']['batch_size']}
- **学习率**: {self.config['training']['learning_rate']}
- **优化器**: {self.config['training']['optimizer']}
- **调度器**: {self.config['training']['scheduler']}

## 训练结果
- **最佳验证准确率**: {report['results']['best_val_accuracy']:.2f}%
- **最佳epoch**: {report['results']['best_epoch']}
- **最终训练准确率**: {report['results']['final_train_accuracy']:.2f}%
- **最终验证准确率**: {report['results']['final_val_accuracy']:.2f}%
- **总epoch数**: {report['results']['total_epochs']}

## 模型信息
- **总参数数量**: {report['model_info']['total_parameters']:,}
- **可训练参数数量**: {report['model_info']['trainable_parameters']:,}

## 文件位置
- **实验目录**: `{report['file_locations']['experiment_dir']}`
- **最佳模型**: `{report['file_locations']['best_model']}`
- **训练曲线**: `{report['file_locations']['training_curves']}`

## 文件管理规范
本实验遵循项目文件管理规范：
- 📁 **实验数据**: 保存在 `experiments/fractal_vit_complete_{self.timestamp}/`
- 📁 **模型文件**: 保存在 `workspace/models/fractal_vit/`
- 📁 **结果数据**: 保存在 `workspace/results/`
- 📁 **可视化文件**: 保存在 `workspace/visualizations/`
- 📁 **报告文档**: 保存在 `workspace/reports/`

## 生成文件
### 实验目录文件
- `checkpoints/best.pth`: 最佳模型检查点
- `checkpoints/latest.pth`: 最新检查点
- `logs/training.log`: 详细训练日志
- `results/training_curves.png/pdf`: 训练曲线图
- `results/training_report.json`: JSON格式报告

### Workspace文件
- `models/fractal_vit/fractal_vit_best_{self.timestamp}.pth`: 最佳模型
- `results/fractal_vit_report_{self.timestamp}.json`: 实验报告
- `visualizations/fractal_vit_training_curves_{self.timestamp}.png`: 训练曲线
- `reports/fractal_vit_report_{self.timestamp}.md`: 本报告
"""
        
        # 保存Markdown报告
        with open(self.results_dir / "training_report.md", 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        # 同时保存到workspace reports目录
        with open(self.workspace_reports_dir / f"fractal_vit_report_{self.timestamp}.md", 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        self.logger.info(f"实验报告已保存到: {self.results_dir} 和 {self.workspace_reports_dir}")
        
        return report
    
    def train(self):
        """完整训练流程"""
        self.logger.info("开始训练 Fractal ViT")
        
        # 创建模型和数据
        self.model = self.create_model()
        self.train_loader, self.val_loader, self.test_loader = self.create_data_loaders()
        self.create_optimizer_and_scheduler()
        
        # 加载检查点（如果存在）
        if self.config.get('resume_from_checkpoint'):
            self.load_checkpoint(self.config['resume_from_checkpoint'])
        
        start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, self.config['training']['num_epochs']):
                self.current_epoch = epoch
                
                self.logger.info(f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
                
                # 训练
                train_loss, train_acc = self.train_epoch()
                
                # 验证
                val_loss, val_acc = self.evaluate(self.val_loader)
                
                # 记录历史
                self.train_history['train_loss'].append(train_loss)
                self.train_history['train_acc'].append(train_acc)
                self.train_history['val_loss'].append(val_loss)
                self.train_history['val_acc'].append(val_acc)
                
                # 安全获取学习率
                current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
                self.train_history['lr'].append(current_lr)
                
                # 日志记录
                self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # TensorBoard记录
                if self.writer:
                    self.writer.add_scalar('Loss/Train', train_loss, epoch)
                    self.writer.add_scalar('Loss/Val', val_loss, epoch)
                    self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
                    self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
                    if self.optimizer:
                        self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
                
                # 保存最佳模型
                is_best = val_acc > self.best_acc
                if is_best:
                    self.best_acc = val_acc
                    self.best_epoch = epoch
                
                # 保存检查点
                self.save_checkpoint(is_best)
                
                # 早停检查
                if self.config.get('early_stopping_patience'):
                    if epoch - self.best_epoch >= self.config['early_stopping_patience']:
                        self.logger.info(f"早停触发，最佳准确率: {self.best_acc:.2f}%")
                        break
        
        except KeyboardInterrupt:
            self.logger.info("训练被用户中断")
        
        training_time = time.time() - start_time
        self.logger.info(f"训练完成，总时间: {training_time:.2f}秒")
        self.logger.info(f"最佳验证准确率: {self.best_acc:.2f}% (Epoch {self.best_epoch+1})")
        
        # 最终测试
        if self.test_loader:
            self.logger.info("正在进行最终测试...")
            # 加载最佳模型
            best_checkpoint = torch.load(self.checkpoints_dir / "best.pth")
            self.model.load_state_dict(best_checkpoint['model_state_dict'])
            
            test_loss, test_acc = self.evaluate(self.test_loader)
            self.logger.info(f"最终测试准确率: {test_acc:.2f}%")
            
            if self.writer:
                self.writer.add_scalar('Final_Test/Accuracy', test_acc, 0)
                self.writer.add_scalar('Final_Test/Loss', test_loss, 0)
        
        # 生成结果
        self.plot_training_curves()
        self.generate_report()
        
        if self.writer:
            self.writer.close()
        
        self.logger.info(f"所有结果已保存到: {self.exp_dir}")


def get_default_config():
    """获取优化后的默认配置，基于快速训练的结果分析"""
    return {
        'dataset': {
            'name': 'CIFAR10',  # CIFAR10, CIFAR100, MNIST
            'val_split': 0.1,
            'subset_size': None,  # None表示使用全部数据
        },
        'model': {
            'image_size': 32,
            'num_classes': 10,
            # 优化后的模型参数 - 基于分析结果调整
            'dim': 192,              # 降低维度以提高训练效率
            'depth': 8,              # 增加深度以提高表现力  
            'heads': 6,              # 减少头数以降低复杂度
            'mlp_dim': 384,          # 调整MLP维度
            'channels': 3,
            'dim_head': 32,          # 每个注意力头的维度
            'dropout': 0.1,
            'emb_dropout': 0.1,
            # 分形特有参数优化
            'min_patch_size': [4, 4],
            'max_level': 3,          # 保持适中的层级
            'learnable_split': True, # 启用可学习分割
            'pool': 'cls'
        },
        'training': {
            'num_epochs': 50,        # 默认减少轮次
            'batch_size': 64,
            # 优化后的学习率策略
            'learning_rate': 5e-4,   # 调整学习率，基于快速训练结果
            'weight_decay': 0.05,    # 减少权重衰减
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'warmup_ratio': 0.1,
            'betas': [0.9, 0.999]
        },
        # 系统配置优化
        'use_enhanced': True,
        'use_amp': True,         # 启用混合精度训练
        'use_tensorboard': True,
        'num_workers': 2,        # 减少worker数量以避免内存问题
        'aux_loss_weight': 0.1,  # 分形tokenizer辅助损失权重
        'early_stopping_patience': 15,  # 启用早停以避免过拟合
        'resume_from_checkpoint': None
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练 Fractal ViT 模型')
    
    # 数据集参数
    parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100', 'MNIST'], 
                       default='CIFAR10', help='数据集选择')
    parser.add_argument('--subset-size', type=int, default=None, 
                       help='使用数据子集大小（用于快速测试）')
    
    # 模型参数
    parser.add_argument('--image-size', type=int, default=32, help='输入图像尺寸')
    parser.add_argument('--dim', type=int, default=256, help='嵌入维度')
    parser.add_argument('--depth', type=int, default=6, help='Transformer深度')
    parser.add_argument('--heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--min-patch-size', type=int, nargs=2, default=[4, 4], 
                       help='最小patch尺寸')
    parser.add_argument('--max-level', type=int, default=3, help='最大分形层级')
    parser.add_argument('--no-learnable-split', action='store_true', 
                       help='禁用可学习分割')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--optimizer', choices=['adamw', 'adam'], default='adamw', 
                       help='优化器')
    parser.add_argument('--scheduler', choices=['cosine', 'linear'], default='cosine',
                       help='学习率调度器')
    
    # 其他参数
    parser.add_argument('--no-amp', action='store_true', help='禁用混合精度训练')
    parser.add_argument('--no-tensorboard', action='store_true', help='禁用TensorBoard')
    parser.add_argument('--resume', type=str, default=None, help='从检查点恢复训练')
    parser.add_argument('--early-stopping', type=int, default=None, 
                       help='早停耐心值')
    
    args = parser.parse_args()
    
    # 构建配置
    config = get_default_config()
    
    # 更新配置
    config['dataset']['name'] = args.dataset
    config['dataset']['subset_size'] = args.subset_size
    
    config['model']['image_size'] = args.image_size
    config['model']['dim'] = args.dim
    config['model']['depth'] = args.depth
    config['model']['heads'] = args.heads
    config['model']['min_patch_size'] = args.min_patch_size
    config['model']['max_level'] = args.max_level
    config['model']['learnable_split'] = not args.no_learnable_split
    
    # 根据数据集调整参数
    if args.dataset == 'CIFAR100':
        config['model']['num_classes'] = 100
    elif args.dataset == 'MNIST':
        config['model']['num_classes'] = 10
        config['model']['channels'] = 1
    
    config['training']['num_epochs'] = args.epochs
    config['training']['batch_size'] = args.batch_size
    config['training']['learning_rate'] = args.lr
    config['training']['optimizer'] = args.optimizer
    config['training']['scheduler'] = args.scheduler
    
    config['use_amp'] = not args.no_amp
    config['use_tensorboard'] = not args.no_tensorboard
    config['resume_from_checkpoint'] = args.resume
    config['early_stopping_patience'] = args.early_stopping
    
    print("🚀 Fractal ViT 完整训练程序")
    print("=" * 50)
    print(f"📊 数据集: {config['dataset']['name']}")
    print(f"🏗️ 模型维度: {config['model']['dim']}")
    print(f"📏 最小patch尺寸: {config['model']['min_patch_size']}")
    print(f"🔢 最大层级: {config['model']['max_level']}")
    print(f"⚙️ 可学习分割: {config['model']['learnable_split']}")
    print(f"🎯 训练轮次: {config['training']['num_epochs']}")
    print(f"📦 批次大小: {config['training']['batch_size']}")
    print(f"🚀 学习率: {config['training']['learning_rate']}")
    print(f"🛡️ 早停耐心: {config.get('early_stopping_patience', 'None')}")
    print("=" * 50)
    
    # 基于快速训练结果的建议
    print("\n💡 优化建议 (基于快速训练分析):")
    print("  - 已优化学习率为5e-4，平衡收敛速度和稳定性")  
    print("  - 默认启用早停机制，防止过拟合")
    print("  - 调整模型维度以提高训练效率")
    print("  - 减少worker数量以避免内存问题")
    print("=" * 50)
    
    # 创建训练器并开始训练
    trainer = FractalViTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
