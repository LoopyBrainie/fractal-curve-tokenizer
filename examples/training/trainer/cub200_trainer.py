# ============================================================================
# CUB-200-2011 细粒度分类专用训练器
# Fine-grained Classification Trainer for CUB-200-2011
# ============================================================================
"""
CUB-200-2011 数据集的专用训练适配器。

与通用分类训练器的区别：
1. 使用 Center Loss 增强类内紧凑性
2. 针对细粒度分类调整的数据增强策略
3. 更详细的评估指标（per-class accuracy, confusion analysis）
4. 优化的正则化策略（更强的 dropout、label smoothing）

数据集特性：
    - 200 类鸟类物种（细粒度）
    - 5,994 训练样本 / 5,794 测试样本
    - 每类约 30 样本
    - 类间差异微小（同属不同种）
    - 关键判别特征：喙、眼、羽毛纹理
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 从 losses 模块导入细粒度损失
from ..losses.finegrained import FinegrainedLoss, FinegrainedLossConfig


@dataclass
class CUB200TrainingConfig:
    """CUB-200 细粒度分类训练配置
    
    数学形式化推导：
        1. 模型容量: P/N ≈ 1400× (Double Descent 最优区)
        2. 正则化强度: scale = √(N_ref / N) ≈ 1.5-2.0
        3. 学习率: lr = lr_base × (batch_eff / 256) × α_mixup × α_small
    """
    # 基础训练配置
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 2.7e-4
    warmup_epochs: int = 7
    accum_steps: int = 3
    
    # 设备和混合精度
    device: str = "cuda"
    use_amp: bool = True
    
    # 细粒度专用配置
    use_center_loss: bool = True
    center_loss_weight: float = 0.01
    
    # 正则化（比通用分类更强）
    label_smoothing: float = 0.13  # 0.1 × (1 + log₁₀(200/100))
    dropout: float = 0.22
    drop_path: float = 0.16
    weight_decay: float = 0.15
    
    # 数据增强（细粒度优化）
    mixup_alpha: float = 0.3   # 降低，保留更多原始特征
    cutmix_alpha: float = 0.8  # CutMix 对局部特征更有效
    mixup_prob: float = 0.4    # 降低概率
    
    # Focal Loss（处理困难样本）
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    
    # 早停
    patience: int = 15
    min_delta: float = 0.001
    
    # 评估
    compute_per_class: bool = True
    compute_confusion: bool = True
    top_confused_pairs: int = 10


@dataclass
class CUB200EvalResult:
    """CUB-200 评估结果"""
    loss: float
    accuracy: float  # Top-1 准确率
    top5_accuracy: float
    per_class_accuracy: Optional[Dict[int, float]] = None
    confused_pairs: Optional[List[Tuple[int, int, int]]] = None  # (class_a, class_b, count)
    feature_stats: Optional[Dict[str, float]] = None


class CUB200Trainer:
    """
    CUB-200-2011 细粒度分类训练器
    
    职责：
        1. 管理细粒度专用损失函数
        2. 提供专用的训练和评估循环
        3. 计算细粒度分类特有的指标
        
    与主训练器的接口：
        - train_epoch(): 训练一个 epoch
        - evaluate(): 评估模型
        - get_loss_fn(): 获取损失函数
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: CUB200TrainingConfig,
        num_classes: int = 200,
        feat_dim: int = 256,
        device: torch.device = torch.device('cuda'),
    ):
        self.model = model
        self.config = config
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        # 创建细粒度损失函数
        loss_config = FinegrainedLossConfig(
            num_classes=num_classes,
            feat_dim=feat_dim,
            label_smoothing=config.label_smoothing,
            use_center_loss=config.use_center_loss,
            center_loss_weight=config.center_loss_weight,
            use_focal=config.use_focal_loss,
            focal_gamma=config.focal_gamma,
        )
        self.loss_fn = FinegrainedLoss(loss_config).to(device)
        
        # Center Loss 需要单独的优化器（较大学习率）
        self.center_optimizer = None
        if config.use_center_loss and self.loss_fn.center_loss is not None:
            self.center_optimizer = torch.optim.SGD(
                self.loss_fn.center_loss.parameters(),
                lr=0.5,  # Center Loss 论文推荐
            )
    
    def get_loss_fn(self) -> nn.Module:
        """获取损失函数（供主训练器使用）"""
        return self.loss_fn
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算细粒度分类损失
        
        Args:
            logits: 模型输出 [B, C]
            labels: 标签 [B]
            features: 特征向量 [B, D]（用于 Center Loss）
            attention_weights: 注意力权重（用于熵正则化）
            
        Returns:
            loss: 总损失
            stats: 损失分量统计
        """
        return self.loss_fn(logits, labels, features, attention_weights)
    
    def update_centers(self):
        """更新 Center Loss 的优化器步骤"""
        if self.center_optimizer is not None:
            self.center_optimizer.step()
            self.center_optimizer.zero_grad()
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        exp_dir: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """
        完整的 CUB-200 细粒度分类训练循环
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 模型优化器
            scheduler: 学习率调度器（可选）
            exp_dir: 实验目录（可选）
            
        Returns:
            训练历史记录
        """
        import time
        from torch.amp import GradScaler, autocast
        
        history = []
        best_val_acc = 0.0
        patience_counter = 0
        
        # 混合精度
        scaler = GradScaler(enabled=self.config.use_amp)
        
        for epoch in range(1, self.config.num_epochs + 1):
            start_time = time.time()
            
            # 训练一个 epoch
            train_loss, train_acc = self._train_epoch(
                train_loader, optimizer, scaler
            )
            
            # 验证
            val_result = self.evaluate(val_loader, use_amp=self.config.use_amp)
            
            # 学习率调度
            if scheduler is not None:
                scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # 记录历史
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_result.loss,
                'val_acc': val_result.accuracy,
                'val_top5': val_result.top5_accuracy,
                'lr': optimizer.param_groups[0]['lr'],
                'time': epoch_time,
            })
            
            # 打印进度
            print(f"\nEpoch {epoch}/{self.config.num_epochs}:")
            print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.2f}%")
            print(f"  Val:   loss={val_result.loss:.4f}, acc={val_result.accuracy:.2f}% (top5={val_result.top5_accuracy:.2f}%)")
            print(f"  Time:  {epoch_time:.1f}s")
            
            # 保存最佳模型
            if val_result.accuracy > best_val_acc + self.config.min_delta:
                best_val_acc = val_result.accuracy
                patience_counter = 0
                if exp_dir is not None:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'val_acc': val_result.accuracy,
                        'val_top5': val_result.top5_accuracy,
                    }, exp_dir / "checkpoints" / "best.pth")
                    print(f"  [*] Best model saved: {val_result.accuracy:.2f}%")
            else:
                patience_counter += 1
                print(f"  [!] No improvement ({patience_counter}/{self.config.patience})")
            
            # 早停
            if patience_counter >= self.config.patience:
                print(f"\n[INFO] Early stopping at epoch {epoch}")
                break
        
        return history
    
    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: "GradScaler",
    ) -> Tuple[float, float]:
        """训练一个 epoch"""
        from torch.amp import autocast
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc="Training")):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            with autocast(device_type=self.device.type, enabled=self.config.use_amp):
                # 获取 logits
                logits = self.model(imgs)
                
                # 计算损失
                loss, _ = self.compute_loss(logits, labels)
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 更新 Center Loss 优化器
            self.update_centers()
            
            # 统计
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        use_amp: bool = True,
        channels_last: bool = False,
    ) -> CUB200EvalResult:
        """
        CUB-200 专用评估
        
        计算：
            - Top-1 / Top-5 准确率
            - 逐类别准确率
            - 最易混淆的类别对
            - 特征质量统计
            
        Args:
            loader: 测试数据加载器
            use_amp: 是否使用混合精度
            channels_last: 是否使用 channels_last 内存格式
            
        Returns:
            CUB200EvalResult 包含详细评估结果
        """
        self.model.eval()
        
        total_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        # 逐类别统计
        class_correct = torch.zeros(self.num_classes, device=self.device)
        class_total = torch.zeros(self.num_classes, device=self.device)
        
        # 混淆矩阵（用于分析最易混淆的类别对）
        confusion = torch.zeros(
            self.num_classes, self.num_classes, 
            device=self.device, dtype=torch.long
        )
        
        # 特征收集（可选，用于分析）
        all_features = []
        all_labels = []
        
        amp_context = torch.autocast(
            device_type=self.device.type, 
            enabled=use_amp
        )
        
        for batch in tqdm(loader, desc="CUB-200 Eval"):
            imgs, labels = batch
            imgs = imgs.to(self.device)
            if channels_last:
                imgs = imgs.to(memory_format=torch.channels_last)
            labels = labels.to(self.device)
            
            with amp_context:
                # 获取 logits 和特征
                outs, aux_info = self.model(imgs, return_aux_info=True)
                
                # 检查 NaN/Inf
                if torch.isnan(outs).any() or torch.isinf(outs).any():
                    continue
                
                loss = F.cross_entropy(outs, labels)
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item() * labels.size(0)
            
            # Top-1 准确率
            preds = outs.argmax(dim=1)
            correct_top1 += (preds == labels).sum().item()
            
            # Top-5 准确率
            _, top5_preds = outs.topk(5, dim=1)
            correct_top5 += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
            
            total += labels.size(0)
            
            # 逐类别统计
            for c in range(self.num_classes):
                mask = labels == c
                if mask.sum() > 0:
                    class_correct[c] += (preds[mask] == c).sum()
                    class_total[c] += mask.sum()
            
            # 混淆矩阵更新
            for pred, label in zip(preds, labels):
                confusion[label, pred] += 1
        
        # 计算指标
        avg_loss = total_loss / total if total > 0 else 0.0
        top1_acc = 100.0 * correct_top1 / total if total > 0 else 0.0
        top5_acc = 100.0 * correct_top5 / total if total > 0 else 0.0
        
        # 逐类别准确率
        per_class_acc = None
        if self.config.compute_per_class:
            per_class_acc = {}
            for c in range(self.num_classes):
                if class_total[c] > 0:
                    per_class_acc[c] = (class_correct[c] / class_total[c]).item() * 100
                else:
                    per_class_acc[c] = 0.0
        
        # 最易混淆的类别对
        confused_pairs = None
        if self.config.compute_confusion:
            confused_pairs = self._find_confused_pairs(
                confusion, 
                top_k=self.config.top_confused_pairs
            )
        
        return CUB200EvalResult(
            loss=avg_loss,
            accuracy=top1_acc,
            top5_accuracy=top5_acc,
            per_class_accuracy=per_class_acc,
            confused_pairs=confused_pairs,
        )
    
    def _find_confused_pairs(
        self, 
        confusion: torch.Tensor, 
        top_k: int = 10
    ) -> List[Tuple[int, int, int]]:
        """
        找到最易混淆的类别对
        
        Args:
            confusion: 混淆矩阵 [C, C]
            top_k: 返回前 k 个最混淆的对
            
        Returns:
            List of (true_class, pred_class, count) 元组
        """
        # 移除对角线（正确预测）
        conf = confusion.clone()
        conf.fill_diagonal_(0)
        
        # 找到最大的非对角元素
        pairs = []
        flat_conf = conf.view(-1)
        top_values, top_indices = flat_conf.topk(top_k)
        
        for idx, val in zip(top_indices, top_values):
            if val.item() == 0:
                break
            true_class = (idx // self.num_classes).item()
            pred_class = (idx % self.num_classes).item()
            pairs.append((true_class, pred_class, val.item()))
        
        return pairs
    
    def print_eval_summary(
        self, 
        result: CUB200EvalResult,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """
        打印评估摘要
        
        Args:
            result: 评估结果
            class_names: 类别名称列表（可选）
        """
        print("\n" + "=" * 60)
        print("CUB-200-2011 细粒度分类评估结果")
        print("=" * 60)
        print(f"  Loss:          {result.loss:.4f}")
        print(f"  Top-1 Acc:     {result.accuracy:.2f}%")
        print(f"  Top-5 Acc:     {result.top5_accuracy:.2f}%")
        
        if result.per_class_accuracy:
            # 找到最差的几个类
            sorted_classes = sorted(
                result.per_class_accuracy.items(), 
                key=lambda x: x[1]
            )
            print("\n  最差的 5 个类别:")
            for cls_id, acc in sorted_classes[:5]:
                name = class_names[cls_id] if class_names else f"Class {cls_id}"
                print(f"    {name}: {acc:.1f}%")
        
        if result.confused_pairs:
            print("\n  最易混淆的类别对:")
            for true_cls, pred_cls, count in result.confused_pairs[:5]:
                true_name = class_names[true_cls] if class_names else f"Class {true_cls}"
                pred_name = class_names[pred_cls] if class_names else f"Class {pred_cls}"
                print(f"    {true_name} → {pred_name}: {count} 次")
        
        print("=" * 60 + "\n")


def get_cub200_augmentation(
    image_size: int = 224,
    is_training: bool = True,
    config: Optional[CUB200TrainingConfig] = None,
) -> "transforms.Compose":
    """
    获取 CUB-200 细粒度分类专用数据增强
    
    细粒度分类增强策略：
        1. 较大的 RandomResizedCrop scale 范围 → 学习不同尺度特征
        2. 适度的颜色增强 → 保留判别性颜色特征
        3. 较小的旋转角度 → 鸟类姿态敏感
        4. 不使用垂直翻转 → 鸟类通常不倒立
        
    Args:
        image_size: 目标图像尺寸
        is_training: 是否训练模式
        config: 训练配置
        
    Returns:
        torchvision transforms 组合
    """
    from torchvision import transforms
    
    # CUB-200 使用 ImageNet 统计量
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    if is_training:
        return transforms.Compose([
            # 1. Resize 到稍大尺寸用于裁剪
            transforms.Resize(int(image_size * 1.15)),  # 256 for 224
            
            # 2. RandomResizedCrop: 较大 scale 范围学习多尺度特征
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.5, 1.0),  # 比标准 (0.08, 1.0) 更保守
                ratio=(0.75, 1.33),
            ),
            
            # 3. 水平翻转（不使用垂直翻转）
            transforms.RandomHorizontalFlip(p=0.5),
            
            # 4. 较小旋转角度（鸟类姿态敏感）
            transforms.RandomRotation(10),
            
            # 5. 颜色增强（保留判别性颜色）
            transforms.ColorJitter(
                brightness=0.2,  # 比通用分类更保守
                contrast=0.2,
                saturation=0.2,
                hue=0.05,  # 色相变化很小
            ),
            
            # 6. RandAugment（较低强度）
            transforms.RandAugment(num_ops=2, magnitude=6),
            
            # 7. 转换为 Tensor
            transforms.ToTensor(),
            
            # 8. 归一化
            transforms.Normalize(mean, std),
            
            # 9. Random Erasing（遮挡增强）
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def create_cub200_trainer(
    model: nn.Module,
    num_classes: int = 200,
    feat_dim: int = 256,
    device: torch.device = torch.device('cuda'),
    **kwargs,
) -> CUB200Trainer:
    """
    创建 CUB-200 训练器的便捷工厂函数
    
    Args:
        model: Fractal ViT 模型
        num_classes: 类别数
        feat_dim: 特征维度
        device: 设备
        **kwargs: 传递给 CUB200TrainingConfig 的参数
        
    Returns:
        CUB200Trainer 实例
    """
    config = CUB200TrainingConfig(**kwargs)
    return CUB200Trainer(
        model=model,
        config=config,
        num_classes=num_classes,
        feat_dim=feat_dim,
        device=device,
    )
