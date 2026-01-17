# ============================================================================
# CUB-200-2011 细粒度分类专用训练器 (独立实现)
# Fine-grained Classification Trainer for CUB-200-2011
# ============================================================================
"""
CUB-200-2011 数据集的专用训练器 - 完全独立实现。

设计原则:
1. 完全解耦 - 不依赖 ModularTrainer 或其他通用训练组件
2. 完整训练循环 - 独立管理训练、验证、检查点、早停
3. 数学形式化 - 所有参数有严格数学推导

数据集特性:
    - 200 类鸟类物种（细粒度）
    - 5,994 训练样本 / 5,794 测试样本
    - 每类约 30 样本（样本稀缺需强正则化）
    - 类间差异微小（同属不同种）
    - 关键判别特征：喙、眼、羽毛纹理

数学形式化:
    Center Loss: L_center = (1/2m) Σ_i ||f_i - c_{y_i}||²
    中心梯度: ∂L/∂c_j = (1/m_j) Σ_{i:y_i=j}(c_j - f_i)
    学习率关系: lr_center = lr_main × ratio (论文推荐 ratio=50)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

from ..losses.finegrained import FinegrainedLoss, FinegrainedLossConfig

logger = logging.getLogger(__name__)


@dataclass
class CUB200TrainingConfig:
    """CUB-200 细粒度分类训练配置

    数学参数推导:
        1. 主学习率: lr = lr_base × (batch_eff / 256)
        2. Center Loss 学习率: lr_center = lr_main × center_lr_ratio
        3. 正则化: scale = √(N_ref / N) ≈ 1.5-2.0

    Attributes:
        batch_size: 批次大小（有效批次 = batch_size × accum_steps）
        num_epochs: 训练轮数
        learning_rate: 主模型学习率
        warmup_epochs: 学习率预热轮数
        accum_steps: 梯度累积步数
        validate_interval: 验证间隔（每 N 个 epoch）
        center_lr_ratio: Center Loss 学习率与主学习率的比率（论文推荐 50）
        center_lr: Center Loss 绝对学习率（若设置则优先使用）
    """
    # 基础训练配置
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 2.7e-4
    warmup_epochs: int = 7
    accum_steps: int = 3
    validate_interval: int = 3  # 每 N 个 epoch 验证一次

    # Center Loss 学习率参数
    center_lr_ratio: float = 50.0  # lr_center / lr_main，论文推荐 50
    center_lr: Optional[float] = None  # 绝对学习率，优先于 ratio

    # 设备和混合精度
    device: str = "cuda"
    use_amp: bool = True
    gradient_clip_norm: Optional[float] = 1.0

    # 细粒度专用配置
    use_center_loss: bool = True
    center_loss_weight: float = 0.01

    # 正则化（比通用分类更强）
    label_smoothing: float = 0.13  # 0.1 × (1 + log₁₀(200/100))
    dropout: float = 0.22
    drop_path: float = 0.16
    weight_decay: float = 0.15

    # Mixup/CutMix 数据增强
    mixup_alpha: float = 0.3
    cutmix_alpha: float = 0.8
    mixup_prob: float = 0.4

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

    # 日志
    log_level: str = "INFO"

    def __post_init__(self):
        """参数验证 - 数学约束"""
        assert 0 < self.batch_size <= 512, f"batch_size={self.batch_size} 必须在 (0, 512]"
        assert 1 <= self.num_epochs <= 1000, f"num_epochs={self.num_epochs} 必须在 [1, 1000]"
        assert self.learning_rate > 0, f"learning_rate={self.learning_rate} 必须 > 0"
        assert 0 <= self.warmup_epochs <= self.num_epochs // 2, \
            f"warmup_epochs={self.warmup_epochs} 必须 <= num_epochs/2"
        assert 1 <= self.accum_steps <= 32, f"accum_steps={self.accum_steps} 必须在 [1, 32]"
        assert 1 <= self.validate_interval <= self.num_epochs, \
            f"validate_interval={self.validate_interval} 无效"
        assert 0 <= self.label_smoothing < 1.0, f"label_smoothing={self.label_smoothing} 必须在 [0, 1)"
        assert 0 <= self.dropout < 1.0, f"dropout={self.dropout} 必须在 [0, 1)"
        assert 0 <= self.drop_path < 1.0, f"drop_path={self.drop_path} 必须在 [0, 1)"
        assert self.center_loss_weight >= 0, f"center_loss_weight={self.center_loss_weight} 必须 >= 0"
        assert 0 < self.center_lr_ratio <= 1000, f"center_lr_ratio={self.center_lr_ratio} 必须在 (0, 1000]"

        # 日志级别
        self.log_level = getattr(logging, self.log_level.upper(), logging.INFO)


@dataclass
class CUB200EvalResult:
    """CUB-200 评估结果"""
    loss: float
    accuracy: float  # Top-1 准确率
    top5_accuracy: float
    per_class_accuracy: Optional[Dict[int, float]] = None
    confused_pairs: Optional[List[Tuple[int, int, int]]] = None  # (class_a, class_b, count)
    feature_stats: Optional[Dict[str, float]] = None
    mca: Optional[float] = None  # Mean Class Accuracy
    intra_inter_ratio: Optional[float] = None  # 类内/类间距离比


@dataclass
class CUB200TrainerState:
    """训练状态 - 完整记录训练过程"""
    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0
    best_epoch: int = 0
    patience_counter: int = 0

    # 当前 epoch 统计
    train_loss: float = 0.0
    train_accuracy: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0

    # 历史记录
    history: Dict[str, List[float]] = field(default_factory=lambda: {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_top5_accuracy": [],
        "learning_rate": [],
        "center_loss": [],
    })


class CUB200Trainer:
    """CUB-200-2011 细粒度分类训练器 - 独立实现

    数学形式化:
        总损失: L_total = L_ce + λ_center × L_center

        梯度累积:
            effective_loss = L_total / accum_steps
            grad = Σ_{i=1}^{accum_steps} ∂L_i/∂θ

        学习率预热:
            lr(epoch) = lr_base × min(epoch / warmup_epochs, 1.0)

    特征:
        1. 完整训练循环 - 独立管理所有训练逻辑
        2. 正确梯度累积 - Center Loss 与主模型同步更新
        3. 验证禁用 AMP - 确保指标精度
        4. 完整检查点 - 包含 scaler、center optimizer 状态
        5. 学习率预热 - 线性预热策略
    """

    def __init__(
        self,
        model: nn.Module,
        config: CUB200TrainingConfig,
        num_classes: int = 200,
        feat_dim: int = 256,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: Fractal ViT 模型
            config: 训练配置
            num_classes: 类别数（CUB-200 = 200）
            feat_dim: 特征维度
            device: 计算设备
        """
        self.model = model
        self.config = config
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        # 设备处理
        if device is None:
            self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)

        # 日志
        logging.basicConfig(level=config.log_level)
        self.logger = logging.getLogger(__name__)

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
        self.loss_fn = FinegrainedLoss(loss_config).to(self.device)

        # Center Loss 优化器（延迟初始化，需要主优化器学习率）
        self.center_optimizer: Optional[torch.optim.Optimizer] = None
        self._center_lr: float = 0.0

        # 训练状态
        self.state = CUB200TrainerState()

        # 混合精度 scaler
        self.scaler: Optional[GradScaler] = GradScaler('cuda', enabled=config.use_amp)

        self.logger.info(f"CUB200Trainer 初始化完成: device={self.device}, num_classes={num_classes}")

    def initialize_center_optimizer(self, main_optimizer: torch.optim.Optimizer) -> None:
        """初始化 Center Loss 优化器

        数学推导:
            论文推荐 lr_center = 0.5，当 lr_main = 0.01
            比率: ratio = lr_center / lr_main = 50

            对于 Fractal ViT: lr_main = config.learning_rate
            lr_center = lr_main × ratio (或使用绝对 center_lr)

        Args:
            main_optimizer: 主模型优化器
        """
        if not self.config.use_center_loss or self.loss_fn.center_loss is None:
            self.logger.info("Center Loss 未启用，跳过初始化")
            return

        # 计算 Center Loss 学习率
        base_lr = main_optimizer.param_groups[0]['lr']

        if self.config.center_lr is not None:
            # 绝对学习率
            self._center_lr = self.config.center_lr
            self.logger.info(f"使用绝对 Center Loss 学习率: {self._center_lr}")
        else:
            # 相对学习率 = lr_main × ratio
            self._center_lr = base_lr * self.config.center_lr_ratio
            self.logger.info(f"计算 Center Loss 学习率: {base_lr} × {self.config.center_lr_ratio} = {self._center_lr}")

        # 创建优化器
        self.center_optimizer = torch.optim.SGD(
            self.loss_fn.center_loss.parameters(),
            lr=self._center_lr,
            momentum=0.9,  # Center Loss 论文推荐动量
        )

        self.logger.info(f"Center Loss 优化器初始化完成: lr={self._center_lr}")

    def _get_warmup_lr(self, epoch: int) -> float:
        """计算学习率（包含预热）

        数学公式:
            lr(epoch) = lr_base × min(epoch / warmup_epochs, 1.0)

        Args:
            epoch: 当前 epoch（从 1 开始）

        Returns:
            当前学习率
        """
        if epoch <= self.config.warmup_epochs:
            warmup_factor = epoch / max(1, self.config.warmup_epochs)
            return self.config.learning_rate * warmup_factor
        return self.config.learning_rate

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算细粒度分类损失

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

    def train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, float, Dict[str, float]]:
        """训练一个 epoch

        数学形式化:
            梯度累积:
                for batch i in accum_steps:
                    loss_i = L(x_i, y_i) / accum_steps
                    loss_i.backward()

                optimizer.step()  # 每 accum_steps 步更新一次

        Args:
            loader: 数据加载器
            optimizer: 主模型优化器

        Returns:
            avg_loss: 平均损失
            accuracy: 训练准确率
            stats: 统计信息
        """
        self.model.train()

        total_loss = 0.0
        total_center_loss = 0.0
        correct = 0
        total = 0

        accum_steps = self.config.accum_steps
        grad_norm = 0.0

        for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc="Training", leave=False)):
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # 累积归一化
            scale_factor = 1.0 / accum_steps

            with autocast(device_type=self.device.type, enabled=self.config.use_amp):
                # 获取 logits
                logits = self.model(imgs)

                # 计算损失
                loss, stats = self.compute_loss(logits, labels)

                # 累积归一化
                loss = loss * scale_factor

            # 反向传播
            self.scaler.scale(loss).backward()

            # 梯度累积步数检查
            if (batch_idx + 1) % accum_steps == 0:
                # 梯度裁剪
                if self.config.gradient_clip_norm is not None:
                    self.scaler.unscale_(optimizer)
                    grad_norm_curr = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                    grad_norm = grad_norm_curr.item()

                # 更新主模型
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # 更新 Center Loss 优化器
                if self.center_optimizer is not None:
                    self.center_optimizer.step()
                    self.center_optimizer.zero_grad()

                self.state.global_step += 1

            # 统计（恢复原始损失值）
            total_loss += loss.item() / scale_factor
            total_center_loss += stats.get('center_loss', 0.0) * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = 100.0 * correct / max(total, 1)
        avg_center_loss = total_center_loss / max(total, 1)

        stats = {
            'center_loss': avg_center_loss,
            'grad_norm': grad_norm,
        }

        return avg_loss, accuracy, stats

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        return_features: bool = False,
    ) -> CUB200EvalResult:
        """评估模型

        注意: 验证阶段禁用 AMP 以确保指标精度

        计算指标:
            - Top-1 / Top-5 准确率
            - 逐类别准确率 (MCA)
            - 混淆矩阵分析
            - 类内/类间距离比

        Args:
            loader: 数据加载器
            return_features: 是否返回特征用于分析

        Returns:
            CUB200EvalResult 包含详细评估结果
        """
        self.model.eval()

        # 修复: 评估时禁用 persistent_workers 以避免多进程兼容性问题
        # 使用安全的数据加载方式
        try:
            # 尝试获取数据集
            eval_dataset = loader.dataset
            # 创建评估用的 DataLoader (禁用多进程)
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.config.batch_size,
                num_workers=0,  # 评估时禁用多进程
                pin_memory=True,
                shuffle=False,
                drop_last=False,
            )
            loader = eval_loader
        except Exception as e:
            self.logger.warning(f"无法创建安全的评估 DataLoader: {e}，使用原始 loader")
            # 如果失败，保持原样

        total_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        # 逐类别统计
        class_correct = torch.zeros(self.num_classes, device=self.device)
        class_total = torch.zeros(self.num_classes, device=self.device)

        # 混淆矩阵
        confusion = torch.zeros(
            self.num_classes, self.num_classes,
            device=self.device, dtype=torch.long
        )

        # 特征收集
        all_features = []
        all_labels = []

        for batch in tqdm(loader, desc="Evaluating", leave=False):
            imgs, labels = batch
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # 验证禁用 AMP
            with autocast(device_type=self.device.type, enabled=False):
                outs = self.model(imgs)

                # 检查 NaN/Inf
                if torch.isnan(outs).any() or torch.isinf(outs).any():
                    self.logger.warning("检测到 NaN/Inf，跳过该批次")
                    continue

                loss = F.cross_entropy(outs, labels)

                # 收集特征
                if return_features and hasattr(self.model, 'get_features'):
                    features = self.model.get_features(imgs)
                    all_features.append(features.cpu())
                    all_labels.append(labels.cpu())

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

            # 混淆矩阵
            for pred, label in zip(preds, labels):
                confusion[label, pred] += 1

        # 计算指标
        avg_loss = total_loss / max(total, 1)
        top1_acc = 100.0 * correct_top1 / max(total, 1)
        top5_acc = 100.0 * correct_top5 / max(total, 1)

        # 逐类别准确率
        per_class_acc = None
        mca = None
        if self.config.compute_per_class:
            per_class_acc = {}
            valid_classes = 0
            acc_sum = 0.0
            for c in range(self.num_classes):
                if class_total[c] > 0:
                    acc = (class_correct[c] / class_total[c]).item() * 100
                    per_class_acc[c] = acc
                    acc_sum += acc
                    valid_classes += 1
                else:
                    per_class_acc[c] = 0.0
            mca = acc_sum / max(valid_classes, 1)

        # 混淆对
        confused_pairs = None
        if self.config.compute_confusion:
            confused_pairs = self._find_confused_pairs(confusion)

        # 类内/类间距离
        intra_inter_ratio = None
        if return_features and len(all_features) > 0:
            features = torch.cat(all_features, dim=0)
            labels = torch.cat(all_labels, dim=0)
            intra_inter_ratio = self._compute_intra_inter_ratio(features, labels)

        return CUB200EvalResult(
            loss=avg_loss,
            accuracy=top1_acc,
            top5_accuracy=top5_acc,
            per_class_accuracy=per_class_acc,
            confused_pairs=confused_pairs,
            mca=mca,
            intra_inter_ratio=intra_inter_ratio,
        )

    def _find_confused_pairs(
        self,
        confusion: torch.Tensor,
        top_k: int = 10,
    ) -> List[Tuple[int, int, int]]:
        """找到最易混淆的类别对

        Args:
            confusion: 混淆矩阵 [C, C]
            top_k: 返回前 k 个

        Returns:
            List of (true_class, pred_class, count)
        """
        conf = confusion.clone()
        conf.fill_diagonal_(0)  # 移除对角线

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

    def _compute_intra_inter_ratio(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """计算类内/类间距离比

        数学公式:
            intra = (1/N) Σ_i ||f_i - c_{y_i}||²  (类内距离)
            inter = (1/C(C-1)/2) Σ_{j<k} ||c_j - c_k||²  (类间距离)

        Args:
            features: 特征 [N, D]
            labels: 标签 [N]

        Returns:
            intra/inter 比率（越小越好）
        """
        unique_labels = labels.unique()
        num_classes = len(unique_labels)
        feat_dim = features.shape[1]

        # 计算类中心
        centers = torch.zeros(num_classes, feat_dim, device=features.device)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            centers[i] = features[mask].mean(dim=0)

        # 类内距离
        intra_dist = 0.0
        for i, label in enumerate(unique_labels):
            mask = labels == label
            class_features = features[mask]
            center = centers[i]
            intra_dist += ((class_features - center) ** 2).sum(dim=1).mean()

        intra_dist = intra_dist / num_classes

        # 类间距离
        inter_dist = 0.0
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                inter_dist += ((centers[i] - centers[j]) ** 2).sum()

        inter_dist = inter_dist / (num_classes * (num_classes - 1) / 2)

        # 返回比率
        return intra_dist / max(inter_dist, 1e-8)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        exp_dir: Optional[Path] = None,
    ) -> Dict[str, List[float]]:
        """完整训练流程

        数学形式化:
            for epoch in 1..num_epochs:
                lr = get_warmup_lr(epoch)
                set_optimizer_lr(optimizer, lr)

                train_loss, train_acc = train_epoch()

                if epoch % validate_interval == 0:
                    val_loss, val_acc = evaluate()

                    if acc > best_acc + min_delta:
                        save_checkpoint()
                        best_acc = acc
                        patience_counter = 0
                    else:
                        patience_counter += 1

                if patience_counter >= patience:
                    break

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 主模型优化器
            scheduler: 学习率调度器（可选）
            exp_dir: 实验目录（可选）

        Returns:
            训练历史记录
        """
        # 初始化 Center Loss 优化器
        self.initialize_center_optimizer(optimizer)

        # 创建检查点目录
        checkpoint_dir = None
        if exp_dir is not None:
            checkpoint_dir = exp_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"检查点目录: {checkpoint_dir}")

        # 记录初始学习率
        initial_lr = optimizer.param_groups[0]['lr']

        for epoch in range(1, self.config.num_epochs + 1):
            self.state.epoch = epoch
            epoch_start = time.time()

            # 学习率预热
            warmup_lr = self._get_warmup_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

            # 训练
            train_loss, train_acc, train_stats = self.train_epoch(train_loader, optimizer)

            # 验证（每 N 个 epoch）
            val_result = None
            if epoch % self.config.validate_interval == 0:
                val_result = self.evaluate(val_loader)

                # 更新最佳指标
                if val_result.accuracy > self.state.best_metric + self.config.min_delta:
                    self.state.best_metric = val_result.accuracy
                    self.state.best_epoch = epoch
                    self.state.patience_counter = 0

                    # 保存最佳检查点
                    if checkpoint_dir is not None:
                        self._save_checkpoint(
                            checkpoint_dir / "best.pth",
                            optimizer,
                            epoch,
                            val_result.accuracy,
                        )
                        self.logger.info(f"保存最佳模型: val_acc={val_result.accuracy:.2f}%")
                else:
                    self.state.patience_counter += 1

                self.state.val_loss = val_result.loss
                self.state.val_accuracy = val_result.accuracy
            else:
                self.state.patience_counter += 1

            # 学习率调度
            if scheduler is not None:
                scheduler.step()

            epoch_time = time.time() - epoch_start

            # 更新历史
            self.state.history["train_loss"].append(train_loss)
            self.state.history["train_accuracy"].append(train_acc)
            self.state.history["learning_rate"].append(warmup_lr)
            self.state.history["center_loss"].append(train_stats.get('center_loss', 0.0))

            if val_result is not None:
                self.state.history["val_loss"].append(val_result.loss)
                self.state.history["val_accuracy"].append(val_result.accuracy)
                self.state.history["val_top5_accuracy"].append(val_result.top5_accuracy)

            # 日志输出
            current_lr = optimizer.param_groups[0]['lr']
            msg = (f"Epoch {epoch}/{self.config.num_epochs} | "
                   f"Time: {epoch_time:.1f}s | "
                   f"LR: {current_lr:.2e} | "
                   f"Train: loss={train_loss:.4f}, acc={train_acc:.2f}%")

            if val_result is not None:
                msg += f" | Val: loss={val_result.loss:.4f}, acc={val_result.accuracy:.2f}%, top5={val_result.top5_accuracy:.2f}%"

            self.logger.info(msg)

            # 早停检查
            if self.state.patience_counter >= self.config.patience:
                self.logger.info(f"早停触发: 连续 {self.config.patience} 个 epoch 无改善")
                break

            # 保存周期检查点
            if checkpoint_dir is not None and epoch % 10 == 0:
                self._save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch}.pth",
                    optimizer,
                    epoch,
                    val_result.accuracy if val_result else 0.0,
                )

        # 保存最终检查点
        if checkpoint_dir is not None:
            self._save_checkpoint(
                checkpoint_dir / "last.pth",
                optimizer,
                self.state.epoch,
                self.state.best_metric,
            )

        self.logger.info(f"训练完成! 最佳准确率: {self.state.best_metric:.2f}% (epoch {self.state.best_epoch})")

        return self.state.history

    def _save_checkpoint(
        self,
        path: Path,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_acc: float,
    ) -> None:
        """保存检查点

        包含完整状态:
            - 模型权重
            - 优化器状态
            - AMP scaler 状态
            - Center Loss 优化器状态
            - 训练配置
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'val_acc': val_acc,
            'best_metric': self.state.best_metric,
            'best_epoch': self.state.best_epoch,
            # 完整的模型架构配置（用于评估时正确重建模型）
            'config': {
                # 训练超参数
                'batch_size': self.config.batch_size,
                'num_epochs': self.config.num_epochs,
                'learning_rate': self.config.learning_rate,
                'accum_steps': self.config.accum_steps,
                'center_lr_ratio': self.config.center_lr_ratio,
                'center_lr': self.config.center_lr,
                'label_smoothing': self.config.label_smoothing,
                'dropout': self.config.dropout,
                'patience': self.config.patience,
                'use_amp': self.config.use_amp,
                'validate_interval': self.config.validate_interval,
                'gradient_clip_norm': self.config.gradient_clip_norm,
                # 模型架构参数
                'num_classes': self.config.num_classes,
                'dim': self.config.dim,
                'depth': self.config.depth,
                'heads': self.config.heads,
                'mlp_dim': self.config.mlp_dim,
                'dim_head': self.config.dim_head,
                'drop_path_rate': self.config.drop_path_rate,
                # Tokenizer 参数
                'num_scales': self.config.num_scales,
                'min_patch_size': self.config.min_patch_size,
                'max_level': self.config.max_level,
                'K_min': self.config.K_min,
                'K_max': self.config.K_max,
                'tokenizer_type': self.config.tokenizer_type,
                'ffn_type': self.config.ffn_type,
                'use_checkpoint': self.config.use_checkpoint,
                'channels': self.config.channels,
            },
        }

        # Center Loss 优化器状态
        if self.center_optimizer is not None:
            checkpoint['center_optimizer_state_dict'] = self.center_optimizer.state_dict()
            checkpoint['center_lr'] = self._center_lr

        torch.save(checkpoint, path)
        self.logger.debug(f"检查点已保存: {path}")

    def load_checkpoint(
        self,
        path: Path,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[int, float]:
        """加载检查点

        Args:
            path: 检查点路径
            optimizer: 优化器（用于恢复状态）

        Returns:
            epoch: 恢复的 epoch
            best_metric: 最佳准确率
        """
        checkpoint = torch.load(path, map_location=self.device)

        # 恢复模型
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 恢复优化器
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 恢复 scaler
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # 恢复 Center Loss 优化器
        if self.center_optimizer is not None and 'center_optimizer_state_dict' in checkpoint:
            self.center_optimizer.load_state_dict(checkpoint['center_optimizer_state_dict'])

        # 恢复状态
        epoch = checkpoint.get('epoch', 0)
        best_metric = checkpoint.get('best_metric', 0.0)
        best_epoch = checkpoint.get('best_epoch', 0)

        self.state.epoch = epoch
        self.state.best_metric = best_metric
        self.state.best_epoch = best_epoch

        self.logger.info(f"恢复检查点: epoch={epoch}, best_acc={best_metric:.2f}%")

        return epoch, best_metric

    def print_eval_summary(
        self,
        result: CUB200EvalResult,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """打印评估摘要"""
        print("\n" + "=" * 60)
        print("CUB-200-2011 细粒度分类评估结果")
        print("=" * 60)
        print(f"  Loss:          {result.loss:.4f}")
        print(f"  Top-1 Acc:     {result.accuracy:.2f}%")
        print(f"  Top-5 Acc:     {result.top5_accuracy:.2f}%")
        if result.mca is not None:
            print(f"  MCA:           {result.mca:.2f}%")
        if result.intra_inter_ratio is not None:
            print(f"  Intra/Inter:   {result.intra_inter_ratio:.4f}")

        if result.per_class_accuracy:
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
    """获取 CUB-200 细粒度分类专用数据增强

    细粒度分类增强策略:
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

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if is_training:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.5, 1.0),
                ratio=(0.75, 1.33),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.RandAugment(num_ops=2, magnitude=6),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
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
    device: Optional[torch.device] = None,
    **kwargs,
) -> CUB200Trainer:
    """创建 CUB-200 训练器的便捷工厂函数

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
