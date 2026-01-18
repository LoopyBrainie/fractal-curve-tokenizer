"""
Modular Trainer System for Fractal Curve ViT
=============================================

设计原则:
1. 模型无关 - 训练器不依赖特定模型实现
2. 组件可插拔 - Loss/Sampler/Metrics/Callbacks 均可替换
3. 配置驱动 - 所有超参数通过 Config 传递
4. 生命周期钩子 - Callback 系统支持自定义扩展

数学形式化:
-----------
训练循环的核心是最小化复合损失:
    L_total = L_task + Σ_i λ_i · L_aux_i

其中:
- L_task: 任务损失 (CE, Focal, CB-CE)
- L_aux: 辅助损失 (FLOPS Budget, Depth Penalty)

训练状态转换:
    State(t+1) = Update(State(t), Grad(L_total))
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Protocol, Union
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time
import json


# ============================================================================
# 协议定义 (Protocol Interfaces)
# ============================================================================

class ModelProtocol(Protocol):
    """模型协议 - 定义训练器期望的模型接口"""
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def parameters(self): ...
    def train(self, mode: bool = True) -> 'ModelProtocol': ...
    def eval(self) -> 'ModelProtocol': ...


class MetricsProtocol(Protocol):
    """指标协议 - 定义指标计算接口"""
    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None: ...
    def compute(self) -> Dict[str, float]: ...
    def reset(self) -> None: ...


class LossProtocol(Protocol):
    """损失函数协议"""
    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: ...


# ============================================================================
# 配置数据类
# ============================================================================

@dataclass
class TrainerConfig:
    """训练器核心配置
    
    数学参数:
    ---------
    - grad_clip_norm: 梯度裁剪阈值 ||g||_2 ≤ clip_norm
    - accumulation_steps: 有效批次 = batch_size × accumulation_steps
    """
    # 设备配置
    device: str = "cuda"
    
    # 训练参数
    num_epochs: int = 100
    gradient_clip_norm: Optional[float] = 1.0
    accumulation_steps: int = 1
    
    # 日志配置
    log_interval: int = 50  # 每 N 个 batch 记录一次
    validate_interval: int = 1  # 每 N 个 epoch 验证一次
    
    # 检查点配置
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    monitor_metric: str = "val_accuracy"  # 监控的指标
    monitor_mode: str = "max"  # "max" or "min"
    
    # 混合精度
    use_amp: bool = True
    
    # 调试选项
    debug: bool = False
    max_batches_per_epoch: Optional[int] = None  # 用于调试


@dataclass
class TrainerState:
    """训练状态 - 完整记录训练过程"""
    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0
    best_epoch: int = 0
    
    # 当前 epoch 统计
    train_loss: float = 0.0
    train_metrics: Dict[str, float] = field(default_factory=dict)
    val_loss: float = 0.0
    val_metrics: Dict[str, float] = field(default_factory=dict)
    
    # 历史记录
    history: Dict[str, List[float]] = field(default_factory=lambda: {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "learning_rate": [],
    })


# ============================================================================
# Callback 系统
# ============================================================================

@dataclass
class CallbackContext:
    """Callback 上下文 - 传递给每个回调的信息"""
    epoch: int
    batch_idx: int
    global_step: int
    
    # 可选数据
    loss: Optional[float] = None
    outputs: Optional[torch.Tensor] = None
    targets: Optional[torch.Tensor] = None
    metrics: Optional[Dict[str, float]] = None
    
    # 控制信号
    stop_training: bool = False


class Callback(ABC):
    """Callback 基类
    
    生命周期钩子:
    - on_train_begin/end: 训练开始/结束
    - on_epoch_begin/end: epoch 开始/结束
    - on_batch_begin/end: batch 开始/结束
    - on_validation_begin/end: 验证开始/结束
    """
    priority: int = 0  # 值越小越先执行
    
    def on_train_begin(self, trainer: 'ModularTrainer', state: TrainerState) -> None:
        pass
    
    def on_train_end(self, trainer: 'ModularTrainer', state: TrainerState) -> None:
        pass
    
    def on_epoch_begin(self, trainer: 'ModularTrainer', ctx: CallbackContext) -> None:
        pass
    
    def on_epoch_end(self, trainer: 'ModularTrainer', ctx: CallbackContext) -> None:
        pass
    
    def on_batch_begin(self, trainer: 'ModularTrainer', ctx: CallbackContext) -> None:
        pass
    
    def on_batch_end(self, trainer: 'ModularTrainer', ctx: CallbackContext) -> None:
        pass
    
    def on_validation_begin(self, trainer: 'ModularTrainer', ctx: CallbackContext) -> None:
        pass
    
    def on_validation_end(self, trainer: 'ModularTrainer', ctx: CallbackContext) -> None:
        pass


class CallbackList:
    """Callback 管理器"""
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = sorted(callbacks or [], key=lambda c: c.priority)
    
    def add(self, callback: Callback) -> None:
        self.callbacks.append(callback)
        self.callbacks.sort(key=lambda c: c.priority)
    
    def _run(self, method_name: str, *args, **kwargs) -> None:
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if method:
                method(*args, **kwargs)
    
    def on_train_begin(self, trainer: 'ModularTrainer', state: TrainerState) -> None:
        self._run('on_train_begin', trainer, state)
    
    def on_train_end(self, trainer: 'ModularTrainer', state: TrainerState) -> None:
        self._run('on_train_end', trainer, state)
    
    def on_epoch_begin(self, trainer: 'ModularTrainer', ctx: CallbackContext) -> None:
        self._run('on_epoch_begin', trainer, ctx)
    
    def on_epoch_end(self, trainer: 'ModularTrainer', ctx: CallbackContext) -> None:
        self._run('on_epoch_end', trainer, ctx)
    
    def on_batch_begin(self, trainer: 'ModularTrainer', ctx: CallbackContext) -> None:
        self._run('on_batch_begin', trainer, ctx)
    
    def on_batch_end(self, trainer: 'ModularTrainer', ctx: CallbackContext) -> None:
        self._run('on_batch_end', trainer, ctx)
    
    def on_validation_begin(self, trainer: 'ModularTrainer', ctx: CallbackContext) -> None:
        self._run('on_validation_begin', trainer, ctx)
    
    def on_validation_end(self, trainer: 'ModularTrainer', ctx: CallbackContext) -> None:
        self._run('on_validation_end', trainer, ctx)


# ============================================================================
# 内置 Callbacks
# ============================================================================

class EarlyStoppingCallback(Callback):
    """早停回调
    
    数学形式:
    如果连续 patience 个 epoch, metric 没有改善 delta 以上，则停止训练。
    
    改善定义:
    - mode="max": metric_new > metric_best + delta
    - mode="min": metric_new < metric_best - delta
    """
    priority = 100  # 在其他回调之后执行
    
    def __init__(
        self,
        monitor: str = "val_accuracy",
        patience: int = 10,
        delta: float = 1e-4,
        mode: str = "max",
    ):
        self.monitor = monitor
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_value = float("-inf") if mode == "max" else float("inf")
    
    def _is_improvement(self, current: float) -> bool:
        if self.mode == "max":
            return current > self.best_value + self.delta
        return current < self.best_value - self.delta
    
    def on_epoch_end(self, trainer: 'ModularTrainer', ctx: CallbackContext) -> None:
        if ctx.metrics is None:
            return
        
        current = ctx.metrics.get(self.monitor)
        if current is None:
            return
        
        if self._is_improvement(current):
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"[EarlyStopping] No improvement for {self.patience} epochs. Stopping.")
                ctx.stop_training = True


class CheckpointCallback(Callback):
    """检查点保存回调"""
    priority = 50
    
    def __init__(
        self,
        checkpoint_dir: str,
        monitor: str = "val_accuracy",
        mode: str = "max",
        save_best_only: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = float("-inf") if mode == "max" else float("inf")
    
    def _is_better(self, current: float) -> bool:
        if self.mode == "max":
            return current > self.best_value
        return current < self.best_value
    
    def on_epoch_end(self, trainer: 'ModularTrainer', ctx: CallbackContext) -> None:
        if ctx.metrics is None:
            return
        
        current = ctx.metrics.get(self.monitor, 0.0)
        
        # 总是保存最新
        if not self.save_best_only:
            path = self.checkpoint_dir / f"checkpoint_epoch_{ctx.epoch}.pt"
            trainer.save_checkpoint(path)
        
        # 保存最佳
        if self._is_better(current):
            self.best_value = current
            path = self.checkpoint_dir / "best_model.pt"
            trainer.save_checkpoint(path)
            print(f"[Checkpoint] New best {self.monitor}: {current:.4f}. Saved to {path}")


class LRSchedulerCallback(Callback):
    """学习率调度回调"""
    priority = 10
    
    def __init__(self, scheduler, step_on: str = "epoch"):
        """
        Args:
            scheduler: PyTorch LR scheduler
            step_on: "epoch" or "batch"
        """
        self.scheduler = scheduler
        self.step_on = step_on
    
    def on_epoch_end(self, trainer: 'ModularTrainer', ctx: CallbackContext) -> None:
        if self.step_on == "epoch":
            self.scheduler.step()
    
    def on_batch_end(self, trainer: 'ModularTrainer', ctx: CallbackContext) -> None:
        if self.step_on == "batch":
            self.scheduler.step()


class ProgressCallback(Callback):
    """进度显示回调"""
    priority = -10  # 最先执行
    
    def __init__(self, log_interval: int = 50):
        self.log_interval = log_interval
        self.epoch_start_time = 0.0
        self.batch_losses: List[float] = []
    
    def on_epoch_begin(self, trainer: 'ModularTrainer', ctx: CallbackContext) -> None:
        self.epoch_start_time = time.time()
        self.batch_losses = []
        print(f"\n{'='*60}")
        print(f"Epoch {ctx.epoch}/{trainer.config.num_epochs}")
        print(f"{'='*60}")
    
    def on_batch_end(self, trainer: 'ModularTrainer', ctx: CallbackContext) -> None:
        if ctx.loss is not None:
            self.batch_losses.append(ctx.loss)
        
        if (ctx.batch_idx + 1) % self.log_interval == 0:
            avg_loss = sum(self.batch_losses[-self.log_interval:]) / min(len(self.batch_losses), self.log_interval)
            lr = trainer.optimizer.param_groups[0]['lr']
            print(f"  Batch {ctx.batch_idx+1} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
    
    def on_epoch_end(self, trainer: 'ModularTrainer', ctx: CallbackContext) -> None:
        elapsed = time.time() - self.epoch_start_time
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in (ctx.metrics or {}).items())
        print(f"\n  Epoch completed in {elapsed:.1f}s")
        print(f"  Metrics: {metrics_str}")


# ============================================================================
# 模块化训练器
# ============================================================================

class ModularTrainer:
    """模块化训练器
    
    设计特点:
    1. 依赖注入 - 所有组件通过构造函数传入
    2. 生命周期钩子 - Callback 系统
    3. 状态分离 - TrainerState 独立于训练逻辑
    4. 设备无关 - 自动处理 CPU/GPU
    
    典型用法:
    ```python
    trainer = ModularTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=FocalLoss(gamma=2.0),
        metrics=ClassificationMetrics(num_classes=200),
        callbacks=[EarlyStoppingCallback(patience=10)],
        config=TrainerConfig(num_epochs=100),
    )
    history = trainer.fit()
    ```
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: LossProtocol,
        metrics: Optional[MetricsProtocol] = None,
        aux_losses: Optional[Dict[str, Callable]] = None,  # 辅助损失 {name: (fn, lambda)}
        callbacks: Optional[List[Callback]] = None,
        config: Optional[TrainerConfig] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.aux_losses = aux_losses or {}
        self.callbacks = CallbackList(callbacks)
        self.config = config or TrainerConfig()
        
        # 状态
        self.state = TrainerState()
        
        # 设备
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        
        # 混合精度
        self.scaler = torch.amp.GradScaler('cuda') if self.config.use_amp else None
    
    def train_epoch(self) -> Dict[str, float]:
        """单轮训练
        
        Returns:
            训练指标字典
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        if self.metrics:
            self.metrics.reset()
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # 调试模式限制 batch 数
            if self.config.max_batches_per_epoch and batch_idx >= self.config.max_batches_per_epoch:
                break
            
            # Callback: batch begin
            ctx = CallbackContext(
                epoch=self.state.epoch,
                batch_idx=batch_idx,
                global_step=self.state.global_step,
            )
            self.callbacks.on_batch_begin(self, ctx)
            
            # 前向传播
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=self.config.use_amp):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                
                # 辅助损失
                for name, (aux_fn, weight) in self.aux_losses.items():
                    aux_loss = aux_fn(outputs, targets)
                    loss = loss + weight * aux_loss
                
                # 梯度累积
                loss = loss / self.config.accumulation_steps
            
            # 反向传播
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度更新 (考虑累积)
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                if self.config.gradient_clip_norm:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.state.global_step += 1
            
            # 统计 (I78: 使用 detach().item() 支持 torch.compile)
            batch_loss = loss.detach().item() * self.config.accumulation_steps
            total_loss += batch_loss
            num_batches += 1
            
            # 更新指标
            if self.metrics:
                self.metrics.update(outputs.detach(), targets)
            
            # Callback: batch end
            ctx.loss = batch_loss
            ctx.outputs = outputs
            ctx.targets = targets
            self.callbacks.on_batch_end(self, ctx)
            
            if ctx.stop_training:
                break
        
        # 计算 epoch 指标
        result = {"train_loss": total_loss / max(num_batches, 1)}
        if self.metrics:
            metrics_result = self.metrics.compute()
            result.update({f"train_{k}": v for k, v in metrics_result.items()})
        
        self.state.train_loss = result["train_loss"]
        self.state.train_metrics = result
        
        return result
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证
        
        Returns:
            验证指标字典
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        if self.metrics:
            self.metrics.reset()
        
        # Callback: validation begin
        ctx = CallbackContext(
            epoch=self.state.epoch,
            batch_idx=0,
            global_step=self.state.global_step,
        )
        self.callbacks.on_validation_begin(self, ctx)
        
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # 验证禁用 AMP 以确保指标精度 (I78: 使用 detach().item() 支持 torch.compile)
            with torch.amp.autocast('cuda', enabled=False):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

            total_loss += loss.detach().item()
            num_batches += 1
            
            if self.metrics:
                self.metrics.update(outputs, targets)
        
        # 计算指标
        result = {"val_loss": total_loss / max(num_batches, 1)}
        if self.metrics:
            metrics_result = self.metrics.compute()
            result.update({f"val_{k}": v for k, v in metrics_result.items()})
        
        self.state.val_loss = result["val_loss"]
        self.state.val_metrics = result
        
        # Callback: validation end
        ctx.metrics = result
        self.callbacks.on_validation_end(self, ctx)
        
        return result
    
    def fit(self) -> Dict[str, List[float]]:
        """完整训练流程
        
        Returns:
            训练历史记录
        """
        # Callback: train begin
        self.callbacks.on_train_begin(self, self.state)
        
        for epoch in range(1, self.config.num_epochs + 1):
            self.state.epoch = epoch
            
            # Callback: epoch begin
            ctx = CallbackContext(
                epoch=epoch,
                batch_idx=0,
                global_step=self.state.global_step,
            )
            self.callbacks.on_epoch_begin(self, ctx)
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = {}
            if epoch % self.config.validate_interval == 0:
                val_metrics = self.validate()
            
            # 更新历史
            all_metrics = {**train_metrics, **val_metrics}
            for key in self.state.history:
                if key in all_metrics:
                    self.state.history[key].append(all_metrics[key])
            
            # 更新 learning rate 历史
            lr = self.optimizer.param_groups[0]['lr']
            self.state.history["learning_rate"].append(lr)
            
            # 更新 best metric
            monitor_value = all_metrics.get(self.config.monitor_metric, 0.0)
            is_best = False
            if self.config.monitor_mode == "max":
                is_best = monitor_value > self.state.best_metric
            else:
                is_best = monitor_value < self.state.best_metric
            
            if is_best:
                self.state.best_metric = monitor_value
                self.state.best_epoch = epoch
            
            # Callback: epoch end
            ctx.metrics = all_metrics
            self.callbacks.on_epoch_end(self, ctx)
            
            if ctx.stop_training:
                print(f"\n[Trainer] Training stopped at epoch {epoch}")
                break
        
        # Callback: train end
        self.callbacks.on_train_end(self, self.state)
        
        return self.state.history
    
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """保存检查点"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state": {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "best_metric": self.state.best_metric,
                "best_epoch": self.state.best_epoch,
            },
            "config": {
                "num_epochs": self.config.num_epochs,
                "gradient_clip_norm": self.config.gradient_clip_norm,
            },
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        state_dict = checkpoint.get("state", {})
        self.state.epoch = state_dict.get("epoch", 0)
        self.state.global_step = state_dict.get("global_step", 0)
        self.state.best_metric = state_dict.get("best_metric", 0.0)
        self.state.best_epoch = state_dict.get("best_epoch", 0)
        
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # 配置
    "TrainerConfig",
    "TrainerState",
    # Callbacks
    "Callback",
    "CallbackContext",
    "CallbackList",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "LRSchedulerCallback",
    "ProgressCallback",
    # 训练器
    "ModularTrainer",
    # CUB-200 细粒度分类训练器
    "CUB200Trainer",
    "CUB200TrainingConfig",
    "CUB200EvalResult",
    "create_cub200_trainer",
    "get_cub200_augmentation",
]

# 导入 CUB-200 细粒度训练器
from .cub200_trainer import (
    CUB200Trainer,
    CUB200TrainingConfig,
    CUB200EvalResult,
    create_cub200_trainer,
    get_cub200_augmentation,
)
