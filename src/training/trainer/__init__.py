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
from typing import Dict, Any, Optional, List, Callable, Protocol, Union, Tuple
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time
import json
import warnings

# I36: 导入 BaseTokenizer 用于 FractalModelProtocol
try:
    from vit_pytorch.tokenizer_streaming import BaseTokenizer
    from vit_pytorch.model_fractal_vit import TrainingStats
except ImportError:
    BaseTokenizer = None  # type: ignore
    TrainingStats = None  # type: ignore

# 延迟导入 ModelArchitectureConfig（避免循环导入）
def _get_model_arch_config():
    from ..config import ModelArchitectureConfig
    return ModelArchitectureConfig

ModelArchitectureConfig = None  # type: ignore


# ============================================================================
# 协议定义 (Protocol Interfaces)
# ============================================================================

class ModelProtocol(Protocol):
    """模型协议 - 定义训练器期望的模型接口"""
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def parameters(self): ...
    def train(self, mode: bool = True) -> 'ModelProtocol': ...
    def eval(self) -> 'ModelProtocol': ...


# I36: Fractal ViT 模型扩展协议
class FractalModelProtocol(ModelProtocol):
    """Fractal ViT 模型协议 - 扩展基础模型协议

    设计原则: 模型定义"我能做什么"，训练器决定"我怎么用你"

    数学形式化:
        - forward() 返回 TrainingStats 用于损失计算和监控
        - configure_training() 接收训练配置，保持模型内部逻辑独立
        - get_splitter_diagnostics() 返回分割器诊断信息
    """
    # 前向传播返回 TrainingStats (单一接口)
    def forward(self, img: torch.Tensor) -> TrainingStats: ...

    def configure_training(self, config: Dict[str, Any]) -> None:
        """配置训练相关参数

        Args:
            config: 配置字典，包含:
                - temperature_annealing: bool - 是否启用温度退火
                - total_steps: int - 总训练步数（用于温度退火调度）
                - temp_start: float - 起始温度
                - temp_end: float - 结束温度
                - aux_loss_weights: Dict[str, float] - 辅助损失权重
        """
        ...

    def get_splitter_diagnostics(self) -> Dict[str, Any]:
        """获取分割器诊断信息

        Returns:
            诊断字典，包含:
                - current_temperature: float - 当前温度
                - depth_distribution: Dict[int, float] - 深度分布
                - quota_allocation: List[float] - 配额分配
                - num_selected: int - 选中的 token 数
        """
        ...

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型诊断信息 (I36-7)

        Returns:
            完整诊断信息:
                - architecture: Dict - 架构参数
                - tokenizer: Dict - tokenizer 配置
                - splitter: Dict - splitter 参数
                - total_params: int - 总参数量
                - trainable_params: int - 可训练参数量
        """
        ...


@dataclass
class ExtraInfoProtocol:
    """辅助信息协议 - 规范 extra_info 的键名约定

    警告: 此协议仅用于文档目的，实际返回为字典
    """
    num_tokens: int                      # Token 数量
    levels_used: List[int]               # 使用的深度列表
    splitter_diagnostics: Dict[str, Any] # 分割器诊断信息
    token_selection_entropy: float       # 分割不确定性
    depth_distribution: Dict[int, float] # 深度分布


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
        arch_config: Optional["ModelArchitectureConfig"] = None,  # 模型架构配置（用于构造 ModelGene）
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
        self.arch_config = arch_config  # 用于构造 ModelGene
        
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
        # CRIT-4: 使用 GPU 张量累积，避免 .item() 同步点
        # 累积公式: L_epoch = sum(L_i * batch_size) / sum(batch_size)
        loss_sum = torch.zeros(1, device=self.device)
        num_samples = 0

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

            batch_size = targets.size(0)

            with torch.amp.autocast('cuda', enabled=self.config.use_amp):
                # 单一接口: forward() 返回 TrainingStats 或 Tensor
                stats = self.model(inputs)
                # 类型检查优先于 hasattr (I112)
                if TrainingStats is not None and isinstance(stats, TrainingStats):
                    outputs = stats.logits
                else:
                    # 兼容旧接口: 可能是 tuple 或纯 Tensor
                    if hasattr(stats, 'logits'):
                        outputs = stats.logits
                    elif isinstance(stats, tuple):
                        outputs = stats[0]
                    else:
                        outputs = stats

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
                # I107-3: 在累积步结束时清理 CUDA 缓存碎片
                # 这有助于防止长时间训练时的显存碎片化
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.state.global_step += 1

            # CRIT-4: 累积到 GPU 张量 (不同步)
            # 记录原始 loss 值用于 callback (不乘 accumulation_steps，保持语义一致性)
            loss_for_callback = loss.detach()
            loss_sum += loss_for_callback * batch_size
            num_samples += batch_size

            # 更新指标 (使用已提取的 outputs)
            if self.metrics:
                self.metrics.update(outputs.detach(), targets)

            # Callback: batch end
            ctx.loss = loss_for_callback.item()  # 仅在 callback 时同步
            ctx.outputs = outputs
            ctx.targets = targets
            self.callbacks.on_batch_end(self, ctx)

            if ctx.stop_training:
                break

        # CRIT-4: Epoch 结束时同步一次
        train_loss = (loss_sum / max(num_samples, 1)).item()

        # 计算 epoch 指标
        result = {"train_loss": train_loss}
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
        # CRIT-4: 使用 GPU 张量累积，避免 .item() 同步点
        val_loss_sum = torch.zeros(1, device=self.device)
        val_num_samples = 0

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

            batch_size = targets.size(0)

            # 验证禁用 AMP 以确保指标精度 (I78: 使用 detach().item() 支持 torch.compile)
            with torch.amp.autocast('cuda', enabled=False):
                model_output = self.model(inputs)
                # P0-Critical: 模型可能返回元组，提取 logits
                if isinstance(model_output, tuple):
                    outputs = model_output[0]
                else:
                    outputs = model_output
                loss = self.loss_fn(outputs, targets)

            # CRIT-4: 累积到 GPU 张量 (不同步)
            val_loss_sum += loss.detach() * batch_size
            val_num_samples += batch_size

            if self.metrics:
                self.metrics.update(outputs, targets)

        # CRIT-4: Epoch 结束时同步一次
        val_loss = (val_loss_sum / max(val_num_samples, 1)).item()

        # 计算指标
        result = {"val_loss": val_loss}
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
        """保存检查点 (使用 ModelGene 自包含格式)

        优先使用 from_config() 从配置构造 ModelGene（单一数据源）。
        如果没有 arch_config，回退到 from_model()。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 构造 ModelGene
        if self.arch_config is not None:
            # 单一数据源：从配置构造
            gene = ModelGene.from_config(
                self.arch_config,
                dataset_name=getattr(self.config, 'dataset', 'unknown'),
                epoch=self.state.epoch,
            )
        else:
            # 回退：从模型提取（通用训练器）
            from training.core.model_gene import ModelGene
            gene = ModelGene.from_model(
                self.model,
                dataset_name=getattr(self.config, 'dataset', 'unknown'),
                epoch=self.state.epoch,
            )

        # 使用共享的 checkpoint 保存函数
        save_checkpoint_with_gene(
            path=path,
            model=self.model,
            gene=gene,
            optimizer_state=self.optimizer.state_dict(),
            epoch=self.state.epoch,
            val_acc=self.state.best_metric,
        )
    
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
    # 协议
    "ModelProtocol",
    "FractalModelProtocol",
    "ExtraInfoProtocol",
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
    CUB200ModularTrainer,  # I36: 新增继承版本
    create_cub200_trainer,
    get_cub200_augmentation,
)
