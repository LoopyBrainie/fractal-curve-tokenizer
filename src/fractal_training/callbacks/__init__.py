"""
Fractal Training Callbacks Module
==================================

提供扩展训练流程的回调机制，包括：
1. WandBCallback - Weights & Biases 实验跟踪
2. SplitterHealthCallback - 分割器健康监控

设计原则:
---------
- 完全可选: 回调不影响核心训练逻辑
- 低耦合: 通过 CallbackContext 接收数据
- 异常安全: 回调失败不中断训练
"""

from __future__ import annotations
import os
import sys
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..trainer import ModularTrainer, TrainerState, CallbackContext

# 延迟导入以处理 wandb 未安装的情况
_wandb_available = None


def _check_wandb_available() -> bool:
    """检查 wandb 是否可用"""
    global _wandb_available
    if _wandb_available is None:
        try:
            import wandb
            _wandb_available = True
        except ImportError:
            _wandb_available = False
    return _wandb_available


# ============================================================================
# WandB 配置增强
# ============================================================================

@dataclass
class WandBCallbackConfig:
    """WandB Callback 详细配置
    
    扩展自 WandBConfig，添加更多控制选项。
    
    指标分组设计:
    -------------
    - train/*: 训练指标 (loss, accuracy, lr)
    - val/*: 验证指标 (loss, accuracy, mca)
    - splitter/*: 分割器健康 (avg_tokens, entropy, temperature)
    - resource/*: 资源使用 (flops, memory)
    - system/*: 系统信息 (gpu_util, step_time)
    
    数学形式化:
    ----------
    记录频率控制:
        - log_freq_batch: 每 N 个 batch 记录一次 (减少 API 调用)
        - log_freq_epoch: 每个 epoch 记录验证指标
    
    带宽优化:
        - commit=False 积累日志，commit=True 批量发送
    """
    
    # 基础配置
    project: str = "fractal-vit"
    entity: Optional[str] = None
    name: Optional[str] = None
    group: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    # 功能开关
    enabled: bool = True
    log_model: bool = True      # 保存模型 artifact
    log_code: bool = False       # 保存代码 (增加上传时间)
    watch_model: bool = False    # 监控梯度 (增加开销)
    watch_freq: int = 1000       # 梯度监控频率
    
    # 日志频率
    log_freq_batch: int = 50     # 每 N batch 记录训练指标
    log_gradients: bool = False  # 记录梯度直方图
    
    # 模型保存策略
    save_best_only: bool = True
    monitor_metric: str = "val_accuracy"
    monitor_mode: str = "max"    # "max" or "min"
    
    # 离线模式
    offline: bool = False        # 离线模式 (无网络时)
    
    # 恢复训练
    resume: Optional[str] = None  # run_id for resuming


# ============================================================================
# WandB Callback 实现
# ============================================================================

class WandBCallback:
    """Weights & Biases 实验跟踪回调
    
    功能:
    -----
    1. 自动记录训练/验证指标
    2. 记录 Splitter 健康状态
    3. 保存最佳模型 artifact
    4. 超参数和配置跟踪
    
    指标命名空间:
    ------------
    ```
    train/loss, train/accuracy, train/lr
    val/loss, val/accuracy, val/mca, val/head_acc, val/tail_acc
    splitter/avg_tokens, splitter/entropy, splitter/temperature, splitter/health_score
    resource/flops, resource/memory_mb
    epoch, global_step
    ```
    
    使用方法:
    --------
    ```python
    wandb_callback = WandBCallback(
        config=WandBCallbackConfig(
            project="fractal-vit",
            name=f"exp_{datetime.now():%Y%m%d_%H%M%S}",
            tags=["baseline", "tiny-imagenet"],
        ),
        experiment_config=experiment_config,  # ExperimentConfig dataclass
    )
    
    trainer = ModularTrainer(
        ...,
        callbacks=[wandb_callback, ...],
    )
    ```
    
    离线使用:
    --------
    ```python
    # 设置环境变量
    os.environ["WANDB_MODE"] = "offline"
    
    # 或使用配置
    wandb_callback = WandBCallback(
        config=WandBCallbackConfig(offline=True),
    )
    
    # 稍后同步: wandb sync ./wandb/offline-run-xxx
    ```
    """
    
    priority = -5  # 在 ProgressCallback 之后执行
    
    def __init__(
        self,
        config: Optional[WandBCallbackConfig] = None,
        experiment_config: Optional[Any] = None,
    ):
        """
        Args:
            config: WandB 回调配置
            experiment_config: 完整实验配置 (用于记录超参数)
        """
        self.config = config or WandBCallbackConfig()
        self.experiment_config = experiment_config
        
        # 运行时状态
        self._run = None
        self._best_metric_value = float("-inf") if self.config.monitor_mode == "max" else float("inf")
        self._step = 0
        
        # 批量日志缓存 (减少 API 调用)
        self._batch_logs: Dict[str, List[float]] = {}
    
    def _init_wandb(self) -> bool:
        """初始化 wandb run"""
        if not _check_wandb_available():
            warnings.warn(
                "wandb not installed. Install with: pip install wandb\n"
                "WandB logging will be disabled."
            )
            return False
        
        if not self.config.enabled:
            return False
        
        import wandb
        
        # 设置离线模式
        if self.config.offline:
            os.environ["WANDB_MODE"] = "offline"
        
        # 准备配置字典
        run_config = {}
        if self.experiment_config is not None:
            if hasattr(self.experiment_config, '__dataclass_fields__'):
                run_config = asdict(self.experiment_config)
            elif isinstance(self.experiment_config, dict):
                run_config = self.experiment_config
        
        # 生成 run name
        run_name = self.config.name
        if run_name is None:
            run_name = f"fractal_vit_{datetime.now():%Y%m%d_%H%M%S}"
        
        # 初始化 run
        try:
            self._run = wandb.init(
                project=self.config.project,
                entity=self.config.entity,
                name=run_name,
                group=self.config.group,
                tags=self.config.tags,
                notes=self.config.notes,
                config=run_config,
                resume=self.config.resume or "allow",
                reinit=True,
            )
            
            print(f"[WandB] Initialized: {self._run.url}")
            return True
            
        except Exception as e:
            warnings.warn(f"[WandB] Failed to initialize: {e}")
            return False
    
    def on_train_begin(self, trainer: 'ModularTrainer', state: 'TrainerState') -> None:
        """训练开始时初始化 WandB"""
        if not self._init_wandb():
            return
        
        import wandb
        
        # 监控模型梯度 (可选)
        if self.config.watch_model and self._run is not None:
            try:
                wandb.watch(
                    trainer.model,
                    log="all" if self.config.log_gradients else "gradients",
                    log_freq=self.config.watch_freq,
                )
                print(f"[WandB] Watching model gradients (freq={self.config.watch_freq})")
            except Exception as e:
                warnings.warn(f"[WandB] Failed to watch model: {e}")
        
        # 记录代码 (可选)
        if self.config.log_code and self._run is not None:
            try:
                wandb.run.log_code(".")
                print("[WandB] Code logged")
            except Exception as e:
                warnings.warn(f"[WandB] Failed to log code: {e}")
    
    def on_batch_end(self, trainer: 'ModularTrainer', ctx: 'CallbackContext') -> None:
        """记录 batch 级指标"""
        if self._run is None:
            return
        
        import wandb
        
        # 累积批次损失
        if ctx.loss is not None:
            if "train_loss" not in self._batch_logs:
                self._batch_logs["train_loss"] = []
            self._batch_logs["train_loss"].append(ctx.loss)
        
        # 每 N 个 batch 记录一次
        if (ctx.batch_idx + 1) % self.config.log_freq_batch == 0:
            log_data = {"global_step": ctx.global_step}
            
            # 平均批次损失
            if self._batch_logs.get("train_loss"):
                avg_loss = sum(self._batch_logs["train_loss"]) / len(self._batch_logs["train_loss"])
                log_data["train/loss"] = avg_loss
                self._batch_logs["train_loss"] = []
            
            # 学习率
            if hasattr(trainer, 'optimizer'):
                log_data["train/lr"] = trainer.optimizer.param_groups[0]['lr']
            
            try:
                wandb.log(log_data, step=ctx.global_step)
            except Exception as e:
                warnings.warn(f"[WandB] Log failed: {e}")
    
    def on_epoch_end(self, trainer: 'ModularTrainer', ctx: 'CallbackContext') -> None:
        """记录 epoch 级指标"""
        if self._run is None or ctx.metrics is None:
            return
        
        import wandb
        
        log_data = {
            "epoch": ctx.epoch,
            "global_step": ctx.global_step,
        }
        
        # 处理指标
        for key, value in ctx.metrics.items():
            if isinstance(value, (int, float)):
                # 自动分配命名空间
                if key.startswith("train_"):
                    namespace = "train"
                    metric_name = key[6:]  # 移除 "train_" 前缀
                elif key.startswith("val_"):
                    namespace = "val"
                    metric_name = key[4:]  # 移除 "val_" 前缀
                elif key.startswith("splitter_") or key.startswith("Splitter"):
                    namespace = "splitter"
                    metric_name = key.replace("splitter_", "").replace("Splitter/", "")
                elif key.startswith("resource_"):
                    namespace = "resource"
                    metric_name = key[9:]
                else:
                    namespace = "metrics"
                    metric_name = key
                
                log_data[f"{namespace}/{metric_name}"] = value
        
        # 从训练器状态提取 Splitter 信息 (如果有)
        splitter_metrics = self._extract_splitter_metrics(trainer)
        if splitter_metrics:
            log_data.update(splitter_metrics)
        
        try:
            wandb.log(log_data, step=ctx.global_step)
        except Exception as e:
            warnings.warn(f"[WandB] Epoch log failed: {e}")
        
        # 检查是否需要保存模型
        if self.config.log_model:
            self._maybe_save_model(trainer, ctx)
    
    def _extract_splitter_metrics(self, trainer: 'ModularTrainer') -> Dict[str, float]:
        """从训练器/模型中提取 Splitter 指标"""
        metrics = {}
        
        try:
            model = trainer.model
            
            # 尝试获取 tokenizer
            tokenizer = getattr(model, 'tokenizer', None)
            if tokenizer is None and hasattr(model, 'module'):  # DataParallel
                tokenizer = getattr(model.module, 'tokenizer', None)
            
            if tokenizer is None:
                return metrics
            
            # 获取 splitter
            splitter = getattr(tokenizer, 'splitter', None)
            if splitter is None or not hasattr(splitter, 'training') or not splitter.training:
                return metrics
            
            # 提取可用指标
            if hasattr(splitter, 'temperature') and splitter.temperature is not None:
                if hasattr(splitter.temperature, 'item'):
                    metrics["splitter/temperature"] = splitter.temperature.item()
                else:
                    metrics["splitter/temperature"] = float(splitter.temperature)
            
            if hasattr(splitter, 'thresholds') and splitter.thresholds is not None:
                thresh = splitter.thresholds
                if hasattr(thresh, 'mean'):
                    metrics["splitter/threshold_mean"] = thresh.mean().item()
                    metrics["splitter/threshold_std"] = thresh.std().item()
            
            # 尝试获取最近的性能统计
            if hasattr(splitter, '_last_perf_stats') and splitter._last_perf_stats:
                stats = splitter._last_perf_stats
                if 'soft_token_count' in stats:
                    metrics["splitter/avg_tokens"] = stats['soft_token_count']
                if 'soft_entropy' in stats:
                    metrics["splitter/entropy"] = stats['soft_entropy']
                if 'entropy_ratio' in stats:
                    metrics["splitter/entropy_ratio"] = stats['entropy_ratio']
        
        except Exception:
            pass  # 静默失败，不影响训练
        
        return metrics
    
    def _maybe_save_model(self, trainer: 'ModularTrainer', ctx: 'CallbackContext') -> None:
        """根据指标决定是否保存模型"""
        if ctx.metrics is None:
            return
        
        current_value = ctx.metrics.get(self.config.monitor_metric)
        if current_value is None:
            return
        
        is_better = False
        if self.config.monitor_mode == "max":
            is_better = current_value > self._best_metric_value
        else:
            is_better = current_value < self._best_metric_value
        
        if not is_better and self.config.save_best_only:
            return
        
        self._best_metric_value = current_value
        
        import wandb
        
        try:
            # 保存模型检查点
            checkpoint_dir = Path(trainer.config.checkpoint_dir) if hasattr(trainer.config, 'checkpoint_dir') else Path("./checkpoints")
            checkpoint_path = checkpoint_dir / "wandb_best_model.pt"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            trainer.save_checkpoint(checkpoint_path)
            
            # 创建 artifact
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}",
                type="model",
                metadata={
                    "epoch": ctx.epoch,
                    self.config.monitor_metric: current_value,
                }
            )
            artifact.add_file(str(checkpoint_path))
            wandb.log_artifact(artifact)
            
            print(f"[WandB] Model artifact saved (epoch {ctx.epoch}, {self.config.monitor_metric}={current_value:.4f})")
            
        except Exception as e:
            warnings.warn(f"[WandB] Failed to save model artifact: {e}")
    
    def on_train_end(self, trainer: 'ModularTrainer', state: 'TrainerState') -> None:
        """训练结束时关闭 WandB"""
        if self._run is None:
            return
        
        import wandb
        
        try:
            # 记录最终统计
            summary = {
                "best_epoch": state.best_epoch,
                "best_metric": state.best_metric,
                "total_epochs": state.epoch,
                "total_steps": state.global_step,
            }
            
            for key, value in summary.items():
                wandb.run.summary[key] = value
            
            wandb.finish()
            print("[WandB] Run finished successfully")
            
        except Exception as e:
            warnings.warn(f"[WandB] Failed to finish run: {e}")


# ============================================================================
# Splitter 健康监控 Callback
# ============================================================================

@dataclass
class SplitterHealthConfig:
    """分割器健康监控配置"""
    
    # 健康阈值
    min_tokens: int = 4
    max_tokens: int = 512
    min_entropy_ratio: float = 0.3
    
    # 警告设置
    warn_on_collapse: bool = True
    warn_on_saturation: bool = True
    log_frequency: int = 1  # 每 N 个 epoch 检查一次


class SplitterHealthCallback:
    r"""分割器健康监控回调
    
    监控指标:
    ---------
    - avg_tokens: 平均 token 数
    - entropy_ratio: 深度熵与最大熵的比值
    - temperature: 当前温度
    - health_score: 综合健康评分
    
    健康评分计算:
    -------------
    $$\text{health} = \min(1, \text{token\_score}) \times \min(1, \text{entropy\_score})$$
    
    其中:
    - token_score = clip((N - N_min) / (N_target - N_min), 0, 1)
    - entropy_score = clip(H / H_max, 0, 1)
    """
    
    priority = 80  # 在检查点保存之后
    
    def __init__(self, config: Optional[SplitterHealthConfig] = None):
        self.config = config or SplitterHealthConfig()
        self._collapse_count = 0
        self._saturation_count = 0
    
    def on_epoch_end(self, trainer: 'ModularTrainer', ctx: 'CallbackContext') -> None:
        """检查分割器健康状态"""
        if ctx.epoch % self.config.log_frequency != 0:
            return
        
        try:
            model = trainer.model
            tokenizer = getattr(model, 'tokenizer', None)
            if tokenizer is None and hasattr(model, 'module'):
                tokenizer = getattr(model.module, 'tokenizer', None)
            
            if tokenizer is None:
                return
            
            splitter = getattr(tokenizer, 'splitter', None)
            if splitter is None:
                return
            
            # 获取健康指标
            health_info = self._compute_health(splitter)
            
            # 添加到 context metrics
            if ctx.metrics is None:
                ctx.metrics = {}
            
            ctx.metrics.update({
                f"splitter_{k}": v for k, v in health_info.items()
            })
            
            # 检查并警告
            if health_info.get("is_collapsed") and self.config.warn_on_collapse:
                self._collapse_count += 1
                print(f"\n⚠️  [SplitterHealth] COLLAPSE detected! "
                      f"avg_tokens={health_info.get('avg_tokens', 'N/A'):.2f} < {self.config.min_tokens}")
            
            if health_info.get("is_saturated") and self.config.warn_on_saturation:
                self._saturation_count += 1
                print(f"\n⚠️  [SplitterHealth] SATURATION detected! "
                      f"avg_tokens={health_info.get('avg_tokens', 'N/A'):.2f} > {self.config.max_tokens}")
        
        except Exception:
            pass  # 静默失败
    
    def _compute_health(self, splitter) -> Dict[str, float]:
        """计算健康指标"""
        result = {}
        
        # 获取最近的性能统计
        stats = getattr(splitter, '_last_perf_stats', {}) or {}
        
        avg_tokens = stats.get('soft_token_count')
        if avg_tokens is not None:
            result["avg_tokens"] = avg_tokens
            result["is_collapsed"] = avg_tokens < self.config.min_tokens
            result["is_saturated"] = avg_tokens > self.config.max_tokens
        
        entropy_ratio = stats.get('entropy_ratio')
        if entropy_ratio is not None:
            result["entropy_ratio"] = entropy_ratio
        
        # 计算综合健康评分
        token_score = 1.0
        if avg_tokens is not None:
            if avg_tokens < self.config.min_tokens:
                token_score = max(0, avg_tokens / self.config.min_tokens)
            elif avg_tokens > self.config.max_tokens:
                token_score = max(0, 1 - (avg_tokens - self.config.max_tokens) / self.config.max_tokens)
        
        entropy_score = entropy_ratio if entropy_ratio is not None else 1.0
        
        result["health_score"] = token_score * min(1.0, entropy_score / self.config.min_entropy_ratio)
        
        return result


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    "WandBCallbackConfig",
    "WandBCallback",
    "SplitterHealthConfig",
    "SplitterHealthCallback",
]
