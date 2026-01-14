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
import time
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
    "LayeredEvaluationCallback",
]


# ============================================================================
# 分层评估回调
# ============================================================================

@dataclass
class LayeredEvaluationCallbackConfig:
    """分层评估回调配置
    
    用于配置训练过程中自动执行的分层评估。
    
    数学形式化
    ==========
    评估触发条件:
        trigger = (epoch % eval_interval == 0) ∨ is_new_best ∨ is_final_epoch
    
    评估层选择:
        layers = enabled_layers ∩ available_layers
    """
    
    # 启用开关
    enabled: bool = True
    
    # 评估层
    enabled_layers: List[str] = field(
        default_factory=lambda: ["L1", "L2", "L4", "L5", "L6"]
    )  # L3 默认禁用（需要额外开销）
    
    # 触发条件
    eval_interval: int = 10  # 每 N 个 epoch 评估一次
    eval_on_best: bool = True  # 新 best 时评估
    eval_on_final: bool = True  # 最后一个 epoch 评估
    
    # 采样
    max_samples: int = 5000
    batch_size: int = 64
    
    # 日志
    log_to_wandb: bool = True
    log_to_console: bool = True
    
    # 保存
    save_report: bool = True
    report_format: str = "json"


class LayeredEvaluationCallback:
    """分层评估训练回调
    
    在训练过程中自动执行分层评估，为模型架构改进提供数据分析支持。
    
    功能
    ====
    1. 周期性分层评估 (L1-L6)
    2. 新 best 模型时自动评估
    3. 评估结果记录到 WandB
    4. 生成详细的 JSON 报告
    
    评估层说明
    ==========
    - L1 (分类): Top-1/5 准确率、MCA、ECE、混淆分析
    - L2 (Tokenizer): Token 数统计、深度分布、空间覆盖
    - L3 (注意力): 注意力熵、Head 利用率 (开销较大)
    - L4 (表示): Fisher 判别比、类别可分性
    - L5 (效率): 延迟、吞吐量、内存
    - L6 (稳定性): 权重范数、NaN/Inf 检测
    
    使用方法
    ========
    >>> callback = LayeredEvaluationCallback(
    ...     config=LayeredEvaluationCallbackConfig(
    ...         eval_interval=5,
    ...         enabled_layers=["L1", "L2", "L5"],
    ...     ),
    ... )
    >>> trainer = ModularTrainer(..., callbacks=[callback])
    
    数据分析输出
    ============
    评估报告包含以下用于架构改进的关键指标:
    
    1. **Tokenizer 效率分析** (L2):
       - 深度分布熵 H_d → 检测深度坍缩
       - Token-内容相关性 ρ → 验证自适应性
    
    2. **表示质量分析** (L4):
       - Fisher 判别比 FDR → 类间分离度
       - 类别可分性 → 识别难分类
    
    3. **资源效率分析** (L5):
       - 组件延迟分解 → 定位瓶颈
       - Accuracy/Token → 效率优化方向
    """
    
    priority = 90  # 在 CheckpointCallback 之后
    
    def __init__(
        self,
        config: Optional[LayeredEvaluationCallbackConfig] = None,
        val_loader=None,
        num_classes: int = 10,
    ):
        """
        Args:
            config: 评估回调配置
            val_loader: 验证数据加载器 (可在 on_train_begin 时设置)
            num_classes: 类别数
        """
        self.config = config or LayeredEvaluationCallbackConfig()
        self.val_loader = val_loader
        self.num_classes = num_classes
        
        # 运行时状态
        self._last_eval_epoch = -1
        self._last_best_metric = float("-inf")
        self._evaluators = {}
        self._reports = []
    
    def on_train_begin(self, trainer: 'ModularTrainer', state: 'TrainerState') -> None:
        """训练开始时初始化评估器"""
        if not self.config.enabled:
            return
        
        # 尝试从 trainer 获取 val_loader
        if self.val_loader is None and hasattr(trainer, 'val_loader'):
            self.val_loader = trainer.val_loader
        
        # 获取 num_classes
        if hasattr(trainer, 'num_classes'):
            self.num_classes = trainer.num_classes
        elif hasattr(trainer, 'config') and hasattr(trainer.config, 'num_classes'):
            self.num_classes = trainer.config.num_classes
        
        # 延迟导入评估器
        try:
            from ..evaluation_layers import (
                ClassificationEvaluator,
                TokenizerEvaluator,
                AttentionEvaluator,
                RepresentationEvaluator,
                EfficiencyEvaluator,
                StabilityEvaluator,
            )
            
            # 初始化评估器
            self._evaluators = {
                "L1": ClassificationEvaluator(self.num_classes),
                "L2": TokenizerEvaluator(),
                "L3": AttentionEvaluator(),
                "L4": RepresentationEvaluator(),
                "L5": EfficiencyEvaluator(),
                "L6": StabilityEvaluator(),
            }
            
            if self.config.log_to_console:
                enabled = ", ".join(self.config.enabled_layers)
                print(f"[LayeredEval] Initialized with layers: {enabled}")
                
        except ImportError as e:
            warnings.warn(f"[LayeredEval] Failed to import evaluators: {e}")
            self.config.enabled = False
    
    def on_epoch_end(self, trainer: 'ModularTrainer', ctx: 'CallbackContext') -> None:
        """Epoch 结束时检查是否需要评估"""
        if not self.config.enabled or not self._evaluators:
            return
        
        should_eval = False
        eval_reason = ""
        
        # 条件1: 周期性评估
        if ctx.epoch > 0 and ctx.epoch % self.config.eval_interval == 0:
            should_eval = True
            eval_reason = f"periodic (epoch {ctx.epoch})"
        
        # 条件2: 新 best
        if self.config.eval_on_best and ctx.metrics:
            current_metric = ctx.metrics.get("val_accuracy", 0)
            if current_metric > self._last_best_metric:
                self._last_best_metric = current_metric
                should_eval = True
                eval_reason = f"new best ({current_metric:.2f}%)"
        
        if should_eval and ctx.epoch != self._last_eval_epoch:
            self._run_evaluation(trainer, ctx, eval_reason)
            self._last_eval_epoch = ctx.epoch
    
    def on_train_end(self, trainer: 'ModularTrainer', state: 'TrainerState') -> None:
        """训练结束时执行最终评估"""
        if not self.config.enabled or not self.config.eval_on_final:
            return
        
        # 构造一个 ctx
        ctx = type('CallbackContext', (), {
            'epoch': state.epoch,
            'global_step': state.global_step,
            'metrics': {'val_accuracy': state.best_metric},
            'batch_idx': 0,
            'loss': 0.0,
        })()
        
        if state.epoch != self._last_eval_epoch:
            self._run_evaluation(trainer, ctx, "final")
    
    def _run_evaluation(
        self,
        trainer: 'ModularTrainer',
        ctx: 'CallbackContext',
        reason: str,
    ) -> None:
        """执行分层评估"""
        import time
        from ..evaluation_layers import LayeredEvaluationReport
        
        start_time = time.time()
        
        if self.config.log_to_console:
            print(f"\n{'='*60}")
            print(f"[LayeredEval] Running evaluation (reason: {reason})")
            print(f"{'='*60}")
        
        model = trainer.model
        device = next(model.parameters()).device
        
        # 确保模型在 eval 模式
        was_training = model.training
        model.eval()
        
        # 创建报告
        report = LayeredEvaluationReport(
            checkpoint_path=f"epoch_{ctx.epoch}",
            dataset_name=getattr(trainer, 'dataset_name', 'unknown'),
            num_samples=len(self.val_loader.dataset) if self.val_loader else 0,
            num_classes=self.num_classes,
            device=str(device),
        )
        
        metrics_for_logging = {}
        
        try:
            # L6: 稳定性 (最先，检查模型健康)
            if "L6" in self.config.enabled_layers and "L6" in self._evaluators:
                report.L6_stability = self._evaluators["L6"].evaluate(model)
                metrics_for_logging.update({
                    "eval/L6_health_score": report.L6_stability.gradient_health_score,
                    "eval/L6_has_nan": int(report.L6_stability.has_nan_weights),
                    "eval/L6_splitter_health": report.L6_stability.splitter_health_score,
                })
                if self.config.log_to_console:
                    print(f"  L6: health={report.L6_stability.gradient_health_score:.2f}")
            
            # L5: 效率
            if "L5" in self.config.enabled_layers and "L5" in self._evaluators:
                if self.val_loader:
                    sample_input = next(iter(self.val_loader))[0][:8].to(device)
                    report.L5_efficiency = self._evaluators["L5"].evaluate(
                        model, sample_input, device
                    )
                    metrics_for_logging.update({
                        "eval/L5_latency_ms": report.L5_efficiency.avg_latency_ms,
                        "eval/L5_throughput": report.L5_efficiency.throughput_samples_per_sec,
                        "eval/L5_memory_mb": report.L5_efficiency.peak_memory_mb,
                        "eval/L5_tokenizer_latency_ms": report.L5_efficiency.tokenizer_latency_ms,
                        "eval/L5_transformer_latency_ms": report.L5_efficiency.transformer_latency_ms,
                    })
                    if self.config.log_to_console:
                        print(f"  L5: latency={report.L5_efficiency.avg_latency_ms:.1f}ms, "
                              f"throughput={report.L5_efficiency.throughput_samples_per_sec:.0f}/s")
            
            # L1: 分类性能
            if "L1" in self.config.enabled_layers and "L1" in self._evaluators:
                if self.val_loader:
                    report.L1_classification = self._evaluators["L1"].evaluate(
                        model, self.val_loader, device
                    )
                    metrics_for_logging.update({
                        "eval/L1_top1_acc": report.L1_classification.top1_accuracy,
                        "eval/L1_top5_acc": report.L1_classification.top5_accuracy,
                        "eval/L1_mca": report.L1_classification.mean_class_accuracy,
                        "eval/L1_ece": report.L1_classification.ece,
                    })
                    if self.config.log_to_console:
                        print(f"  L1: top1={report.L1_classification.top1_accuracy:.2f}%, "
                              f"mca={report.L1_classification.mean_class_accuracy:.2f}%")
            
            # L2: Tokenizer
            if "L2" in self.config.enabled_layers and "L2" in self._evaluators:
                if self.val_loader:
                    report.L2_tokenizer = self._evaluators["L2"].evaluate(
                        model, self.val_loader, device
                    )
                    metrics_for_logging.update({
                        "eval/L2_avg_tokens": report.L2_tokenizer.avg_tokens,
                        "eval/L2_depth_entropy": report.L2_tokenizer.depth_entropy,
                        "eval/L2_spatial_coverage": report.L2_tokenizer.spatial_coverage_ratio,
                        "eval/L2_content_correlation": report.L2_tokenizer.content_token_correlation,
                    })
                    if self.config.log_to_console:
                        print(f"  L2: tokens={report.L2_tokenizer.avg_tokens:.1f}, "
                              f"entropy={report.L2_tokenizer.depth_entropy:.3f}")
            
            # L3: 注意力 (开销较大，默认禁用)
            if "L3" in self.config.enabled_layers and "L3" in self._evaluators:
                if self.val_loader:
                    report.L3_attention = self._evaluators["L3"].evaluate(
                        model, self.val_loader, device, max_batches=10
                    )
                    metrics_for_logging.update({
                        "eval/L3_avg_entropy": report.L3_attention.avg_entropy,
                        "eval/L3_dead_head_ratio": report.L3_attention.dead_head_ratio,
                    })
                    if self.config.log_to_console:
                        print(f"  L3: entropy={report.L3_attention.avg_entropy:.3f}")
            
            # L4: 表示
            if "L4" in self.config.enabled_layers and "L4" in self._evaluators:
                if self.val_loader:
                    report.L4_representation = self._evaluators["L4"].evaluate(
                        model, self.val_loader, device
                    )
                    metrics_for_logging.update({
                        "eval/L4_fisher_ratio": report.L4_representation.fisher_discriminant_ratio,
                        "eval/L4_avg_separability": report.L4_representation.avg_separability,
                    })
                    if self.config.log_to_console:
                        print(f"  L4: FDR={report.L4_representation.fisher_discriminant_ratio:.2f}")
        
        except Exception as e:
            warnings.warn(f"[LayeredEval] Evaluation error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 恢复训练模式
            if was_training:
                model.train()
        
        # 记录到 WandB
        if self.config.log_to_wandb:
            self._log_to_wandb(metrics_for_logging, ctx.global_step)
        
        # 保存报告
        if self.config.save_report:
            report.evaluation_time_sec = time.time() - start_time
            self._save_report(trainer, report, ctx.epoch)
        
        self._reports.append(report)
        
        if self.config.log_to_console:
            elapsed = time.time() - start_time
            print(f"[LayeredEval] Completed in {elapsed:.1f}s")
            print(f"{'='*60}\n")
    
    def _log_to_wandb(self, metrics: Dict[str, float], step: int) -> None:
        """记录指标到 WandB"""
        if not _check_wandb_available():
            return
        
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics, step=step)
        except Exception as e:
            warnings.warn(f"[LayeredEval] WandB logging failed: {e}")
    
    def _save_report(
        self,
        trainer: 'ModularTrainer',
        report,
        epoch: int,
    ) -> None:
        """保存评估报告"""
        try:
            import json
            
            # 确定保存路径
            if hasattr(trainer, 'config') and hasattr(trainer.config, 'checkpoint_dir'):
                save_dir = Path(trainer.config.checkpoint_dir)
            else:
                save_dir = Path("./checkpoints")
            
            save_dir = save_dir / "evaluations"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            report_path = save_dir / f"layered_eval_epoch_{epoch:04d}.json"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False, default=str)
            
            if self.config.log_to_console:
                print(f"  Report saved: {report_path}")
                
        except Exception as e:
            warnings.warn(f"[LayeredEval] Failed to save report: {e}")
    
    def get_reports(self) -> List:
        """获取所有评估报告"""
        return self._reports
