"""Numerical Defense Module

Provides numerical stability protection:
- Gradient NaN/Inf detection
- Anomaly detection context
- Automatic gradient skipping on numerical issues
- NaN auto-investigation with debug info dumping
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Dict, Any, List, Callable
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..metrics.collector import MetricsCollector


class AnomalyDetectionContext:
    """Context manager for PyTorch anomaly detection

    Enables torch.autograd.set_detect_anomaly() within a scope.
    Useful for debugging gradient issues.

    Example:
        with AnomalyDetectionContext(model):
            loss = model(inputs)
            loss.backward()
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._previous_state = None

    def __enter__(self):
        if self.enabled:
            self._previous_state = torch.is_anomaly_enabled()
            torch.set_detect_anomaly(True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self._previous_state is not None:
            torch.set_detect_anomaly(self._previous_state)
        return False


class GradientValidator:
    """Validate gradients for numerical issues

    Checks for NaN, Inf, and extreme values in gradients.
    Can automatically skip optimizer steps on detected issues.

    Example:
        validator = GradientValidator(model, skip_on_issue=True)

        # After backward:
        should_skip = validator.check_gradients()
        if not should_skip:
            optimizer.step()
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        skip_on_issue: bool = True,
        log_warnings: bool = True,
        collector: Optional["MetricsCollector"] = None,
    ):
        self.model = model
        self.skip_on_issue = skip_on_issue
        self.log_warnings = log_warnings
        self.collector = collector

        self.issue_count = 0
        self.nan_count = 0
        self.inf_count = 0
        self.issues_history: List[Dict[str, Any]] = []

    def check_gradients(self) -> bool:
        """Check all model gradients for issues

        Returns:
            True if gradients are valid (can proceed with optimizer step)
            False if gradients have issues (should skip)
        """
        if self.model is None:
            return True

        has_nan = False
        has_inf = False

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad

                # Check NaN
                if torch.isnan(grad).any():
                    has_nan = True
                    self.nan_count += 1
                    self._record_issue(name, "nan", grad.norm().item())

                # Check Inf
                if torch.isinf(grad).any():
                    has_inf = True
                    self.inf_count += 1
                    self._record_issue(name, "inf", grad.norm().item())

        has_issue = has_nan or has_inf
        if has_issue:
            self.issue_count += 1

            if self.log_warnings:
                self._log_warning(has_nan, has_inf)

        # Emit to MetricsCollector if available
        if self.collector is not None:
            self.collector.record("nan_count", self.nan_count)
            self.collector.record("inf_count", self.inf_count)
            self.collector.record("issue_count", self.issue_count)

        # Return True if valid (proceed), False if skip
        return not (self.skip_on_issue and has_issue)

    def _record_issue(self, name: str, issue_type: str, norm: float):
        """Record issue for history"""
        self.issues_history.append({
            "name": name,
            "type": issue_type,
            "norm": norm,
        })

    def _log_warning(self, has_nan: bool, has_inf: bool):
        """Log warning message"""
        issues = []
        if has_nan:
            issues.append("NaN")
        if has_inf:
            issues.append("Inf")

        print(f"[WARNING] Gradient issue detected: {', '.join(issues)} "
              f"(total issues: {self.issue_count})")

    def get_gradient_stats(self) -> Dict[str, Any]:
        """Get gradient statistics

        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_issues": self.issue_count,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "has_issues": self.issue_count > 0,
        }

        # Current gradient norms
        if self.model is not None:
            norms = []
            for param in self.model.parameters():
                if param.grad is not None:
                    norms.append(param.grad.norm().item())

            if norms:
                stats["current_grad_norm"] = norms[0] if len(norms) == 1 else norms
                stats["max_grad_norm"] = max(norms)
                stats["min_grad_norm"] = min(norms)

        return stats

    def has_recent_issues(self, window: int = 10) -> bool:
        """Check if there were issues in recent steps

        Args:
            window: Number of recent steps to check

        Returns:
            True if issues found in recent history
        """
        recent = self.issues_history[-window:]
        return len(recent) > 0

    def reset(self) -> None:
        """Reset counters and history"""
        self.issue_count = 0
        self.nan_count = 0
        self.inf_count = 0
        self.issues_history.clear()


class NumericalDefender:
    """Complete numerical defense system

    Combines anomaly detection and gradient validation.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        detect_anomaly: bool = False,
        skip_on_nan: bool = True,
        check_frequency: int = 1,
        collector: Optional["MetricsCollector"] = None,
    ):
        self.model = model
        self.detect_anomaly = detect_anomaly
        self.skip_on_nan = skip_on_nan
        self.check_frequency = check_frequency
        self.collector = collector

        self.validator = GradientValidator(model, skip_on_issue=skip_on_nan, collector=collector)
        self.anomaly_context = AnomalyDetectionContext(enabled=detect_anomaly)

        self.step_count = 0

    def should_check(self) -> bool:
        """Check if we should validate gradients this step"""
        return self.step_count % self.check_frequency == 0

    def pre_backward(self):
        """Hook to call before backward"""
        pass  # Could enable anomaly detection here if needed

    def post_backward(self) -> bool:
        """Hook to call after backward, returns whether to skip optimizer step

        Returns:
            True if should proceed with optimizer step
            False if should skip
        """
        self.step_count += 1

        if self.should_check():
            return self.validator.check_gradients()
        return True

    def pre_step(self):
        """Hook to call before optimizer step"""
        if self.detect_anomaly:
            self.anomaly_context.__enter__()

    def post_step(self):
        """Hook to call after optimizer step"""
        if self.detect_anomaly:
            self.anomaly_context.__exit__(None, None, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get defense statistics"""
        return {
            "validator": self.validator.get_gradient_stats(),
            "step_count": self.step_count,
            "skip_rate": self.validator.issue_count / max(self.step_count, 1),
        }

    def reset(self) -> None:
        """Reset defender state"""
        self.validator.reset()
        self.step_count = 0


def check_tensor_numerical_health(
    tensor: torch.Tensor,
    name: str = "tensor",
    raise_on_issue: bool = False,
) -> Dict[str, Any]:
    """Check a tensor for numerical health

    Args:
        tensor: Tensor to check
        name: Name for logging
        raise_on_issue: Whether to raise exception on issues

    Returns:
        Dictionary of health metrics
    """
    result = {
        "name": name,
        "has_nan": torch.isnan(tensor).any().item(),
        "has_inf": torch.isinf(tensor).any().item(),
        "min": tensor.min().item() if tensor.numel() > 0 else None,
        "max": tensor.max().item() if tensor.numel() > 0 else None,
        "mean": tensor.mean().item() if tensor.numel() > 0 else None,
        "std": tensor.std().item() if tensor.numel() > 0 else None,
    }

    has_issues = result["has_nan"] or result["has_inf"]

    if has_issues and raise_on_issue:
        raise ValueError(f"Numerical issue in {name}: NaN={result['has_nan']}, Inf={result['has_inf']}")

    return result


# =============================================================================
# Activation Stats Collector (用于捕获前向传播中的激活值统计)
# =============================================================================


class ActivationStatsCollector:
    """使用 forward hooks 记录关键节点的激活值统计"""

    def __init__(
        self,
        model: nn.Module,
        target_modules: Optional[List[str]] = None,
        collector: Optional["MetricsCollector"] = None,
    ):
        """初始化激活值收集器

        Args:
            model: 要监控的模型
            target_modules: 要监控的模块名称子串列表（如 ["splitter", "manifold", "decoder"]）
            collector: Optional MetricsCollector for unified metrics pipeline
        """
        self.model = model
        self.target_modules = target_modules or ["splitter", "manifold", "decoder", "entmax", "density", "mlp_head", "head", "classifier"]
        self.collector = collector
        self.hooks: List[Callable] = []
        self.stats: Dict[str, Dict[str, float]] = {}
        self._register_hooks()

    def _register_hooks(self):
        """注册 forward hooks

        注意: torch.compile 与 forward hooks 不兼容 (torch._dynamo.exc.InternalTorchDynamoError:
        FakeRootModule 无法解析 hook closure 中捕获的 cell 变量)。
        如果模型已被 torch.compile 包装 (OptimizedModule)，则跳过 hook 注册。
        """
        # 检测模型是否已被 torch.compile 包装
        # torch.compile 返回 torch._dynamo.eval_frame.OptimizedModule
        if hasattr(self.model, '_orig_mod'):
            # 模型已被 torch.compile 包装，hook 会导致 FakeRootModule 错误
            return

        def create_hook(name: str):
            def hook(module, input, output):
                # 处理输出
                if isinstance(output, torch.Tensor):
                    self._record_tensor_stats(name, output)
                elif isinstance(output, (tuple, list)):
                    for i, o in enumerate(output):
                        if isinstance(o, torch.Tensor):
                            self._record_tensor_stats(f"{name}_{i}", o)
            return hook

        for name, module in self.model.named_modules():
            # 检查模块名是否匹配目标
            if any(target in name.lower() for target in self.target_modules):
                hook = module.register_forward_hook(create_hook(name))
                self.hooks.append(hook)

    def _record_tensor_stats(self, name: str, tensor: torch.Tensor):
        """记录张量统计"""
        if tensor.numel() == 0:
            return

        # 使用 detach() 避免追踪梯度
        t = tensor.detach()

        # 计算统计（处理不同维度）
        if t.dim() > 2:
            t_flat = t.flatten(start_dim=2)
        elif t.dim() == 2:
            t_flat = t
        else:
            t_flat = t.unsqueeze(0)

        # 计算统计（延迟 .item() 调用以避免同步）
        # 在 get_stats() 时再转换为 Python scalars
        self.stats[name] = {
            "mean": t_flat.mean(),
            "std": t_flat.std(),
            "min": t_flat.min(),
            "max": t_flat.max(),
            "norm": t_flat.norm(),
            "has_nan": torch.isnan(t_flat).any(),
            "has_inf": torch.isinf(t_flat).any(),
        }

        # Emit to MetricsCollector if available
        if self.collector is not None:
            mean_val = t_flat.mean().item()
            std_val = t_flat.std().item()
            self.collector.record(f"activation_{name}_mean", mean_val)
            self.collector.record(f"activation_{name}_std", std_val)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """获取收集的统计信息（延迟转换避免同步）"""
        # 延迟转换所有 tensor 为 Python scalars
        result = {}
        for name, stat in self.stats.items():
            result[name] = {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in stat.items()
            }
        return result

    def clear(self):
        """清除统计信息"""
        self.stats.clear()

    def remove_hooks(self):
        """移除所有 hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


# =============================================================================
# NaN Auto-Investigation (自动取证功能)
# =============================================================================


class NaNAutoInvestigation:
    """NaN 自动取证器 - 当检测到 NaN 时自动收集调试信息"""

    def __init__(
        self,
        model: nn.Module,
        debug_dir: str = "experiments/debug",
        enabled: bool = True,
    ):
        self.model = model
        self.debug_dir = Path(debug_dir)
        self.enabled = enabled

        # 创建调试目录
        if self.enabled:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        # 激活值收集器
        self.activation_collector = ActivationStatsCollector(model) if enabled else None

        # 记录计数器
        self.investigation_count = 0

    def investigate(
        self,
        epoch: int,
        step: int,
        loss_value: float,
        pre_clip_grad_norm: float,
        input_stats: Optional[Dict[str, float]] = None,
        classification_logits_stats: Optional[Dict[str, float]] = None,
        feature_stats: Optional[Dict[str, float]] = None,
        amp_loss_scale: Optional[float] = None,
        learning_rate: Optional[float] = None,
        loss_components: Optional[Dict[str, float]] = None,
    ) -> Optional[Path]:
        """执行 NaN 调查并保存调试信息

        Args:
            epoch: 当前 epoch
            step: 当前 step
            loss_value: 损失值
            pre_clip_grad_norm: 裁剪前的梯度范数
            input_stats: 输入数据统计（可选）
            classification_logits_stats: 分类 logits 统计（可选），注意：这是 outputs.logits 而非 splitter 内部 logits
            feature_stats: 特征模长统计（可选），mlp_head 前的激活值
            amp_loss_scale: AMP 损失缩放因子（可选）
            learning_rate: 学习率（可选）
            loss_components: 损失分量统计（可选）

        Returns:
            保存的调试文件路径
        """
        if not self.enabled:
            return None

        self.investigation_count += 1
        report: Dict[str, Any] = {
            "meta": {
                "epoch": epoch,
                "step": step,
                "loss_value": loss_value,
                "pre_clip_grad_norm": pre_clip_grad_norm,
                "investigation_count": self.investigation_count,
            },
            "input_data_summary": input_stats or {},
            "classification_logits_stats": classification_logits_stats or {},
            "training_env": {
                "amp_loss_scale": amp_loss_scale,
                "learning_rate": learning_rate,
            },
            "loss_components": loss_components or {},
            "feature_stats": feature_stats or {},
        }

        # 自动诊断结论
        report["diagnosis"] = self._diagnose_nan(
            input_stats=input_stats,
            classification_logits_stats=classification_logits_stats,
            feature_stats=feature_stats,
            loss_components=loss_components,
            amp_loss_scale=amp_loss_scale,
        )

        # A. 梯度热力图分析
        report["gradient_heatmap"] = self._analyze_gradient_heatmap()

        # B. 激活值统计
        if self.activation_collector:
            report["activation_stats"] = self.activation_collector.get_stats()
            self.activation_collector.clear()

        # C. 找到第一个出现 NaN 的层
        report["first_nan_layer"] = self._find_first_nan_layer()

        # D. 参数范围分析
        report["param_ranges"] = self._analyze_param_ranges()

        # 保存到文件
        filename = f"nan_snapshot_epoch_{epoch:04d}_step_{step:06d}.json"
        filepath = self.debug_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"[DEBUG] NaN investigation report saved to: {filepath}")
        return filepath

    def _analyze_gradient_heatmap(self) -> List[Dict[str, Any]]:
        """分析每层的梯度范数（梯度热力图）"""
        layers_data = []

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()

                has_nan = torch.isnan(grad).any().item()
                has_inf = torch.isinf(grad).any().item()

                # 计算梯度范数
                grad_norm = grad.norm().item() if not has_nan and not has_inf else float('nan')

                # 计算参数范围
                param_min = param.detach().min().item()
                param_max = param.detach().max().item()

                layers_data.append({
                    "layer_name": name,
                    "grad_norm": grad_norm,
                    "has_nan": has_nan,
                    "has_inf": has_inf,
                    "param_range": [param_min, param_max],
                    "param_shape": list(param.shape),
                    "param_numel": param.numel(),
                })

        # 按梯度范数排序（异常的在前）
        layers_data.sort(key=lambda x: float('inf') if x["has_nan"] or x["has_inf"] else -x["grad_norm"])

        return layers_data

    def _find_first_nan_layer(self) -> Optional[Dict[str, Any]]:
        """找到第一个出现 NaN 的层（反向传播顺序）"""
        # PyTorch 反向传播是从输出到输入，所以后面的层先有梯度
        # 我们按参数顺序找最后一个出现 NaN 的（接近 loss 的）
        for name, param in reversed(list(self.model.named_parameters())):
            if param.grad is not None and torch.isnan(param.grad).any():
                grad = param.grad.detach()
                return {
                    "layer_name": name,
                    "grad_norm": grad.norm().item(),
                    "nan_count": torch.isnan(grad).sum().item(),
                    "total_params": grad.numel(),
                    "nan_ratio": torch.isnan(grad).sum().item() / grad.numel(),
                }
        return None

    def _diagnose_nan(
        self,
        input_stats: Optional[Dict[str, float]] = None,
        classification_logits_stats: Optional[Dict[str, float]] = None,
        feature_stats: Optional[Dict[str, float]] = None,
        loss_components: Optional[Dict[str, float]] = None,
        amp_loss_scale: Optional[float] = None,
    ) -> Dict[str, Any]:
        """自动诊断 NaN 根因
        
        诊断规则:
        | 记录项        | 如果发现...               | 结论                    |
        |---------------|--------------------------|------------------------|
        | Input Data    | has_nan: true           | 数据清洗有问题          |
        | Splitter      | max > 100               | Entmax 溢出            |
        | Feature Norm  | max > 1e3               | Head前需LayerNorm      |
        | AMP Scale     | scale=1.0 仍溢出        | 权重初始化/LR问题      |
        | Loss Break    | 某项 > 1e6              | 该Loss项异常           |
        | Pre-clip Norm | NaN 但 Input 正常        | 梯度爆炸，需调低 LR    |
        """
        diagnosis = []
        root_cause = "unknown"

        # R1: 输入数据问题
        if input_stats and input_stats.get("images_has_nan"):
            diagnosis.append("[R1] INPUT_NAN: 输入数据包含 NaN，可能是数据清洗问题或坏图")
            root_cause = "input_data"

        # R2: 分类 Logits 异常 (注意：这是 outputs.logits，不是 splitter 内部 logits)
        if classification_logits_stats:
            classification_logits_stats.get("logits_max", 0)
            classification_logits_stats.get("logits_min", 0)
            # I-AUDIT: 分类 logits 较大是正常的（尤其是类别多时），不再诊断为溢出
            # 但 NaN/Inf 仍然是异常的
            if classification_logits_stats.get("logits_has_nan"):
                diagnosis.append("[R2a] CLASS_LOGITS_NAN: 分类 logits 包含 NaN")
                root_cause = "classification_logits_nan"
            if classification_logits_stats.get("logits_has_inf"):
                diagnosis.append("[R2b] CLASS_LOGITS_INF: 分类 logits 包含 Inf")
                root_cause = "classification_logits_inf"

        # R3: 特征模长异常 (Feature Norm)
        if feature_stats:
            max_feat = feature_stats.get("max", 0)
            feature_stats.get("mean", 0)
            if max_feat > 1e3:
                diagnosis.append(f"[R3a] FEATURE_NORM_HIGH: 特征 max={max_feat:.2e} > 1e3，mlp_head 前需 LayerNorm 或缩放")
                root_cause = "feature_norm_high"
            if feature_stats.get("has_nan"):
                diagnosis.append("[R3b] FEATURE_NAN: mlp_head 前特征包含 NaN")
                root_cause = "feature_nan"
            if feature_stats.get("has_inf"):
                diagnosis.append("[R3c] FEATURE_INF: mlp_head 前特征包含 Inf")
                root_cause = "feature_inf"

        # R4: AMP 状态检查
        if amp_loss_scale is not None and amp_loss_scale <= 1.0:
            diagnosis.append(f"[R4] AMP_SCALE_LOW: Loss Scale={amp_loss_scale:.1f} 已降至最低仍溢出，需检查权重初始化或学习率")
            if root_cause == "unknown":
                root_cause = "amp_scale_low"

        # R5: 损失项爆炸 (Loss Breakdown)
        if loss_components:
            for name, value in loss_components.items():
                if name != "total" and abs(value) > 1e6:
                    diagnosis.append(f"[R5] LOSS_EXPLODE: {name}={value:.2e} 异常大，可能是该损失权重设置不当")
                    root_cause = "loss_explode"

        # R6: 默认诊断（输入正常但梯度 NaN）
        if root_cause == "unknown":
            diagnosis.append("[R6] GRAD_EXPLODE: 输入/特征/Loss 均正常但梯度 NaN，可能是学习率过高或梯度裁剪不足")
            root_cause = "grad_explode"

        return {
            "diagnosis": diagnosis,
            "root_cause": root_cause,
        }

    def _analyze_param_ranges(self) -> Dict[str, List[float]]:
        """分析模型参数范围"""
        ranges = {}
        for name, param in self.model.named_parameters():
            p = param.detach()
            std_val = p.std().item() if p.numel() > 1 else 0.0
            ranges[name] = [p.min().item(), p.max().item(), p.mean().item(), std_val]
        return ranges


def dump_debug_info(
    model: nn.Module,
    epoch: int,
    step: int,
    loss_value: float,
    pre_clip_grad_norm: float,
    debug_dir: str = "experiments/debug",
    input_stats: Optional[Dict[str, float]] = None,
    classification_logits_stats: Optional[Dict[str, float]] = None,
    feature_stats: Optional[Dict[str, float]] = None,
    amp_loss_scale: Optional[float] = None,
    learning_rate: Optional[float] = None,
    loss_components: Optional[Dict[str, float]] = None,
    enabled: bool = True,
) -> Optional[Path]:
    """Dump debug info when NaN is detected (standalone function)

    Args:
        model: Model to analyze
        epoch: Current epoch
        step: Current step
        loss_value: Loss value
        pre_clip_grad_norm: Gradient norm before clipping
        debug_dir: Directory to save debug info
        input_stats: Input data statistics
        classification_logits_stats: Splitter logits statistics
        feature_stats: Feature norm statistics before mlp_head
        amp_loss_scale: AMP loss scale
        learning_rate: Learning rate
        loss_components: Loss component breakdown
        enabled: Whether to actually dump

    Returns:
        Path to saved debug file, or None if disabled
    """
    if not enabled:
        return None

    investigator = NaNAutoInvestigation(model=model, debug_dir=debug_dir, enabled=True)
    return investigator.investigate(
        epoch=epoch,
        step=step,
        loss_value=loss_value,
        pre_clip_grad_norm=pre_clip_grad_norm,
        input_stats=input_stats,
        classification_logits_stats=classification_logits_stats,
        feature_stats=feature_stats,
        amp_loss_scale=amp_loss_scale,
        learning_rate=learning_rate,
        loss_components=loss_components,
    )


__all__ = [
    "AnomalyDetectionContext",
    "GradientValidator",
    "NumericalDefender",
    "check_tensor_numerical_health",
    "ActivationStatsCollector",
    "NaNAutoInvestigation",
    "dump_debug_info",
]
