"""Gradient Flow Monitoring Module

Provides detailed gradient flow monitoring for all model layers.
Tracks gradient norms for analysis and debugging.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional
import torch
import torch.nn as nn
from collections import defaultdict

if TYPE_CHECKING:
    from ..metrics.collector import MetricsCollector


class GradientMonitor:
    """Monitor gradient flow through model layers

    Tracks:
    - Per-layer gradient norms
    - Gradient statistics (mean, std, min, max)
    - Gradient flow between layers
    - I150-3 ENHANCEMENT: 层级（Layer-wise）梯度统计，特别是 ManifoldDecoder

    Example:
        monitor = GradientMonitor(model)

        # In training loop:
        loss.backward()
        grad_stats = monitor.compute_grad_norms()
        monitor.reset()
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        record_layer_norms: bool = True,
        hooks_enabled: bool = True,
        track_key_components: bool = True,
        collector: Optional["MetricsCollector"] = None,
    ):
        """Initialize GradientMonitor

        Args:
            model: The model to monitor
            record_layer_norms: Whether to record per-layer norms
            hooks_enabled: Whether to register backward hooks
            track_key_components: I150-3: 是否追踪关键组件（ManifoldDecoder等）
            collector: Optional MetricsCollector for unified metrics pipeline
        """
        self.model = model
        self.record_layer_norms = record_layer_norms
        self.hooks_enabled = hooks_enabled
        self.track_key_components = track_key_components
        self.collector = collector

        # Storage
        self.layer_norms: Dict[str, List[float]] = defaultdict(list)
        self.grad_norms_by_step: List[float] = []
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        # I-OPT: 延迟 .item() - 在 backward 期间存储原始 tensor，归一化后转换
        self._pending_grad_norms: Dict[str, List[torch.Tensor]] = defaultdict(list)

        # I150-3: 关键组件梯度统计
        self.component_grad_norms: Dict[str, List[float]] = defaultdict(list)
        self.component_grad_stats: Dict[str, Dict[str, float]] = {}

    def register_hooks(self, model: nn.Module) -> None:
        """Register backward hooks to capture gradients

        Args:
            model: The model to monitor
        """
        if not self.hooks_enabled:
            return

        # Remove old hooks
        self.clear_hooks()

        def create_hook(name: str):
            def hook(grad: torch.Tensor) -> None:
                if grad is not None:
                    # I-OOM FIX: detach() 断开梯度图，防止显存泄漏
                    self._pending_grad_norms[name].append(grad.norm().detach())
            return hook

        # Register hooks on parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                handle = param.register_post_accumulate_grad_hook(create_hook(name))
                self._hooks.append(handle)

    def compute_grad_norms(self) -> Dict[str, float]:
        """Compute current gradient norms for all parameters

        Returns:
            Dictionary of parameter name -> gradient norm
        """
        if self.model is None:
            return {}

        norms = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                norms[name] = param.grad.norm().item()

        # Emit to MetricsCollector if available
        if self.collector is not None:
            for name, norm in norms.items():
                self.collector.record(f"layer_grad_norm_{name}", norm)

        return norms

    def compute_total_grad_norm(self) -> float:
        """Compute total gradient norm (for clipping)

        Returns:
            Total gradient norm
        """
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_norm += param.grad.norm().item() ** 2
        return total_norm ** 0.5

    def finalize(self) -> None:
        """I-OPT: 在 backward 结束后调用，将 pending tensor norms 批量转换为 Python float

        在训练循环中的正确使用位置:
            loss.backward()
            monitor.finalize()  # <-- 在这里调用
            grad_stats = monitor.compute_grad_norms()
            monitor.reset()

        这样将 N 个 .item() 同步（每个参数一次）合并为 N 个顺序 .item() 调用，
        GPU 会重叠所有 tensor 的计算，减少等待时间。
        """
        # I-OPT: 将所有 pending tensor norms 批量转换为 Python float
        for name, norm_tensors in self._pending_grad_norms.items():
            for t in norm_tensors:
                self.layer_norms[name].append(t.item())
        self._pending_grad_norms.clear()

    def get_layer_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each layer's gradients

        Returns:
            Dictionary of layer -> {mean, std, min, max, count}
        """
        stats = {}
        for name, norms in self.layer_norms.items():
            if norms:
                stats[name] = {
                    "mean": sum(norms) / len(norms),
                    "std": (sum((n - sum(norms)/len(norms))**2 for n in norms) / len(norms)) ** 0.5,
                    "min": min(norms),
                    "max": max(norms),
                    "count": len(norms),
                }
        return stats

    # I150-3: 新增方法 - 追踪关键组件的梯度
    def compute_component_grad_norms(self) -> Dict[str, float]:
        """I150-3: 计算关键组件的梯度范数

        Deprecated: 使用 layer-packaged 架构后，组件梯度通过 auxiliary_outputs
        直接从各层获取，不再使用 KEY_COMPONENT_KEYWORDS 进行字符串匹配。

        Returns:
            空字典（保持接口兼容）
        """
        return {}
        """I150-3: 获取关键组件的梯度统计

        Returns:
            Dictionary of component -> {mean, std, min, max, latest}
        """
        stats = {}
        for comp_name, norms in self.component_grad_norms.items():
            if norms:
                mean_val = sum(norms) / len(norms)
                stats[comp_name] = {
                    "mean": mean_val,
                    "std": (sum((n - mean_val)**2 for n in norms) / len(norms)) ** 0.5,
                    "min": min(norms),
                    "max": max(norms),
                    "latest": norms[-1],
                }
        return stats

    def get_manifold_decoder_grad_stats(self) -> Dict[str, float]:
        """I150-3: 专门获取 ManifoldDecoder 的梯度统计

        Returns:
            Dictionary with max, min, mean, std of ManifoldDecoder gradients
        """
        manifold_norms = self.component_grad_norms.get("manifold_decoder", [])
        if not manifold_norms:
            return {"max": 0.0, "min": 0.0, "mean": 0.0, "std": 0.0, "latest": 0.0}

        mean_val = sum(manifold_norms) / len(manifold_norms)
        return {
            "max": max(manifold_norms),
            "min": min(manifold_norms),
            "mean": mean_val,
            "std": (sum((n - mean_val)**2 for n in manifold_norms) / len(manifold_norms)) ** 0.5,
            "latest": manifold_norms[-1],
        }

    def get_top_k_norms(self, k: int = 10) -> List[tuple]:
        """Get top k layers by gradient norm

        Args:
            k: Number of top layers to return

        Returns:
            List of (name, norm) tuples
        """
        current_norms = self.compute_grad_norms()
        sorted_norms = sorted(current_norms.items(), key=lambda x: x[1], reverse=True)
        return sorted_norms[:k]

    def detect_gradient_issues(self) -> Dict[str, List[str]]:
        """Detect common gradient issues

        Returns:
            Dictionary of issue type -> affected layers
        """
        issues = {
            "vanishing": [],  # Gradient norm < 1e-7
            "exploding": [],   # Gradient norm > 100
            "nan": [],         # Gradient is NaN
            "zero": [],        # Gradient is exactly zero
        }

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                norm = grad.norm().item()

                if torch.isnan(grad).any():
                    issues["nan"].append(name)
                elif norm < 1e-7:
                    issues["vanishing"].append(name)
                elif norm > 100:
                    issues["exploding"].append(name)
                elif norm == 0:
                    issues["zero"].append(name)

        return issues

    def reset(self) -> None:
        """Reset accumulated statistics"""
        self.layer_norms.clear()
        self.grad_norms_by_step.clear()
        # I150-3: 重置关键组件梯度统计
        self.component_grad_norms.clear()
        self.component_grad_stats.clear()

    def clear_hooks(self) -> None:
        """Remove all registered hooks"""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def __del__(self):
        """Cleanup hooks on deletion"""
        self.clear_hooks()


class GradientStatisticsTracker:
    """Track gradient statistics over time

    Aggregates gradient statistics across multiple steps.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history: List[float] = []

    def add(self, norm: float) -> None:
        """Add a gradient norm value"""
        self.history.append(norm)
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def get_statistics(self) -> Dict[str, float]:
        """Get statistics over the window

        Returns:
            Dictionary of statistics
        """
        if not self.history:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        return {
            "mean": sum(self.history) / len(self.history),
            "std": (sum((x - sum(self.history)/len(self.history))**2 for x in self.history) / len(self.history)) ** 0.5,
            "min": min(self.history),
            "max": max(self.history),
        }

    def reset(self) -> None:
        """Clear history"""
        self.history.clear()


__all__ = [
    "GradientMonitor",
    "GradientStatisticsTracker",
    "EnhancedGradientMonitor",
]


class EnhancedGradientMonitor:
    """增强梯度监控器 - 实时追踪 R1-R8 风险指标

    设计说明:
    - 使用 register_full_backward_hook 被动监听梯度
    - entmax_sparsity 从 TrainingStats 获取（不依赖 model.last_probs）
    - 计算开销约 3-5%

    使用方式:
        from training.monitor.gradient_monitor import EnhancedGradientMonitor

        monitor = EnhancedGradientMonitor(model)

        # 训练循环中
        loss.backward()
        monitor.check_from_stats(outputs)  # outputs 是 TrainingStats

        if monitor.should_rollback(threshold_sparsity=0.3):
            model.load_state_dict(saved_state)
    """

    def __init__(
        self,
        model,
        warn_ratio: float = 1e-6,
        crit_ratio: float = 1e-1,
    ):
        """
        Args:
            model: 要监控的模型
            warn_ratio: 梯度消失警告阈值 (grad/weight ratio)
            crit_ratio: 梯度爆炸严重阈值 (grad/weight ratio)
        """
        from typing import Any

        self.model = model
        self.warn_ratio = warn_ratio
        self.crit_ratio = crit_ratio
        self.layer_norms: Dict[str, List[float]] = {}
        # I-OPT: 延迟 .item() - backward 期间存储 tensor，最后批量转换
        self._pending_grad_norms: Dict[str, List[torch.Tensor]] = {}
        self._pending_grad_ratios: List[torch.Tensor] = []  # 存储 ratio tensor
        self.hooks: List[Any] = []
        self._register_hooks()

    def _register_hooks(self):
        """注册关键模块的梯度监控"""
        for name, module in self.model.named_modules():
            if len(list(module.parameters())) > 0:
                h = module.register_full_backward_hook(self._make_hook(name))
                self.hooks.append(h)

    def _make_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            if grad_output is None or len(grad_output) == 0:
                return
            grad = grad_output[0]
            if not isinstance(grad, torch.Tensor) or grad.numel() == 0:
                return

            # I-OOM FIX: detach() 断开梯度图，防止显存泄漏
            self._pending_grad_norms.setdefault(name, []).append(grad.norm().detach())

            # I-OPT: 延迟 ratio 计算，改为存储 tensor 在 finalize 时检查
            # 原实现: 在 hook 内调用 named_parameters() + .norm() + .item() 多次同步
            for n, p in module.named_parameters():
                if p.grad is not None:
                    ratio = p.grad.norm() / (p.norm() + 1e-8)
                    self._pending_grad_ratios.append((name, n, ratio.detach()))

        return hook

    def finalize(self) -> None:
        """I-OPT: 在 backward 结束后调用，批量转换 pending tensor norms

        在训练循环中的正确使用位置:
            loss.backward()
            monitor.finalize()  # <-- 在这里调用
            report = monitor.get_report()

        批量 .item() 的优势：
        - 将所有 GPU-CPU 同步合并执行
        - 避免在 backward hook 内直接调用 .item()
        """
        # 转换 norm tensors
        for name, norm_tensors in self._pending_grad_norms.items():
            for t in norm_tensors:
                self.layer_norms.setdefault(name, []).append(t.item())
        self._pending_grad_norms.clear()

        # 处理 vanishing/exploding ratio 检查
        for name, n, ratio_tensor in self._pending_grad_ratios:
            ratio_val = ratio_tensor.item()  # 批量 .item()，GPU 计算已重叠
            if ratio_val < self.warn_ratio:
                print(f"[VANISH] {name}.{n}: ratio={ratio_val:.2e}")
            elif ratio_val > self.crit_ratio:
                print(f"[EXPLODE] {name}.{n}: ratio={ratio_val:.2e}")
        self._pending_grad_ratios.clear()

    def check_from_stats(self, stats) -> None:
        """从 TrainingStats 获取 entmax 稀疏度

        Args:
            stats: TrainingStats 对象或类似结构，需有 split_probs 属性
        """
        if hasattr(stats, 'split_probs') and stats.split_probs is not None:
            zero_ratio = (stats.split_probs < 1e-4).float().mean().item()
            self.layer_norms.setdefault('entmax_sparsity', []).append(zero_ratio)

    def get_report(self) -> dict:
        """生成梯度健康报告

        Returns:
            包含各模块梯度统计的字典
        """
        import numpy as np

        report = {}
        for name, values in self.layer_norms.items():
            if len(values) == 0:
                continue
            arr = np.array(values[-100:])  # 只取最近100个样本
            report[name] = {
                'mean': float(arr.mean()),
                'std': float(arr.std()),
                'min': float(arr.min()),
                'max': float(arr.max()),
            }
        return report

    def should_rollback(self, threshold_sparsity: float = 0.3) -> bool:
        """基于稀疏度判断是否需要回滚

        Args:
            threshold_sparsity: 稀疏度阈值，超过则回滚

        Returns:
            True 如果应该回滚
        """
        import numpy as np

        if 'entmax_sparsity' in self.layer_norms:
            recent = np.array(self.layer_norms['entmax_sparsity'][-10:])
            if len(recent) > 0 and recent.mean() > threshold_sparsity:
                print(f"[ROLLBACK] Entmax sparsity {recent.mean():.2%} > {threshold_sparsity:.2%}")
                return True
        return False

    def remove_hooks(self):
        """移除所有注册的 hooks"""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
