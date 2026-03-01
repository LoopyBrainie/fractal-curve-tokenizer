"""Gradient Flow Monitoring Module

Provides detailed gradient flow monitoring for all model layers.
Tracks gradient norms for analysis and debugging.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set
import torch
import torch.nn as nn
from collections import defaultdict


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

    # I150-3: 关键组件名称关键字（用于识别特定模块的梯度）
    KEY_COMPONENT_KEYWORDS = {
        "manifold_decoder": ["manifold", "decoder", "poincare", "hyperbolic"],
        "entmax": ["entmax", "sparsemax", "alpha_entmax"],
        "splitter": ["splitter", "gumbel", "tokenizer", "selector"],
        "backbone": ["transformer", "encoder", "blocks", "attn", "mlp"],
        "embedding": ["embed", "patch_embed", "cls_token", "pos_embed"],
    }

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        record_layer_norms: bool = True,
        hooks_enabled: bool = True,
        track_key_components: bool = True,
    ):
        """Initialize GradientMonitor

        Args:
            model: The model to monitor
            record_layer_norms: Whether to record per-layer norms
            hooks_enabled: Whether to register backward hooks
            track_key_components: I150-3: 是否追踪关键组件（ManifoldDecoder等）
        """
        self.model = model
        self.record_layer_norms = record_layer_norms
        self.hooks_enabled = hooks_enabled
        self.track_key_components = track_key_components

        # Storage
        self.layer_norms: Dict[str, List[float]] = defaultdict(list)
        self.grad_norms_by_step: List[float] = []
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

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
                    norm = grad.norm().item()
                    self.layer_norms[name].append(norm)
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

        特别追踪:
        - ManifoldDecoder (流形解码器)
        - Entmax (稀疏注意力)
        - Splitter (分形分割器)
        - Backbone (Transformer主干)
        - Embedding (嵌入层)

        Returns:
            Dictionary of component name -> gradient norm
        """
        if self.model is None:
            return {}

        # 初始化组件梯度
        component_norms: Dict[str, float] = {
            "manifold_decoder": 0.0,
            "entmax": 0.0,
            "splitter": 0.0,
            "backbone": 0.0,
            "embedding": 0.0,
            "other": 0.0,
        }

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm().item()
                matched = False

                # 根据关键字匹配组件
                name_lower = name.lower()
                for comp_name, keywords in self.KEY_COMPONENT_KEYWORDS.items():
                    if any(kw in name_lower for kw in keywords):
                        component_norms[comp_name] += norm
                        matched = True
                        break

                if not matched:
                    component_norms["other"] += norm

        # 记录到统计中
        if self.track_key_components:
            for comp_name, norm in component_norms.items():
                if norm > 0:
                    self.component_grad_norms[comp_name].append(norm)

        return component_norms

    def get_component_grad_statistics(self) -> Dict[str, Dict[str, float]]:
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
]
