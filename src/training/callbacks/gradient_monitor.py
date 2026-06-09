"""GradientMonitorCallback — 梯度监控 (PR3 of trainer refactor)

替换原 `src.training.monitor.gradient_monitor.GradientMonitor` (501 行,
包含 dead `EnhancedGradientMonitor` 类 + CF-3 `register_full_backward_hook`
调用)。

设计要点 (per Q3 + alpha-e-r2 + Oracle CF-3 强制):
  - 单一来源: 只使用 `register_post_accumulate_grad_hook`, 严禁
    `register_full_backward_hook` (CF-3 已修正)
  - TrainerCallback ABC 继承: `post_backward` hook 是骨架唯一调用点
  - I-OPT 保留: 延迟 .item() 转换, `finalize()` 批量 GPU→CPU 同步
  - 内存泄漏修复: `__del__` 显式清理 hook handles
  - SVD off (per Q3 + beta-A-r2 D4): `compute_geometry_spectral_norms` 删除
  - EnhancedGradientMonitor 整类删除 (dead code, 0 callers)

数学形式 (per-layer 梯度范数):
    norm_l = ||∇W_l||_2 = sqrt(Σ (∂L/∂W_l[i,j,...])²)
    total_norm = sqrt(Σ_l norm_l²)

I-OPT 关键:
  - backward 期间存储 grad.norm().detach() 的 tensor 引用 (不立即 .item())
  - finalize() 一次性将所有 tensor norms 转为 float, 合并 N 个 GPU→CPU 同步
  - 防止每 step 累积 tensor 引用导致显存泄漏 (REPLACE not EXTEND)

See:
  - plan: C:\\Users\\LamKo\\.claude\\plans\\fluffy-watching-turing.md §3 PR3
  - Oracle CF-3: gradient_monitor.py 旧 line 398 register_full_backward_hook
    已彻底删除, 单一来源 register_post_accumulate_grad_hook
  - F-X3: PR3 commit 显式 diff 删除 register_full_backward_hook 调用
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from .base import TrainerCallback

if TYPE_CHECKING:
    from .base import TrainerContext


class GradientMonitorCallback(TrainerCallback):
    """梯度监控 callback (替代旧 GradientMonitor)。

    通过 `register_post_accumulate_grad_hook` 在每个参数梯度累加后捕获
    范数, 暂存为 tensor, finalize() 阶段批量 .item() 转换。
    严禁使用 `register_full_backward_hook` (CF-3)。

    Example:
        monitor = GradientMonitorCallback(model, record_layer_norms=True)
        monitor.register_hooks(model)
        ...
        loss.backward()
        monitor.finalize()                  # 批量 GPU→CPU 同步
        stats = monitor.get_statistics()   # 读取最近缓存
        monitor.reset()                     # 清理 pending refs
    """

    priority: int = 10  # 监控类后置 hook, 在 NaNGuard 之后

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        record_layer_norms: bool = True,
        hooks_enabled: bool = True,
        log_interval: int = 100,
    ):
        """Initialize GradientMonitorCallback.

        Args:
            model: nn.Module (Optional; register_hooks() 时必须提供)
            record_layer_norms: 是否记录每层梯度范数
            hooks_enabled: 是否注册 hooks (False = 纯手动 record_gradients)
            log_interval: 日志间隔 (steps)
        """
        self.model = model
        self.record_layer_norms = record_layer_norms
        self.hooks_enabled = hooks_enabled
        self.log_interval = log_interval

        # I-OPT 存储: pending tensor norms (backward 期间写入, finalize 转 float)
        # 单一来源: register_post_accumulate_grad_hook (CF-3 强制)
        self._pending_grad_norms: Dict[str, List[torch.Tensor]] = defaultdict(list)
        # I-OPT 缓存: finalized 的 tensor 副本 (compute_* 在 finalize 后读取此处)
        # REPLACE not EXTEND (I-OOM FIX)
        self._cached_norm_tensors: Dict[str, List[torch.Tensor]] = defaultdict(list)
        # Float 形态的 layer_norms (供外部读取, 例如日志)
        self.layer_norms: Dict[str, List[float]] = defaultdict(list)
        # 总体梯度范数历史
        self.grad_norms_by_step: List[float] = []

        # Hook handles (显式管理, __del__ 清理)
        self._hooks: List[RemovableHandle] = []

    # === Hook 管理 (CF-3: 单一来源 register_post_accumulate_grad_hook) ===

    def register_hooks(self, model: nn.Module) -> None:
        """注册梯度 hook (在每个 parameter 上)。

        唯一来源: `param.register_post_accumulate_grad_hook`。
        严禁调用 `module.register_full_backward_hook` (CF-3)。
        """
        if not self.hooks_enabled:
            return

        # 清理旧 hooks (防累积)
        self.clear_hooks()

        def create_hook(name: str):
            def hook(grad: torch.Tensor) -> None:
                if grad is not None:
                    # I-OOM FIX: detach() 断开梯度图, 防止显存泄漏
                    self._pending_grad_norms[name].append(grad.norm().detach())
            return hook

        for name, param in model.named_parameters():
            if param.requires_grad:
                handle = param.register_post_accumulate_grad_hook(create_hook(name))
                self._hooks.append(handle)

    def clear_hooks(self) -> None:
        """移除所有已注册的 hooks (泄漏修复)。"""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def __del__(self):
        """析构时清理 hooks (修复原 __del__ 泄漏)。"""
        try:
            self.clear_hooks()
        except Exception:
            # 析构路径上不抛异常
            pass

    # === TrainerCallback hook implementations ===

    def post_backward(self, ctx: "TrainerContext") -> None:
        """backward() 之后调用, finalize pending tensor norms。

        这是 PR5 骨架中的调用点, 替代原 `monitor.finalize()` 显式调用。
        """
        self.finalize()

    # === 数据流 (I-OPT 核心) ===

    def finalize(self) -> None:
        """在 backward 结束后调用, 批量 GPU→CPU 同步。

        将 _pending_grad_norms 中的 tensor norms 缓存到 _cached_norm_tensors,
        并写入 layer_norms (float 形式供外部读取)。
        REPLACE not EXTEND (I-OOM FIX): 防止每 step 累积 tensor 引用。

        使用方式:
            loss.backward()
            monitor.finalize()        # <-- 在这里调用
            stats = monitor.get_statistics()  # finalize 后读取
            monitor.reset()           # 下一步前清理
        """
        # REPLACE not EXTEND: 清空缓存, 然后从 pending 复制
        self._cached_norm_tensors.clear()
        for name, norm_tensors in self._pending_grad_norms.items():
            # 复制 list (避免 finalize 后清空 pending 影响 cache)
            self._cached_norm_tensors[name] = list(norm_tensors)
        self._pending_grad_norms.clear()

        # 转换为 float 写入 layer_norms (供外部日志读取)
        for name, norm_tensors in self._cached_norm_tensors.items():
            for t in norm_tensors:
                if isinstance(t, torch.Tensor):
                    self.layer_norms[name].append(t.item())
                else:
                    # 已转 float (defensive)
                    self.layer_norms[name].append(float(t))

        # 记录总体梯度范数
        total = self.compute_total_grad_norm()
        self.grad_norms_by_step.append(total)
        # 限制历史长度 (防 O(N) 累积)
        if len(self.grad_norms_by_step) > 1000:
            self.grad_norms_by_step = self.grad_norms_by_step[-1000:]

    def reset(self) -> None:
        """清理 pending refs (每 step 末尾调用, 防显存泄漏)。

        注意: `layer_norms` (float 列表) 保留以供外部日志, 不在此清空。
        """
        self._pending_grad_norms.clear()
        self._cached_norm_tensors.clear()

    # === 统计读取 (finalize 后) ===

    def compute_grad_norms(self) -> Dict[str, float]:
        """计算每层当前梯度范数 (从 _cached_norm_tensors 读, finalize 后)。

        Returns:
            dict: {param_name: avg_norm_in_step}
        """
        norms: Dict[str, float] = {}
        for name, norm_tensors in self._cached_norm_tensors.items():
            if norm_tensors:
                norms[name] = sum(t.item() for t in norm_tensors) / len(norm_tensors)
        return norms

    def compute_total_grad_norm(self) -> float:
        """计算总梯度范数 (用于 grad clip)。

        Returns:
            float: ||∇W||_total = sqrt(Σ_l norm_l²)
        """
        total_sq = 0.0
        for norm_tensors in self._cached_norm_tensors.values():
            for t in norm_tensors:
                total_sq += t.item() ** 2
        return total_sq ** 0.5

    def get_layer_statistics(self) -> Dict[str, Dict[str, float]]:
        """每层累积历史统计 (mean, std, min, max, count)。

        Returns:
            dict: {layer: {mean, std, min, max, count}}
        """
        stats: Dict[str, Dict[str, float]] = {}
        for name, norms in self.layer_norms.items():
            if norms:
                mean = sum(norms) / len(norms)
                variance = sum((n - mean) ** 2 for n in norms) / len(norms)
                stats[name] = {
                    "mean": mean,
                    "std": variance ** 0.5,
                    "min": min(norms),
                    "max": max(norms),
                    "count": len(norms),
                }
        return stats

    def get_top_k_norms(self, k: int = 10) -> List[tuple]:
        """返回梯度范数 top-k 层 (降序)。

        Args:
            k: 返回数量

        Returns:
            list of (name, norm) tuples, sorted desc
        """
        current_norms = self.compute_grad_norms()
        sorted_norms = sorted(current_norms.items(), key=lambda x: x[1], reverse=True)
        return sorted_norms[:k]

    def detect_gradient_issues(self) -> Dict[str, List[str]]:
        """检测常见梯度问题 (NaN / Inf / vanishing / exploding / zero)。

        Returns:
            dict: {issue_type: [layer_names]}
        """
        issues: Dict[str, List[str]] = {
            "vanishing": [],
            "exploding": [],
            "nan": [],
            "zero": [],
        }
        if self.model is None:
            return issues

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad
            if torch.isnan(grad).any():
                issues["nan"].append(name)
            elif torch.isinf(grad).any():
                issues["exploding"].append(name)
            else:
                norm = grad.norm().item()
                if norm < 1e-7:
                    issues["vanishing"].append(name)
                elif norm > 100:
                    issues["exploding"].append(name)
                elif norm == 0:
                    issues["zero"].append(name)

        return issues


__all__ = [
    "GradientMonitorCallback",
]
