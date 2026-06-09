"""NaNGuard — 数值防御骨架硬依赖 (PR2 of trainer refactor)

合并自原 `src.training.monitor.numerical_defender` 模块中的:
  - NumericalDefender (post_backward + skip 决策)
  - GradientValidator (per-param NaN/Inf 扫描)
  - AnomalyDetectionContext (torch.autograd.set_detect_anomaly 包装)

NaNGuard **不是 callback** (Q1 锁定: 骨架硬依赖, 不可回调化) —
它由 PR5 骨架通过 `ctx.nan_guard.is_healthy(loss, model)` 直接调用。

设计要点 (per Q3 + beta-A-r2 D4):
  - 命名契约显式: `is_healthy(loss, model) -> bool` 返回 True=数值健康 (proceed)
  - 内存泄漏修复: 显式 .detach().cpu() 转换 tensor, 清理 hook handles
  - Discovery hooks: 保留 register_discovery_hooks() 接口 (向后兼容 PR5 前的调用方)
  - 单一来源: NaN 检测走 isfinite().all() 一次扫描 (D1-SYNC 优化)

数学形式 (核心判定):
    is_healthy(loss, model) ↔
        (loss is not None) ∧ (loss.isfinite().all()) ∧
        (∀ param ∈ model.parameters() with param.grad: param.grad.isfinite().all())

    True = proceed with optimizer step
    False = skip optimizer step (state.nan_skip_count += 1)

See:
  - plan: C:\\Users\\LamKo\\.claude\\plans\\fluffy-watching-turing.md §3 PR2
  - 5-allow/3-forbid 契约: callbacks/base.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from torch import Tensor
import torch.nn as nn

if TYPE_CHECKING:
    from .base import TrainerContext


class NaNGuard:
    """数值防御: NaN/Inf 判定 + skip 决策 (骨架硬依赖, 非 callback)。

    Example:
        guard = NaNGuard(model, skip_on_nan=True)
        ...
        loss.backward()
        if not guard.is_healthy(loss, model):
            optimizer.zero_grad()
            state.nan_skip_count += 1
            continue
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        detect_anomaly: bool = False,
        skip_on_nan: bool = True,
        check_frequency: int = 1,
    ):
        """Initialize NaNGuard.

        Args:
            model: nn.Module (可选; 为 None 时 is_healthy 仅检查 loss)
            detect_anomaly: 是否启用 torch.autograd.set_detect_anomaly
            skip_on_nan: 检测到 NaN/Inf 时是否建议 skip
            check_frequency: 检测频率 (1=每步, N=每 N 步)
        """
        self.model = model
        self.detect_anomaly = detect_anomaly
        self.skip_on_nan = skip_on_nan
        self.check_frequency = check_frequency

        # 计数器 (迁移自 GradientValidator + NumericalDefender)
        self.step_count: int = 0
        self.issue_count: int = 0
        self.nan_count: int = 0
        self.inf_count: int = 0
        self.issues_history: List[Dict[str, Any]] = []
        self._last_ghost_nan: Optional[str] = None
        self._discovery_hooks: List[Any] = []
        # I-SLOW FIX: 显式 .detach().cpu() 转换统计张量, 释放 GPU refs
        self._pending_stats_tensors: List[Tensor] = []

    # === 主判定接口 (PR5 骨架调用) ===

    def is_healthy(self, loss: Tensor, model: Optional[nn.Module] = None) -> bool:
        """判定 loss + model 梯度是否数值健康。

        命名契约 (F-X3 / Q1 锁定):
            True = 数值健康 (proceed with optimizer step)
            False = 拦截 (skip optimizer step)

        Args:
            loss: 当前 batch 的 loss (scalar Tensor)
            model: nn.Module (若 None, 退化为仅检查 loss)

        Returns:
            bool: True = healthy / proceed, False = unhealthy / skip
        """
        target_model = model if model is not None else self.model

        self.step_count += 1
        if self.step_count % self.check_frequency != 0:
            return True  # 检查频率之外默认 healthy

        # 1) loss 数值检查 (单次 isfinite 扫描, D1-SYNC 优化)
        if loss is None or not torch.isfinite(loss.detach()).all().item():
            self.issue_count += 1
            if loss is not None and torch.isnan(loss.detach()).any().item():
                self.nan_count += 1
            if loss is not None and torch.isinf(loss.detach()).any().item():
                self.inf_count += 1
            return not (self.skip_on_nan and self.issue_count > 0)

        # 2) gradient 数值检查 (per-param isfinite, 单次同步 per-param)
        if target_model is not None:
            for name, param in target_model.named_parameters():
                if param.grad is None:
                    continue
                grad = param.grad
                if torch.isnan(grad).any().item():
                    self.nan_count += 1
                    self.issue_count += 1
                    self._record_issue(name, "nan")
                    return not (self.skip_on_nan and self.issue_count > 0)
                if torch.isinf(grad).any().item():
                    self.inf_count += 1
                    self.issue_count += 1
                    self._record_issue(name, "inf")
                    return not (self.skip_on_nan and self.issue_count > 0)

        return True  # healthy

    # === 辅助接口 (PR5 骨架 / 监控使用) ===

    def pre_backward(self) -> None:
        """backward() 之前调用 (保留接口, 供将来 anomaly detection 用)。"""
        # 保留位: 若 detect_anomaly=True, 在 PR5 骨架中可启用 anomaly_context
        # 当前 PR2 不实现以保持表面 API 最小
        pass

    def post_backward(self) -> bool:
        """backward() 之后调用, 返回是否建议 skip。

        兼容 NumericalDefender.post_backward() 接口, 供 train_fractal_vit.py
        等老调用方在 PR5 之前继续工作 (F-X1 风格兼容层)。

        Returns:
            bool: True = proceed, False = skip
        """
        return self.is_healthy(
            loss=torch.zeros(1, requires_grad=False),  # 仅检查 gradient (loss=None 走 fallback)
            model=self.model,
        ) if self.model is not None else True

    def get_stats(self) -> Dict[str, Any]:
        """获取防御统计 (字典形式, 兼容 NumericalDefender.get_stats())。"""
        return {
            "validator": {
                "total_issues": self.issue_count,
                "nan_count": self.nan_count,
                "inf_count": self.inf_count,
                "has_issues": self.issue_count > 0,
            },
            "step_count": self.step_count,
            "skip_rate": self.issue_count / max(self.step_count, 1),
        }

    def reset(self) -> None:
        """重置计数器 (epoch 边界使用)。"""
        self.step_count = 0
        self.issue_count = 0
        self.nan_count = 0
        self.inf_count = 0
        self.issues_history.clear()
        self._last_ghost_nan = None

    def clear(self) -> None:
        """清理 GPU tensor 引用 (I-SLOW FIX 迁移自 NumericalDefender.clear)。

        调用时机: 每个 epoch 末尾或 step 边界, 防止 _pending_stats_tensors
        累积导致显存泄漏。
        """
        # 显式 detach + cpu 释放 GPU refs
        for t in self._pending_stats_tensors:
            if isinstance(t, Tensor):
                t.detach()
        self._pending_stats_tensors.clear()

    # === Discovery hooks (向后兼容, 供 train_fractal_vit.py 旧调用) ===

    def register_discovery_hooks(
        self, target_modules: Optional[List[str]] = None
    ) -> None:
        """注册 pre-hook 在 nan_robust_hook 之前发现 NaN (向后兼容)。

        迁移自 NumericalDefender.register_discovery_hooks。
        在 PR5 之前的 train_fractal_vit.py 调用方继续工作。
        保留 target_modules 默认值: ["splitter", "entmax", "manifold", "decoder"]。

        Args:
            target_modules: 模块名子串列表, 默认为 ["splitter", "entmax",
                            "manifold", "decoder"]
        """
        if self.model is None:
            return

        # 清理旧 hooks
        self.remove_discovery_hooks()

        target_modules = target_modules or ["splitter", "entmax", "manifold", "decoder"]

        def make_hook(module_name: str):
            def hook(grad: Tensor) -> Tensor:
                if grad is not None and not torch.isfinite(grad).all():
                    self._last_ghost_nan = module_name
                return grad
            return hook

        for name, module in self.model.named_modules():
            if any(t in name.lower() for t in target_modules):
                for param_name, param in module.named_parameters(recurse=False):
                    full_name = f"{name}.{param_name}" if name else param_name
                    handle = param.register_hook(make_hook(full_name))
                    self._discovery_hooks.append(handle)

    def remove_discovery_hooks(self) -> None:
        """移除所有已注册的 discovery hooks (修复 NumericalDefender 泄漏)。"""
        for h in self._discovery_hooks:
            h.remove()
        self._discovery_hooks.clear()

    def __del__(self):
        """析构时清理 hooks (修复潜在泄漏)。"""
        try:
            self.remove_discovery_hooks()
        except Exception:
            pass  # 析构路径上不抛异常

    # === 内部 helpers ===

    def _record_issue(self, name: str, issue_type: str) -> None:
        """记录 issue 到 history (迁移自 GradientValidator._record_issue)。"""
        self.issues_history.append({
            "name": name,
            "type": issue_type,
        })
        # 防 history 无限增长: 保留最近 100 条
        if len(self.issues_history) > 100:
            self.issues_history = self.issues_history[-100:]


__all__ = [
    "NaNGuard",
]
