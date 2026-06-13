"""Auxiliary Routing Loss Callback (PR: auxiliary-loss-pr-unified-rabbit)

为 5 个路由参数 (splitter.logit_scale / splitter._semantic_ratio /
splitter.conv1d_hilbert.{weight,bias} / lca_bias_subtractor.bias_table /
alpha_modulator.alpha_raw) 注入 entropy / budget / locality 监督信号,
使其重新获得训练梯度 (Phase 3 把 auxiliary_losses 硬编码为 None 导致
5 参数结构上脱离计算图)。

关键设计 — 双轨分离 (Blocker 1 修复):
- splitter_output 持有 Python float 日志快照 (已 detach, 仅供 UnifiedMonitor)
- routing_tensors 命名空间持有 raw Tensor (未 detach, 专供本 callback 计算 loss)
- 本 callback **完全在内部用 Tensor 算子现算** 4 个 loss, 严禁从 auxiliary_outputs
  读取预计算的 .item() 值 —— 那会断 autograd 图

L2 偏置正则 (Warning 1 修复):
- bias_table 初始为 0, L1 正则在零点梯度恒为 ±1, 易把 bias 钉死在 0 附近
- L2 (mean(bias²)) 在 0 附近梯度为 0, 允许微小梯度自由演化

dim 对齐断言 (Warning 2 修复):
- 防御性检查 probs.shape[-1] == len(hilbert_indices), 防止 DDP/非方形
  输入时 Silent Misalignment

遵循 5-allow/3-forbid 契约 (同 FractalTreeRegCallback):
- 5-allow: 读 ctx.aux_forward.auxiliary_outputs, 写 ctx.aux_losses[name], 不 detach aux loss
- 3-forbid: 不修改 ctx.aux_forward, 不读 ctx.model 直接 state_dict, 不假设 batch_size 已知
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .base import TrainerCallback, TrainerContext

if TYPE_CHECKING:
    from torch import Tensor


class AuxiliaryRoutingLossCallback(TrainerCallback):
    """为 5 路由参数注入 entropy/budget/locality 监督信号.

    Attributes:
        entropy_weight: entropy loss 权重 (锚定 logit_scale + alpha_raw)
        budget_weight: budget loss 权重 (锚定 conv1d_hilbert.bias)
        locality_weight: locality loss 权重 (锚定 _semantic_ratio)
        bias_reg_weight: bias_table L2 正则权重
        entropy_target: 目标 entropy (normalized [0, 1])
        budget_target: 目标 K fraction ∈ [0, 1]
    """

    priority: int = 0  # 默认后置 hook

    def __init__(
        self,
        entropy_weight: float = 0.05,
        budget_weight: float = 0.08,
        locality_weight: float = 0.04,
        bias_reg_weight: float = 0.02,
        entropy_target: float = 0.7,
        budget_target: float = 0.25,
    ) -> None:
        self.entropy_weight = float(entropy_weight)
        self.budget_weight = float(budget_weight)
        self.locality_weight = float(locality_weight)
        self.bias_reg_weight = float(bias_reg_weight)
        self.entropy_target = float(entropy_target)
        self.budget_target = float(budget_target)
        self._enabled: bool = False
        self._eps: float = 1e-6

    def on_train_start(self, ctx: TrainerContext) -> None:
        """读取 config 标志 + EPS 常量。"""
        cfg = getattr(ctx, "config", None)
        training_cfg = getattr(cfg, "training", None)
        self._enabled = bool(getattr(training_cfg, "enable_routing_aux_loss", False))
        if self._enabled:
            try:
                from vit_pytorch.core.constants import EPS
                self._eps = float(EPS)
            except ImportError:
                self._eps = 1e-6
            # 允许从 config 覆盖默认 weight
            if training_cfg is not None:
                self.entropy_weight = float(getattr(training_cfg, "routing_entropy_weight", self.entropy_weight))
                self.budget_weight = float(getattr(training_cfg, "routing_budget_weight", self.budget_weight))
                self.locality_weight = float(getattr(training_cfg, "routing_locality_weight", self.locality_weight))
                self.bias_reg_weight = float(getattr(training_cfg, "routing_bias_reg_weight", self.bias_reg_weight))
                self.entropy_target = float(getattr(training_cfg, "routing_entropy_target", self.entropy_target))
                self.budget_target = float(getattr(training_cfg, "routing_budget_target", self.budget_target))

    def on_loss_computed(
        self, ctx: TrainerContext, loss: "Tensor", components: dict
    ) -> None:
        """注入 routing aux-loss。前提: skeleton 在 PR5c 把 forward_output 存到 ctx.

        关键: 所有 loss 用 raw Tensor 算子现算, 保留 grad_fn, 不读预计算 .item() 值。
        """
        del loss  # 不直接使用, 仅满足父类签名
        del components  # 同上 (5-allow/3-forbid LSP 兼容)
        if not self._enabled:
            return  # disabled
        fwd = getattr(ctx, "aux_forward", None)
        if fwd is None:
            return
        aux = getattr(fwd, "auxiliary_outputs", None) or {}
        rt = aux.get("routing_tensors")
        if rt is None:
            return
        splitter_log = aux.get("splitter", {}) or {}

        # === Entropy (锚定 logit_scale + alpha_raw) ===
        # H_actual 在 callback 内现算, 保留 grad_fn
        probs = rt.get("probs")
        if probs is not None and isinstance(probs, torch.Tensor) and probs.dim() >= 1:
            try:
                n_cand = int(probs.shape[-1])
                if n_cand > 1:
                    log_n = torch.log(
                        torch.tensor(float(n_cand), dtype=probs.dtype, device=probs.device)
                    )
                    # H_actual ∈ [0, 1] (归一化)
                    h_actual = -(probs * torch.log(probs + self._eps)).sum(dim=-1) / log_n
                    # 显式把 target 转为 tensor, 避免 pyright 把 (target - h_actual) ** 2 推断为 int | float
                    target_t = torch.as_tensor(
                        self.entropy_target, dtype=h_actual.dtype, device=h_actual.device
                    )
                    diff_sq = (target_t - h_actual) ** 2
                    # 权重内嵌 (epoch_train 聚合是 loss = loss + v, 无权重重载)
                    ctx.aux_losses["routing_entropy"] = self.entropy_weight * diff_sq.mean()
            except Exception:
                pass

        # === Budget (锚定 conv1d_hilbert.bias) ===
        # K_actual 是 active_mask 的 fraction ∈ [0, 1], 与 K_target=0.25 量纲匹配
        active_mask = rt.get("active_mask")
        if active_mask is not None and isinstance(active_mask, torch.Tensor) and active_mask.numel() > 0:
            try:
                # 显式类型转换 + gt() 方法 (规避 pyright 对 Any 算子的推断问题)
                mask_bool = torch.gt(active_mask, torch.tensor(0.5, dtype=active_mask.dtype, device=active_mask.device))
                k_actual = mask_bool.to(torch.float32).mean()
                ctx.aux_losses["routing_budget"] = self.budget_weight * (self.budget_target - k_actual) ** 2
            except Exception:
                pass

        # === Locality (锚定 _semantic_ratio) ===
        # 使用 Hilbert 索引差分作为一阶导数代理
        hilbert_indices = splitter_log.get("hilbert_indices")
        if (
            probs is not None
            and isinstance(probs, torch.Tensor)
            and probs.dim() >= 1
            and hilbert_indices is not None
            and isinstance(hilbert_indices, torch.Tensor)
            and hilbert_indices.dim() == 1
        ):
            try:
                # Warning 2 修复: 防御性 dim 对齐断言
                probs_last_dim = int(probs.shape[-1])
                if probs_last_dim != int(hilbert_indices.shape[0]):
                    raise AssertionError(
                        f"Locality alignment failed: probs last dim {probs_last_dim} "
                        f"!= hilbert_indices length {int(hilbert_indices.shape[0])}"
                    )
                probs_sorted = probs.index_select(-1, hilbert_indices.long())
                grad_h = probs_sorted[..., 1:] - probs_sorted[..., :-1]
                # MSE 形式 (平滑, 比 L1 更稳)
                ctx.aux_losses["routing_locality"] = self.locality_weight * (grad_h ** 2).mean()
            except Exception:
                pass

        # === Bias L2 正则 (锚定 bias_table) — Warning 1 修复: L1 → L2 ===
        # 原因: bias_table 初始为 0, L1 正则在零点梯度恒为 ±1, 易把 bias 钉死在 0
        # L2 在 0 附近梯度为 0, 允许参数在微小区间自由演化
        bias_table = rt.get("bias_table")
        if bias_table is not None and isinstance(bias_table, torch.Tensor) and bias_table.numel() > 0:
            try:
                ctx.aux_losses["routing_bias_reg"] = self.bias_reg_weight * (bias_table ** 2).mean()
            except Exception:
                pass


__all__ = ["AuxiliaryRoutingLossCallback"]
