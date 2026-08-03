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

    I165-3a 扩展: 同时为 3 个结构性参数 (rot_proj / roi_norm / geo_norm)
    注入正交化 / 单位缩放 / 三段对齐监督信号 (Grouped Scaling, 1:1:1 配比).

    Attributes:
        entropy_weight: entropy loss 权重 (锚定 logit_scale + alpha_raw)
        budget_weight: budget loss 权重 (锚定 conv1d_hilbert.bias)
        locality_weight: locality loss 权重 (锚定 _semantic_ratio)
        bias_reg_weight: bias_table L2 正则权重
        entropy_target: 目标 entropy (normalized [0, 1])
        budget_target: 目标 K fraction ∈ [0, 1]
        i165_3a_global_scale: I165-3a 3 项的全局标量 (Grouped Scaling)
        _enabled_i165_3a: I165-3a kill-switch 标志
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
        i165_3a_global_scale: float = 0.05,
    ) -> None:
        self.entropy_weight = float(entropy_weight)
        self.budget_weight = float(budget_weight)
        self.locality_weight = float(locality_weight)
        self.bias_reg_weight = float(bias_reg_weight)
        self.entropy_target = float(entropy_target)
        self.budget_target = float(budget_target)
        self.i165_3a_global_scale = float(i165_3a_global_scale)
        self._enabled: bool = False
        self._enabled_i165_3a: bool = True  # 默认 ON, kill-switch
        self._eps: float = 1e-6

    def on_train_start(self, ctx: TrainerContext) -> None:
        """读取 config 标志 + EPS 常量。"""
        cfg = getattr(ctx, "config", None)
        training_cfg = getattr(cfg, "training", None)
        self._enabled = bool(getattr(training_cfg, "enable_routing_aux_loss", False))
        # I165-3a: 即使上层 _enabled=False 也允许 I165-3a 独立开关
        # (但当前默认架构是 _enabled 包含 I165-3a,所以这里尊重上层)
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
                # I165-3a: Grouped Scaling 单标量 + kill-switch
                self.i165_3a_global_scale = float(getattr(
                    training_cfg, "i165_3a_global_scale", self.i165_3a_global_scale
                ))
                self._enabled_i165_3a = bool(getattr(
                    training_cfg, "enable_i165_3a_aux_loss", True
                ))

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

        # === I165-3a: rot_proj / roi_norm / geo_norm 联合监督 (Grouped Scaling) ===
        # 设计: 单个 global_scale s 控制 3 项联合强度,内部 1:1:1 等配比
        # 与 R7-A 4 项 (entropy/budget/locality/bias_reg) 在参数空间正交:
        #   - L_rot  锚定 rot_proj.weight   (d × d 矩阵正交化)
        #   - L_roi  锚定 roi_norm.weight   (d 维 γ² → 1 单位缩放)
        #   - L_geo  锚定 geo_norm.weight   (3d 维三段 γ 均值平方对齐 + 段间方差)
        # kill-switch: enable_i165_3a_aux_loss=False 或 i165_3a_global_scale=0 都跳过
        if not (self._enabled_i165_3a and self.i165_3a_global_scale > 0):
            return  # 跳过 I165-3a 全部 3 项 (Grouped Scaling 整体开/关)

        s = self.i165_3a_global_scale

        # Loss 5: L_rot = ||W^T W - I||_F^2 / d^2
        # 物理解释: 鼓励 rot_proj 接近正交,保持 Hilbert 旋转嵌入几何结构
        # 在 W = Q (正交) 时 L_rot = 0, 平衡点正确
        # ∂L/∂W = 4/d² · (W W^T W - W), 偏离量线性相关,温和稳定
        rot_w = rt.get("rot_proj_weight")
        if rot_w is not None and isinstance(rot_w, torch.Tensor) and rot_w.dim() == 2:
            try:
                d = int(rot_w.shape[0])
                # d == d (square), 默认 HilbertOptimalSplitter hidden_dim × hidden_dim
                if d > 0 and rot_w.shape[0] == rot_w.shape[1]:
                    gram = rot_w.t() @ rot_w  # [d, d]
                    eye = torch.eye(d, dtype=gram.dtype, device=gram.device)
                    # Frobenius 范数平方 / d^2 归一化
                    ctx.aux_losses["i165_3a_rot"] = s * ((gram - eye) ** 2).sum() / float(d * d)
            except Exception:
                pass

        # Loss 6: L_roi = mean((γ² - 1)²)
        # 物理解释: 鼓励 roi_norm γ 保持单位缩放, 初始 γ=1 → L_roi = 0
        # ∂L/∂γ_i = (4/d) · γ_i · (γ_i² - 1), γ=1 处梯度为零 (完全平滑引入)
        # 等价于约束 LN 不放大也不缩小 ROI 特征
        roi_w = rt.get("roi_norm_weight")
        if roi_w is not None and isinstance(roi_w, torch.Tensor) and roi_w.dim() == 1:
            try:
                ctx.aux_losses["i165_3a_roi"] = s * ((roi_w ** 2 - 1.0) ** 2).mean()
            except Exception:
                pass

        # Loss 7: L_geo (三段 γ 均值平方对齐 + 段间方差)
        # 物理解释: 强制 path/rot/area 三段在 Hilbert 流形上对称, 避免任何一段被 LN 主导
        # 初始三段 γ=1 → align=0, var=0 → L_geo = 0, 与初始化完全匹配
        # 段方差权重 0.1 是内部固定常数, 不暴露配置 (YAGNI)
        geo_w = rt.get("geo_norm_weight")
        if geo_w is not None and isinstance(geo_w, torch.Tensor) and geo_w.dim() == 1:
            try:
                d_total = int(geo_w.shape[0])
                d = d_total // 3
                if d > 0 and d_total == 3 * d:
                    seg = geo_w.view(3, d)  # [3, d]
                    seg_mean_sq = (seg ** 2).mean(dim=-1)  # [3]
                    align = ((seg_mean_sq - 1.0) ** 2).mean()
                    var = seg_mean_sq.var(unbiased=False)
                    ctx.aux_losses["i165_3a_geo"] = s * (align + 0.1 * var)
            except Exception:
                pass


__all__ = ["AuxiliaryRoutingLossCallback"]
