"""NaNDumpCallback — NaN 发生时调试信息取证 (PR2 of trainer refactor)

迁移自原 `src.training.monitor.numerical_defender.NaNAutoInvestigation`。
仅在 `--debug` 或 `--debug-nan-dump` 显式开启时挂入,默认不挂入
(避免每个 step 的 JSON dump I/O 开销)。

设计要点 (per Q3):
  - 实际 callback (在 ctx.callbacks 列表中)
  - 通过 ctx.should_skip_step 判定 NaN (NaNGuard 写入)
  - 默认 disable=True (只挂入, 不真的写文件), 需 args.debug=True 才真正 dump
  - 与 NaNGuard 解耦: NaNGuard 只判 skip, NaNDumpCallback 只记录

数学形式 (诊断规则, 迁移自 _diagnose_nan):
    R1: input_data.has_nan → DATA_NAN
    R2: classification logits NaN/Inf → CLASS_LOGITS_NAN / CLASS_LOGITS_INF
    R3: feature norm > 1e3 → FEATURE_NORM_HIGH (head 前需 LayerNorm)
    R4: amp_loss_scale ≤ 1.0 仍溢出 → AMP_SCALE_LOW (LR 或 init 问题)
    R5: 某 loss 项 > 1e6 → LOSS_EXPLODE (权重设置不当)
    R6: 默认 → GRAD_EXPLODE (LR 过高或 grad clip 不足)

See:
  - plan: C:\\Users\\LamKo\\.claude\\plans\\fluffy-watching-turing.md §3 PR2
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

# 运行时 import: TrainerCallback 是 NaNDumpCallback 的基类, 必须 module-level
from .base import TrainerCallback

if TYPE_CHECKING:
    from .base import TrainerContext


def _tensor_to_python(val: Any) -> Any:
    """递归将 tensor → Python 标量 (用于 JSON 序列化)。"""
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            return val.item()
        return val.detach().cpu().tolist()
    if isinstance(val, dict):
        return {k: _tensor_to_python(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_tensor_to_python(v) for v in val]
    return val


class NaNDumpCallback(TrainerCallback):
    """NaN 发生时自动收集调试信息。

    Args:
        model: nn.Module
        debug_dir: 调试输出目录
        enabled: 是否真的写文件 (默认 False — 仅当 args.debug=True 才启用)
        target_modules: 激活值收集目标模块名子串
    """

    priority: int = -10  # 早于普通 callback, 紧跟 NaNGuard 判定之后

    def __init__(
        self,
        model: torch.nn.Module,
        debug_dir: str = "experiments/debug",
        enabled: bool = False,
        target_modules: Optional[list] = None,
    ):
        self.model = model
        self.debug_dir = Path(debug_dir)
        self.enabled = enabled
        self.target_modules = target_modules or [
            "splitter", "manifold", "decoder", "entmax",
            "density", "mlp_head", "head", "classifier",
        ]

        # I-OOM FIX: 显式 .detach().cpu() 释放 GPU tensor 引用
        self._activation_hooks: list = []
        self._activation_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self.investigation_count: int = 0

        if self.enabled:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    # === TrainerCallback hook implementations ===

    def on_train_start(self, ctx: "TrainerContext") -> None:
        """训练开始时挂 forward hooks (仅 enabled 时)。"""
        if self.enabled:
            self._register_activation_hooks()

    def on_train_end(self, ctx: "TrainerContext") -> None:
        """训练结束时清理 hooks。"""
        self._remove_activation_hooks()

    def on_batch_end(self, ctx: "TrainerContext") -> None:
        """每个 batch 末尾检查 NaN, 若发生则 dump。

        通过 ctx.should_skip_step 判定 (由 NaNGuard 写入)。
        """
        if not self.enabled:
            return
        if not ctx.should_skip_step:
            return
        # ctx.should_skip_step == True → NaN 发生, 取证
        self.investigate(ctx)

    # === 调查与 dump ===

    def investigate(self, ctx: Any) -> Optional[Path]:
        """执行 NaN 调查并保存 JSON 报告 (核心取证函数)。

        ctx 可以是 TrainerContext (PR5 骨架传入) 或 SimpleNamespace
        (PR2 兼容层, epoch_train.py 旧调用方构造)。运行时走 duck typing
        (访问 ctx.epoch / ctx.global_step / ctx.metrics / ctx.loss_components /
        ctx.nan_guard)。

        Returns:
            Path: 报告文件路径, 若未启用则返回 None
        """
        if not self.enabled:
            return None

        self.investigation_count += 1

        # 收集梯度热力图
        gradient_heatmap = self._analyze_gradient_heatmap()
        first_nan_layer = self._find_first_nan_layer()
        param_ranges = self._analyze_param_ranges()

        # 收集激活值统计 (显式 detach 释放 GPU refs)
        activation_stats_serializable: Dict[str, Any] = {}
        for name, stats in self._activation_stats.items():
            activation_stats_serializable[name] = {
                k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else v
                for k, v in stats.items()
            }
        self._activation_stats.clear()  # I-SLOW FIX: 清空释放

        # 报告主体
        report = {
            "meta": {
                "epoch": ctx.epoch,
                "global_step": ctx.global_step,
                "investigation_count": self.investigation_count,
                "nan_guard_stats": ctx.nan_guard.get_stats() if ctx.nan_guard else {},
            },
            "diagnosis": self._diagnose(ctx),
            "gradient_heatmap": gradient_heatmap,
            "first_nan_layer": first_nan_layer,
            "param_ranges": param_ranges,
            "activation_stats": activation_stats_serializable,
        }

        # 序列化 (递归处理嵌套 tensor)
        report_serializable = _tensor_to_python(report)

        # 写文件
        filename = f"nan_snapshot_epoch_{ctx.epoch:04d}_step_{ctx.global_step:06d}.json"
        filepath = self.debug_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report_serializable, f, indent=2, ensure_ascii=False)

        return filepath

    # === 内部 helpers ===

    def _register_activation_hooks(self) -> None:
        """注册 forward hooks 收集激活值 (enabled 时)。"""
        # torch.compile 兼容性: 不在 OptimizedModule 上注册 hooks
        if hasattr(self.model, "_orig_mod"):
            return

        def make_hook(name: str):
            def hook(module, _input, output):
                t = output if isinstance(output, torch.Tensor) else None
                if t is None or t.numel() == 0:
                    return
                t = t.detach()
                if t.dim() > 2:
                    t_flat = t.flatten(start_dim=2)
                elif t.dim() == 2:
                    t_flat = t
                else:
                    t_flat = t.unsqueeze(0)
                # 存储为 tensor, get_stats 时再转 python
                self._activation_stats[name] = {
                    "mean": t_flat.mean(),
                    "std": t_flat.std(),
                    "min": t_flat.min(),
                    "max": t_flat.max(),
                    "norm": t_flat.norm(),
                }
            return hook

        for name, module in self.model.named_modules():
            if any(t in name.lower() for t in self.target_modules):
                h = module.register_forward_hook(make_hook(name))
                self._activation_hooks.append(h)

    def _remove_activation_hooks(self) -> None:
        """移除所有 activation hooks。"""
        for h in self._activation_hooks:
            h.remove()
        self._activation_hooks.clear()

    def _analyze_gradient_heatmap(self) -> list:
        """每层梯度范数 + NaN/Inf 标记 (sorted: 异常优先)。"""
        layers = []
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            has_nan = torch.isnan(grad).any().item()
            has_inf = torch.isinf(grad).any().item()
            grad_norm = float("nan") if (has_nan or has_inf) else grad.norm().item()
            param_min = param.detach().min().item()
            param_max = param.detach().max().item()
            layers.append({
                "layer_name": name,
                "grad_norm": grad_norm,
                "has_nan": has_nan,
                "has_inf": has_inf,
                "param_range": [param_min, param_max],
                "param_shape": list(param.shape),
            })
        # 异常在前
        layers.sort(
            key=lambda x: float("inf") if x["has_nan"] or x["has_inf"] else -x["grad_norm"]
        )
        return layers

    def _find_first_nan_layer(self) -> Optional[Dict[str, Any]]:
        """找到反向传播中最先 NaN 的层 (按 named_parameters 倒序扫描)。"""
        for name, param in reversed(list(self.model.named_parameters())):
            if param.grad is not None and torch.isnan(param.grad).any().item():
                grad = param.grad.detach()
                nan_count = torch.isnan(grad).sum().item()
                return {
                    "layer_name": name,
                    "grad_norm": grad.norm().item(),
                    "nan_count": nan_count,
                    "total_params": grad.numel(),
                    "nan_ratio": nan_count / grad.numel(),
                }
        return None

    def _analyze_param_ranges(self) -> Dict[str, list]:
        """每参数 [min, max, mean, std] (NaN 安全)。"""
        ranges = {}
        for name, param in self.model.named_parameters():
            p = param.detach()
            std_val = p.std().item() if p.numel() > 1 else 0.0
            ranges[name] = [
                p.min().item(),
                p.max().item(),
                p.mean().item(),
                std_val,
            ]
        return ranges

    def _diagnose(self, ctx: Any) -> Dict[str, Any]:
        """自动诊断 NaN 根因 (规则引擎)。

        Returns:
            dict: {"diagnosis": [...], "root_cause": "..."}
        """
        diagnosis: list = []
        root_cause = "unknown"

        # R1: input NaN (通过 ctx.metrics['train/inputs_has_nan'] 推断)
        if ctx.metrics.get("train/inputs_has_nan", False):
            diagnosis.append("[R1] INPUT_NAN: 输入数据包含 NaN")
            root_cause = "input_data"

        # R2: 分类 logits NaN/Inf
        logits_has_nan = ctx.metrics.get("train/logits_has_nan", False)
        logits_has_inf = ctx.metrics.get("train/logits_has_inf", False)
        if logits_has_nan:
            diagnosis.append("[R2a] CLASS_LOGITS_NAN")
            root_cause = "classification_logits_nan"
        if logits_has_inf:
            diagnosis.append("[R2b] CLASS_LOGITS_INF")
            root_cause = "classification_logits_inf"

        # R3: feature norm 异常
        feat_max = ctx.metrics.get("train/feature_max", 0.0)
        feat_has_nan = ctx.metrics.get("train/feature_has_nan", False)
        feat_has_inf = ctx.metrics.get("train/feature_has_inf", False)
        if feat_max > 1e3:
            diagnosis.append(f"[R3a] FEATURE_NORM_HIGH: max={feat_max:.2e} > 1e3")
            root_cause = "feature_norm_high"
        if feat_has_nan:
            diagnosis.append("[R3b] FEATURE_NAN")
            root_cause = "feature_nan"
        if feat_has_inf:
            diagnosis.append("[R3c] FEATURE_INF")
            root_cause = "feature_inf"

        # R4: AMP scale 低
        amp_scale = ctx.metrics.get("train/amp_loss_scale")
        if amp_scale is not None and amp_scale <= 1.0:
            diagnosis.append(f"[R4] AMP_SCALE_LOW: scale={amp_scale:.1f}")
            if root_cause == "unknown":
                root_cause = "amp_scale_low"

        # R5: loss 爆炸
        for name, value in ctx.loss_components.items():
            if name == "total" or not isinstance(value, (int, float)):
                continue
            if abs(value) > 1e6:
                diagnosis.append(f"[R5] LOSS_EXPLODE: {name}={value:.2e}")
                root_cause = "loss_explode"

        # R6: 默认 (梯度爆炸)
        if root_cause == "unknown":
            diagnosis.append("[R6] GRAD_EXPLODE: 输入/特征/loss 均正常但梯度 NaN")
            root_cause = "grad_explode"

        return {"diagnosis": diagnosis, "root_cause": root_cause}


__all__ = [
    "NaNDumpCallback",
]
