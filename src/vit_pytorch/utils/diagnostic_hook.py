"""
NaN 诊断钩子 (DiagnosticHook)

用于精确定位 Fractal ViT 训练过程中 NaN 产生位置的诊断工具。

功能
----
1. 监控每个 Transformer Block forward 后的有限性
2. 记录 ManifoldDecoder 输出的 manifold_bias 统计信息
3. 监控 Entmax 输入 Logits，检测极端值和零概率输出

用法
----
hook = DiagnosticHook()
model = hook.register_hooks(model)

# 在训练循环中获取诊断报告
if hook.has_nan_detected():
    report = hook.get_diagnostic_report()
    print(report)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LayerDiagnostics:
    """单层诊断结果"""
    layer_idx: int
    layer_name: str

    # 有限性检查
    is_finite: bool = True
    has_nan: bool = False
    has_inf: bool = False

    # 输出统计
    output_mean: float = 0.0
    output_std: float = 0.0
    output_min: float = 0.0
    output_max: float = 0.0

    # Manifold Bias 统计
    manifold_bias_max: float = 0.0
    manifold_bias_min: float = 0.0
    manifold_bias_mean: float = 0.0

    # Entmax 诊断
    entmax_input_max: float = 0.0
    entmax_input_min: float = 0.0
    entmax_input_std: float = 0.0
    entmax_output_all_zero: bool = False


@dataclass
class DiagnosticReport:
    """完整诊断报告"""
    batch_idx: int
    total_layers: int
    layers: List[LayerDiagnostics] = field(default_factory=list)

    # 汇总信息
    first_nan_layer: Optional[int] = None
    first_nan_layer_name: Optional[str] = None
    nan_propagation_path: List[str] = field(default_factory=list)

    # Splitter 诊断
    splitter_logits_max: float = 0.0
    splitter_logits_min: float = 0.0
    splitter_probs_all_zero: bool = False

    # 建议
    suggestions: List[str] = field(default_factory=list)


class DiagnosticHook:
    """
    NaN 诊断钩子

    通过注册到模型的各层来监控数值稳定性问题。
    """

    def __init__(
        self,
        enabled: bool = True,
        log_frequency: int = 10,
        stop_on_first_nan: bool = True,
    ):
        """
        参数
        ----
        enabled: 是否启用诊断
        log_frequency: 每隔多少 batch 输出一次统计信息
        stop_on_first_nan: 检测到首个 NaN 后是否停止训练
        """
        self.enabled = enabled
        self.log_frequency = log_frequency
        self.stop_on_first_nan = stop_on_first_nan

        # 诊断状态
        self._has_nan = False
        self._batch_count = 0
        self._layer_hooks: List[Any] = []
        self._diagnostics: Dict[str, LayerDiagnostics] = {}

        # Splitter 状态
        self._splitter_logits: Optional[torch.Tensor] = None
        self._splitter_probs: Optional[torch.Tensor] = None

        # 原始 forward 方法缓存
        self._original_forwards: Dict[str, Any] = {}

    def has_nan_detected(self) -> bool:
        """是否检测到 NaN"""
        return self._has_nan

    def get_diagnostic_report(self) -> Optional[DiagnosticReport]:
        """获取诊断报告"""
        if not self._diagnostics:
            return None

        # 找到第一个出现 NaN 的层
        first_nan_layer = None
        first_nan_layer_name = None
        for name, diag in self._diagnostics.items():
            if diag.has_nan:
                first_nan_layer = diag.layer_idx
                first_nan_layer_name = diag.layer_name
                break

        # 构建报告
        report = DiagnosticReport(
            batch_idx=self._batch_count,
            total_layers=len(self._diagnostics),
            layers=list(self._diagnostics.values()),
            first_nan_layer=first_nan_layer,
            first_nan_layer_name=first_nan_layer_name,
        )

        # 添加建议
        if first_nan_layer is not None:
            report.suggestions.append(
                f"NaN 首次出现在层 {first_nan_layer} ({first_nan_layer_name})"
            )
            if "manifold" in first_nan_layer_name.lower():
                report.suggestions.append(
                    "建议检查 GeometricLatentDecoder 中的 poincare_distance 函数"
                )
            if "attention" in first_nan_layer_name.lower():
                report.suggestions.append(
                    "建议检查注意力计算中的数值稳定性"
                )

        return report

    def register_hooks(self, model: nn.Module) -> nn.Module:
        """
        注册诊断钩子到模型

        参数
        ----
        model: FractalViT 模型实例

        返回
        ----
        添加了钩子的模型
        """
        if not self.enabled:
            return model

        # 注册 Transformer Block 钩子
        if hasattr(model, 'transformer'):
            self._register_transformer_hooks(model.transformer)

        # 注册 Splitter 钩子
        if hasattr(model, 'splitter'):
            self._register_splitter_hooks(model.splitter)

        # 注册 Tokenizer 钩子
        if hasattr(model, 'tokenizer'):
            self._register_tokenizer_hooks(model.tokenizer)

        logger.info(f"DiagnosticHook: 已注册 {len(self._layer_hooks)} 个钩子")

        return model

    def _register_transformer_hooks(self, transformer: nn.Module) -> None:
        """注册 Transformer 层钩子"""
        for idx, module in enumerate(transformer.modules()):
            if isinstance(module, (torch.nn.Module)):
                module_name = f"transformer_block_{idx}"

                # 保存原始 forward
                original_forward = module.forward
                self._original_forwards[module_name] = original_forward

                # 包装 forward
                def make_wrapper(idx, orig_forward, name):
                    def wrapper(self_, *args, **kwargs):
                        result = orig_forward(*args, **kwargs)
                        self._diagnose_layer_output(idx, name, result)
                        return result
                    return wrapper

                module.forward = make_wrapper(idx, original_forward, module_name)
                self._layer_hooks.append((module_name, module))

    def _register_splitter_hooks(self, splitter: nn.Module) -> None:
        """注册 Splitter 钩子 - 监控 Entmax 输入"""
        if hasattr(splitter, 'forward'):
            original_forward = splitter.forward

            def wrapper(*args, **kwargs):
                result = original_forward(*args, **kwargs)
                self._diagnose_splitter_output(result)
                return result

            splitter.forward = wrapper
            self._layer_hooks.append(("splitter", splitter))

    def _register_tokenizer_hooks(self, tokenizer: nn.Module) -> None:
        """注册 Tokenizer 钩子"""
        if hasattr(tokenizer, 'forward'):
            original_forward = tokenizer.forward

            def wrapper(*args, **kwargs):
                result = original_forward(*args, **kwargs)
                # Tokenizer 输出诊断
                if hasattr(result, 'features'):
                    features = result.features
                    self._diagnose_tensor("tokenizer_features", features)
                return result

            tokenizer.forward = wrapper
            self._layer_hooks.append(("tokenizer", tokenizer))

    def _diagnose_layer_output(
        self,
        idx: int,
        name: str,
        output: torch.Tensor,
    ) -> None:
        """诊断单层输出"""
        if not isinstance(output, torch.Tensor):
            return

        is_finite = torch.isfinite(output).all().item()
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()

        # 计算统计信息
        output_mean = output.float().mean().item()
        output_std = output.float().std().item()
        output_min = output.float().min().item()
        output_max = output.float().max().item()

        diag = LayerDiagnostics(
            layer_idx=idx,
            layer_name=name,
            is_finite=is_finite,
            has_nan=has_nan,
            has_inf=has_inf,
            output_mean=output_mean,
            output_std=output_std,
            output_min=output_min,
            output_max=output_max,
        )

        self._diagnostics[name] = diag

        # 检测到 NaN
        if has_nan and not self._has_nan:
            self._has_nan = True
            logger.warning(
                f"[DIAGNOSTIC] NaN 首次检测到: layer={name}, "
                f"output_stats=[mean={output_mean:.4f}, std={output_std:.4f}, "
                f"min={output_min:.4f}, max={output_max:.4f}]"
            )

    def _diagnose_splitter_output(self, result: Any) -> None:
        """诊断 Splitter 输出"""
        if hasattr(result, 'logits') and result.logits is not None:
            logits = result.logits
            self._splitter_logits = logits

            logits_max = logits.float().max().item()
            logits_min = logits.float().min().item()

            # 检查是否有有效梯度
            if hasattr(result, 'probs') and result.probs is not None:
                probs = result.probs
                self._splitter_probs = probs

                probs_all_zero = (probs.float().sum() < 1e-6).item()
                if probs_all_zero:
                    logger.warning(
                        f"[DIAGNOSTIC] Splitter probabilities 几乎全为 0! "
                        f"logits=[min={logits_min:.4f}, max={logits_max:.4f}]"
                    )

    def _diagnose_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """通用张量诊断"""
        if not isinstance(tensor, torch.Tensor):
            return

        is_finite = torch.isfinite(tensor).all().item()
        has_nan = torch.isnan(tensor).any().item()

        if has_nan and not self._has_nan:
            self._has_nan = True
            logger.warning(f"[DIAGNOSTIC] NaN 检测: {name}")

    def diagnose_manifold_bias(
        self,
        bias: torch.Tensor,
        layer_name: str = "unknown",
    ) -> None:
        """
        诊断 ManifoldDecoder 输出的 bias

        参数
        ----
        bias: GeometricLatentDecoder 输出的偏置张量 [B, H, N, N]
        layer_name: 层名称
        """
        if bias is None or not isinstance(bias, torch.Tensor):
            return

        bias_max = bias.float().max().item()
        bias_min = bias.float().min().item()
        bias_mean = bias.float().mean().item()

        # 记录到诊断
        key = f"manifold_bias_{layer_name}"
        if key in self._diagnostics:
            self._diagnostics[key].manifold_bias_max = bias_max
            self._diagnostics[key].manifold_bias_min = bias_min
            self._diagnostics[key].manifold_bias_mean = bias_mean

        # 检查极端值
        if bias_max > 100 or bias_min < -100:
            logger.warning(
                f"[DIAGNOSTIC] Manifold bias 极端值: layer={layer_name}, "
                f"max={bias_max:.4f}, min={bias_min:.4f}, mean={bias_mean:.4f}"
            )

    def diagnose_entmax_input(
        self,
        logits: torch.Tensor,
        layer_name: str = "unknown",
    ) -> None:
        """
        诊断 Entmax 输入的 Logits

        参数
        ----
        logits: Entmax 前的 logits 张量
        layer_name: 层名称
        """
        if logits is None or not isinstance(logits, torch.Tensor):
            return

        logits_max = logits.float().max().item()
        logits_min = logits.float().min().item()
        logits_std = logits.float().std().item()

        # 记录到诊断
        key = f"entmax_input_{layer_name}"
        if key in self._diagnostics:
            self._diagnostics[key].entmax_input_max = logits_max
            self._diagnostics[key].entmax_input_min = logits_min
            self._diagnostics[key].entmax_input_std = logits_std

        # 检查极端值（可能导致 Entmax 输出全 0）
        if logits_max - logits_min > 1000:
            logger.warning(
                f"[DIAGNOSTIC] Entmax input 极端值范围: layer={layer_name}, "
                f"max={logits_max:.4f}, min={logits_min:.4f}, std={logits_std:.4f}"
            )

    def log_statistics(self) -> None:
        """输出诊断统计信息"""
        if not self.enabled or self._batch_count % self.log_frequency != 0:
            return

        logger.info(f"[DIAGNOSTIC] Batch {self._batch_count} 统计:")

        for name, diag in self._diagnostics.items():
            if not diag.is_finite:
                logger.warning(
                    f"  {name}: NaN={diag.has_nan}, Inf={diag.has_inf}, "
                    f"output=[mean={diag.output_mean:.4f}, std={diag.output_std:.4f}]"
                )

    def reset(self) -> None:
        """重置诊断状态"""
        self._has_nan = False
        self._batch_count = 0
        self._diagnostics.clear()

    def increment_batch(self) -> None:
        """增加 batch 计数"""
        self._batch_count += 1
        self.log_statistics()


def create_diagnostic_hook(
    enabled: bool = True,
    log_frequency: int = 10,
) -> DiagnosticHook:
    """
    创建诊断钩子工厂函数

    参数
    ----
    enabled: 是否启用
    log_frequency: 日志输出频率

    返回
    ----
    DiagnosticHook 实例
    """
    return DiagnosticHook(
        enabled=enabled,
        log_frequency=log_frequency,
    )
