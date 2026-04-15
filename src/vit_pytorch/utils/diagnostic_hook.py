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
        self._needs_full_report = False
        self._batch_count = 0
        self._layer_hooks: List[Any] = []
        self._handles: List[Any] = []  # register_forward_hook 的句柄
        self._diagnostics: Dict[str, Optional[LayerDiagnostics]] = {}
        self._raw_diagnostics: Dict[str, Dict[str, Any]] = {}  # D1-SYNC: GPU tensors

        # Splitter 状态
        self._splitter_logits: Optional[torch.Tensor] = None
        self._splitter_probs: Optional[torch.Tensor] = None

        # 原始 forward 方法缓存
        self._original_forwards: Dict[str, Any] = {}

    def has_nan_detected(self) -> bool:
        """是否检测到 NaN"""
        return self._has_nan

    def get_diagnostic_report(self) -> Optional[DiagnosticReport]:
        """获取诊断报告 - D1-SYNC: 在此处统一做 .item()"""
        if not self._diagnostics and not self._raw_diagnostics:
            return None

        # D1-SYNC: 批量转换 - 只在报告生成时同步一次
        layers = []
        first_nan_layer = None
        first_nan_layer_name = None
        splitter_logits_max = 0.0
        splitter_logits_min = 0.0
        splitter_probs_all_zero = False

        for name, raw in self._raw_diagnostics.items():
            if 'layer_idx' in raw:
                # 层诊断 - LayerDiagnostics
                diag = LayerDiagnostics(
                    layer_idx=raw['layer_idx'],
                    layer_name=raw['layer_name'],
                    is_finite=raw['is_finite'].item(),
                    has_nan=raw['has_nan'].item(),
                    has_inf=raw['has_inf'].item(),
                    output_mean=raw['output_mean'].item(),
                    output_std=raw['output_std'].item(),
                    output_min=raw['output_min'].item(),
                    output_max=raw['output_max'].item(),
                )
                layers.append(diag)
                self._diagnostics[name] = diag  # 更新缓存

                if diag.has_nan and first_nan_layer is None:
                    first_nan_layer = diag.layer_idx
                    first_nan_layer_name = diag.layer_name
            elif name == 'splitter_logits':
                # Splitter logits 诊断
                splitter_logits_max = raw['logits_max'].item()
                splitter_logits_min = raw['logits_min'].item()
                # 检查 probs 是否全零
                if self._splitter_probs is not None:
                    splitter_probs_all_zero = (self._splitter_probs.float().sum() < 1e-6).item()
            elif name.startswith('manifold_bias_'):
                # Manifold bias 诊断
                if self._diagnostics.get(name) is None:
                    self._diagnostics[name] = LayerDiagnostics(layer_idx=-1, layer_name=name)
                self._diagnostics[name].manifold_bias_max = raw['bias_max'].item()
                self._diagnostics[name].manifold_bias_min = raw['bias_min'].item()
                self._diagnostics[name].manifold_bias_mean = raw['bias_mean'].item()
            elif name.startswith('entmax_input_'):
                # Entmax input 诊断
                if self._diagnostics.get(name) is None:
                    self._diagnostics[name] = LayerDiagnostics(layer_idx=-1, layer_name=name)
                self._diagnostics[name].entmax_input_max = raw['logits_max'].item()
                self._diagnostics[name].entmax_input_min = raw['logits_min'].item()
                self._diagnostics[name].entmax_input_std = raw['logits_std'].item()

        # 构建报告
        report = DiagnosticReport(
            batch_idx=self._batch_count,
            total_layers=len(self._diagnostics),
            layers=layers,
            first_nan_layer=first_nan_layer,
            first_nan_layer_name=first_nan_layer_name,
            splitter_logits_max=splitter_logits_max,
            splitter_logits_min=splitter_logits_min,
            splitter_probs_all_zero=splitter_probs_all_zero,
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

    def remove_hooks(self) -> None:
        """移除所有注册的 forward hooks"""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        logger.info("DiagnosticHook: 已移除所有钩子")

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
        """注册 Transformer 层钩子 - 使用 register_forward_hook 更安全"""
        # 使用 transformer.named_modules() 但只注册顶层 blocks 的 hook
        # 不替换 forward，只添加观察钩子
        for idx, module in enumerate(transformer.children()):
            # 跳过非模块
            if not isinstance(module, nn.Module):
                continue

            module_name = f"transformer_block_{idx}"

            # 使用 register_forward_hook 添加诊断
            # 这是 PyTorch 原生支持的，不会破坏模块内部调用
            def make_hook(idx, name):
                def hook(module, input, output):
                    # output 可能是 tensor 或 tuple
                    if isinstance(output, torch.Tensor):
                        self._diagnose_layer_output(idx, name, output)
                    elif isinstance(output, tuple):
                        # 取第一个元素（通常是 tensor）
                        for item in output:
                            if isinstance(item, torch.Tensor):
                                self._diagnose_layer_output(idx, name, item)
                                break
                return hook

            handle = module.register_forward_hook(make_hook(idx, module_name))
            self._handles.append(handle)
            self._layer_hooks.append((module_name, module))

    def _register_splitter_hooks(self, splitter: nn.Module) -> None:
        """注册 Splitter 钩子 - 使用 register_forward_hook"""
        # 使用 forward hook 而不是替换 forward 方法
        def hook(module, input, output):
            self._diagnose_splitter_output(output)

        handle = splitter.register_forward_hook(hook)
        self._handles.append(handle)
        self._layer_hooks.append(("splitter", splitter))

    def _register_tokenizer_hooks(self, tokenizer: nn.Module) -> None:
        """注册 Tokenizer 钩子 - 使用 register_forward_hook"""
        # 使用 forward hook 而不是替换 forward 方法
        def hook(module, input, output):
            # Tokenizer 输出诊断
            if hasattr(output, 'features'):
                features = output.features
                self._diagnose_tensor("tokenizer_features", features)

        handle = tokenizer.register_forward_hook(hook)
        self._handles.append(handle)
        self._layer_hooks.append(("tokenizer", tokenizer))

    def _diagnose_layer_output(
        self,
        idx: int,
        name: str,
        output: torch.Tensor,
    ) -> None:
        """诊断单层输出 - D1-SYNC: 延迟 .item() 到 get_diagnostic_report()"""
        if not isinstance(output, torch.Tensor):
            return

        # 收集 GPU tensor 统计量，不调用 .item() (避免同步)
        # 只做快速 NaN/Inf 检查（GPU tensor 比较）
        has_nan = torch.isnan(output).any()
        has_inf = torch.isinf(output).any()

        # 检测到 NaN/Inf 时立即记录（使用 GPU tensor，不等待 .item()）
        if has_nan and not self._has_nan:
            self._has_nan = True
            # 延迟到 get_diagnostic_report() 时再做 .item()
            # 但先标记需要报告
            self._needs_full_report = True

        # 存储原始 GPU tensors，在 get_diagnostic_report() 时统一 .item()
        self._raw_diagnostics[name] = {
            'layer_idx': idx,
            'layer_name': name,
            'is_finite': torch.isfinite(output).all(),
            'has_nan': has_nan,
            'has_inf': has_inf,
            'output_mean': output.float().mean(),
            'output_std': output.float().std(),
            'output_min': output.float().min(),
            'output_max': output.float().max(),
        }

        # 延迟到 get_diagnostic_report()
        if name not in self._diagnostics:
            self._diagnostics[name] = None  # placeholder

    def _diagnose_splitter_output(self, result: Any) -> None:
        """诊断 Splitter 输出 - D1-SYNC: 延迟 .item()"""
        if hasattr(result, 'logits') and result.logits is not None:
            logits = result.logits
            self._splitter_logits = logits

            # D1-SYNC: 存储 GPU tensor，在 get_diagnostic_report() 时 .item()
            self._raw_diagnostics['splitter_logits'] = {
                'logits_max': logits.float().max(),
                'logits_min': logits.float().min(),
            }

            # 检查是否有有效梯度
            if hasattr(result, 'probs') and result.probs is not None:
                probs = result.probs
                self._splitter_probs = probs

                probs_all_zero = probs.float().sum() < 1e-6
                # 检测到零概率时标记（GPU tensor check，不等待 .item()）
                if probs_all_zero and not self._has_nan:
                    self._has_nan = True
                    self._needs_full_report = True

    def _diagnose_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """通用张量诊断 - D1-SYNC: 延迟 .item()"""
        if not isinstance(tensor, torch.Tensor):
            return

        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()

        # 检测到 NaN 时标记（不立即 .item()）
        if has_nan and not self._has_nan:
            self._has_nan = True
            self._needs_full_report = True

    def diagnose_manifold_bias(
        self,
        bias: torch.Tensor,
        layer_name: str = "unknown",
    ) -> None:
        """
        诊断 ManifoldDecoder 输出的 bias - D1-SYNC: 延迟 .item()

        参数
        ----
        bias: GeometricLatentDecoder 输出的偏置张量 [B, H, N, N]
        layer_name: 层名称
        """
        if bias is None or not isinstance(bias, torch.Tensor):
            return

        # D1-SYNC: 存储 GPU tensors，在 get_diagnostic_report() 时 .item()
        key = f"manifold_bias_{layer_name}"
        self._raw_diagnostics[key] = {
            'bias_max': bias.float().max(),
            'bias_min': bias.float().min(),
            'bias_mean': bias.float().mean(),
        }
        if key not in self._diagnostics:
            self._diagnostics[key] = None  # placeholder

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

        # D1-SYNC: 存储 GPU tensors，在 get_diagnostic_report() 时 .item()
        key = f"entmax_input_{layer_name}"
        self._raw_diagnostics[key] = {
            'logits_max': logits.float().max(),
            'logits_min': logits.float().min(),
            'logits_std': logits.float().std(),
        }
        if key not in self._diagnostics:
            self._diagnostics[key] = None  # placeholder

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
