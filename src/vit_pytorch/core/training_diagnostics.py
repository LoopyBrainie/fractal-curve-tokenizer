# -*- coding: utf-8 -*-
r"""Train/Eval Loss 診斷工具

用於診斷 train loss = 90+ vs eval loss = 4 的不一致問題。

使用方式:
    from vit_pytorch.core.training_diagnostics import register_loss_hooks

    diagnostics = register_loss_hooks(model)
    # 訓練後打印 diagnostics.stats
"""

from __future__ import annotations

import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Optional


class LossDiagnostics:
    """Train/Eval Loss 診斷工具

    統計 train/eval 階段的 logits 分佈，幫助診斷 loss 不一致問題。
    """

    def __init__(self):
        self.stats: Dict[str, List[Dict]] = defaultdict(list)
        self.handles: List[nn.Module._forward_hooks] = []
        self._is_train: bool = True

    def register_hooks(self, model: nn.Module) -> "LossDiagnostics":
        """為模型註冊 forward hooks

        Args:
            model: PyTorch 模型

        Returns:
            self
        """
        def make_hook(name: str, is_train_fn):
            def hook(module, input, output):
                # 檢測模型是否在 train 模式
                self._is_train = is_train_fn()

                # 提取 logits
                if hasattr(output, 'logits'):
                    logits = output.logits
                elif isinstance(output, torch.Tensor):
                    logits = output
                else:
                    return

                # 記錄統計信息
                self.stats[f'{name}_{"train" if self._is_train else "eval"}'].append({
                    'max': logits.max().item(),
                    'min': logits.min().item(),
                    'mean': logits.mean().item(),
                    'std': logits.std().item(),
                    'shape': logits.shape,
                })
            return hook

        # 為分類頭註冊 hook
        for name, module in model.named_modules():
            if 'head' in name.lower() or 'classifier' in name.lower():
                handle = module.register_forward_hook(
                    make_hook(name, lambda: model.training)
                )
                self.handles.append(handle)

        return self

    def remove_hooks(self) -> None:
        """移除所有 hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def summary(self) -> str:
        """生成診斷摘要"""
        lines = ["=" * 60]
        lines.append("Train/Eval Loss Diagnostics Summary")
        lines.append("=" * 60)

        for key, values in self.stats.items():
            if not values:
                continue

            avg_max = sum(v['max'] for v in values) / len(values)
            avg_min = sum(v['min'] for v in values) / len(values)
            avg_mean = sum(v['mean'] for v in values) / len(values)
            avg_std = sum(v['std'] for v in values) / len(values)

            lines.append(f"\n{key}:")
            lines.append(f"  avg_max:  {avg_max:>10.4f}")
            lines.append(f"  avg_min:  {avg_min:>10.4f}")
            lines.append(f"  avg_mean: {avg_mean:>10.4f}")
            lines.append(f"  avg_std:  {avg_std:>10.4f}")
            lines.append(f"  samples: {len(values)}")

            # 診斷建議
            if avg_max > 15:
                lines.append(f"  ⚠️  WARNING: logits max > 15 可能導致 softmax 溢出!")
            if abs(avg_mean) > 10:
                lines.append(f"  ⚠️  WARNING: logits mean 異常偏離 0")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


def register_loss_hooks(model: nn.Module) -> LossDiagnostics:
    """為模型註冊 Loss 診斷 hooks

    Args:
        model: PyTorch 模型

    Returns:
        LossDiagnostics 實例
    """
    diagnostics = LossDiagnostics()
    diagnostics.register_hooks(model)
    return diagnostics
