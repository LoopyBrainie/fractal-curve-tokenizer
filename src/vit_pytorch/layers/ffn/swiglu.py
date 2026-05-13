# -*- coding: utf-8 -*-
r"""自适应分形前馈网络模块

数学形式化
============

SwiGLU FFN (LLaMA/PaLM 风格):
    SwiGLU(x) = W_out · (Swish(W_gate · x) ⊙ (W_value · x))
    其中 Swish(x) = x · σ(x), σ 是 sigmoid

### SiLU 与 Swish 等价性
**SiLU (Sigmoid Linear Unit)** 和 **Swish** 是同一激活函数的两个名称：

.. math::
    \text{SiLU}(x) = \text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}

其中 :math:`\sigma(x) = \frac{1}{1 + e^{-x}}` 是 sigmoid 函数。

**PyTorch 实现**:
- `F.silu(x)` - PyTorch 内置 SiLU 激活函数
- `nn.SiLU()` - SiLU 模块

**使用场景**:
- SwiGLU 中的 gate 分支: `F.silu(self.w_gate(x))` (第112行)
- Adapter 中的非线性: `nn.SiLU()` (第193行)

层级自适应 (Level Adaptation):
    Output = (1 - α_d) · FFN(x) + α_d · Adapter([x; E_level(d)])
    其中:
    - α_d = sigmoid(MixingWeights[d]) ∈ (0, 1)，每个深度独立门控
    - Adapter 是小型 MLP
    - [;] 表示拼接
    
    Sigmoid 语义:
    - sigmoid(0) = 0.5: 50% FFN + 50% Adapter (初始状态)
    - sigmoid(+∞) → 1: 100% Adapter (深度特化)
    - sigmoid(-∞) → 0: 100% FFN (深度无关)

复杂度分析
----------
SwiGLUFFN:
    时间: O(B · N · D · D_ff)  — 3次矩阵乘法 (W_gate, W_value, W_out)
    空间: O(B · N · D_ff)      — gate/value 中间张量
    
AdaptiveFractalFeedForward (with level adaptation):
    时间: O(B · N · D · D_ff) + O(B · N · D)  — FFN + Level Adapter
    空间: O(B · N · D_ff) + O(L_max · D)      — 中间张量 + level embedding

其中: B=batch, N=seq_len, D=dim, D_ff (SwiGLU: 4 × dim, LLaMA 风格)

类对照表
----------
+---------------------------+------------------------------------------+
| 类                         | 数学定义                                   |
+===========================+==========================================+
| SwiGLUFFN                 | x → W_out(Swish(W_g x) ⊙ W_v x)         |
| AdaptiveFractalFeedForward| x → (1-α)FFN(x) + α Adapter(x,d)       |
+---------------------------+------------------------------------------+

FFN 变体选项 (ffn_type):
- 'gelu': 标准 GELU FFN（原始，向后兼容）
- 'swiglu': SwiGLU FFN（轻量级，无层级自适应）

废弃特性:
- use_feature_gating: SwiGLU 已内置门控
- Dynamic Activation: 消融实验表明熵 > 90%，无效
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_pytorch.core.levels_info import LevelsInfo  # I98-4


# FFN type literal for type checking
FFNType = Literal['gelu', 'swiglu']


def _align_to_multiple(value: int, multiple: int = 64) -> int:
    """对齐整数到最近的 multiple 倍数（向上取整）。

    用于确保 SwiGLU 隐藏层维度是 Tensor Core 高效计算的形状。
    NVIDIA GPU Tensor Core 在矩阵维度是 64/128 的整数倍时效率最优。
    """
    return ((value + multiple - 1) // multiple) * multiple


class SwiGLUFFN(nn.Module):
    """独立的 SwiGLU 实现（参考 LLaMA/PaLM）。

    SwiGLU(x) = (Swish(W_gate · x) ⊙ (W_value · x)) · W_out

    优势:
    - 内置门控机制，无需额外 feature_gate
    - 梯度流动更平滑
    - 参数量与 GELU FFN 相当（通过调整 hidden_dim）

    Args:
        dim: 输入/输出维度
        hidden_dim: 隐藏层维度（建议为原始 hidden_dim 的 2/3）
        dropout: Dropout 比率
        bias: 是否使用偏置
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        # Tensor Core 对齐: 隐藏层维度向上取整到 64 的倍数
        # 例如: dim=256, hidden_dim=682 (8/3扩展) → 对齐到 704
        self.hidden_dim = _align_to_multiple(hidden_dim, multiple=64)

        self.w_gate = nn.Linear(dim, self.hidden_dim, bias=bias)
        self.w_value = nn.Linear(dim, self.hidden_dim, bias=bias)
        self.w_out = nn.Linear(self.hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。
        
        Args:
            x: 输入张量，形状为 [B, S, D]
            
        Returns:
            输出张量，形状为 [B, S, D]
        """
        gate = F.silu(self.w_gate(x))  # Swish 激活
        value = self.w_value(x)
        hidden = gate * value  # 元素级门控
        return self.dropout(self.w_out(hidden))


class AdaptiveFractalFeedForward(nn.Module):
    """自适应前馈网络（简化版）。

    支持两种 FFN 类型:
    - 'gelu': 标准 GELU FFN（向后兼容）
    - 'swiglu': SwiGLU FFN（LLaMA 风格，推荐）

    Args:
        dim: 输入/输出维度
        hidden_dim: 隐藏层维度
        dropout: Dropout 比率
        max_level: 最大层级深度（仅用于参数兼容）
        ffn_type: FFN 变体 ('gelu' 或 'swiglu')
        bias: 线性层是否使用偏置
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        max_level: int = 8,
        ffn_type: FFNType = 'swiglu',
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.max_level = max_level
        self.ffn_type = ffn_type

        # 标准 LayerNorm
        self.norm = nn.LayerNorm(dim)

        # FFN 主网络
        if ffn_type == 'swiglu':
            swiglu_hidden = hidden_dim
            self.swiglu = SwiGLUFFN(dim, swiglu_hidden, dropout, bias=bias)
            self.main_net = None
        else:
            self.swiglu = None
            self.main_net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout),
            )

        # 诊断缓存
        self._diagnostic_cache: Dict[str, Any] = {}

    def clear_diagnostics(self) -> None:
        """清除诊断缓冲区。"""
        self._diagnostic_cache.clear()

    def forward(self, x: torch.Tensor, levels_info: Optional[LevelsInfo] = None) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状为 [B, S, D]。
            levels_info: LevelsInfo 实例（保留参数兼容性，当前未使用）。

        Returns:
            输出张量，形状为 [B, S, D]。
        """
        x_norm = self.norm(x)

        if self.ffn_type == 'swiglu':
            assert self.swiglu is not None
            main_out = self.swiglu(x_norm)
        else:
            assert self.main_net is not None
            main_out = self.main_net(x_norm)

        self._diagnostic_cache["main_ffn_norm"] = main_out.norm().detach()

        return main_out

    @property
    @torch._dynamo.disable
    def ffn_output(self) -> dict:
        """FFN 层诊断输出。

        遵循 Layer-Packaged -> Trainer-Unpacked 哲学。

        诊断字段:
            - main_ffn_norm: 主 FFN 输出范数
            - ffn_type: FFN 类型标志
        """
        output = {}

        if self._diagnostic_cache:
            for k, v in self._diagnostic_cache.items():
                if isinstance(v, torch.Tensor):
                    output[k] = v.detach().cpu().item()
                else:
                    output[k] = v
            self._diagnostic_cache.clear()

        output["ffn_type"] = self.ffn_type

        return output
