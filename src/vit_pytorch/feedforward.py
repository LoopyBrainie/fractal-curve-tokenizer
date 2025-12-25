# -*- coding: utf-8 -*-
"""自适应分形前馈网络模块

数学形式化
============

SwiGLU FFN (LLaMA/PaLM 风格):
    SwiGLU(x) = W_out · (Swish(W_gate · x) ⊙ (W_value · x))
    其中 Swish(x) = x · σ(x), σ 是 sigmoid

层级自适应 (Level Adaptation):
    Output = (1 - α_d) · FFN(x) + α_d · Adapter([x; E_level(d)])
    其中:
    - α_d = softmax(MixingWeights)_d
    - Adapter 是小型 MLP
    - [;] 表示拼接

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
- 'swiglu_level': SwiGLU + Level Adaptation（推荐，最佳平衡）

废弃特性:
- use_feature_gating: SwiGLU 已内置门控
- Dynamic Activation: 消融实验表明熵 > 90%，无效
"""

from __future__ import annotations

import warnings
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import extract_depths


# FFN type literal for type checking
FFNType = Literal['gelu', 'swiglu', 'swiglu_level']


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
        self.hidden_dim = hidden_dim
        
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_value = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_out = nn.Linear(hidden_dim, dim, bias=bias)
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
    """Adaptive feed-forward block aware of tokenizer hierarchy.
    
    This module applies a feed-forward transformation to the input,
    with optional depth-aware level adaptation.
    
    Supports multiple FFN types:
    - 'gelu': Standard GELU FFN (original, for backward compatibility)
    - 'swiglu': SwiGLU FFN (LLaMA-style, lightweight)
    - 'swiglu_level': SwiGLU + Level Adaptation (recommended, best balance)
    
    Args:
        dim: Input/output dimension.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout rate.
        max_level: Maximum hierarchical level for embeddings.
        use_level_adaptation: Whether to use level-aware adaptation (only for 'gelu').
        ffn_type: FFN variant to use ('gelu', 'swiglu', 'swiglu_level').
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        max_level: int = 50,
        use_level_adaptation: bool = True,
        ffn_type: FFNType = 'swiglu_level',
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.max_level = max_level
        self.ffn_type = ffn_type
        
        # For SwiGLU variants, level_adaptation is controlled by ffn_type
        if ffn_type in ('swiglu', 'swiglu_level'):
            self.use_level_adaptation = (ffn_type == 'swiglu_level')
        else:
            self.use_level_adaptation = use_level_adaptation

        self.norm = nn.LayerNorm(dim)
        
        # ========== FFN 主网络 ==========
        if ffn_type in ('swiglu', 'swiglu_level'):
            # SwiGLU: 调整 hidden_dim 以保持参数量相当
            # 原始 GELU: dim -> hidden_dim -> dim (2 * dim * hidden_dim 参数)
            # SwiGLU: dim -> swiglu_hidden * 3 线性层 (3 * dim * swiglu_hidden 参数)
            # 为匹配参数量: swiglu_hidden = hidden_dim * 2 / 3
            swiglu_hidden = (hidden_dim * 2) // 3
            self.swiglu = SwiGLUFFN(dim, swiglu_hidden, dropout)
            self.main_net = None
        else:
            # 原始 GELU FFN
            self.swiglu = None
            self.main_net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout),
            )
        
        # ========== Level Adaptation（仅对 swiglu_level 或 GELU + use_level_adaptation）==========
        if self.use_level_adaptation:
            self.level_embedding: Optional[nn.Embedding] = nn.Embedding(max_level + 1, dim)
            
            # 根据 FFN 类型选择 adapter 的 hidden_dim
            adapter_hidden = (hidden_dim * 2) // 3 // 2 if ffn_type == 'swiglu_level' else hidden_dim // 2
            self.shared_level_adapter: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(dim * 2, adapter_hidden),
                nn.SiLU() if ffn_type == 'swiglu_level' else nn.ReLU(),
                nn.Linear(adapter_hidden, dim),
                nn.Dropout(dropout),
            )
            # P1-1: 初始化为 0，使 sigmoid(0)=0.5 作为中性起点
            # 语义: α_d = σ(w_d)，50% main FFN + 50% level adapter
            self.level_mixing_weights: Optional[nn.Parameter] = nn.Parameter(torch.zeros(max_level + 1))
        else:
            self.level_embedding = None
            self.shared_level_adapter = None
            self.level_mixing_weights = None
    
    def _apply_level_adaptation(
        self, 
        x_norm: torch.Tensor, 
        main_out: torch.Tensor, 
        levels_info: torch.Tensor,
        batch: int,
        seq_len: int,
    ) -> torch.Tensor:
        """应用层级自适应。
        
        Args:
            x_norm: 归一化后的输入
            main_out: FFN 主网络输出
            levels_info: 层级信息
            batch: batch size
            seq_len: 序列长度
            
        Returns:
            层级自适应后的输出
        """
        assert self.level_embedding is not None
        assert self.shared_level_adapter is not None
        assert self.level_mixing_weights is not None
        
        if levels_info.dim() == 2:
            # (Seq, Info) -> broadcast to batch
            depths = extract_depths(levels_info, self.max_level)
            level_embs = self.level_embedding(depths).unsqueeze(0).expand(batch, -1, -1)
            # P1-1 修复: 使用 sigmoid 替代错误的 softmax(dim=0)
            # 数学形式: α_d = σ(w_d) ∈ (0,1)，每个深度独立控制 adapter 权重
            mixing_weights = torch.sigmoid(self.level_mixing_weights[depths]).view(1, seq_len, 1)
        else:
            # (Batch, Seq, Info)
            depths = extract_depths(levels_info, self.max_level)
            level_embs = self.level_embedding(depths)
            # P1-1 修复: 使用 sigmoid 替代错误的 softmax(dim=1)
            mixing_weights = torch.sigmoid(self.level_mixing_weights[depths]).unsqueeze(-1)
        
        adapter_input = torch.cat([x_norm, level_embs], dim=-1)
        level_adapted = self.shared_level_adapter(adapter_input)
        
        return main_out * (1 - mixing_weights) + level_adapted * mixing_weights

    def forward(self, x: torch.Tensor, levels_info: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播。
        
        Args:
            x: 输入张量，形状为 [B, S, D]。
            levels_info: 层级信息，形状为 (S, Info) 或 (B, S, Info)。
            
        Returns:
            输出张量，形状为 [B, S, D]。
        """
        batch, seq_len, _ = x.shape
        x_norm = self.norm(x)

        # ========== FFN 主网络 ==========
        if self.ffn_type in ('swiglu', 'swiglu_level'):
            assert self.swiglu is not None
            main_out = self.swiglu(x_norm)
        else:
            assert self.main_net is not None
            main_out = self.main_net(x_norm)

        # ========== Level Adaptation ==========
        if self.use_level_adaptation and levels_info is not None and levels_info.numel() > 0:
            main_out = self._apply_level_adaptation(x_norm, main_out, levels_info, batch, seq_len)

        return main_out
