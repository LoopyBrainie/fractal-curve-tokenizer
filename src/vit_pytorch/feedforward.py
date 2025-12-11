# -*- coding: utf-8 -*-
"""Adaptive fractal-aware feed-forward network module.

This module implements AdaptiveFractalFeedForward, a feed-forward block
that is aware of the hierarchical structure from the fractal tokenizer,
applying depth-dependent scaling to the hidden representations.

Supported FFN types:
- 'gelu': Standard GELU FFN (original)
- 'swiglu': SwiGLU FFN (LLaMA-style, recommended)
- 'swiglu_level': SwiGLU + Level Adaptation (best balance)
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
    
    Note: `use_feature_gating` is deprecated and ignored when `ffn_type` is 
    'swiglu' or 'swiglu_level', as SwiGLU already has built-in gating.
    The Dynamic Activation mechanism has been removed due to ablation results
    showing it degenerates to near-uniform distribution (entropy > 90%).
    
    Args:
        dim: Input/output dimension.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout rate.
        max_level: Maximum hierarchical level for embeddings.
        use_level_adaptation: Whether to use level-aware adaptation (only for 'gelu').
        use_feature_gating: DEPRECATED - ignored for SwiGLU variants.
        ffn_type: FFN variant to use ('gelu', 'swiglu', 'swiglu_level').
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        max_level: int = 50,
        use_level_adaptation: bool = True,
        use_feature_gating: bool = True,
        ffn_type: FFNType = 'swiglu_level',
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.max_level = max_level
        self.ffn_type = ffn_type
        
        # For SwiGLU variants, feature_gating is built-in and level_adaptation
        # is controlled by ffn_type, not the boolean flag
        if ffn_type in ('swiglu', 'swiglu_level'):
            self.use_level_adaptation = (ffn_type == 'swiglu_level')
            self.use_feature_gating = False  # SwiGLU has built-in gating
            # 注意：不再对默认参数发出警告，只有在用户显式传入这些参数时
            # 才需要考虑警告，但由于无法区分，我们静默忽略
        else:
            self.use_level_adaptation = use_level_adaptation
            self.use_feature_gating = use_feature_gating

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
            self.level_mixing_weights: Optional[nn.Parameter] = nn.Parameter(torch.ones(max_level + 1))
        else:
            self.level_embedding = None
            self.shared_level_adapter = None
            self.level_mixing_weights = None
        
        # ========== Feature Gating + Dynamic Activation（仅对 GELU 模式）==========
        # 注意：消融实验表明 Dynamic Activation 熵 > 90%，接近均匀分布，
        # 说明模型没有学到有意义的激活选择。保留此代码仅为向后兼容。
        if self.use_feature_gating:
            self.feature_gate: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, hidden_dim),
                nn.Sigmoid(),
            )
            self.activation_selector: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(dim, 3), nn.Softmax(dim=-1)
            )
        else:
            self.feature_gate = None
            self.activation_selector = None

    def _apply_dynamic_activation(self, x: torch.Tensor, activation_weights: torch.Tensor) -> torch.Tensor:
        """应用动态加权的激活函数组合（已废弃，仅保留向后兼容）。
        
        Warning: 消融实验表明此机制无效（熵 > 90%），建议使用 SwiGLU 变体。
        """
        gelu_out = F.gelu(x)
        relu_out = F.relu(x)
        swish_out = x * torch.sigmoid(x)
        return (
            activation_weights[:, :, 0:1] * gelu_out
            + activation_weights[:, :, 1:2] * relu_out
            + activation_weights[:, :, 2:3] * swish_out
        )
    
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
            mixing_weights = F.softmax(self.level_mixing_weights[depths], dim=0).view(1, seq_len, 1)
        else:
            # (Batch, Seq, Info)
            depths = extract_depths(levels_info, self.max_level)
            level_embs = self.level_embedding(depths)
            mixing_weights = F.softmax(self.level_mixing_weights[depths], dim=1).unsqueeze(-1)
        
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

        # ========== Feature Gating（仅 GELU 模式且启用）==========
        if self.use_feature_gating and self.feature_gate is not None and self.main_net is not None:
            assert self.activation_selector is not None
            gates = self.feature_gate(x_norm)
            hidden = F.linear(x_norm, self.main_net[0].weight, self.main_net[0].bias)
            gated_hidden = hidden * gates
            activation_weights = self.activation_selector(x_norm)
            activated_hidden = self._apply_dynamic_activation(gated_hidden, activation_weights)
            main_out = F.linear(activated_hidden, self.main_net[3].weight, self.main_net[3].bias)
            main_out = self.main_net[4](main_out)

        return main_out
