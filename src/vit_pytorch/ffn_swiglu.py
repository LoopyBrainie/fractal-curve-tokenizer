# -*- coding: utf-8 -*-
"""自适应分形前馈网络模块

数学形式化
============

SwiGLU FFN (LLaMA/PaLM 风格):
    SwiGLU(x) = W_out · (Swish(W_gate · x) ⊙ (W_value · x))
    其中 Swish(x) = x · σ(x), σ 是 sigmoid

### SiLU 与 Swish 等价性
**SiLU (Sigmoid Linear Unit)** 和 **Swish** 是同一激活函数的两个名称：

$$\text{SiLU}(x) = \text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

其中 $\sigma(x) = \frac{1}{1 + e^{-x}}$ 是 sigmoid 函数。

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

from .levels_info import LevelsInfo  # I98-4


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
    
    P11-2 修复: 参数 max_depth 现在应传入与 tokenizer.max_depth 一致的值，
    而非硬编码的 50。这确保 level_embedding 和 level_mixing_weights 的
    Embedding 表大小与实际使用的深度范围匹配，减少约 90% 的参数浪费。
    
    Args:
        dim: Input/output dimension.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout rate.
        max_depth: Maximum hierarchical level for embeddings (P11-2: should match tokenizer.max_depth).
        use_level_adaptation: Whether to use level-aware adaptation (only for 'gelu').
        ffn_type: FFN variant to use ('gelu', 'swiglu', 'swiglu_level').
        bias: Whether to use bias in linear layers (default: False for SwiGLU).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        max_depth: int = 8,  # P11-2: 默认改为 8，应由上层传入实际 max_depth
        use_level_adaptation: bool = True,
        ffn_type: FFNType = 'swiglu_level',
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.max_depth = max_depth
        self.ffn_type = ffn_type
        
        # For SwiGLU variants, level_adaptation is controlled by ffn_type
        if ffn_type in ('swiglu', 'swiglu_level'):
            self.use_level_adaptation = (ffn_type == 'swiglu_level')
        else:
            self.use_level_adaptation = use_level_adaptation

        # I106-2: 层级感知归一化 (方案D)
        # 移除标准 LayerNorm，添加层级感知的 gamma/beta 参数
        # 数学形式: x_norm = (x - μ) / σ * γ[d] + β[d]
        # 其中 d 是 token 的深度层级
        # I101-3: 深度边界处理 - 使用 max_depth+2 以支持 padding sentinel
        # index 0..max_depth: 有效深度, index max_depth+1: padding
        self.ffn_gamma = nn.Embedding(max_depth + 2, dim)
        self.ffn_beta = nn.Embedding(max_depth + 2, dim)

        # 初始化为恒等变换: γ=1, β=0
        nn.init.ones_(self.ffn_gamma.weight)
        nn.init.zeros_(self.ffn_beta.weight)
        
        # ========== FFN 主网络 ==========
        if ffn_type in ('swiglu', 'swiglu_level'):
            # I34-19: LLaMA 风格 SwiGLU - Dff = 4 * dim
            # 移除与 GELU 的参数匹配约束，简化设计
            swiglu_hidden = dim * 4
            self.swiglu = SwiGLUFFN(dim, swiglu_hidden, dropout, bias=bias)
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
            self.level_embedding: Optional[nn.Embedding] = nn.Embedding(max_depth + 1, dim)
            
            # I34-19: LLaMA 风格 adapter - D_adapter = dim / 2
            adapter_hidden = dim // 2 if ffn_type == 'swiglu_level' else hidden_dim // 2
            self.shared_level_adapter: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(dim * 2, adapter_hidden),
                nn.SiLU() if ffn_type == 'swiglu_level' else nn.ReLU(),
                nn.Linear(adapter_hidden, dim),
                nn.Dropout(dropout),
            )
            # P1-1: 初始化为 0，使 sigmoid(0)=0.5 作为中性起点
            # 语义: α_d = σ(w_d)，50% main FFN + 50% level adapter
            self.level_mixing_weights: Optional[nn.Parameter] = nn.Parameter(torch.zeros(max_depth + 1))
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

        # I98-4: 使用 LevelsInfo.depths，并 clamp 负值（padding sentinel）
        depths = levels_info.depths  # [B, S]
        # I98-4: clamp depths to [0, max_depth] to handle padding sentinel (-1)
        depths_clamped = depths.clamp(min=0, max=self.max_depth)
        level_embs = self.level_embedding(depths_clamped)
        # P1-1 修复: 使用 sigmoid 替代错误的 softmax(dim=1)
        mixing_weights = torch.sigmoid(self.level_mixing_weights[depths_clamped]).unsqueeze(-1)

        adapter_input = torch.cat([x_norm, level_embs], dim=-1)
        level_adapted = self.shared_level_adapter(adapter_input)
        
        return main_out * (1 - mixing_weights) + level_adapted * mixing_weights

    def forward(self, x: torch.Tensor, levels_info: Optional[LevelsInfo] = None) -> torch.Tensor:
        """前向传播。

        I98-4: levels_info 参数类型从 torch.Tensor 改为 LevelsInfo
        I106-2: 使用层级感知归一化替代标准 LayerNorm

        数学形式:
            x_norm = (x - μ) / σ * γ[d] + β[d]

        Args:
            x: 输入张量，形状为 [B, S, D]。
            levels_info: LevelsInfo 实例（可选）。

        Returns:
            输出张量，形状为 [B, S, D]。
        """
        batch, seq_len, _ = x.shape

        # ========== I106-2: 层级感知归一化 ==========
        # 获取深度信息
        if levels_info is None or levels_info.data.numel() == 0:
            depths = torch.zeros(batch, seq_len, dtype=torch.long, device=x.device)
        else:
            depths = levels_info.depths  # [B, S], 包含 padding sentinel -1
            # I101-3: 深度边界处理 - padding (-1) 映射到 max_depth+1
            # 有效深度 0..max_depth 保持不变
            depths = depths.where(depths >= 0, torch.tensor(self.max_depth + 1, device=depths.device))
            depths = depths.clamp(min=0, max=self.max_depth + 1)

        # 标准 LayerNorm 计算
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)

        # 层级感知仿射变换
        gamma = self.ffn_gamma(depths)  # [B, S, D]
        beta = self.ffn_beta(depths)    # [B, S, D]
        x_norm = x_norm * gamma + beta

        # ========== FFN 主网络 ==========
        if self.ffn_type in ('swiglu', 'swiglu_level'):
            assert self.swiglu is not None
            main_out = self.swiglu(x_norm)
        else:
            assert self.main_net is not None
            main_out = self.main_net(x_norm)

        # ========== Level Adaptation ==========
        if self.use_level_adaptation and levels_info is not None and levels_info.data.numel() > 0:
            main_out = self._apply_level_adaptation(x_norm, main_out, levels_info, batch, seq_len)

        return main_out
