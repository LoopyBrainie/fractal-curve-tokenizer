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
- 'swiglu_level': SwiGLU + Level Adaptation（推荐，最佳平衡）

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
FFNType = Literal['gelu', 'swiglu', 'swiglu_level']


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
    """Adaptive feed-forward block aware of tokenizer hierarchy.
    
    This module applies a feed-forward transformation to the input,
    with optional depth-aware level adaptation.
    
    Supports multiple FFN types:
    - 'gelu': Standard GELU FFN (original, for backward compatibility)
    - 'swiglu': SwiGLU FFN (LLaMA-style, lightweight)
    - 'swiglu_level': SwiGLU + Level Adaptation (recommended, best balance)
    
    P11-2 修复: 参数 max_level 现在应传入与 tokenizer.max_level 一致的值，
    而非硬编码的 50。这确保 level_embedding 和 level_mixing_weights 的
    Embedding 表大小与实际使用的深度范围匹配，减少约 90% 的参数浪费。
    
    Args:
        dim: Input/output dimension.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout rate.
        max_level: Maximum hierarchical level for embeddings (P11-2: should match tokenizer.max_level).
        use_level_adaptation: Whether to use level-aware adaptation (only for 'gelu').
        ffn_type: FFN variant to use ('gelu', 'swiglu', 'swiglu_level').
        bias: Whether to use bias in linear layers (default: False for SwiGLU).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        max_level: int = 8,  # P11-2: 默认改为 8，应由上层传入实际 max_level
        use_level_adaptation: bool = True,
        ffn_type: FFNType = 'swiglu_level',
        bias: bool = False,
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

        # I106-2: 层级感知归一化 (方案D)
        # 移除标准 LayerNorm，添加层级感知的 gamma/beta 参数
        # 数学形式: x_norm = (x - μ) / σ * γ[d] + β[d]
        # 其中 d 是 token 的深度层级
        # I101-3: 深度边界处理 - 使用 max_level+2 以支持 padding sentinel
        # index 0..max_level: 有效深度, index max_level+1: padding
        self.ffn_gamma = nn.Embedding(max_level + 2, dim)
        self.ffn_beta = nn.Embedding(max_level + 2, dim)

        # 初始化为恒等变换: γ=1, β=0
        # I-NAN: 改为小值初始化，避免固定值阻断梯度
        nn.init.normal_(self.ffn_gamma.weight, mean=0, std=0.01)
        nn.init.normal_(self.ffn_beta.weight, mean=0, std=0.01)
        
        # ========== FFN 主网络 ==========
        if ffn_type in ('swiglu', 'swiglu_level'):
            # I34-19: LLaMA 风格 SwiGLU - 默认 Dff = 4 * dim
            # I140: 支持可配置的 expansion ratio（兼容旧 checkpoint）
            # 注意: 使用 hidden_dim 参数而非硬编码 dim * 4
            swiglu_hidden = hidden_dim  # 使用传入的 hidden_dim，支持 2.67x 等不同 expansion
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
            self.level_embedding: Optional[nn.Embedding] = nn.Embedding(max_level + 1, dim)

            # I34-19: LLaMA 风格 adapter - D_adapter = dim / 2
            # I140: 支持可配置的 adapter expansion（兼容旧 checkpoint）
            # 原始设计: adapter_hidden = dim // 2 (0.5x)
            # 但 checkpoint 可能使用不同 expansion，根据 hidden_dim 比例计算
            base_adapter_hidden = hidden_dim // 2 if hidden_dim > dim else dim // 2
            # 确保至少有一个合理的最小值
            adapter_hidden = max(base_adapter_hidden, 128)
            self.shared_level_adapter: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(dim * 2, adapter_hidden),
                nn.SiLU() if ffn_type == 'swiglu_level' else nn.ReLU(),
                nn.Linear(adapter_hidden, dim),
                nn.Dropout(dropout),
            )
            # P1-1: 初始化为 0，使 sigmoid(0)=0.5 作为中性起点
            # 语义: α_d = σ(w_d)，50% main FFN + 50% level adapter
            # I-NAN: 改为小值初始化
            self.level_mixing_weights: Optional[nn.Parameter] = nn.Parameter(
                torch.randn(max_level + 1) * 0.01
            )
        else:
            self.level_embedding = None
            self.shared_level_adapter = None
            self.level_mixing_weights = None

        # === 诊断数据记录器（Layer-Packaged -> Trainer-Unpacked 架构）===
        # 使用统一缓存替代散落的 _last_xxx 变量，避免 DDP 不同步问题
        # D1-AUDIT FIX: 类型改为 Dict[str, Any]，存储 GPU tensor 以避免 forward 内 .item()
        self._diagnostic_cache: Dict[str, Any] = {}
        # 保留旧变量以兼容现有逻辑（将在 ffn_output 中整合）
        self._last_adapter_norm: Optional[torch.Tensor] = None
        self._last_level_mixing_weights: Optional[torch.Tensor] = None

    def clear_diagnostics(self) -> None:
        """清除诊断缓冲区，防止显存泄漏

        I-OOM FIX: 在每次 forward 结束后调用，
        确保 _last_xxx 引用不累积导致显存泄漏
        """
        self._last_adapter_norm = None
        self._last_level_mixing_weights = None

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
        # I98-4: clamp depths to [0, max_level] to handle padding sentinel (-1)
        depths_clamped = depths.clamp(min=0, max=self.max_level)
        level_embs = self.level_embedding(depths_clamped)
        # P1-1 修复: 使用 sigmoid 替代错误的 softmax(dim=1)
        mixing_weights = torch.sigmoid(self.level_mixing_weights[depths_clamped]).unsqueeze(-1)

        adapter_input = torch.cat([x_norm, level_embs], dim=-1)
        level_adapted = self.shared_level_adapter(adapter_input)

        # 记录诊断数据
        self._last_adapter_norm = level_adapted.norm().detach()
        self._last_level_mixing_weights = mixing_weights.detach()

        # === P1: 增强诊断缓存（D1-AUDIT FIX: 保持 GPU tensor）===
        self._diagnostic_cache.clear()  # 防止内存泄漏：每次 forward 清空
        adapter_norm_val = level_adapted.norm().detach()
        self._diagnostic_cache["adapter_norm"] = adapter_norm_val
        # 混合权重分布统计
        self._diagnostic_cache["level_mixing_min"] = mixing_weights.min().detach()  # GPU tensor
        self._diagnostic_cache["adapter_dominance"] = (mixing_weights > 0.5).float().mean().detach()  # GPU tensor
        # === P1: 计算贡献比率（带数值安全 clamp）===
        # I-NAN: 训练初期 main_ffn_norm 因 gamma=0.01 可能极小，
        # adapter_norm / tiny_value 会产生极大离群点，clamp 防止离群值
        # D1-AUDIT FIX: main_norm 现在是 tensor，torch.where 处理条件
        # D4-AUDIT FIX: 使用 zeros_like 替代 torch.tensor() 创建新 tensor
        main_norm = self._diagnostic_cache.get("main_ffn_norm", torch.zeros_like(adapter_norm_val))
        contribution = torch.where(
            main_norm > 1e-6,
            (adapter_norm_val / main_norm.clamp(min=1e-6)).clamp(max=100.0),
            torch.zeros_like(adapter_norm_val)
        )
        self._diagnostic_cache["contribution_ratio"] = contribution.detach()

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
            # I101-3: 深度边界处理 - padding (-1) 映射到 max_level+1
            # 有效深度 0..max_level 保持不变
            # D4-AUDIT FIX: 使用 new_full 替代 torch.tensor() 创建标量
            depths = depths.where(depths >= 0, depths.new_full((), self.max_level + 1))
            depths = depths.clamp(min=0, max=self.max_level + 1)

        # Task 4 重构: 使用 F.layer_norm 启用 torch.compile Kernel Fusion
        # 分离标准归一化与层级感知仿射变换，使 PyTorch 识别为融合模式
        x_norm = F.layer_norm(x, [self.dim], weight=None, bias=None)

        # 层级感知仿射变换 (level_offset)
        # 注意: 在 F.layer_norm 之后应用，确保融合内核仅包含归一化
        gamma = self.ffn_gamma(depths)  # [B, S, D]
        beta = self.ffn_beta(depths)    # [B, S, D]
        x_norm = x_norm * gamma + beta

        # === P0: 捕获 gamma/beta 运行时统计（D1-AUDIT FIX: GPU tensor）===
        self._diagnostic_cache["ffn_gamma_mean"] = gamma.mean().detach()
        self._diagnostic_cache["ffn_gamma_std"] = gamma.std().detach()
        self._diagnostic_cache["ffn_beta_mean"] = beta.mean().detach()
        self._diagnostic_cache["ffn_beta_std"] = beta.std().detach()

        # ========== FFN 主网络 ==========
        if self.ffn_type in ('swiglu', 'swiglu_level'):
            assert self.swiglu is not None
            main_out = self.swiglu(x_norm)
        else:
            assert self.main_net is not None
            main_out = self.main_net(x_norm)

        # === P0: 捕获主 FFN 输出范数（D1-AUDIT FIX: GPU tensor）===
        self._diagnostic_cache["main_ffn_norm"] = main_out.norm().detach()

        # ========== Level Adaptation ==========
        if self.use_level_adaptation and levels_info is not None and levels_info.data.numel() > 0:
            main_out = self._apply_level_adaptation(x_norm, main_out, levels_info, batch, seq_len)

        # I-OOM FIX: 清除诊断引用，防止显存泄漏
        self.clear_diagnostics()

        return main_out

    @property
    @torch._dynamo.disable  # 🌟 修复：禁止 Dynamo 追踪此属性，防止 Guard 失败导致重编译泄漏
    def ffn_output(self) -> dict:
        """FFN 层的增强诊断包裹

        遵循 Layer-Packaged -> Trainer-Unpacked 哲学。
        使用扁平键名格式（展平后变为 train/ffn_0/{key}）。

        D1-AUDIT FIX: 所有值现在为 GPU tensor，
        由 flatten_layer_outputs() 在 post_forward() 统一调用 .item()。

        I-OOM FIX: 使用 Disable & Flush 模式：
        - @torch._dynamo.disable 屏蔽追踪
        - 读取后立即 .cpu().item() 迁移到 CPU
        - 读取后立即置 None 斩断计算图引用

        诊断字段:
            - ffn_gamma_mean/std, ffn_beta_mean/std: 层级感知归一化参数
            - level_mixing_mean/std/min/max: 层级混合权重分布
            - adapter_dominance: 混合权重 > 0.5 的 Token 比例
            - main_ffn_norm: 主 FFN 输出范数
            - adapter_norm: Adapter 输出范数
            - contribution_ratio: Adapter 相对于 FFN 的贡献率
            - ffn_type: FFN 类型标志
        """
        output = {}

        # 1. 层级感知归一化参数（CPU 迁移）
        if hasattr(self, 'ffn_gamma'):
            output["ffn_gamma_mean"] = self.ffn_gamma.weight.mean().item()
            output["ffn_gamma_std"] = self.ffn_gamma.weight.std().item()
        if hasattr(self, 'ffn_beta'):
            output["ffn_beta_mean"] = self.ffn_beta.weight.mean().item()
            output["ffn_beta_std"] = self.ffn_beta.weight.std().item()

        # 2. 运行时诊断缓存（CPU 迁移 + 清空）
        if self._diagnostic_cache:
            for k, v in self._diagnostic_cache.items():
                if isinstance(v, torch.Tensor):
                    output[k] = v.detach().cpu().item()
                else:
                    output[k] = v
            self._diagnostic_cache.clear()  # 🌟 立即清空缓存释放计算图

        # 3. _last_level_mixing_weights（取后即清 + CPU 迁移）
        if getattr(self, '_last_level_mixing_weights', None) is not None:
            w = self._last_level_mixing_weights.detach().cpu()
            output["level_mixing_mean"] = w.mean().item()
            output["level_mixing_std"] = w.std().item()
            output["level_mixing_max"] = w.max().item()
            self._last_level_mixing_weights = None  # 🌟 斩断幽灵引用

        # 4. _last_adapter_norm（取后即清 + CPU 迁移）
        if getattr(self, '_last_adapter_norm', None) is not None:
            output["adapter_norm_fallback"] = self._last_adapter_norm.detach().cpu().item()
            self._last_adapter_norm = None  # 🌟 斩断幽灵引用

        # 5. FFN 类型标志
        output["ffn_type"] = self.ffn_type

        return output
