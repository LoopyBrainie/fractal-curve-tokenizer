# -*- coding: utf-8 -*-
"""
分形视觉 Transformer (Fractal ViT)

数学形式化
============

完整前向传播:
    1. Tokenization:  (T, L) = Tokenizer(I)
       其中 I ∈ R^{B × C × H × W}, T ∈ R^{B × N × D}, L ∈ Z^{B × N}
    
    2. 位置编码:      T' = T + E_pos(T, L)
    
    3. CLS + Dropout: X = Dropout([CLS; T'])
    
    4. Transformer:   X' = Transformer(X, L)
    
    5. 池化:          z = Pool(X')
       - cls:  z = X'[:, 0]
       - mean: z = mean(X'[:, 1:])
    
    6. 分类:          ŷ = MLP(z)

Tokenizer 选项
--------------
+---------------+-------------------------------+------------------+
| tokenizer_type| 实现                           | 特点              |
+===============+===============================+==================+
| streaming_v3  | StreamingFractalTokenizerV3   | Variable Depth   |
|               |                               | Tokens (推荐)    |
+---------------+-------------------------------+------------------+
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embed_fractal_position import FractalPositionEmbedding
from .tokenizer_streaming import StreamingFractalTokenizerV3
from .base_tokenizer import BaseTokenizer, TokenizerOutput
from .block_transformer import FractalTransformer, FFNType
from .utils import pair


# Tokenizer 类型定义
TokenizerType = Literal["streaming_v3"]


class FractalCurveViT(nn.Module):
    """分形曲线视觉 Transformer。
    
    核心特性：
    - 可学习的分割决策网络（6特征输入）
    - 真正的 Hilbert 曲线递归算法（支持多方向）
    - 动态层级管理（最大 50 层）
    - 智能 patch 处理和特征增强
    - 边缘检测和纹理复杂度分析
    - 自适应多尺度处理
    
    P11-2 修复: max_level 参数现在默认为 None，将自动从 tokenizer.max_depth 获取。
    这确保所有 Embedding 表大小与实际使用的深度范围匹配，减少约 90% 的参数浪费。
    
    Attributes:
        image_size: 输入图像尺寸
        num_classes: 分类类别数
        dim: 模型维度
        pool: 池化策略 ('cls', 'mean' 或其他)
        max_level: 最大递归层级 (从 tokenizer.max_depth 自动获取)
        tokenizer: 图像 tokenizer
        token_processor: token 处理器
        pos_embedding: 位置编码
        transformer: Transformer 模块
    """

    def __init__(
        self,
        *,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        num_classes: int,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 1024,
        pool: str = "cls",
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        # I30-17: 替换 Tuple[int, int] 为单一 int
        # max_depth 由 min_patch_size 和 image_size 自动计算: floor(log2(min(H, W) / min_patch_size))
        min_patch_size: Union[int, Tuple[int, int]] = 4,
        max_level: Optional[int] = None,  # P11-2: 默认 None，从 tokenizer.max_depth 自动获取
        # I30-17: 废弃 num_scales 参数，改为动态计算
        num_scales: Optional[int] = None,  # 已废弃，仅用于向后兼容
        use_hilbert_encoding: bool = True,
        use_spatial_encoding: bool = True,
        use_checkpoint: bool = False,
        drop_path_rate: float = 0.0,
        ffn_type: FFNType = 'swiglu_level',
        tokenizer: Optional[BaseTokenizer] = None,
        position_embedding: Optional[FractalPositionEmbedding] = None,
        # Streaming Tokenizer 配置
        tokenizer_type: TokenizerType = "streaming_v3",
        # P6-2: LCA 温度配置
        lca_temperature: Optional[float] = 1.5,
        learnable_temperature: bool = True,
        # I23-2: Token 数量约束
        K_min: int = 8,
        K_max: int = 64,
        # I27: 子模块 Dropout 配置
        # 设计原理: 允许统一控制正则化强度，避免硬编码值
        splitter_dropout: Optional[float] = None,  # None = 跟随主 dropout
        pos_dropout: Optional[float] = None,       # None = 跟随主 dropout * 0.5
        # I31-3: 面积编码配置
        use_area_encoding: bool = False,           # 启用面积增强位置编码
        use_affine_modulation: bool = False,       # 启用仿射调制注意力偏置
        fourier_levels: int = 4,                   # 傅里叶特征级别数
    ) -> None:
        """初始化 FractalCurveViT。

        Args:
            image_size: 输入图像尺寸（整数或 (H, W) 元组），None 表示动态分辨率
            num_classes: 分类类别数
            dim: 模型嵌入维度
            depth: Transformer 层数
            heads: 注意力头数
            mlp_dim: MLP 隐藏层维度
            pool: 池化策略 ('cls', 'mean' 或混合)
            channels: 输入图像通道数
            dim_head: 每个注意力头的维度
            dropout: Dropout 比率
            emb_dropout: 嵌入层 Dropout 比率
            min_patch_size: (I30-17) 目标最小 patch 大小，用于动态计算 max_depth
                max_depth = floor(log2(min(H, W) / min_patch_size))
            max_level: 最大递归层级（P11-2: 默认 None，自动从 tokenizer.max_depth 获取）
            num_scales: (已废弃) 使用 min_patch_size 替代
            use_hilbert_encoding: 是否使用 Hilbert 编码
            use_spatial_encoding: 是否使用空间编码
            ffn_type: FFN 变体 ('gelu', 'swiglu', 'swiglu_level')
            tokenizer: 自定义 tokenizer（可选，若提供则忽略 tokenizer_type）
            position_embedding: 自定义位置编码（可选）
            tokenizer_type: tokenizer 类型 ("streaming_v3" - Variable Depth Tokens)
            lca_temperature: (P6-2) LCA 偏置温度参数，默认 1.5
                - None: 不使用温度缩放 (兼容模式)
                - float: 温度初始值
            learnable_temperature: (P6-2) 是否使温度可学习
            K_min: (I23-2) GumbelTopKSplitter 最小 token 数量硬下界
            K_max: (I23-2) GumbelTopKSplitter 最大 token 数量
            splitter_dropout: (I27) Splitter MLP dropout
                - None: 自动 = min(dropout, 0.15)  # Splitter 不宜过高
                - float: 显式指定
            pos_dropout: (I27) Position Embedding dropout
                - None: 自动 = dropout * 0.5  # 信息瓶颈需保守
                - float: 显式指定
            use_area_encoding: (I31-3) 启用面积增强位置编码，默认 False
            use_affine_modulation: (I31-3) 启用仿射调制注意力偏置，默认 False
            fourier_levels: (I31-3) 傅里叶特征级别数，用于面积编码，默认 4

        Note:
            I30-17: 已废弃 num_scales 参数。现在使用 min_patch_size 动态计算 max_depth:
                max_depth = max(0, floor(log2(min(H, W) / min_patch_size)))

            I78: 支持 image_size=None 实现真正的动态分辨率输入。
                初始化时使用估算的 image_size（基于 min_patch_size），
                前向传播时根据实际输入尺寸动态调整深度。
        """
        super().__init__()

        self.num_classes = num_classes
        self.dim = dim
        self.pool = pool
        # P11-2: max_level 将在 tokenizer 创建后从 tokenizer.max_depth 获取
        self.use_checkpoint = use_checkpoint
        self.ffn_type = ffn_type
        
        # 保存 tokenizer 类型
        self.tokenizer_type = tokenizer_type
        self._is_streaming = True  # 现在所有 tokenizer 都是 streaming 模式
        self.lca_temperature = lca_temperature
        self.learnable_temperature = learnable_temperature

        # ====================================================================
        # I27: 子模块 Dropout 配置 (避免硬编码)
        # ====================================================================
        # 数学依据:
        #   - Splitter MLP 敏感: 过高 dropout 导致分割决策不稳定
        #   - Position Embedding 是信息瓶颈: 需保守正则化
        #
        # 推导公式:
        #   splitter_dropout = min(dropout, 0.15)  # cap at 0.15
        #   pos_dropout = dropout * 0.5            # half of main dropout
        # ====================================================================
        effective_splitter_dropout = splitter_dropout if splitter_dropout is not None else min(dropout, 0.15)
        effective_pos_dropout = pos_dropout if pos_dropout is not None else (dropout * 0.5)

        # I30-17: 处理 min_patch_size 的向后兼容
        # 支持旧 API: min_patch_size=(4, 4)
        if isinstance(min_patch_size, tuple):
            effective_min_patch_size = min_patch_size[0]
        else:
            effective_min_patch_size = min_patch_size

        # I78: 处理动态分辨率模式 (必须在使用 effective_min_patch_size 之后)
        if image_size is None:
            # 使用 min_patch_size 估算默认图像尺寸用于初始化
            # 估算公式: min(H, W) ≈ min_patch_size × 2^6 = min_patch_size × 64
            # max_depth 将由 StreamingFractalTokenizerV3 根据实际输入图像自动计算
            estimated_size = effective_min_patch_size * 64
            self.image_size = pair(estimated_size)
            self._dynamic_image_size = True
        else:
            self.image_size = pair(image_size)
            self._dynamic_image_size = False

        # === Tokenizer 选择逻辑 ===
        if tokenizer is not None:
            # 用户提供自定义 tokenizer，直接使用
            pass
        elif tokenizer_type == "streaming_v3":
            # I30-17: 使用动态深度计算，废弃 num_scales
            # 兼容旧代码: 如果显式指定 num_scales，使用 max_depth = num_scales - 1
            if num_scales is not None:
                # 向后兼容: 使用 num_scales 计算 max_depth
                max_depth_v3 = num_scales - 1
                tokenizer = StreamingFractalTokenizerV3(
                    image_size=self.image_size,
                    channels=channels,
                    d_model=dim,
                    base_patch_size=effective_min_patch_size,
                    max_depth=max_depth_v3,  # 使用显式指定的 max_depth
                    K_min=K_min,
                    K_max=K_max,
                    splitter_dropout=effective_splitter_dropout,  # I27: 可配置
                )
            else:
                # I30-17: 新方式，使用 min_patch_size 动态计算 max_depth
                # max_depth = floor(log2(min(H, W) / min_patch_size))，无硬限制
                tokenizer = StreamingFractalTokenizerV3(
                    image_size=self.image_size,
                    channels=channels,
                    d_model=dim,
                    base_patch_size=effective_min_patch_size,
                    min_patch_size=effective_min_patch_size,  # I30-17: 目标最小 patch
                    K_min=K_min,
                    K_max=K_max,
                    splitter_dropout=effective_splitter_dropout,  # I27: 可配置
                )
        else:
            raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}. Use 'streaming_v3'.")

        # 允许外部访问统一接口
        self.tokenizer = tokenizer
        # 兼容旧属性名
        self.fractal_tokenizer = tokenizer

        # P11-2 修复: 从 tokenizer 动态获取 max_depth 作为 max_level
        # 这确保 Embedding 表大小与实际使用的深度范围匹配
        if max_level is None:
            if hasattr(tokenizer, 'max_depth'):
                max_level = tokenizer.max_depth
            else:
                # 向后兼容: 若 tokenizer 没有 max_depth 属性，使用保守默认值
                max_level = 8
        self.max_level = max_level

        # Streaming tokenizer 已直接输出 D-dim embeddings，无需 token_processor
        self.token_processor = None

        # 高级分形位置编码
        # I27: 传递 pos_dropout 到 Position Embedding
        # I31-3: 支持面积增强位置编码
        if position_embedding is None:
            if use_area_encoding:
                from .embed_fractal_position import AreaEnhancedPositionEmbedding
                position_embedding = AreaEnhancedPositionEmbedding(
                    dim=dim,
                    max_level=max_level,
                    fourier_levels=fourier_levels,
                    use_hilbert_encoding=use_hilbert_encoding,
                    use_spatial_encoding=use_spatial_encoding,
                    dropout=effective_pos_dropout,  # I27: 可配置
                )
            else:
                position_embedding = FractalPositionEmbedding(
                    dim=dim,
                    max_level=max_level,
                    max_seq_len=10000,
                    use_hilbert_encoding=use_hilbert_encoding,
                    use_spatial_encoding=use_spatial_encoding,
                    dropout=effective_pos_dropout,  # I27: 可配置
                )

        self.pos_embedding = position_embedding
        self.position_embedding = position_embedding

        # CLS token和dropout
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # I30-11: 已删除 Mixed Pooling (pooling_selector)
        # 当前仅支持 cls, mean, weighted 三种池化策略
        # 保留 aux_loss_weight 缓冲以保持向后兼容性
        self.register_buffer("aux_loss_weight", torch.tensor(0.0))

        # I78: 预分配零损失张量，避免每次调用 get_entropy_loss 时重复分配
        self.register_buffer("_zero_loss", torch.tensor(0.0))

        # 分形Transformer
        # I31-3: 传递 use_affine_modulation 和 fourier_levels
        self.transformer = FractalTransformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            max_level=max_level,
            drop_path_rate=drop_path_rate,
            ffn_type=ffn_type,
            use_checkpoint=use_checkpoint,
            lca_temperature=lca_temperature,
            learnable_temperature=learnable_temperature,
            use_affine_modulation=use_affine_modulation,
            fourier_levels=fourier_levels,
        )

        # 分类头
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, num_classes),
        )
        
        # 权重初始化 - 关键改进，防止类别偏差
        self._init_weights()

    def _init_weights(self):
        """初始化权重 - 使用正确的方差缩放
        
        I17 修复: 解决 logits 输出过小导致的模型坍缩问题
        
        问题分析:
            原实现使用固定 std=0.02 的 trunc_normal 初始化，
            不考虑 fan_in，导致:
            - 信号在 MLP Head 中逐层衰减
            - 最终 logits std 仅 ~0.05 (期望 ~1.0)
            - 所有类别 logits 过于接近
            - 模型坍缩到单一类别预测
            
        修复方案:
            分类头使用 Xavier 初始化 (考虑 fan_in + fan_out)
            保证信号在前向传播中保持稳定
            
        参考:
            - Xavier/Glorot: Var(W) = 2 / (fan_in + fan_out)
            - 这确保输入和输出的方差大致相等
        """
        # CLS token: 使用较小的标准差
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 分类头：使用 Xavier 初始化
        # I17 修复: 使用 xavier_uniform 替代固定 std 的 trunc_normal
        for module in self.mlp_head.modules():
            if isinstance(module, nn.Linear):
                # Xavier 初始化: std = sqrt(2 / (fan_in + fan_out))
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Transformer 层权重初始化
        self._init_transformer_weights()
    
    def _init_transformer_weights(self):
        """初始化 Transformer 层的权重"""
        for module in self.transformer.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # 禁用 torch.compile 以支持可变长度 tokens
    @torch._dynamo.disable
    def _prepare_tokens(
        self, img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], "TokenizerOutput"]:
        """准备 tokens 和进行 padding。

        数学形式化：
            T, L = S(I) where N varies per image in variable_tokens mode

        P9-5 优化：
            原实现: O(B) Python 循环进行 padding
            新实现: 使用 TokenizerOutput 的预填充缓存，O(1) 张量操作

        P11-3 改进：
            返回 TokenizerOutput 以便后续获取 regions 信息

        P12-2 优化：
            lengths 返回 Tensor[B] 而非 List[int]，避免 _create_attention_mask 转换开销

        I78: 动态分辨率支持
            当 image_size=None 时，根据实际输入图像大小动态更新 tokenizer 候选区域

        Args:
            img: 输入图像 [B, C, H, W]

        Returns:
            (padded_tokens, padded_levels, lengths, levels_list, token_output):
            - padded_tokens: tokens [B, MaxN, Dim] (padded)
            - padded_levels: 层级信息 [B, MaxN, InfoDim]
            - lengths: Tensor[B] 每个样本的实际 token 数量
            - levels_list: 原始层级列表（用于辅助输出）
            - token_output: TokenizerOutput (P11-3: 用于获取 regions)
        """
        # I78: 动态分辨率支持 - 根据实际输入更新 tokenizer 候选区域
        if self._dynamic_image_size:
            actual_size = (img.shape[2], img.shape[3])  # (H, W)
            self.tokenizer.update_candidates(actual_size)

        # Streaming tokenizer 直接输出 D-dim embeddings
        token_output = self.tokenizer.tokenize(img)
        
        # P9-5 优化: 使用预填充缓存接口，避免 O(B) Python 循环
        info_dim = self.max_level + 4
        padded_tokens, lengths = token_output.get_padded_tokens()
        padded_levels = token_output.get_padded_levels(info_dim)
        levels_list = token_output.levels_list()
        
        # I24-14: 最终防御层 - 无条件 clamp (torch.compile 安全)
        # 不使用 .item() 或数据依赖的 if，直接 clamp
        lengths = lengths.clamp(min=1)
        
        return padded_tokens, padded_levels, lengths, levels_list, token_output

    def _apply_position_and_cls(
        self,
        padded_tokens: torch.Tensor,
        padded_levels: torch.Tensor,
        regions: Optional[torch.Tensor] = None,
        image_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """添加位置编码和 CLS token。

        Args:
            padded_tokens: 填充后的 tokens [B, MaxLen, Dim]
            padded_levels: 填充后的层级信息 [B, MaxLen, InfoDim]
            regions: (I31-3) 区域边界张量，形状 [B, N, 4]
            image_size: (I31-3) 图像尺寸，可以是整数或 (W, H) 元组

        Returns:
            (x, padded_levels):
            - x: 带位置编码和 CLS 的序列 [B, 1+MaxLen, Dim]
            - padded_levels: 更新后的层级信息（包含 CLS） [B, 1+MaxLen, InfoDim]
        """
        batch_size = padded_tokens.shape[0]
        device = padded_tokens.device

        # I31-3: 传递 regions 和 image_size 给位置编码器（用于面积编码）
        pos_emb = self.pos_embedding(padded_levels, regions=regions, image_size=image_size)
        x = padded_tokens + pos_emb

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        cls_level = torch.zeros(batch_size, 1, padded_levels.shape[-1], dtype=torch.long, device=device)
        padded_levels = torch.cat([cls_level, padded_levels], dim=1)

        x = self.dropout(x)

        return x, padded_levels

    def _create_attention_mask(
        self,
        batch_size: int,
        seq_len: int,
        lengths: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建 attention mask（P12-2 向量化优化）。
        
        数学形式化:
            设 lengths = [L_1, ..., L_B]，序列长度 S (含 CLS)。
            Key Padding Mask 定义为:
                M_{b,s} = 1  当且仅当 s > L_b (即 padding 位置)
            
            向量化实现:
                positions = [0, 1, ..., S-1] ∈ Z^{1×S}
                lengths   = [L_1, ..., L_B]^T ∈ Z^{B×1}
                M = (positions > lengths)  # 广播比较 → Z^{B×S}
        
        复杂度分析:
            原实现: O(B) Python 循环 + B 次 GPU slice assignment
            新实现: O(1) 广播比较，单次 GPU kernel
            加速比: ~5-10x (随 B 增大)
        
        Args:
            batch_size: batch 大小
            seq_len: 序列长度（包含 CLS）
            lengths: Tensor[B] 每个样本的有效 token 数量（不含 CLS）
            device: 设备
            
        Returns:
            (attn_mask, key_padding_mask):
            - attn_mask: attention mask [B, 1, 1, Seq]
            - key_padding_mask: padding mask [B, Seq]
        """
        # P12-2: 向量化实现 - 使用广播比较替代 O(B) 循环
        # positions[s] > lengths[b] 等价于 s >= lengths[b] + 1 (即 padding 位置)
        positions = torch.arange(seq_len, device=device)  # [S]
        # 广播: [1, S] > [B, 1] → [B, S]
        key_padding_mask = positions.unsqueeze(0) > lengths.unsqueeze(1)

        attn_mask = ~key_padding_mask
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)

        return attn_mask, key_padding_mask

    def _apply_pooling(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
        split_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """应用池化策略。

        I30-11: 支持 weighted 池化，利用 GumbelTopKSplitter 的 split_probs 作为权重。

        Args:
            x: transformer 输出 [B, Seq, Dim]
            key_padding_mask: padding mask [B, Seq]
            split_probs: [B, N] 分割概率，用于加权池化

        Returns:
            pooled: 池化后的表示 [B, Dim]
        """
        if self.pool == "cls":
            return x[:, 0]
        elif self.pool == "mean":
            # 标准 mean pooling
            token_x = x[:, 1:]
            token_mask = ~key_padding_mask[:, 1:]
            token_x = token_x * token_mask.unsqueeze(-1).float()
            sum_x = token_x.sum(dim=1)
            valid_counts = token_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            return sum_x / valid_counts
        elif self.pool == "weighted":
            # I30-11: 加权池化，利用 split_probs 作为 token 重要性权重
            # 数学形式: z = sum(w_i * x_i) / sum(w_i), 其中 w_i = split_prob_i
            if split_probs is None:
                # 回退到 mean pooling
                token_x = x[:, 1:]
                token_mask = ~key_padding_mask[:, 1:]
                token_x = token_x * token_mask.unsqueeze(-1).float()
                sum_x = token_x.sum(dim=1)
                valid_counts = token_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
                return sum_x / valid_counts

            token_x = x[:, 1:]  # [B, N, D] - 排除 CLS
            token_mask = ~key_padding_mask[:, 1:]  # [B, N] - 排除 CLS

            # I30-11: split_probs 形状为 [B, N]，与 token_x/token_mask 对齐
            # 无需再切片，直接使用
            token_probs = split_probs  # [B, N]

            # 有效性 mask
            token_probs = token_probs * token_mask.float()

            # 权重归一化: w_norm = w / sum(w)
            weight_sum = token_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            normalized_weights = token_probs / weight_sum  # [B, N]

            # 加权平均: z = sum(w_i * x_i)
            weighted = (token_x * normalized_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]

            return weighted
        else:
            raise ValueError(f"Unknown pool type: {self.pool}")

    def _prepare_auxiliary_output(
        self,
        batch_size: int,
        lengths: torch.Tensor,
        levels_list: List[torch.Tensor],
        pooled: torch.Tensor,
        return_aux_info: bool,
        return_features: bool,
    ) -> Tuple[List[Dict[str, Any]], List[torch.Tensor]]:
        """准备辅助输出。
        
        Args:
            batch_size: batch 大小
            lengths: Tensor[B] 有效 token 数量
            levels_list: 层级信息列表
            pooled: 池化后的表示
            return_aux_info: 是否返回辅助信息
            return_features: 是否返回特征
            
        Returns:
            (aux_infos, features_list)
            
        性能优化 (P-OPT-5):
            - aux_info 仅在验证/调试时使用，保持简单实现
            - 使用 non_blocking 转移减少同步等待
        """
        aux_infos: List[Dict[str, Any]] = []
        features_list: List[torch.Tensor] = []

        if return_aux_info:
            # P-OPT-5: 批量获取 lengths 到 CPU，避免多次 .item() 调用
            lengths_cpu = lengths.to('cpu', non_blocking=True)
            
            for i in range(batch_size):
                l = levels_list[i]
                if l.numel() > 0:
                    depths = l[:, 0]
                    # P-OPT-5: unique 操作在 GPU 上执行，结果再转 CPU
                    unique = depths.unique().to('cpu', non_blocking=True).tolist()
                    aux_infos.append({"num_tokens": int(lengths_cpu[i].item()), "levels_used": unique})
                else:
                    aux_infos.append({"num_tokens": 0})
        
        if return_features:
            # 直接返回 pooled 表示作为每个样本的特征
            features_list = [pooled[i] for i in range(pooled.shape[0])]

        return aux_infos, features_list

    def forward(
        self,
        img: torch.Tensor,
        return_attention: bool = False,
        return_aux_info: bool = False,
        return_features: bool = False,
        return_tokens: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, List[Dict[str, Any]]],
        Tuple[torch.Tensor, List[torch.Tensor]],
        Tuple[torch.Tensor, List[Dict[str, Any]], List[torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # (logits, tokens, lengths)
    ]:
        """前向传播。
        
        Args:
            img: 输入图像，形状为 [B, C, H, W]
            return_attention: 是否返回注意力权重（已弃用）
            return_aux_info: 是否返回辅助信息
            return_features: 是否返回特征
            return_tokens: 是否返回 transformer 输出 tokens (用于困难样本挖掘)
            
        Returns:
            根据参数返回不同类型：
            - 默认：分类 logits [B, num_classes]
            - return_aux_info=True：(logits, aux_infos)
            - return_features=True：(logits, features)
            - return_tokens=True: (logits, tokens [B, N, D], lengths [B])
            - 两者都为 True：(logits, aux_infos, features)
        """
        batch_size = img.shape[0]
        device = img.device

        # 1. 准备 tokens (P11-3: 返回 token_output 用于获取 regions)
        padded_tokens, padded_levels, lengths, levels_list, token_output = self._prepare_tokens(img)
        
        # P11-3: 获取 regions 和 image_size 用于正确的 LCA 偏置计算
        regions, image_size = token_output.get_padded_regions()

        # 2. 添加位置编码和 CLS token
        # I31-3: 传递 regions 和 image_size 用于面积编码
        x, padded_levels = self._apply_position_and_cls(
            padded_tokens, padded_levels, regions=regions, image_size=image_size
        )
        
        # P11-3: 为 regions 添加 CLS 对应的零填充
        if regions is not None:
            cls_region = torch.zeros(batch_size, 1, 4, dtype=regions.dtype, device=device)
            regions = torch.cat([cls_region, regions], dim=1)

        # 3. 创建 attention mask
        attn_mask, key_padding_mask = self._create_attention_mask(
            batch_size, x.shape[1], lengths, device
        )

        # 4. Transformer 处理 (P11-3: 传递 regions 和 image_size)
        x = self.transformer(x, padded_levels, attn_mask, regions=regions, image_size=image_size)
        
        # I30-2: 保存 transformer 输出用于困难样本挖掘 (排除 CLS token)
        transformer_tokens = x[:, 1:]  # [B, N, D] 排除 CLS

        # I30-11: 获取 split_probs 用于加权池化
        split_probs = token_output.get_padded_split_probs()  # [B, N] 或 None

        # 5. 池化
        pooled = self._apply_pooling(x, key_padding_mask, split_probs)
        pooled = self.to_latent(pooled)
        final_output = self.mlp_head(pooled)
        
        # I30-2: Token 输出 (用于 HilbertAwareHardMining)
        if return_tokens:
            return final_output, transformer_tokens, lengths

        # 6. 辅助输出
        if return_aux_info or return_features:
            aux_infos, features_list = self._prepare_auxiliary_output(
                batch_size, lengths, levels_list, pooled, return_aux_info, return_features
            )
            
            if return_aux_info and return_features:
                return final_output, aux_infos, features_list
            if return_aux_info:
                return final_output, aux_infos
            if return_features:
                return final_output, features_list

        return final_output

    def get_tokenizer_loss(
        self,
        reward: Optional[float] = None,
        baseline: Optional[float] = None,
        entropy_coef: float = 0.01,
    ) -> torch.Tensor:
        """
        获取 tokenizer 的策略梯度损失 (REINFORCE)
        
        Args:
            reward: 外部提供的奖励信号（如 -classification_loss）
                    正值鼓励当前分割策略，负值惩罚
            baseline: 基线值（未使用，保留接口兼容性）
            entropy_coef: 熵正则化系数（未使用，保留接口兼容性）
        
        Returns:
            torch.Tensor: 零损失（Variable Depth Tokenizer 完全可微分）
        """
        # StreamingFractalTokenizerV3 使用可微分的 Region Pooling
        # 无需策略梯度损失 (I78: 使用预分配零张量避免重复分配)
        return self._zero_loss.detach().to(self.aux_loss_weight.device)
    
    def clear_tokenizer_cache(self) -> None:
        """清空 tokenizer 的动作缓存，应在每个 batch 结束后调用"""
        if hasattr(self.tokenizer, "clear_saved_actions"):
            self.tokenizer.clear_saved_actions()

    def analyze_tokenization(self, img: torch.Tensor) -> Dict[str, Any]:
        """分析 tokenization 过程，返回详细统计信息。
        
        Args:
            img: 输入图像，形状为 [B, C, H, W]
            
        Returns:
            包含以下键的字典：
            - batch_size: 批次大小
            - per_image_stats: 每张图像的统计信息列表
            - overall_stats: 整体统计信息
        """
        with torch.no_grad():
            token_output = self.tokenizer.tokenize(img)
            legacy_output = token_output.to_legacy()
            tokens_list = legacy_output.tokens
            levels_list = legacy_output.levels

            analysis: Dict[str, Any] = {"batch_size": len(tokens_list), "per_image_stats": [], "overall_stats": {}}

            all_levels = []
            total_tokens = 0

            for i, (tokens, levels) in enumerate(zip(tokens_list, levels_list)):
                if tokens.numel() == 0:
                    image_stats = {
                        "num_tokens": 0,
                        "levels_used": [],
                        "max_level": 0,
                        "level_distribution": [],
                    }
                else:
                    depths = levels[:, 0] if levels.numel() > 0 else levels.new_empty(0)
                    unique_levels = depths.unique().tolist()
                    max_level = depths.max().item() if depths.numel() > 0 else 0

                    image_stats = {
                        "num_tokens": tokens.shape[0],
                        "levels_used": unique_levels,
                        "max_level": max_level,
                        "level_distribution": torch.bincount(depths.long()).tolist() if depths.numel() > 0 else [],
                    }

                    all_levels.extend(unique_levels)
                    total_tokens += tokens.shape[0]

                analysis["per_image_stats"].append(image_stats)

            if all_levels:
                level_tensor = torch.tensor(all_levels, device=img.device)
                analysis["overall_stats"] = {
                    "total_tokens": total_tokens,
                    "avg_tokens_per_image": total_tokens / len(tokens_list),
                    "unique_levels_used": sorted(list(set(all_levels))),
                    "max_level_overall": max(all_levels),
                    "level_usage_distribution": dict(zip(*torch.unique(level_tensor.cpu(), return_counts=True))),
                }
            else:
                analysis["overall_stats"] = {
                    "total_tokens": 0,
                    "avg_tokens_per_image": 0,
                    "unique_levels_used": [],
                    "max_level_overall": 0,
                    "level_usage_distribution": {},
                }

            return analysis


# 保持向后兼容性的别名
EnhancedFractalViT = FractalCurveViT
NextGenerationFractalViT = FractalCurveViT
