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

from typing import Any, Dict, List, Literal, Optional, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embed_fractal_position import FractalPositionEmbedding
from .tokenizer_streaming import StreamingFractalTokenizerV3
from .base_tokenizer import BaseTokenizer, TokenizerOutput
from .block_transformer import FractalTransformer, FFNType
from .utils import pair
from .constants import DIVISION_EPSILON, PROB_EPSILON
from .config import AttentionEncoderConfig  # I98-3


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
        # I98-2: 依赖注入参数（可选）
        splitter: Optional[Any] = None,
        tokenizer: Optional[BaseTokenizer] = None,
        transformer: Optional[Any] = None,
        position_embedding: Optional[FractalPositionEmbedding] = None,
        mlp_head: Optional[nn.Sequential] = None,
        cls_token: Optional[nn.Parameter] = None,
        # I98-2: 内部状态标记（由工厂函数设置）
        dynamic_image_size: bool = False,
        # 配置参数（用于向后兼容）
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        num_classes: int = 1000,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 1024,
        pool: str = "cls",
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        min_patch_size: Union[int, Tuple[int, int]] = 4,
        max_level: Optional[int] = None,
        num_scales: Optional[int] = None,
        use_hilbert_encoding: bool = True,
        use_spatial_encoding: bool = True,
        use_checkpoint: bool = False,
        drop_path_rate: float = 0.0,
        ffn_type: FFNType = 'swiglu_level',
        tokenizer_type: TokenizerType = "streaming_v3",
        lca_temperature: Optional[float] = 1.5,
        learnable_temperature: bool = True,
        K_min: int = 8,
        K_max: int = 64,
        pos_dropout: Optional[float] = None,
        use_area_encoding: bool = False,
        use_affine_modulation: bool = True,
        fourier_levels: int = 4,
        encoder_config: Optional[AttentionEncoderConfig] = None,
        quota_learnable: Optional[bool] = None,
        lca_fp16: bool = False,  # I104-3: 使用 FP16 存储 LCA embedding
    ) -> None:
        """初始化 FractalCurveViT。

        I98-2: 支持依赖注入模式。
        - 若提供 injected 组件（splitter, transformer 等），则使用注入的组件
        - 若未提供，则自动创建组件（向后兼容模式）

        Args:
            splitter: (I98-2) 注入的 Splitter 实例
            tokenizer: 自定义 tokenizer（可选，若提供则忽略 tokenizer_type）
            transformer: (I98-2) 注入的 Transformer 实例
            position_embedding: 自定义位置编码（可选）
            mlp_head: (I98-2) 注入的 MLP Head 实例
            cls_token: (I98-2) 注入的 CLS token
            dynamic_image_size: (I98-2) 内部标记：是否动态分辨率模式
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
            min_patch_size: 目标最小 patch 大小，用于动态计算 max_depth
            max_level: 最大递归层级
            num_scales: (已废弃) 使用 min_patch_size 替代
            use_hilbert_encoding: 是否使用 Hilbert 编码
            use_spatial_encoding: 是否使用空间编码
            ffn_type: FFN 变体 ('gelu', 'swiglu', 'swiglu_level')
            tokenizer_type: Tokenizer 类型
            lca_temperature: LCA 温度
            learnable_temperature: 是否可学习温度
            K_min: 最少 token 数
            K_max: 最多 token 数
            pos_dropout: 位置编码 dropout
            use_area_encoding: 启用面积增强位置编码
            use_affine_modulation: 启用 ShapeScaleEncoder 仿射调制
            fourier_levels: 傅里叶特征级别数
            encoder_config: 注意力编码器配置
            quota_learnable: 可学习配额控制

        Note:
            I30-17: 已废弃 num_scales 参数。现在使用 min_patch_size 动态计算 max_depth:
                max_depth = max(0, floor(log2(min(H, W) / min_patch_size)))

            I78: 支持 image_size=None 实现真正的动态分辨率输入。
                初始化时使用估算的 image_size（基于 min_patch_size），
                前向传播时根据实际输入尺寸动态调整深度。

            I98-2: 支持依赖注入模式。若提供 injected 组件（splitter, transformer 等），
                则使用注入的组件；否则自动创建组件（向后兼容模式）。
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
        self.lca_fp16 = lca_fp16  # I104-3

        # ====================================================================
        # I27: 子模块 Dropout 配置 (避免硬编码)
        # ====================================================================
        # 数学依据:
        #   - Splitter MLP 敏感: 过高 dropout 导致分割决策不稳定
        #   - Position Embedding 是信息瓶颈: 需保守正则化
        #
        # 推导公式:
        #   pos_dropout = dropout * 0.5            # half of main dropout
        # ====================================================================
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
            self._cached_image_size = None  # 用于缓存上次更新的尺寸
        else:
            self.image_size = pair(image_size)
            self._dynamic_image_size = False
            self._cached_image_size = None

        # I98-2: 优先级 - 使用注入的组件 > 动态参数创建 > 默认配置
        # 依赖注入模式: 检查是否提供了 injected 组件

        # === Splitter ===
        if splitter is not None:
            # I98-2: 使用注入的 Splitter
            self.splitter = splitter
        else:
            # 动态创建 Splitter（向后兼容）
            from .gumbel_topk_splitter import GumbelTopKSplitter
            from .config import SplitterConfig

            # I98-1: 确定 max_depth_limit (根据 tokenizer 或默认值)
            max_depth_limit = 8  # 默认值
            if tokenizer is not None:
                if hasattr(tokenizer, 'max_depth'):
                    max_depth_limit = tokenizer.max_depth
            elif num_scales is not None:
                max_depth_limit = num_scales - 1

            splitter_config = SplitterConfig(
                feature_dim=dim,
                min_patch_size=effective_min_patch_size,
                max_depth_limit=max_depth_limit,
                hidden_dim=64,
                intermediate_dim=64,
                pool_size=4,
                K_min=K_min,
                K_max=K_max,
                use_dynamic_k=True,
                dropout=min(dropout, 0.15),
                enable_learnable_quota=quota_learnable if quota_learnable is not None else True,
            )
            self.splitter = GumbelTopKSplitter(
                config=splitter_config,
                image_size=self.image_size,
            )

        # === Tokenizer ===
        if tokenizer is not None:
            # 用户提供自定义 tokenizer，直接使用
            pass
        elif tokenizer_type == "streaming_v3":
            if num_scales is not None:
                max_depth_v3 = num_scales - 1
                tokenizer = StreamingFractalTokenizerV3(
                    image_size=self.image_size,
                    channels=channels,
                    d_model=dim,
                    base_patch_size=effective_min_patch_size,
                    max_depth=max_depth_v3,
                )
            else:
                tokenizer_kwargs = dict(
                    image_size=self.image_size,
                    channels=channels,
                    d_model=dim,
                    base_patch_size=effective_min_patch_size,
                    min_patch_size=effective_min_patch_size,
                )
                if max_level is not None:
                    tokenizer_kwargs['max_depth'] = max_level
                tokenizer = StreamingFractalTokenizerV3(**tokenizer_kwargs)
        else:
            raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}. Use 'streaming_v3'.")

        self.tokenizer = tokenizer
        self.fractal_tokenizer = tokenizer

        # P11-2 修复: 从 tokenizer 动态获取 max_depth 作为 max_level
        if max_level is None:
            if hasattr(tokenizer, 'max_depth'):
                max_level = tokenizer.max_depth
            else:
                max_level = 8
        self.max_level = max_level

        self.token_processor = None

        # === Position Embedding ===
        if position_embedding is None:
            # I98-2: 动态创建位置编码（向后兼容）
            # I27: 传递 pos_dropout 到 Position Embedding
            # I31-3: 支持面积增强位置编码
            if use_area_encoding:
                from .embed_fractal_position import AreaEnhancedPositionEmbedding
                position_embedding = AreaEnhancedPositionEmbedding(
                    dim=dim,
                    max_level=max_level,
                    fourier_levels=fourier_levels,
                    use_hilbert_encoding=use_hilbert_encoding,
                    use_spatial_encoding=use_spatial_encoding,
                    dropout=effective_pos_dropout,
                )
            else:
                position_embedding = FractalPositionEmbedding(
                    dim=dim,
                    max_level=max_level,
                    max_seq_len=10000,
                    use_hilbert_encoding=use_hilbert_encoding,
                    use_spatial_encoding=use_spatial_encoding,
                    dropout=effective_pos_dropout,
                )

        self.pos_embedding = position_embedding
        self.position_embedding = position_embedding

        # === CLS Token ===
        if cls_token is not None:
            self.cls_token = cls_token
        else:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        # I30-11: 已删除 Mixed Pooling
        self.register_buffer("aux_loss_weight", torch.tensor(0.0))
        self.register_buffer("_zero_loss", torch.tensor(0.0))

        # === Transformer ===
        if transformer is not None:
            # I98-2: 使用注入的 Transformer
            self.transformer = transformer
        else:
            # 动态创建 Transformer（向后兼容）
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
                encoder_config=encoder_config,
                use_fp16=lca_fp16,  # I104-3
            )

        # === MLP Head ===
        if mlp_head is not None:
            self.mlp_head = mlp_head
        else:
            # 动态创建 MLP Head（向后兼容）
            self.to_latent = nn.Identity()
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, mlp_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim // 2, num_classes),
            )
            self.num_classes = num_classes
            self.dim = dim
            self.depth = depth
            self.heads = heads
            self.mlp_dim = mlp_dim

        # 权重初始化 - 关键改进，防止类别偏差
        self._init_weights()

    # I98-2: 特征提取器抽象 - 封装 tokenizer.shared_conv
    @property
    def _feature_extractor(self) -> nn.Module:
        """特征提取器抽象 (I98-2).

        封装 tokenizer.shared_conv，对外提供特征提取能力。
        未来可替换为独立的特征提取模块。

        对于自定义 tokenizer (如 DummyTokenizer)，如果没有 shared_conv，
        则直接返回 tokenizer 本身（假设已处理特征提取）。

        Returns:
            特征提取模块 (通常为 nn.Conv2d)
        """
        # I98-2: 优先使用 shared_conv，兼容自定义 tokenizer
        if hasattr(self.tokenizer, 'shared_conv'):
            return self.tokenizer.shared_conv
        else:
            # 自定义 tokenizer 已处理特征提取，直接返回
            return self.tokenizer

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
        # I78: 动态分辨率支持 - 根据实际输入更新 splitter 候选区域
        # I99-10: 仅在尺寸变化时更新，避免不必要的计算
        if self._dynamic_image_size:
            actual_size = (img.shape[2], img.shape[3])  # (H, W)
            if actual_size != self._cached_image_size:
                self.splitter._update_candidates(actual_size)
                self._cached_image_size = actual_size

        # I98-1: Pipeline 架构 - 先调用 Splitter，再调用 Tokenizer
        # Splitter 决策哪些区域需要细分
        features = self._feature_extractor(img)

        # I98-2: 检测 tokenizer 是否需要 split_result
        # 对于自定义 tokenizer (没有 shared_conv)，假设不需要 split_result
        needs_split_result = hasattr(self.tokenizer, 'shared_conv')

        if needs_split_result:
            split_result = self.splitter(
                features,
                image_size=(img.shape[2], img.shape[3]),
                hard=not self.training,
            )
            # Tokenizer 使用 Splitter 的结果进行 embedding
            token_output = self.tokenizer.tokenize(img, split_result)
        else:
            # 自定义 tokenizer 已处理所有逻辑
            token_output = self.tokenizer.tokenize(img)

        # P9-5 优化: 使用预填充缓存接口，避免 O(B) Python 循环
        # I98-4: info_dim = max_level + 1 对应 levels_info 的 (depth + paths) 结构
        info_dim = self.max_level + 1
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
    ) -> Tuple[torch.Tensor, "LevelsInfo"]:
        """添加位置编码和 CLS token。

        I98-4: 返回 LevelsInfo 而非 raw tensor

        Args:
            padded_tokens: 填充后的 tokens [B, MaxLen, Dim]
            padded_levels: 填充后的层级信息 [B, MaxLen, max_level+1]
            regions: (I31-3) 区域边界张量，形状 [B, N, 4]
            image_size: (I31-3) 图像尺寸，可以是整数或 (W, H) 元组

        Returns:
            (x, levels_info):
            - x: 带位置编码和 CLS 的序列 [B, 1+MaxLen, Dim]
            - levels_info: LevelsInfo 实例（包含 CLS）
        """
        batch_size = padded_tokens.shape[0]
        device = padded_tokens.device

        # I98-4: 将 raw tensor 转换为 LevelsInfo
        from .levels_info import LevelsInfo
        levels_info = LevelsInfo(data=padded_levels, max_depth=self.max_level)

        # I31-3: 传递 regions 和 image_size 给位置编码器（用于面积编码）
        pos_emb = self.pos_embedding(levels_info, regions=regions, image_size=image_size)
        x = padded_tokens + pos_emb

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # I98-4: 构造包含 CLS 的 LevelsInfo
        cls_level = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        cls_depths = levels_info.depths  # (B, MaxLen)
        all_depths = torch.cat([cls_level, cls_depths], dim=1)  # (B, 1+MaxLen)
        # all_paths shape: (B, 1+MaxLen, max_level)
        all_paths = torch.zeros(batch_size, all_depths.shape[1], self.max_level, dtype=torch.long, device=device)
        all_paths[:, 1:, :] = levels_info.paths  # (B, 1+MaxLen, max_level)

        levels_info_with_cls = LevelsInfo.from_arrays(all_depths, all_paths, max_depth=self.max_level)

        x = self.dropout(x)

        return x, levels_info_with_cls

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
            valid_counts = token_mask.sum(dim=1, keepdim=True).float().clamp(min=DIVISION_EPSILON)
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
                valid_counts = token_mask.sum(dim=1, keepdim=True).float().clamp(min=DIVISION_EPSILON)
                return sum_x / valid_counts

            token_x = x[:, 1:]  # [B, N, D] - 排除 CLS
            token_mask = ~key_padding_mask[:, 1:]  # [B, N] - 排除 CLS

            # I30-11: split_probs 形状为 [B, N]，与 token_x/token_mask 对齐
            # 无需再切片，直接使用
            token_probs = split_probs  # [B, N]

            # 有效性 mask
            token_probs = token_probs * token_mask.float()

            # 权重归一化: w_norm = w / sum(w)
            weight_sum = token_probs.sum(dim=-1, keepdim=True).clamp(min=PROB_EPSILON)
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
        split_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Dict[str, Any]], List[torch.Tensor]]:
        """准备辅助输出。

        Args:
            batch_size: batch 大小
            lengths: Tensor[B] 有效 token 数量
            levels_list: 层级信息列表
            pooled: 池化后的表示
            return_aux_info: 是否返回辅助信息
            return_features: 是否返回特征
            split_probs: Tensor[B, N] 分割概率（可选）

        Returns:
            (aux_infos, features_list)

        性能优化 (P-OPT-5):
            - aux_info 仅在验证/调试时使用，保持简单实现
            - 使用 non_blocking 转移减少同步等待
        """
        aux_infos: List[Dict[str, Any]] = []
        features_list: List[torch.Tensor] = []

        if return_aux_info:
            # M3: 添加缺失字段 (token_selection_entropy, depth_distribution)
            # P-OPT-5: 批量获取 lengths 到 CPU，避免多次 .item() 调用
            lengths_cpu = lengths.to('cpu', non_blocking=True)

            # P-OPT: 向量化深度分布计算
            max_depth = self.tokenizer.max_depth if hasattr(self, 'tokenizer') else 8
            max_depth_range = max_depth + 1

            # 预分配所有样本的深度分布
            all_depth_counts = torch.zeros(batch_size, max_depth_range, dtype=torch.float32, device='cpu')
            all_levels_used: List[List[int]] = [[] for _ in range(batch_size)]

            for i in range(batch_size):
                l = levels_list[i]
                if l.numel() > 0:
                    # 提取深度并过滤负值
                    depths = l[:, 0].to(dtype=torch.int64, device='cpu')
                    depths = depths[depths >= 0]
                    if depths.numel() > 0:
                        # 使用 bincount 向量化计数
                        d_max = int(depths.max().item())
                        d_max = min(d_max, max_depth)
                        counts = torch.bincount(depths, minlength=max_depth_range)[:max_depth_range].float()
                        all_depth_counts[i] = counts
                        # 记录使用的层级
                        all_levels_used[i] = [d for d in range(d_max + 1) if counts[d] > 0]

            # 计算归一化分布
            depth_sums = all_depth_counts.sum(dim=1, keepdim=True).clamp(min=1e-8)
            normalized_counts = all_depth_counts / depth_sums

            for i in range(batch_size):
                num_tokens = int(lengths_cpu[i].item())

                # 构建稀疏分布字典
                depth_distribution: Dict[int, float] = {}
                for d in all_levels_used[i]:
                    val = normalized_counts[i, d].item()
                    if val > 0:
                        depth_distribution[d] = val

                aux_info = {
                    "num_tokens": num_tokens,
                    "levels_used": all_levels_used[i],
                    "depth_distribution": depth_distribution,
                    # I78: 添加缺失的 splitter_diagnostics 字段（与 FractalModelProtocol 对齐）
                    "splitter_diagnostics": self.get_splitter_diagnostics(),
                }

                # M3: 计算选择熵 (token_selection_entropy)
                if split_probs is not None:
                    probs_i = split_probs[i, :lengths[i]]
                    # 避免 log(0)
                    probs_safe = probs_i.clamp(min=PROB_EPSILON)
                    entropy = -(probs_safe * torch.log(probs_safe)).sum().item()
                    aux_info["token_selection_entropy"] = entropy

                aux_infos.append(aux_info)

        if return_features:
            # 直接返回 pooled 张量 [B, D] 而非 List[Tensor]
            # P1 Fix: 避免冗余的 list comprehension + stack 操作
            features_tensor = pooled

        return aux_infos, features_tensor if return_features else None

    # H2: Protocol compliance - overload signatures for type checkers
    @overload
    def forward(
        self,
        img: torch.Tensor,
        return_aux_info: bool = False,
        return_features: bool = False,
        return_tokens: bool = False,
    ) -> torch.Tensor: ...

    @overload
    def forward(
        self,
        img: torch.Tensor,
        return_aux_info: bool = True,
        return_features: bool = False,
        return_tokens: bool = False,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]: ...

    @overload
    def forward(
        self,
        img: torch.Tensor,
        return_aux_info: bool = False,
        return_features: bool = True,
        return_tokens: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def forward(
        self,
        img: torch.Tensor,
        return_aux_info: bool = True,
        return_features: bool = True,
        return_tokens: bool = False,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]], torch.Tensor]: ...

    @overload
    def forward(
        self,
        img: torch.Tensor,
        return_aux_info: bool = False,
        return_features: bool = False,
        return_tokens: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def forward(
        self,
        img: torch.Tensor,
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
        # P0-2: 自动转换 channels_last 内存格式以优化卷积性能
        # 检测输入是否为 4D 且是 channels_first (stride(1) != stride(2))
        if (img.dim() == 4 and
            img.stride(1) != img.stride(2) and
            hasattr(self, '_channels_last_enabled') and
            self._channels_last_enabled):
            img = img.to(memory_format=torch.channels_last)

        batch_size = img.shape[0]
        device = img.device

        # 1. 准备 tokens (P11-3: 返回 token_output 用于获取 regions)
        padded_tokens, padded_levels, lengths, levels_list, token_output = self._prepare_tokens(img)
        
        # P11-3: 获取 regions 和 image_size 用于正确的 LCA 偏置计算
        regions, image_size = token_output.get_padded_regions()

        # 2. 添加位置编码和 CLS token
        # I31-3: 传递 regions 和 image_size 用于面积编码
        # I98-4: 返回 (x, levels_info) 而非 (x, padded_levels)
        x, levels_info = self._apply_position_and_cls(
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
        # I98-4: 传递 LevelsInfo 而非 raw tensor
        x = self.transformer(x, levels_info, attn_mask, regions=regions, image_size=image_size)
        
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
            aux_infos, features_tensor = self._prepare_auxiliary_output(
                batch_size, lengths, levels_list, pooled, return_aux_info, return_features,
                split_probs=split_probs
            )

            if return_aux_info and return_features:
                return final_output, aux_infos, features_tensor
            if return_aux_info:
                return final_output, aux_infos
            if return_features:
                return final_output, features_tensor

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
        """清空 tokenizer 和 splitter 的内部状态，应在每个 batch 结束后调用 (I98-2).

        封装 tokenizer 和 splitter 的状态清理，提供清晰的公共接口。
        """
        # I98-2: 清理 tokenizer 状态（如果有）
        if hasattr(self.tokenizer, "clear_saved_actions"):
            self.tokenizer.clear_saved_actions()
        elif hasattr(self.tokenizer, "reset_state"):
            self.tokenizer.reset_state()

        # I98-2: 清理 splitter 状态（如果有）
        if hasattr(self.splitter, "clear_cache"):
            self.splitter.clear_cache()

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
            # I98-2 Bug 修复: 使用完整 pipeline，需要 split_result 参数
            features = self._feature_extractor(img)
            split_result = self.splitter(
                features,
                image_size=(img.shape[2], img.shape[3]),
                hard=True,
            )
            token_output = self.tokenizer.tokenize(img, split_result)
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

    # =========================================================================
    # I36: FractalModelProtocol 接口实现
    # =========================================================================

    def get_extra_info(
        self,
        img: torch.Tensor,
        return_aux_info: bool = True,
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, Any]]]]:
        """获取辅助信息 (I36-3: FractalModelProtocol 实现)

        Args:
            img: 输入图像 [B, C, H, W]
            return_aux_info: 是否返回 aux_info

        Returns:
            logits: 分类输出 [B, num_classes]
            aux_infos: 辅助信息列表，每个元素对应一个样本
        """
        return self.forward(img, return_aux_info=return_aux_info)

    def configure_training(self, config: Dict[str, Any]) -> None:
        """配置训练相关参数 (I36-2: 解耦设计, I99-对齐修复)

        设计原则: 训练器通过协议接口配置，不直接访问内部实现

        I99修复: 统一 Splitter 访问路径，同时支持:
            - 新架构: self.splitter (I98-2 依赖注入模式)
            - 旧架构: self.tokenizer.splitter

        Args:
            config: 配置字典，包含:
                - temperature_annealing: bool - 是否启用温度退火
                - total_steps: int - 总训练步数（必须提供，否则使用默认值可能不正确）
                - temp_start: float - 起始温度
                - temp_end: float - 结束温度
                - aux_loss_weights: Dict[str, float] - 辅助损失权重
        """
        # 统一 Splitter 访问路径 (I99-对齐修复)
        splitter = None

        # 优先检查新架构: self.splitter (I98-2 依赖注入模式)
        if hasattr(self, 'splitter'):
            splitter = self.splitter
        # 兼容旧架构: self.tokenizer.splitter
        elif hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'splitter'):
            splitter = self.tokenizer.splitter

        # 温度退火配置
        if config.get('temperature_annealing') and splitter is not None:
            if hasattr(splitter, 'enable_temperature_annealing'):
                # I78: 修复 - 必须明确要求 total_steps，避免使用错误的默认值
                if 'total_steps' not in config:
                    import warnings
                    warnings.warn(
                        "configure_training: 'total_steps' not in config. "
                        "Using default 10000 which may be incorrect for your training run. "
                        "Please pass total_steps explicitly.",
                        UserWarning,
                        stacklevel=2
                    )
                total_steps = config.get('total_steps', 10000)
                splitter.enable_temperature_annealing(
                    total_steps=total_steps,
                    T_start=config.get('temp_start', 1.0),
                    T_end=config.get('temp_end', 0.5),
                    schedule=config.get('schedule', 'exponential'),
                )

        # 辅助损失权重配置
        if 'aux_loss_weights' in config:
            weights = config['aux_loss_weights']
            # 可以在此处更新内部辅助损失权重
            # 目前使用默认值，留作扩展接口

    def get_splitter_diagnostics(self) -> Dict[str, Any]:
        """获取分割器诊断信息 (I36-3: FractalModelProtocol 实现, I99-对齐修复)

        I99修复: 统一 Splitter 访问路径，同时支持:
            - 新架构: self.splitter (I98-2 依赖注入模式)
            - 旧架构: self.tokenizer.splitter

        Returns:
            诊断字典，包含:
                - current_temperature: float - 当前温度
                - depth_distribution: Dict[int, float] - 深度分布 (L3: 优化类型)
                - quota_allocation: List[float] - 配额分配
                - num_selected: int - 选中的 token 数
                - has_splitter: bool - 是否有分割器 (L3: 新增)
                - splitter_type: str - 分割器类型 (L3: 新增)
        """
        diagnostics: Dict[str, Any] = {
            'current_temperature': 1.0,
            'depth_distribution': {},  # Dict[int, float]
            'quota_allocation': [],   # List[float]
            'num_selected': 0,
            'has_splitter': False,
            'splitter_type': 'None',
        }

        # 统一 Splitter 访问路径 (I99-对齐修复)
        splitter = None

        # 优先检查新架构: self.splitter (I98-2 依赖注入模式)
        if hasattr(self, 'splitter'):
            splitter = self.splitter
        # 兼容旧架构: self.tokenizer.splitter
        elif hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'splitter'):
            splitter = self.tokenizer.splitter

        if splitter is not None:
            diagnostics['has_splitter'] = True
            diagnostics['splitter_type'] = type(splitter).__name__

            # 当前温度
            if hasattr(splitter, 'get_current_temperature'):
                diagnostics['current_temperature'] = splitter.get_current_temperature()

            # 深度分布
            if hasattr(splitter, 'get_depth_distribution'):
                diagnostics['depth_distribution'] = splitter.get_depth_distribution()

            # 配额分配
            if hasattr(splitter, 'quota_logits') and hasattr(splitter, '_current_max_depth'):
                D = splitter._current_max_depth + 1
                quota = torch.softmax(splitter.quota_logits[:D], dim=0)
                diagnostics['quota_allocation'] = quota.detach().cpu().tolist()

            # 选中的 token 数
            if hasattr(splitter, '_avg_selected'):
                diagnostics['num_selected'] = int(splitter._avg_selected)

        return diagnostics

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型诊断信息 (I36-7: 完整诊断)

        Returns:
            完整诊断信息:
                - architecture: Dict - 架构参数
                - tokenizer: Dict - tokenizer 配置
                - splitter: Dict - splitter 参数
                - total_params: int - 总参数量
                - trainable_params: int - 可训练参数量
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # 架构参数
        architecture = {
            'num_classes': self.num_classes,
            'dim': self.dim,
            'depth': self.depth,
            'heads': self.heads,
            'mlp_dim': self.mlp_dim,
            'pool': self.pool,
        }

        # Tokenizer 配置
        tokenizer_info: Dict[str, Any] = {}
        if hasattr(self, 'tokenizer'):
            t = self.tokenizer
            tokenizer_info = {
                'type': type(t).__name__,
            }
            if hasattr(t, 'max_depth'):
                tokenizer_info['max_depth'] = t.max_depth
            if hasattr(t, 'K_min'):
                tokenizer_info['K_min'] = t.K_min
            if hasattr(t, 'K_max'):
                tokenizer_info['K_max'] = t.K_max

        # Splitter 参数 (I99: 统一访问路径 - 先检查 self.splitter，再回退到 self.tokenizer.splitter)
        splitter_info: Dict[str, Any] = {}
        if hasattr(self, 'splitter'):
            s = self.splitter
        elif hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'splitter'):
            s = self.tokenizer.splitter
        else:
            s = None
        if s is not None:
            splitter_info = {
                'type': type(s).__name__,
            }
            if hasattr(s, 'quota_logits'):
                splitter_info['quota_dim'] = s.quota_logits.shape[0]
            if hasattr(s, 'enable_temperature_annealing'):
                splitter_info['has_temp_annealing'] = True

        return {
            'architecture': architecture,
            'tokenizer': tokenizer_info,
            'splitter': splitter_info,
            'total_params': total_params,
            'trainable_params': trainable_params,
        }

    # =========================================================================
    # I35: torch.compile 优化支持
    # =========================================================================

    def compile(
        self,
        mode: str = "max-autotune",
        dynamic: bool = False,
        fullgraph: bool = False,
    ) -> "FractalCurveViT":
        """编译模型以获得最佳性能 (I35)。

        使用 torch.compile 优化模型，支持多种优化模式：
        - "default": 基础优化
        - "reduce-overhead": 减少开销优化
        - "max-autotune": 自动调优最优 kernel (推荐)

        自动配置以下 PyTorch 2.4 优化：
        - TF32 (TensorFloat-32) 加速矩阵运算
        - cuDNN SDP (Scaled Dot-Product Attention)
        - Inductor max-autotune 优化

        Args:
            mode: 编译模式
            dynamic: 是否启用动态形状支持 (用于可变分辨率)
            fullgraph: 是否要求完整图编译

        Returns:
            编译后的模型

        Example:
            >>> model = FractalCurveViT(image_size=224, num_classes=1000)
            >>> model = model.compile(mode="max-autotune")
            >>> # 或对于动态分辨率
            >>> model = model.compile(mode="reduce-overhead", dynamic=True)
        """
        import torch

        # P0-3: 启用 TF32 (Ampere+ GPU) - 约 10x 矩阵运算加速
        if torch.backends.cuda.is_built():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

            # 启用 cuDNN SDP - 使用 cuDNN 内核的 Flash Attention
            torch.backends.cuda.enable_cudnn_sdp(True)

        # 配置 inductor 优化
        torch._inductor.config.max_autotune = True
        torch._inductor.config.cudnn_sdp = True  # 启用 cuDNN attention
        torch._inductor.config.coordinate_descent_tuning = True

        # 编译模型
        self = torch.compile(self, mode=mode, dynamic=dynamic, fullgraph=fullgraph)

        return self

    def enable_channels_last(self) -> "FractalCurveViT":
        """启用 channels_last 内存格式以优化卷积性能 (I35)。

        将模型和输入转换为 channels_last 格式，可提升卷积操作性能。
        自动在 forward 方法中转换输入格式。

        Returns:
            配置后的模型

        Example:
            >>> model = FractalCurveViT(image_size=224, num_classes=1000)
            >>> model = model.enable_channels_last()
            >>> # 输入自动转换为 channels_last
        """
        self.to(memory_format=torch.channels_last)
        # P0-2: 设置标志以在 forward 中自动转换输入
        self._channels_last_enabled = True
        return self


# 保持向后兼容性的别名
EnhancedFractalViT = FractalCurveViT
NextGenerationFractalViT = FractalCurveViT


# =============================================================================
# I98-2: 工厂函数 - 依赖注入模式
# =============================================================================

def create_fractal_vit(
    image_size: Optional[Union[int, Tuple[int, int]]] = None,
    num_classes: int = 1000,
    *,
    # 模型维度配置
    dim: int = 512,
    depth: int = 6,
    heads: int = 8,
    mlp_dim: int = 1024,
    # 图像处理配置
    channels: int = 3,
    min_patch_size: Union[int, Tuple[int, int]] = 4,
    max_level: Optional[int] = None,
    # Tokenizer 配置
    use_hilbert_encoding: bool = True,
    use_spatial_encoding: bool = True,
    # Splitter 配置
    K_min: int = 8,
    K_max: int = 64,
    enable_learnable_quota: Optional[bool] = None,
    # Transformer 配置
    dropout: float = 0.0,
    emb_dropout: float = 0.0,
    drop_path_rate: float = 0.0,
    ffn_type: FFNType = "swiglu_level",
    use_checkpoint: bool = False,
    # 位置编码配置
    use_area_encoding: bool = False,
    use_affine_modulation: bool = True,
    fourier_levels: int = 4,
    # 输出配置
    pool: str = "cls",
    # 编码器配置 (I98-3)
    encoder_config: Optional[AttentionEncoderConfig] = None,
    # I104-3: FP16 存储 LCA embedding
    lca_fp16: bool = False,
) -> FractalCurveViT:
    """创建 FractalCurveViT 实例的工厂函数。

    工厂函数模式：组件通过依赖注入组合，支持任意 Protocol 实现替换。

    数学形式:
        Model = FractalViT(
            splitter=GumbelTopKSplitter(Config_S),
            tokenizer=StreamingFractalTokenizerV3(Config_T),
            transformer=FractalTransformer(Config_A),
            position_embedding=FractalPositionEmbedding(Config_P),
            mlp_head=MLP_Head(Config_H)
        )

    Args:
        image_size: 输入图像尺寸，None 表示动态分辨率
        num_classes: 分类类别数
        dim: 模型嵌入维度
        depth: Transformer 层数
        heads: 注意力头数
        mlp_dim: MLP 隐藏层维度
        channels: 输入图像通道数
        min_patch_size: 最小 patch 大小（用于动态计算 max_depth）
        max_level: 最大递归层级，None 表示自动计算
        use_hilbert_encoding: 是否使用 Hilbert 编码
        use_spatial_encoding: 是否使用空间编码
        K_min: 最少 token 数量
        K_max: 最多 token 数量
        enable_learnable_quota: 是否启用可学习配额
        dropout: Dropout 比率
        emb_dropout: 嵌入层 Dropout 比率
        drop_path_rate: Drop path 比率
        ffn_type: FFN 类型
        use_checkpoint: 是否使用梯度检查点
        use_area_encoding: 是否使用面积编码
        use_affine_modulation: 是否使用仿射调制
        fourier_levels: 傅里叶特征级别数
        pool: 池化策略 ('cls', 'mean', 'weighted')
        encoder_config: 注意力编码器配置

    Returns:
        FractalCurveViT: 配置好的模型实例

    Example:
        >>> # 动态分辨率模式
        >>> model = create_fractal_vit(
        ...     image_size=None,
        ...     num_classes=200,
        ...     dim=384,
        ...     depth=8,
        ... )
        >>> # 固定分辨率模式
        >>> model = create_fractal_vit(
        ...     image_size=224,
        ...     num_classes=1000,
        ...     dim=512,
        ... )
    """
    from .gumbel_topk_splitter import GumbelTopKSplitter
    from .config import SplitterConfig

    # 处理 min_patch_size
    if isinstance(min_patch_size, tuple):
        effective_min_patch_size = min_patch_size[0]
    else:
        effective_min_patch_size = min_patch_size

    # 处理 image_size
    if image_size is None:
        # 动态分辨率模式
        estimated_size = effective_min_patch_size * 64
        actual_image_size = pair(estimated_size)
        dynamic_image_size = True
    else:
        actual_image_size = pair(image_size)
        dynamic_image_size = False

    # 确定 max_level
    if max_level is None:
        max_level = 8  # 默认值

    # 创建 Splitter
    splitter_config = SplitterConfig(
        feature_dim=dim,
        min_patch_size=effective_min_patch_size,
        max_depth_limit=max_level,
        hidden_dim=64,
        intermediate_dim=64,
        pool_size=4,
        K_min=K_min,
        K_max=K_max,
        use_dynamic_k=True,
        dropout=min(dropout, 0.15),
        enable_learnable_quota=enable_learnable_quota if enable_learnable_quota is not None else True,
    )
    splitter = GumbelTopKSplitter(
        config=splitter_config,
        image_size=actual_image_size,
    )

    # 创建 Tokenizer
    tokenizer = StreamingFractalTokenizerV3(
        image_size=actual_image_size,
        channels=channels,
        d_model=dim,
        base_patch_size=effective_min_patch_size,
        max_depth=max_level,
        min_patch_size=effective_min_patch_size,
    )

    # 创建 Position Embedding
    if use_area_encoding:
        from .embed_fractal_position import AreaEnhancedPositionEmbedding
        position_embedding = AreaEnhancedPositionEmbedding(
            dim=dim,
            max_level=max_level,
            fourier_levels=fourier_levels,
            use_hilbert_encoding=use_hilbert_encoding,
            use_spatial_encoding=use_spatial_encoding,
            dropout=emb_dropout if emb_dropout > 0 else dropout * 0.5,
        )
    else:
        position_embedding = FractalPositionEmbedding(
            dim=dim,
            max_level=max_level,
            max_seq_len=10000,
            use_hilbert_encoding=use_hilbert_encoding,
            use_spatial_encoding=use_spatial_encoding,
            dropout=emb_dropout if emb_dropout > 0 else dropout * 0.5,
        )

    # 创建 Transformer
    transformer = FractalTransformer(
        dim=dim,
        depth=depth,
        heads=heads,
        dim_head=dim // heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        max_level=max_level,
        drop_path_rate=drop_path_rate,
        ffn_type=ffn_type,
        use_checkpoint=use_checkpoint,
        lca_temperature=1.5,
        learnable_temperature=True,
        use_affine_modulation=use_affine_modulation,
        fourier_levels=fourier_levels,
        encoder_config=encoder_config,
        use_fp16=lca_fp16,  # I104-3
    )

    # 创建 MLP Head
    mlp_head = nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, mlp_dim // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_dim // 2, num_classes),
    )

    # 创建 CLS token
    cls_token = nn.Parameter(torch.randn(1, 1, dim))

    # 组装 Model
    model = FractalCurveViT(
        # 注入组件
        splitter=splitter,
        tokenizer=tokenizer,
        transformer=transformer,
        position_embedding=position_embedding,
        mlp_head=mlp_head,
        cls_token=cls_token,
        # 配置
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        pool=pool,
        image_size=actual_image_size,
        max_level=max_level,
        dynamic_image_size=dynamic_image_size,
    )

    return model
