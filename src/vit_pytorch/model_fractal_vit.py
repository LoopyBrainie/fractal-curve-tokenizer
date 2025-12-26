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
    
    Attributes:
        image_size: 输入图像尺寸
        num_classes: 分类类别数
        dim: 模型维度
        pool: 池化策略 ('cls', 'mean' 或其他)
        max_level: 最大递归层级
        tokenizer: 图像 tokenizer
        token_processor: token 处理器
        pos_embedding: 位置编码
        transformer: Transformer 模块
    """

    def __init__(
        self,
        *,
        image_size: Union[int, Tuple[int, int]],
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
        min_patch_size: Tuple[int, int] = (4, 4),
        max_level: int = 50,
        use_hilbert_encoding: bool = True,
        use_spatial_encoding: bool = True,
        use_checkpoint: bool = False,
        drop_path_rate: float = 0.0,
        ffn_type: FFNType = 'swiglu_level',
        tokenizer: Optional[BaseTokenizer] = None,
        position_embedding: Optional[FractalPositionEmbedding] = None,
        # Streaming Tokenizer 配置
        tokenizer_type: TokenizerType = "streaming_v3",
        num_scales: int = 4,
        # Hilbert Bias 配置
        hilbert_bias_mode: str = 'lca',
        low_rank_r: int = 32,
    ) -> None:
        """初始化 FractalCurveViT。
        
        Args:
            image_size: 输入图像尺寸（整数或 (H, W) 元组）
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
            min_patch_size: 最小 patch 尺寸
            max_level: 最大递归层级
            use_hilbert_encoding: 是否使用 Hilbert 编码
            use_spatial_encoding: 是否使用空间编码
            ffn_type: FFN 变体 ('gelu', 'swiglu', 'swiglu_level')
            tokenizer: 自定义 tokenizer（可选，若提供则忽略 tokenizer_type）
            position_embedding: 自定义位置编码（可选）
            tokenizer_type: tokenizer 类型 ("streaming_v3" - Variable Depth Tokens)
            num_scales: 多尺度金字塔层数
            hilbert_bias_mode: Hilbert Bias 计算模式
                - 'lca': LCA 嵌入表（推荐，~40参数，显式几何意义）
                - 'low_rank': 低秩分解（显存友好，~50K参数）
                - 'hierarchical': 分层计算（可解释性强）
            low_rank_r: 低秩分解的秩参数（仅当 hilbert_bias_mode='low_rank' 时有效）
        """
        super().__init__()

        self.image_size = pair(image_size)
        self.num_classes = num_classes
        self.dim = dim
        self.pool = pool
        self.max_level = max_level
        self.use_checkpoint = use_checkpoint
        self.ffn_type = ffn_type
        
        # 保存 tokenizer 类型
        self.tokenizer_type = tokenizer_type
        self._is_streaming = True  # 现在所有 tokenizer 都是 streaming 模式
        self.hilbert_bias_mode = hilbert_bias_mode
        self.low_rank_r = low_rank_r

        # === Tokenizer 选择逻辑 ===
        if tokenizer is not None:
            # 用户提供自定义 tokenizer，直接使用
            pass
        elif tokenizer_type == "streaming_v3":
            base_ps = min_patch_size[0]
            max_depth_v3 = num_scales - 1  # num_scales 个尺度对应 max_depth = num_scales - 1
            tokenizer = StreamingFractalTokenizerV3(
                image_size=self.image_size,
                channels=channels,
                d_model=dim,
                base_patch_size=base_ps,
                max_depth=max_depth_v3,
            )
        else:
            raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}. Use 'streaming_v3'.")

        # 允许外部访问统一接口
        self.tokenizer = tokenizer
        # 兼容旧属性名
        self.fractal_tokenizer = tokenizer

        # Streaming tokenizer 已直接输出 D-dim embeddings，无需 token_processor
        self.token_processor = None

        # 高级分形位置编码
        if position_embedding is None:
            position_embedding = FractalPositionEmbedding(
                dim=dim,
                max_level=max_level,
                max_seq_len=10000,
                use_hilbert_encoding=use_hilbert_encoding,
                use_spatial_encoding=use_spatial_encoding,
            )

        self.pos_embedding = position_embedding
        self.position_embedding = position_embedding

        # CLS token和dropout
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # 混合池化选择器 (用于 pool 不是 'cls' 或 'mean' 时)
        # 输入: [B, D, N]，输出: [B, 2] 表示 (cls_weight, mean_weight)
        self.pooling_selector = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [B, D, 1]
            nn.Flatten(),              # [B, D]
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1),
        )
        
        # 策略梯度损失权重 (保留接口兼容性，实际值为0因为使用Gumbel-Softmax)
        self.register_buffer("aux_loss_weight", torch.tensor(0.0))

        # 分形Transformer
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
            hilbert_bias_mode=hilbert_bias_mode,
            low_rank_r=low_rank_r,
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
        """初始化权重 - 遵循 ViT 标准初始化"""
        # CLS token: 使用较小的标准差
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 分类头：使用较小的标准差初始化，最后一层更小
        for module in self.mlp_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # 最后一层（分类层）使用更小的标准差
        for module in reversed(list(self.mlp_head.modules())):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                break
        
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
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[torch.Tensor]]:
        """准备 tokens 和进行 padding。
        
        数学形式化：
            T, L = S(I) where N varies per image in variable_tokens mode
        
        Args:
            img: 输入图像 [B, C, H, W]
            
        Returns:
            (padded_tokens, padded_levels, lengths, levels_list):
            - padded_tokens: tokens [B, MaxN, Dim] (padded)
            - padded_levels: 层级信息 [B, MaxN, InfoDim]
            - lengths: 每个样本的实际 token 数量
            - levels_list: 原始层级列表（用于辅助输出）
        """
        batch_size = img.shape[0]
        device = img.device

        # Streaming tokenizer 直接输出 D-dim embeddings
        token_output = self.tokenizer.tokenize(img)
        
        # 使用 TokenizerOutput 的标准方法获取数据
        tokens_list = token_output.tokens_list()   # List[Tensor[N_i, D]]
        levels_raw = token_output.levels_list()    # List[Tensor[N_i, info_len]]
        
        # 获取每个样本的实际 token 数量
        lengths = [t.shape[0] for t in tokens_list]
        max_len = max(lengths)
        
        # 使用 pad_sequence 处理可变长度 tokens
        # pad_sequence 默认 batch_first=False，需要转置
        padded_tokens = torch.nn.utils.rnn.pad_sequence(
            tokens_list, batch_first=True, padding_value=0.0
        )  # [B, MaxN, D]
        
        # 构建 level info
        info_dim = self.max_level + 4
        if levels_raw[0].dim() == 1:
            # 如果是 1D，需要扩展并 padding
            padded_levels = torch.zeros(
                batch_size, max_len, info_dim,
                dtype=torch.long, device=device
            )
            for i, lv in enumerate(levels_raw):
                n = lv.shape[0]
                padded_levels[i, :n, 0] = lv
        else:
            # 已经是 2D [N_i, info_len]，需要 padding
            padded_levels = torch.zeros(
                batch_size, max_len, info_dim,
                dtype=torch.long, device=device
            )
            for i, lv in enumerate(levels_raw):
                n = lv.shape[0]
                info_len = lv.shape[1]
                copy_len = min(info_len, info_dim)
                padded_levels[i, :n, :copy_len] = lv[:, :copy_len]
        
        levels_list = levels_raw
        
        return padded_tokens, padded_levels, lengths, levels_list

    def _apply_position_and_cls(
        self,
        padded_tokens: torch.Tensor,
        padded_levels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """添加位置编码和 CLS token。
        
        Args:
            padded_tokens: 填充后的 tokens [B, MaxLen, Dim]
            padded_levels: 填充后的层级信息 [B, MaxLen, InfoDim]
            
        Returns:
            (x, padded_levels):
            - x: 带位置编码和 CLS 的序列 [B, 1+MaxLen, Dim]
            - padded_levels: 更新后的层级信息（包含 CLS） [B, 1+MaxLen, InfoDim]
        """
        batch_size = padded_tokens.shape[0]
        device = padded_tokens.device

        pos_emb = self.pos_embedding(padded_levels)
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
        lengths: List[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建 attention mask。
        
        Args:
            batch_size: batch 大小
            seq_len: 序列长度（包含 CLS）
            lengths: 每个样本的有效 token 数量（不含 CLS）
            device: 设备
            
        Returns:
            (attn_mask, key_padding_mask):
            - attn_mask: attention mask [B, 1, 1, Seq]
            - key_padding_mask: padding mask [B, Seq]
        """
        key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        
        for i, length in enumerate(lengths):
            if length + 1 < seq_len:
                key_padding_mask[i, length + 1:] = True

        attn_mask = ~key_padding_mask
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)

        return attn_mask, key_padding_mask

    def _apply_pooling(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """应用池化策略。
        
        Args:
            x: transformer 输出 [B, Seq, Dim]
            key_padding_mask: padding mask [B, Seq]
            
        Returns:
            pooled: 池化后的表示 [B, Dim]
        """
        if self.pool == "cls":
            return x[:, 0]
        elif self.pool == "mean":
            token_x = x[:, 1:]
            token_mask = ~key_padding_mask[:, 1:]
            token_x = token_x * token_mask.unsqueeze(-1).float()
            sum_x = token_x.sum(dim=1)
            valid_counts = token_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            return sum_x / valid_counts
        else:
            # 混合池化
            pooling_weights = self.pooling_selector(x.transpose(1, 2))
            cls_pooled = x[:, 0]
            
            token_x = x[:, 1:]
            token_mask = ~key_padding_mask[:, 1:]
            token_x = token_x * token_mask.unsqueeze(-1).float()
            sum_x = token_x.sum(dim=1)
            valid_counts = token_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            mean_pooled = sum_x / valid_counts
            
            return pooling_weights[:, 0:1] * cls_pooled + pooling_weights[:, 1:2] * mean_pooled

    def _prepare_auxiliary_output(
        self,
        batch_size: int,
        lengths: List[int],
        levels_list: List[torch.Tensor],
        pooled: torch.Tensor,
        return_aux_info: bool,
        return_features: bool,
    ) -> Tuple[List[Dict[str, Any]], List[torch.Tensor]]:
        """准备辅助输出。
        
        Args:
            batch_size: batch 大小
            lengths: 有效 token 数量列表
            levels_list: 层级信息列表
            pooled: 池化后的表示
            return_aux_info: 是否返回辅助信息
            return_features: 是否返回特征
            
        Returns:
            (aux_infos, features_list)
        """
        aux_infos: List[Dict[str, Any]] = []
        features_list: List[torch.Tensor] = []

        if return_aux_info:
            for i in range(batch_size):
                l = levels_list[i]
                if l.numel() > 0:
                    depths = l[:, 0]
                    unique = depths.unique().tolist()
                    aux_infos.append({"num_tokens": lengths[i], "levels_used": unique})
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
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, List[Dict[str, Any]]],
        Tuple[torch.Tensor, List[torch.Tensor]],
        Tuple[torch.Tensor, List[Dict[str, Any]], List[torch.Tensor]],
    ]:
        """前向传播。
        
        Args:
            img: 输入图像，形状为 [B, C, H, W]
            return_attention: 是否返回注意力权重（已弃用）
            return_aux_info: 是否返回辅助信息
            return_features: 是否返回特征
            
        Returns:
            根据参数返回不同类型：
            - 默认：分类 logits [B, num_classes]
            - return_aux_info=True：(logits, aux_infos)
            - return_features=True：(logits, features)
            - 两者都为 True：(logits, aux_infos, features)
        """
        batch_size = img.shape[0]
        device = img.device

        # 1. 准备 tokens
        padded_tokens, padded_levels, lengths, levels_list = self._prepare_tokens(img)

        # 2. 添加位置编码和 CLS token
        x, padded_levels = self._apply_position_and_cls(padded_tokens, padded_levels)

        # 3. 创建 attention mask
        attn_mask, key_padding_mask = self._create_attention_mask(
            batch_size, x.shape[1], lengths, device
        )

        # 4. Transformer 处理
        x = self.transformer(x, padded_levels, attn_mask)

        # 5. 池化
        pooled = self._apply_pooling(x, key_padding_mask)
        pooled = self.to_latent(pooled)
        final_output = self.mlp_head(pooled)

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
        # 无需策略梯度损失
        return torch.tensor(0.0, device=self.aux_loss_weight.device)
    
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
