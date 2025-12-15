"""Token processor module for fractal ViT.

This module contains the EnhancedFractalTokenProcessor class, which handles
token processing with multi-scale features, edge information, and texture analysis.

.. deprecated:: 0.4.0
    此模块已废弃。功能已集成到 StreamingFractalTokenizer。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# 使用父包的导入
from vit_pytorch.features import TokenFeatures, compute_token_features
from vit_pytorch.tokenization import BaseTokenProcessor, TokenSequence, TokenizerOutput


class EnhancedFractalTokenProcessor(BaseTokenProcessor):
    """
    增强的分形token处理器，对齐tokenizer的特征提取能力
    支持多尺度特征、边缘信息和纹理分析
    
    .. deprecated:: 0.4.0
        推荐使用 :class:`StreamingFractalTokenizer`，其将 patch 分割与特征处理
        统一为单次前向，无需单独的 TokenProcessor。
        使用 ``NextGenerationFractalViT(tokenizer_type='streaming')`` 启用。
        此类将在 v1.0 中移除。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        min_patch_size: Tuple[int, int] = (4, 4),
        max_level: int = 50,
        use_feature_enhancement: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.min_patch_size = min_patch_size
        self.max_level = max_level
        self.use_feature_enhancement = use_feature_enhancement

        # 基础token投影
        self.token_norm = nn.LayerNorm(input_dim)
        self.token_projection = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
        )

        # 层级类型嵌入
        self.level_type_embedding = nn.Embedding(max_level + 1, output_dim)

        # 特征增强网络（模拟tokenizer的特征提取）
        if use_feature_enhancement:
            # 统计特征处理器
            self.stats_processor = nn.Sequential(
                nn.Linear(2, output_dim // 4),  # 输入：方差和均值
                nn.ReLU(),
                nn.Linear(output_dim // 4, output_dim // 4),
            )

            # 边缘特征处理器
            self.edge_processor = nn.Sequential(
                nn.Linear(1, output_dim // 4),  # 输入：边缘密度
                nn.ReLU(),
                nn.Linear(output_dim // 4, output_dim // 4),
            )

            # 空间特征处理器
            self.spatial_processor = nn.Sequential(
                nn.Linear(2, output_dim // 4),  # 输入：高度和宽度
                nn.ReLU(),
                nn.Linear(output_dim // 4, output_dim // 4),
            )

            # 层级特征处理器
            self.level_processor = nn.Sequential(
                nn.Linear(1, output_dim // 4),  # 输入：层级
                nn.ReLU(),
                nn.Linear(output_dim // 4, output_dim // 4),
            )

            # 特征融合网络
            self.feature_fusion = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )

        # 多尺度适配器
        # REFACTORED: Replaced ModuleList with shared adapter + level embedding
        self.level_embedding = nn.Embedding(max_level + 1, output_dim)
        self.shared_scale_adapter = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),  # Input: concatenated token + level_emb
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

        # 动态权重网络
        self.dynamic_weighting = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, 1),
            nn.Sigmoid(),
        )

    def _extract_token_features(
        self, tokens: torch.Tensor, level: int, patch_size: Tuple[int, int]
    ) -> TokenFeatures:
        return compute_token_features(tokens, level=level, patch_size=patch_size)

    def process(
        self,
        batch: TokenizerOutput,
        patch_info_list: Optional[List[Dict[str, Any]]] = None,
    ) -> TokenizerOutput:
        processed_sequences: List[TokenSequence] = []

        if patch_info_list and len(patch_info_list) != len(batch):
            raise ValueError(
                f"TokenProcessor.process: patch_info_list length ({len(patch_info_list)}) "
                f"does not match token sequence count ({len(batch)}). "
                f"Hint: Ensure patch_info_list has one entry per image in the batch."
            )

        for index, sequence in enumerate(batch):
            tokens = sequence.tokens
            levels = sequence.get_levels()
            metadata = dict(sequence.metadata)

            if tokens.numel() == 0:
                empty = torch.empty(0, self.output_dim, device=tokens.device)
                if levels is not None and "levels" not in metadata:
                    metadata["levels"] = levels
                processed_sequences.append(TokenSequence(tokens=empty, metadata=metadata))
                continue

            level_tensor = (
                levels
                if levels is not None
                else torch.empty(0, 0, dtype=torch.long, device=tokens.device)
            )

            # 基础处理
            norm_tokens = self.token_norm(tokens)
            proj_tokens = self.token_projection(norm_tokens)

            # 添加层级类型嵌入
            if level_tensor.numel() > 0 and level_tensor.shape[0] > 0:
                depths = level_tensor[:, 0].clamp(0, self.max_level)
                type_emb = self.level_type_embedding(depths)
                proj_tokens = proj_tokens + type_emb

            if self.use_feature_enhancement and level_tensor.numel() > 0:
                patch_info = (
                    patch_info_list[index]
                    if patch_info_list
                    else sequence.metadata.get("patch_info", {})
                )
                patch_height = patch_info.get("height", self.min_patch_size[0])
                patch_width = patch_info.get("width", self.min_patch_size[1])
                level_value = (
                    int(level_tensor[0, 0].item()) if level_tensor.shape[0] > 0 else 0
                )

                features = self._extract_token_features(
                    tokens, level_value, (patch_height, patch_width)
                )

                stats_feat = self.stats_processor(features.stats)
                edge_feat = self.edge_processor(features.edge)
                spatial_feat = self.spatial_processor(features.spatial)
                level_feat = self.level_processor(features.level)

                enhanced_feat = torch.cat(
                    [stats_feat, edge_feat, spatial_feat, level_feat], dim=-1
                )
                enhanced_feat = self.feature_fusion(enhanced_feat)
                proj_tokens = proj_tokens + enhanced_feat * 0.3

            # 多尺度自适应
            if level_tensor.numel() > 0 and level_tensor.shape[0] > 0:
                # Vectorized implementation
                depths = level_tensor[:, 0].clamp(0, self.max_level).long()

                # 1. Get level embeddings: (seq_len, dim)
                level_embs = self.level_embedding(depths)

                # 2. Concatenate: (seq_len, dim * 2)
                adapter_input = torch.cat([proj_tokens, level_embs], dim=-1)

                # 3. Apply shared adapter
                final_tokens = self.shared_scale_adapter(adapter_input)
            else:
                final_tokens = proj_tokens

            weights = self.dynamic_weighting(final_tokens)
            final_tokens = final_tokens * (0.8 + 0.4 * weights)

            if levels is not None and "levels" not in metadata:
                metadata["levels"] = levels

            processed_sequences.append(TokenSequence(tokens=final_tokens, metadata=metadata))

        return TokenizerOutput(processed_sequences)
