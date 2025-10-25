from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import TokenFeatures, compute_token_features
from .fractal_curve_tokenizer import FractalHilbertTokenizer
from .positional import AdvancedFractalPositionEmbedding
from .tokenization import BaseTokenProcessor, BaseTokenizer, TokenSequence, TokenizerOutput
from .transformer import EnhancedFractalTransformer
from .utils import create_attention_mask, pair


class EnhancedFractalTokenProcessor(BaseTokenProcessor):
    """
    增强的分形token处理器，对齐tokenizer的特征提取能力
    支持多尺度特征、边缘信息和纹理分析
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
        self.scale_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(output_dim, output_dim),
                    nn.LayerNorm(output_dim),
                    nn.GELU(),
                )
                for _ in range(max_level + 1)
            ]
        )

        # 动态权重网络
        self.dynamic_weighting = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, 1),
            nn.Sigmoid(),
        )

    def _extract_token_features(self, tokens: torch.Tensor, level: int, patch_size: Tuple[int, int]) -> TokenFeatures:
        return compute_token_features(tokens, level=level, patch_size=patch_size)

    def process(
        self,
        batch: TokenizerOutput,
        patch_info_list: Optional[List[Dict[str, Any]]] = None,
    ) -> TokenizerOutput:
        processed_sequences: List[TokenSequence] = []

        if patch_info_list and len(patch_info_list) != len(batch):
            raise ValueError("Length of patch_info_list must match the number of token sequences.")

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

            level_tensor = levels if levels is not None else torch.empty(0, 0, dtype=torch.long, device=tokens.device)

            # 基础处理
            norm_tokens = self.token_norm(tokens)
            proj_tokens = self.token_projection(norm_tokens)

            # 添加层级类型嵌入
            if level_tensor.numel() > 0 and level_tensor.shape[0] > 0:
                depths = level_tensor[:, 0].clamp(0, self.max_level)
                type_emb = self.level_type_embedding(depths)
                proj_tokens = proj_tokens + type_emb

            if self.use_feature_enhancement and level_tensor.numel() > 0:
                patch_info = patch_info_list[index] if patch_info_list else sequence.metadata.get("patch_info", {})
                patch_height = patch_info.get("height", self.min_patch_size[0])
                patch_width = patch_info.get("width", self.min_patch_size[1])
                level_value = int(level_tensor[0, 0].item()) if level_tensor.shape[0] > 0 else 0

                features = self._extract_token_features(tokens, level_value, (patch_height, patch_width))

                stats_feat = self.stats_processor(features.stats)
                edge_feat = self.edge_processor(features.edge)
                spatial_feat = self.spatial_processor(features.spatial)
                level_feat = self.level_processor(features.level)

                enhanced_feat = torch.cat([stats_feat, edge_feat, spatial_feat, level_feat], dim=-1)
                enhanced_feat = self.feature_fusion(enhanced_feat)
                proj_tokens = proj_tokens + enhanced_feat * 0.3

            # 多尺度自适应
            if level_tensor.numel() > 0 and level_tensor.shape[0] > 0:
                adapted = []
                for j, token in enumerate(proj_tokens):
                    if j < level_tensor.shape[0]:
                        level_idx = int(level_tensor[j, 0].clamp(0, self.max_level).item())
                        adapted_token = self.scale_adapters[level_idx](token.unsqueeze(0)).squeeze(0)
                    else:
                        adapted_token = token
                    adapted.append(adapted_token)

                final_tokens = torch.stack(adapted)
            else:
                final_tokens = proj_tokens

            weights = self.dynamic_weighting(final_tokens)
            final_tokens = final_tokens * (0.8 + 0.4 * weights)

            if levels is not None and "levels" not in metadata:
                metadata["levels"] = levels

            processed_sequences.append(TokenSequence(tokens=final_tokens, metadata=metadata))

        return TokenizerOutput(processed_sequences)


class NextGenerationFractalViT(nn.Module):
    """
    下一代分形ViT，完全对齐增强的FractalHilbertTokenizer特性：

    核心特性：
    - 可学习的分割决策网络（6特征输入）
    - 真正的Hilbert曲线递归算法（支持多方向）
    - 动态层级管理（最大50层）
    - 智能patch处理和特征增强
    - 边缘检测和纹理复杂度分析
    - 自适应多尺度处理
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
        learnable_split: bool = True,
        adaptive_threshold: float = 0.5,
        use_hilbert_encoding: bool = True,
        use_spatial_encoding: bool = True,
        use_feature_enhancement: bool = True,
        use_dynamic_depth: bool = False,
        tokenizer: Optional[BaseTokenizer] = None,
        token_processor: Optional[BaseTokenProcessor] = None,
        position_embedding: Optional["AdvancedFractalPositionEmbedding"] = None,
    ):
        super().__init__()

        self.image_size = pair(image_size)
        self.num_classes = num_classes
        self.dim = dim
        self.pool = pool
        self.max_level = max_level
        self.use_dynamic_depth = use_dynamic_depth

        if tokenizer is None:
            tokenizer = FractalHilbertTokenizer(
                min_patch_size=min_patch_size,
                max_level=max_level,
                learnable_split=learnable_split,
                adaptive_threshold=adaptive_threshold,
            )

        # 允许外部访问统一接口
        self.tokenizer = tokenizer
        # 兼容旧属性名
        self.fractal_tokenizer = tokenizer

        # 计算token维度
        patch_dim = channels * min_patch_size[0] * min_patch_size[1]

        # 增强的token处理器
        if token_processor is None:
            token_processor = EnhancedFractalTokenProcessor(
                input_dim=patch_dim,
                output_dim=dim,
                min_patch_size=min_patch_size,
                max_level=max_level,
                use_feature_enhancement=use_feature_enhancement,
            )

        self.token_processor = token_processor

        # 高级分形位置编码
        if position_embedding is None:
            position_embedding = AdvancedFractalPositionEmbedding(
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

        # 增强的分形Transformer
        self.transformer = EnhancedFractalTransformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            max_level=max_level,
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

        # 层级权重（动态平衡不同层级token的贡献）
        self.level_weights = nn.Parameter(torch.ones(max_level + 1))

        # 可学习的池化策略选择
        self.pooling_selector = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(dim, 2),  # CLS vs Mean pooling
            nn.Softmax(dim=-1),
        )

        # 辅助损失权重
        self.aux_loss_weight = nn.Parameter(torch.tensor(0.1))

        # 特征分析器
        self.feature_analyzer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 6),
        )

    def forward(
        self,
        img: torch.Tensor,
        return_attention: bool = False,
        return_aux_info: bool = False,
        return_features: bool = False,
    ):
        batch_size = img.shape[0]
        device = img.device

        # 1. 分形tokenization
        token_output = self.tokenizer.tokenize(img)
        processed_output = self.token_processor(token_output) if self.token_processor is not None else token_output
        legacy_output = processed_output.to_legacy()
        tokens_list = legacy_output.tokens
        levels_list = legacy_output.levels

        if len(tokens_list) != batch_size:
            raise ValueError("Tokenizer output sequence count does not match batch size.")

        # 2. 批量处理
        batch_outputs = []
        aux_infos = []
        features_list = []

        for b in range(batch_size):
            tokens = tokens_list[b]
            levels = levels_list[b]

            if tokens.device != device:
                tokens = tokens.to(device)
            if levels.device != device:
                levels = levels.to(device)

            if tokens.numel() == 0 or tokens.shape[0] == 0:
                batch_outputs.append(torch.zeros(1, self.num_classes, device=device))
                if return_aux_info:
                    aux_infos.append({
                        "num_tokens": 0,
                        "levels_used": [],
                        "token_distribution": torch.zeros(self.max_level + 1, device=device),
                    })
                if return_features:
                    features_list.append(torch.zeros(6, device=device))
                continue

            # 3. Token处理和投影
            x = tokens.unsqueeze(0)
            level_info = levels

            # 4. 位置编码
            if level_info.numel() > 0:
                seq_positions = torch.arange(x.shape[1], device=device)
                pos_emb = self.pos_embedding(level_info, seq_positions)
                x = x + pos_emb.unsqueeze(0)

            # 5. 添加CLS token
            cls_tokens = self.cls_token.expand(1, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            if level_info.numel() > 0:
                cls_level = torch.zeros(1, level_info.shape[1], device=device, dtype=level_info.dtype)
                level_info = torch.cat([cls_level, level_info], dim=0)

            x = self.dropout(x)

            # 6. 创建注意力掩码
            attention_mask = None
            if level_info.numel() > 0:
                token_levels = level_info[1:] if level_info.shape[0] > 1 else torch.empty(0, 0, device=device)
                base_mask = create_attention_mask([token_levels], device)
                attention_mask = torch.ones(1, x.shape[1], x.shape[1], device=device)
                if base_mask.numel() > 0:
                    seq_tokens = base_mask.shape[-1]
                    attention_mask[:, 1 : seq_tokens + 1, 1 : seq_tokens + 1] = base_mask[0, :seq_tokens, :seq_tokens]

            # 7. Transformer处理
            x = self.transformer(x, level_info, attention_mask, self.use_dynamic_depth)

            # 8. 池化策略选择
            if self.pool == "cls":
                pooled = x[:, 0].squeeze(0)
            elif self.pool == "mean":
                if x.shape[1] > 1:
                    if level_info.numel() > 0 and level_info.shape[0] > 1:
                        token_levels = level_info[1:, 0].clamp(0, self.max_level)
                        weights = F.softmax(self.level_weights[token_levels], dim=0)
                        pooled = torch.sum(x[0, 1:] * weights.unsqueeze(-1), dim=0)
                    else:
                        pooled = x[:, 1:].mean(dim=1).squeeze(0)
                else:
                    pooled = x[:, 0].squeeze(0)
            else:
                pooling_weights = self.pooling_selector(x.transpose(1, 2))
                cls_pooled = x[:, 0]
                mean_pooled = x[:, 1:].mean(dim=1) if x.shape[1] > 1 else cls_pooled
                pooled = (pooling_weights[0, 0] * cls_pooled + pooling_weights[0, 1] * mean_pooled).squeeze(0)

            pooled = self.to_latent(pooled)
            output = self.mlp_head(pooled.unsqueeze(0))

            batch_outputs.append(output)

            if return_aux_info:
                if level_info.numel() > 0 and level_info.shape[0] > 1:
                    depths = level_info[1:, 0]
                    unique_levels = depths.unique().tolist()
                    max_level_for_bincount = min(self.max_level, depths.max().item()) if depths.numel() > 0 else 0
                    token_dist = torch.bincount(depths.long(), minlength=int(max_level_for_bincount + 1)).float()
                    full_token_dist = torch.zeros(self.max_level + 1, device=device)
                    full_token_dist[: token_dist.shape[0]] = token_dist
                else:
                    unique_levels = []
                    full_token_dist = torch.zeros(self.max_level + 1, device=device)

                aux_info = {
                    "num_tokens": tokens.shape[0],
                    "levels_used": unique_levels,
                    "token_distribution": full_token_dist,
                }
                aux_infos.append(aux_info)

            if return_features:
                features = self.feature_analyzer(pooled)
                features_list.append(features)

        final_output = torch.cat(batch_outputs, dim=0)

        if return_aux_info and return_features:
            return final_output, aux_infos, features_list
        if return_aux_info:
            return final_output, aux_infos
        if return_features:
            return final_output, features_list
        return final_output

    def get_tokenizer_loss(self) -> torch.Tensor:
        """获取tokenizer的辅助损失（可学习分割决策的正则化）"""
        if hasattr(self.tokenizer, "split_decision") and self.tokenizer.split_decision is not None:
            weight_reg = sum(p.pow(2).sum() for p in self.tokenizer.split_decision.parameters())
            level_balance_loss = torch.var(self.level_weights)

            total_loss = weight_reg * 0.001 + level_balance_loss * 0.01

            return total_loss * self.aux_loss_weight

        return torch.tensor(0.0, requires_grad=True, device=self.aux_loss_weight.device)

    def analyze_tokenization(self, img: torch.Tensor) -> Dict[str, Any]:
        """分析tokenization过程，返回详细统计信息"""
        with torch.no_grad():
            token_output = self.tokenizer.tokenize(img)
            legacy_output = token_output.to_legacy()
            tokens_list = legacy_output.tokens
            levels_list = legacy_output.levels

            analysis = {"batch_size": len(tokens_list), "per_image_stats": [], "overall_stats": {}}

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
EnhancedFractalViT = NextGenerationFractalViT


class SimpleFractalViT(nn.Module):
    """简化版本，保持向后兼容性，使用合理默认参数"""

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
        max_level: int = 5,
    ):
        super().__init__()

        self.enhanced_model = NextGenerationFractalViT(
            image_size=image_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            min_patch_size=min_patch_size,
            max_level=max_level,
            learnable_split=False,
            use_hilbert_encoding=True,
            use_spatial_encoding=True,
            use_feature_enhancement=False,
            use_dynamic_depth=False,
        )

        self.tokenizer = self.enhanced_model.tokenizer
        self.fractal_tokenizer = self.enhanced_model.fractal_tokenizer
        self.token_processor = self.enhanced_model.token_processor
        self.pos_embedding = self.enhanced_model.pos_embedding
        self.cls_token = self.enhanced_model.cls_token
        self.dropout = self.enhanced_model.dropout
        self.transformer = self.enhanced_model.transformer
        self.pool = pool
        self.to_latent = self.enhanced_model.to_latent
        self.mlp_head = self.enhanced_model.mlp_head

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.enhanced_model(img, return_attention=False, return_aux_info=False, return_features=False)
