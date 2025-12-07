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
        # REFACTORED: Replaced ModuleList with shared adapter + level embedding
        self.level_embedding = nn.Embedding(max_level + 1, output_dim)
        self.shared_scale_adapter = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim), # Input: concatenated token + level_emb
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
                # Vectorized implementation
                depths = level_tensor[:, 0].clamp(0, self.max_level).long()
                
                # 1. Get level embeddings: (seq_len, dim)
                level_embs = self.level_embedding(depths)
                
                # 2. Expand to batch size if needed (assuming proj_tokens is [batch, seq_len, dim])
                # Note: proj_tokens comes from processing a single sequence in the loop, so it's (seq_len, dim)
                # Wait, the loop iterates over batch. So proj_tokens is (seq_len, dim).
                
                # 3. Concatenate: (seq_len, dim * 2)
                adapter_input = torch.cat([proj_tokens, level_embs], dim=-1)
                
                # 4. Apply shared adapter
                final_tokens = self.shared_scale_adapter(adapter_input)
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
                channels=channels,
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

        # 2. 准备 Batch Padding
        # 过滤掉空 Token 的情况 (虽然理论上不应该发生，但为了健壮性)
        valid_indices = []
        valid_tokens = []
        valid_levels = []
        lengths = []

        for i in range(batch_size):
            t = tokens_list[i]
            l = levels_list[i]
            if t.numel() > 0:
                valid_indices.append(i)
                valid_tokens.append(t)
                valid_levels.append(l)
                lengths.append(t.shape[0])
            else:
                # 处理空图片的情况: 创建一个 dummy token
                dummy_token = torch.zeros(1, self.dim, device=device)
                dummy_level = torch.zeros(1, self.max_level + 4, dtype=torch.long, device=device) # 假设 info_len 足够
                valid_indices.append(i)
                valid_tokens.append(dummy_token)
                valid_levels.append(dummy_level)
                lengths.append(1)

        # 使用 pad_sequence 进行对齐
        # padded_tokens: (B, Max_Len, Dim)
        padded_tokens = torch.nn.utils.rnn.pad_sequence(valid_tokens, batch_first=True)
        
        # padded_levels: (B, Max_Len, Info_Dim)
        # 注意: levels 的 info_len 可能不一致，需要先统一 info_len
        max_info_len = max([l.shape[1] for l in valid_levels])
        uniform_levels = []
        for l in valid_levels:
            if l.shape[1] < max_info_len:
                padding = torch.zeros(l.shape[0], max_info_len - l.shape[1], dtype=torch.long, device=device)
                l = torch.cat([l, padding], dim=1)
            uniform_levels.append(l)
            
        padded_levels = torch.nn.utils.rnn.pad_sequence(uniform_levels, batch_first=True, padding_value=0)

        # 3. 位置编码 (Batch 处理)
        # padded_levels: (B, Max_Len, Info_Dim)
        # 我们需要生成 sequence_positions: (B, Max_Len)
        max_len = padded_tokens.shape[1]
        seq_positions = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # AdvancedFractalPositionEmbedding 原生支持 (..., info_len) 输入
        # 因此可以直接传入 (B, MaxLen, Info) 格式，无需 flatten/reshape
        pos_emb = self.pos_embedding(padded_levels)  # (B, MaxLen, Dim)
        
        x = padded_tokens + pos_emb

        # 4. 添加 CLS Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (B, 1, Dim)
        x = torch.cat((cls_tokens, x), dim=1) # (B, 1+MaxLen, Dim)

        # 更新 Levels (添加 CLS 的 level info，全 0)
        cls_level = torch.zeros(batch_size, 1, padded_levels.shape[-1], dtype=torch.long, device=device)
        padded_levels = torch.cat([cls_level, padded_levels], dim=1) # (B, 1+MaxLen, Info_Dim)

        x = self.dropout(x)

        # 5. 创建 Attention Mask
        # True 表示被 Mask (不参与计算)，False 表示保留
        # 初始全 False (保留)
        # 形状: (B, 1+MaxLen) -> 扩展为 (B, 1, 1, 1+MaxLen) 或 (B, 1+MaxLen, 1+MaxLen)
        # PyTorch MultiheadAttention 的 key_padding_mask 是 (B, S)
        # 但我们的 Transformer Block 内部可能用了自定义 Attention
        
        # 构建 key_padding_mask: (B, 1+MaxLen)
        # CLS token (index 0) 永远有效
        key_padding_mask = torch.zeros(batch_size, x.shape[1], dtype=torch.bool, device=device)
        
        for i, length in enumerate(lengths):
            # 有效长度是 length，加上 CLS 是 length + 1
            # 所以从 length + 1 开始 mask
            if length + 1 < x.shape[1]:
                key_padding_mask[i, length + 1:] = True

        # 转换 mask 为 attention 矩阵所需的形状 (B, 1, Seq, Seq) 或 (B, Seq, Seq)
        # 我们的 Attention 模块接受 attention_mask
        # 如果是 True/False mask, 通常 True 表示 Mask 掉
        # 但在 utils.create_attention_mask 中，通常返回的是 1/0 mask (1保留, 0 mask)
        # 让我们检查一下 attention.py 的实现:
        # dots.masked_fill_(~attention_mask.bool(), mask_value)
        # 这意味着 attention_mask 必须是: True(保留), False(Mask掉)
        
        attn_mask = ~key_padding_mask # (B, Seq) -> True 保留
        # 扩展为 (B, 1, 1, Seq) 以便广播? 
        # Attention 内部: dots (B, H, N, N)
        # 我们需要 (B, 1, 1, N) 或者 (B, 1, N, N)
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, Seq)
        # 这样只会 mask Key，Query 都能关注到 Key
        
        # 6. Transformer 处理 (Batch 模式)
        # 注意: transformer.py 需要能处理 Batch 的 levels_info
        x = self.transformer(x, padded_levels, attn_mask, self.use_dynamic_depth)

        # 7. 池化策略
        if self.pool == "cls":
            pooled = x[:, 0]
        elif self.pool == "mean":
            # 只对非 Padding 部分求平均
            # x: (B, 1+MaxLen, Dim)
            # mask: (B, 1+MaxLen) True=Valid
            token_x = x[:, 1:] # (B, MaxLen, Dim)
            token_mask = ~key_padding_mask[:, 1:] # (B, MaxLen) True=Valid
            
            # 将 Padding 部分置 0
            token_x = token_x * token_mask.unsqueeze(-1).float()
            
            # 求和
            sum_x = token_x.sum(dim=1) # (B, Dim)
            
            # 有效数量
            valid_counts = token_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            
            pooled = sum_x / valid_counts
        else:
            # 混合池化
            pooling_weights = self.pooling_selector(x.transpose(1, 2)) # (B, 2)
            cls_pooled = x[:, 0]
            
            # Mean pooling logic
            token_x = x[:, 1:]
            token_mask = ~key_padding_mask[:, 1:]
            token_x = token_x * token_mask.unsqueeze(-1).float()
            sum_x = token_x.sum(dim=1)
            valid_counts = token_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            mean_pooled = sum_x / valid_counts
            
            pooled = pooling_weights[:, 0:1] * cls_pooled + pooling_weights[:, 1:2] * mean_pooled

        pooled = self.to_latent(pooled)
        final_output = self.mlp_head(pooled)

        # 8. 辅助信息 (可选)
        # 初始化为空列表，确保变量已定义
        aux_infos = []
        features_list = []

        if return_aux_info or return_features:
            if return_aux_info:
                for i in range(batch_size):
                    l = levels_list[i]
                    if l.numel() > 0:
                        depths = l[:, 0]
                        unique = depths.unique().tolist()
                        # ... 简化统计 ...
                        aux_infos.append({"num_tokens": lengths[i], "levels_used": unique})
                    else:
                        aux_infos.append({"num_tokens": 0})
            
            if return_features:
                # 批量计算 features
                batch_features = self.feature_analyzer(pooled)
                features_list = [f for f in batch_features]

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
            baseline: 基线值，用于减少方差。如果为 None，则不使用基线
            entropy_coef: 熵正则化系数，鼓励探索（默认 0.01）
        
        Returns:
            torch.Tensor: 策略梯度损失
        """
        loss = torch.tensor(0.0, device=self.aux_loss_weight.device)
        
        # 1. REINFORCE 策略梯度损失
        if hasattr(self.tokenizer, "saved_log_probs") and len(self.tokenizer.saved_log_probs) > 0:
            log_probs = torch.stack(self.tokenizer.saved_log_probs)
            
            # 真正的策略梯度：-reward * log_prob
            # reward > 0 时，增加该动作的概率
            # reward < 0 时，减少该动作的概率
            if reward is not None:
                # 计算优势函数 (advantage)
                advantage = reward if baseline is None else (reward - baseline)
                # 策略梯度：最大化 reward * log_prob，即最小化 -reward * log_prob
                policy_loss = -advantage * log_probs.mean()
                loss = loss + policy_loss
            
            # 2. 熵正则化 (鼓励探索，防止策略过早收敛)
            if hasattr(self.tokenizer, "saved_entropies") and len(self.tokenizer.saved_entropies) > 0:
                entropies = torch.stack(self.tokenizer.saved_entropies)
                # 负号：最大化熵 = 最小化负熵
                entropy_loss = -entropy_coef * entropies.mean()
                loss = loss + entropy_loss
        
        # 3. 分割决策网络的权重正则化 (防止过拟合)
        if hasattr(self.tokenizer, "split_decision") and self.tokenizer.split_decision is not None:
            weight_reg = sum(p.pow(2).sum() for p in self.tokenizer.split_decision.parameters())
            loss = loss + weight_reg * 1e-5

        return loss * self.aux_loss_weight
    
    def clear_tokenizer_cache(self) -> None:
        """清空 tokenizer 的动作缓存，应在每个 batch 结束后调用"""
        if hasattr(self.tokenizer, "clear_saved_actions"):
            self.tokenizer.clear_saved_actions()

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
