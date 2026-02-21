# -*- coding: utf-8 -*-
"""
双路径Fractal ViT架构 (I162-1)

数学形式化
============

核心设计：Dual-Path Architecture

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      DualPathFractalViT_v2                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Image → Quadtree → Tokens                                            │
    │                               │                                         │
    │                               ▼                                         │
    │   ┌─────────────────────────────────────────────────────────────────┐  │
    │   │                      Dual-Path Processing                        │  │
    │   │                                                                       │  │
    │   │   Path 1: Direct Pathway           Path 2: Pattern Pathway      │  │
    │   │   ┌─────────────────────┐        ┌─────────────────────────┐   │  │
    │   │   │ Tokens              │        │ Tokens → Hilbert重排     │   │  │
    │   │   │        ↓            │        │ → 1D卷积(多尺度)         │   │  │
    │   │   │ Position Encoding   │        │ → Pattern Features       │   │  │
    │   │   │        ↓            │        │           ↓             │   │  │
    │   │   │ Transformer Blocks  │        │ Position Encoding       │   │  │
    │   │   └─────────────────────┘        │        ↓                │   │  │
    │   │              ↓                   │ Transformer Blocks      │   │  │
    │   │        [T_token]                 └─────────────────────────┘   │  │
    │   │              ↓                                ↓                   │  │
    │   │              └────────────────┬───────────────────────        │  │
    │   │                               ↓                               │  │
    │   │                    ┌─────────────────────┐                 │  │
    │   │                    │   Feature Fusion    │                 │  │
    │   │                    │ F = [T_token; T_pattern]             │  │
    │   │                    │ F = LayerNorm(F) → FFN              │  │
    │   │                    └─────────────────────┘                 │  │
    │   └─────────────────────────────────────────────────────────────────┘  │
    │                               ↓                                         │
    │                    Output (Classification)                              │
    └─────────────────────────────────────────────────────────────────────────┘

数学性质
--------
1. Hilbert局部性保证:
   ∀i, ||pos_i - pos_{i+1}||_2 ≤ √2
   → 1D卷积相邻感受野对应2D空间邻域

2. 双路径互补性:
   - Direct: 保持原始Token语义信息
   - Pattern: 捕捉空间异质性模式
   - Fusion: F = f_direct ⊕ f_pattern

3. 计算效率:
   - Direct Path: O(L × N² × d)  (标准Transformer)
   - Pattern Path: O(L × N × k × d)  (k << N)
   - Fusion: O(N × d)

与原架构的关系
--------------
   FractalCurveViT (原) = DualPathFractalViT (pattern_path_enabled=False)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_pytorch.layers.embeddings.fractal_path import BitFlippedPositionEncoder
from vit_pytorch.modules.tokenizer import StreamingFractalTokenizerV3
from vit_pytorch.modules.base_tokenizer import BaseTokenizer, TokenizerOutput
from vit_pytorch.modules.transformer_block import FractalTransformer, FFNType
from vit_pytorch.core.utils import pair
from vit_pytorch.core.constants import (
    DIVISION_EPSILON, PROB_EPSILON,
    LOGIT_CLAMP_BOUND,
)
from vit_pytorch.core.pattern_encoder import (
    HilbertPatternEncoder,
    HilbertPatternEncoderLight,
    create_hilbert_pattern_encoder,
)


@dataclass
class TrainingStatsV2:
    """V2架构的训练统计信息（扩展版）"""
    logits: torch.Tensor
    num_tokens: Union[int, List[int], torch.Tensor]
    depth_used: int
    depth_distribution: Dict[int, float]
    features: torch.Tensor
    transformer_tokens: torch.Tensor

    # V2新增字段
    pattern_features: Optional[torch.Tensor] = None  # [B, N, D]
    direct_features: Optional[torch.Tensor] = None   # [B, N, D]


class DualPathFractalViT(nn.Module):
    """双路径Fractal ViT (I162-1 最佳实现)

    同时利用Hilbert序的两种价值：
    1. 副产品：LCA深度 → Attention偏置约束（保持局部性）
    2. 本体价值：Hilbert序 → 模式发现（捕捉空间异质性）

    与Wang et al. (2021)方法的对齐：
        Wang: 图像 → Hilbert序 → CNN → 特征
        本架构: Token → Hilbert序 → 1D卷积 → 模式特征

    核心等价：两者都在Hilbert序上做滑动窗口特征提取

    Args:
        pattern_encoder_mode: 模式编码器类型 ("light", "standard", "multihead")
        pattern_window_sizes: 模式编码器卷积核大小
        pattern_dropout: 模式编码器dropout
    """

    def __init__(
        self,
        *,
        # 注入组件（可选）
        splitter: Optional[Any] = None,
        tokenizer: Optional[BaseTokenizer] = None,
        transformer: Optional[Any] = None,
        position_embedding: Optional[BitFlippedPositionEncoder] = None,
        mlp_head: Optional[nn.Sequential] = None,
        cls_token: Optional[nn.Parameter] = None,
        # 架构参数
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        num_classes: int = 1000,
        dim: int = 512,
        num_layers: int = 6,
        heads: int = 8,
        mlp_dim: int = 1024,
        pool: str = "weighted",
        channels: int = 3,
        dim_head: int = 64,
        # V2新增：模式编码器参数
        pattern_encoder_mode: str = "light",
        pattern_window_sizes: Tuple[int, ...] = (3, 7, 15),
        pattern_dim: Optional[int] = None,
        pattern_dropout: float = 0.0,
        pattern_scale: float = 1.0,
        # Dropout参数（V2：默认值0，保持确定性）
        tokenizer_dropout: float = 0.0,
        transformer_dropout: float = 0.0,
        emb_dropout: float = 0.0,
        min_patch_size: Union[int, Tuple[int, int]] = 4,
        # Hilbert编码参数
        use_hilbert_encoding: bool = True,
        use_spatial_encoding: bool = True,
        use_checkpoint: bool = False,
        ffn_type: FFNType = 'swiglu_level',
        # 其他参数...
        token_coverage_min: float = 0.01,
        token_coverage_max: Optional[float] = None,
        target_ratio: float = 0.5,
        use_area_encoding: bool = False,
        use_affine_modulation: bool = True,
        fourier_levels: int = 4,
        encoder_config: Optional[Any] = None,
        quota_learnable: Optional[bool] = None,
        quota_entropy_weight: float = 0.5,  # P3-FIX: 统一为 0.5
        lca_fp16: bool = False,
        splitter_hidden_dim: Optional[int] = None,
        splitter_feature_dim: Optional[int] = None,
        splitter_pool_size: Optional[int] = None,
        splitter_temp_start: Optional[float] = None,
        splitter_temp_end: Optional[float] = None,
        splitter_type: str = 'gumbel_topk',
        use_semantic_splitter: bool = False,
        semantic_splitter_config: Optional[Any] = None,
        depth_scale_range: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.dim = dim
        self.pool = pool
        self.use_checkpoint = use_checkpoint
        self.ffn_type = ffn_type

        # V2新增：模式编码器配置
        self.pattern_encoder_mode = pattern_encoder_mode
        self.pattern_window_sizes = pattern_window_sizes
        self.pattern_dim = pattern_dim or dim

        # I150-4: pattern_scale 改为可学习参数（原来是 float，无法学习）
        self.pattern_scale = nn.Parameter(torch.tensor(pattern_scale, dtype=torch.float32))

        # Token覆盖率参数
        self.token_coverage_min = token_coverage_min

        # 1. 创建Tokenizer（如果未注入）
        if tokenizer is None:
            tokenizer = StreamingFractalTokenizerV3(
                image_size=image_size,
                channels=channels,
                dim=dim,
                patch_size=None,
                patch_size_list=None,
                overlapped=False,
                use_hilbert=True,
                min_patch_size=min_patch_size,
                depth_scale_range=depth_scale_range,
                splitter_type=splitter_type,
                splitter_hidden_dim=splitter_hidden_dim,
                splitter_feature_dim=splitter_feature_dim,
                splitter_pool_size=splitter_pool_size,
                temp_start=splitter_temp_start,
                temp_end=splitter_temp_end,
                use_semantic=use_semantic_splitter,
                semantic_config=semantic_splitter_config,
            )
        self.tokenizer = tokenizer

        # 获取max_level
        self.max_level = self.tokenizer.max_level

        # 2. 位置编码 (v6.0: 使用 BitFlippedPositionEncoder)
        if position_embedding is None:
            position_embedding = BitFlippedPositionEncoder(
                dim=dim,
                max_level=self.max_level,
                grid_size=256,
            )
        self.pos_embedding = position_embedding

        # 3. V2新增：模式编码器
        if pattern_encoder_mode == "disabled":
            self.pattern_encoder = None
        else:
            self.pattern_encoder = create_hilbert_pattern_encoder(
                dim=self.pattern_dim,
                mode=pattern_encoder_mode,
                window_sizes=pattern_window_sizes,
                out_dim=self.pattern_dim,
                dropout=pattern_dropout,
            )
            # 初始化缩放因子
            if self.pattern_encoder is not None and hasattr(self.pattern_encoder, 'scale'):
                nn.init.constant_(self.pattern_encoder.scale, pattern_scale)

        # 4. CLS Token
        if cls_token is None:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        else:
            self.cls_token = cls_token

        # 5. Transformer
        if transformer is None:
            transformer = FractalTransformer(
                dim=dim,
                depth=num_layers,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                dropout=transformer_dropout,
                ffn_type=ffn_type,
                encoder_config=encoder_config,
                quota_learnable=quota_learnable,
                quota_entropy_weight=quota_entropy_weight,
                lca_fp16=lca_fp16,
                max_level=self.max_level,
            )
        self.transformer = transformer

        # 6. 池化层
        self.pool_type = pool

        # 7. MLP Head
        if mlp_head is None:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim * 2 if self.pattern_encoder else dim),
                nn.Linear(dim * 2 if self.pattern_encoder else dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_dim, num_classes),
                LogitsClamp(),  # V2：保持钳制
            )
        else:
            self.mlp_head = mlp_head

    def forward(self, x: torch.Tensor) -> TrainingStatsV2:
        """前向传播

        Args:
            x: [B, C, H, W] 输入图像

        Returns:
            TrainingStatsV2: 包含logits和所有中间特征
        """
        B = x.shape[0]

        # 1. Tokenization
        token_output: TokenizerOutput = self.tokenizer(x)
        tokens = token_output.tokens  # [B, N, D]
        levels_info = token_output.levels_info  # [B, N, D+1]

        # 2. 位置编码 (v6.0: BitFlippedPositionEncoder 返回 pos_emb 和 geometry_emb)
        pos_emb, geometry_emb = self.pos_embedding(levels_info)
        tokens = tokens + pos_emb  # 基础位置编码

        # 3. 添加CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_with_cls = torch.cat([cls_tokens, tokens], dim=1)  # [B, N+1, D]

        # v6.0: 为 CLS 添加零几何嵌入
        cls_geometry = torch.zeros(B, 1, self.dim, device=tokens.device)
        geometry_emb_with_cls = torch.cat([cls_geometry, geometry_emb], dim=1)  # [B, N+1, D]

        # ============================================================
        # V2新增：双路径处理
        # ============================================================

        if self.pattern_encoder is not None:
            # 获取Hilbert排序
            hilbert_order = self._get_hilbert_order(levels_info, tokens.shape[1] - 1)

            # Path 2: 模式编码分支
            # 使用原始tokens（不含CLS）进行模式编码
            pattern_features = self.pattern_encoder(
                tokens,  # [B, N, D]（不含CLS）
                hilbert_order
            )  # [B, N, D]

            # 在CLS位置填充零，保持维度一致
            pattern_features_with_cls = torch.cat([
                torch.zeros(B, 1, self.pattern_dim, device=tokens.device),
                pattern_features
            ], dim=1)  # [B, N+1, D]

            # ============================================================
            # 双路径并行处理
            # ============================================================

            # Path 1: 直接分支 → 原始Transformer (v6.0: 注入 geometry_emb)
            direct_out = self.transformer(
                x_with_cls,
                levels_info=levels_info,
                geometry_emb=geometry_emb_with_cls,
            )  # [B, N+1, D]

            # Path 2: 模式分支 → 需要构建模式增强的输入
            # 将模式特征融入Transformer
            # 方法：残差连接
            pattern_with_cls = pattern_features_with_cls * self.pattern_scale
            pattern_out = self.transformer(
                pattern_with_cls,
                levels_info=levels_info,
                geometry_emb=geometry_emb_with_cls,
            )  # [B, N+1, D]

            # ============================================================
            # 特征融合
            # ============================================================

            # 从两个分支获取特征（不含CLS）
            direct_token_feats = direct_out[:, 1:, :]  # [B, N, D]
            pattern_token_feats = pattern_out[:, 1:, :]  # [B, N, D]

            # 拼接两个分支
            fused = torch.cat([direct_token_feats, pattern_token_feats], dim=-1)  # [B, N, 2D]

            # 添加回CLS位置
            cls_feat = direct_out[:, 0:1, :]  # [B, 1, D]
            fused_with_cls = torch.cat([cls_feat, fused], dim=1)  # [B, N+1, 2D]

            # 使用CLS进行分类
            pooled = fused_with_cls[:, 0, :]  # [B, 2D]

            # 记录模式特征用于诊断
            pattern_features_for_stats = pattern_features

        else:
            # V1兼容模式：无模式编码器 (v6.0: 注入 geometry_emb)
            direct_out = self.transformer(
                x_with_cls,
                levels_info=levels_info,
                geometry_emb=geometry_emb_with_cls,
            )  # [B, N+1, D]

            pooled = direct_out[:, 0, :]  # [B, D]
            fused = direct_out[:, 1:, :]  # [B, N, D]
            pattern_features_for_stats = None

        # ============================================================
        # 分类头
        # ============================================================

        logits = self.mlp_head(pooled)  # [B, num_classes]

        # ============================================================
        # 统计信息
        # ============================================================

        # 深度分布
        depth_dist = self._compute_depth_distribution(levels_info)

        stats = TrainingStatsV2(
            logits=logits,
            num_tokens=token_output.num_tokens,
            depth_used=token_output.depth_used,
            depth_distribution=depth_dist,
            features=pooled,
            transformer_tokens=fused,
            pattern_features=pattern_features_for_stats,
            direct_features=direct_out[:, 1:, :] if self.pattern_encoder else None,
        )

        return stats

    def _get_hilbert_order(
        self,
        levels_info: Any,
        num_tokens: int
    ) -> torch.Tensor:
        """从levels_info提取或计算Hilbert排序

        Args:
            levels_info: LevelsInfo对象或原始张量
            num_tokens: Token数量（不含CLS）

        Returns:
            [N] Hilbert排序索引
        """
        # 尝试从tokenizer获取预计算的Hilbert排序
        if hasattr(self.tokenizer, 'get_hilbert_order'):
            try:
                order = self.tokenizer.get_hilbert_order(num_tokens)
                return order
            except Exception:
                pass

        # 回退：使用默认排序
        return torch.arange(num_tokens, device=levels_info.device)

    def _compute_depth_distribution(self, levels_info: Any) -> Dict[int, float]:
        """计算深度分布

        Args:
            levels_info: LevelsInfo对象

        Returns:
            {深度: 比例}的字典
        """
        depths = levels_info.data[:, :, 0]  # [B, N]
        valid_mask = depths >= 0
        valid_depths = depths[valid_mask]

        if len(valid_depths) == 0:
            return {}

        depth_counts = torch.bincount(valid_depths.long(), minlength=self.max_level + 1)
        total = len(valid_depths)

        return {
            d: (count / total).item()
            for d, count in enumerate(depth_counts)
            if count > 0
        }

    def get_pattern_encoder(self) -> Optional[HilbertPatternEncoder]:
        """获取模式编码器，用于诊断"""
        return self.pattern_encoder

    def set_pattern_scale(self, scale: float):
        """设置模式特征的缩放因子"""
        if self.pattern_encoder is not None and hasattr(self.pattern_encoder, 'scale'):
            self.pattern_encoder.scale.data.fill_(scale)

    @property
    def has_pattern_encoder(self) -> bool:
        """检查是否启用了模式编码器"""
        return self.pattern_encoder is not None and self.pattern_encoder_mode != "disabled"


class FractalCurveViTV2(DualPathFractalViT):
    """FractalCurveViT的V2版本（别名）

    提供与原版相同的接口，但默认启用模式编码器。
    """

    def __init__(
        self,
        *,
        # V2参数（带默认值）
        pattern_encoder_mode: str = "light",
        pattern_window_sizes: Tuple[int, ...] = (3, 7),
        pattern_scale: float = 0.5,
        # 其他参数（保持与原版一致）
        **kwargs
    ):
        # 设置默认值
        kwargs.setdefault('pattern_encoder_mode', pattern_encoder_mode)
        kwargs.setdefault('pattern_window_sizes', pattern_window_sizes)
        kwargs.setdefault('pattern_scale', pattern_scale)

        super().__init__(**kwargs)


# 向后兼容别名
FractalCurveViT_V2 = DualPathFractalViT
