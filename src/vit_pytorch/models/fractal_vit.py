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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, overload
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_pytorch.layers.embeddings.fractal_path import BitFlippedPositionEncoder
from vit_pytorch.modules.tokenizer import StreamingFractalTokenizerV3
from vit_pytorch.modules.base_tokenizer import BaseTokenizer, TokenizerOutput
from vit_pytorch.modules.transformer_block import FractalTransformer, FFNType
from vit_pytorch.core.utils import pair
from vit_pytorch.core.constants import (
    EPS, DIVISION_EPSILON, PROB_EPSILON,
    compute_max_level, compute_num_candidates, compute_k_bounds,
    K_COVERAGE_MAX_HARD,
    SPLITTER_TEMP_START, SPLITTER_TEMP_END,
    LOGIT_CLAMP_BOUND,  # I147: 添加钳制边界导入
)
from vit_pytorch.core.config import AttentionEncoderConfig, SemanticSplitterConfig  # I98-3, I110-5
from vit_pytorch.core.pattern_encoder import (
    HilbertPatternEncoder,
    HilbertPatternEncoderLight,
    create_hilbert_pattern_encoder,
)  # I162-1
from vit_pytorch.core.pattern_plugin import (
    HilbertPatternPlugin,
    create_hilbert_pattern_plugin,
)  # I162-1: 双路径插件


# =============================================================================
# I147: Logits 钳制层 - 解决训练损失异常 (~82)
# =============================================================================
class LogitsClamp(nn.Module):
    """分类 Logits 钳制层 - 确保数值稳定性

    数学形式化
    =============
        y = clamp(x, min=-bound, max=bound)

    作用
    ----
        - 防止 logits 数值溢出导致 CE 损失爆炸
        - 保持 softmax 梯度有效性 (|z| ≤ 10 → ∂p/∂z ≥ 5e-5)
        - 替代 Focal Loss 的数值稳定化

    最佳实践
    =======
        - bound = LOGIT_CLAMP_BOUND = 10.0
        - 添加在 MLP Head 最后一层
        - 确保 train/eval 输出一致
    """
    __slots__ = ('bound',)

    def __init__(self, bound: float = LOGIT_CLAMP_BOUND):
        super().__init__()
        self.bound = bound

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=-self.bound, max=self.bound)

    def extra_repr(self) -> str:
        return f"bound={self.bound}"


# I112-6: LazyDiagnostics 延迟 diagnostics 包装器
# 避免 torch.compile 中 cudagraphs 的 CPU 同步问题
class LazyDiagnostics:
    """延迟计算的 diagnostics 包装器。

    设计原则: 避免 forward 关键路径中的 CPU 同步操作。
    实际计算延迟到访问时进行（训练循环不在 cudagraphs 范围内）。
    """
    __slots__ = ('_model', '_filled')

    def __init__(self, model):
        self._model = model
        self._filled = False

    def _ensure_filled(self):
        if not self._filled:
            self._model._fill_lazy_diagnostics(self)

    def __repr__(self):
        self._ensure_filled()
        return repr(self._filled)

    def __getitem__(self, key):
        self._ensure_filled()
        return self._filled[key]

    def get(self, key, default=None):
        self._ensure_filled()
        return self._filled.get(key, default)

    def __bool__(self):
        self._ensure_filled()
        return bool(self._filled)

    def __contains__(self, key):
        self._ensure_filled()
        return key in self._filled


@dataclass
class TrainingStats:
    """统一训练统计信息 (替代 aux_info 字典)

    设计原则: 单一接口，配置驱动，无废弃参数
    """
    # === 必需字段 ===
    logits: torch.Tensor              # [B, num_classes]
    # I139: num_tokens 现在支持 int (单样本) 或 List[int] (多样本批次)
    # I141: 添加 torch.Tensor 支持，避免 forward 中的 .cpu() 调用导致 cudagraphs 失败
    num_tokens: Union[int, List[int], torch.Tensor]  # Token 数量
    # I99-1 OPT: depth_used 支持 Tensor 类型以避免 CPU 同步
    depth_used: Union[int, torch.Tensor]  # 使用的深度
    depth_distribution: Dict[int, float]  # 深度分布
    features: torch.Tensor            # [B, dim] 池化特征
    transformer_tokens: torch.Tensor  # [B, N, dim] Transformer token

    # === 可选字段 ===
    shared_features: Optional[torch.Tensor] = None  # [B, d_model, H/p, W/p]
    splitter_entropy: float = 0.0
    temperature: float = 1.0

    # I110-7: 语义分裂器损失
    semantic_loss: Optional[torch.Tensor] = None  # 语义冗余损失
    child_features: Optional[torch.Tensor] = None  # 预测的子节点特征 [B, N, 4, D]
    redundancy: Optional[torch.Tensor] = None  # 冗余性分数 [B, N]

    # === 向后兼容字段 (I112) ===
    aux_infos: Optional[List[Dict[str, Any]]] = None  # 评估层期望的 aux_infos 格式
    ema_stats: Optional[torch.Tensor] = None           # I99-1: EMA buffer 统计信息
    split_info: Dict[str, Any] = field(default_factory=dict)  # 分割决策详情

    def validate(self) -> None:
        """数学约束验证"""
        # I139: 支持列表和张量类型的 num_tokens
        if isinstance(self.num_tokens, (list, tuple)):
            for i, n in enumerate(self.num_tokens):
                assert 0 <= n <= 4096, f"Batch[{i}] Token 数异常: {n}"
        elif isinstance(self.num_tokens, torch.Tensor):
            # P-OPT: 使用张量比较避免 .item()/.tolist() 同步
            # 仅在异常时提取违规值，使用 .detach() 避免梯度追踪
            tokens_valid = (self.num_tokens >= 0) & (self.num_tokens <= 4096)
            assert tokens_valid.all(), (
                f"Token 数异常: "
                f"min={float(self.num_tokens.min()):.0f}, "
                f"max={float(self.num_tokens.max()):.0f}, "
                f"out_of_range={(~tokens_valid).sum()} 个"
            )
        else:
            assert 0 <= self.num_tokens <= 4096, f"Token 数异常: {self.num_tokens}"
        # P4-FIX: 支持 depth_used 为 Tensor 类型
        if isinstance(self.depth_used, torch.Tensor):
            depth_valid = (self.depth_used >= 0) & (self.depth_used <= 50)
            assert depth_valid.all(), f"深度越界: {self.depth_used}"
        else:
            assert 0 <= self.depth_used <= 50, f"深度越界: {self.depth_used}"
        if self.depth_distribution:
            total = sum(self.depth_distribution.values())
            assert abs(total - 1.0) < 1e-5, f"分布未归一化: {total}"


class FractalCurveViT(nn.Module):
    """分形曲线视觉 Transformer。
    
    核心特性：
    - 可学习的分割决策网络（6特征输入）
    - 真正的 Hilbert 曲线递归算法（支持多方向）
    - 动态层级管理（最大 50 层）
    - 智能 patch 处理和特征增强
    - 边缘检测和纹理复杂度分析
    - 自适应多尺度处理
    
    P0 修复: max_level 参数统一为标准数学术语（分形四叉树递归深度），
    与 FractalConfig, LevelsInfo, StreamingFractalTokenizerV3 保持一致。
    这确保所有 Embedding 表大小与实际使用的深度范围匹配，减少约 90% 的参数浪费。

    Attributes:
        image_size: 输入图像尺寸
        num_classes: 分类类别数
        dim: 模型维度
        pool: 池化策略 ('cls', 'mean' 或其他)
        max_level: 最大递归深度 (统一使用 max_level)
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
        position_embedding: Optional[BitFlippedPositionEncoder] = None,
        mlp_head: Optional[nn.Sequential] = None,
        cls_token: Optional[nn.Parameter] = None,
        # I98-2: 内部状态标记（由工厂函数设置）
        dynamic_image_size: bool = False,
        # 配置参数
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        num_classes: int = 1000,
        dim: int = 512,
        num_layers: int = 6,  # Transformer 层数
        heads: int = 8,
        mlp_dim: int = 1024,
        pool: str = "weighted",
        channels: int = 3,
        dim_head: int = 64,
        # I120-2: 分离 dropout 配置
        # tokenizer_dropout: Tokenizer/Splitter dropout，必须为 0.0 (确定性)
        # transformer_dropout: Transformer dropout，默认为 0.1 (正则化)
        # I130-2: Hilbert 最佳实现 - 禁用 Dropout 和 DropPath
        # 理由: Dropout/DropPath 破坏 Hilbert 曲线的确定性保证
        # 预期效果: train/eval max_diff: 3.64 → <0.05
        tokenizer_dropout: float = 0.0,
        transformer_dropout: float = 0.0,
        emb_dropout: float = 0.0,
        min_patch_size: Union[int, Tuple[int, int]] = 4,
        # 注意: max_level 是变参数，完全由模型架构内部根据 image_size 和 min_patch_size 动态计算
        # 不再作为外部参数传入，确保训练/评估模型结构完全一致
        use_hilbert_encoding: bool = True,
        use_spatial_encoding: bool = True,
        use_checkpoint: bool = False,
        drop_path_rate: float = 0.0,
        ffn_type: FFNType = 'swiglu_level',
        # I122-2: 移除 lca_temperature，由 hilbert_bias_scale 统一缩放
        # I33: 覆盖率参数（用于在不同分辨率下正确复算 K 值）
        # K_min/K_max 现在作为计算属性，不再是直接参数
        # I113-2: token_coverage_max 已废弃，使用 target_ratio 替代
        token_coverage_min: float = 0.01,
        # I113-2: token_coverage_max 已废弃，保留仅用于向后兼容
        token_coverage_max: Optional[float] = None,
        # I113-2: target_ratio - L1 相对参数，由外部配置传递
        target_ratio: float = 0.5,
        pos_dropout: Optional[float] = None,
        use_area_encoding: bool = False,
        use_affine_modulation: bool = True,
        fourier_levels: int = 4,
        encoder_config: Optional[AttentionEncoderConfig] = None,
        quota_learnable: Optional[bool] = None,
        quota_entropy_weight: float = 0.5,  # I165-1: 增加熵权重以驱动深度分布变化 (原0.01)
        lca_fp16: bool = False,  # I104-3: 使用 FP16 存储 LCA embedding
        # I140: Splitter 架构参数
        splitter_hidden_dim: Optional[int] = None,
        splitter_feature_dim: Optional[int] = None,
        splitter_pool_size: Optional[int] = None,
        # I145: Splitter 温度参数 (用于温度退火)
        splitter_temp_start: Optional[float] = None,
        splitter_temp_end: Optional[float] = None,
        # I130-3: Splitter 类型选择 (I145: 新增 semantic_redundancy 支持)
        splitter_type: str = 'gumbel_topk',  # 'gumbel_topk', 'deterministic_neighbor', 'semantic_redundancy'
        # I110-7: 语义分裂器配置
        use_semantic_splitter: bool = False,
        semantic_splitter_config: Optional[SemanticSplitterConfig] = None,
        # P6-1: 深度缩放参数 (传递给 HilbertPatchEmbed)
        depth_scale_range: Optional[Tuple[float, float]] = None,
        # I162-1: Hilbert 模式编码器参数
        use_pattern_encoder: bool = False,  # 是否启用模式编码器
        pattern_encoder_mode: str = "light",  # "light", "standard", "multihead"
        pattern_encoder_window_sizes: Optional[Tuple[int, ...]] = None,  # 多尺度窗口大小
        # I162-1: 双路径插件参数
        use_pattern_plugin: bool = False,  # 是否启用双路径插件 (替代串行模式)
        pattern_plugin_config: Optional[dict] = None,  # 插件配置字典

        # Scheme C: Structured Manifold Bias 参数
        use_geometry_field: bool = False,  # 是否启用几何流形场
        geometry_field_dim: Optional[int] = None,  # 几何流形场维度 (默认等于 dim)
        manifold_bias_scale: float = 1.0,  # 流形偏置缩放因子
    ) -> None:
        """初始化 FractalCurveViT。

        I98-2: 支持依赖注入模式。
        - 若提供 injected 组件（splitter, transformer 等），则使用注入的组件
        - 若未提供，则自动创建组件（向后兼容模式）

        Args:
            splitter: (I98-2) 注入的 Splitter 实例
            tokenizer: 自定义 tokenizer（可选，若提供则使用自定义 tokenizer）
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
            pool: 池化策略 ('weighted' 或 'mean')
            channels: 输入图像通道数
            dim_head: 每个注意力头的维度
            tokenizer_dropout: Tokenizer/Splitter dropout 比率，必须为 0.0 (确定性)
            transformer_dropout: Transformer dropout 比率，默认为 0.1 (正则化)
            emb_dropout: 嵌入层 Dropout 比率
            min_patch_size: 目标最小 patch 大小，用于动态计算 max_level
            max_level: 最大递归深度 (P0: 统一使用 max_level)
            use_hilbert_encoding: 是否使用 Hilbert 编码
            use_spatial_encoding: 是否使用空间编码
            ffn_type: FFN 变体 ('gelu', 'swiglu', 'swiglu_level')
            I122-2: 移除 lca_temperature，由 hilbert_bias_scale 统一缩放
            K_min: 最少 token 数
            K_max: 最多 token 数
            pos_dropout: 位置编码 dropout
            use_area_encoding: 启用面积增强位置编码
            use_affine_modulation: 启用 ShapeScaleEncoder 仿射调制
            fourier_levels: 傅里叶特征级别数
            encoder_config: 注意力编码器配置
            quota_learnable: 可学习配额控制
            depth_scale_range: P6-1 深度缩放范围 (σ_min, σ_max)，传递给 HilbertPatchEmbed

        Note:
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
        # P0 修复: max_level 将在 tokenizer 创建后从 tokenizer.max_level 获取
        self.use_checkpoint = use_checkpoint
        self.ffn_type = ffn_type

        # I33: 存储覆盖率参数（用于 K 值计算）
        # I113-2: token_coverage_max 已废弃，保留仅用于向后兼容
        self.token_coverage_min = token_coverage_min
        if token_coverage_max is not None:
            import warnings
            warnings.warn(
                "token_coverage_max 已废弃，将在未来版本中移除。 "
                "请使用 config.target_ratio 替代。",
                DeprecationWarning,
                stacklevel=2
            )
            self._deprecated_token_coverage_max = token_coverage_max
        else:
            self._deprecated_token_coverage_max = None

        # I24-2: 可学习配额参数
        self.quota_learnable = quota_learnable
        self.quota_entropy_weight = quota_entropy_weight

        # I140: Splitter 架构参数
        self.splitter_hidden_dim = splitter_hidden_dim
        self.splitter_feature_dim = splitter_feature_dim
        self.splitter_pool_size = splitter_pool_size

        # I122-2: 移除 lca_temperature，由 hilbert_bias_scale 统一缩放
        self.lca_fp16 = lca_fp16  # I104-3

        # P6-1: 深度缩放参数
        self.depth_scale_range = depth_scale_range

        # I162-1: Hilbert 模式编码器配置
        self.use_pattern_encoder = use_pattern_encoder
        self.pattern_encoder = None
        if use_pattern_encoder:
            window_sizes = pattern_encoder_window_sizes or (3, 7)
            if pattern_encoder_mode == "light":
                self.pattern_encoder = create_hilbert_pattern_encoder(
                    dim=dim,
                    mode=pattern_encoder_mode,
                    kernel_size=window_sizes[0] if window_sizes else 7,
                    out_dim=dim,
                )
            else:
                self.pattern_encoder = create_hilbert_pattern_encoder(
                    dim=dim,
                    mode=pattern_encoder_mode,
                    window_sizes=window_sizes,
                    out_dim=dim,
                )

        # I162-1: 双路径插件配置
        self.use_pattern_plugin = use_pattern_plugin
        self.pattern_plugin = None
        if use_pattern_plugin:
            self.pattern_plugin = create_hilbert_pattern_plugin(
                dim=dim,
                enabled=True,
                pattern_dim=dim,
                pattern_scale=1.0,
                pattern_encoder=self.pattern_encoder,  # 复用已有的模式编码器
            )

        # ====================================================================
        # I120-2: 子模块 Dropout 配置 (确定性 + 正则化分离)
        # ====================================================================
        # 数学依据:
        #   - Splitter MLP: dropout=0.0 (确定性，无随机性)
        #   - Position Embedding: dropout=0.0 (确定性)
        #   - Transformer: dropout=0.1 (正则化)
        #
        # 推导公式:
        #   pos_dropout = transformer_dropout * 0.5  # half of transformer dropout
        # ====================================================================
        effective_pos_dropout = pos_dropout if pos_dropout is not None else (transformer_dropout * 0.5)

        # I30-17: 处理 min_patch_size 的向后兼容
        # 支持旧 API: min_patch_size=(4, 4)
        if isinstance(min_patch_size, tuple):
            effective_min_patch_size = min_patch_size[0]
        else:
            effective_min_patch_size = min_patch_size

        # 存储几何配置和编码选项
        self.min_patch_size = effective_min_patch_size
        self.use_hilbert_encoding = use_hilbert_encoding
        self.use_spatial_encoding = use_spatial_encoding
        self.use_area_encoding = use_area_encoding
        self.use_affine_modulation = use_affine_modulation
        self.fourier_levels = fourier_levels
        # I120-2: 分离 dropout 配置
        self.tokenizer_dropout = tokenizer_dropout
        self.transformer_dropout = transformer_dropout
        self.emb_dropout = emb_dropout
        self.drop_path_rate = drop_path_rate

        # I113-2: target_ratio - L1 相对参数，由外部配置传递
        self.target_ratio = target_ratio

        # I78: 处理动态分辨率模式 (必须在使用 effective_min_patch_size 之后)
        if image_size is None:
            # 使用 min_patch_size 估算默认图像尺寸用于初始化
            # 估算公式: min(H, W) ≈ min_patch_size × 2^6 = min_patch_size × 64
            # max_level 将由 StreamingFractalTokenizerV3 根据实际输入图像自动计算
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
            # I98-1: 确定 max_level_limit (根据 tokenizer 或默认值)
            # I164-1: 使用 max_level_limit=8 (已通过分块处理优化)
            max_level_limit = 8
            if tokenizer is not None:
                if hasattr(tokenizer, 'max_level'):
                    max_level_limit = tokenizer.max_level

            # I130-3: 根据 splitter_type 创建不同的 Splitter (I145: 添加 semantic_redundancy)
            if splitter_type == 'deterministic_neighbor':
                # DeterministicNeighborSplitter: 100%梯度覆盖率，完整邻居传播
                from vit_pytorch.layers.splitters.deterministic_neighbor import (
                    DeterministicNeighborSplitter,
                    DeterministicNeighborSplitterConfig,
                )

                splitter_config = DeterministicNeighborSplitterConfig(
                    feature_dim=splitter_feature_dim or dim,
                    max_level_limit=max_level_limit,
                    hidden_dim=splitter_hidden_dim or 64,
                    # 覆盖率参数
                    coverage_min=token_coverage_min,
                    coverage_base=token_coverage_max if token_coverage_max else K_COVERAGE_MAX_HARD,
                    # 温度参数
                    temperature_init=splitter_temp_start if splitter_temp_start is not None else SPLITTER_TEMP_START,
                    temperature_min=splitter_temp_end if splitter_temp_end is not None else SPLITTER_TEMP_END,
                    # 配额参数
                    enable_learnable_quota=quota_learnable if quota_learnable is not None else True,
                    locality_weight=0.1,
                    entropy_weight=0.01,
                )
                self.splitter = DeterministicNeighborSplitter(
                    config=splitter_config,
                    image_size=self.image_size,
                    feature_dim=splitter_feature_dim or dim,
                )
            elif splitter_type == 'semantic_redundancy':
                # SemanticRedundancySplitter: 语义冗余性感知分裂
                from vit_pytorch.layers.splitters.semantic_redundancy import (
                    SemanticRedundancySplitter,
                )

                self.splitter = SemanticRedundancySplitter(
                    feature_dim=splitter_feature_dim or dim,
                    hidden_dim=splitter_hidden_dim or 128,
                    max_level_limit=max_level_limit,
                    gumbel_temp_start=splitter_temp_start if splitter_temp_start is not None else SPLITTER_TEMP_START,
                    gumbel_temp_end=splitter_temp_end if splitter_temp_end is not None else SPLITTER_TEMP_END,
                    learnable_temperature=True,
                )
            else:
                # 默认使用 GumbelTopKSplitter
                from vit_pytorch.layers.splitters.gumbel_topk import GumbelTopKSplitter
                from vit_pytorch.core.config import SplitterConfig

                splitter_config = SplitterConfig(
                    feature_dim=splitter_feature_dim or dim,
                    min_patch_size=effective_min_patch_size,
                    max_level_limit=max_level_limit,
                    hidden_dim=splitter_hidden_dim or 64,
                    intermediate_dim=(splitter_hidden_dim or 64) // 2,
                    pool_size=splitter_pool_size or 4,
                    # I113-2: K 边界由 config 内部根据 coverage_min/coverage_max_hard 自动计算
                    use_dynamic_k=True,
                    # I120-2: dropout 始终为 0.0 (Tokenizer 确定性)
                    dropout=0.0,
                    enable_learnable_quota=quota_learnable if quota_learnable is not None else True,
                    quota_entropy_weight=quota_entropy_weight,
                    # I33: 传递覆盖率参数
                    coverage_min=token_coverage_min,
                    coverage_max_hard=token_coverage_max if token_coverage_max else K_COVERAGE_MAX_HARD,
                    # I120-3: 选中率均衡配额 (解决深度分布单一化)
                    # I165-1: 禁用 rate_balanced 以启用可学习的 ContinuousQuotaAllocator
                    enable_rate_balanced_quota=False,
                    # I165-1: 启用分层自适应配额 (根据图像内容动态调整深度分布)
                    enable_hierarchical_quota=True,
                    # I145: 传递温度参数
                    temperature_init=splitter_temp_start if splitter_temp_start is not None else SPLITTER_TEMP_START,
                    temperature_min=splitter_temp_end if splitter_temp_end is not None else SPLITTER_TEMP_END,
                    # v6.1: Soft-Threshold 课程学习配置
                    enable_soft_threshold=True,
                    soft_threshold_max=0.5,
                    soft_threshold_schedule='linear',
                )
                self.splitter = GumbelTopKSplitter(
                    config=splitter_config,
                    image_size=self.image_size,
                )

        # P-OPT: 缓存是否为 SemanticRedundancySplitter，避免每次 forward 都做 isinstance 检查
        from vit_pytorch.layers.splitters.semantic_redundancy import SemanticRedundancySplitter
        self._is_semantic_splitter = isinstance(self.splitter, SemanticRedundancySplitter)

        # === HilbertTopologyCache (Step 3 优化) ===
        # O(1) Tensor Lookup for Hilbert 坐标转换
        # 必须在 tokenizer 创建之前创建
        from vit_pytorch.core.hilbert_topology_cache import HilbertTopologyCache
        self.hilbert_cache = HilbertTopologyCache()

        # === Tokenizer ===
        if tokenizer is None:
            # 创建 StreamingFractalTokenizerV3
            # max_level 完全由 tokenizer 内部根据 image_size 和 min_patch_size 动态计算
            tokenizer = StreamingFractalTokenizerV3(
                image_size=self.image_size,
                channels=channels,
                d_model=dim,
                base_patch_size=effective_min_patch_size,
                min_patch_size=effective_min_patch_size,
                depth_scale_range=self.depth_scale_range,
                hilbert_cache=self.hilbert_cache,  # Step 3: O(1) Hilbert lookup
            )

        # I110-7: 配置语义分裂器（必须在 tokenizer 赋值之前）
        self._use_semantic_splitter = use_semantic_splitter
        self._semantic_splitter_config = semantic_splitter_config
        self._semantic_splitter: Optional[nn.Module] = None
        self._semantic_loss_fn: Optional[nn.Module] = None

        if use_semantic_splitter and semantic_splitter_config is not None:
            # 配置 tokenizer 使用语义分裂器
            tokenizer.use_semantic_splitter(config=semantic_splitter_config)
            # 创建语义分裂器实例
            self._semantic_splitter = tokenizer.get_semantic_splitter()
            self._semantic_loss_fn = tokenizer.get_semantic_loss_fn()

        self.tokenizer = tokenizer

        # I98-1: 设置 tokenizer 对 model 的弱引用，避免循环引用导致递归遍历失败
        tokenizer._model = weakref.ref(self)

        # Step 3: 确保 tokenizer 有 hilbert_cache（如果外部传入 tokenizer）
        if not hasattr(tokenizer, '_hilbert_cache') or tokenizer._hilbert_cache is None:
            tokenizer._hilbert_cache = self.hilbert_cache

        # 从 tokenizer 动态获取 max_level（变参数）
        if hasattr(tokenizer, 'max_level'):
            computed_max_level = tokenizer.max_level
        else:
            computed_max_level = 8  # 默认值
        self.max_level = computed_max_level

        self.token_processor = None

        # === Position Embedding (v6.0: 使用 BitFlippedPositionEncoder) ===
        if position_embedding is None:
            position_embedding = BitFlippedPositionEncoder(
                dim=dim,
                max_level=self.max_level,
                grid_size=256,
            )

        self.pos_embedding = position_embedding
        self.position_embedding = position_embedding

        # === Scheme C: GeometryField ===
        self.use_geometry_field = use_geometry_field
        self.manifold_bias_scale = manifold_bias_scale
        self.geometry_field_dim = geometry_field_dim or dim
        self.geometry_field = None

        if use_geometry_field:
            from vit_pytorch.layers.embeddings.fractal_position import GeometryField
            self.geometry_field = GeometryField(
                dim=self.geometry_field_dim,
                max_level=self.max_level,
                heads=heads,
            )

        # === CLS Token ===
        if cls_token is not None:
            self.cls_token = cls_token
        else:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # I120-2: emb_dropout 始终为 0.0 (确定性位置编码)
        self.emb_dropout_module = nn.Dropout(emb_dropout)

        # I30-11: 已删除 Mixed Pooling
        self.register_buffer("aux_loss_weight", torch.tensor(0.0))
        self.register_buffer("_zero_loss", torch.tensor(0.0))

        # === Transformer ===
        if transformer is not None:
            # I98-2: 使用注入的 Transformer
            self.transformer = transformer
        else:
            # 动态创建 Transformer（向后兼容）
            # 使用 self.max_level（从 tokenizer 获取的变参数）
            # I120-2: 使用 transformer_dropout 而非 dropout
            self.transformer = FractalTransformer(
                dim=dim,
                depth=num_layers,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                dropout=transformer_dropout,
                max_level=self.max_level,
                drop_path_rate=drop_path_rate,
                ffn_type=ffn_type,
                use_checkpoint=use_checkpoint,
                use_affine_modulation=use_affine_modulation,
                fourier_levels=fourier_levels,
                encoder_config=encoder_config,
                use_fp16=lca_fp16,  # I104-3
            )

        # === 一致性检查：确保 tokenizer 和 transformer 使用相同的 max_level ===
        # I145: 防止配置不一致导致的权重不匹配问题
        tokenizer_max_level = getattr(self.tokenizer, 'max_level', None)
        transformer_max_level = getattr(self.transformer, 'max_level', None)

        if tokenizer_max_level is not None and transformer_max_level is not None:
            if tokenizer_max_level != transformer_max_level:
                raise ValueError(
                    f"[MODEL] max_level 不一致: tokenizer={tokenizer_max_level}, "
                    f"transformer={transformer_max_level}。\n"
                    f"  这通常是由于外部传入的组件配置不正确导致的。\n"
                    f"  请确保 tokenizer 和 transformer 使用相同的 max_level。"
                )

        # === MLP Head ===
        # I162-1: 双路径插件模式下，输入维度翻倍
        mlp_input_dim = dim * 2 if use_pattern_plugin else dim
        if mlp_head is not None:
            self.mlp_head = mlp_head
        else:
            # 动态创建 MLP Head
            # I120-2: 使用 transformer_dropout 而非 dropout
            # I147: 添加 LogitsClamp 解决训练损失异常 (~82)
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(mlp_input_dim),
                nn.Linear(mlp_input_dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(transformer_dropout),
                nn.Linear(mlp_dim, num_classes),
                LogitsClamp(LOGIT_CLAMP_BOUND),  # I147: 钳制 logits 防止损失爆炸
            )
            self.num_classes = num_classes
            self.dim = dim
            self.num_layers = num_layers
            self.heads = heads
            self.mlp_dim = mlp_dim

        # 权重初始化 - 关键改进，防止类别偏差
        self._init_weights()

    # =========================================================================
    # TIER 2: 变参数计算属性 (Variables)
    # =========================================================================
    # 这些属性根据模型参数动态计算，确保 K 值在不同分辨率下正确复算。

    @property
    def K_min(self) -> int:
        """最少 token 数量（Tier 2: 变参数）。

        数学形式化:
            K_min = max(K_MIN_HARD, ceil(N_candidates × coverage_min))
            N_candidates = Σ(4^d), d=0..max_level

        计算时机:
            - 访问时动态计算
            - 基于存储的 coverage 参数和 max_level

        Returns:
            最少 token 数量
        """
        max_level = self.max_level if self.max_level is not None else 8
        # I113-2: 使用 _deprecated_token_coverage_max 兼容旧代码
        K_min, _ = compute_k_bounds(
            max_level=max_level,
            token_coverage_min=self.token_coverage_min,
            token_coverage_max=getattr(self, '_deprecated_token_coverage_max', None),
            image_size=min(self.image_size) if self.image_size else None,
            target_ratio=self.target_ratio,  # I113-2: 使用外部传递的 target_ratio
        )
        return K_min

    @property
    def K_max(self) -> int:
        """最多 token 数量（Tier 2: 变参数）。

        数学形式化:
            K_max = min(K_MAX_HARD, ceil(N_candidates × coverage_max × scale))
            N_candidates = Σ(4^d), d=0..max_level
            scale = sqrt(min(H, W) / 224)

        计算时机:
            - 访问时动态计算
            - 基于存储的 coverage 参数、max_level 和 image_size

        Returns:
            最多 token 数量
        """
        max_level = self.max_level if self.max_level is not None else 8
        # I113-2: 使用 _deprecated_token_coverage_max 兼容旧代码
        _, K_max = compute_k_bounds(
            max_level=max_level,
            token_coverage_min=self.token_coverage_min,
            token_coverage_max=getattr(self, '_deprecated_token_coverage_max', None),
            image_size=min(self.image_size) if self.image_size else None,
            target_ratio=self.target_ratio,  # I113-2: 使用外部传递的 target_ratio
        )
        return K_max

    @property
    def num_candidates(self) -> int:
        """候选节点总数（Tier 2: 变参数）。

        数学形式化:
            N_candidates = Σ(4^d), d=0..max_level = (4^(max_level+1) - 1) / 3

        Returns:
            四叉树候选节点总数
        """
        max_level = self.max_level if self.max_level is not None else 8
        return compute_num_candidates(max_level)

    def get_model_config(self) -> Dict[str, Any]:
        """获取模型参数字典（Tier 3: 参数快照）。

        用于 checkpoint 序列化，确保模型可复现。

        Note:
            max_level 是变参数，由模型架构根据 image_size 和 min_patch_size 动态计算，
            不保存到 config 中。重建模型时会自动计算。

        Returns:
            包含所有 Tier 3 参数的字典
        """
        # I145: 安全获取 quota_learnable（可能在某些配置中不存在）
        quota_learnable = getattr(self, 'quota_learnable', None)

        return {
            # 核心架构
            'num_classes': self.num_classes,
            'dim': self.dim,
            'num_layers': self.num_layers,
            'heads': self.heads,
            'mlp_dim': self.mlp_dim,
            # 几何配置
            'image_size': self.image_size,
            'min_patch_size': self.min_patch_size,
            # Note: max_level 是变参数，由模型架构动态计算，不保存
            # 覆盖率预算 (I33)
            'token_coverage_min': self.token_coverage_min,
            'token_coverage_max': getattr(self, '_deprecated_token_coverage_max', None),
            # 编码选项
            # I122-2: 移除 lca_temperature，由 hilbert_bias_scale 统一缩放
            'use_hilbert_encoding': self.use_hilbert_encoding,
            'use_spatial_encoding': self.use_spatial_encoding,
            'use_area_encoding': self.use_area_encoding,
            'use_affine_modulation': self.use_affine_modulation,
            'fourier_levels': self.fourier_levels,
            # I120-2: 分离 dropout 配置
            'tokenizer_dropout': self.tokenizer_dropout,
            'transformer_dropout': self.transformer_dropout,
            'emb_dropout': self.emb_dropout,
            'drop_path_rate': self.drop_path_rate,
            # FFN
            'ffn_type': self.ffn_type,
            # I24-2: 可学习配额
            'quota_learnable': quota_learnable,
            # Scheme C: 几何流形场参数
            'use_geometry_field': getattr(self, 'use_geometry_field', False),
            'geometry_field_dim': getattr(self, 'geometry_field_dim', None),
            'manifold_bias_scale': getattr(self, 'manifold_bias_scale', 1.0),
        }

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], "TokenizerOutput", torch.Tensor, Optional[Any]]:
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

        I107-7: 返回 features 避免重复计算
            原始流程: feature_extractor → splitter → tokenizer (再次计算 shared_conv)
            优化后: feature_extractor → splitter → tokenizer (复用 features)

        I170: 返回 split_result 用于提取语义分裂器的 redundancy 和 child_features

        Args:
            img: 输入图像 [B, C, H, W]

        Returns:
            (padded_tokens, padded_levels, lengths, levels_list, token_output, features, split_result):
            - padded_tokens: tokens [B, MaxN, Dim] (padded)
            - padded_levels: 层级信息 [B, MaxN, InfoDim]
            - lengths: Tensor[B] 每个样本的实际 token 数量
            - levels_list: 原始层级列表（用于辅助输出）
            - token_output: TokenizerOutput (P11-3: 用于获取 regions)
            - features: 共享特征图 [B, d_model, H/p, W/p] (I107-7: 避免重复计算)
            - split_result: Splitter 返回的分裂结果（可选，用于提取语义信息）
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

        split_result = None  # I170: 初始化 split_result

        if needs_split_result:
            # I130-2: Hilbert 最佳实现 - DeterministicTopK 模式始终使用硬选择
            # 关键修复: 对于确定性模式，hard 参数不影响选择逻辑
            # DeterminativeTopK 模式下，硬掩码和软掩码基于相同的确定性概率
            use_hard = True  # 始终使用硬选择以确保确定性

            # I130-3: SemanticRedundancySplitter 需要 3D 输入 [B, N, D]
            # P-OPT: 使用缓存的 _is_semantic_splitter 避免 isinstance 检查
            if self._is_semantic_splitter:
                # features: [B, d_model, H_feat, W_feat] -> [B, N, d_model]
                B, C, H_feat, W_feat = features.shape
                features = features.view(B, C, H_feat * W_feat).transpose(1, 2)  # [B, N, C]

            split_result = self.splitter(
                features,
                image_size=(img.shape[2], img.shape[3]),
                hard=use_hard,
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

        # I107-7: 返回 features 避免训练循环中重复计算 shared_conv
        # I170: 返回 split_result 用于提取语义分裂器的 redundancy 和 child_features
        return padded_tokens, padded_levels, lengths, levels_list, token_output, features, split_result

    def _apply_position_and_cls(
        self,
        padded_tokens: torch.Tensor,
        padded_levels: torch.Tensor,
        regions: Optional[torch.Tensor] = None,
        image_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, "LevelsInfo", torch.Tensor]:
        """添加位置编码和 CLS token。

        I98-4: 返回 LevelsInfo 而非 raw tensor
        v6.0: 返回 geometry_emb_with_cls 用于 Attention 注入

        Args:
            padded_tokens: 填充后的 tokens [B, MaxLen, Dim]
            padded_levels: 填充后的层级信息 [B, MaxLen, max_level+1]
            regions: (I31-3) 区域边界张量，形状 [B, N, 4]
            image_size: (I31-3) 图像尺寸，可以是整数或 (W, H) 元组

        Returns:
            (x, levels_info, geometry_emb_with_cls):
            - x: 带位置编码和 CLS 的序列 [B, 1+MaxLen, Dim]
            - levels_info: LevelsInfo 实例（包含 CLS）
            - geometry_emb_with_cls: 带 CLS 的几何嵌入 [B, 1+MaxLen, Dim]
        """
        batch_size = padded_tokens.shape[0]
        device = padded_tokens.device

        # I98-4: 将 raw tensor 转换为 LevelsInfo
        from vit_pytorch.core.levels_info import LevelsInfo
        levels_info = LevelsInfo(data=padded_levels, max_level=self.max_level)

        # v6.0: BitFlippedPositionEncoder 返回 pos_emb 和 geometry_emb
        pos_emb, geometry_emb = self.pos_embedding(levels_info)
        x = padded_tokens + pos_emb

        # v6.0: 为 CLS 添加零几何嵌入
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cls_geometry = torch.zeros(batch_size, 1, self.dim, device=x.device)
        geometry_emb_with_cls = torch.cat([cls_geometry, geometry_emb], dim=1)  # [B, N+1, D]

        x = torch.cat((cls_tokens, x), dim=1)

        # I98-4: 构造包含 CLS 的 LevelsInfo
        cls_level = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        cls_depths = levels_info.depths  # (B, MaxLen)
        all_depths = torch.cat([cls_level, cls_depths], dim=1)  # (B, 1+MaxLen)
        # all_paths shape: (B, 1+MaxLen, max_level)
        all_paths = torch.zeros(batch_size, all_depths.shape[1], self.max_level, dtype=torch.long, device=device)
        all_paths[:, 1:, :] = levels_info.paths  # (B, 1+MaxLen, max_level)

        levels_info_with_cls = LevelsInfo.from_arrays(all_depths, all_paths, max_level=self.max_level)

        x = self.emb_dropout_module(x)

        # v6.0: 返回 geometry_emb_with_cls
        return x, levels_info_with_cls, geometry_emb_with_cls

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

        数学形式:
        - mean: z = (1/N) * Σ_i x_i
        - weighted: z = Σ_i (w_i / Σ_j w_j) * x_i, 其中 w_i = split_prob_i

        Args:
            x: transformer 输出 [B, Seq, Dim]
            key_padding_mask: padding mask [B, Seq]
            split_probs: [B, N] 分割概率，用于加权池化

        Returns:
            pooled: 池化后的表示 [B, Dim]
        """
        token_x = x[:, 1:]  # [B, N, D] - 排除 CLS
        token_mask = ~key_padding_mask[:, 1:]  # [B, N] - 排除 CLS

        if self.pool == "weighted" and split_probs is not None:
            # I30-11: 加权池化，利用 split_probs 作为 token 重要性权重
            token_probs = split_probs * token_mask.float()
            weight_sum = token_probs.sum(dim=-1, keepdim=True).clamp(min=PROB_EPSILON)
            normalized_weights = token_probs / weight_sum
            return (token_x * normalized_weights.unsqueeze(-1)).sum(dim=1)
        else:
            # mean pooling 作为默认
            masked_x = token_x * token_mask.unsqueeze(-1).float()
            valid_counts = token_mask.sum(dim=1, keepdim=True).float().clamp(min=DIVISION_EPSILON)
            return masked_x.sum(dim=1) / valid_counts

    @torch._dynamo.disable(recursive=False)
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
            # P-OPT: 完全在 GPU 上计算，避免 CPU 同步
            # 根因: .to('cpu') 会导致 torch.compile 的 cudagraphs 失败
            max_level = self.tokenizer.max_level if hasattr(self, 'tokenizer') else 8
            max_level_range = max_level + 1
            B = len(levels_list)

            # P-OPT: 向量化深度分布计算 - 完全 GPU 计算
            if levels_list and lengths.numel() == B:
                # 获取最大 token 数量 (安全处理空列表)
                max_tokens = 0
                for l in levels_list:
                    if l.numel() > 0:
                        max_tokens = max(max_tokens, l.size(0))
                max_tokens = min(max_tokens, 256)

                # 空列表保护
                if max_tokens == 0:
                    # I144: 批量转换避免循环中的 .item()
                    # I141: 添加 non_blocking=True 避免同步阻塞
                    lengths_cpu = lengths.cpu(non_blocking=True) if lengths.is_cuda else lengths
                    lengths_list = lengths_cpu.tolist()
                    # I145: 使用模块级 LazyDiagnostics，避免 forward 中的 CPU 同步
                    lazy_diag = LazyDiagnostics(self)
                    for i in range(B):
                        aux_infos.append({
                            "num_tokens": lengths_list[i],
                            "levels_used": [],
                            "depth_distribution": {},
                            "splitter_diagnostics": lazy_diag,
                        })
                    return aux_infos, None

                # 填充深度矩阵 [B, max_tokens] - 全部在 GPU
                padded_depths = torch.full((B, max_tokens), -1,
                                           dtype=torch.long, device=lengths.device)
                valid_counts = []

                # P-OPT: levels_list 填充循环
                # 注意: levels_list 是 Python list of tensors (各元素形状不同)
                # 无法完全向量化，但循环体已使用张量操作，O(B) 开销可忽略
                # B 通常 8-32，此循环开销 < 0.1ms
                for i, l in enumerate(levels_list):
                    if l.numel() > 0:
                        depths = l[:, 0].long()
                        depths = depths[depths >= 0]
                        n = min(depths.size(0), max_tokens)
                        padded_depths[i, :n] = depths[:n]
                        valid_counts.append(n)
                    else:
                        valid_counts.append(0)  # 确保 valid_counts 长度始终等于 B

                # I145: 向量化 depth 计数 - 使用 scatter_add 替代 Python 循环
                # 原始实现:
                # for d in range(max_level_range):
                #     mask = (padded_depths == d)
                #     all_depth_counts[:, d] = mask.sum(dim=1, dtype=torch.float32)
                # 向量化实现:
                depths_for_count = padded_depths.clamp(min=0)  # [B, max_tokens], padding (-1) -> 0
                # I99-1 FIX: torch.compile 保护 - clamp 到有效深度范围
                depths_for_count = depths_for_count.clamp(max=max_level_range - 1)
                # 有效位置掩码 (排除 padding)
                valid_pos_mask = padded_depths >= 0
                # 使用 scatter_add: counts[batch, depth] = sum over valid positions with that depth
                all_depth_counts = torch.zeros(B, max_level_range, dtype=torch.float32, device=lengths.device)
                all_depth_counts.scatter_add_(dim=1, index=depths_for_count, src=valid_pos_mask.float())

                # 归一化分布
                depth_sums = all_depth_counts.sum(dim=1, keepdim=True).clamp(min=EPS)
                normalized_counts = all_depth_counts / depth_sums

                # I144: 向量化计算所有 num_tokens - 完全 GPU 计算，避免循环中的 .item()
                # lengths 是 GPU tensor，直接在 GPU 上操作
                lengths_cpu = lengths.cpu(non_blocking=True) if lengths.is_cuda else lengths  # 只在需要时同步一次
                lengths_list = lengths_cpu.tolist()  # 单次批量转换

                # I144: 向量化计算 entropy - 完全在 GPU 上计算，避免 Python 循环
                # 原始实现 (Python循环):
                # for i in range(B):
                #     probs_i = split_probs[i, :lengths[i]]
                #     probs_safe = probs_i + (probs_i == 0).float() * PROB_EPSILON
                #     entropy_i = -(probs_safe * torch.log(probs_safe)).sum()
                #     entropies_gpu.append(entropy_i)
                # 向量化实现:
                if split_probs is not None:
                    # 创建有效位置掩码 [B, max_tokens]
                    max_len = split_probs.size(1)
                    valid_mask = torch.arange(max_len, device=split_probs.device).unsqueeze(0) < lengths.unsqueeze(1)
                    # 掩码概率，填充为 1.0 (log(1)=0，不影响求和)
                    probs_masked = torch.where(valid_mask, split_probs, torch.ones_like(split_probs))
                    probs_safe = probs_masked + (probs_masked == 0).float() * PROB_EPSILON
                    # 计算每个样本的熵 [B]
                    entropies_gpu = -(probs_safe * torch.log(probs_safe)).sum(dim=1)
                    # 最后一次性转换为 Python float
                    entropies_list = entropies_gpu.tolist()

                # P-OPT: 向量化构建 levels_used_list - 避免 Python 循环
                # 原始: for i in range(B): nonzero.tolist()
                # 向量化: 批量处理所有 batch
                depth_counts_sliced = all_depth_counts[:, :max_level + 1]  # [B, max_level+1]
                has_tokens_mask = depth_counts_sliced.sum(dim=1) > 0  # [B]
                # nonzero 返回每个 batch 中非零元素的索引
                levels_used_list = []
                for i in range(B):
                    if has_tokens_mask[i]:
                        nonzero = depth_counts_sliced[i].nonzero(as_tuple=True)[0]
                        levels_used_list.append(nonzero.tolist())
                    else:
                        levels_used_list.append([])

                # I145: 使用模块级 LazyDiagnostics，延迟到首次访问时计算
                lazy_diag = LazyDiagnostics(self)

                for i in range(B):
                    num_tokens = lengths_list[i]  # 使用预转换的 Python list
                    levels_used = levels_used_list[i]

                    # P1 Fix: 使用已定义的 has_tokens_mask 替代未定义的 valid_bool
                    if not has_tokens_mask[i] or not levels_used:
                        aux_info = {
                            "num_tokens": num_tokens,
                            "levels_used": [],
                            "depth_distribution": {},
                            "splitter_diagnostics": lazy_diag,
                        }
                        if split_probs is not None:
                            aux_info["token_selection_entropy"] = 0.0
                        aux_infos.append(aux_info)
                        continue

                    d_max = min(levels_used[-1], max_level)
                    depth_distribution = {
                        d: float(normalized_counts[i, d])
                        for d in levels_used if d <= d_max
                    }

                    aux_info = {
                        "num_tokens": num_tokens,
                        "levels_used": levels_used,
                        "depth_distribution": depth_distribution,
                        "splitter_diagnostics": lazy_diag,
                    }

                    if split_probs is not None:
                        aux_info["token_selection_entropy"] = entropies_list[i]

                    aux_infos.append(aux_info)

        if return_features:
            # 直接返回 pooled 张量 [B, D] 而非 List[Tensor]
            # P1 Fix: 避免冗余的 list comprehension + stack 操作
            features_tensor = pooled

        return aux_infos, features_tensor if return_features else None

    @overload
    def forward(self, img: torch.Tensor) -> TrainingStats: ...

    def forward(
        self,
        img: torch.Tensor,
    ) -> Union[torch.Tensor, TrainingStats]:
        """前向传播。

        设计原则: 单一接口，无废弃参数

        Args:
            img: 输入图像，形状为 [B, C, H, W]

        Returns:
            TrainingStats: 包含 logits, features, num_tokens, depth_distribution 等
        """
        # P0-2: 自动转换 channels_last 内存格式以优化卷积性能
        if (img.dim() == 4 and
            not img.is_contiguous(memory_format=torch.channels_last) and
            hasattr(self, '_channels_last_enabled') and
            self._channels_last_enabled):
            img = img.to(memory_format=torch.channels_last)

        batch_size = img.shape[0]
        device = img.device

        # 1. 准备 tokens
        # I107-7: _prepare_tokens 现在返回 features (避免重复计算)
        # I170: _prepare_tokens 返回 split_result 用于提取语义分裂器信息
        padded_tokens, padded_levels, lengths, levels_list, token_output, features, split_result = self._prepare_tokens(img)

        # I170: 提取语义分裂器的 redundancy 和 child_features
        redundancy = None
        child_features = None
        if self._is_semantic_splitter and split_result is not None:
            redundancy = split_result.redundancy  # [B, N]
            child_features = split_result.child_features  # [B, N, 4, D]

        # P11-3: 获取 regions 和 image_size 用于正确的 LCA 偏置计算
        regions, image_size = token_output.get_padded_regions()

        # I162-1: Hilbert 模式编码 - 在添加 CLS 之前获取 Hilbert 索引
        hilbert_order = None
        if self.pattern_encoder is not None or self.pattern_plugin is not None:
            from vit_pytorch.core.levels_info import LevelsInfo
            temp_levels_info = LevelsInfo(data=padded_levels, max_level=self.max_level)
            hilbert_indices = temp_levels_info.get_hilbert_indices()  # [B, MaxLen]
            # 将 Hilbert 索引转换为排序位置 (使用第一个样本的排序，对所有batch通用)
            hilbert_order = torch.argsort(hilbert_indices[0], dim=0)  # [MaxLen]
            # 裁剪到实际 token 数量 (P-OPT: avoid .item() sync)
            actual_num_tokens = lengths[0] if lengths.dim() > 0 else lengths
            actual_num_tokens = actual_num_tokens.clamp(max=hilbert_order.shape[0])
            hilbert_order = hilbert_order[:actual_num_tokens]

        # 2. 添加位置编码和 CLS token (v6.0: 同时获取 geometry_emb)
        x, levels_info, geometry_emb_with_cls = self._apply_position_and_cls(
            padded_tokens, padded_levels, regions=regions, image_size=image_size
        )

        # Scheme C: 如果启用 GeometryField，计算流形场偏置并与现有 geometry_emb 融合
        # 注意: levels_info 已经包含 CLS，所以 manifold_emb 形状已经是 [B, N+1, dim]
        if self.use_geometry_field and self.geometry_field is not None:
            # 计算几何流形场编码 (levels_info 已包含 CLS)
            manifold_emb = self.geometry_field(levels_info)  # [B, N+1, dim]

            # 缩放并融合到现有的 geometry_emb
            # geometry_emb_with_cls 形状: [B, N+1, dim]
            geometry_emb_with_cls = geometry_emb_with_cls + self.manifold_bias_scale * manifold_emb

        # P11-3: 为 regions 添加 CLS 对应的零填充
        if regions is not None:
            cls_region = torch.zeros(batch_size, 1, 4, dtype=regions.dtype, device=device)
            regions = torch.cat([cls_region, regions], dim=1)

        # 3. 创建 attention mask
        attn_mask, key_padding_mask = self._create_attention_mask(
            batch_size, x.shape[1], lengths, device
        )

        # ============================================================
        # I162-1: 双路径插件模式 vs 串行模式
        # ============================================================
        _plugin_mode = self.pattern_plugin is not None

        if _plugin_mode:
            # === 双路径插件模式 (真正的并行处理) ===
            # 使用插件进行双路径处理
            fused_tokens, pooled = self.pattern_plugin(
                direct_tokens=x,  # [B, N+1, D] 含 CLS
                levels_info=levels_info,
                geometry_emb=geometry_emb_with_cls,
                transformer=self.transformer,
                hilbert_order=hilbert_order,
            )
            transformer_tokens = fused_tokens  # [B, N, 2D]
            final_output = self.mlp_head(pooled)  # [B, num_classes]
            pooled_for_stats = pooled  # 保存用于 stats

        elif self.pattern_encoder is not None and hilbert_order is not None:
            # === 串行模式 (原有逻辑，残差连接) ===
            # 使用与实际 token 数量匹配的 Hilbert 排序索引
            # x 形状: [B, 1+MaxLen, D], 跳过 CLS 后: [B, MaxLen, D]
            actual_num_tokens = x.shape[1] - 1  # 减去 CLS
            max_len = min(hilbert_order.shape[0], actual_num_tokens)
            hilbert_order_trimmed = hilbert_order[:max_len]
            x_tokens = x[:, 1:max_len+1, :]
            # 应用模式编码器
            pattern_features = self.pattern_encoder(x_tokens, hilbert_order_trimmed)
            # 残差连接并重建完整序列
            x_enhanced = x[:, 1:max_len+1, :] + pattern_features
            x = torch.cat([x[:, :1, :], x_enhanced, x[:, max_len+1:, :]], dim=1)

            # 标记为非插件模式
            _plugin_mode = False

        else:
            # 标记为非插件模式
            _plugin_mode = False

        # 如果不是插件模式，继续执行原有的 transformer 处理逻辑
        if not _plugin_mode:
            # 4. Transformer 处理 (v6.0: 注入 geometry_emb)
            x = self.transformer(
                x, levels_info, attn_mask,
                regions=regions, image_size=image_size,
                geometry_emb=geometry_emb_with_cls,
            )

            # 获取 transformer 输出 (排除 CLS token)
            transformer_tokens = x[:, 1:]

            # I30-11: 获取 split_probs 用于加权池化
            split_probs = token_output.get_padded_split_probs()

            # 5. 池化 + 分类
            pooled = self._apply_pooling(x, key_padding_mask, split_probs)
            final_output = self.mlp_head(pooled)
            pooled_for_stats = pooled

        # 6. 构建 TrainingStats
        # I141: 直接返回 GPU tensor，避免 .cpu() 调用导致 cudagraphs 失败
        # 训练器负责转换为 Python list
        num_tokens_tensor = lengths  # GPU tensor

        # 构建 split_info 字典
        split_info = {
            'levels_list': levels_list,
            'batch_size': batch_size,
        }

        # [Gradient Monitor] 添加 selected_mask 到 split_info 用于死节点检测
        # 检查 split_result 是否有 selected_mask 属性（GumbelTopKResult 有）
        if split_result is not None and hasattr(split_result, 'selected_mask'):
            # 使用 detach() 避免引入额外的梯度追踪
            split_info['selected_mask'] = split_result.selected_mask.detach()
            # 同时保存选中次数（按 batch 维度求和）
            split_info['selection_counts'] = split_result.selected_mask.sum(dim=0).detach()

        # I162-1: 仅在非插件模式下调用需要 split_probs 的函数
        if not _plugin_mode:
            aux_infos, _ = self._prepare_auxiliary_output(
                batch_size, lengths, levels_list, pooled, return_aux_info=False, return_features=False,
                split_probs=split_probs
            )

        # P2-FIX: 计算实际的 depth_distribution 而非空字典
        # 基于 _prepare_auxiliary_output 中的逻辑，避免 GPU-CPU 同步
        # I162-1: 插件模式下也需要计算深度分布
        depth_dist: Dict[int, float] = {}
        if levels_list and len(levels_list) > 0:
            # 计算 batch 平均深度分布
            max_level_range = self.max_level + 1
            depth_counts = torch.zeros(batch_size, max_level_range, dtype=torch.long, device=lengths.device)

            for i, levels in enumerate(levels_list):
                if levels.numel() > 0:
                    depths = levels[:, 0].long()
                    depths = depths[depths >= 0]
                    valid_depths = depths[depths < max_level_range]
                    if valid_depths.numel() > 0:
                        depth_counts[i].index_add_(0, valid_depths, torch.ones_like(valid_depths))

            # 归一化为概率分布
            total_counts = depth_counts.sum(dim=1, keepdim=True).clamp(min=1)
            normalized_counts = depth_counts.float() / total_counts.float()

            # 计算 batch 平均分布
            avg_distribution = normalized_counts.mean(dim=0)

            # I150-3-FIX: 确保归一化到 1.0
            depth_sum = avg_distribution.sum().clamp(min=1e-8)
            avg_distribution = avg_distribution / depth_sum

            depth_dist = {d: float(avg_distribution[d]) for d in range(max_level_range) if avg_distribution[d] > 0}

        # I135: 辅助函数 - 递归展平嵌套结构，提取所有整数值
        # P-OPT: 避免在 forward 中使用 .cpu()，使用 GPU 计算 max_level
        # 从 levels_list 推断主要使用的深度
        # I142: 使用向量化操作替代 for 循环中的 int() 转换，避免 CPU 同步
        # I145: 已知限制 - int() 转换仍会导致 CPU 同步，影响 torch.compile cudagraphs
        # I99-1 OPT: 修改 TrainingStats.depth_used 类型为 Union[int, Tensor]
        # 保持 Tensor 格式避免 CPU 同步，训练器负责在需要时转换
        if levels_list and len(levels_list) > 0:
            # 向量化计算所有 depths 的最大值
            all_max_levels = []
            for levels in levels_list:
                if levels.numel() > 0:
                    # 获取每个样本的最大深度
                    all_max_levels.append(levels[:, 0].max())
            if all_max_levels:
                # 使用 torch.stack 和 max，保持 GPU Tensor 格式
                max_level_tensor = torch.stack(all_max_levels).max()
                depth_used = max_level_tensor  # 保持 Tensor，避免 CPU 同步
            else:
                depth_used = torch.tensor(0, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.long)
        else:
            depth_used = torch.tensor(0, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.long)

        # I107-7: 在 training 模式下返回 shared_features供 auxiliary loss 使用
        return_features = features if self.training else None

        # I141: num_tokens 直接使用 GPU tensor，训练器负责转换
        # 保持原始 tensor 格式，避免 .cpu() 调用
        # I170: 添加 redundancy 和 child_features 到 TrainingStats
        stats = TrainingStats(
            logits=final_output,
            num_tokens=num_tokens_tensor,  # GPU tensor，避免 CPU 同步
            depth_used=depth_used,
            depth_distribution=depth_dist,
            features=pooled_for_stats,  # I162-1: 支持双路径模式下的 2D 特征
            transformer_tokens=transformer_tokens,
            shared_features=return_features,  # I107-7: 避免训练循环重复计算
            split_info=split_info,
            redundancy=redundancy,  # I170: 语义分裂器的冗余性分数
            child_features=child_features,  # I170: 语义分裂器的子节点特征
        )

        return stats

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
        # I99-1 OPT: 添加 non_blocking=True 进行异步传输
        return self._zero_loss.detach().to(self.aux_loss_weight.device, non_blocking=True)
    
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

        # I110-7: 清理语义分裂器状态（如果有）
        if hasattr(self, '_semantic_splitter') and self._semantic_splitter is not None:
            # SemanticRedundancySplitter 无缓存需要清理
            pass

    # =====================================================================
    # I110-7: 语义分裂器接口
    # =====================================================================
    @property
    def use_semantic_splitter(self) -> bool:
        """是否使用语义分裂器"""
        return self._use_semantic_splitter

    def get_semantic_splitter(self) -> Optional[nn.Module]:
        """获取语义分裂器实例"""
        return self._semantic_splitter

    def get_semantic_loss_fn(self) -> Optional[nn.Module]:
        """获取语义损失函数"""
        return self._semantic_loss_fn

    def compute_semantic_loss(
        self,
        parent_features: torch.Tensor,
        child_features: torch.Tensor,
        split_decision: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """计算语义冗余损失 (I110-7)

        Args:
            parent_features: [B, N, D] 父节点特征
            child_features: [B, N, 4, D] 子节点特征
            split_decision: [B, N] 分裂决策

        Returns:
            包含 loss, diversity_loss, reconstruction_loss 的字典
        """
        if self._semantic_loss_fn is None:
            return {'loss': torch.tensor(0.0, device=parent_features.device)}

        return self._semantic_loss_fn(parent_features, child_features, split_decision)

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
            # P-OPT: SemanticRedundancySplitter 需要 3D 输入
            if self._is_semantic_splitter:
                B, C, H_feat, W_feat = features.shape
                features = features.view(B, C, H_feat * W_feat).transpose(1, 2)
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
                    # P-OPT: avoid .item() in loop - compute max once
                    max_level = int(depths.max().item()) if depths.numel() > 0 else 0

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
                    # I141: 添加 non_blocking=True 避免同步阻塞
                    "level_usage_distribution": dict(zip(*torch.unique(level_tensor.cpu(non_blocking=True), return_counts=True))),
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

        # I121-2: 辅助损失权重配置 (修复配置链路断裂)
        if 'aux_loss_weights' in config and splitter is not None:
            weights = config['aux_loss_weights']

            # 应用稀疏性权重 (熵损失)
            if 'sparsity' in weights and hasattr(splitter, '_entropy_weight_base'):
                splitter._entropy_weight_base = weights['sparsity']

            # 应用弹性预算权重 (外部因子)
            if 'elastic' in weights and hasattr(splitter, '_elastic_budget_factor'):
                splitter._elastic_budget_factor = weights['elastic']

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
                - gradient_coverage: float - 梯度覆盖率 (I36 Phase 3)
                - temperature_status: str - 温度状态 (健康/过低)
        """
        diagnostics: Dict[str, Any] = {
            'current_temperature': 1.0,
            'depth_distribution': {},  # Dict[int, float]
            'quota_allocation': [],   # List[float]
            'num_selected': 0,
            'has_splitter': False,
            'splitter_type': 'None',
            'gradient_coverage': 0.0,
            'temperature_status': 'unknown',
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
                current_temp = splitter.get_current_temperature()
                # 提取标量用于日志和比较
                current_temp_val = current_temp.detach().cpu().float().item()
                diagnostics['current_temperature'] = current_temp_val

                # I36 Phase 3: 温度状态检查
                if current_temp_val < 0.3:
                    diagnostics['temperature_status'] = 'low_risk'  # T < 0.3 可能导致梯度消失
                elif current_temp_val < 0.5:
                    diagnostics['temperature_status'] = 'healthy'  # 健康范围
                else:
                    diagnostics['temperature_status'] = 'high_explore'  # 高温度，探索性强

            # 深度分布 - 兼容多种返回格式 (P0-2 修复)
            # Splitter 可能返回:
            # - Dict[int, float]: 深度 -> 百分比 (base_splitter 协议)
            # - Dict[str, Any]: {'pi': [...], 'entropy': ..., 'kl_from_uniform': ...} (GumbelTopKSplitter)
            raw_depth_dist = splitter.get_depth_distribution()
            if isinstance(raw_depth_dist, dict):
                if 'pi' in raw_depth_dist and raw_depth_dist['pi'] is not None:
                    # GumbelTopKSplitter 格式: {'pi': [D], ...}
                    pi = raw_depth_dist['pi']
                    diagnostics['depth_distribution'] = {
                        int(d): float(p) for d, p in enumerate(pi) if p > 0
                    }
                else:
                    # 已经是 Dict[int, float] 格式或空字典
                    # 过滤掉无法转换为整数的键
                    diagnostics['depth_distribution'] = {
                        int(k): float(v) for k, v in raw_depth_dist.items()
                        if self._is_convertible_to_int(k)
                    }

            # 配额分配
            if hasattr(splitter, 'quota_logits') and hasattr(splitter, '_current_max_level'):
                D = splitter._current_max_level + 1
                quota = torch.softmax(splitter.quota_logits[:D], dim=0)
                # I145: 延迟 CPU 转换，保留为 GPU tensor 或 detach 后转换
                # 训练时仅记录标量值，避免 cudagraphs 失败
                diagnostics['quota_allocation'] = quota.detach().cpu().tolist() if quota.numel() <= 32 else []

            # 选中的 token 数
            if hasattr(splitter, '_avg_selected'):
                diagnostics['num_selected'] = int(splitter._avg_selected)

            # I36 Phase 3: 梯度覆盖率估算
            # 数学: 梯度覆盖率 ≈ K_selected / N_candidates (Top-K 选择)
            # GumbelTopK 使用分层选择，估算为选中 token 比例
            if hasattr(splitter, '_avg_selected') and hasattr(splitter, '_num_candidates'):
                num_selected = splitter._avg_selected
                num_candidates = splitter._num_candidates
                if num_candidates > 0:
                    diagnostics['gradient_coverage'] = float(num_selected / num_candidates)

        return diagnostics

    def _fill_lazy_diagnostics(self, lazy_obj: Any) -> None:
        """填充延迟的 diagnostics (I145: 避免 cudagraphs CPU 同步)

        当 LazyDiagnostics 首次被访问时调用，延迟计算真实的 diagnostics
        """
        lazy_obj._filled = self.get_splitter_diagnostics()

    def _is_convertible_to_int(self, key) -> bool:
        """检查键是否可以转换为整数 (P0-2 修复辅助方法)"""
        try:
            int(key)
            return True
        except (ValueError, TypeError):
            return False

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
            'num_layers': self.num_layers,
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
            if hasattr(t, 'max_level'):
                tokenizer_info['max_level'] = t.max_level
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
            # I145: 修复 - 添加异常处理防止断言错误
            try:
                torch.backends.cuda.enable_cudnn_sdp(True)
            except Exception as e:
                import warnings
                warnings.warn(f"无法启用 cuDNN SDP: {e}，使用默认后端")

        # 配置 inductor 优化
        try:
            torch._inductor.config.max_autotune = True
            torch._inductor.config.cudnn_sdp = True  # 启用 cuDNN attention
            torch._inductor.config.coordinate_descent_tuning = True
        except Exception as e:
            import warnings
            warnings.warn(f"Inductor 配置失败: {e}")

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

