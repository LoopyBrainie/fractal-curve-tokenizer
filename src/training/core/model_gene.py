#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModelGene - 模型架构基因

包含重建模型所需的全部参数，确保 checkpoint 自包含。

Author: Claude
Date: 2026-01-27
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import math
import torch
import torch.nn as nn

try:
    from vit_pytorch.config import SemanticSplitterConfig
except ImportError:
    SemanticSplitterConfig = None  # 类型提示用，实际使用时确保已安装


@dataclass
class ModelGene:
    """模型架构基因 - 包含重建模型所需的全部参数

    用途：
        1. 保存到 checkpoint 中，使模型配置自包含
        2. 从 checkpoint 恢复模型时，无需外部 config.json
        3. 确保训练器和评估器使用完全相同的模型配置

    使用方式
    ========
    >>> # 保存 checkpoint 时
    >>> gene = ModelGene.from_model(model, dataset_name='cifar10', epoch=10)
    >>> checkpoint = {
    ...     'model_state_dict': model.state_dict(),
    ...     'model_gene': gene.to_dict(),
    ...     ...
    ... }
    >>>
    >>> # 加载 checkpoint 时
    >>> gene = ModelGene.from_dict(checkpoint['model_gene'])
    >>> model = gene.build_model()
    """

    # ==================== 核心架构参数 ====================
    dim: int                       # 模型维度
    num_layers: int                # Transformer 层数 (原 depth)
    heads: int                     # 注意力头数
    mlp_dim: int                   # FFN 隐藏层维度
    num_classes: int               # 分类类别数
    image_size: Optional[int]      # 图像尺寸 (None = 动态分辨率)
    channels: int = 3              # 输入通道数

    # ==================== Tokenizer 参数 ====================
    pool: str = "weighted"         # 池化类型
    min_patch_size: int = 4        # 最小 patch 尺寸
    # 注意: max_level 是变参数，由模型架构根据 image_size 和 min_patch_size 动态计算
    # 不保存到 ModelGene 中，确保训练/评估模型结构完全一致

    # I33: 相对预算参数 (替代绝对 K_min/K_max)
    # 保存覆盖率而非绝对 K 值，确保评估时能正确复算
    # I145: 修复默认值与 ModelArchitectureConfig 一致
    token_coverage_min: float = 0.01   # α = 1% 最小覆盖率 (向后兼容)
    token_coverage_max: float = 0.25   # β = 25% 最大覆盖率 (向后兼容)
    # I145: 统一命名覆盖率字段
    coverage_min: float = 0.01          # α = 1% 最小覆盖率 (与 HilbertSplitterConfig 一致)
    coverage_max_hard: float = 0.25     # β = 25% 最大覆盖率 (与 HilbertSplitterConfig 一致)

    # I145: Splitter K 值限制（从 SplitterConfig 提取）
    K_min_abs: int = 8                 # 绝对最小采样数
    K_max_hard: int = 8192             # 绝对最大采样数 (与 K_MAX_HARD_LIMIT 一致)
    coverage_base: float = 0.12         # 基准覆盖率
    splitter_temp_start: float = 1.0    # 初始温度 (与 constants.SPLITTER_TEMP_START 一致)
    splitter_temp_end: float = 0.4      # I145: 最终温度 (与 constants.SPLITTER_TEMP_END 一致)

    # ==================== 正则化参数 ====================
    # I148: 修复默认值与 ModelArchitectureConfig 和 train_fractal_vit.py 一致
    # 2026-02-07: 更新 emb_dropout=0.0，与 argparse --emb-dropout 默认值一致
    # dropout=0.1 (transformer_dropout), emb_dropout=0.0, drop_path_rate=0.25
    dropout: float = 0.1           # Dropout 比率 (与 args --transformer-dropout 一致)
    emb_dropout: float = 0.0        # 嵌入 dropout (与 args --emb-dropout 一致)
    drop_path_rate: float = 0.25   # Drop path 比率 (与 args --drop-path 一致)

    # ==================== 编码选项 ====================
    use_hilbert_encoding: bool = True   # 使用 Hilbert 编码
    use_spatial_encoding: bool = True   # 使用空间编码
    use_checkpoint: bool = False        # 使用梯度检查点

    # ==================== FFN 选项 ====================
    ffn_type: str = "swiglu_level"  # FFN 类型

    # I122-2: lca_temperature 已移除，由 hilbert_bias_scale × √d_k 统一缩放
    # lca_temperature: float = 1.5      # LCA 温度
    # learnable_temperature: bool = True  # 可学习温度

    # ==================== I24-2: 可学习配额 ====================
    quota_learnable: Optional[bool] = None  # 是否启用可学习配额
    quota_entropy_weight: float = 0.01  # 配额熵正则化权重

    # ==================== I31-3: 形状-尺度编码 ====================
    use_area_encoding: bool = False      # 使用面积编码
    use_affine_modulation: bool = True   # 使用仿射调制
    fourier_levels: int = 4              # Fourier 级别数

    # ==================== Splitter 关键参数 (I140) ====================
    # 注意: 这些是模型内部的默认值，如果为 None 则使用以下值
    splitter_hidden_dim: Optional[int] = None   # Splitter MLP 隐藏层维度 (None → 64)
    splitter_feature_dim: Optional[int] = None  # Splitter 特征维度 (None → 使用 dim)
    splitter_pool_size: Optional[int] = None    # Splitter 池化大小 (None → 4)

    # ==================== I136: Elastic Budget 配置 ====================
    # 这些参数控制训练时的弹性覆盖率约束，评估时需要保持一致
    elastic_coverage_min: float = 0.03   # 最小弹性覆盖率
    elastic_coverage_max: float = 0.25   # 最大弹性覆盖率
    elastic_lambda_over: float = 0.1    # 超额惩罚系数
    elastic_lambda_under: float = 0.01  # 低额惩罚系数

    # ==================== I110-7: 语义分裂器参数 ====================
    use_semantic_splitter: bool = False  # 是否使用 SemanticRedundancySplitter
    semantic_splitter_config: Optional[Dict[str, Any]] = None  # 语义分裂器配置字典
    semantic_loss_weight: float = 0.1  # 语义分裂器损失权重

    # ==================== 训练元信息 ====================
    dataset_name: str = ""          # 数据集名称
    training_epochs: int = 0        # 总训练轮数
    checkpoint_epoch: int = 0       # Checkpoint 对应的轮数

    # ==================== 内部状态 (不序列化) ====================
    _model_config: Optional[Dict[str, Any]] = field(default=None, repr=False)

    # ==================== 参数层级元数据 (不序列化) ====================
    # 用于记录参数的分类：hyperparameter, parameter, variable
    _parameter_tier: Dict[str, str] = field(default_factory=dict, repr=False)

    # ==================== 序列化/反序列化 ====================

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典 (用于保存到 checkpoint)

        遵循三层参数策略:
        - 第一层（架构参数）: dim, num_layers, heads, mlp_dim, image_size, channels
        - 第二层（变参数）: max_level - 不保存，由模型架构内部计算
        - 第三层（超参数）: token_coverage_*, dropout, ffn_type 等
        """
        return {
            # ========== 第一层：架构参数 ==========
            'dim': self.dim,
            'num_layers': self.num_layers,
            'heads': self.heads,
            'mlp_dim': self.mlp_dim,
            'num_classes': self.num_classes,
            'image_size': self.image_size,
            'channels': self.channels,

            # ========== Tokenizer 参数 ==========
            'pool': self.pool,
            'min_patch_size': self.min_patch_size,
            # 注意: max_level 是变参数，由模型架构动态计算，不保存
            # I33: 保存覆盖率而非绝对 K 值（确保不同分辨率下正确复算）
            'token_coverage_min': self.token_coverage_min,
            'token_coverage_max': self.token_coverage_max,
            # I145: 统一命名覆盖率字段
            'coverage_min': self.coverage_min,
            'coverage_max_hard': self.coverage_max_hard,

            # ========== I145: Splitter K 值限制 ==========
            'K_min_abs': self.K_min_abs,
            'K_max_hard': self.K_max_hard,
            'coverage_base': self.coverage_base,
            'splitter_temp_start': self.splitter_temp_start,
            'splitter_temp_end': self.splitter_temp_end,

            # ========== 正则化参数 ==========
            'dropout': self.dropout,
            'emb_dropout': self.emb_dropout,
            'drop_path_rate': self.drop_path_rate,

            # ========== 编码选项 ==========
            'use_hilbert_encoding': self.use_hilbert_encoding,
            'use_spatial_encoding': self.use_spatial_encoding,
            'use_checkpoint': self.use_checkpoint,

            # ========== FFN 选项 ==========
            'ffn_type': self.ffn_type,

            # I122-2: lca_temperature 已移除
            # 'lca_temperature': self.lca_temperature,
            # 'learnable_temperature': self.learnable_temperature,

            # ========== I24-2: 可学习配额 ==========
            'quota_learnable': self.quota_learnable,
            'quota_entropy_weight': self.quota_entropy_weight,

            # ========== I31-3: 形状-尺度编码 ==========
            'use_area_encoding': self.use_area_encoding,
            'use_affine_modulation': self.use_affine_modulation,
            'fourier_levels': self.fourier_levels,

            # ========== I140: Splitter 关键参数 ==========
            'splitter_hidden_dim': self.splitter_hidden_dim,
            'splitter_feature_dim': self.splitter_feature_dim,
            'splitter_pool_size': self.splitter_pool_size,

            # ========== I136: Elastic Budget 配置 ==========
            'elastic_coverage_min': self.elastic_coverage_min,
            'elastic_coverage_max': self.elastic_coverage_max,
            'elastic_lambda_over': self.elastic_lambda_over,
            'elastic_lambda_under': self.elastic_lambda_under,

            # ========== I110-7: 语义分裂器 ==========
            'use_semantic_splitter': self.use_semantic_splitter,
            'semantic_splitter_config': self.semantic_splitter_config,
            'semantic_loss_weight': self.semantic_loss_weight,

            # ========== 训练元信息 ==========
            'dataset_name': self.dataset_name,
            'training_epochs': self.training_epochs,
            'checkpoint_epoch': self.checkpoint_epoch,

            # ========== 参数层级元数据（不序列化但保留） ==========
            '_parameter_tier': self._parameter_tier,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelGene":
        """从字典反序列化 (从 checkpoint 加载)

        支持新旧两种 checkpoint 格式的迁移:
        - 新格式: 使用 token_coverage_min/token_coverage_max
        - 旧格式: 使用 K_min/K_max 绝对值
        """
        # 处理嵌套结构 (ExperimentConfig 保存为 {"model": {...}, "training": {...}})
        if 'model' in d and isinstance(d['model'], dict):
            d = d['model']

        # 创建副本避免修改原始字典
        d = d.copy()

        # I145: 处理字段映射 - 旧版使用 depth，新版使用 num_layers
        if 'depth' in d and 'num_layers' not in d:
            d['num_layers'] = d['depth']

        # 注意: max_level/max_depth 是变参数，由模型架构内部动态计算
        # 不再从 checkpoint 读取或迁移

        # I33: 迁移支持 - 旧版 checkpoint 使用 K_min/K_max
        # 如果有旧的 K_min/K_max 字段但没有新的 token_coverage_* 字段，则进行迁移
        if ('token_coverage_min' not in d or 'token_coverage_max' not in d) and 'K_min' in d:
            # 从 K_min/K_max 和 max_depth 计算覆盖率
            K_min = d['K_min']
            K_max = d['K_max']

            # 获取 max_level（兼容新旧字段名）
            max_level = d.get('tokenizer_max_level') or d.get('max_depth') or 6

            # 计算候选节点总数 N = (4^(max_level+1) - 1) / 3
            N = (4 ** (max_level + 1) - 1) // 3

            # 使用 image_size 计算 scale 因子
            # 原始 K 值通常是针对特定分辨率计算的
            image_size = d.get('image_size', 224)
            if isinstance(image_size, tuple):
                image_size = min(image_size)
            scale = math.sqrt(image_size / 224)

            # 反推覆盖率: coverage = K / (N * scale)
            # 考虑分辨率 scale 因子
            token_coverage_min = K_min / (N * scale)
            token_coverage_max = K_max / (N * scale)

            # 钳制到合理范围
            token_coverage_min = max(0.001, min(0.5, token_coverage_min))
            token_coverage_max = max(token_coverage_min + 0.001, min(0.5, token_coverage_max))

            d['token_coverage_min'] = token_coverage_min
            d['token_coverage_max'] = token_coverage_max

        # 再次检查必需字段
        if 'token_coverage_min' not in d or 'token_coverage_max' not in d:
            raise ValueError(
                "Checkpoint 中缺少必需字段 token_coverage_min 和 token_coverage_max，"
                "且无法从旧版 K_min/K_max 迁移。"
                " 请使用新版训练脚本重新训练。"
            )

        # 过滤掉不存在的字段 (向后兼容)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_d = {k: v for k, v in d.items() if k in valid_fields}

        return cls(**filtered_d)

    # ==================== 模型构建 ====================

    def build_model(self) -> nn.Module:
        """从 ModelGene 构建 FractalCurveViT 模型

        三层参数策略（完全信任模型架构源码）：
        - 第一层（架构参数）: dim, num_layers, heads, mlp_dim, min_patch_size, image_size
        - 第二层（动态参数）: max_level → 由 FractalCurveViT → StreamingFractalTokenizerV3 动态计算
        - 第三层（超参数）: token_coverage, dropout, ffn_type 等

        关键：完全信任模型架构源码，不手动创建任何子组件。
        FractalCurveViT 会根据 image_size 和 min_patch_size 自动计算 max_level。

        注意: max_level 是变参数，完全由模型架构内部计算，不从外部传入。
              确保训练/评估模型结构完全一致。
        """
        from vit_pytorch import FractalCurveViT

        # 传递所有保存的参数，让模型架构创建所有组件
        # I120-2: 使用分离的 dropout 参数
        model = FractalCurveViT(
            image_size=self.image_size,
            num_classes=self.num_classes,
            dim=self.dim,
            num_layers=self.num_layers,
            heads=self.heads,
            dim_head=self.dim // self.heads,  # 确保与 heads 一致
            mlp_dim=self.mlp_dim,
            pool=self.pool,
            channels=self.channels,
            tokenizer_dropout=self.dropout,  # 复用 dropout 字段作为 tokenizer_dropout
            transformer_dropout=self.dropout,
            emb_dropout=self.emb_dropout,
            min_patch_size=self.min_patch_size,
            # max_level 不传递，由模型架构内部动态计算
            use_hilbert_encoding=self.use_hilbert_encoding,
            use_spatial_encoding=self.use_spatial_encoding,
            use_checkpoint=self.use_checkpoint,
            drop_path_rate=self.drop_path_rate,
            ffn_type=self.ffn_type,
            # I122-2: lca_temperature 已移除
            # lca_temperature=self.lca_temperature,
            # learnable_temperature=self.learnable_temperature,
            # 不传递自定义 splitter，让 FractalCurveViT 自己创建
            splitter=None,
            token_coverage_min=self.token_coverage_min,
            token_coverage_max=self.token_coverage_max,
            pos_dropout=None,
            use_area_encoding=self.use_area_encoding,
            use_affine_modulation=self.use_affine_modulation,
            fourier_levels=self.fourier_levels,
            quota_learnable=self.quota_learnable,
            quota_entropy_weight=self.quota_entropy_weight,
            # I140: Splitter 架构参数
            splitter_hidden_dim=self.splitter_hidden_dim,
            splitter_feature_dim=self.splitter_feature_dim,
            splitter_pool_size=self.splitter_pool_size,
            # I145: Splitter 温度参数
            splitter_temp_start=self.splitter_temp_start,
            splitter_temp_end=self.splitter_temp_end,
            # I110-7: 语义分裂器配置
            use_semantic_splitter=self.use_semantic_splitter,
            semantic_splitter_config=(
                SemanticSplitterConfig(**self.semantic_splitter_config)
                if self.semantic_splitter_config and SemanticSplitterConfig else None
            ),
        )

        return model

    # ==================== 从模型提取 ====================

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        dataset_name: str = "",
        epoch: int = 0,
        verbose: bool = False,
    ) -> "ModelGene":
        """从模型实例提取架构基因

        注意：这是向后兼容方法。首选使用 from_config() 直接从配置构造。

        Args:
            model: FractalCurveViT 模型实例
            dataset_name: 数据集名称
            epoch: 当前 epoch
            verbose: 是否打印详细信息
        """
        # 提取 dropout 值
        # I122-1: FractalCurveViT 使用分离的 dropout 参数
        # 优先使用 model.transformer_dropout，其次是 model.dropout (兼容旧模型)
        dropout_val = getattr(model, 'transformer_dropout', None)
        if dropout_val is None:
            dropout_attr = getattr(model, 'dropout', None)
            if dropout_attr is not None:
                if isinstance(dropout_attr, (int, float)):
                    dropout_val = dropout_attr
                elif hasattr(dropout_attr, 'p'):  # PyTorch Dropout module
                    dropout_val = dropout_attr.p
                else:
                    dropout_val = 0.1
            else:
                dropout_val = 0.1

        # 提取基础参数
        gene = cls(
            dim=model.dim,
            num_layers=getattr(model, 'num_layers', 6),  # I145: 简化 - 模型使用 num_layers
            heads=model.heads,
            mlp_dim=model.mlp_dim,
            num_classes=cls._detect_num_classes(model),
            image_size=getattr(model, 'image_size', None),
            channels=getattr(model, 'channels', 3),
            pool=getattr(model, 'pool', 'weighted'),
            dropout=dropout_val,
            emb_dropout=getattr(model, 'emb_dropout', 0.1),
            drop_path_rate=getattr(model, 'drop_path_rate', 0.15),
            use_hilbert_encoding=getattr(model, 'use_hilbert_encoding', True),
            use_spatial_encoding=getattr(model, 'use_spatial_encoding', True),
            use_checkpoint=getattr(model, 'use_checkpoint', False),
            ffn_type=getattr(model, 'ffn_type', 'swiglu_level'),
            # I122-2: lca_temperature 已移除
            # lca_temperature=getattr(model, 'lca_temperature', 1.5),
            # learnable_temperature=getattr(model, 'learnable_temperature', True),
            use_area_encoding=getattr(model, 'use_area_encoding', False),
            use_affine_modulation=getattr(model, 'use_affine_modulation', True),
            fourier_levels=getattr(model, 'fourier_levels', 4),
            dataset_name=dataset_name,
            checkpoint_epoch=epoch,
        )

        # 从 tokenizer 提取参数
        if hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
            gene.min_patch_size = getattr(tokenizer, 'min_patch_size', 4)

            # I33: 从 SplitterConfig 提取覆盖率参数
            # 这是一级参数，用于在不同分辨率下正确复算 K 值
            # 优先从 model.splitter.config 提取（最可靠）
            if hasattr(model, 'splitter') and hasattr(model.splitter, 'config') and model.splitter.config is not None:
                config = model.splitter.config
                # I145: 修复 - SplitterConfig 使用 coverage_min/max_hard 而非 token_coverage_min/max
                gene.token_coverage_min = getattr(config, 'coverage_min', 0.01)
                gene.token_coverage_max = getattr(config, 'coverage_max_hard', 0.25)  # I109-3
                # I145: 提取 K 值限制参数
                gene.K_min_abs = getattr(config, 'K_min_abs', 8)
                gene.K_max_hard = getattr(config, 'K_max_hard', 8192)
                gene.coverage_base = getattr(config, 'coverage_base', 0.12)
                # I145: 提取温度参数
                gene.splitter_temp_start = getattr(config, 'temperature_init', 1.0)
                gene.splitter_temp_end = getattr(config, 'temperature_min', 0.1)
            # 备选：从 tokenizer.splitter.config 提取
            elif hasattr(tokenizer, 'splitter') and hasattr(tokenizer.splitter, 'config') and tokenizer.splitter.config is not None:
                config = tokenizer.splitter.config
                gene.token_coverage_min = getattr(config, 'coverage_min', 0.01)
                gene.token_coverage_max = getattr(config, 'coverage_max_hard', 0.25)  # I109-3
                gene.K_min_abs = getattr(config, 'K_min_abs', 8)
                gene.K_max_hard = getattr(config, 'K_max_hard', 8192)
                gene.coverage_base = getattr(config, 'coverage_base', 0.12)
                gene.splitter_temp_start = getattr(config, 'temperature_init', 1.0)
                gene.splitter_temp_end = getattr(config, 'temperature_min', 0.1)
            else:
                raise ValueError(
                    "无法从模型提取 token_coverage_* 参数。"
                    " 请确保模型使用包含 SplitterConfig 的 GumbelTopKSplitter。"
                )

            # I145: 分离提取两个 max_level
            # 注意: max_level 是变参数，由模型架构内部计算，不保存到 ModelGene
            # 如果需要获取 max_level，应直接从 model.max_level 读取

        # 从 splitter 提取参数 (I140)
        if hasattr(model, 'splitter'):
            splitter = model.splitter
            gene.splitter_hidden_dim = cls._detect_splitter_param(splitter, 'hidden_dim')
            gene.splitter_pool_size = cls._detect_splitter_param(splitter, 'pool_size')
            gene.splitter_feature_dim = cls._detect_splitter_param(splitter, 'feature_dim')

            # I110-7: 提取语义分裂器配置
            # I145: 修复 - 只在 semantic_splitter 实际启用时提取配置
            model_use_semantic = getattr(model, '_use_semantic_splitter', False)
            gene.use_semantic_splitter = model_use_semantic

            if model_use_semantic and hasattr(splitter, 'config') and splitter.config is not None:
                config = splitter.config
                if hasattr(config, 'to_dict'):
                    gene.semantic_splitter_config = config.to_dict()
                elif isinstance(config, dict):
                    gene.semantic_splitter_config = config

        # I145: 从模型权重推断真实的架构配置（解决训练代码与权重不一致的问题）
        # 某些架构参数（如 heads）可能在权重中与模型属性不一致
        actual_heads = cls._detect_actual_heads_from_model(model)
        if actual_heads is not None:
            gene.heads = actual_heads
            if verbose:
                print(f"  [I145] 从模型权重检测到真实的 heads={actual_heads}")

            # I145: 可学习配额（从 Splitter.config 或 GumbelTopKSplitter._enable_learnable_quota 提取）
            if hasattr(splitter, 'config') and splitter.config is not None:
                gene.quota_learnable = getattr(splitter.config, 'enable_learnable_quota', None)
            elif hasattr(splitter, '_enable_learnable_quota'):
                gene.quota_learnable = splitter._enable_learnable_quota

        return gene

    @classmethod
    def from_config(
        cls,
        config: "ModelArchitectureConfig",
        dataset_name: str = "",
        epoch: int = 0,
    ) -> "ModelGene":
        """从 ModelArchitectureConfig 构造 ModelGene（推荐方式）

        这是保存 checkpoint 时构造 ModelGene 的首选方法。
        训练器在构建模型时已经持有配置，直接从配置构造 ModelGene，
        而不是从已创建的模型中重新提取（违反单一数据源原则）。

        Args:
            config: ModelArchitectureConfig 对象
            dataset_name: 数据集名称
            epoch: 当前 epoch

        Returns:
            ModelGene 对象
        """
        # I145: 从 config 获取温度参数（支持 TrainingConfig 和 args 两种格式）
        # TrainingConfig 使用 splitter_temp_start/end，args 使用其他命名
        splitter_temp_start = getattr(config, 'splitter_temp_start', None)
        if splitter_temp_start is None:
            # 兼容旧版 config 或直接从 args 获取
            splitter_temp_start = getattr(config, 'temperature_init', 1.0)

        splitter_temp_end = getattr(config, 'splitter_temp_end', None)
        if splitter_temp_end is None:
            # I145 修复: 使用 constants.SPLITTER_TEMP_END (0.4) 而非错误的 0.1
            from vit_pytorch.constants import SPLITTER_TEMP_END
            splitter_temp_end = getattr(config, 'temperature_min', SPLITTER_TEMP_END)

        # 使用 getattr 处理可选字段（兼容不同版本的 ModelArchitectureConfig）
        gene = cls(
            dim=config.dim,
            num_layers=config.num_layers,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
            num_classes=config.num_classes,
            image_size=config.image_size,
            channels=config.channels,
            pool=config.pool,
            min_patch_size=config.min_patch_size,
            dropout=config.dropout,
            emb_dropout=config.emb_dropout,
            drop_path_rate=config.drop_path_rate,
            # 使用 getattr 兼容可选字段
            use_hilbert_encoding=getattr(config, 'use_hilbert_encoding', True),
            use_spatial_encoding=getattr(config, 'use_spatial_encoding', True),
            use_checkpoint=config.use_checkpoint,
            ffn_type=config.ffn_type,
            # I122-2: lca_temperature 已移除
            # lca_temperature=config.lca_temperature if config.lca_temperature is not None else 1.5,
            # learnable_temperature=config.learnable_temperature,
            token_coverage_min=config.token_coverage_min,
            token_coverage_max=config.token_coverage_max,
            # I145: 统一覆盖率字段
            coverage_min=getattr(config, 'coverage_min', config.token_coverage_min),
            coverage_max_hard=getattr(config, 'coverage_max_hard', config.token_coverage_max),
            # Splitter 温度参数（I145: 确保从训练配置正确保存）
            splitter_temp_start=splitter_temp_start,
            splitter_temp_end=splitter_temp_end,
            use_area_encoding=config.use_area_encoding,
            use_affine_modulation=config.use_affine_modulation,
            fourier_levels=config.fourier_levels,
            quota_learnable=config.quota_learnable,
            quota_entropy_weight=getattr(config, 'quota_entropy_weight', 0.01),
            # I140: Splitter 架构参数
            splitter_hidden_dim=getattr(config, 'splitter_hidden_dim', None),
            splitter_feature_dim=getattr(config, 'splitter_feature_dim', None),
            splitter_pool_size=getattr(config, 'splitter_pool_size', None),
            # I136: Elastic Budget 配置
            elastic_coverage_min=getattr(config, 'elastic_coverage_min', 0.03),
            elastic_coverage_max=getattr(config, 'elastic_coverage_max', 0.25),
            elastic_lambda_over=getattr(config, 'elastic_lambda_over', 0.1),
            elastic_lambda_under=getattr(config, 'elastic_lambda_under', 0.01),
            # I110-7: 语义分裂器配置
            use_semantic_splitter=getattr(config, 'use_semantic_splitter', False),
            semantic_splitter_config=getattr(config, 'semantic_splitter_config', None),
            semantic_loss_weight=getattr(config, 'semantic_loss_weight', 0.1),
            dataset_name=dataset_name,
            checkpoint_epoch=epoch,
        )

        return gene

    @staticmethod
    def _detect_num_classes(model: nn.Module) -> int:
        """从模型检测类别数 - 找最后一个 head 层的输出维度"""
        # 优先使用模型的 num_classes 属性
        if hasattr(model, 'num_classes'):
            return model.num_classes

        # 备选: 从参数名推断
        for name, param in model.named_parameters():
            if '.weight' in name or '.bias' in name:
                # 找 mlp_head 的最后一个 linear 层 (shape[0] 应该是类别数)
                if 'mlp_head' in name and param.shape[0] <= 1000:
                    # 检查是否是最后一个 linear 层（输出层）
                    # 输出层的 shape[0] 通常等于 num_classes
                    return param.shape[0]
                # 找 head 层的输出维度
                if name == 'head.4.weight' or name == 'head.4.bias':
                    return param.shape[0]
        return 10  # 默认值

    @staticmethod
    def _detect_splitter_param(splitter: nn.Module, param_name: str) -> Optional[Any]:
        """从 splitter 检测参数"""
        if param_name == 'hidden_dim':
            # 从 complexity_mlp 第一层检测
            if hasattr(splitter, 'complexity_mlp') and len(splitter.complexity_mlp) > 0:
                first_layer = splitter.complexity_mlp[0]
                if hasattr(first_layer, 'weight'):
                    return first_layer.weight.shape[0]
        elif param_name == 'pool_size':
            # 从特征图推断
            if hasattr(splitter, 'pool_size'):
                return splitter.pool_size
            # 或从 hidden_dim 和 feature_dim 推断
            hidden_dim = ModelGene._detect_splitter_param(splitter, 'hidden_dim')
            feature_dim = ModelGene._detect_splitter_param(splitter, 'feature_dim')
            if hidden_dim and feature_dim:
                # 假设 input_dim = feature_dim * pool_size^2
                import math
                ratio = feature_dim / hidden_dim if hidden_dim > 0 else 1
                pool_size = int(math.sqrt(1 / ratio)) if ratio > 0 else 4
                return max(1, pool_size)
        elif param_name == 'feature_dim':
            # 通常等于 model.dim
            if hasattr(splitter, 'feature_dim'):
                return splitter.feature_dim
        return None

    @staticmethod
    def _detect_actual_heads_from_model(model: nn.Module) -> Optional[int]:
        """I145: 从模型权重推断真实的 heads 数

        某些架构参数（如 heads）可能在权重中与模型属性不一致。
        通过检查多个权重的形状，可以推断实际使用的 heads 数。

        注意: to_qkv.weight shape = [3*dim_head*heads, dim] 无法唯一确定 heads，
        因为多个 heads 值可能产生相同的 to_qkv 形状。
        需要结合其他权重（如 _level_scale_raw）来唯一确定。

        Returns:
            实际的 heads 数，如果无法推断则返回 None
        """
        # 方法1: 从 _level_scale_raw embedding 直接获取 heads
        # _level_scale_raw.weight shape = [num_levels, heads]
        for name, param in model.named_parameters():
            if '_level_scale_raw.weight' in name and len(param.shape) == 2:
                # param.shape[1] = heads
                if param.shape[1] in [1, 2, 4, 6, 8, 12, 16]:
                    return param.shape[1]

        # 方法2: 从 relative_pos_embedding 获取 heads
        for name, param in model.named_parameters():
            if 'relative_pos_embedding.weight' in name and len(param.shape) == 2:
                # param.shape[1] = heads
                if param.shape[1] in [1, 2, 4, 6, 8, 12, 16]:
                    return param.shape[1]

        # 方法3: 回退到 to_qkv（不够准确，但对于某些配置有效）
        for name, param in model.named_parameters():
            if '.attention.to_qkv.weight' in name:
                if len(param.shape) == 2:
                    dim = param.shape[1]
                    # 遍历可能的 heads 值，检查 dim 是否能被整除
                    for heads in [1, 2, 4, 6, 8, 12, 16]:
                        if dim % heads == 0:
                            dim_head = dim // heads
                            expected_qkv_dim = 3 * dim_head * heads
                            if param.shape[0] == expected_qkv_dim:
                                return heads
                break
        return None

    # ==================== 验证 ====================

    def validate(self) -> bool:
        """验证 ModelGene 的有效性

        I145: 添加配置一致性验证，确保与常量定义对齐。
        """
        errors = []
        warnings = []

        # 必填参数检查
        if self.dim <= 0:
            errors.append(f"dim must be positive, got {self.dim}")
        if self.num_layers <= 0:
            errors.append(f"num_layers must be positive, got {self.num_layers}")
        if self.heads <= 0:
            errors.append(f"heads must be positive, got {self.heads}")
        if self.mlp_dim <= 0:
            errors.append(f"mlp_dim must be positive, got {self.mlp_dim}")
        if self.num_classes <= 0:
            errors.append(f"num_classes must be positive, got {self.num_classes}")

        # 兼容性检查
        if self.min_patch_size < 1:
            errors.append(f"min_patch_size must be >= 1, got {self.min_patch_size}")

        # I33: 覆盖率参数验证 (数学约束: 0 < α < β < 1)
        if not (0 < self.token_coverage_min < self.token_coverage_max < 1):
            errors.append(
                f"覆盖率约束违反: 0 < {self.token_coverage_min} < {self.token_coverage_max} < 1"
            )

        # I145: 温度参数验证
        from vit_pytorch.constants import SPLITTER_TEMP_START, SPLITTER_TEMP_END, TEMPERATURE_MIN
        if self.splitter_temp_start < TEMPERATURE_MIN:
            warnings.append(
                f"splitter_temp_start ({self.splitter_temp_start}) < TEMPERATURE_MIN ({TEMPERATURE_MIN}), "
                f"可能导致梯度消失问题"
            )
        if self.splitter_temp_end < TEMPERATURE_MIN:
            errors.append(
                f"splitter_temp_end ({self.splitter_temp_end}) < TEMPERATURE_MIN ({TEMPERATURE_MIN}), "
                f"这会导致梯度消失"
            )
        if self.splitter_temp_start < self.splitter_temp_end:
            errors.append(
                f"splitter_temp_start ({self.splitter_temp_start}) < splitter_temp_end ({self.splitter_temp_end}), "
                f"温度应该从高到低退火"
            )

        # I145: K 边界验证
        from vit_pytorch.constants import K_MIN_HARD_LIMIT, K_MAX_HARD_LIMIT
        if self.K_min_abs < K_MIN_HARD_LIMIT:
            warnings.append(
                f"K_min_abs ({self.K_min_abs}) < K_MIN_HARD_LIMIT ({K_MIN_HARD_LIMIT}), "
                f"将被自动调整"
            )
        if self.K_max_hard > K_MAX_HARD_LIMIT:
            warnings.append(
                f"K_max_hard ({self.K_max_hard}) > K_MAX_HARD_LIMIT ({K_MAX_HARD_LIMIT}), "
                f"将被自动调整"
            )

        # I145: Splitter 架构参数验证
        if self.splitter_hidden_dim is not None and self.splitter_hidden_dim <= 0:
            errors.append(f"splitter_hidden_dim must be positive, got {self.splitter_hidden_dim}")
        if self.splitter_feature_dim is not None and self.splitter_feature_dim <= 0:
            errors.append(f"splitter_feature_dim must be positive, got {self.splitter_feature_dim}")
        if self.splitter_pool_size is not None and self.splitter_pool_size <= 0:
            errors.append(f"splitter_pool_size must be positive, got {self.splitter_pool_size}")

        if errors:
            raise ValueError(f"Invalid ModelGene: {'; '.join(errors)}")

        if warnings:
            import warnings as _warnings
            for w in warnings:
                _warnings.warn(f"[ModelGene Validation] {w}")

        return True

    # ==================== 字符串表示 ====================

    def __str__(self) -> str:
        """可读字符串表示"""
        return (
            f"ModelGene(dim={self.dim}, num_layers={self.num_layers}, heads={self.heads}, "
            f"mlp_dim={self.mlp_dim}, num_classes={self.num_classes}, "
            f"image_size={self.image_size}, dataset={self.dataset_name})"
        )

    def __repr__(self) -> str:
        return f"ModelGene({self.to_dict()})"
