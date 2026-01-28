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

import torch
import torch.nn as nn


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
    depth: int                     # Transformer 层数
    heads: int                     # 注意力头数
    mlp_dim: int                   # FFN 隐藏层维度
    num_classes: int               # 分类类别数
    image_size: Optional[int]      # 图像尺寸 (None = 动态分辨率)
    channels: int = 3              # 输入通道数

    # ==================== Tokenizer 参数 ====================
    pool: str = "weighted"         # 池化类型
    min_patch_size: int = 4        # 最小 patch 尺寸
    max_depth: Optional[int] = None  # 最大深度 (从 tokenizer 获取)
    K_min: int = 16                # 最小 token 数
    K_max: int = 64                # 最大 token 数

    # ==================== 正则化参数 ====================
    dropout: float = 0.1           # Dropout 比率
    emb_dropout: float = 0.1       # 嵌入 dropout
    drop_path_rate: float = 0.15   # Drop path 比率

    # ==================== 编码选项 ====================
    use_hilbert_encoding: bool = True   # 使用 Hilbert 编码
    use_spatial_encoding: bool = True   # 使用空间编码
    use_checkpoint: bool = False        # 使用梯度检查点

    # ==================== FFN 选项 ====================
    ffn_type: str = "swiglu_level"  # FFN 类型

    # ==================== LCA 参数 ====================
    lca_temperature: float = 1.5      # LCA 温度
    learnable_temperature: bool = True  # 可学习温度

    # ==================== I24-2: 可学习配额 ====================
    quota_learnable: Optional[bool] = None  # 是否启用可学习配额
    quota_entropy_weight: float = 0.01  # 配额熵正则化权重

    # ==================== I31-3: 形状-尺度编码 ====================
    use_area_encoding: bool = False      # 使用面积编码
    use_affine_modulation: bool = True   # 使用仿射调制
    fourier_levels: int = 4              # Fourier 级别数

    # ==================== Splitter 关键参数 (I140) ====================
    splitter_hidden_dim: Optional[int] = None   # Splitter MLP 隐藏层维度
    splitter_feature_dim: Optional[int] = None  # Splitter 特征维度 (通常 = dim)
    splitter_pool_size: Optional[int] = None    # Splitter 池化大小

    # ==================== 训练元信息 ====================
    dataset_name: str = ""          # 数据集名称
    training_epochs: int = 0        # 总训练轮数
    checkpoint_epoch: int = 0       # Checkpoint 对应的轮数

    # ==================== 内部状态 (不序列化) ====================
    _model_config: Optional[Dict[str, Any]] = field(default=None, repr=False)

    # ==================== 序列化/反序列化 ====================

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典 (用于保存到 checkpoint)"""
        return {
            # 核心架构参数
            'dim': self.dim,
            'depth': self.depth,
            'heads': self.heads,
            'mlp_dim': self.mlp_dim,
            'num_classes': self.num_classes,
            'image_size': self.image_size,
            'channels': self.channels,

            # Tokenizer 参数
            'pool': self.pool,
            'min_patch_size': self.min_patch_size,
            'max_depth': self.max_depth,
            'K_min': self.K_min,
            'K_max': self.K_max,

            # 正则化
            'dropout': self.dropout,
            'emb_dropout': self.emb_dropout,
            'drop_path_rate': self.drop_path_rate,

            # 编码选项
            'use_hilbert_encoding': self.use_hilbert_encoding,
            'use_spatial_encoding': self.use_spatial_encoding,
            'use_checkpoint': self.use_checkpoint,

            # FFN
            'ffn_type': self.ffn_type,

            # LCA
            'lca_temperature': self.lca_temperature,
            'learnable_temperature': self.learnable_temperature,

            # I24-2
            'quota_learnable': self.quota_learnable,
            'quota_entropy_weight': self.quota_entropy_weight,

            # I31-3
            'use_area_encoding': self.use_area_encoding,
            'use_affine_modulation': self.use_affine_modulation,
            'fourier_levels': self.fourier_levels,

            # Splitter (I140)
            'splitter_hidden_dim': self.splitter_hidden_dim,
            'splitter_feature_dim': self.splitter_feature_dim,
            'splitter_pool_size': self.splitter_pool_size,

            # 训练元信息
            'dataset_name': self.dataset_name,
            'training_epochs': self.training_epochs,
            'checkpoint_epoch': self.checkpoint_epoch,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelGene":
        """从字典反序列化 (从 checkpoint 加载)"""
        # 处理嵌套结构 (ExperimentConfig 保存为 {"model": {...}, "training": {...}})
        if 'model' in d and isinstance(d['model'], dict):
            d = d['model']

        # 过滤掉不存在的字段 (向后兼容)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_d = {k: v for k, v in d.items() if k in valid_fields}

        return cls(**filtered_d)

    # ==================== 模型构建 ====================

    def build_model(self) -> nn.Module:
        """从 ModelGene 构建 FractalCurveViT 模型"""
        from vit_pytorch import FractalCurveViT
        from vit_pytorch.gumbel_topk_splitter import create_gumbel_topk_from_config

        # 处理 image_size（可能是 int 或 tuple）
        if self.image_size is None:
            img_size_for_splitter = None
        elif isinstance(self.image_size, tuple):
            img_size_for_splitter = self.image_size
        else:
            img_size_for_splitter = (self.image_size, self.image_size)

        # 构建 Splitter
        splitter = create_gumbel_topk_from_config(
            feature_dim=self.splitter_feature_dim or self.dim,
            min_patch_size=self.min_patch_size,
            max_depth_limit=self.max_depth,
            hidden_dim=self.splitter_hidden_dim,
            pool_size=self.splitter_pool_size,
            K_min=self.K_min,
            K_max=self.K_max,
            image_size=img_size_for_splitter,
        )

        # 构建模型
        model = FractalCurveViT(
            image_size=self.image_size,
            num_classes=self.num_classes,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            mlp_dim=self.mlp_dim,
            pool=self.pool,
            channels=self.channels,
            dropout=self.dropout,
            emb_dropout=self.emb_dropout,
            min_patch_size=self.min_patch_size,
            max_depth=None,  # 从 tokenizer 获取
            use_hilbert_encoding=self.use_hilbert_encoding,
            use_spatial_encoding=self.use_spatial_encoding,
            use_checkpoint=self.use_checkpoint,
            drop_path_rate=self.drop_path_rate,
            ffn_type=self.ffn_type,
            lca_temperature=self.lca_temperature,
            learnable_temperature=self.learnable_temperature,
            splitter=splitter,
            K_min=self.K_min,
            K_max=self.K_max,
            pos_dropout=None,
            use_area_encoding=self.use_area_encoding,
            use_affine_modulation=self.use_affine_modulation,
            fourier_levels=self.fourier_levels,
            quota_learnable=self.quota_learnable,
        )

        return model

    # ==================== 从模型提取 ====================

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        dataset_name: str = "",
        epoch: int = 0,
    ) -> "ModelGene":
        """从模型实例提取架构基因

        Args:
            model: FractalCurveViT 模型实例
            dataset_name: 数据集名称
            epoch: 当前 epoch
        """
        # 提取 dropout 值（可能是 float 或 Dropout 对象）
        dropout_val = getattr(model, 'dropout', 0.1)
        if hasattr(dropout_val, 'p'):  # PyTorch Dropout module
            dropout_val = dropout_val.p
        elif not isinstance(dropout_val, (int, float)):
            dropout_val = 0.1

        # 提取基础参数
        gene = cls(
            dim=model.dim,
            depth=model.depth,
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
            lca_temperature=getattr(model, 'lca_temperature', 1.5),
            learnable_temperature=getattr(model, 'learnable_temperature', True),
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
            gene.K_min = getattr(tokenizer, 'K_min', 16)
            gene.K_max = getattr(tokenizer, 'K_max', 64)

            # max_depth 从 tokenizer 获取
            if hasattr(tokenizer, 'max_depth'):
                gene.max_depth = tokenizer.max_depth

        # 从 splitter 提取参数 (I140)
        if hasattr(model, 'splitter'):
            splitter = model.splitter
            gene.splitter_hidden_dim = cls._detect_splitter_param(splitter, 'hidden_dim')
            gene.splitter_pool_size = cls._detect_splitter_param(splitter, 'pool_size')
            gene.splitter_feature_dim = cls._detect_splitter_param(splitter, 'feature_dim')

            # 可学习配额
            if hasattr(splitter, 'quota_learnable'):
                gene.quota_learnable = splitter.quota_learnable

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

    # ==================== 验证 ====================

    def validate(self) -> bool:
        """验证 ModelGene 的有效性"""
        errors = []

        # 必填参数检查
        if self.dim <= 0:
            errors.append(f"dim must be positive, got {self.dim}")
        if self.depth <= 0:
            errors.append(f"depth must be positive, got {self.depth}")
        if self.heads <= 0:
            errors.append(f"heads must be positive, got {self.heads}")
        if self.mlp_dim <= 0:
            errors.append(f"mlp_dim must be positive, got {self.mlp_dim}")
        if self.num_classes <= 0:
            errors.append(f"num_classes must be positive, got {self.num_classes}")

        # 兼容性检查
        if self.min_patch_size < 1:
            errors.append(f"min_patch_size must be >= 1, got {self.min_patch_size}")
        if self.K_min < 1:
            errors.append(f"K_min must be >= 1, got {self.K_min}")
        if self.K_max < self.K_min:
            errors.append(f"K_max must be >= K_min, got K_max={self.K_max} < K_min={self.K_min}")

        if errors:
            raise ValueError(f"Invalid ModelGene: {errors}")
        return True

    # ==================== 字符串表示 ====================

    def __str__(self) -> str:
        """可读字符串表示"""
        return (
            f"ModelGene(dim={self.dim}, depth={self.depth}, heads={self.heads}, "
            f"mlp_dim={self.mlp_dim}, num_classes={self.num_classes}, "
            f"image_size={self.image_size}, dataset={self.dataset_name})"
        )

    def __repr__(self) -> str:
        return f"ModelGene({self.to_dict()})"
