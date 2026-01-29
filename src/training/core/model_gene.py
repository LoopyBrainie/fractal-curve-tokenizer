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
    # I145: 分离两个 max_level（之前是同一个参数控制两个不同组件）
    # - tokenizer_max_level: 用于 Splitter/Tokenizer（动态计算，通常基于图像尺寸）
    # - transformer_max_level: 用于 Transformer/lca_embedding（从 args.max_level 获取）
    tokenizer_max_level: Optional[int] = None  # Tokenizer/Splitter 用的 max_level
    transformer_max_level: Optional[int] = None  # Transformer/lca_embedding 用的 max_level

    # I33: 相对预算参数 (替代绝对 K_min/K_max)
    # 保存覆盖率而非绝对 K 值，确保评估时能正确复算
    token_coverage_min: float = 0.01   # α = 1% 最小覆盖率
    token_coverage_max: float = 0.05   # β = 5% 最大覆盖率

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

    # ==================== 参数层级元数据 (不序列化) ====================
    # 用于记录参数的分类：hyperparameter, parameter, variable
    _parameter_tier: Dict[str, str] = field(default_factory=dict, repr=False)

    # ==================== 序列化/反序列化 ====================

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典 (用于保存到 checkpoint)"""
        return {
            # 核心架构参数
            'dim': self.dim,
            'num_layers': self.num_layers,
            'heads': self.heads,
            'mlp_dim': self.mlp_dim,
            'num_classes': self.num_classes,
            'image_size': self.image_size,
            'channels': self.channels,

            # Tokenizer 参数
            'pool': self.pool,
            'min_patch_size': self.min_patch_size,
            # I145: 分离保存两个 max_level
            'tokenizer_max_level': self.tokenizer_max_level,
            'transformer_max_level': self.transformer_max_level,
            # I33: 保存覆盖率而非绝对 K 值
            'token_coverage_min': self.token_coverage_min,
            'token_coverage_max': self.token_coverage_max,

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

            # 参数层级元数据
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

        # I145: 处理向后兼容 - 旧版 checkpoint 只有 max_depth 字段
        if 'max_depth' in d:
            if 'tokenizer_max_level' not in d and 'transformer_max_level' not in d:
                # 旧版格式：max_depth 同时用于 tokenizer 和 transformer
                d['tokenizer_max_level'] = d.get('max_depth')
                d['transformer_max_level'] = d.get('max_depth')

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

        关键：K 值从 coverage 复算，确保评估时与训练时的相对预算一致。
        """
        from vit_pytorch import FractalCurveViT
        from vit_pytorch.gumbel_topk_splitter import create_gumbel_topk_from_config

        # 处理 image_size（可能是 int 或 tuple）
        if self.image_size is None:
            img_size_for_splitter = None
        elif isinstance(self.image_size, tuple):
            img_size_for_splitter = self.image_size
        else:
            img_size_for_splitter = (self.image_size, self.image_size)

        # I145: dim_head 必须等于 dim // heads，确保 to_qkv 形状正确
        dim_head = self.dim // self.heads

        # I33: 从 coverage 复算 K 值
        # 公式: K = coverage * max_patches = coverage * (image_size/min_patch_size)^2
        if img_size_for_splitter is not None:
            img_h, img_w = img_size_for_splitter
            img_size_for_budget = min(img_h, img_w)
            max_patches = (img_size_for_budget // self.min_patch_size) ** 2
            K_min_computed = max(4, int(max_patches * self.token_coverage_min))
            K_max_computed = int(max_patches * self.token_coverage_max)
        else:
            # 动态分辨率：使用默认 224 作为参考
            max_patches = (224 // self.min_patch_size) ** 2
            K_min_computed = max(4, int(max_patches * self.token_coverage_min))
            K_max_computed = int(max_patches * self.token_coverage_max)

        # I145: 使用 tokenizer_max_level 构建 Splitter
        # 注意：只有当值不为 None 时才传递，否则使用 create_gumbel_topk_from_config 的默认值
        splitter_kwargs = dict(
            feature_dim=self.splitter_feature_dim if self.splitter_feature_dim else self.dim,
            min_patch_size=self.min_patch_size,
            K_min=K_min_computed,
            K_max=K_max_computed,
            image_size=img_size_for_splitter,
            # I33: 传递覆盖率参数
            token_coverage_min=self.token_coverage_min,
            token_coverage_max_hard=self.token_coverage_max,
        )
        if self.tokenizer_max_level is not None:
            splitter_kwargs['max_level_limit'] = self.tokenizer_max_level
        if self.splitter_hidden_dim is not None:
            splitter_kwargs['hidden_dim'] = self.splitter_hidden_dim
        if self.splitter_pool_size is not None:
            splitter_kwargs['pool_size'] = self.splitter_pool_size

        splitter = create_gumbel_topk_from_config(**splitter_kwargs)

        # I145: 使用 transformer_max_level 构建模型
        # 如果有 transformer_max_level，则传入；否则设为 None（让模型从 tokenizer 获取）
        model_max_level = self.transformer_max_level if self.transformer_max_level is not None else None

        # 构建模型
        # 注意：K_min/K_max 现在是 FractalCurveViT 的计算属性，
        # 由 token_coverage_* 和 max_level 动态计算，不再需要显式传递
        model = FractalCurveViT(
            image_size=self.image_size,
            num_classes=self.num_classes,
            dim=self.dim,
            num_layers=self.num_layers,
            heads=self.heads,
            dim_head=dim_head,  # I145: 关键修复，确保与训练一致
            mlp_dim=self.mlp_dim,
            pool=self.pool,
            channels=self.channels,
            dropout=self.dropout,
            emb_dropout=self.emb_dropout,
            min_patch_size=self.min_patch_size,
            max_level=model_max_level,  # I145: 使用 transformer_max_level
            use_hilbert_encoding=self.use_hilbert_encoding,
            use_spatial_encoding=self.use_spatial_encoding,
            use_checkpoint=self.use_checkpoint,
            drop_path_rate=self.drop_path_rate,
            ffn_type=self.ffn_type,
            lca_temperature=self.lca_temperature,
            learnable_temperature=self.learnable_temperature,
            splitter=splitter,
            # I33: 传递覆盖率参数，K 值由模型内部计算
            token_coverage_min=self.token_coverage_min,
            token_coverage_max=self.token_coverage_max,
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
        verbose: bool = False,
    ) -> "ModelGene":
        """从模型实例提取架构基因

        Args:
            model: FractalCurveViT 模型实例
            dataset_name: 数据集名称
            epoch: 当前 epoch
            verbose: 是否打印详细信息
        """
        # 提取 dropout 值
        # 优先使用 model.dropout（如果它是 float），否则从 Dropout 模块提取
        dropout_attr = getattr(model, 'dropout', None)
        if dropout_attr is not None:
            if isinstance(dropout_attr, (int, float)):
                dropout_val = dropout_attr
            elif hasattr(dropout_attr, 'p'):  # PyTorch Dropout module
                dropout_val = dropout_attr.p
            else:
                dropout_val = getattr(model, 'emb_dropout', 0.1)
        else:
            dropout_val = getattr(model, 'emb_dropout', 0.1)

        # 提取基础参数
        gene = cls(
            dim=model.dim,
            num_layers=getattr(model, 'num_layers', model.depth if hasattr(model, 'depth') else 6),
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

            # I33: 从 SplitterConfig 提取覆盖率参数
            # 这是一级参数，用于在不同分辨率下正确复算 K 值
            # 优先从 model.splitter.config 提取（最可靠）
            if hasattr(model, 'splitter') and hasattr(model.splitter, 'config') and model.splitter.config is not None:
                config = model.splitter.config
                gene.token_coverage_min = getattr(config, 'token_coverage_min', 0.01)
                gene.token_coverage_max = getattr(config, 'token_coverage_max_hard', 0.05)
            # 备选：从 tokenizer.splitter.config 提取
            elif hasattr(tokenizer, 'splitter') and hasattr(tokenizer.splitter, 'config') and tokenizer.splitter.config is not None:
                config = tokenizer.splitter.config
                gene.token_coverage_min = getattr(config, 'token_coverage_min', 0.01)
                gene.token_coverage_max = getattr(config, 'token_coverage_max_hard', 0.05)
            else:
                raise ValueError(
                    "无法从模型提取 token_coverage_* 参数。"
                    " 请确保模型使用包含 SplitterConfig 的 GumbelTopKSplitter。"
                )

            # I145: 分离提取两个 max_level
            # tokenizer_max_level: 从 tokenizer 获取（动态计算的 Splitter 深度）
            if hasattr(tokenizer, 'max_level'):
                gene.tokenizer_max_level = tokenizer.max_level

            # transformer_max_level: 从 model 获取（传入 Transformer 的深度）
            gene.transformer_max_level = getattr(model, 'max_level', None)

        # 从 splitter 提取参数 (I140)
        if hasattr(model, 'splitter'):
            splitter = model.splitter
            gene.splitter_hidden_dim = cls._detect_splitter_param(splitter, 'hidden_dim')
            gene.splitter_pool_size = cls._detect_splitter_param(splitter, 'pool_size')
            gene.splitter_feature_dim = cls._detect_splitter_param(splitter, 'feature_dim')

        # I145: 从模型权重推断真实的架构配置（解决训练代码与权重不一致的问题）
        # 某些架构参数（如 heads）可能在权重中与模型属性不一致
        actual_heads = cls._detect_actual_heads_from_model(model)
        if actual_heads is not None:
            gene.heads = actual_heads
            if verbose:
                print(f"  [I145] 从模型权重检测到真实的 heads={actual_heads}")

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
        """验证 ModelGene 的有效性"""
        errors = []

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

        if errors:
            raise ValueError(f"Invalid ModelGene: {errors}")
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
