#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共享 Checkpoint 加载模块

统一训练器和评估器的模型加载逻辑，确保最大一致性。

功能
====
1. **load_checkpoint()** - 加载 checkpoint 文件，提取 model_gene 和 state_dict
2. **build_model()** - 从 checkpoint 构建模型（封装 ModelGene.build_model）
3. **load_model()** - 一键加载模型（加载 checkpoint + 构建模型 + 加载权重）

使用方式
=======
>>> from training.core.checkpoint import load_checkpoint, load_model
>>>
>>> # 方式1: 分离加载（需要更多控制时）
>>> ckpt = load_checkpoint("path/to/checkpoint.pth", device="cuda")
>>> model = build_model(ckpt['model_gene'])
>>> model.load_state_dict(ckpt['model_state_dict'], strict=False)
>>>
>>> # 方式2: 一键加载（推荐）
>>> model, gene = load_model(
...     "path/to/checkpoint.pth",
...     device="cuda",
...     strict=False,
... )
>>> print(f"Loaded {gene.dim}d-{gene.depth}l model, epoch {gene.checkpoint_epoch}")

Author: Claude
Date: 2026-01-27
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


# ============================================================================
# 类型定义
# ============================================================================

CheckpointType = Dict[str, Any]
ModelType = nn.Module


# ============================================================================
# 核心加载函数
# ============================================================================

def load_checkpoint(
    checkpoint_path: str,
    *,
    device: str = "cpu",
    weights_only: bool = False,
) -> CheckpointType:
    """
    加载 checkpoint 文件

    Args:
        checkpoint_path: checkpoint 文件路径
        device: 设备映射
        weights_only: 是否只加载权重（用于 torch.load 的 weights_only 参数）

    Returns:
        checkpoint 字典，包含:
        - model_state_dict: 模型权重
        - model_gene: 模型基因（如果存在）
        - epoch: 训练轮数
        - val_acc: 验证准确率
        - val_loss: 验证损失
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[CHECKPOINT] Loading: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=weights_only,
    )

    # 提取基本信息
    epoch = checkpoint.get('epoch', 0)
    val_acc = checkpoint.get('val_acc', 0.0)

    # 检查是否有 model_gene
    if 'model_gene' in checkpoint:
        print(f"  [OK] Found embedded model_gene (epoch={epoch}, val_acc={val_acc:.2f}%)")
    else:
        print(f"  [WARN] No model_gene found (legacy checkpoint)")

    return checkpoint


def build_model(
    gene_dict: Union[Dict[str, Any], "ModelGene"],
    *,
    device: Optional[torch.device] = None,
) -> Tuple[ModelType, "ModelGene"]:
    """
    从 ModelGene 构建模型

    Args:
        gene_dict: ModelGene 字典或 ModelGene 对象
        device: 可选设备

    Returns:
        (model, gene) 元组
    """
    from training.core.model_gene import ModelGene

    # 转换为 ModelGene 对象
    if isinstance(gene_dict, dict):
        gene = ModelGene.from_dict(gene_dict)
    else:
        gene = gene_dict

    # 构建模型
    model = gene.build_model()

    # 移动到设备
    if device is not None:
        model = model.to(device)

    return model, gene


def load_model(
    gene_dict_or_path: Union[str, Dict[str, Any]],
    *,
    state_dict: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    strict: bool = False,
    verbose: bool = True,
) -> Tuple[ModelType, "ModelGene"]:
    """
    一键加载模型（推荐方式）

    封装所有加载逻辑，确保训练器和评估器使用完全一致的代码。

    支持两种调用方式:
    1. 从文件加载: load_model("path/to/checkpoint.pth", device="cuda")
    2. 从已加载的数据加载: load_model(gene_dict, state_dict=state_dict, device="cuda")

    Args:
        gene_dict_or_path: ModelGene 字典或 checkpoint 文件路径
        state_dict: 模型权重字典（当 gene_dict_or_path 是路径时可选）
        device: 设备
        strict: 是否严格加载权重
        verbose: 是否打印详细信息

    Returns:
        (model, gene) 元组
    """
    from training.core.model_gene import ModelGene

    # 判断是路径还是字典
    if isinstance(gene_dict_or_path, str):
        # 方式1: 从文件加载
        checkpoint = load_checkpoint(gene_dict_or_path, device=device)

        # 检查 model_gene
        if 'model_gene' not in checkpoint:
            # 回退到旧版加载方式（从 state_dict 推断架构）
            if verbose:
                print(f"  [WARN] No model_gene found, using legacy inference from state_dict")
            model, legacy_info = load_model_legacy(
                gene_dict_or_path,
                device=device,
                strict=strict,
                verbose=verbose,
            )
            # 返回一个推断的 ModelGene
            inferred_gene = ModelGene(
                dim=legacy_info.get('dim', 256),
                depth=legacy_info.get('depth', 4),
                heads=legacy_info.get('heads', 4),
                mlp_dim=legacy_info.get('mlp_dim', 512),
                num_classes=legacy_info.get('num_classes', 10),
                image_size=legacy_info.get('image_size', 32),
                channels=legacy_info.get('channels', 3),
                dataset_name=legacy_info.get('dataset_name', 'unknown'),
                checkpoint_epoch=checkpoint.get('epoch', 0),
            )
            return model, inferred_gene

        gene_dict = checkpoint['model_gene']
        _state_dict = checkpoint.get('model_state_dict', checkpoint)
    else:
        # 方式2: 从已加载的数据加载
        gene_dict = gene_dict_or_path
        _state_dict = state_dict or {}

    # 转换为 ModelGene 对象
    gene = ModelGene.from_dict(gene_dict)

    if verbose:
        print(f"  ModelGene: {gene}")
        print(f"  Epoch: {gene.checkpoint_epoch}, Dataset: {gene.dataset_name}")

    # 构建模型
    model = gene.build_model()

    # 移动到设备
    model = model.to(torch.device(device))

    # 加载权重
    if _state_dict:
        # 过滤掉非 Tensor 类型的键
        tensor_state = {k: v for k, v in _state_dict.items() if isinstance(v, torch.Tensor)}
        missing, unexpected = model.load_state_dict(tensor_state, strict=strict)

        if missing and verbose:
            print(f"  [WARN] Missing keys: {len(missing)}")
        if unexpected and verbose:
            print(f"  [WARN] Unexpected keys: {len(unexpected)}")

    if verbose:
        print(f"  [OK] Model loaded successfully")

    return model, gene


# ============================================================================
# 旧 checkpoint 兼容（Legacy Support）
# ============================================================================

def load_model_legacy(
    checkpoint_path: str,
    *,
    device: str = "cpu",
    config_path: Optional[str] = None,
    strict: bool = False,
    verbose: bool = True,
) -> Tuple[ModelType, Dict[str, Any]]:
    """
    从旧版 checkpoint 加载模型（无 model_gene）

    Args:
        checkpoint_path: checkpoint 文件路径
        device: 设备
        config_path: 可选的外部配置文件路径
        strict: 是否严格加载
        verbose: 是否打印详细信息

    Returns:
        (model, config) 元组
    """
    from vit_pytorch import FractalCurveViT

    checkpoint = load_checkpoint(checkpoint_path, device=device)

    # 提取配置
    raw_config = None
    if 'config' in checkpoint:
        raw_config = checkpoint['config']
    elif 'model_config' in checkpoint:
        raw_config = checkpoint['model_config']
    elif config_path:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path) as f:
                raw_config = json.load(f)

    if raw_config is None:
        raise ValueError(
            f"Cannot find model config in checkpoint: {checkpoint_path}\n"
            f"Please provide config_path or retrain with updated script."
        )

    # 处理配置格式
    if hasattr(raw_config, '__dict__'):
        raw_config = vars(raw_config)
    elif hasattr(raw_config, '_asdict'):
        raw_config = raw_config._asdict()

    # 处理嵌套结构
    if 'model' in raw_config and isinstance(raw_config['model'], dict):
        config = raw_config['model']
    else:
        config = raw_config

    # 合并训练配置（如果有）
    if 'training' in raw_config and isinstance(raw_config['training'], dict):
        for key, value in raw_config['training'].items():
            if key not in config or config[key] is None:
                config[key] = value

    # 构建模型
    if verbose:
        print(f"  Building model from legacy config...")

    model = _build_model_from_config(config)

    # 加载权重
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    if isinstance(state_dict, dict):
        tensor_state = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    else:
        tensor_state = state_dict

    missing, unexpected = model.load_state_dict(tensor_state, strict=strict)

    if missing and verbose:
        print(f"  [WARN] Missing keys: {len(missing)}")
    if unexpected and verbose:
        print(f"  [WARN] Unexpected keys: {len(unexpected)}")

    if verbose:
        print(f"  [OK] Legacy model loaded")

    return model, config


def _build_model_from_config(config: Dict[str, Any]) -> ModelType:
    """从配置字典构建模型（兼容旧版配置格式）"""
    from vit_pytorch import FractalCurveViT
    from vit_pytorch.gumbel_topk_splitter import create_gumbel_topk_from_config

    # 提取配置值，设置合理的默认值
    dim = config.get('dim', 256)
    depth = config.get('depth', 8)
    heads = config.get('heads', 8)
    mlp_dim = config.get('mlp_dim', 512)
    num_classes = config.get('num_classes', 10)
    image_size = config.get('image_size', 32)

    # Tokenizer 配置
    pool = config.get('pool', 'weighted')
    min_patch_size = config.get('min_patch_size', 4)
    K_min = config.get('K_min', 16)
    K_max = config.get('K_max', 64)
    max_depth = config.get('max_depth', None)

    # Splitter 配置
    splitter_hidden_dim = config.get('splitter_hidden_dim', None)
    splitter_feature_dim = config.get('splitter_feature_dim', None)
    splitter_pool_size = config.get('splitter_pool_size', None)

    # 构建 Splitter
    splitter = create_gumbel_topk_from_config(
        feature_dim=splitter_feature_dim or dim,
        min_patch_size=min_patch_size,
        max_depth_limit=max_depth,
        hidden_dim=splitter_hidden_dim,
        pool_size=splitter_pool_size,
        K_min=K_min,
        K_max=K_max,
        image_size=(image_size, image_size) if image_size else None,
    )

    # 构建模型
    model = FractalCurveViT(
        image_size=image_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        pool=pool,
        channels=config.get('channels', 3),
        dropout=config.get('dropout', 0.1),
        emb_dropout=config.get('emb_dropout', 0.1),
        min_patch_size=min_patch_size,
        max_depth=None,  # 从 tokenizer 获取
        use_hilbert_encoding=config.get('use_hilbert_encoding', True),
        use_spatial_encoding=config.get('use_spatial_encoding', True),
        use_checkpoint=config.get('use_checkpoint', False),
        drop_path_rate=config.get('drop_path_rate', 0.15),
        ffn_type=config.get('ffn_type', 'swiglu_level'),
        lca_temperature=config.get('lca_temperature', 1.5),
        learnable_temperature=config.get('learnable_temperature', True),
        splitter=splitter,
        K_min=K_min,
        K_max=K_max,
        pos_dropout=None,
        use_area_encoding=config.get('use_area_encoding', False),
        use_affine_modulation=config.get('use_affine_modulation', True),
        fourier_levels=config.get('fourier_levels', 4),
        quota_learnable=config.get('quota_learnable', None),
    )

    return model


# ============================================================================
# 工具函数
# ============================================================================

def save_checkpoint_with_gene(
    path: Path,
    model: nn.Module,
    gene: "ModelGene",
    optimizer_state: Optional[Dict[str, Any]] = None,
    epoch: int = 0,
    val_acc: float = 0.0,
    val_loss: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    保存包含 model_gene 的 checkpoint

    Args:
        path: 保存路径
        model: 模型
        gene: ModelGene 对象
        optimizer_state: 优化器状态
        epoch: 当前轮数
        val_acc: 验证准确率
        val_loss: 验证损失
        extra: 额外信息
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_state,
        'val_acc': val_acc,
        'val_loss': val_loss,
        'model_gene': gene.to_dict(),
    }

    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, path)

    print(f"[CHECKPOINT] Saved: {path}")
    print(f"  Epoch: {epoch}, Val Acc: {val_acc:.2f}%, Gene: {gene}")


def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    获取 checkpoint 信息（不加载模型）

    Args:
        checkpoint_path: checkpoint 文件路径

    Returns:
        checkpoint 信息字典
    """
    checkpoint = load_checkpoint(checkpoint_path, weights_only=True)

    info = {
        'path': str(checkpoint_path),
        'epoch': checkpoint.get('epoch', 0),
        'val_acc': checkpoint.get('val_acc', 0.0),
        'val_loss': checkpoint.get('val_loss', 0.0),
        'has_gene': 'model_gene' in checkpoint,
    }

    if 'model_gene' in checkpoint:
        gene = ModelGene.from_dict(checkpoint['model_gene'])
        info.update({
            'dim': gene.dim,
            'depth': gene.depth,
            'heads': gene.heads,
            'num_classes': gene.num_classes,
            'dataset': gene.dataset_name,
        })

    return info


# 延迟导入 ModelGene（避免循环导入）
def __getattr__(name: str):
    if name == "ModelGene":
        from training.core.model_gene import ModelGene
        return ModelGene
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
