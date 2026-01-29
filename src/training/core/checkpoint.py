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

    三层参数策略：
    - 第一层（架构参数）: 从 checkpoint 读取，构建模型
    - 第二层（动态参数）: 模型架构动态计算（max_level）
    - 第三层（超参数）: 从 checkpoint 读取，应用到模型

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

    # ========== 构建模型 ==========
    # 注意: max_level 是变参数，完全由模型架构内部计算，不从外部传入
    # 构建模型（只使用第一层架构参数，max_level 由模型架构处理）
    model = gene.build_model()

    # 移动到设备
    model = model.to(torch.device(device))

    # 加载权重
    if _state_dict:
        # 过滤掉非 Tensor 类型的键
        tensor_state = {k: v for k, v in _state_dict.items() if isinstance(v, torch.Tensor)}

        # 首先清理 torch.compile 产生的 _orig_mod. 前缀
        cleaned_state = {k.replace('_orig_mod.', ''): v for k, v in tensor_state.items()}

        # 尝试加载权重，处理形状不匹配的情况
        try:
            missing, unexpected = model.load_state_dict(cleaned_state, strict=strict)
        except RuntimeError:
            # 部分加载：只加载形状匹配的权重
            model_dict = model.state_dict()
            loaded_keys = []
            for name, param in cleaned_state.items():
                if name in model_dict and model_dict[name].shape == param.shape:
                    model_dict[name] = param
                    loaded_keys.append(name)
            model.load_state_dict(model_dict, strict=False)
            missing = [k for k in cleaned_state.keys() if k not in loaded_keys]
            unexpected = list(set(model_dict.keys()) - set(loaded_keys))

            loaded_count = len(loaded_keys)
            total_model_keys = len(model_dict)
            load_ratio = loaded_count / total_model_keys * 100 if total_model_keys > 0 else 0

            if verbose:
                print(f"  [I145] 部分加载完成: {loaded_count}/{total_model_keys} ({load_ratio:.1f}%)")
                if load_ratio < 80:
                    print(f"  [WARN] 仅加载了 {load_ratio:.1f}% 的权重")

        if missing and verbose:
            print(f"  [WARN] Missing keys: {len(missing)}")
            if len(missing) > 0:
                print(f"  [WARN] Missing key examples: {list(missing)[:5]}")
        if unexpected and verbose:
            print(f"  [WARN] Unexpected keys: {len(unexpected)}")
            if len(unexpected) > 0:
                print(f"  [WARN] Unexpected key examples: {list(unexpected)[:5]}")

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

    警告：此函数现在会直接报错，因为旧版格式不包含 token_coverage_* 字段。
    请使用新版训练脚本重新训练，新版 checkpoint 会保存覆盖率参数而非绝对 K 值。

    Args:
        checkpoint_path: checkpoint 文件路径
        device: 设备
        config_path: 可选的外部配置文件路径
        strict: 是否严格加载
        verbose: 是否打印详细信息

    Returns:
        (model, config) 元组

    Raises:
        ValueError: 旧版 checkpoint 不再支持加载
    """
    # I33: 强制迁移检查
    raise ValueError(
        "旧版 checkpoint 不再支持加载。"
        " 旧版格式使用 K_min/K_max 绝对值，无法在不同分辨率下正确复算。"
        " 请使用包含 token_coverage_* 字段的新版 checkpoint，"
        " 或使用新版训练脚本重新训练。"
    )


def _build_model_from_config(config: Dict[str, Any]) -> ModelType:
    """从配置字典构建模型

    强制迁移检查：配置必须包含 token_coverage_* 字段。
    旧版配置格式使用 K_min/K_max 绝对值，无法在不同分辨率下正确复算。

    Args:
        config: 配置字典

    Returns:
        构建的模型

    Raises:
        ValueError: 旧版配置格式不支持
    """
    from vit_pytorch import FractalCurveViT
    from vit_pytorch.gumbel_topk_splitter import create_gumbel_topk_from_config

    # I33: 强制迁移检查 - 必须有 token_coverage_* 字段
    if 'token_coverage_min' not in config or 'token_coverage_max' not in config:
        # 检查是否有旧的 K_min/K_max 字段（用于生成更友好的错误信息）
        has_old_format = 'K_min' in config or 'K_max' in config
        if has_old_format:
            raise ValueError(
                "旧版配置格式不支持：请使用包含 token_coverage_* 字段的新版配置。"
                " 旧版使用 K_min/K_max 绝对值，无法在不同分辨率下正确复算。"
                " 请使用新版训练脚本重新训练。"
            )
        else:
            raise ValueError(
                "配置中缺少必需字段 token_coverage_min 和 token_coverage_max。"
                " 请使用新版训练脚本重新训练。"
            )

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
    # I33: 从 coverage 复算 K 值
    token_coverage_min = config['token_coverage_min']
    token_coverage_max = config['token_coverage_max']
    max_patches = (image_size // min_patch_size) ** 2 if image_size else (224 // min_patch_size) ** 2
    K_min = max(4, int(max_patches * token_coverage_min))
    K_max = int(max_patches * token_coverage_max)
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

def _infer_model_config_from_state_dict(state_dict: Dict[str, Any], verbose: bool = False) -> Optional[Dict[str, Any]]:
    """I145: 从 state_dict 推断完整模型配置

    分析 state_dict 中的权重形状，推断完整的模型参数。
    处理 torch.compile 产生的 _orig_mod. 前缀。

    关键推断逻辑:
    - lca_embedding.weight shape = [max_depth+1, heads] → 直接得到 heads 和 max_depth
    - to_qkv.weight shape = [3 * dim_head * heads, dim] → 只用于推断 dim
    - depth_embedding.weight shape = [max_depth+1, dim] → dim（备用）
    - complexity_mlp.0.weight shape = [hidden_dim, feature_dim * pool_size^2] → hidden_dim, pool_size

    注意：无法从 to_qkv.weight 唯一确定 heads，因为:
    - to_qkv.shape = [3 * dim_head * heads, dim]
    - dim_head = dim // heads
    - 所以 to_qkv.shape = [3 * (dim // heads) * heads, dim]
    - 多个 (heads, dim_head) 组合可能产生相同 shape

    Returns:
        dict with keys: dim, depth, heads, mlp_dim, max_depth, hidden_dim, feature_dim, pool_size
    """
    import math

    result = {
        'dim': None,
        'depth': None,
        'heads': None,
        'mlp_dim': None,
        'max_depth': None,  # 用于 Transformer/Attention
        'tokenizer_max_depth': None,  # 用于 Tokenizer/Splitter
        'hidden_dim': None,
        'feature_dim': None,
        'pool_size': None,
    }

    # 清理 _orig_mod. 前缀
    cleaned_keys = {k.replace('_orig_mod.', ''): k for k in state_dict.keys()}

    # 1. 首先从 lca_embedding 推断真实的 heads（最可靠的方法）
    # lca_embedding.weight shape = [max_depth+1, heads] → 直接得到 heads
    # 注意：lca_embedding 第二个维度是 attention heads（与 lca_heads 不同）
    lca_heads_detected = None
    for key in cleaned_keys.keys():
        if 'lca_embedding' in key and 'weight' in key:
            orig_key = cleaned_keys[key]
            tensor = state_dict[orig_key]
            if len(tensor.shape) == 2:
                # lca_embedding.weight shape = [max_depth+1, heads]
                # 第二个维度直接就是 heads 数
                lca_heads_detected = tensor.shape[1]
                result['max_depth'] = tensor.shape[0] - 1  # Transformer 用的 max_depth
                result['heads'] = lca_heads_detected
                if verbose:
                    print(f"  [I145] 发现 lca_embedding: {key}, shape={tuple(tensor.shape)}")
                    print(f"  [I145]   推断: transformer_max_depth={result['max_depth']}, heads={result['heads']}")
                break

    # 2. 从 to_qkv 推断 dim（作为补充验证）
    # to_qkv.weight shape = [3 * dim_head * heads, dim] = [3 * (dim // heads) * heads, dim]
    # 由于无法从 to_qkv 唯一确定 heads，我们只推断 dim
    for key in cleaned_keys.keys():
        if '.attention.to_qkv.weight' in key:
            orig_key = cleaned_keys[key]
            tensor = state_dict[orig_key]
            if len(tensor.shape) == 2:
                # dim = to_qkv.shape[1]
                result['dim'] = tensor.shape[1]
                if verbose:
                    print(f"  [I145] 发现 to_qkv: {key}, shape={tuple(tensor.shape)}")
                    print(f"  [I145]   推断: dim={result['dim']}")
                # 如果之前没有检测到 heads，尝试从 to_qkv 推断
                if result['heads'] is None and lca_heads_detected is None:
                    # 使用常见的 heads 值进行验证
                    dim = result['dim']
                    for heads in [1, 2, 4, 6, 8, 12, 16]:
                        dim_head = dim // heads
                        expected_qkv_dim = 3 * dim_head * heads
                        if tensor.shape[0] == expected_qkv_dim:
                            result['heads'] = heads
                            if verbose:
                                print(f"  [I145]   从 to_qkv 推断 heads={heads}")
                            break
                break

    # 3. 从 splitter.threshold_offsets 推断 tokenizer 的 max_depth（最可靠）
    # threshold_offsets.shape = [max_depth] → tokenizer max_depth = shape[0]
    for key in cleaned_keys.keys():
        if 'splitter.threshold_offsets' in key:
            orig_key = cleaned_keys[key]
            tensor = state_dict[orig_key]
            if len(tensor.shape) == 1:
                result['tokenizer_max_depth'] = tensor.shape[0] - 1
                if verbose:
                    print(f"  [I145] 发现 splitter.threshold_offsets: {key}, shape={tuple(tensor.shape)}")
                    print(f"  [I145]   推断: tokenizer_max_depth={result['tokenizer_max_depth']}")
                break

    # 5. 从 complexity_mlp.0.weight 推断 hidden_dim, feature_dim, pool_size
    for key in cleaned_keys.keys():
        if 'complexity_mlp' in key and '.0.weight' in key:
            orig_key = cleaned_keys[key]
            tensor = state_dict[orig_key]
            if len(tensor.shape) == 2:
                # hidden_dim = out_features
                result['hidden_dim'] = tensor.shape[0]
                # in_features = feature_dim * pool_size^2
                in_features = tensor.shape[1]
                # 推断 feature_dim 和 pool_size
                if result.get('dim') is not None:
                    feature_dim = result['dim']
                    if in_features % feature_dim == 0:
                        pool_sq = in_features // feature_dim
                        pool_size = int(math.sqrt(pool_sq)) if pool_sq > 0 else 1
                        if pool_size * pool_size == pool_sq:
                            result['feature_dim'] = feature_dim
                            result['pool_size'] = pool_size
                if verbose := False:
                    print(f"  [I145] 发现 complexity_mlp.0: {key}, shape={tuple(tensor.shape)}")
                    print(f"  [I145]   推断: hidden_dim={result['hidden_dim']}, feature_dim={result['feature_dim']}, pool_size={result['pool_size']}")
            break

    # 6. 推断 depth（Transformer 层数）
    depth_count = 0
    for key in cleaned_keys.keys():
        if '.attention.to_qkv.weight' in key:
            depth_count += 1
    if depth_count > 0:
        result['depth'] = depth_count

    return result if any(v is not None for v in result.values()) else None


def _infer_splitter_config_from_state_dict(state_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """I145: 从 state_dict 推断 Splitter 配置

    分析 state_dict 中的权重形状，推断正确的 Splitter 参数。

    关键推断逻辑:
    - complexity_mlp.0.weight shape = [hidden_dim, feature_dim * pool_size^2]
    - 已知 feature_dim 通常等于 model.dim

    Returns:
        dict with keys: hidden_dim, feature_dim, pool_size (may be None if not found)
    """
    import math

    result = {
        'hidden_dim': None,
        'feature_dim': None,
        'pool_size': None,
    }

    # 1. 查找 complexity_mlp.0.weight 的形状，推断 hidden_dim 和 pool_size
    # shape = [hidden_dim, feature_dim * pool_size^2]
    for key in state_dict.keys():
        if 'complexity_mlp' in key and '.0.weight' in key:
            tensor = state_dict[key]
            if len(tensor.shape) == 2:
                # hidden_dim = out_features
                result['hidden_dim'] = tensor.shape[0]
                # 计算 feature_dim * pool_size^2 = in_features
                in_features = tensor.shape[1]
                # 常见配置: feature_dim ∈ {192, 256, 320, 384}
                # 推断 feature_dim 和 pool_size
                if result.get('hidden_dim') is not None:
                    # 尝试从 in_features 推断 pool_size
                    # in_features = feature_dim * pool_size^2
                    # 常见组合:
                    #   feature_dim=256, pool_size=4 → 256*16=4096
                    #   feature_dim=256, pool_size=2 → 256*4=1024
                    #   feature_dim=384, pool_size=4 → 384*16=6144
                    #   feature_dim=192, pool_size=4 → 192*16=3072
                    for feature_dim in [192, 256, 320, 384]:
                        if in_features % (feature_dim * feature_dim) == 0:
                            pool_sq = in_features // feature_dim
                            pool_size = int(math.sqrt(pool_sq))
                            if pool_size * pool_size == pool_sq:
                                result['feature_dim'] = feature_dim
                                result['pool_size'] = pool_size
                                break
                        elif in_features % feature_dim == 0:
                            pool_sq = in_features // feature_dim
                            pool_size = int(math.sqrt(pool_sq)) if pool_sq > 0 else 1
                            if pool_size * pool_size == pool_sq:
                                result['feature_dim'] = feature_dim
                                result['pool_size'] = pool_size
                                break
                if verbose := False:  # debug
                    print(f"  [I145] 发现 complexity_mlp.0: {key}, shape={tuple(tensor.shape)}")
                    print(f"  [I145]   推断: hidden_dim={result['hidden_dim']}, feature_dim={result['feature_dim']}, pool_size={result['pool_size']}")
            break

    # 2. 如果 pool_size 仍未知，尝试从其他层推断
    if result['pool_size'] is None:
        for key in state_dict.keys():
            if 'depth_proj' in key and 'weight' in key:
                tensor = state_dict[key]
                if len(tensor.shape) == 1:
                    # depth_proj.weight shape = [feature_dim]
                    result['feature_dim'] = tensor.shape[0]
                    break

    return result if any(v is not None for v in result.values()) else None


def _diagnose_splitter_config(state_dict: Dict[str, Any], gene: "ModelGene") -> None:
    """I145: 从 state_dict 推断 Splitter 配置，帮助诊断权重不匹配问题"""
    print(f"\n  [I145] 尝试从 state_dict 推断 Splitter 配置...")

    # 查找 splitter 相关的 key
    splitter_keys = [k for k in state_dict.keys() if 'splitter' in k.lower() or 'mlp' in k.lower()]

    if not splitter_keys:
        print(f"  [I145] 未找到 splitter 相关的 state_dict keys")
        return

    # 尝试从 MLP 层推断 hidden_dim
    for key in splitter_keys:
        if '.0.' in key and 'weight' in key:
            tensor = state_dict[key]
            if len(tensor.shape) == 2:
                # tensor.shape = [out_features, in_features]
                inferred_hidden = tensor.shape[0]
                inferred_in = tensor.shape[1]
                print(f"  [I145] 发现 MLP 层: {key}")
                print(f"  [I145]   推断 in_features={inferred_in}, out_features={inferred_hidden}")
                print(f"  [I145]   建议设置: splitter_hidden_dim={inferred_hidden}")

    # 查找 complexity_logits 相关层
    for key in state_dict.keys():
        if 'complexity' in key.lower() and 'weight' in key:
            tensor = state_dict[key]
            if len(tensor.shape) == 2:
                print(f"  [I145] 发现 complexity 层: {key}")
                print(f"  [I145]   shape={tuple(tensor.shape)}")
                print(f"  [I145]   建议设置: splitter_feature_dim={tensor.shape[1]}")

    print(f"\n  [I145] 提示: 使用命令行参数覆盖 Splitter 配置:")
    print(f"  [I145]   --splitter-hidden-dim <值> --splitter-feature-dim <值> --splitter-pool-size <值>")


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
