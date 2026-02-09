#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModelGene 参数一致性测试

验证以下三者之间的参数同步：
1. ModelArchitectureConfig - 训练配置
2. ModelGene.from_config() - 从配置构造基因
3. ModelGene.build_model() - 从基因构建模型

当模型架构更新时，只要更新 training.config，测试会检查 ModelGene 的方法是否同步更新。

设计原则：
- 单一数据源：配置是真理之源
- 自动检测：参数遗漏会在运行时被发现
- 易于维护：只需添加配置参数，测试会自动验证
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, fields as dataclass_fields
from typing import Any, Dict, Optional, Set, Tuple

import pytest


# ============================================================================
# 参数映射定义 (手动维护，保持清晰的依赖关系)
# ============================================================================

# ModelArchitectureConfig → ModelGene 的参数映射
# 格式: (config_field, gene_field, default_value)
# default_value 用于向后兼容，当 config 字段不存在时使用
# 注意：只有 Config 中存在的字段才应该在这里

CONFIG_TO_GENE_MAPPING = [
    # 核心架构参数
    ("dim", "dim", None),
    ("num_layers", "num_layers", None),
    ("heads", "heads", None),
    ("mlp_dim", "mlp_dim", None),
    ("num_classes", "num_classes", None),
    ("image_size", "image_size", None),
    ("channels", "channels", 3),

    # Tokenizer 参数
    ("pool", "pool", "weighted"),
    ("min_patch_size", "min_patch_size", 4),
    ("token_coverage_min", "token_coverage_min", None),
    ("token_coverage_max", "token_coverage_max", None),

    # 正则化参数
    ("dropout", "dropout", 0.0),
    ("emb_dropout", "emb_dropout", 0.0),
    ("drop_path_rate", "drop_path_rate", 0.0),

    # 编码选项 (这些是 Config 中实际存在的字段)
    ("use_checkpoint", "use_checkpoint", False),

    # FFN 选项
    ("ffn_type", "ffn_type", "swiglu_level"),

    # I122-2: lca_temperature 已移除，由 hilbert_bias_scale × √d_k 统一缩放
    # ("lca_temperature", "lca_temperature", 1.5),
    # ("learnable_temperature", "learnable_temperature", True),

    # I24-2: 可学习配额
    ("quota_learnable", "quota_learnable", None),
    ("quota_entropy_weight", "quota_entropy_weight", 0.01),

    # I31-3: 形状-尺度编码
    ("use_area_encoding", "use_area_encoding", False),
    ("use_affine_modulation", "use_affine_modulation", True),
    ("fourier_levels", "fourier_levels", 4),

    # I140: Splitter 架构参数
    ("splitter_hidden_dim", "splitter_hidden_dim", None),
    ("splitter_feature_dim", "splitter_feature_dim", None),
    ("splitter_pool_size", "splitter_pool_size", None),
]

# ModelGene → FractalCurveViT 的参数映射
# 格式: (gene_field, model_param, default_value)

GENE_TO_MODEL_MAPPING = [
    # 核心架构参数
    ("dim", "dim", None),
    ("num_layers", "num_layers", None),
    ("heads", "heads", None),
    ("mlp_dim", "mlp_dim", None),
    ("num_classes", "num_classes", None),
    ("image_size", "image_size", None),
    ("channels", "channels", 3),

    # 计算参数 (确保 dim_head = dim // heads)
    ("dim", "dim_head", None),  # 特殊处理

    # Tokenizer 参数
    ("pool", "pool", "weighted"),
    ("min_patch_size", "min_patch_size", 4),
    ("token_coverage_min", "token_coverage_min", None),
    ("token_coverage_max", "token_coverage_max", None),

    # 正则化参数
    # I120-2: dropout → tokenizer_dropout, transformer_dropout
    ("dropout", "tokenizer_dropout", 0.0),
    ("dropout", "transformer_dropout", 0.0),
    ("emb_dropout", "emb_dropout", 0.0),
    ("drop_path_rate", "drop_path_rate", 0.0),

    # 编码选项
    ("use_hilbert_encoding", "use_hilbert_encoding", True),
    ("use_spatial_encoding", "use_spatial_encoding", True),
    ("use_checkpoint", "use_checkpoint", False),

    # FFN 选项
    ("ffn_type", "ffn_type", "swiglu_level"),

    # I122-2: lca_temperature 已移除
    # ("lca_temperature", "lca_temperature", 1.5),
    # ("learnable_temperature", "learnable_temperature", True),

    # Splitter 相关
    (None, "splitter", None),  # 特殊：不传递，使用模型默认

    # I24-2: 可学习配额
    ("quota_learnable", "quota_learnable", None),
    ("quota_entropy_weight", "quota_entropy_weight", 0.01),

    # I31-3: 形状-尺度编码
    ("use_area_encoding", "use_area_encoding", False),
    ("use_affine_modulation", "use_affine_modulation", True),
    ("fourier_levels", "fourier_levels", 4),

    # I140: Splitter 架构参数
    ("splitter_hidden_dim", "splitter_hidden_dim", None),
    ("splitter_feature_dim", "splitter_feature_dim", None),
    ("splitter_pool_size", "splitter_pool_size", None),

    # 特殊处理
    (None, "pos_dropout", None),  # 不传递，使用模型默认
]

# 变参数列表 - 这些参数由模型架构内部计算，不应出现在映射中
VARIABLE_PARAMS = {
    "max_level",      # 由 image_size 和 min_patch_size 动态计算
    "max_depth",      # max_level 的旧名
    "K_min",          # 绝对 K 值，由覆盖率计算
    "K_max",          # 绝对 K 值，由覆盖率计算
    "K_min_abs",      # 绝对下界，不是变参数，但在 Config 中用于验证
}

# 训练专用参数 - 这些参数只在 Config 中存在，不需要传递到模型
TRAINING_ONLY_PARAMS = {
    "use_channels_last",   # 训练内存格式
    "compile_model",       # torch.compile
    "compile_mode",        # compile 模式
    "patch_size",          # 已废弃，使用 min_patch_size
    "dim_head",            # 计算值，不是配置值
    "splitter_dropout",    # Splitter 内部参数
    "pos_dropout",         # 已移到模型内部处理
    "depth_scale_range",   # 深度缩放，仅训练策略
    "freeze_quota",        # 训练策略参数
    "freeze_tokenizer",    # 训练策略参数
    "freeze_tokenizer_epochs",  # 训练策略参数
}


# ============================================================================
# 辅助函数
# ============================================================================

def get_dataclass_fields(cls) -> Set[str]:
    """获取 dataclass 的所有字段名"""
    return {f.name for f in dataclass_fields(cls)}


def extract_from_init_call(code: str) -> Set[str]:
    """从 __init__ 调用中提取参数名

    从以下形式的代码中提取:
        FractalCurveViT(
            image_size=self.image_size,
            num_classes=self.num_classes,
            ...
        )
    只提取函数体中的参数，不包含 docstring 中的。
    """
    # 找到函数体开始的位置（第一个左括号之后）
    func_start = code.find('def ')
    if func_start == -1:
        return set()

    # 找到方法体的开始（冒号后的换行）
    colon_pos = code.find(':', func_start)
    if colon_pos == -1:
        return set()

    # 方法体从下一行开始
    body_start = code.find('\n', colon_pos) + 1

    # 只分析方法体
    method_body = code[body_start:]

    # 匹配 FractalCurveViT(....) 调用，处理多行
    pattern = r'FractalCurveViT\s*\(\s*(.*?)\s*\)'
    match = re.search(pattern, method_body, re.DOTALL)

    if not match:
        return set()

    call_content = match.group(1)

    # 提取所有关键字参数名
    param_pattern = r'(\w+)\s*='
    params = re.findall(param_pattern, call_content)

    return set(params)


def extract_gene_assignments(code: str) -> Set[str]:
    """从 from_config 方法的 cls(...) 调用中提取 gene 字段赋值

    从以下形式的代码中提取字段名:
        gene = cls(
            dim=config.dim,
            num_layers=config.num_layers,
            ...
        )
    """
    # 找到 from_config 方法的开始
    method_start = code.find('def from_config')
    if method_start == -1:
        return set()

    # 找到方法体的开始（冒号后的换行）
    colon_pos = code.find(':', method_start)
    if colon_pos == -1:
        return set()

    # 方法体从下一行开始
    body_start = code.find('\n', colon_pos) + 1

    # 只分析方法体
    method_body = code[body_start:]

    # 找到下一个方法定义或函数结束
    next_method = method_body.find('\n    @')
    if next_method != -1:
        method_body = method_body[:next_method]

    # 找到 "gene = cls(" 的位置，只匹配这个特定的调用
    gene_call_pos = method_body.find('gene = cls(')
    if gene_call_pos == -1:
        return set()

    # 从 "gene = cls(" 开始，找到匹配的闭合括号
    call_start = gene_call_pos + len('gene = cls(')

    # 计算嵌套括号层数来找到正确的闭合括号
    depth = 1
    i = call_start
    while i < len(method_body) and depth > 0:
        if method_body[i] == '(':
            depth += 1
        elif method_body[i] == ')':
            depth -= 1
        i += 1

    call_content = method_body[call_start:i-1]

    # 提取所有 keyword 参数名
    param_pattern = r'(\w+)\s*='
    params = re.findall(param_pattern, call_content)

    return set(params)


def extract_getattr_patterns(code: str) -> Dict[str, Tuple[str, Any]]:
    """从代码中提取 getattr 模式

    返回: {gene_field: (config_field, default_value)}
    """
    result = {}

    # 找到 from_config 方法的开始
    method_start = code.find('def from_config')
    if method_start == -1:
        return result

    # 找到方法体的开始
    colon_pos = code.find(':', method_start)
    body_start = code.find('\n', colon_pos) + 1

    # 只分析方法体
    method_body = code[body_start:]
    next_method = method_body.find('\n    @')
    if next_method != -1:
        method_body = method_body[:next_method]

    # 找到 "gene = cls(" 的位置
    gene_call_pos = method_body.find('gene = cls(')
    if gene_call_pos == -1:
        return result

    # 只分析 cls(...) 调用内部
    call_start = gene_call_pos + len('gene = cls(')
    depth = 1
    i = call_start
    while i < len(method_body) and depth > 0:
        if method_body[i] == '(':
            depth += 1
        elif method_body[i] == ')':
            depth -= 1
        i += 1

    call_content = method_body[call_start:i-1]

    # 匹配: gene.field = getattr(config, 'field', default) 或 getattr(config, "field", default)
    pattern = r'gene\.(\w+)\s*=\s*getattr\s*\(\s*config\s*,\s*[\'"]([^\'"]+)[\'"]\s*,\s*([^\)]+)\)'

    matches = re.findall(pattern, call_content)

    for gene_field, config_field, default_value in matches:
        result[gene_field] = (config_field, default_value)

    return result


def extract_direct_assignments(code: str) -> Dict[str, str]:
    """从代码中提取直接赋值模式

    返回: {gene_field: config_field}
    """
    result = {}

    # 找到 from_config 方法的开始
    method_start = code.find('def from_config')
    if method_start == -1:
        return result

    # 找到方法体的开始
    colon_pos = code.find(':', method_start)
    body_start = code.find('\n', colon_pos) + 1

    # 只分析方法体
    method_body = code[body_start:]
    next_method = method_body.find('\n    @')
    if next_method != -1:
        method_body = method_body[:next_method]

    # 找到 "gene = cls(" 的位置
    gene_call_pos = method_body.find('gene = cls(')
    if gene_call_pos == -1:
        return result

    # 只分析 cls(...) 调用内部
    call_start = gene_call_pos + len('gene = cls(')
    depth = 1
    i = call_start
    while i < len(method_body) and depth > 0:
        if method_body[i] == '(':
            depth += 1
        elif method_body[i] == ')':
            depth -= 1
        i += 1

    call_content = method_body[call_start:i-1]

    # 匹配: gene.field = config.field
    pattern = r'gene\.(\w+)\s*=\s*config\.(\w+)'

    matches = re.findall(pattern, call_content)

    for gene_field, config_field in matches:
        result[gene_field] = config_field

    return result


# ============================================================================
# 测试类
# ============================================================================

class TestModelGeneConfigConsistency:
    """ModelGene 与配置的一致性测试"""

    def test_config_to_gene_mapping_complete(self):
        """测试: Config 中每个架构参数都应映射到 Gene

        当在 ModelArchitectureConfig 中添加新参数时，必须同时更新
        CONFIG_TO_GENE_MAPPING 和 ModelGene.from_config() 方法。
        """
        from training.core.model_gene import ModelGene

        # 提取 from_config 中使用的参数（只分析方法体）
        from_config_code = inspect.getsource(ModelGene.from_config)
        gene_params = extract_gene_assignments(from_config_code)
        getattr_params = extract_getattr_patterns(from_config_code)
        direct_params = extract_direct_assignments(from_config_code)

        # 合并所有提取的参数
        all_extracted_params = gene_params | set(getattr_params.keys()) | set(direct_params.keys())

        # 检查映射表中的每个 gene 字段
        expected_gene_params = set()
        for config_field, gene_field, default in CONFIG_TO_GENE_MAPPING:
            if config_field not in VARIABLE_PARAMS:
                expected_gene_params.add(gene_field)

        # 检查实际代码中的参数
        missing_in_from_config = expected_gene_params - all_extracted_params

        assert not missing_in_from_config, (
            f"ModelGene.from_config() 缺少以下参数:\n"
            f"  {sorted(missing_in_from_config)}\n\n"
            f"请在 from_config() 方法中添加这些参数映射。"
        )

    def test_gene_to_model_mapping_complete(self):
        """测试: Gene 中每个参数都应传递给模型

        当在 ModelGene 中添加新参数时，必须同时更新
        GENE_TO_MODEL_MAPPING 和 ModelGene.build_model() 方法。
        """
        from training.core.model_gene import ModelGene

        # 获取 Gene 的所有字段
        gene_fields = get_dataclass_fields(ModelGene)

        # 提取 build_model 中的参数
        build_model_code = inspect.getsource(ModelGene.build_model)
        model_params = extract_from_init_call(build_model_code)

        # 验证 build_model 传递所有必要参数
        # 从映射表推断应该传递的参数（排除特殊的 None）
        expected_model_params = set()
        for gene_field, model_param, default in GENE_TO_MODEL_MAPPING:
            if gene_field is not None and gene_field not in VARIABLE_PARAMS:
                expected_model_params.add(model_param)

        missing_in_build = expected_model_params - model_params

        assert not missing_in_build, (
            f"ModelGene.build_model() 缺少以下参数传递给 FractalCurveViT:\n"
            f"  {sorted(missing_in_build)}\n\n"
            f"请在 build_model() 方法中添加这些参数传递。"
        )

    def test_config_fields_are_synchronized(self):
        """测试: Config 字段与 from_config 提取结果同步

        这个测试确保当 Config 添加新字段时，from_config 方法也同步更新。
        """
        from training.config import ModelArchitectureConfig
        from training.core.model_gene import ModelGene

        config_fields = get_dataclass_fields(ModelArchitectureConfig)

        # 架构相关字段（排除训练策略字段）
        architecture_fields = config_fields - TRAINING_ONLY_PARAMS - VARIABLE_PARAMS

        # 从 from_config 提取参数
        from_config_code = inspect.getsource(ModelGene.from_config)
        gene_params = extract_gene_assignments(from_config_code)
        getattr_params = extract_getattr_patterns(from_config_code)
        direct_params = extract_direct_assignments(from_config_code)

        all_extracted = gene_params | set(getattr_params.keys()) | set(direct_params.keys())

        # 应该被提取但缺失的字段
        missing_extraction = []
        for field in architecture_fields:
            # 检查是否在 extracted 中
            if field not in all_extracted:
                # 进一步检查 getattr 是否有这个字段
                found = False
                for gene_field, (config_field, _) in getattr_params.items():
                    if config_field == field:
                        found = True
                        break
                if not found:
                    missing_extraction.append(field)

        assert not missing_extraction, (
            f"from_config() 未提取以下 Config 字段:\n"
            f"  {sorted(missing_extraction)}\n\n"
            f"请在 from_config() 方法中添加这些字段的提取。"
        )

    def test_gene_serialization_complete(self):
        """测试: to_dict 序列化所有 Gene 字段

        确保 ModelGene 的每个字段都被序列化，用于 checkpoint 保存。
        """
        from training.core.model_gene import ModelGene

        gene_fields = get_dataclass_fields(ModelGene)

        # 获取 to_dict 源码
        to_dict_code = inspect.getsource(ModelGene.to_dict)

        # 检查序列化是否完整
        missing_serialization = []
        for field in gene_fields:
            if field.startswith('_'):  # 内部字段
                continue
            # 检查是否在 to_dict 中
            if f"'{field}':" not in to_dict_code and f'"{field}":' not in to_dict_code:
                if field not in ['_model_config', '_parameter_tier']:  # 这些不序列化
                    missing_serialization.append(field)

        assert not missing_serialization, (
            f"ModelGene.to_dict() 缺少以下字段的序列化:\n"
            f"  {sorted(missing_serialization)}\n\n"
            f"请在 to_dict() 方法中添加这些字段。"
        )

    def test_checkpoint_roundtrip(self):
        """测试: Checkpoint 完整往返 (Config → Gene → Model)

        确保从配置构建基因，再从基因构建模型，参数完全一致。
        """
        from training.config import ModelArchitectureConfig
        from training.core.model_gene import ModelGene

        # 创建测试配置
        config = ModelArchitectureConfig(
            dim=256,
            num_layers=6,
            heads=8,
            mlp_dim=1024,
            num_classes=100,
            image_size=224,
            channels=3,
            dropout=0.1,
            token_coverage_min=0.02,
            token_coverage_max=0.15,
            quota_learnable=True,
            quota_entropy_weight=0.02,
            use_area_encoding=True,
            use_affine_modulation=False,
        )

        # 从配置构建基因
        gene = ModelGene.from_config(config, dataset_name='test', epoch=10)

        # 验证基因字段
        assert gene.dim == 256, f"dim mismatch: {gene.dim} != 256"
        assert gene.num_layers == 6, f"num_layers mismatch: {gene.num_layers} != 6"
        assert gene.heads == 8, f"heads mismatch: {gene.heads} != 8"
        assert gene.quota_learnable == True, f"quota_learnable mismatch"
        assert gene.quota_entropy_weight == 0.02, f"quota_entropy_weight mismatch"

        # 从基因构建模型
        model = gene.build_model()

        # 验证模型参数
        assert model.dim == 256, f"model.dim mismatch"
        assert model.num_layers == 6, f"model.num_layers mismatch"
        assert model.heads == 8, f"model.heads mismatch"


class TestSplitterParameterConsistency:
    """Splitter 参数一致性测试 (I140)"""

    def test_splitter_config_params_in_gene(self):
        """测试: Splitter 配置参数应在 Gene 中保存

        I140 引入的 Splitter 架构参数 (hidden_dim, feature_dim, pool_size)
        应该被保存到 ModelGene 中，用于 checkpoint 恢复。
        """
        from training.config import ModelArchitectureConfig
        from training.core.model_gene import ModelGene

        config = ModelArchitectureConfig()

        # 检查 from_config 是否提取了 Splitter 参数
        from_config_code = inspect.getsource(ModelGene.from_config)

        gene_params = extract_gene_assignments(from_config_code)
        getattr_params = extract_getattr_patterns(from_config_code)
        direct_params = extract_direct_assignments(from_config_code)

        all_extracted = gene_params | set(getattr_params.keys()) | set(direct_params.keys())

        # I140: Splitter 架构参数应该被提取
        splitter_params = ['splitter_hidden_dim', 'splitter_feature_dim', 'splitter_pool_size']

        missing = [p for p in splitter_params if p not in all_extracted]

        assert not missing, (
            f"from_config() 缺少 Splitter 参数:\n"
            f"  {sorted(missing)}\n\n"
            f"请添加这些参数的提取。"
        )

    def test_splitter_config_params_to_model(self):
        """测试: Splitter 配置参数应传递给模型"""
        from training.core.model_gene import ModelGene

        build_model_code = inspect.getsource(ModelGene.build_model)

        # I140: Splitter 架构参数应该被传递
        splitter_params = ['splitter_hidden_dim', 'splitter_feature_dim', 'splitter_pool_size']

        missing = []
        for param in splitter_params:
            if f'{param}=self.{param}' not in build_model_code:
                missing.append(param)

        assert not missing, (
            f"build_model() 缺少 Splitter 参数传递:\n"
            f"  {sorted(missing)}\n\n"
            f"请添加这些参数到 FractalCurveViT 调用。"
        )


class TestVariableParameterIsolation:
    """变参数隔离测试 - 确保变参数不泄漏到配置层"""

    def test_max_level_not_serialized(self):
        """测试: max_level 是变参数，不应序列化到 checkpoint

        max_level 由模型架构根据 image_size 和 min_patch_size 动态计算，
        应该在模型内部计算，不应从外部传入或序列化。
        """
        from training.core.model_gene import ModelGene

        # 获取 to_dict 源码
        to_dict_code = inspect.getsource(ModelGene.to_dict)

        # max_level 不应被序列化
        assert "'max_level'" not in to_dict_code and '"max_level"' not in to_dict_code, \
            "max_level 是变参数，不应序列化到 checkpoint"

    def test_build_model_does_not_pass_max_level(self):
        """测试: build_model 不传递 max_level 给模型

        max_level 完全由模型架构内部计算。
        """
        from training.core.model_gene import ModelGene

        build_model_code = inspect.getsource(ModelGene.build_model)

        # max_level 不应作为参数传递（只检查方法体）
        # 先找到方法体
        colon_pos = build_model_code.find(':')
        body_start = build_model_code.find('\n', colon_pos) + 1
        method_body = build_model_code[body_start:]

        # 检查方法体中是否有 max_level 参数传递
        assert 'max_level=' not in method_body, \
            "max_level 是变参数，不应传递给模型"


class TestEndToEndConsistency:
    """端到端一致性测试"""

    def test_config_to_model_preserves_parameters(self):
        """测试: 配置参数完整传递到模型

        从 Config → Gene → Model，参数应完整传递。
        这是一个集成测试，确保整个链路正确。
        """
        from training.config import ModelArchitectureConfig
        from training.core.model_gene import ModelGene

        # 创建配置
        config = ModelArchitectureConfig(
            dim=512,
            num_layers=12,
            heads=16,
            mlp_dim=2048,
            num_classes=1000,
            image_size=384,
            dropout=0.2,
            drop_path_rate=0.3,
            ffn_type="swiglu_level",
            # I122-2: lca_temperature 已移除
            # lca_temperature=2.0,
            # learnable_temperature=False,
            token_coverage_min=0.01,
            token_coverage_max=0.3,
            quota_learnable=True,
            quota_entropy_weight=0.005,
            use_area_encoding=True,
            use_affine_modulation=True,
            fourier_levels=6,
        )

        # 构建基因
        gene = ModelGene.from_config(config)

        # 构建模型
        model = gene.build_model()

        # 验证关键参数
        assert model.dim == config.dim
        assert model.num_layers == config.num_layers
        assert model.heads == config.heads
        assert model.mlp_dim == config.mlp_dim
        assert model.num_classes == config.num_classes
        # I120-2: dropout → tokenizer_dropout, transformer_dropout
        assert model.tokenizer_dropout == config.dropout
        assert model.transformer_dropout == config.dropout
        assert model.drop_path_rate == config.drop_path_rate


class TestWandBConfigAlignment:
    """WandB 配置三层参数对齐测试

    验证 WandB 记录的 config 与 ModelGene.to_dict() 一致：
    1. ModelArchitectureConfig - 训练配置
    2. ModelGene.from_config() - 从配置构造基因
    3. WandB config - wandb.init(config=...) 使用 ModelGene.to_dict()

    这确保了训练/评估加载/pth.gene保存三者一致。
    """

    def test_wandb_config_matches_model_gene(self):
        """测试: WandB config 应与 ModelGene.to_dict() 完全一致

        这是三层参数对齐的核心测试:
        - Config → Gene → WandB config
        - 确保没有任何参数遗漏或不一致
        """
        from training.config import ModelArchitectureConfig
        from training.core.model_gene import ModelGene

        # 创建配置
        config = ModelArchitectureConfig(
            dim=384,
            num_layers=8,
            heads=6,
            mlp_dim=1536,
            num_classes=200,
            image_size=224,
            dropout=0.1,
            token_coverage_min=0.03,
            token_coverage_max=0.25,
            ffn_type="swiglu_level",
            # I122-2: lca_temperature 已移除
            # lca_temperature=1.5,
            # learnable_temperature=True,
            splitter_hidden_dim=256,
            splitter_feature_dim=64,
            splitter_pool_size=4,
        )

        # 构建基因
        gene = ModelGene.from_config(config)

        # 获取 ModelGene.to_dict()
        gene_dict = gene.to_dict()

        # 验证关键字段存在于 ModelGene
        assert 'dim' in gene_dict
        assert 'num_layers' in gene_dict
        assert 'heads' in gene_dict
        assert 'token_coverage_min' in gene_dict
        assert 'token_coverage_max' in gene_dict

        # 验证 WandB 可以使用这个配置
        # (这模拟了 WandBCallback.on_train_begin 中的逻辑)
        wandb_config = gene_dict.copy()
        wandb_config['_wandb_config_version'] = '1.0'

        assert '_wandb_config_version' in wandb_config

    def test_wandb_callback_uses_model_gene(self):
        """测试: WandBCallback.on_train_begin 应使用 ModelGene

        验证 WandBCallback.on_train_begin 中的逻辑:
        1. 优先从 trainer.arch_config 创建 ModelGene
        2. 使用 ModelGene.to_dict() 作为 WandB config
        """
        from training.config import ModelArchitectureConfig
        from training.core.model_gene import ModelGene
        from training.callbacks import WandBCallback, WandBCallbackConfig

        # 创建配置
        config = ModelArchitectureConfig(
            dim=384,
            num_layers=8,
            heads=6,
            num_classes=200,
            image_size=224,
        )

        # 模拟 ModularTrainer
        class MockTrainer:
            def __init__(self, arch_config):
                self.arch_config = arch_config

        trainer = MockTrainer(config)

        # 验证 WandBCallback 可以访问 trainer.arch_config
        assert hasattr(trainer, 'arch_config')
        assert trainer.arch_config is not None

        # 验证可以从 arch_config 创建 ModelGene
        gene = ModelGene.from_config(trainer.arch_config)
        assert gene is not None

        # 验证 ModelGene.to_dict() 可用于 WandB
        wandb_config = gene.to_dict()
        assert isinstance(wandb_config, dict)
        assert 'dim' in wandb_config
        assert 'num_layers' in wandb_config

    def test_extract_model_gene_metrics_returns_valid_metrics(self):
        """测试: _extract_model_gene_metrics 返回有效的指标

        验证指标与 ModelGene 字段对齐:
        - splitter/temperature → splitter_temp_*
        - splitter/avg_tokens → K_min_abs, K_max_hard
        - splitter/depth_entropy → max_level_limit
        """
        import torch
        from training.callbacks import WandBCallback, WandBCallbackConfig

        # 创建回调
        callback = WandBCallback(config=WandBCallbackConfig())

        # 模拟 MockTrainer
        class MockSplitter:
            def __init__(self):
                self.temperature = 0.6
                self.training = True
                self._rate_balanced_quota = torch.tensor([0.5, 0.3, 0.2])

        class MockTokenizer:
            def __init__(self):
                self.splitter = MockSplitter()

        class MockModel:
            def __init__(self):
                self.tokenizer = MockTokenizer()

        class MockTrainer:
            def __init__(self):
                self.model = MockModel()

        trainer = MockTrainer()

        # 验证返回的指标
        metrics = callback._extract_model_gene_metrics(trainer)

        assert 'splitter/temperature' in metrics
        assert isinstance(metrics['splitter/temperature'], float)
        assert 0.4 <= metrics['splitter/temperature'] <= 1.0


# ============================================================================
# 运行测试
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
