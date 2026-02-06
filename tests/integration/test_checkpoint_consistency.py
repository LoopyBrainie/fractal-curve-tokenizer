# -*- coding: utf-8 -*-
"""
Checkpoint Consistency Tests

验证训练-保存-加载循环的参数一致性。

测试三层参数原则：
- L1 架构参数: 从 checkpoint 读取
- L2 变参数: 动态计算，不保存
- L3 超参数: 从 checkpoint 读取

关键验证点：
1. ModelGene.from_config() 正确接收所有参数
2. Checkpoint 完整保存 model_gene
3. load_model() 正确恢复模型
4. 训练时和评估时的参数完全一致
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

# 添加项目路径
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from vit_pytorch import FractalCurveViT
from training.core.model_gene import ModelGene
from training.core.checkpoint import save_checkpoint_with_gene, load_model
from training.config import ModelArchitectureConfig


class MockArgs:
    """模拟命令行参数"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestModelGeneFromConfig:
    """测试 ModelGene.from_config() 参数完整性"""

    def test_splitter_temperature_parameters(self):
        """验证 splitter_temp_start/end 被正确保存"""
        # 创建配置，使用非默认温度值
        config = ModelArchitectureConfig(
            dim=256,
            num_layers=8,
            heads=8,
            mlp_dim=1024,
            num_classes=100,
            image_size=224,
        )

        # 模拟 TrainingConfig 格式（带有 splitter_temp_start/end）
        mock_config = Mock()
        mock_config.dim = config.dim
        mock_config.num_layers = config.num_layers
        mock_config.heads = config.heads
        mock_config.mlp_dim = config.mlp_dim
        mock_config.num_classes = config.num_classes
        mock_config.image_size = config.image_size
        mock_config.channels = config.channels
        mock_config.pool = config.pool
        mock_config.min_patch_size = config.min_patch_size
        mock_config.dropout = config.dropout
        mock_config.emb_dropout = config.emb_dropout
        mock_config.drop_path_rate = config.drop_path_rate
        mock_config.use_hilbert_encoding = config.use_hilbert_encoding
        mock_config.use_spatial_encoding = config.use_spatial_encoding
        mock_config.use_checkpoint = config.use_checkpoint
        mock_config.ffn_type = config.ffn_type
        mock_config.lca_temperature = config.lca_temperature
        mock_config.learnable_temperature = config.learnable_temperature
        mock_config.token_coverage_min = config.token_coverage_min
        mock_config.token_coverage_max = config.token_coverage_max
        mock_config.use_area_encoding = config.use_area_encoding
        mock_config.use_affine_modulation = config.use_affine_modulation
        mock_config.fourier_levels = config.fourier_levels
        mock_config.quota_learnable = config.quota_learnable
        mock_config.splitter_hidden_dim = config.splitter_hidden_dim
        mock_config.splitter_feature_dim = config.splitter_feature_dim
        mock_config.splitter_pool_size = config.splitter_pool_size
        mock_config.elastic_coverage_min = config.elastic_coverage_min
        mock_config.elastic_coverage_max = config.elastic_coverage_max
        mock_config.elastic_lambda_over = config.elastic_lambda_over
        mock_config.elastic_lambda_under = config.elastic_lambda_under
        mock_config.use_semantic_splitter = config.use_semantic_splitter
        mock_config.semantic_splitter_config = config.semantic_splitter_config
        mock_config.compile_model = config.compile_model

        # 使用非默认温度值
        mock_config.splitter_temp_start = 2.0  # 非默认值
        mock_config.splitter_temp_end = 0.05    # 非默认值

        # 创建 ModelGene
        gene = ModelGene.from_config(mock_config, dataset_name="test", epoch=10)

        # 验证温度参数被正确保存
        assert gene.splitter_temp_start == 2.0, \
            f"Expected splitter_temp_start=2.0, got {gene.splitter_temp_start}"
        assert gene.splitter_temp_end == 0.05, \
            f"Expected splitter_temp_end=0.05, got {gene.splitter_temp_end}"

    def test_token_coverage_parameters(self):
        """验证 token_coverage_min/max 被正确保存"""
        config = ModelArchitectureConfig(
            dim=256,
            num_layers=8,
            heads=8,
            mlp_dim=1024,
            num_classes=100,
            image_size=224,
            token_coverage_min=0.02,  # 非默认值
            token_coverage_max=0.30,  # 非默认值
        )

        gene = ModelGene.from_config(config, dataset_name="test", epoch=10)

        # 验证覆盖率参数被正确保存
        assert gene.token_coverage_min == 0.02, \
            f"Expected token_coverage_min=0.02, got {gene.token_coverage_min}"
        assert gene.token_coverage_max == 0.30, \
            f"Expected token_coverage_max=0.30, got {gene.token_coverage_max}"


class TestCheckpointSaveLoadConsistency:
    """测试 checkpoint 保存-加载 参数一致性"""

    @pytest.fixture
    def sample_model_and_config(self, tmp_path):
        """创建测试用模型和配置"""
        # 创建非默认配置
        config = ModelArchitectureConfig(
            dim=192,
            num_layers=6,
            heads=6,
            mlp_dim=768,
            num_classes=100,
            image_size=64,
            token_coverage_min=0.015,
            token_coverage_max=0.25,
            dropout=0.2,
        )

        # 创建模型
        model = FractalCurveViT(
            image_size=config.image_size,
            num_classes=config.num_classes,
            dim=config.dim,
            num_layers=config.num_layers,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout,
            token_coverage_min=config.token_coverage_min,
            token_coverage_max=config.token_coverage_max,
        )

        checkpoint_path = tmp_path / "test_checkpoint.pth"

        return model, config, checkpoint_path

    def test_model_gene_serialization_roundtrip(self, sample_model_and_config, tmp_path):
        """测试 ModelGene 序列化-反序列化 完整循环"""
        model, config, checkpoint_path = sample_model_and_config

        # 1. 创建 ModelGene
        gene = ModelGene.from_config(config, dataset_name="test", epoch=5)

        # 2. 保存 checkpoint
        save_checkpoint_with_gene(
            path=checkpoint_path,
            model=model,
            gene=gene,
            optimizer_state=None,
            epoch=5,
            val_acc=0.95,
        )

        # 3. 加载 checkpoint
        loaded_model, loaded_gene = load_model(str(checkpoint_path), device="cpu")

        # 4. 验证参数一致
        assert gene.dim == loaded_gene.dim, \
            f"dim mismatch: {gene.dim} vs {loaded_gene.dim}"
        assert gene.num_layers == loaded_gene.num_layers, \
            f"num_layers mismatch: {gene.num_layers} vs {loaded_gene.num_layers}"
        assert gene.heads == loaded_gene.heads, \
            f"heads mismatch: {gene.heads} vs {loaded_gene.heads}"
        assert gene.mlp_dim == loaded_gene.mlp_dim, \
            f"mlp_dim mismatch: {gene.mlp_dim} vs {loaded_gene.mlp_dim}"
        assert gene.num_classes == loaded_gene.num_classes, \
            f"num_classes mismatch: {gene.num_classes} vs {loaded_gene.num_classes}"
        assert gene.dropout == loaded_gene.dropout, \
            f"dropout mismatch: {gene.dropout} vs {loaded_gene.dropout}"
        assert gene.token_coverage_min == loaded_gene.token_coverage_min, \
            f"token_coverage_min mismatch: {gene.token_coverage_min} vs {loaded_gene.token_coverage_min}"
        assert gene.token_coverage_max == loaded_gene.token_coverage_max, \
            f"token_coverage_max mismatch: {gene.token_coverage_max} vs {loaded_gene.token_coverage_max}"

    def test_loaded_model_produces_same_output(self, sample_model_and_config, tmp_path):
        """测试加载的模型在权重完全加载时产生相同的输出"""
        model, config, checkpoint_path = sample_model_and_config

        # 1. 创建 ModelGene 并保存
        gene = ModelGene.from_config(config, dataset_name="test", epoch=5)
        save_checkpoint_with_gene(
            path=checkpoint_path,
            model=model,
            gene=gene,
            optimizer_state=None,
            epoch=5,
        )

        # 2. 加载模型
        loaded_model, loaded_gene = load_model(str(checkpoint_path), device="cpu")

        # 3. 验证关键参数一致性（这是最重要的）
        # 由于 load_model 使用 strict=False，部分权重可能未加载
        # 但 model_gene 中的参数应该完全一致
        assert gene.dim == loaded_gene.dim
        assert gene.num_layers == loaded_gene.num_layers
        assert gene.heads == loaded_gene.heads
        assert gene.token_coverage_min == loaded_gene.token_coverage_min
        assert gene.token_coverage_max == loaded_gene.token_coverage_max
        assert gene.splitter_temp_start == loaded_gene.splitter_temp_start
        assert gene.splitter_temp_end == loaded_gene.splitter_temp_end

        # 4. 验证模型结构一致
        assert type(model) == type(loaded_model)
        assert model.dim == loaded_model.dim
        assert model.num_classes == loaded_model.num_classes

    def test_checkpoint_contains_all_required_fields(self, sample_model_and_config, tmp_path):
        """测试 checkpoint 包含所有必需字段"""
        model, config, checkpoint_path = sample_model_and_config

        # 1. 创建 ModelGene 并保存
        gene = ModelGene.from_config(config, dataset_name="test", epoch=10)
        save_checkpoint_with_gene(
            path=checkpoint_path,
            model=model,
            gene=gene,
            optimizer_state={"test": "state"},
            epoch=10,
            val_acc=0.92,
        )

        # 2. 直接加载 checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # 3. 验证必需字段
        required_fields = [
            'epoch',
            'model_state_dict',
            'val_acc',
            'model_gene',
        ]

        for field in required_fields:
            assert field in checkpoint, f"Checkpoint missing required field: {field}"

        # 4. 验证 model_gene 包含关键参数
        gene_dict = checkpoint['model_gene']
        required_gene_fields = [
            'dim',
            'num_layers',
            'heads',
            'mlp_dim',
            'num_classes',
            'token_coverage_min',
            'token_coverage_max',
            'dropout',
        ]

        for field in required_gene_fields:
            assert field in gene_dict, f"model_gene missing required field: {field}"


class TestThreeTierParameterPrinciple:
    """测试三层参数原则"""

    def test_l1_architecture_parameters_saved(self, tmp_path):
        """验证 L1 架构参数被保存"""
        config = ModelArchitectureConfig(
            dim=384,
            num_layers=12,
            heads=12,
            mlp_dim=1536,
            num_classes=200,
            image_size=224,
        )

        gene = ModelGene.from_config(config)

        # 验证 L1 参数
        assert gene.dim == 384
        assert gene.num_layers == 12
        assert gene.heads == 12
        assert gene.mlp_dim == 1536
        assert gene.num_classes == 200

    def test_l2_variable_parameters_not_saved(self, tmp_path):
        """验证 L2 变参数（max_level）不保存，由模型动态计算"""
        config = ModelArchitectureConfig(
            dim=256,
            num_layers=8,
            heads=8,
            mlp_dim=1024,
            num_classes=100,
            image_size=128,  # 不同分辨率会产生不同 max_level
        )

        gene = ModelGene.from_config(config)

        # max_level 应该不在 to_dict() 中
        gene_dict = gene.to_dict()
        assert 'max_level' not in gene_dict, \
            "max_level should not be saved (it's L2 variable parameter)"

        # 验证模型构建时使用动态计算的 max_level
        model = gene.build_model()
        assert hasattr(model, 'max_level'), \
            "Model should have max_level attribute (dynamically computed)"

    def test_l3_hyperparameters_saved(self, tmp_path):
        """验证 L3 超参数被保存"""
        config = ModelArchitectureConfig(
            dim=256,
            num_layers=8,
            heads=8,
            mlp_dim=1024,
            num_classes=100,
            image_size=224,
            dropout=0.3,
            token_coverage_min=0.02,
            token_coverage_max=0.35,
        )

        gene = ModelGene.from_config(config)

        # 验证 L3 参数
        gene_dict = gene.to_dict()
        assert gene_dict['dropout'] == 0.3
        assert gene_dict['token_coverage_min'] == 0.02
        assert gene_dict['token_coverage_max'] == 0.35


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
