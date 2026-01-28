# -*- coding: utf-8 -*-
"""
L4 Application Tests: 架构一致性 (I112)

测试内容:
- 模型 forward 返回 TrainingStats 类型
- TrainingStats 包含所有必需字段
- 训练器能正确处理模型输出
- TrainingStats.validate() 数学约束
- 评估模块与训练器推理逻辑一致性 (eval.py vs train_fractal_vit.py)

对应模块:
- vit_pytorch.model_fractal_vit
- training.core.inference_wrapper
- training.core.checkpoint
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path

from vit_pytorch import FractalCurveViT
from vit_pytorch.model_fractal_vit import TrainingStats
from training.core.inference_wrapper import (
    inference,
    evaluate,
    EvalResult,
    InferenceStats,
    extract_logits,
    wrap_stats,
)
from training.core.checkpoint import (
    save_checkpoint_with_gene,
    load_checkpoint,
    load_model,
)
from training.core.model_gene import ModelGene


class TestModelOutputInterface:
    """模型输出接口测试."""

    def test_model_returns_trainingstats_type(self):
        """验证模型 forward 返回 TrainingStats 类型."""
        model = FractalCurveViT(
            image_size=224,
            num_classes=1000,
            dim=256,
            depth=4,
            heads=4,
        )
        x = torch.randn(2, 3, 224, 224)

        output = model(x)

        assert isinstance(output, TrainingStats), \
            f"模型应返回 TrainingStats，实际返回 {type(output)}"

    def test_trainingstats_required_fields(self):
        """验证 TrainingStats 包含所有必需字段."""
        model = FractalCurveViT(
            image_size=224,
            num_classes=1000,
            dim=256,
            depth=4,
            heads=4,
        )
        x = torch.randn(2, 3, 224, 224)

        output = model(x)

        required_fields = [
            'logits',
            'num_tokens',
            'depth_used',
            'depth_distribution',
            'features',
            'transformer_tokens',
        ]
        for field in required_fields:
            assert hasattr(output, field), f"TrainingStats 缺少字段: {field}"
            assert getattr(output, field) is not None, f"字段 {field} 不能为 None"

    def test_trainingstats_optional_fields_exist(self):
        """验证 TrainingStats 可选字段存在且有默认值."""
        model = FractalCurveViT(
            image_size=224,
            num_classes=1000,
            dim=256,
            depth=4,
            heads=4,
        )
        x = torch.randn(2, 3, 224, 224)

        output = model(x)

        # 可选字段应有默认值
        assert hasattr(output, 'shared_features')
        assert hasattr(output, 'splitter_entropy')
        assert hasattr(output, 'temperature')
        assert hasattr(output, 'aux_infos')
        assert hasattr(output, 'ema_stats')
        assert hasattr(output, 'split_info')

    def test_trainingstats_tensor_shapes(self):
        """验证 TrainingStats Tensor 形状正确."""
        batch_size = 4
        num_classes = 100
        dim = 256
        model = FractalCurveViT(
            image_size=224,
            num_classes=num_classes,
            dim=dim,
            depth=4,
            heads=4,
        )
        x = torch.randn(batch_size, 3, 224, 224)

        output = model(x)

        assert output.logits.shape == (batch_size, num_classes)
        assert output.features.shape == (batch_size, dim)
        assert output.transformer_tokens.ndim == 3  # [B, N, dim]
        # I141: num_tokens 支持 int, List[int], 或 torch.Tensor
        assert isinstance(output.num_tokens, (list, torch.Tensor))
        if isinstance(output.num_tokens, torch.Tensor):
            assert output.num_tokens.shape == (batch_size,)
        else:
            assert len(output.num_tokens) == batch_size
        assert 0 <= output.depth_used <= 50


class TestTrainingStatsValidation:
    """TrainingStats 数学约束验证测试."""

    def test_validate_valid_stats(self):
        """验证有效统计能通过验证."""
        valid_stats = TrainingStats(
            logits=torch.randn(2, 1000),
            num_tokens=100,
            depth_used=4,
            depth_distribution={0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4},
            features=torch.randn(2, 512),
            transformer_tokens=torch.randn(2, 100, 512),
        )
        # 不应抛出异常
        valid_stats.validate()

    def test_validate_invalid_num_tokens(self):
        """验证无效 num_tokens 会被检测."""
        invalid_stats = TrainingStats(
            logits=torch.randn(2, 1000),
            num_tokens=5000,  # > 4096
            depth_used=4,
            depth_distribution={0: 1.0},
            features=torch.randn(2, 512),
            transformer_tokens=torch.randn(2, 100, 512),
        )
        with pytest.raises(AssertionError, match="Token 数异常"):
            invalid_stats.validate()

    def test_validate_invalid_depth(self):
        """验证无效 depth_used 会被检测."""
        invalid_stats = TrainingStats(
            logits=torch.randn(2, 1000),
            num_tokens=100,
            depth_used=60,  # > 50
            depth_distribution={0: 1.0},
            features=torch.randn(2, 512),
            transformer_tokens=torch.randn(2, 100, 512),
        )
        with pytest.raises(AssertionError, match="深度越界"):
            invalid_stats.validate()

    def test_validate_unnormalized_distribution(self):
        """验证未归一化分布会被检测."""
        invalid_stats = TrainingStats(
            logits=torch.randn(2, 1000),
            num_tokens=100,
            depth_used=2,
            depth_distribution={0: 0.5, 1: 0.3},  # 总和 0.8 != 1.0
            features=torch.randn(2, 512),
            transformer_tokens=torch.randn(2, 100, 512),
        )
        with pytest.raises(AssertionError, match="分布未归一化"):
            invalid_stats.validate()


class TestBackwardCompatibility:
    """向后兼容性测试."""

    def test_trainingstats_with_optional_fields(self):
        """验证 TrainingStats 可接受所有可选字段."""
        stats = TrainingStats(
            logits=torch.randn(2, 1000),
            num_tokens=100,
            depth_used=4,
            depth_distribution={0: 1.0},
            features=torch.randn(2, 512),
            transformer_tokens=torch.randn(2, 100, 512),
            shared_features=torch.randn(2, 128, 14, 14),
            splitter_entropy=0.5,
            temperature=0.8,
            aux_infos=[{'num_tokens': 100}],
            ema_stats=torch.randn(10),
            split_info={'region_0': [0, 1, 2]},
        )

        assert stats.shared_features is not None
        assert stats.splitter_entropy == 0.5
        assert stats.temperature == 0.8
        assert stats.aux_infos == [{'num_tokens': 100}]
        assert stats.ema_stats is not None
        assert stats.split_info == {'region_0': [0, 1, 2]}

    def test_trainingstats_default_values(self):
        """验证 TrainingStats 默认值正确."""
        stats = TrainingStats(
            logits=torch.randn(2, 1000),
            num_tokens=100,
            depth_used=4,
            depth_distribution={0: 1.0},
            features=torch.randn(2, 512),
            transformer_tokens=torch.randn(2, 100, 512),
        )

        assert stats.shared_features is None
        assert stats.splitter_entropy == 0.0
        assert stats.temperature == 1.0
        assert stats.aux_infos is None
        assert stats.ema_stats is None
        assert stats.split_info == {}


class TestInferenceWrapperConsistency:
    """推理包装器一致性测试 - 确保 eval.py 与 train_fractal_vit.py 使用相同推理逻辑."""

    def test_extract_logits_from_trainingstats(self):
        """验证 extract_logits 能从 TrainingStats 提取 logits."""
        model = FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=4,
            mlp_dim=128,
        )
        x = torch.randn(2, 3, 32, 32)

        output = model(x)
        logits = extract_logits(output)

        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (2, 10)

    def test_extract_logits_from_tuple(self):
        """验证 extract_logits 能从旧版 tuple 格式提取 logits (向后兼容)."""
        logits = torch.randn(2, 10)
        num_tokens = 16
        stats = (logits, num_tokens)

        extracted = extract_logits(stats)

        assert torch.equal(extracted, logits)

    def test_extract_logits_from_raw_tensor(self):
        """验证 extract_logits 能直接从 Tensor 提取 logits."""
        logits = torch.randn(2, 10)

        extracted = extract_logits(logits)

        assert torch.equal(extracted, logits)

    def test_wrap_stats_preserves_logits(self):
        """验证 wrap_stats 保留 logits 并包装为 InferenceStats."""
        model = FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=4,
            mlp_dim=128,
        )
        x = torch.randn(2, 3, 32, 32)

        output = model(x)
        logits = extract_logits(output)
        stats = wrap_stats(output, logits)

        assert isinstance(stats, InferenceStats)
        assert torch.equal(stats.logits, logits)
        assert stats.depth_used >= 0
        assert isinstance(stats.depth_distribution, dict)

    def test_inference_function_returns_correct_format(self):
        """验证 inference() 函数返回正确格式."""
        model = FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=4,
            mlp_dim=128,
        )
        x = torch.randn(2, 3, 32, 32)
        labels = torch.randint(0, 10, (2,))

        result = inference(model, x, labels, device=torch.device('cpu'))

        # 有标签时返回 (loss, logits, InferenceStats)
        assert isinstance(result, tuple)
        assert len(result) == 3
        loss, out_logits, stats = result
        assert isinstance(loss, torch.Tensor)
        assert isinstance(out_logits, torch.Tensor)
        assert isinstance(stats, InferenceStats)

    def test_inference_without_labels(self):
        """验证 inference() 无标签时返回 InferenceStats."""
        model = FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=4,
            mlp_dim=128,
        )
        x = torch.randn(2, 3, 32, 32)

        result = inference(model, x, device=torch.device('cpu'))

        assert isinstance(result, InferenceStats)


class TestEvalResultFormat:
    """EvalResult 格式测试 - 确保与训练器 evaluate() 返回格式兼容."""

    def test_eval_result_has_all_fields(self):
        """验证 EvalResult 包含所有必需字段."""
        result = EvalResult(
            accuracy=50.0,
            avg_loss=1.5,
            num_samples=100,
        )

        assert hasattr(result, 'accuracy')
        assert hasattr(result, 'avg_loss')
        assert hasattr(result, 'num_samples')
        assert hasattr(result, 'top5_accuracy')
        assert hasattr(result, 'per_class_accuracy')
        assert hasattr(result, 'ece')

    def test_eval_result_optional_fields(self):
        """验证 EvalResult 可选字段有默认值."""
        result = EvalResult(
            accuracy=50.0,
            avg_loss=1.5,
            num_samples=100,
        )

        assert result.top5_accuracy == 0.0
        assert result.per_class_accuracy is None
        assert result.ece == 0.0

    def test_eval_result_with_per_class_accuracy(self):
        """验证 EvalResult 支持逐类别准确率."""
        per_class = {i: 50.0 + i for i in range(10)}
        result = EvalResult(
            accuracy=55.0,
            avg_loss=1.2,
            num_samples=100,
            per_class_accuracy=per_class,
        )

        assert result.per_class_accuracy is not None
        assert len(result.per_class_accuracy) == 10


class TestCheckpointConsistency:
    """Checkpoint 一致性测试 - 确保训练器和评估器使用相同加载逻辑."""

    def test_save_and_load_checkpoint_with_gene(self):
        """验证 checkpoint 能正确保存和加载 ModelGene."""
        model = FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=4,
            mlp_dim=128,
        )

        gene = ModelGene.from_model(model, dataset_name='test', epoch=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / 'test_checkpoint.pth'

            # 保存 checkpoint
            save_checkpoint_with_gene(
                path=ckpt_path,
                model=model,
                gene=gene,
                optimizer_state=None,
                epoch=5,
                val_acc=75.0,
                val_loss=0.5,
            )

            # 加载 checkpoint
            loaded_ckpt = load_checkpoint(str(ckpt_path))

            assert 'model_gene' in loaded_ckpt
            assert loaded_ckpt['epoch'] == 5
            assert loaded_ckpt['val_acc'] == 75.0

            # 验证 ModelGene
            loaded_gene = ModelGene.from_dict(loaded_ckpt['model_gene'])
            assert loaded_gene.dim == 64
            assert loaded_gene.depth == 2
            assert loaded_gene.num_classes == 10

    def test_load_model_gene_preserves_architecture(self):
        """验证 ModelGene 能保留并重建模型架构配置."""
        model = FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=4,
            mlp_dim=128,
            dropout=0.1,
        )

        gene = ModelGene.from_model(model, dataset_name='test', epoch=10)

        # 验证 ModelGene 包含正确的架构参数
        assert gene.dim == 64
        assert gene.depth == 2
        assert gene.heads == 4
        assert gene.mlp_dim == 128
        assert gene.num_classes == 10
        assert gene.dataset_name == 'test'
        assert gene.checkpoint_epoch == 10

        # 验证从 ModelGene 能重建模型（不加载权重）
        rebuilt_model = gene.build_model()
        assert isinstance(rebuilt_model, torch.nn.Module)
        assert rebuilt_model.dim == 64
        assert rebuilt_model.depth == 2
        assert rebuilt_model.num_classes == 10

    def test_checkpoint_self_contained(self):
        """验证 checkpoint 是自包含的 (包含完整模型配置)."""
        # 创建包含完整配置的 ModelGene（包含所有 splitter 参数）
        gene = ModelGene(
            dim=64,
            depth=2,
            heads=4,
            mlp_dim=128,
            num_classes=10,
            image_size=32,
            channels=3,
            pool='weighted',
            dropout=0.2,
            max_depth=4,              # 必需：最大深度
            splitter_feature_dim=64,  # 必需：特征维度
            splitter_hidden_dim=64,   # 必需：隐藏层维度
            splitter_pool_size=4,     # 必需：池化大小
            dataset_name='test',
            checkpoint_epoch=3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / 'self_contained.pth'

            # 直接创建模型并保存
            model = gene.build_model()
            save_checkpoint_with_gene(
                path=ckpt_path,
                model=model,
                gene=gene,
                epoch=3,
            )

            # 加载 - 不需要外部 config.json
            loaded_ckpt = load_checkpoint(str(ckpt_path))

            assert 'model_gene' in loaded_ckpt
            loaded_gene = ModelGene.from_dict(loaded_ckpt['model_gene'])
            # 验证 ModelGene 被正确保存和加载
            assert loaded_gene.dim == 64
            assert loaded_gene.dropout == 0.2  # 验证 dropout 被保存
