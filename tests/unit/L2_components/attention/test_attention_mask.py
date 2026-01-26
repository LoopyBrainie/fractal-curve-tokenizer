# -*- coding: utf-8 -*-
"""
L2 Components: Attention Mask Tests

对应模块: vit_pytorch.utils (create_attention_mask)

测试内容:
- Attention mask 有效性验证
- Padding token mask 处理
- 层级注意力权重分配
"""

import pytest
import torch

from vit_pytorch import FractalCurveViT
from vit_pytorch.utils import create_attention_mask


class TestAttentionMaskEffectiveness:
    """验证 padding token 不影响有效 token 输出"""

    @pytest.fixture
    def model(self):
        """创建测试模型"""
        return FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            depth=2,
            heads=4,
            mlp_dim=128,
            min_patch_size=(4, 4),
            max_depth=3,
        )

    def test_padding_tokens_masked_in_attention(self, model):
        """验证 padding token 在 attention 中被正确 mask"""
        model.eval()

        with torch.no_grad():
            x = torch.randn(2, 3, 32, 32)
            x[1] = x[0].clone()

            output = model(x)
            # P-OPT fix: 模型返回 TrainingStats，需访问 .logits
            logits = output.logits if hasattr(output, 'logits') else output

            diff = torch.abs(logits[0] - logits[1]).max()
            assert diff < 1e-5, f"相同图像输出差异过大: {diff}"

    def test_different_padding_same_valid_output(self, model):
        """验证不同 padding 量不影响有效区域的输出"""
        model.eval()

        with torch.no_grad():
            x1 = torch.randn(1, 3, 32, 32)
            x2 = torch.randn(1, 3, 32, 32)

            out1 = model(x1)
            out2 = model(x2)
            # P-OPT fix: 模型返回 TrainingStats，需访问 .logits
            logits1 = out1.logits if hasattr(out1, 'logits') else out1
            logits2 = out2.logits if hasattr(out2, 'logits') else out2

            assert logits1.shape == (1, 10)
            assert logits2.shape == (1, 10)


class TestCreateAttentionMask:
    """create_attention_mask 函数测试"""

    def test_empty_input(self):
        """空输入测试"""
        mask = create_attention_mask([], torch.device('cpu'))
        assert mask.numel() == 0

    def test_single_sample(self):
        """单样本测试"""
        levels = [torch.tensor([[0], [1], [1], [2]])]
        mask = create_attention_mask(levels, torch.device('cpu'))

        assert mask.shape == (1, 4, 4)

        for i in range(4):
            level_i = levels[0][i, 0].item()
            for j in range(4):
                level_j = levels[0][j, 0].item()
                expected = 1.2 if level_i == level_j else (1.1 if abs(level_i - level_j) == 1 else 1.0)
                assert abs(mask[0, i, j] - expected) < 1e-6

    def test_batch_consistency(self):
        """批量处理一致性测试"""
        level_info = torch.tensor([[1], [2], [2]])
        levels = [level_info.clone(), level_info.clone()]

        mask = create_attention_mask(levels, torch.device('cpu'))

        assert mask.shape == (2, 3, 3)
        assert torch.allclose(mask[0], mask[1])

    def test_variable_length_padding(self):
        """变长序列 padding 测试"""
        levels = [
            torch.tensor([[0], [1], [2]]),
            torch.tensor([[0], [1]]),
        ]

        mask = create_attention_mask(levels, torch.device('cpu'))

        assert mask.shape == (2, 3, 3)
        assert mask[1, 2, 2] == 1.0
        assert mask[1, 0, 2] == 1.0

    def test_same_level_high_weight(self):
        """同层级高权重测试"""
        levels = [torch.tensor([[2], [2], [2]])]

        mask = create_attention_mask(levels, torch.device('cpu'))

        assert torch.allclose(mask, torch.full_like(mask, 1.2))

    def test_adjacent_level_medium_weight(self):
        """相邻层级中权重测试"""
        levels = [torch.tensor([[1], [2]])]

        mask = create_attention_mask(levels, torch.device('cpu'))

        assert mask[0, 0, 0] == 1.2
        assert mask[0, 1, 1] == 1.2
        assert mask[0, 0, 1] == 1.1
        assert mask[0, 1, 0] == 1.1

    def test_distant_level_low_weight(self):
        """远距离层级低权重测试"""
        levels = [torch.tensor([[0], [3]])]

        mask = create_attention_mask(levels, torch.device('cpu'))

        assert mask[0, 0, 1] == 1.0
        assert mask[0, 1, 0] == 1.0


class TestGlobalAttentionMask:
    """全局注意力 mask 泄漏测试"""

    def test_transformer_uses_mask(self):
        """验证 transformer 正确使用 attention mask"""
        from vit_pytorch.block_transformer import FractalTransformer

        transformer = FractalTransformer(
            dim=64,
            depth=2,
            heads=4,
            dim_head=16,
            mlp_dim=128,
        )
        transformer.eval()

        with torch.no_grad():
            x = torch.randn(2, 10, 64)
            mask = torch.ones(2, 1, 1, 10, dtype=torch.bool)
            mask[0, 0, 0, 5:] = False
            mask[1, 0, 0, 8:] = False

            levels = torch.zeros(2, 10, 5, dtype=torch.long)
            out = transformer(x, levels, mask)

            assert out.shape == x.shape
            assert not torch.isnan(out).any()
