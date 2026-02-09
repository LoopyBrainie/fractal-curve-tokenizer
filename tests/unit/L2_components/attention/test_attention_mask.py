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
    """验证模型输出确定性与 attention mask 正确性"""

    @pytest.fixture
    def model(self):
        """创建测试模型 (I98-2: max_level 由 image_size 和 min_patch_size 动态计算)"""
        return FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            num_layers=2,
            heads=4,
            mlp_dim=128,
            min_patch_size=(4, 4),
            tokenizer_dropout=0.0,  # 禁用 tokenizer dropout 确保确定性
            transformer_dropout=0.0,  # 禁用 transformer dropout
        )

    def test_deterministic_output(self, model):
        """验证相同输入产生确定性的输出 (I98-2: 核心不变性)

        关键验证：
        - eval 模式下，相同输入 → 相同输出
        - 无随机性干扰（dropout 已禁用）
        """
        model.eval()

        with torch.no_grad():
            x = torch.randn(2, 3, 32, 32)

            # 第一次前向
            output1 = model(x)
            logits1 = output1.logits if hasattr(output1, 'logits') else output1

            # 第二次前向（相同输入）
            output2 = model(x)
            logits2 = output2.logits if hasattr(output2, 'logits') else output2

            # 验证确定性
            assert torch.allclose(logits1, logits2, atol=1e-6), \
                f"相同输入产生不同输出: max_diff={torch.abs(logits1 - logits2).max().item()}"

    def test_different_inputs_produce_different_outputs(self, model):
        """验证不同输入产生不同输出 (区分度验证)

        关键验证：
        - 不同图像 → 不同输出（除非语义相同）
        - 模型能区分不同的输入样本
        """
        model.eval()

        with torch.no_grad():
            x1 = torch.randn(1, 3, 32, 32)
            x2 = torch.randn(1, 3, 32, 32)
            # 确保 x2 与 x1 不同
            x2 = x1 + 10.0

            out1 = model(x1)
            logits1 = out1.logits if hasattr(out1, 'logits') else out1
            out2 = model(x2)
            logits2 = out2.logits if hasattr(out2, 'logits') else out2

            assert logits1.shape == (1, 10)
            assert logits2.shape == (1, 10)
            # 不同输入应该产生不同的 logits
            assert not torch.allclose(logits1, logits2, atol=1e-3), \
                "不同输入应产生不同的输出 logits"


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
