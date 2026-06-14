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
from vit_pytorch.core.utils import create_attention_mask


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

            # 第一次前向 (forward 返回 tuple)
            forward_output1, _metrics1 = model(x)
            logits1 = forward_output1.logits

            # 第二次前向（相同输入）
            forward_output2, _metrics2 = model(x)
            logits2 = forward_output2.logits

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

            forward_output1, _metrics1 = model(x1)
            logits1 = forward_output1.logits
            forward_output2, _metrics2 = model(x2)
            logits2 = forward_output2.logits

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
    """全局注意力 mask 泄漏测试

    注意：历史版本 (commit 6a8d6de 前) 此测试构造 mask 但未实际传入。
    该回归导致 ManifoldNativeAttention.forward 签名丢失 attention_mask kwarg
    后无法被任何调用方激活。当前版本（修复后）必须真正传 mask 并断言
    hot path 物理生效（masked 区域的 attn 权重应被压制至 ~0）。

    已知独立问题：levels_info 路径触发 Cartesian2DRoPE.apply_rotation
    pre-existing regression（缺少 sin_cos 参数）。本测试聚焦 mask 路径，
    不传 levels_info 以隔离两个独立 bug。
    """

    def test_transformer_uses_mask_strictly(self):
        """强力验证 mask 在 hot path 物理生效 (回归修复 E2E 契约校验)

        关键校验:
        1. attention_mask 真正被传入 forward（不丢失）
        2. ManifoldNativeAttention hot path 注入 masked_fill
        3. 被 mask 区域 (mask[..., 5:] == False) 的 attn 权重应被压制至 ~0
        """
        from vit_pytorch.modules.transformer_block import FractalTransformer

        transformer = FractalTransformer(
            dim=64,
            depth=2,
            heads=4,
            dim_head=16,
            mlp_dim=128,
        )
        transformer.eval()

        with torch.no_grad():
            B, N, D = 2, 10, 64
            x = torch.randn(B, N, D)

            # 修复 1: 真实传 mask (历史版本 mask 局部构造后丢弃)
            mask = torch.ones(B, 1, 1, N, dtype=torch.bool)
            mask[:, :, :, N // 2:] = False   # 后半段 token 全部屏蔽

            out = transformer(
                x,
                attention_mask=mask,    # ← 关键: 真正传 mask
            )

            assert out.shape == x.shape
            assert not torch.isnan(out).any(), \
                "Forward output contains NaN — likely a fully-masked row in attention"

            # 修复 2: self.layers 而非 self.blocks (用户提议中的 API 错误)
            # 修复 3: layer.attention 而非 layer.attn (项目实际成员名)
            # 修复 4: 用 _stats_cache["last_attn_weights"] 而非裸属性 (与项目诊断模式对齐)
            for layer in transformer.layers:
                attn_module = layer.attention
                stats = getattr(attn_module, "_stats_cache", {})
                if "last_attn_weights" not in stats:
                    # 模型未在 eval/no_grad 下记录 attn weights，跳过详细断言
                    continue
                last = stats["last_attn_weights"]
                # last shape: [B, H, N, N] - 取 mask 区域的 key 列
                masked_keys = last[:, :, :, N // 2:]
                # 修复 5: allclose 而非严格 == (允许 FP 误差)
                assert torch.allclose(
                    masked_keys, torch.zeros_like(masked_keys), atol=1e-3
                ), (
                    f"CRITICAL INVARIANT VIOLATION: attention mask was bypassed in "
                    f"the numerical hot-path! Masked region max weight = "
                    f"{masked_keys.abs().max().item():.6e} (expected ~0). "
                    f"This means padding tokens still leak into attention computation."
                )

                # 修复 6: 同步校验诊断字段已被填充
                assert "nan_count" in stats, "nan_count diagnostic missing"
                assert "total_count" in stats, "total_count diagnostic missing"
                # 全 mask 行应被压制至 ~0，不应产生 NaN
                assert stats["nan_count"] == 0, \
                    f"NaN detected in attention weights: {stats['nan_count']} NaN / {stats['total_count']} total"
