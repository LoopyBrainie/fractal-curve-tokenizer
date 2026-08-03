# -*- coding: utf-8 -*-
"""
ShapeStabilizer 单元测试

验证形状稳定器的正确性：
1. 非线性桶映射
2. Padding 梯度为零
3. K > max_tokens 鲁棒性
4. torch.compile 兼容性
"""

import sys
sys.path.insert(0, 'src')

import torch
import pytest
from vit_pytorch.core.shape_stabilizer import ShapeStabilizer


class TestShapeStabilizer:

    def setup_method(self):
        """每个测试前初始化 ShapeStabilizer"""
        self.stabilizer = ShapeStabilizer(
            low_frequency_buckets=(128, 256, 512),
            high_frequency_buckets=(1024, 2048, 4096, 8192),
        )

    def test_nonlinear_bucket_mapping(self):
        """验证非线性桶的正确映射"""
        test_cases = [
            (50, 128),     # < 128 → 128
            (128, 128),    # = 128 → 128
            (200, 256),    # 128 < x ≤ 256 → 256
            (512, 512),    # = 512 → 512
            (600, 1024),   # 512 < x ≤ 1024 → 1024
            (1024, 1024),  # = 1024 → 1024
            (2048, 2048),  # = 2048 → 2048
            (8192, 8192),  # = 8192 → 8192
            (10000, 8192), # > max → 回退到 8192
        ]
        for k, expected in test_cases:
            result = self.stabilizer.get_nearest_bucket(k)
            assert result == expected, f"K={k}: expected {expected}, got {result}"

    def test_padding_gradient_zero(self):
        """验证 Padding 区域的梯度贡献为 0"""
        B, K_actual, D = 2, 100, 64
        tokens = torch.randn(B, K_actual, D, requires_grad=True)
        levels = torch.randint(0, 8, (B, K_actual))
        lengths = torch.tensor([K_actual, K_actual])

        tokens_padded, _, _ = self.stabilizer.pad_to_bucket(tokens, levels, lengths)

        # 模拟 Transformer 前向
        out = tokens_padded.sum(dim=1)  # [B, D]
        out.sum().backward()

        # 验证：Padding 区域的梯度必须为 0
        padding_grad = tokens.grad[:, K_actual:, :]
        assert torch.allclose(padding_grad, torch.zeros_like(padding_grad)), \
            f"Padding gradient must be zero, got {padding_grad.abs().sum()}"

    def test_robustness_k_exceeds_max(self):
        """验证 K > max_tokens 时的鲁棒性"""
        B, D = 2, 64
        # K = 9000 超过最大桶 8192
        tokens = torch.randn(B, 9000, D)
        levels = torch.randint(0, 8, (B, 9000))
        lengths = torch.tensor([9000, 8500])

        # 不会崩溃，返回最大桶 8192
        out, _, _ = self.stabilizer.pad_to_bucket(tokens, levels, lengths)
        assert out.shape[1] == 8192

    def test_masked_mean_pooling(self):
        """验证 Masked Mean Pooling 使用正确的分母

        注意：这是测试 masked mean 的数学公式，不是 ShapeStabilizer 的功能。
        """
        B, K, D = 2, 100, 64
        tokens = torch.randn(B, K, D)
        lengths = torch.tensor([80, 60])  # 不等长

        # 正确的 masked mean：只 sum 有效 token
        valid_mask = torch.arange(K, device=tokens.device).unsqueeze(0) < lengths.unsqueeze(1)
        # [B, K]: True for valid positions

        # Masked sum
        masked_sum = (tokens * valid_mask.unsqueeze(-1)).sum(dim=1)  # [B, D]
        lengths_expanded = lengths.unsqueeze(-1).float()  # [B, 1]
        mean_correct = masked_sum / lengths_expanded  # [B, D]

        # 验证
        expected_0 = tokens[0, :80].mean(dim=0)
        expected_1 = tokens[1, :60].mean(dim=0)

        assert torch.allclose(mean_correct[0], expected_0, atol=1e-6)
        assert torch.allclose(mean_correct[1], expected_1, atol=1e-6)

    def test_attention_mask_boolean_type(self):
        """验证 Attention Mask 使用布尔类型（避免 -inf 数值溢出）"""
        B = 2
        lengths = torch.tensor([100, 80])
        K = max(lengths.tolist())  # 128
        K_bucket = 256

        # 创建原始 mask [B, 1, K, K]
        original_mask = torch.ones(B, 1, K, K, dtype=torch.bool)

        # Padding 后应该是布尔类型
        padded_mask = self.stabilizer._pad_attention_mask(original_mask, lengths, K_bucket)

        assert padded_mask.dtype == torch.bool, f"Expected bool, got {padded_mask.dtype}"
        assert padded_mask.shape == (B, 1, K_bucket, K_bucket)
        # 验证 padding 区域为 True（基于 lengths）
        # batch 0: lengths[0]=100，所以位置 >= 100 是 padding
        # batch 1: lengths[1]=80，所以位置 >= 80 是 padding
        assert padded_mask[0, 0, 100, 100].item() == True  # batch 0 padding 位置
        assert padded_mask[1, 0, 80, 80].item() == True    # batch 1 padding 位置

    def test_rope_coordinates_padding(self):
        """验证 RoPE 坐标 Padding（避免 NaN）"""
        B, K = 2, 128
        D = 2  # D=2 表示 (x, y) 坐标
        coords = torch.randn(B, K, D)
        levels = torch.randint(0, 8, (B, K))
        lengths = torch.tensor([100, 80])  # 注意：lengths 不影响 _pad_coordinates 的 pad_len
        K_bucket = 256

        coords_padded, levels_padded = self.stabilizer._pad_coordinates(
            coords, levels, lengths, K_bucket
        )

        # 验证 padding 后坐标维度正确
        assert coords_padded.shape == (B, K_bucket, D)
        assert levels_padded.shape == (B, K_bucket)

        # pad_len = K_bucket - K = 256 - 128 = 128
        # padding 从索引 K=128 开始，到 K_bucket=256
        pad_start = K  # 128
        assert torch.allclose(coords_padded[0, pad_start:], torch.zeros(pad_start, 2, device=coords.device))
        assert torch.allclose(coords_padded[1, pad_start:], torch.zeros(pad_start, 2, device=coords.device))

        # 验证 padding 区域的深度为 -1
        assert torch.all(levels_padded[0, pad_start:] == -1)
        assert torch.all(levels_padded[1, pad_start:] == -1)

    @pytest.mark.skipif(sys.platform == "win32", reason="torch.compile inductor Unicode path issue on Windows")
    def test_torch_compile_compatibility(self):
        """验证 ShapeStabilizer 可以被 torch.compile 捕获"""
        @torch.compile
        def forward_compiled(tokens, levels, lengths):
            return self.stabilizer.pad_to_bucket(tokens, levels, lengths)

        tokens = torch.randn(2, 100, 64)
        levels = torch.randint(0, 8, (2, 100))
        lengths = torch.tensor([100, 80])

        # 应该不触发 .item() 错误
        out, _, _ = forward_compiled(tokens, levels, lengths)
        assert out.shape[1] == 256  # 最近的桶

        # 验证梯度仍然为 0
        out.sum().backward()
        padding_grad = tokens.grad[:, 100:, :]
        assert torch.allclose(padding_grad, torch.zeros_like(padding_grad))

    def test_no_padding_needed(self):
        """验证 K 恰好等于桶大小时不进行 padding（保留对象身份）"""
        B, K, D = 2, 256, 64
        tokens = torch.randn(B, K, D)
        levels = torch.randint(0, 8, (B, K))
        lengths = torch.tensor([256, 256])

        tokens_padded, levels_padded, lengths_returned = self.stabilizer.pad_to_bucket(
            tokens, levels, lengths
        )

        # 形状应该不变
        assert tokens_padded.shape == tokens.shape
        assert levels_padded.shape == levels.shape

        # 严格的 is 身份比较：Python 标量早返命中，无任何 tensor 分配
        assert tokens_padded is tokens
        assert levels_padded is levels
        assert lengths_returned is lengths

    def test_batch_different_lengths(self):
        """验证批次中不同样本长度的情况"""
        B, K, D = 3, 200, 64
        tokens = torch.randn(B, K, D)
        levels = torch.randint(0, 8, (B, K))
        lengths = torch.tensor([150, 180, 200])  # 不同长度

        tokens_padded, levels_padded, lengths_out = self.stabilizer.pad_to_bucket(
            tokens, levels, lengths
        )

        # 所有样本应该被 padding 到相同长度 (256)
        assert tokens_padded.shape[1] == 256
        assert levels_padded.shape[1] == 256

        # lengths 应该保持不变
        assert torch.equal(lengths_out, lengths)

    def test_levels_padding_value(self):
        """验证 levels padding 使用 -1 值

        注意：pad_to_bucket 对整个批次使用相同的 k_bucket（基于 max(lengths)），
        而不是每个样本单独计算。
        """
        B, K, D = 2, 100, 64
        tokens = torch.randn(B, K, D)
        levels = torch.ones(B, K, dtype=torch.long) * 5  # 所有深度为 5
        lengths = torch.tensor([100, 80])

        tokens_padded, levels_padded, _ = self.stabilizer.pad_to_bucket(tokens, levels, lengths)

        # max(lengths) = 100 -> nearest bucket = 128
        # pad_len = 128 - 100 = 28
        # 所以 padding 从索引 100 开始，到 128 结束
        pad_start = 100  # K
        assert levels_padded.shape == (B, 128)

        # 前 K=100 个 token 保持不变
        assert torch.all(levels_padded[0, :pad_start] == 5)
        assert torch.all(levels_padded[1, :pad_start] == 5)

        # padding 部分应该是 -1
        assert torch.all(levels_padded[0, pad_start:] == -1)
        assert torch.all(levels_padded[1, pad_start:] == -1)
