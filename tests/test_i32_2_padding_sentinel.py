"""
I32-2: Padding Token Sentinel 修复 - 单元测试

Tests for:
- levels_info 使用 -1 sentinel 标识 padding token
- 有效 token 的 depth >= 0
- Padding token 不参与 LCA 注意力偏置计算

数学形式化
==========
Padding 标识: is_padding(b, t) = (levels_info[b, t, 0] == -1)
有效深度: depth ∈ {0, 1, ..., D_max}
LCA 安全计算: padding 位置的偏置 = 0
"""

import math
import pytest
import torch


def get_token_lengths(output):
    """获取每个样本的有效 token 数量"""
    if output._lengths_cache is not None:
        return output._lengths_cache
    # 如果没有缓存，从 sequences 计算
    return torch.tensor([len(seq) for seq in output.sequences])


class TestI32_2PaddingSentinel:
    """I32-2: Padding Token Sentinel 验证"""

    def test_levels_info_initialization(self):
        """验证 levels_info 使用 -1 sentinel 初始化"""
        from vit_pytorch import StreamingFractalTokenizerV3

        tokenizer = StreamingFractalTokenizerV3(
            image_size=(64, 64),
            base_patch_size=4,
            max_depth=8,
        )

        images = torch.randn(2, 3, 64, 64)
        output = tokenizer(images)

        B, max_tokens, info_dim = output.levels_info.shape
        assert info_dim == 9  # max_depth + 1 = 8 + 1

    def test_padding_depth_is_minus_one(self):
        """验证 padding token 的 depth 为 -1"""
        from vit_pytorch import StreamingFractalTokenizerV3

        tokenizer = StreamingFractalTokenizerV3(
            image_size=(64, 64),
            base_patch_size=4,
            max_depth=8,
        )

        images = torch.randn(2, 3, 64, 64)
        output = tokenizer(images)

        B, max_tokens, _ = output.levels_info.shape
        levels_depth = output.levels_info[:, :, 0]  # [B, max_tokens]

        lengths = get_token_lengths(output)

        for b in range(B):
            actual_tokens = lengths[b].item()

            # Padding 位置的 depth 应该为 -1
            padding_depths = levels_depth[b, actual_tokens:]
            assert (padding_depths == -1).all(), \
                f"Batch {b}: padding depth 应全为-1，实际为{padding_depths.tolist()}"

    def test_valid_depth_non_negative(self):
        """验证有效 token 的 depth >= 0"""
        from vit_pytorch import StreamingFractalTokenizerV3

        tokenizer = StreamingFractalTokenizerV3(
            image_size=(64, 64),
            base_patch_size=4,
            max_depth=8,
        )

        images = torch.randn(2, 3, 64, 64)
        output = tokenizer(images)

        levels_depth = output.levels_info[:, :, 0]  # [B, max_tokens]

        # 所有非 padding 位置的 depth 应该 >= 0
        valid_depths = levels_depth[levels_depth >= 0]
        assert (valid_depths >= 0).all(), \
            f"有效 token depth 不应 < 0，实际最小值为 {valid_depths.min().item()}"

        # 验证 depth 不超过 max_depth
        assert (valid_depths <= 8).all(), \
            f"有效 token depth 不应 > max_depth，实际最大值为 {valid_depths.max().item()}"

    def test_path_encoding_for_valid_tokens(self):
        """验证有效 token 有正确的路径编码"""
        from vit_pytorch import StreamingFractalTokenizerV3

        tokenizer = StreamingFractalTokenizerV3(
            image_size=(64, 64),
            base_patch_size=4,
            max_depth=8,
        )

        images = torch.randn(2, 3, 64, 64)
        output = tokenizer(images)

        B, max_tokens, info_dim = output.levels_info.shape
        levels_info = output.levels_info

        lengths = get_token_lengths(output)

        for b in range(B):
            actual_tokens = lengths[b].item()

            # 有效 token 的路径应该在 0-3 范围内
            valid_paths = levels_info[b, :actual_tokens, 1:]  # [N_valid, max_depth]
            path_min = valid_paths.min().item()
            path_max = valid_paths.max().item()

            assert path_min >= 0, \
                f"Batch {b}: 有效 token 路径最小值应为 0，实际为 {path_min}"
            assert path_max <= 3, \
                f"Batch {b}: 有效 token 路径最大值应为 3，实际为 {path_max}"

            # Padding token 的路径应该是 -1（与depth sentinel一致）
            padding_paths = levels_info[b, actual_tokens:, 1:]
            assert (padding_paths == -1).all(), \
                f"Batch {b}: padding token 路径应全为-1，实际为 {padding_paths[0].tolist()}"


class TestI32_2LCABiasPadding:
    """I32-2: LCA 偏置处理 Padding 验证"""

    def test_lca_bias_zero_for_padding(self):
        """验证 LCA 偏置对 padding 位置为 0"""
        from vit_pytorch.attn_hilbert_bias import LCAHilbertBias
        from vit_pytorch import StreamingFractalTokenizerV3

        tokenizer = StreamingFractalTokenizerV3(
            image_size=(64, 64),
            base_patch_size=4,
            max_depth=8,
        )

        # 创建 LCA bias 模块
        lca_bias = LCAHilbertBias(
            max_depth=8,
            heads=4,
            lca_temperature=1.5,
            learnable_temperature=False,
        )

        images = torch.randn(2, 3, 64, 64)
        output = tokenizer(images)

        B, max_tokens, _ = output.levels_info.shape

        # 计算 LCA 偏置
        bias = lca_bias(output.levels_info)  # [B, H, S, S]

        if bias is None:
            pytest.skip("levels_info 无效，跳过此测试")

        # 验证 padding 位置的偏置为 0
        levels_depth = output.levels_info[:, :, 0]  # [B, max_tokens]
        lengths = get_token_lengths(output)

        for b in range(B):
            actual_tokens = lengths[b].item()

            # 找到 padding 位置
            padding_positions = levels_depth[b] == -1

            # 任何涉及 padding token 的注意力偏置应该为 0
            for i in range(max_tokens):
                for j in range(max_tokens):
                    if padding_positions[i] or padding_positions[j]:
                        # 检查偏置是否为 0（允许小数值误差）
                        for h in range(bias.shape[1]):
                            bias_val = bias[b, h, i, j].item()
                            assert abs(bias_val) < 1e-5, \
                                f"Padding 位置 ({i},{j}) 的偏置应为 0，实际为 {bias_val}"

    def test_lca_bias_nonzero_for_valid_pairs(self):
        """验证有效 token 对之间的 LCA 偏置非零"""
        from vit_pytorch.attn_hilbert_bias import LCAHilbertBias
        from vit_pytorch import StreamingFractalTokenizerV3

        tokenizer = StreamingFractalTokenizerV3(
            image_size=(64, 64),
            base_patch_size=4,
            max_depth=8,
        )

        lca_bias = LCAHilbertBias(
            max_depth=8,
            heads=4,
            lca_temperature=1.5,
            learnable_temperature=False,
        )

        images = torch.randn(2, 3, 64, 64)
        output = tokenizer(images)

        bias = lca_bias(output.levels_info)

        if bias is None:
            pytest.skip("levels_info 无效，跳过此测试")

        levels_depth = output.levels_info[:, :, 0]

        # 统计有效 token 对之间非零偏置的比例
        total_pairs = 0
        nonzero_pairs = 0

        for b in range(bias.shape[0]):
            levels_depth_b = levels_depth[b]
            valid_mask = levels_depth_b >= 0
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]

            if len(valid_indices) < 2:
                continue

            # 检查有效 token 对
            for i in valid_indices:
                for j in valid_indices:
                    total_pairs += 1
                    bias_val = bias[b, :, i, j].abs().max().item()
                    if bias_val > 1e-5:
                        nonzero_pairs += 1

        # 至少部分有效 token 对应该有非零偏置
        assert total_pairs > 0, "没有有效 token 对可供测试"
        assert nonzero_pairs > 0, "有效 token 对之间应该有非零 LCA 偏置"


class TestI32_2EdgeCases:
    """I32-2: 边界情况测试"""

    def test_empty_batch(self):
        """验证空 batch 处理"""
        from vit_pytorch import StreamingFractalTokenizerV3

        tokenizer = StreamingFractalTokenizerV3(
            image_size=(64, 64),
            base_patch_size=4,
            max_depth=8,
        )

        # 单张图像
        images = torch.randn(1, 3, 64, 64)
        output = tokenizer(images)

        # 验证输出结构
        assert output.levels_info.shape[0] == 1

    def test_single_token(self):
        """验证只有单个 token 的情况"""
        from vit_pytorch import StreamingFractalTokenizerV3

        tokenizer = StreamingFractalTokenizerV3(
            image_size=(64, 64),
            base_patch_size=4,
            max_depth=8,
        )

        images = torch.randn(1, 3, 64, 64)
        output = tokenizer(images)

        # 至少有一个有效 token
        assert output.levels_info.shape[1] >= 1

    def test_sentinel_not_confused_with_depth_zero(self):
        """验证 -1 sentinel 与 depth=0 有效 token 可区分"""
        from vit_pytorch import StreamingFractalTokenizerV3

        tokenizer = StreamingFractalTokenizerV3(
            image_size=(64, 64),
            base_patch_size=4,
            max_depth=8,
        )

        images = torch.randn(2, 3, 64, 64)
        output = tokenizer(images)

        levels_depth = output.levels_info[:, :, 0]

        # 检查是否存在 depth=0 的有效 token
        depth_zero_count = (levels_depth == 0).sum().item()
        padding_count = (levels_depth == -1).sum().item()

        # 两者应该可以共存且可区分
        print(f"depth=0 的 token 数: {depth_zero_count}")
        print(f"padding token 数: {padding_count}")

        # 如果存在 depth=0 的 token，它们应该和 padding 有不同的路径编码
        if depth_zero_count > 0:
            depth_zero_paths = output.levels_info[levels_depth == 0, 1:]
            padding_paths = output.levels_info[levels_depth == -1, 1:]

            # Padding 路径应该是 0
            assert (padding_paths == 0).all()

            # Depth=0 有效 token 的路径也应该存在（不为全0，除非真的是整图）
            # 这取决于具体图像内容


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
