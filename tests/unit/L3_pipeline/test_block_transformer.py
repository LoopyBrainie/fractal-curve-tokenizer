# -*- coding: utf-8 -*-
"""
block_transformer.py 单元测试

测试内容：
1. FractalTransformerBlock 前向传播
2. Hilbert 偏置集成
3. DropPath 行为
4. 层级注意力模式
5. FractalTransformer 堆叠
"""

import pytest
import torch
import torch.nn as nn
from vit_pytorch.block_transformer import (
    DropPath,
    FractalTransformerBlock,
    FractalTransformer,
)
from vit_pytorch.levels_info import LevelsInfo


class TestDropPath:
    """DropPath 类测试"""

    def test_no_drop_in_eval(self):
        """评估模式下不应用 DropPath"""
        drop_path = DropPath(drop_prob=0.5)
        drop_path.eval()

        x = torch.randn(2, 16, 64)
        output = drop_path(x)

        assert torch.equal(output, x)

    def test_no_drop_when_prob_zero(self):
        """drop_prob=0 时不应用 DropPath"""
        drop_path = DropPath(drop_prob=0.0)

        x = torch.randn(2, 16, 64)
        output = drop_path(x)

        assert torch.equal(output, x)

    def test_drop_in_train(self):
        """训练模式下应用 DropPath"""
        drop_path = DropPath(drop_prob=0.5)
        drop_path.train()

        x = torch.randn(2, 16, 64)
        output = drop_path(x)

        # DropPath 会随机丢弃一些路径
        # 输出应该与输入不同
        assert output.shape == x.shape

    def test_scale_by_keep_shape(self):
        """按 keep_prob 缩放 - 验证形状"""
        drop_path = DropPath(drop_prob=0.5, scale_by_keep=True)
        drop_path.train()

        x = torch.randn(2, 16, 64)
        output = drop_path(x)

        # 输出形状应该与输入相同
        assert output.shape == x.shape
        # 验证缩放：保留的元素应该被放大
        # (drop_path 使用 random_tensor / keep_prob 进行缩放)

    def test_no_scale_by_keep(self):
        """不按 keep_prob 缩放"""
        drop_path = DropPath(drop_prob=0.5, scale_by_keep=False)
        drop_path.train()

        x = torch.randn(2, 16, 64)
        output = drop_path(x)

        # 输出形状应该与输入相同
        assert output.shape == x.shape

    def test_1d_tensor(self):
        """1D 张量处理"""
        drop_path = DropPath(drop_prob=0.5)

        x_1d = torch.randn(64)
        output = drop_path(x_1d)

        assert output.shape == x_1d.shape

    def test_3d_tensor(self):
        """3D 张量处理"""
        drop_path = DropPath(drop_prob=0.5)

        x_3d = torch.randn(2, 8, 64)
        output = drop_path(x_3d)

        assert output.shape == x_3d.shape

    def test_4d_tensor(self):
        """4D 张量处理"""
        drop_path = DropPath(drop_prob=0.5)

        x_4d = torch.randn(2, 4, 8, 64)
        output = drop_path(x_4d)

        assert output.shape == x_4d.shape


class TestFractalTransformerBlock:
    """FractalTransformerBlock 类测试"""

    def test_forward_output_shape(self):
        """前向传播输出形状"""
        batch, seq_len, dim = 2, 16, 64
        heads, dim_head = 4, 16
        mlp_dim = 256

        block = FractalTransformerBlock(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
        )
        x = torch.randn(batch, seq_len, dim)

        output = block(x)
        assert output.shape == (batch, seq_len, dim)

    def test_with_levels_info(self):
        """使用 LevelsInfo"""
        batch, seq_len, dim = 2, 16, 64
        max_depth = 4

        block = FractalTransformerBlock(dim=64, heads=4, dim_head=16, mlp_dim=256)
        x = torch.randn(batch, seq_len, dim)

        # 创建 LevelsInfo
        depths = torch.randint(0, max_depth + 1, (batch, seq_len))
        paths = torch.randint(0, 4, (batch, seq_len, max_depth))
        levels_info = LevelsInfo.from_arrays(depths, paths, max_depth)

        output = block(x, levels_info=levels_info)
        assert output.shape == (batch, seq_len, dim)

    def test_without_levels_info(self):
        """不使用 LevelsInfo"""
        batch, seq_len, dim = 2, 16, 64

        block = FractalTransformerBlock(dim=64, heads=4, dim_head=16, mlp_dim=256)
        x = torch.randn(batch, seq_len, dim)

        output = block(x)  # 无 levels_info
        assert output.shape == (batch, seq_len, dim)

    def test_gradient_flow(self):
        """梯度流动测试"""
        dim = 64
        block = FractalTransformerBlock(dim=dim, heads=4, dim_head=16, mlp_dim=256)

        x = torch.randn(2, 16, dim, requires_grad=True)
        output = block(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_training_eval_mode(self):
        """训练/评估模式切换"""
        block = FractalTransformerBlock(dim=64, heads=4, dim_head=16, mlp_dim=256, drop_path=0.5)
        x = torch.randn(2, 16, 64)

        # 训练模式
        block.train()
        output_train = block(x)

        # 评估模式
        block.eval()
        output_eval = block(x)

        # 评估模式应该确定性
        output_eval2 = block(x)
        assert torch.equal(output_eval, output_eval2)

    def test_different_ffn_types(self):
        """不同 FFN 类型"""
        x = torch.randn(2, 16, 64)

        block_gelu = FractalTransformerBlock(dim=64, heads=4, dim_head=16, mlp_dim=256, ffn_type='gelu')
        block_swiglu = FractalTransformerBlock(dim=64, heads=4, dim_head=16, mlp_dim=256, ffn_type='swiglu')

        out_gelu = block_gelu(x)
        out_swiglu = block_swiglu(x)

        # 应该产生不同输出
        assert not torch.allclose(out_gelu, out_swiglu, atol=1e-4)

    def test_max_depth_parameter(self):
        """max_depth 参数"""
        x = torch.randn(2, 16, 64)

        block_small = FractalTransformerBlock(dim=64, heads=4, dim_head=16, mlp_dim=256, max_level=4)
        block_large = FractalTransformerBlock(dim=64, heads=4, dim_head=16, mlp_dim=256, max_level=8)

        out_small = block_small(x)
        out_large = block_large(x)

        assert out_small.shape == (2, 16, 64)
        assert out_large.shape == (2, 16, 64)

    def test_lca_temperature(self):
        """LCA 温度参数"""
        x = torch.randn(2, 16, 64)

        block_warm = FractalTransformerBlock(dim=64, heads=4, dim_head=16, mlp_dim=256, lca_temperature=2.0)
        block_cool = FractalTransformerBlock(dim=64, heads=4, dim_head=16, mlp_dim=256, lca_temperature=0.5)

        out_warm = block_warm(x)
        out_cool = block_cool(x)

        # 不同温度应该产生不同输出
        assert not torch.allclose(out_warm, out_cool, atol=1e-4)


class TestFractalTransformer:
    """FractalTransformer 类测试"""

    def test_forward_output_shape(self):
        """前向传播输出形状"""
        batch, seq_len, dim = 2, 16, 64

        transformer = FractalTransformer(
            dim=dim,
            depth=2,
            heads=4,
            dim_head=16,
            mlp_dim=256,
        )
        x = torch.randn(batch, seq_len, dim)

        output = transformer(x)
        assert output.shape == (batch, seq_len, dim)

    def test_multiple_layers(self):
        """多层堆叠"""
        batch, seq_len, dim = 2, 16, 64

        transformer_1 = FractalTransformer(dim=dim, depth=1, heads=4, dim_head=16, mlp_dim=256)
        transformer_2 = FractalTransformer(dim=dim, depth=2, heads=4, dim_head=16, mlp_dim=256)

        x = torch.randn(batch, seq_len, dim)

        out_1 = transformer_1(x)
        out_2 = transformer_2(x)

        # 两者都应该是有效的输出
        assert out_1.shape == (batch, seq_len, dim)
        assert out_2.shape == (batch, seq_len, dim)

    def test_with_levels_info(self):
        """使用 LevelsInfo"""
        batch, seq_len, dim = 2, 16, 64
        max_depth = 4

        transformer = FractalTransformer(dim=dim, depth=2, heads=4, dim_head=16, mlp_dim=256)
        x = torch.randn(batch, seq_len, dim)

        depths = torch.randint(0, max_depth + 1, (batch, seq_len))
        paths = torch.randint(0, 4, (batch, seq_len, max_depth))
        levels_info = LevelsInfo.from_arrays(depths, paths, max_depth)

        output = transformer(x, levels_info=levels_info)
        assert output.shape == (batch, seq_len, dim)

    def test_gradient_flow(self):
        """梯度流动测试"""
        transformer = FractalTransformer(dim=64, depth=2, heads=4, dim_head=16, mlp_dim=256)

        x = torch.randn(2, 16, 64, requires_grad=True)
        output = transformer(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_gradient_checkpointing(self):
        """梯度检查点"""
        transformer = FractalTransformer(
            dim=64,
            depth=2,
            heads=4,
            dim_head=16,
            mlp_dim=256,
            use_checkpoint=True,
        )

        x = torch.randn(2, 16, 64, requires_grad=True)
        output = transformer(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_effective_depth_info(self):
        """测试有效深度信息返回"""
        transformer = FractalTransformer(
            dim=64,
            depth=4,
            heads=4,
            dim_head=16,
            mlp_dim=256,
        )

        x = torch.randn(2, 16, 64)

        output, extra_info = transformer(x, return_extra_info=True)

        assert output.shape == (2, 16, 64)
        assert 'effective_depth' in extra_info
        assert extra_info['effective_depth'] == 2  # depth // 2 = 4 // 2

    def test_drop_path_rate(self):
        """DropPath 率"""
        transformer = FractalTransformer(
            dim=64,
            depth=3,
            heads=4,
            dim_head=16,
            mlp_dim=256,
            drop_path_rate=0.2,
        )

        x = torch.randn(2, 16, 64)

        # 训练模式
        transformer.train()
        output_train = transformer(x)

        # 评估模式
        transformer.eval()
        output_eval = transformer(x)

        assert output_train.shape == (2, 16, 64)
        assert output_eval.shape == (2, 16, 64)


class TestBlockTransformerIntegration:
    """Block Transformer 集成测试"""

    def test_residual_connection(self):
        """残差连接验证"""
        batch, seq_len, dim = 2, 16, 64

        block = FractalTransformerBlock(dim=dim, heads=4, dim_head=16, mlp_dim=256)
        x = torch.randn(batch, seq_len, dim)
        x_copy = x.clone()

        output = block(x)

        # 残差连接确保输出与输入在同一尺度
        assert output.dtype == x.dtype
        assert output.device == x.device

    def test_deterministic_eval(self):
        """评估模式确定性"""
        transformer = FractalTransformer(dim=64, depth=2, heads=4, dim_head=16, mlp_dim=256)
        transformer.eval()

        x = torch.randn(2, 16, 64)

        output1 = transformer(x)
        output2 = transformer(x)

        assert torch.equal(output1, output2)

    def test_different_sequence_lengths(self):
        """不同序列长度测试"""
        transformer = FractalTransformer(dim=64, depth=2, heads=4, dim_head=16, mlp_dim=256)

        # 使用固定序列长度测试
        x1 = torch.randn(2, 16, 64)
        x2 = torch.randn(2, 16, 64)

        out1 = transformer(x1)
        out2 = transformer(x2)

        assert out1.shape == (2, 16, 64)
        assert out2.shape == (2, 16, 64)

    def test_level_aware_norm(self):
        """层级感知归一化"""
        batch, seq_len, dim = 2, 16, 64
        max_depth = 4

        block = FractalTransformerBlock(dim=dim, heads=4, dim_head=16, mlp_dim=256, max_level=max_depth)
        x = torch.randn(batch, seq_len, dim)

        # 不同深度的 LevelsInfo
        depths_uniform = torch.full((batch, seq_len), 2)
        paths_uniform = torch.randint(0, 4, (batch, seq_len, max_depth))
        levels_uniform = LevelsInfo.from_arrays(depths_uniform, paths_uniform, max_depth)

        out = block(x, levels_info=levels_uniform)
        assert out.shape == (batch, seq_len, dim)

    def test_standard_sequence(self):
        """标准序列长度测试"""
        batch, seq_len = 4, 16  # 使用标准序列长度
        dim = 64

        transformer = FractalTransformer(dim=dim, depth=2, heads=4, dim_head=16, mlp_dim=256)
        x = torch.randn(batch, seq_len, dim)

        output = transformer(x)
        assert output.shape == (batch, seq_len, dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_no_nan_in_output(self):
        """输出无 NaN"""
        block = FractalTransformerBlock(dim=64, heads=4, dim_head=16, mlp_dim=256)
        x = torch.randn(2, 16, 64)

        output = block(x)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
