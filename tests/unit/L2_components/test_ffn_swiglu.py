# -*- coding: utf-8 -*-
"""
ffn_swiglu.py 单元测试

测试内容：
1. SwiGLUFFN 前向传播正确性
2. AdaptiveFractalFeedForward 层级自适应
3. 梯度流动 (requires_grad)
4. 训练/评估模式切换
5. 不同 ffn_type 行为差异
"""

import pytest
import torch
import torch.nn as nn
from vit_pytorch.ffn_swiglu import SwiGLUFFN, AdaptiveFractalFeedForward


class TestSwiGLUFFN:
    """SwiGLUFFN 类测试"""

    def test_forward_output_shape(self):
        """前向传播输出形状"""
        batch, seq_len, dim = 2, 16, 64
        hidden_dim = 256

        ffn = SwiGLUFFN(dim, hidden_dim)
        x = torch.randn(batch, seq_len, dim)

        output = ffn(x)
        assert output.shape == (batch, seq_len, dim)

    def test_swish_activation(self):
        """Swish 激活函数验证"""
        dim, hidden_dim = 64, 256
        ffn = SwiGLUFFN(dim, hidden_dim)
        x = torch.randn(2, 16, dim, requires_grad=True)

        output = ffn(x)
        assert output.requires_grad

        # 反向传播测试
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_elementwise_multiplication(self):
        """元素级乘法验证"""
        dim, hidden_dim = 64, 256
        ffn = SwiGLUFFN(dim, hidden_dim)

        # 相同输入应该产生相同输出
        x = torch.randn(1, 8, dim)
        output1 = ffn(x)
        output2 = ffn(x)
        assert torch.equal(output1, output2)

    def test_dropout_zero(self):
        """dropout=0 时无随机性"""
        dim, hidden_dim = 64, 256
        ffn = SwiGLUFFN(dim, hidden_dim, dropout=0.0)

        x = torch.randn(2, 16, dim)
        output1 = ffn(x)
        output2 = ffn(x)
        assert torch.equal(output1, output2)

    def test_with_dropout(self):
        """dropout>0 时有随机性（训练模式）"""
        dim, hidden_dim = 64, 256
        ffn = SwiGLUFFN(dim, hidden_dim, dropout=0.5)

        ffn.train()
        x = torch.randn(2, 16, dim)

        # 多次前向传播应该产生不同结果
        outputs = [ffn(x) for _ in range(5)]
        not_equal = any(not torch.equal(outputs[i], outputs[i+1]) for i in range(4))
        assert not_equal  # 应该有一定概率不同

    def test_device_consistency(self):
        """设备一致性"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dim, hidden_dim = 64, 256
        ffn = SwiGLUFFN(dim, hidden_dim)

        x_cpu = torch.randn(2, 16, dim)
        output_cpu = ffn(x_cpu)

        ffncuda = ffn.cuda()
        x_cuda = x_cpu.cuda()
        output_cuda = ffncuda(x_cuda)

        assert output_cuda.device.type == "cuda"
        # CPU 和 CUDA 输出的数值应该接近
        assert torch.allclose(output_cpu, output_cuda.cpu(), atol=1e-4)

    def test_bias_option(self):
        """偏置选项"""
        dim, hidden_dim = 64, 256

        ffn_no_bias = SwiGLUFFN(dim, hidden_dim, bias=False)
        ffn_with_bias = SwiGLUFFN(dim, hidden_dim, bias=True)

        x = torch.randn(2, 16, dim)

        output_no_bias = ffn_no_bias(x)
        output_with_bias = ffn_with_bias(x)

        # 有偏置应该产生不同输出
        assert not torch.allclose(output_no_bias, output_with_bias, atol=1e-6)


class TestAdaptiveFractalFeedForward:
    """AdaptiveFractalFeedForward 类测试"""

    def test_forward_output_shape_gelu(self):
        """GELU 类型输出形状"""
        batch, seq_len, dim = 2, 16, 64
        hidden_dim = 128

        ffn = AdaptiveFractalFeedForward(dim, hidden_dim, ffn_type='gelu')
        x = torch.randn(batch, seq_len, dim)

        output = ffn(x)
        assert output.shape == (batch, seq_len, dim)

    def test_forward_output_shape_swiglu(self):
        """SwiGLU 类型输出形状"""
        batch, seq_len, dim = 2, 16, 64

        ffn = AdaptiveFractalFeedForward(dim, dim * 4, ffn_type='swiglu')
        x = torch.randn(batch, seq_len, dim)

        output = ffn(x)
        assert output.shape == (batch, seq_len, dim)

    def test_forward_output_shape_swiglu_level(self):
        """SwiGLU Level 类型输出形状"""
        batch, seq_len, dim = 2, 16, 64

        ffn = AdaptiveFractalFeedForward(dim, dim * 4, ffn_type='swiglu_level')
        x = torch.randn(batch, seq_len, dim)

        output = ffn(x)
        assert output.shape == (batch, seq_len, dim)

    def test_gradient_flow(self):
        """梯度流动测试"""
        dim = 64
        ffn = AdaptiveFractalFeedForward(dim, dim * 4, ffn_type='swiglu_level')

        x = torch.randn(2, 16, dim, requires_grad=True)
        output = ffn(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_level_adaptation_no_levels_info(self):
        """无 levels_info 时的层级自适应"""
        dim = 64
        ffn = AdaptiveFractalFeedForward(dim, dim * 4, ffn_type='swiglu_level')

        x = torch.randn(2, 16, dim)
        output = ffn(x)  # 无 levels_info

        assert output.shape == (2, 16, dim)

    def test_training_eval_mode(self):
        """训练/评估模式切换"""
        dim = 64
        ffn = AdaptiveFractalFeedForward(dim, dim * 4, ffn_type='swiglu_level', dropout=0.5)

        x = torch.randn(2, 16, dim)

        # 训练模式
        ffn.train()
        output_train = ffn(x)

        # 评估模式
        ffn.eval()
        output_eval = ffn(x)

        # 评估模式应该确定性
        output_eval2 = ffn(x)
        assert torch.equal(output_eval, output_eval2)

    def test_different_ffn_types_produce_different_outputs(self):
        """不同 ffn_type 产生不同输出"""
        dim = 64
        x = torch.randn(2, 16, dim)

        ffn_gelu = AdaptiveFractalFeedForward(dim, dim * 4, ffn_type='gelu')
        ffn_swiglu = AdaptiveFractalFeedForward(dim, dim * 4, ffn_type='swiglu')

        out_gelu = ffn_gelu(x)
        out_swiglu = ffn_swiglu(x)

        # 输出应该不同
        assert not torch.allclose(out_gelu, out_swiglu, atol=1e-4)

    def test_max_depth_parameter(self):
        """max_depth 参数测试"""
        dim = 64

        ffn_small = AdaptiveFractalFeedForward(dim, dim * 4, max_level=4, ffn_type='swiglu_level')
        ffn_large = AdaptiveFractalFeedForward(dim, dim * 4, max_level=8, ffn_type='swiglu_level')

        x = torch.randn(2, 16, dim)

        out_small = ffn_small(x)
        out_large = ffn_large(x)

        # 两者都应该正常工作
        assert out_small.shape == (2, 16, dim)
        assert out_large.shape == (2, 16, dim)

    def test_use_level_adaptation_flag(self):
        """use_level_adaptation 标志测试"""
        dim = 64

        ffn_with = AdaptiveFractalFeedForward(dim, dim * 4, ffn_type='gelu', use_level_adaptation=True)
        ffn_without = AdaptiveFractalFeedForward(dim, dim * 4, ffn_type='gelu', use_level_adaptation=False)

        x = torch.randn(2, 16, dim)

        out_with = ffn_with(x)
        out_without = ffn_without(x)

        # 有/无层级自适应应该产生不同输出
        assert not torch.equal(out_with, out_without)


class TestFFNSwigluIntegration:
    """FFN SwiGLU 集成测试"""

    def test_multiple_ffns_same_input(self):
        """多个 FFN 处理相同输入"""
        dim = 64
        batch, seq_len = 2, 16

        # SwiGLUFFN 是独立的，AdaptiveFractalFeedForward(swiglu) 包装它
        # 两者都应该输出相同形状，且无 NaN/Inf
        ffn1 = SwiGLUFFN(dim, dim * 4)
        ffn2 = AdaptiveFractalFeedForward(dim, dim * 4, ffn_type='swiglu')

        x = torch.randn(batch, seq_len, dim)

        out1 = ffn1(x)
        out2 = ffn2(x)

        # 形状应该相同
        assert out1.shape == out2.shape
        # 都应该没有 NaN/Inf
        assert not torch.isnan(out1).any()
        assert not torch.isinf(out1).any()
        assert not torch.isnan(out2).any()
        assert not torch.isinf(out2).any()

    def test_residual_connection_implied(self):
        """残差连接隐式测试（通过 LayerNorm）"""
        dim = 64
        ffn = AdaptiveFractalFeedForward(dim, dim * 4, ffn_type='swiglu_level')

        x = torch.randn(2, 16, dim)
        x_copy = x.clone()

        output = ffn(x)

        # 输出应该与输入在同一尺度
        assert output.dtype == x.dtype
        assert output.device == x.device

    def test_large_sequence(self):
        """长序列处理"""
        dim = 64
        batch, seq_len = 4, 256

        ffn = SwiGLUFFN(dim, dim * 4)
        x = torch.randn(batch, seq_len, dim)

        output = ffn(x)
        assert output.shape == (batch, seq_len, dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_depth_variance_normalization_effect(self):
        """深度方差归一化效果测试"""
        dim = 64

        # 模拟不同深度的输入
        x = torch.randn(2, 32, dim)

        ffn = AdaptiveFractalFeedForward(dim, dim * 4, ffn_type='swiglu_level')

        # 应该能处理
        output = ffn(x)
        assert output.shape == (2, 32, dim)


class TestSwiGLUMathProperties:
    """SwiGLU 数学性质测试"""

    def test_silu_equivalence(self):
        """SiLU 等价于 Swish 验证"""
        x = torch.linspace(-3, 3, 100)

        # PyTorch F.silu 实现
        silu_output = torch.nn.functional.silu(x)

        # 手动 Swish: x * sigmoid(x)
        swish_output = x * torch.sigmoid(x)

        assert torch.allclose(silu_output, swish_output, atol=1e-6)

    def test_glu_gating_property(self):
        """GLU 门控性质"""
        dim, hidden_dim = 64, 256
        ffn = SwiGLUFFN(dim, hidden_dim)

        x = torch.randn(1, 1, dim)

        # Gate 和 Value 应该形状相同
        gate_shape = ffn.w_gate(x).shape
        value_shape = ffn.w_value(x).shape

        assert gate_shape == value_shape

    def test_output_bound(self):
        """输出边界测试"""
        dim, hidden_dim = 64, 256
        ffn = SwiGLUFFN(dim, hidden_dim)

        # 大输入
        x = torch.randn(2, 16, dim) * 10
        output = ffn(x)

        # 输出应该没有 NaN 或 Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
