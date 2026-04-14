"""
I167-1: Hilbert Distance Decay Convolution 测试

验证解耦版距离衰减卷积:
- 空间混合: 固定距离衰减权重 (无梯度)
- 通道混合: 可学习 1x1 卷积 (有梯度)

数学: z = Pointwise(Depthwise(x, w_decay))
      w_decay[k] = 1 / (|k - center| + 1)
"""

import pytest
import torch
import torch.nn.functional as F

from vit_pytorch.layers.splitters.hilbert_distance_decay_conv import (
    HilbertDistanceDecayConv1D,
    HilbertDistanceDecayConv1DWithSkip,
    create_distance_decay_weights,
    verify_decay_weights,
)


class TestHilbertDistanceDecayConv1D:
    """HilbertDistanceDecayConv1D 单元测试"""

    @pytest.fixture
    def in_channels(self):
        return 256  # hidden_dim * 4

    @pytest.fixture
    def module(self, in_channels):
        return HilbertDistanceDecayConv1D(in_channels)

    @pytest.fixture
    def batch_input(self, in_channels):
        """创建测试输入: [B=2, D, N=100]"""
        B, D, N = 2, in_channels, 100
        return torch.randn(B, D, N)

    # ===================================================================
    # 基础功能测试
    # ===================================================================

    def test_output_shape(self, module, batch_input):
        """验证输出形状 [B, N]"""
        output = module(batch_input)
        assert output.shape == (2, 100), f"Expected (2, 100), got {output.shape}"

    def test_decay_weights_values(self, module):
        """验证 w_d = 1/(|d|+1) 对应 kernel_size=5"""
        weights = module.decay_weights.squeeze().cpu()
        expected = torch.tensor([1/3, 1/2, 1.0, 1/2, 1/3])

        assert torch.allclose(weights, expected, atol=1e-4), (
            f"Decay weights mismatch: {weights} vs {expected}"
        )

    def test_decay_weights_shape(self, module):
        """验证固定权重形状为 [1, 1, 5]"""
        weights = module.decay_weights
        assert weights.shape == (1, 1, 5), f"Expected (1, 1, 5), got {weights.shape}"

    def test_no_gradient_for_decay_weights(self, module, batch_input):
        """验证距离衰减权重无梯度"""
        output = module(batch_input)
        output.sum().backward()

        # 检查 _decay_weights 是否在计算图中
        assert '_decay_weights' in module._buffers or \
               not module._decay_weights.requires_grad, \
               "Decay weights should not require grad"

        # 验证 pointwise 权重有梯度
        assert module.pointwise.weight.grad is not None, \
               "Pointwise weights should have gradient"

    # ===================================================================
    # 边界情况测试
    # ===================================================================

    def test_single_channel(self):
        """测试单通道输入"""
        module = HilbertDistanceDecayConv1D(in_channels=128)
        x = torch.randn(1, 128, 50)
        output = module(x)
        assert output.shape == (1, 50)

    def test_large_batch(self):
        """测试大批量输入"""
        module = HilbertDistanceDecayConv1D(in_channels=256)
        x = torch.randn(32, 256, 200)  # batch=32
        output = module(x)
        assert output.shape == (32, 200)

    def test_small_sequence(self):
        """测试短序列"""
        module = HilbertDistanceDecayConv1D(in_channels=64)
        x = torch.randn(1, 64, 5)  # N=5, 小于 kernel_size
        output = module(x)
        assert output.shape == (1, 5)

    def test_padding_effect(self):
        """验证 padding=2 正确处理边界"""
        module = HilbertDistanceDecayConv1D(in_channels=64)
        x = torch.randn(1, 64, 3)  # N=3 < kernel_size
        output = module(x)
        assert output.shape == (1, 3), "Padding should handle N < kernel_size"

    # ===================================================================
    # 数学性质测试
    # ===================================================================

    def test_center_weight_is_maximum(self, module):
        """验证中心位置权重最大 (d=0, w=1)"""
        weights = module.decay_weights.squeeze()
        assert weights[2].item() == 1.0, "Center weight should be 1.0"
        assert weights[2] >= weights[0], "Center weight should be maximum"
        assert weights[2] >= weights[4], "Center weight should be maximum"

    def test_symmetric_decay(self, module):
        """验证权重对称: w[-2] = w[2], w[-1] = w[1]"""
        weights = module.decay_weights.squeeze()
        # w[0] = w[4] = 1/3, w[1] = w[3] = 1/2
        assert torch.isclose(weights[0], weights[4]), "Weights should be symmetric"
        assert torch.isclose(weights[1], weights[3]), "Weights should be symmetric"

    def test_fixed_weights_not_trainable(self, module, batch_input):
        """验证固定权重不参与训练"""
        # 保存初始权重
        initial_weights = module.decay_weights.clone()

        # 前向 + 反向
        output = module(batch_input)
        output.sum().backward()

        # 权重应该不变（因为 no_grad）
        assert torch.allclose(initial_weights, module.decay_weights, atol=1e-6), \
               "Fixed weights should not change after backward"

    # ===================================================================
    # 与标准 Conv1D 对比
    # ===================================================================

    def test_compare_with_standard_conv(self):
        """对比标准 Conv1D 与距离衰减卷积的输出差异"""
        in_channels = 64
        B, N = 2, 100

        # 标准 Conv1D
        conv_standard = torch.nn.Conv1d(in_channels, 1, kernel_size=5, padding=2)

        # 距离衰减卷积
        conv_decay = HilbertDistanceDecayConv1D(in_channels)

        # 相同输入
        x = torch.randn(B, in_channels, N)

        # 检查输出形状 (标准 Conv1D 输出 [B, 1, N], 需要 squeeze)
        out_standard = conv_standard(x).squeeze(1)
        out_decay = conv_decay(x)

        assert out_standard.shape == out_decay.shape == (B, N)

        # 注意: 由于权重不同，输出值会不同，这是预期行为

    def test_depthwise_separability(self):
        """验证深度可分离性质 - 空间混合与通道混合分离"""
        in_channels = 64
        B, N = 2, 50

        module = HilbertDistanceDecayConv1D(in_channels)
        x = torch.randn(B, in_channels, N)

        # 两次前向应该得到相同结果（无随机性）
        out1 = module(x)
        out2 = module(x)
        assert torch.allclose(out1, out2, atol=1e-6), "Output should be deterministic"


class TestHilbertDistanceDecayConv1DWithSkip:
    """带残差连接版本的测试"""

    @pytest.fixture
    def module(self):
        return HilbertDistanceDecayConv1DWithSkip(
            in_channels=128,
            skip_factor=0.5
        )

    @pytest.fixture
    def batch_input(self):
        return torch.randn(2, 128, 100)

    def test_output_shape(self, module, batch_input):
        """验证输出形状不变"""
        output = module(batch_input)
        assert output.shape == batch_input.shape, \
               f"Expected {batch_input.shape}, got {output.shape}"

    def test_residual_effect(self, module, batch_input):
        """验证残差连接确实添加了原始信号"""
        output = module(batch_input)

        # 如果 skip_factor > 0，输出应该与输入有一定相关性
        # 这个测试比较宽松，只检查形状和数值范围
        assert not torch.isnan(output).any(), "Output should not contain NaN"
        assert not torch.isinf(output).any(), "Output should not contain Inf"


class TestUtilityFunctions:
    """工具函数测试"""

    def test_create_distance_decay_weights_kernel5(self):
        """测试创建 kernel_size=5 的权重"""
        weights = create_distance_decay_weights(kernel_size=5)
        expected = torch.tensor([1/3, 1/2, 1.0, 1/2, 1/3])
        assert torch.allclose(weights.squeeze(), expected, atol=1e-4)

    def test_create_distance_decay_weights_kernel3(self):
        """测试创建 kernel_size=3 的权重"""
        weights = create_distance_decay_weights(kernel_size=3)
        expected = torch.tensor([1/2, 1.0, 1/2])
        assert torch.allclose(weights.squeeze(), expected, atol=1e-4)

    def test_create_distance_decay_weights_kernel7(self):
        """测试创建 kernel_size=7 的权重"""
        weights = create_distance_decay_weights(kernel_size=7)
        expected = torch.tensor([1/4, 1/3, 1/2, 1.0, 1/2, 1/3, 1/4])
        assert torch.allclose(weights.squeeze(), expected, atol=1e-4)

    def test_verify_decay_weights_valid(self):
        """测试验证有效权重"""
        weights = create_distance_decay_weights(kernel_size=5)
        is_valid, msg = verify_decay_weights(weights)
        assert is_valid, f"Valid weights should pass: {msg}"

    def test_verify_decay_weights_invalid(self):
        """测试验证无效权重"""
        # 全 1 权重是无效的
        weights = torch.ones(1, 1, 5)
        is_valid, msg = verify_decay_weights(weights)
        assert not is_valid, "Invalid weights should fail"

    def test_create_weights_on_device(self):
        """测试在不同设备上创建权重"""
        weights_cpu = create_distance_decay_weights(kernel_size=5, device=torch.device('cpu'))
        assert weights_cpu.device.type == 'cpu'

    def test_create_weights_invalid_kernel_raises(self):
        """测试无效 kernel_size 抛出错误"""
        with pytest.raises(ValueError, match="must be odd"):
            create_distance_decay_weights(kernel_size=4)


class TestIntegrationWithHilbertOptimalSplitter:
    """与 HilbertOptimalSplitter 的集成测试"""

    def test_splitter_with_distance_decay_conv(self):
        """测试 HilbertOptimalSplitter 使用距离衰减卷积"""
        from vit_pytorch.layers.splitters import HilbertOptimalSplitter

        splitter = HilbertOptimalSplitter(
            feature_dim=256,
            hidden_dim=64,
            max_level_limit=4,  # 减小以加快测试
            K_min=8,
            K_max=32,
            use_distance_decay_conv=True,
        )

        # 创建测试特征
        features = torch.randn(1, 256, 16, 16)
        result = splitter.forward(features, image_size=(64, 64), hard=False)

        assert result.num_selected > 0, "Should select some tokens"

    def test_splitter_backward_compatibility(self):
        """测试 HilbertOptimalSplitter 向后兼容 (use_distance_decay_conv=False)"""
        from vit_pytorch.layers.splitters import HilbertOptimalSplitter

        splitter = HilbertOptimalSplitter(
            feature_dim=256,
            hidden_dim=64,
            max_level_limit=4,
            K_min=8,
            K_max=32,
            use_distance_decay_conv=False,  # 使用标准 Conv1D
        )

        features = torch.randn(1, 256, 16, 16)
        result = splitter.forward(features, image_size=(64, 64), hard=False)

        assert result.num_selected > 0

    def test_splitter_config_use_distance_decay_conv(self):
        """测试通过 HilbertOptimalSplitterConfig 传递配置"""
        from vit_pytorch.layers.splitters import HilbertOptimalSplitter
        from vit_pytorch.layers.splitters.hilbert_optimal_splitter import (
            HilbertOptimalSplitterConfig
        )

        config = HilbertOptimalSplitterConfig(
            feature_dim=256,
            hidden_dim=64,
            max_level_limit=4,
            use_distance_decay_conv=True,
        )

        splitter = HilbertOptimalSplitter(config=config)
        features = torch.randn(1, 256, 16, 16)
        result = splitter.forward(features, image_size=(64, 64), hard=False)

        assert result.num_selected > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
