"""I167-1: Hilbert Distance Decay Convolution

解耦版 Hilbert 流形卷积：
- 空间混合 (Spatial): Depthwise Conv1D with fixed distance decay weights (0 params)
- 通道混合 (Channel): Pointwise 1x1 Conv (D params)

数学形式:
    z = Pointwise(Depthwise(x, w_decay))
    w_decay[k] = 1 / (|k - center| + 1)

参考: NAP/MAT 论文距离衰减权重 w_d = 1/(|d|+1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple


class HilbertDistanceDecayConv1D(nn.Module):
    """解耦版 Hilbert 距离衰减卷积

    将标准 Conv1D(k=5) 分解为：
    1. Depthwise: 固定距离衰减权重，强制局部性先验 (0 参数)
    2. Pointwise: 可学习的通道混合 (D 参数)

    参数:
        in_channels: 输入通道数 (即 hidden_dim * 4)

    输入形状:
        x: [B, D, N] - 沿 Hilbert 序列的特征，D = in_channels

    输出形状:
        z: [B, N] - 每个位置的 logit
    """

    def __init__(self, in_channels: int, kernel_size: int = 5):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Pointwise 1x1 conv: D -> 1 (可学习, D 参数)
        self.pointwise = nn.Conv1d(in_channels, 1, kernel_size=1)

        # 固定距离衰减权重 (只读, 不注册为参数)
        self._register_fixed_decay_weights()

    def _register_fixed_decay_weights(self) -> None:
        """注册固定距离衰减权重: w_d = 1/(|d|+1)

        权重形状: [1, 1, kernel_size]
        对于 kernel_size=5: [-2, -1, 0, 1, 2] -> [1/3, 1/2, 1, 1/2, 1/3]
        """
        offsets = torch.arange(
            -self.padding, self.padding + 1, dtype=torch.float32
        )
        weights = 1.0 / (offsets.abs() + 1.0)  # [K]
        self.register_buffer(
            '_decay_weights',
            weights.view(1, 1, -1),
            persistent=False  # 不保存到 state_dict
        )

    @property
    def decay_weights(self) -> torch.Tensor:
        """返回固定距离衰减权重 (只读)"""
        return self._decay_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D, N] - 沿 Hilbert 序列的特征

        Returns:
            z: [B, N] - 每个位置的 logit
        """
        B, D, N = x.shape
        assert D == self.in_channels, (
            f"Expected in_channels={self.in_channels}, got {D}"
        )

        # Step 1: Depthwise with fixed distance decay
        # Note: _decay_weights is a buffer (not a Parameter), so it automatically
        # has requires_grad=False. No need for torch.no_grad() wrapper -
        # gradients will flow to x but not to the fixed decay weights.
        z = F.conv1d(
            x,
            weight=self._decay_weights.expand(self.in_channels, 1, -1),
            padding=self.padding,
            groups=self.in_channels
        )  # [B, D, N]

        # Step 2: Pointwise 1x1 conv (learnable)
        z = self.pointwise(z)  # [B, 1, N]

        return z.squeeze(1)  # [B, N]

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"decay=fixed(w_d=1/(|d|+1))"
        )

    @property
    def output_dict(self) -> Dict[str, Any]:
        """返回 Decay Conv 层诊断指标（用于日志系统）

        I167-1 公理验证:
            - decay_weights_center 应接近 1.0
            - 若偏离说明距离衰减先验被破坏
        """
        return {
            "decay_weights_mean": self._decay_weights.mean().item(),
            "decay_weights_min": self._decay_weights.min().item(),
            "decay_weights_max": self._decay_weights.max().item(),
            # I167-1: center 权重应为 1.0，偏离说明公理被破坏
            "decay_weights_center": self._decay_weights[0, 0, self.padding].item(),
        }


class HilbertDistanceDecayConv1DWithSkip(nn.Module):
    """带残差连接的版本

    适用于需要保留原始信号的场景:
        z = x + Pointwise(Depthwise(x, w_decay))

    注意: 这个类主要用于测试目的，实际的 HilbertOptimalSplitter
    不使用残差连接。
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 5,
        skip_factor: float = 0.5
    ):
        super().__init__()
        self.decay_conv = HilbertDistanceDecayConv1D(in_channels, kernel_size)
        self.skip_factor = skip_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D, N] - 沿 Hilbert 序列的特征
        Returns:
            z: [B, D, N] - 带残差连接的特征
        """
        # 距离衰减卷积 (输入 [B, D, N], 输出 [B, N])
        conv_out = self.decay_conv(x)  # [B, N]

        # 将 [B, N] 扩展为 [B, D, N] 以匹配输入
        conv_out = conv_out.unsqueeze(1).expand_as(x)  # [B, D, N]

        # 残差连接: out = x + skip_factor * conv_out
        return x + self.skip_factor * conv_out


def create_distance_decay_weights(
    kernel_size: int = 5,
    device: torch.device = None,
) -> torch.Tensor:
    """工具函数: 创建距离衰减权重

    Args:
        kernel_size: 卷积核大小 (必须为奇数)
        device: 目标设备

    Returns:
        weights: [1, 1, kernel_size] 固定衰减权重
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")

    pad = kernel_size // 2
    offsets = torch.arange(-pad, pad + 1, dtype=torch.float32)
    weights = 1.0 / (offsets.abs() + 1.0)
    weights = weights.view(1, 1, -1)

    if device is not None:
        weights = weights.to(device)

    return weights


def verify_decay_weights(weights: torch.Tensor) -> Tuple[bool, str]:
    """验证距离衰减权重的正确性

    Args:
        weights: [1, 1, K] 衰减权重

    Returns:
        (is_valid, message): 是否有效及原因
    """
    expected_values = {
        5: [1/3, 1/2, 1.0, 1/2, 1/3],
        3: [1/2, 1.0, 1/2],
        7: [1/4, 1/3, 1/2, 1.0, 1/2, 1/3, 1/4],
    }

    k = weights.shape[-1]
    if k not in expected_values:
        return False, f"Unsupported kernel_size={k}"

    expected = torch.tensor(expected_values[k])
    actual = weights.squeeze().cpu()

    if torch.allclose(actual, expected, atol=1e-4):
        return True, f"Valid decay weights for kernel_size={k}"
    else:
        diff = (actual - expected).abs().max().item()
        return False, f"Invalid weights, max_diff={diff:.6f}"
