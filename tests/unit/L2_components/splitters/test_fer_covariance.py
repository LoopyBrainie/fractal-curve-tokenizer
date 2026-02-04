"""
FER协方差计算单元测试 (I112-2)

测试覆盖:
1. n=1 边界情况 (协方差应为0)
2. MLE vs Bessel校正一致性
3. 协方差计算的数值稳定性
"""

import pytest
import torch
import torch.nn.functional as F


class TestFERCovariance:
    """FER协方差计算单元测试"""

    def test_covariance_n1_boundary(self):
        """I112-2: 测试 n=1 边界情况协方差计算

        数学验证:
        - n=1 时，中心化后 X_c = X - mean(X) = 0
        - X_c^T X_c = 0 (零矩阵)
        - MLE: Σ = 0 / 1 = 0 (数学自洽)
        """
        D = 384
        X = torch.randn(1, D)

        # 中心化
        X_c = X - X.mean(dim=0, keepdim=True)

        # MLE协方差估计 (修复后)
        n = X_c.shape[0]
        cov_mle = (X_c.T @ X_c) / (n + 1e-6)

        # 验证: n=1 时协方差应为0
        assert cov_mle.abs().sum() < 1e-5, \
            f"n=1 时协方差应为0, got {cov_mle.abs().sum().item():.6f}"

    def test_covariance_n2_minimal(self):
        """I112-2: 测试 n=2 最小有效样本情况

        数学验证:
        - n=2 时，Bessel校正使用 1/(n-1) = 1
        - MLE使用 1/n = 0.5
        - 差异: 50% (理论预期)
        """
        D = 64
        X = torch.randn(2, D)

        X_c = X - X.mean(dim=0, keepdim=True)

        # MLE协方差
        cov_mle = (X_c.T @ X_c) / (2 + 1e-6)

        # Bessel协方差
        cov_bessel = (X_c.T @ X_c) / (1 + 1e-6)

        # MLE应该是Bessel的一半
        ratio = (cov_mle / (cov_bessel + 1e-8)).mean().item()
        assert abs(ratio - 0.5) < 0.01, \
            f"MLE应为Bessel的0.5倍, got {ratio:.4f}"

    def test_covariance_mle_consistency(self):
        """I112-2: 测试 MLE 协方差随 n 的变化

        数学验证:
        - n=100 时，MLE与Bessel差异约 1%
        - n=1000 时，差异约 0.1%
        """
        D = 128
        X = torch.randn(100, D)

        X_c = X - X.mean(dim=0, keepdim=True)

        # MLE协方差
        cov_mle = (X_c.T @ X_c) / (100 + 1e-6)

        # Bessel协方差
        cov_bessel = (X_c.T @ X_c) / (99 + 1e-6)

        # 计算相对差异
        diff_ratio = (cov_mle - cov_bessel).abs().sum() / cov_bessel.abs().sum()

        assert diff_ratio < 0.02, \
            f"n=100 时差异应<2%, got {diff_ratio.item()*100:.2f}%"

    def test_covariance_large_n(self):
        """I112-2: 测试大样本量时 MLE 与 Bessel 趋同"""
        D = 256
        X = torch.randn(1000, D)

        X_c = X - X.mean(dim=0, keepdim=True)

        # MLE vs Bessel
        cov_mle = (X_c.T @ X_c) / (1000 + 1e-6)
        cov_bessel = (X_c.T @ X_c) / (999 + 1e-6)

        diff_ratio = (cov_mle - cov_bessel).abs().sum() / cov_bessel.abs().sum()

        assert diff_ratio < 0.002, \
            f"n=1000 时差异应<0.2%, got {diff_ratio.item()*100:.3f}%"

    def test_covariance_gradient_flow(self):
        """I112-2: 测试协方差计算的梯度流"""
        D = 64
        X = torch.randn(16, D, requires_grad=True)

        X_c = X - X.mean(dim=0, keepdim=True)

        # 协方差计算
        n = X_c.shape[0]
        cov = (X_c.T @ X_c) / (n + 1e-6)

        # Frobenius范数损失
        loss = cov.sum()
        loss.backward()

        # 验证梯度存在
        assert X.grad is not None, "梯度应存在"
        assert X.grad.abs().sum() > 0, "梯度应非零"

    def test_covariance_spectral_properties(self):
        """I112-2: 测试协方差矩阵的谱性质

        数学验证:
        - 协方差矩阵应该是对称的
        - 应该是半正定的
        """
        D = 32
        X = torch.randn(50, D)

        X_c = X - X.mean(dim=0, keepdim=True)
        cov = (X_c.T @ X_c) / (50 + 1e-6)

        # 1. 对称性
        assert torch.allclose(cov, cov.T, atol=1e-6), "协方差矩阵应是对称的"

        # 2. 半正定性 (特征值非负)
        eigenvalues = torch.linalg.eigvalsh(cov)
        assert (eigenvalues >= -1e-6).all(), "特征值应非负"


class TestFERLossIntegration:
    """FER损失集成测试"""

    def test_feature_reconstruction_loss_shape(self):
        """测试特征重构损失输入输出形状"""
        B, N, D = 4, 16, 384
        original = torch.randn(B, N, D)
        reconstructed = torch.randn(B, N, D)

        # 简单MSE损失验证
        mean_loss = F.mse_loss(reconstructed.mean(dim=(0, 1)), original.mean(dim=(0, 1)))
        var_loss = F.mse_loss(reconstructed.var(dim=(0, 1)), original.var(dim=(0, 1)))

        assert mean_loss.dim() == 0, "损失应为标量"
        assert var_loss.dim() == 0, "损失应为标量"

    def test_channel_correlation_loss_shape(self):
        """测试通道相关性损失形状"""
        B, N, D = 2, 8, 128
        original = torch.randn(B, N, D)
        reconstructed = torch.randn(B, N, D)

        # 展平
        original_flat = original.flatten(0, 1)  # [B*N, D]
        recon_flat = reconstructed.flatten(0, 1)

        # 中心化
        original_flat = original_flat - original_flat.mean(dim=0, keepdim=True)
        recon_flat = recon_flat - recon_flat.mean(dim=0, keepdim=True)

        # 协方差计算 (MLE)
        n = original_flat.shape[0]
        original_cov = (original_flat.mT @ original_flat) / (n + 1e-6)
        recon_cov = (recon_flat.mT @ recon_flat) / (n + 1e-6)

        # 验证形状
        assert original_cov.shape == (D, D), f"期望 {(D, D)}, 实际 {original_cov.shape}"
        assert recon_cov.shape == (D, D), f"期望 {(D, D)}, 实际 {recon_cov.shape}"

        # MSE损失
        corr_loss = F.mse_loss(recon_cov, original_cov)
        assert corr_loss.dim() == 0, "相关性损失应为标量"
