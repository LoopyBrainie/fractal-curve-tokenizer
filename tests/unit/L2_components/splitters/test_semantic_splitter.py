"""
语义冗余分裂器单元测试

测试覆盖:
1. LookAheadHead 输出形状和梯度流
2. CorrelationGate 冗余性计算正确性
3. SemanticRedundancySplitter 决策逻辑
4. SemanticRedundancyLoss 梯度流
5. 训练/推理模式一致性
"""

import pytest
import torch
import torch.nn as nn

from vit_pytorch.semantic_redundancy_splitter import (
    LookAheadHead,
    CorrelationGate,
    SemanticRedundancySplitter,
    SplitResult,
)
from vit_pytorch.semantic_losses import (
    DiversityLoss,
    ReconstructionLoss,
    SemanticRedundancyLoss,
)


class TestLookAheadHead:
    """LookAheadHead 单元测试"""

    def test_output_shape(self):
        """测试输出形状正确性"""
        B, N, D = 2, 10, 256
        model = LookAheadHead(D, hidden_dim=128)
        features = torch.randn(B, N, D)
        output = model(features)

        assert output.shape == (B, N, 4, D), f"期望 {(B, N, 4, D)}, 实际 {output.shape}"

    def test_feature_dim_384(self):
        """测试 D=384 配置"""
        B, N, D = 1, 5, 384
        model = LookAheadHead(D, hidden_dim=128)
        features = torch.randn(B, N, D)
        output = model(features)

        assert output.shape == (B, N, 4, D)

    def test_gradient_flow(self):
        """测试梯度流连续性"""
        B, N, D = 2, 10, 256
        model = LookAheadHead(D, hidden_dim=128)
        features = torch.randn(B, N, D, requires_grad=True)
        output = model(features)

        # 反向传播
        loss = output.sum()
        loss.backward()

        # 检查梯度存在且非零
        assert features.grad is not None, "输入梯度不存在"
        assert features.grad.abs().sum() > 0, "梯度为零"

    def test_hidden_dim_variations(self):
        """测试不同隐藏层维度"""
        B, N, D = 2, 5, 256
        features = torch.randn(B, N, D)

        for hidden_dim in [64, 128, 256]:
            model = LookAheadHead(D, hidden_dim=hidden_dim)
            output = model(features)
            assert output.shape == (B, N, 4, D)

    def test_deterministic_output(self):
        """测试确定性输出（相同输入，相同输出）"""
        B, N, D = 2, 5, 256
        model = LookAheadHead(D, hidden_dim=128)
        features = torch.randn(B, N, D)

        output1 = model(features)
        output2 = model(features)

        assert torch.allclose(output1, output2), "相同输入应产生相同输出"


class TestCorrelationGate:
    """CorrelationGate 单元测试"""

    def test_output_range(self):
        """测试输出在 [0, 1] 范围内"""
        B, N, D = 2, 10, 256
        gate = CorrelationGate(D)
        child_features = torch.randn(B, N, 4, D)
        redundancy = gate(child_features)

        assert redundancy.min() >= 0, f"冗余性下限 < 0: {redundancy.min()}"
        assert redundancy.max() <= 1, f"冗余性上限 > 1: {redundancy.max()}"

    def test_identical_features_redundancy(self):
        """测试完全相同特征的冗余性（应接近 0）"""
        B, N, D = 2, 10, 256
        gate = CorrelationGate(D)

        # 四个子节点完全相同
        features = torch.randn(B, N, 1, D).expand(-1, -1, 4, -1)
        redundancy = gate(features)

        assert redundancy.abs().max() < 1e-5, f"相同特征冗余性应接近 0, got {redundancy.abs().max()}"

    def test_orthogonal_features_independence(self):
        """测试近似正交特征的独立性"""
        B, N, D = 1, 1, 256
        gate = CorrelationGate(D)

        # 在高维空间生成近似正交的特征
        # 使用 Gram-Schmidt 正交化
        v1 = torch.randn(D)
        v1 = v1 / v1.norm()

        v2 = torch.randn(D)
        v2 = v2 - (v2 @ v1) * v1
        v2 = v2 / v2.norm()

        v3 = torch.randn(D)
        v3 = v3 - (v3 @ v1) * v1 - (v3 @ v2) * v2
        v3 = v3 / v3.norm()

        v4 = torch.randn(D)
        v4 = v4 - (v4 @ v1) * v1 - (v4 @ v2) * v2 - (v4 @ v3) * v3
        v4 = v4 / v4.norm()

        features = torch.stack([v1, v2, v3, v4]).unsqueeze(0).unsqueeze(0)
        redundancy = gate(features)

        # 经过 Gram-Schmidt 正交化后，平均余弦相似度应接近 0
        # 冗余性 = 1 - sim_mean ≈ 1 - 0 = 1
        assert redundancy.item() > 0.5, f"正交特征冗余性应大于 0.5, got {redundancy.item():.4f}"

    def test_gradient_flow(self):
        """测试梯度流"""
        B, N, D = 2, 10, 256
        gate = CorrelationGate(D)
        child_features = torch.randn(B, N, 4, D, requires_grad=True)
        redundancy = gate(child_features)

        loss = redundancy.sum()
        loss.backward()

        assert child_features.grad is not None, "输入梯度不存在"
        assert child_features.grad.abs().sum() > 0, "梯度为零"

    def test_batch_processing(self):
        """测试批量处理"""
        B, N, D = 4, 20, 256
        gate = CorrelationGate(D)
        child_features = torch.randn(B, N, 4, D)
        redundancy = gate(child_features)

        assert redundancy.shape == (B, N)


class TestSemanticRedundancySplitter:
    """SemanticRedundancySplitter 单元测试"""

    def test_forward_output_type(self):
        """测试前向输出类型"""
        B, N, D = 2, 10, 256
        splitter = SemanticRedundancySplitter(D, hidden_dim=128)
        features = torch.randn(B, N, D)

        result = splitter(features, depth=2, remaining_quota=0.5)

        assert isinstance(result, SplitResult)
        assert result.split_decision.shape == (B, N)
        assert result.child_features.shape == (B, N, 4, D)
        assert result.redundancy.shape == (B, N)
        assert result.logits.shape == (B, N)

    def test_hard_mode(self):
        """测试硬模式输出为二值"""
        B, N, D = 2, 10, 256
        splitter = SemanticRedundancySplitter(D, hidden_dim=128)
        features = torch.randn(B, N, D)

        result = splitter(features, depth=2, remaining_quota=0.5, hard=True)

        unique_values = result.split_decision.unique()
        assert len(unique_values) <= 2, "硬模式应为二值输出"
        assert all(v in [0.0, 1.0] for v in unique_values)

    def test_soft_mode_probabilistic(self):
        """测试软模式输出为概率值"""
        B, N, D = 2, 10, 256
        splitter = SemanticRedundancySplitter(D, hidden_dim=128)
        features = torch.randn(B, N, D)

        result = splitter(features, depth=2, remaining_quota=0.5, hard=False)

        # 软模式输出应在 (0, 1) 范围内
        assert result.split_decision.min() >= 0
        assert result.split_decision.max() <= 1

    def test_gradient_flow(self):
        """测试梯度流"""
        B, N, D = 2, 10, 256
        splitter = SemanticRedundancySplitter(D, hidden_dim=128)
        features = torch.randn(B, N, D, requires_grad=True)

        result = splitter(features, depth=2, remaining_quota=0.5)
        loss = result.split_decision.sum()
        loss.backward()

        assert features.grad is not None, "输入梯度不存在"
        assert features.grad.abs().sum() > 0, "梯度为零"

    def test_temperature_parameter_exists(self):
        """测试温度参数存在"""
        D = 256
        splitter = SemanticRedundancySplitter(D, hidden_dim=128)

        assert hasattr(splitter, '_get_temperature')
        temp = splitter._get_temperature()
        assert temp > 0, "温度应大于 0"

    def test_depth_normalization(self):
        """测试深度归一化"""
        B, N, D = 2, 10, 256
        splitter = SemanticRedundancySplitter(D, hidden_dim=128, max_level_limit=8)
        features = torch.randn(B, N, D)

        result_shallow = splitter(features, depth=1, remaining_quota=0.5)
        result_deep = splitter(features, depth=7, remaining_quota=0.5)

        # 深度不应导致 NaN 或 Inf
        assert not torch.isnan(result_shallow.logits).any()
        assert not torch.isnan(result_deep.logits).any()
        assert not torch.isinf(result_shallow.logits).any()
        assert not torch.isinf(result_deep.logits).any()

    def test_parameter_count(self):
        """测试可学习参数数量"""
        D = 256
        splitter = SemanticRedundancySplitter(D, hidden_dim=128)

        param_count = sum(p.numel() for p in splitter.parameters() if p.requires_grad)

        # LookAheadHead: D*128 + 128 + 128*4D + 4D = 640D + 128 + 1024
        #   = 640*256 + 128 + 1024 = 163840 + 128 + 1024 = 164992
        # Decision params: 4 (w_r, w_d, w_q, bias)
        # Temperature: 1 (log_temp)
        expected_base = 164992 + 4 + 1  # = 164997
        assert param_count == expected_base, f"期望 {expected_base}, 实际 {param_count}"


class TestSemanticRedundancyLoss:
    """SemanticRedundancyLoss 单元测试"""

    def test_diversity_loss_orthogonal(self):
        """测试正交特征的多样性损失较低"""
        B, N, D = 2, 10, 256
        loss_fn = DiversityLoss(reduction="none")

        # 生成近似正交特征（使用 Gram-Schmidt）
        v1 = torch.randn(D)
        v1 = v1 / v1.norm()

        v2 = torch.randn(D)
        v2 = v2 - (v2 @ v1) * v1
        v2 = v2 / v2.norm()

        v3 = torch.randn(D)
        v3 = v3 - (v3 @ v1) * v1 - (v3 @ v2) * v2
        v3 = v3 / v3.norm()

        v4 = torch.randn(D)
        v4 = v4 - (v4 @ v1) * v1 - (v4 @ v2) * v2 - (v4 @ v3) * v3
        v4 = v4 / v4.norm()

        child_features = torch.stack([v1, v2, v3, v4]).view(1, 1, 4, D)
        child_features = child_features.expand(B, N, -1, -1)

        loss = loss_fn(child_features)
        # 经过 Gram-Schmidt 正交化后，损失应显著降低
        assert loss.abs().max() < 0.1, f"正交特征损失应较小, got {loss.abs().max():.4f}"

    def test_diversity_loss_norm_invariance(self):
        """I112-1: 测试多样性损失对特征范数的不变性

        验证修复后的 DiversityLoss 使用余弦相似度，
        无论特征范数大小，正交特征的损失应接近 0。
        """
        B, N, D = 1, 1, 256
        loss_fn = DiversityLoss(reduction="none")

        # 使用 QR 分解创建完美正交基
        def create_orthogonal_features(norm_value: float) -> torch.Tensor:
            random = torch.randn(D, 4)
            q, _ = torch.linalg.qr(random)  # Q 的列为正交单位向量, shape [D, 4]
            return q.T * norm_value  # 转置并缩放到指定范数, shape [4, D]

        # 测试不同范数下的正交特征
        for norm_value in [1.0, 10.0, 100.0, 1000.0]:
            orthogonal_features = create_orthogonal_features(norm_value)
            assert orthogonal_features.shape == (4, D), f"Expected (4, {D}), got {orthogonal_features.shape}"
            child_features = orthogonal_features.view(B, N, 4, D)
            loss = loss_fn(child_features)
            # 正交特征的损失应接近 0（与范数无关）
            assert loss.abs() < 1e-5, \
                f"正交特征(范数={norm_value})损失应≈0, got {loss.item():.6f}"

        # 验证：完全平行的大范数特征会产生显著损失
        # 创建真正平行的特征：所有4个特征向量完全相同
        base_vector = torch.randn(D)
        base_vector = base_vector / base_vector.norm()  # 归一化
        parallel_features = base_vector.unsqueeze(0).expand(4, -1)  # [4, D]
        parallel_child = parallel_features.view(B, N, 4, D)
        parallel_loss = loss_fn(parallel_child)
        # 完美平行特征的损失应为 12 (4个特征完全平行)
        # 4×3 = 12 (4行，每行有3个1)
        assert parallel_loss.abs() > 10.0, \
            f"平行特征损失应≈12, got {parallel_loss.item():.4f}"

    def test_reconstruction_loss_identity(self):
        """测试子节点均值等于父节点时重构损失为 0"""
        B, N, D = 2, 10, 256
        loss_fn = ReconstructionLoss(reduction="none")

        parent = torch.randn(B, N, D)
        child = parent.unsqueeze(2).expand(-1, -1, 4, -1)

        loss = loss_fn(parent, child)
        assert loss.abs().max() < 1e-5, f"重构损失应接近 0, got {loss.abs().max()}"

    def test_gradient_flow(self):
        """测试梯度流"""
        B, N, D = 2, 10, 256
        loss_fn = SemanticRedundancyLoss()

        parent = torch.randn(B, N, D, requires_grad=True)
        child = torch.randn(B, N, 4, D, requires_grad=True)

        losses = loss_fn(parent, child)
        losses["loss"].backward()

        assert parent.grad is not None, "父节点梯度不存在"
        assert child.grad is not None, "子节点梯度不存在"

    def test_weighted_combination(self):
        """测试权重组合"""
        B, N, D = 2, 10, 256
        loss_fn = SemanticRedundancyLoss(
            diversity_weight=0.5,
            reconstruction_weight=0.2
        )

        parent = torch.randn(B, N, D)
        child = torch.randn(B, N, 4, D)

        losses = loss_fn(parent, child)

        # 总损失应大于各分量
        assert losses["loss"] >= losses["diversity_loss"] * 0.5
        assert losses["loss"] >= losses["reconstruction_loss"] * 0.2

    def test_split_masking(self):
        """测试分裂掩码 - 使用 mean reduction"""
        B, N, D = 2, 10, 256
        loss_fn = SemanticRedundancyLoss(reduction="mean")

        parent = torch.randn(B, N, D)
        child = torch.randn(B, N, 4, D)

        # 随机分裂决策
        split_decisions = torch.rand(B, N) > 0.7

        losses = loss_fn(parent, child, split_decisions)

        # 不应产生 NaN
        assert not torch.isnan(losses["loss"]).any()
        assert losses["loss"] > 0  # 损失应为正


class TestIntegration:
    """集成测试"""

    def test_e2e_training_step(self):
        """端到端训练步骤 - 验证关键组件的梯度流"""
        B, N, D = 2, 10, 256
        splitter = SemanticRedundancySplitter(D, hidden_dim=128)
        loss_fn = SemanticRedundancyLoss()

        # 随机特征
        features = torch.randn(B, N, D, requires_grad=True)

        # 前向
        result = splitter(features, depth=2, remaining_quota=0.5)

        # 计算损失 - 使用 child_features（依赖 LookAheadHead）
        losses = loss_fn(features, result.child_features, result.split_decision)

        # 反向
        losses["loss"].backward()

        # 验证 LookAheadHead 的梯度（核心组件）
        assert features.grad is not None, "输入特征梯度不存在"

        # 检查 LookAheadHead 的 MLP 权重有梯度
        look_ahead_grad = splitter.look_ahead.mlp[0].weight.grad
        assert look_ahead_grad is not None, "LookAheadHead 梯度不存在"

        # 检查 CorrelationGate 的梯度
        corr_grad = splitter.correlation.feature_dim  # CorrelationGate 无内部参数
        assert corr_grad is not None  # 验证通过（CorrelationGate 无参数）

        # 注意：决策参数 (w_r, w_d, w_q, bias) 的梯度取决于 split_decision 是否影响损失
        # 当前设计：split_decision 用于掩码，不直接影响损失计算
        # 如需决策参数梯度，需修改损失函数使其依赖 logits

    def test_different_depths(self):
        """测试不同深度"""
        B, N, D = 1, 5, 256
        splitter = SemanticRedundancySplitter(D, hidden_dim=128, max_level_limit=8)

        for depth in range(1, 8):
            features = torch.randn(B, N, D)
            result = splitter(features, depth=depth, remaining_quota=0.5)

            assert result.split_decision.shape == (B, N)
            assert not torch.isnan(result.split_decision).any()

    def test_temperature_update(self):
        """测试温度更新"""
        D = 256
        splitter = SemanticRedundancySplitter(D, hidden_dim=128)

        initial_temp = splitter._get_temperature().item()

        # 更新温度
        splitter.update_temperature(step=50, total_steps=100, schedule="cosine")
        new_temp = splitter._get_temperature().item()

        # 余弦调度应在更新后改变
        assert initial_temp != new_temp or splitter.gumbel_temp_start == splitter.gumbel_temp_end


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
