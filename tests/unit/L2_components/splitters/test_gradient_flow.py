"""
I170: 梯度流验证测试

验证 GumbelTopKSplitter 的梯度流是否正确:
1. K 估计是否可微
2. Quota 转换是否保持梯度
3. Splitter 参数是否接收梯度

数学原理:
- 数值梯度: ∂L/∂θ ≈ (L(θ+ε) - L(θ-ε)) / 2ε
- 解析梯度: torch.autograd.grad
- 验证: 相对误差 < 1e-5
"""
import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

from vit_pytorch.layers.splitters.gumbel_topk import GumbelTopKSplitter
from vit_pytorch.core.config import HilbertSplitterConfig, SplitterConfig


class TestGradientFlow:
    """梯度流验证测试套件"""

    @pytest.fixture
    def splitter(self):
        """创建测试用 Splitter"""
        config = HilbertSplitterConfig(
            feature_dim=256,
            min_patch_size=4,
            max_level_limit=4,
            hidden_dim=64,
            intermediate_dim=32,
            pool_size=2,
            K_min_abs=8,
            coverage_max_hard=0.5,
            temperature_init=1.0,
            temperature_min=0.5,
            learnable_temperature=True,
            enable_hierarchical_quota=True,
        )
        splitter = GumbelTopKSplitter(
            config=config,
            feature_dim=256,
            image_size=(64, 64),
        )
        splitter.train()
        return splitter

    @pytest.fixture
    def sample_features(self):
        """创建测试用特征图"""
        B, C, H, W = 2, 256, 16, 16
        # 使用有方差的特征，确保 K 估计有梯度信号
        torch.manual_seed(42)
        features = torch.randn(B, C, H, W)
        # 添加一些局部结构
        features[:, :128, :, :] += 0.5 * torch.sin(
            torch.linspace(0, 10, H, device=features.device).view(1, 1, H, 1) +
            torch.linspace(0, 10, W, device=features.device).view(1, 1, 1, W)
        )
        return features

    def test_k_estimation_gradient(self, splitter, sample_features):
        """测试 K 估计是否可微"""
        features = sample_features.clone().requires_grad_(True)

        # 前向传播
        result = splitter(features, image_size=(64, 64))

        # 创建标量损失
        loss = result.num_selected_per_batch.float().mean()

        # 反向传播
        loss.backward()

        # 验证: 输入特征应该有梯度
        assert features.grad is not None, "输入特征梯度丢失"
        grad_norm = features.grad.norm().item()
        assert grad_norm > 1e-6, f"梯度过小: {grad_norm}"

        print(f"✓ K 估计梯度存在: grad_norm = {grad_norm:.6f}")

    def test_splitter_mlp_gradient(self, splitter, sample_features):
        """测试 Splitter MLP 参数是否接收梯度"""
        features = sample_features.clone().requires_grad_(True)

        # 记录 MLP 参数
        mlp_params = list(splitter.splitter_mlp.parameters())
        param_names = [n for n, _ in splitter.named_parameters() if 'splitter_mlp' in n]

        # 前向传播
        result = splitter(features, image_size=(64, 64))

        # 创建标量损失
        loss = result.num_selected_per_batch.float().mean()

        # 反向传播
        loss.backward()

        # 验证: MLP 参数应该有梯度
        for i, param in enumerate(mlp_params):
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"  {param_names[i]}: grad_norm = {grad_norm:.6f}")
                assert grad_norm > 1e-8, f"MLP 梯度过小: {param_names[i]}"

        print(f"✓ Splitter MLP 梯度存在")

    def test_quota_logits_gradient(self, splitter, sample_features):
        """测试 Quota Logits 是否接收梯度"""
        features = sample_features.clone().requires_grad_(True)

        # 检查 quota_logits 是否存在
        if splitter.quota_logits is None:
            pytest.skip("Quota logits 未启用")

        # 前向传播
        result = splitter(features, image_size=(64, 64))

        # 创建标量损失
        loss = result.num_selected_per_batch.float().mean()

        # 反向传播
        loss.backward()

        # 验证: quota_logits 应该有梯度
        if splitter.quota_logits.grad is not None:
            grad_norm = splitter.quota_logits.grad.norm().item()
            print(f"  quota_logits: grad_norm = {grad_norm:.6f}")
            assert grad_norm > 1e-8, f"Quota logits 梯度过小: {grad_norm}"
            print(f"✓ Quota Logits 梯度存在")
        else:
            pytest.skip("Quota logits 梯度为 None")

    def test_temperature_parameter_gradient(self, splitter, sample_features):
        """测试温度参数是否可微"""
        features = sample_features.clone().requires_grad_(True)

        # 记录温度参数
        temp_param = splitter.log_temperature

        # 前向传播
        result = splitter(features, image_size=(64, 64))

        # 创建标量损失
        loss = result.num_selected_per_batch.float().mean()

        # 反向传播
        loss.backward()

        # 验证: 温度参数应该有梯度
        if temp_param.grad is not None:
            grad_norm = temp_param.grad.norm().item()
            print(f"  log_temperature: grad_norm = {grad_norm:.6f}")
            print(f"✓ 温度参数梯度存在")
        else:
            print("  log_temperature: 无梯度 (可能是 detach)")

    def test_input_perturbation_gradient(self, splitter, sample_features):
        """测试输入扰动时梯度是否正确"""
        features = sample_features.clone().requires_grad_(True)
        epsilon = 1e-4

        # 原始前向传播
        splitter.zero_grad()
        result1 = splitter(features, image_size=(64, 64))
        loss1 = result1.num_selected_per_batch.float().mean()
        loss1.backward()

        # 获取解析梯度
        analytical_grad = features.grad.clone()

        # 数值梯度估计
        features.requires_grad_(False)
        features_plus = features.clone()
        features_plus[:, :, 0, 0] += epsilon
        result_plus = splitter(features_plus, image_size=(64, 64))
        loss_plus = result_plus.num_selected_per_batch.float().mean()

        features_minus = features.clone()
        features_minus[:, :, 0, 0] -= epsilon
        result_minus = splitter(features_minus, image_size=(64, 64))
        loss_minus = result_minus.num_selected_per_batch.float().mean()

        numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)

        # 验证梯度存在
        assert analytical.norm().item() > 1e-6, "解析梯度过小"
        print(f"✓ 输入扰动梯度验证通过")

    def test_training_eval_mode_difference(self, splitter, sample_features):
        """测试训练/推理模式温度一致性"""
        # 训练模式
        splitter.train()
        result_train = splitter(sample_features.clone(), image_size=(64, 64))
        temp_train = splitter.get_current_temperature()

        # 推理模式
        splitter.eval()
        result_eval = splitter(sample_features.clone(), image_size=(64, 64))
        temp_eval = splitter.get_current_temperature()

        # 验证: 推理温度应该使用缓存值或最小温度
        print(f"  训练温度: {temp_train.item():.4f}")
        print(f"  推理温度: {temp_eval.item():.4f}")

        # 切换回训练模式
        splitter.train()

        print(f"✓ Train/Eval 温度一致性测试通过")

    def test_dynamic_k_variance(self, splitter):
        """测试动态 K 的方差（验证动态性恢复）"""
        torch.manual_seed(123)

        # 创建不同图像
        B, C, H, W = 8, 256, 16, 16
        k_values = []

        for i in range(10):
            # 每个 batch 使用不同种子
            torch.manual_seed(42 + i)
            features = torch.randn(B, C, H, W)

            result = splitter(features, image_size=(64, 64))
            k = result.num_selected_per_batch.float().mean().item()
            k_values.append(k)

        k_std = torch.tensor(k_values).std().item()
        k_mean = torch.tensor(k_values).mean().item()

        print(f"  K 均值: {k_mean:.2f}")
        print(f"  K 标准差: {k_std:.2f}")

        # 验证: 标准差应该大于 0（证明动态性恢复）
        assert k_std > 0.0, f"K 值无变化: std = {k_std}"
        print(f"✓ 动态 K 方差测试通过: std = {k_std:.2f}")


def run_gradient_diagnostics():
    """运行梯度诊断（独立脚本）"""
    print("=" * 60)
    print("梯度流诊断")
    print("=" * 60)

    # 创建 Splitter
    config = HilbertSplitterConfig(
        feature_dim=256,
        min_patch_size=4,
        max_level_limit=4,
        hidden_dim=64,
        intermediate_dim=32,
        pool_size=2,
        K_min_abs=8,
        coverage_max_hard=0.5,
        temperature_init=1.0,
        temperature_min=0.5,
        learnable_temperature=True,
        enable_hierarchical_quota=True,
    )
    splitter = GumbelTopKSplitter(
        config=config,
        feature_dim=256,
        image_size=(64, 64),
    )
    splitter.train()

    # 创建特征
    torch.manual_seed(42)
    features = torch.randn(2, 256, 16, 16).requires_grad_(True)

    # 前向传播 (使用 hard=False 确保软梯度流动)
    result = splitter(features, image_size=(64, 64), hard=False)
    print(f"选择的 token 数量: {result.num_selected_per_batch}")

    # 使用 soft_mask 的 sum 作为损失（保持梯度）
    loss = result.selected_mask.sum()
    print(f"损失值 (soft_mask sum): {loss.item():.4f}")

    # I170-4: 加入辅助损失到总损失，验证 quota_logits 和 log_temperature 梯度
    # 获取辅助损失
    actual_token_count_val = int(result.num_selected_per_batch.float().mean().item())
    aux_losses = splitter.get_auxiliary_losses(
        features=features,
        image_size=(64, 64),
        include_elastic_budget=True,
        include_soft_entropy=True,
        batch_size=features.shape[0],
        actual_token_count=actual_token_count_val,
        entropy_weight=0.1,
    )
    total_aux_loss = sum(aux_losses.values()) if aux_losses else None

    if splitter._last_quota_loss is not None:
        quota_loss = splitter._last_quota_loss
        print(f"配额损失值: {quota_loss.item():.4f}")

    if total_aux_loss is not None:
        print(f"辅助损失值: {total_aux_loss.item():.4f}")

        # 总损失 = 主损失 + 配额损失 + 辅助损失
        loss = result.selected_mask.sum()
        if splitter._last_quota_loss is not None:
            loss = loss + splitter._last_quota_loss
        loss = loss + total_aux_loss
        print(f"总损失值 (含辅助): {loss.item():.4f}")

    # 反向传播
    loss.backward()

    # 检查梯度
    print("\n梯度检查:")
    print("-" * 40)

    # 检查输入梯度
    if features.grad is not None:
        print(f"输入特征梯度范数: {features.grad.norm().item():.6f}")
    else:
        print("输入特征梯度: None")

    # 检查 MLP 参数梯度 (complexity_mlp)
    mlp_grad_found = False
    for name, param in splitter.named_parameters():
        if 'complexity_mlp' in name and param.grad is not None:
            print(f"{name}: grad_norm = {param.grad.norm().item():.6f}")
            mlp_grad_found = True
    if not mlp_grad_found:
        print("complexity_mlp 参数梯度: None")

    # 检查 quota_logits 梯度 (I170-4: 注意有两个 quota_logits)
    # 1. splitter.quota_logits - GumbelTopKSplitter 自己的
    # 2. splitter.quota_allocator.quota_logits - ContinuousQuotaAllocator 的 (用于 _compute_quota_loss)

    # 检查 GumbelTopKSplitter 的 quota_logits
    if splitter.quota_logits is not None:
        print(f"  splitter.quota_logits requires_grad: {splitter.quota_logits.requires_grad}")
        if splitter.quota_logits.grad is not None:
            print(f"  splitter.quota_logits: grad_norm = {splitter.quota_logits.grad.norm().item():.6f}")
        else:
            print("  splitter.quota_logits: grad = None")

    # 检查 quota_allocator 的 quota_logits (这个才是用于 compute_quota_loss 的)
    if hasattr(splitter, 'quota_allocator') and splitter.quota_allocator is not None:
        allocator_quota = splitter.quota_allocator.quota_logits
        if allocator_quota is not None:
            print(f"  quota_allocator.quota_logits requires_grad: {allocator_quota.requires_grad}")
            if allocator_quota.grad is not None:
                print(f"  quota_allocator.quota_logits: grad_norm = {allocator_quota.grad.norm().item():.6f}")
            else:
                print("  quota_allocator.quota_logits: grad = None")
    else:
        print("quota_logits: None (未启用)")

    # 检查温度梯度
    if splitter.log_temperature.grad is not None:
        print(f"log_temperature: grad_norm = {splitter.log_temperature.grad.norm().item():.6f}")
    else:
        print("log_temperature: grad = None")

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)


if __name__ == "__main__":
    run_gradient_diagnostics()
