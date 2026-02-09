"""Fractal Training 模块单元测试

验证内容:
1. ClassBalancedSampler 数学正确性
2. FocalLoss 与标准 CE 的关系
3. ClassificationMetrics 计算正确性
"""

import pytest
import torch
from collections import Counter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

# 设置 matplotlib 后端为 Agg (无 GUI)，必须在导入 pyplot 之前
import matplotlib
matplotlib.use('Agg')

from training.samplers import (
    ClassBalancedSampler,
    ProgressiveSampler,
    compute_effective_sample_weights,
)
from training.losses import FocalLoss, ClassBalancedCrossEntropy, FocalClassBalancedLoss
from training.metrics import ClassificationMetrics
from training.schedulers import (
    FLOPSConfig,
    compute_transformer_flops,
    FLOPSBudgetLoss,
    DepthWeightedBudgetLoss,
    BudgetScheduler,
    estimate_baseline_flops,
)


class TestClassBalancedSampler:
    """ClassBalancedSampler 数学验证"""
    
    def test_uniform_sampling_when_beta_zero(self):
        """β=0 时应该是均匀采样"""
        # 极度不平衡: 类别0有100个，类别1有10个
        labels = [0] * 100 + [1] * 10
        
        sampler = ClassBalancedSampler(labels, beta=0.0)
        
        # 采样并统计 (多次迭代)
        all_sampled = []
        for _ in range(100):
            indices = list(sampler)
            sampled_labels = [labels[i] for i in indices]
            all_sampled.extend(sampled_labels)
        counts = Counter(all_sampled)
        
        # β=0 时，采样比例应接近原始比例 (100:10 = 10:1)
        ratio = counts[0] / max(counts[1], 1)
        assert 8 < ratio < 12, f"Expected ratio ~10, got {ratio:.2f}"
    
    def test_inverse_frequency_when_beta_one(self):
        """β=1 时应该是逆频率采样 (各类别等概率)"""
        labels = [0] * 100 + [1] * 10
        
        sampler = ClassBalancedSampler(labels, beta=1.0)
        
        # 采样并统计 (多次迭代)
        all_sampled = []
        for _ in range(100):
            indices = list(sampler)
            sampled_labels = [labels[i] for i in indices]
            all_sampled.extend(sampled_labels)
        counts = Counter(all_sampled)
        
        # β=1 时，各类别采样次数应接近相等
        ratio = counts[0] / max(counts[1], 1)
        assert 0.8 < ratio < 1.2, f"Expected ratio ~1, got {ratio:.2f}"
    
    def test_intermediate_beta(self):
        """β=0.5 应该是中间状态"""
        labels = [0] * 100 + [1] * 10
        
        sampler = ClassBalancedSampler(labels, beta=0.5)
        
        # 采样并统计 (多次迭代)
        all_sampled = []
        for _ in range(100):
            indices = list(sampler)
            sampled_labels = [labels[i] for i in indices]
            all_sampled.extend(sampled_labels)
        counts = Counter(all_sampled)
        
        # β=0.5 时，比例应该在 1 和 10 之间
        # 理论: sqrt(100)/sqrt(10) ≈ 3.16
        ratio = counts[0] / max(counts[1], 1)
        assert 2 < ratio < 5, f"Expected ratio ~3.16, got {ratio:.2f}"
    
    def test_get_class_sampling_probs(self):
        """验证类别采样概率计算"""
        labels = [0, 0, 0, 1]  # 类别0: 3个, 类别1: 1个
        
        sampler = ClassBalancedSampler(labels, beta=1.0)
        stats = sampler.get_statistics()
        probs = stats["sampled_probs"]
        
        # 类别0: 3个样本，每个权重 1/3，总权重 3 * (1/3) = 1
        # 类别1: 1个样本，每个权重 1/1，总权重 1 * 1 = 1
        # 类别0概率: 1/2, 类别1概率: 1/2
        assert abs(probs[0] - 0.5) < 0.01
        assert abs(probs[1] - 0.5) < 0.01


class TestProgressiveSampler:
    """ProgressiveSampler 渐进调度验证"""
    
    def test_beta_schedule(self):
        """验证 β 随 epoch 变化"""
        labels = [0] * 100 + [1] * 10
        
        sampler = ProgressiveSampler(
            labels,
            beta_start=0.9,
            beta_end=0.3,
            total_epochs=100,
        )
        
        # 初始 β (注意: ProgressiveSampler 使用 beta 而不是 current_beta)
        assert abs(sampler.beta - 0.9) < 0.01
        
        # 中间
        sampler.set_epoch(50)
        assert abs(sampler.beta - 0.6) < 0.01
        
        # 最终
        sampler.set_epoch(99)
        assert abs(sampler.beta - 0.3) < 0.02  # 稍宽松的容差


class TestFocalLoss:
    """FocalLoss 数学验证"""
    
    def test_gamma_zero_equals_ce(self):
        """γ=0 时 Focal Loss 应等于标准 CE"""
        torch.manual_seed(42)
        
        logits = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))
        
        focal = FocalLoss(gamma=0.0)
        ce = torch.nn.CrossEntropyLoss()
        
        focal_loss = focal(logits, targets)
        ce_loss = ce(logits, targets)
        
        assert torch.allclose(focal_loss, ce_loss, atol=1e-5), \
            f"Focal(γ=0) = {focal_loss:.4f}, CE = {ce_loss:.4f}"
    
    def test_focal_reduces_easy_sample_weight(self):
        """Focal Loss 应该降低易分类样本的权重"""
        # 创建一个易分类的样本 (高置信度)
        logits_easy = torch.tensor([[10.0, 0.0, 0.0]])  # 高置信度预测类别0
        logits_hard = torch.tensor([[0.5, 0.3, 0.2]])   # 低置信度预测类别0
        targets = torch.tensor([0])
        
        focal = FocalLoss(gamma=2.0, reduction='none')
        
        loss_easy = focal(logits_easy, targets)
        loss_hard = focal(logits_hard, targets)
        
        # 易分类样本的损失应该更小
        assert loss_easy < loss_hard, \
            f"Easy loss ({loss_easy:.4f}) should be < Hard loss ({loss_hard:.4f})"
    
    def test_focal_vs_ce_ratio(self):
        """验证 Focal Loss 对难样本的相对保留"""
        logits_easy = torch.tensor([[5.0, 0.0]])  # p ≈ 0.993
        logits_hard = torch.tensor([[0.5, 0.0]])  # p ≈ 0.622
        targets = torch.tensor([0])
        
        focal = FocalLoss(gamma=2.0, reduction='none')
        ce = torch.nn.CrossEntropyLoss(reduction='none')
        
        # Focal / CE 比值
        focal_easy = focal(logits_easy, targets)
        focal_hard = focal(logits_hard, targets)
        ce_easy = ce(logits_easy, targets)
        ce_hard = ce(logits_hard, targets)
        
        ratio_easy = focal_easy / ce_easy
        ratio_hard = focal_hard / ce_hard
        
        # 难样本的 Focal/CE 比值应该更大
        assert ratio_hard > ratio_easy, \
            f"Hard ratio ({ratio_hard:.4f}) should be > Easy ratio ({ratio_easy:.4f})"


class TestClassBalancedCrossEntropy:
    """ClassBalancedCrossEntropy 数学验证"""

    def test_weight_calculation(self):
        """验证有效样本数权重计算"""
        class_counts = torch.tensor([100.0, 10.0, 1.0])
        beta = 0.9999

        weights = compute_effective_sample_weights(class_counts, beta)

        # 样本越少，权重越大
        assert weights[2] > weights[1] > weights[0], \
            f"Weights should be increasing: {weights.tolist()}"

        # 权重平均值应为 1
        assert abs(weights.mean() - 1.0) < 0.01

    def test_loss_matches_weighted_ce(self):
        """ClassBalancedCrossEntropy 应该等于加权 CE"""
        torch.manual_seed(42)

        class_counts = torch.tensor([100.0, 50.0, 10.0])
        logits = torch.randn(32, 3)
        targets = torch.randint(0, 3, (32,))

        cb_ce = ClassBalancedCrossEntropy(class_counts, beta=0.9999)
        
        # 手动计算权重并使用标准 CE
        weights = compute_effective_sample_weights(class_counts, 0.9999)
        # 归一化
        weights_normalized = weights / weights.sum() * len(weights)
        standard_ce = torch.nn.CrossEntropyLoss(weight=weights_normalized)
        
        loss1 = cb_ce(logits, targets)
        loss2 = standard_ce(logits, targets)
        
        assert torch.allclose(loss1, loss2, atol=1e-4), \
            f"CB-CE = {loss1:.4f}, Weighted CE = {loss2:.4f}"


class TestClassificationMetrics:
    """ClassificationMetrics 计算验证"""
    
    def test_top1_accuracy(self):
        """验证 Top-1 准确率计算"""
        metrics = ClassificationMetrics(num_classes=3)
        
        # 6 个样本，4 个正确
        logits = torch.tensor([
            [1.0, 0.0, 0.0],  # 预测0，正确
            [0.0, 1.0, 0.0],  # 预测1，正确
            [0.0, 0.0, 1.0],  # 预测2，正确
            [0.0, 1.0, 0.0],  # 预测1，正确
            [1.0, 0.0, 0.0],  # 预测0，错误 (真实1)
            [0.0, 1.0, 0.0],  # 预测1，错误 (真实2)
        ])
        targets = torch.tensor([0, 1, 2, 1, 1, 2])
        
        metrics.update(logits, targets)
        result = metrics.compute()
        
        assert abs(result.top1_accuracy - 4/6) < 0.01
    
    def test_mean_class_accuracy(self):
        """验证 MCA 计算"""
        metrics = ClassificationMetrics(num_classes=3)
        
        # 类别0: 2/2 = 100%
        # 类别1: 1/2 = 50%
        # 类别2: 0/2 = 0%
        # MCA = (100% + 50% + 0%) / 3 = 50%
        logits = torch.tensor([
            [1.0, 0.0, 0.0],  # 预测0，真实0 ✓
            [1.0, 0.0, 0.0],  # 预测0，真实0 ✓
            [0.0, 1.0, 0.0],  # 预测1，真实1 ✓
            [0.0, 0.0, 1.0],  # 预测2，真实1 ✗
            [0.0, 1.0, 0.0],  # 预测1，真实2 ✗
            [0.0, 1.0, 0.0],  # 预测1，真实2 ✗
        ])
        targets = torch.tensor([0, 0, 1, 1, 2, 2])
        
        metrics.update(logits, targets)
        result = metrics.compute()
        
        # Top-1 = 3/6 = 50%
        assert abs(result.top1_accuracy - 0.5) < 0.01
        
        # MCA = (1.0 + 0.5 + 0.0) / 3 = 0.5
        assert abs(result.mean_class_accuracy - 0.5) < 0.01
        
        # Per-class
        assert abs(result.per_class_accuracy[0] - 1.0) < 0.01
        assert abs(result.per_class_accuracy[1] - 0.5) < 0.01
        assert abs(result.per_class_accuracy[2] - 0.0) < 0.01
    
    def test_top1_vs_mca_difference(self):
        """演示 Top-1 和 MCA 的差异"""
        metrics = ClassificationMetrics(num_classes=2)
        
        # 不平衡场景: 类别0有90个全对，类别1有10个全错
        logits_0 = torch.zeros(90, 2)
        logits_0[:, 0] = 1.0  # 预测类别0
        targets_0 = torch.zeros(90, dtype=torch.long)  # 真实类别0
        
        logits_1 = torch.zeros(10, 2)
        logits_1[:, 0] = 1.0  # 预测类别0 (错误!)
        targets_1 = torch.ones(10, dtype=torch.long)   # 真实类别1
        
        metrics.update(logits_0, targets_0)
        metrics.update(logits_1, targets_1)
        result = metrics.compute()
        
        # Top-1 = 90/100 = 90%
        assert abs(result.top1_accuracy - 0.9) < 0.01
        
        # MCA = (100% + 0%) / 2 = 50%
        assert abs(result.mean_class_accuracy - 0.5) < 0.01
        
        print(f"\n演示 Top-1 vs MCA 差异:")
        print(f"  Top-1: {result.top1_accuracy:.1%}")
        print(f"  MCA:   {result.mean_class_accuracy:.1%}")
        print(f"  差异:  {result.top1_accuracy - result.mean_class_accuracy:.1%}")


class TestIntegration:
    """组件集成测试"""
    
    def test_sampler_with_loss(self):
        """采样器 + 损失函数集成"""
        # 模拟不平衡数据
        labels = [0] * 100 + [1] * 10
        class_counts = torch.tensor([100.0, 10.0])
        
        # 创建采样器
        sampler = ClassBalancedSampler(labels, beta=0.5)
        
        # 创建损失函数
        loss_fn = FocalClassBalancedLoss(class_counts, gamma=2.0, beta=0.9999)
        
        # 模拟训练 (需要 requires_grad=True)
        logits = torch.randn(32, 2, requires_grad=True)
        targets = torch.randint(0, 2, (32,))
        
        loss = loss_fn(logits, targets)
        
        assert loss.requires_grad
        assert loss.item() > 0


class TestFLOPSComputation:
    """FLOPS 计算验证"""
    
    def test_flops_scales_quadratically_with_tokens(self):
        """FLOPS 应该随 token 数量超线性增长 (因为 O(N²) 注意力)"""
        config = FLOPSConfig(embed_dim=256, num_layers=12)
        
        flops_64 = compute_transformer_flops(64, config)
        flops_128 = compute_transformer_flops(128, config)
        flops_256 = compute_transformer_flops(256, config)
        
        # token 翻倍，FLOPS 应该超过翻倍
        ratio_1 = flops_128 / flops_64
        ratio_2 = flops_256 / flops_128
        
        assert ratio_1 > 2.0, f"Expected ratio > 2, got {ratio_1:.2f}"
        assert ratio_2 > 2.0, f"Expected ratio > 2, got {ratio_2:.2f}"
        
        print(f"\nFLOPS 缩放验证:")
        print(f"  N=64:  {flops_64:,}")
        print(f"  N=128: {flops_128:,} (ratio: {ratio_1:.2f}x)")
        print(f"  N=256: {flops_256:,} (ratio: {ratio_2:.2f}x)")
    
    def test_baseline_flops_estimation(self):
        """验证基准 FLOPS 估算"""
        config = FLOPSConfig(embed_dim=384, num_layers=12)
        
        # ViT-S/16 on ImageNet-224
        baseline = estimate_baseline_flops(224, 16, config)
        
        # 参考: ViT-S 约 4.6G FLOPS
        # 我们的估算应该在同一数量级
        assert 1e9 < baseline < 1e11, f"Baseline FLOPS {baseline:,} out of expected range"
        
        print(f"\n基准 FLOPS 估算:")
        print(f"  ViT-S/16 @ 224: {baseline:,.0f} ({baseline/1e9:.2f}G)")


class TestBudgetLoss:
    """预算损失验证"""
    
    def test_flops_budget_zero_when_under_budget(self):
        """FLOPS 未超预算时损失为 0"""
        config = FLOPSConfig(embed_dim=256, num_layers=12)
        budget = estimate_baseline_flops(224, 16, config) * 2  # 2x 余量
        
        loss_fn = FLOPSBudgetLoss(budget, lambda_weight=0.01, config=config)
        
        token_counts = torch.tensor([64, 64, 64, 64])
        loss = loss_fn(token_counts)
        
        assert loss.item() < 1e-6, f"Expected ~0 loss, got {loss.item()}"
    
    def test_flops_budget_positive_when_over_budget(self):
        """FLOPS 超预算时损失 > 0"""
        config = FLOPSConfig(embed_dim=256, num_layers=12)
        budget = estimate_baseline_flops(64, 16, config)  # 很小的预算
        
        loss_fn = FLOPSBudgetLoss(budget, lambda_weight=0.01, config=config)
        
        token_counts = torch.tensor([256, 256, 256, 256])  # 超预算
        loss = loss_fn(token_counts)
        
        assert loss.item() > 0, f"Expected loss > 0, got {loss.item()}"
    
    def test_depth_weighted_budget(self):
        """深度加权预算验证"""
        loss_fn = DepthWeightedBudgetLoss(
            budget_tokens=50.0,
            alpha=0.5,
            lambda_weight=0.01,
            max_depth=4,
        )
        
        # 浅层 tokens (应该权重低)
        shallow_depths = torch.zeros(32, dtype=torch.long)  # 全深度0
        token_counts = torch.tensor([32])
        
        loss_shallow = loss_fn(shallow_depths, token_counts)
        
        # 深层 tokens (应该权重高)
        deep_depths = torch.full((32,), 4, dtype=torch.long)  # 全深度4
        
        loss_deep = loss_fn(deep_depths, token_counts)
        
        # 深层应该产生更高损失
        assert loss_deep > loss_shallow, \
            f"Deep loss ({loss_deep:.4f}) should be > Shallow loss ({loss_shallow:.4f})"


class TestBudgetScheduler:
    """预算调度器验证"""
    
    def test_cosine_schedule(self):
        """Cosine 调度验证"""
        scheduler = BudgetScheduler(
            budget_max=100.0,
            budget_min=50.0,
            total_epochs=100,
            schedule="cosine",
        )
        
        # 初始 (epoch 0)
        budget_0 = scheduler.step(0)
        assert abs(budget_0 - 100.0) < 0.1
        
        # 中间 (epoch 50)
        budget_50 = scheduler.step(50)
        assert 70 < budget_50 < 80  # 约 75
        
        # 最终 (epoch 99)
        budget_99 = scheduler.step(99)
        assert abs(budget_99 - 50.0) < 0.1
    
    def test_linear_schedule(self):
        """Linear 调度验证"""
        scheduler = BudgetScheduler(
            budget_max=100.0,
            budget_min=50.0,
            total_epochs=100,
            schedule="linear",
        )
        
        budget_0 = scheduler.step(0)
        budget_50 = scheduler.step(50)
        budget_99 = scheduler.step(99)
        
        assert abs(budget_0 - 100.0) < 0.1
        assert abs(budget_50 - 75.0) < 1.0
        assert abs(budget_99 - 50.0) < 1.0


# ============================================================================
# Trainer 模块测试
# ============================================================================

from training.trainer import (
    TrainerConfig,
    TrainerState,
    Callback,
    CallbackContext,
    EarlyStoppingCallback,
    ModularTrainer,
)

from training.config import (
    ExperimentConfig,
    DataConfig,
    LossConfig,
    ConfigLoader,
)


class TestTrainerConfig:
    """Trainer 配置测试"""
    
    def test_default_config(self):
        """默认配置应该有效"""
        config = TrainerConfig()
        assert config.num_epochs == 100
        assert config.gradient_clip_norm == 1.0
        assert config.use_amp is True
    
    def test_custom_config(self):
        """自定义配置"""
        config = TrainerConfig(
            num_epochs=50,
            gradient_clip_norm=0.5,
            use_amp=False,
        )
        assert config.num_epochs == 50
        assert config.gradient_clip_norm == 0.5
        assert config.use_amp is False


class TestEarlyStopping:
    """早停回调测试"""
    
    def test_stops_after_patience(self):
        """超过 patience 后应该停止"""
        callback = EarlyStoppingCallback(
            monitor="val_accuracy",
            patience=3,
            mode="max",
        )
        
        # 模拟 5 个 epoch 无改善
        trainer = None  # 早停不需要 trainer
        
        for epoch in range(5):
            ctx = CallbackContext(
                epoch=epoch,
                batch_idx=0,
                global_step=epoch * 100,
                metrics={"val_accuracy": 0.5},  # 固定值，无改善
            )
            callback.on_epoch_end(trainer, ctx)
            
            if epoch < 3:
                assert not ctx.stop_training
            else:
                assert ctx.stop_training
                break
    
    def test_no_stop_when_improving(self):
        """持续改善时不应停止"""
        callback = EarlyStoppingCallback(
            monitor="val_accuracy",
            patience=3,
            mode="max",
        )
        
        for epoch in range(10):
            ctx = CallbackContext(
                epoch=epoch,
                batch_idx=0,
                global_step=epoch * 100,
                metrics={"val_accuracy": 0.5 + epoch * 0.01},  # 持续改善
            )
            callback.on_epoch_end(None, ctx)
            assert not ctx.stop_training


class TestExperimentConfig:
    """实验配置测试"""
    
    def test_default_experiment_config(self):
        """默认实验配置"""
        config = ExperimentConfig()
        assert config.name == "default_experiment"
        assert config.data.batch_size == 128
        assert config.loss.type == "cross_entropy"
    
    def test_nested_config_modification(self):
        """嵌套配置修改"""
        config = ExperimentConfig(
            name="test_experiment",
            data=DataConfig(batch_size=64, sampler_type="class_balanced"),
            loss=LossConfig(type="focal", focal_gamma=3.0),
        )
        assert config.data.batch_size == 64
        assert config.data.sampler_type == "class_balanced"
        assert config.loss.focal_gamma == 3.0


class TestConfigLoader:
    """配置加载器测试"""
    
    def test_from_dict(self):
        """从字典加载配置"""
        loader = ConfigLoader()
        config = loader.from_dict({
            "name": "test",
            "data": {"batch_size": 256},
            "loss": {"type": "focal", "focal_gamma": 2.5},
        })
        
        assert config.name == "test"
        assert config.data.batch_size == 256
        assert config.loss.type == "focal"
        assert config.loss.focal_gamma == 2.5


class TestModularTrainer:
    """模块化训练器集成测试"""
    
    def test_trainer_creation(self):
        """训练器创建"""
        # 简单模型
        model = torch.nn.Linear(10, 2)
        
        # 简单数据集
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10),
            torch.randint(0, 2, (100,))
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=10)
        
        # 创建训练器
        trainer = ModularTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=torch.optim.Adam(model.parameters()),
            loss_fn=torch.nn.CrossEntropyLoss(),
            config=TrainerConfig(
                num_epochs=1,
                use_amp=False,
                device="cpu",
            ),
        )
        
        assert trainer.model is model
        assert trainer.config.num_epochs == 1
    
    def test_single_epoch_training(self):
        """单轮训练执行"""
        model = torch.nn.Linear(10, 2)
        dataset = torch.utils.data.TensorDataset(
            torch.randn(50, 10),
            torch.randint(0, 2, (50,))
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=10)
        
        trainer = ModularTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=torch.optim.Adam(model.parameters()),
            loss_fn=torch.nn.CrossEntropyLoss(),
            config=TrainerConfig(
                num_epochs=1,
                use_amp=False,
                device="cpu",
            ),
        )
        
        # 执行单轮训练
        metrics = trainer.train_epoch()
        
        assert "train_loss" in metrics
        assert metrics["train_loss"] > 0
    
    def test_validation(self):
        """验证执行"""
        model = torch.nn.Linear(10, 2)
        dataset = torch.utils.data.TensorDataset(
            torch.randn(50, 10),
            torch.randint(0, 2, (50,))
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=10)
        
        trainer = ModularTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=torch.optim.Adam(model.parameters()),
            loss_fn=torch.nn.CrossEntropyLoss(),
            config=TrainerConfig(
                num_epochs=1,
                use_amp=False,
                device="cpu",
            ),
        )
        
        metrics = trainer.validate()
        
        assert "val_loss" in metrics
        assert metrics["val_loss"] > 0


# ============================================================================
# WandB Callback 测试
# ============================================================================

class TestWandBCallback:
    """WandB Callback 单元测试
    
    注意: 这些测试不需要实际的 wandb 连接，
    测试的是回调的初始化和配置逻辑。
    """
    
    def test_import_and_config(self):
        """验证模块可以正确导入"""
        from training.callbacks import (
            WandBCallbackConfig,
            WandBCallback,
            SplitterHealthConfig,
            SplitterHealthCallback,
        )
        
        # 创建配置
        config = WandBCallbackConfig(
            project="test-project",
            entity="test-entity",
            name="test-run",
            tags=["test", "unit-test"],
            enabled=False,  # 不实际连接
        )
        
        assert config.project == "test-project"
        assert config.entity == "test-entity"
        assert config.enabled is False
    
    def test_callback_creation_disabled(self):
        """禁用模式下创建回调"""
        from training.callbacks import WandBCallbackConfig, WandBCallback
        
        config = WandBCallbackConfig(enabled=False)
        callback = WandBCallback(config=config)
        
        # 回调应该可以创建但不会初始化 wandb
        assert callback._run is None
        assert callback.config.enabled is False
    
    def test_splitter_health_config(self):
        """分割器健康监控配置"""
        from training.callbacks import SplitterHealthConfig, SplitterHealthCallback
        
        config = SplitterHealthConfig(
            min_tokens=8,
            max_tokens=256,
            min_entropy_ratio=0.5,
        )
        
        callback = SplitterHealthCallback(config=config)
        
        assert callback.config.min_tokens == 8
        assert callback.config.max_tokens == 256
    
    def test_health_computation(self):
        """健康评分计算验证"""
        from training.callbacks import SplitterHealthConfig, SplitterHealthCallback
        
        config = SplitterHealthConfig(min_tokens=4, max_tokens=256)
        callback = SplitterHealthCallback(config=config)
        
        # 模拟 splitter 对象
        class MockSplitter:
            _last_perf_stats = {
                'soft_token_count': 32,
                'entropy_ratio': 0.8,
            }
        
        health = callback._compute_health(MockSplitter())
        
        assert health["avg_tokens"] == 32
        assert health["entropy_ratio"] == 0.8
        assert health["is_collapsed"] is False
        assert health["is_saturated"] is False
        assert 0 < health["health_score"] <= 1.0
    
    def test_health_collapse_detection(self):
        """崩溃检测"""
        from training.callbacks import SplitterHealthConfig, SplitterHealthCallback
        
        config = SplitterHealthConfig(min_tokens=8)
        callback = SplitterHealthCallback(config=config)
        
        class MockSplitter:
            _last_perf_stats = {'soft_token_count': 1.0, 'entropy_ratio': 0.1}
        
        health = callback._compute_health(MockSplitter())
        
        assert health["is_collapsed"] is True
        assert health["health_score"] < 0.2
    
    def test_wandb_config_from_main_module(self):
        """从主模块导入 WandB 组件"""
        from training import (
            WandBCallbackConfig,
            WandBCallback,
            SplitterHealthCallback,
        )
        
        # 验证导出正确
        assert WandBCallbackConfig is not None
        assert WandBCallback is not None
        assert SplitterHealthCallback is not None


class TestVisualization:
    """可视化模块单元测试"""
    
    def test_visualization_imports(self):
        """验证可视化模块导入"""
        from training.visualization import (
            plot_per_class_accuracy,
            plot_confusion_matrix,
            plot_token_distribution,
            plot_depth_distribution,
            VisualizationConfig,
            ExperimentVisualizer,
        )
        
        assert plot_per_class_accuracy is not None
        assert plot_confusion_matrix is not None
        assert plot_token_distribution is not None
        assert plot_depth_distribution is not None
        assert VisualizationConfig is not None
        assert ExperimentVisualizer is not None
    
    def test_visualization_config(self):
        """可视化配置测试"""
        from training.visualization import VisualizationConfig
        
        config = VisualizationConfig(
            save_dir="test_output",
            format="png",
            dpi=100,
            auto_save=False,
        )
        
        assert config.save_dir == "test_output"
        assert config.format == "png"
        assert config.dpi == 100
        assert config.auto_save is False
    
    def test_get_save_path(self):
        """保存路径生成测试"""
        from training.visualization import VisualizationConfig
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = VisualizationConfig(save_dir=tmpdir, format="pdf")
            
            path = config.get_save_path("test_plot", timestamp=False)
            
            assert path.suffix == ".pdf"
            assert "test_plot" in path.name
    
    def test_plot_per_class_accuracy(self):
        """Per-class accuracy 热图生成测试"""
        pytest.importorskip("seaborn")
        from training.visualization import plot_per_class_accuracy, VisualizationConfig
        import matplotlib.pyplot as plt
        
        # 模拟数据
        per_class_acc = {i: 0.5 + 0.05 * (i % 10) for i in range(20)}
        
        config = VisualizationConfig(auto_save=False)
        fig = plot_per_class_accuracy(per_class_acc, config=config)
        
        assert fig is not None
        assert len(fig.axes) > 0
        
        plt.close(fig)
    
    def test_plot_token_distribution(self):
        """Token 分布直方图测试"""
        pytest.importorskip("seaborn")
        from training.visualization import plot_token_distribution, VisualizationConfig
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 模拟 token 数据
        np.random.seed(42)
        token_counts = list(np.random.randint(10, 100, size=200))
        
        config = VisualizationConfig(auto_save=False)
        fig = plot_token_distribution(token_counts, n_min=20, n_max=80, config=config)
        
        assert fig is not None
        assert len(fig.axes) == 2  # 直方图 + 箱线图
        
        plt.close(fig)
    
    def test_plot_depth_distribution(self):
        """深度分布柱状图测试"""
        pytest.importorskip("seaborn")
        from training.visualization import plot_depth_distribution, VisualizationConfig
        import matplotlib.pyplot as plt
        
        # 模拟深度数据
        depth_counts = {0: 100, 1: 200, 2: 300, 3: 150, 4: 50}
        
        config = VisualizationConfig(auto_save=False)
        fig = plot_depth_distribution(depth_counts, config=config)
        
        assert fig is not None
        
        plt.close(fig)
    
    def test_plot_training_curves(self):
        """训练曲线测试"""
        pytest.importorskip("seaborn")
        from training.visualization import plot_training_curves, VisualizationConfig
        import matplotlib.pyplot as plt
        
        # 模拟训练历史
        history = {
            "train_loss": [1.0, 0.8, 0.6, 0.5, 0.4],
            "val_loss": [1.1, 0.9, 0.7, 0.65, 0.6],
            "train_accuracy": [0.3, 0.5, 0.6, 0.7, 0.75],
            "val_accuracy": [0.25, 0.45, 0.55, 0.6, 0.62],
        }
        
        config = VisualizationConfig(auto_save=False)
        fig = plot_training_curves(history, config=config)
        
        assert fig is not None
        
        plt.close(fig)
    
    def test_experiment_visualizer_creation(self):
        """ExperimentVisualizer 创建测试"""
        from training.visualization import ExperimentVisualizer, VisualizationConfig
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = ExperimentVisualizer(
                run_id="test_run",
                base_dir=tmpdir,
                config=VisualizationConfig(auto_save=False),
            )
            
            assert visualizer.run_id == "test_run"
            assert "visualizations" in str(visualizer.vis_dir)
    
    def test_visualization_from_main_module(self):
        """从主模块导入可视化组件"""
        from training import (
            VisualizationConfig,
            ExperimentVisualizer,
            plot_per_class_accuracy,
            plot_token_distribution,
        )
        
        assert VisualizationConfig is not None
        assert ExperimentVisualizer is not None
        assert plot_per_class_accuracy is not None
        assert plot_token_distribution is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

