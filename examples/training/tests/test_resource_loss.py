# -*- coding: utf-8 -*-
"""
单元测试：资源统计和资源损失

测试覆盖
========
1. 资源统计接口
   - 深度加权 token 数计算
   - FLOPS 计算
   - 深度熵计算

2. 资源损失函数
   - FLOPS 约束损失
   - Token 数量约束损失
   - 深度熵正则损失
   - 损失组合

3. 边界情况
   - 空深度分布
   - 单一深度
   - 极端参数值
"""

import unittest
import numpy as np
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from examples.training.core.resource_stats import ModelResourceStats
from examples.training.losses.resource_loss import (
    ResourceAwareLoss,
    compute_depth_entropy,
    get_weighted_token_count,
)


class TestDepthEntropy(unittest.TestCase):
    """测试深度熵计算"""
    
    def test_uniform_distribution_maximum_entropy(self):
        """均匀分布应有最大熵"""
        # 5 个深度，均匀分布
        dist = torch.ones(5) * 20.0  # 每个深度 20 tokens
        H = compute_depth_entropy(dist)
        
        # 理论最大熵: log(5)
        H_max_theory = np.log(5)
        
        self.assertAlmostEqual(H.item(), H_max_theory, places=4,
                               msg="均匀分布应达到最大熵")
    
    def test_collapsed_distribution_zero_entropy(self):
        """坍缩分布应有零熵"""
        # 所有 token 在深度 0
        dist = torch.tensor([100.0, 0.0, 0.0, 0.0, 0.0])
        H = compute_depth_entropy(dist)
        
        self.assertAlmostEqual(H.item(), 0.0, places=4,
                               msg="坍缩分布应有零熵")
    
    def test_binary_distribution_known_entropy(self):
        """二分布的熵可以手动计算验证"""
        # 60-40 分布
        p1, p2 = 0.6, 0.4
        dist = torch.tensor([60.0, 40.0])
        H = compute_depth_entropy(dist)
        
        # 理论值: H = -0.6*log(0.6) - 0.4*log(0.4)
        H_theory = -(p1 * np.log(p1) + p2 * np.log(p2))
        
        self.assertAlmostEqual(H.item(), H_theory, places=4)
    
    def test_entropy_range(self):
        """熵应该在合理范围内"""
        for _ in range(10):
            # 随机分布
            dist = torch.rand(5) * 100
            dist = dist / dist.sum() * 100  # 归一化到总和 100
            
            H = compute_depth_entropy(dist)
            H_max = np.log(5)
            
            self.assertGreaterEqual(H.item(), 0.0)
            self.assertLessEqual(H.item(), H_max + 0.01)  # 允许小误差


class TestWeightedTokenCount(unittest.TestCase):
    """测试深度加权 token 数计算"""
    
    def test_zero_alpha_equals_total(self):
        """α=0 时应等于总 token 数"""
        dist = torch.tensor([50.0, 30.0, 20.0])
        weighted = get_weighted_token_count(dist, alpha=0.0)
        
        self.assertAlmostEqual(weighted.item(), 100.0, places=4,
                               msg="α=0 时加权数应等于总数")
    
    def test_positive_alpha_increases_weight(self):
        """正 α 应增加深层 token 的权重"""
        shallow_dist = torch.tensor([50.0, 30.0, 20.0])  # 浅层为主
        deep_dist = torch.tensor([20.0, 30.0, 50.0])     # 深层为主
        
        weighted_shallow = get_weighted_token_count(shallow_dist, alpha=0.1)
        weighted_deep = get_weighted_token_count(deep_dist, alpha=0.1)
        
        self.assertGreater(weighted_deep.item(), weighted_shallow.item(),
                           msg="深层分布应有更高的加权数")
    
    def test_alpha_magnitude_effect(self):
        """更大的 α 应产生更大的惩罚"""
        dist = torch.tensor([20.0, 30.0, 50.0])  # 深层为主
        
        weighted_01 = get_weighted_token_count(dist, alpha=0.1)
        weighted_02 = get_weighted_token_count(dist, alpha=0.2)
        
        self.assertGreater(weighted_02.item(), weighted_01.item(),
                           msg="更大的 α 应产生更大的加权数")
    
    def test_manual_calculation(self):
        """手动计算验证"""
        dist = torch.tensor([50.0, 30.0, 20.0])
        alpha = 0.1
        
        # 手动计算
        expected = (50.0 * np.exp(0.0) +
                    30.0 * np.exp(0.1) +
                    20.0 * np.exp(0.2))
        
        weighted = get_weighted_token_count(dist, alpha=alpha)
        
        self.assertAlmostEqual(weighted.item(), expected, places=2)


class TestFLOPSLoss(unittest.TestCase):
    """测试 FLOPS 约束损失"""
    
    def setUp(self):
        self.loss_fn = ResourceAwareLoss(
            flops_budget=5e9,
            lambda_flops=1.0,
            lambda_token=0.0,
            lambda_entropy=0.0,
        )
    
    def test_below_budget_zero_loss(self):
        """低于预算应无损失"""
        stats = ModelResourceStats(total_flops=4e9)
        loss = self.loss_fn(stats)
        
        self.assertAlmostEqual(loss.item(), 0.0, places=5,
                               msg="低于预算应无 FLOPS 损失")
    
    def test_at_budget_zero_loss(self):
        """恰好预算应无损失"""
        stats = ModelResourceStats(total_flops=5e9)
        loss = self.loss_fn(stats)
        
        self.assertAlmostEqual(loss.item(), 0.0, places=5)
    
    def test_over_budget_nonzero_loss(self):
        """超预算应有损失"""
        stats = ModelResourceStats(total_flops=6e9)
        loss = self.loss_fn(stats)
        
        # L = (6e9/5e9 - 1)² = 0.2² = 0.04
        expected_loss = 0.04
        
        self.assertAlmostEqual(loss.item(), expected_loss, places=4)
    
    def test_quadratic_penalty(self):
        """损失应呈二次增长"""
        stats_20 = ModelResourceStats(total_flops=6e9)  # +20%
        stats_50 = ModelResourceStats(total_flops=7.5e9)  # +50%
        
        loss_20 = self.loss_fn(stats_20).item()
        loss_50 = self.loss_fn(stats_50).item()
        
        # 二次关系: loss_50 / loss_20 = (0.5/0.2)² = 6.25
        ratio = loss_50 / loss_20
        self.assertAlmostEqual(ratio, 6.25, places=2)


class TestTokenLoss(unittest.TestCase):
    """测试 Token 数量约束损失"""
    
    def setUp(self):
        self.loss_fn = ResourceAwareLoss(
            token_budget=100,
            depth_weight_alpha=0.1,
            lambda_flops=0.0,
            lambda_token=1.0,
            lambda_entropy=0.0,
        )
    
    def test_below_budget_zero_loss(self):
        """低于预算应无损失"""
        # 浅层分布: 计算加权值
        dist = [50, 30, 10, 0, 0]
        # weighted = 50*1.0 + 30*1.105 + 10*1.221 ≈ 95.4 < 100
        stats = ModelResourceStats(token_depth_distribution=dist)
        loss = self.loss_fn(stats)
        
        # 95.4 < 100 预算，应该无损失
        self.assertLess(loss.item(), 0.01,
                        msg="低于预算应无 token 损失")
    
    def test_depth_weighted_penalty(self):
        """深层分布应受更大惩罚"""
        shallow_stats = ModelResourceStats(token_depth_distribution=[50, 30, 20])
        deep_stats = ModelResourceStats(token_depth_distribution=[20, 30, 50])
        
        loss_shallow = self.loss_fn(shallow_stats).item()
        loss_deep = self.loss_fn(deep_stats).item()
        
        self.assertGreater(loss_deep, loss_shallow,
                           msg="深层分布应有更大的 token 损失")


class TestEntropyLoss(unittest.TestCase):
    """测试深度熵正则损失"""
    
    def setUp(self):
        self.loss_fn = ResourceAwareLoss(
            lambda_flops=0.0,
            lambda_token=0.0,
            lambda_entropy=1.0,
            target_entropy_ratio=1.0,  # 要求均匀分布
        )
    
    def test_uniform_distribution_zero_loss(self):
        """均匀分布应无熵损失"""
        stats = ModelResourceStats(token_depth_distribution=[20, 20, 20, 20, 20])
        loss = self.loss_fn(stats)
        
        self.assertLess(loss.item(), 0.01,
                        msg="均匀分布应有接近零的熵损失")
    
    def test_collapsed_distribution_high_loss(self):
        """坍缩分布应有高熵损失"""
        stats = ModelResourceStats(token_depth_distribution=[100, 0, 0, 0, 0])
        loss = self.loss_fn(stats)
        
        # H_actual = 0, H_target = log(5) ≈ 1.609
        # L = (0 - 1.609)² ≈ 2.59
        expected_loss = np.log(5) ** 2
        
        self.assertAlmostEqual(loss.item(), expected_loss, places=2)
    
    def test_partial_collapse(self):
        """部分坍缩应有中等损失"""
        # 两个深度各 50%
        stats = ModelResourceStats(token_depth_distribution=[50, 50, 0, 0, 0])
        loss = self.loss_fn(stats)
        
        # H_actual = log(2) ≈ 0.693
        # H_target = log(5) ≈ 1.609
        # L = (0.693 - 1.609)² ≈ 0.84
        H_actual = np.log(2)
        H_target = np.log(5)
        expected_loss = (H_actual - H_target) ** 2
        
        self.assertAlmostEqual(loss.item(), expected_loss, places=2)


class TestCombinedLoss(unittest.TestCase):
    """测试损失组合"""
    
    def test_all_components(self):
        """测试所有损失项同时工作"""
        loss_fn = ResourceAwareLoss(
            flops_budget=5e9,
            token_budget=100,
            depth_weight_alpha=0.1,
            lambda_flops=0.1,
            lambda_token=0.01,
            lambda_entropy=0.05,
        )
        
        stats = ModelResourceStats(
            total_flops=6e9,  # 超预算 20%
            token_depth_distribution=[50, 30, 20, 10, 5],
            depth_entropy=1.2,
        )
        
        loss, components = loss_fn(stats, return_components=True)
        
        # 验证所有分量都非零
        self.assertGreater(components["L_flops"], 0.0)
        self.assertGreater(components["L_token"], 0.0)
        self.assertGreater(components["L_entropy"], 0.0)
        
        # 验证总损失是加权和
        expected_total = (
            0.1 * components["L_flops"] +
            0.01 * components["L_token"] +
            0.05 * components["L_entropy"]
        )
        
        self.assertAlmostEqual(loss.item(), expected_total, places=5)
    
    def test_diagnostics(self):
        """测试诊断信息"""
        loss_fn = ResourceAwareLoss(flops_budget=5e9, token_budget=128)
        
        stats = ModelResourceStats(
            total_flops=6e9,
            token_depth_distribution=[50, 30, 20],
        )
        
        diagnostics = loss_fn.get_diagnostics(stats)
        
        # 验证诊断字段存在
        self.assertIn("flops_usage_percent", diagnostics)
        self.assertIn("token_usage_percent", diagnostics)
        self.assertIn("depth_distribution_percent", diagnostics)
        
        # 验证百分比计算
        self.assertAlmostEqual(diagnostics["flops_usage_percent"], 120.0, places=1)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def test_empty_depth_distribution(self):
        """空深度分布应能处理"""
        loss_fn = ResourceAwareLoss()
        stats = ModelResourceStats(token_depth_distribution=[])
        
        # 不应抛出异常
        loss = loss_fn(stats)
        self.assertIsInstance(loss, torch.Tensor)
    
    def test_single_depth(self):
        """单一深度应能处理"""
        loss_fn = ResourceAwareLoss()
        stats = ModelResourceStats(token_depth_distribution=[100])
        
        # 应跳过熵损失（无法计算）
        loss = loss_fn(stats)
        self.assertIsInstance(loss, torch.Tensor)
    
    def test_zero_tokens(self):
        """零 token 应能处理"""
        loss_fn = ResourceAwareLoss()
        stats = ModelResourceStats(
            avg_tokens_per_image=0.0,
            token_depth_distribution=[0, 0, 0],
        )
        
        loss = loss_fn(stats)
        self.assertIsInstance(loss, torch.Tensor)
    
    def test_extreme_alpha(self):
        """极端 alpha 值应能处理"""
        dist = torch.tensor([50.0, 30.0, 20.0])
        
        # α = 0
        w0 = get_weighted_token_count(dist, alpha=0.0)
        self.assertAlmostEqual(w0.item(), 100.0, places=4)
        
        # α = 1.0
        w1 = get_weighted_token_count(dist, alpha=1.0)
        self.assertGreater(w1.item(), 100.0)  # 应该显著增加


class TestGradientFlow(unittest.TestCase):
    """测试梯度流"""
    
    def test_flops_loss_gradient(self):
        """FLOPS 损失应有梯度"""
        # 创建可微分的 total_flops (模拟从模型参数计算)
        flops_param = torch.tensor(6e9, requires_grad=True)
        
        loss_fn = ResourceAwareLoss(
            flops_budget=5e9,
            lambda_flops=1.0,
            lambda_token=0.0,
            lambda_entropy=0.0,
        )
        
        # 手动计算损失
        flops_ratio = flops_param / 5e9
        loss = F.relu(flops_ratio - 1.0) ** 2
        
        loss.backward()
        
        self.assertIsNotNone(flops_param.grad)
        self.assertNotEqual(flops_param.grad.item(), 0.0)
    
    def test_token_loss_gradient(self):
        """Token 损失应有梯度"""
        # 创建可微分的深度分布
        depth_dist = torch.tensor([50.0, 30.0, 20.0], requires_grad=True)
        
        weighted = get_weighted_token_count(depth_dist, alpha=0.1)
        loss = F.relu(weighted - 100.0) ** 2
        
        loss.backward()
        
        self.assertIsNotNone(depth_dist.grad)
        # 验证梯度不全为零
        self.assertGreater(depth_dist.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    # 运行所有测试
    unittest.main(verbosity=2)
