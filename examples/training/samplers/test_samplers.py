# -*- coding: utf-8 -*-
"""
类别平衡采样器测试
"""

import pytest
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch

from samplers.balanced_sampler import ClassBalancedSampler, ProgressiveSampler


class TestClassBalancedSampler:
    """ClassBalancedSampler 单元测试"""
    
    def test_basic_functionality(self):
        """测试基本功能"""
        # 创建不平衡数据集: [0]*40 + [1]*6 + [2]*4
        targets = [0] * 40 + [1] * 6 + [2] * 4
        sampler = ClassBalancedSampler(targets, beta=0.9)
        
        assert len(sampler) == 50
        assert sampler.num_classes == 3
        assert sampler.beta == 0.9
    
    def test_boost_ratios(self):
        """测试尾部类别提升比例"""
        # 极度不平衡: [0]*80 + [1]*15 + [2]*5
        targets = [0] * 80 + [1] * 15 + [2] * 5
        sampler = ClassBalancedSampler(targets, beta=0.9)
        
        stats = sampler.get_statistics()
        
        # 尾部类别 (class 2) 的提升比例应该最大
        boost_ratios = stats["boost_ratios"]
        assert boost_ratios[2] > boost_ratios[1] > boost_ratios[0]
        
        # β=0.9 时，尾部类别提升应该显著 (>5x)
        assert boost_ratios[2] > 5.0
    
    def test_beta_extremes(self):
        """测试 beta 极端值"""
        targets = [0] * 50 + [1] * 30 + [2] * 20
        
        # β=0: 均匀采样
        sampler_uniform = ClassBalancedSampler(targets, beta=0.0)
        stats_uniform = sampler_uniform.get_statistics()
        
        # 采样概率应该接近原始分布
        np.testing.assert_allclose(
            stats_uniform["sampled_probs"],
            stats_uniform["original_probs"],
            rtol=0.01
        )
        
        # β=1: 完全逆频率
        sampler_inverse = ClassBalancedSampler(targets, beta=1.0)
        stats_inverse = sampler_inverse.get_statistics()
        
        # 提升比例应该 ∝ 1/n_c
        boost_ratios = np.array(stats_inverse["boost_ratios"])
        expected_boost = np.array([20/50, 20/30, 20/20])  # 归一化到最小类
        np.testing.assert_allclose(
            boost_ratios / boost_ratios.min(),
            expected_boost / expected_boost.min(),
            rtol=0.1
        )
    
    def test_sampling_distribution(self):
        """测试采样分布统计特性"""
        targets = [0] * 70 + [1] * 20 + [2] * 10
        sampler = ClassBalancedSampler(targets, beta=0.8, replacement=True)
        
        # 采样 10000 次
        samples = []
        for _ in range(100):  # 100 个 epoch
            samples.extend(list(sampler))
        
        samples = np.array(samples)
        sampled_targets = np.array(targets)[samples]
        
        # 统计各类别采样频率
        unique, counts = np.unique(sampled_targets, return_counts=True)
        empirical_probs = counts / counts.sum()
        
        # 应该接近理论采样概率
        stats = sampler.get_statistics()
        np.testing.assert_allclose(
            empirical_probs,
            stats["sampled_probs"],
            atol=0.02  # 允许 2% 误差
        )
    
    def test_invalid_beta(self):
        """测试非法 beta 值"""
        targets = [0] * 10 + [1] * 10
        
        with pytest.raises(ValueError):
            ClassBalancedSampler(targets, beta=-0.1)
        
        with pytest.raises(ValueError):
            ClassBalancedSampler(targets, beta=1.5)


class TestProgressiveSampler:
    """ProgressiveSampler 单元测试"""
    
    def test_beta_schedule(self):
        """测试 beta 调度"""
        targets = [0] * 50 + [1] * 30 + [2] * 20
        sampler = ProgressiveSampler(
            targets,
            beta_start=0.9,
            beta_end=0.3,
            total_epochs=100
        )
        
        # Epoch 0: β = 0.9
        sampler.set_epoch(0)
        assert abs(sampler.beta - 0.9) < 1e-6
        
        # Epoch 50: β = 0.6 (中点)
        sampler.set_epoch(50)
        assert abs(sampler.beta - 0.6) < 1e-6
        
        # Epoch 100: β = 0.3
        sampler.set_epoch(100)
        assert abs(sampler.beta - 0.3) < 1e-6
    
    def test_boost_ratio_decay(self):
        """测试提升比例随训练衰减"""
        targets = [0] * 80 + [1] * 15 + [2] * 5
        sampler = ProgressiveSampler(
            targets,
            beta_start=0.9,
            beta_end=0.1,
            total_epochs=100
        )
        
        # 记录尾部类别的提升比例
        boost_ratios_over_time = []
        for epoch in [0, 25, 50, 75, 100]:
            sampler.set_epoch(epoch)
            stats = sampler.get_statistics()
            boost_ratios_over_time.append(stats["boost_ratios"][2])  # 最尾部类
        
        # 提升比例应该单调递减
        assert all(
            boost_ratios_over_time[i] > boost_ratios_over_time[i+1]
            for i in range(len(boost_ratios_over_time) - 1)
        )
        
        # 起始提升应该很高，最终提升应该接近 1
        assert boost_ratios_over_time[0] > 5.0
        assert boost_ratios_over_time[-1] < 2.0
    
    def test_schedule_info(self):
        """测试调度信息导出"""
        targets = [0] * 50 + [1] * 50
        sampler = ProgressiveSampler(targets, total_epochs=10)
        
        schedule_info = sampler.get_schedule_info()
        
        assert "epochs" in schedule_info
        assert "betas" in schedule_info
        assert len(schedule_info["epochs"]) == 11  # 0~10
        assert len(schedule_info["betas"]) == 11


def test_integration_with_dataloader():
    """测试与 DataLoader 集成"""
    # 创建模拟数据集
    num_samples = 100
    features = torch.randn(num_samples, 10)
    targets = torch.cat([
        torch.zeros(70, dtype=torch.long),   # 类别 0: 70
        torch.ones(20, dtype=torch.long),    # 类别 1: 20
        torch.full((10,), 2, dtype=torch.long)  # 类别 2: 10
    ])
    
    dataset = TensorDataset(features, targets)
    
    # 使用 ClassBalancedSampler
    sampler = ClassBalancedSampler(targets.tolist(), beta=0.8)
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)
    
    # 测试迭代
    batch_count = 0
    for batch_features, batch_targets in dataloader:
        assert batch_features.shape[0] <= 10
        assert batch_targets.shape[0] <= 10
        batch_count += 1
    
    assert batch_count == 10  # 100 samples / batch_size 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
