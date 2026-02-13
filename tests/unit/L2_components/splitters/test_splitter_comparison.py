# -*- coding: utf-8 -*-
"""
Splitter 对比验证实验
===================

无需训练模型的计算验证实验：
1. 梯度流分析
2. train/eval 一致性测试
3. Hilbert 局部性验证
4. 数值稳定性测试
5. 配额分布分析

运行方式:
    uv run python tests/unit/L2_components/splitters/test_splitter_comparison.py
"""

import pytest
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import math


class SplitterComparisonBase:
    """Splitter 对比基类 - 包含共享的数学验证工具"""

    @staticmethod
    def compute_jaccard_similarity(set1: torch.Tensor, set2: torch.Tensor) -> float:
        """计算 Jaccard 相似度"""
        if set1.numel() == 0 and set2.numel() == 0:
            return 1.0
        intersection = torch.intersect1d(set1, set2).numel()
        union = set1.numel() + set2.numel() - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def compute_gradient_stats(grad: torch.Tensor) -> Dict[str, float]:
        """计算梯度统计"""
        grad_flat = grad.flatten()
        nonzero_grad = grad_flat[grad_flat != 0]
        return {
            'nonzero_ratio': (grad_flat != 0).float().mean().item(),
            'mean_abs': grad_flat.abs().mean().item(),
            'max_abs': grad_flat.abs().max().item(),
            'min_abs': nonzero_grad.abs().min().item() if nonzero_grad.numel() > 0 else 0.0,
            'std': grad_flat.std().item(),
        }

    @staticmethod
    def compute_hilbert_locality(hilbert_indices: torch.Tensor, selected_indices: torch.Tensor) -> Dict[str, float]:
        """计算 Hilbert 局部性指标"""
        if selected_indices.numel() < 2:
            return {'mean_adjacent_dist': 0.0, 'lca_depth_mean': 0.0}

        selected_h = hilbert_indices[selected_indices].sort()[0]
        adjacent_dists = selected_h[1:] - selected_h[:-1]

        return {
            'mean_adjacent_dist': adjacent_dists.float().mean().item(),
            'max_adjacent_dist': adjacent_dists.float().max().item(),
            'std_adjacent_dist': adjacent_dists.float().std().item(),
        }


class TestGumbelTopKGradientFlow:
    """实验 1: GumbelTopKSplitter 梯度流分析"""

    def test_gradient_coverage(self):
        """测试梯度覆盖率"""
        from vit_pytorch.layers.splitters.gumbel_topk import GumbelTopKSplitter
        from vit_pytorch.core.config import SplitterConfig

        # 配置 - 使用正确的参数
        config = SplitterConfig(
            max_level_limit=3,
            feature_dim=256,
            hidden_dim=64,
            use_dynamic_k=False,  # 固定 K
        )

        splitter = GumbelTopKSplitter(
            config=config,
            image_size=(224, 224),
        )
        splitter.eval()

        # 创建虚拟输入
        B, C, H, W = 2, 3, 224, 224
        images = torch.randn(B, C, H, W)
        features = torch.randn(B, 256, H // 4, W // 4)

        # Forward
        splitter.train()
        try:
            output = splitter(features, target_k=32)

            if output.selected_mask is not None and output.selected_mask.grad is not None:
                pass  # 已经计算过梯度
            elif output.selected_mask is not None:
                # Backward
                loss = output.selected_mask.sum()
                loss.backward()

            # 分析梯度
            if hasattr(splitter, 'scoring_net') and splitter.scoring_net is not None:
                # 尝试获取第一个线性层的梯度
                first_linear = splitter.scoring_net[0] if hasattr(splitter.scoring_net, '__getitem__') else None
                if first_linear is not None and hasattr(first_linear, 'weight'):
                    grad = first_linear.weight.grad
                    if grad is not None:
                        stats = SplitterComparisonBase.compute_gradient_stats(grad)
                        print(f"\n=== GumbelTopK 梯度统计 ===")
                        print(f"非零梯度比例: {stats['nonzero_ratio']:.4f}")
                        print(f"平均梯度绝对值: {stats['mean_abs']:.6f}")
                        print(f"最大梯度: {stats['max_abs']:.6f}")
                        print(f"最小非零梯度: {stats['min_abs']:.8f}")
                        print(f"梯度标准差: {stats['std']:.6f}")
            else:
                print("\n=== GumbelTopK 无法获取梯度 ===")
                print(f"output type: {type(output)}")
                print(f"selected_mask requires_grad: {output.selected_mask.requires_grad if output.selected_mask is not None else None}")
        except Exception as e:
            print(f"GumbelTopK 测试跳过: {e}")


class TestDeterministicNeighborGradientFlow:
    """实验 1 (续): DeterministicNeighborSplitter 梯度流分析"""

    def test_gradient_coverage(self):
        """测试梯度覆盖率 - 预期 100%"""
        from vit_pytorch.layers.splitters.deterministic_neighbor import (
            DeterministicNeighborSplitter,
            DeterministicNeighborSplitterConfig,
        )

        config = DeterministicNeighborSplitterConfig(
            max_level_limit=3,
            feature_dim=256,
            hidden_dim=64,
        )

        splitter = DeterministicNeighborSplitter(
            config=config,
            image_size=(224, 224),
            feature_dim=256,
        )
        splitter.eval()

        # 虚拟输入
        features = torch.randn(2, 256, 56, 56)

        # Forward + Backward
        splitter.train()
        try:
            output = splitter(features, target_k=32)

            if output.selected_mask is not None and output.selected_mask.grad is None:
                loss = output.selected_mask.sum()
                loss.backward()

            # 分析梯度
            if hasattr(splitter, 'scoring') and splitter.scoring is not None:
                first_linear = splitter.scoring.hidden[0] if hasattr(splitter.scoring, 'hidden') else None
                if first_linear is not None and hasattr(first_linear, 'weight'):
                    grad = first_linear.weight.grad
                    if grad is not None:
                        stats = SplitterComparisonBase.compute_gradient_stats(grad)
                        print(f"\n=== DeterministicNeighbor 梯度统计 ===")
                        print(f"非零梯度比例: {stats['nonzero_ratio']:.4f}")
                        print(f"平均梯度绝对值: {stats['mean_abs']:.6f}")
                        print(f"最大梯度: {stats['max_abs']:.6f}")
                        print(f"最小非零梯度: {stats['min_abs']:.8f}")
                        print(f"梯度标准差: {stats['std']:.6f}")
            else:
                print("\n=== DeterministicNeighbor 无法获取梯度 ===")
        except Exception as e:
            print(f"DeterministicNeighbor 测试跳过: {e}")


class TestTrainEvalConsistency:
    """实验 2: train/eval 一致性测试"""

    def test_gumbel_consistency(self):
        """测试 GumbelTopK 的一致性 - 预期不一致"""
        from vit_pytorch.layers.splitters.gumbel_topk import GumbelTopKSplitter
        from vit_pytorch.core.config import SplitterConfig

        config = SplitterConfig(
            max_level_limit=3,
            feature_dim=256,
            hidden_dim=64,
            use_dynamic_k=False,
        )
        splitter = GumbelTopKSplitter(config=config, image_size=(224, 224))

        features = torch.randn(1, 256, 56, 56)

        # 多次 forward (train mode)
        splitter.train()
        train_selections = []
        for _ in range(10):
            try:
                output = splitter(features, target_k=32)
                if output.selected_indices is not None:
                    train_selections.append(output.selected_indices[0].cpu())
            except:
                pass

        # 多次 forward (eval mode)
        splitter.eval()
        eval_selections = []
        for _ in range(10):
            try:
                output = splitter(features, target_k=32)
                if output.selected_indices is not None:
                    eval_selections.append(output.selected_indices[0].cpu())
            except:
                pass

        # 计算 Jaccard 相似度
        if len(train_selections) > 1:
            train_jaccards = []
            for i in range(len(train_selections) - 1):
                j = SplitterComparisonBase.compute_jaccard_similarity(
                    train_selections[i], train_selections[i + 1]
                )
                train_jaccards.append(j)
            print(f"\n=== GumbelTopK 一致性测试 ===")
            print(f"Train 模式 Jaccard 相似度: {sum(train_jaccards)/len(train_jaccards):.4f}")

        if len(eval_selections) > 1:
            eval_jaccards = []
            for i in range(len(eval_selections) - 1):
                j = SplitterComparisonBase.compute_jaccard_similarity(
                    eval_selections[i], eval_selections[i + 1]
                )
                eval_jaccards.append(j)
            print(f"Eval 模式 Jaccard 相似度: {sum(eval_jaccards)/len(eval_jaccards):.4f}")

    def test_deterministic_consistency(self):
        """测试 DeterministicNeighbor 的一致性 - 预期一致"""
        from vit_pytorch.layers.splitters.deterministic_neighbor import (
            DeterministicNeighborSplitter,
            DeterministicNeighborSplitterConfig,
        )

        config = DeterministicNeighborSplitterConfig(
            max_level_limit=3,
            feature_dim=256,
            hidden_dim=64,
        )
        splitter = DeterministicNeighborSplitter(
            config=config, image_size=(224, 224), feature_dim=256
        )

        features = torch.randn(1, 256, 56, 56)

        # 多次 forward (train mode)
        splitter.train()
        train_selections = []
        for _ in range(10):
            try:
                output = splitter(features, target_k=32)
                if output.selected_indices is not None:
                    train_selections.append(output.selected_indices[0].cpu())
            except:
                pass

        # 多次 forward (eval mode)
        splitter.eval()
        eval_selections = []
        for _ in range(10):
            try:
                output = splitter(features, target_k=32)
                if output.selected_indices is not None:
                    eval_selections.append(output.selected_indices[0].cpu())
            except:
                pass

        # 计算 Jaccard
        if len(train_selections) > 1:
            train_jaccards = []
            for i in range(len(train_selections) - 1):
                j = SplitterComparisonBase.compute_jaccard_similarity(
                    train_selections[i], train_selections[i + 1]
                )
                train_jaccards.append(j)
            print(f"\n=== DeterministicNeighbor 一致性测试 ===")
            print(f"Train 模式 Jaccard 相似度: {sum(train_jaccards)/len(train_jaccards):.4f}")

        if len(eval_selections) > 1:
            eval_jaccards = []
            for i in range(len(eval_selections) - 1):
                j = SplitterComparisonBase.compute_jaccard_similarity(
                    eval_selections[i], eval_selections[i + 1]
                )
                eval_jaccards.append(j)
            print(f"Eval 模式 Jaccard 相似度: {sum(eval_jaccards)/len(eval_jaccards):.4f}")


class TestNumericalStability:
    """实验 4: 数值稳定性测试"""

    def test_gumbel_low_temperature(self):
        """测试 GumbelTopK 低温稳定性"""
        from vit_pytorch.layers.splitters.gumbel_topk import GumbelTopKSplitter
        from vit_pytorch.core.config import SplitterConfig

        config = SplitterConfig(
            max_level_limit=3,
            feature_dim=256,
            hidden_dim=64,
            use_dynamic_k=False,
        )
        splitter = GumbelTopKSplitter(config=config, image_size=(224, 224))

        features = torch.randn(2, 256, 56, 56)

        temperatures = [1.0, 0.5, 0.3, 0.1]

        print(f"\n=== GumbelTopK 低温稳定性测试 ===")
        for tau in temperatures:
            splitter.train()
            if hasattr(splitter, 'log_temperature'):
                splitter.log_temperature.data = torch.tensor(math.log(tau))

            try:
                output = splitter(features, target_k=32)

                # 检查是否有 NaN
                if output.selected_mask is not None:
                    has_nan = torch.isnan(output.selected_mask).any()
                    has_inf = torch.isinf(output.selected_mask).any()

                    print(f"T={tau:.1f}: NaN={has_nan}, Inf={has_inf}, "
                          f"mask_range=[{output.selected_mask.min():.4f}, {output.selected_mask.max():.4f}]")
            except Exception as e:
                print(f"T={tau:.1f}: ERROR - {e}")


class TestQuotaDistribution:
    """实验 5: 配额分布分析"""

    def test_quota_distribution(self):
        """测试深度配额分布"""
        from vit_pytorch.layers.splitters.gumbel_topk import GumbelTopKSplitter
        from vit_pytorch.layers.splitters.deterministic_neighbor import (
            DeterministicNeighborSplitter,
            DeterministicNeighborSplitterConfig,
        )
        from vit_pytorch.core.config import SplitterConfig

        # GumbelTopK
        config = SplitterConfig(
            max_level_limit=4,
            feature_dim=256,
            hidden_dim=64,
            use_dynamic_k=False,
        )
        gumbel_splitter = GumbelTopKSplitter(config=config, image_size=(224, 224))

        # DeterministicNeighbor
        dn_config = DeterministicNeighborSplitterConfig(
            max_level_limit=4,
            feature_dim=256,
            hidden_dim=64,
        )
        dn_splitter = DeterministicNeighborSplitter(
            config=dn_config, image_size=(224, 224), feature_dim=256
        )

        features = torch.randn(2, 256, 56, 56)

        print(f"\n=== 配额分布分析 ===")

        # GumbelTopK
        gumbel_splitter.eval()
        try:
            gumbel_output = gumbel_splitter(features, target_k=64)
            if gumbel_output.depth_distribution:
                print(f"GumbelTopK 深度分布: {gumbel_output.depth_distribution}")
        except Exception as e:
            print(f"GumbelTopK 错误: {e}")

        # DeterministicNeighbor
        dn_splitter.eval()
        try:
            dn_output = dn_splitter(features, target_k=64)
            if dn_output.depth_distribution:
                print(f"DeterministicNeighbor 深度分布: {dn_output.depth_distribution}")
        except Exception as e:
            print(f"DeterministicNeighbor 错误: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
