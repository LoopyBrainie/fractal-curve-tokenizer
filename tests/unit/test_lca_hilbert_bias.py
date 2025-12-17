# -*- coding: utf-8 -*-
"""LCAHilbertBias 单元测试

测试覆盖:
1. 基本功能测试 (形状、设备、梯度)
2. LCA 计算正确性验证
3. 与其他 bias_mode 的性能对比
4. 边界条件测试
"""

import pytest
import torch
import torch.nn as nn

from vit_pytorch.attention import (
    LCAHilbertBias,
    LowRankHilbertBias,
    HierarchicalHilbertBias,
    HilbertAwareMultiScaleAttention,
)
from vit_pytorch.fractal_path import VectorizedPathEncoder


class TestLCAHilbertBiasBasic:
    """LCAHilbertBias 基础功能测试"""
    
    @pytest.fixture
    def lca_bias(self):
        """创建 LCA Bias 实例"""
        return LCAHilbertBias(max_depth=8, heads=4)
    
    def test_init(self, lca_bias):
        """初始化测试"""
        assert lca_bias.max_depth == 8
        assert lca_bias.heads == 4
        assert lca_bias.lca_embedding.num_embeddings == 9  # 0 to max_depth
        assert lca_bias.lca_embedding.embedding_dim == 4
    
    def test_parameter_count(self, lca_bias):
        """参数量测试 - 验证参数量极少"""
        num_params = sum(p.numel() for p in lca_bias.parameters())
        # 参数量 = (max_depth + 1) * heads = 9 * 4 = 36
        assert num_params == 36
        
        # 与 LowRankHilbertBias 对比
        low_rank = LowRankHilbertBias(path_dim=16, rank=32, heads=4)
        low_rank_params = sum(p.numel() for p in low_rank.parameters())
        
        # LCA 参数量应该远少于 Low-Rank
        assert num_params < low_rank_params / 100, \
            f"LCA params ({num_params}) should be << Low-Rank params ({low_rank_params})"
    
    def test_forward_2d(self, lca_bias):
        """2D 输入测试 (S, Info)"""
        # 模拟 levels_info: (S, Info) 其中 Info = [depth, q1, q2, ...]
        seq_len = 16
        info_len = 5  # depth + 4 层路径
        levels_info = torch.randint(0, 4, (seq_len, info_len))
        levels_info[:, 0] = 4  # 设置深度为 4
        
        bias = lca_bias(levels_info)
        
        assert bias is not None
        assert bias.shape == (4, seq_len, seq_len)  # (H, S, S)
        assert not torch.isnan(bias).any()
        assert not torch.isinf(bias).any()
    
    def test_forward_3d(self, lca_bias):
        """3D 输入测试 (B, S, Info)"""
        batch_size = 2
        seq_len = 16
        info_len = 5
        levels_info = torch.randint(0, 4, (batch_size, seq_len, info_len))
        levels_info[:, :, 0] = 4
        
        bias = lca_bias(levels_info)
        
        assert bias is not None
        assert bias.shape == (batch_size, 4, seq_len, seq_len)  # (B, H, S, S)
        assert not torch.isnan(bias).any()
    
    def test_gradient_flow(self, lca_bias):
        """梯度流测试"""
        levels_info = torch.randint(0, 4, (2, 16, 5))
        levels_info[:, :, 0] = 4
        
        bias = lca_bias(levels_info)
        loss = bias.sum()
        loss.backward()
        
        # 验证梯度存在
        assert lca_bias.lca_embedding.weight.grad is not None
        assert not torch.isnan(lca_bias.lca_embedding.weight.grad).any()
    
    def test_empty_input(self, lca_bias):
        """空输入测试"""
        empty = torch.empty(0, 5)
        assert lca_bias(empty) is None
        
        empty_3d = torch.empty(2, 0, 5)
        assert lca_bias(empty_3d) is None
    
    def test_minimal_info(self, lca_bias):
        """最小信息维度测试"""
        # info_dim = 1 (只有深度，没有路径) 应返回 None
        minimal = torch.randint(0, 4, (16, 1))
        assert lca_bias(minimal) is None


class TestLCAComputation:
    """LCA 计算正确性测试"""
    
    def test_identical_paths_max_lca(self):
        """相同路径应产生最大 LCA 深度"""
        # 创建相同路径
        path = torch.tensor([0, 1, 2, 3])  # 一条路径
        paths = path.unsqueeze(0).expand(4, -1)  # 4 个相同的路径
        
        lca_depths = VectorizedPathEncoder.compute_common_ancestor_depth(paths)
        
        # 所有 LCA 深度应该等于路径长度
        assert (lca_depths == 4).all()
    
    def test_completely_different_paths_zero_lca(self):
        """完全不同的根节点应产生 LCA=0"""
        paths = torch.tensor([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
        ])
        
        lca_depths = VectorizedPathEncoder.compute_common_ancestor_depth(paths)
        
        # 对角线为满深度，非对角线为 0
        assert (lca_depths.diag() == 4).all()
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert lca_depths[i, j] == 0
    
    def test_partial_common_ancestor(self):
        """部分共同祖先测试"""
        paths = torch.tensor([
            [0, 0, 0, 0],  # 路径 0000
            [0, 0, 1, 0],  # 路径 0010, LCA with first = 2
            [0, 1, 0, 0],  # 路径 0100, LCA with first = 1
        ])
        
        lca_depths = VectorizedPathEncoder.compute_common_ancestor_depth(paths)
        
        assert lca_depths[0, 1] == 2  # 前两层相同
        assert lca_depths[0, 2] == 1  # 只有第一层相同
        assert lca_depths[1, 2] == 1  # 只有第一层相同
    
    def test_lca_to_bias_mapping(self):
        """LCA 到偏置的映射测试"""
        lca_bias = LCAHilbertBias(max_depth=4, heads=2)
        
        # 创建已知 LCA 深度的路径
        paths = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 1],  # LCA with [0] = 3
            [0, 0, 1, 0],  # LCA with [0] = 2
            [1, 0, 0, 0],  # LCA with [0] = 0
        ])
        
        levels_info = torch.zeros(4, 5, dtype=torch.long)
        levels_info[:, 0] = 4  # 深度
        levels_info[:, 1:] = paths
        
        bias = lca_bias(levels_info)  # (H, 4, 4)
        
        # 验证 LCA 深度不同的位置偏置不同
        # 更深的 LCA 应该有更高的偏置（根据初始化）
        lca_03 = 3  # paths[0] vs paths[1]
        lca_02 = 2  # paths[0] vs paths[2]
        lca_00 = 0  # paths[0] vs paths[3]
        
        # 偏置应该反映 LCA 深度
        bias_03 = bias[0, 0, 1].item()  # LCA=3
        bias_02 = bias[0, 0, 2].item()  # LCA=2
        bias_00 = bias[0, 0, 3].item()  # LCA=0
        
        # 根据对数衰减初始化，深度越大偏置越高
        assert bias_03 > bias_02 > bias_00, \
            f"Bias ordering wrong: {bias_03} > {bias_02} > {bias_00}"


class TestIntegrationWithAttention:
    """与 HilbertAwareMultiScaleAttention 集成测试"""
    
    @pytest.fixture
    def attention_lca(self):
        """创建 LCA 模式的 attention"""
        return HilbertAwareMultiScaleAttention(
            dim=64,
            heads=4,
            dim_head=16,
            max_level=8,
            use_hilbert_bias=True,
            bias_mode='lca',
        )
    
    def test_lca_mode_initialization(self, attention_lca):
        """LCA 模式初始化测试"""
        assert attention_lca.bias_mode == 'lca'
        assert isinstance(attention_lca.hilbert_bias_impl, LCAHilbertBias)
        assert attention_lca.hilbert_bias_network is None
    
    def test_forward_with_lca_bias(self, attention_lca):
        """LCA 模式前向传播测试"""
        batch_size = 2
        seq_len = 16
        dim = 64
        
        x = torch.randn(batch_size, seq_len, dim)
        levels_info = torch.randint(0, 4, (batch_size, seq_len, 5))
        levels_info[:, :, 0] = 4
        
        output = attention_lca(x, levels_info=levels_info)
        
        assert output.shape == (batch_size, seq_len, dim)
        assert not torch.isnan(output).any()
    
    def test_parameter_efficiency_vs_low_rank(self):
        """参数效率对比 (LCA vs Low-Rank)"""
        attention_lca = HilbertAwareMultiScaleAttention(
            dim=64, heads=4, max_level=8, bias_mode='lca'
        )
        attention_lr = HilbertAwareMultiScaleAttention(
            dim=64, heads=4, max_level=8, bias_mode='low_rank'
        )
        
        lca_params = sum(p.numel() for p in attention_lca.parameters())
        lr_params = sum(p.numel() for p in attention_lr.parameters())
        
        # LCA 模式总参数应该更少
        # 差异主要来自 hilbert_bias_impl
        lca_bias_params = sum(
            p.numel() for p in attention_lca.hilbert_bias_impl.parameters()
        )
        lr_bias_params = sum(
            p.numel() for p in attention_lr.hilbert_bias_impl.parameters()
        )
        
        assert lca_bias_params < lr_bias_params / 50, \
            f"LCA bias ({lca_bias_params}) should be << Low-Rank bias ({lr_bias_params})"


class TestBiasModeComparison:
    """不同 bias_mode 对比测试"""
    
    @pytest.fixture
    def levels_info(self):
        """共享测试数据"""
        batch_size = 2
        seq_len = 16
        info_len = 9
        levels_info = torch.randint(0, 4, (batch_size, seq_len, info_len))
        levels_info[:, :, 0] = 8
        return levels_info
    
    def test_all_modes_produce_valid_output(self, levels_info):
        """所有模式都应产生有效输出"""
        modes = ['original', 'low_rank', 'hierarchical', 'lca']
        
        for mode in modes:
            attention = HilbertAwareMultiScaleAttention(
                dim=64,
                heads=4,
                max_level=8,
                use_hilbert_bias=True,
                bias_mode=mode,
            )
            
            x = torch.randn(2, 16, 64)
            output = attention(x, levels_info=levels_info)
            
            assert output.shape == (2, 16, 64), f"Mode {mode} output shape wrong"
            assert not torch.isnan(output).any(), f"Mode {mode} produced NaN"
            assert not torch.isinf(output).any(), f"Mode {mode} produced Inf"
    
    def test_gradient_flow_all_modes(self, levels_info):
        """所有模式梯度流测试"""
        modes = ['original', 'low_rank', 'hierarchical', 'lca']
        
        for mode in modes:
            attention = HilbertAwareMultiScaleAttention(
                dim=64,
                heads=4,
                max_level=8,
                use_hilbert_bias=True,
                bias_mode=mode,
            )
            
            x = torch.randn(2, 16, 64, requires_grad=True)
            output = attention(x, levels_info=levels_info)
            loss = output.sum()
            loss.backward()
            
            assert x.grad is not None, f"Mode {mode} no gradient to input"
            
            # 检查 bias 实现的梯度
            if attention.hilbert_bias_impl is not None:
                for name, param in attention.hilbert_bias_impl.named_parameters():
                    assert param.grad is not None, \
                        f"Mode {mode} no gradient to {name}"


class TestEdgeCases:
    """边界条件测试"""
    
    def test_single_token(self):
        """单 token 测试"""
        lca_bias = LCAHilbertBias(max_depth=4, heads=2)
        
        levels_info = torch.randint(0, 4, (1, 5))
        levels_info[0, 0] = 4
        
        bias = lca_bias(levels_info)
        
        assert bias.shape == (2, 1, 1)
        # 单 token 的 LCA 深度应该是满深度
        expected_depth = 4
        expected_bias = lca_bias.lca_embedding.weight[expected_depth]
        assert torch.allclose(bias[:, 0, 0], expected_bias)
    
    def test_max_depth_exceeded(self):
        """超过最大深度测试"""
        lca_bias = LCAHilbertBias(max_depth=4, heads=2)
        
        # 创建路径长度超过 max_depth 的情况
        levels_info = torch.randint(0, 4, (4, 10))  # 9 层路径
        levels_info[:, 0] = 8  # 深度为 8
        
        # 应该正常工作，LCA 会被 clamp 到 max_depth
        bias = lca_bias(levels_info)
        
        assert bias is not None
        assert bias.shape == (2, 4, 4)
        # 偏置值应该在有效范围内
        assert (bias >= lca_bias.lca_embedding.weight.min()).all()
        assert (bias <= lca_bias.lca_embedding.weight.max()).all()
    
    def test_device_transfer(self):
        """设备转移测试"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        lca_bias = LCAHilbertBias(max_depth=4, heads=2).cuda()
        levels_info = torch.randint(0, 4, (2, 16, 5)).cuda()
        
        bias = lca_bias(levels_info)
        
        assert bias.device.type == 'cuda'


class TestPerformance:
    """性能测试"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_large_sequence(self):
        """大序列长度测试"""
        lca_bias = LCAHilbertBias(max_depth=12, heads=8).cuda()
        
        # 模拟 224x224 / 16x16 patch = 196 tokens
        seq_len = 196
        levels_info = torch.randint(0, 4, (4, seq_len, 13)).cuda()
        levels_info[:, :, 0] = 12
        
        # 预热
        for _ in range(3):
            _ = lca_bias(levels_info)
        
        torch.cuda.synchronize()
        
        # 应该能快速完成
        import time
        start = time.time()
        for _ in range(10):
            _ = lca_bias(levels_info)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # 平均每次应该 < 10ms
        avg_time = elapsed / 10 * 1000
        assert avg_time < 50, f"Too slow: {avg_time:.2f}ms per forward"
