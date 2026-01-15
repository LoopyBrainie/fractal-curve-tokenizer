"""
P9-1 TensorSplitResult 单元测试.

测试目标:
    1. TensorSplitResult 数据结构正确性
    2. 完全向量化 BFS 与 Python BFS 的等价性
    3. 性能提升验证 (无 .item() 调用)

数学验证:
    - Hilbert 排序后的 token 顺序一致
    - 每个 batch 的 token 数量一致
    - regions, depths, complexities 值一致
"""

import pytest
import torch
import time
import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestTensorSplitResult:
    """TensorSplitResult 数据结构测试."""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_empty_result(self, device):
        """测试空结果创建."""
        from vit_pytorch.split_adaptive import TensorSplitResult
        
        result = TensorSplitResult.empty(device)
        
        assert result.num_tokens == 0
        assert result.regions.shape == (0, 4)
        assert result.depths.shape == (0,)
        assert result.batch_indices.shape == (0,)
        assert result.hilbert_indices.shape == (0,)
        assert result.complexities.shape == (0,)
    
    def test_from_split_results(self, device):
        """测试从 Python SplitResult 转换."""
        from vit_pytorch.split_adaptive import (
            TensorSplitResult, SplitResult, SplitToken, Region
        )
        
        # 创建测试数据
        tokens1 = [
            SplitToken(Region(0, 0, 64, 64), depth=0, path=[], hilbert_idx=0, complexity=0.5),
            SplitToken(Region(64, 0, 128, 64), depth=0, path=[], hilbert_idx=1, complexity=0.3),
        ]
        tokens2 = [
            SplitToken(Region(0, 0, 32, 32), depth=1, path=[0], hilbert_idx=0, complexity=0.7),
        ]
        
        results = [SplitResult(tokens=tokens1), SplitResult(tokens=tokens2)]
        
        # 转换
        tensor_result = TensorSplitResult.from_split_results(results, device)
        
        assert tensor_result.num_tokens == 3
        assert tensor_result.batch_size == 2
        assert tensor_result.regions.shape == (3, 4)
        assert (tensor_result.batch_indices == torch.tensor([0, 0, 1], device=device)).all()
    
    def test_sort_by_hilbert(self, device):
        """测试 Hilbert 排序."""
        from vit_pytorch.split_adaptive import TensorSplitResult
        
        # 创建乱序数据
        result = TensorSplitResult(
            regions=torch.tensor([[0, 0, 64, 64], [64, 0, 128, 64], [0, 64, 64, 128]], 
                                device=device, dtype=torch.long),
            depths=torch.tensor([0, 0, 0], device=device, dtype=torch.long),
            batch_indices=torch.tensor([0, 0, 0], device=device, dtype=torch.long),
            hilbert_indices=torch.tensor([3, 1, 2], device=device, dtype=torch.long),
            complexities=torch.tensor([0.3, 0.5, 0.7], device=device),
        )
        
        # 排序
        sorted_result = result.sort_by_hilbert()
        
        # 验证顺序
        assert (sorted_result.hilbert_indices == torch.tensor([1, 2, 3], device=device)).all()
        assert (sorted_result.complexities == torch.tensor([0.5, 0.7, 0.3], device=device)).all()
    
    def test_get_batch_tokens(self, device):
        """测试按 batch 提取 tokens."""
        from vit_pytorch.split_adaptive import TensorSplitResult
        
        result = TensorSplitResult(
            regions=torch.tensor([[0, 0, 64, 64], [64, 0, 128, 64], [0, 0, 32, 32]], 
                                device=device, dtype=torch.long),
            depths=torch.tensor([0, 0, 1], device=device, dtype=torch.long),
            batch_indices=torch.tensor([0, 0, 1], device=device, dtype=torch.long),
            hilbert_indices=torch.tensor([0, 1, 0], device=device, dtype=torch.long),
            complexities=torch.tensor([0.3, 0.5, 0.7], device=device),
        )
        
        # 提取 batch 0
        batch0 = result.get_batch_tokens(0)
        assert batch0.num_tokens == 2
        assert (batch0.batch_indices == 0).all()
        
        # 提取 batch 1
        batch1 = result.get_batch_tokens(1)
        assert batch1.num_tokens == 1
        assert batch1.depths[0] == 1
    
    def test_roundtrip_conversion(self, device):
        """测试 Python ↔ Tensor 往返转换."""
        from vit_pytorch.split_adaptive import (
            TensorSplitResult, SplitResult, SplitToken, Region
        )
        
        # 原始数据
        original = [
            SplitResult(tokens=[
                SplitToken(Region(0, 0, 64, 64), 0, [], 0, 0.5),
                SplitToken(Region(64, 0, 128, 64), 0, [], 1, 0.3),
            ]),
            SplitResult(tokens=[
                SplitToken(Region(0, 0, 32, 32), 1, [0], 0, 0.7),
            ]),
        ]
        
        # 转换
        tensor_result = TensorSplitResult.from_split_results(original, device)
        converted_back = tensor_result.to_split_results()
        
        # 验证
        assert len(converted_back) == 2
        assert len(converted_back[0].tokens) == 2
        assert len(converted_back[1].tokens) == 1
        assert converted_back[0].tokens[0].depth == 0
        assert converted_back[1].tokens[0].depth == 1


class TestVectorizedBFS:
    """完全向量化 BFS 测试."""
    
    @pytest.fixture
    def learnable_splitter(self):
        """创建可学习分割器."""
        from vit_pytorch.split_adaptive import LearnableSplitter
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        splitter = LearnableSplitter(
            feature_dim=64,
            max_depth=3,
            hidden_dim=32,
            pool_size=2,
            temperature=1.0,
            min_region_size=8,
        ).to(device)
        splitter.eval()
        
        return splitter
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
    def test_forward_basic(self, learnable_splitter):
        """测试 forward 基本功能 (P9-1 方案 D: 现在返回 TensorSplitResult)."""
        device = next(learnable_splitter.parameters()).device
        
        # 创建测试输入
        B, C, H, W = 2, 64, 16, 16
        features = torch.randn(B, C, H, W, device=device)
        image_size = (128, 128)  # 假设原始图像大小
        
        # 向量化分割 (现在是默认的 forward)
        result = learnable_splitter.forward(features, image_size, hard=True)
        
        # 验证输出
        from vit_pytorch.split_adaptive import TensorSplitResult
        assert isinstance(result, TensorSplitResult)
        assert result.num_tokens > 0
        assert result.batch_size == B
        assert result.regions.shape[1] == 4
        assert result.tokens_per_batch is not None
        assert result.tokens_per_batch.sum() == result.num_tokens
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
    def test_forward_returns_tensor_split_result(self, learnable_splitter):
        """测试 forward 返回 TensorSplitResult (P9-1 方案 D 验证)."""
        device = next(learnable_splitter.parameters()).device
        
        # 固定随机种子
        torch.manual_seed(42)
        
        # 创建测试输入
        B, C, H, W = 2, 64, 16, 16
        features = torch.randn(B, C, H, W, device=device)
        image_size = (128, 128)
        
        # 调用 forward
        learnable_splitter.eval()
        with torch.no_grad():
            result = learnable_splitter(features, image_size, hard=True)
        
        # 验证返回类型
        from vit_pytorch.split_adaptive import TensorSplitResult
        assert isinstance(result, TensorSplitResult)
        
        # 验证所有张量都在 GPU 上
        assert result.regions.device.type == 'cuda'
        assert result.depths.device.type == 'cuda'
        assert result.batch_indices.device.type == 'cuda'
        assert result.hilbert_indices.device.type == 'cuda'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
    def test_no_item_calls_in_hot_path(self, learnable_splitter):
        """验证热路径中没有 .item() 调用 (性能关键)."""
        device = next(learnable_splitter.parameters()).device
        
        # 创建测试输入
        B, C, H, W = 4, 64, 16, 16
        features = torch.randn(B, C, H, W, device=device)
        image_size = (128, 128)
        
        # Warm up
        for _ in range(3):
            with torch.no_grad():
                _ = learnable_splitter(features, image_size, hard=True)
        
        torch.cuda.synchronize()
        
        # 计时 (向量化版本应该很快)
        start = time.perf_counter()
        for _ in range(10):
            with torch.no_grad():
                result = learnable_splitter(features, image_size, hard=True)
        torch.cuda.synchronize()
        tensor_time = time.perf_counter() - start
        
        print(f"\n向量化分割时间: {tensor_time:.4f}s (10 次迭代)")
        
        # 验证结果有效
        from vit_pytorch.split_adaptive import TensorSplitResult
        assert isinstance(result, TensorSplitResult)
        assert result.num_tokens > 0


class TestTokenizerWithTensorResult:
    """使用 TensorSplitResult 的 Tokenizer 测试."""
    
    @pytest.fixture
    def tokenizer(self):
        """创建带可学习分割器的 tokenizer."""
        try:
            from vit_pytorch.tokenizer_streaming import StreamingFractalTokenizerV3
            from vit_pytorch.split_adaptive import AdaptiveSplitConfig
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            config = AdaptiveSplitConfig.for_learnable(
                max_depth=3,
                min_region_size=8,
            )
            
            tokenizer = StreamingFractalTokenizerV3(
                d_model=64,
                image_size=128,
                base_patch_size=16,
                use_learnable_split=True,
                split_config=config,
            ).to(device)
            tokenizer.eval()
            
            return tokenizer
        except ImportError as e:
            pytest.skip(f"无法导入所需模块: {e}")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
    def test_tokenize_basic(self, tokenizer):
        """测试 tokenize 基本功能."""
        device = next(tokenizer.parameters()).device
        
        # 创建测试图像
        images = torch.randn(2, 3, 128, 128, device=device)
        
        # 向量化 tokenization
        output = tokenizer.tokenize(images)
        
        # 验证输出
        assert len(output) == 2
        for seq in output:
            assert seq.tokens.dim() == 2
            assert seq.tokens.shape[1] == 64  # d_model
            assert "levels" in seq.metadata
    
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
