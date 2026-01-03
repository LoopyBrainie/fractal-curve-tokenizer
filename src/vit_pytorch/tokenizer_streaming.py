# -*- coding: utf-8 -*-
"""
Streaming Fractal Tokenizer V3 (Unified Architecture)

数学形式化
============

统一架构 (P7 重构):
    F = SharedConv(I)              # 共享特征提取
    Regions = Splitter(F or I)     # 可学习分割
    T = Embed(F, Regions)          # 区域池化 + 深度编码

Tokenization 过程:
    T: R^{B × C × H × W} → (T ∈ R^{B × N × D}, L ∈ Z^{B × N})

Variable Depth Tokenization:
    N ∈ [N_min, N_max] 根据图像内容自适应
    Regions = LearnableSplit(F)
    Token_i = Pool(F[R_i]) * σ_d + E_d

可学习分割 (Scheme L - LearnableSplitter):
    C_θ(R) = σ(MLP(ROI-Align(F, R)))   # 可学习复杂度评估
    
    相比已移除的规则方法的优势:
    - 端到端可微分 (Gumbel-Softmax + STE)
    - 无复杂度饱和 (MLP vs 方差公式)
    - 可学习阈值: τ_d = τ_{base,d} + δ_d
    - O(D) BFS vs O(N·4^D) DP 复杂度

复杂度分析
----------
StreamingFractalTokenizerV3:
    时间: O(C · H · W) + O(N_max · D × k²)
          ├─ 特征提取 (Conv):      O(C · H · W)         — 共享卷积
          └─ 可学习分割 (MLP):     O(N_max · D × k²)    — ROI + MLP
    空间: O(C · H · W) + O(N_max · D)
          ├─ 特征图:  O(C · H · W)
          └─ Token:   O(N_max · D)

其中: C=channels, H×W=image_size, N_max=max_tokens, D=d_model

Note:
    Scheme B (BalancedGreedySplitter) 和 Scheme C (FixedBudgetDPSplitter)
    已被移除，因为 LearnableSplitter 提供了更优的功能。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .base_tokenizer import BaseTokenizer, TokenizerOutput, TokenSequence
from .config_fractal import FractalConfig
from .embed_fractal_path import VectorizedPathEncoder  # I12-3: 用于计算路径


class StreamingFractalTokenizerV3(BaseTokenizer):
    """Variable Depth Tokenizer with Learnable Quadtree Splitting.
    
    数学形式化
    ==========
    
    统一架构 (P7 重构):
        F = SharedConv(I)              # 共享特征提取 (提升到 Tokenizer 级别)
        Regions = Splitter(F, I)       # 可学习分割
        Token_i = Pool(F[R_i]) * σ_d + E_d  # 区域池化 + 深度编码
    
    可学习分割 (LearnableSplitter):
        C_θ(R) = σ(MLP(ROI-Align(F, R)))   # 可学习复杂度
        p_split = σ((C_θ - τ_d) / T)        # 软分割决策
        z ~ Gumbel-Softmax                   # 可微分采样
    
    梯度流:
        L_cls → T → F → SharedConv (端到端)
        L_cls → T → Splitter → MLP (可学习分割)
    
    Args:
        image_size: 输入图像尺寸
        channels: 图像通道数
        d_model: 输出嵌入维度
        base_patch_size: 最细粒度 patch 大小
        max_depth: 最大四叉树深度
        use_hilbert_order: 是否使用 Hilbert 曲线排序
        target_tokens: 目标 token 数量
        enforce_balance: 是否强制 2:1 平衡约束
        depth_scale_range: (P6-1) 深度缩放范围 (σ_min, σ_max)
        gamma: 可学习分割器的阈值衰减因子 γ ∈ (0,1)
        learnable_temperature: 可学习分割的初始温度
        use_gumbel: 是否使用 Gumbel-Softmax
        
    Note:
        Scheme B (BalancedGreedySplitter) 和 Scheme C (FixedBudgetDPSplitter)
        已被移除。LearnableSplitter 提供了完全覆盖的功能并增加了:
        - 端到端可微分性
        - 无复杂度饱和
        - 可学习阈值
    """
    
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        channels: int = 3,
        d_model: int = 256,
        base_patch_size: int = 4,
        max_depth: int = 4,
        use_hilbert_order: bool = True,
        target_tokens: Optional[int] = None,
        enforce_balance: bool = True,
        depth_scale_range: Optional[Tuple[float, float]] = (0.5, 2.0),
        gamma: float = 0.85,
        learnable_temperature: float = 1.0,
        use_gumbel: bool = True,
    ) -> None:
        super().__init__()
        
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        
        self.image_size = image_size
        self.channels = channels
        self.d_model = d_model
        self.base_patch_size = base_patch_size
        self.max_depth = max_depth
        self.use_hilbert_order = use_hilbert_order
        self._use_learnable_split = True  # Always use LearnableSplitter
        
        # =====================================================================
        # Hilbert-Native Patch Embedding (包含 SharedConv)
        # =====================================================================
        from .embed_hilbert_patch import HilbertNativePatchEmbed
        self.patch_embed = HilbertNativePatchEmbed(
            channels=channels,
            dim=d_model,
            base_patch_size=base_patch_size,
            max_depth=max_depth,
            conv_layers=2,
            use_batch_norm=True,
            depth_scale_range=depth_scale_range,
        )
        
        # =====================================================================
        # LearnableSplitter (Scheme B/C have been removed)
        # =====================================================================
        # Mathematical justification for removal:
        # - Scheme B: C(R) = Var/(Var+σ₀²) saturates as Var → ∞
        # - Scheme C: O(N·4^D) DP complexity, not differentiable
        # - Scheme L: C_θ(R) = σ(MLP(ROI-Align(F, R))) - no saturation, O(D) BFS
        # =====================================================================
        from .split_adaptive import LearnableSplitter
        
        # P11-14: 确保 min_region_size >= 2 * base_patch_size
        # 这保证最小区域至少覆盖 2×2 = 4 个特征像素,
        # 使 ROI-Align 的 4×4 采样网格能获得有意义的空间信息
        safe_min_region_size = max(base_patch_size * 2, 8)
        
        self.splitter = LearnableSplitter(
            feature_dim=d_model,
            max_depth=max_depth,
            hidden_dim=64,
            pool_size=4,
            temperature=learnable_temperature,
            use_gumbel=use_gumbel,
            enforce_balance=enforce_balance,
            min_region_size=safe_min_region_size,
            init_tau_base=0.5,
            init_tau_gamma=gamma,
        )
        
        self._last_split_stats: Optional[Dict[str, Any]] = None
        self._last_depth_count_matrix: Optional[torch.Tensor] = None  # P11-9: 向量化缓存
        self._last_features: Optional[torch.Tensor] = None
    
    @property
    def patch_sizes(self) -> List[int]:
        """兼容性属性: 从 base_patch_size 和 max_depth 计算等效的 patch 大小列表.
        
        数学形式:
            patch_sizes[d] = base_patch_size × 2^d, d ∈ [0, max_depth]
            
        例如: base_patch_size=4, max_depth=4
            → patch_sizes = [4, 8, 16, 32, 64]
            
        Note:
            这是为了向后兼容旧版评估脚本。V3 tokenizer 使用可变深度 token,
            实际 patch 大小由 LearnableSplitter 动态决定。
        """
        return [self.base_patch_size * (2 ** d) for d in range(self.max_depth + 1)]
    
    @property 
    def shared_conv(self) -> nn.Module:
        """获取共享卷积层 (用于可学习分割)."""
        return self.patch_embed.shared_conv
    
    @classmethod
    def from_config(
        cls,
        config: "FractalConfig",
        channels: int = 3,
        d_model: int = 256,
    ) -> "StreamingFractalTokenizerV3":
        """从 FractalConfig 创建 Tokenizer 实例."""
        return cls(
            image_size=config.image_size,
            channels=channels,
            d_model=d_model,
            base_patch_size=config.patch_sizes[0] if config.patch_sizes else 4,
            max_depth=config.max_depth,
            use_hilbert_order=True,
        )
    
    def tokenize(self, images: torch.Tensor) -> TokenizerOutput:
        """Variable Depth tokenization (P9-1 方案 D: 完全向量化).
        
        数学形式化:
            1. F = SharedConv(I)                    # 特征提取
            2. TensorResult = Splitter.forward(F)   # 纯张量分割 (P9-1)
            3. T = _embed_with_tensor_result(F, TensorResult)  # 纯张量嵌入
            
        性能特性 (LearnableSplitter):
            - O(D) GPU kernels 替代 O(N×D) Python 循环
            - 预期加速: ~8x
            
        对于规则分割器，回退到旧实现。
        """
        if images.dim() != 4:
            raise ValueError(
                f"StreamingFractalTokenizerV3.tokenize expects 4D input [B, C, H, W], "
                f"got {images.dim()}D tensor."
            )
        
        B, C, H, W = images.shape
        device = images.device
        
        # 1. 提取共享特征图
        features = self.shared_conv(images)  # [B, d_model, H/p, W/p]
        self._last_features = features
        
        # 2. Adaptive/Learnable Splitting
        if self._use_learnable_split:
            # P9-1: 使用完全向量化的 forward()
            from .split_adaptive import LearnableSplitter, TensorSplitResult
            assert isinstance(self.splitter, LearnableSplitter)
            
            tensor_result: TensorSplitResult = self.splitter(
                features,
                image_size=(H, W),
                hard=not self.training,
            )
            
            # P11-3 优化: 使用 non_blocking=True 减少 GPU-CPU 同步阻塞
            # 统计收集在 no_grad 块内，不影响梯度，但仍需数据传输
            # non_blocking 允许 CUDA 流并行，减少等待时间
            with torch.no_grad():
                tokens_per_batch = tensor_result.tokens_per_batch
                if tokens_per_batch is not None:
                    # P11-3: 异步传输到 CPU (使用 .to() 支持 non_blocking)
                    num_tokens_list = tokens_per_batch.to('cpu', non_blocking=True).tolist()
                else:
                    # Fallback: 使用 bincount (P11-2 优化的一致性)
                    tokens_per_batch = torch.bincount(
                        tensor_result.batch_indices, 
                        minlength=B
                    )
                    num_tokens_list = tokens_per_batch.to('cpu', non_blocking=True).tolist()
                
                # 计算 depth distribution (P9-6 向量化优化)
                # 使用批量操作减少 .item() 调用次数从 O(B × max_depth) 到 O(B)
                depth_dists = []
                max_d = self.max_depth + 1
                depths = tensor_result.depths
                batch_indices = tensor_result.batch_indices
                
                # 一次性计算所有 (batch, depth) 组合的计数
                # 使用 one-hot encoding + scatter_add
                if tensor_result.num_tokens > 0:
                    # 创建 [B, max_depth+1] 的计数矩阵
                    count_matrix = torch.zeros(B, max_d, dtype=torch.long, device=device)
                    # 使用 index_add 在每个 (batch, depth) 位置累加 1
                    flat_idx = batch_indices * max_d + depths.clamp(max=max_d - 1)
                    ones = torch.ones_like(flat_idx)
                    count_matrix.view(-1).scatter_add_(0, flat_idx, ones)
                    
                    # P11-9: 保存张量以供 get_scale_entropy 使用
                    self._last_depth_count_matrix = count_matrix
                    
                    # P11-3: 异步传输到 CPU (使用 .to() 支持 non_blocking)
                    count_matrix_cpu = count_matrix.to('cpu', non_blocking=True).numpy()
                    
                    # P11-4 保留: Python 循环构建 dict 结构
                    # 这是必要的，因为输出格式需要稀疏字典表示
                    for b in range(B):
                        dist = {}
                        for d in range(max_d):
                            count = int(count_matrix_cpu[b, d])
                            if count > 0:
                                dist[d] = count
                        depth_dists.append(dist)
                else:
                    depth_dists = [{} for _ in range(B)]
                    self._last_depth_count_matrix = None  # P11-9: 清除缓存
            
            self._last_split_stats = {
                'num_tokens': num_tokens_list,
                'depth_distributions': depth_dists,
            }
            
            # 3. 纯张量嵌入 (P11-3: 额外返回 regions)
            tokens, levels_info, padded_regions = self._embed_with_tensor_result(features, tensor_result)
            
            # 4. 构建输出 (P9-5: 保留已 padding 的张量作为缓存)
            sequences = []
            for b in range(B):
                num_tokens = num_tokens_list[b]
                seq = TokenSequence(
                    tokens=tokens[b, :num_tokens],
                    metadata={
                        "levels": levels_info[b, :num_tokens],
                        "split_stats": {
                            "num_tokens": num_tokens,
                            "depth_distribution": depth_dists[b],
                        },
                    },
                )
                sequences.append(seq)
            
            # P9-5/P12-2 优化: 传入已 padding 的张量缓存，避免 model 中重复 padding
            # P11-3: 新增 _regions_cache 和 _image_size_cache 用于正确的 LCA 偏置计算
            # P12-2: _lengths_cache 直接存储为 Tensor，避免后续 List->Tensor 转换
            lengths_tensor = torch.tensor(num_tokens_list, dtype=torch.long, device=device)
            return TokenizerOutput(
                sequences=sequences,
                _padded_tokens_cache=tokens,
                _padded_levels_cache=levels_info,
                _lengths_cache=lengths_tensor,
                _regions_cache=padded_regions,
                _image_size_cache=self.image_size,
            )
        
        else:
            # 规则分割: 使用原始实现
            split_results = self.splitter.split_batch(images)
            
            self._last_split_stats = {
                'num_tokens': [sr.num_tokens for sr in split_results],
                'depth_distributions': [sr.depth_distribution for sr in split_results],
            }
            
            # Hilbert-Native Patch Embedding
            tokens, levels_info = self._embed_with_features(features, split_results)
            
            # 构建输出
            sequences = []
            for b in range(B):
                num_tokens = split_results[b].num_tokens
                seq = TokenSequence(
                    tokens=tokens[b, :num_tokens],
                    metadata={
                        "levels": levels_info[b, :num_tokens],
                        "split_stats": {
                            "num_tokens": num_tokens,
                            "depth_distribution": split_results[b].depth_distribution,
                        },
                    },
                )
                sequences.append(seq)
            
            return TokenizerOutput(sequences)
    
    def tokenize_tensor(self, images: torch.Tensor) -> TokenizerOutput:
        """兼容别名: 已弃用，请使用 tokenize().
        
        P9-1 方案 D 实施后，tokenize() 已完全向量化。
        保留此方法仅为向后兼容。
        """
        import warnings
        warnings.warn(
            "tokenize_tensor() is deprecated. Use tokenize() instead. "
            "tokenize() is now fully vectorized for LearnableSplitter.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.tokenize(images)
    
    def _embed_with_features(
        self,
        features: torch.Tensor,
        split_results: List,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用预计算的特征进行嵌入 (避免重复计算 SharedConv).
        
        这是 patch_embed.forward() 的优化版本，跳过 SharedConv。
        """
        from .split_adaptive import SplitResult
        
        B = features.shape[0]
        device = features.device
        dtype = features.dtype
        dim = self.d_model
        
        # 确定最大 token 数量
        token_counts = [sr.num_tokens for sr in split_results]
        max_tokens = max(token_counts) if token_counts else 1
        total_tokens = sum(token_counts)
        
        if total_tokens == 0:
            tokens = torch.zeros(B, 1, dim, device=device, dtype=dtype)
            levels_info = torch.zeros(B, 1, self.max_depth + 1, dtype=torch.long, device=device)
            return self.patch_embed.norm(tokens), levels_info
        
        # 收集所有 region 的 boxes 和 depths
        all_boxes = []
        all_depths = []
        all_levels_info = []
        batch_indices = []
        token_indices = []
        
        p = self.base_patch_size
        for b, sr in enumerate(split_results):
            for i, token in enumerate(sr.tokens):
                fx1 = token.region.x1 / p
                fy1 = token.region.y1 / p
                fx2 = token.region.x2 / p
                fy2 = token.region.y2 / p
                
                fx2 = max(fx1 + 0.5, fx2)
                fy2 = max(fy1 + 0.5, fy2)
                
                all_boxes.append([b, fx1, fy1, fx2, fy2])
                all_depths.append(min(token.depth, self.max_depth))
                all_levels_info.append(token.to_levels_info(self.max_depth))
                batch_indices.append(b)
                token_indices.append(i)
        
        boxes_tensor = torch.tensor(all_boxes, device=device, dtype=dtype)
        depths_tensor = torch.tensor(all_depths, device=device, dtype=torch.long)
        
        # ROI-Align
        try:
            from torchvision.ops import roi_align
            pooled = roi_align(
                features,
                boxes_tensor,
                output_size=(1, 1),
                spatial_scale=1.0,
                aligned=True,
            ).squeeze(-1).squeeze(-1)
        except ImportError:
            pooled = self.patch_embed._fallback_roi_pool(
                features, boxes_tensor, 
                features.shape[2], features.shape[3]
            )
        
        # 深度编码
        scales = self.patch_embed.depth_scale[depths_tensor]
        embeds = self.patch_embed.depth_embed(depths_tensor)
        all_tokens = pooled * scales.unsqueeze(-1) + embeds
        
        # 分配到输出 buffer (P-PERF-3: 向量化分配)
        tokens = torch.zeros(B, max_tokens, dim, device=device, dtype=dtype)
        levels_info = torch.zeros(B, max_tokens, self.max_depth + 1, dtype=torch.long, device=device)
        
        # 使用高级索引进行向量化分配
        batch_idx_tensor = torch.tensor(batch_indices, device=device, dtype=torch.long)
        token_idx_tensor = torch.tensor(token_indices, device=device, dtype=torch.long)
        
        # 向量化 token 分配 (确保 dtype 匹配，支持 AMP 混合精度)
        tokens[batch_idx_tensor, token_idx_tensor] = all_tokens.to(dtype)
        
        # 向量化 levels_info 分配
        levels_info_tensor = torch.tensor(all_levels_info, device=device, dtype=torch.long)
        levels_info[batch_idx_tensor, token_idx_tensor] = levels_info_tensor
        
        return self.patch_embed.norm(tokens), levels_info
    
    def _embed_with_tensor_result(
        self,
        features: torch.Tensor,
        tensor_result: "TensorSplitResult",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """使用 TensorSplitResult 进行嵌入 (P9-1 完全向量化版本).
        
        P11-3 改进: 额外返回 padded_regions 张量用于正确的 LCA 偏置计算。
        
        数学形式化
        ==========
        
        传统实现:
            T_embed = O(N) Python 循环 + O(N) torch.tensor() 调用
            同步点: ~3N 次 (每个 token 创建 3 个小张量)
            
        向量化实现:
            T_embed = O(1) 张量操作
            同步点: 0 次 (所有数据已在 GPU)
            
        预期加速: ~10x (消除所有 Python 循环)
        
        Args:
            features: [B, C, H', W'] 预计算的特征图
            tensor_result: TensorSplitResult 纯张量分割结果
            
        Returns:
            (tokens, levels_info, padded_regions):
            - tokens: [B, MaxN, D] 嵌入后的 tokens
            - levels_info: [B, MaxN, max_depth+1] 层级信息
            - padded_regions: [B, MaxN, 4] 区域边界 (P11-3 新增)
        """
        from .split_adaptive import TensorSplitResult
        
        B = features.shape[0]
        device = features.device
        dtype = features.dtype
        dim = self.d_model
        
        N_total = tensor_result.num_tokens
        if N_total == 0:
            tokens = torch.zeros(B, 1, dim, device=device, dtype=dtype)
            levels_info = torch.zeros(B, 1, self.max_depth + 1, dtype=torch.long, device=device)
            padded_regions = torch.zeros(B, 1, 4, dtype=torch.long, device=device)
            return self.patch_embed.norm(tokens), levels_info, padded_regions
        
        # 计算每个 batch 的最大 token 数量
        if tensor_result.tokens_per_batch is not None:
            max_tokens = int(tensor_result.tokens_per_batch.max().item())
        else:
            # 回退: 计算每个 batch 的 token 数
            max_tokens = 0
            for b in range(B):
                n = (tensor_result.batch_indices == b).sum()
                max_tokens = max(max_tokens, int(n.item()))
        
        # ====================================================================
        # 构建 ROI boxes (纯张量操作)
        # ====================================================================
        # regions: [N, 4] -> (x1, y1, x2, y2)
        p = self.base_patch_size
        regions = tensor_result.regions.float()
        batch_indices = tensor_result.batch_indices
        
        # boxes: [N, 5] -> (batch_idx, x1, y1, x2, y2) (scaled)
        boxes = torch.zeros(N_total, 5, device=device, dtype=dtype)
        boxes[:, 0] = batch_indices.float()
        boxes[:, 1] = regions[:, 0] / p  # x1
        boxes[:, 2] = regions[:, 1] / p  # y1
        boxes[:, 3] = regions[:, 2] / p  # x2  
        boxes[:, 4] = regions[:, 3] / p  # y2
        
        # 确保最小尺寸
        boxes[:, 3] = torch.maximum(boxes[:, 1] + 0.5, boxes[:, 3])
        boxes[:, 4] = torch.maximum(boxes[:, 2] + 0.5, boxes[:, 4])
        
        # ====================================================================
        # ROI-Align (批量)
        # ====================================================================
        from torchvision.ops import roi_align
        pooled = roi_align(
            features,
            boxes,
            output_size=(1, 1),
            spatial_scale=1.0,
            aligned=True,
        ).squeeze(-1).squeeze(-1)  # [N, C]
        
        # ====================================================================
        # 深度编码 (向量化)
        # ====================================================================
        depths = tensor_result.depths.clamp(max=self.max_depth)
        scales = self.patch_embed.depth_scale[depths]  # [N]
        embeds = self.patch_embed.depth_embed(depths)   # [N, D]
        all_tokens = pooled * scales.unsqueeze(-1) + embeds  # [N, D]
        
        # ====================================================================
        # 向量化分配到输出 buffer
        # ====================================================================
        tokens = torch.zeros(B, max_tokens, dim, device=device, dtype=dtype)
        levels_info = torch.zeros(B, max_tokens, self.max_depth + 1, dtype=torch.long, device=device)
        padded_regions = torch.zeros(B, max_tokens, 4, dtype=torch.long, device=device)  # P11-3
        
        # 计算每个 token 在其 batch 内的索引
        # P12-3: 利用 batch_indices 已按 (batch_idx, hilbert_idx) 排序的特性
        # 使用 cummax 传播段起始位置，实现 O(N) 单次遍历的向量化计算
        if N_total == 0:
            token_positions = torch.zeros(0, dtype=torch.long, device=device)
        else:
            # 检测段边界: S_i = 1 当 i=0 或 b_i ≠ b_{i-1}
            segment_starts = torch.cat([
                torch.ones(1, device=device, dtype=torch.long),
                (batch_indices[1:] != batch_indices[:-1]).long()
            ])
            
            # 全局位置索引
            global_positions = torch.arange(N_total, device=device)
            
            # 段起始位置传播: cummax(i * S_i) 获取每个位置所属段的起始索引
            segment_start_indices = (global_positions * segment_starts).cummax(dim=0)[0]
            
            # 段内位置 = 全局位置 - 段起始位置
            token_positions = global_positions - segment_start_indices
        
        # 向量化分配
        tokens[batch_indices, token_positions] = all_tokens.to(dtype)
        
        # =====================================================================
        # I12-3 修复: 从 regions 计算四叉树路径填充 levels_info
        # 原问题: levels_info[:, :, 1:] 全为零，导致所有 token 共享相同的路径编码
        # 修复方案: 使用 VectorizedPathEncoder.compute_paths_from_regions 计算正确路径
        # =====================================================================
        levels_info[batch_indices, token_positions, 0] = depths
        
        # 计算四叉树路径 (基于区域中心的空间位置)
        # regions: [N_total, 4] -> paths: [N_total, max_depth]
        if N_total > 0:
            # 获取图像尺寸 (Hilbert 曲线要求方形，使用较大边)
            img_size = max(self.image_size) if isinstance(self.image_size, tuple) else self.image_size
            paths = VectorizedPathEncoder.compute_paths_from_regions(
                tensor_result.regions,  # [N_total, 4]
                img_size,
                self.max_depth
            )  # [N_total, max_depth]
            
            # 填充路径到 levels_info
            # levels_info 格式: [depth, path[0], path[1], ..., path[max_depth-1]]
            levels_info[batch_indices, token_positions, 1:] = paths
        
        # P11-3: 分配 regions 到 padded buffer
        padded_regions[batch_indices, token_positions] = tensor_result.regions
        
        return self.patch_embed.norm(tokens), levels_info, padded_regions
    
    def forward(self, images: torch.Tensor) -> TokenizerOutput:
        """前向传播，等价于 tokenize."""
        return self.tokenize(images)
    
    @torch.no_grad()
    def get_split_stats(self) -> Optional[Dict[str, Any]]:
        """获取最近一次分割的统计信息."""
        if self._last_split_stats is None:
            return None
        
        # 添加 depth_entropy 到统计信息
        stats = self._last_split_stats.copy()
        stats['depth_entropy'] = self.get_scale_entropy()
        return stats
    
    def get_entropy_loss(self) -> Optional[torch.Tensor]:
        """获取熵正则化损失 (用于可学习分割器)."""
        if not self._use_learnable_split:
            return None
        
        from .split_adaptive import LearnableSplitter
        assert isinstance(self.splitter, LearnableSplitter)
        
        if self._last_split_stats is None:
            return None
        
        # 构造 SplitResult 列表用于熵计算
        from .split_adaptive import SplitResult, SplitToken, Region
        results = []
        for i, dist in enumerate(self._last_split_stats['depth_distributions']):
            # 重建简化的 SplitResult
            tokens = []
            for depth, count in dist.items():
                for _ in range(count):
                    tokens.append(SplitToken(
                        region=Region(0, 0, 1, 1),  # 占位
                        depth=depth,
                        path=[],
                        hilbert_idx=0,
                        complexity=0.0,
                    ))
            results.append(SplitResult(tokens=tokens))
        
        return self.splitter.get_entropy_loss(results)
    
    def get_learnable_split_loss(
        self,
        lambda_entropy: float = 0.1,
        lambda_budget: float = 0.01,
        target_tokens: int = 64,
    ) -> Optional[torch.Tensor]:
        """获取可学习分割器的辅助损失.
        
        数学形式化:
            L_split = λ₁ · L_entropy + λ₂ · L_budget + L_reg
            
        其中:
            L_entropy = -H(depth_distribution)  # 鼓励多尺度
            L_budget = ReLU(N - N_target)²      # 预算约束
            L_reg = 阈值正则化                   # 防止坍塌
        
        Args:
            lambda_entropy: 熵损失权重
            lambda_budget: 预算损失权重  
            target_tokens: 目标 token 数
            
        Returns:
            可微分的辅助损失 (用于多任务训练)
        """
        if not self._use_learnable_split:
            return None
        
        from .split_adaptive import LearnableSplitter
        assert isinstance(self.splitter, LearnableSplitter)
        
        # 阈值正则化 (可微分)
        reg_loss = self.splitter.get_threshold_regularization_loss()
        
        # 深度损失 (需要特征图)
        if self._last_features is not None:
            depth_loss = self.splitter.get_differentiable_depth_loss(
                self._last_features, 
                self.image_size,
            )
        else:
            depth_loss = torch.tensor(0.0, device=reg_loss.device)
        
        return lambda_entropy * depth_loss + reg_loss
    
    def get_scale_entropy(self) -> Optional[float]:
        """获取尺度分布熵值.
        
        P11-9 优化: 使用向量化计算，避免 O(B × D) Python 循环。
        
        数学形式:
            H = -∑_d p_d log(p_d)
            其中 p_d = count_d / ∑_d' count_d'
        """
        if self._last_split_stats is None:
            return None
        
        # P11-9: 如果有张量缓存，使用向量化计算
        if self._last_depth_count_matrix is not None:
            count_matrix = self._last_depth_count_matrix  # [B, max_d]
            total_counts = count_matrix.sum(dim=0).float()  # [max_d]
            total = total_counts.sum()
            if total == 0:
                return None
            probs = total_counts / total
            # 避免 log(0)
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum().item()
            return entropy
        
        # Fallback: 原始 Python 字典方式 (规则分割器路径)
        total_dist: Dict[int, int] = {}
        for dist in self._last_split_stats['depth_distributions']:
            for d, count in dist.items():
                total_dist[d] = total_dist.get(d, 0) + count
        
        total = sum(total_dist.values())
        if total == 0:
            return None
        
        entropy = 0.0
        for count in total_dist.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log(p)
        
        return entropy
    
    @torch.no_grad()
    def compute_scale_distribution(self, images: torch.Tensor) -> Dict[str, Any]:
        """计算深度分布统计信息.
        
        P11-9 优化: 优先使用张量缓存进行向量化计算。
        """
        _ = self.tokenize(images)
        
        if self._last_split_stats is None:
            return {
                'scale_ratios': {},
                'entropy': 0.0,
                'max_entropy': 0.0,
                'dominant_scale': self.base_patch_size,
                'depth_distribution': {},
            }
        
        # P11-9: 优先使用向量化路径
        if self._last_depth_count_matrix is not None:
            count_matrix = self._last_depth_count_matrix  # [B, max_d]
            total_counts = count_matrix.sum(dim=0)  # [max_d]
            total_tokens = int(total_counts.sum().item())
            
            if total_tokens == 0:
                return {
                    'scale_ratios': {},
                    'entropy': 0.0,
                    'max_entropy': 0.0,
                    'dominant_scale': self.base_patch_size,
                    'depth_distribution': {},
                }
            
            # 转换为 Python dict (用于返回值兼容性)
            total_counts_cpu = total_counts.cpu().numpy()
            total_dist = {d: int(c) for d, c in enumerate(total_counts_cpu) if c > 0}
            
            # 向量化计算 scale_ratios
            scale_ratios: Dict[int, float] = {}
            for depth, count in total_dist.items():
                ps = self.base_patch_size * (2 ** (self.max_depth - depth))
                scale_ratios[ps] = count / total_tokens
            
            # 向量化熵计算
            probs = total_counts.float() / total_tokens
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum().item()
            
            num_depths = self.max_depth + 1
            max_entropy = math.log(num_depths) if num_depths > 1 else 0.0
            
            dominant_depth = int(total_counts.argmax().item())
            dominant_scale = self.base_patch_size * (2 ** (self.max_depth - dominant_depth))
            
            return {
                'scale_ratios': scale_ratios,
                'entropy': entropy,
                'max_entropy': max_entropy,
                'dominant_scale': dominant_scale,
                'depth_distribution': total_dist,
            }
        
        # Fallback: 原始 Python 字典方式 (规则分割器路径)
        total_dist: Dict[int, int] = {}
        for dist in self._last_split_stats['depth_distributions']:
            for d, count in dist.items():
                total_dist[d] = total_dist.get(d, 0) + count
        
        total_tokens = sum(total_dist.values())
        if total_tokens == 0:
            return {
                'scale_ratios': {},
                'entropy': 0.0,
                'max_entropy': 0.0,
                'dominant_scale': self.base_patch_size,
                'depth_distribution': {},
            }
        
        scale_ratios = {}
        for depth, count in total_dist.items():
            ps = self.base_patch_size * (2 ** (self.max_depth - depth))
            scale_ratios[ps] = count / total_tokens
        
        entropy = 0.0
        for count in total_dist.values():
            p = count / total_tokens
            if p > 0:
                entropy -= p * math.log(p)
        
        num_depths = self.max_depth + 1
        max_entropy = math.log(num_depths) if num_depths > 1 else 0.0
        
        dominant_depth = max(total_dist.keys(), key=lambda d: total_dist[d])
        dominant_scale = self.base_patch_size * (2 ** (self.max_depth - dominant_depth))
        
        return {
            'scale_ratios': scale_ratios,
            'entropy': entropy,
            'max_entropy': max_entropy,
            'dominant_scale': dominant_scale,
            'depth_distribution': total_dist,
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练状态统计信息."""
        stats = {
            'tokenizer_version': 'v3_unified',
            'architecture': 'shared_conv + learnable_split + hilbert_embed',
            'split_scheme': 'learnable',  # Only LearnableSplitter remains
            'max_depth': self.max_depth,
            'learnable_split': True,  # Always true after Scheme B/C removal
        }
        
        if self._last_split_stats:
            avg_tokens = sum(self._last_split_stats['num_tokens']) / len(self._last_split_stats['num_tokens'])
            stats['avg_tokens_per_image'] = avg_tokens
            stats['depth_entropy'] = self.get_scale_entropy()
        
        # Always use LearnableSplitter
        from .split_adaptive import LearnableSplitter
        assert isinstance(self.splitter, LearnableSplitter)
        splitter_stats = self.splitter.get_split_statistics()
        stats['learnable_thresholds'] = splitter_stats['thresholds'].tolist()
        stats['learnable_temperature'] = splitter_stats['temperature'].item()
        
        return stats
    
    def set_split_temperature(self, temperature: float) -> None:
        """设置可学习分割器的温度 (用于退火调度).
        
        数学形式化:
            T(t) = T_start · (T_end / T_start)^(t / T_total)
        """
        if self._use_learnable_split:
            from .split_adaptive import LearnableSplitter
            assert isinstance(self.splitter, LearnableSplitter)
            self.splitter.set_temperature(temperature)
    
    def reset_split_statistics(self) -> None:
        """重置可学习分割器的统计信息."""
        if self._use_learnable_split:
            from .split_adaptive import LearnableSplitter
            assert isinstance(self.splitter, LearnableSplitter)
            self.splitter.reset_statistics()
    
    def get_temperature_scheduler(
        self,
        T_start: float = 1.0,
        T_end: float = 0.3,  # P11-11: 0.1 → 0.3 安全下界
        schedule: str = 'exponential',
        warmup_steps: int = 0,
    ):
        """获取温度退火调度器 (仅可学习分割器).
        
        数学形式化:
            T(t) = T_start · (T_end / T_start)^(t / total_steps)
            
        推荐参数 (P11-11 修复后):
            T_start = 1.0: 探索充分，100% 样本有活跃梯度
            T_end = 0.3:   决策确定性 51%，92% 样本有活跃梯度
            
            注意: T_end < 0.3 会导致训练后期梯度稀疏（当 z 分布偏移时）
            
        用法:
            scheduler = tokenizer.get_temperature_scheduler()
            scheduler.set_total_steps(epochs * steps_per_epoch)
            
            for step in training_loop:
                loss = model(batch)
                ...
                scheduler.step()  # 自动更新温度
        
        Args:
            T_start: 初始温度 (默认 1.0)
            T_end: 最终温度 (默认 0.3, P11-11 安全下界)
            schedule: 调度策略 ('exponential', 'linear', 'cosine')
            warmup_steps: 热身步数，期间保持 T_start
            
        Returns:
            TemperatureScheduler 实例
            
        Raises:
            ValueError: 如果不是可学习分割器
        """
        from .split_adaptive import LearnableSplitter, TemperatureScheduler
        assert isinstance(self.splitter, LearnableSplitter)
        
        return TemperatureScheduler(
            splitter=self.splitter,
            T_start=T_start,
            T_end=T_end,
            schedule=schedule,
            warmup_steps=warmup_steps,
        )

    def get_multi_layer_depth_loss(
        self,
        features: Tensor,
        image_size: Optional[Tuple[int, int]] = None,
        max_eval_depth: Optional[int] = None,
        weight_decay_factor: float = 0.5,
        target_entropy: float = 0.693,
        return_details: bool = False,
    ):
        """获取多层可微分深度损失 (仅可学习分割器).
        
        数学形式化:
            L_multi = Σ_d w_d · (H_target - H̄_d)²
            
        为什么需要多层损失:
            单层损失仅在根区域评估，深层阈值 τ₁, τ₂, τ₃... 无梯度信号。
            多层损失在每层的规则网格评估，确保所有阈值可学习。
            
        梯度覆盖率 (β=0.5, D=3):
            d=0: 51.6%, d=1: 25.8%, d=2: 12.9%, d=3: 6.5%
            总覆盖: 96.8%
            
        用法:
            loss, details = tokenizer.get_multi_layer_depth_loss(
                features, image_size, return_details=True
            )
            
            # TensorBoard 可视化
            for d, (l, e, p) in enumerate(zip(
                details['loss_per_depth'],
                details['entropy_per_depth'], 
                details['p_split_per_depth']
            )):
                writer.add_scalar(f'depth/loss_d{d}', l, step)
                writer.add_scalar(f'depth/entropy_d{d}', e, step)
                writer.add_scalar(f'depth/p_split_d{d}', p, step)
                
        Args:
            features: [B, C, H', W'] 特征图
            image_size: (H, W) 图像尺寸，默认使用 self.image_size
            max_eval_depth: 最大评估深度 (默认 min(max_depth, 3))
            weight_decay_factor: 权重衰减因子 β (默认 0.5)
            target_entropy: 目标熵 (默认 0.693 = ln(2))
            return_details: 是否返回各层详情
            
        Returns:
            loss: 标量损失
            details (if return_details): 各层损失详情字典
            
        Raises:
            ValueError: 如果不是可学习分割器
        """
        from .split_adaptive import LearnableSplitter
        assert isinstance(self.splitter, LearnableSplitter)
        
        if image_size is None:
            image_size = self.image_size
        
        return self.splitter.get_multi_layer_depth_loss(
            features=features,
            image_size=image_size,
            max_eval_depth=max_eval_depth,
            weight_decay_factor=weight_decay_factor,
            target_entropy=target_entropy,
            return_details=return_details,
        )
