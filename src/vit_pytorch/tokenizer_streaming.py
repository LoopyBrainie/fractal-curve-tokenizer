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
from .constants import LOG_EPSILON, PROB_EPSILON  # I12-7: 数值稳定性常量
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
        # I30-17: 替换固定 max_depth 为动态计算
        # 支持 Union[int, Tuple[int, int]] 用于向后兼容
        min_patch_size: Union[int, Tuple[int, int]] = 4,
        # 保留 max_depth 用于向后兼容 (可选，如果指定则使用该值)
        max_depth: Optional[int] = None,
        use_hilbert_order: bool = True,
        target_tokens: Optional[int] = None,
        enforce_balance: bool = True,
        depth_scale_range: Optional[Tuple[float, float]] = (0.5, 2.0),
        gamma: float = 0.85,
        learnable_temperature: float = 1.0,
        use_gumbel: bool = True,
        # I20: Gumbel-Top-K 参数
        K_min: int = 8,
        K_max: int = 64,
        # I27-1: Splitter Dropout 配置
        # 数学依据: Splitter MLP 规模 ~64×128，推荐 p ∈ [0.1, 0.15]
        # 过高 dropout 会降低分割决策质量，过低则正则化不足
        splitter_dropout: float = 0.1,
        # I30-10: SplitterConfig 统一配置
        splitter_config: Optional["SplitterConfig"] = None,
    ) -> None:
        super().__init__()

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        # I30-17: 处理 min_patch_size 的向后兼容
        if isinstance(min_patch_size, tuple):
            # 旧 API: min_patch_size=(4, 4)，取第一个值
            effective_min_patch_size = min_patch_size[0]
        else:
            effective_min_patch_size = min_patch_size

        self.image_size = image_size
        self.channels = channels
        self.d_model = d_model
        self.base_patch_size = base_patch_size
        self.use_hilbert_order = use_hilbert_order
        self._use_learnable_split = True  # Legacy flag, always True

        # I30-17: 动态深度计算
        self.min_patch_size = effective_min_patch_size  # 存储规范化后的值

        # 动态计算 max_depth (用于 splitter)
        # 公式: L_max = max(0, floor(log2(min(H, W) / min_patch_size)))
        from .depth_utils import compute_max_depth
        self._computed_max_depth = compute_max_depth(
            image_size, effective_min_patch_size
        )

        # I30-17-EXT: 确定最终使用的 max_depth
        # 优先级: 显式指定 max_depth > 动态计算
        self.max_depth = max_depth if max_depth is not None else self._computed_max_depth

        # =====================================================================
        # Hilbert-Native Patch Embedding (包含 SharedConv)
        # =====================================================================
        from .embed_hilbert_patch import HilbertNativePatchEmbed
        # I30-17-EXT: 使用动态计算的 max_depth
        # 对于 embedding 层，需要在 __init__ 时确定深度
        # 使用 self.max_depth (可能由用户显式指定，也可能由动态计算得到)
        self.patch_embed = HilbertNativePatchEmbed(
            channels=channels,
            dim=d_model,
            base_patch_size=base_patch_size,
            max_depth=self.max_depth,  # 保持使用计算后的深度
            conv_layers=2,
            use_batch_norm=True,
            depth_scale_range=depth_scale_range,
        )

        # =====================================================================
        # I20: GumbelTopKSplitter (替代 LearnableSplitter)
        # =====================================================================
        # 数学形式化分析结论 (2026-01-10):
        #
        # Scheme A (LearnableSplitter/BFS+STE):
        #   - Hilbert 局部性: 100% ✓
        #   - 梯度覆盖: ~25% ✗ (BFS 串行依赖导致深层饥饿)
        #   - 崩塌风险: 高 (τ/T > 3 时不可逆崩塌)
        #
        # Scheme D (GumbelTopKSplitter):
        #   - Hilbert 局部性: 100% ✓
        #   - 梯度覆盖: 100% ✓ (STE 使所有 85 候选都有梯度)
        #   - 崩塌风险: 低 (无串行依赖)
        #   - LCA 兼容性: 完全 (每个 token 精确对应一个四叉树节点)
        #
        # 结论: Scheme D 是 Hilbert Curve ViT 的最优分割器
        # =====================================================================
        from .gumbel_topk_splitter import GumbelTopKSplitter
        from .config import SplitterConfig

        # I30-10: 使用 SplitterConfig 或传统参数
        if splitter_config is not None:
            # 使用 SplitterConfig
            self.splitter = GumbelTopKSplitter(
                config=splitter_config,
                image_size=image_size,
            )
        else:
            # 使用传统参数（向后兼容）
            # I27-1: 使用传入的 splitter_dropout 而非硬编码值
            # 允许训练器统一控制正则化强度
            # I30-17-EXT: 使用动态计算的 min_patch_size 和 max_depth
            self.splitter = GumbelTopKSplitter(
                feature_dim=d_model,
                min_patch_size=effective_min_patch_size,  # I30-17-EXT: 动态深度计算
                max_depth_limit=self._computed_max_depth,
                hidden_dim=64,
                pool_size=4,
                temperature=learnable_temperature,
                K_min=K_min,
                K_max=K_max,
                dropout=splitter_dropout,  # I27-1: 可配置
                use_dynamic_k=True,
                image_size=image_size,
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
            
        性能特性 (GumbelTopKSplitter - I20):
            - 100% Hilbert 局部性
            - 100% 梯度覆盖 (STE)
            - 无串行依赖
            - 预期加速: ~8x (vs Python 循环)
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
        
        # 2. I20: GumbelTopKSplitter 分割
        from .gumbel_topk_splitter import GumbelTopKSplitter, GumbelTopKResult
        from .split_adaptive import TensorSplitResult
        
        split_result = self.splitter(
            features,
            image_size=(H, W),
            hard=not self.training,
        )
        
        # I20: GumbelTopKResult → TensorSplitResult 转换
        # I30-11: raw_probs 在下面直接使用 split_result.probs 获取
        if isinstance(split_result, GumbelTopKResult):
            tensor_result = split_result.to_tensor_split_result()
        elif isinstance(split_result, TensorSplitResult):
            tensor_result = split_result
        else:
            raise ValueError(f"Unexpected split result type: {type(split_result)}")
        
        # 统计收集 (no_grad)
        with torch.no_grad():
            # I24-14 修复: 使用无条件张量操作 (torch.compile 安全)
            # 原问题: 数据依赖的 if 语句在 torch.compile 下可能被跳过
            # 解决: 无条件计算 bincount，然后无条件 clamp
            
            # bincount 需要至少一个元素，使用 torch.where 处理空情况
            # 创建一个始终有效的 batch_indices (添加一个 dummy 0)
            batch_indices_safe = tensor_result.batch_indices
            if batch_indices_safe.numel() == 0:
                # 极端边界情况：完全没有 token
                batch_indices_safe = torch.zeros(1, dtype=torch.long, device=device)
            
            tokens_per_batch = torch.bincount(batch_indices_safe, minlength=B)
            
            # I24-14: 无条件 clamp (torch.compile 安全)
            # 不使用 .item() 或数据依赖的 if，直接 clamp
            tokens_per_batch = tokens_per_batch.clamp(min=1)
            num_tokens_list = tokens_per_batch.to('cpu', non_blocking=True).tolist()
            
            # 计算 depth distribution
            # P-OPT-3: 使用向量化操作，避免 Python for 循环
            depth_dists = []
            max_d = self.max_depth + 1
            depths = tensor_result.depths
            batch_indices = tensor_result.batch_indices
            
            if tensor_result.num_tokens > 0:
                count_matrix = torch.zeros(B, max_d, dtype=torch.long, device=device)
                # I78: 添加 min clamp 防止负索引 (M3 修复)
                flat_idx = batch_indices * max_d + depths.clamp(min=0, max=max_d - 1)
                ones = torch.ones_like(flat_idx)
                count_matrix.view(-1).scatter_add_(0, flat_idx, ones)
                
                self._last_depth_count_matrix = count_matrix
                
                # P-OPT-3: 延迟转换到 CPU，使用 non_blocking
                # 仅在实际需要 depth_dists 时才转换（统计信息通常只用于日志）
                # 将 dict 构建移到 _build_depth_dists_lazy 方法
                self._depth_count_matrix_for_stats = count_matrix
                depth_dists = None  # 延迟构建
            else:
                depth_dists = [{} for _ in range(B)]
                self._last_depth_count_matrix = None
                self._depth_count_matrix_for_stats = None
        
        # P-OPT-3: 延迟构建 depth_dists (仅在需要时转换)
        if depth_dists is None:
            depth_dists = self._build_depth_dists_lazy(B)
        
        self._last_split_stats = {
            'num_tokens': num_tokens_list,
            'depth_distributions': depth_dists,
        }
        
        # 3. 纯张量嵌入
        # I30-11: 传递 raw_probs 用于构建 padded_split_probs
        raw_probs = split_result.probs if isinstance(split_result, GumbelTopKResult) else None
        tokens, levels_info, padded_regions, padded_split_probs = self._embed_with_tensor_result(
            features, tensor_result, raw_probs
        )
        
        # 4. 构建输出 (P-OPT-4: 向量化输出构建，避免 Python for 循环)
        # TokenSequence 对象仍需构建，但使用预计算的张量切片
        sequences = []
        for b in range(B):
            num_tokens = num_tokens_list[b]
            seq = TokenSequence(
                tokens=tokens[b, :num_tokens],
                metadata={
                    "levels": levels_info[b, :num_tokens],
                    "split_stats": {
                        "num_tokens": num_tokens,
                        "depth_distribution": depth_dists[b] if depth_dists else {},
                    },
                },
            )
            sequences.append(seq)
        
        # P9-5/P12-2 优化: 传入已 padding 的张量缓存，避免 model 中重复 padding
        # I20: 简化输出构建
        # I24-14: num_tokens_list 已在前面 clamp 过，直接使用
        # I30-11: 添加 _split_probs_cache 用于加权池化
        lengths_tensor = torch.tensor(num_tokens_list, dtype=torch.long, device=device)

        return TokenizerOutput(
            sequences=sequences,
            _padded_tokens_cache=tokens,
            _padded_levels_cache=levels_info,
            _lengths_cache=lengths_tensor,
            _regions_cache=padded_regions,
            _image_size_cache=self.image_size,
            _split_probs_cache=padded_split_probs,
        )
    
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
    
    def _tokenize_continuous(
        self,
        features: torch.Tensor,
        probs_result,  # ShallowCandidateProbs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """I10-19: 连续松弛tokenization.
        
        数学形式化
        ==========
        
        架构边界:
            Splitter: P = g_θ(features, candidate_regions)  ← 概率
            Tokenizer: E = f_φ(features, candidate_regions) ← embeddings
            Tokens: T = fuse(E, P)                          ← 连续加权融合
            
        关键特性:
            - 无循环依赖: E和P独立计算，T仅依赖(E, P)
            - 完全可微: 所有操作纯张量，梯度流畅
            - 参数正交: θ_split ⊥ θ_embed
        
        连续融合公式:
            t_d = (1 - p_d) · embed_d + p_d · Σ_{i∈children} α_i · t_child_i
            
        其中:
            p_d: 分割概率 (来自Splitter)
            embed_d: 区域embedding (本方法计算)
            α_i: 子区域归一化权重 (按照Hilbert顺序)
            
        Args:
            features: 共享特征图 [B, C, H_feat, W_feat]
            probs_result: ShallowCandidateProbs (包含概率信息)
            
        Returns:
            (tokens, levels_info, padded_regions, num_tokens_list):
            - tokens: [B, MaxN, D]
            - levels_info: [B, MaxN, max_depth+1]
            - padded_regions: [B, MaxN, 4]
            - num_tokens_list: [B]
        """
        from .split_adaptive import ShallowCandidateProbs
        assert isinstance(probs_result, ShallowCandidateProbs)
        
        B = features.shape[0]
        device = features.device
        dtype = features.dtype
        dim = self.d_model
        
        # =====================================================================
        # Step 1: 预计算所有候选区域的embeddings (向量化)
        # 数学说明: E = f_φ(features, candidate_regions) 独立于 P
        # 
        # 性能优化: 消除 Python for 循环，使用广播构建 boxes
        # =====================================================================
        N_candidates = probs_result.num_candidates
        candidate_regions = probs_result.candidate_regions  # [N_candidates, 4]
        
        # 向量化构建 boxes: [B×N_candidates, 5] -> (batch_idx, x1, y1, x2, y2)
        p = self.base_patch_size
        
        # 创建 batch 索引 [B, N_candidates] → flatten
        batch_indices = torch.arange(B, device=device, dtype=dtype).unsqueeze(1).expand(-1, N_candidates)  # [B, N]
        batch_indices_flat = batch_indices.reshape(-1)  # [B*N]
        
        # 扩展区域坐标 [N_candidates, 4] → [B, N_candidates, 4] → [B*N, 4]
        regions_scaled = candidate_regions.float() / p  # [N, 4]
        regions_expanded = regions_scaled.unsqueeze(0).expand(B, -1, -1)  # [B, N, 4]
        regions_flat = regions_expanded.reshape(B * N_candidates, 4)  # [B*N, 4]
        
        # 组合成 boxes [B*N, 5]
        boxes = torch.cat([batch_indices_flat.unsqueeze(1), regions_flat], dim=1)  # [B*N, 5]
        
        # 确保最小尺寸
        boxes[:, 3] = torch.maximum(boxes[:, 1] + 0.5, boxes[:, 3])
        boxes[:, 4] = torch.maximum(boxes[:, 2] + 0.5, boxes[:, 4])
        
        # ROI-Align (批量)
        from torchvision.ops import roi_align
        pooled = roi_align(
            features,
            boxes,
            output_size=(1, 1),
            spatial_scale=1.0,
            aligned=True,
        ).squeeze(-1).squeeze(-1)  # [B×N_candidates, C]
        
        # Reshape回 [B, N_candidates, C]
        pooled = pooled.view(B, N_candidates, -1)
        
        # 深度编码 (向量化)
        candidate_depths = probs_result.candidate_depths.clamp(max=self.max_depth)  # [N_candidates]
        scales = self.patch_embed.depth_scale[candidate_depths]  # [N_candidates]
        embeds = self.patch_embed.depth_embed(candidate_depths)   # [N_candidates, D]
        
        # 广播到batch维度: [B, N_candidates, D]
        candidate_embeddings = pooled * scales.unsqueeze(0).unsqueeze(-1) + embeds.unsqueeze(0)
        
        # =====================================================================
        # Step 2: 连续加权融合
        # 使用ShallowParallelEvaluator的get_continuous_tokens方法
        # 
        # I10-18增强: 传递threshold参数控制token数量
        # I18-3修复: 提高训练threshold从0.01到0.1
        #   - threshold=0.01 导致所有85个候选都通过 (cumulative_prob > 0.01)
        #   - threshold=0.1 可以有效过滤低权重候选，预期30-60 tokens
        #   - 梯度流通过累积概率保持，无需所有候选都显式保留
        # =====================================================================
        threshold = 0.1 if self.training else 0.1  # 训练和推理使用相同阈值保证一致性
        
        continuous_tokens, token_weights = self.splitter.shallow_evaluator.get_continuous_tokens(
            features=features,                          # [B, C, H_feat, W_feat]
            embeddings=candidate_embeddings,            # [B, N_candidates, D]
            probs=probs_result.probs,                   # [B, N_candidates]
            cumulative_probs=probs_result.cumulative_probs,  # [B, N_candidates]
            embed_dim=dim,                              # D
            threshold=threshold,
        )  # [B, N_output, D], [B, N_output]
        
        # N_output 是实际输出token数量 (通常小于N_candidates)
        B_out, N_output, D_out = continuous_tokens.shape
        assert B_out == B and D_out == dim
        
        # =====================================================================
        # Step 3: 构建输出
        # I18-3 修复: 使用实际有效token数而非输出张量维度
        # =====================================================================
        # 从 evaluator 缓存获取每个batch的有效token数
        if hasattr(self.splitter.shallow_evaluator, '_last_valid_counts'):
            valid_counts = self.splitter.shallow_evaluator._last_valid_counts  # [B]
            num_tokens_list = valid_counts.tolist()
        else:
            # Fallback: 使用 token_weights 计算有效数量
            # 有效token的weight > 0 (非padding)
            num_tokens_list = (token_weights > PROB_EPSILON).sum(dim=1).tolist()
        
        # levels_info: 使用evaluator缓存的深度信息
        # I10-18增强: 从get_continuous_tokens获取实际深度
        levels_info = torch.zeros(B, N_output, self.max_depth + 1, dtype=torch.long, device=device)
        
        if hasattr(self.splitter.shallow_evaluator, '_last_token_depths'):
            token_depths = self.splitter.shallow_evaluator._last_token_depths  # [B, N_output]
            # levels_info[:, :, 0] 存储深度值
            levels_info[:, :, 0] = token_depths.clamp(max=self.max_depth)
        
        # padded_regions: 使用候选区域的前N_output个
        padded_regions = torch.zeros(B, N_output, 4, dtype=torch.long, device=device)
        padded_regions[:, :, :] = candidate_regions[:N_output].unsqueeze(0).expand(B, -1, -1)
        
        return continuous_tokens, levels_info, padded_regions, num_tokens_list
    
    def _build_depth_dists_lazy(self, B: int) -> List[Dict[int, int]]:
        """延迟构建 depth distribution dicts (P-OPT-3).
        
        性能优化:
            - 仅在实际需要时才从 GPU 拷贝到 CPU
            - 使用 non_blocking=True 避免同步等待
            - 使用 numpy 向量化操作替代 Python 循环
        
        Args:
            B: batch size
            
        Returns:
            List of depth distribution dicts
        """
        if self._depth_count_matrix_for_stats is None:
            return [{} for _ in range(B)]
        
        count_matrix = self._depth_count_matrix_for_stats
        max_d = count_matrix.shape[1]
        
        # 转换到 CPU (同步方式，确保数据完整)
        # 注意: 使用 non_blocking=True 会导致 torch.compile 下的竞态条件
        count_matrix_cpu = count_matrix.cpu().numpy()
        
        # 向量化构建 dicts
        depth_dists = []
        for b in range(B):
            row = count_matrix_cpu[b]
            # 使用 numpy where 找到非零元素
            nonzero_mask = row > 0
            nonzero_indices = nonzero_mask.nonzero()[0]
            dist = {int(d): int(row[d]) for d in nonzero_indices}
            depth_dists.append(dist)
        
        return depth_dists

    def _embed_with_tensor_result(
        self,
        features: torch.Tensor,
        tensor_result: "TensorSplitResult",
        raw_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """使用 TensorSplitResult 进行嵌入 (P9-1 完全向量化版本).

        I30-11: 额外返回 padded_split_probs 用于加权池化。

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
        
        # 计算每个 batch 的最大 token 数量 (I78: 优化为张量操作)
        if tensor_result.tokens_per_batch is not None:
            max_tokens = int(tensor_result.tokens_per_batch.max())
        else:
            # 回退: 使用 bincount 计算每个 batch 的 token 数
            token_counts = torch.bincount(tensor_result.batch_indices, minlength=B)
            max_tokens = int(token_counts.max())
        
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

        # I30-11: 构建 padded_split_probs [B, max_tokens]
        # 使用 hilbert_indices 将 raw_probs 映射到正确位置
        padded_split_probs = None
        if raw_probs is not None and N_total > 0 and max_tokens > 0:
            # hilbert_indices 给出每个 selected token 在原始 candidate 列表中的索引
            hilbert_indices = tensor_result.hilbert_indices  # [N_total]
            split_probs_padded = torch.zeros(B, max_tokens, dtype=raw_probs.dtype, device=device)

            # 向量化分配: split_probs_padded[batch_idx, token_pos] = raw_probs[batch_idx, hilbert_idx]
            split_probs_padded[batch_indices, token_positions] = raw_probs[
                batch_indices, hilbert_indices
            ]
            padded_split_probs = split_probs_padded

        return self.patch_embed.norm(tokens), levels_info, padded_regions, padded_split_probs

    def forward(self, images: torch.Tensor) -> TokenizerOutput:
        """前向传播，等价于 tokenize."""
        return self.tokenize(images)

    # I78: 动态分辨率支持
    def update_candidates(self, image_size: Tuple[int, int]):
        """根据输入尺寸动态更新候选区域。

        用于支持动态分辨率输入，无需在初始化时指定固定 image_size。

        Args:
            image_size: (H, W) 输入图像尺寸
        """
        from .gumbel_topk_splitter import GumbelTopKSplitter

        if isinstance(self.splitter, GumbelTopKSplitter):
            self.splitter._update_candidates(image_size)
            # I78: 重置 EMA 统计量以避免维度不匹配
            self.splitter._running_mu = None
            self.splitter._running_sigma = None

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
        """获取熵正则化损失 (用于可学习分割器).
        
        I20: 支持 LearnableSplitter 和 GumbelTopKSplitter 两种分割器。
        """
        if not self._use_learnable_split:
            return None
        
        from .split_adaptive import LearnableSplitter
        from .gumbel_topk_splitter import GumbelTopKSplitter
        
        if self._last_split_stats is None:
            return None
        
        # GumbelTopKSplitter: 使用其内置的熵损失方法
        if isinstance(self.splitter, GumbelTopKSplitter):
            return self.splitter.get_depth_entropy_loss()
        
        # 其他分割器类型不支持
        return None
    
    @torch.no_grad()
    def get_scale_entropy(self) -> Optional[float]:
        """获取尺度分布熵值 (I78: 添加 no_grad 避免梯度计算).

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
            # I78: 使用 clamp 防止除零 (M4 修复)
            total_safe = total.clamp(min=1e-8)
            probs = total_counts / total_safe
            # 避免 log(0)
            log_probs = torch.log(probs + LOG_EPSILON)
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
            log_probs = torch.log(probs + LOG_EPSILON)
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
        """获取训练状态统计信息.
        
        I20: 支持 LearnableSplitter 和 GumbelTopKSplitter 两种分割器。
        """
        from .split_adaptive import LearnableSplitter
        from .gumbel_topk_splitter import GumbelTopKSplitter
        
        # 确定分割器类型
        if isinstance(self.splitter, GumbelTopKSplitter):
            split_scheme = 'gumbel_topk'
        elif isinstance(self.splitter, LearnableSplitter):
            split_scheme = 'learnable'
        else:
            split_scheme = 'unknown'
        
        stats = {
            'tokenizer_version': 'v3_unified',
            'architecture': 'shared_conv + learnable_split + hilbert_embed',
            'split_scheme': split_scheme,
            'max_depth': self.max_depth,
            'learnable_split': True,
        }
        
        if self._last_split_stats:
            avg_tokens = sum(self._last_split_stats['num_tokens']) / len(self._last_split_stats['num_tokens'])
            stats['avg_tokens_per_image'] = avg_tokens
            stats['depth_entropy'] = self.get_scale_entropy()
        
        # 获取分割器特定的统计信息 (仅 GumbelTopKSplitter)
        if isinstance(self.splitter, GumbelTopKSplitter):
            stats['learnable_thresholds'] = self.splitter.thresholds.tolist()
            stats['learnable_temperature'] = self.splitter.current_temperature
        
        return stats
    
    def set_split_temperature(self, temperature: float) -> None:
        """设置可学习分割器的温度 (用于退火调度).
        
        数学形式化:
            T(t) = T_start · (T_end / T_start)^(t / T_total)
            
        I20: 支持 LearnableSplitter 和 GumbelTopKSplitter 两种分割器。
        """
        if self._use_learnable_split:
            from .gumbel_topk_splitter import GumbelTopKSplitter
            
            if isinstance(self.splitter, GumbelTopKSplitter):
                self.splitter.set_temperature(temperature)
    
    def reset_split_statistics(self) -> None:
        """重置可学习分割器的统计信息.
        
        I20: 支持 LearnableSplitter 和 GumbelTopKSplitter 两种分割器。
        """
        if self._use_learnable_split:
            # GumbelTopKSplitter 暂无 reset_statistics 方法
            pass

