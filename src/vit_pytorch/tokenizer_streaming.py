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

import dataclasses
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .base_tokenizer import BaseTokenizer, TokenizerOutput, TokenSequence
from .config import FractalConfig, SemanticSplitterConfig  # I97-5: 合并 config_fractal.py, I110-5: 语义配置
from .constants import LOG_EPSILON, PROB_EPSILON, LEARNABLE_QUOTA_ENABLED  # I12-7: 数值稳定性常量
from .embed_fractal_path import VectorizedPathEncoder  # I12-3: 用于计算路径
from .semantic_redundancy_splitter import SplitResult  # I110-6: 语义分裂器结果


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
        max_level: 最大四叉树深度
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
        # I30-17: 替换固定 max_level 为动态计算
        # 支持 Union[int, Tuple[int, int]] 用于向后兼容
        min_patch_size: Union[int, Tuple[int, int]] = 4,
        max_level: Optional[int] = None,
        use_hilbert_order: bool = True,
        # I98-1: 移除 Splitter 相关参数
        # Splitter 现在是独立组件，通过 tokenize() 参数传入
        # 移除: K_min, K_max, splitter_dropout, splitter_config, enable_learnable_quota
        # 保留 depth_scale_range (用于 patch_embed)
        depth_scale_range: Optional[Tuple[float, float]] = (0.5, 2.0),
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

        # 动态计算 max_level (用于 splitter)
        # 公式: L_max = max(0, floor(log2(min(H, W) / min_patch_size)))
        from .depth_utils import compute_max_level
        self._computed_max_level = compute_max_level(
            image_size, effective_min_patch_size
        )

        # I30-17-EXT: 确定最终使用的 max_level
        # 优先级: 显式指定 max_level > 动态计算
        self.max_level = max_level if max_level is not None else self._computed_max_level

        # =====================================================================
        # Hilbert-Native Patch Embedding (包含 SharedConv)
        # =====================================================================
        from .embed_hilbert_patch import HilbertNativePatchEmbed
        # I30-17-EXT: 使用动态计算的 max_level
        # 对于 embedding 层，需要在 __init__ 时确定深度
        # 使用 self.max_level (可能由用户显式指定，也可能由动态计算得到)
        self.patch_embed = HilbertNativePatchEmbed(
            channels=channels,
            dim=d_model,
            base_patch_size=base_patch_size,
            max_level=self.max_level,  # 保持使用计算后的深度
            conv_layers=2,
            use_batch_norm=True,
            depth_scale_range=depth_scale_range,
        )

        # =====================================================================
        # I98-1: Splitter 现在是独立组件，不在 Tokenizer 内部创建
        #
        # 架构变更:
        #   - 之前: Tokenizer 内部创建 GumbelTopKSplitter
        #   - 现在: Splitter 通过 tokenize() 参数外部传入
        #
        # 职责分离:
        #   - Tokenizer: 特征提取 + 区域编码 + Hilbert 排序
        #   - Splitter: 决策哪些区域需要细分
        # =====================================================================
        # self.splitter 已移除，改为外部传入
        # 保留统计变量用于 tokenize 输出
        self._last_split_stats: Optional[Dict[str, Any]] = None
        self._tokens_per_batch: Optional[torch.Tensor] = None  # 延迟 GPU 计算
        self._count_matrix_cache: Optional[torch.Tensor] = None  # P-OPT: 延迟构建 depth_distributions
        # I107-2: 移除调试缓存变量 (_last_features, _last_depth_count_matrix)
        # 这些仅用于调试，会导致显存泄露

        # =====================================================================
        # I110-6: 语义冗余分裂器支持
        # =====================================================================
        # 语义分裂器相关属性（I110-6）
        self._use_semantic_splitter: bool = False
        self._semantic_config: Optional[SemanticSplitterConfig] = None
        self._semantic_loss_fn: Optional[nn.Module] = None
        # 缓存用于语义分裂的区域边界 [N, 4] (初始全图区域)
        self._region_bounds_cache: Optional[torch.Tensor] = None

    @property
    def patch_sizes(self) -> List[int]:
        """兼容性属性: 从 base_patch_size 和 max_level 计算等效的 patch 大小列表.

        数学形式:
            patch_sizes[d] = base_patch_size × 2^d, d ∈ [0, max_level]

        例如: base_patch_size=4, max_level=4
            → patch_sizes = [4, 8, 16, 32, 64]
            
        Note:
            这是为了向后兼容旧版评估脚本。V3 tokenizer 使用可变深度 token,
            实际 patch 大小由 LearnableSplitter 动态决定。
        """
        return [self.base_patch_size * (2 ** d) for d in range(self.max_level + 1)]
    
    @property
    def shared_conv(self) -> nn.Module:
        """获取共享卷积层 (用于可学习分割)."""
        return self.patch_embed.shared_conv

    # =====================================================================
    # I110-6: 语义冗余分裂器配置方法
    # =====================================================================
    def use_semantic_splitter(
        self,
        config: Optional[SemanticSplitterConfig] = None,
        loss_fn: Optional[nn.Module] = None,
    ) -> None:
        """配置使用语义冗余分裂器 (I110-6)

        Args:
            config: SemanticSplitterConfig 配置（默认使用参数）
            loss_fn: 语义损失函数（默认使用配置中的权重创建）
        """
        from .semantic_redundancy_splitter import SemanticRedundancySplitter
        from .semantic_losses import SemanticRedundancyLoss

        self._use_semantic_splitter = True
        self._semantic_config = config if config is not None else SemanticSplitterConfig()
        self._semantic_config.validate()

        # 创建损失函数
        if loss_fn is not None:
            self._semantic_loss_fn = loss_fn
        else:
            self._semantic_loss_fn = SemanticRedundancyLoss(
                diversity_weight=self._semantic_config.diversity_weight,
                reconstruction_weight=self._semantic_config.reconstruction_weight,
            )

    def get_semantic_splitter(self) -> Optional["SemanticRedundancySplitter"]:
        """获取语义分裂器实例（用于模型前向）"""
        if not self._use_semantic_splitter or self._semantic_config is None:
            return None

        from .semantic_redundancy_splitter import SemanticRedundancySplitter

        return SemanticRedundancySplitter(
            feature_dim=self.d_model,
            hidden_dim=self._semantic_config.hidden_dim,
            max_level_limit=self.max_level,
            gumbel_temp_start=self._semantic_config.gumbel_temp_start,
            gumbel_temp_end=self._semantic_config.gumbel_temp_end,
            learnable_temperature=self._semantic_config.learnable_temperature,
        )

    def get_semantic_loss_fn(self) -> Optional[nn.Module]:
        """获取语义损失函数"""
        return self._semantic_loss_fn

    def _get_initial_region_bounds(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """获取初始区域边界（全图）[B*N_initial, 4]"""
        if self._region_bounds_cache is not None:
            return self._region_bounds_cache

        H, W = self.image_size
        # 初始只有一个区域：整个图像
        # 格式: [x0, y0, x1, y1]
        initial_bounds = torch.tensor([0, 0, W, H], dtype=torch.float32, device=device)
        self._region_bounds_cache = initial_bounds
        return initial_bounds

    # =====================================================================
    # I110-6: SplitResult 转换方法
    # =====================================================================
    def convert_split_result_to_tensor(
        self,
        split_result: SplitResult,
        features: torch.Tensor,
    ) -> "TensorSplitResult":
        """将 SplitResult 转换为 TensorSplitResult (I110-6)

        语义分裂器返回的是 split_decision（二值掩码），需要转换为
        实际的 regions/depths/batch_indices 用于 tokenization。

        Args:
            split_result: SplitResult 包含 split_decision [B, N]
            features: [B, C, H, W] 输入图像

        Returns:
            TensorSplitResult: 兼容 TensorSplitResult 格式的分割结果
        """
        from .gumbel_topk_splitter import TensorSplitResult

        B, C, H, W = features.shape
        device = features.device
        split_decision = split_result.split_decision  # [B, N]

        # 递归构建四叉树区域
        # 初始区域：整个图像
        all_regions = []  # [total_regions, 4]
        all_depths = []  # [total_regions]
        all_batch_indices = []  # [total_regions]

        # BFS 构建四叉树
        queue = []  # (bounds, depth, batch_idx)
        for b in range(B):
            bounds = self._get_initial_region_bounds(B, device)
            queue.append((bounds, 0, b))

        while queue:
            bounds, depth, b_idx = queue.pop(0)

            if depth >= self.max_level:
                # 达到最大深度，添加为叶子节点
                region_idx = len(all_regions)
                all_regions.append(bounds)
                all_depths.append(depth)
                all_batch_indices.append(b_idx)
                continue

            # 计算当前区域的索引
            # 这里简化处理：假设区域按 BFS 顺序排列
            # 实际需要更复杂的索引映射
            region_idx_in_batch = len([r for r, d, b in queue if b == b_idx]) + \
                                  len([r for r, d, b in queue if b != b_idx]) + \
                                  len([1 for r, d, b in all_regions if b == b_idx])

            # 检查是否应该分裂
            if region_idx_in_batch < split_decision.shape[1]:
                should_split = split_decision[b_idx, region_idx_in_batch] > 0.5
            else:
                should_split = False

            if should_split:
                # 分裂为四个子区域
                x0, y0, x1, y1 = bounds.tolist()
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2

                # 添加四个子区域到队列
                child_bounds = [
                    torch.tensor([x0, y0, cx, cy], dtype=torch.float32, device=device),  # 左上
                    torch.tensor([cx, y0, x1, cy], dtype=torch.float32, device=device),  # 右上
                    torch.tensor([x0, cy, cx, y1], dtype=torch.float32, device=device),  # 左下
                    torch.tensor([cx, cy, x1, y1], dtype=torch.float32, device=device),  # 右下
                ]

                for child_bound in child_bounds:
                    queue.append((child_bound, depth + 1, b_idx))
            else:
                # 不分裂，添加为叶子节点
                all_regions.append(bounds)
                all_depths.append(depth)
                all_batch_indices.append(b_idx)

        # 转换为张量
        if len(all_regions) == 0:
            regions = torch.zeros(0, 4, dtype=torch.float32, device=device)
            depths = torch.zeros(0, dtype=torch.long, device=device)
            batch_indices = torch.zeros(0, dtype=torch.long, device=device)
        else:
            regions = torch.stack(all_regions)
            depths = torch.tensor(all_depths, dtype=torch.long, device=device)
            batch_indices = torch.tensor(all_batch_indices, dtype=torch.long, device=device)

        # 计算 Hilbert 索引
        from .curve_hilbert import xy_to_hilbert_distance
        hilbert_indices = xy_to_hilbert_distance(
            (regions[:, 0] + regions[:, 2]) / 2,  # center_x
            (regions[:, 1] + regions[:, 3]) / 2,  # center_y
            max_bits=16,
        ).to(device)

        # 复杂度使用冗余性分数
        complexities = split_result.redundancy.view(-1) if split_result.redundancy.numel() > 0 else \
                       torch.zeros(len(regions), dtype=torch.float32, device=device)

        return TensorSplitResult(
            regions=regions,
            depths=depths,
            batch_indices=batch_indices,
            hilbert_indices=hilbert_indices,
            complexities=complexities,
        )
    
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
            max_level=config.max_level,
            use_hilbert_order=True,
        )
    
    def tokenize(
        self,
        images: torch.Tensor,
        split_result: "GumbelTopKResult",
    ) -> TokenizerOutput:
        """Variable Depth tokenization (P9-1 方案 D: 完全向量化).

        I98-1 架构变更:
            - Splitter 从外部传入，不再内部创建
            - 职责分离: Tokenizer 只负责 embedding，Splitter 负责分割决策

        数学形式化:
            1. F = SharedConv(I)                    # 特征提取
            2. T = _embed_with_tensor_result(F, TensorResult)  # 纯张量嵌入
               其中 TensorResult 来自外部 Splitter

        Args:
            images: [B, C, H, W] 输入图像
            split_result: 外部 Splitter 的输出结果
                包含: regions, depths, batch_indices, hilbert_indices,
                      selected_mask, logits, probs

        Returns:
            TokenizerOutput: 包含 tokens, levels_info, regions 等

        I35: 支持 channels_last 内存格式以优化卷积性能
        """
        from .gumbel_topk_splitter import GumbelTopKResult
        from .gumbel_topk_splitter import TensorSplitResult

        if images.dim() != 4:
            raise ValueError(
                f"StreamingFractalTokenizerV3.tokenize expects 4D input [B, C, H, W], "
                f"got {images.dim()}D tensor."
            )

        B, C, H, W = images.shape
        device = images.device

        # I35: 转换为 channels_last 以优化卷积性能
        # I108-2: 使用 is_contiguous() 正确检测内存格式，而非错误的 stride 比较
        if images.dim() == 4 and not images.is_contiguous(memory_format=torch.channels_last):
            images = images.to(memory_format=torch.channels_last)

        # 1. 提取共享特征图
        features = self.shared_conv(images)  # [B, d_model, H/p, W/p]
        # I107-2: 移除调试缓存，防止显存泄露
        # 调试可视化可通过回调钩子实现，不应在核心代码中持有引用

        # 2. I98-1: 使用外部传入的 split_result
        # GumbelTopKResult → TensorSplitResult 转换
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
            # P-OPT: 保持 GPU 计算，支持 torch.compile + cudagraphs
            max_tokens_per_batch = tokens_per_batch.max()  # 保持在 GPU

            # 计算 depth distribution
            # P-OPT-3: 使用向量化操作，避免 Python for 循环
            depth_dists = []
            max_d = self.max_level + 1
            depths = tensor_result.depths
            batch_indices = tensor_result.batch_indices

            if tensor_result.num_tokens > 0:
                count_matrix = torch.zeros(B, max_d, dtype=torch.long, device=device)
                # I78: 添加 min clamp 防止负索引 (M3 修复)
                flat_idx = batch_indices * max_d + depths.clamp(min=0, max=max_d - 1)
                ones = torch.ones_like(flat_idx)
                count_matrix.view(-1).scatter_add_(0, flat_idx, ones)
            else:
                count_matrix = torch.zeros(B, max_d, dtype=torch.long, device=device)

        # P-OPT: 延迟 depth_distributions 构建，避免 forward 路径中的 .cpu() 调用
        # depth_dists 仅用于日志记录，不应在 forward 路径中阻塞
        self._last_split_stats = {
            'num_tokens': None,  # 延迟计算，按需在外部转换
            'depth_distributions': None,  # 延迟构建
        }
        # 缓存 count_matrix 用于延迟构建 depth_distributions
        self._count_matrix_cache: Optional[torch.Tensor] = count_matrix if tensor_result.num_tokens > 0 else None
        # 清空之前的缓存，强制重新计算
        self._num_tokens_list = None
        # 缓存 tokens_per_batch 用于 get_training_stats (延迟平均计算)
        self._tokens_per_batch: Optional[torch.Tensor] = tokens_per_batch

        # I78: 计算 max_tokens 用于 _embed_with_tensor_result
        # I99-1 FIX: 使用每个 batch 的最大 token 数，而非总 token 数
        # N_total 是实际选择的 token 总数，但 max_tokens 应该是每个 batch 的最大 token 数
        N_total = tensor_result.num_tokens
        max_tokens = max_tokens_per_batch  # 每个 batch 的最大 token 数

        # 3. 纯张量嵌入
        # I30-11: 传递 raw_probs 用于构建 padded_split_probs
        # I78-2: 传递 selected_mask 用于替换 threshold 机制
        raw_probs = split_result.probs if isinstance(split_result, GumbelTopKResult) else None
        selected_mask = split_result.selected_mask if isinstance(split_result, GumbelTopKResult) else None
        tokens, levels_info, padded_regions, padded_split_probs = self._embed_with_tensor_result(
            features, tensor_result, raw_probs, max_tokens=max_tokens,
            selected_mask=selected_mask
        )

        # 4. 构建输出 (P-OPT-4: 向量化输出构建，避免 Python for 循环)
        # TokenSequence 对象仍需构建，但使用预计算的张量切片
        # P-OPT: 从 count_matrix 直接在 GPU 上构建 depth_distribution，避免 .cpu()
        # I145: 修复 GPU 同步问题 - 批量转换 tokens_per_batch 到 CPU
        # 原始: int(tokens_per_batch[b].item()) 在循环中调用 B 次 .item()
        # 修复: 一次性转换到 CPU，再在循环中使用 Python 值
        tokens_per_batch_cpu = tokens_per_batch.cpu().tolist() if tokens_per_batch.is_cuda else tokens_per_batch.tolist()
        sequences = []
        for b in range(B):
            # P-OPT: 使用预转换的 CPU 值，避免 GPU 同步
            num_tokens = tokens_per_batch_cpu[b]
            # 从 count_matrix[b] 在 GPU 上构建 depth_distribution
            row = count_matrix[b]  # [max_d]
            nonzero_mask = row > 0
            nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0]
            depth_dist = {int(d): int(row[d]) for d in nonzero_indices}
            seq = TokenSequence(
                tokens=tokens[b, :num_tokens],
                metadata={
                    "levels": levels_info[b, :num_tokens],
                    "split_stats": {
                        "num_tokens": num_tokens,
                        "depth_distribution": depth_dist,
                    },
                },
            )
            sequences.append(seq)

        # P9-5/P12-2 优化: 传入已 padding 的张量缓存，避免 model 中重复 padding
        # I20: 简化输出构建
        # I24-14: 使用 tokens_per_batch 作为 lengths_tensor (已在 GPU 上)
        lengths_tensor = tokens_per_batch.long()

        # P-OPT: 使用延迟构建，不传递 sequences
        # sequences 会在首次访问时通过 _build_sequences_from_cache() 延迟构建
        return TokenizerOutput(
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
            levels_info = torch.zeros(B, 1, self.max_level + 1, dtype=torch.long, device=device)
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
                all_depths.append(min(token.depth, self.max_level))
                all_levels_info.append(token.to_levels_info(self.max_level))
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
        levels_info = torch.zeros(B, max_tokens, self.max_level + 1, dtype=torch.long, device=device)
        
        # 使用高级索引进行向量化分配
        batch_idx_tensor = torch.tensor(batch_indices, device=device, dtype=torch.long)
        token_idx_tensor = torch.tensor(token_indices, device=device, dtype=torch.long)
        
        # 向量化 token 分配 (确保 dtype 匹配，支持 AMP 混合精度)
        tokens[batch_idx_tensor, token_idx_tensor] = all_tokens.to(dtype)
        
        # 向量化 levels_info 分配
        levels_info_tensor = torch.tensor(all_levels_info, device=device, dtype=torch.long)
        levels_info[batch_idx_tensor, token_idx_tensor] = levels_info_tensor
        
        return self.patch_embed.norm(tokens), levels_info

    def _build_depth_dists_lazy(self, B: int, count_matrix: Optional[torch.Tensor] = None) -> List[Dict[int, int]]:
        """延迟构建 depth distribution dicts (P-OPT-3).

        性能优化:
            - 仅在实际需要时才从 GPU 拷贝到 CPU
            - 使用 non_blocking=True 避免同步等待
            - 使用 numpy 向量化操作替代 Python 循环

        Args:
            B: batch size
            count_matrix: depth count matrix [B, max_d], 如果为 None 则返回空 dicts

        Returns:
            List of depth distribution dicts
        """
        if count_matrix is None:
            return [{} for _ in range(B)]

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
        max_tokens: int = 1,
        selected_mask: Optional[torch.Tensor] = None,
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
            max_tokens: int 实际输出的 token 数量 (来自 get_continuous_tokens)

        Returns:
            (tokens, levels_info, padded_regions):
            - tokens: [B, MaxN, D] 嵌入后的 tokens
            - levels_info: [B, MaxN, max_level+1] 层级信息
            - padded_regions: [B, MaxN, 4] 区域边界 (P11-3 新增)
        """
        from .gumbel_topk_splitter import TensorSplitResult
        
        B = features.shape[0]
        device = features.device
        dtype = features.dtype
        dim = self.d_model
        
        N_total = tensor_result.num_tokens
        if N_total == 0:
            tokens = torch.zeros(B, 1, dim, device=device, dtype=dtype)
            # I32-2: 使用-1 sentinel标识padding token，避免与有效depth=0混淆
            levels_info = torch.full((B, 1, self.max_level + 1), -1, dtype=torch.long, device=device)
            padded_regions = torch.zeros(B, 1, 4, dtype=torch.long, device=device)
            return self.patch_embed.norm(tokens), levels_info, padded_regions

        # I78: max_tokens 由调用者从 continuous_tokens.shape[1] 传入
        # 避免重复计算，确保与实际输出形状一致

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
        # I34-7 Fix: Add min=0 boundary protection to prevent negative depth index errors
        depths = tensor_result.depths.clamp(min=0, max=self.max_level)
        scales = self.patch_embed.depth_scale[depths]  # [N]
        embeds = self.patch_embed.depth_embed(depths)   # [N, D]
        all_tokens = pooled * scales.unsqueeze(-1) + embeds  # [N, D]
        
        # ====================================================================
        # 向量化分配到输出 buffer
        # ====================================================================
        # I99-1: 防御性检查 - 确保 max_tokens 至少为 1
        max_tokens_safe = max(1, max_tokens)
        tokens = torch.zeros(B, max_tokens_safe, dim, device=device, dtype=dtype)
        # I32-2: 使用-1 sentinel标识padding token，避免与有效depth=0混淆
        levels_info = torch.full((B, max_tokens_safe, self.max_level + 1), -1, dtype=torch.long, device=device)
        padded_regions = torch.zeros(B, max_tokens_safe, 4, dtype=torch.long, device=device)  # P11-3

        # I99-1: 修复 batch 独立性 - 确保按 (batch_idx, hilbert_idx) 排序
        # 问题: batch_indices 可能未按 batch 分组，导致 token 位置计算错误
        # 解决: 显式排序所有相关张量
        if N_total > 0:
            # 获取 hilbert_indices 用于排序
            hilbert_idx_for_sort = tensor_result.hilbert_indices

            # I99-1 FIX: torch.lexsort 可能不可用，使用 torch.argsort 替代
            # lexsort 的语义是: 先按最后一列排序，再按倒数第二列排序...
            # 所以我们先按 hilbert_indices 排序 (稳定)，再按 batch_indices 排序 (稳定)
            hilbert_order = torch.argsort(hilbert_idx_for_sort, stable=True)
            batch_order = torch.argsort(batch_indices[hilbert_order], stable=True)
            sort_indices = hilbert_order[batch_order]

            # 重新排列所有张量
            batch_indices_sorted = batch_indices[sort_indices]
            all_tokens_sorted = all_tokens[sort_indices]
            depths_sorted = depths[sort_indices]
            regions_sorted = tensor_result.regions[sort_indices]

            # 如果有 raw_probs 和 hilbert_indices，也要重新排列
            if raw_probs is not None:
                raw_probs_sorted = raw_probs[batch_indices, hilbert_idx_for_sort][sort_indices]
            else:
                raw_probs_sorted = None

            # 更新引用
            batch_indices = batch_indices_sorted
            all_tokens = all_tokens_sorted
            depths = depths_sorted
            tensor_result_regions_sorted = regions_sorted
            raw_probs = raw_probs_sorted
        else:
            tensor_result_regions_sorted = tensor_result.regions
            raw_probs = None

        # 计算每个 token 在其 batch 内的索引
        # 排序后: batch_indices 严格按 batch 分组 [0,0,...,0, 1,1,...,1, ...]
        if N_total == 0:
            token_positions = torch.zeros(0, dtype=torch.long, device=device)
        else:
            # I99-1 FIX: 使用更简单直接的方法计算 batch 内位置
            # batch_starts[b] = batch b 在全局数组中的起始位置
            batch_starts = torch.zeros(B, dtype=torch.long, device=device)
            batch_counts = torch.bincount(batch_indices, minlength=B)
            # 计算每个 batch 的起始位置 (前缀和)
            batch_starts[1:] = batch_counts[:-1].cumsum(dim=0)

            # token_positions = 全局位置 - 该 batch 的起始位置
            global_positions = torch.arange(N_total, device=device)
            token_positions = global_positions - batch_starts[batch_indices]

            # I99-1: 防御性边界检查 - 钳制 token_positions 到 [0, max_tokens_safe-1]
            # 防止由于 splitter 异常导致的越界访问
            token_positions = token_positions.clamp(min=0, max=max_tokens_safe - 1)

        # I99-1: 防御性检查 - 确保 batch_indices 在有效范围内
        if N_total > 0:
            batch_indices = batch_indices.clamp(min=0, max=B - 1)

        # 向量化分配
        tokens[batch_indices, token_positions] = all_tokens.to(dtype)
        
        # =====================================================================
        # I12-3 修复: 从 regions 计算四叉树路径填充 levels_info
        # 原问题: levels_info[:, :, 1:] 全为零，导致所有 token 共享相同的路径编码
        # 修复方案: 使用 VectorizedPathEncoder.compute_paths_from_regions 计算正确路径
        # =====================================================================
        levels_info[batch_indices, token_positions, 0] = depths

        # 计算四叉树路径 (基于区域中心的空间位置)
        # regions: [N_total, 4] -> paths: [N_total, max_level]
        if N_total > 0:
            # 获取图像尺寸 (Hilbert 曲线要求方形，使用较大边)
            img_size = max(self.image_size) if isinstance(self.image_size, tuple) else self.image_size
            paths = VectorizedPathEncoder.compute_paths_from_regions(
                tensor_result_regions_sorted,  # [N_total, 4] - 使用排序后的 regions
                img_size,
                self.max_level
            )  # [N_total, max_level]

            # 填充路径到 levels_info
            # levels_info 格式: [depth, path[0], path[1], ..., path[max_level-1]]
            levels_info[batch_indices, token_positions, 1:] = paths

        # P11-3: 分配 regions 到 padded buffer
        padded_regions[batch_indices, token_positions] = tensor_result_regions_sorted

        # I30-11: 构建 padded_split_probs [B, max_tokens]
        # I99-1: 使用排序后的 raw_probs
        padded_split_probs = None
        if raw_probs is not None and N_total > 0 and max_tokens_safe > 0:
            split_probs_padded = torch.zeros(B, max_tokens_safe, dtype=raw_probs.dtype, device=device)

            # 向量化分配: split_probs_padded[batch_idx, token_pos] = raw_probs_sorted[...]
            split_probs_padded[batch_indices, token_positions] = raw_probs
            padded_split_probs = split_probs_padded

        return self.patch_embed.norm(tokens), levels_info, padded_regions, padded_split_probs

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

    @torch.no_grad()
    def get_scale_entropy(self) -> Optional[float]:
        """获取尺度分布熵值 (I78: 添加 no_grad 避免梯度计算).

        I107-2: 移除了向量化张量缓存，仅使用 Python dict 方式计算。

        数学形式:
            H = -∑_d p_d log(p_d)
            其中 p_d = count_d / ∑_d' count_d'
        """
        if self._count_matrix_cache is None:
            return None

        # P-OPT: 从 GPU tensor 直接计算熵，避免 .cpu()
        count_matrix = self._count_matrix_cache  # [B, max_d]
        # 计算每个 batch 的分布，然后取平均
        row_sums = count_matrix.sum(dim=1, keepdim=True).clamp(min=1)
        probs = count_matrix / row_sums  # [B, max_d]
        # 计算熵: H = -sum(p * log(p))
        # 避免 log(0) 问题
        probs_safe = probs + (probs == 0).float() * 1e-10
        entropy_per_batch = -(probs_safe * probs_safe.log()).sum(dim=1)
        return float(entropy_per_batch.mean().item())

    @torch.no_grad()
    def compute_scale_distribution(
        self,
        images: torch.Tensor,
        split_result: Optional["GumbelTopKResult"] = None,
    ) -> Dict[str, Any]:
        """计算深度分布统计信息.

        I98-1: 需要外部传入 split_result，因为 Splitter 已不再是内部组件。
        I107-2: 移除了缓存依赖，不再使用 _last_features 和 _last_depth_count_matrix。

        Args:
            images: [B, C, H, W] 输入图像
            split_result: 外部 Splitter 的输出结果（可选，如果未提供则尝试自动获取）

        Returns:
            Dict[str, Any]: 深度分布统计信息
        """
        from .gumbel_topk_splitter import GumbelTopKResult

        # I98-1: 如果未提供 split_result，尝试从外部获取
        if split_result is None:
            # 尝试从 model.splitter 获取 (使用 weakref 避免循环引用)
            if hasattr(self, '_model') and callable(self._model):
                model_ref = self._model()
                if model_ref is not None and hasattr(model_ref, 'splitter'):
                    model = model_ref
                    # 获取特征图
                    features = self.shared_conv(images)
                    # 调用 splitter
                    split_result = model.splitter(features, image_size=(images.shape[2], images.shape[3]))

            # 如果仍然无法获取 split_result，返回空统计
            if split_result is None:
                return {
                    'scale_ratios': {},
                    'entropy': 0.0,
                    'max_entropy': 0.0,
                    'dominant_scale': self.base_patch_size,
                    'depth_distribution': {},
                    'warning': 'split_result not available - tokenizer has no access to splitter'
                }

        _ = self.tokenize(images, split_result)

        if self._last_split_stats is None:
            return {
                'scale_ratios': {},
                'entropy': 0.0,
                'max_entropy': 0.0,
                'dominant_scale': self.base_patch_size,
                'depth_distribution': {},
            }

        # I112: 使用 _count_matrix_cache 计算深度分布
        if self._count_matrix_cache is None:
            return {
                'scale_ratios': {},
                'entropy': 0.0,
                'max_entropy': 0.0,
                'dominant_scale': self.base_patch_size,
                'depth_distribution': {},
            }

        # 从 count_matrix_cache 计算总分布
        count_matrix = self._count_matrix_cache  # [B, max_d]
        # 对所有 batch 求和
        total_counts = count_matrix.sum(dim=0)  # [max_d]
        total_dist = {int(d): int(total_counts[d]) for d in range(len(total_counts)) if total_counts[d] > 0}

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
            ps = self.base_patch_size * (2 ** (self.max_level - depth))
            scale_ratios[ps] = count / total_tokens

        entropy = 0.0
        for count in total_dist.values():
            p = count / total_tokens
            if p > 0:
                entropy -= p * math.log(p)

        num_depths = self.max_level + 1
        max_entropy = math.log(num_depths) if num_depths > 1 else 0.0

        dominant_depth = max(total_dist.keys(), key=lambda d: total_dist[d])
        dominant_scale = self.base_patch_size * (2 ** (self.max_level - dominant_depth))

        return {
            'scale_ratios': scale_ratios,
            'entropy': entropy,
            'max_entropy': max_entropy,
            'dominant_scale': dominant_scale,
            'depth_distribution': total_dist,
        }

    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练状态统计信息.

        I98-1: 移除了 splitter 特定统计信息.
        Splitter 统计信息现在由外部管理 (model.splitter.get_diagnostics()).

        Returns:
            Dict[str, Any]: 基础统计信息（不含 splitter 相关）
        """
        stats = {
            'tokenizer_version': 'v3_unified',
            'architecture': 'shared_conv + hilbert_embed',
            'max_level': self.max_level,
            'learnable_split': True,
        }

        if self._last_split_stats:
            # P-OPT: 从 GPU tensor 计算平均，避免 .to('cpu')
            if self._tokens_per_batch is not None:
                avg_tokens = self._tokens_per_batch.float().mean().item()
                stats['avg_tokens_per_image'] = avg_tokens
                stats['depth_entropy'] = self.get_scale_entropy()

        return stats


