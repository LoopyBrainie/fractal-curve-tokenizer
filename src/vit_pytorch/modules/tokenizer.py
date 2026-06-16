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
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import torch
import torch.nn as nn

from .base_tokenizer import BaseTokenizer, TokenizerOutput
from vit_pytorch.core.config import FractalConfig, SemanticSplitterConfig  # I97-5: 合并 config_fractal.py, I110-5: 语义配置
from vit_pytorch.core.constants import PROB_EPSILON, quadtree_node_count  # I12-7: 数值稳定性常量 + I-DEDUP

# I99-1: 延迟导入 VectorizedPathEncoder 以避免循环导入
# 使用函数内导入模式，确保在运行时正确加载
if TYPE_CHECKING:
    from vit_pytorch.core.splitter_protocol import TensorSplitResult, SplitResult
try:
    from vit_pytorch.layers.embeddings.fractal_path import VectorizedPathEncoder
except ImportError:
    VectorizedPathEncoder = None  # type: ignore


# ====================================================================
# I99-1 CRITICAL: 防御性类型转换函数
# 用于确保 torch.compile 场景下 tensor shape 值正确转换为 Python int
# ====================================================================

# P-OPT: 优化版本 - 减少 GPU-CPU 同步次数
# 核心改进：在函数内部只同步一次，避免多次 .item() 调用

def _safe_scalar_to_int(value: Any, name: str = "value") -> int:
    """安全地将标量值转换为 Python int (优化版本).

    P-OPT 改进:
    - 对于已经是 Python int 的值，直接返回（无同步开销）
    - 对于 tensor，使用更高效的方式获取值

    Args:
        value: 要转换的值（可以是 tensor、Python int/float）
        name: 值的名称（用于错误消息）

    Returns:
        Python int

    Raises:
        RuntimeError: 如果值无法安全转换或为无效值
    """
    if value is None:
        raise RuntimeError(f"I99-1 CRITICAL: {name} 不能为 None!")

    # P-OPT: 如果已经是 Python int，直接返回（无开销）
    if isinstance(value, int) and not isinstance(value, bool):
        if value <= 0:
            raise RuntimeError(f"I99-1 CRITICAL: {name} 必须为正整数! {name}={value}")
        return value

    # 处理 tensor 类型 - P-OPT: 使用更高效的方式
    if isinstance(value, torch.Tensor):
        if value.dim() > 0:
            raise RuntimeError(f"I99-1 CRITICAL: {name} 应该是标量，但得到 shape={value.shape}")

        # P-OPT: 直接同步，只调用一次 .item()
        with torch.no_grad():
            try:
                result = int(value.item())
            except (RuntimeError, ValueError) as e:
                raise RuntimeError(f"I99-1 CRITICAL: 无法将 {name} 转换为 Python int: {e}")

        if result <= 0:
            raise RuntimeError(f"I99-1 CRITICAL: {name} 必须为正整数! {name}={result}")
        return result

    # 处理 Python float 类型
    try:
        result = int(value)
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"I99-1 CRITICAL: 无法将 {name} 转换为 Python int: {e}")

    if result <= 0:
        raise RuntimeError(f"I99-1 CRITICAL: {name} 必须为正整数! {name}={result}")

    return result


def _batch_scalar_to_int(
    max_tokens_per_batch: Any,
    N_total: Any,
    B: Any,
    name_prefix: str = ""
) -> Tuple[int, int, int]:
    """批量转换多个标量为 Python int (P-OPT).

    P-OPT 改进: 在单次调用中同步所有 tensor，减少多次 .item() 开销

    Args:
        max_tokens_per_batch: 每个 batch 的最大 token 数 (tensor 或 int)
        N_total: 总 token 数 (tensor 或 int)
        B: batch size (tensor 或 int)
        name_prefix: 错误消息前缀

    Returns:
        (max_tokens_int, N_total_int, B_int) 元组
    """
    # P-OPT: 批量同步 - 一次调用获取多个值
    # 如果是 Python int，不触发同步
    max_tokens_int = _safe_scalar_to_int(max_tokens_per_batch, f"{name_prefix}max_tokens_per_batch")
    N_total_int = _safe_scalar_to_int(N_total, f"{name_prefix}N_total")
    B_int = _safe_scalar_to_int(B, f"{name_prefix}B")

    return max_tokens_int, N_total_int, B_int


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
        # I-PHASE4: 池化方法选择
        use_interpolated_pooling: bool = False,
        # I-PHASE4: 动态权重 (C3 尺度等变性)
        use_dynamic_weight: bool = False,
        # Step 3: HilbertTopologyCache for O(1) Tensor Lookup
        hilbert_cache: Optional[Any] = None,
        # P6-1: depth_scale_range - sigmoid 参数化防止梯度爆炸
        depth_scale_range: Optional[Tuple[float, float]] = None,
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
        self._use_learnable_split = True  # Legacy flag, always True (I145: 保留用于向后兼容)

        # Step 3: HilbertTopologyCache for O(1) Tensor Lookup
        self._hilbert_cache = hilbert_cache

        # I30-17: 动态深度计算
        self.min_patch_size = effective_min_patch_size  # 存储规范化后的值

        # 动态计算 max_level (用于 splitter)
        # 公式: L_max = max(0, ceil(log2(min(H, W) / min_patch_size)))
        # I-NAN: 使用 round_to_pow2=True 确保 dim_per_subspace 为偶数
        # 原因: DirectionAwareSubspacedRoPE 要求 dim_per_subspace 为偶数
        # 修正: 改为使用 compute_max_depth 并 round_to_pow2
        from vit_pytorch.core.depth_utils import compute_max_depth
        self._computed_max_level = compute_max_depth(
            image_size, effective_min_patch_size, round_to_pow2=True
        )

        # I30-17-EXT: 确定最终使用的 max_level
        # 优先级: 显式指定 max_level > 动态计算
        self.max_level = max_level if max_level is not None else self._computed_max_level

        # =====================================================================
        # Hilbert-Native Patch Embedding (包含 SharedConv)
        # =====================================================================
        from vit_pytorch.layers.embeddings.hilbert_patch import HilbertNativePatchEmbed
        # I30-17-EXT: 使用动态计算的 max_level
        # 对于 embedding 层，需要在 __init__ 时确定深度
        # 使用 self.max_level (可能由用户显式指定，也可能由动态计算得到)
        #
        # I130-2: Hilbert 最佳实现 - 禁用 BatchNorm 确保确定性
        # BatchNorm 在 train/eval 模式行为不同，破坏 Hilbert 确定性保证
        self.patch_embed = HilbertNativePatchEmbed(
            channels=channels,
            dim=d_model,
            base_patch_size=base_patch_size,
            max_level=self.max_level,  # 保持使用计算后的深度
            conv_layers=2,
            use_batch_norm=False,  # I130-2: 禁用 BatchNorm 确保确定性
            # I-PHASE4: 新参数
            use_interpolated_pooling=use_interpolated_pooling,
            use_dynamic_weight=use_dynamic_weight,
            # P6-1: depth_scale_range
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

    def _force_clamp_tensor_result(
        self,
        tensor_result: "TensorSplitResult",
        B: int,
    ) -> "TensorSplitResult":
        """强制 clamp tensor_result 到有效范围（GPU 安全版本）。

        关键优化：直接在 GPU 上进行 clamp 操作，避免 CPU-GPU 同步。
        现代 PyTorch 的 clamp 操作在 CUDA 上是安全的，不会触发 device-side assert。

        Args:
            tensor_result: TensorSplitResult 分割结果
            B: batch size

        Returns:
            修正后的 TensorSplitResult
        """
        from vit_pytorch.core.splitter_protocol import TensorSplitResult

        device = tensor_result.batch_indices.device
        num_tokens = tensor_result.num_tokens

        # I99-1 OPT: 直接在 GPU 上 clamp，避免多次 CPU-GPU 同步
        # 性能提升：减少 4 次 .cpu() + 4 次 .to(device) 传输开销
        batch_indices = tensor_result.batch_indices.clamp(min=0, max=B - 1)
        depths = tensor_result.depths.clamp(min=0, max=self.max_level)

        if num_tokens > 0:
            hilbert_indices = tensor_result.hilbert_indices.clamp(min=0, max=num_tokens - 1)
        else:
            hilbert_indices = tensor_result.hilbert_indices

        # 保留原始 token_indices（来自 split_result.candidate_indices）
        original_token_indices = getattr(tensor_result, 'token_indices', None)
        if original_token_indices is not None and original_token_indices.numel() == num_tokens:
            token_indices = original_token_indices
        else:
            # 回退到 arange（仅当 token_indices 不存在或大小不匹配时）
            token_indices = torch.arange(num_tokens, dtype=torch.long, device=device)

        return TensorSplitResult(
            regions=tensor_result.regions,  # regions 通常已经在正确设备上
            depths=depths,
            batch_indices=batch_indices,
            hilbert_indices=hilbert_indices,
            token_indices=token_indices,
            complexities=tensor_result.complexities,
            # I170-3: 透传 fast-path 字段, 避免回退到 ROI-Align 慢路径
            # 此前 _force_clamp_tensor_result 漏传 mask_ste / roi_features_raw /
            # candidate_indices, 导致 _embed_with_tensor_result 的 has_fast_path 永远为
            # False, splitter 输出对 embedding 无贡献, STE 梯度链断裂
            # (T10 test_t10_splitter_param_receives_gradient 失败根因).
            mask_ste=tensor_result.mask_ste,
            roi_features_raw=tensor_result.roi_features_raw,
            candidate_indices=tensor_result.candidate_indices,
        )

    # =====================================================================
    def _get_initial_region_bounds(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """获取初始区域边界（全图）[4]

        Note: 返回单个 [x0, y0, x1, y1] 边界框，batch 维度在 BFS 循环中通过 b_idx 追踪。
              所有 batch 元素共享相同的初始边界（全图）。
        """
        if self._region_bounds_cache is not None:
            return self._region_bounds_cache

        H, W = self.image_size
        # 初始只有一个区域：整个图像
        # 格式: [x0, y0, x1, y1]
        initial_bounds = torch.tensor([0, 0, W, H], dtype=dtype, device=device)
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

        P-OPT: 使用预分配 tensor 替代 Python list，避免频繁内存分配

        Args:
            split_result: SplitResult 包含 split_decision [B, N]
            features: [B, C, H, W] 输入图像

        Returns:
            TensorSplitResult: 兼容 TensorSplitResult 格式的分割结果
        """
        from vit_pytorch.core.splitter_protocol import TensorSplitResult

        B, C, H, W = features.shape
        device = features.device
        split_decision = split_result.split_decision  # [B, N]

        # P-OPT: 预分配最大可能大小的 buffer
        # 最大节点数: 每个 batch 有 quadtree_node_count(max_level) 个节点 (四叉树)
        max_nodes_per_batch = quadtree_node_count(self.max_level)
        max_total_nodes = max_nodes_per_batch * B

        # 预分配 buffer
        regions_buffer = torch.zeros(max_total_nodes, 4, dtype=features.dtype, device=device)
        depths_buffer = torch.zeros(max_total_nodes, dtype=torch.long, device=device)
        batch_buffer = torch.zeros(max_total_nodes, dtype=torch.long, device=device)

        # P-OPT: BFS 使用数组 + 索引替代 Python deque，避免对象创建开销
        # 队列使用 CPU 列表存储 (queue_bounds, depth, b_idx)，避免 GPU-CPU 同步
        # bounds_queue 仅存储 bounds tensor 引用，不做同步操作
        bounds_queue: List[torch.Tensor] = []
        depth_queue: List[int] = []
        batch_queue: List[int] = []

        initial_bounds = self._get_initial_region_bounds(device, dtype=features.dtype)

        # 使用 Python list 追踪每个 batch 的区域索引
        batch_region_counters = [0] * B
        count = 0  # 实际使用的节点数

        # 初始化队列: B 个根节点
        for b in range(B):
            bounds_queue.append(initial_bounds)
            depth_queue.append(0)
            batch_queue.append(b)

        # BFS 构建四叉树 - 使用列表索引替代 deque.popleft()
        queue_start = 0
        while queue_start < len(bounds_queue):
            bounds = bounds_queue[queue_start]
            depth = depth_queue[queue_start]
            b_idx = batch_queue[queue_start]
            queue_start += 1

            # 获取当前 batch 的区域索引
            region_idx_in_batch = batch_region_counters[b_idx]
            batch_region_counters[b_idx] += 1

            if depth >= self.max_level:
                # 达到最大深度，添加到 buffer
                regions_buffer[count] = bounds
                depths_buffer[count] = depth
                batch_buffer[count] = b_idx
                count += 1
                continue

            # 检查是否应该分裂
            # I-OPT: 使用 GPU tensor 直接做条件判断，避免 .cpu() 强制同步
            if region_idx_in_batch < split_decision.shape[1]:
                should_split = split_decision[b_idx, region_idx_in_batch] > 0.5
            else:
                should_split = False

            if should_split and count + 4 < max_total_nodes:
                # 计算子边界
                x0, y0, x1, y1 = bounds.unbind()
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2

                # 直接在 GPU 上创建子边界张量
                child1 = torch.stack([x0, y0, cx, cy], dim=0)
                child2 = torch.stack([cx, y0, x1, cy], dim=0)
                child3 = torch.stack([x0, cy, cx, y1], dim=0)
                child4 = torch.stack([cx, cy, x1, y1], dim=0)

                # 入队 4 个子节点
                bounds_queue.append(child1)
                depth_queue.append(depth + 1)
                batch_queue.append(b_idx)

                bounds_queue.append(child2)
                depth_queue.append(depth + 1)
                batch_queue.append(b_idx)

                bounds_queue.append(child3)
                depth_queue.append(depth + 1)
                batch_queue.append(b_idx)

                bounds_queue.append(child4)
                depth_queue.append(depth + 1)
                batch_queue.append(b_idx)
            else:
                # 添加为叶子节点
                regions_buffer[count] = bounds
                depths_buffer[count] = depth
                batch_buffer[count] = b_idx
                count += 1

        # 裁剪到实际大小
        if count == 0:
            regions = torch.zeros(0, 4, dtype=torch.float32, device=device)
            depths = torch.zeros(0, dtype=torch.long, device=device)
            batch_indices = torch.zeros(0, dtype=torch.long, device=device)
            token_indices = torch.zeros(0, dtype=torch.long, device=device)
        else:
            regions = regions_buffer[:count]
            depths = depths_buffer[:count]
            batch_indices = batch_buffer[:count]
            # token_indices 是顺序索引 (0, 1, 2, ..., N-1)，用于正确的概率索引
            token_indices = torch.arange(count, dtype=torch.long, device=device)

        # 计算 Hilbert 索引用于排序 (I113-18: 使用 HilbertScanner)
        # P-OPT: 使用 self.max_level 替代 depths.max().item() 避免 GPU-CPU 同步
        # 性能影响: 排序精度略有下降，但避免前向传播中的 .item() 同步
        from vit_pytorch.core.curve_hilbert import HilbertScanner
        # 关键优化: 使用预计算的 max_level 而非每次都同步获取实际最大值
        hilbert_depth = self.max_level
        # Step 3: 传递 HilbertTopologyCache 进行 O(1) Tensor Lookup
        hilbert_indices = HilbertScanner.region_to_hilbert_index(
            regions[:, 0],  # x0
            regions[:, 1],  # y0
            regions[:, 2],  # x1
            regions[:, 3],  # y1
            hilbert_depth,  # P-OPT: 使用 max_level 避免 .item() 同步
            H,
            W,
            hilbert_cache=self._hilbert_cache,  # Step 3 优化
        )

        # 复杂度使用冗余性分数
        # D1-AUDIT FIX: 使用 regions.shape[0] 替代 len(regions) 避免 GPU-CPU 同步
        n_regions = regions.shape[0]
        complexities = split_result.redundancy.view(-1) if split_result.redundancy.numel() > 0 else \
                       torch.zeros(n_regions, dtype=torch.float32, device=device)

        return TensorSplitResult(
            regions=regions.long(),  # I131-1: 转换为 long 以匹配 padded_regions 的 dtype
            depths=depths,
            batch_indices=batch_indices,
            hilbert_indices=hilbert_indices,
            token_indices=token_indices,
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
        split_result: "SplitResult",
        features: Optional[torch.Tensor] = None,  # I107-7: 预计算特征，跳过 shared_conv
    ) -> TokenizerOutput:
        """Variable Depth tokenization (P9-1 方案 D: 完全向量化).

        I98-1 架构变更:
            - Splitter 从外部传入，不再内部创建
            - 职责分离: Tokenizer 只负责 embedding，Splitter 负责分割决策

        I107-7 优化:
            - 接受外部预计算的 features 跳过 shared_conv 双重计算
            - 验证 4 个运行时不变量: shape / device / dtype / contiguity
            - 默认 None 保持向后兼容（旧调用方零侵入）

        数学形式化:
            1. F = SharedConv(I)                    # 特征提取（features=None 路径）
               或 F = F̃                             # features ≠ None 路径
            2. T = _embed_with_tensor_result(F, TensorResult)  # 纯张量嵌入
               其中 TensorResult 来自外部 Splitter

        Args:
            images: [B, C, H, W] 输入图像
            split_result: 外部 Splitter 的输出结果
                包含: regions, depths, batch_indices, hilbert_indices,
                      selected_mask, logits, probs
            features: [B, d_model, H/p, W/p] 可选。预计算的特征图
                （通常来自 self._feature_extractor(images)）。传入时跳过
                self.shared_conv(images) 重复计算，实现 I107-7 优化。
                必须满足 4 个不变量:
                  I1 shape:     == (B, d_model, H/base_patch_size, W/base_patch_size)
                  I2 device:    == images.device
                  I3 dtype:     == images.dtype  (防 AMP 静默 upcast)
                  I4 contiguity: ∈ {default, channels_last}，其他 layout 自动 .contiguous() 归一化
                传 None 则由 tokenize 内部计算（向后兼容路径）。

        Returns:
            TokenizerOutput: 包含 tokens, levels_info, regions 等

        I35: 支持 channels_last 内存格式以优化卷积性能
        """
        from vit_pytorch.core.splitter_protocol import SplitResult, TensorSplitResult

        if images.dim() != 4:
            raise ValueError(
                f"StreamingFractalTokenizerV3.tokenize expects 4D input [B, C, H, W], "
                f"got {images.dim()}D tensor."
            )

        B, C, H, W = images.shape
        device = images.device

        # 1. 提取或复用共享特征图
        # I107-7: 若调用方传入预计算的 features，跳过 shared_conv 双重计算
        if features is None:
            # 旧路径（向后兼容）：转 channels_last → shared_conv
            # I35: 转换为 channels_last 以优化卷积性能
            # I108-2: 使用 is_contiguous() 正确检测内存格式，而非错误的 stride 比较
            if images.dim() == 4 and not images.is_contiguous(memory_format=torch.channels_last):
                images = images.to(memory_format=torch.channels_last)
            features = self.shared_conv(images)  # [B, d_model, H/p, W/p]
            # I107-2: 移除调试缓存，防止显存泄露
            # 调试可视化可通过回调钩子实现，不应在核心代码中持有引用
        else:
            # 新路径（I107-7）：审计 4 个运行时不变量
            # I1: 形状不变量
            p = self.base_patch_size
            expected_shape = (B, self.d_model, H // p, W // p)
            if tuple(features.shape) != expected_shape:
                raise ValueError(
                    f"StreamingFractalTokenizerV3.tokenize features shape 不匹配: "
                    f"got {tuple(features.shape)}, expected {expected_shape} "
                    f"(B={B}, d_model={self.d_model}, H/p={H // p}, W/p={W // p}). "
                    f"传 None 以自动计算，或确认 features 来自 self._feature_extractor(images)。"
                )
            # I2: 设备不变量（防跨设备 ROI-Align 静默失败 / CUDA assert）
            if features.device != device:
                raise ValueError(
                    f"StreamingFractalTokenizerV3.tokenize features device 不匹配: "
                    f"got {features.device}, expected {device}. "
                    f"请调用 features.to({device}) 后再传入。"
                )
            # I3: 数据类型不变量（防 AMP 静默 upcast，参 T10 keystone abaec33）
            if features.dtype != images.dtype:
                raise ValueError(
                    f"StreamingFractalTokenizerV3.tokenize features dtype 不匹配: "
                    f"got {features.dtype}, expected {images.dtype}. "
                    f"请检查 autocast 上下文或显式 features.to({images.dtype})。"
                )
            # I4: 连续性不变量（非阻塞，自动归一化）
            # ROI-Align 需要 contiguous_format；若 features 既非 default contig 也非
            # channels_last，强制 contiguous 避免下游静默失败
            if not (features.is_contiguous() or
                    features.is_contiguous(memory_format=torch.channels_last)):
                features = features.contiguous()

        # 2. I98-1: 使用外部传入的 split_result
        # H1SS 返回 SplitResult → TensorSplitResult
        if isinstance(split_result, TensorSplitResult):
            tensor_result = split_result
        elif split_result is not None:
            # H1SS 返回 SplitResult 类型
            if isinstance(split_result, SplitResult):
                tensor_result = TensorSplitResult.from_split_result(split_result)
            else:
                raise ValueError(f"Unexpected split result type: {type(split_result)}")

        # I99-1 FIX: 使用强制 clamp 版本处理 tensor_result
        # 始终确保 batch_indices 和 depths 在有效范围内，避免 CUDA assert
        tensor_result = self._force_clamp_tensor_result(tensor_result, B)

        # 统计收集 (no_grad)
        with torch.no_grad():
            # I24-14 修复: 使用无条件张量操作 (torch.compile 安全)
            # 原问题: 数据依赖的 if 语句在 torch.compile 下可能被跳过
            # 解决: 无条件计算 bincount，然后无条件 clamp
            
            # bincount 需要至少一个元素，使用 torch.where 处理空情况
            # I99-1 CRITICAL: 必须先 clamp 再 bincount!
            batch_indices_for_bincount = tensor_result.batch_indices.clamp(min=0, max=B - 1)
            if batch_indices_for_bincount.numel() == 0:
                # 极端边界情况：完全没有 token
                batch_indices_for_bincount = torch.zeros(1, dtype=torch.long, device=device)

            tokens_per_batch = torch.bincount(batch_indices_for_bincount, minlength=B)

            # I24-14: 无条件 clamp (torch.compile 安全)
            # 不使用 .item() 或数据依赖的 if，直接 clamp
            tokens_per_batch = tokens_per_batch.clamp(min=1)
            # P-OPT: 保持 GPU 计算，支持 torch.compile + cudagraphs
            max_tokens_per_batch = tokens_per_batch.max()  # 保持在 GPU

            # 计算 depth distribution
            # P-OPT-3: 使用向量化操作，避免 Python for 循环
            max_d = self.max_level + 1
            depths = tensor_result.depths
            batch_indices = tensor_result.batch_indices

            if tensor_result.num_tokens > 0:
                count_matrix = torch.zeros(B, max_d, dtype=torch.long, device=device)
                # I99-1: clamp batch_indices 到 [0, B-1] 防止 scatter_add_ 越界
                batch_indices_clamped = batch_indices.clamp(min=0, max=B - 1)
                # I78: 添加 min clamp 防止负索引 (M3 修复)
                flat_idx_base = batch_indices_clamped * max_d
                depths_clamped = depths.clamp(min=0, max=max_d - 1)
                flat_idx = flat_idx_base + depths_clamped
                # I99-1: 额外的 clamp 保护 scatter_add_ index
                flat_idx = flat_idx.clamp(min=0, max=B * max_d - 1)
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
        max_tokens_int = _safe_scalar_to_int(max_tokens_per_batch, "max_tokens_per_batch")  # P0-FIX: 提前转换

        # 3. 纯张量嵌入
        # I30-11: 传递 raw_probs 用于构建 padded_split_probs
        # I78-2: 传递 selected_mask 用于替换 threshold 机制
        raw_probs = split_result.probs
        selected_mask = split_result.selected_mask
        tokens, levels_info, padded_regions, padded_split_probs = self._embed_with_tensor_result(
            features, tensor_result, raw_probs, max_tokens=max_tokens_int,  # P0-FIX: 传入 Python int
            selected_mask=selected_mask
        )

        # P9-5/P12-2 优化: 传入已 padding 的张量缓存，避免 model 中重复 padding
        # I20: 简化输出构建
        # I24-14: 使用 tokens_per_batch 作为 lengths_tensor (已在 GPU 上)
        lengths_tensor = tokens_per_batch.long()

        # P-OPT: 使用延迟构建，不传递 sequences
        # sequences 会在首次访问时通过 _build_sequences_from_cache() 延迟构建
        # 注意：原代码中本地构建 sequences 但未使用的死代码已移除，避免 .tolist() 同步
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
        
        # D1-AUDIT FIX: 使用 torch.as_tensor 避免不必要的数据拷贝
        # torch.tensor 会创建新拷贝，torch.as_tensor 尽可能复用已有内存
        boxes_tensor = torch.as_tensor(all_boxes, device=device, dtype=dtype)

        # P-OPT: 直接替换 NaN/Inf，移除 .any() 同步检查
        # 批量操作保持 GPU 利用率，避免 GPU-CPU 同步
        nan_mask = torch.isnan(boxes_tensor)
        inf_mask = torch.isinf(boxes_tensor)
        # 使用 fused 操作避免显式 .any() 调用 - 始终应用替换
        boxes_tensor = torch.where(nan_mask | inf_mask, torch.zeros_like(boxes_tensor), boxes_tensor)

        # ROI-Align
        try:
            from torchvision.ops import roi_align
            # I99-1: ROIAlign 需要 channels_first 格式
            if features.dim() == 4 and features.is_contiguous(memory_format=torch.channels_last):
                features_roi = features.to(memory_format=torch.contiguous_format)
            else:
                features_roi = features
            pooled = roi_align(
                features_roi,
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
        
        # 深度编码已移至 DirectionAwareSubspacedRoPE（注意力层）
        # tokenizer 仅输出 patch embedding，由 C+ RoPE 处理深度/位置编码
        all_tokens = pooled
        
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

        count_matrix.shape[1]
        
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
        max_tokens: int = 1,  # P0-FIX: 现在确保传入 Python int
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

        B = features.shape[0]
        B_int = _safe_scalar_to_int(B, "B")
        device = features.device
        dtype = features.dtype
        dim = self.d_model

        N_total = tensor_result.num_tokens

        # I99-1 CRITICAL: N_total 可能是 tensor，使用 _safe_scalar_to_int 确保正确比较
        N_total_int = _safe_scalar_to_int(N_total, "N_total")

        if N_total_int == 0:
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

        # P-OPT: 直接替换 NaN/Inf，移除 .any() 同步检查
        # 批量操作保持 GPU 利用率，避免 GPU-CPU 同步
        nan_mask = torch.isnan(regions)
        inf_mask = torch.isinf(regions)
        # 使用 fused 操作避免显式 .any() 调用 - 始终应用替换
        regions = torch.where(nan_mask | inf_mask, torch.zeros_like(regions), regions)

        # I99-1: clamp regions 到有效图像边界
        # P-OPT: 直接在原始张量上 clamp_()，避免不必要的 .clone() 内存分配
        # 修复: 分别使用宽度和高度进行clamp，支持非方形图像
        if isinstance(self.image_size, tuple):
            img_w, img_h = self.image_size
        else:
            img_w = img_h = self.image_size
        # D3-AUDIT FIX: 使用非 in-place clamp 替代 clamp_()，避免在视图上操作破坏梯度跟踪
        # 原实现: regions[:, 0].clamp_(...) — in-place 操作在视图上可能破坏梯度
        regions = regions.clone()  # 先 clone 避免修改原始视图
        regions[:, 0] = regions[:, 0].clamp(min=0, max=img_w)  # x1: 使用宽度
        regions[:, 1] = regions[:, 1].clamp(min=0, max=img_h)  # y1: 使用高度
        regions[:, 2] = regions[:, 2].clamp(min=0, max=img_w)  # x2: 使用宽度
        regions[:, 3] = regions[:, 3].clamp(min=0, max=img_h)  # y2: 使用高度

        # I99-1 CRITICAL: 验证 batch_indices 值范围（在 clamp 之前）
        # P-OPT: 使用向量化布尔运算，避免 GPU-CPU 同步
        # torch.all/any 保持张量在 GPU 上，不触发 .item() 同步
        batch_indices_raw = tensor_result.batch_indices
        if batch_indices_raw.numel() > 0:
            # P-OPT: 仅在调试模式验证 batch_indices 范围
            # clamp 已经提供了安全保证，避免每次 forward 都触发 GPU-CPU 同步
            if getattr(self, '_debug_mode', False):
                valid_min = (batch_indices_raw >= 0).all()
                valid_max = (batch_indices_raw < B_int).all()
                if not (valid_min and valid_max):
                    batch_min = int(batch_indices_raw.min().item()) if batch_indices_raw.numel() > 0 else "N/A"
                    batch_max = int(batch_indices_raw.max().item()) if batch_indices_raw.numel() > 0 else "N/A"
                    raise RuntimeError(
                        f"I99-1 CRITICAL: batch_indices 包含无效值! "
                        f"范围=[{batch_min}, {batch_max}], B={B_int}, N_total={N_total}"
                    )

        # I99-1: 防御性 clamp batch_indices (额外保护)
        batch_indices = tensor_result.batch_indices.clamp(min=0, max=B_int - 1)
        
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
        # 特征提取: 优先使用 Splitter 预计算的 roi_features_raw（避免重复 ROI-Align）
        # ====================================================================
        has_fast_path = (
            hasattr(tensor_result, 'roi_features_raw')
            and tensor_result.roi_features_raw is not None
            and hasattr(tensor_result, 'mask_ste')
            and tensor_result.mask_ste is not None
            and hasattr(tensor_result, 'candidate_indices')
            and tensor_result.candidate_indices is not None
        )

        if has_fast_path:
            # 快速路径: tokens = roi_features_raw * mask_ste 权重乘法
            roi_raw = tensor_result.roi_features_raw  # [B, N_candidates, d_model]
            mask_ste = tensor_result.mask_ste  # [B, N_candidates]
            candidate_indices = tensor_result.candidate_indices  # [M] 选中 token 在候选池中的索引
            batch_idx_for_token = batch_indices.clamp(min=0, max=B_int - 1)  # [M]

            # 应用 STE 权重: tokens_all[b, n, :] * mask_ste[b, n]
            tokens_all = roi_raw * mask_ste.unsqueeze(-1)  # [B, N, d_model]

            # 从 tokens_all 中按 (batch, candidate_idx) 提取选中的 token 特征
            pooled = tokens_all[batch_idx_for_token, candidate_indices, :]  # [M, d_model]
        else:
            # 回退路径: 传统 ROI-Align
            from torchvision.ops import roi_align
            if features.dim() == 4 and features.is_contiguous(memory_format=torch.channels_last):
                features_roi = features.to(memory_format=torch.contiguous_format)
            else:
                features_roi = features

            nan_mask = torch.isnan(boxes)
            inf_mask = torch.isinf(boxes)
            combined_mask = nan_mask | inf_mask
            boxes = torch.where(combined_mask, torch.zeros_like(boxes), boxes)

            if boxes.device != features_roi.device:
                boxes = boxes.to(features_roi.device, non_blocking=True)

            pooled = roi_align(
                features_roi, boxes,
                output_size=(1, 1), spatial_scale=1.0, aligned=True,
            ).squeeze(-1).squeeze(-1)  # [N_total, C]
        
        # ====================================================================
        # 深度编码 (向量化)
        # ====================================================================
        # I34-7 Fix: Add min=0 boundary protection to prevent negative depth index errors
        depths = tensor_result.depths.clamp(min=0, max=self.max_level)
        # 深度编码已移至 DirectionAwareSubspacedRoPE（注意力层）
        # tokenizer 仅输出 patch embedding，由 C+ RoPE 处理深度/位置编码
        all_tokens = pooled

        # ====================================================================
        # 向量化分配到输出 buffer
        # ====================================================================
        # I99-1: 防御性检查 - 确保 max_tokens 至少为 1 且足够容纳所有 token
        # I99-1 FIX: max_tokens 是 scalar tensor，需要提取 Python int
        # 计算每个 batch 需要的最小 token 数（向上取整）
        max(1, (N_total + B_int - 1) // B_int)  # 向上取整确保足够

        # P0-FIX: max_tokens 已经是 Python int，无需类型检查
        # 移除 isinstance 检查以支持 torch.compile + CUDA Graphs
        max_tokens_int = max(1, int(max_tokens))

        tokens = torch.zeros(B, max_tokens_int, dim, device=device, dtype=dtype)
        # I32-2: 使用-1 sentinel标识padding token，避免与有效depth=0混淆
        levels_info = torch.full((B, max_tokens_int, self.max_level + 1), -1, dtype=torch.long, device=device)
        padded_regions = torch.zeros(B, max_tokens_int, 4, dtype=torch.long, device=device)  # P11-3

        # I99-1: 修复 batch 独立性 - 确保按 (batch_idx, hilbert_idx) 排序
        # 问题: batch_indices 可能未按 batch 分组，导致 token 位置计算错误
        # 解决: 显式排序所有相关张量
        if N_total > 0:
            # I99-1 CRITICAL: 验证所有张量长度一致性
            expected_len = N_total
            tensors_to_check = [
                ("hilbert_indices", tensor_result.hilbert_indices),
                ("batch_indices", tensor_result.batch_indices),
                ("depths", tensor_result.depths),
                ("regions", tensor_result.regions),
            ]
            for name, tensor in tensors_to_check:
                if tensor.shape[0] != expected_len:
                    raise RuntimeError(
                        f"I99-1 CRITICAL: {name}.shape[0]={tensor.shape[0]} != N_total={expected_len}"
                    )

            # 获取 hilbert_indices 用于排序
            hilbert_idx_for_sort = tensor_result.hilbert_indices

            # I99-1 FIX: torch.lexsort 可能不可用，使用 torch.argsort 替代
            # lexsort 的语义是: 先按最后一列排序，再按倒数第二列排序...
            # 所以我们先按 hilbert_indices 排序 (稳定)，再按 batch_indices 排序 (稳定)
            hilbert_order = torch.argsort(hilbert_idx_for_sort, stable=True)

            # I99-1 CRITICAL: 验证 hilbert_order 和 batch_indices 长度一致
            if hilbert_order.shape[0] != batch_indices.shape[0]:
                raise RuntimeError(
                    f"I99-1 CRITICAL: hilbert_order.shape[0]={hilbert_order.shape[0]} != "
                    f"batch_indices.shape[0]={batch_indices.shape[0]}"
                )

            batch_order = torch.argsort(batch_indices[hilbert_order], stable=True)

            # I99-1 CRITICAL: 验证 batch_order 和 hilbert_order 长度一致
            if batch_order.shape[0] != hilbert_order.shape[0]:
                raise RuntimeError(
                    f"I99-1 CRITICAL: batch_order.shape[0]={batch_order.shape[0]} != "
                    f"hilbert_order.shape[0]={hilbert_order.shape[0]}"
                )

            sort_indices = hilbert_order[batch_order]

            # 重新排列所有张量
            batch_indices_sorted = batch_indices[sort_indices]
            all_tokens_sorted = all_tokens[sort_indices]
            depths_sorted = depths[sort_indices]
            regions_sorted = tensor_result.regions[sort_indices]

            # I113-18 修复: 在排序后立即计算 token_positions
            # 排序后 batch_indices_sorted 分组为 [0,0,...,0,1,1,...,1,...]
            # 每个 token 在其 batch 内的位置用 bincount+cumsum 计算
            if N_total_int > 0:
                batch_counts = torch.bincount(batch_indices_sorted, minlength=B_int)
                batch_offsets = torch.cat([
                    torch.zeros(1, device=device, dtype=torch.long),
                    batch_counts.cumsum(0)[:-1]
                ])
                token_positions_sorted = torch.arange(N_total_int, device=device, dtype=torch.long)
                token_positions = token_positions_sorted - batch_offsets[batch_indices_sorted]
            else:
                token_positions = torch.zeros(0, dtype=torch.long, device=device)

            # 如果有 raw_probs，使用 token_positions 进行正确的概率索引
            if raw_probs is not None:
                raw_probs_sorted = raw_probs[batch_indices_sorted, token_positions]
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
        # I99-1 CRITICAL: N_total 可能是 tensor，需要先转换为 Python int
        # 然后统一使用 Python int 进行比较，避免 torch.compile 导致的类型问题
        N_total_int = _safe_scalar_to_int(N_total, "N_total")
        N_total_is_zero = (N_total_int == 0)

        # I99-1 CRITICAL: 防御性断言 - 验证 N_total_int 在合理范围内
        # 这可以在最早的阶段捕获异常值
        # 改回 if-raise 模式，确保不会被 Python -O 标志跳过
        if not (0 <= N_total_int <= B_int * max_tokens_int * 10):
            raise RuntimeError(
                f"N_total_int out of reasonable range: {N_total_int}, B={B_int}, max_tokens={max_tokens_int}"
            )

        # 排序后: batch_indices 严格按 batch 分组 [0,0,...,0, 1,1,...,1, ...]
        if N_total_is_zero:
            token_positions = torch.zeros(0, dtype=torch.long, device=device)
            batch_indices_safe = torch.zeros(0, dtype=torch.long, device=device)
        else:
            # I99-1 CRITICAL: 使用 torch.sort 方法确保精确的 batch 分组
            # 问题: 原整除法 + clamp 假设 token 在 batch 间均匀分布
            # 解决: 使用排序方法，精确计算每个 token 的位置
            batch_indices_sorted, sort_order = torch.sort(batch_indices)
            token_positions_sorted = torch.arange(N_total_int, device=device, dtype=torch.long)

            # 通过排序映射恢复原始顺序
            batch_indices_safe = batch_indices_sorted
            token_positions = torch.zeros_like(token_positions_sorted)
            token_positions[sort_order] = token_positions_sorted

            # I99-1 OPT: 向量化验证，避免每个 batch 单独同步
            # P-OPT: 仅在调试模式进行完整的验证检查，避免 GPU-CPU 同步
            if getattr(self, '_debug_mode', False):
                batch_indices_valid = (batch_indices_safe >= 0).all() & (batch_indices_safe < B_int).all()
                token_indices_valid = (token_positions >= 0).all()
                if not (batch_indices_valid and token_indices_valid):
                    batch_min = int(batch_indices_safe.min().item()) if batch_indices_valid else "N/A"
                    batch_max = int(batch_indices_safe.max().item()) if batch_indices_valid else "N/A"
                    token_min = int(token_positions.min().item()) if token_indices_valid else "N/A"
                    token_max = int(token_positions.max().item()) if token_indices_valid else "N/A"
                    raise RuntimeError(
                        f"I99-1 CRITICAL: batch_indices 或 token_positions 无效! "
                        f"batch范围=[{batch_min}, {batch_max}], token范围=[{token_min}, {token_max}], "
                        f"B={B_int}, N_total={N_total_int}"
                    )

            # I99-1: 防御性 clamp - 确保 token_positions 在安全范围内
            # 这是一个额外的保护层，防止 splitter 异常
            token_positions = token_positions.clamp(min=0, max=max_tokens_int - 1)

        # I99-1 OPT: 批量验证，避免每个样本单独检查
        # P-OPT: 仅在调试模式进行完整的验证检查，避免 GPU-CPU 同步
        if getattr(self, '_debug_mode', False):
            if N_total_int > 0 and B_int > 0 and max_tokens_int > 0:
                batch_ok = (batch_indices_safe >= 0).all() & (batch_indices_safe < B_int).all()
                token_ok = (token_positions >= 0).all() & (token_positions < max_tokens_int).all()
                if not (batch_ok and token_ok):
                    batch_min = int(batch_indices_safe.min().item()) if batch_ok else "N/A"
                    batch_max = int(batch_indices_safe.max().item()) if batch_ok else "N/A"
                    token_min = int(token_positions.min().item()) if token_ok else "N/A"
                    token_max = int(token_positions.max().item()) if token_ok else "N/A"
                    raise RuntimeError(
                        f"I99-1 CRITICAL: batch_indices 或 token_positions 越界! "
                        f"batch范围=[{batch_min}, {batch_max}], token范围=[{token_min}, {token_max}], "
                        f"B_int={B_int}, max_tokens_int={max_tokens_int}, N_total_int={N_total_int}"
                    )

        # 验证 tokens tensor 形状
        expected_tokens_shape = (B, max_tokens_int, dim)
        if tokens.shape != expected_tokens_shape:
            raise RuntimeError(
                f"I99-1 CRITICAL: tokens 形状错误! "
                f"expected={expected_tokens_shape}, actual={tokens.shape}"
            )

        # 验证 all_tokens 形状
        if all_tokens.shape[0] != N_total_int:
            raise RuntimeError(
                f"I99-1 CRITICAL: all_tokens 形状错误! "
                f"expected N_total_int={N_total_int}, actual={all_tokens.shape[0]}"
            )

        # I99-1 CRITICAL: 最终安全检查 - 验证 N_total 是否超过容量
        # 这是最后一道防线
        if N_total_int > B_int * max_tokens_int:
            raise RuntimeError(
                f"I99-1 CRITICAL: N_total({N_total_int}) > B_int({B_int}) * max_tokens_int({max_tokens_int})!"
            )

        # 向量化分配 (使用 clamp 后的 batch_indices_safe)
        tokens[batch_indices_safe, token_positions] = all_tokens.to(dtype)

        # =====================================================================
        # I12-3 修复: 从 regions 计算四叉树路径填充 levels_info
        # 原问题: levels_info[:, :, 1:] 全为零，导致所有 token 共享相同的路径编码
        # 修复方案: 使用 VectorizedPathEncoder.compute_paths_from_regions 计算正确路径
        # =====================================================================
        levels_info[batch_indices_safe, token_positions, 0] = depths

        # 计算四叉树路径 (基于区域中心的空间位置)
        # regions: [N_total, 4] -> paths: [N_total, max_level]
        if N_total_int > 0:
            # I99-1 FIX: 防御性处理 image_size 为 None 或 0 的情况
            if self.image_size is None:
                # 动态分辨率模式：从特征图获取尺寸
                img_size = max(features.shape[-2], features.shape[-1])
            elif isinstance(self.image_size, tuple):
                img_size = max(self.image_size)
            else:
                img_size = self.image_size

            # I99-1: 防御性检查 - 确保 img_size 有效 (>= 1)
            if img_size is None or img_size <= 0:
                img_size = 64  # 安全默认值

            paths = VectorizedPathEncoder.compute_paths_from_regions(
                tensor_result_regions_sorted,  # [N_total, 4] - 使用排序后的 regions
                img_size,
                self.max_level
            )  # [N_total, max_level]

            # 填充路径到 levels_info
            # levels_info 格式: [depth, path[0], path[1], ..., path[max_level-1]]
            levels_info[batch_indices_safe, token_positions, 1:] = paths

        # P11-3: 分配 regions 到 padded buffer
        padded_regions[batch_indices_safe, token_positions] = tensor_result_regions_sorted

        # I30-11: 构建 padded_split_probs [B, max_tokens]
        # I99-1: 使用排序后的 raw_probs
        padded_split_probs = None
        if raw_probs is not None and N_total > 0 and max_tokens_int > 0:
            split_probs_padded = torch.zeros(B, max_tokens_int, dtype=raw_probs.dtype, device=device)

            # 向量化分配: split_probs_padded[batch_idx, token_pos] = raw_probs_sorted[...]
            # I99-1: 使用 batch_indices_safe 确保索引安全
            split_probs_padded[batch_indices_safe, token_positions] = raw_probs
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
        # I99-1 OPT: 使用 PROB_EPSILON 替代硬编码 1e-10
        probs_safe = probs + (probs == 0).float() * PROB_EPSILON
        entropy_per_batch = -(probs_safe * probs_safe.log()).sum(dim=1)
        return float(entropy_per_batch.mean().item())

    @torch.no_grad()
    def compute_scale_distribution(
        self,
        images: torch.Tensor,
        split_result: Optional["SplitResult"] = None,
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
        # I98-1: 如果未提供 split_result，尝试从外部获取
        if split_result is None:
            # 尝试从 model.splitter 获取 (使用 weakref 避免循环引用)
            if hasattr(self, '_model') and callable(self._model):
                model_ref = self._model()
                if model_ref is not None and hasattr(model_ref, 'splitter'):
                    model = model_ref
                    # 获取特征图
                    features = self.shared_conv(images)  # [B, C, H, W]

                    # H1SS 使用 4D 输入 [B, C, H, W]
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


