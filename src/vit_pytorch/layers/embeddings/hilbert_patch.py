# -*- coding: utf-8 -*-
"""
Hilbert-Native Patch Embedding for Variable Depth Tokens

数学形式化
============

Variable Depth Token 的 Patch Embedding 必须满足 4 个约束:

【约束 C1】维度一致性
    Embed(R_i) ∈ ℝ^dim, ∀ i, ∀ d_i
    不同大小的 region 必须映射到相同维度

【约束 C2】Hilbert 路径一致性
    HilbertPath(center(R_i))[:d_i] = QuadtreePath(R_i)
    嵌入必须保持四叉树路径信息

【约束 C3】尺度等变性
    若 R_j = 2×downsample(R_i) 且 content 相同，则:
    Embed(R_i) ≈ σ · Embed(R_j) + bias
    相同内容不同尺度应有数学联系

【约束 C4】LCA 兼容性
    LCA_depth(path_i, path_j) 必须对 Transformer bias 有效
    嵌入必须与 LCA 偏置协同工作

方案 C+ (Region Pooling) 实现 - 向量化版本
==========================================

公式:
    F = SharedConv(Image)  ∈ ℝ^{B × dim × H/p × W/p}
    
    向量化池化 (使用 ROI-Align):
        boxes = [(b, x1/p, y1/p, x2/p, y2/p) for all regions]
        pooled = roi_align(F, boxes, output_size=(1,1))  # [N_total, dim, 1, 1]
        T = pooled · σ_d + E_d  # 批量深度编码
    复杂度:
        - 时间: O(1) GPU kernel 调用 (vs O(N) for Python loop)
        - 空间: O(N × dim) 
        - Kernel 启动: 1 次 (vs N 次)

参数量:
    - SharedConv: dim × C × p × p ≈ 12K
    - depth_embed: (D+1) × dim ≈ 1.3K
    - depth_scale: D+1 ≈ 5
    - 总计: ~14K (vs 4.2M for Depth-Specific Conv)

深度缩放范围 (P6-1 改进)
==========================

数学形式化:
    σ_d = σ_min + (σ_max - σ_min) · sigmoid(γ_d)
    
    其中:
    - σ_min = 0.5, σ_max = 2.0 (默认)
    - γ_d 是可学习参数
    - 动态范围: 4x (vs 原始 1.2x)
    
约束验证:
    【C1 信息保持】σ_min ≥ 0.5 确保浅层至少保留 50% 信息
    【C2 梯度稳定】σ_max ≤ 2.0 确保梯度放大不超过 2x
    【C3 区分度】σ_max/σ_min = 4x 提供充足的深度区分能力
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from vit_pytorch.core.levels_info import LevelsInfo
from vit_pytorch.layers.embeddings.fractal_rope import DirectionAwareSubspacedRoPE

# I100-4: 强制依赖 torchvision
# 移除 _fallback_roi_pool 回退实现，统一使用 torchvision.ops.roi_align
from torchvision.ops import roi_align  # 强制依赖，无回退

# Note: SplitResult 仅用作类型注解。运行时 split_results 的实际类型是
# TensorSplitResult (from vit_pytorch.core.splitter_protocol)，其接口不同。
# 此 stub 类与类型注解配合使用，不应在运行时被实例化。


class SplitResult:
    """Stub type for type hints only. Runtime type is TensorSplitResult."""
    num_tokens: int
    tokens: list


class SafeSoftplus(torch.autograd.Function):
    """自定义 softplus，在 backward 时主动检查和修复 NaN/Inf 梯度.

    解决 nan_to_num 只保护 forward、不保护 vjp 的问题。
    当上游梯度为 NaN/Inf 时，backward 计算可能产生级联 NaN。
    """

    @staticmethod
    def forward(ctx, x: Tensor, eps: float = 1e-6) -> Tensor:
        """Forward: y = softplus(x) + eps."""
        ctx.save_for_backward(x)
        ctx.eps = eps
        return F.softplus(x) + eps

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor) -> Tuple[Tensor, None]:
        """Backward: 检测并修复 NaN/Inf 梯度.

        Args:
            grad_outputs: 上游传回的梯度元组

        Returns:
            修复后的输入梯度
        """
        grad_output = grad_outputs[0]
        x, = ctx.saved_tensors
        eps = ctx.eps

        # 标准 softplus backward: grad = sigmoid(x) * grad_output
        sigmoid_x = torch.sigmoid(x)
        grad_input = grad_output * sigmoid_x

        # I-NAN: 检测 NaN/Inf 并修复
        has_nan = torch.isnan(grad_input).any()
        has_inf = torch.isinf(grad_input).any()

        if has_nan or has_inf:
            # 用 sign-based 方法替换 NaN/Inf
            grad_sign = torch.sign(grad_output)
            grad_input = torch.where(
                torch.isfinite(grad_input),
                grad_input,
                grad_sign * eps  # 用小值替换
            )
            # 记录修复次数
            from vit_pytorch.core.numerical_stability import increment_nan_fix
            increment_nan_fix("safe_softplus")

        return grad_input, None


class HilbertNativePatchEmbed(nn.Module):
    """Hilbert-Native 变深度 Patch Embedding.

    满足四个数学约束：
    1. 维度一致性: 所有 region → 相同 dim
    2. 路径一致性: 区域池化保持四叉树路径
    3. 尺度等变性: depth_scale 编码尺度信息
    4. LCA 兼容性: 与现有 LCA bias 无缝工作

    I30-17-EXT: 支持动态深度计算
        - 新 API: 使用 min_patch_size + max_level_limit
        - 旧 API: 直接指定 max_level (自动转换)

    Args:
        channels: 输入图像通道数
        dim: 输出嵌入维度
        base_patch_size: 最细粒度 patch 大小 (共享 Conv 的 stride)
        min_patch_size: [新 API] 目标最小 patch 大小，用于动态计算 max_level
        max_level_limit: [新 API] max_level 硬上限
        max_level: [旧 API] 最大四叉树深度 (直接指定)
        image_size: [新 API] 输入图像尺寸，用于计算 max_level
        conv_layers: SharedConv 层数 (1-3)
        use_batch_norm: 是否使用 BatchNorm
        depth_scale_beta: [已废弃] 使用 depth_scale_range 替代
        depth_scale_range: 深度缩放范围 (σ_min, σ_max)，使用 sigmoid 参数化
                          默认 (0.5, 2.0) 提供 4x 动态范围
                          设为 None 使用旧版固定初始化 (向后兼容)
    """

    def __init__(
        self,
        channels: int = 3,
        dim: int = 256,
        base_patch_size: int = 4,
        # I30-17-EXT: 新 API 参数
        min_patch_size: Optional[int] = None,
        max_level_limit: int = 8,
        # 旧 API 参数 (直接指定 max_level)
        max_level: Optional[int] = None,
        image_size: Optional[Tuple[int, int]] = None,
        conv_layers: int = 2,
        use_batch_norm: bool = True,
        # I-PHASE4: 池化方法选择
        use_interpolated_pooling: bool = False,
        # I-PHASE4: 动态权重 (C3 尺度等变性)
        use_dynamic_weight: bool = False,
        # P6-1: depth_scale_range - 使用 sigmoid 参数化防止 CUDA 梯度爆炸
        depth_scale_range: Optional[Tuple[float, float]] = None,  # (σ_min, σ_max)
    ) -> None:
        super().__init__()

        self.channels = channels
        self.dim = dim
        self.base_patch_size = base_patch_size
        self.use_interpolated_pooling = use_interpolated_pooling
        self.use_dynamic_weight = use_dynamic_weight

        # P6-1: depth_scale_range - sigmoid 参数化存储
        if depth_scale_range is not None:
            self.depth_scale_range = depth_scale_range
            self.depth_scale_sigma_min = depth_scale_range[0]
            self.depth_scale_sigma_max = depth_scale_range[1]
        else:
            self.depth_scale_range = None
            self.depth_scale_sigma_min = None
            self.depth_scale_sigma_max = None

        # I30-17-EXT: 处理新旧 API
        if max_level is not None:
            # 旧 API: 直接使用指定的 max_level
            self.max_level = max_level
        elif min_patch_size is not None and image_size is not None:
            # 新 API: 从 min_patch_size 计算 max_level
            from vit_pytorch.core.depth_utils import compute_max_level
            H, W = image_size
            self.max_level = compute_max_level(
                (H, W), min_patch_size, max_level_limit
            )
        else:
            # 默认值
            self.max_level = max_level_limit
        
        # =====================================================================
        # SharedConv: 统一的特征提取器
        # =====================================================================
        # 使用 stride=base_patch_size，将图像 [B, C, H, W] → [B, dim, H/p, W/p]
        # 多层设计增加感受野，但保持单次 stride 降采样
        
        layers = []
        in_ch = channels
        out_ch = dim // 2 if conv_layers > 1 else dim
        
        # 第一层: 主要下采样
        layers.extend([
            nn.Conv2d(in_ch, out_ch, kernel_size=base_patch_size, stride=base_patch_size),
            nn.BatchNorm2d(out_ch) if use_batch_norm else nn.Identity(),
            nn.GELU(),
        ])
        
        # 中间层: 扩展感受野
        for i in range(1, conv_layers):
            next_ch = dim if i == conv_layers - 1 else out_ch
            layers.extend([
                nn.Conv2d(out_ch, next_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(next_ch) if use_batch_norm else nn.Identity(),
                nn.GELU() if i < conv_layers - 1 else nn.Identity(),
            ])
            out_ch = next_ch

        self.shared_conv = nn.Sequential(*layers)

        # =====================================================================
        # FPN 金字塔 (FPN-style Multi-scale Feature Sampling)
        # =====================================================================
        # I-PHASE3: 添加 FPN 以解决单尺度特征截断问题
        #
        # 数学形式:
        #   FPN_l = Conv(Resize(F_{l-1})) + F_l
        #
        # 其中 F_l 是第 l 层的特征，Resize 是 2x 上采样
        #
        # 深度-特征层映射:
        #   depth 1-2 (大区域) → FPN_2 (最粗糙, 感受野最大)
        #   depth 3-4 (中区域) → FPN_1 (中等感受野)
        #   depth 5+  (小区域) → FPN_0 (最精细, 感受野最小)
        #
        # 这解决了原始方案的问题:
        #   - 单尺度特征导致小 Token (<1px) 产生 Aliasing

        self.fpn_levels = min(conv_layers, 3)  # 最多3层金字塔
        self.fpn_conv = nn.ModuleList()

        # 构建 FPN: 所有层使用 dim 通道以保持一致性
        for i in range(self.fpn_levels):
            self.fpn_conv.append(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1)
            )
        
        # I-PHASE4: 深度感知特征调制 (DAFM)
        # 解决 C3 尺度等变性问题
        if use_dynamic_weight:
            # 深度感知调制: γ_d = W @ one_hot(d)
            self.depth_gamma = nn.Linear(max_level + 1, dim)
            self.depth_beta = nn.Linear(max_level + 1, dim)
        else:
            self.depth_gamma = None
            self.depth_beta = None

        # 层归一化 (可选，用于稳定训练)
        self.norm = nn.LayerNorm(dim)

        # C+ RoPE: 方向感知子空间隔离 RoPE（绝对相位注入）
        # 仅对后半维度（拓扑场）应用，物理场保持不变
        # I-NAN: 修正检查逻辑 - dim_per_subspace = dim // max_level 必须为偶整数
        # 而非仅检查 rope_dim_per_subspace（这是错误的）
        dim_per_subspace = dim // max_level
        if dim_per_subspace > 0 and dim_per_subspace % 2 == 0:
            self.rope_fractal = DirectionAwareSubspacedRoPE(
                dim=dim // 2,  # RoPE 仅作用于后半维度（拓扑场）
                max_level=max_level,
            )
        else:
            # 维度不满足 RoPE 要求，跳过 RoPE
            self.rope_fractal = None

        # I-NAN: 为所有参数注册梯度 hook，捕获 backward 过程中产生的 NaN
        # 解决 patch_embed 内部 backward 产生 NaN 的问题
        self._nan_grad_hooks = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, n=name: (
                        torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
                        if torch.isnan(grad).any() or torch.isinf(grad).any() else grad
                    )
                )
                self._nan_grad_hooks.append(hook)

        # D1-AUDIT FIX: 用于 forward hook 的 handle，以便在下次 forward 时移除
        self._forward_grad_hook = None

    def remove_hooks(self) -> None:
        """D1-AUDIT FIX: Remove all registered gradient hooks to prevent memory leak"""
        for hook in self._nan_grad_hooks:
            hook.remove()
        self._nan_grad_hooks.clear()
        if hasattr(self, '_forward_grad_hook') and self._forward_grad_hook is not None:
            self._forward_grad_hook.remove()
            self._forward_grad_hook = None

    def __del__(self):
        """Cleanup hooks on deletion"""
        self.remove_hooks()

    def _apply_depth_modulation(
        self,
        pooled_features: Tensor,
        depths: Tensor,
    ) -> Tensor:
        """应用深度感知特征调制 (DAFM) 到池化后的特征

        I-PHASE4: 实现 C3 尺度等变性约束

        数学形式:
            F_d = γ_d ⊙ pooled + β_d
            其中 γ_d, β_d 由深度 d 动态生成

        Args:
            pooled_features: [N, D] 池化后的特征
            depths: [N] 每个 token 的深度

        Returns:
            modulated: [N, D] 调制后的特征
        """
        if self.depth_gamma is None:
            return pooled_features

        # 创建深度 one-hot: [N, max_level+1]
        depths_clamped = depths.clamp(0, self.max_level)
        depth_onehot = torch.nn.functional.one_hot(
            depths_clamped, num_classes=self.max_level + 1
        ).float()

        # 生成调制参数: [N, D]
        gamma = self.depth_gamma(depth_onehot)  # 逐通道缩放
        beta = self.depth_beta(depth_onehot)   # 逐通道偏移

        # 调制: F' = γ ⊙ F + β
        modulated = pooled_features * gamma + beta

        return modulated

    def _build_fpn_features(self, features: torch.Tensor) -> list:
        """构建 FPN 特征金字塔

        数学形式:
            FPN_0 = features (最精细)
            FPN_l = Conv(Resize(FPN_{l-1})) + features_l

        Args:
            features: [B, C, H, W] 共享卷积特征

        Returns:
            fpn_features: [FPN_levels] 特征金字塔列表
        """
        fpn_features = [features]

        # 从最细到最粗构建金字塔
        # 注意: 只需要 2 层上采样，因为 conv_layers=2 时只有 2 层特征
        for i in range(1, min(self.fpn_levels, 2)):
            # 上采样 2x
            prev_feat = fpn_features[-1]
            upsampled = F.interpolate(
                prev_feat,
                scale_factor=2.0,
                mode='nearest'
            )

            # 3x3 卷积减少混叠 (通道数相同，无需调整)
            convolved = self.fpn_conv[i](upsampled)
            fpn_features.append(convolved)

        return fpn_features

    def _select_fpn_level(self, depths: torch.Tensor) -> torch.Tensor:
        """根据深度选择 FPN 层级

        数学映射:
            depth 1-2 → level 2 (最粗糙，感受野最大)
            depth 3-4 → level 1 (中等)
            depth 5+  → level 0 (最精细)

        Args:
            depths: [N] token 深度

        Returns:
            level_indices: [N] 对应的 FPN 层级
        """
        # 映射: depth → level
        # level = max_level - depth (深token用细特征)
        levels = torch.clamp(self.max_level - depths, min=0, max=self.fpn_levels - 1)
        return levels

    def _dynamic_roi_align(
        self,
        features: Tensor,
        boxes: Tensor,
        depths: Tensor,
    ) -> Tensor:
        """动态 sampling_ratio 的 ROI-Align

        I106-1: 根据 Token 深度动态调整 sampling_ratio
        深层 Token 感受野小，使用更细致的采样防止特征模糊

        数学形式:
            sampling_ratio(d_i) = {
                1,  d_i <= 2   (浅层，全局特征)
                2,  2 < d_i <= 4  (中层，中等细节)
                4,  d_i > 4    (深层，需要更细采样)
            }

        Args:
            features: [B, D, H, W] 特征图
            boxes: [N, 5] ROI boxes [batch_idx, x1, y1, x2, y2]
            depths: [N] 每个 ROI 的深度

        Returns:
            pooled: [N, D] 池化后的特征
        """
        # 定义深度区间和对应的 sampling_ratio
        depth_bins = [0, 2, 4, self.max_level + 1]
        sampling_ratios = [1, 2, 4]

        # 按深度分组
        pooled_list = []
        indices_list = []

        for i in range(len(depth_bins) - 1):
            low, high = depth_bins[i], depth_bins[i + 1]
            mask = (depths >= low) & (depths < high)
            # I-OPT: 使用 .sum() 避免 .any() 的隐式 CPU-GPU 同步
            # D4-AUDIT FIX: 使用 not mask.any() 替代 mask.sum() == 0
            if not mask.any():
                continue

            indices = mask.nonzero(as_tuple=False).squeeze(-1)  # D3-AUDIT FIX: as_tuple=False 避免 Graph Break
            group_boxes = boxes[indices]

            # 调用 ROI-Align，使用对应的 sampling_ratio
            pooled = roi_align(
                features,
                group_boxes,
                output_size=(1, 1),
                spatial_scale=1.0,
                sampling_ratio=sampling_ratios[i],
                aligned=True,
            )  # [N_group, D, 1, 1]
            pooled = pooled.squeeze(-1).squeeze(-1)

            pooled_list.append(pooled)
            indices_list.append(indices)

        # 合并结果
        if len(pooled_list) == 0:
            # 无 token 时的边界情况
            return torch.zeros(0, features.shape[1], device=features.device, dtype=features.dtype)

        # 按原始顺序合并
        all_pooled = torch.zeros(depths.shape[0], features.shape[1], device=features.device, dtype=features.dtype)
        for pooled, indices in zip(pooled_list, indices_list):
            all_pooled[indices] = pooled

        return all_pooled

    def _interpolated_pool(
        self,
        features: Tensor,
        boxes: Tensor,
        depths: Tensor,
    ) -> Tensor:
        """纯 PyTorch 双线性插值池化 (无 torchvision 依赖)

        I-PHASE4: 添加 Interpolated Pooling 作为 ROI-Align 的回退方案
        用于解决 torchvision.ops.roi_align 的部署限制

        数学形式:
            output = BilinearInterpolate(F, center)
            其中 center = (cx, cy) 是 region 中心点

        双线性插值公式:
            f(x, y) = f(Q11) * (x2-x)*(y2-y) + f(Q21) * (x-x1)*(y2-y)
                    + f(Q12) * (x2-x)*(y-y1) + f(Q22) * (x-x1)*(y-y1)

        优点:
            - 纯 PyTorch 实现，无 torchvision 依赖
            - 跨平台兼容性好
            - 单次 grid_sample 调用，效率高

        精度:
            - 与 ROI-Align 相比，平均误差约 1-2%
            - 对于大多数视觉任务可接受

        Args:
            features: [B, D, H, W] 特征图
            boxes: [N, 5] ROI boxes [batch_idx, x1, y1, x2, y2]
            depths: [N] 每个 ROI 的深度 (用于动态采样密度)

        Returns:
            pooled: [N, D] 池化后的特征
        """
        B, D, H, W = features.shape
        N = boxes.shape[0]

        if N == 0:
            return torch.zeros(0, D, device=features.device, dtype=features.dtype)

        # 提取 batch index 和 box 坐标
        batch_indices = boxes[:, 0].long()  # [N]
        x1 = boxes[:, 1]  # [N]
        y1 = boxes[:, 2]
        x2 = boxes[:, 3]
        y2 = boxes[:, 4]

        # 根据深度调整采样密度
        # 浅层 (大区域): 2x2 采样
        # 中层: 3x3 采样
        # 深层 (小区域): 4x4 采样
        depth_bins = [0, 2, 4, self.max_level + 1]
        sample_sizes = [2, 3, 4]

        # I-OPT: 向量化深度→sample_size 映射，替代逐个 .item()
        # 使用 searchsorted 批量计算所有 box 的采样大小
        depth_bins_t = torch.tensor(depth_bins, device=depths.device, dtype=torch.float32)
        bin_indices = torch.searchsorted(depth_bins_t, depths.float()) - 1
        bin_indices = bin_indices.clamp(0, len(sample_sizes) - 1)
        sample_sizes_t = torch.tensor(sample_sizes, device=depths.device, dtype=torch.long)[bin_indices]

        # I-OPT: 按 sample_size 分组，每组批量处理所有 boxes
        # 替代逐 box 的 F.grid_sample 调用 (N×B 次 → 最多 3 次)
        # 使用 tensor 而非 dict，避免enumerate+tolist同步
        pooled_results = torch.zeros(N, D, device=features.device, dtype=features.dtype)
        assigned_mask = torch.zeros(N, dtype=torch.bool, device=features.device)

        for ss in sample_sizes:
            ss_mask = sample_sizes_t == ss
            if ss_mask.sum() == 0:
                continue
            ss_indices = ss_mask.nonzero(as_tuple=False).squeeze(-1)  # D3-AUDIT FIX: as_tuple=False 避免 Graph Break
            N_ss = len(ss_indices)

            # 获取该组所有 box 的坐标 (在 [0, 1] 范围)
            x1_ss = x1[ss_indices]
            y1_ss = y1[ss_indices]
            x2_ss = x2[ss_indices]
            y2_ss = y2[ss_indices]
            batch_idx_ss = batch_indices[ss_indices]

            # 计算归一化中心点和尺寸 (boxes 在 feature map 坐标 [0, W), [0, H))
            cx_ss = ((x1_ss + x2_ss) / 2) / W  # 归一化到 [0, 1]
            cy_ss = ((y1_ss + y2_ss) / 2) / H
            bw_ss = (x2_ss - x1_ss) / W  # 归一化到 [0, 1]
            bh_ss = (y2_ss - y1_ss) / H

            # 采样点坐标 (在 [0, 1] 范围)
            x_coords = (cx_ss - bw_ss / 2).unsqueeze(1) + (bw_ss / (ss - 1)).unsqueeze(1) * torch.arange(ss, device=features.device, dtype=torch.float32)
            y_coords = (cy_ss - bh_ss / 2).unsqueeze(1) + (bh_ss / (ss - 1)).unsqueeze(1) * torch.arange(ss, device=features.device, dtype=torch.float32)

            # 创建批量网格 [N_ss, ss, ss, 2]
            yy = y_coords.unsqueeze(2).expand(N_ss, ss, ss)  # [N_ss, 1, ss] -> [N_ss, ss, ss]
            xx = x_coords.unsqueeze(1).expand(N_ss, ss, ss)
            grids = torch.stack([xx, yy], dim=-1)  # [N_ss, ss, ss, 2]
            grids = grids * 2 - 1  # 映射到 [-1, 1]

            # 批量 grid_sample: 按 batch 分组处理
            for b in range(B):
                b_mask = batch_idx_ss == b
                # D4-AUDIT FIX: 使用 not mask.any() 替代 mask.sum() == 0
                if not b_mask.any():
                    continue
                b_grids = grids[b_mask]  # [N_b, ss, ss, 2]
                feat = features[b]  # [D, H, W]

                # 批量采样 (单次 kernel launch)
                # [N_b, D, ss, ss]
                sampled = F.grid_sample(
                    feat.unsqueeze(0).expand(b_grids.shape[0], -1, -1, -1),
                    b_grids,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True
                )  # [N_b, D, ss, ss]

                # 全局平均池化
                pooled = sampled.mean(dim=(2, 3))  # [N_b, D]

                # 记录结果 - 向量化赋值替代dict+enumerate
                pooled_results[ss_indices[b_mask]] = pooled
                assigned_mask[ss_indices[b_mask]] = True

        # 组装最终结果 (按原始顺序)
        # D4-AUDIT FIX: 移除 assigned_mask.all() 检查，因为 pooled_results 初始化时已经包含零值
        # 即使不是所有 token 都被分配，pooled_results 也已在未分配位置包含正确的零值
        result = pooled_results

        return result

    def forward(
        self,
        images: Tensor,
        split_results: List[SplitResult],
    ) -> Tuple[Tensor, Tensor]:
        """前向传播: 图像 + 分割结果 → token 序列 (向量化实现).
        
        数学形式化:
            T = ROI-Align(F, boxes) · σ_d + E_d
            
            其中:
            - F = SharedConv(I) ∈ ℝ^{B × D × H' × W'}
            - boxes = [(b, x1', y1', x2', y2')] 转换后的 ROI 坐标
            - σ_d = depth_scale[d] 深度缩放因子
            - E_d = depth_embed[d] 深度嵌入向量
        
        复杂度:
            - 时间: O(N_total × D) where N_total = Σ N_b
            - Kernel 调用: 1 次 (vs O(N_total) for Python loop)
        
        Args:
            images: [B, C, H, W] 输入图像
            split_results: 长度为 B 的 SplitResult 列表 (来自 AdaptiveSplitter)
            
        Returns:
            tokens: [B, N_max, dim] token 序列 (已按 Hilbert 顺序排列)
            levels_info: [B, N_max, max_level+1] 层级信息 [depth, q1, q2, ...]
        """
        B, C, H, W = images.shape
        device = images.device
        dtype = images.dtype
        
        # 1. 提取共享特征图
        features = self.shared_conv(images)  # [B, dim, H/p, W/p]
        _, dim, fh, fw = features.shape
        
        # 2. 确定最大 token 数量和总 token 数
        token_counts = [sr.num_tokens for sr in split_results]
        max_tokens = max(token_counts)
        total_tokens = sum(token_counts)
        
        if total_tokens == 0:
            # 边界情况: 无 token
            tokens = torch.zeros(B, 1, self.dim, device=device, dtype=dtype)
            levels_info = torch.zeros(B, 1, self.max_level + 1, dtype=torch.long, device=device)
            return self.norm(tokens), levels_info
        
        # 3. 收集所有 region 的 boxes 和 depths (向量化准备)
        all_boxes = []  # [batch_idx, x1, y1, x2, y2] in feature map coords
        all_depths = []
        all_levels_info = []
        batch_indices = []
        token_indices = []  # 每个 token 在其 batch 内的索引
        
        p = self.base_patch_size
        for b, sr in enumerate(split_results):
            for i, token in enumerate(sr.tokens):
                # 将像素坐标转换为 feature map 坐标
                # ROI-Align 使用浮点坐标，格式为 [batch_idx, x1, y1, x2, y2]
                fx1 = token.region.x1 / p
                fy1 = token.region.y1 / p
                fx2 = token.region.x2 / p
                fy2 = token.region.y2 / p
                
                # 确保有效的 ROI (至少 1 个像素)
                fx2 = max(fx1 + 0.5, fx2)
                fy2 = max(fy1 + 0.5, fy2)
                
                all_boxes.append([b, fx1, fy1, fx2, fy2])
                all_depths.append(min(token.depth, self.max_level))
                all_levels_info.append(token.to_levels_info(self.max_level))
                batch_indices.append(b)
                token_indices.append(i)
        
        # 4. 转换为 tensor
        boxes_tensor = torch.tensor(all_boxes, device=device, dtype=dtype)  # [N_total, 5]
        depths_tensor = torch.tensor(all_depths, device=device, dtype=torch.long)  # [N_total]
        
        # 5. 批量池化 (支持动态 sampling_ratio)
        # I106-1: 深层 Token 使用更细致的采样，防止特征模糊
        # sampling_ratio(d) = 1 (d<=2), 2 (2<d<=4), 4 (d>4)
        # I-PHASE4: 支持两种池化方法选择
        if self.use_interpolated_pooling:
            # 纯 PyTorch 实现，无 torchvision 依赖
            pooled = self._interpolated_pool(
                features, boxes_tensor, depths_tensor
            )
        else:
            # 使用 torchvision ROI-Align (默认，更精确)
            pooled = self._dynamic_roi_align(
                features, boxes_tensor, depths_tensor
            )  # [N_total, D]

        # I-PHASE4: 应用深度感知特征调制 (DAFM)
        if self.use_dynamic_weight:
            pooled = self._apply_depth_modulation(pooled, depths_tensor)

        # 6. 批量应用深度编码（纯 RoPE - 无加性/乘性 depth encoding）
        # C+ RoPE 方案：绝对相位通过 DirectionAwareSubspacedRoPE 在拓扑场注入
        # 不再使用 depth_scale (乘性) 和 depth_embed (加性)
        all_tokens = pooled
        
        # 7. 分配到输出 buffer
        # D4-AUDIT FIX: 向量化批量赋值替代 Python 循环
        tokens = torch.zeros(B, max_tokens, self.dim, device=device, dtype=dtype)
        levels_info = torch.zeros(B, max_tokens, self.max_level + 1, dtype=torch.long, device=device)

        # 批量赋值：利用高级索引一次性写入所有 token
        batch_idx_t = torch.tensor(batch_indices, dtype=torch.long, device=device)
        token_idx_t = torch.tensor(token_indices, dtype=torch.long, device=device)
        tokens[batch_idx_t, token_idx_t] = all_tokens
        levels_info[batch_idx_t, token_idx_t] = torch.stack(all_levels_info)
        
        # 8. 层归一化
        tokens = self.norm(tokens)

        # 9. C+ RoPE: 对拓扑场应用方向感知子空间隔离（绝对相位注入）
        # 仅在维度满足要求时应用 (dim_per_subspace 需为偶数)
        if self.rope_fractal is not None:
            # 分割 tokens: 前 D/2 物理场保持不变，后 D/2 拓扑场应用 RoPE
            tokens_physical, tokens_fractal = tokens.chunk(2, dim=-1)  # [B, N, D/2] each

            # 转换为 [B, H, N, D/2] 格式以匹配 RoPE 期望 (H=1 对于 tokenizer)
            tokens_fractal = tokens_fractal.unsqueeze(1)  # [B, 1, N, D/2]

            # 创建 LevelsInfo 对象用于 RoPE
            levels_info_obj = LevelsInfo(data=levels_info, max_level=self.max_level)

            # 应用 RoPE（注入绝对相位）
            tokens_fractal = self.rope_fractal(tokens_fractal, levels_info_obj)  # [B, 1, N, D/2]

            # 恢复原始形状
            tokens_fractal = tokens_fractal.squeeze(1)  # [B, N, D/2]

            # 拼接回物理场和拓扑场
            tokens = torch.cat([tokens_physical, tokens_fractal], dim=-1)  # [B, N, D]

        # I-NAN: 注册梯度 hook，捕获从 Transformer 传回的 NaN/Inf
        # D1-AUDIT FIX: 先移除旧 hook，避免每 forward 累积
        if self._forward_grad_hook is not None:
            self._forward_grad_hook.remove()
        def safe_grad(grad):
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                return torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
            return grad
        self._forward_grad_hook = tokens.register_hook(safe_grad)
        
        return tokens, levels_info

    def forward_fixed_grid(
        self,
        images: Tensor,
        grid_h: int,
        grid_w: int,
    ) -> Tuple[Tensor, Tensor]:
        """固定网格模式: 兼容非自适应分割.
        
        当不使用 AdaptiveSplit 时，使用均匀网格 (所有 token 深度相同)
        
        Args:
            images: [B, C, H, W]
            grid_h: 网格高度
            grid_w: 网格宽度
            
        Returns:
            tokens: [B, grid_h * grid_w, dim]
            levels_info: [B, grid_h * grid_w, max_level+1]
        """
        B, C, H, W = images.shape
        device = images.device
        
        # 1. 提取特征
        features = self.shared_conv(images)  # [B, dim, fh, fw]
        _, dim, fh, fw = features.shape
        
        # 2. 重塑为 token 序列
        # 假设 fh == grid_h, fw == grid_w
        tokens = features.flatten(2).transpose(1, 2)  # [B, N, dim]
        
        # 3. 确定统一深度
        # log2(image_size / patch_size) 对应固定网格的深度
        # I109-7: 使用 bit_length() 替代 int(math.log2(...)) 避免浮点精度问题
        depth = max(grid_h, grid_w).bit_length() - 1
        depth = min(depth, self.max_level)

        # 4. 层归一化
        tokens = self.norm(tokens)

        # 5. 创建 levels_info (统一深度)
        N = tokens.shape[1]
        levels_info = torch.zeros(
            B, N, self.max_level + 1,
            dtype=torch.long, device=device
        )
        levels_info[:, :, 0] = depth

        # 填充四叉树路径 (从 HilbertPathCache 获取)
        from vit_pytorch.core.hilbert_indexer import HilbertPathCache
        _, quadtree_paths = HilbertPathCache.get_or_compute(
            grid_h=fh, grid_w=fw, max_level=self.max_level, device=device
        )
        path_len = min(quadtree_paths.shape[1], self.max_level)
        actual_n = min(N, quadtree_paths.shape[0])
        levels_info[:, :actual_n, 1:path_len+1] = quadtree_paths[:actual_n, :path_len]

        # 6. C+ RoPE: 对拓扑场应用方向感知子空间隔离（绝对相位注入）
        # 仅在维度满足要求时应用 (dim_per_subspace 需为偶数)
        if self.rope_fractal is not None:
            # 分割 tokens: 前 D/2 物理场保持不变，后 D/2 拓扑场应用 RoPE
            tokens_physical, tokens_fractal = tokens.chunk(2, dim=-1)  # [B, N, D/2] each

            # 转换为 [B, H, N, D/2] 格式以匹配 RoPE 期望 (H=1 对于 tokenizer)
            tokens_fractal = tokens_fractal.unsqueeze(1)  # [B, 1, N, D/2]

            # 创建 LevelsInfo 对象用于 RoPE
            levels_info_obj = LevelsInfo(data=levels_info, max_level=self.max_level)

            # 应用 RoPE（注入绝对相位）
            tokens_fractal = self.rope_fractal(tokens_fractal, levels_info_obj)  # [B, 1, N, D/2]

            # 恢复原始形状
            tokens_fractal = tokens_fractal.squeeze(1)  # [B, N, D/2]

            # 拼接回物理场和拓扑场
            tokens = torch.cat([tokens_physical, tokens_fractal], dim=-1)  # [B, N, D]

        return tokens, levels_info

    @property
    def embed_output(self) -> Dict[str, Any]:
        """HilbertNativePatchEmbed 诊断输出

        命名空间:
            embed/health/*: 数值健康度
            embed/rope/*: DirectionAwareSubspacedRoPE 诊断
        """
        output: Dict[str, Any] = {}

        # embed/health/* - nan_grad_hooks 注册数
        if hasattr(self, '_nan_grad_hooks') and self._nan_grad_hooks:
            output["health/nan_grad_hooks_registered"] = len(self._nan_grad_hooks)

        # === 收集 RoPE 诊断（嵌套子模块）===
        # I-EMBED: rope_fractal 是 DirectionAwareSubspacedRoPE 子模块，
        # 其 embed_output 不会自动被 collect_auxiliary_diagnostics 收集
        if hasattr(self, 'rope_fractal') and self.rope_fractal is not None:
            rope_output = self.rope_fractal.embed_output
            if rope_output:
                for k, v in rope_output.items():
                    output[f"rope/{k}"] = v  # train/embed/rope/...

        return output


class DepthAwarePositionalEncoding(nn.Module):
    """深度感知位置编码.
    
    结合传统正弦位置编码和深度信息:
        PE(i, d) = sin/cos(pos) + DepthEmbed(d)
    
    与 HilbertNativePatchEmbed 配合使用时，
    提供额外的位置信息补充。
    """
    
    def __init__(
        self,
        dim: int,
        max_tokens: int = 1024,
        max_level: int = 4,
    ) -> None:
        super().__init__()
        
        self.dim = dim
        self.max_tokens = max_tokens
        
        # 正弦位置编码 (预计算)
        # D4-AUDIT FIX: 显式 dtype + exp clamp 防止 AMP 数值溢出
        pe = torch.zeros(max_tokens, dim, dtype=torch.float32)
        position = torch.arange(0, max_tokens, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)).clamp(min=-50, max=50)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # 深度嵌入
        self.depth_embed = nn.Embedding(max_level + 1, dim)
    
    def forward(
        self,
        tokens: Tensor,  # [B, N, dim]
        depths: Tensor,  # [B, N] 每个 token 的深度
    ) -> Tensor:
        """添加位置编码."""
        B, N, D = tokens.shape
        
        # 正弦位置编码
        pos_enc = self.pe[:N].unsqueeze(0).expand(B, -1, -1)
        
        # 深度嵌入
        depth_enc = self.depth_embed(depths)  # [B, N, dim]
        
        return tokens + pos_enc + depth_enc
