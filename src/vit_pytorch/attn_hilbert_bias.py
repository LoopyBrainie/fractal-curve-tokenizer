# -*- coding: utf-8 -*-
"""
Hilbert 感知多尺度注意力模块

数学形式化
============

标准多头注意力:
    Attention(Q, K, V) = softmax(QK^T / √d_k) · V

Hilbert 感知注意力:
    HilbertAttn(Q, K, V) = softmax(QK^T / √d_k · σ_scale + B_hilbert + B_level) · V

偏置项
------
1. LCA Hilbert Bias (最近公共祖先):
   B_hilbert[i,j] = LCAEmbed(LCA(i,j))
   利用四叉树 LCA 深度直接编码空间距离，参数极少 (~100)

2. Level Bias (相对层级偏置):
   B_level[i,j] = Embedding(clamp(d_i - d_j + L, 0, 2L))

3. Level Scaling (层级缩放):
   σ_scale(d) = LevelScaleEmb(d)
   深层 token 使用较小缩放

类对照表
----------
+-------------------------------+------------------------------------------+
| 类                             | 数学定义                                   |
+===============================+==========================================+
| LCAHilbertBias                | B[i,j] = LCAEmbed(LCA(i,j))              |
| HilbertAwareMultiScaleAttention| Attn + B_hilbert + B_level              |
+-------------------------------+------------------------------------------+

P11-8 简化: 移除未使用的 LowRankHilbertBias 和 HierarchicalHilbertBias
"""

from __future__ import annotations

import math
import weakref
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .constants import HILBERT_BIAS_SCALE, LEVEL_BIAS_SCALE
from .utils import extract_depths, normalize_levels_info
from .embed_fractal_path import VectorizedPathEncoder
from .depth_utils import (
    compute_region_shape_scale,
    compute_shape_scale_similarity,
    compute_normalized_area,
    compute_area_similarity,
)


class HilbertBiasBase(ABC, nn.Module):
    """Hilbert Bias 抽象基类。
    
    统一处理 2D/3D levels_info 维度转换，子类只需实现核心计算逻辑。
    
    数学形式化
    ----------
    输入规范化:
        L' = normalize(L)  where  L ∈ R^{(S, Info)} → L' ∈ R^{(1, S, Info)}
                                  L ∈ R^{(B, S, Info)} → L' = L
    
    输出形状约定:
        - 2D 输入 → (H, S, S) 输出
        - 3D 输入 → (B, H, S, S) 输出
    
    设计原则 (P5-8):
        内部统一使用 3D 格式 (B, S, Info) 处理，避免代码重复
    """
    
    def forward(self, levels_info: torch.Tensor) -> Optional[torch.Tensor]:
        """计算 Hilbert Bias。
        
        自动处理 2D/3D 输入，维护输出形状兼容性。
        
        Args:
            levels_info: 层级信息张量
                - 2D: (S, Info) 单样本格式
                - 3D: (B, S, Info) 批量格式
            
        Returns:
            偏置矩阵:
                - 2D 输入 → (H, S, S)
                - 3D 输入 → (B, H, S, S)
            若输入无效则返回 None
        """
        if levels_info.numel() == 0:
            return None
        
        # 记录原始维度以决定输出形状
        was_2d = levels_info.dim() == 2
        
        # 统一规范化为 3D: (B, S, Info)
        levels_info_3d = normalize_levels_info(levels_info)
        
        # 调用子类实现的核心计算
        bias = self._compute_bias_3d(levels_info_3d)
        
        if bias is None:
            return None
        
        # 若原始输入为 2D，移除 batch 维度: (B, H, S, S) → (H, S, S)
        if was_2d:
            bias = bias.squeeze(0)
        
        return bias
    
    @abstractmethod
    def _compute_bias_3d(self, levels_info: torch.Tensor) -> Optional[torch.Tensor]:
        """核心计算逻辑（子类实现）。
        
        Args:
            levels_info: 规范化后的 3D 张量 (B, S, Info)
            
        Returns:
            偏置矩阵 (B, H, S, S) 或 None
        """
        ...


class LCAHilbertBias(HilbertBiasBase):
    """基于最近公共祖先 (LCA) 的 Hilbert Bias 实现。
    
    数学原理
    ========
    利用四叉树编码的核心性质: LCA 深度直接编码空间距离。
    
    定理 (LCA-距离等价性):
        对于四叉树编码的两个 token i, j:
        LCA(i, j) = ℓ  ⟹  ‖pos_i - pos_j‖_∞ ≤ N / 2^ℓ
        
    其中 N 是网格边长，ℓ 是 LCA 深度。
    
    偏置公式:
        B[i,j] = τ_h · LCAEmbed(LCA(i,j))
        
    其中 LCAEmbed: {0,1,...,D} → R^H 是可学习的嵌入表，
    τ_h 是 per-head 可学习温度参数。
    
    P11-3 修复: 从 regions 直接计算路径
    ===================================
    问题: 原始设计中 levels_info 的路径部分全为 0，导致 LCA 失效。
    
    根本原因:
        - TensorSplitResult 只存储 hilbert_indices 和 regions，不存储路径
        - 代码注释声称 "path 可从 hilbert_idx 恢复" 是数学错误
        - Hilbert index XOR ≠ Quadtree LCA (验证仅 56% 一致性)
    
    解决方案:
        添加 forward_from_regions() 方法，从 regions 向量化计算真实四叉树路径，
        然后计算正确的 LCA 深度矩阵。
    
    P6-2 改进: 可学习温度参数
    =========================
    问题: 原始 LCA 偏置范围 [0, 1]，相对 attention logit (σ≈1) 可能偏弱。
    
    解决方案: 引入 per-head 可学习温度 τ_h:
        B'[h,i,j] = τ_h · LCAEmbed(LCA(i,j))
    
    数学分析:
    - 信噪比: SNR = τ · ΔB / σ_logit = τ (当 ΔB=1, σ≈1)
    - 默认 τ=1.5 提供 1.5σ 的空间先验，对应 e^1.5 ≈ 4.5x 注意力偏好
    - 使用 softplus 确保 τ > 0: τ_h = softplus(γ_h)
    - 初始化 γ_h = log(e^1.5 - 1) ≈ 1.176 使 τ_h ≈ 1.5
    
    复杂度分析
    ==========
    - 参数量: O((D+1) × H + H) ≈ 128 + 8 (vs Low-Rank ~50K)
    - 计算量: O(N² × D) 用于 LCA 计算 (P1-6: 支持缓存避免重复计算)
    - 显存: O(N²) 用于偏置矩阵
    
    优势
    ====
    1. 显式几何意义: LCA 深度 ⟺ 空间距离
    2. 参数极少: ~100× 少于 Low-Rank
    3. 无需学习距离: 距离信息由编码结构直接提供
    4. 可解释性强: 偏置值可直接对应空间邻近程度
    5. [P6-2] 自适应强度: 每个 head 可学习最优的空间偏好强度
    6. [P11-3] 语义正确: 从 regions 直接计算真实四叉树 LCA
    """
    
    def __init__(
        self, 
        max_depth: int, 
        heads: int,
        lca_temperature: Optional[float] = 1.5,
        learnable_temperature: bool = True,
    ) -> None:
        """初始化 LCA Hilbert Bias。
        
        Args:
            max_depth: 最大四叉树深度 (决定 LCA 取值范围)
            heads: 注意力头数
            lca_temperature: LCA 偏置温度参数初始值
                - None: 不使用温度缩放 (兼容旧版，等效 τ=1)
                - float: 温度初始值，推荐 1.5
            learnable_temperature: 是否使温度可学习
                - True: per-head 可学习温度 (推荐)
                - False: 固定温度值
        """
        super().__init__()
        self.max_depth = max_depth
        self.heads = heads
        
        # LCA 深度嵌入表: depth ∈ {0, 1, ..., max_depth} → R^heads
        # 深度 0 表示完全不同的根节点，深度 max_depth 表示相邻或相同
        self.lca_embedding = nn.Embedding(max_depth + 1, heads)
        
        # P6-2: 可学习温度参数
        # 数学: τ_h = softplus(γ_h), 初始化使 τ_h ≈ lca_temperature
        self._lca_temperature_init = lca_temperature
        self._learnable_temperature = learnable_temperature
        self._init_temperature(lca_temperature, learnable_temperature)
        
        # P11-1 修复: LCA 深度矩阵缓存
        # I30-9: 使用 data_ptr + PyTorch 版本校验
        # 替代原 WeakRef 方案，解决身份检查无法捕获原地修改的问题
        #
        # 缓存结构: {data_ptr: (data_ptr, torch_version, lca_depths)}
        # 缓存命中条件: data_ptr 匹配 AND PyTorch 版本号匹配
        #
        # PyTorch 版本号 (._version) 在每次 in-place 操作时自动递增
        # 数学保证: hit ⇒ V(T_cache) = V(T_input) ⇒ D_cache = f(T_input)
        self._lca_cache_inputs: Dict[int, Tuple[int, int, torch.Tensor]] = {}
        
        # 初始化: 深度越大（越邻近）偏置越高
        # 使用对数衰减初始化，符合 Hilbert 曲线的 √ 局部性
        self._init_weights()
    
    def _init_temperature(
        self, 
        lca_temperature: Optional[float], 
        learnable: bool
    ) -> None:
        """初始化温度参数。
        
        P6-2 数学推导:
        - 使用 softplus: τ = log(1 + exp(γ))
        - 求逆: γ = log(exp(τ) - 1)
        - 对于 τ=1.5: γ = log(e^1.5 - 1) ≈ 1.176
        
        数值稳定性:
        - 当 τ < ln(2) ≈ 0.693 时, exp(τ) - 1 < 1, log 参数趋近 0
        - 使用 softplus_inverse 的稳定形式: γ = τ + log(1 - exp(-τ))
        - 此公式对所有 τ > 0 数值稳定
        
        Args:
            lca_temperature: 目标温度值，None 表示禁用
            learnable: 是否可学习
        """
        import math
        
        if lca_temperature is None:
            # 兼容模式: 无温度缩放
            self._lca_temp_raw: Optional[nn.Parameter] = None
            self._lca_temp_fixed: Optional[float] = None
        elif learnable:
            # 可学习模式: per-head 温度
            # I24-ALIGN: 数值稳定的 softplus 逆变换
            # 标准公式 γ = log(exp(τ) - 1) 在 τ < 0.693 时不稳定
            # 使用等价形式: γ = τ + log(1 - exp(-τ))
            # 对于 τ → 0+: γ → -∞ (正确)
            # 对于 τ → ∞: γ → τ (正确)
            if lca_temperature > 20:
                # 大温度: softplus 饱和，γ ≈ τ
                init_raw = lca_temperature
            else:
                # 通用稳定公式
                init_raw = lca_temperature + math.log(1 - math.exp(-lca_temperature))
            self._lca_temp_raw = nn.Parameter(
                torch.full((self.heads,), init_raw)
            )
            self._lca_temp_fixed = None
        else:
            # 固定模式
            self._lca_temp_raw = None
            self._lca_temp_fixed = lca_temperature
    
    @property
    def lca_temperature(self) -> Optional[torch.Tensor]:
        """获取当前 LCA 温度值。
        
        Returns:
            (H,) 温度向量，若禁用则返回 None
        """
        if self._lca_temp_raw is not None:
            # 可学习: softplus 确保正值
            return F.softplus(self._lca_temp_raw)
        elif self._lca_temp_fixed is not None:
            # 固定值
            return torch.tensor(self._lca_temp_fixed)
        else:
            # 禁用
            return None
    
    def _init_weights(self) -> None:
        """初始化 LCA 嵌入权重。
        
        采用对数衰减初始化:
            embed[d] ∝ log(1 + d) / log(1 + max_depth)
            
        这样深层 (邻近) token 获得更高的初始偏置。
        """
        with torch.no_grad():
            depths = torch.arange(self.max_depth + 1, dtype=torch.float32)
            # 归一化对数深度: [0, 1]
            log_depths = torch.log1p(depths) / torch.log1p(
                torch.tensor(float(self.max_depth))
            )
            # 广播到所有 heads，加小随机扰动
            init_values = log_depths.unsqueeze(1).expand(-1, self.heads)
            self.lca_embedding.weight.copy_(init_values)
            # 添加小随机扰动以打破对称性
            self.lca_embedding.weight.add_(
                torch.randn_like(self.lca_embedding.weight) * 0.02
            )
    
    def _compute_bias_3d(self, levels_info: torch.Tensor) -> Optional[torch.Tensor]:
        """计算基于 LCA 的 Hilbert Bias（核心 3D 实现）。
        
        数学形式:
            LCA[b,i,j] = sum_d prod_{k<=d} 1[p_i[k] = p_j[k]]
            Bias[b,i,j] = Embedding(LCA[b,i,j])
        
        P1-6 优化: LCA 深度矩阵缓存
            - 同一 batch 的 levels_info 在所有 Transformer 层间共享
            - 使用 data_ptr 作为缓存键，避免重复计算
            - 理论加速: 6层时约 6x
        
        复杂度: O(B·N²·D) 但无 Python 循环开销
        
        Args:
            levels_info: (B, S, Info) 规范化后的层级信息
            
        Returns:
            (B, H, S, S) 偏置矩阵，若无效则返回 None
        """
        batch_size, seq_len, info_dim = levels_info.shape
        if info_dim <= 1:
            return None

        # 提取四叉树路径: (B, S, Path)
        paths = levels_info[:, :, 1:].long()

        # I30-5: 路径值验证 + 警告
        # 四叉树路径值必须是 0-3 (对应四个象限: 左上, 右上, 左下, 右下)
        path_min = paths.min().item()
        path_max = paths.max().item()
        if path_max > 3 or path_min < 0:
            warnings.warn(
                f"[I30-5] levels_info path values out of range: "
                f"[{path_min}, {path_max}], expected [0, 3]. "
                f"Clipping will be applied. "
                f"This may indicate a tokenizer bug.",
                RuntimeWarning,
                stacklevel=2
            )
        paths = paths.clamp(0, 3)  # 安全保护仍保留

        # I30-9: 使用 data_ptr + PyTorch 版本校验
        # 解决 WeakRef 身份检查无法捕获原地修改的问题
        #
        # 缓存命中条件:
        #   data_ptr 匹配 AND PyTorch 版本号匹配
        #
        # PyTorch 版本号 (._version) 在每次 in-place 操作时自动递增
        # 这确保了原地修改后的张量能正确触发缓存失效
        data_ptr = levels_info.data_ptr()
        torch_version = levels_info._version if hasattr(levels_info, '_version') else 0
        cache_hit = False

        if data_ptr in self._lca_cache_inputs:
            cached_ptr, cached_version, cached_lca = self._lca_cache_inputs[data_ptr]

            # 双重校验: data_ptr 匹配 AND 版本匹配
            if cached_ptr == data_ptr and cached_version == torch_version:
                cache_hit = True
                lca_depths = cached_lca

        if not cache_hit:
            # 缓存未命中，计算 LCA
            lca_depths = VectorizedPathEncoder.compute_common_ancestor_depth(paths)
            lca_depths = lca_depths.clamp(0, self.max_depth)

            # 更新缓存: 使用当前 PyTorch 版本号
            self._lca_cache_inputs[data_ptr] = (data_ptr, torch_version, lca_depths)
        
        # 批量嵌入: (B, S, S, H)
        bias = self.lca_embedding(lca_depths)
        
        # P6-2: 应用温度缩放
        # 数学: B'[h,i,j] = τ_h · B[h,i,j]
        temperature = self.lca_temperature
        if temperature is not None:
            # temperature: (H,) -> (1, 1, 1, H) for broadcasting
            temp_scale = temperature.to(bias.device).view(1, 1, 1, -1)
            bias = bias * temp_scale
        
        # 调整形状: (B, H, S, S)
        return bias.permute(0, 3, 1, 2)
    
    def clear_cache(self) -> None:
        """清除 LCA 缓存。

        在以下情况调用:
        - 开始新的 batch 前
        - 评估/推理前后
        - 内存清理时

        I30-9: 使用 data_ptr + PyTorch 版本校验后，此方法清空缓存。
        """
        self._lca_cache_inputs.clear()
    
    def forward_from_regions(
        self,
        regions: torch.Tensor,
        image_size: int,
    ) -> Optional[torch.Tensor]:
        """从 regions 直接计算 LCA Hilbert Bias (P11-3 修复)。
        
        这是推荐的调用方式，绕过有问题的 levels_info 接口。
        
        数学原理:
            1. 从区域边界计算中心点: cx = (x1+x2)/2, cy = (y1+y2)/2
            2. 向量化计算四叉树路径: path[d] = bit(cx, D-d) + 2*bit(cy, D-d)
            3. 向量化计算 LCA: LCA[i,j] = len(common_prefix(path_i, path_j))
            4. 查表获取偏置: B[i,j] = LCAEmbed(LCA[i,j])
        
        Args:
            regions: 区域边界张量
                - 2D: (N, 4) 格式 [x1, y1, x2, y2]，单样本
                - 3D: (B, N, 4) 格式，批量
            image_size: 图像边长 (假设正方形)
            
        Returns:
            偏置矩阵:
                - 2D 输入 → (H, S, S)
                - 3D 输入 → (B, H, S, S)
            若输入无效则返回 None
        """
        if regions.numel() == 0:
            return None
        
        # 记录原始维度
        was_2d = regions.dim() == 2
        if was_2d:
            regions = regions.unsqueeze(0)  # [1, N, 4]
        
        B, N, _ = regions.shape

        # 转换为整数坐标 (确保 bit shift 操作正确)
        regions = regions.long()

        # 从 regions 计算四叉树路径
        paths = VectorizedPathEncoder.compute_paths_from_regions(
            regions, image_size, self.max_depth
        )  # [B, N, max_depth]
        
        # 计算 LCA 深度矩阵
        lca_depths = VectorizedPathEncoder.compute_common_ancestor_depth(paths)
        lca_depths = lca_depths.clamp(0, self.max_depth)  # [B, N, N]
        
        # 批量嵌入: (B, N, N, H)
        bias = self.lca_embedding(lca_depths)
        
        # P6-2: 应用温度缩放
        temperature = self.lca_temperature
        if temperature is not None:
            temp_scale = temperature.to(bias.device).view(1, 1, 1, -1)
            bias = bias * temp_scale
        
        # 调整形状: (B, H, S, S)
        bias = bias.permute(0, 3, 1, 2)
        
        # 若原始输入为 2D，移除 batch 维度
        if was_2d:
            bias = bias.squeeze(0)  # (H, S, S)
        
        return bias


class   HilbertAwareMultiScaleAttention(nn.Module):
    """Hilbert 曲线感知的多尺度注意力机制。

    通过编码层级深度和 Hilbert 路径关系来调制注意力权重。
    
    P11-2 修复: max_level 参数现在应传入与 tokenizer.max_depth 一致的值，
    确保 Embedding 表大小与实际使用的深度范围匹配，减少约 90% 的参数浪费。
    
    P11-8 简化: 移除 bias_mode 和 low_rank_r 参数，仅保留 LCA 模式。
    
    Attributes:
        heads: 注意力头数
        dim_head: 每个头的维度
        max_level: 最大层级 (应与 tokenizer.max_depth 匹配)
        use_hilbert_bias: 是否使用 Hilbert 偏置
        use_level_scaling: 是否使用层级缩放
        scale: 注意力缩放因子
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        max_level: int = 8,  # P11-2: 默认改为 8，应由上层传入实际 max_depth
        use_hilbert_bias: bool = True,
        use_level_scaling: bool = True,
        lca_temperature: Optional[float] = 1.5,
        learnable_temperature: bool = True,
        # I31-3: 仿射调制参数
        use_affine_modulation: bool = False,
        fourier_levels: int = 4,
    ) -> None:
        """初始化 HilbertAwareMultiScaleAttention。

        P11-2 修复: 参数 max_level 现在应传入与 tokenizer.max_depth 一致的值，
        而非硬编码的 50。这确保 level_scale 和 relative_pos_embedding 的
        Embedding 表大小与实际使用的深度范围匹配，减少约 90% 的参数浪费。

        P11-8 简化: 移除 bias_mode 和 low_rank_r 参数，仅保留 LCA 模式。

        I31-3: 添加仿射调制支持，通过面积信息调制注意力偏置。

        Args:
            dim: 输入维度
            heads: 注意力头数
            dim_head: 每个头的维度
            dropout: Dropout 比率
            max_level: 最大层级 (P11-2: should match tokenizer.max_depth)
            use_hilbert_bias: 是否使用 Hilbert 路径偏置 (使用 LCA 模式)
            use_level_scaling: 是否使用层级缩放
            lca_temperature: (P6-2) LCA 偏置温度参数，默认 1.5
                - None: 不使用温度缩放 (兼容模式)
                - float: 温度初始值
            learnable_temperature: (P6-2) 是否使温度可学习
            use_affine_modulation: (I31-3) 是否使用仿射调制偏置，默认 False
            fourier_levels: (I31-3) 傅里叶频率级别数，默认 4
        """
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.max_level = max_level
        self.use_hilbert_bias = use_hilbert_bias
        self.use_level_scaling = use_level_scaling
        self.use_affine_modulation = use_affine_modulation  # I31-3

        # I24-11: 注意力权重存储开关 (默认关闭以节省内存)
        # 评估时设为 True 以支持 attention 可视化和分析
        self.store_attn_weights: bool = False
        self._last_attn_weights: Optional[torch.Tensor] = None

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # P11-8 简化: 仅使用 LCA 模式
        if use_hilbert_bias:
            self.hilbert_bias_impl: Optional[nn.Module] = LCAHilbertBias(
                max_depth=max_level,
                heads=heads,
                lca_temperature=lca_temperature,
                learnable_temperature=learnable_temperature,
            )
        else:
            self.hilbert_bias_impl = None

        # I31-3: 仿射调制偏置 (可选)
        if use_affine_modulation:
            self.affine_modulated_bias: Optional[nn.Module] = AffineModulatedBias(
                dim=dim,
                max_depth=max_level,
                enable_area_modulation=True,
                fourier_levels=fourier_levels,
            )
        else:
            self.affine_modulated_bias = None

        if use_level_scaling:
            # P11-4 修复: 使用 Softplus 约束确保 level_scale > 0
            # 原设计使用 N(1.0, 0.1) 初始化，但无正性约束，有偏梯度可导致负值
            # 新设计: softplus(_level_scale_raw) ∈ (0, +∞)
            # 初始化 x=0.54 使得 softplus(0.54) ≈ 1.0
            self._level_scale_raw: Optional[nn.Embedding] = nn.Embedding(max_level + 1, heads)
            nn.init.constant_(self._level_scale_raw.weight, 0.54)  # softplus(0.54) ≈ 1.0
        else:
            self._level_scale_raw = None

        # P11-7 修复: 初始化使 softplus(0.54) ≈ 1.0，与 level_scale 保持一致
        self._scale_weights_raw = nn.Parameter(torch.full((heads,), 0.54))
        self.relative_pos_embedding = nn.Embedding(2 * max_level + 1, heads)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def _compute_hilbert_bias(
        self,
        levels_info: Optional[torch.Tensor] = None,
        regions: Optional[torch.Tensor] = None,
        image_size: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """计算基于 Hilbert 路径的注意力偏置。
        
        根据 bias_mode 调用不同的实现：
        - 'lca': LCA 嵌入表（推荐，支持 regions 直接计算）
        - 'low_rank': 低秩分解
        - 'hierarchical': 分层计算
        
        P11-3 改进: 当提供 regions + image_size 时，使用 forward_from_regions
        直接从区域边界计算正确的四叉树 LCA，绕过有问题的 levels_info 路径。
        
        Args:
            levels_info: 层级信息张量，形状为 (Seq, Info) 或 (Batch, Seq, Info)
                        [已废弃，路径部分全为 0]
            regions: (P11-3) 区域边界张量，形状为 (B, N, 4)
                     格式 [x1, y1, x2, y2]
            image_size: (P11-3) 图像边长，与 regions 配合使用
            
        Returns:
            Hilbert 偏置张量，形状为 (H, S, S) 或 (B, H, S, S)，若无效则返回 None
        """
        if not self.use_hilbert_bias:
            return None

        if self.hilbert_bias_impl is None:
            return None
        
        # P11-3: 优先使用 regions 直接计算 (LCA 模式)
        # P11-8: 简化后仅支持 LCA 模式
        if regions is not None and image_size is not None:
            if isinstance(self.hilbert_bias_impl, LCAHilbertBias):
                return self.hilbert_bias_impl.forward_from_regions(regions, image_size)
        
        # 回退到 levels_info
        if levels_info is not None and levels_info.numel() > 0:
            return self.hilbert_bias_impl(levels_info)
        
        return None

    def _compute_level_bias(self, levels_info: torch.Tensor) -> Optional[torch.Tensor]:
        """计算基于层级差异的相对位置偏置。
        
        Args:
            levels_info: 层级信息张量，形状为 (Seq, Info) 或 (Batch, Seq, Info)
            
        Returns:
            层级偏置张量，形状为 (H, S, S) 或 (B, H, S, S)，若无效则返回 None
        """
        if levels_info.numel() == 0:
            return None

        if levels_info.dim() == 2:
            # Old behavior: (Seq, Info)
            depths = extract_depths(levels_info, self.max_level)
            level_diff = depths.unsqueeze(0) - depths.unsqueeze(1)
            level_diff = level_diff.clamp(-self.max_level, self.max_level) + self.max_level
            rel_pos_bias = self.relative_pos_embedding(level_diff)
            return rel_pos_bias.permute(2, 0, 1) # (H, S, S)
        else:
            # New behavior: (Batch, Seq, Info)
            depths = extract_depths(levels_info, self.max_level) # (B, S)
            level_diff = depths.unsqueeze(2) - depths.unsqueeze(1) # (B, S, S)
            level_diff = level_diff.clamp(-self.max_level, self.max_level) + self.max_level
            rel_pos_bias = self.relative_pos_embedding(level_diff) # (B, S, S, H)
            return rel_pos_bias.permute(0, 3, 1, 2) # (B, H, S, S)

    def forward(
        self,
        x: torch.Tensor,
        levels_info: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        regions: Optional[torch.Tensor] = None,
        image_size: Optional[int] = None,
    ) -> torch.Tensor:
        """前向传播。
        
        P11-3 改进: 新增 regions 和 image_size 参数，用于直接从区域边界
        计算正确的四叉树 LCA 偏置，绕过 levels_info 中全为 0 的路径问题。
        
        Args:
            x: 输入张量，形状为 [B, N, D]
            levels_info: 层级信息（可选，用于 depth 提取和 level bias）
            attention_mask: 注意力掩码（可选）
            regions: (P11-3) 区域边界张量，形状为 [B, N, 4]，
                     格式 [x1, y1, x2, y2]，用于计算正确的 Hilbert LCA 偏置
            image_size: (P11-3) 图像边长，与 regions 配合使用
            
        Returns:
            输出张量，形状为 [B, N, D]
        """
        batch, _, _ = x.shape

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # STAB-3: 使用 softplus 约束 scale_weights 在 (0, +∞)，防止负数或过大
        # softplus(0) ≈ 0.69，接近 1.0，且训练中可学习调整
        scale_weights = F.softplus(self._scale_weights_raw)
        dots = dots * scale_weights.view(1, -1, 1, 1)

        if self.use_level_scaling and levels_info is not None and levels_info.numel() > 0:
            # Type guard: guaranteed non-None when use_level_scaling is True
            assert self._level_scale_raw is not None
            
            depths = extract_depths(levels_info, self.max_level)
            if levels_info.dim() == 2:
                # P11-4: Softplus 约束确保 level_scales ∈ (0, +∞)
                level_scales = F.softplus(self._level_scale_raw(depths))
                level_scales = level_scales.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
            else:
                level_scales = F.softplus(self._level_scale_raw(depths))  # (B, S, H)
                level_scales = level_scales.permute(0, 2, 1).unsqueeze(-1)  # (B, H, S, 1)
            
            dots = dots * level_scales

        if levels_info is not None:
            # I31-3: 仿射调制优先 (当启用且 regions 可用时)
            if self.use_affine_modulation and regions is not None and image_size is not None:
                affine_bias = self.affine_modulated_bias(regions, image_size)
                if affine_bias is not None:
                    # affine_bias: [B, dim, N, N]
                    # 需要广播到 [B, H, N, N]，通过对 dim 维度求均值
                    # 因为 dim = heads * head_dim，求均值后形状保持 [B, dim, N, N]
                    # 然后需要重塑为 [B, H, N, N]
                    if affine_bias.dim() == 4:
                        # 对 dim 维度求平均，得到 [B, dim, N, N]
                        affine_bias_avg = affine_bias.mean(dim=1, keepdim=True)  # [B, 1, N, N]
                        # 广播到所有 heads
                        affine_bias_final = affine_bias_avg.expand(-1, self.heads, -1, -1)  # [B, H, N, N]
                        dots = dots + affine_bias_final * HILBERT_BIAS_SCALE
            else:
                hilbert_bias = self._compute_hilbert_bias(
                    levels_info=levels_info,
                    regions=regions,
                    image_size=image_size,
                )
                if hilbert_bias is not None:
                    # hilbert_bias: (H, S, S) or (B, H, S, S)
                    if hilbert_bias.dim() == 3:
                        dots = dots + hilbert_bias.unsqueeze(0) * HILBERT_BIAS_SCALE
                    else:
                        dots = dots + hilbert_bias * HILBERT_BIAS_SCALE

            level_bias = self._compute_level_bias(levels_info)
            if level_bias is not None:
                # level_bias: (H, S, S) or (B, H, S, S)
                if level_bias.dim() == 3:
                    dots = dots + level_bias.unsqueeze(0) * LEVEL_BIAS_SCALE
                else:
                    dots = dots + level_bias * LEVEL_BIAS_SCALE

        if attention_mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            dots.masked_fill_(~attention_mask.bool(), mask_value)

        attn = self.attend(dots)
        
        # I24-11: 条件存储注意力权重 (评估时启用)
        if self.store_attn_weights:
            self._last_attn_weights = attn.detach()
        
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# ==================== I31: 形状-尺度编码器 ====================


class ShapeScaleEncoder(nn.Module):
    """形状-尺度编码器 (I31)

    数学形式化
    ==========

    将区域的几何特征编码为注意力偏置修正。

    特征定义:
        纵横比: r = log(w/h)  (对数变换，对称处理)
        面积:   s = (w/W) * (h/H)  (归一化到 [0, 1])

    门控机制:
        g = sigmoid(MLP([r; s]))
        输出: o = r * g + s * (1-g)

    门控选择理由 (I31-1):
        1. 自适应特征选择: 根据输入决定纵横比/面积的重要性
        2. 可解释性: g > 0.5 表示纵横比更重要
        3. 梯度有界: ∂g/∂x = g(1-g) * w ∈ (0, 0.25)

    组合偏置:
        B_final = B_LCA + τ * B_shape_scale
        τ = shape_scale_weight (零初始化，渐进启用)

    属性
    ----
    aspect_proj : nn.Sequential
        纵横比投影: 1 -> hidden -> dim/4
    area_proj : nn.Sequential
        面积投影: 1 -> hidden -> dim/4
    gate_net : nn.Sequential
        门控网络: 2*(dim/4) -> hidden -> 1 (Sigmoid)
    combine : nn.Linear
        组合层: dim/2 -> dim
    shape_scale_weight : nn.Parameter
        可学习权重 (零初始化)
    """

    def __init__(self, dim: int, hidden_dim: int = 32):
        """初始化形状-尺度编码器。

        参数
        ----
        dim : int
            输出嵌入维度
        hidden_dim : int, optional
            隐藏层维度，默认 32
        """
        super().__init__()

        # 特征投影: 1 -> hidden -> dim/4
        self.aspect_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim // 4)
        )

        self.area_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim // 4)
        )

        # 门控网络: 完整组合特征 -> hidden -> 1
        # 输入是 ar_emb 和 na_emb 的拼接 [B, N, D/2]，即完整的 32 维 (当 D=64 时)
        self.gate_net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 组合层: dim/2 -> dim
        self.combine = nn.Linear(dim // 2, dim)

        # 可学习权重 (零初始化，渐进启用)
        self.shape_scale_weight = nn.Parameter(torch.zeros(1))

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重。"""
        # 初始化所有 Linear 层
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        regions: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """计算形状-尺度嵌入。

        参数
        ----
        regions : torch.Tensor
            区域边界张量，形状 [B, N, 4]
            格式: [x1, y1, x2, y2]
        image_size : Tuple[int, int]
            (W, H) 图像尺寸

        返回
        ----
        torch.Tensor
            形状-尺度嵌入，形状 [B, N, dim]
        """
        B, N, _ = regions.shape
        W, H = image_size

        # 计算纵横比和面积 [B, N]
        aspect_ratios, normalized_areas = compute_region_shape_scale(
            regions, (W, H), epsilon=1e-8
        )

        # 编码 [B, N, D/4]
        ar_emb = self.aspect_proj(aspect_ratios.unsqueeze(-1))
        na_emb = self.area_proj(normalized_areas.unsqueeze(-1))

        # 门控计算 [B, N, D/2]
        combined = torch.cat([ar_emb, na_emb], dim=-1)
        gate = self.gate_net(combined)  # [B, N, 1]

        # 门控组合: 分别门控后拼接
        # gate 控制 ar_emb 和 na_emb 的相对权重
        # 输出维度保持 D/2，以匹配 combine 层的输入
        gated_ar = ar_emb * gate  # [B, N, D/4]
        gated_na = na_emb * (1 - gate)  # [B, N, D/4]
        gated = torch.cat([gated_ar, gated_na], dim=-1)  # [B, N, D/2]

        # 最终输出 [B, N, D]
        output = self.combine(gated)

        # 应用可学习权重
        output = output * self.shape_scale_weight

        return output


class LCAHilbertBiasWithShapeScale(nn.Module):
    """带形状-尺度修正的 LCA Hilbert 偏置 (I31)

    数学形式化
    ==========

    基础 LCA 偏置:
        B_LCA[i,j] = LCAEmbed(LCA(i,j))

    形状-尺度偏置:
        B_SS[i,j] = <ShapeScaleEncoder(c_i), ShapeScaleEncoder(c_j)>

    组合偏置:
        B_final[i,j] = B_LCA[i,j] + τ * B_SS[i,j]
        其中 τ = shape_scale_weight (零初始化)

    零初始化保证:
        τ = 0 时，B_final = B_LCA (退化为标准 LCA 偏置)

    属性
    ----
    lca_embedding : nn.Embedding
        LCA 深度嵌入层
    shape_scale_encoder : ShapeScaleEncoder
        形状-尺度编码器
    shape_scale_weight : nn.Parameter
        可学习组合权重
    """

    def __init__(
        self,
        dim: int,
        max_depth: int,
        lca_embedding_dim: Optional[int] = None,
        enable_shape_scale: bool = True,
        shape_scale_dim: Optional[int] = None,
        shape_scale_hidden_dim: int = 32,
    ):
        """初始化带形状-尺度修正的 LCA Hilbert 偏置。

        参数
        ----
        dim : int
            注意力维度
        max_depth : int
            最大四叉树深度
        lca_embedding_dim : int, optional
            LCA 嵌入维度，默认等于 dim
        enable_shape_scale : bool, optional
            是否启用形状-尺度修正，默认 True
        shape_scale_dim : int, optional
            形状-尺度嵌入维度，默认等于 dim
        shape_scale_hidden_dim : int, optional
            形状-尺度编码器隐藏层维度，默认 32
        """
        super().__init__()

        # LCA 嵌入层
        lca_embed_dim = lca_embedding_dim or dim
        self.lca_embedding = nn.Embedding(
            num_embeddings=max_depth + 1,
            embedding_dim=lca_embed_dim
        )

        # 形状-尺度编码器
        self.enable_shape_scale = enable_shape_scale
        if enable_shape_scale:
            self.shape_scale_encoder = ShapeScaleEncoder(
                dim=shape_scale_dim or dim,
                hidden_dim=shape_scale_hidden_dim
            )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重。"""
        # LCA 嵌入: 深度越大（越邻近）偏置越高
        with torch.no_grad():
            for d in range(self.lca_embedding.num_embeddings):
                # 对数衰减初始化: depth d -> scale log(d+1)
                scale = 0.1 * (1 + torch.log(torch.tensor(d + 1.0)))
                self.lca_embedding.weight[d].fill_(scale)

    def forward_from_regions(
        self,
        regions: torch.Tensor,
        image_size: int,
    ) -> torch.Tensor:
        """从 regions 计算 LCA Hilbert 偏置。

        参数
        ----
        regions : torch.Tensor
            区域边界张量，形状 [B, N, 4]
        image_size : int
            图像边长

        返回
        ----
        torch.Tensor
            偏置矩阵，形状 [B, H, N, N] 或 [B, dim, N, N]
        """
        # 委托给现有实现
        return self._compute_lca_bias_from_regions(regions, image_size)

    def _compute_lca_bias_from_regions(
        self,
        regions: torch.Tensor,
        image_size: int,
    ) -> torch.Tensor:
        """从 regions 计算 LCA 偏置 (内部方法)。"""
        B, N, _ = regions.shape

        if N == 0:
            return torch.zeros(B, 1, N, N, device=regions.device)

        # 转换为整数坐标 (VectorizedPathEncoder 需要整数)
        regions_int = regions.long()

        # 从 regions 计算四叉树路径
        paths = VectorizedPathEncoder.compute_paths_from_regions(
            regions_int, image_size, self.lca_embedding.num_embeddings - 1
        )

        # 计算 LCA 深度矩阵
        lca_depths = VectorizedPathEncoder.compute_common_ancestor_depth(paths)
        lca_depths = lca_depths.clamp(0, self.lca_embedding.num_embeddings - 1)

        # 嵌入 [B, N, N, dim]
        bias = self.lca_embedding(lca_depths)

        # 调整形状 [B, dim, N, N]
        return bias.permute(0, 3, 1, 2)

    def forward_with_shape_scale(
        self,
        regions: torch.Tensor,
        image_size: int,
    ) -> torch.Tensor:
        """带形状-尺度修正的偏置计算。

        参数
        ----
        regions : torch.Tensor
            区域边界张量，形状 [B, N, 4]
        image_size : int
            图像边长

        返回
        ----
        torch.Tensor
            组合偏置，形状 [B, dim, N, N]
        """
        B, N, _ = regions.shape

        # 1. LCA 偏置 [B, dim, N, N]
        lca_bias = self._compute_lca_bias_from_regions(regions, image_size)

        if not self.enable_shape_scale:
            return lca_bias

        # 2. 形状-尺度编码 [B, N, dim]
        shape_emb = self.shape_scale_encoder(regions, (image_size, image_size))

        # 3. 形状-尺度相似性矩阵 [B, N, N]
        shape_sim = torch.bmm(shape_emb, shape_emb.transpose(-2, -1))

        # 4. 投影到 dim 维度 [B, dim, N, N]
        # 复制 shape_sim 到所有 dim 维度
        shape_sim = shape_sim.unsqueeze(1)  # [B, 1, N, N]
        shape_sim = shape_sim.expand(-1, lca_bias.size(1), -1, -1)

        # 5. 可学习的组合权重
        tau = self.shape_scale_encoder.shape_scale_weight  # [1]

        # 组合偏置: B_final = B_LCA + τ * B_SS
        combined_bias = lca_bias + tau * shape_sim

        return combined_bias


# ==================== I31-3: 面积编码与仿射调制 ====================


class AreaEncoder(nn.Module):
    """面积编码器 (I31-3)

    数学形式化
    ==========

    面积归一化公式 (用户指定):

        .. math::
            f_{{area}} = \\frac{{\\log(s_{{patch}} + 1)}}{{\\log(S_{{total}} + 1)}}

    傅里叶特征 (NeRF-style):
        γ(f) = [sin(2^k π f), cos(2^k π f)]_{k=0}^{L-1}

    MLP 投影:
        E = W_2 · GELU(W_1 · γ(f))

    属性
    ----
    fourier_levels : int
        傅里叶频率级别数，默认 4
    fourier_dim : int
        傅里叶特征维度 = 2 × fourier_levels
    fourier_proj : nn.Linear
        傅里叶特征投影层
    mlp : nn.Sequential
        MLP 投影层
    area_weight : nn.Parameter
        可学习权重 (零初始化)
    """

    def __init__(
        self,
        dim: int,
        fourier_levels: int = 4,
        hidden_dim: int = 32,
    ):
        """初始化面积编码器。

        参数
        ----
        dim : int
            输出嵌入维度
        fourier_levels : int, optional
            傅里叶频率级别数，默认 4
        hidden_dim : int, optional
            隐藏层维度，默认 32
        """
        super().__init__()
        self.dim = dim
        self.fourier_levels = fourier_levels
        self.fourier_dim = fourier_levels * 2  # sin + cos

        # 傅里叶特征投影: 2L -> hidden
        self.fourier_proj = nn.Linear(self.fourier_dim, hidden_dim)

        # MLP 投影: hidden -> dim
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

        # 可学习权重 (零初始化)
        self.area_weight = nn.Parameter(torch.zeros(1))

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        regions: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """计算面积嵌入。

        参数
        ----
        regions : torch.Tensor
            区域边界张量，形状 [B, N, 4]
            格式: [x1, y1, x2, y2]
        image_size : int or Tuple[int, int]
            图像尺寸，可以是整数或 (W, H) 元组

        返回
        ----
        torch.Tensor
            面积嵌入，形状 [B, N, dim]
        """
        B, N, _ = regions.shape
        # 处理 image_size 格式：支持 int 或 (W, H) 元组
        if isinstance(image_size, int):
            image_size_tuple = (image_size, image_size)
        else:
            image_size_tuple = image_size
        W, H = image_size_tuple

        # 1. 计算归一化面积分数
        area_scores = compute_normalized_area(regions, image_size, epsilon=1e-8)
        # area_scores: [B, N]

        # 2. 傅里叶特征编码
        fourier_features = []
        for k in range(self.fourier_levels):
            freq = 2 ** k
            fourier_features.append(torch.sin(freq * math.pi * area_scores))
            fourier_features.append(torch.cos(freq * math.pi * area_scores))
        gamma = torch.stack(fourier_features, dim=-1)  # [B, N, 2L]

        # 3. MLP 投影
        area_emb = self.mlp(self.fourier_proj(gamma))

        # 4. 应用可学习权重
        area_emb = area_emb * self.area_weight

        return area_emb


class AffineModulatedBias(nn.Module):
    """仿射调制注意力偏置 (I31-3)

    数学形式化
    ==========

    仿射调制公式 (用户指定):
        B_{\text{final}} = γ(s_i, s_j) ⊙ B_{\text{spatial}} + β(s_i, s_j)

    其中:
        γ(s_i, s_j) = σ(MLP_γ(p_s))     # 缩放因子
        β(s_i, s_j) = MLP_β(p_s)        # 偏置因子
        p_s = area_emb[i] · area_emb[j] # 面积相似性

    残差连接:
        B_{\text{final}} = B_{\text{spatial}} + α · (γ ⊙ B_{\text{spatial}} + β - B_{\text{spatial}})

    属性
    ----
    lca_embedding : nn.Embedding
        LCA 深度嵌入层
    area_encoder : AreaEncoder
        面积编码器
    scale_net : nn.Sequential
        缩放网络: area_sim → [0, 1]
    bias_net : nn.Sequential
        偏置网络: area_sim → (-1, 1)
    residual_alpha : nn.Parameter
        残差权重 (零初始化)
    """

    def __init__(
        self,
        dim: int,
        max_depth: int,
        enable_area_modulation: bool = True,
        fourier_levels: int = 4,
    ):
        """初始化仿射调制偏置。

        参数
        ----
        dim : int
            注意力维度
        max_depth : int
            最大四叉树深度
        enable_area_modulation : bool, optional
            是否启用面积调制，默认 True
        fourier_levels : int, optional
            傅里叶频率级别数，默认 4
        """
        super().__init__()
        self.dim = dim
        self.max_depth = max_depth
        self.enable_area_modulation = enable_area_modulation

        # 空间偏置 (LCA)
        self.lca_embedding = nn.Embedding(
            num_embeddings=max_depth + 1,
            embedding_dim=dim
        )

        # 面积编码器和调制网络
        if enable_area_modulation:
            self.area_encoder = AreaEncoder(
                dim=dim,
                fourier_levels=fourier_levels,
                hidden_dim=32
            )

            # 傅里叶特征维度
            fourier_dim = fourier_levels * 2  # sin + cos

            # 缩放网络: fourier_features → [0, 1]
            self.scale_net = nn.Sequential(
                nn.Linear(fourier_dim, dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, dim),
                nn.Sigmoid()  # γ ∈ (0, 1)
            )

            # 偏置网络: fourier_features → ℝ
            self.bias_net = nn.Sequential(
                nn.Linear(fourier_dim, dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, dim),
                nn.Tanh()  # β ∈ (-1, 1)
            )

            # 残差权重 (零初始化)
            self.residual_alpha = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        """初始化权重。"""
        # LCA 嵌入: 深度越大（越邻近）偏置越高
        with torch.no_grad():
            for d in range(self.lca_embedding.num_embeddings):
                scale = 0.1 * (1 + torch.log(torch.tensor(d + 1.0)))
                self.lca_embedding.weight[d].fill_(scale)

    def _compute_lca_bias_from_regions(
        self,
        regions: torch.Tensor,
        image_size: int,
    ) -> torch.Tensor:
        """从 regions 计算 LCA 偏置 (内部方法)。"""
        B, N, _ = regions.shape

        if N == 0:
            return torch.zeros(B, 1, N, N, device=regions.device)

        # 转换为整数坐标
        regions_int = regions.long()

        # 从 regions 计算四叉树路径
        paths = VectorizedPathEncoder.compute_paths_from_regions(
            regions_int, image_size, self.max_depth
        )

        # 计算 LCA 深度矩阵
        lca_depths = VectorizedPathEncoder.compute_common_ancestor_depth(paths)
        lca_depths = lca_depths.clamp(0, self.max_depth)

        # 嵌入 [B, N, N, dim]
        bias = self.lca_embedding(lca_depths)

        # 调整形状 [B, dim, N, N]
        return bias.permute(0, 3, 1, 2)

    def forward(
        self,
        regions: torch.Tensor,
        image_size: int,
    ) -> torch.Tensor:
        """计算仿射调制偏置。

        参数
        ----
        regions : torch.Tensor
            区域边界张量，形状 [B, N, 4]
        image_size : int
            图像边长

        返回
        ----
        torch.Tensor
            仿射调制偏置，形状 [B, dim, N, N]
        """
        B, N, _ = regions.shape

        # 1. 空间偏置 (LCA) [B, dim, N, N]
        lca_bias = self._compute_lca_bias_from_regions(regions, image_size)

        if not self.enable_area_modulation:
            return lca_bias

        # 2. 面积编码 [B, N, dim]
        # image_size 已经是 (W, H) 元组，无需再包装
        area_emb = self.area_encoder(regions, image_size)

        # 3. 计算面积相似性矩阵 [B, N, N]
        # p_s[i,j] = area_emb[i] · area_emb[j]
        area_sim = torch.bmm(area_emb, area_emb.transpose(-2, -1))

        # 4. 傅里叶特征编码面积相似性
        fourier_features = []
        for k in range(self.area_encoder.fourier_levels):
            freq = 2 ** k
            fourier_features.append(torch.sin(freq * math.pi * area_sim))
            fourier_features.append(torch.cos(freq * math.pi * area_sim))
        gamma_features = torch.stack(fourier_features, dim=-1)  # [B, N, N, 2L]

        # 5. 仿射调制
        # 展平 [B, N, N, 2L] -> [B*N*N, 2L] 以便通过 Linear 层
        B, N, N, fourier_dim = gamma_features.shape
        gamma_flat = gamma_features.view(-1, fourier_dim)

        # 通过网络
        gamma = self.scale_net(gamma_flat)  # [B*N*N, dim]
        beta = self.bias_net(gamma_flat)    # [B*N*N, dim]

        # 恢复形状 [B, N, N, dim]
        gamma = gamma.view(B, N, N, self.dim)
        beta = beta.view(B, N, N, self.dim)

        # 调整维度以匹配 lca_bias: [B, dim, N, N]
        gamma = gamma.permute(0, 3, 1, 2)
        beta = beta.permute(0, 3, 1, 2)

        # 6. 仿射变换
        modulated = gamma * lca_bias + beta

        # 7. 残差连接
        combined_bias = lca_bias + self.residual_alpha * (modulated - lca_bias)

        return combined_bias

