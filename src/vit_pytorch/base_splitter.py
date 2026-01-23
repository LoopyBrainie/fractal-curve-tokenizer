"""
Splitter 核心接口定义

数学形式化
==========

Hilbert Curve ViT 的 Splitter 组件负责决策哪些区域需要进一步细分。

组件职责抽象:
    - Splitter: $S: \mathbb{R}^{N×D} → \{0,1\}^N$ (决策函数)
    - 输入: 候选区域的特征表示
    - 输出: 二值选择 mask (是否选中该区域)

与 Tokenizer 的关系:
    - 当前设计 (错误): Tokenizer 内部创建 Splitter
    - 正确设计: Splitter 作为独立组件，Tokenizer 使用其决策结果

    T ← S: Tokenizer 使用 Splitter 的输出，但不创建 Splitter

接口设计原则:
    1. 最小接口: 只暴露必要方法
    2. 协议而非抽象类: 使用 Protocol 实现 duck typing
    3. 类型安全: 完整的类型注解

作者: Claude Code
日期: 2026-01-22
"""

from __future__ import annotations

from typing import Protocol, Dict, Any, Tuple, Optional
from torch import Tensor


class BaseSplitter(Protocol):
    """
    Hilbert Curve ViT Splitter 核心接口协议。

    该协议定义了 Splitter 组件必须实现的最小接口。
    遵循 Protocol 设计模式，支持 duck typing 实现。

    核心方法:
        - forward(): 执行分割决策
        - update_candidates(): 更新候选区域（当 image_size 变化时）

    温度控制方法:
        - set_temperature(): 设置 Gumbel 温度
        - get_current_temperature(): 获取当前温度

    诊断方法:
        - get_diagnostics(): 获取诊断信息
        - get_depth_distribution(): 获取深度分布

    与 Tokenizer 的关系:
        - Tokenizer 不持有 Splitter 引用
        - Splitter 输出作为 Tokenizer 的输入
        - 两者通过 SplitResult 数据结构解耦
    """

    def forward(
        self,
        features: Tensor,
        image_size: Optional[Tuple[int, int]] = None,
        hard: bool = False,
    ) -> "SplitResult":
        """
        执行分割决策：哪些候选区域应该被选中。

        数学形式:
            对每个候选区域 i:
            - 计算复杂度分数 s_i = MLP(ROI_i)
            - 应用 Gumbel 扰动: g_i ~ Gumbel(0, 1)
            - Top-K 选择: selected = TopK(s_i + g_i, K)

        Args:
            features: [B, C, H_feat, W_feat] 特征图
            image_size: (H, W) 原始图像尺寸（可选，用于动态更新区域）
            hard: 是否使用硬决策（推理模式）

        Returns:
            SplitResult: 分割结果，包含选中区域信息

        Note:
            - 训练模式返回 STE mask（支持梯度）
            - 推理模式返回硬二值 mask
        """
        ...

    def update_candidates(self, image_size: Tuple[int, int]) -> None:
        """
        根据输入尺寸动态更新候选区域。

        数学形式:
            L_max = min(max_depth_limit, max(0, floor(log2(min(H, W) / min_patch_size))))

        Args:
            image_size: (H, W) 输入图像尺寸

        Note:
            - 当 image_size 变化时（如动态分辨率输入）
            - 需要重新计算候选区域坐标和深度
        """
        ...

    # =========================================================================
    # 温度控制接口
    # =========================================================================

    def set_temperature(self, temperature: float) -> None:
        """
        设置 Gumbel 温度参数。

        数学形式:
            τ 控制探索-利用权衡:
            - τ → ∞: 均匀随机选择
            - τ → 0: 确定性选择 argmax

        Args:
            temperature: 温度值 τ > 0

        Note:
            - 通常 τ ∈ [0.1, 1.0]
            - 温度退火：从高 τ 开始，逐渐降低到目标 τ
        """
        ...

    def get_current_temperature(self) -> float:
        """
        获取当前 Gumbel 温度值。

        Returns:
            float: 当前温度 τ > 0
        """
        ...

    # =========================================================================
    # 诊断接口
    # =========================================================================

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        获取 Splitter 诊断信息。

        Returns:
            Dict[str, Any]: 包含以下键的字典:
                - current_temperature: float - 当前温度
                - depth_distribution: Dict[int, float] - 深度分布
                - quota_allocation: List[float] - 配额分配
                - num_selected: int - 选中的 token 数
                - splitter_type: str - Splitter 类型名
        """
        ...

    def get_depth_distribution(self) -> Dict[int, float]:
        """
        获取深度分布统计。

        数学形式:
            P(d) = count(depth=d) / total_selected

        Returns:
            Dict[int, float]: 深度 → 比例映射
        """
        ...

    # =========================================================================
    # 属性接口
    # =========================================================================

    @property
    def max_depth_limit(self) -> int:
        """
        获取最大深度限制。

        Returns:
            int: 最大深度 L_max
        """
        ...

    @property
    def num_candidates(self) -> int:
        """
        获取当前候选区域数量。

        数学形式:
            N = Σ_{d=0}^{L_max} 4^d = (4^{L_max+1} - 1) / 3

        Returns:
            int: 候选区域总数 N
        """
        ...

    @property
    def training(self) -> bool:
        """
        获取当前训练/评估模式。

        Returns:
            bool: True 表示训练模式
        """
        ...


# =============================================================================
# 数据类型定义
# =============================================================================

class SplitResult:
    """
    分割器输出的标准数据结构。

    字段说明:
        - regions: 选中区域的边界坐标 [M, 4]
        - depths: 每个区域的深度 [M]
        - batch_indices: 每个区域的 batch 索引 [M]
        - hilbert_indices: Hilbert 曲线排序索引 [M]
        - selected_mask: 选中掩码 [B, N]

    数学形式化:
        M = |{i : selected_mask[i] = 1}| (选中的 token 数量)
    """

    regions: Tensor           # [M, 4] 坐标 (x0, y0, x1, y1)
    depths: Tensor            # [M] 深度值
    batch_indices: Tensor     # [M] batch 索引
    hilbert_indices: Tensor   # [M] Hilbert 索引
    selected_mask: Optional[Tensor] = None  # [B, N] 选中掩码
    logits: Optional[Tensor] = None         # [B, N] 原始 logits
    probs: Optional[Tensor] = None          # [B, N] 分割概率

    def __init__(
        self,
        regions: Tensor,
        depths: Tensor,
        batch_indices: Tensor,
        hilbert_indices: Tensor,
        selected_mask: Optional[Tensor] = None,
        logits: Optional[Tensor] = None,
        probs: Optional[Tensor] = None,
    ):
        """
        初始化 SplitResult。

        Args:
            regions: [M, 4] 选中区域坐标
            depths: [M] 区域深度
            batch_indices: [M] batch 索引
            hilbert_indices: [M] Hilbert 曲线索引
            selected_mask: [B, N] 二值选中掩码
            logits: [B, N] 原始 logits（可选）
            probs: [B, N] 分割概率（可选）
        """
        import torch

        self.regions = regions
        self.depths = depths
        self.batch_indices = batch_indices
        self.hilbert_indices = hilbert_indices
        self.selected_mask = selected_mask
        self.logits = logits
        self.probs = probs

        # 验证形状一致性
        M = regions.shape[0]
        assert depths.shape[0] == M, f"depths 形状不匹配: {depths.shape[0]} vs {M}"
        assert batch_indices.shape[0] == M, f"batch_indices 形状不匹配"
        assert hilbert_indices.shape[0] == M, f"hilbert_indices 形状不匹配"

    @property
    def num_selected(self) -> int:
        """
        获取选中的 token 数量。

        Returns:
            int: M = regions.shape[0]
        """
        return self.regions.shape[0]

    @property
    def tokens_per_batch(self) -> Tensor:
        """
        计算每个 batch 选中的 token 数量。

        Returns:
            Tensor: [B] 每个 batch 的 token 数量
        """
        import torch
        if self.batch_indices.numel() == 0:
            return torch.zeros(1, dtype=torch.long, device=self.batch_indices.device)
        return torch.bincount(
            self.batch_indices,
            minlength=int(self.batch_indices.max().item() + 1)
        )
