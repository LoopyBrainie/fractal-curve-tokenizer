"""
Splitter 接口协议定义（功能分组版本）- L1 Foundation

数学形式化
==========

Hilbert Curve ViT 的 Splitter 组件负责决策哪些区域需要进一步细分。

核心职责抽象:
    Splitter: S: (F, R, K) → SplitResult
    - 输入: 候选区域的特征表示 F，区域集合 R，预算约束 K
    - 输出: 选中区域及其元数据（包含 Hilbert 索引）

接口设计原则:
    1. 正交分解: Core/Annealing/Metrics 三层独立
    2. 渐进式实现: 可选择实现子集
    3. 类型安全: 完整的类型注解 + 运行时验证

版本: Protocol v2.0 (移动到 L1 Foundation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Dict, Any, Tuple, Optional, Literal
from torch import Tensor


# =============================================================================
# CoreSplitter: 核心决策逻辑（必须实现）
# =============================================================================

class CoreSplitter(Protocol):
    """
    Splitter 核心决策接口。

    数学形式:
        S: (F, R, K) → SplitResult
        其中:
        - F ∈ R^{B×C×H_f×W_f} 是特征空间
        - R = QuadtreeDecompose(L_max) 是候选区域集合
        - K 是 token 预算约束

    约束:
        1. Hilbert 局部性: 结果包含 hilbert_indices 保证空间邻近性
        2. 预算有界: K_min ≤ |SplitResult| ≤ K_max
        3. 可微性: 训练模式支持梯度流 (STE)
    """

    def forward(
        self,
        features: Tensor,
        image_size: Optional[Tuple[int, int]] = None,
        hard: bool = False,
    ) -> "SplitResult":
        """
        执行分割决策。

        数学:
            logits = MLP(ROI_aligned(F, R))
            selected = TopK_Gumbel(logits, K)
            result = {R_i | i ∈ selected}

        Args:
            features: [B, C, H_feat, W_feat] 特征图
            image_size: (H, W) 原始图像尺寸（可选，用于动态更新区域）
            hard: 是否使用硬决策（推理模式）

        Returns:
            SplitResult: 选中区域及其元数据

        Note:
            - 训练模式返回 STE mask（支持梯度）
            - 推理模式返回硬二值 mask
        """
        ...

    def update_candidates(self, image_size: Tuple[int, int]) -> None:
        """
        根据输入尺寸动态更新候选区域。

        数学:
            L_max = min(max_level_limit, floor(log2(min(H, W) / min_patch_size)))
            N = Σ_{d=0}^{L_max} 4^d = (4^{L_max+1} - 1) / 3

        Args:
            image_size: (H, W) 输入图像尺寸

        Note:
            - 当 image_size 变化时（如动态分辨率输入）
            - 需要重新计算候选区域坐标和深度
        """
        ...

    @property
    def max_level_limit(self) -> int:
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

        数学:
            N = Σ_{d=0}^{L_max} 4^d = (4^{L_max+1} - 1) / 3

        Returns:
            int: 候选区域总数 N
        """
        ...

    @property
    def is_training(self) -> bool:
        """
        获取当前训练/评估模式。

        Returns:
            bool: True 表示训练模式

        Note:
            使用 is_training 而非 training 以避免与 nn.Module.training 冲突
        """
        ...


# =============================================================================
# AnnealingSplitter: 退火控制（可选，用于训练）
# =============================================================================

class AnnealingSplitter(Protocol):
    """
    温度和偏置退火接口。

    数学形式:
        τ(t) = τ_start × (τ_end/τ_start)^(t/T)  # 指数退火
        τ(t) = τ_start - (τ_start - τ_end) × t/T  # 线性退火
        τ(t) = τ_end + (τ_start - τ_end) × cos²(π × t / (2T))  # 余弦退火

    温度 τ 的物理意义:
        - τ → ∞: 均匀随机探索
        - τ → 0: 确定性利用
        - 梯度强度: ∂p/∂z ∝ 1/τ
    """

    def set_temperature(self, temperature: float) -> None:
        """
        设置 Gumbel 温度参数。

        数学:
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

    def get_current_temperature(self) -> Tensor:
        """
        获取当前 Gumbel 温度张量。

        Returns:
            Tensor: 当前温度 τ ∈ [T_min, ∞)，GPU 张量
        """
        ...

    def set_annealing_schedule(
        self,
        schedule: Literal["linear", "exponential", "cosine"],
        start: float,
        end: float,
        total_steps: int,
    ) -> None:
        """
        配置退火调度。

        数学:
            linear: τ(t) = start + (end - start) × t/total_steps
            exponential: τ(t) = start × (end/start)^(t/total_steps)
            cosine: τ(t) = end + (start - end) × cos²(π × t / (2×total_steps))

        Args:
            schedule: 退火调度类型
            start: 起始温度
            end: 终止温度
            total_steps: 总步数
        """
        ...

    def set_explore_bias(self, bias: float) -> None:
        """
        设置探索偏置。

        数学:
            logits'_i = logits_i + b_explore
            其中 b_explore ∈ [0, 1] 控制探索程度

        Args:
            bias: 探索偏置值
        """
        ...


# =============================================================================
# MetricsSplitter: 诊断指标和正则化（可选，用于训练和分析）
# =============================================================================

class MetricsSplitter(Protocol):
    """
    诊断指标和正则化接口。

    数学形式:
        H(π) = -Σ_d π_d log(π_d)           # 配额熵
        Var(d) = Σ_d π_d × (d - E[d])²     # 深度方差
        coverage = K_selected / N_total     # 覆盖率
        adaptive_coverage = coverage × min(H, W) / ref_size  # 自适应覆盖率

    正则化目标:
        L_total = L_task + λ_H × H(π) + λ_V × Var(d)
        - 熵正则化: 鼓励配额分布多样性
        - 方差正则化: 避免深度聚集
    """

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

    def get_quota_logits(self) -> Optional[Tensor]:
        """
        获取可学习配额 logits。

        数学:
            φ = [φ_0, φ_1, ..., φ_{D-1}]
            π_d = softmax(φ)_d

        Returns:
            None 如果未启用可学习配额
        """
        ...

    def get_quota_probs(self) -> Optional[Tensor]:
        """
        获取配额概率分布。

        数学:
            π = softmax(φ)

        Returns:
            None 如果未启用可学习配额
        """
        ...

    def get_entropy_loss(self) -> Tensor:
        """
        获取配额熵正则化损失。

        数学:
            L_entropy = -Σ_d π_d × log(π_d + ε)
            鼓励配额分布保持多样性

        Returns:
            Tensor: 熵损失值
        """
        ...

    def get_variance_regularization(self) -> Tensor:
        """
        获取深度方差正则化损失。

        数学:
            L_var = Σ_d π_d × (d - E[d])²
            E[d] = Σ_d d × π_d
            避免配额过度集中于特定深度

        Returns:
            Tensor: 方差正则化损失值
        """
        ...

    def get_coverage_stats(self) -> Dict[str, float]:
        """
        获取覆盖率统计。

        数学:
            raw_coverage = K_selected / N_total
            adaptive_coverage = raw_coverage × sqrt(min(H, W) / ref_size)

        Returns:
            Dict[str, float]: {
                'raw_coverage': float,
                'adaptive_coverage': float,
                'K_selected': int,
                'N_total': int,
            }
        """
        ...


# =============================================================================
# SplitResult: 标准输出数据结构
# =============================================================================

class SplitResult:
    """
    分割器输出的标准数据结构。

    字段说明:
        - regions: 选中区域的边界坐标 [M, 4]
        - depths: 每个区域的深度 [M]
        - batch_indices: 每个区域的 batch 索引 [M]
        - hilbert_indices: Hilbert 曲线排序索引 [M]（选中子集的 Hilbert 排序）
        - selected_mask: 选中掩码 [B, N]
        - K_soft: 可微分的软 K 值（STE 直通估计）用于 aux_budget 损失

    数学形式化:
        M = |{i : selected_mask[i] = 1}| (选中的 token 数量)
        N = 所有候选区域数量
        HilbertOrder: HilbertCurve(R_selected) → [0, M-1]（用于 TensorSplitResult 兼容）

    注意:
        - probs/logits/selected_mask: [B, N]（所有候选区域）
        - hilbert_indices: [M]（仅选中子集，用于 tokenizer 排序）
        - 若需 TV Loss 在全量空间计算，使用 splitter.hilbert_indices [N]
    """

    regions: Tensor           # [M, 4] 坐标 (x0, y0, x1, y1)
    depths: Tensor            # [M] 深度值
    batch_indices: Tensor     # [M] batch 索引
    hilbert_indices: Tensor   # [M] Hilbert 索引（选中子集的 Hilbert 排序）
    selected_mask: Optional[Tensor] = None  # [B, N] 选中掩码
    logits: Optional[Tensor] = None         # [B, N] 原始 logits
    probs: Optional[Tensor] = None          # [B, N] 分割概率
    K_soft: Optional[Tensor] = None         # [1] or [B] 可微分 K 值 (STE)

    def __init__(
        self,
        regions: Tensor,
        depths: Tensor,
        batch_indices: Tensor,
        hilbert_indices: Tensor,
        selected_mask: Optional[Tensor] = None,
        logits: Optional[Tensor] = None,
        probs: Optional[Tensor] = None,
        K_soft: Optional[Tensor] = None,
    ):
        """
        初始化 SplitResult。

        Args:
            regions: [M, 4] 选中区域坐标
            depths: [M] 区域深度
            batch_indices: [M] batch 索引
            hilbert_indices: [M] Hilbert 曲线索引（选中子集的 Hilbert 排序）
            selected_mask: [B, N] 二值选中掩码
            logits: [B, N] 原始 logits（可选）
            probs: [B, N] 分割概率（可选）
            K_soft: 可微分 K 值（STE 直通估计）（可选）
        """
        import torch

        self.regions = regions
        self.depths = depths
        self.batch_indices = batch_indices
        self.hilbert_indices = hilbert_indices
        self.selected_mask = selected_mask
        self.logits = logits
        self.probs = probs
        self.K_soft = K_soft

        # I: 添加 split_decision 别名以兼容 tokenizer
        # split_decision 用于语义分裂器，selected_mask 用于 H1SS
        self.split_decision = selected_mask

        # 验证形状一致性
        M = regions.shape[0]
        assert depths.shape[0] == M, f"depths 形状不匹配: {depths.shape[0]} vs {M}"
        assert batch_indices.shape[0] == M, f"batch_indices 形状不匹配"
        # 注意: hilbert_indices 现在是 [N]（所有候选区域），不再等于 M

    @property
    def num_selected(self) -> int:
        """
        获取选中的 token 数量。

        Returns:
            int: M = regions.shape[0]
        """
        return self.regions.shape[0]

    def tokens_per_batch(self, B: int) -> Tensor:
        """
        计算每个 batch 选中的 token 数量。

        数学:
            K_b = Σ_i 1{batch_indices[i] = b}

        Args:
            B: batch size

        Returns:
            Tensor: [B] 每个 batch 的 token 数量
        """
        import torch
        if self.batch_indices.numel() == 0:
            return torch.zeros(B, dtype=torch.long, device=self.batch_indices.device)
        # I99-1 FIX: clamp batch_indices 防止 bincount 越界
        batch_indices_clamped = self.batch_indices.clamp(min=0, max=B - 1)
        return torch.bincount(
            batch_indices_clamped,
            minlength=B
        )

    @property
    def splitter_output(self) -> Dict[str, Any]:
        """返回 Splitter 层的诊断包裹

        关键原则：只包含该 Splitter 实际产生的数据。
        废弃的 TopK 特有字段（top_k, temperature, gumbel_hard）不出现。

        诊断字段说明:
            - 基础指标: 所有 Splitter 都有的通用指标
            - H1SS 特有: entropy, budget_loss, tree_consistency, locality_score,
                         jump_loss, iou_mean, iou_std, alpha
            - H-Entmax 特有: grad_coverage

        Returns:
            Dict[str, Any]: 诊断指标字典
        """
        output: Dict[str, Any] = {}

        # === 所有 Splitter 都有的基础指标 ===
        if self.selected_mask is not None:
            total = self.selected_mask.numel()
            active = self.selected_mask.sum().item()
            output["active_ratio"] = active / total if total > 0 else 0.0
            output["active_count"] = int(active)

        if self.logits is not None:
            output["logits_mean"] = float(self.logits.mean().item())
            output["logits_std"] = float(self.logits.std().item())
            output["logits_abs_mean"] = float(self.logits.abs().mean().item())

        if self.probs is not None:
            import torch
            eps = 1e-8
            output["probs_max"] = float(self.probs.max().item())
            output["probs_entropy"] = float(
                -(self.probs * torch.log(self.probs + eps)).sum().item()
            )

        # === H1SS / HilbertOptimalSplitter 特有的辅助损失 ===
        # 这些字段来自 get_auxiliary_losses()，由 HilbertOptimalSplitter 调用时填充
        if hasattr(self, "entropy") and self.entropy is not None:
            entropy_val = self.entropy
            output["entropy"] = float(entropy_val.item()) if hasattr(entropy_val, "item") else float(entropy_val)

        if hasattr(self, "budget_loss") and self.budget_loss is not None:
            budget_val = self.budget_loss
            output["budget_loss"] = float(budget_val.item()) if hasattr(budget_val, "item") else float(budget_val)

        if hasattr(self, "tree_consistency") and self.tree_consistency is not None:
            tree_val = self.tree_consistency
            output["tree_consistency"] = float(tree_val.item()) if hasattr(tree_val, "item") else float(tree_val)

        if hasattr(self, "locality_score") and self.locality_score is not None:
            output["locality_score"] = float(self.locality_score)

        if hasattr(self, "jump_loss") and self.jump_loss is not None:
            output["jump_loss"] = float(self.jump_loss)

        if hasattr(self, "iou_mean") and self.iou_mean is not None:
            output["iou_mean"] = float(self.iou_mean)

        if hasattr(self, "iou_std") and self.iou_std is not None:
            output["iou_std"] = float(self.iou_std)

        # === Alpha 参数（所有 Splitter 都暴露，包括 H1SS 固定 alpha） ===
        if hasattr(self, "alpha") and self.alpha is not None:
            alpha_val = self.alpha
            output["alpha"] = float(alpha_val.item()) if hasattr(alpha_val, "item") else float(alpha_val)

        return output


# =============================================================================
# TensorSplitResult: Tokenizer内部使用的数据结构
# =============================================================================

@dataclass
class TensorSplitResult:
    """
    Tokenizer 内部使用的分割结果数据结构。

    这是 H1SS SplitResult 到 tokenizer 内部格式的适配器。

    字段说明:
        - regions: 选中区域的边界坐标 [M, 4]
        - depths: 每个区域的深度 [M]
        - batch_indices: 每个区域的 batch 索引 [M]
        - hilbert_indices: Hilbert 曲线排序索引 [M]
        - token_indices: token 索引 [M]
        - complexities: 复杂度分数 [M]
    """
    regions: Tensor           # [M, 4] 坐标 (x0, y0, x1, y1)
    depths: Tensor            # [M] 深度值
    batch_indices: Tensor     # [M] batch 索引
    hilbert_indices: Tensor   # [M] Hilbert 索引
    token_indices: Tensor     # [M] token 索引
    complexities: Tensor      # [M] 复杂度分数

    @property
    def num_tokens(self) -> int:
        """获取 token 总数"""
        return self.regions.shape[0]

    @classmethod
    def from_split_result(cls, split_result: SplitResult) -> "TensorSplitResult":
        """
        从 SplitResult 创建 TensorSplitResult。

        Args:
            split_result: H1SS 返回的 SplitResult

        Returns:
            TensorSplitResult
        """
        import torch

        regions = split_result.regions
        depths = split_result.depths
        batch_indices = split_result.batch_indices
        hilbert_indices = split_result.hilbert_indices

        # 创建 token_indices
        token_indices = torch.arange(
            len(regions),
            dtype=torch.long,
            device=regions.device
        )

        # 从 probs 或 logits 计算 complexities
        if split_result.probs is not None and split_result.probs.numel() > 0:
            complexities = split_result.probs.view(-1)
        elif split_result.logits is not None and split_result.logits.numel() > 0:
            complexities = split_result.logits.view(-1)
        else:
            complexities = torch.zeros(
                len(regions),
                dtype=torch.float32,
                device=regions.device
            )

        return cls(
            regions=regions.long(),
            depths=depths,
            batch_indices=batch_indices,
            hilbert_indices=hilbert_indices,
            token_indices=token_indices,
            complexities=complexities,
        )


# =============================================================================
# 运行时 Protocol 验证工具
# =============================================================================

def validate_core_splitter(splitter: Any, name: str = "Splitter") -> None:
    """
    验证 CoreSplitter 契约。

    Args:
        splitter: 要验证的对象
        name: 对象名称（用于错误信息）

    Raises:
        AssertionError: 如果缺少必需方法
    """
    required_methods = ['forward', 'update_candidates']
    required_properties = ['max_level_limit', 'num_candidates', 'is_training']

    for method in required_methods:
        assert hasattr(splitter, method), f"{name} 缺少必需方法: {method}"
        assert callable(getattr(splitter, method)), f"{name}.{method} 不是可调用的"

    for prop in required_properties:
        assert hasattr(splitter, prop), f"{name} 缺少必需属性: {prop}"


def validate_annealing_splitter(splitter: Any, name: str = "Splitter") -> None:
    """
    验证 AnnealingSplitter 契约（软验证，可选实现）。

    Args:
        splitter: 要验证的对象
        name: 对象名称（用于错误信息）
    """
    optional_methods = ['set_temperature', 'get_current_temperature']

    for method in optional_methods:
        if hasattr(splitter, method):
            assert callable(getattr(splitter, method)), f"{name}.{method} 不是可调用的"


def validate_metrics_splitter(splitter: Any, name: str = "Splitter") -> None:
    """
    验证 MetricsSplitter 契约（软验证，可选实现）。

    Args:
        splitter: 要验证的对象
        name: 对象名称（用于错误信息）
    """
    optional_methods = [
        'get_diagnostics',
        'get_depth_distribution',
        'get_quota_logits',
        'get_quota_probs',
        'get_entropy_loss',
        'get_variance_regularization',
        'get_coverage_stats',
    ]

    for method in optional_methods:
        if hasattr(splitter, method):
            assert callable(getattr(splitter, method)), f"{name}.{method} 不是可调用的"


def validate_splitter(splitter: Any, name: str = "Splitter") -> None:
    """
    完整验证 Splitter 契约。

    验证顺序:
        1. CoreSplitter (必需)
        2. AnnealingSplitter (可选)
        3. MetricsSplitter (可选)

    Args:
        splitter: 要验证的对象
        name: 对象名称（用于错误信息）
    """
    validate_core_splitter(splitter, name)
    validate_annealing_splitter(splitter, name)
    validate_metrics_splitter(splitter, name)
