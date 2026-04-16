# -*- coding: utf-8 -*-
"""
Fractal Curve ViT 综合分析器

此模块提供对 Fractal Curve ViT 模型各组件的全面分析功能:

1. 分词分析 (Tokenization Analysis)
   - 四叉树分割可视化
   - 深度分布统计
   - Hilbert 索引分布

2. 注意力分析 (Attention Analysis)
   - 深度感知注意力矩阵
   - LCA 偏置可视化
   - Head-wise 分析

3. 位置编码分析 (Position Encoding Analysis)
   - Depth embedding 相似度
   - Path embedding 验证
   - LCA embed 分析

4. 梯度流分析 (Gradient Flow Analysis)
   - 各层梯度幅度
   - 端到端可微性验证

数学背景
========
Fractal Curve ViT 与标准 ViT 的核心差异:

| 维度 | 标准 ViT | Fractal Curve ViT |
|------|----------|-------------------|
| Tokenization | 固定 N×N patches | 自适应四叉树分割 |
| Token 数量 | 固定 (196 for 224×224) | 可变 (8-64) |
| 空间排序 | 栅格顺序 | Hilbert 曲线 |
| 位置编码 | 绝对/相对位置 | LCA Hilbert 偏置 |

使用方法
========
```python
from examples.analysis.fractal_vit_analyzer import FractalViTAnalyzer

# 创建分析器
analyzer = FractalViTAnalyzer(model, device='cuda')

# 分析分词过程
token_results = analyzer.analyze_tokenization(images)

# 分析注意力
attn_results = analyzer.analyze_attention(images)

# 生成综合报告
report = analyzer.generate_report(images)
```
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 尝试导入可视化模块
try:
    from .visualization.hilbert_splitter import (
        visualize_quadtree_split,
        plot_hilbert_curve,
        compare_orderings
    )
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    print("Warning: Visualization modules not available")


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class TokenizationResult:
    """分词分析结果"""
    tokens: torch.Tensor  # [B, N, D] token 嵌入
    levels_info: torch.Tensor  # [B, N, max_depth+1] 深度和路径信息
    hilbert_indices: torch.Tensor  # [B, N] Hilbert 排序索引
    depths: torch.Tensor  # [B, N] 每个 token 的深度
    num_tokens: int  # token 数量
    depth_distribution: Dict[int, int]  # 深度分布统计
    regions: List = field(default_factory=list)  # 分割区域


@dataclass
class AttentionResult:
    """注意力分析结果"""
    attention_weights: torch.Tensor  # [B, H, N, N] 注意力权重
    depth_matrix: torch.Tensor  # [D+1, D+1] 深度对聚合矩阵
    diagonal_dominance: float  # 对角线主导性
    decay_rate: float  # 跨尺度衰减率
    symmetry_score: float  # 对称性分数
    per_head_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)


@dataclass
class PositionEncodingResult:
    """位置编码分析结果"""
    depth_embed: torch.Tensor  # [D+1, dim] 深度嵌入
    depth_similarity: torch.Tensor  # [D+1, D+1] 余弦相似度矩阵
    separation_score: float  # 分离度分数
    hierarchy_score: float  # 层次性分数
    lca_embed_norm: float  # LCA 嵌入范数


@dataclass
class GradientFlowResult:
    """梯度流分析结果"""
    layer_gradients: Dict[str, torch.Tensor]  # 各层梯度
    gradient_norms: Dict[str, float]  # 各层梯度范数
    has_nan_gradients: bool  # 是否有 NaN 梯度
    gradient_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class AnalysisReport:
    """综合分析报告"""
    tokenization: TokenizationResult
    attention: AttentionResult
    position_encoding: PositionEncodingResult
    gradient_flow: GradientFlowResult
    metrics: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# 综合分析器
# =============================================================================

class FractalViTAnalyzer:
    """
    Fractal Curve ViT 综合分析器

    提供对模型各组件的深入分析，包括分词、注意力、位置编码和梯度流。
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        max_depth: int = 4,
        hooks_dir: Optional[str] = None
    ):
        """
        初始化分析器

        Args:
            model: Fractal Curve ViT 模型
            device: 计算设备 ('cuda' 或 'cpu')
            max_depth: 最大深度
            hooks_dir: 中间结果保存目录 (可选)
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.max_depth = max_depth
        self.hooks_dir = hooks_dir

        # 将模型移至设备
        self.model = self.model.to(self.device)
        self.model.eval()

        # 注册 hooks 用于提取中间结果
        self.hooks = {}
        self.intermediate_results = {}

        if hooks_dir:
            import os
            os.makedirs(hooks_dir, exist_ok=True)

    def _register_hooks(self):
        """注册 forward hooks 用于提取中间结果"""
        # 注册 tokenizer hooks
        if hasattr(self.model, 'fractal_tokenizer'):
            tokenizer = self.model.fractal_tokenizer
            self.hooks['tokenizer'] = tokenizer.register_forward_hook(
                self._make_hook_fn('tokenizer')
            )

        # 注册 transformer hooks
        if hasattr(self.model, 'transformer'):
            for i, block in enumerate(self.model.transformer.blocks):
                self.hooks[f'block_{i}'] = block.register_forward_hook(
                    self._make_hook_fn(f'block_{i}')
                )

        # 注册 attention hooks
        if hasattr(self.model, 'fractal_tokenizer'):
            tokenizer = self.model.fractal_tokenizer
            if hasattr(tokenizer, 'hilbert_attention'):
                self.hooks['attention'] = tokenizer.hilbert_attention.register_forward_hook(
                    self._make_hook_fn('attention')
                )

    def _make_hook_fn(self, name: str):
        """创建 hook 函数"""
        def hook(module, input, output):
            self.intermediate_results[name] = output
        return hook

    def _remove_hooks(self):
        """移除所有注册的 hooks"""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()

    # =========================================================================
    # 分词分析
    # =========================================================================

    def analyze_tokenization(
        self,
        images: torch.Tensor,
        visualize: bool = True,
        save_path: Optional[str] = None
    ) -> TokenizationResult:
        """
        分析分词过程

        数学背景:
        =========
        分词过程:

        1. 特征提取: F = SharedConv(I) ∈ R^{B, D, H', W'}

        2. 自适应分割: 使用 GumbelTopKSplitter
           - 评估 N 个候选区域
           - 基于复杂度选择 K 个区域
           - 保持树一致性约束

        3. Hilbert 排序: tokens = sort(regions, by=hilbert_indices)

        4. 深度编码: token = Pool(F[R]) × depth_scale[d] + depth_embed[d]

        Args:
            images: 输入图像 [B, C, H, W]
            visualize: 是否可视化
            save_path: 保存路径 (可选)

        Returns:
            TokenizationResult: 分词分析结果
        """
        self.model.eval()
        images = images.to(self.device)

        with torch.no_grad():
            # 注册 hooks
            self._register_hooks()

            # 前向传播
            output = self.model(images)

            # 提取分词结果
            if hasattr(output, 'tokenizer_output'):
                tokenizer_output = output.tokenizer_output
            elif hasattr(output, 'tokens'):
                tokenizer_output = output
            else:
                # 尝试从 intermediate results 获取
                tokenizer_output = self.intermediate_results.get('tokenizer')

            # 清理 hooks
            self._remove_hooks()

        # 解析结果
        if tokenizer_output is not None:
            if hasattr(tokenizer_output, 'tokens'):
                tokens = tokenizer_output.tokens
            else:
                tokens = tokenizer_output[0] if isinstance(tokenizer_output, tuple) else tokenizer_output

            if hasattr(tokenizer_output, 'levels_info'):
                levels_info = tokenizer_output.levels_info
            else:
                levels_info = None

            if hasattr(tokenizer_output, 'hilbert_indices'):
                hilbert_indices = tokenizer_output.hilbert_indices
            else:
                hilbert_indices = None

            if hasattr(tokenizer_output, 'batch_indices'):
                pass
            else:
                pass

            # 计算深度分布
            if levels_info is not None:
                depths = levels_info[:, :, 0].long()  # [B, N]
            else:
                depths = torch.zeros(tokens.shape[:2], dtype=torch.long, device=self.device)

            # 统计深度分布
            depth_distribution = {}
            for b in range(images.shape[0]):
                depth_counts = torch.bincount(depths[b].clamp(0, self.max_depth))
                for d in range(len(depth_counts)):
                    if depth_counts[d] > 0:
                        depth_distribution[d] = depth_distribution.get(d, 0) + depth_counts[d].item()

            result = TokenizationResult(
                tokens=tokens,
                levels_info=levels_info,
                hilbert_indices=hilbert_indices,
                depths=depths,
                num_tokens=tokens.shape[1] if tokens.dim() > 1 else 0,
                depth_distribution=depth_distribution,
                regions=[]
            )
        else:
            # 如果无法获取分词结果，返回空结果
            result = TokenizationResult(
                tokens=torch.zeros(1, 32, 384, device=self.device),
                levels_info=torch.zeros(1, 32, 5, device=self.device),
                hilbert_indices=torch.arange(32, device=self.device).unsqueeze(0),
                depths=torch.zeros(1, 32, dtype=torch.long, device=self.device),
                num_tokens=32,
                depth_distribution={0: 32}
            )

        # 可视化
        if visualize and HAS_VISUALIZATION:
            if images.shape[0] >= 1:
                img = images[0].cpu()
                if hilbert_indices is not None:
                    hilbert_indices[0] if hilbert_indices.dim() > 1 else hilbert_indices
                else:
                    torch.arange(result.num_tokens)

                if levels_info is not None:
                    depths_for_viz = levels_info[:, :, 0].long()[0]
                else:
                    depths_for_viz = depths[0] if depths.dim() > 1 else depths

                if save_path:
                    visualize_quadtree_split(
                        img,
                        result.regions if result.regions else [],
                        depths_for_viz,
                        save_path=save_path
                    )

        return result

    # =========================================================================
    # 注意力分析
    # =========================================================================

    def analyze_attention(
        self,
        images: torch.Tensor,
        layer_idx: int = -1,
        per_head: bool = True,
        visualize: bool = True,
        save_path: Optional[str] = None
    ) -> AttentionResult:
        """
        分析注意力模式

        数学背景:
        =========
        Hilbert 感知注意力:

        A_ij = softmax(Q_i · K_j / √d_k + τ_h · LCAEmbed(LCA(i,j)) + B_level)

        其中:
        - LCA(i,j) = 公共前缀深度 (0 到 max_depth)
        - B_level = level_scales[|d_i - d_j|]

        深度对聚合矩阵:
        M[d_q, d_k] = mean_{i,j: d_i=d_q, d_j=d_k} A[i,j]

        Args:
            images: 输入图像 [B, C, H, W]
            layer_idx: 分析的层索引
            per_head: 是否逐 head 分析
            visualize: 是否可视化
            save_path: 保存路径 (可选)

        Returns:
            AttentionResult: 注意力分析结果
        """
        self.model.eval()
        images = images.to(self.device)

        # 注册 hooks
        self._register_hooks()

        with torch.no_grad():
            output = self.model(images)

        # 提取注意力权重
        attention = None
        if 'attention' in self.intermediate_results:
            attention = self.intermediate_results['attention']
        elif hasattr(output, 'attention_weights'):
            attention = output.attention_weights

        # 提取深度信息
        tokenizer_output = None
        if hasattr(output, 'tokenizer_output'):
            tokenizer_output = output.tokenizer_output
        elif hasattr(output, 'levels_info'):
            tokenizer_output = output

        if tokenizer_output is not None and hasattr(tokenizer_output, 'levels_info'):
            levels_info = tokenizer_output.levels_info
            depths = levels_info[:, :, 0].long()  # [B, N]
        else:
            N = 32  # 默认 token 数量
            depths = torch.zeros(images.shape[0], N, dtype=torch.long, device=self.device)

        # 清理 hooks
        self._remove_hooks()

        # 计算深度对聚合矩阵
        if attention is not None:
            # 取平均 head
            if attention.dim() == 4:
                attention_mean = attention.mean(dim=1)  # [B, N, N]
            else:
                attention_mean = attention

            # 计算深度矩阵
            depth_matrix = self._compute_depth_attention_matrix(attention_mean, depths)
            diagonal_dominance = self._compute_diagonal_dominance(depth_matrix)
            decay_rate = self._compute_decay_rate(depth_matrix)
            symmetry_score = self._compute_symmetry(depth_matrix)

            per_head_metrics = {}
            if per_head and attention.dim() == 4:
                for h in range(attention.shape[1]):
                    head_attn = attention[:, h]  # [B, N, N]
                    head_matrix = self._compute_depth_attention_matrix(head_attn.mean(dim=0), depths[0])
                    per_head_metrics[h] = {
                        'diagonal_dominance': self._compute_diagonal_dominance(head_matrix),
                        'decay_rate': self._compute_decay_rate(head_matrix),
                        'symmetry': self._compute_symmetry(head_matrix)
                    }
        else:
            # 默认值
            depth_matrix = torch.zeros(self.max_depth + 1, self.max_depth + 1, device=self.device)
            diagonal_dominance = 1.0 / (self.max_depth + 1)
            decay_rate = 1.0
            symmetry_score = 0.0
            per_head_metrics = {}

        result = AttentionResult(
            attention_weights=attention if attention is not None else torch.zeros(1, 4, 32, 32),
            depth_matrix=depth_matrix,
            diagonal_dominance=diagonal_dominance,
            decay_rate=decay_rate,
            symmetry_score=symmetry_score,
            per_head_metrics=per_head_metrics
        )

        return result

    def _compute_depth_attention_matrix(
        self,
        attention: torch.Tensor,
        depths: torch.Tensor
    ) -> torch.Tensor:
        """计算深度对聚合注意力矩阵"""
        if attention.dim() == 3:
            attention = attention.mean(dim=0)  # [N, N]

        N = attention.shape[0]
        D = self.max_depth + 1

        # 确保 depths 在有效范围内
        depths = depths.long().clamp(0, self.max_depth)
        if depths.dim() == 2:
            depths = depths[0]  # 取第一个样本

        # 构建深度对索引
        depth_i = depths.unsqueeze(1).expand(N, N)  # [N, N]
        depth_j = depths.unsqueeze(0).expand(N, N)  # [N, N]

        # 分组累加
        sum_matrix = torch.zeros(D, D, device=attention.device)
        count_matrix = torch.zeros(D, D, device=attention.device)

        flat_idx = depth_i * D + depth_j
        flat_sum = torch.zeros(D * D, device=attention.device)
        flat_count = torch.zeros(D * D, device=attention.device)

        flat_sum.scatter_add_(0, flat_idx.flatten(), attention.flatten())
        flat_count.scatter_add_(0, flat_idx.flatten(), torch.ones_like(attention).flatten())

        sum_matrix = flat_sum.view(D, D)
        count_matrix = flat_count.view(D, D)

        # 计算平均值
        avg_matrix = sum_matrix / (count_matrix + 1e-8)

        return avg_matrix

    def _compute_diagonal_dominance(self, depth_matrix: torch.Tensor) -> float:
        """计算对角线主导性"""
        diag_sum = depth_matrix.diag().sum().item()
        total_sum = depth_matrix.sum().item()
        expected = 1.0 / depth_matrix.shape[0]
        return diag_sum / (total_sum + 1e-8) if total_sum > 0 else expected

    def _compute_decay_rate(self, depth_matrix: torch.Tensor) -> float:
        """计算跨尺度衰减率"""
        decay_rates = []
        D = depth_matrix.shape[0]
        for d in range(D):
            center_val = depth_matrix[d, d].item()
            neighbor_vals = []
            if d > 0:
                neighbor_vals.append(depth_matrix[d, d-1].item())
            if d < D - 1:
                neighbor_vals.append(depth_matrix[d, d+1].item())
            if neighbor_vals:
                avg_neighbor = sum(neighbor_vals) / len(neighbor_vals)
                if avg_neighbor > 1e-8:
                    decay_rates.append(center_val / avg_neighbor)
        return np.mean(decay_rates) if decay_rates else 1.0

    def _compute_symmetry(self, depth_matrix: torch.Tensor) -> float:
        """计算对称性分数"""
        return torch.norm(depth_matrix - depth_matrix.T, p='fro').item()

    # =========================================================================
    # 位置编码分析
    # =========================================================================

    def analyze_position_encoding(self) -> PositionEncodingResult:
        """
        分析位置编码

        数学背景:
        =========
        位置编码由三部分组成:

        1. 深度嵌入: depth_embed[d] ∈ R^dim
           - 可学习的嵌入向量
           - 每个深度一个向量

        2. 路径嵌入: path_embed[p] ∈ R^dim
           - 四叉树路径编码
           - p ∈ {0, 1, 2, 3} 表示象限

        3. LCA 偏置: lca_embed[l] ∈ R^dim
           - 基于公共前缀深度的嵌入
           - l ∈ [0, max_depth]

        Returns:
            PositionEncodingResult: 位置编码分析结果
        """
        # 提取嵌入
        depth_embed = None
        lca_embed = None

        if hasattr(self.model, 'fractal_tokenizer'):
            tokenizer = self.model.fractal_tokenizer
            if hasattr(tokenizer, 'depth_embed'):
                depth_embed = tokenizer.depth_embed.weight.detach()
            if hasattr(tokenizer, 'path_embed'):
                tokenizer.path_embed.weight.detach()
            if hasattr(tokenizer, 'lca_embed'):
                lca_embed = tokenizer.lca_embed.weight.detach()

        if depth_embed is None:
            # 默认值
            D = self.max_depth + 1
            dim = 384
            depth_embed = torch.randn(D, dim)

        # 计算深度嵌入相似度矩阵
        normed = F.normalize(depth_embed, p=2, dim=-1)
        similarity = torch.mm(normed, normed.T)

        # 计算分离度 (1 - 平均非对角相似度)
        mask_offdiag = ~torch.eye(similarity.shape[0], dtype=torch.bool)
        offdiag_mean = similarity[mask_offdiag].mean().item()
        separation_score = 1.0 - offdiag_mean

        # 计算层次性分数 (相邻深度更相似)
        adjacent_sims = []
        distant_sims = []
        D = similarity.shape[0]
        for d in range(D):
            for delta in range(1, D):
                if d + delta < D:
                    sim = similarity[d, d + delta].item()
                    if delta == 1:
                        adjacent_sims.append(sim)
                    else:
                        distant_sims.append(sim)
        hierarchy_score = np.mean(adjacent_sims) - np.mean(distant_sims)

        # LCA 嵌入范数
        lca_norm = 0.0
        if lca_embed is not None:
            lca_norm = lca_embed.norm(p=2).item() / lca_embed.shape[0]

        result = PositionEncodingResult(
            depth_embed=depth_embed,
            depth_similarity=similarity,
            separation_score=separation_score,
            hierarchy_score=hierarchy_score,
            lca_embed_norm=lca_norm
        )

        return result

    # =========================================================================
    # 梯度流分析
    # =========================================================================

    def analyze_gradient_flow(
        self,
        images: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loss_fn: Optional[nn.Module] = None
    ) -> GradientFlowResult:
        """
        分析梯度流

        数学背景:
        =========
        梯度流分析验证:

        1. 端到端可微性: 所有参数都有梯度
           - ∂L/∂θ ≠ 0 对于所有可学习参数 θ

        2. 梯度幅度分布:
           - 识别梯度消失/爆炸问题
           - 归一化: ||∂L/∂x|| / (||x|| + ε)

        3. STE 梯度验证:
           - GumbelTopK 使用 straight-through estimator
           - 验证梯度正确传播

        Args:
            images: 输入图像 [B, C, H, W]
            targets: 目标标签 [B] (可选，用于计算损失)
            loss_fn: 损失函数 (可选)

        Returns:
            GradientFlowResult: 梯度流分析结果
        """
        self.model.train()

        if targets is None:
            # 创建虚拟目标
            getattr(self.model, 'num_classes', 1000)
            targets = torch.zeros(images.shape[0], dtype=torch.long, device=self.device)

        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        images = images.to(self.device)
        targets = targets.to(self.device)

        # 前向传播
        output = self.model(images)

        # 检查输出维度
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        # 计算损失
        loss = loss_fn(logits, targets)

        # 反向传播
        self.model.zero_grad()
        loss.backward()

        # 提取梯度
        layer_gradients = {}
        gradient_norms = {}
        gradient_stats = {}
        has_nan = False

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                layer_gradients[name] = grad

                # 计算梯度范数
                grad_norm = grad.norm(p=2).item()
                gradient_norms[name] = grad_norm

                # 计算梯度统计
                gradient_stats[name] = {
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'min': grad.min().item(),
                    'max': grad.max().item(),
                    'has_nan': torch.isnan(grad).any().item(),
                    'has_inf': torch.isinf(grad).any().item()
                }

                if gradient_stats[name]['has_nan']:
                    has_nan = True

        # 清理
        self.model.zero_grad()

        result = GradientFlowResult(
            layer_gradients=layer_gradients,
            gradient_norms=gradient_norms,
            has_nan_gradients=has_nan,
            gradient_stats=gradient_stats
        )

        return result

    # =========================================================================
    # 综合报告生成
    # =========================================================================

    def generate_report(
        self,
        images: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> AnalysisReport:
        """
        生成综合分析报告

        Args:
            images: 输入图像 [B, C, H, W]
            targets: 目标标签 [B] (可选)

        Returns:
            AnalysisReport: 综合分析报告
        """
        print("\n" + "=" * 70)
        print(" Fractal Curve ViT Analysis Report")
        print("=" * 70)

        # 1. 分词分析
        print("\n[1] Tokenization Analysis")
        print("-" * 50)
        token_result = self.analyze_tokenization(images, visualize=False)
        print(f"   Token count: {token_result.num_tokens}")
        print(f"   Depth distribution: {token_result.depth_distribution}")

        # 2. 注意力分析
        print("\n[2] Attention Analysis")
        print("-" * 50)
        attn_result = self.analyze_attention(images, visualize=False)
        print(f"   Diagonal dominance: {attn_result.diagonal_dominance:.4f}")
        print(f"   Decay rate: {attn_result.decay_rate:.4f}")
        print(f"   Symmetry score: {attn_result.symmetry_score:.4f}")

        # 3. 位置编码分析
        print("\n[3] Position Encoding Analysis")
        print("-" * 50)
        pos_result = self.analyze_position_encoding()
        print(f"   Separation score: {pos_result.separation_score:.4f}")
        print(f"   Hierarchy score: {pos_result.hierarchy_score:.4f}")
        print(f"   LCA embed norm: {pos_result.lca_embed_norm:.4f}")

        # 4. 梯度流分析
        print("\n[4] Gradient Flow Analysis")
        print("-" * 50)
        grad_result = self.analyze_gradient_flow(images, targets)
        print(f"   Parameters with gradients: {len(grad_result.gradient_norms)}")
        print(f"   NaN gradients: {grad_result.has_nan_gradients}")

        # 计算平均梯度范数
        if grad_result.gradient_norms:
            avg_norm = np.mean(list(grad_result.gradient_norms.values()))
            print(f"   Average gradient norm: {avg_norm:.6f}")

        print("\n" + "=" * 70)

        # 汇总指标
        metrics = {
            'token_count': token_result.num_tokens,
            'diagonal_dominance': attn_result.diagonal_dominance,
            'decay_rate': attn_result.decay_rate,
            'separation_score': pos_result.separation_score,
            'hierarchy_score': pos_result.hierarchy_score,
            'has_nan_gradients': grad_result.has_nan_gradients
        }

        return AnalysisReport(
            tokenization=token_result,
            attention=attn_result,
            position_encoding=pos_result,
            gradient_flow=grad_result,
            metrics=metrics
        )

    # =========================================================================
    # 可视化辅助方法
    # =========================================================================

    def visualize_all(
        self,
        images: torch.Tensor,
        save_dir: Optional[str] = None
    ):
        """
        生成所有可视化

        Args:
            images: 输入图像 [B, C, H, W]
            save_dir: 保存目录 (可选)
        """
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)

        # 1. 分词可视化
        print("\nGenerating tokenization visualization...")
        token_result = self.analyze_tokenization(images, visualize=False)
        if HAS_VISUALIZATION:
            save_path = f"{save_dir}/tokenization.png" if save_dir else None
            if images.shape[0] >= 1:
                img = images[0].cpu()
                visualize_quadtree_split(
                    img,
                    [],
                    token_result.depths[0] if token_result.depths.dim() > 1 else token_result.depths,
                    save_path=save_path
                )

        # 2. Hilbert 曲线可视化
        print("Generating Hilbert curve visualization...")
        if HAS_VISUALIZATION:
            save_path = f"{save_dir}/hilbert_curve.png" if save_dir else None
            plot_hilbert_curve(order=4, save_path=save_path)

        # 3. 注意力热图
        print("Generating attention heatmap...")
        attn_result = self.analyze_attention(images, visualize=False)
        self._visualize_attention(attn_result, save_dir)

        # 4. 位置编码相似度
        print("Generating position encoding similarity...")
        pos_result = self.analyze_position_encoding()
        self._visualize_position_encoding(pos_result, save_dir)

        print("\nAll visualizations generated!")

    def _visualize_attention(
        self,
        result: AttentionResult,
        save_dir: Optional[str] = None
    ):
        """可视化注意力"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 6))
        matrix = result.depth_matrix.cpu().numpy()
        D = matrix.shape[0]

        sns.heatmap(
            matrix,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            xticklabels=[f'd={d}' for d in range(D)],
            yticklabels=[f'd={d}' for d in range(D)],
            ax=ax
        )

        ax.set_xlabel('Key Depth (d_k)')
        ax.set_ylabel('Query Depth (d_q)')
        ax.set_title('Depth-Pair Attention Matrix')

        plt.tight_layout()

        if save_dir:
            plt.savefig(f"{save_dir}/attention_depth_matrix.png", dpi=150, bbox_inches='tight')

        plt.show()

    def _visualize_position_encoding(
        self,
        result: PositionEncodingResult,
        save_dir: Optional[str] = None
    ):
        """可视化位置编码"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 6))
        similarity = result.depth_similarity.cpu().numpy()
        D = similarity.shape[0]

        sns.heatmap(
            similarity,
            annot=True,
            fmt='.2f',
            cmap='RdYlBu_r',
            xticklabels=[f'd={d}' for d in range(D)],
            yticklabels=[f'd={d}' for d in range(D)],
            ax=ax,
            square=True
        )

        ax.set_xlabel('Depth')
        ax.set_ylabel('Depth')
        ax.set_title(f'Depth Embedding Similarity\n(Separation: {result.separation_score:.3f})')

        plt.tight_layout()

        if save_dir:
            plt.savefig(f"{save_dir}/depth_similarity.png", dpi=150, bbox_inches='tight')

        plt.show()


# =============================================================================
# 演示函数
# =============================================================================

def demo_analyzer():
    """演示分析器使用"""
    print("\n" + "=" * 70)
    print(" FractalViTAnalyzer Demo")
    print("=" * 70)

    # 尝试加载模型
    try:
        from vit_pytorch import FractalCurveViT

        model = FractalCurveViT(
            image_size=64,
            num_classes=100,
            dim=256,
            depth=4,
            heads=4
        )

        # 创建分析器
        analyzer = FractalViTAnalyzer(model, device='cpu')

        # 创建测试图像
        images = torch.randn(2, 3, 64, 64)

        # 生成综合报告
        analyzer.generate_report(images)

        # 可视化
        print("\nGenerating visualizations...")
        analyzer.visualize_all(images, save_dir='./analysis_output')

    except ImportError as e:
        print(f"Could not load model: {e}")
        print("Running with synthetic data...")

        # 使用合成数据演示
        analyzer = FractalViTAnalyzer(None, device='cpu')

        # 演示各个分析函数
        print("\n[1] Position Encoding Analysis Demo")
        pos_result = analyzer.analyze_position_encoding()
        print(f"   Separation score: {pos_result.separation_score:.4f}")
        print(f"   Hierarchy score: {pos_result.hierarchy_score:.4f}")

    print("\n[2] Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FractalViT Analyzer Demo")
    parser.add_argument("--output-dir", type=str, default="./analysis_output",
                       help="Directory to save visualizations")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save visualizations")

    args = parser.parse_args()

    demo_analyzer()
