# -*- coding: utf-8 -*-
r"""梯度传播监控工具

用于监控模型训练过程中的梯度流动、死节点检测和损失函数曲率分析。

使用方式:
    from vit_pytorch.core.gradient_monitor import GradientMonitor

    monitor = GradientMonitor(model)
    # 训练循环中
    loss.backward()
    monitor.record()  # 记录梯度
    monitor.record_selection(split_info)  # 记录节点选中情况

    # 定期输出诊断
    if step % 100 == 0:
        print(monitor.generate_report())
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple
import warnings


@dataclass
class GradientFlowStats:
    """单层梯度统计"""
    name: str
    grad_norm_history: List[float] = field(default_factory=list)
    grad_max_history: List[float] = field(default_factory=list)
    grad_zero_count: int = 0
    total_samples: int = 0

    @property
    def avg_grad_norm(self) -> float:
        if not self.grad_norm_history:
            return 0.0
        return sum(self.grad_norm_history) / len(self.grad_norm_history)

    @property
    def max_grad_norm(self) -> float:
        if not self.grad_max_history:
            return 0.0
        return max(self.grad_max_history)

    @property
    def zero_grad_ratio(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.grad_zero_count / self.total_samples


class GradientFlowMonitor:
    """梯度流监控器

    为模型每一层注册 backward hook，收集梯度范数统计。
    """

    def __init__(self, model: nn.Module, enabled: bool = True):
        """
        Args:
            model: 要监控的模型
            enabled: 是否启用监控
        """
        self.model = model
        self.enabled = enabled
        self.stats: Dict[str, GradientFlowStats] = {}
        self.hooks: List[Any] = []
        self._layer_names: List[str] = []

        if enabled:
            self._register_hooks()

    def _register_hooks(self):
        """为模型每一层注册 backward hook"""
        self._layer_names = []
        for name, module in self.model.named_modules():
            # 跳过无参数的层
            if len(list(module.parameters())) == 0:
                continue

            self._layer_names.append(name)
            self.stats[name] = GradientFlowStats(name=name)

            def make_hook(layer_name: str):
                def hook(module, grad_input, grad_output):
                    if not self.enabled:
                        return
                    # grad_output 是元组，取第一个元素
                    grad = grad_output[0]
                    if grad is None:
                        return

                    stats = self.stats[layer_name]
                    stats.total_samples += 1

                    # 计算梯度范数
                    norm = grad.norm().item()
                    max_val = grad.abs().max().item()

                    stats.grad_norm_history.append(norm)
                    stats.grad_max_history.append(max_val)

                    # 记录梯度为0的情况
                    if norm < 1e-8:
                        stats.grad_zero_count += 1

                return hook

            # 注册 backward hook
            handle = module.register_full_backward_hook(make_hook(name))
            self.hooks.append(handle)

    def disable(self):
        """禁用监控"""
        self.enabled = False

    def enable(self):
        """启用监控"""
        self.enabled = True

    def reset(self):
        """重置统计信息"""
        for stats in self.stats.values():
            stats.grad_norm_history.clear()
            stats.grad_max_history.clear()
            stats.grad_zero_count = 0
            stats.total_samples = 0

    def remove_hooks(self):
        """移除所有 hooks"""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def generate_table(self, top_k: int = 20) -> str:
        """生成梯度范数表格

        Args:
            top_k: 显示的最多层数

        Returns:
            Markdown 格式的表格字符串
        """
        if not self.stats:
            return "无梯度数据"

        # 按平均梯度范数排序
        sorted_stats = sorted(
            self.stats.values(),
            key=lambda s: s.avg_grad_norm,
            reverse=True
        )[:top_k]

        lines = [
            "### 梯度流可视化",
            "",
            "| 层级 | 平均梯度范数 | 最大梯度范数 | 零梯度比例 |",
            "|------|-------------|-------------|-----------|"
        ]

        for stats in sorted_stats:
            name = stats.name.split('.')[-1]  # 取最后一级名称
            avg_norm = f"{stats.avg_grad_norm:.6f}"
            max_norm = f"{stats.max_grad_norm:.6f}"
            zero_ratio = f"{stats.zero_grad_ratio:.2%}"
            lines.append(f"| {name} | {avg_norm} | {max_norm} | {zero_ratio} |")

        return "\n".join(lines)


class DeadNodeDetector:
    """死节点检测器

    追踪每个候选节点被选中的次数，检测长期不被选中的节点。
    """

    def __init__(self, num_candidates: int, device: torch.device = None):
        """
        Args:
            num_candidates: 候选节点总数
            device: 张量设备
        """
        self.num_candidates = num_candidates
        self.device = device or torch.device('cpu')
        self.selection_counts = torch.zeros(num_candidates, device=self.device)
        self.total_batches = 0

    def record_selection(self, selected_mask: torch.Tensor):
        """记录一次选中

        Args:
            selected_mask: [B, N] 选中掩码
        """
        if selected_mask.device != self.device:
            selected_mask = selected_mask.to(self.device)

        # 累加每个节点的选中次数
        self.selection_counts += selected_mask.sum(dim=0).detach()
        self.total_batches += 1

    def record_from_split_info(self, split_info: Dict[str, Any]):
        """从 split_info 中提取选中信息

        Args:
            split_info: TrainingStats.split_info 字典
        """
        if 'selection_counts' in split_info:
            # 已经有预计算的选中计数
            counts = split_info['selection_counts']
            if isinstance(counts, torch.Tensor):
                self.selection_counts += counts.to(self.device)
            else:
                self.selection_counts += torch.tensor(counts, device=self.device)
            self.total_batches += 1

    def detect_dead_nodes(self, threshold: int = 0) -> Tuple[torch.Tensor, float]:
        """检测死节点

        Args:
            threshold: 选中次数阈值，低于等于此值的节点视为死节点

        Returns:
            (死节点索引, 死节点比例)
        """
        dead_mask = self.selection_counts <= threshold
        # D1-AUDIT FIX: 使用 torch.where 替代 nonzero(as_tuple=True) 避免 Graph Break
        dead_indices = torch.where(dead_mask)[0]
        # D1-AUDIT FIX: 延迟 .item() 到后处理，移除同步点
        dead_ratio = dead_mask.float().mean()
        return dead_indices, dead_ratio

    def get_selection_distribution(self) -> Dict[str, float]:
        """获取选中次数分布统计

        Returns:
            分布统计字典
        """
        counts = self.selection_counts.cpu().numpy()
        return {
            'mean': float(counts.mean()),
            'std': float(counts.std()),
            'min': int(counts.min()),
            'max': int(counts.max()),
            'median': float(torch.median(self.selection_counts).item()),
        }

    def generate_report(self) -> str:
        """生成死节点检测报告"""
        dead_indices, dead_ratio = self.detect_dead_nodes(threshold=0)
        dist = self.get_selection_distribution()

        lines = [
            "### 死节点检测",
            "",
            f"- 总候选节点数: {self.num_candidates}",
            f"- 死节点数量: {len(dead_indices)} ({dead_ratio:.2%})",
            f"- 平均选中次数: {dist['mean']:.2f}",
            f"- 选中次数标准差: {dist['std']:.2f}",
            f"- 选中次数范围: [{dist['min']}, {dist['max']}]",
        ]

        if len(dead_indices) > 0:
            # 显示前10个死节点索引
            display_indices = dead_indices[:10].tolist()
            lines.append(f"- 死节点索引示例: {display_indices}" +
                        ("..." if len(dead_indices) > 10 else ""))

            # 建议
            if dead_ratio > 0.1:
                lines.append("")
                lines.append("⚠️ **建议**: 死节点比例超过10%，考虑：")
                lines.append("  - 增加熵正则化系数 (entropy_coef)")
                lines.append("  - 调整 Gumbel 温度参数")
                lines.append("  - 使用学习率预热")
            else:
                lines.append("")
                lines.append("✅ 死节点比例在正常范围内")
        else:
            lines.append("")
            lines.append("✅ 未检测到死节点")

        return "\n".join(lines)

    def reset(self):
        """重置统计"""
        self.selection_counts.zero_()
        self.total_batches = 0


class LossCurvatureAnalyzer:
    """损失函数曲率分析器

    计算两个损失函数的梯度方向一致性（余弦相似度）。
    """

    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        Args:
            model: 模型
            device: 设备
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.history: List[float] = []

    def compute_gradient_similarity(
        self,
        loss_budget: torch.Tensor,
        loss_ce: torch.Tensor,
        retain_graph: bool = True,
    ) -> float:
        """计算两个损失的梯度余弦相似度

        Args:
            loss_budget: LagrangianBudgetLoss
            loss_ce: CrossEntropyLoss
            retain_graph: 是否保留计算图

        Returns:
            余弦相似度 (-1 到 1)
        """
        self.model.eval()  # 需要.eval()以确保梯度正确
        self.model.zero_grad()

        # 获取所有参数
        params = list(self.model.parameters())
        param_grads_budget = torch.autograd.grad(
            loss_budget, params, retain_graph=retain_graph, allow_unused=True
        )
        param_grads_ce = torch.autograd.grad(
            loss_ce, params, retain_graph=False, allow_unused=True
        )

        # 展平并连接梯度
        grads_budget = torch.cat([
            g.flatten() if g is not None else torch.zeros(p.numel(), device=self.device)
            for g, p in zip(param_grads_budget, params)
        ])
        grads_ce = torch.cat([
            g.flatten() if g is not None else torch.zeros(p.numel(), device=self.device)
            for g, p in zip(param_grads_ce, params)
        ])

        # 计算余弦相似度
        cos_sim = F.cosine_similarity(
            grads_budget.unsqueeze(0),
            grads_ce.unsqueeze(0),
            dim=1
        ).item()

        self.history.append(cos_sim)
        return cos_sim

    def generate_report(self) -> str:
        """生成损失曲率分析报告"""
        if not self.history:
            return "### 损失函数曲率\n\n无数据"

        avg_sim = sum(self.history) / len(self.history)
        latest_sim = self.history[-1]

        lines = [
            "### 损失函数曲率",
            "",
            "- LagrangianBudgetLoss ↔ CrossEntropyLoss",
            f"- 当前余弦相似度: {latest_sim:.4f}",
            f"- 历史平均相似度: {avg_sim:.4f}",
        ]

        # 解释相似度含义
        if latest_sim > 0.8:
            status = "✅ 梯度方向高度一致，损失函数协同良好"
        elif latest_sim > 0.5:
            status = "⚠️ 梯度方向基本一致，存在一定冲突"
        elif latest_sim > 0.0:
            status = "⚠️ 梯度方向差异较大，可能相互抵消"
        else:
            status = "❌ 梯度方向相反，损失函数相互对抗"

        lines.append(f"- 状态: {status}")

        return "\n".join(lines)

    def reset(self):
        """重置历史"""
        self.history.clear()


class GradientRepairAdvisor:
    """梯度修复建议器

    检测梯度问题并提供修复建议。
    """

    def __init__(self, model: nn.Module = None):
        self.model = model
        self.issues: List[Dict[str, str]] = []

    def analyze_zero_gradients(
        self,
        grad_stats: Dict[str, GradientFlowStats],
    ) -> List[Dict[str, str]]:
        """分析零梯度问题

        Args:
            grad_stats: 梯度统计字典

        Returns:
            问题列表
        """
        issues = []

        for name, stats in grad_stats.items():
            if stats.zero_grad_ratio > 0.5:
                # 超过50%的梯度为0，这是严重问题
                issue = {
                    'layer': name,
                    'problem': f"零梯度比例 {stats.zero_grad_ratio:.1%}",
                    'suggestion': "检查是否使用了 detach() 或 requires_grad=False",
                }
                issues.append(issue)

        self.issues = issues
        return issues

    def generate_report(self) -> str:
        """生成修复建议报告"""
        lines = [
            "### 修复建议",
        ]

        if not self.issues:
            lines.append("")
            lines.append("✅ 未发现需要修复的梯度问题")
        else:
            lines.append("")
            for issue in self.issues:
                lines.append(f"⚠️ 层级: {issue['layer']}")
                lines.append(f"   问题: {issue['problem']}")
                lines.append(f"   建议: {issue['suggestion']}")
                lines.append("")

        return "\n".join(lines)


class GradientMonitor:
    """梯度监控综合类

    整合梯度流监控、死节点检测、损失曲率分析和修复建议。
    """

    def __init__(
        self,
        model: nn.Module,
        num_candidates: int = None,
        enabled: bool = True,
        log_interval: int = 100,
    ):
        """
        Args:
            model: 监控的模型
            num_candidates: 候选节点数量（用于死节点检测）
            enabled: 是否启用
            log_interval: 日志输出间隔
        """
        self.model = model
        self.log_interval = log_interval

        # 初始化子监控器
        self.grad_monitor = GradientFlowMonitor(model, enabled=enabled)
        self.curvature_analyzer = LossCurvatureAnalyzer(model)
        self.repair_advisor = GradientRepairAdvisor(model)

        # 死节点检测器（需要知道候选节点数量）
        if num_candidates is not None:
            self.dead_node_detector = DeadNodeDetector(num_candidates)
        else:
            self.dead_node_detector = None
            warnings.warn(
                "未提供 num_candidates，死节点检测将被跳过",
                UserWarning
            )

    def record(self):
        """记录当前梯度（backward 后调用）"""
        # GradientFlowMonitor 通过 hooks 自动记录
        pass

    def record_selection(self, selected_mask: torch.Tensor):
        """记录节点选中情况

        Args:
            selected_mask: [B, N] 选中掩码
        """
        if self.dead_node_detector is not None:
            self.dead_node_detector.record_selection(selected_mask)

    def record_selection_from_split_info(self, split_info: Dict[str, Any]):
        """从 split_info 记录选中情况

        Args:
            split_info: TrainingStats.split_info
        """
        if self.dead_node_detector is not None:
            self.dead_node_detector.record_from_split_info(split_info)

    def compute_curvature(
        self,
        loss_budget: torch.Tensor,
        loss_ce: torch.Tensor,
    ) -> float:
        """计算损失函数曲率

        Args:
            loss_budget: 预算损失
            loss_ce: 交叉熵损失

        Returns:
            余弦相似度
        """
        return self.curvature_analyzer.compute_gradient_similarity(
            loss_budget, loss_ce
        )

    def generate_report(self, step: int = None) -> str:
        """生成完整的诊断报告

        Args:
            step: 当前步数

        Returns:
            报告字符串
        """
        lines = []

        if step is not None:
            lines.append(f"[Gradient Monitor] Step {step}")
            lines.append("")

        # 1. 梯度流可视化
        lines.append(self.grad_monitor.generate_table())
        lines.append("")

        # 2. 死节点检测
        if self.dead_node_detector is not None:
            lines.append(self.dead_node_detector.generate_report())
            lines.append("")

        # 3. 损失函数曲率
        lines.append(self.curvature_analyzer.generate_report())
        lines.append("")

        # 4. 修复建议
        self.repair_advisor.analyze_zero_gradients(self.grad_monitor.stats)
        lines.append(self.repair_advisor.generate_report())

        return "\n".join(lines)

    def reset(self):
        """重置所有统计"""
        self.grad_monitor.reset()
        if self.dead_node_detector is not None:
            self.dead_node_detector.reset()
        self.curvature_analyzer.reset()
        self.issues.clear()

    def disable(self):
        """禁用监控"""
        self.grad_monitor.disable()

    def enable(self):
        """启用监控"""
        self.grad_monitor.enable()

    def __del__(self):
        """清理资源"""
        self.grad_monitor.remove_hooks()


def create_gradient_monitor(
    model: nn.Module,
    splitter_type: str = 'hilbert_optimal',
    **kwargs
) -> GradientMonitor:
    """创建梯度监控器（工厂函数）

    Args:
        model: 模型
        splitter_type: splitter 类型 ('hilbert_optimal')
        **kwargs: 其他参数

    Returns:
        GradientMonitor 实例
    """
    # 尝试从模型中推断候选节点数量
    num_candidates = None

    # 查找 tokenizer.splitter
    if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'splitter'):
        splitter = model.tokenizer.splitter
        if hasattr(splitter, 'num_candidates'):
            num_candidates = splitter.num_candidates
        elif hasattr(splitter, 'N'):
            num_candidates = splitter.N

    return GradientMonitor(
        model=model,
        num_candidates=num_candidates,
        **kwargs
    )
