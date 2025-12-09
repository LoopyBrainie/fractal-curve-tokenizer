# -*- coding: utf-8 -*-
"""FFN 变体消融实验

对比以下配置的性能差异：
1. Baseline: 仅 GELU 主网络
2. + Level Adaptation: 添加层级自适应
3. + Feature Gating: 添加特征门控
4. Full: 完整配置 (Level Adaptation + Feature Gating)
5. SwiGLU: 用 SwiGLU 替代 GELU + Feature Gating

评估指标：
- 参数量
- 前向/反向传播时间
- 内存占用
- 训练损失曲线（可选）
- 激活分布分析

Usage:
    uv run python tests/benchmarks/ablation_ffn_variants.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "src")

from vit_pytorch.feedforward import AdaptiveFractalFeedForward
from vit_pytorch.utils import extract_depths


# ============================================================================
# SwiGLU 实现（用于对比）
# ============================================================================

class SwiGLUFFN(nn.Module):
    """独立的 SwiGLU 实现（参考 LLaMA）
    
    SwiGLU(x) = (Swish(W_gate · x) ⊙ (W_value · x)) · W_out
    
    相比 GELU FFN:
    - 参数量相近（通过调整 hidden_dim）
    - 内置门控机制，无需额外 feature_gate
    - 梯度流动更平滑
    """
    
    def __init__(
        self, 
        dim: int, 
        hidden_dim: int, 
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        # 为保持参数量相当，SwiGLU 的 hidden_dim 通常为原始的 2/3
        self.hidden_dim = hidden_dim
        
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_value = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_out = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, levels_info: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = self.norm(x)
        gate = F.silu(self.w_gate(x_norm))  # Swish 激活
        value = self.w_value(x_norm)
        hidden = gate * value  # 元素级门控
        return self.dropout(self.w_out(hidden))


class SwiGLUWithLevelAdaptation(nn.Module):
    """SwiGLU + 层级自适应（方案 A 的实现）"""
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        max_level: int = 50,
    ):
        super().__init__()
        self.max_level = max_level
        
        # SwiGLU 主网络
        swiglu_hidden = (hidden_dim * 2) // 3
        self.norm = nn.LayerNorm(dim)
        self.w_gate = nn.Linear(dim, swiglu_hidden, bias=False)
        self.w_value = nn.Linear(dim, swiglu_hidden, bias=False)
        self.w_out = nn.Linear(swiglu_hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # 层级自适应（保留）
        self.level_embedding = nn.Embedding(max_level + 1, dim)
        self.level_adapter = nn.Sequential(
            nn.Linear(dim * 2, swiglu_hidden // 2),
            nn.SiLU(),
            nn.Linear(swiglu_hidden // 2, dim),
            nn.Dropout(dropout),
        )
        self.level_mixing = nn.Parameter(torch.ones(max_level + 1))
    
    def forward(self, x: torch.Tensor, levels_info: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        x_norm = self.norm(x)
        
        # SwiGLU 主路径
        gate = F.silu(self.w_gate(x_norm))
        value = self.w_value(x_norm)
        main_out = self.dropout(self.w_out(gate * value))
        
        # 层级自适应
        if levels_info is not None and levels_info.numel() > 0:
            depths = extract_depths(levels_info, self.max_level)
            if depths.dim() == 1:
                depths = depths.unsqueeze(0).expand(batch, -1)
            
            level_embs = self.level_embedding(depths)
            adapter_input = torch.cat([x_norm, level_embs], dim=-1)
            level_out = self.level_adapter(adapter_input)
            
            mixing = F.softmax(self.level_mixing[depths], dim=1).unsqueeze(-1)
            main_out = main_out * (1 - mixing) + level_out * mixing
        
        return main_out


# ============================================================================
# 消融实验配置
# ============================================================================

@dataclass
class FFNConfig:
    """FFN 配置"""
    name: str
    use_level_adaptation: bool = False
    use_feature_gating: bool = False
    ffn_class: str = "adaptive"  # "adaptive" | "swiglu" | "swiglu_level"
    
    def create_ffn(self, dim: int, hidden_dim: int, dropout: float = 0.0, max_level: int = 50) -> nn.Module:
        if self.ffn_class == "swiglu":
            swiglu_hidden = (hidden_dim * 2) // 3
            return SwiGLUFFN(dim, swiglu_hidden, dropout)
        elif self.ffn_class == "swiglu_level":
            return SwiGLUWithLevelAdaptation(dim, hidden_dim, dropout, max_level)
        else:
            return AdaptiveFractalFeedForward(
                dim=dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                max_level=max_level,
                use_level_adaptation=self.use_level_adaptation,
                use_feature_gating=self.use_feature_gating,
            )


ABLATION_CONFIGS = [
    FFNConfig(name="1_baseline_gelu", use_level_adaptation=False, use_feature_gating=False),
    FFNConfig(name="2_level_adaptation", use_level_adaptation=True, use_feature_gating=False),
    FFNConfig(name="3_feature_gating", use_level_adaptation=False, use_feature_gating=True),
    FFNConfig(name="4_full_adaptive", use_level_adaptation=True, use_feature_gating=True),
    FFNConfig(name="5_swiglu_only", ffn_class="swiglu"),
    FFNConfig(name="6_swiglu_level", ffn_class="swiglu_level"),
]


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    config_name: str
    param_count: int
    forward_time_ms: float
    backward_time_ms: float
    memory_mb: float
    activation_stats: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# 分析函数
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def analyze_activation_distribution(model: nn.Module, x: torch.Tensor, levels_info: torch.Tensor) -> Dict[str, float]:
    """分析激活函数权重分布（仅对 AdaptiveFractalFeedForward 有效）"""
    stats = {}
    
    if hasattr(model, 'activation_selector') and hasattr(model, 'use_feature_gating') and model.use_feature_gating:
        model.eval()
        with torch.no_grad():
            x_norm = model.norm(x)
            weights = model.activation_selector(x_norm)  # (B, S, 3)
            
            # 计算各激活函数的平均权重
            mean_weights = weights.mean(dim=(0, 1))
            stats["gelu_weight"] = mean_weights[0].item()
            stats["relu_weight"] = mean_weights[1].item()
            stats["swish_weight"] = mean_weights[2].item()
            
            # 计算熵（衡量分布均匀程度）
            entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean().item()
            stats["entropy"] = entropy
            stats["max_entropy"] = 1.0986  # log(3)
            stats["entropy_ratio"] = entropy / 1.0986
    
    return stats


def benchmark_ffn(
    config: FFNConfig,
    dim: int = 384,
    hidden_dim: int = 768,
    batch_size: int = 8,
    seq_len: int = 256,
    num_warmup: int = 5,
    num_runs: int = 20,
    device: str = "cpu",
) -> BenchmarkResult:
    """对单个 FFN 配置进行基准测试"""
    
    # 创建模型
    ffn = config.create_ffn(dim, hidden_dim).to(device)
    ffn.train()
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)
    levels_info = torch.randint(0, 50, (batch_size, seq_len, 8), device=device)
    
    # 参数量
    param_count = count_parameters(ffn)
    
    # 预热
    for _ in range(num_warmup):
        out = ffn(x, levels_info)
        loss = out.sum()
        loss.backward()
        x.grad = None
    
    # 前向传播计时
    if device == "cuda":
        torch.cuda.synchronize()
    
    forward_times = []
    for _ in range(num_runs):
        x.grad = None
        start = time.perf_counter()
        out = ffn(x, levels_info)
        if device == "cuda":
            torch.cuda.synchronize()
        forward_times.append((time.perf_counter() - start) * 1000)
    
    # 反向传播计时
    backward_times = []
    for _ in range(num_runs):
        x = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)
        out = ffn(x, levels_info)
        loss = out.sum()
        
        start = time.perf_counter()
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        backward_times.append((time.perf_counter() - start) * 1000)
    
    # 内存占用
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)
        out = ffn(x, levels_info)
        loss = out.sum()
        loss.backward()
        memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        memory_mb = 0.0
    
    # 激活分布分析
    x_eval = torch.randn(batch_size, seq_len, dim, device=device)
    activation_stats = analyze_activation_distribution(ffn, x_eval, levels_info)
    
    return BenchmarkResult(
        config_name=config.name,
        param_count=param_count,
        forward_time_ms=sum(forward_times) / len(forward_times),
        backward_time_ms=sum(backward_times) / len(backward_times),
        memory_mb=memory_mb,
        activation_stats=activation_stats,
    )


def run_ablation_study(
    dim: int = 384,
    hidden_dim: int = 768,
    batch_size: int = 8,
    seq_len: int = 256,
    device: str = "cpu",
) -> List[BenchmarkResult]:
    """运行完整消融实验"""
    
    print("=" * 70)
    print("FFN 变体消融实验")
    print("=" * 70)
    print(f"配置: dim={dim}, hidden_dim={hidden_dim}, batch={batch_size}, seq={seq_len}")
    print(f"设备: {device}")
    print("=" * 70)
    print()
    
    results = []
    baseline_params = None
    baseline_forward = None
    
    for config in ABLATION_CONFIGS:
        print(f"测试: {config.name}...")
        result = benchmark_ffn(
            config=config,
            dim=dim,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )
        results.append(result)
        
        if baseline_params is None:
            baseline_params = result.param_count
            baseline_forward = result.forward_time_ms
        
        # 计算相对变化
        param_delta = (result.param_count - baseline_params) / baseline_params * 100
        time_delta = (result.forward_time_ms - baseline_forward) / baseline_forward * 100
        
        print(f"  参数量: {result.param_count:,} ({param_delta:+.1f}% vs baseline)")
        print(f"  前向: {result.forward_time_ms:.2f}ms ({time_delta:+.1f}%)")
        print(f"  反向: {result.backward_time_ms:.2f}ms")
        
        if result.activation_stats:
            stats = result.activation_stats
            print(f"  激活权重: GELU={stats.get('gelu_weight', 0):.3f}, "
                  f"ReLU={stats.get('relu_weight', 0):.3f}, "
                  f"Swish={stats.get('swish_weight', 0):.3f}")
            print(f"  熵: {stats.get('entropy', 0):.4f} / {stats.get('max_entropy', 0):.4f} "
                  f"({stats.get('entropy_ratio', 0)*100:.1f}%)")
        print()
    
    return results


def print_summary_table(results: List[BenchmarkResult]) -> None:
    """打印汇总表格"""
    
    print("=" * 90)
    print("汇总表格")
    print("=" * 90)
    
    # 找到 baseline
    baseline = results[0]
    
    # 表头
    print(f"{'配置':<25} {'参数量':>12} {'相对变化':>10} {'前向(ms)':>10} {'反向(ms)':>10}")
    print("-" * 90)
    
    for r in results:
        param_delta = (r.param_count - baseline.param_count) / baseline.param_count * 100
        print(f"{r.config_name:<25} {r.param_count:>12,} {param_delta:>+9.1f}% "
              f"{r.forward_time_ms:>10.2f} {r.backward_time_ms:>10.2f}")
    
    print("-" * 90)
    print()
    
    # 激活分析（仅对有 feature_gating 的配置）
    print("=" * 90)
    print("Dynamic Activation 分析 (仅适用于 feature_gating 配置)")
    print("=" * 90)
    
    for r in results:
        if r.activation_stats:
            stats = r.activation_stats
            print(f"\n{r.config_name}:")
            print(f"  GELU 权重:  {stats.get('gelu_weight', 0):.4f}")
            print(f"  ReLU 权重:  {stats.get('relu_weight', 0):.4f}")
            print(f"  Swish 权重: {stats.get('swish_weight', 0):.4f}")
            print(f"  熵值:       {stats.get('entropy', 0):.4f} / {stats.get('max_entropy', 0):.4f}")
            print(f"  熵比例:     {stats.get('entropy_ratio', 0)*100:.1f}% (100% = 完全均匀)")
    
    print()


def analyze_feature_gating_value() -> None:
    """深入分析 feature_gating 的价值"""
    
    print("=" * 70)
    print("Feature Gating 深入分析")
    print("=" * 70)
    
    dim, hidden_dim = 384, 768
    batch_size, seq_len = 4, 128
    
    # 创建两个模型：有/无 feature_gating
    ffn_with_gate = AdaptiveFractalFeedForward(
        dim=dim, hidden_dim=hidden_dim,
        use_level_adaptation=False, use_feature_gating=True
    )
    
    ffn_no_gate = AdaptiveFractalFeedForward(
        dim=dim, hidden_dim=hidden_dim,
        use_level_adaptation=False, use_feature_gating=False
    )
    
    # 参数量对比
    params_with = count_parameters(ffn_with_gate)
    params_without = count_parameters(ffn_no_gate)
    
    print(f"\n参数量对比:")
    print(f"  无 feature_gating: {params_without:,}")
    print(f"  有 feature_gating: {params_with:,}")
    print(f"  增加: {params_with - params_without:,} ({(params_with-params_without)/params_without*100:.1f}%)")
    
    # 分析未训练状态下的激活分布
    print(f"\n未训练状态下的激活权重分布:")
    x = torch.randn(batch_size, seq_len, dim)
    levels = torch.randint(0, 50, (batch_size, seq_len, 8))
    
    ffn_with_gate.eval()
    with torch.no_grad():
        x_norm = ffn_with_gate.norm(x)
        weights = ffn_with_gate.activation_selector(x_norm)
        
        print(f"  GELU:  {weights[:, :, 0].mean():.4f} ± {weights[:, :, 0].std():.4f}")
        print(f"  ReLU:  {weights[:, :, 1].mean():.4f} ± {weights[:, :, 1].std():.4f}")
        print(f"  Swish: {weights[:, :, 2].mean():.4f} ± {weights[:, :, 2].std():.4f}")
        
        # 计算熵
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
        max_entropy = 1.0986  # log(3)
        print(f"\n  熵: {entropy:.4f} / {max_entropy:.4f} ({entropy/max_entropy*100:.1f}%)")
        print(f"  结论: {'接近均匀分布，未体现输入自适应性' if entropy/max_entropy > 0.95 else '存在一定偏好'}")
    
    # 分析 feature_gate 的门控效果
    print(f"\n特征门控分析:")
    with torch.no_grad():
        gates = ffn_with_gate.feature_gate(x_norm)  # (B, S, hidden_dim)
        
        print(f"  门控值范围: [{gates.min():.4f}, {gates.max():.4f}]")
        print(f"  门控均值:   {gates.mean():.4f} ± {gates.std():.4f}")
        
        # 门控的稀疏性（有多少门被激活 > 0.5）
        active_ratio = (gates > 0.5).float().mean()
        print(f"  激活比例 (>0.5): {active_ratio*100:.1f}%")
    
    print()


def main() -> None:
    """主函数"""
    
    # 检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 运行消融实验
    results = run_ablation_study(
        dim=384,
        hidden_dim=768,
        batch_size=8,
        seq_len=256,
        device=device,
    )
    
    # 打印汇总
    print_summary_table(results)
    
    # 深入分析 feature_gating
    analyze_feature_gating_value()
    
    # 最终建议
    print("=" * 70)
    print("消融实验结论")
    print("=" * 70)
    
    baseline = results[0]
    full = next(r for r in results if "full" in r.config_name)
    swiglu_level = next(r for r in results if "swiglu_level" in r.config_name)
    
    print(f"""
📊 参数效率分析:
  • Full 配置增加 {(full.param_count-baseline.param_count)/baseline.param_count*100:.1f}% 参数
  • SwiGLU + Level 相比 Full 减少 {(full.param_count-swiglu_level.param_count)/full.param_count*100:.1f}% 参数

⚡ 计算效率分析:
  • Feature Gating 导致计算开销增加
  • SwiGLU 的前向传播速度与 baseline 接近

🔬 Feature Gating 分析:
  • 未训练状态下激活权重接近均匀分布
  • 这表明需要充分训练才能体现其价值
  • 但也意味着：可能只是增加了参数，未必提供真正的自适应性

💡 建议:
  1. 如果优先参数效率：使用 SwiGLU + Level Adaptation
  2. 如果优先验证原始设计：保留 Full 配置进行训练验证
  3. 可以设计对比实验：训练后比较 Full vs SwiGLU+Level 的精度
""")


if __name__ == "__main__":
    main()
