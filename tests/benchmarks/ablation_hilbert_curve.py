# -*- coding: utf-8 -*-
"""Hilbert Curve ViT 消融实验

验证 Hilbert Curve ViT 的三个核心假设：
1. H1: Hilbert 排序优于光栅排序
2. H2: LCA Bias 优于 Low-Rank Bias  
3. H3: 多尺度自适应优于固定尺度

实验矩阵：
┌─────────┬──────────┬────────────┬─────────────┬─────────────────┐
│ 实验    │ Hilbert  │ Bias Mode  │ 尺度选择    │ 预期对比        │
├─────────┼──────────┼────────────┼─────────────┼─────────────────┤
│ E1-Base │ Raster   │ None       │ Fixed 16×16 │ Baseline        │
│ E2-Hilb │ Hilbert  │ None       │ Fixed       │ H1 验证         │
│ E3-LR   │ Hilbert  │ low_rank   │ Fixed       │ Bias 效果       │
│ E4-LCA  │ Hilbert  │ lca        │ Fixed       │ H2 验证         │
│ E5-Adap │ Hilbert  │ lca        │ Adaptive    │ H3 验证         │
└─────────┴──────────┴────────────┴─────────────┴─────────────────┘

Usage:
    # 快速测试 (3 epochs, CIFAR-10 子集)
    uv run python tests/benchmarks/ablation_hilbert_curve.py --quick
    
    # 完整消融实验 (20 epochs)
    uv run python tests/benchmarks/ablation_hilbert_curve.py --epochs 20
    
    # 只运行特定实验
    uv run python tests/benchmarks/ablation_hilbert_curve.py --experiments E1-Base E4-LCA
    
    # 保存结果到文件
    uv run python tests/benchmarks/ablation_hilbert_curve.py --output results.json

数学形式化
==========

假设 H1 (Hilbert 局部性):
    Hilbert 曲线满足 ‖H(d₁) - H(d₂)‖₂ ≤ C·|d₁ - d₂|^(1/2)
    因此在 1D 序列中相邻的 token 在 2D 空间中也相近，
    这有助于注意力机制捕捉局部特征。

假设 H2 (LCA 距离等价性):
    LCA(i,j) = ℓ  ⟹  ‖pos_i - pos_j‖_∞ ≤ N/2^ℓ
    LCA 深度直接编码空间距离，无需学习，参数量从 ~50K 降至 ~36。

假设 H3 (自适应 Token 分配):
    复杂区域应使用更多 token (细粒度 patch)，
    简单区域应使用更少 token (粗粒度 patch)，
    从而在固定计算预算下最大化信息保留。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

# Handle import paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vit_pytorch import FractalCurveViT

# Optional: torchvision for CIFAR-10
try:
    from torchvision import datasets, transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


# ============================================================================
# 实验配置
# ============================================================================

@dataclass
class ExperimentConfig:
    """单个消融实验的配置"""
    name: str
    description: str
    
    # 核心消融变量
    use_hilbert_order: bool = True
    bias_mode: Optional[str] = None  # None, 'low_rank', 'lca'
    use_hilbert_bias: bool = True
    adaptive_scale: bool = False  # True = streaming_v2, False = fixed patch
    
    # 模型超参数 (保持一致以控制变量)
    dim: int = 64
    depth: int = 4
    heads: int = 4
    mlp_dim: int = 128
    
    # 训练超参数
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    
    def __post_init__(self):
        # 验证配置
        if self.bias_mode is not None and self.bias_mode not in ['low_rank', 'lca', 'hierarchical']:
            raise ValueError(f"Unknown bias_mode: {self.bias_mode}")


@dataclass
class ExperimentResult:
    """实验结果"""
    config_name: str
    
    # 训练指标
    train_losses: List[float] = field(default_factory=list)
    train_accs: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_accs: List[float] = field(default_factory=list)
    
    # 最终指标
    best_val_acc: float = 0.0
    best_epoch: int = 0
    final_val_acc: float = 0.0
    
    # 计算指标
    total_params: int = 0
    bias_params: int = 0
    avg_epoch_time: float = 0.0
    throughput: float = 0.0  # images/sec
    
    # Token 分析 (仅 adaptive 模式)
    avg_tokens: float = 0.0
    token_std: float = 0.0
    complexity_correlation: float = 0.0


# ============================================================================
# 预定义实验配置
# ============================================================================

ABLATION_CONFIGS = {
    "E1-Base": ExperimentConfig(
        name="E1-Base",
        description="Baseline: Raster order, no Hilbert bias, fixed 16x16 patches",
        use_hilbert_order=False,
        bias_mode=None,
        use_hilbert_bias=False,
        adaptive_scale=False,
    ),
    "E2-Hilbert": ExperimentConfig(
        name="E2-Hilbert",
        description="Hilbert order, no bias (验证 H1: Hilbert 排序价值)",
        use_hilbert_order=True,
        bias_mode=None,
        use_hilbert_bias=False,
        adaptive_scale=False,
    ),
    "E3-LowRank": ExperimentConfig(
        name="E3-LowRank",
        description="Hilbert order + Low-Rank bias",
        use_hilbert_order=True,
        bias_mode='low_rank',
        use_hilbert_bias=True,
        adaptive_scale=False,
    ),
    "E4-LCA": ExperimentConfig(
        name="E4-LCA",
        description="Hilbert order + LCA bias (验证 H2: LCA vs Low-Rank)",
        use_hilbert_order=True,
        bias_mode='lca',
        use_hilbert_bias=True,
        adaptive_scale=False,
    ),
    "E5-Adaptive": ExperimentConfig(
        name="E5-Adaptive",
        description="Hilbert + LCA + Adaptive scale (验证 H3: 多尺度自适应)",
        use_hilbert_order=True,
        bias_mode='lca',
        use_hilbert_bias=True,
        adaptive_scale=True,
    ),
}


# ============================================================================
# 数据加载
# ============================================================================

def get_cifar10_loaders(
    batch_size: int = 64,
    subset_size: Optional[int] = None,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, int]:
    """获取 CIFAR-10 数据加载器
    
    Args:
        batch_size: 批次大小
        subset_size: 子集大小 (用于快速测试)
        num_workers: 数据加载线程数
    
    Returns:
        train_loader, val_loader, num_classes
    """
    if not HAS_TORCHVISION:
        raise ImportError("请安装 torchvision: pip install torchvision")
    
    # CIFAR-10 标准化
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    data_root = PROJECT_ROOT / "workspace" / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    
    train_dataset = datasets.CIFAR10(
        root=str(data_root),
        train=True,
        download=True,
        transform=train_transform,
    )
    
    val_dataset = datasets.CIFAR10(
        root=str(data_root),
        train=False,
        download=True,
        transform=val_transform,
    )
    
    # 创建子集
    if subset_size is not None:
        train_indices = torch.randperm(len(train_dataset))[:subset_size].tolist()
        val_indices = torch.randperm(len(val_dataset))[:subset_size // 5].tolist()
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, 10


# ============================================================================
# 模型创建
# ============================================================================

def create_model(config: ExperimentConfig, num_classes: int = 10) -> nn.Module:
    """根据配置创建模型
    
    Args:
        config: 实验配置
        num_classes: 分类类别数
    
    Returns:
        配置好的 FractalCurveViT 模型
    """
    # 确定 tokenizer 类型
    if config.adaptive_scale:
        tokenizer_type = 'streaming_v2'
    else:
        tokenizer_type = 'streaming'  # Fixed scale streaming
    
    # 创建模型
    model = FractalCurveViT(
        image_size=32,  # CIFAR-10
        num_classes=num_classes,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        mlp_dim=config.mlp_dim,
        channels=3,
        # Tokenizer 配置
        tokenizer_type=tokenizer_type,
        min_patch_size=(4, 4),
        max_level=6,
        # Hilbert 排序
        use_hilbert_order=config.use_hilbert_order,
        # 注意力偏置配置
        use_hilbert_bias=config.use_hilbert_bias,
        bias_mode=config.bias_mode if config.use_hilbert_bias else 'low_rank',
        # FFN 配置
        ffn_type='swiglu_level',
        # 正则化
        dropout=0.1,
        drop_path_rate=0.1,
    )
    
    return model


def count_bias_params(model: nn.Module) -> int:
    """统计 Hilbert Bias 相关参数量"""
    bias_params = 0
    for name, param in model.named_parameters():
        if 'hilbert_bias' in name or 'lca_embedding' in name:
            bias_params += param.numel()
    return bias_params


# ============================================================================
# 训练与评估
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def analyze_tokens(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """分析 Token 分布 (仅用于 adaptive 模式)"""
    model.eval()
    token_counts = []
    complexities = []
    
    for images, _ in loader:
        images = images.to(device)
        
        # 获取 token 数量
        _, aux_info = model(images, return_aux_info=True)
        if 'num_tokens' in aux_info:
            token_counts.extend(aux_info['num_tokens'])
        
        # 估计图像复杂度 (使用方差作为代理)
        batch_complexity = images.view(images.size(0), -1).var(dim=1)
        complexities.extend(batch_complexity.cpu().tolist())
    
    if not token_counts:
        return {'avg_tokens': 0, 'token_std': 0, 'complexity_correlation': 0}
    
    token_counts = np.array(token_counts)
    complexities = np.array(complexities)
    
    # 计算相关性
    if len(token_counts) > 1 and token_counts.std() > 0:
        correlation = np.corrcoef(token_counts, complexities)[0, 1]
    else:
        correlation = 0.0
    
    return {
        'avg_tokens': float(token_counts.mean()),
        'token_std': float(token_counts.std()),
        'complexity_correlation': float(correlation) if not np.isnan(correlation) else 0.0,
    }


def run_experiment(
    config: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    verbose: bool = True,
) -> ExperimentResult:
    """运行单个消融实验"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"实验: {config.name}")
        print(f"描述: {config.description}")
        print(f"{'='*60}")
    
    # 创建模型
    model = create_model(config, num_classes)
    model = model.to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    bias_params = count_bias_params(model)
    
    if verbose:
        print(f"总参数量: {total_params:,}")
        print(f"Bias 参数量: {bias_params:,}")
    
    # 优化器和调度器
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # 训练结果
    result = ExperimentResult(config_name=config.name)
    result.total_params = total_params
    result.bias_params = bias_params
    
    epoch_times = []
    best_val_acc = 0.0
    best_epoch = 0
    
    # 训练循环
    for epoch in range(config.epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        result.train_losses.append(train_loss)
        result.train_accs.append(train_acc)
        result.val_losses.append(val_loss)
        result.val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
        
        if verbose:
            print(f"Epoch {epoch+1}/{config.epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.1f}%, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.1f}% "
                  f"[{epoch_time:.1f}s]")
    
    result.best_val_acc = best_val_acc
    result.best_epoch = best_epoch
    result.final_val_acc = result.val_accs[-1] if result.val_accs else 0.0
    result.avg_epoch_time = np.mean(epoch_times)
    
    # 计算吞吐量
    total_images = len(train_loader.dataset) * config.epochs
    total_time = sum(epoch_times)
    result.throughput = total_images / total_time if total_time > 0 else 0.0
    
    # Token 分析 (仅 adaptive 模式)
    if config.adaptive_scale:
        token_stats = analyze_tokens(model, val_loader, device)
        result.avg_tokens = token_stats['avg_tokens']
        result.token_std = token_stats['token_std']
        result.complexity_correlation = token_stats['complexity_correlation']
    
    if verbose:
        print(f"\n最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch+1})")
        print(f"平均 Epoch 时间: {result.avg_epoch_time:.1f}s")
        print(f"吞吐量: {result.throughput:.1f} images/sec")
    
    return result


# ============================================================================
# 结果分析
# ============================================================================

def analyze_results(results: Dict[str, ExperimentResult]) -> Dict[str, Any]:
    """分析消融实验结果"""
    analysis = {
        'summary': {},
        'hypothesis_tests': {},
        'recommendations': [],
    }
    
    # 提取关键指标
    for name, result in results.items():
        analysis['summary'][name] = {
            'best_val_acc': result.best_val_acc,
            'final_val_acc': result.final_val_acc,
            'total_params': result.total_params,
            'bias_params': result.bias_params,
            'throughput': result.throughput,
        }
    
    # 假设检验
    if 'E1-Base' in results and 'E2-Hilbert' in results:
        diff = results['E2-Hilbert'].best_val_acc - results['E1-Base'].best_val_acc
        analysis['hypothesis_tests']['H1_hilbert_ordering'] = {
            'baseline': results['E1-Base'].best_val_acc,
            'hilbert': results['E2-Hilbert'].best_val_acc,
            'improvement': diff,
            'significant': abs(diff) > 1.0,  # > 1% 视为显著
            'conclusion': 'Hilbert 排序有效' if diff > 1.0 else 
                         'Hilbert 排序无效' if diff < -1.0 else '无显著差异',
        }
    
    if 'E3-LowRank' in results and 'E4-LCA' in results:
        diff = results['E4-LCA'].best_val_acc - results['E3-LowRank'].best_val_acc
        param_reduction = (results['E3-LowRank'].bias_params - results['E4-LCA'].bias_params) / max(results['E3-LowRank'].bias_params, 1)
        analysis['hypothesis_tests']['H2_lca_vs_lowrank'] = {
            'low_rank_acc': results['E3-LowRank'].best_val_acc,
            'lca_acc': results['E4-LCA'].best_val_acc,
            'improvement': diff,
            'param_reduction': param_reduction * 100,
            'low_rank_params': results['E3-LowRank'].bias_params,
            'lca_params': results['E4-LCA'].bias_params,
            'conclusion': f'LCA 参数减少 {param_reduction*100:.1f}%，精度{"提升" if diff > 0 else "下降"} {abs(diff):.2f}%',
        }
    
    if 'E4-LCA' in results and 'E5-Adaptive' in results:
        diff = results['E5-Adaptive'].best_val_acc - results['E4-LCA'].best_val_acc
        analysis['hypothesis_tests']['H3_adaptive_scale'] = {
            'fixed_acc': results['E4-LCA'].best_val_acc,
            'adaptive_acc': results['E5-Adaptive'].best_val_acc,
            'improvement': diff,
            'complexity_correlation': results['E5-Adaptive'].complexity_correlation,
            'conclusion': '自适应有效' if diff > 1.0 and results['E5-Adaptive'].complexity_correlation > 0.3 else
                         '自适应无效或相关性弱',
        }
    
    # 生成建议
    if 'H1_hilbert_ordering' in analysis['hypothesis_tests']:
        h1 = analysis['hypothesis_tests']['H1_hilbert_ordering']
        if h1['improvement'] > 1.0:
            analysis['recommendations'].append(
                f"✅ 保留 Hilbert 排序：提升 {h1['improvement']:.2f}% 验证准确率"
            )
        else:
            analysis['recommendations'].append(
                f"⚠️ Hilbert 排序收益有限 ({h1['improvement']:.2f}%)，考虑简化"
            )
    
    if 'H2_lca_vs_lowrank' in analysis['hypothesis_tests']:
        h2 = analysis['hypothesis_tests']['H2_lca_vs_lowrank']
        if h2['improvement'] >= -0.5:  # LCA 精度持平或更好
            analysis['recommendations'].append(
                f"✅ 使用 LCA Bias：参数减少 {h2['param_reduction']:.1f}%，精度变化 {h2['improvement']:+.2f}%"
            )
        else:
            analysis['recommendations'].append(
                f"⚠️ LCA Bias 精度下降 {abs(h2['improvement']):.2f}%，考虑保留 Low-Rank"
            )
    
    return analysis


def print_analysis(analysis: Dict[str, Any]) -> None:
    """打印分析结果"""
    print("\n" + "="*70)
    print("消融实验分析报告")
    print("="*70)
    
    # 摘要表格
    print("\n📊 实验结果摘要:")
    print("-"*70)
    print(f"{'实验':<15} {'最佳验证Acc':>12} {'总参数':>12} {'Bias参数':>10} {'吞吐量':>12}")
    print("-"*70)
    for name, summary in analysis['summary'].items():
        print(f"{name:<15} {summary['best_val_acc']:>11.2f}% {summary['total_params']:>12,} "
              f"{summary['bias_params']:>10,} {summary['throughput']:>10.1f}/s")
    print("-"*70)
    
    # 假设检验
    print("\n🔬 假设检验结果:")
    for test_name, result in analysis['hypothesis_tests'].items():
        print(f"\n  {test_name}:")
        print(f"    结论: {result['conclusion']}")
    
    # 建议
    print("\n💡 优化建议:")
    for rec in analysis['recommendations']:
        print(f"  {rec}")
    
    print("\n" + "="*70)


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hilbert Curve ViT 消融实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速测试
  uv run python tests/benchmarks/ablation_hilbert_curve.py --quick
  
  # 完整实验
  uv run python tests/benchmarks/ablation_hilbert_curve.py --epochs 20
  
  # 只运行特定实验
  uv run python tests/benchmarks/ablation_hilbert_curve.py --experiments E1-Base E4-LCA
        """
    )
    
    parser.add_argument(
        '--quick', action='store_true',
        help='快速测试模式 (3 epochs, 5000 样本子集)'
    )
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='训练 epochs 数 (默认: 10)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='批次大小 (默认: 64)'
    )
    parser.add_argument(
        '--experiments', nargs='+', default=None,
        help='要运行的实验名称 (默认: 全部)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='结果输出文件路径 (JSON)'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='设备 (cuda/cpu/auto)'
    )
    
    args = parser.parse_args()
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 快速测试模式
    if args.quick:
        args.epochs = 3
        subset_size = 5000
        print("快速测试模式: 3 epochs, 5000 样本子集")
    else:
        subset_size = None
    
    # 加载数据
    print("\n加载 CIFAR-10 数据...")
    train_loader, val_loader, num_classes = get_cifar10_loaders(
        batch_size=args.batch_size,
        subset_size=subset_size,
    )
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    
    # 选择实验
    if args.experiments:
        configs = {name: ABLATION_CONFIGS[name] for name in args.experiments 
                   if name in ABLATION_CONFIGS}
    else:
        configs = ABLATION_CONFIGS.copy()
    
    # 更新 epochs
    for config in configs.values():
        config.epochs = args.epochs
        config.batch_size = args.batch_size
    
    # 运行实验
    results = {}
    for name, config in configs.items():
        try:
            result = run_experiment(
                config, train_loader, val_loader, num_classes, device
            )
            results[name] = result
        except Exception as e:
            print(f"\n❌ 实验 {name} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 分析结果
    if results:
        analysis = analyze_results(results)
        print_analysis(analysis)
        
        # 保存结果
        if args.output:
            output_data = {
                'results': {name: asdict(r) for name, r in results.items()},
                'analysis': analysis,
            }
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存到: {args.output}")
    
    return results


if __name__ == '__main__':
    main()
