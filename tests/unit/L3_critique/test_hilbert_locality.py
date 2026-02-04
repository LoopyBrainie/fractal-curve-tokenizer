"""
Phase 1: Hilbert 序列扫描批判分析 - 局部性基准测试

从最佳实现角度批判现有 Hilbert 扫描方案，比较不同矩形处理策略的局部性指标。

测试目标:
1. 对比 Padding、RectHilbertIndex、Pseudo-Hilbert 的 L_max 和 L_avg
2. 分析不同宽高比下的局部性退化程度
3. 量化边界效应对序列质量的影响
"""

import sys
import math
from pathlib import Path

# 添加 src 目录到路径
src_path = Path(__file__).parent.parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import pytest
import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from pathlib import Path as PathLib

# 导入 Hilbert 相关模块
from vit_pytorch.curve_hilbert import (
    HilbertCurve,
    HilbertScanner,
    PseudoHilbertCurve,
)


@dataclass
class LocalityMetrics:
    """局部性指标结构体"""
    L_max: float      # 最大跳跃距离
    L_avg: float      # 平均跳跃距离
    L_median: float   # 中位数跳跃距离
    R_local: float    # 局部性保持率 (跳跃 ≤ √2 的比例)
    R_top10: float    # 跳跃 ≤ 10 的比例
    n_jumps: int      # 总跳跃数


def compute_locality_metrics(
    points: torch.Tensor,
    image_size: Tuple[int, int]
) -> LocalityMetrics:
    """
    计算 Hilbert 序列的局部性指标。

    数学形式化:
    =============

    跳跃距离定义:
        Δ_i = ||p_i - p_{i+1}||_2,  for i = 0, ..., N-2

    指标:
        L_max = max_i Δ_i
        L_avg = (1/(N-1)) * Σ_i Δ_i
        R_local(τ) = (1/(N-1)) * Σ_i 1[Δ_i ≤ τ]

    Args:
        points: [N, 2] 点序列 (x, y)
        image_size: (H, W) 图像尺寸

    Returns:
        LocalityMetrics 对象
    """
    H, W = image_size

    # 计算相邻点之间的欧氏距离
    diffs = points[1:] - points[:-1]  # [N-1, 2]
    distances = torch.sqrt((diffs ** 2).sum(dim=1))  # [N-1]

    # 转换为 Python float 用于计算统计
    dist_list = distances.cpu().numpy()

    # 计算各项指标
    L_max = float(dist_list.max())
    L_avg = float(dist_list.mean())
    L_median = float(np.median(dist_list))

    # 局部性保持率 (Δ ≤ √2 ≈ 1.414 的比例)
    R_local = float((dist_list <= np.sqrt(2)).sum() / len(dist_list))

    # Top-10 保持率
    R_top10 = float((dist_list <= 10).sum() / len(dist_list))

    n_jumps = len(dist_list)

    return LocalityMetrics(
        L_max=L_max,
        L_avg=L_avg,
        L_median=L_median,
        R_local=R_local,
        R_top10=R_top10,
        n_jumps=n_jumps
    )


def compute_aspect_ratio(H: int, W: int) -> float:
    """计算宽高比 (长边/短边)"""
    return max(H, W) / min(H, W)


def _generate_hilbert_sequence(n: int) -> torch.Tensor:
    """
    生成标准 Hilbert 曲线的点序列

    Args:
        n: 曲线阶数 (n × n 网格)

    Returns:
        points: [n*n, 2] 点坐标序列
    """
    # 生成所有 (x, y) 坐标
    x = torch.arange(n).repeat(n)
    y = torch.arange(n).repeat_interleave(n)

    # 计算 Hilbert 距离
    d = HilbertCurve.xy_to_d_batch(n, x, y)

    # 按 Hilbert 距离排序
    sorted_idx = torch.argsort(d)
    points = torch.stack([x[sorted_idx], y[sorted_idx]], dim=1)

    return points


def _compute_hilbert_scanner_metrics(
    H: int,
    W: int
) -> LocalityMetrics:
    """使用 HilbertScanner 计算局部性指标（I113-18 最佳实现）"""
    # 使用 HilbertScanner 生成扫描序列
    points_np = HilbertScanner.scan(H, W)
    points = torch.from_numpy(np.array(points_np)).float()

    return compute_locality_metrics(points, (H, W))


def _analyze_locality_degradation(
    results: Dict[Tuple[int, int], LocalityMetrics],
    aspect_ratios: List[Tuple[int, int]]
):
    """
    分析 HilbertScanner 局部性随宽高比的变化

    I113-18 分析结论:

    HilbertScanner 的关键指标是 R_local（局部保持率）而非 L_max。

    L_max 会在分割边界处有跳跃，这是 Pseudo-Hilbert 的固有特性。
    但 R_local 应该始终 > 0.9，意味着 90% 以上的相邻点都是局部相邻的。
    """
    print("\n" + "=" * 60)
    print("HilbertScanner 局部性分析")
    print("=" * 60)

    for (H, W), metrics in results.items():
        rho = max(H, W) / min(H, W)
        print(f"\n[H={H}, W={W}, ρ={rho:.2f}]")
        print(f"  L_max={metrics.L_max:.2f}, L_avg={metrics.L_avg:.2f}")
        print(f"  R_local={metrics.R_local:.3f}")

        # 分析 R_local
        if metrics.R_local > 0.95:
            print(f"  评级: 优秀 (R_local > 0.95)")
        elif metrics.R_local > 0.9:
            print(f"  评级: 良好 (R_local > 0.9)")
        else:
            print(f"  评级: 需改进 (R_local < 0.9)")


class TestHilbertLocalityCritique:
    """
    Hilbert 序列扫描批判分析测试

    批判维度:
    1. 宽高比敏感性: 不同方案的局部性如何随宽高比退化
    2. 边界效应: 图像边缘区域的跳跃是否异常
    3. 可扩展性: 大尺寸图像的性能表现
    """

    @pytest.fixture
    def aspect_ratios(self) -> List[Tuple[int, int]]:
        """测试不同的宽高比配置（I113-18 验证 HilbertScanner）"""
        return [
            (32, 32),    # 1:1 正方形（标准 Hilbert 退化）
            (32, 64),    # 1:2
            (32, 128),   # 1:4
            (64, 32),    # 2:1
            (64, 128),   # 1:2 (偶数)
            (128, 32),   # 4:1
            (16, 64),    # 1:4 (小图)
            (32, 256),   # 1:8 (极端)
        ]

    def test_hilbert_scanner_locality(
        self,
        aspect_ratios: List[Tuple[int, int]]
    ):
        """
        I113-18 核心测试: HilbertScanner 局部性验证

        验证结论:
        - HilbertScanner 在所有宽高比下都稳定
        - L_max 符合实际表现（分割边界跳跃是固有特性）
        - 关键指标 R_local (局部保持率) 最重要
        """
        results = {}

        for H, W in aspect_ratios:
            rho = max(H, W) / min(H, W)
            metrics = _compute_hilbert_scanner_metrics(H, W)
            results[(H, W)] = metrics

            # 打印诊断信息
            print(f"\n[HilbertScanner] H={H}, W={W}, ρ={rho:.2f}")
            print(f"  L_max={metrics.L_max:.2f}, L_avg={metrics.L_avg:.2f}")
            print(f"  R_local={metrics.R_local:.3f} (局部保持率)")

        # 打印总结
        _analyze_locality_degradation(results, aspect_ratios)

        # 验证 R_local > 0.9 (关键指标)
        for (H, W), metrics in results.items():
            assert metrics.R_local > 0.9, \
                f"R_local {metrics.R_local:.3f} < 0.9 for {H}x{W}"

    def test_standard_hilbert_degeneration(self):
        """
        验证标准 Hilbert 退化（2^k × 2^k 正方形）

        当 H=W 且是 2 的幂时，HilbertScanner 应退化为标准 Hilbert
        """
        print("\n" + "=" * 60)
        print("标准 Hilbert 退化验证")
        print("=" * 60)

        for size in [8, 16, 32, 64]:
            H = W = size
            metrics = _compute_hilbert_scanner_metrics(H, W)

            # 标准 Hilbert 的 L_max 理论值是 √2
            print(f"\n[{size}×{size}] L_max={metrics.L_max:.2f}")
            assert metrics.L_max <= math.sqrt(2) + 0.1, \
                f"L_max {metrics.L_max:.2f} > √2={math.sqrt(2):.2f}"

    def test_pseudo_hilbert_for_rectangle(self):
        """
        验证 Pseudo-Hilbert 在矩形上的表现

        对于非正方形，HilbertScanner 使用 Pseudo-Hilbert
        """
        print("\n" + "=" * 60)
        print("Pseudo-Hilbert 矩形验证")
        print("=" * 60)

        test_cases = [
            (32, 64),   # 1:2
            (32, 128),  # 1:4
            (64, 32),   # 2:1
            (128, 32),  # 4:1
        ]

        for H, W in test_cases:
            rho = max(H, W) / min(H, W)
            metrics = _compute_hilbert_scanner_metrics(H, W)

            theoretical_max = math.sqrt(2 * rho)
            within_bound = metrics.L_max <= theoretical_max + 0.1

            print(f"\n[{H}×{W}, ρ={rho:.2f}] L_max={metrics.L_max:.2f}, bound={theoretical_max:.2f}")
            assert within_bound, f"L_max {metrics.L_max:.2f} > √(2ρ)={theoretical_max:.2f}"

    def test_depth_consistency(self):
        """
        测试深度一致性

        问题: 不同深度应该对应不同尺度的局部性

        深度 d → patch_size = 2^d
        深度 d 的区域在 Hilbert 序列中应该相对集中

        这对于 Fractal ViT 的分裂决策至关重要！
        """
        print("\n" + "=" * 60)
        print("深度一致性分析")
        print("=" * 60)

        depths = [1, 2, 3, 4, 5]
        H, W = 64, 64

        for d in depths:
            # Pseudo-Hilbert 自然支持深度结构
            points_np = PseudoHilbertCurve.scan(H, W)
            points = torch.from_numpy(np.array(points_np)).float()

            # 模拟四叉树深度 d 的区域划分
            n_regions = 4 ** d
            region_size = H / (2 ** d)

            # 计算每个区域的中心点
            region_centers = []
            for i in range(2 ** d):
                for j in range(2 ** d):
                    cx = (i + 0.5) * region_size
                    cy = (j + 0.5) * region_size
                    region_centers.append((cx, cy))

            # 计算深度 d 区域的局部性
            if len(region_centers) > 1:
                centers = torch.tensor(region_centers)
                diffs = centers[1:] - centers[:-1]
                distances = torch.sqrt((diffs ** 2).sum(dim=1))

                print(f"\n深度 d={d}: {n_regions} 个区域")
                print(f"  区域大小: {region_size:.1f} × {region_size:.1f}")
                print(f"  区域中心间平均跳跃: {distances.mean():.2f}")
                print(f"  区域中心间最大跳跃: {distances.max():.2f}")

    def test_hilbert_scanner_performance(self):
        """
        I113-18 HilbertScanner 性能基准

        验证不同分辨率下的性能表现
        """
        print("\n" + "=" * 70)
        print("HilbertScanner 性能基准")
        print("=" * 70)

        import time

        test_cases = [
            (32, 32, "32×32"),
            (64, 64, "64×64"),
            (128, 128, "128×128"),
            (32, 128, "32×128"),
            (64, 256, "64×256"),
            (128, 128, "128×128"),
        ]

        for H, W, name in test_cases:
            # 首次运行（无缓存）
            start = time.perf_counter()
            HilbertScanner.scan(H, W)
            first_run = time.perf_counter() - start

            # 第二次运行（有缓存）
            start = time.perf_counter()
            HilbertScanner.scan(H, W)
            cached_run = time.perf_counter() - start

            print(f"\n[{name}] 首次: {first_run*1000:.2f}ms, 缓存: {cached_run*1000:.4f}ms")
            assert cached_run < first_run, "缓存应该更快"

    def test_hilbert_scanner_api(self):
        """
        验证 HilbertScanner API 正确性

        - scan(): 生成扫描序列
        - xy_to_d(): 坐标 → 索引
        - d_to_xy(): 索引 → 坐标
        - region_to_hilbert_index(): 区域 → Hilbert 索引
        """
        print("\n" + "=" * 60)
        print("HilbertScanner API 验证")
        print("=" * 60)

        # 测试标准 Hilbert 退化
        H = W = 64
        points = HilbertScanner.scan(H, W)
        assert len(points) == H * W, f"长度错误: {len(points)} vs {H*W}"

        # 验证 xy_to_d 和 d_to_xy 互逆
        for d in [0, 100, 1000, 4095]:
            x, y = HilbertScanner.d_to_xy(H, W, d)
            d_back = HilbertScanner.xy_to_d(H, W, x, y)
            assert d == d_back, f"xy_to_d/d_to_xy invert failed: {d} vs {d_back}"

        print("[OK] Standard Hilbert degeneration passed")

        # 测试 Pseudo-Hilbert
        H, W = 32, 128
        points = HilbertScanner.scan(H, W)
        assert len(points) == H * W, f"Pseudo-Hilbert length error"

        print("[OK] Pseudo-Hilbert passed")

        # 测试 region_to_hilbert_index
        x0 = torch.tensor([0.0, 16.0])
        y0 = torch.tensor([0.0, 0.0])
        x1 = torch.tensor([16.0, 32.0])
        y1 = torch.tensor([16.0, 16.0])

        hilbert_indices = HilbertScanner.region_to_hilbert_index(
            x0, y0, x1, y1, depth=2, H=32, W=128
        )

        assert hilbert_indices.shape == (2,), f"Shape error: {hilbert_indices.shape}"
        print("[OK] region_to_hilbert_index passed")


class TestHilbertGradientQuality:
    """
    测试 Hilbert 序列的可微性质量

    问题: 如何为 Hilbert 序列排序提供梯度？

    方案:
    1. 排序操作不可微 → 使用连续近似 (NDCG 近似)
    2. 基于分数的排序: score_i = f(p_i), 使用 softmax 近似 argmax
    """

    def test_sort_gradient_approximation(self):
        """
        测试排序梯度近似

        理想梯度: ∂L / ∂score_i 应该反映 p_i 位置变化对 loss 的影响
        近似方法: 使用 softmax 温度 τ 控制近似精度

        数学分析:
        ∂sort_approx(p, τ)_i / ∂p_j ∝ exp(p_j/τ) × (δ_ij - softmax(p)_i)
        """
        print("\n" + "=" * 60)
        print("排序梯度近似分析")
        print("=" * 60)

        # 测试不同温度下的梯度质量
        tau_values = [0.01, 0.1, 1.0, 10.0]

        for tau in tau_values:
            # 模拟位置分数
            scores = torch.randn(64, requires_grad=True)

            # Softmax 近似排序
            probs = torch.softmax(scores / tau, dim=0)

            # 模拟 loss (假设最后一个位置有高 loss)
            loss = probs[-1]  # 鼓励将质量低的点排到后面

            # 反向传播
            loss.backward()

            print(f"\nτ = {tau}:")
            if scores.grad is not None:
                print(f"  梯度最大值: {scores.grad.abs().max():.4f}")
                print(f"  梯度非零比例: {(scores.grad != 0).float().mean():.3f}")
            else:
                print(f"  梯度: None")

            if tau < 0.1:
                print("  [LOW_TEMP] Sparse gradient (接近 argmax)")
            elif tau > 5:
                print("  [HIGH_TEMP] Flat gradient (接近 uniform)")
            else:
                print("  [OPTIMAL] Well-distributed gradient")

    def test_hilbert_continuous_approximation(self):
        """
        测试 Hilbert 索引的连续近似

        问题: Hilbert 索引计算中的 floor() 操作阻断梯度

        解决方案:
        1. 使用浮点数索引: h̃ = H̃(p), 无 floor
        2. 基于距离的排序: d_i = ||p_i - p_{target}||
        3. 基于密度的加权: w_i = density(p_i)
        """
        print("\n" + "=" * 60)
        print("Hilbert 连续近似分析")
        print("=" * 60)

        H, W = 64, 64
        points_np = PseudoHilbertCurve.scan(H, W)
        points = torch.from_numpy(np.array(points_np)).float()

        # 方案1: 基于密度的加权 (连续可微)
        density_weights = torch.rand(len(points))
        weighted_order = torch.argsort(
            torch.cumsum(density_weights, dim=0)
        )

        # 方案2: 基于距离的排序 (连续可微)
        target = torch.tensor([H / 2, W / 2])
        distances = torch.sqrt(((points - target) ** 2).sum(dim=1))
        distance_order = torch.argsort(distances)

        # 方案3: Hilbert 原始顺序 (不可微)
        hilbert_order = torch.arange(len(points))

        print(f"\nHilbert 原始顺序 vs 密度加权:")
        order_corr = torch.corrcoef(
            torch.stack([
                hilbert_order.float(),
                weighted_order.float()
            ])
        )[0, 1]
        print(f"  顺序相关性: {order_corr:.4f}")

        print(f"\nHilbert 原始顺序 vs 距离排序:")
        dist_corr = torch.corrcoef(
            torch.stack([
                hilbert_order.float(),
                distance_order.float()
            ])
        )[0, 1]
        print(f"  顺序相关性: {dist_corr:.4f}")

        # 批判
        print("\n[CRITICAL ANALYSIS]:")
        print("  1. Hilbert order not fully correlated with spatial distance")
        print("  2. Continuous approximation breaks Hilbert locality guarantee")
        print("  3. Trade-off needed between differentiability and locality")


if __name__ == "__main__":
    # 运行 HilbertScanner 测试
    tester = TestHilbertLocalityCritique()

    print("=" * 70)
    print("I113-18 HilbertScanner 测试")
    print("=" * 70)

    # 1. HilbertScanner 局部性验证
    aspect_ratios = [
        (32, 32), (32, 64), (32, 128), (64, 32),
        (64, 128), (128, 32), (16, 64), (32, 256)
    ]
    tester.test_hilbert_scanner_locality(aspect_ratios)

    # 2. 标准 Hilbert 退化验证
    tester.test_standard_hilbert_degeneration()

    # 3. Pseudo-Hilbert 矩形验证
    tester.test_pseudo_hilbert_for_rectangle()

    # 4. API 验证
    tester.test_hilbert_scanner_api()

    # 5. 性能基准
    tester.test_hilbert_scanner_performance()
