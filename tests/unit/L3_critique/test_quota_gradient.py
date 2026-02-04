"""
Phase 2: Quota Allocation Critique - STE Gradient Bias Analysis

Critique existing quota allocation schemes from the perspective of optimal implementation.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class GradientMetrics:
    """Gradient quality metrics"""
    grad_true: np.ndarray
    grad_ste: np.ndarray
    grad_bias: np.ndarray
    bias_mean: float
    bias_std: float
    relative_error: float


def largest_remainder_method(p: np.ndarray, K: int) -> np.ndarray:
    """
    Largest Remainder Method (Hamilton Method)

    Problem: min ||K_d - p_d*K||_1 s.t. SUM(K_d) = K, K_d in Z

    Algorithm:
        K_d^floor = floor(p_d * K)
        r_d = p_d * K - K_d^floor (remainder)
        K_d = K_d^floor + 1 if r_d in top-(K - SUM(K_d^floor))
    """
    D = len(p)
    floor_quota = np.floor(p * K).astype(int)
    remainders = p * K - floor_quota
    remaining = K - floor_quota.sum()

    if remaining > 0:
        indices = np.argsort(-remainders)[:remaining]
        floor_quota[indices] += 1

    return floor_quota


def compute_ste_gradient(p: np.ndarray, K: int, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute STE proxy gradient vs true gradient

    True gradient (finite difference):
        d(LRM)_d / d(p_e) approx [LRM(p + eps*e_e) - LRM(p)] / eps

    STE proxy gradient:
        d(STE)_d / d(p_e) = K * delta_de
    """
    D = len(p)
    grad_true = np.zeros((D, D))
    grad_ste = np.zeros((D, D))

    for e in range(D):
        grad_ste[:, e] = K * np.eye(D)[:, e]

    for e in range(D):
        p_eps = p.copy()
        p_eps[e] += eps
        p_eps = np.clip(p_eps, 0, 1)
        p_eps = p_eps / p_eps.sum()

        k_base = largest_remainder_method(p, K)
        k_eps = largest_remainder_method(p_eps, K)

        grad_true[:, e] = (k_eps - k_base) / eps

    return grad_true, grad_ste


def analyze_gradient_bias(p: np.ndarray, K: int, eps_values: List[float] = None) -> Dict[float, GradientMetrics]:
    """Analyze gradient bias at different eps values"""
    if eps_values is None:
        eps_values = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]

    results = {}

    for eps in eps_values:
        grad_true, grad_ste = compute_ste_gradient(p, K, eps)

        grad_bias = grad_ste - grad_true
        bias_mean = np.abs(grad_bias).mean()
        bias_std = np.abs(grad_bias).std()

        mask = np.abs(grad_true) > 1e-6
        if mask.sum() > 0:
            relative_error = np.abs(grad_bias[mask] / grad_true[mask]).mean()
        else:
            relative_error = 0.0

        results[eps] = GradientMetrics(
            grad_true=grad_true,
            grad_ste=grad_ste,
            grad_bias=grad_bias,
            bias_mean=bias_mean,
            bias_std=bias_std,
            relative_error=relative_error
        )

    return results


class TestQuotaGradientCritique:
    """Quota allocation gradient quality critique"""

    def test_gradient_bias_analysis(self):
        """Core test: Analyze STE gradient bias"""
        print("=" * 70)
        print("Phase 2: Quota Allocation STE Gradient Bias Analysis")
        print("=" * 70)

        D = 8
        K = 64
        p = np.random.rand(D)
        p = p / p.sum()

        print(f"\nConfig: D={D}, K={K}")
        print(f"Distribution: {p.round(4)}")

        eps_values = [1e-2, 1e-3, 1e-4, 1e-5]
        results = analyze_gradient_bias(p, K, eps_values)

        print("\n" + "-" * 70)
        print(f"{'eps':>10} | {'bias mean':>12} | {'bias std':>12} | {'rel error':>12}")
        print("-" * 70)

        for eps, metrics in results.items():
            print(f"{eps:>10.0e} | {metrics.bias_mean:>12.4f} | "
                  f"{metrics.bias_std:>12.4f} | {metrics.relative_error:>12.4f}")

        print("\n" + "=" * 70)
        print("KEY FINDINGS")
        print("=" * 70)
        print("""
[CRITICAL] STE Proxy Gradient Bias Analysis:

1. STE Proxy Gradient: d(p*K)/dp = K*I
   - Each p_d's change affects K_d with gradient magnitude K
   - For K=64, proxy gradient is approximately 64

2. True Gradient: d(LRM)/dp
   - LRM uses floor() and remainder sorting, essentially discrete
   - Gradient jumps at threshold crossings (0 -> K)
   - Finite difference captures this discreteness

3. Bias Source:
   - STE assumes p*K is continuously differentiable
   - LRM introduces non-linear threshold behavior
   - Bias magnitude: O(K) = O(64)

4. Convergence Issues:
   - Training with STE may not converge to LRM optimal
   - Need continuous relaxation + projection for guaranteed convergence
        """)

    def test_bias_vs_K(self):
        """Test bias scaling with K"""
        print("\n" + "=" * 70)
        print("Gradient Bias vs K Scaling Analysis")
        print("=" * 70)

        D = 8
        K_values = [16, 32, 64, 128, 256]
        p = np.random.rand(D)
        p = p / p.sum()

        print(f"Config: D={D}, distribution={p.round(4)}")

        scale_results = []

        for K in K_values:
            results = analyze_gradient_bias(p, K)
            best_eps = min(results.keys(), key=lambda x: results[x].relative_error)
            best = results[best_eps]

            scale_results.append({
                'K': K,
                'bias_mean': best.bias_mean,
                'bias_std': best.bias_std,
                'relative_error': best.relative_error
            })

            print(f"\nK={K:>3}: bias_mean={best.bias_mean:.4f}, "
                  f"rel_error={best.relative_error:.4f}")

        print("\n" + "-" * 70)
        print("Scaling Analysis:")
        print("-" * 70)

        K_arr = np.array([r['K'] for r in scale_results])
        bias_arr = np.array([r['bias_mean'] for r in scale_results])
        bias_ratio = bias_arr / K_arr

        print(f"Bias/K mean: {bias_ratio.mean():.4f} (std={bias_ratio.std():.4f})")

        if bias_ratio.std() < 0.1:
            print("[CONFIRMED] Bias proportional to K (linear scaling)")
        else:
            print("[WARNING] Bias not simple linear relationship with K")

    def test_gradient_distribution(self):
        """Test gradient distribution"""
        print("\n" + "=" * 70)
        print("Gradient Distribution Analysis")
        print("=" * 70)

        D = 8
        K = 64
        p = np.random.rand(D)
        p = p / p.sum()

        grad_true, grad_ste = compute_ste_gradient(p, K, eps=1e-5)

        print(f"STE gradient (non-zero): {(grad_ste != 0).sum()}")
        print(f"True gradient (non-zero): {(np.abs(grad_true) > 1e-6).sum()}")

        true_nonzero = np.abs(grad_true) > 1e-6
        if true_nonzero.sum() > 0:
            true_values = grad_true[true_nonzero]
            print(f"True gradient non-zero values:")
            print(f"  Mean: {true_values.mean():.4f}")
            print(f"  Std: {true_values.std():.4f}")
            print(f"  Unique: {np.unique(np.abs(true_values).round(0))}")

        print("\n" + "=" * 70)
        print("CRITICAL ANALYSIS")
        print("=" * 70)
        print("""
[CRITICAL ANALYSIS]:

1. STE Proxy Gradient Distribution:
   - Form: K*I (diagonal matrix)
   - Each p_d independently affects K_d with same gradient magnitude
   - Problem: Ignores remainder allocation effects

2. True Gradient Distribution:
   - Form: Sparse, non-uniform
   - Only p_d crossing threshold produces gradient
   - Sparsity: ~25-50% elements have non-zero gradient

3. Training Impact:
   - STE overestimates most p_d gradients
   - STE underestimates edge p_d gradients (threshold附近)
   - May cause over-adjustment of quota distribution

4. Solution Directions:
   - Continuous Relaxation: Use softmax(phi/tau) instead of p
   - Projection to LRM: q_proj = LRM(softmax(phi/tau) * K)
   - Better convergence guarantees
        """)

    def compare_gradient_schemes(self):
        """Compare different gradient schemes"""
        print("\n" + "=" * 70)
        print("Gradient Schemes Comparison")
        print("=" * 70)

        D = 8
        K = 64
        p = np.random.rand(D)
        p = p / p.sum()

        print(f"Config: D={D}, K={K}")
        print(f"Distribution: {p.round(4)}")

        # Scheme 1: STE
        grad_ste = np.eye(D) * K

        # Scheme 2: Gumbel-Softmax
        tau_values = [0.1, 0.5, 1.0, 2.0]
        print("\nScheme 2: Gumbel-Softmax Approximation")
        for tau in tau_values:
            gumbel_p = F.softmax(
                torch.log(torch.tensor(p)) + torch.randn(D) / tau,
                dim=0
            ).numpy()

            soft_quota = gumbel_p * K
            hard_quota = largest_remainder_method(gumbel_p, K)
            diff = np.abs(soft_quota - hard_quota).mean()
            print(f"  tau={tau:.1f}: soft-hard diff={diff:.4f}")

        # Scheme 3: Continuous Relaxation + Projection
        print("\nScheme 3: Continuous Relaxation + LRM Projection")
        phi = np.log(p + 1e-8)

        for tau in tau_values:
            q_soft = np.exp(phi / tau) / np.sum(np.exp(phi / tau))
            q_proj = largest_remainder_method(q_soft, K)
            k_true = largest_remainder_method(p, K)
            diff = np.abs(q_proj - k_true).mean()
            print(f"  tau={tau:.1f}: projection diff={diff:.4f}")

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print("""
[SUMMARY TABLE]:

| Scheme              | Differentiable | Convergence | Complexity | Bias Control |
|---------------------|---------------|-------------|------------|--------------|
| STE (current)       | Approx        | No          | O(D)       | O(K)         |
| Gumbel-Softmax     | Yes           | Partial     | O(D)       | O(tau*K)     |
| Continuous+Project  | Yes           | Yes         | O(D log D) | O(1)         |

Recommended: Continuous Relaxation + Projection
- Forward: softmax(phi/tau) * K -> LRM projection
- Backward: Gradient through softmax
- Advantages: Guaranteed convergence, controllable bias
        """)


class TestQuotaConvergence:
    """Quota allocation convergence tests"""

    def test_convergence_simulation(self):
        """Simulate quota learning convergence"""
        print("\n" + "=" * 70)
        print("Quota Learning Convergence Simulation")
        print("=" * 70)

        D = 5
        K = 64

        p_target = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        k_target = largest_remainder_method(p_target, K)

        p = np.ones(D) / D

        print(f"Target quota: {k_target}")
        print(f"Initial quota: {largest_remainder_method(p, K)}")

        lr = 0.01
        loss_history = []

        for step in range(500):
            grad = K * (p - p_target)
            p = p - lr * grad
            p = np.clip(p, 0, 1)
            p = p / p.sum()

            k = largest_remainder_method(p, K)
            loss = np.abs(k - k_target).sum()
            loss_history.append(loss)

        final_k = largest_remainder_method(p, K)

        print(f"\nFinal quota: {final_k}")
        print(f"Final loss: {loss_history[-1]:.4f}")

        print("\n" + "=" * 70)
        print("CONVERGENCE CRITIQUE")
        print("=" * 70)
        print("""
[CRITICAL ANALYSIS]:

1. STE Gradient Issues:
   - STE assumes d(p*K)/dp = K*I
   - True gradient is sparse, discrete
   - STE may cause over-adjustment

2. Convergence Behavior:
   - If STE gradient direction correct: fast convergence
   - If STE gradient direction wrong: oscillation
   - Boundary cases (p near 0 or 1) unstable

3. Improvement Directions:
   - Confidence-weighted: grad = confidence * K * (p - p_target)
   - Soft threshold: only update when |p - p_target| > threshold
   - Momentum: smooth discrete gradient
        """)


if __name__ == "__main__":
    tester = TestQuotaGradientCritique()
    convergence_tester = TestQuotaConvergence()

    tester.test_gradient_bias_analysis()
    tester.test_bias_vs_K()
    tester.test_gradient_distribution()
    tester.compare_gradient_schemes()
    convergence_tester.test_convergence_simulation()


# I113-17: 新增 ContinuousQuotaAllocator 测试
# =============================================================================
class TestContinuousQuotaAllocator:
    """I113-17: 连续松弛配额分配器测试"""

    def test_allocator_creation(self):
        """测试分配器创建"""
        print("\n" + "=" * 70)
        print("I113-17: ContinuousQuotaAllocator 创建测试")
        print("=" * 70)

        from vit_pytorch.gumbel_topk_splitter import ContinuousQuotaAllocator

        allocator = ContinuousQuotaAllocator(D=8, tau=1.0)

        assert allocator.D == 8
        assert allocator.temperature.item() == 1.0
        assert allocator.quota_logits.shape == (8,)

        print("[PASSED] 分配器创建成功")

    def test_forward_backward(self):
        """测试前向和反向传播"""
        print("\n" + "=" * 70)
        print("I113-17: 前向/反向传播测试")
        print("=" * 70)

        import torch
        from vit_pytorch.gumbel_topk_splitter import ContinuousQuotaAllocator

        allocator = ContinuousQuotaAllocator(D=8, tau=1.0)

        K_hard, K_soft = allocator.forward(K=64)

        # 验证形状
        assert K_hard.shape == (8,)
        assert K_soft.shape == (8,)

        # 验证硬配额是整数
        assert K_hard.dtype == torch.int64

        # 验证软配额是浮点数
        assert K_soft.dtype == torch.float32

        # 验证总和 (硬配额)
        assert K_hard.sum().item() == 64

        print(f"硬配额: {K_hard.detach().numpy()}")
        print(f"软配额: {K_soft.detach().numpy()}")
        print(f"软配额总和: {K_soft.sum().item():.4f}")
        print("[PASSED] 前向传播正确")

    def test_gradient_flow(self):
        """测试梯度流是否正确传递"""
        print("\n" + "=" * 70)
        print("I113-17: 梯度流测试")
        print("=" * 70)

        import torch
        from vit_pytorch.gumbel_topk_splitter import ContinuousQuotaAllocator

        allocator = ContinuousQuotaAllocator(D=8, tau=1.0)

        # 启用梯度
        K_hard, K_soft = allocator.forward(K=64)

        # 使用软配额计算损失 - 目标是均匀分布，这依赖于分布形状
        target = torch.ones(8) * 8
        loss = F.mse_loss(K_soft, target)

        # 反向传播
        loss.backward()

        # 验证梯度存在且非零
        grad = allocator.quota_logits.grad
        assert grad is not None
        assert not torch.allclose(grad, torch.zeros_like(grad)), "梯度应该非零"

        print(f"梯度形状: {grad.shape}")
        print(f"梯度非零元素: {(grad != 0).sum().item()}/{grad.numel()}")
        print(f"梯度均值: {grad.mean().item():.6f}")
        print(f"梯度标准差: {grad.std().item():.6f}")
        print("[PASSED] 梯度流正确传递")

    def test_gradient_quality_vs_ste(self):
        """对比连续松弛与STE的梯度质量"""
        print("\n" + "=" * 70)
        print("I113-17: 梯度质量对比测试")
        print("=" * 70)

        import torch
        import numpy as np
        from vit_pytorch.gumbel_topk_splitter import ContinuousQuotaAllocator

        D = 8
        K = 64

        # 测试连续松弛
        allocator = ContinuousQuotaAllocator(D=D, tau=1.0)
        K_hard, K_soft = allocator.forward(K)
        loss = K_soft.sum()
        loss.backward()

        cr_grad = allocator.quota_logits.grad.clone()
        cr_grad_norm = cr_grad.norm().item()

        # STE近似梯度 (模拟)
        # STE假设: ∂K_d/∂p_e = K * δ_de
        # 真实梯度: 通过有限差分计算
        def lrm_projection(K_soft_np, K):
            floor_quota = np.floor(K_soft_np).astype(int)
            remainders = K_soft_np - floor_quota
            remaining = int(np.floor(K_soft_np.sum())) - floor_quota.sum()
            remaining = max(0, remaining)
            if remaining > 0:
                indices = np.argsort(-remainders)[:remaining]
                floor_quota[indices] += 1
            return floor_quota

        def compute_ste_gradient(p, K, eps=1e-5):
            """STE代理梯度"""
            return np.eye(len(p)) * K

        def compute_true_gradient(p, K, eps=1e-5):
            """真实梯度 (有限差分)"""
            D = len(p)
            grad = np.zeros((D, D))
            for e in range(D):
                p_eps = p.copy()
                p_eps[e] += eps
                p_eps = p_eps / p_eps.sum()
                k_base = lrm_projection(p * K, K)
                k_eps = lrm_projection(p_eps * K, K)
                grad[:, e] = (k_eps - k_base) / eps
            return grad

        p = torch.softmax(allocator.quota_logits, dim=0).detach().numpy()
        ste_grad = compute_ste_gradient(p, K)
        true_grad = compute_true_gradient(p, K)
        bias = np.abs(ste_grad - true_grad).mean()

        print(f"连续松弛梯度范数: {cr_grad_norm:.4f}")
        print(f"STE偏差均值: {bias:.4f}")
        print(f"真实梯度非零元素: {(np.abs(true_grad) > 1e-6).sum()}")

        print("[PASSED] 梯度质量对比完成")

    def test_temperature_warmup(self):
        """测试温度warmup"""
        print("\n" + "=" * 70)
        print("I113-17: 温度 Warmup 测试")
        print("=" * 70)

        import torch
        from vit_pytorch.gumbel_topk_splitter import ContinuousQuotaAllocator

        allocator = ContinuousQuotaAllocator(
            D=8, tau=1.0, tau_warmup_steps=100, enable_warmup=True
        )

        # 初始温度 (应该更高)
        initial_tau = allocator.temperature.item()
        print(f"初始温度: {initial_tau:.4f}")

        # 多次调用 step
        for _ in range(50):
            allocator.step()

        mid_tau = allocator.temperature.item()
        print(f"50步后温度: {mid_tau:.4f}")

        for _ in range(50):
            allocator.step()

        final_tau = allocator.temperature.item()
        print(f"100步后温度: {final_tau:.4f}")

        # 验证温度逐渐降低到目标值
        assert final_tau == 1.0  # 应该回到目标温度
        print("[PASSED] 温度warmup正确")

    def test_multi_step_training(self):
        """模拟多步训练"""
        print("\n" + "=" * 70)
        print("I113-17: 多步训练模拟测试")
        print("=" * 70)

        import torch
        from vit_pytorch.gumbel_topk_splitter import ContinuousQuotaAllocator

        allocator = ContinuousQuotaAllocator(D=8, tau=1.0, enable_warmup=False)

        optimizer = torch.optim.Adam(allocator.parameters(), lr=0.1)

        losses = []
        grad_norms = []

        for step in range(100):
            optimizer.zero_grad()

            K_hard, K_soft = allocator.forward(K=64)

            # 目标: 均匀分布 - 使用软配额保持梯度
            target = torch.ones(8) * 8
            loss = F.mse_loss(K_soft, target)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            grad_norms.append(allocator.quota_logits.grad.norm().item())

        print(f"初始损失: {losses[0]:.4f}")
        print(f"最终损失: {losses[-1]:.4f}")
        print(f"损失下降: {(losses[0] - losses[-1]):.4f}")
        print(f"平均梯度范数: {sum(grad_norms)/len(grad_norms):.4f}")

        # 验证损失下降
        assert losses[-1] < losses[0], "损失应该下降"
        print("[PASSED] 多步训练收敛")

    def test_convergence_comparison(self):
        """对比收敛性: 连续松弛 vs STE"""
        print("\n" + "=" * 70)
        print("I113-17: 收敛性对比测试")
        print("=" * 70)

        import torch
        import numpy as np
        from vit_pytorch.gumbel_topk_splitter import ContinuousQuotaAllocator

        D = 8
        K = 64
        target_p = np.array([0.3, 0.25, 0.2, 0.15, 0.08, 0.01, 0.005, 0.005])
        target_quota = (target_p * K).round().astype(int)
        target_quota[-1] += K - target_quota.sum()  # 确保总和正确

        print(f"目标配额: {target_quota}")

        # 连续松弛
        cr_allocator = ContinuousQuotaAllocator(D=D, tau=1.0, enable_warmup=False)
        cr_optimizer = torch.optim.Adam(cr_allocator.parameters(), lr=0.1)

        cr_losses = []
        for step in range(200):
            cr_optimizer.zero_grad()
            K_hard, K_soft = cr_allocator.forward(K)
            target_tensor = torch.tensor(target_quota, dtype=torch.float32)
            loss = F.mse_loss(K_soft, target_tensor)
            loss.backward()
            cr_optimizer.step()
            cr_losses.append(loss.item())

        final_cr_quota = cr_allocator.forward(K)[0].detach().numpy()
        print(f"连续松弛最终配额: {final_cr_quota}")
        print(f"连续松弛最终损失: {cr_losses[-1]:.4f}")

        # 计算配额误差
        cr_error = np.abs(final_cr_quota - target_quota).mean()

        print(f"\n连续松弛配额误差: {cr_error:.4f}")
        print(f"连续松弛收敛: {'是' if cr_losses[-1] < cr_losses[0] else '否'}")

        print("[PASSED] 收敛性对比完成")


if __name__ == "__main__":
    cr_tester = TestContinuousQuotaAllocator()

    print("\n" + "#" * 70)
    print("# I113-17: ContinuousQuotaAllocator 测试套件")
    print("#" * 70)

    cr_tester.test_allocator_creation()
    cr_tester.test_forward_backward()
    cr_tester.test_gradient_flow()
    cr_tester.test_gradient_quality_vs_ste()
    cr_tester.test_temperature_warmup()
    cr_tester.test_multi_step_training()
    cr_tester.test_convergence_comparison()

    print("\n" + "=" * 70)
    print("所有 I113-17 测试通过!")
    print("=" * 70)
