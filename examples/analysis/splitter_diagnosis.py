"""
数学形式化批判分析：当前 Splitter 问题诊断

核心问题：阈值 τ 与复杂度 C(R) 的数值范围不匹配
"""
import sys
import torch
import numpy as np
sys.path.insert(0, 'src')

from vit_pytorch.split_adaptive import (
    BalancedGreedySplitter, 
    AdaptiveSplitConfig, 
    IntegralImageCache, 
    Region, 
    ComplexityEstimator
)

def analyze_complexity_distribution():
    """分析不同类型图像的复杂度分布."""
    print("=" * 70)
    print("1. 复杂度分布分析 (Complexity Distribution Analysis)")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # 创建不同类型的测试图像
    test_images = {
        "uniform_gray": torch.ones(3, 64, 64) * 0.5,
        "pure_noise": torch.rand(3, 64, 64),
        "gaussian_noise": torch.randn(3, 64, 64).clamp(0, 1),
        "half_split": torch.cat([torch.zeros(3, 64, 32), torch.ones(3, 64, 32)], dim=2),
        "checkerboard_8x8": create_checkerboard(64, 8),
        "gradient_h": torch.linspace(0, 1, 64).view(1, 1, 64).expand(3, 64, 64),
        "gradient_v": torch.linspace(0, 1, 64).view(1, 64, 1).expand(3, 64, 64),
    }
    
    # 默认配置
    config = AdaptiveSplitConfig.scheme_b(max_depth=2, tau_0=0.15, gamma=0.85)
    
    print(f"\n配置参数:")
    print(f"  α = {config.alpha} (variance weight)")
    print(f"  σ₀² = {config.sigma_0_sq}")
    print(f"  g₀² = {config.g_0_sq}")
    print(f"  τ₀ = {config.tau_0}, γ = {config.gamma}")
    print(f"  阈值: τ₀={config.tau_0:.3f}, τ₁={config.get_threshold(1):.3f}, τ₂={config.get_threshold(2):.3f}")
    
    print(f"\n各类图像的复杂度值:")
    print(f"{'图像类型':<20} {'C_var':<12} {'C_grad':<12} {'C_total':<12} {'vs τ₀'}")
    print("-" * 70)
    
    for name, img in test_images.items():
        cache = IntegralImageCache(img)
        est = ComplexityEstimator(config, root_area=64*64)
        
        region = Region(0, 0, 64, 64)
        var = cache.compute_variance(region)
        grad = cache.compute_gradient_energy(region)
        
        c_var = var / (var + config.sigma_0_sq)
        c_grad = grad / (grad + config.g_0_sq)
        c_total = config.alpha * c_var + (1 - config.alpha) * c_grad
        
        vs_tau = "SPLIT" if c_total > config.tau_0 else "KEEP"
        print(f"{name:<20} {c_var:<12.4f} {c_grad:<12.4f} {c_total:<12.4f} {vs_tau}")

def create_checkerboard(size, block_size):
    """创建棋盘格图像."""
    img = torch.zeros(3, size, size)
    for i in range(size // block_size):
        for j in range(size // block_size):
            if (i + j) % 2 == 0:
                img[:, i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = 1.0
    return img

def analyze_threshold_sensitivity():
    """分析阈值敏感性."""
    print("\n" + "=" * 70)
    print("2. 阈值敏感性分析 (Threshold Sensitivity Analysis)")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # 模拟真实图像：混合平坦和纹理区域
    img = torch.randn(3, 64, 64).clamp(0, 1)
    img[:, :16, :16] = 0.5  # 左上角平坦
    img[:, :16, 48:] = 0.3  # 右上角平坦
    
    print(f"\n不同 τ₀ 下的分割结果:")
    print(f"{'τ₀':<10} {'depth=0':<10} {'depth=1':<10} {'depth=2':<10} {'tokens':<10} {'entropy'}")
    print("-" * 60)
    
    for tau_0 in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90, 0.95, 0.99]:
        config = AdaptiveSplitConfig.scheme_b(max_depth=2, tau_0=tau_0, gamma=0.85)
        splitter = BalancedGreedySplitter(config)
        result = splitter.split(img)
        
        dist = result.depth_distribution
        d0 = dist.get(0, 0)
        d1 = dist.get(1, 0)
        d2 = dist.get(2, 0)
        entropy = result.depth_entropy
        
        print(f"{tau_0:<10.2f} {d0:<10} {d1:<10} {d2:<10} {result.num_tokens:<10} {entropy:.3f}")

def analyze_mathematical_mismatch():
    """分析数学公式的不匹配问题."""
    print("\n" + "=" * 70)
    print("3. 数学不匹配分析 (Mathematical Mismatch Analysis)")
    print("=" * 70)
    
    # 问题核心：C(R) ∈ [0, 1]，但实际值集中在高端
    # 因为 C_var = Var / (Var + σ₀²) 对于大多数真实图像 Var >> σ₀²
    
    print(f"""
问题诊断:
---------
1. 归一化公式: C_var = Var / (Var + σ₀²)
   - 当 Var >> σ₀² 时，C_var → 1.0
   - 对于随机噪声图像，Var ≈ 0.08，σ₀² = 0.01
   - 因此 C_var = 0.08 / 0.09 = 0.89 >> τ₀ = 0.15
   
2. 这导致几乎所有区域都会被分割到最大深度

3. 根本问题：τ₀ 的"复杂度"语义与 C(R) 的数值范围不匹配
   - τ₀ = 0.15 意味着"15% 复杂度以下不分割"
   - 但实际图像的复杂度几乎都 > 0.8
   
4. 解决方案对比:
   A) 提高 τ₀ 到 0.9+ → 失去对低复杂度图像的敏感性
   B) 提高 σ₀² 到 0.1+ → 改变归一化语义
   C) 使用可学习的复杂度预测器 → 端到端学习最优分割
""")
    
    # 验证：计算不同 σ₀² 下的复杂度
    print("验证：σ₀² 对复杂度的影响")
    print(f"{'σ₀²':<12} {'噪声图像 C_var':<20} {'平坦图像 C_var':<20}")
    print("-" * 55)
    
    var_noise = 0.08  # 典型噪声图像方差
    var_flat = 0.001  # 典型平坦区域方差
    
    for sigma_0_sq in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]:
        c_noise = var_noise / (var_noise + sigma_0_sq)
        c_flat = var_flat / (var_flat + sigma_0_sq)
        print(f"{sigma_0_sq:<12.3f} {c_noise:<20.4f} {c_flat:<20.4f}")

def propose_learnable_solution():
    """提出可学习方案."""
    print("\n" + "=" * 70)
    print("4. 可学习 Splitter 方案设计")
    print("=" * 70)
    
    print("""
方案设计：LearnableSplitter
===========================

数学形式化:
-----------
1. 复杂度预测网络 (Complexity Network):
   
   C_θ(R) = σ(MLP(Pool(Conv(I, R))))
   
   其中:
   - Conv: 轻量级卷积特征提取 (与 PatchEmbed 共享)
   - Pool: ROI-Align 区域池化
   - MLP: 2 层 MLP，输出标量
   - σ: sigmoid 确保 C ∈ [0, 1]

2. 可微分分割决策:
   
   P_split(R, d) = σ((C_θ(R) - τ_d) / T)
   
   其中:
   - τ_d = τ_0 · γ^d (可学习阈值)
   - T: 温度参数 (训练时 T=1，推理时 T→0)

3. 训练目标:
   
   L = L_cls + λ₁ · L_entropy + λ₂ · L_budget
   
   其中:
   - L_cls: 分类损失 (主任务)
   - L_entropy: 鼓励尺度多样性
     L_entropy = -∑_d P(d) log P(d)
   - L_budget: token 数量约束
     L_budget = max(0, N - N_budget)²

4. Gumbel-Softmax 可微分采样:
   
   对于每个区域 R，分割决策:
   z = Gumbel-Softmax([P_keep, P_split], τ)
   
   这允许梯度流过离散分割决策

优势:
-----
1. 端到端学习：分割策略与分类目标对齐
2. 自适应阈值：无需手动调参 τ₀, σ₀², g₀²
3. 保持 Hilbert 局部性：分割后仍按 Hilbert 排序
4. 向后兼容：可作为 BalancedGreedySplitter 的可学习替代

实现策略:
---------
1. 初始阶段：使用规则 splitter 预热
2. 混合阶段：规则 + 学习混合
3. 完全可学习：端到端训练
""")

if __name__ == "__main__":
    analyze_complexity_distribution()
    analyze_threshold_sensitivity()
    analyze_mathematical_mismatch()
    propose_learnable_solution()
