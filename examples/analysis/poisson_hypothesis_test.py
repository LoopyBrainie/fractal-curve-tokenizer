"""
I133-1: Poisson假设验证实验

验证token计数是否服从Poisson分布。

运行方式:
    uv run python examples/analysis/poisson_hypothesis_test.py
"""

import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
from collections import Counter
from scipy import stats
from vit_pytorch import FractalCurveViT


def collect_token_counts(
    num_samples: int = 1000,
    image_size: int = 64,
    batch_size: int = 16,
) -> np.ndarray:
    """
    收集模型forward的token计数

    Args:
        num_samples: 收集的样本数量
        image_size: 图像大小
        batch_size: 批次大小

    Returns:
        token计数数组
    """
    print(f"初始化模型 (image_size={image_size})...")

    # 创建模型
    model = FractalCurveViT(
        image_size=image_size,
        num_classes=10,
        dim=128,
        num_layers=4,
        heads=4,
    )
    model.eval()

    # 收集token计数
    token_counts = []

    print(f"收集 {num_samples} 个样本的token计数...")

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            x = torch.randn(current_batch_size, 3, image_size, image_size)

            # Forward pass
            out = model(x)

            # 获取token计数 - 支持多种格式
            num_tokens = out.num_tokens

            if isinstance(num_tokens, torch.Tensor):
                # GPU tensor
                num_tokens = num_tokens.cpu().numpy()
            elif isinstance(num_tokens, (list, int)):
                if isinstance(num_tokens, int):
                    num_tokens = [num_tokens]
                num_tokens = np.array(num_tokens)

            token_counts.extend(num_tokens.tolist())

            if (i + batch_size) % 200 == 0:
                print(f"  进度: {i + batch_size}/{num_samples}")

    return np.array(token_counts)


def analyze_poisson_hypothesis(counts: np.ndarray) -> dict:
    """
    分析token计数是否符合Poisson分布

    Args:
        counts: token计数数组

    Returns:
        分析结果字典
    """
    n = len(counts)
    mean = counts.mean()
    var = counts.var()
    std = counts.std()

    # 分散指数 (Index of Dispersion)
    # Poisson: E[var] = E[mean] => I approx 1
    dispersion_index = var / mean if mean > 0 else 0

    # 基本统计量
    result = {
        'n': n,
        'mean': mean,
        'var': var,
        'std': std,
        'min': counts.min(),
        'max': counts.max(),
        'dispersion_index': dispersion_index,
    }

    print("\n" + "="*60)
    print("Poisson Hypothesis Test Results")
    print("="*60)
    print(f"Sample count:       {n}")
    print(f"Mean (E[K]):        {mean:.2f}")
    print(f"Variance (Var):    {var:.2f}")
    print(f"Std deviation:      {std:.2f}")
    print(f"Range:              [{counts.min()}, {counts.max()}]")
    print(f"Dispersion Index I: {dispersion_index:.4f}")
    print("-"*60)

    # 分散指数分析
    print("\n[Dispersion Index Analysis]")
    print(f"  I = Var/Mean = {dispersion_index:.4f}")

    if abs(dispersion_index - 1.0) < 0.2:
        print("  -> I approx 1: Consistent with Poisson (Var approx Mean)")
    elif dispersion_index > 1.0:
        print("  -> I > 1: Over-dispersion detected")
        print("    Recommendation: Use Negative Binomial distribution")
    else:
        print("  -> I < 1: Under-dispersion detected")
        print("    Recommendation: Use alternative distribution")

    # KS检验
    print("\n[Kolmogorov-Smirnov Test]")
    # 使用均值作为Poisson参数
    ks_stat, ks_pvalue = stats.kstest(counts, 'poisson', args=(mean,))
    result['ks_stat'] = ks_stat
    result['ks_pvalue'] = ks_pvalue

    print(f"  KS statistic:    {ks_stat:.4f}")
    print(f"  p-value:         {ks_pvalue:.4f}")

    if ks_pvalue > 0.05:
        print("  -> p > 0.05: Cannot reject Poisson hypothesis")
    else:
        print("  -> p < 0.05: Rejects Poisson hypothesis (alpha=0.05)")

    # 频率分布
    print("\n[Frequency Distribution]")
    counter = Counter(counts)
    sorted_counts = sorted(counter.items())

    # 显示频率分布
    freq_lines = []
    for k, c in sorted_counts[:15]:  # 只显示前15个
        freq_lines.append(f"  K={k:2d}: {c:4d} ({100*c/n:.1f}%)")
    print("\n".join(freq_lines))

    if len(sorted_counts) > 15:
        print(f"  ... (共 {len(sorted_counts)} 个不同的K值)")

    # 与理论Poisson分布对比
    print("\n[Comparison with Theoretical Poisson]")
    unique_ks = sorted(counter.keys())
    observed_freq = [counter[k] / n for k in unique_ks]

    # 理论Poisson概率
    theoretical_prob = [stats.poisson.pmf(k, mean) for k in unique_ks]

    # 显示对比
    print("  K     Observed    Theoretical(Poisson)")
    print("  " + "-"*40)
    for i, k in enumerate(unique_ks[:10]):
        obs = observed_freq[i]
        theo = theoretical_prob[i]
        print(f"  {k:2d}    {obs:.4f}       {theo:.4f}")

    # 卡方检验 (简化版)
    print("\n[Chi-square Goodness-of-fit Test]")

    # 按区间合并
    bins = [(0, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, float('inf'))]
    observed_bins = []
    expected_bins = []

    for low, high in bins:
        # 观测频数
        obs_count = sum(counter[k] for k in unique_ks if low <= k < high)
        observed_bins.append(obs_count)

        # 期望频数 (Poisson)
        if high == float('inf'):
            exp_prob = 1 - stats.poisson.cdf(low - 1, mean)
        else:
            exp_prob = stats.poisson.cdf(high - 1, mean) - stats.poisson.cdf(low - 1, mean)
        expected_bins.append(exp_prob * n)

    # 合并频数小于5的区间
    min_expected = 5
    obs_merged = []
    exp_merged = []

    obs_tmp = 0
    exp_tmp = 0

    for obs, exp in zip(observed_bins, expected_bins):
        obs_tmp += obs
        exp_tmp += exp
        if exp_tmp >= min_expected:
            obs_merged.append(obs_tmp)
            exp_merged.append(exp_tmp)
            obs_tmp = 0
            exp_tmp = 0

    if obs_tmp > 0:
        if len(exp_merged) > 0:
            obs_merged[-1] += obs_tmp
            exp_merged[-1] += exp_tmp
        else:
            obs_merged.append(obs_tmp)
            exp_merged.append(exp_tmp)

    # 计算卡方统计量
    chi2_stat = sum((o - e) ** 2 / e for o, e in zip(obs_merged, exp_merged) if e > 0)
    df = max(len(obs_merged) - 1 - 1, 1)  # 类别数 - 1 - 估计参数数(1)
    chi2_pvalue = 1 - stats.chi2.cdf(chi2_stat, df)

    result['chi2_stat'] = chi2_stat
    result['chi2_pvalue'] = chi2_pvalue

    print(f"  Chi-square stat: {chi2_stat:.4f}")
    print(f"  Degrees of freedom: {df}")
    print(f"  p-value:       {chi2_pvalue:.4f}")

    if chi2_pvalue > 0.05:
        print("  -> p > 0.05: Cannot reject Poisson hypothesis")
    else:
        print("  -> p < 0.05: Rejects Poisson hypothesis")

    # 结论
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)

    conclusions = []

    if abs(dispersion_index - 1.0) < 0.2:
        conclusions.append("[OK] Dispersion index near 1, supports Poisson hypothesis")
    elif dispersion_index > 1.5:
        conclusions.append("[X] Dispersion index >> 1, over-dispersion detected")
    else:
        conclusions.append("[!] Dispersion index < 1, under-dispersion detected")

    if ks_pvalue > 0.05:
        conclusions.append("[OK] KS test: cannot reject Poisson hypothesis")
    else:
        conclusions.append("[X] KS test: rejects Poisson hypothesis")

    if chi2_pvalue > 0.05:
        conclusions.append("[OK] Chi-square: cannot reject Poisson hypothesis")
    else:
        conclusions.append("[X] Chi-square: rejects Poisson hypothesis")

    for c in conclusions:
        print(f"  {c}")

    # 综合判断
    reject_count = sum(1 for c in conclusions if c.startswith("[X]"))

    if reject_count >= 2:
        print("\n[CONCLUSION]: Poisson hypothesis LIKELY INVALID")
        print("  Recommendation: Consider Negative Binomial or alternative loss function")
        result['conclusion'] = 'reject'
    elif reject_count == 1:
        print("\n[CONCLUSION]: Borderline case, requires further analysis")
        result['conclusion'] = 'uncertain'
    else:
        print("\n[CONCLUSION]: Poisson hypothesis LARGELY VALID")
        result['conclusion'] = 'accept'

    return result


def main():
    """主函数"""
    print("="*60)
    print("I133-1: Poisson假设验证实验")
    print("="*60)

    # 收集token计数
    # 使用较小的样本数和batch_size以快速验证
    counts = collect_token_counts(
        num_samples=1000,
        image_size=64,
        batch_size=16,
    )

    # 分析
    result = analyze_poisson_hypothesis(counts)

    # 保存结果
    output_path = 'experiments/poisson_hypothesis_result.npz'
    np.savez(output_path, counts=counts)
    print(f"\n数据已保存到: {output_path}")

    return result


if __name__ == '__main__':
    main()
