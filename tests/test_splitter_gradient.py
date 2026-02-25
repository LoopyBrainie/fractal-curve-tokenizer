"""
Splitter 梯度流动测试 - 直接测试版

直接测试 density_field 的梯度能否通过 Entmax 回传。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, 'src')


def entmax_beta(scores: torch.Tensor, alpha: float = 1.5, dim: int = -1) -> torch.Tensor:
    """Entmax 稀疏激活"""
    if alpha == 2.0:
        return F.softmax(scores, dim=dim)

    scores_norm = scores - scores.max(dim=dim, keepdim=True)[0]

    if abs(alpha - 1.5) < 0.01:
        # Sparsemax
        sorted_scores, _ = torch.sort(scores_norm, dim=dim, descending=True)
        cumsum = torch.cumsum(sorted_scores, dim=dim)
        tau = (cumsum - 1) / torch.arange(1, sorted_scores.size(dim) + 1, device=scores.device).float().view(1, -1)
        tau = torch.clamp(tau.max(dim=dim, keepdim=True)[0], min=0)
        probs = F.relu(scores_norm - tau)
        probs = probs / probs.sum(dim=dim, keepdim=True).clamp(min=1e-8)
    else:
        probs = F.softmax(scores_norm * (alpha - 1), dim=dim)

    return probs


def test_gradient_flow():
    """直接测试 density_field -> logits -> entmax -> loss 的梯度"""
    print("=" * 70)
    print("Density Field 梯度流动直接测试")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 测试不同 Alpha 值
    alphas = [1.2, 1.5, 1.8, 2.0]

    for alpha in alphas:
        print(f"\n{'='*50}")
        print(f"Testing Alpha = {alpha}")
        print(f"{'='*50}")

        # 创建简化的 Splitter 模拟
        # 1. density_field: features -> density
        density_field = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        ).to(device)

        # 初始化 bias = -1.1 (sigmoid ≈ 0.25)
        with torch.no_grad():
            density_field[2].bias.fill_(-1.1)

        # 2. tree_logits: features -> logits
        tree_logits = nn.Linear(256, 16, bias=False).to(device)

        # 3. 输入特征
        features = torch.randn(4, 16, 256, device=device)

        # ===== 前向传播 =====
        # density_field 计算密度
        density = density_field(features.mean(dim=1, keepdim=True))  # [4, 1, 1]

        # tree_logits 计算
        logits = tree_logits(features)  # [4, 16, 16]

        # 用 density 作为 weight 乘以 logits
        # 模拟: K = density * sum(probs)
        probs = entmax_beta(logits, alpha=alpha, dim=1)  # [4, 16, 16]

        # 期望 token 数 = density * probs.sum()
        K_expected = density.squeeze(-1) * probs.sum(dim=(1, 2))  # [4]

        # 损失 = 目标 K (32) 与期望 K 的差异
        loss = (K_expected - 32).abs().mean()

        # ===== 反向传播 =====
        loss.backward()

        # ===== 检查梯度 =====
        print(f"Loss: {loss.item():.4f}")
        print(f"K_expected: {K_expected.detach().cpu().numpy()}")

        # density_field 梯度
        if density_field[2].weight.grad is not None:
            df_grad_norm = density_field[2].weight.grad.norm().item()
            df_bias_grad = density_field[2].bias.grad.norm().item()
            print(f"density_field.weight grad: {df_grad_norm:.8f}")
            print(f"density_field.bias grad:   {df_bias_grad:.8f}")
        else:
            print("density_field.weight grad: None")

        # tree_logits 梯度
        if tree_logits.weight.grad is not None:
            tl_grad_norm = tree_logits.weight.grad.norm().item()
            print(f"tree_logits.weight grad: {tl_grad_norm:.8f}")
        else:
            print("tree_logits.weight grad: None")

        # probs 信息
        probs_detached = probs.detach()
        nonzero_ratio = (probs_detached > 1e-6).float().mean().item()
        print(f"probs non-zero ratio: {nonzero_ratio:.2%}")

        # 检查梯度是否为0
        if df_grad_norm < 1e-8:
            print(f"[!] WARNING: density_field 梯度为 0!")
        else:
            print(f"[OK] density_field 梯度正常")

    # ===== 结论 =====
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("""
如果 density_field 梯度始终为 0，说明 Entmax 导致了硬截断，
梯度无法回传到 density_field。

建议修改:
1. 在 HilbertOptimalSplitterConfig 中:
   entmax_alpha_init: 1.5 -> 1.2

2. 或在 fractal_vit.py 配置:
   entmax_alpha_init: 1.5 -> 1.2
""")
    print("=" * 70)


if __name__ == "__main__":
    test_gradient_flow()
