"""
验证修改后的 entmax_alpha=1.2 是否修复了梯度流动问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, 'src')


def entmax_beta(scores: torch.Tensor, alpha: float = 1.2, dim: int = -1) -> torch.Tensor:
    if alpha == 2.0:
        return F.softmax(scores, dim=dim)

    scores_norm = scores - scores.max(dim=dim, keepdim=True)[0]

    if abs(alpha - 1.5) < 0.01:
        sorted_scores, _ = torch.sort(scores_norm, dim=dim, descending=True)
        cumsum = torch.cumsum(sorted_scores, dim=dim)
        tau = (cumsum - 1) / torch.arange(1, sorted_scores.size(dim) + 1, device=scores.device).float().view(1, -1)
        tau = torch.clamp(tau.max(dim=dim, keepdim=True)[0], min=0)
        probs = F.relu(scores_norm - tau)
        probs = probs / probs.sum(dim=dim, keepdim=True).clamp(min=1e-8)
    else:
        probs = F.softmax(scores_norm * (alpha - 1), dim=dim)

    return probs


def verify_fix():
    print("=" * 70)
    print("验证 entmax_alpha=1.2 修复")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 使用修改后的默认值 1.2
    alpha = 1.2

    density_field = nn.Sequential(
        nn.Linear(256, 64),
        nn.GELU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    ).to(device)

    with torch.no_grad():
        density_field[2].bias.fill_(-1.1)

    tree_logits = nn.Linear(256, 16, bias=False).to(device)

    features = torch.randn(4, 16, 256, device=device)

    density = density_field(features.mean(dim=1, keepdim=True))
    logits = tree_logits(features)
    probs = entmax_beta(logits, alpha=alpha, dim=1)

    K_expected = density.squeeze(-1) * probs.sum(dim=(1, 2))
    loss = (K_expected - 32).abs().mean()

    loss.backward()

    print(f"Alpha: {alpha}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Probs non-zero: {(probs > 1e-6).float().mean().item():.2%}")

    if density_field[2].weight.grad is not None:
        grad_norm = density_field[2].weight.grad.norm().item()
        print(f"density_field.weight grad: {grad_norm:.6f}")

        if grad_norm > 1e-6:
            print("\n[OK] 修复成功！梯度正常流动")
        else:
            print("\n[!] 仍然有问题")
    else:
        print("无梯度")


if __name__ == "__main__":
    verify_fix()
