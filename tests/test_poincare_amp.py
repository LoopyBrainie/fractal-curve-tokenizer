"""
Poincaré 距离 AMP 模拟测试

模拟真实训练场景：在 autocast 环境下运行，
检查前向传播中间值的精度问题。
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Tuple

# 导入修复后的函数
import sys
sys.path.insert(0, 'src')
from vit_pytorch.layers.attention.manifold_decoder import poincare_distance as poincare_distance_fixed


def poincare_distance_amp(
    coords: torch.Tensor,
    image_size: Tuple[int, int],
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """
    使用修复后的版本进行测试
    """
    return poincare_distance_fixed(coords, image_size, epsilon)

    normalized = coords_fp32.clone()
    normalized[..., 0] = (coords_fp32[..., 0] - cx) / (cx + epsilon)
    normalized[..., 1] = (coords_fp32[..., 1] - cy) / (cy + epsilon)

    r = torch.norm(normalized, dim=-1, keepdim=True)
    r_safe = r.clamp(min=epsilon)

    # 关键：双重保险确保 u_norm < 1
    u_numerator = torch.tanh(r_safe / 2) * normalized
    u_denominator = (r_safe + epsilon)
    u = u_numerator / u_denominator

    # 归一化确保 ||u|| < 1
    u_norm = torch.norm(u, dim=-1, keepdim=True).clamp(min=epsilon)
    u = u / u_norm * torch.tanh(r_safe / 2).clamp(max=0.9999)

    u_i = u.unsqueeze(2)
    u_j = u.unsqueeze(1)
    diff_norm_sq = torch.sum((u_i - u_j) ** 2, dim=-1)

    u_norm_sq = torch.sum(u ** 2, dim=-1)
    u_norm_sq_i = u_norm_sq.unsqueeze(2)
    u_norm_sq_j = u_norm_sq.unsqueeze(1)

    # 增强保护
    denom_i = (1 - u_norm_sq_i).clamp(min=epsilon)
    denom_j = (1 - u_norm_sq_j).clamp(min=epsilon)
    denominator = denom_i * denom_j

    numerator = 2 * diff_norm_sq
    x = 1 + numerator / denominator

    # acosh 定义域保护
    x = x.clamp(min=1.0 + epsilon)

    distance = torch.acosh(x)

    if was_2d:
        distance = distance.squeeze(0)

    return distance.to(orig_dtype)


def trace_intermediate_values(coords, image_size, name):
    """追踪中间计算值，检测精度损失"""
    print(f"\n{'='*60}")
    print(f"追踪: {name}")
    print(f"{'='*60}")

    coords_fp32 = coords.float()
    coords_fp16 = coords.half()

    epsilon = 1e-5

    # ============ FP32 追踪 ============
    print("\n[FP32] 中间值:")
    W, H = image_size
    cx, cy = W / 2, H / 2

    # 添加batch维度
    coords_batched_fp32 = coords_fp32.unsqueeze(0)  # [1, N, 2]

    normalized = coords_batched_fp32.clone()
    normalized[..., 0] = (coords_batched_fp32[..., 0] - cx) / (cx + epsilon)
    normalized[..., 1] = (coords_batched_fp32[..., 1] - cy) / (cy + epsilon)

    r = torch.norm(normalized, dim=-1, keepdim=True)
    r_safe = r.clamp(min=epsilon)
    print(f"  r (到中心距离): min={r_safe.min():.8f}, max={r_safe.max():.8f}")

    u = torch.tanh(r_safe / 2) * (normalized / (r_safe + epsilon))
    u_norm = torch.norm(u, dim=-1, keepdim=True)
    print(f"  ||u||: min={u_norm.min():.8f}, max={u_norm.max():.8f}")

    u_i = u.unsqueeze(2)
    u_j = u.unsqueeze(1)
    diff_norm_sq = torch.sum((u_i - u_j) ** 2, dim=-1)
    print(f"  ||u-v||^2: min={diff_norm_sq.min():.10f}, max={diff_norm_sq.max():.10f}")

    u_norm_sq = torch.sum(u ** 2, dim=-1)
    u_norm_sq_i = u_norm_sq.unsqueeze(2)
    u_norm_sq_j = u_norm_sq.unsqueeze(1)

    numerator = 2 * diff_norm_sq
    denominator = (1 - u_norm_sq_i) * (1 - u_norm_sq_j)
    denominator = denominator.clamp(min=epsilon)

    print(f"  numerator: min={numerator.min():.10f}, max={numerator.max():.10f}")
    print(f"  denominator: min={denominator.min():.10f}, max={denominator.max():.10f}")

    x = 1 + numerator / denominator
    x = x.clamp(min=1 + epsilon)
    print(f"  x (acosh输入): min={x.min():.10f}, max={x.max():.10f}")

    distance_fp32 = torch.acosh(x)
    print(f"  acosh(x): min={distance_fp32.min():.8f}, max={distance_fp32.max():.8f}")

    # ============ FP16 追踪 ============
    print("\n[FP16] 中间值:")
    coords_batched_fp16 = coords_fp16.unsqueeze(0)  # [1, N, 2]

    normalized = coords_batched_fp16.clone()
    normalized[..., 0] = (coords_batched_fp16[..., 0] - cx) / (cx + epsilon)
    normalized[..., 1] = (coords_batched_fp16[..., 1] - cy) / (cy + epsilon)

    r = torch.norm(normalized, dim=-1, keepdim=True)
    r_safe = r.clamp(min=epsilon)
    print(f"  r: min={r_safe.min():.8f}, max={r_safe.max():.8f}")

    u = torch.tanh(r_safe / 2) * (normalized / (r_safe + epsilon))
    u_norm = torch.norm(u, dim=-1, keepdim=True)
    print(f"  ||u||: min={u_norm.min():.8f}, max={u_norm.max():.8f}")

    u_i = u.unsqueeze(2)
    u_j = u.unsqueeze(1)
    diff_norm_sq = torch.sum((u_i - u_j) ** 2, dim=-1)
    print(f"  ||u-v||^2: min={diff_norm_sq.min():.10f}, max={diff_norm_sq.max():.10f}")

    u_norm_sq = torch.sum(u ** 2, dim=-1)
    u_norm_sq_i = u_norm_sq.unsqueeze(2)
    u_norm_sq_j = u_norm_sq.unsqueeze(1)

    numerator = 2 * diff_norm_sq
    denominator = (1 - u_norm_sq_i) * (1 - u_norm_sq_j)
    denominator = denominator.clamp(min=epsilon)

    print(f"  numerator: min={numerator.min():.10f}, max={numerator.max():.10f}")
    print(f"  denominator: min={denominator.min():.10f}, max={denominator.max():.10f}")

    x = 1 + numerator / denominator
    x = x.clamp(min=1 + epsilon)
    print(f"  x (acosh输入): min={x.min():.10f}, max={x.max():.10f}")

    distance_fp16 = torch.acosh(x)
    print(f"  acosh(x): min={distance_fp16.min():.8f}, max={distance_fp16.max():.8f}")

    # 差异
    diff = (distance_fp32 - distance_fp16.float()).abs()
    print(f"\n[差异] |FP32 - FP16|: max={diff.max():.8f}, mean={diff.mean():.8f}")

    return distance_fp32, distance_fp16


def test_autocast_simulation():
    """模拟 autocast 环境"""
    print("="*70)
    print("AMP autocast 模拟测试")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    image_size = (64, 64)

    # 测试场景 1: 极端边缘
    print("\n" + "="*50)
    print("场景1: 极端边缘像素")
    print("="*50)

    coords_extreme = torch.tensor([
        [32.0, 32.0],     # 中心
        [63.9, 63.9],     # 右上角 (极端)
        [0.1, 63.9],      # 左上角
        [63.9, 0.1],      # 右下角
    ], device=device)

    trace_intermediate_values(coords_extreme, image_size, "极端边缘像素")

    # 测试场景 2: 接近重合
    print("\n" + "="*50)
    print("场景2: 接近重合的像素")
    print("="*50)

    coords_close = torch.tensor([
        [32.0, 32.0],
        [32.001, 32.0],   # 极小偏移
        [32.0, 32.001],
        [31.999, 32.0],
    ], device=device)

    trace_intermediate_values(coords_close, image_size, "接近重合像素")

    # 测试场景 3: 大批量随机点
    print("\n" + "="*50)
    print("场景3: 大批量随机点 (256 tokens)")
    print("="*50)

    torch.manual_seed(123)
    # 生成类似真实训练时的 token 分布
    coords_random = torch.rand(256, 2, device=device) * 64

    # FP32
    result_fp32 = poincare_distance_amp(coords_random.float(), image_size)
    nan_fp32 = torch.isnan(result_fp32).sum().item()
    inf_fp32 = torch.isinf(result_fp32).sum().item()

    # FP16
    result_fp16 = poincare_distance_amp(coords_random.half(), image_size)
    nan_fp16 = torch.isnan(result_fp16).sum().item()
    inf_fp16 = torch.isinf(result_fp16).sum().item()

    # 零值检测
    zero_fp32 = (result_fp32 < 1e-6).sum().item()
    zero_fp16 = (result_fp16 < 1e-6).sum().item()

    print(f"FP32: NaN={nan_fp32}, Inf={inf_fp32}, 零值={zero_fp32}")
    print(f"FP16: NaN={nan_fp16}, Inf={inf_fp16}, 零值={zero_fp16}")

    # 差异分析
    diff = (result_fp32 - result_fp16.float()).abs()
    print(f"FP32-FP16 差异: max={diff.max():.6f}, mean={diff.mean():.6f}")

    # 找出差异最大的位置
    max_idx = diff.argmax()
    max_idx_flat = max_idx.item()
    i, j = max_idx_flat // 256, max_idx_flat % 256
    print(f"最大差异位置: ({i}, {j}), FP32={result_fp32[i,j]:.6f}, FP16={result_fp16[i,j]:.6f}")

    # 测试场景 4: 梯度反向传播测试
    print("\n" + "="*50)
    print("场景4: 梯度反向传播 (检测 NaN in gradients)")
    print("="*50)

    # FP32 梯度测试
    coords_grad_fp32 = torch.randn(32, 2, device=device) * 32 + 32
    coords_grad_fp32.requires_grad_(True)
    out_fp32 = poincare_distance_amp(coords_grad_fp32, image_size)
    loss_fp32 = out_fp32.sum()
    loss_fp32.backward()

    grad_fp32 = coords_grad_fp32.grad.clone()
    has_nan_grad_fp32 = torch.isnan(grad_fp32).any().item()
    has_inf_grad_fp32 = torch.isinf(grad_fp32).any().item()

    print(f"FP32 梯度: NaN={has_nan_grad_fp32}, Inf={has_inf_grad_fp32}")
    print(f"FP32 梯度范数: {grad_fp32.norm():.6f}")

    # FP16 梯度测试
    coords_grad_fp16 = torch.randn(32, 2, device=device) * 32 + 32
    coords_grad_fp16.requires_grad_(True)
    out_fp16 = poincare_distance_amp(coords_grad_fp16, image_size)
    loss_fp16 = out_fp16.sum()
    loss_fp16.backward()

    grad_fp16 = coords_grad_fp16.grad.float()
    has_nan_grad_fp16 = torch.isnan(grad_fp16).any().item()
    has_inf_grad_fp16 = torch.isinf(grad_fp16).any().item()

    print(f"FP16 梯度: NaN={has_nan_grad_fp16}, Inf={has_inf_grad_fp16}")
    print(f"FP16 梯度范数: {grad_fp16.norm():.6f}")

    # 梯度差异
    grad_diff = (grad_fp32 - grad_fp16).abs()
    print(f"梯度差异: max={grad_diff.max():.6f}, mean={grad_diff.mean():.6f}")

    print("\n" + "="*70)
    print("测试完成")
    print("="*70)


if __name__ == "__main__":
    test_autocast_simulation()
