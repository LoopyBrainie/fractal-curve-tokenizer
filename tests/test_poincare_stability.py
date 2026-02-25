"""
Poincaré 距离数值稳定性测试

测试 poincare_distance 函数在 FP16 和 FP32 下的表现差异，
验证极端边界条件（边缘像素、重合像素）下的数值稳定性。
"""

import torch
import numpy as np
from typing import Tuple


# ==================== 原始实现 (来自 manifold_decoder.py) ====================

def poincare_distance_original(
    coords: torch.Tensor,
    image_size: Tuple[int, int],
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """原始实现 (当前版本)"""
    was_2d = coords.dim() == 2
    if was_2d:
        coords = coords.unsqueeze(0)

    B, N, _ = coords.shape
    W, H = image_size

    cx, cy = W / 2, H / 2

    normalized = coords.clone()
    normalized[..., 0] = (coords[..., 0] - cx) / (cx + epsilon)
    normalized[..., 1] = (coords[..., 1] - cy) / (cy + epsilon)

    r = torch.norm(normalized, dim=-1, keepdim=True)
    u = torch.tanh(r / 2) * (normalized / (r + epsilon))

    u_i = u.unsqueeze(2)
    u_j = u.unsqueeze(1)
    diff_norm_sq = torch.sum((u_i - u_j) ** 2, dim=-1)

    u_norm_sq = torch.sum(u ** 2, dim=-1)
    u_norm_sq_i = u_norm_sq.unsqueeze(2)
    u_norm_sq_j = u_norm_sq.unsqueeze(1)

    numerator = 2 * diff_norm_sq
    denominator = (1 - u_norm_sq_i) * (1 - u_norm_sq_j)
    denominator = denominator.clamp(min=epsilon)

    x = 1 + numerator / denominator
    x = x.clamp(min=1 + epsilon)

    distance = torch.acosh(x)

    if was_2d:
        distance = distance.squeeze(0)

    return distance


# ==================== 修复实现 (强制 FP32 + 增强保护) ====================

def poincare_distance_fp32_safe(
    coords: torch.Tensor,
    image_size: Tuple[int, int],
    epsilon: float = 1e-7,  # 改为 1e-7 确保 acosh 稳定
) -> torch.Tensor:
    """
    FP32 安全版本：强制内部计算使用 FP32

    数学保证：
        - acosh(x) 的定义域: x >= 1
        - 当 x -> 1+ 时，导数 d(acosh)/dx = 1/sqrt(x^2-1) -> inf
        - 使用 epsilon = 1e-7 确保 x >= 1 + 1e-7，避免导数爆炸
    """
    # 强制转换为 FP32 进行计算
    orig_dtype = coords.dtype
    coords_fp32 = coords.float()

    was_2d = coords_fp32.dim() == 2
    if was_2d:
        coords_fp32 = coords_fp32.unsqueeze(0)

    B, N, _ = coords_fp32.shape
    W, H = image_size

    cx, cy = W / 2, H / 2

    normalized = coords_fp32.clone()
    normalized[..., 0] = (coords_fp32[..., 0] - cx) / (cx + epsilon)
    normalized[..., 1] = (coords_fp32[..., 1] - cy) / (cy + epsilon)

    # 计算到中心的距离
    r = torch.norm(normalized, dim=-1, keepdim=True)

    # 关键修复1: 防止 r = 0 (原点) 导致的除零
    r_safe = r.clamp(min=epsilon)

    # 关键修复2: tanh(r/2) 确保 ||u|| < 1，但使用 clamp 双重保险
    u = torch.tanh(r_safe / 2) * (normalized / (r_safe + epsilon))
    u_norm = torch.norm(u, dim=-1, keepdim=True)
    u = u / (u_norm + epsilon) * torch.tanh(r_safe / 2).clamp(max=0.9999)

    # 计算 ||u - v||^2
    u_i = u.unsqueeze(2)
    u_j = u.unsqueeze(1)
    diff_norm_sq = torch.sum((u_i - u_j) ** 2, dim=-1)

    # 计算 ||u||^2 和 ||v||^2
    u_norm_sq = torch.sum(u ** 2, dim=-1)
    u_norm_sq_i = u_norm_sq.unsqueeze(2)
    u_norm_sq_j = u_norm_sq.unsqueeze(1)

    # 关键修复3: 增强 denominator 保护
    denom_i = (1 - u_norm_sq_i).clamp(min=epsilon)
    denom_j = (1 - u_norm_sq_j).clamp(min=epsilon)
    denominator = denom_i * denom_j

    # 双曲距离公式
    numerator = 2 * diff_norm_sq
    x = 1 + numerator / denominator

    # 关键修复4: 强制 x >= 1 + epsilon (acosh 定义域保护)
    x = x.clamp(min=1.0 + epsilon)

    distance = torch.acosh(x)

    if was_2d:
        distance = distance.squeeze(0)

    # 保持原始 dtype
    return distance.to(orig_dtype)


# ==================== 测试用例 ====================

def test_edge_cases():
    """测试极端边界情况"""
    print("=" * 70)
    print("Poincaré 距离数值稳定性测试")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")

    # 图像尺寸 (TinyImageNet 64x64)
    image_size = (64, 64)

    # 测试用例 1: 边缘像素 (||u|| -> 1.0)
    print("\n" + "-" * 50)
    print("测试1: 边缘像素 (||u|| -> 1.0)")
    print("-" * 50)

    # 4个角点
    edge_coords = torch.tensor([
        [0.5, 0.5],      # 中心
        [0.99, 0.5],     # 接近右边缘
        [0.5, 0.99],     # 接近上边缘
        [0.99, 0.99],    # 右上角 (最极端)
    ], device=device)

    # 转换为实际像素坐标
    edge_coords_px = edge_coords * 64

    for coords, name in [(edge_coords_px, "edge")]:
        print(f"\n坐标 (像素): {coords.tolist()}")

        # FP32 测试
        result_fp32 = poincare_distance_original(coords.float(), image_size)
        print(f"  FP32 结果:\n  {result_fp32}")
        print(f"  FP32 NaN: {torch.isnan(result_fp32).any()}")
        print(f"  FP32 Inf: {torch.isinf(result_fp32).any()}")

        # FP16 测试
        result_fp16 = poincare_distance_original(coords.half(), image_size)
        print(f"  FP16 结果:\n  {result_fp16}")
        print(f"  FP16 NaN: {torch.isnan(result_fp16).any()}")
        print(f"  FP16 Inf: {torch.isinf(result_fp16).any()}")

        # FP16 vs FP32 差异
        if not torch.isnan(result_fp16).any():
            diff = (result_fp32.float() - result_fp16.float()).abs()
            print(f"  FP16-FP32 最大差异: {diff.max().item():.6f}")

    # 测试用例 2: 重合像素 (||u - v|| -> 0)
    print("\n" + "-" * 50)
    print("测试2: 重合像素 (||u - v|| -> 0)")
    print("-" * 50)

    # 非常接近的两个点
    close_coords = torch.tensor([
        [32.0, 32.0],
        [32.01, 32.0],  # 微小偏移
        [32.0, 32.01],
        [31.99, 32.0],  # 另一个方向
    ], device=device)

    for coords in [close_coords]:
        print(f"\n坐标: {coords.tolist()}")

        result_fp32 = poincare_distance_original(coords.float(), image_size)
        result_fp16 = poincare_distance_original(coords.half(), image_size)

        print(f"  FP32 结果:\n  {result_fp32}")
        print(f"  FP32 NaN: {torch.isnan(result_fp32).any()}")

        print(f"  FP16 结果:\n  {result_fp16}")
        print(f"  FP16 NaN: {torch.isnan(result_fp16).any()}")

    # 测试用例 3: 批量极端输入
    print("\n" + "-" * 50)
    print("测试3: 批量极端输入 (100个随机边缘点)")
    print("-" * 50)

    # 生成随机边缘点
    torch.manual_seed(42)
    batch_coords = []
    for _ in range(100):
        # 随机选择一条边
        side = torch.randint(0, 4, (1,)).item()
        if side == 0:  # 上边
            x = torch.rand(1).item() * 64
            y = 63.9  # 非常接近边缘
        elif side == 1:  # 下边
            x = torch.rand(1).item() * 64
            y = 0.1
        elif side == 2:  # 左边
            x = 0.1
            y = torch.rand(1).item() * 64
        else:  # 右边
            x = 63.9
            y = torch.rand(1).item() * 64
        batch_coords.append([x, y])

    batch_coords = torch.tensor(batch_coords, device=device)

    # FP32 批量测试
    result_fp32 = poincare_distance_original(batch_coords.float(), image_size)
    nan_fp32 = torch.isnan(result_fp32).sum().item()
    inf_fp32 = torch.isinf(result_fp32).sum().item()

    # FP16 批量测试
    result_fp16 = poincare_distance_original(batch_coords.half(), image_size)
    nan_fp16 = torch.isnan(result_fp16).sum().item()
    inf_fp16 = torch.isinf(result_fp16).sum().item()

    print(f"FP32: NaN={nan_fp32}/100, Inf={inf_fp32}/100")
    print(f"FP16: NaN={nan_fp16}/100, Inf={inf_fp16}/100")

    # 测试用例 4: 修复版本对比
    print("\n" + "-" * 50)
    print("测试4: FP32 安全版本 vs 原始版本")
    print("-" * 50)

    test_coords = torch.tensor([
        [32.0, 32.0],
        [63.9, 63.9],
        [0.1, 63.9],
        [63.9, 0.1],
    ], device=device)

    # 原始 FP16
    result_orig_fp16 = poincare_distance_original(test_coords.half(), image_size)

    # 修复版本 (FP16 输入 -> FP32 计算 -> FP16 输出)
    result_fix_fp16 = poincare_distance_fp32_safe(test_coords.half(), image_size)

    # 修复版本 (FP32 输入)
    result_fix_fp32 = poincare_distance_fp32_safe(test_coords.float(), image_size)

    print(f"\n原始 FP16:\n{result_orig_fp16}")
    print(f"修复 FP16:\n{result_fix_fp16}")
    print(f"修复 FP32:\n{result_fix_fp32}")

    print(f"\n原始 FP16 NaN: {torch.isnan(result_orig_fp16).any()}")
    print(f"修复 FP16 NaN: {torch.isnan(result_fix_fp16).any()}")
    print(f"修复 FP32 NaN: {torch.isnan(result_fix_fp32).any()}")

    # 对比差异
    diff = (result_fix_fp32.float() - result_fix_fp16.float()).abs()
    print(f"修复版本 FP16-FP32 差异: max={diff.max().item():.10f}")

    # 测试用例 5: acosh 输入边界测试
    print("\n" + "-" * 50)
    print("测试5: acosh 输入边界 (x -> 1+)")
    print("-" * 50)

    # 测试不同 epsilon 值
    epsilons = [1e-8, 1e-7, 1e-6, 1e-5]

    for eps in epsilons:
        # 构造 x = 1 + eps * random
        x = 1.0 + eps * torch.rand(100, device=device)
        x = x.clamp(min=1.0 + eps)

        try:
            # 原始: 直接用 x
            y_orig = torch.acosh(x.float())

            # 修复: 确保 x >= 1 + eps
            x_safe = x.clamp(min=1.0 + 1e-7)
            y_safe = torch.acosh(x_safe.float())

            print(f"epsilon={eps:.0e}: x范围=[{x.min():.10f}, {x.max():.10f}]")
            print(f"  原始 acosh NaN: {torch.isnan(y_orig).any()}")
            print(f"  修复 acosh NaN: {torch.isnan(y_safe).any()}")

        except Exception as e:
            print(f"epsilon={eps:.0e}: 错误 - {e}")

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    test_edge_cases()
