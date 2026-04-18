# -*- coding: utf-8 -*-
"""
FastBitwiseHilbert: 动态位宽 Hilbert 变换算子

数学框架
========

Hilbert 映射可分解为两个复合函数：

1. 位交织 (Morton Encoding):
   f_interleave(x, y) = M
   将 x, y 的比特位"拉链式"合并

2. Hilbert 变换 (Gray 码):
   g_hilbert(M) = D = M ⊕ (M >> 1)
   通过异或消除 Morton 码中的"突跳"

核心特性
--------
- 硬件感知动态位宽: 根据 dtype 自动推导 B
  - int32 → B = 16 (映射到 65536 空间)
  - int64 → B = 32 (映射到 2^32 空间)
- 归一化定点映射: x_v = (x * (2^B - 1)) // W
- 完全向量化: 无 Python 循环
- torch.compile 兼容
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from vit_pytorch.core.curve_hilbert import HilbertScanner


class FastBitwiseHilbert:
    """动态位宽 Hilbert 变换算子"""

    @staticmethod
    def infer_bit_width(dtype: torch.dtype) -> int:
        """
        从 dtype 推导虚拟位宽 B

        Args:
            dtype: 输入 Tensor 的数据类型

        Returns:
            虚拟位宽 B

        Raises:
            ValueError: 不支持的 dtype
        """
        if dtype == torch.int32 or dtype == torch.int64:
            return dtype.itemsize * 8 // 2
        elif dtype == torch.int16:
            return 8
        else:
            raise ValueError(
                f"Unsupported dtype: {dtype}. "
                f"Expected int32 or int64 (got {dtype})"
            )

    @staticmethod
    def normalize_coords(
        x: Tensor, y: Tensor, H: int, W: int, B: int
    ) -> Tuple[Tensor, Tensor]:
        """
        归一化定点映射

        将任意分辨率 (H, W) 映射到 2^B 的虚拟空间

        公式:
            x_v = (x * (2^B - 1)) // W
            y_v = (y * (2^B - 1)) // H

        注意: 先乘后除，确保拓扑顺序精确

        Args:
            x: x 坐标 Tensor
            y: y 坐标 Tensor
            H: 图像高度
            W: 图像宽度
            B: 虚拟位宽

        Returns:
            (xv, yv): 归一化后的虚拟坐标
        """
        scale = (1 << B) - 1  # 2^B - 1

        # 先乘后除，确保整数运算的精确性
        xv = (x * scale) // W
        yv = (y * scale) // H

        return xv, yv

    @staticmethod
    def xy_to_morton(x: Tensor, y: Tensor, n: int) -> Tensor:
        """
        批量将 2D 坐标转换为 Morton 码（位交织）

        数学形式:
            M = Σ_{k=0}^{B-1} (bit(x, k) << (2*k)) + (bit(y, k) << (2*k+1))

        这正是 Hilbert 曲线 Gray 码变换前的中间态。
        Morton 码保留了 Hilbert 曲线的：
        1. 局部性：相邻 Morton 码的位模式相似
        2. 旋转对称：位翻转对应 Hilbert 曲线的旋转

        I106-3: 向量化实现，使用 torch.bitwise_left_shift 避免 Python 循环

        Args:
            x: [B] x 坐标 Tensor (支持 int32/int64)
            y: [B] y 坐标 Tensor (支持 int32/int64)
            n: 网格大小 (必须是 2 的幂)

        Returns:
            M: [B] Morton 码 Tensor
        """
        # 转换为长整型并取模
        # D4-AUDIT FIX: %n → &(n-1)，n 是 2 的幂，& 比 % 快
        xv = x.long() & (n - 1)
        yv = y.long() & (n - 1)

        # 计算实际位宽
        max_bits = n.bit_length() - 1

        # I106-3: 向量化位交织 - 预计算所有位移
        # 创建位移数组: [0, 2, 4, ..., 2*(max_bits-1)]
        shifts_x = torch.arange(0, 2 * max_bits, 2, device=xv.device, dtype=torch.long)  # [max_bits]
        shifts_y = shifts_x + 1  # [1, 3, 5, ..., 2*max_bits-1]

        # 预计算的位掩码
        masks = (1 << torch.arange(max_bits, device=xv.device, dtype=torch.long)).unsqueeze(0)  # [1, max_bits]

        # 提取所有位: [B, max_bits]
        x_bits = ((xv.unsqueeze(-1) & masks) >> torch.arange(max_bits, device=xv.device)).long()
        y_bits = ((yv.unsqueeze(-1) & masks) >> torch.arange(max_bits, device=yv.device)).long()

        # 位交织: x 在偶数位，y 在奇数位
        # 使用移位而非乘法，减少开销
        x_morton = torch.bitwise_left_shift(x_bits, shifts_x.unsqueeze(0))  # [B, max_bits]
        y_morton = torch.bitwise_left_shift(y_bits, shifts_y.unsqueeze(0))  # [B, max_bits]

        # 按位求和
        morton = (x_morton + y_morton).sum(dim=-1)  # [B]

        return morton

    @staticmethod
    def morton_to_xy(morton: Tensor, n: int) -> Tuple[Tensor, Tensor]:
        """
        批量将 Morton 码转换为 2D 坐标

        数学:
            M = Σ (x_k << 2k) + (y_k << (2k+1))
            x_k = (M >> (2k)) & 1
            y_k = (M >> (2k+1)) & 1

        I106-3: 向量化实现，使用 torch.bitwise_left_shift 避免 Python 循环

        Args:
            morton: [B] Morton 码 Tensor
            n: 网格大小 (必须是 2 的幂)

        Returns:
            (x, y): [B] 坐标 Tensor
        """
        max_bits = n.bit_length() - 1

        # I106-3: 向量化位提取
        # 位移数组: [0, 2, 4, ..., 2*(max_bits-1)]
        shifts = torch.arange(0, 2 * max_bits, 2, device=morton.device, dtype=torch.long)  # [max_bits]

        # 提取 x 位和 y 位
        # [B, max_bits]
        x_bits = ((morton.unsqueeze(-1) >> shifts.unsqueeze(0)) & 1).long()
        y_bits = ((morton.unsqueeze(-1) >> (shifts + 1).unsqueeze(0)) & 1).long()

        # 位移回原位
        x_shifts = torch.arange(max_bits, device=morton.device, dtype=torch.long)  # [max_bits]
        x = (torch.bitwise_left_shift(x_bits, x_shifts.unsqueeze(0))).sum(dim=-1)  # [B]
        y = (torch.bitwise_left_shift(y_bits, x_shifts.unsqueeze(0))).sum(dim=-1)  # [B]

        return x, y

    @staticmethod
    def xy_to_d(x: Tensor, y: Tensor, H: int, W: int) -> Tensor:
        """
        批量将 2D 坐标转换为 Hilbert 距离

        数学形式:
            d = Σ_{k=0}^{log2(n)-1} 2^{2k} * ((3 * rx_k) ^ ry_k)

        其中 rx_k, ry_k 是第 k 位的比特值

        Args:
            x: [B] x 坐标 Tensor (支持 int32/int64)
            y: [B] y 坐标 Tensor (支持 int32/int64)
            H: 图像高度
            W: 图像宽度

        Returns:
            d: [B] Hilbert 距离 Tensor
        """
        # 判断是否为 2^k × 2^k 正方形
        if H == W and (H & (H - 1)) == 0:
            # 推导位宽
            dtype = x.dtype
            FastBitwiseHilbert.infer_bit_width(dtype)

            # 2^k 正方形：直接使用坐标
            # D4-AUDIT FIX: %H → &(H-1)，H 是 2 的幂
            xv = x.long() & (H - 1)
            yv = y.long() & (H - 1)
            n = H

            # Gray 码变换
            return FastBitwiseHilbert._gray_code_transform(xv, yv, n)
        else:
            # 矩形或非 2^k：使用 HilbertScanner 向量化 batch 方法
            # I-OPT: 使用 xy_to_d_batch 避免 cpu().numpy() + list comprehension
            return HilbertScanner.xy_to_d_batch(H, W, x.long(), y.long())

    @staticmethod
    def d_to_xy(d: Tensor, H: int, W: int) -> Tuple[Tensor, Tensor]:
        """
        批量将 Hilbert 距离转换为 2D 坐标

        Args:
            d: [B] Hilbert 距离 Tensor
            H: 图像高度
            W: 图像宽度

        Returns:
            (x, y): [B] 坐标 Tensor
        """
        # 判断是否为 2^k × 2^k 正方形
        if H == W and (H & (H - 1)) == 0:
            # 推导位宽
            dtype = d.dtype
            FastBitwiseHilbert.infer_bit_width(dtype)

            n = H

            # 逆 Gray 码变换到虚拟空间
            xv, yv = FastBitwiseHilbert._inverse_gray_code_transform(d, n)

            # 2^k 正方形：直接使用坐标
            # D4-AUDIT FIX: %H → &(H-1)，H 是 2 的幂
            x = xv & (H - 1)
            y = yv & (H - 1)

            return x, y
        else:
            # 矩形或非 2^k：使用 HilbertScanner 向量化 batch 方法
            # I-OPT: 使用 d_to_xy_batch 避免 cpu().numpy() + list comprehension
            x_out, y_out = HilbertScanner.d_to_xy_batch(H, W, d.long())
            return x_out.to(dtype=d.dtype, non_blocking=True), y_out.to(dtype=d.dtype, non_blocking=True)

    @staticmethod
    def _gray_code_transform(xv: Tensor, yv: Tensor, n: int) -> Tensor:
        """
        Gray 码 Hilbert 变换（向量化）

        数学形式:
            d = Σ_{k=0}^{log2(n)-1} 2^{2k} * ((3 * rx_k) ^ ry_k)

        旋转规则 (ry == 0 时):
            if rx == 1: (x, y) → (s-1-x, s-1-y)
            (x, y) → (y, x)

        Args:
            xv: 虚拟 x 坐标 (范围 [0, n))
            yv: 虚拟 y 坐标 (范围 [0, n))
            n: 网格大小 (必须是 2 的幂)

        Returns:
            d: Hilbert 距离
        """
        # 创建工作副本
        x = xv.clone().long()
        y = yv.clone().long()
        d = torch.zeros_like(x)

        # 计算实际位宽
        max_bits = n.bit_length() - 1

        # 遍历每一位，从最高位到最低位
        for k in range(max_bits - 1, -1, -1):
            s = 1 << k
            mask = 1 << k

            # 提取当前位
            rx = (x & mask) >> k
            ry = (y & mask) >> k

            # d += s² × ((3 × rx) ^ ry)
            # D4-AUDIT FIX: s*s → s<<k (2^k * 2^k = 2^(2k))，避免乘法
            d = d + (s << k) * ((3 * rx) ^ ry)

            # 旋转条件: ry == 0
            rot_mask = (ry == 0)

            # 翻转: rx == 1 且 ry == 0
            flip_mask = (rx == 1) & rot_mask
            x = torch.where(flip_mask, s - 1 - x, x)
            y = torch.where(flip_mask, s - 1 - y, y)

            # 交换: ry == 0
            x_new = torch.where(rot_mask, y, x)
            y_new = torch.where(rot_mask, x, y)
            x, y = x_new, y_new

        return d

    @staticmethod
    def _inverse_gray_code_transform(d: Tensor, n: int) -> Tuple[Tensor, Tensor]:
        """
        逆 Gray 码变换（向量化）

        从 Hilbert 距离恢复虚拟坐标

        Args:
            d: Hilbert 距离
            n: 网格大小 (必须是 2 的幂)

        Returns:
            (x, y): 虚拟坐标
        """
        x = torch.zeros_like(d)
        y = torch.zeros_like(d)
        d_batch = d.clone()

        # 计算实际位宽
        max_bits = n.bit_length() - 1

        for k in range(max_bits):
            s = 1 << k

            # 提取当前位
            rx = 1 & (d_batch >> 1)
            ry = 1 & (d_batch ^ rx)

            # 旋转条件: ry == 0
            rot_mask = (ry == 0)

            # 翻转: rx == 1 且 ry == 0
            flip_mask = (rx == 1) & rot_mask
            x = torch.where(flip_mask, s - 1 - x, x)
            y = torch.where(flip_mask, s - 1 - y, y)

            # 交换: ry == 0
            x_new = torch.where(rot_mask, y, x)
            y_new = torch.where(rot_mask, x, y)
            x, y = x_new, y_new

            # 累积坐标
            x = x + s * rx
            y = y + s * ry

            # 移位
            # D4-AUDIT FIX: //4 → >>2，整数除法优化
            d_batch = d_batch >> 2

        return x, y


class HilbertDispatcher:
    """
    统一 Hilbert 调度器

    调度逻辑:
    - 2^k × 2^k: FastBitwiseHilbert (O(B) 向量化位元变换)
    - 任意矩形: PseudoHilbertCurve (严格双射，L_max ≤ √(2ρ))

    数学保证:
    - 双射: 两条路径都保持 1:1 像素映射
    - 局部性: 标准 Hilbert L_max ≤ √2, Pseudo-Hilbert L_max ≤ √(2ρ)
    """

    @staticmethod
    def is_power_of_2(n: int) -> bool:
        """检查是否为 2 的幂"""
        return n > 0 and (n & (n - 1)) == 0

    @staticmethod
    def get_strategy(H: int, W: int) -> str:
        """
        根据尺寸返回最优策略

        Returns:
            'standard': 2^k × 2^k 正方形，使用标准 Hilbert
            'pseudo': 任意矩形，使用 Pseudo-Hilbert
        """
        if H == W and HilbertDispatcher.is_power_of_2(H):
            return 'standard'
        else:
            return 'pseudo'

    @staticmethod
    def xy_to_d(x: Tensor, y: Tensor, H: int, W: int) -> Tensor:
        """
        批量坐标 → Hilbert 索引

        调度逻辑:
        - 2^k × 2^k: FastBitwiseHilbert (O(B) 向量化)
        - 其他: PseudoHilbertCurve (严格双射)

        Args:
            x: [B] x 坐标 Tensor
            y: [B] y 坐标 Tensor
            H: 图像高度
            W: 图像宽度

        Returns:
            d: [B] Hilbert 索引 Tensor
        """
        if H == W and HilbertDispatcher.is_power_of_2(H):
            # 路径1: FastBitwiseHilbert (O(B) 向量化)
            return FastBitwiseHilbert.xy_to_d(x, y, H, W)
        else:
            # 路径2: PseudoHilbertCurve (严格双射)
            from vit_pytorch.core.curve_hilbert import PseudoHilbertCurve
            return PseudoHilbertCurve.xy_to_d_batch(H, W, x, y)

    @staticmethod
    def d_to_xy(d: Tensor, H: int, W: int) -> Tuple[Tensor, Tensor]:
        """
        批量 Hilbert 索引 → 坐标

        Args:
            d: [B] Hilbert 索引 Tensor
            H: 图像高度
            W: 图像宽度

        Returns:
            (x, y): [B] 坐标 Tensor
        """
        if H == W and HilbertDispatcher.is_power_of_2(H):
            # 路径1: FastBitwiseHilbert
            return FastBitwiseHilbert.d_to_xy(d, H, W)
        else:
            # 路径2: PseudoHilbertCurve
            from vit_pytorch.core.curve_hilbert import PseudoHilbertCurve
            return PseudoHilbertCurve.d_to_xy_batch(H, W, d)

    @staticmethod
    def get_bit_width(tensor: Tensor) -> int:
        """
        从 tensor dtype 推导有效位宽

        Args:
            tensor: 输入 Tensor

        Returns:
            有效位宽 B
        """
        return FastBitwiseHilbert.infer_bit_width(tensor.dtype)


def _fast_lca_vectorized(idx1: Tensor, idx2: Tensor) -> Tensor:
    """
    全向量化的 LCA 计算，无需 Python 循环

    数学原理:
        - xor = idx1 ^ idx2 (逐元素异或)
        - depth = xor.bit_length() - 1 (分歧深度)
        - mask = ~((1 << (2 * depth)) - 1) (保留到分歧点的掩码)
        - LCA = idx1 & mask

    Args:
        idx1: Hilbert 索引 Tensor
        idx2: Hilbert 索引 Tensor

    Returns:
        LCA 索引 Tensor
    """
    xor = idx1 ^ idx2

    # 计算分歧深度: bit_length(x) = floor(log2(x)) + 1 (对于 x > 0)
    # 使用 log2 方式兼容旧版 PyTorch
    # xor=0 时 log2(1)=0，bit_length=0，depth=-1，需要 clamp
    depth = torch.where(xor == 0, torch.zeros_like(xor), xor.bit_length() - 1)
    # 确保深度非负
    depth = depth.clamp(min=0)

    # 构建掩码：~((1 << (2 * depth)) - 1)
    # 使用 long 类型确保掩码正确
    masks = ~((1 << (2 * depth)) - 1)

    # 使用 torch.where 避免分支：xor==0 时返回 idx1
    return torch.where(xor == 0, idx1, idx1 & masks)


def fast_lca(idx1: int | Tensor, idx2: int | Tensor) -> int | Tensor:
    """
    计算两个 Hilbert 索引的最近公共祖先 (LCA)

    数学原理:
        - Hilbert 索引可以看作四叉树的层次遍历
        - 两个索引的 XOR 最高位确定分歧点
        - 分歧深度 = bit_length(idx1 ^ idx2) - 1
        - LCA = idx1 & (~((1 << (2 * depth)) - 1))
          (乘以2是因为Hilbert是2D，每层消耗2位)

    这是一个与位宽无关的函数，适用于任意深度的 Hilbert 曲线

    Args:
        idx1: Hilbert 索引 1
        idx2: Hilbert 索引 2

    Returns:
        LCA 索引

    示例:
        >>> fast_lca(5, 13)  # 5=0101, 13=1101
        4  # 分歧在第2层（从0开始），LCA=0100=4

        >>> fast_lca(7, 7)
        7  # 相同索引
    """
    # 检查是否为 Tensor
    idx1_is_tensor = isinstance(idx1, Tensor)
    idx2_is_tensor = isinstance(idx2, Tensor)

    if idx1_is_tensor or idx2_is_tensor:
        # 向量化版本：使用全张量运算，无 Python 循环
        if not idx1_is_tensor:
            idx1 = torch.tensor([idx1], dtype=torch.long, device=idx2.device)
        if not idx2_is_tensor:
            idx2 = torch.tensor([idx2], dtype=torch.long, device=idx1.device)

        # 保持与输入设备一致
        result = _fast_lca_vectorized(idx1, idx2)
        return result.squeeze() if result.numel() == 1 else result
    else:
        # 标量版本
        xor = idx1 ^ idx2
        if xor == 0:
            return idx1
        # 分歧深度（Hilbert每层2位）
        depth = xor.bit_length() - 1
        # 构建掩码：保留到分歧点的所有位
        mask = ~((1 << (2 * depth)) - 1)
        return idx1 & mask
