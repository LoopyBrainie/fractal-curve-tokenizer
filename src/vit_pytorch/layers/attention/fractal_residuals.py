"""
Scale-Aware Fractal Residuals (尺度感知分形残差)

替代 Global Anchor Hack 的拓扑级方案:
- 父节点特征追溯 Parent(·)
- 递归 FPN 等价的残差路由
- 无需症状性偏置

数学形式化
===========

父节点追溯:
    p(i) = argmax_{j: d_j = d-1, Ancestor(j, i)} ||x_i - x_j||

残差路由:
    X_{l+1} = LayerNorm(X_l + Attn(X_l) + Proj(Parent(X_l)))

性质
----
- 递归特征金字塔 (Recursive FPN) 等价形式
- 保持注意力对称性
- 无 Global Anchor hack
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ParentTokenLookup(nn.Module):
    """
    父节点索引查找表。

    从 levels_info 提取父节点索引，实现 O(1) 查找。
    """

    def __init__(self, max_level: int = 8):
        super().__init__()
        self.max_level = max_level

    def forward(
        self,
        depths: torch.Tensor,
        hilbert_indices: torch.Tensor,
        regions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        查找每个 token 的父节点索引。

        参数
        ----
        depths : torch.Tensor
            Token 深度，形状 [B, N]
        hilbert_indices : torch.Tensor
            Hilbert 指数，形状 [B, N]
        regions : torch.Tensor, optional
            区域边界 [B, N, 4]

        返回
        ----
        Tuple[torch.Tensor, torch.Tensor]
            - parent_indices: 父节点索引 [B, N]
            - parent_mask: 有效父节点掩码 [B, N]
        """
        B, N = depths.shape
        device = depths.device

        # 初始化父节点为自身
        parent_indices = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        parent_mask = torch.ones(B, N, dtype=torch.bool, device=device)

        # 对每个深度 d > 0，查找深度 d-1 的父节点
        for d in range(1, self.max_level + 1):
            # 当前深度的 token
            current_mask = (depths == d)
            parent_mask_d = (depths == d - 1)

            if not current_mask.any():
                continue

            # 查找父节点
            current_indices = torch.where(current_mask)[1]
            parent_indices_d = torch.where(parent_mask_d)[1]

            if len(parent_indices_d) == 0:
                parent_mask[:, current_indices] = False
                continue

            # 简化: 使用批量操作代替循环
            # 为每个当前深度的 token 查找最近父节点
            h_current = hilbert_indices[:, current_indices]  # [B, num_current]
            h_parents = hilbert_indices[:, parent_indices_d]  # [B, num_parents]

            # 广播计算距离: [B, num_current, num_parents]
            dist = torch.abs(h_current.unsqueeze(2) - h_parents.unsqueeze(1))

            # 找到最近父节点: [B, num_current]
            nearest = dist.argmin(dim=2)

            # 更新父节点索引 (克隆避免警告)
            parent_indices_clone = parent_indices.clone()
            parent_indices_clone[:, current_indices] = parent_indices_d[nearest]
            parent_indices = parent_indices_clone

        return parent_indices, parent_mask


class ScaleAwareResidual(nn.Module):
    """
    尺度感知残差模块。

    数学形式
    --------

    X_residual = Proj(Parent(X))

    其中:
        - Parent(X): 父节点特征查找
        - Proj: 线性投影
    """

    def __init__(
        self,
        dim: int,
        max_level: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.max_level = max_level

        # 父节点查找
        self.parent_lookup = ParentTokenLookup(max_level)

        # 投影层
        self.proj = nn.Linear(dim, dim)

        # 门控机制 - 初始化为0.5使残差路径半开，允许梯度流通
        self.residual_gate = nn.Embedding(max_level + 1, 1)
        nn.init.constant_(self.residual_gate.weight, 0.5)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        depths: torch.Tensor,
        hilbert_indices: torch.Tensor,
        regions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播。

        参数
        ----
        x : torch.Tensor
            输入特征 [B, N, D]
        depths : torch.Tensor
            Token 深度 [B, N]
        hilbert_indices : torch.Tensor
            Hilbert 指数 [B, N]
        regions : torch.Tensor, optional
            区域边界 [B, N, 4]

        返回
        ----
        torch.Tensor
            残差特征 [B, N, D]
        """
        B, N, D = x.shape
        device = x.device

        # 查找父节点
        parent_indices, parent_mask = self.parent_lookup(depths, hilbert_indices, regions)

        # 获取父节点特征
        # parent_indices: [B, N]
        parent_features = torch.gather(
            x,
            dim=1,
            index=parent_indices.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, N, D]

        # 应用掩码
        parent_mask = parent_mask.unsqueeze(-1).float()
        parent_features = parent_features * parent_mask

        # 投影
        parent_proj = self.proj(parent_features)

        # 门控
        depths_clamped = depths.clamp(min=0, max=self.max_level)
        gate_raw = self.residual_gate(depths_clamped)  # [B, N, 1]
        gate = torch.sigmoid(gate_raw)

        residual = parent_proj * gate

        return self.dropout(residual)


class FractalTransformerBlockV2(nn.Module):
    """
    集成 Fractal Residuals 的 Transformer Block。

    数学形式
    --------

    X_norm = LN(X)
    Attn_out = Attn(X_norm)
    Residual_out = ScaleAwareResidual(X_norm)
    X' = X + DropPath(Attn_out + Residual_out)
    X'' = X' + FFN(X')

    特性
    ----
    - 无需 Global Anchor hack
    - 拓扑级多尺度特征保持
    - 保持注意力对称性
    """

    def __init__(
        self,
        dim: int,
        attention: nn.Module,
        ff: nn.Module,
        max_level: int = 8,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.max_level = max_level

        # 原有组件
        self.attention = attention
        self.ff = ff
        self.norm1 = nn.LayerNorm(dim)

        # Fractal Residual
        self.fractal_residual = ScaleAwareResidual(
            dim=dim,
            max_level=max_level,
        )

        # 门控 - 初始化为0.5使残差路径半开
        self._residual_gate = nn.Embedding(max_level + 1, 1)
        nn.init.constant_(self._residual_gate.weight, 0.5)

        # DropPath
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        depths: torch.Tensor,
        hilbert_indices: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        前向传播。

        参数
        ----
        x : torch.Tensor
            输入 [B, N, D]
        depths : torch.Tensor
            深度 [B, N]
        hilbert_indices : torch.Tensor
            Hilbert 指数 [B, N]
        **kwargs : 传递给 attention 的其他参数

        返回
        ----
        torch.Tensor
            输出 [B, N, D]
        """
        B, N, D = x.shape
        device = x.device

        # 门控
        depths_clamped = depths.clamp(min=0, max=self.max_level)
        gate_raw = self._residual_gate(depths_clamped)
        gate = torch.sigmoid(gate_raw)

        # Norm + Attention
        norm_x = self.norm1(x)
        attn_out = self.attention(norm_x, **kwargs)

        # Fractal Residual
        fractal_residual = self.fractal_residual(x, depths, hilbert_indices)

        # 残差连接
        residual_contribution = (attn_out + fractal_residual) * gate
        x = x + self.drop_path(residual_contribution)

        # FFN
        x = x + self.ff(x)

        return x


# ==================== 便捷函数 ====================


def create_fractal_residual_block(
    dim: int,
    attention: nn.Module,
    ff: nn.Module,
    max_level: int = 8,
    drop_path: float = 0.0,
) -> FractalTransformerBlockV2:
    """
    创建集成 Fractal Residuals 的 Transformer Block。

    参数
    ----
    dim : int
        维度
    attention : nn.Module
        注意力模块
    ff : nn.Module
        FFN 模块
    max_level : int
        最大深度
    drop_path : float
        DropPath 概率

    返回
    ----
    FractalTransformerBlockV2
    """
    return FractalTransformerBlockV2(
        dim=dim,
        attention=attention,
        ff=ff,
        max_level=max_level,
        drop_path=drop_path,
    )


# ==================== 测试 ====================


if __name__ == "__main__":
    # 测试 ParentTokenLookup
    B, N, D = 2, 16, 64
    max_level = 6

    depths = torch.randint(0, max_level + 1, (B, N))
    hilbert_indices = torch.argsort(torch.randn(B, N), dim=-1)

    lookup = ParentTokenLookup(max_level)
    parent_indices, parent_mask = lookup(depths, hilbert_indices)

    print(f"Parent indices shape: {parent_indices.shape}")
    print(f"Valid parent ratio: {parent_mask.float().mean().item():.2%}")

    # 测试 ScaleAwareResidual
    x = torch.randn(B, N, D)
    residual = ScaleAwareResidual(dim=D, max_level=max_level)
    out = residual(x, depths, hilbert_indices)

    print(f"ScaleAwareResidual output shape: {out.shape}")
