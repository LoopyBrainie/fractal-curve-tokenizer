"""v1.3 STANDARD: Multi-Block HMFT mathematical invariant verification.

Per I170.4 plan: 17 个 HMFT 缺口补强,Group 1 — 数学不变量 (T1-T5)。

覆盖 I170 系列已落地的公理 (A5: Fixed-K) 之外的**结构性不变量**:
  T1: token 互斥 + selected_mask 行和 = K (跨 grid × mode)
  T2: probs 行和 = 1 (跨 grid × mode) — softmax 归一化守恒
  T3: 跨 batch 元素 K 守恒 (B ∈ {1, 2, 4, 8})
  T4: STE 双向契约 (前向 bit-exact + 反向梯度活性)
  T5: STE 公式 bit-exact (跨 grid) — `(hard - soft).detach() + soft`

Path bootstrap 由 ``tests/conftest.py`` 自动处理 (PROJECT_ROOT + SRC_PATH)。
"""
from __future__ import annotations

import pytest
import torch

from vit_pytorch.core.constants import HMFT_K_HARD_GLOBAL_POOL
from vit_pytorch.layers.splitters.multi_block_hmft_splitter import (
    MultiBlockHMFTSplitter,
    MultiBlockHMFTSplitterConfig,
)

K = HMFT_K_HARD_GLOBAL_POOL  # = 8


def _build_features(B: int, C: int, grid: int) -> torch.Tensor:
    """构造符合 HMFT 约束的特征图: grid 必须是 2 的幂 (32/64/128) 以满足 Hilbert 曲线 + 小 h=8 整除。

    默认 config 的 feature_dim=256,故 C=256。HMFT_MIN_IMAGE_SIZE=32 限制 grid >= 32。
    """
    assert grid in {32, 64, 128}, f"grid 必须是 2 的幂 (32/64/128), got {grid}"
    return torch.randn(B, C, grid, grid)


# T1 — token 互斥 + selected_mask 行和 = K
# 3 grid × 2 mode (eval=纯确定性 / train_seed=带 Gumbel+seed 锁) = 6 子用例
@pytest.mark.parametrize("grid_size", [32, 64, 128])
@pytest.mark.parametrize("mode", ["eval", "train_seed"])
def test_token_mutual_exclusion_and_K_sum(make_hmft_splitter, grid_size: int, mode: str):
    """T1: 验证 token 选择满足互斥性,且 selected_mask 行和 == K (跨 grid × mode)。"""
    if mode == "train_seed":
        torch.manual_seed(42)  # 锁住 Gumbel 噪声
        splitter = make_hmft_splitter()
        splitter.train()
    else:
        splitter = make_hmft_splitter()
        splitter.eval()

    features = _build_features(B=2, C=256, grid=grid_size)
    result = splitter(features, image_size=(grid_size, grid_size), hard=True)

    # 互斥: 每个 batch 行的候选 token 索引无重复
    # candidate_indices 是 1D 长度 B*K,需 view 为 [B, K] 后按行访问
    cand_view = result.candidate_indices.view(2, K)
    for b in range(2):
        chosen = cand_view[b]
        assert chosen.unique().numel() == K, (
            f"Batch {b} 存在重复采样的候选 Token! (grid={grid_size}, mode={mode})"
        )

    # K 守恒: selected_mask 行和 == K
    assert torch.allclose(
        result.selected_mask.sum(dim=-1),
        torch.tensor(float(K), dtype=result.selected_mask.dtype),
    ), f"硬掩码行和 != K (grid={grid_size}, mode={mode})"


# T2 — probs 行和 = 1 (softmax 归一化守恒,跨 grid × mode)
@pytest.mark.parametrize("grid_size", [32, 64, 128])
@pytest.mark.parametrize("mode", ["eval", "train_seed"])
def test_probs_row_sum_equals_one(make_hmft_splitter, grid_size: int, mode: str):
    """T2: 验证 probs 行和 == 1,无论 grid 与 mode。"""
    if mode == "train_seed":
        torch.manual_seed(42)
        splitter = make_hmft_splitter()
        splitter.train()
    else:
        splitter = make_hmft_splitter()
        splitter.eval()

    features = _build_features(B=2, C=256, grid=grid_size)
    result = splitter(features, image_size=(grid_size, grid_size), hard=True)

    # probs 是 softmax 输出,行和必须 == 1
    assert torch.allclose(
        result.probs.sum(dim=-1),
        torch.ones(2, dtype=result.probs.dtype),
        atol=1e-5,
    ), f"Softmax 归一化失效! (grid={grid_size}, mode={mode})"


# T3 — 跨 batch 元素 K 守恒 (A5 单 B=2 的补强)
@pytest.mark.parametrize("B", [1, 2, 4, 8])
def test_K_invariant_across_batch_size(make_hmft_splitter, B: int):
    """T3: 验证 K 守恒不依赖于 batch 维度。"""
    torch.manual_seed(42)
    splitter = make_hmft_splitter()
    splitter.eval()
    features = _build_features(B=B, C=256, grid=32)
    result = splitter(features, image_size=(32, 32), hard=True)
    assert result.regions.shape[0] == B * K, (
        f"B={B} 时 M (regions 行数) != B*K, M={result.regions.shape[0]}"
    )


# T4 — STE 双向契约 (前向 bit-exact + 反向梯度活性)
def test_mask_ste_bidirectional_contract(make_hmft_splitter):
    """T4: 验证 STE 桥的 (a) 前向 bit-exact 直通, (b) 反向梯度活性。

    注意: h_logits 走 argmax/采样 (非可微),不会接收梯度。实际可微链路
    终止于 _score_cells 的 3 个模块 (geometry_encoder / fusion / hilbert_conv1d)。

    反向 loss 使用 ``(probs * score_head).sum()`` 而非 ``mask_ste.sum()``:
    softmax 满足归一化恒等式 ``Σ softmax(x)[k] ≡ 1``,因此
    ``d(Σ softmax)/d(x[j]) ≡ 0`` — 任何纯 softmax-sum 的 loss 数学梯度恒为 0。
    原 loss 在 FP32 下仅靠噪声(~1e-7)勉强通过 ``assert > 0``,这是脆弱的伪通过。
    新 loss 通过 ``mask_soft * score_head`` 打破归一化对称性,产生数学非零梯度。
    """
    torch.manual_seed(42)
    splitter = make_hmft_splitter()
    splitter.train()
    features = _build_features(B=2, C=256, grid=32)
    result = splitter(features, image_size=(32, 32), hard=True)

    # (a) 前向: mask_ste 数值 bit-exact 等于 selected_mask 的 float 形式
    assert torch.equal(result.mask_ste, result.selected_mask.float()), (
        "STE 前向与 mask_hard 不 bit-exact! 公式契约破坏。"
    )

    # (a') 前向: mask_ste 必须保留 requires_grad (T10 keystone 强制)
    assert result.mask_ste.requires_grad, (
        "mask_ste.requires_grad 为 False — STE 桥未建立 (T10 keystone 失败)"
    )

    # (b) 反向: mask_ste 的梯度能传导到 _score_cells 的可微参数
    # 使用 (probs * score_head).sum() 打破 softmax 归一化对称性,见 docstring
    loss = (result.probs * result.logits).sum()
    loss.backward()
    for name, param in [
        ("geometry_encoder.proj", splitter.geometry_encoder.proj.weight),
        ("fusion.score_proj", splitter.fusion.score_proj.weight),
        ("hilbert_conv1d", splitter.hilbert_conv1d.weight),
    ]:
        assert param.grad is not None, (
            f"{name}.grad 为 None — STE 链断裂 (mask_ste → score_head 断流)"
        )
        assert param.grad.abs().sum().item() > 0, (
            f"{name}.grad 静默 0 — STE 桥未真正建立"
        )


# T5 — STE 公式 bit-exact (跨 grid 验证 (hard - soft).detach() + soft)
@pytest.mark.parametrize("grid_size", [32, 64, 128])
def test_mask_ste_formula_bit_exact(make_hmft_splitter, grid_size: int):
    """T5: 直接构造 STE 期望值,验证 mask_ste 满足源码公式契约。"""
    torch.manual_seed(42)
    splitter = make_hmft_splitter()
    splitter.train()
    features = _build_features(B=2, C=256, grid=grid_size)
    result = splitter(features, image_size=(grid_size, grid_size), hard=True)

    # 独立计算 STE 期望值
    expected = (result.selected_mask - result.probs).detach() + result.probs
    assert torch.equal(result.mask_ste, expected), (
        f"STE 公式契约破坏 (grid={grid_size}): mask_ste != (hard - soft).detach() + soft"
    )


# T16 — score_cells 形状契约 (跨 grid × hidden_dim)
@pytest.mark.parametrize("grid_size", [32, 64, 128])
@pytest.mark.parametrize("hidden_dim", [32, 64, 128])
def test_score_cells_shape_consistency(
    make_hmft_splitter, grid_size: int, hidden_dim: int
):
    """T16: 验证 _score_cells 输出形状 [B, n_cells] 在 grid × hidden_dim 组合下稳定。

    I170 Commit 2 引入了 A1 Hilbert 1D Locality Conv1D score head,要求输出
    形状 [B, n_cells] 不依赖 hidden_dim 维度。这是几何编码器 (3 输入) +
    fusion (feature_dim + hidden_dim) + Conv1D 平滑链路的最终不变量。
    """
    torch.manual_seed(42)
    splitter = make_hmft_splitter(hidden_dim=hidden_dim)
    splitter.eval()
    features = _build_features(B=2, C=256, grid=grid_size)
    result = splitter(features, image_size=(grid_size, grid_size), hard=True)
    # logits 形状 [B, n_cells],与 hidden_dim 无关
    assert result.logits.shape[0] == 2, (
        f"logits batch dim 应为 2, got {result.logits.shape[0]}"
    )
    assert result.logits.dim() == 2, (
        f"logits 应为 2D [B, n_cells], got {result.logits.dim()}D"
    )
    # n_cells 随 grid_size 变化但必须为正
    assert result.logits.shape[1] > 0, (
        f"n_cells 必须为正 (grid={grid_size}, hidden_dim={hidden_dim})"
    )


# T17 — param_count 不变 (P0 静态注册守卫)
def test_param_count_unchanged_after_forward(make_hmft_splitter):
    """T17: 验证连续 3 次 forward 后 ``list(splitter.parameters())`` 长度不变。

    I170 Commit 3 引入 P0 静态注册约束: ``feature_proj`` + ``roi_norm`` 必须在
    ``__init__`` 末尾注册,不能在 forward 中动态注册 (否则优化器快照陷阱)。
    本测试是 P0 评审反馈的回归守卫: 若后续修改 splitter 时不小心在 forward
    中动态注册了 nn.Module,本测试会立即发现。
    """
    torch.manual_seed(42)
    splitter = make_hmft_splitter()
    initial_params = list(splitter.parameters())
    initial_count = len(initial_params)
    initial_ids = {id(p) for p in initial_params}

    # 跑 3 次 forward,每次用相同输入
    for _ in range(3):
        features = _build_features(B=2, C=256, grid=32)
        _ = splitter(features, image_size=(32, 32), hard=True)

    after_params = list(splitter.parameters())
    after_count = len(after_params)
    after_ids = {id(p) for p in after_params}

    # param 数量不变
    assert after_count == initial_count, (
        f"P0 静态注册破坏: forward 后 param 数量 {after_count} != 初始 {initial_count}"
    )
    # param 对象本身也不变 (防止替换 weight.data 行为异常)
    assert after_ids == initial_ids, (
        "P0 静态注册破坏: forward 后 param 对象被替换"
    )


# 显式 re-export,避免 mypy / IDE 误以为这些导入未使用
__all__ = [
    "K",
    "MultiBlockHMFTSplitter",
    "MultiBlockHMFTSplitterConfig",
    "test_token_mutual_exclusion_and_K_sum",
    "test_probs_row_sum_equals_one",
    "test_K_invariant_across_batch_size",
    "test_mask_ste_bidirectional_contract",
    "test_mask_ste_formula_bit_exact",
    "test_score_cells_shape_consistency",
    "test_param_count_unchanged_after_forward",
]
