"""
T10 Splitter Gradient Flow — 正式 L2 回归测试 (seed-agnostic by construction)

本测试从 R7 集成期间的诊断脚手架 ``tests/test_t10_splitter_grad.py``
迁移到 L2_components/ 正式位置, 并进行两项关键强化:

强化 1 — seed-agnostic by construction (吸收 review 建议):
    原 fixture ``default_splitter`` 烘入 ``torch.manual_seed(0)``, 单一
    seed 不足以证明测试结构稳健。本测试把 seed 提升为
    ``@pytest.fixture(params=T10_SEEDS)`` 参数化, 3 个 grad 测试自动
    展开为 7 × 3 = 21 个 sub-tests。任何 seed 失败, pytest 报告精确定位
    (e.g. ``test_xxx[12345]``), 取代外部 sed 循环对工作区的污染。

强化 2 — hard-zero 防御 (吸收 review 建议):
    单一 ``> 1e-6`` 阈值不足以捕获 grad 接通但被 ``X * 0.0`` 链路退化为
    物理零的静默失效 (stealth break)。组合断言升级为::

        assert p.grad is not None          # 接通
        assert p.grad.abs().sum() > 0.0    # 非物理零 (hard-zero 防御)
        assert p.grad.abs().sum() > 1e-6   # 数值健康

    Gumbel-STE softmax 塌缩到 near-one-hot 时, top-K 选中位置仍走 hard
    mask 旁路, 非零, 所以 ``> 0.0`` 不会产生 false positive。

设计意图:
    - 本测试只检查**图结构**: STE 路径是否接通, 7 个核心参数是否获得
      非零梯度。
    - **不在**测试里检查数值健康度 (HEALTHY/MARGINAL/NUMERICAL_COLLAPSE);
      那是训练时的数值问题, 应在独立 callback 或 stress test 里检查。
    - 5 个路由通路参数 (rot_proj/area_proj/roi_norm/geo_norm/conv1d_hilbert)
      在生产 R7 配置 (auxiliary_losses = None) 下结构上断流, 属设计限制;
      本测试在 splitter 独立路径下用 ``logits.sum() + probs.sum()`` loss
      验证它们获得强梯度, 证明 STE 链路本身完整。

对应 IMPROVEMENT_PLAN.md:
    - §一 I165-3: 验证 STE 链路, 7 个核心参数获得非零梯度。
    - §四 R7 design: T10 L2 回归测试实现。
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from vit_pytorch.layers.splitters.hilbert_optimal_splitter import (
    HilbertOptimalSplitter,
)


# 多 seed 验证: 7 seeds 覆盖小/中/大数, 含紧邻对照 (0/1 易暴露种子消耗顺序敏感)
T10_SEEDS = (0, 1, 7, 42, 100, 12345, 99999)


# T10: 在 splitter 独立路径下通过 logits+probs loss 接通的 2 个核心表示参数
T10_NAMED_PARAM_PREFIXES = (
    'feature_proj',
    'depth_embedding',
)

# 路由通路参数: 在 splitter 独立路径下通过 conv1d_hilbert → logits 接通
T10_ROUTING_PARAM_PREFIXES = (
    'rot_proj',
    'area_proj',
    'roi_norm',
    'geo_norm',
    'conv1d_hilbert',
)


def collect_trainable_params(
    splitter: HilbertOptimalSplitter,
) -> "OrderedDict[str, torch.nn.Parameter]":
    """枚举所有 requires_grad=True 的 leaf Parameter (按命名顺序)."""
    seen: set = set()
    out: "OrderedDict[str, torch.nn.Parameter]" = OrderedDict()
    for name, p in splitter.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen:
            continue
        seen.add(id(p))
        out[name] = p
    return out


def _assert_grad_connected_and_nonzero(
    name: str, p: torch.nn.Parameter, seed: int
) -> None:
    """组合断言: grad 接通 + 非物理零 + 数值健康。

    Args:
        name: 参数名 (用于错误信息)
        p: leaf Parameter
        seed: 当前 parametrize seed (用于错误信息定位)
    """
    assert p.grad is not None, (
        f"[seed={seed}] {name} grad is None — STE 链路断裂"
    )
    grad_abs_sum = p.grad.abs().sum().item()
    # Hard-zero 防御: 任何 bit-level 零化都触发 (stealth break 防护)
    assert grad_abs_sum > 0.0, (
        f"[seed={seed}] {name} grad is hard zero (stealth break) — "
        f"grad 接通但被 X * 0.0 链路退化为物理零"
    )
    # 数值健康阈值: 主路径接通后物理梯度稳定在 1e-4 ~ 1e-2 量级
    assert grad_abs_sum > 1e-6, (
        f"[seed={seed}] {name} 梯度 L1={grad_abs_sum:.4e} ≤ 1e-6, "
        f"虽然非零但过弱, 可能处于数值下溢状态"
    )


def _require_grad(
    name: str, p: torch.nn.Parameter, seed: int
) -> torch.Tensor:
    """断言 p.grad 非 None 并返回类型收窄后的 Tensor。

    与 _assert_grad_connected_and_nonzero 不同: 此 helper 只做**结构性**
    断言 (grad 接通), 不做数值健康度检查。用于 (b) 类"至少非零"循环
    而非 (a) 类"必须强梯度"断言。

    Args:
        name: 参数名 (用于错误信息)
        p: leaf Parameter
        seed: 当前 parametrize seed (用于错误信息定位)

    Returns:
        p.grad (类型收窄为 Tensor)
    """
    assert p.grad is not None, (
        f"[seed={seed}] {name} grad is None — STE 链路断裂"
    )
    return p.grad


def _compute_splitter_loss(out) -> torch.Tensor:
    """从 SplitResult 构造 splitter 独立路径 loss: logits.sum() + probs.sum().

    该 loss 直接消费 splitter 内部的 logits 和 probs, 验证 STE 链路完整
    (即 7 个核心参数都能获得强梯度)。在生产 R7 配置
    (auxiliary_losses = None) 下, 这条路径不被使用, 但作为图结构健全
    性检查依然有效。

    Args:
        out: HilbertOptimalSplitter.forward() 返回的 SplitResult

    Returns:
        loss: scalar Tensor
    """
    # 类型收窄: SplitResult.logits/probs 字段类型是 Optional[Tensor],
    # 在 splitter 独立路径下两者必非 None, 用 assert 收窄为 Tensor。
    assert out.logits is not None, (
        "out.logits must be non-None after HilbertOptimalSplitter.forward()"
    )
    assert out.probs is not None, (
        "out.probs must be non-None after HilbertOptimalSplitter.forward()"
    )
    return out.logits.float().sum() + out.probs.float().sum()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(params=T10_SEEDS)
def splitter_seed(request) -> int:
    """parametrize seed — 7 个 seed 覆盖各种随机模式。"""
    return request.param


@pytest.fixture
def default_splitter(splitter_seed: int) -> HilbertOptimalSplitter:
    """生产配置 splitter, seed 来自 parametrize。

    feature_dim=256, hidden_dim=64, max_level=6, K_fixed=16 — 与 R7 集成
    期间使用的一致, 保证测试反映生产行为。
    """
    torch.manual_seed(splitter_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(splitter_seed)
    splitter = HilbertOptimalSplitter(
        feature_dim=256,
        hidden_dim=64,
        max_level_limit=6,
        K_fixed=16,
    )
    splitter.train()
    return splitter


@pytest.fixture
def batched_features() -> torch.Tensor:
    """B=2 验证 Batch 维度无 broadcasting 冲突。

    注: 不显式依赖 splitter_seed, 信任 default_splitter 先于
    batched_features 实例化已设过全局 RNG state (pytest fixture
    按签名顺序实例化)。
    """
    return torch.randn(2, 256, 16, 16)


# =============================================================================
# T10 测试用例
# =============================================================================


def test_t10_representation_params_have_strong_grad(
    splitter_seed: int,
    default_splitter: HilbertOptimalSplitter,
    batched_features: torch.Tensor,
) -> None:
    """T10 主断言 (表示通路): feature_proj + depth_embedding 必须获得强梯度。

    噪声底线 1e-6: 主路径接通后物理梯度稳定在 1e-4 ~ 1e-2 量级,
    1e-8 容易受 FP32/AMP 数值下溢干扰。

    注: 本测试只对 2 个**表示通路**参数做硬断言。5 个**路由通路**参数
    (rot_proj/area_proj/roi_norm/geo_norm/conv1d_hilbert) 在 R7 契约
    (auxiliary_losses = None) 下结构性断流, 属符合预期的设计限制,
    由 test_t10_routing_params_have_strong_grad_in_splitter_path 单独
    在 splitter 独立路径下验证。
    """
    params = collect_trainable_params(default_splitter)
    for p in params.values():
        p.grad = None

    image_size = (224, 224)
    out = default_splitter.forward(
        batched_features, image_size=image_size, hard=False
    )
    loss = _compute_splitter_loss(out)
    loss.backward()

    for prefix in T10_NAMED_PARAM_PREFIXES:
        matching = [(n, p) for n, p in params.items() if n.startswith(prefix)]
        assert matching, f"未找到以 '{prefix}' 开头的参数"
        for name, p in matching:
            _assert_grad_connected_and_nonzero(name, p, splitter_seed)


def test_t10_routing_params_have_strong_grad_in_splitter_path(
    splitter_seed: int,
    default_splitter: HilbertOptimalSplitter,
    batched_features: torch.Tensor,
) -> None:
    """T10 路由通路 (splitter 独立路径): 5 个路由参数必须获得强梯度。

    数学依据:
        loss = logits.sum() + probs.sum() (splitter 独立 loss)
        路由参数路径: ... → combined → conv1d_hilbert → logits → loss
        此路径消费 logits, 因此 5 个路由参数都应获得强梯度。

    重要边界:
        本测试验证 splitter 内部 STE 链路完整, 不代表生产路径 (CE loss)
        下的行为。在 FractalCurveViT 完整模型 + CE loss 路径下, 这 5
        个参数结构上断流 (R7 契约 auxiliary_losses = None), 属设计
        限制, 详见 I165-3a。
    """
    params = collect_trainable_params(default_splitter)
    for p in params.values():
        p.grad = None

    out = default_splitter.forward(
        batched_features, image_size=(224, 224), hard=False
    )
    loss = _compute_splitter_loss(out)
    loss.backward()

    for prefix in T10_ROUTING_PARAM_PREFIXES:
        matching = [(n, p) for n, p in params.items() if n.startswith(prefix)]
        assert matching, f"未找到以 '{prefix}' 开头的路由参数"
        for name, p in matching:
            _assert_grad_connected_and_nonzero(name, p, splitter_seed)


@pytest.mark.xfail(
    reason=(
        "HilbertOptimalSplitter.forward() 当前不填充 roi_features 字段 (T10 fix 待实现, "
        "SplitResult 构造 line 803-814 缺 roi_features= 参数)。此测试反映期望协议, "
        "修复后移除 xfail 标记。"
    ),
    strict=True,
)
def test_t10_split_result_has_roi_features_field(
    default_splitter: HilbertOptimalSplitter,
    batched_features: torch.Tensor,
) -> None:
    """T10 协议层断言: SplitResult.roi_features 字段存在且 shape 匹配 hidden_dim。

    注: 此测试不显式使用 splitter_seed, 但通过 default_splitter
    间接被 7-seed parametrize 展开 (协议层 + seed 鲁棒性同时验证)。

    当前状态: xfail — production code 未填充 roi_features 字段,
    见 I165-3 后续工作。
    """
    out = default_splitter.forward(
        batched_features, image_size=(224, 224), hard=False
    )
    assert hasattr(out, 'roi_features'), (
        "SplitResult 必须暴露 roi_features 字段 (T10 fix)"
    )
    assert out.roi_features is not None, "roi_features 不应为 None"
    assert out.roi_features.shape[-1] == default_splitter.hidden_dim, (
        f"roi_features 末维 {out.roi_features.shape[-1]} ≠ hidden_dim "
        f"{default_splitter.hidden_dim}"
    )


@pytest.mark.xfail(
    reason=(
        "同 test_t10_split_result_has_roi_features_field — roi_features 字段未填充, "
        "本测试验证 feature_dim != hidden_dim 时的双字段降级路径也受影响。"
    ),
    strict=True,
)
def test_t10_backward_compat_feature_dim_ne_hidden_dim() -> None:
    """T10 向后兼容: feature_dim != hidden_dim 时降级路径生效。

    模拟 3 个 shape-mismatched 单元测试的配置:
    feature_dim=256, hidden_dim=64 走 roi_features_raw 路径。

    当前状态: xfail — roi_features 字段未填充, 本测试的双字段断言
    不能通过, 见 I165-3 后续工作。
    """
    torch.manual_seed(0)
    # 构造 feature_dim=128, hidden_dim=32 的"测试错配" splitter
    splitter = HilbertOptimalSplitter(
        feature_dim=128,
        hidden_dim=32,
        max_level_limit=4,
        K_fixed=8,
    )
    splitter.train()

    features = torch.randn(2, 128, 16, 16)
    out = splitter.forward(features, image_size=(224, 224), hard=False)

    # 协议层: roi_features 仍存在 (post-projection, hidden_dim=32)
    assert out.roi_features is not None
    assert out.roi_features.shape[-1] == 32

    # 协议层: roi_features_raw 仍存在 (pre-projection, feature_dim=128)
    assert out.roi_features_raw is not None
    assert out.roi_features_raw.shape[-1] == 128

    # 关键: 两个字段维度不同, tokenizer 应走 roi_features_raw fallback 路径
    assert out.roi_features.shape[-1] != out.roi_features_raw.shape[-1]


def test_t10_100pct_param_grad_coverage(
    splitter_seed: int,
    default_splitter: HilbertOptimalSplitter,
    batched_features: torch.Tensor,
) -> None:
    """T10 全覆盖断言: 2 个表示通路参数 100% 强梯度(>1e-6), 其他参数非零即可。

    注: scalar scale factor (如 logit_scale) 经 softmax 饱和区后,
    物理梯度稳定在 1e-7 量级属于正确行为, 不强求 1e-6。
    """
    params = collect_trainable_params(default_splitter)
    for p in params.values():
        p.grad = None

    out = default_splitter.forward(
        batched_features, image_size=(224, 224), hard=False
    )
    loss = _compute_splitter_loss(out)
    loss.backward()

    # (a) 2 个表示通路参数必须 100% 强梯度 (>1e-6)
    core_strong = 0
    core_total = 0
    for prefix in T10_NAMED_PARAM_PREFIXES:
        for name, p in params.items():
            if not name.startswith(prefix):
                continue
            core_total += 1
            _assert_grad_connected_and_nonzero(name, p, splitter_seed)
            # _assert_grad_connected_and_nonzero 已断言 grad 非 None
            # 再用 _require_grad 拿收窄后的 Tensor, 避免 IDE 报警
            g = _require_grad(name, p, splitter_seed)
            l1 = g.abs().sum().item()
            if l1 > 1e-6:
                core_strong += 1

    assert core_strong == core_total, (
        f"[seed={splitter_seed}] 表示通路参数中仅 "
        f"{core_strong}/{core_total} 强梯度(>1e-6). "
        f"STE 链路对核心表示参数存在弱化."
    )

    # (b) 所有可训练参数至少非零 (弱梯度也算)
    # _require_grad 返回 Tensor 类型, IDE 收窄通过
    total = len(params)
    nonzero = 0
    for name, p in params.items():
        g = _require_grad(name, p, splitter_seed)
        if g.abs().sum().item() > 0:
            nonzero += 1
    assert nonzero == total, (
        f"[seed={splitter_seed}] 仅 {nonzero}/{total} splitter 参数有非零梯度, "
        f"{(total - nonzero)} 个完全断流."
    )
