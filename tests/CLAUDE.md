# tests/CLAUDE.md

> 🔄 **Reconstruction in progress** — for binding workflow rules see root `CLAUDE.md` § "Reconstruction Workflow".

测试目录的 Claude Code 指引。本文件**只涵盖测试特定约定**,通用开发规范见根
[`CLAUDE.md`](../CLAUDE.md)。

## 1. 路径与目录结构

- **`src/`** 自动加入 `sys.path` (由 `tests/conftest.py:41-48` 处理)
- **测试层级**: 测试目录按根 `CLAUDE.md` §Module Hierarchy 的 L1→L4 分层命名
- **conftest 继承**:
  - `tests/conftest.py` — 项目级 fixture (`seeded`, `seeded_rng`, `device`, `default_splitter_config` 等)
  - `tests/unit/conftest.py` — L1-L4 单元测试共享 fixture
  - `tests/unit/L2_components/splitters/conftest.py` — splitter 专属 fixture (`make_hmft_splitter` 等)
  - **就近原则**: 越具体的 conftest 越优先;测试文件应优先 import **最近** 的 conftest
- **deprecated_v13/**: 旧版 v13 测试,只用于历史追溯,不在 CI 中运行

## 2. Pytest 约定

### 运行命令

(快速 sanity check 见根 `CLAUDE.md`)

```bash
uv run pytest tests/ -v                              # 全套
uv run pytest tests/unit/L2_components/splitters/    # 单目录
```

### Fixture 命名约定

- 工厂 fixture 用 `make_<name>` (如 `make_hmft_splitter`)
- 单一 fixture 用名词 (如 `seeded`, `device`, `default_splitter_config`)
- 配置字典 fixture 用 `<scope>_config` 后缀 (如 `default_splitter_config`)

### 测试文件命名约定

- 文件名格式: `test_<issue_id>_<feature>.py` (如 `test_i170_4_ste_bridge.py`)
- `<issue_id>` 对应 `IMPROVEMENT_PLAN.md` 或 git history 中的 Issue 编号
- `<feature>` 是该 Issue 修复/实现的核心特性简写

### 断言风格

- **优先使用描述性错误消息**: `assert ..., f"{grid=} 触发 A5 守卫..."`
- **shape 检查**: 用 `(B, n_cells, dim)` 而非 `(2, 16, 256)`
- **数值检查**: `torch.allclose` + `atol/rtol`;严格相等用 `torch.equal` (bit-exact 测试)
- **数值稳定性**: 使用 `vit_pytorch.core.constants.EPS`(见根 `CLAUDE.md` §Conventions),不写 magic number

## 3. ⚠️ STE 桥梯度活性测试约定 (I170.4 重要 fix)

> **历史教训**: `test_a1_gradient_flows_to_geometry_modules` 和
> `test_mask_ste_bidirectional_contract` (T4) 历史上因 **softmax 归一化恒等式**
> 出现伪通过 — 数学梯度恒为 0,仅靠 FP32 subnormal 噪声 (~1e-7) 通过 `assert > 0`。
> 见 `C:\Users\LamKo\.claude\plans\test-a1-gradient-flows-to-geometry-modul-fluffy-moler.md`
> 获取完整诊断。

### 数学恒等式 (反模式根源)

$$\sum_k \text{softmax}(x)[k] \equiv 1 \implies \frac{\partial}{\partial x[j]} \sum_k \text{softmax}(x)[k] \equiv 0$$

**任何"纯 softmax-sum"的 loss 数学梯度恒为 0**,包括:
- `loss = mask_ste.sum()`            ← 仓库历史用法 ❌
- `loss = mask_soft.sum()`           ← ❌
- `loss = -mask_soft.log().sum()`    ← log 不打破归一化 ❌

### ✅ 正确约定: 使用 `tests/conftest.py::ste_gradient_loss`

```python
from tests.conftest import ste_gradient_loss

def test_my_ste_gradient(make_hmft_splitter):
    torch.manual_seed(42)
    splitter = make_hmft_splitter()
    splitter.train()
    features = torch.randn(2, 256, 32, 32)
    result = splitter(features, image_size=(32, 32), hard=True)

    # T10 keystone: STE 桥必须保留 requires_grad
    assert result.mask_ste.requires_grad, "T10 keystone 失败"

    # 使用 ste_gradient_loss helper 而非手写 (probs * logits).sum()
    ste_gradient_loss(result).backward()

    # 此时 3 模块的 .grad.abs().sum() 应该比 FP32 epsilon (1e-7) 大 7 个数量级
    assert splitter.hilbert_conv1d.weight.grad.abs().sum() > 0
```

### 配套使用约定 (缺一不可)

1. **`torch.manual_seed(42)`** — 锁定 4 个随机源(h_logits / multinomial / Gumbel / features)
2. **`splitter.train()`** — 启用 `_is_training_mode` 标志(虽然 hard=True 时不影响 Gumbel)
3. **`hard=True`** — 跳过 Gumbel 噪声注入,避免 max-over-N 极值在 FP32 下让 softmax 退化为 one-hot
4. **`mask_ste.requires_grad` 断言** — T10 keystone 强制,确保 STE 桥未断流

### 已修复的脆弱测试 (不要回退)

- ❌ `mask_ste.sum()` → ✅ `(probs * logits).sum()`:
  - `tests/unit/L2_components/splitters/test_multi_block_hmft_a1.py`
  - `tests/unit/L2_components/splitters/test_multi_block_hmft_math_invariants.py` (T4)

## 4. 调试与排错约定

### 梯度链诊断模板

当 `param.grad` 异常(如 `None`、`0.0`、或 subnormal)时,按以下顺序诊断:

```python
# 1. 验证 graph requires_grad 链路 (T10 keystone)
assert result.mask_ste.requires_grad, "T10 keystone 失败"

# 2. 验证中间张量有梯度
score_head.retain_grad()
mask_soft.retain_grad()

# 3. 验证 leaf 节点有梯度 (vs None)
assert splitter.xxx.weight.grad is not None

# 4. 检查梯度数量级 — subnormal (< 1e-30) 暗示 softmax 退化
print(f"grad.abs().sum() = {param.grad.abs().sum().item():.2e}")
```

### 已知陷阱

- **`.data = tensor`**: 绕过 autograd,破坏 Parameter 引用追踪 — 用 `.copy_()` 替代
- **`device` 不匹配**: `torch.tensor(...)` 默认 CPU,用 `device=param.device` 防御
- **Gumbel + softmax 退化**: Gumbel max-over-N 极值在 FP32 下会让 softmax 退化为 one-hot,
  `d(Σ softmax)/dx` 严格为 0 → 改用 `hard=True` 跳过 Gumbel
- **seeded RNG state 污染**: `pytest-randomly` 改变测试顺序时,RNG state 可能漂移,
  用 `torch.manual_seed(42)` 显式锁定;对于需要固定 h_logits 的测试,使用
  `splitter.h_logits.copy_(...)` 而非依赖 softmax 概率

## 5. 编写新测试的检查清单

- [ ] 测试位置正确?L1 (foundation) / L2 (components) / L3 (pipeline) / L4 (application)
- [ ] 工厂 fixture 就近使用?(`make_<name>`)
- [ ] 导入路径遵守 L1→L4 分层?(见根 `CLAUDE.md` §Import Rules)
- [ ] 数值稳定性?(见根 `CLAUDE.md` §Conventions)
- [ ] 梯度测试?用 `ste_gradient_loss` 而非 `mask_ste.sum()`
- [ ] 描述性错误消息?`assert ..., f"上下文: {grid_size=}, {mode=}, ..."`
- [ ] deterministic?`torch.manual_seed(42)` 在 train-mode 测试首行
- [ ] Parametrize 边界?验证 grid ∈ {32, 64, 128} (A5 守卫)

## 6. 引用

- 根 [`CLAUDE.md`](../CLAUDE.md) — L1-L4 分层、导入规则、`forward()` 契约
- `tests/conftest.py:329-361` — `ste_gradient_loss` 函数定义
- `IMPROVEMENT_PLAN.md:96, 105-110` — STE 桥修复历史
