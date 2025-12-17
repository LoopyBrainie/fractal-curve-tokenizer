# 第十章：测试与质量保证

项目采用了 `pytest` 框架进行全面的测试。

## 10.1 测试目录结构

```text
tests/
├── unit/                   # 单元测试：测试单个组件
│   ├── test_fractal_vit.py
│   ├── test_streaming_tokenizer.py
│   ├── test_attention.py
│   ├── test_feedforward.py
│   ├── test_hilbert.py
│   ├── test_positional.py
│   └── test_utils.py
├── integration/            # 集成测试：测试完整流程
│   ├── test_training.py
│   └── test_system.py
├── benchmarks/             # 性能基准测试
│   ├── benchmark_fractal_vit.py
│   └── compare_fractal_vs_standard.py
└── conftest.py             # Pytest 配置和 Fixtures
```

---

## 10.2 单元测试覆盖范围

### Tokenizer 测试 (`test_streaming_tokenizer.py`)
- ✅ 不同输入尺寸和 Batch Size 下的输出形状
- ✅ Hilbert 重排序的正确性
- ✅ V1 和 V2 的输出一致性
- ✅ 多尺度特征提取

### 模型测试 (`test_fractal_vit.py`)
- ✅ 前向传播形状正确性
- ✅ 不同 tokenizer_type 的支持
- ✅ Batch 处理（Padding 和 Masking）
- ✅ 位置编码注入

### 注意力测试 (`test_attention.py`)
- ✅ 不同 bias_mode 的正确性
- ✅ Low-Rank 分解精度
- ✅ Mask 有效性

### FFN 测试 (`test_feedforward.py`)
- ✅ SwiGLU 输出形状
- ✅ 层级自适应机制
- ✅ 不同 ffn_type 的支持

### Hilbert 测试 (`test_hilbert.py`)
- ✅ d ↔ (x,y) 双向映射
- ✅ 边界情况处理
- ✅ 缓存机制

### 工具测试 (`test_utils.py`)
- ✅ `extract_depths` 维度处理
- ✅ `normalize_levels_info` 升维
- ✅ `create_attention_mask` 向量化

---

## 10.3 运行测试

```bash
# 运行所有测试
uv run pytest tests/ -v

# 运行单元测试
uv run pytest tests/unit/ -v

# 运行特定测试文件
uv run pytest tests/unit/test_streaming_tokenizer.py -v

# 运行带覆盖率
uv run pytest tests/ --cov=vit_pytorch --cov-report=html
```

---

## 10.4 基准测试与评估

### 核心对比 (`compare_fractal_vs_standard.py`)

与标准 ViT 的公平对比：
- **对比对象**: `NextGenerationFractalViT` vs `StandardViT`
- **指标**: 参数量、显存占用、推理延迟、吞吐量

### 综合性能基准 (`benchmark_fractal_vit.py`)

深入分析各项性能指标：
- **Tokenizer 效率**: 纯 Tokenizer 的吞吐量 (img/s)
- **端到端性能**: 完整模型的前向/反向传播速度
- **显存分析**: 详细的显存占用分布

---

## 10.5 废弃模块测试

废弃模块的测试位于 `tests/unit/test_deprecated.py`：
- ✅ 延迟导入正确性
- ✅ DeprecationWarning 发出
- ✅ 功能完整性（向后兼容）

---

## 10.6 测试状态

| 测试类别 | 数量 | 状态 |
| :--- | :--- | :--- |
| 单元测试 | ~100 | ✅ 通过 |
| 集成测试 | ~20 | ✅ 通过 |
| 基准测试 | ~10 | ✅ 通过 |
| **总计** | **~130** | **✅ 全部通过** |

---

## 10.7 持续集成

建议的 CI/CD 配置：

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - run: uv run pytest tests/ -v --tb=short
```
