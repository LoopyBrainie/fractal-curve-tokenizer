# 第十章：测试与质量保证

## 10.1 概述

本项目使用 `pytest` 进行全面测试，涵盖单元测试、集成测试和性能基准测试。

---

## 10.2 测试目录结构

```
tests/
├── conftest.py                 # Pytest 夹具和配置
├── README.md                   # 测试文档
├── unit/                       # 单元测试
│   ├── test_fractal_vit.py
│   ├── test_streaming_tokenizer.py
│   ├── test_attention.py
│   ├── test_feedforward.py
│   ├── test_hilbert.py
│   ├── test_positional.py
│   └── test_utils.py
├── integration/                # 集成测试
│   ├── test_training.py
│   └── test_system.py
└── benchmarks/                 # 性能基准测试
    ├── benchmark_fractal_vit.py
    └── compare_fractal_vs_standard.py
```

---

## 10.3 单元测试覆盖

### 分词器测试（`test_streaming_tokenizer.py`）

| 测试 | 描述 |
|:-----|:------------|
| 输出形状验证 | 各种输入大小和批次大小 |
| Hilbert 重排序 | Hilbert 曲线排序的正确性 |
| V3 输出格式 | TokenizerOutput 结构验证 |
| 变深度分割 | 四叉树分割正确性 |
| 深度分布 | 验证多尺度 token 生成 |

### 模型测试（`test_fractal_vit.py`）

| 测试 | 描述 |
|:-----|:------------|
| 前向传播形状 | 输出维度正确性 |
| 分词器集成 | 不同分词器类型 |
| 批次处理 | 填充和掩码 |
| 位置编码 | 正确注入 |

### 注意力测试（`test_attention.py`）

| 测试 | 描述 |
|:-----|:------------|
| 偏置模式 | LCA、低秩、层次化 |
| LCA 参数计数 | 验证 ~100 个参数 |
| 低秩精度 | 分解精度 |
| 掩码有效性 | 注意力掩码应用 |

### FFN 测试（`test_feedforward.py`）

| 测试 | 描述 |
|:-----|:------------|
| SwiGLU 输出形状 | 维度保持 |
| 级别自适应 | 深度相关处理 |
| FFN 类型 | GELU、SwiGLU、SwiGLU+Level |

### Hilbert 测试（`test_hilbert.py`）

| 测试 | 描述 |
|:-----|:------------|
| d ↔ (x,y) 映射 | 双向正确性 |
| 边界情况 | 边缘坐标 |
| 缓存 | 索引缓存机制 |

### 工具测试（`test_utils.py`）

| 测试 | 描述 |
|:-----|:------------|
| `extract_depths` | 维度处理 |
| `normalize_levels_info` | 2D→3D 转换 |
| `create_attention_mask` | 向量化掩码创建 |

---

## 10.4 运行测试

### 基本命令

```bash
# 运行所有测试
uv run pytest tests/ -v

# 仅运行单元测试
uv run pytest tests/unit/ -v

# 运行特定测试文件
uv run pytest tests/unit/test_streaming_tokenizer.py -v

# 运行特定测试函数
uv run pytest tests/unit/test_attention.py::test_lca_hilbert_bias -v
```

### 带覆盖率

```bash
# 生成覆盖率报告
uv run pytest tests/ --cov=vit_pytorch --cov-report=html

# 查看报告
open htmlcov/index.html
```

### 并行执行

```bash
# 并行运行测试（需要 pytest-xdist）
uv run pytest tests/ -n auto -v
```

---

## 10.5 CUDA 调试指南

### CUDA 设备端断言

当遇到 CUDA 设备端断言错误（如 `IndexKernel.cu:92 index out of bounds`）时，设置以下环境变量以同步报告错误：

```bash
# Windows PowerShell
$env:CUDA_LAUNCH_BLOCKING = "1"
$env:PYTORCH_NO_CUDA_MEMORY_CACHING = "1"

# Linux/macOS Bash
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
```

### 常见 CUDA 错误

| 错误 | 原因 | 解决方案 |
|:------|:------|:---------|
| `index out of bounds` | batch_indices 包含超出范围的值 | 检查 clamp 逻辑 |
| `cublasLt` | 数值不稳定 | 检查输入归一化 |
| `memcpy` | 内存访问违规 | 检查张量形状匹配 |

### 调试技巧

1. **使用 `.item()` 进行 CPU 端验证**：避免在 CUDA 张量上调用 `.min()`/`.max()`
2. **使用 `torch.where` 进行条件 clamp**：避免触发断言的归约操作
3. **添加诊断信息**：`torch.where(condition, valid_value, safe_default)`

---

## 10.6 基准测试

### 核心比较（`compare_fractal_vs_standard.py`）

将 `FractalCurveViT` 与标准 ViT 进行比较：

| 指标 | 描述 |
|:-------|:------------|
| 参数数量 | 总可学习参数 |
| 内存使用 | 峰值 GPU 内存 |
| 推理延迟 | 前向传播时间 |
| 吞吐量 | 每秒图像数 |

### 综合基准（`benchmark_fractal_vit.py`）

| 组件 | 指标 |
|:----------|:--------|
| 分词器 | 吞吐量 (img/s)、内存 |
| 注意力 | FLOPS、延迟 |
| 端到端 | 前向/反向时间 |

---

## 10.7 测试状态

| 类别 | 数量 | 状态 |
|:---------|:------|:-------|
| 单元测试 | ~200 | ✓ 通过 |
| 集成测试 | ~50 | ✓ 通过 |
| 基准测试 | ~45 | ✓ 通过 |
| **总计** | **295+** | **✓ 全部通过** |

---

## 10.8 编写测试

### 测试夹具（`conftest.py`）

```python
import pytest
import torch

@pytest.fixture
def sample_image():
    return torch.randn(2, 3, 224, 224)

@pytest.fixture
def sample_levels_info():
    levels = torch.zeros(2, 64, 5, dtype=torch.long)
    levels[:, :, 0] = torch.randint(0, 5, (2, 64))
    return levels

@pytest.fixture
def model_config():
    return {
        'image_size': 224,
        'num_classes': 10,
        'dim': 192,
        'depth': 3,
        'heads': 3,
    }
```

### 示例测试

```python
def test_forward_shape(sample_image, model_config):
    model = FractalCurveViT(**model_config)
    output = model(sample_image)

    assert output.shape == (2, model_config['num_classes'])

def test_lca_bias_parameters():
    bias = LCAHilbertBias(num_heads=6, max_lca_depth=10)
    param_count = sum(p.numel() for p in bias.parameters())

    # 应大约有 100 个参数
    assert param_count < 200
```

---

## 10.9 持续集成

### GitHub Actions 配置

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Install dependencies
        run: uv sync

      - name: Run tests
        run: uv run pytest tests/ -v --tb=short

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        if: always()
```

### 预提交钩子

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: uv run pytest tests/unit/ -v --tb=short
        language: system
        pass_filenames: false
```

> **下一章**: [11_issues_roadmap.md](11_issues_roadmap.md) - 开发历史
