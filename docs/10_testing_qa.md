# Chapter 10: Testing and Quality Assurance

## 10.1 Overview

This project uses `pytest` for comprehensive testing, covering unit tests, integration tests, and performance benchmarks.

---

## 10.2 Test Directory Structure

```
tests/
├── conftest.py                 # Pytest fixtures and configuration
├── README.md                   # Test documentation
├── unit/                       # Unit tests
│   ├── test_fractal_vit.py
│   ├── test_streaming_tokenizer.py
│   ├── test_attention.py
│   ├── test_feedforward.py
│   ├── test_hilbert.py
│   ├── test_positional.py
│   └── test_utils.py
├── integration/                # Integration tests
│   ├── test_training.py
│   └── test_system.py
└── benchmarks/                 # Performance benchmarks
    ├── benchmark_fractal_vit.py
    └── compare_fractal_vs_standard.py
```

---

## 10.3 Unit Test Coverage

### Tokenizer Tests (`test_streaming_tokenizer.py`)

| Test | Description |
|:-----|:------------|
| Output shape validation | Various input sizes and batch sizes |
| Hilbert reordering | Correctness of Hilbert curve ordering |
| V3 output format | TokenizerOutput structure validation |
| Variable depth splitting | Quadtree split correctness |
| Depth distribution | Validate multi-scale token generation |

### Model Tests (`test_fractal_vit.py`)

| Test | Description |
|:-----|:------------|
| Forward pass shape | Output dimension correctness |
| Tokenizer integration | Different tokenizer types |
| Batch processing | Padding and masking |
| Position encoding | Correct injection |

### Attention Tests (`test_attention.py`)

| Test | Description |
|:-----|:------------|
| Bias modes | LCA, low-rank, hierarchical |
| LCA parameter count | Verify ~100 parameters |
| Low-rank accuracy | Decomposition precision |
| Mask validity | Attention mask application |

### FFN Tests (`test_feedforward.py`)

| Test | Description |
|:-----|:------------|
| SwiGLU output shape | Dimension preservation |
| Level adaptation | Depth-dependent processing |
| FFN types | GELU, SwiGLU, SwiGLU+Level |

### Hilbert Tests (`test_hilbert.py`)

| Test | Description |
|:-----|:------------|
| d ↔ (x,y) mapping | Bidirectional correctness |
| Boundary cases | Edge coordinates |
| Caching | Index cache mechanism |

### Utility Tests (`test_utils.py`)

| Test | Description |
|:-----|:------------|
| `extract_depths` | Dimension handling |
| `normalize_levels_info` | 2D→3D conversion |
| `create_attention_mask` | Vectorized mask creation |

---

## 10.4 Running Tests

### Basic Commands

```bash
# Run all tests
uv run pytest tests/ -v

# Run unit tests only
uv run pytest tests/unit/ -v

# Run specific test file
uv run pytest tests/unit/test_streaming_tokenizer.py -v

# Run specific test function
uv run pytest tests/unit/test_attention.py::test_lca_hilbert_bias -v
```

### With Coverage

```bash
# Generate coverage report
uv run pytest tests/ --cov=vit_pytorch --cov-report=html

# View report
open htmlcov/index.html
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
uv run pytest tests/ -n auto -v
```

---

## 10.5 CUDA Debugging Guide

### CUDA Device-Side Assert

When encountering CUDA device-side assert errors (such as `IndexKernel.cu:92 index out of bounds`), set the following environment variables to report errors synchronously:

```bash
# Windows PowerShell
$env:CUDA_LAUNCH_BLOCKING = "1"
$env:PYTORCH_NO_CUDA_MEMORY_CACHING = "1"

# Linux/macOS Bash
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
```

### Common CUDA Errors

| Error | Cause | Solution |
|:------|:------|:---------|
| `index out of bounds` | batch_indices contains out-of-range values | Check clamp logic |
| `cublasLt` | Numerical instability | Check input normalization |
| `memcpy` | Memory access violation | Check tensor shape matching |

### Debugging Tips

1. **Use `.item()` for CPU-side validation**: Avoid calling `.min()`/`.max()` on CUDA tensors
2. **Use `torch.where` for conditional clamp**: Avoid reduction operations triggering assert
3. **Add diagnostic information**: `torch.where(condition, valid_value, safe_default)`

---

## 10.6 Benchmarks

### Core Comparison (`compare_fractal_vs_standard.py`)

Compares `FractalCurveViT` against standard ViT:

| Metric | Description |
|:-------|:------------|
| Parameter count | Total learnable parameters |
| Memory usage | Peak GPU memory |
| Inference latency | Forward pass time |
| Throughput | Images per second |

### Comprehensive Benchmark (`benchmark_fractal_vit.py`)

| Component | Metrics |
|:----------|:--------|
| Tokenizer | Throughput (img/s), memory |
| Attention | FLOPS, latency |
| End-to-end | Forward/backward time |

---

## 10.6 Test Status

| Category | Count | Status |
|:---------|:------|:-------|
| Unit tests | ~200 | ✓ Passing |
| Integration tests | ~50 | ✓ Passing |
| Benchmarks | ~45 | ✓ Passing |
| **Total** | **295+** | **✓ All Passing** |

---

## 10.7 Writing Tests

### Test Fixtures (`conftest.py`)

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

### Example Test

```python
def test_forward_shape(sample_image, model_config):
    model = FractalCurveViT(**model_config)
    output = model(sample_image)
    
    assert output.shape == (2, model_config['num_classes'])

def test_lca_bias_parameters():
    bias = LCAHilbertBias(num_heads=6, max_lca_depth=10)
    param_count = sum(p.numel() for p in bias.parameters())
    
    # Should be approximately 100 parameters
    assert param_count < 200
```

---

## 10.8 Continuous Integration

### GitHub Actions Configuration

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

### Pre-commit Hooks

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

> **Next**: [11_issues_roadmap.md](11_issues_roadmap.md) - Development History
