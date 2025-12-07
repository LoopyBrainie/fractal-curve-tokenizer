# 第十章：测试与质量保证

项目采用了 `pytest` 框架进行全面的测试。

## 10.1 测试目录结构

```text
tests/
├── unit/                   # 单元测试：测试单个组件的功能
│   ├── test_fractal_vit.py
│   ├── test_tokenizer.py
│   ├── test_utils_module.py
│   └── ...
├── integration/            # 集成测试：测试完整流程
│   ├── test_training.py
│   └── test_system.py
├── benchmarks/             # 性能基准测试
│   └── compare_fractal_vs_standard.py
└── conftest.py             # Pytest 配置和 Fixtures
```

## 10.2 单元测试覆盖范围

*   **Tokenizer 测试 (`test_tokenizer.py`)**:
    *   验证不同输入尺寸和 Batch Size 下的输出形状。
    *   验证极端长宽比下的处理能力。
    *   验证自适应分割逻辑（四分法/二分法）。
    *   验证可学习分割的阈值行为。
*   **模型测试 (`test_fractal_vit.py`)**:
    *   **自适应能力**: 验证复杂图像比简单图像产生更多的 Token。
    *   **Batch 处理**: 验证 Padding 和 Masking 机制在变长序列下的正确性。
    *   **位置编码**: 验证不同路径产生不同的 Embedding。
    *   **Mask 有效性**: 验证 Padding Token 不会影响有效 Token 的计算结果。
*   **工具测试 (`test_utils_module.py`)**:
    *   验证特征计算和 Mask 生成函数的正确性。

## 10.3 基准测试

`tests/benchmarks/compare_fractal_vs_standard.py` 提供了一个公平的对比脚本。
*   **对比对象**: `NextGenerationFractalViT` vs 标准 `StandardViT` (基于 PyTorch 原生 Transformer)。
*   **指标**:
    *   参数量 (Parameters)。
    *   峰值显存占用 (Peak Memory)。
    *   推理延迟 (Inference Latency)。
    *   吞吐量 (Throughput)。
*   **目的**: 量化分形 Tokenizer 带来的性能开销与收益。
