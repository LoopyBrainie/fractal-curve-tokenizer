# 第十章：测试与质量保证

项目采用了 `pytest` 框架进行全面的测试。

## 10.1 测试目录结构

```text
tests/
├── unit/                   # 单元测试：测试单个组件的功能
│   ├── test_fractal_vit.py
│   ├── test_tokenizer.py
│   ├── test_utils.py
│   ├── test_hilbert.py
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
    *   **批处理一致性**: 验证 BFS 批处理模式与递归模式输出完全一致。
    *   **Hilbert 顺序**: 验证批处理模式下仍能保持正确的 Hilbert 遍历顺序。
*   **模型测试 (`test_fractal_vit.py`)**:
    *   **自适应能力**: 验证复杂图像比简单图像产生更多的 Token。
    *   **Batch 处理**: 验证 Padding 和 Masking 机制在变长序列下的正确性。
    *   **位置编码**: 验证不同路径产生不同的 Embedding。
    *   **Mask 有效性**: 验证 Padding Token 不会影响有效 Token 的计算结果。
*   **工具测试 (`test_utils.py`)**:
    *   验证统一的深度提取 (`extract_depths`) 和层级规范化函数。
    *   验证特征计算和 Mask 生成函数的正确性。
*   **Hilbert 测试 (`test_hilbert.py`)**:
    *   验证 Hilbert 曲线生成的正确性和缓存机制。

## 10.3 基准测试与评估

`tests/benchmarks/` 目录包含了一套完整的性能评估工具。

### 1. 核心对比 (`compare_fractal_vs_standard.py`)
提供与标准 ViT 的公平对比。
*   **对比对象**: `NextGenerationFractalViT` vs `StandardViT` (PyTorch 原生实现)。
*   **指标**: 参数量、显存占用、推理延迟、吞吐量。
*   **目的**: 量化分形 Tokenizer 带来的性能开销与收益。

### 2. 综合性能基准 (`benchmark_fractal_vit.py`)
深入分析 Fractal ViT 的各项性能指标。
*   **测试项**:
    *   **Tokenizer 效率**: 纯 Tokenizer 的吞吐量 (img/s)。
    *   **端到端性能**: 完整模型的前向/反向传播速度。
    *   **显存分析**: 详细的显存占用分布。
*   **特性**: 支持不同 Batch Size 和 Image Size 的压力测试。

### 3. 预训练评估 (`evaluate_pretrained.py`)
用于评估已训练模型的性能。
*   **功能**: 加载 Checkpoint 并在指定数据集上运行验证。
*   **可视化**: 支持生成预测结果的可视化网格 (`--visualize`)。
*   **指标**: Top-1 Accuracy, Top-5 Accuracy, Loss。

### 4. 收敛性检查 (`check_convergence.py`)
用于快速验证模型是否具备学习能力。
*   **方法**: 在极小数据集（如 100 张图）上过拟合。
*   **判定**: 如果 Loss 能迅速下降到接近 0，说明模型架构无严重 Bug。

