# 测试文件夹结构说明

经过清理和重新组织，测试文件夹现在具有清晰的结构：

## 📁 unit_tests/ - 单元测试

针对具体模块和函数的单元测试：

- `test_fractal_hilbert_tokenizer.py` - 分形Hilbert分词器核心功能测试
- `test_hilbert_verification.py` - Hilbert曲线算法验证
- `test_improved_hilbert.py` - 改进的Hilbert算法测试
- `test_hilbert_effectiveness.py` - Hilbert曲线有效性测试
- `test_adaptive_tokenization.py` - 自适应分词功能测试
- `test_spatial_locality.py` - 空间局部性测试

## 📁 integration_tests/ - 集成测试

测试整个系统的集成功能：

- `test_complete_system.py` - 完整系统集成测试
- `test_enhanced_fractal_vit.py` - 增强分形ViT测试
- `test_universal_tokenizer.py` - 通用分词器测试
- `test_unlimited_subdivision.py` - 无限细分功能测试
- `final_unlimited_subdivision_test.py` - 最终无限细分测试
- `test_fractal_zigzag_manim.py` - Manim可视化测试

## 📁 training_scripts/ - 训练脚本

用于模型训练和验证的脚本：

- `train_fractal_vit.py` - 分形ViT训练脚本
- `train_enhanced_fractal_vit.py` - 增强分形ViT训练脚本
- `train_fractal_cifar10.py` - CIFAR-10数据集训练脚本
- `train_standard_vit_baseline.py` - 标准ViT基线训练脚本

## 📁 benchmarks/ - 性能基准测试

模型性能对比和基准测试：

- `benchmark_fractal_vit.py` - 分形ViT性能基准测试
- `compare_fractal_vs_standard.py` - 分形vs标准ViT对比测试

## 已删除的文件
为了避免冗余和混乱，以下文件已被删除：

### 重复的Hilbert测试文件：
- `verify_hilbert.py`
- `test_new_hilbert.py` 
- `test_hilbert_advanced.py`
- `test_hilbert_detailed.py`
- `test_hilbert_order.py`

### 调试和演示文件：
- `debug_unlimited_subdivision.py`
- `demo_new_features.py`
- `hilbert_fix_demo.py`
- `test.py` (基础ViT测试)

### 重复的训练文件：
- `train_fractal_demo.py`
- `quick_fractal_vs_standard_train.py`
- `train_unlimited_fractal_vs_standard.py`
- `train_vit_cifar10.py`
- `test_vit_cifar10.py`

### 重复的模型测试：
- `test_fractal_model.py`
- `test_fractal_inference.py`
- `test_large_image_subdivision.py`
- `test_fractal_zigzag_all_levels.py`

### 缓存文件：
- `__pycache__/` 整个文件夹

## 运行测试
```bash
# 运行单元测试
python -m pytest unit_tests/

# 运行集成测试
python -m pytest integration_tests/

# 运行特定测试
python unit_tests/test_fractal_hilbert_tokenizer.py
```

## 训练模型
```bash
# 训练分形ViT
python training_scripts/train_fractal_vit.py

# 训练CIFAR-10
python training_scripts/train_fractal_cifar10.py
```

## 性能基准测试
```bash
# 运行性能对比
python benchmarks/compare_fractal_vs_standard.py

# 运行基准测试
python benchmarks/benchmark_fractal_vit.py
```

---

清理日期: 2025年8月17日
清理前文件数: 37
清理后文件数: 18 + 1 README
