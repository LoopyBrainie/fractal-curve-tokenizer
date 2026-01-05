# Visualization (Architecture-Driven, No Training Data Needed)

This folder visualizes **Fractal Curve ViT** purely基于模型架构与源码：Hilbert 曲线、可学习四叉树 token 化、LCA 偏置等，不依赖任何训练结果或日志。

## 文件

- `CRITICAL_ANALYSIS.md`: 数学与架构批判阅读笔记。
- `seaborn_viz.py`: 生成出版级静态图（Hilbert 路径、合成四叉树、LCA 偏置热力图、深度分布、标度曲线）。
- `manim_viz.py`: 动画场景（HilbertCurveScene、AdaptiveQuadTreeScene、HilbertTraversalScene）。
- `run_viz.py`: 统一入口，运行静态图并打印 manim 命令示例。

## 依赖

```bash
pip install seaborn matplotlib numpy manim
```

## 运行静态可视化（推荐）

```bash
python examples/visualization/run_viz.py --max-depth 4 --static
```
输出位于 `workspace/visualizations/`。

## 运行 Manim 动画

```bash
manim -pql --media_dir workspace/visualizations/manim_media examples/visualization/manim_viz.py AdaptiveQuadTreeScene
manim -pql --media_dir workspace/visualizations/manim_media examples/visualization/manim_viz.py HilbertCurveScene
```

## 可视化要点

1) **Hilbert 曲线局部性**：1D 序列与 2D 网格的局部保持。
2) **自适应四叉树 token 化**：合成场驱动的分割阈值，展示层级与 Hilbert 序列顺序。
3) **LCA 注意力偏置**：按四叉树路径精确计算 LCA 深度矩阵，体现层级亲缘度。
