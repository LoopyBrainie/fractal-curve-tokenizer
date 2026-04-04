# -*- coding: utf-8 -*-
"""
HTML 报告生成器

生成包含所有分析结果的可交互 HTML 报告。
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class ReportSection:
    """报告章节"""
    title: str
    content: str
    figures: List[str] = None
    order: int = 0


class HTMLReportGenerator:
    """
    HTML 报告生成器

    生成包含图表、分析结果和数学公式的可交互 HTML 报告。
    """

    def __init__(
        self,
        title: str = "Fractal Curve ViT Analysis Report",
        author: str = "Analysis Module"
    ):
        self.title = title
        self.author = author
        self.sections = []
        self.figures = []

    def add_section(self, section: ReportSection):
        """添加章节"""
        self.sections.append(section)

    def add_figure(self, figure_path: str, caption: str = ""):
        """添加图片"""
        self.figures.append({
            'path': figure_path,
            'caption': caption
        })

    def generate_html(self, output_path: str):
        """
        生成 HTML 报告

        Args:
            output_path: 输出路径
        """
        # HTML 模板
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            border-radius: 10px;
        }}
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        section {{
            background: white;
            padding: 30px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        section h3 {{
            color: #764ba2;
            margin-top: 20px;
            margin-bottom: 15px;
        }}
        .math {{
            font-family: 'Times New Roman', serif;
            font-style: italic;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            overflow-x: auto;
        }}
        .math-display {{
            text-align: center;
            font-size: 1.1em;
            padding: 20px;
            background: #f0f0f0;
            border-radius: 5px;
            margin: 20px 0;
        }}
        figure {{
            margin: 20px 0;
            text-align: center;
        }}
        figure img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }}
        figcaption {{
            margin-top: 10px;
            font-style: italic;
            color: #666;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .highlight {{
            background: #fff3cd;
            padding: 2px 5px;
            border-radius: 3px;
        }}
        .success {{
            color: #28a745;
            font-weight: bold;
        }}
        .warning {{
            color: #ffc107;
            font-weight: bold;
        }}
        .error {{
            color: #dc3545;
            font-weight: bold;
        }}
        .code {{
            font-family: 'Courier New', monospace;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        .toc {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .toc h3 {{
            margin-top: 0;
        }}
        .toc ul {{
            list-style-type: none;
        }}
        .toc li {{
            margin: 10px 0;
        }}
        .toc a {{
            color: #667eea;
            text-decoration: none;
        }}
        .toc a:hover {{
            text-decoration: underline;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .metric-card .value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .metric-card .label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            margin-top: 30px;
        }}
        .comparison-table {{
            margin: 20px 0;
        }}
        .comparison-table th:first-child {{
            background: #764ba2;
        }}
        .vs-badge {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            margin: 0 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{self.title}</h1>
            <p>Author: {self.author}</p>
            <p>Generated by Fractal Curve ViT Analysis Module</p>
        </header>

        <div class="toc">
            <h3>Table of Contents</h3>
            <ul>
"""

        # 添加目录
        for i, section in enumerate(self.sections):
            html_content += f'                <li><a href="#section-{i}">{section.title}</a></li>\n'

        html_content += """            </ul>
        </div>

        <!-- Main Content -->
"""

        # 添加章节
        for i, section in enumerate(self.sections):
            html_content += f"""
        <section id="section-{i}">
            <h2>{section.title}</h2>
            {section.content}
        </section>
"""

        # 添加图片
        if self.figures:
            html_content += """
        <section>
            <h2>Visualizations</h2>
            <div class="figures">
"""
            for fig in self.figures:
                html_content += f"""
                <figure>
                    <img src="{fig['path']}" alt="{fig['caption']}">
                    <figcaption>{fig['caption']}</figcaption>
                </figure>
"""
            html_content += """
            </div>
        </section>
"""

        html_content += """
        <footer>
            <p>Generated by Fractal Curve ViT Analysis Module</p>
            <p>For more information, visit the project repository.</p>
        </footer>
    </div>
</body>
</html>
"""

        # 保存文件
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Generated HTML report: {output_path}")

        return output_path


def generate_fractal_vit_report(
    model_name: str = "Fractal Curve ViT",
    analysis_results: Dict[str, Any] = None,
    figures: List[str] = None,
    output_path: str = "./analysis_report.html"
) -> str:
    """
    生成 Fractal ViT 分析报告

    Args:
        model_name: 模型名称
        analysis_results: 分析结果字典
        figures: 图表路径列表
        output_path: 输出路径

    Returns:
        输出文件路径
    """
    generator = HTMLReportGenerator(
        title=f"{model_name} Analysis Report",
        author="Analysis Module"
    )

    # 添加介绍章节
    intro_content = f"""
    <h3>模型概述</h3>
    <p>{model_name} 是一种创新的视觉 Transformer 架构，采用自适应分词和 Hilbert 曲线排序。</p>

    <h3>核心创新</h3>
    <ul>
        <li><strong>自适应分词</strong>: 基于四叉树的自适应 token 生成，根据图像复杂度动态调整</li>
        <li><strong>Hilbert 曲线排序</strong>: 使用空间填充曲线保持局部性，比栅格顺序更优</li>
        <li><strong>LCA 位置编码</strong>: O(log N) 参数替代 O(N²)，大幅减少位置参数量</li>
    </ul>

    <h3>数学形式</h3>
    <div class="math-display">
        Token = ROI_Align(F, R) × depth_scale[d] + depth_embed[d]
    </div>

    <h3>与标准 ViT 对比</h3>
    <table class="comparison-table">
        <tr>
            <th>特性</th>
            <th>标准 ViT</th>
            <th>Fractal Curve ViT</th>
        </tr>
        <tr>
            <td>Token 数量</td>
            <td>固定 (196 for 224×224)</td>
            <td>可变 (8-64)</td>
        </tr>
        <tr>
            <td>位置编码</td>
            <td>O(N²) 参数</td>
            <td>O(log N) 参数</td>
        </tr>
        <tr>
            <td>空间排序</td>
            <td>栅格顺序</td>
            <td>Hilbert 曲线</td>
        </tr>
        <tr>
            <td>多尺度</td>
            <td>单一尺度</td>
            <td>4 尺度同时建模</td>
        </tr>
    </table>
    """

    generator.add_section(ReportSection(
        title="模型介绍",
        content=intro_content,
        order=0
    ))

    # 添加分析结果章节
    if analysis_results:
        results_content = "<div class='metrics-grid'>"

        for metric_name, value in analysis_results.items():
            results_content += f"""
            <div class="metric-card">
                <div class="value">{value}</div>
                <div class="label">{metric_name}</div>
            </div>
            """

        results_content += "</div>"
        generator.add_section(ReportSection(
            title="分析结果",
            content=results_content,
            order=1
        ))

    # 添加数学公式章节
    math_content = """
    <h3>核心公式</h3>

    <h4>1. Hilbert 曲线映射</h4>
    <div class="math">
        H: [0, n²) ↔ [0, n) × [0, n)
    </div>

    <h4>2. 深度感知 Token 编码</h4>
    <div class="math">
        Token_d = ROI_Align(F, R_d) × depth_scale[d] + depth_embed[d]
    </div>

    <h4>3. LCA 注意力偏置</h4>
    <div class="math">
        A_ij = softmax(Q_i · K_j / √d + τ_h · LCAEmbed(LCA(i,j)))
    </div>

    <h4>4. 深度分布损失</h4>
    <div class="math">
        L_depth = KL(P_actual || P_target)
    </div>
    """
    generator.add_section(ReportSection(
        title="数学形式化",
        content=math_content,
        order=2
    ))

    # 添加可视化章节
    if figures:
        for fig_path in figures:
            generator.add_figure(fig_path, "Analysis Visualization")

    # 生成报告
    return generator.generate_html(output_path)


def generate_comparison_report(
    fractal_metrics: Dict[str, float],
    standard_metrics: Dict[str, float],
    output_path: str = "./comparison_report.html"
) -> str:
    """
    生成对比分析报告

    Args:
        fractal_metrics: Fractal ViT 指标
        standard_metrics: 标准 ViT 指标
        output_path: 输出路径

    Returns:
        输出文件路径
    """
    generator = HTMLReportGenerator(
        title="ViT Comparison Report",
        author="Analysis Module"
    )

    # 对比表格
    table_rows = ""
    for metric in fractal_metrics:
        f_val = fractal_metrics[metric]
        s_val = standard_metrics.get(metric, "N/A")
        f_better = "success" if f_val > s_val else ""
        s_better = "success" if s_val > f_val else ""

        table_rows += f"""
        <tr>
            <td>{metric}</td>
            <td class="{f_better}">{f_val}</td>
            <td class="{s_better}">{s_val}</td>
        </tr>
        """

    comparison_content = f"""
    <h3>性能对比</h3>
    <table>
        <tr>
            <th>指标</th>
            <th>Fractal Curve ViT</th>
            <th>Standard ViT</th>
        </tr>
        {table_rows}
    </table>

    <h3>关键发现</h3>
    <ul>
        <li>Token 数量减少: {(1 - fractal_metrics.get('token_count', 1) / standard_metrics.get('token_count', 1)) * 100:.1f}%</li>
        <li>位置参数量减少: {(1 - fractal_metrics.get('position_params', 1) / max(standard_metrics.get('position_params', 1), 1)) * 100:.1f}%</li>
        <li>空间局部性改善: {fractal_metrics.get('locality_score', 0) - standard_metrics.get('locality_score', 0):.3f}</li>
    </ul>

    <h3>结论</h3>
    <p>Fractal Curve ViT 在保持可比性能的同时，显著减少了计算开销和参数量。</p>
    """

    generator.add_section(ReportSection(
        title="对比分析",
        content=comparison_content,
        order=0
    ))

    return generator.generate_html(output_path)


if __name__ == "__main__":
    # 演示
    print("HTML Report Generator Demo")

    # 生成示例报告
    fractal_metrics = {
        'token_count': 32,
        'position_params': 128,
        'locality_score': 0.85,
        'accuracy': 0.75
    }

    standard_metrics = {
        'token_count': 196,
        'position_params': 38416,
        'locality_score': 1.2,
        'accuracy': 0.76
    }

    # 生成对比报告
    output_path = generate_comparison_report(
        fractal_metrics,
        standard_metrics,
        output_path="./example_comparison_report.html"
    )

    print(f"Generated: {output_path}")
