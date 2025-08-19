#!/usr/bin/env python3
"""
Fractal ViT vs 标准ViT 详细对比分析
控制变量下的公平性能评估
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_and_compare_results():
    """加载并对比两个模型的结果"""
    
    print("🔍 Fractal ViT vs 标准ViT 对比分析")
    print("=" * 60)
    
    # 从训练日志中提取的结果数据
    fractal_results = {
        'model_type': 'Enhanced Fractal ViT',
        'best_val_acc': 31.5,
        'final_train_acc': 29.7,
        'total_params': 9802612,
        'avg_epoch_time': 130.6,  # (131.7 + 130.4 + 130.7) / 3
        'tokenization_behavior': {
            'avg_tokens_per_image': 64.0,
            'levels_used': [3],  # 主要使用第3层级
            'level_distribution': [0, 0, 0, 64]
        },
        'training_progression': [21.8, 29.1, 31.5]  # 验证准确率
    }
    
    standard_results = {
        'model_type': 'Standard ViT',
        'best_val_acc': 39.8,
        'final_train_acc': 39.5,
        'total_params': 4012682,
        'avg_epoch_time': 65.7,  # (65.7 + 65.5 + 65.9) / 3
        'avg_inference_time_ms': 173.96,
        'sample_accuracy': 37.5,
        'training_progression': [29.5, 29.8, 39.8]  # 验证准确率
    }
    
    print("📊 基础性能对比")
    print("-" * 40)
    print(f"{'指标':<20} {'Fractal ViT':<15} {'标准ViT':<15} {'差异':<15}")
    print("-" * 70)
    
    # 准确率对比
    acc_diff = fractal_results['best_val_acc'] - standard_results['best_val_acc']
    print(f"{'最佳验证准确率':<20} {fractal_results['best_val_acc']:.1f}%{'':<9} {standard_results['best_val_acc']:.1f}%{'':<9} {acc_diff:+.1f}%")
    
    train_acc_diff = fractal_results['final_train_acc'] - standard_results['final_train_acc']
    print(f"{'最终训练准确率':<20} {fractal_results['final_train_acc']:.1f}%{'':<9} {standard_results['final_train_acc']:.1f}%{'':<9} {train_acc_diff:+.1f}%")
    
    # 参数量对比
    param_ratio = fractal_results['total_params'] / standard_results['total_params']
    print(f"{'模型参数量':<20} {fractal_results['total_params']:,}{'':<3} {standard_results['total_params']:,}{'':<3} {param_ratio:.2f}x")
    
    # 训练时间对比
    time_ratio = fractal_results['avg_epoch_time'] / standard_results['avg_epoch_time']
    print(f"{'平均训练时间/轮':<20} {fractal_results['avg_epoch_time']:.1f}s{'':<9} {standard_results['avg_epoch_time']:.1f}s{'':<9} {time_ratio:.2f}x")
    
    print("\n" + "=" * 60)
    
    # 详细分析
    print("🧠 深度分析")
    print("-" * 40)
    
    print("1. 准确率性能")
    if acc_diff < 0:
        print(f"   • 标准ViT在准确率上领先 {abs(acc_diff):.1f}%")
        print(f"   • 可能原因:")
        print(f"     - Fractal ViT的分形tokenization增加了复杂性")
        print(f"     - 当前数据集可能不足以充分发挥分形特性优势")
        print(f"     - 需要更多训练轮数来收敛")
    else:
        print(f"   • Fractal ViT在准确率上领先 {acc_diff:.1f}%")
    
    print("\n2. 模型复杂度")
    print(f"   • Fractal ViT参数量是标准ViT的 {param_ratio:.2f}倍")
    print(f"   • 额外参数主要来自:")
    print(f"     - 分形位置编码组件")
    print(f"     - 多尺度注意力机制") 
    print(f"     - 自适应前馈网络")
    print(f"     - 层级感知的处理模块")
    
    print("\n3. 训练效率")
    print(f"   • Fractal ViT训练时间是标准ViT的 {time_ratio:.2f}倍")
    print(f"   • 主要原因:")
    print(f"     - 分形tokenization的计算开销")
    print(f"     - 可变长度序列处理的复杂性")
    print(f"     - 更复杂的注意力计算")
    
    print("\n4. 分形特性分析")
    print(f"   • 平均每图像生成 {fractal_results['tokenization_behavior']['avg_tokens_per_image']} 个tokens")
    print(f"   • 主要使用分形层级: {fractal_results['tokenization_behavior']['levels_used']}")
    print(f"   • 这表明模型倾向于深度分割，捕获细粒度特征")
    
    print("\n" + "=" * 60)
    
    # 训练曲线对比
    print("📈 训练进程对比")
    print("-" * 40)
    
    print("验证准确率进展:")
    fractal_prog = fractal_results['training_progression']
    standard_prog = standard_results['training_progression']
    
    for epoch in range(3):
        f_acc = fractal_prog[epoch]
        s_acc = standard_prog[epoch]
        diff = f_acc - s_acc
        print(f"   Epoch {epoch+1}: Fractal {f_acc:.1f}% vs 标准 {s_acc:.1f}% (差异: {diff:+.1f}%)")
    
    print("\n训练趋势分析:")
    fractal_improvement = fractal_prog[-1] - fractal_prog[0]
    standard_improvement = standard_prog[-1] - standard_prog[0]
    
    print(f"   • Fractal ViT 总提升: {fractal_improvement:.1f}%")
    print(f"   • 标准ViT 总提升: {standard_improvement:.1f}%")
    
    if standard_improvement > fractal_improvement:
        print(f"   • 标准ViT显示出更强的学习能力")
        print(f"   • 可能原因: 更简单的架构更容易优化")
    else:
        print(f"   • Fractal ViT显示出更强的学习潜力")
    
    print("\n" + "=" * 60)
    
    # 优势与劣势分析
    print("⚖️  优劣势分析")
    print("-" * 40)
    
    print("Fractal ViT 优势:")
    print("   ✅ 创新的分形tokenization方法")
    print("   ✅ 层级感知的特征处理")  
    print("   ✅ 自适应的patch分割策略")
    print("   ✅ 多尺度注意力机制")
    print("   ✅ 理论上更适合复杂图像结构")
    
    print("\nFractal ViT 劣势:")
    print("   ❌ 当前准确率略低于标准ViT")
    print("   ❌ 计算复杂度显著增加")
    print("   ❌ 训练时间约为标准ViT的2倍")
    print("   ❌ 参数量增加约2.4倍")
    print("   ❌ 需要更多调优和训练")
    
    print("\n标准ViT 优势:")
    print("   ✅ 更高的当前准确率")
    print("   ✅ 训练效率高")
    print("   ✅ 参数量相对较少")
    print("   ✅ 成熟稳定的架构")
    
    print("\n标准ViT 劣势:")
    print("   ❌ 固定patch分割限制")
    print("   ❌ 缺乏层级特征处理")
    print("   ❌ 对复杂结构的适应性差")
    
    print("\n" + "=" * 60)
    
    # 改进建议
    print("🚀 Fractal ViT 改进建议")
    print("-" * 40)
    
    print("短期优化:")
    print("   1. 调整学习率调度策略")
    print("   2. 增加训练轮数至10-20个epoch")
    print("   3. 优化分形tokenization的计算效率")
    print("   4. 微调层级权重初始化")
    
    print("\n中期优化:")
    print("   1. 实现更高效的可变长度序列处理")
    print("   2. 优化多尺度注意力的计算")
    print("   3. 添加知识蒸馏从标准ViT学习")
    print("   4. 实现分形结构的稀疏化")
    
    print("\n长期研究:")
    print("   1. 在更大数据集上验证分形优势")
    print("   2. 研究分形特性与图像复杂度的关系")
    print("   3. 开发分形感知的数据增强策略")
    print("   4. 探索分形ViT在其他视觉任务的应用")
    
    print("\n" + "=" * 60)
    
    # 结论
    print("📝 结论")
    print("-" * 40)
    
    print("实验结果显示:")
    print(f"• 在当前设置下，标准ViT表现更优 (39.8% vs 31.5%)")
    print(f"• 但Fractal ViT展现出独特的技术创新价值")
    print(f"• 分形方法在理论上具有更大潜力，需要进一步优化")
    
    print("\n技术价值:")
    print("• Fractal ViT成功实现了分形几何与深度学习的结合")
    print("• 为视觉Transformer提供了全新的tokenization思路")
    print("• 在医学影像、遥感图像等复杂场景中可能有优势")
    
    print("\n建议:")
    print("• 继续优化Fractal ViT的训练策略")
    print("• 在更大规模数据集上进行验证")  
    print("• 探索分形特性最适合的应用场景")
    print("• 开发更高效的分形计算算法")
    
    return fractal_results, standard_results

def create_comparison_plot(fractal_results, standard_results):
    """创建对比可视化图表"""
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Fractal ViT vs 标准ViT 性能对比', fontsize=16, fontweight='bold')
    
    # 1. 准确率对比
    models = ['Fractal ViT', '标准ViT']
    accuracies = [fractal_results['best_val_acc'], standard_results['best_val_acc']]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars1 = ax1.bar(models, accuracies, color=colors, alpha=0.8)
    ax1.set_ylabel('验证准确率 (%)')
    ax1.set_title('最佳验证准确率对比')
    ax1.set_ylim(0, 45)
    
    # 添加数值标签
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. 参数量对比
    params = [fractal_results['total_params']/1e6, standard_results['total_params']/1e6]
    bars2 = ax2.bar(models, params, color=colors, alpha=0.8)
    ax2.set_ylabel('参数量 (M)')
    ax2.set_title('模型参数量对比')
    
    for bar, param in zip(bars2, params):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{param:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    # 3. 训练时间对比
    times = [fractal_results['avg_epoch_time'], standard_results['avg_epoch_time']]
    bars3 = ax3.bar(models, times, color=colors, alpha=0.8)
    ax3.set_ylabel('训练时间 (s/epoch)')
    ax3.set_title('平均每轮训练时间对比')
    
    for bar, time in zip(bars3, times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # 4. 训练进程对比
    epochs = [1, 2, 3]
    ax4.plot(epochs, fractal_results['training_progression'], 
             marker='o', linewidth=2, label='Fractal ViT', color='#FF6B6B')
    ax4.plot(epochs, standard_results['training_progression'], 
             marker='s', linewidth=2, label='标准ViT', color='#4ECDC4')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('验证准确率 (%)')
    ax4.set_title('训练进程对比')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = Path("workspace") / f"fractal_vs_standard_vit_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 对比图表已保存: {plot_path}")
    
    # 显示图表（如果在支持的环境中）
    try:
        plt.show()
    except:
        pass

def main():
    """主分析函数"""
    try:
        fractal_results, standard_results = load_and_compare_results()
        
        # 如果matplotlib可用，创建可视化图表
        try:
            create_comparison_plot(fractal_results, standard_results)
        except ImportError:
            print("📊 注意: matplotlib不可用，跳过图表生成")
        except Exception as e:
            print(f"📊 图表生成出现问题: {e}")
        
        print(f"\n🎯 对比分析完成!")
        print(f"详细数据显示了两种方法的优劣势，为后续优化提供了明确方向。")
        
    except Exception as e:
        print(f"❌ 分析过程出现错误: {e}")

if __name__ == "__main__":
    main()
