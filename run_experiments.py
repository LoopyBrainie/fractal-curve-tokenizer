#!/usr/bin/env python3
"""
快速启动脚本 - 不同配置的实验
"""

import subprocess
import sys
from pathlib import Path

def run_experiment(config_name, **kwargs):
    """运行实验配置"""
    print(f"\n🚀 启动实验: {config_name}")
    print("-" * 50)
    
    cmd = [sys.executable, "train_fractal_vs_standard_cifar10.py"]
    
    for key, value in kwargs.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {config_name} 实验完成")
    except subprocess.CalledProcessError as e:
        print(f"❌ {config_name} 实验失败: {e}")
        return False
    
    return True

def main():
    """主函数 - 定义不同的实验配置"""
    
    experiments = {
        "quick_test": {
            "epochs": 5,
            "batch_size": 64,
            "lr": 1e-3,
            "seed": 42
        },
        "standard_experiment": {
            "epochs": 100,
            "batch_size": 128,
            "lr": 3e-4,
            "seed": 42
        },
        "large_batch": {
            "epochs": 100,
            "batch_size": 256,
            "lr": 5e-4,
            "seed": 42
        },
        "small_batch": {
            "epochs": 100,
            "batch_size": 64,
            "lr": 2e-4,
            "seed": 42
        },
        "different_seed": {
            "epochs": 100,
            "batch_size": 128,
            "lr": 3e-4,
            "seed": 2024
        }
    }
    
    print("🎯 Fractal ViT vs Standard ViT 实验套件")
    print("可用的实验配置:")
    
    for i, (name, config) in enumerate(experiments.items(), 1):
        print(f"  {i}. {name}: {config}")
    
    print(f"  {len(experiments) + 1}. 自定义实验")
    print(f"  0. 退出")
    
    while True:
        try:
            choice = input(f"\n请选择实验配置 (1-{len(experiments) + 1}, 0=退出): ").strip()
            
            if choice == "0":
                print("👋 退出实验")
                break
                
            elif choice in [str(i) for i in range(1, len(experiments) + 1)]:
                exp_name = list(experiments.keys())[int(choice) - 1]
                config = experiments[exp_name]
                
                if run_experiment(exp_name, **config):
                    print(f"\n🎉 {exp_name} 实验已完成!")
                else:
                    print(f"\n❌ {exp_name} 实验失败!")
                    
            elif choice == str(len(experiments) + 1):
                # 自定义实验
                print("\n⚙️ 自定义实验配置:")
                
                try:
                    epochs = int(input("训练轮次 (默认100): ") or "100")
                    batch_size = int(input("批次大小 (默认128): ") or "128")
                    lr = float(input("学习率 (默认3e-4): ") or "3e-4")
                    seed = int(input("随机种子 (默认42): ") or "42")
                    
                    custom_config = {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "lr": lr,
                        "seed": seed
                    }
                    
                    if run_experiment("custom", **custom_config):
                        print(f"\n🎉 自定义实验已完成!")
                    else:
                        print(f"\n❌ 自定义实验失败!")
                        
                except ValueError as e:
                    print(f"❌ 输入错误: {e}")
                    
            else:
                print("❌ 无效选择，请重新输入")
                
        except KeyboardInterrupt:
            print("\n👋 用户取消实验")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main()
