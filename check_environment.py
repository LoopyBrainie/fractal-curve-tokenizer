#!/usr/bin/env python3
"""
环境检查和依赖安装脚本
"""

import sys
import subprocess
import importlib
import torch

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python版本过低: {sys.version}")
        print("需要Python 3.8或更高版本")
        return False
    else:
        print(f"✅ Python版本: {sys.version}")
        return True

def check_cuda():
    """检查CUDA可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("⚠️  CUDA不可用，将使用CPU训练（速度较慢）")
            return False
    except ImportError:
        print("❌ PyTorch未安装")
        return False

def check_dependencies():
    """检查依赖包"""
    required_packages = [
        'torch',
        'torchvision', 
        'numpy',
        'matplotlib',
        'seaborn',
        'tqdm',
        'einops'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 未安装")
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(packages):
    """安装缺失的依赖"""
    if not packages:
        return True
    
    print(f"\n📦 安装缺失的依赖: {', '.join(packages)}")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install"
        ] + packages)
        print("✅ 依赖安装完成")
        return True
    except subprocess.CalledProcessError:
        print("❌ 依赖安装失败")
        return False

def main():
    """主检查函数"""
    print("🔍 环境检查开始...")
    print("="*50)
    
    # 检查Python版本
    if not check_python_version():
        return False
    
    print()
    
    # 检查依赖
    missing = check_dependencies()
    
    print()
    
    # 检查CUDA
    cuda_available = check_cuda()
    
    print()
    
    # 安装缺失依赖
    if missing:
        print("📦 发现缺失依赖，开始安装...")
        if not install_dependencies(missing):
            return False
    
    print("="*50)
    print("✅ 环境检查完成!")
    
    if cuda_available:
        print("🚀 推荐运行完整实验 (100轮):")
        print("   python train_fractal_vs_standard_cifar10.py")
    else:
        print("🚀 推荐运行快速测试 (5轮):")
        print("   python train_fractal_vs_standard_cifar10.py --epochs 5")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
