#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分层评估器 (Layered Evaluator)

本模块实现 FractalCurveViT 的完整分层评估系统，作为独立的评估入口点。

使用方式
=========

命令行使用::

    python layered_evaluator.py --checkpoint path/to/checkpoint.pt --dataset cifar10
    python layered_evaluator.py --checkpoint path/to/checkpoint.pt --dataset tiny-imagenet --output report.json

编程接口::

    from layered_evaluator import LayeredEvaluator
    
    evaluator = LayeredEvaluator(
        checkpoint_path="path/to/checkpoint.pt",
        dataset_name="cifar10",
    )
    report = evaluator.run_full_evaluation()
    report.save_json("evaluation_report.json")

评估架构
=========

分层评估遵循模型架构层次：

    ┌─────────────────────────────────────────────────────────────┐
    │                    LayeredEvaluator                          │
    │  ┌─────────────────────────────────────────────────────────┐ │
    │  │  L6: StabilityEvaluator   (权重健康、数值稳定性)         │ │
    │  ├─────────────────────────────────────────────────────────┤ │
    │  │  L5: EfficiencyEvaluator  (延迟、吞吐量、内存)           │ │
    │  ├─────────────────────────────────────────────────────────┤ │
    │  │  L4: RepresentationEvaluator (Fisher 比、可分性)         │ │
    │  ├─────────────────────────────────────────────────────────┤ │
    │  │  L3: AttentionEvaluator   (注意力熵、Head 利用率)        │ │
    │  ├─────────────────────────────────────────────────────────┤ │
    │  │  L2: TokenizerEvaluator   (Token 数、深度分布)           │ │
    │  ├─────────────────────────────────────────────────────────┤ │
    │  │  L1: ClassificationEvaluator (准确率、ECE、混淆矩阵)     │ │
    │  └─────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────┘

Author: GitHub Copilot
Date: 2026-01-11
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
import warnings
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 过滤非关键警告
warnings.filterwarnings("ignore", message="Truncating the start/stop/step of slice")
warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom is <= 0")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# 添加项目路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "examples" / "training"))

# 导入评估层
from evaluation_layers import (
    LayeredEvaluationReport,
    L1ClassificationMetrics,
    L2TokenizerMetrics,
    L3AttentionMetrics,
    L4RepresentationMetrics,
    L5EfficiencyMetrics,
    L6StabilityMetrics,
    ClassificationEvaluator,
    TokenizerEvaluator,
    AttentionEvaluator,
    RepresentationEvaluator,
    EfficiencyEvaluator,
    StabilityEvaluator,
)


# ============================================================================
# 数据集配置
# ============================================================================

SUPPORTED_DATASETS = {
    'cifar10': {
        'num_classes': 10,
        'image_size': 32,
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2470, 0.2435, 0.2616],
    },
    'cifar100': {
        'num_classes': 100,
        'image_size': 32,
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761],
    },
    'mnist': {
        'num_classes': 10,
        'image_size': 28,
        'mean': [0.1307],
        'std': [0.3081],
    },
    'tiny-imagenet': {
        'num_classes': 200,
        'image_size': 64,
        'mean': [0.4802, 0.4481, 0.3975],
        'std': [0.2302, 0.2265, 0.2262],
    },
}


# ============================================================================
# 数据集下载工具
# ============================================================================

def download_with_progress(url: str, dest: Path, desc: str = "Downloading") -> bool:
    """带进度条的下载函数"""
    try:
        # 获取文件大小
        with urllib.request.urlopen(url, timeout=30) as response:
            total_size = int(response.headers.get('Content-Length', 0))
        
        # 下载
        downloaded = 0
        block_size = 8192
        
        with urllib.request.urlopen(url, timeout=30) as response:
            with open(dest, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        f.write(buffer)
                        downloaded += len(buffer)
                        pbar.update(len(buffer))
        
        return True
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_tiny_imagenet(data_root: Path) -> bool:
    """下载并设置 Tiny ImageNet
    
    数据集信息:
    - 200 类，每类 500 张训练图像
    - 训练集: 100,000 张 64x64 图像
    - 验证集: 10,000 张图像
    """
    target_dir = data_root / "tiny-imagenet-200"
    
    # 检查是否已存在
    if (target_dir / "train").exists() and (target_dir / "val").exists():
        train_classes = len(list((target_dir / "train").iterdir()))
        val_has_classes = any((target_dir / "val").iterdir())
        if train_classes >= 200 and val_has_classes:
            print(f"[OK] Tiny ImageNet already exists at {target_dir}")
            return True
    
    print("\n" + "="*60)
    print("Downloading Tiny ImageNet Dataset")
    print("="*60)
    print(f"  Target: {target_dir}")
    print(f"  Size: ~237MB")
    print("="*60 + "\n")
    
    zip_path = data_root / "tiny-imagenet-200.zip"
    
    # 检查已缓存的 zip 是否有效
    if zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                if zf.testzip() is not None:
                    raise zipfile.BadZipFile("Corrupted zip file")
                if len(zf.namelist()) < 100:
                    raise zipfile.BadZipFile("Incomplete zip file")
            print(f"[OK] Using cached zip: {zip_path}")
        except (zipfile.BadZipFile, Exception) as e:
            print(f"[WARN] Cached zip is invalid: {e}")
            print("[*] Removing corrupted file and re-downloading...")
            zip_path.unlink()
    
    # 尝试多个下载源
    urls = [
        "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
        "https://image-net.org/data/tiny-imagenet-200.zip",
    ]
    
    if not zip_path.exists():
        download_success = False
        for i, url in enumerate(urls):
            print(f"[{i+1}/{len(urls)}] Trying: {url}")
            if download_with_progress(url, zip_path, "Tiny ImageNet"):
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        if zf.testzip() is not None:
                            raise zipfile.BadZipFile("Downloaded file is corrupted")
                    download_success = True
                    print("[OK] Download complete and verified")
                    break
                except zipfile.BadZipFile as e:
                    print(f"[WARN] Downloaded file is invalid: {e}")
                    if zip_path.exists():
                        zip_path.unlink()
            print(f"[WARN] Failed, trying next source...")
        
        if not download_success:
            print("\n[ERROR] All download sources failed.")
            print("Please download manually from:")
            print("  http://cs231n.stanford.edu/tiny-imagenet-200.zip")
            print(f"And place it at: {zip_path}")
            return False
    
    # 解压
    print("\nExtracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            total = len(zf.namelist())
            with tqdm(total=total, desc="Extracting", unit="files") as pbar:
                for member in zf.namelist():
                    zf.extract(member, data_root)
                    pbar.update(1)
        print("[OK] Extraction complete")
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        return False
    
    # 组织验证集（原始格式是所有图片在一个文件夹）
    val_dir = target_dir / "val"
    val_images_dir = val_dir / "images"
    
    if val_images_dir.exists():
        print("\nOrganizing validation set by class...")
        val_annotations = val_dir / "val_annotations.txt"
        
        if val_annotations.exists():
            with open(val_annotations, 'r') as f:
                lines = f.readlines()
            
            for line in tqdm(lines, desc="Organizing"):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_name, class_id = parts[0], parts[1]
                    class_dir = val_dir / class_id / "images"
                    class_dir.mkdir(parents=True, exist_ok=True)
                    src = val_images_dir / img_name
                    dst = class_dir / img_name
                    if src.exists() and not dst.exists():
                        shutil.move(str(src), str(dst))
            
            if val_images_dir.exists():
                shutil.rmtree(val_images_dir)
            
            print("[OK] Validation set organized")
        else:
            print("[WARN] val_annotations.txt not found")
    
    # 验证
    train_classes = len(list((target_dir / "train").iterdir()))
    val_classes = len([d for d in (target_dir / "val").iterdir() if d.is_dir()])
    print(f"\n[OK] Dataset ready:")
    print(f"  Train classes: {train_classes}")
    print(f"  Val classes: {val_classes}")
    
    # 清理 zip
    if zip_path.exists():
        zip_path.unlink()
        print("[OK] Cleaned up zip file")
    
    return True


def ensure_dataset_available(dataset_name: str, data_root: Path) -> bool:
    """确保数据集可用，如果不存在则下载
    
    Args:
        dataset_name: 数据集名称
        data_root: 数据根目录
        
    Returns:
        bool: 数据集是否可用
    """
    data_root.mkdir(parents=True, exist_ok=True)
    
    if dataset_name == 'cifar10':
        # CIFAR-10 会由 torchvision 自动下载
        return True
    elif dataset_name == 'cifar100':
        # CIFAR-100 会由 torchvision 自动下载
        return True
    elif dataset_name == 'mnist':
        # MNIST 会由 torchvision 自动下载
        return True
    elif dataset_name == 'tiny-imagenet':
        return download_tiny_imagenet(data_root)
    else:
        print(f"[WARN] Unknown dataset: {dataset_name}")
        return False


# ============================================================================
# JSON 编码器
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    """支持 NumPy 类型的 JSON 编码器"""
    
    def default(self, o):
        import numpy as np
        
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


# ============================================================================
# 主评估器类
# ============================================================================

class LayeredEvaluator:
    """分层评估器主类
    
    提供完整的分层评估功能，按照模型架构层次进行评估：
    
    - L1: 分类性能 (准确率、校准误差、混淆分析)
    - L2: Tokenizer 行为 (Token 数、深度分布、空间覆盖)
    - L3: 注意力机制 (注意力熵、Head 利用率)
    - L4: 特征表示 (Fisher 判别比、类别可分性)
    - L5: 资源效率 (延迟、吞吐量、内存)
    - L6: 训练稳定性 (权重范数、数值稳定性)
    
    示例
    -----
    >>> evaluator = LayeredEvaluator(
    ...     checkpoint_path="experiments/run_1/best_model.pt",
    ...     dataset_name="cifar10",
    ... )
    >>> report = evaluator.run_full_evaluation()
    >>> report.save_json("report.json")
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        dataset_name: str = "cifar10",
        batch_size: int = 64,
        num_workers: int = 4,
        device: Optional[str] = None,
        data_root: Optional[str] = None,
    ):
        """初始化分层评估器
        
        Parameters
        ----------
        checkpoint_path : str
            模型 checkpoint 文件路径
        dataset_name : str
            数据集名称 (cifar10, cifar100, mnist, tiny-imagenet)
        batch_size : int
            评估批次大小
        num_workers : int
            数据加载线程数
        device : str, optional
            设备 (cuda/cpu)，默认自动检测
        data_root : str, optional
            数据根目录，默认为项目 data 目录
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # 设备设置
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 数据路径
        if data_root is None:
            self.data_root = PROJECT_ROOT / "data"
        else:
            self.data_root = Path(data_root)
        
        # 验证
        self._validate_config()
        
        # 加载模型和数据
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.dataset_config = None
        
    def _validate_config(self):
        """验证配置"""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        if self.dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {self.dataset_name}. "
                f"Supported: {list(SUPPORTED_DATASETS.keys())}"
            )
    
    def _load_model(self) -> nn.Module:
        """加载模型 checkpoint"""
        print(f"Loading checkpoint: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # 获取模型配置
        if 'config' in checkpoint:
            config = checkpoint['config']
        elif 'model_config' in checkpoint:
            config = checkpoint['model_config']
        else:
            # 尝试从同目录加载 config.json
            config_path = self.checkpoint_path.parent / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
            else:
                raise ValueError("Cannot find model config in checkpoint or config.json")
        
        # 将配置转为字典 (如果是 dataclass 或 namespace)
        if hasattr(config, '__dict__'):
            config = vars(config)
        elif hasattr(config, '_asdict'):
            config = config._asdict()
        
        # 从 checkpoint 的 config 中检测数据集，如果与用户指定不同则警告
        ckpt_dataset = config.get('dataset', self.dataset_name)
        if ckpt_dataset != self.dataset_name:
            print(f"⚠️  Warning: Checkpoint was trained on '{ckpt_dataset}', but you specified '{self.dataset_name}'")
            print(f"    Overriding to use '{ckpt_dataset}' from checkpoint config")
            self.dataset_name = ckpt_dataset
            if self.dataset_name not in SUPPORTED_DATASETS:
                raise ValueError(f"Dataset '{self.dataset_name}' from checkpoint is not supported")
        
        # 更新 dataset_config
        self.dataset_config = SUPPORTED_DATASETS[self.dataset_name]
        
        # 构建模型
        from vit_pytorch import FractalCurveViT
        
        # 获取 tokenizer 相关配置
        # 支持旧格式的配置名称映射
        num_scales = config.get('num_scales', 4)
        min_patch_size = config.get('min_patch_size', 4)
        if isinstance(min_patch_size, int):
            min_patch_size = (min_patch_size, min_patch_size)
        
        # 从数据集配置获取 num_classes
        num_classes = config.get('num_classes', self.dataset_config['num_classes'])
        image_size = config.get('image_size', self.dataset_config['image_size'])
        
        model = FractalCurveViT(
            image_size=image_size,
            num_classes=num_classes,
            dim=config.get('dim', 256),
            depth=config.get('depth', 6),
            heads=config.get('heads', 8),
            mlp_dim=config.get('mlp_dim', config.get('dim', 256) * 4),
            pool=config.get('pool', 'cls'),
            channels=config.get('channels', 3),
            dim_head=config.get('dim_head', 64),
            dropout=config.get('dropout', 0.1),
            emb_dropout=config.get('emb_dropout', 0.1),
            min_patch_size=min_patch_size,
            max_level=config.get('max_level', None),
            use_hilbert_encoding=config.get('use_hilbert_encoding', True),
            use_spatial_encoding=config.get('use_spatial_encoding', True),
            use_checkpoint=config.get('use_checkpoint', False),
            drop_path_rate=config.get('drop_path_rate', 0.0),
            ffn_type=config.get('ffn_type', 'swiglu_level'),
            tokenizer_type=config.get('tokenizer_type', 'streaming_v3'),
            num_scales=num_scales,
            lca_temperature=config.get('lca_temperature', 1.5),
            learnable_temperature=config.get('learnable_temperature', True),
        )
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # 直接是 state_dict
            state_dict = checkpoint
        
        # 处理 torch.compile() 产生的 _orig_mod. 前缀
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            print("Detected torch.compile() checkpoint, stripping '_orig_mod.' prefix...")
            state_dict = {
                k.replace('_orig_mod.', ''): v 
                for k, v in state_dict.items()
            }
        
        # 加载权重 (strict=False 以处理可能的细微差异)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing {len(missing_keys)} keys in state_dict")
            # 只打印前 5 个
            for k in missing_keys[:5]:
                print(f"  - {k}")
            if len(missing_keys) > 5:
                print(f"  ... and {len(missing_keys) - 5} more")
        
        if unexpected_keys:
            print(f"Warning: Unexpected {len(unexpected_keys)} keys in state_dict")
            for k in unexpected_keys[:5]:
                print(f"  - {k}")
            if len(unexpected_keys) > 5:
                print(f"  ... and {len(unexpected_keys) - 5} more")
        
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded successfully. Device: {self.device}")
        
        return model
    
    def _load_data(self) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """加载评估数据，如果数据集不存在则自动下载"""
        print(f"Loading dataset: {self.dataset_name}")
        
        self.dataset_config = SUPPORTED_DATASETS[self.dataset_name]
        config = self.dataset_config
        
        # 确保数据集可用（自动下载）
        if not ensure_dataset_available(self.dataset_name, self.data_root):
            raise RuntimeError(f"Failed to ensure dataset '{self.dataset_name}' is available")
        
        # 评估用 transform (不需要数据增强)
        if self.dataset_name == 'mnist':
            test_transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(config['mean'], config['std']),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 转为 3 通道
            ])
        else:
            test_transform = transforms.Compose([
                transforms.Resize(max(config['image_size'], 32)),
                transforms.ToTensor(),
                transforms.Normalize(config['mean'], config['std']),
            ])
        
        # 加载数据集
        data_root = self.data_root
        
        if self.dataset_name == 'cifar10':
            train_dataset = datasets.CIFAR10(
                root=str(data_root),
                train=True,
                download=True,
                transform=test_transform,  # 评估时不需要增强
            )
            test_dataset = datasets.CIFAR10(
                root=str(data_root),
                train=False,
                download=True,
                transform=test_transform,
            )
            
        elif self.dataset_name == 'cifar100':
            train_dataset = datasets.CIFAR100(
                root=str(data_root),
                train=True,
                download=True,
                transform=test_transform,
            )
            test_dataset = datasets.CIFAR100(
                root=str(data_root),
                train=False,
                download=True,
                transform=test_transform,
            )
            
        elif self.dataset_name == 'mnist':
            train_dataset = datasets.MNIST(
                root=str(data_root),
                train=True,
                download=True,
                transform=test_transform,
            )
            test_dataset = datasets.MNIST(
                root=str(data_root),
                train=False,
                download=True,
                transform=test_transform,
            )
            
        elif self.dataset_name == 'tiny-imagenet':
            train_path = data_root / "tiny-imagenet-200" / "train"
            val_path = data_root / "tiny-imagenet-200" / "val"
            
            if not train_path.exists() or not val_path.exists():
                raise FileNotFoundError(
                    f"Tiny ImageNet not found at {data_root / 'tiny-imagenet-200'}. "
                    "Please download it first."
                )
            
            train_dataset = datasets.ImageFolder(str(train_path), transform=test_transform)
            test_dataset = datasets.ImageFolder(str(val_path), transform=test_transform)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        # 创建 DataLoader
        loader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.num_workers > 0 and torch.cuda.is_available(),
            'shuffle': False,  # 评估不需要 shuffle
        }
        
        if self.num_workers > 0:
            loader_kwargs['persistent_workers'] = True
            loader_kwargs['prefetch_factor'] = 4
        
        train_loader = DataLoader(train_dataset, **loader_kwargs)
        test_loader = DataLoader(test_dataset, **loader_kwargs)
        
        print(f"[OK] Data: train={len(train_dataset)}, test={len(test_dataset)}")
        
        return train_loader, test_loader, test_loader
    
    def run_full_evaluation(
        self,
        skip_layers: Optional[List[str]] = None,
    ) -> LayeredEvaluationReport:
        """执行完整分层评估
        
        Parameters
        ----------
        skip_layers : list of str, optional
            跳过的层，例如 ['L3', 'L5']
            
        Returns
        -------
        LayeredEvaluationReport
            完整的分层评估报告
        """
        start_time = time.time()
        skip_layers = skip_layers or []
        
        print("=" * 60)
        print("FractalCurveViT Layered Evaluation")
        print("=" * 60)
        
        # 加载模型和数据
        self.model = self._load_model()
        self.train_loader, self.val_loader, self.test_loader = self._load_data()
        
        eval_loader = self.test_loader or self.val_loader
        
        # 创建报告
        report = LayeredEvaluationReport(
            checkpoint_path=str(self.checkpoint_path),
            dataset_name=self.dataset_name,
            num_samples=len(eval_loader.dataset),
            num_classes=self.dataset_config['num_classes'],
            device=str(self.device),
        )
        
        # L6: 稳定性评估 (最先进行，检查模型健康)
        if 'L6' not in skip_layers:
            print("\n[L6] Stability Evaluation...")
            stability_eval = StabilityEvaluator()
            report.L6_stability = stability_eval.evaluate(self.model)
            print(f"  - NaN weights: {report.L6_stability.has_nan_weights}")
            print(f"  - Inf weights: {report.L6_stability.has_inf_weights}")
            print(f"  - Health score: {report.L6_stability.gradient_health_score:.2f}")
        
        # L5: 效率评估
        if 'L5' not in skip_layers:
            print("\n[L5] Efficiency Evaluation...")
            efficiency_eval = EfficiencyEvaluator()
            sample_input = next(iter(eval_loader))[0][:8]  # 用 8 张图测试
            report.L5_efficiency = efficiency_eval.evaluate(
                self.model, sample_input, self.device
            )
            print(f"  - Avg latency: {report.L5_efficiency.avg_latency_ms:.2f} ms")
            print(f"  - Throughput: {report.L5_efficiency.throughput_samples_per_sec:.1f} samples/s")
            print(f"  - Peak memory: {report.L5_efficiency.peak_memory_mb:.1f} MB")
        
        # L1: 分类性能评估
        if 'L1' not in skip_layers:
            print("\n[L1] Classification Evaluation...")
            classification_eval = ClassificationEvaluator(
                num_classes=self.dataset_config['num_classes']
            )
            report.L1_classification = classification_eval.evaluate(
                self.model, eval_loader, self.device
            )
            print(f"  - Top-1 Accuracy: {report.L1_classification.top1_accuracy:.2f}%")
            print(f"  - Top-5 Accuracy: {report.L1_classification.top5_accuracy:.2f}%")
            print(f"  - Mean Class Accuracy: {report.L1_classification.mean_class_accuracy:.2f}%")
            print(f"  - ECE: {report.L1_classification.ece:.2f}%")
        
        # L2: Tokenizer 行为评估
        if 'L2' not in skip_layers:
            print("\n[L2] Tokenizer Evaluation...")
            tokenizer_eval = TokenizerEvaluator()
            report.L2_tokenizer = tokenizer_eval.evaluate(
                self.model, eval_loader, self.device
            )
            print(f"  - Avg tokens: {report.L2_tokenizer.avg_tokens:.1f}")
            print(f"  - Token range: [{report.L2_tokenizer.min_tokens}, {report.L2_tokenizer.max_tokens}]")
            print(f"  - Depth entropy: {report.L2_tokenizer.depth_entropy:.3f}")
            print(f"  - Spatial coverage: {report.L2_tokenizer.spatial_coverage_ratio:.2%}")
        
        # L3: 注意力机制评估
        if 'L3' not in skip_layers:
            print("\n[L3] Attention Evaluation...")
            attention_eval = AttentionEvaluator()
            report.L3_attention = attention_eval.evaluate(
                self.model, eval_loader, self.device
            )
            if report.L3_attention.avg_entropy > 0:
                print(f"  - Avg entropy: {report.L3_attention.avg_entropy:.3f}")
                print(f"  - Dead head ratio: {report.L3_attention.dead_head_ratio:.2%}")
            else:
                print("  - (Attention metrics not available - hook not captured)")
        
        # L4: 特征表示评估
        if 'L4' not in skip_layers:
            print("\n[L4] Representation Evaluation...")
            representation_eval = RepresentationEvaluator()
            report.L4_representation = representation_eval.evaluate(
                self.model, eval_loader, self.device
            )
            print(f"  - Fisher Discriminant Ratio: {report.L4_representation.fisher_discriminant_ratio:.2f}")
            print(f"  - Feature mean norm: {report.L4_representation.feature_mean_norm:.2f}")
            print(f"  - Avg separability: {report.L4_representation.avg_separability:.2f}")
        
        # 完成
        report.evaluation_time_sec = time.time() - start_time
        
        # 生成摘要
        report.summary = self._generate_summary(report)
        
        print("\n" + "=" * 60)
        print(f"Evaluation completed in {report.evaluation_time_sec:.1f}s")
        print("=" * 60)
        
        return report
    
    def _generate_summary(self, report: LayeredEvaluationReport) -> Dict[str, Any]:
        """生成评估摘要"""
        summary = {
            'overall_health': 'healthy',
            'key_metrics': {
                'accuracy': report.L1_classification.top1_accuracy,
                'calibration_error': report.L1_classification.ece,
                'avg_tokens': report.L2_tokenizer.avg_tokens,
                'throughput': report.L5_efficiency.throughput_samples_per_sec,
            },
            'warnings': [],
            'recommendations': [],
        }
        
        # 检查问题
        if report.L6_stability.has_nan_weights or report.L6_stability.has_inf_weights:
            summary['overall_health'] = 'critical'
            summary['warnings'].append("Model has NaN/Inf weights!")
        
        if report.L1_classification.ece > 10:
            summary['warnings'].append(f"High calibration error: {report.L1_classification.ece:.1f}%")
            summary['recommendations'].append("Consider label smoothing or temperature scaling")
        
        if report.L2_tokenizer.depth_entropy < 0.5:
            summary['warnings'].append("Low depth diversity - tokenizer may be collapsing")
            summary['recommendations'].append("Check tokenizer splitter training")
        
        if report.L4_representation.fisher_discriminant_ratio < 1.0:
            summary['warnings'].append("Low class separability in feature space")
            summary['recommendations'].append("Consider more training or stronger augmentation")
        
        return summary
    
    def save_report(
        self,
        report: LayeredEvaluationReport,
        output_path: Union[str, Path],
    ) -> None:
        """保存评估报告到 JSON 文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_dict = report.to_dict()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        print(f"Report saved to: {output_path}")


# ============================================================================
# 命令行入口
# ============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="FractalCurveViT Layered Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python layered_evaluator.py --checkpoint path/to/checkpoint.pt --dataset cifar10
  python layered_evaluator.py --checkpoint path/to/checkpoint.pt --dataset tiny-imagenet --output report.json
  python layered_evaluator.py --checkpoint path/to/checkpoint.pt --dataset cifar100 --skip L3 L5
        """,
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint',
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='cifar10',
        choices=list(SUPPORTED_DATASETS.keys()),
        help='Dataset name (default: cifar10)',
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output JSON report path (default: auto-generated)',
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=64,
        help='Evaluation batch size (default: 64)',
    )
    
    parser.add_argument(
        '--num-workers', '-w',
        type=int,
        default=4,
        help='Data loader workers (default: 4)',
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda/cpu, default: auto)',
    )
    
    parser.add_argument(
        '--skip',
        type=str,
        nargs='+',
        default=[],
        help='Layers to skip (e.g., --skip L3 L5)',
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        default=None,
        help='Data root directory (default: project/data)',
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建评估器
    evaluator = LayeredEvaluator(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        data_root=args.data_root,
    )
    
    # 执行评估
    report = evaluator.run_full_evaluation(skip_layers=args.skip)
    
    # 确定输出路径 (使用评估器中可能已被更新的 dataset_name)
    if args.output:
        output_path = args.output
    else:
        checkpoint_dir = Path(args.checkpoint).parent
        # 使用评估器中实际使用的数据集名称
        actual_dataset = evaluator.dataset_name
        output_path = checkpoint_dir / f"layered_evaluation_{actual_dataset}.json"
    
    # 保存报告
    evaluator.save_report(report, output_path)
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Overall Health: {report.summary.get('overall_health', 'unknown').upper()}")
    print("\nKey Metrics:")
    for key, value in report.summary.get('key_metrics', {}).items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.2f}")
        else:
            print(f"  - {key}: {value}")
    
    if report.summary.get('warnings'):
        print("\nWarnings:")
        for warning in report.summary['warnings']:
            print(f"  ⚠️  {warning}")
    
    if report.summary.get('recommendations'):
        print("\nRecommendations:")
        for rec in report.summary['recommendations']:
            print(f"  💡 {rec}")


if __name__ == '__main__':
    main()
