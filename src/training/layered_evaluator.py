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
from dataclasses import asdict, dataclass, field
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

# 导入评估层
from .evaluation_layers import (
    LayeredEvaluationReport,
    L1ClassificationMetrics,
    L2TokenizerMetrics,
    L3AttentionMetrics,
    L4RepresentationMetrics,
    L5EfficiencyMetrics,
    L6StabilityMetrics,
    L7SplitterMetrics,
    L8GradientFlowMetrics,
    ClassificationEvaluator,
    TokenizerEvaluator,
    AttentionEvaluator,
    RepresentationEvaluator,
    EfficiencyEvaluator,
    StabilityEvaluator,
    SplitterEvaluator,
    GradientFlowEvaluator,
)

# 导入 CUB-200 专用评估器
try:
    # 使用完整导入路径以避免相对导入问题
    from training.trainer.cub200_trainer import (
        CUB200Trainer,
        CUB200EvalResult,
        CUB200TrainingConfig,
        create_cub200_trainer,
    )
    CUB200_AVAILABLE = True
except ImportError as e:
    CUB200_AVAILABLE = False
    CUB200_IMPORT_ERROR = str(e)


# ============================================================================
# CUB-200 细粒度分类专用评估层
# ============================================================================

@dataclass
class L9FinegrainedMetrics:
    """CUB-200 细粒度分类评估指标

    数学形式化:
        - MCA = (1/C) Σ_c Acc(c)  (平均类准确率)
        - 类内/类间距离比: intra/inter
        - Center Loss 统计: 类中心紧凑性
        - 混淆熵: 衡量分类不确定性

    评估维度:
        1. 整体性能: Top-1/Top-5, MCA
        2. 细粒度指标: Center Loss, 类中心距离
        3. 混淆分析: 最常混淆的类别对
        4. 类别诊断: 易分/难分/缺失类别
        5. 特征空间: 类内/类间距离比
    """
    # ==================== 基础指标 ====================
    top1_accuracy: float = 0.0
    top5_accuracy: float = 0.0
    mean_class_accuracy: float = 0.0  # MCA
    loss: float = 0.0

    # ==================== 细粒度专用指标 ====================
    center_loss: Optional[float] = None
    avg_center_distance: Optional[float] = None

    # ==================== 逐类别统计 ====================
    per_class_accuracy: Optional[np.ndarray] = None  # [C] 逐类别准确率
    per_class_precision: Optional[np.ndarray] = None  # [C] 逐类别精确率
    per_class_recall: Optional[np.ndarray] = None  # [C] 逐类别召回率
    confusion_matrix: Optional[np.ndarray] = None  # [C, C] 混淆矩阵

    # ==================== 混淆分析 ====================
    confused_pairs: Optional[List[Tuple[str, str, int]]] = None  # [(pred, true, count), ...]
    confusion_entropy: Optional[float] = None  # 混淆熵 (分类不确定性)
    most_confused_pairs: List[Tuple[str, str, float]] = field(default_factory=list)  # 高度混淆的类别对

    # ==================== 类内/类间距离分析 ====================
    intra_class_distance: Optional[float] = None  # 平均类内距离
    inter_class_distance: Optional[float] = None  # 平均类间距离
    intra_inter_ratio: Optional[float] = None  # 类内/类间距离比 (越小越好)
    class_centers: Optional[np.ndarray] = None  # [C, D] 类中心

    # ==================== 类别诊断 ====================
    missing_classes: List[int] = field(default_factory=list)  # 准确率为0的类别
    easy_classes: List[Tuple[str, float]] = field(default_factory=list)  # [(class_name, acc), ...]
    hard_classes: List[Tuple[str, float]] = field(default_factory=list)  # [(class_name, acc), ...]
    unbalanced_classes: List[Tuple[str, float, float]] = field(default_factory=list)  # [(name, acc, ideal)]

    # ==================== 困难样本分析 ====================
    hard_samples: List[Tuple[int, int, int, float]] = field(default_factory=list)
    # [(sample_idx, true_label, pred_label, confidence), ...]

    # ==================== 特征统计 ====================
    feature_stats: Optional[Dict[str, float]] = None
    feature_mean_norm: Optional[float] = None
    feature_std_norm: Optional[float] = None

    # ==================== 训练历史摘要 ====================
    best_epoch: Optional[int] = None
    total_epochs: int = 0
    training_time_hours: Optional[float] = None


class FinegrainedClassificationEvaluator:
    """CUB-200 细粒度分类专用评估器

    使用 CUB200Trainer 的 evaluate 方法进行深入评估。

    评估内容:
        - Top-1/Top-5 准确率
        - 平均类准确率 (MCA)
        - 逐类别准确率/精确率/召回率
        - Center Loss 统计（如果可用）
        - 混淆分析（最常混淆的类别对）
        - 类内/类间距离比
        - 困难样本分析
    """

    def __init__(
        self,
        num_classes: int = 200,
        class_names: Optional[List[str]] = None,
    ):
        """初始化细粒度分类评估器

        Args:
            num_classes: 类别数 (CUB-200 为 200)
            class_names: 类别名称列表
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        center_loss_fn: Optional[nn.Module] = None,
        return_features: bool = True,
    ) -> L9FinegrainedMetrics:
        """执行细粒度分类评估

        Args:
            model: FractalCurveViT 模型
            test_loader: 测试数据加载器
            device: 计算设备
            center_loss_fn: Center Loss 函数（可选，用于计算类中心距离）
            return_features: 是否返回特征

        Returns:
            L9FinegrainedMetrics 包含所有评估指标
        """
        import logging
        logger = logging.getLogger(__name__)

        if not CUB200_AVAILABLE:
            logger.warning("CUB200Trainer 不可用，跳过细粒度评估")
            return L9FinegrainedMetrics()

        metrics = L9FinegrainedMetrics()

        try:
            # 使用 CUB200Trainer 进行评估
            batch_size = test_loader.batch_size or 32
            config = CUB200TrainingConfig(
                batch_size=batch_size,
            )

            trainer = CUB200Trainer(
                model=model,
                config=config,
                device=device,
            )

            # 执行评估
            result = trainer.evaluate(test_loader, return_features=return_features)

            # ==================== 基础指标 ====================
            metrics.top1_accuracy = result.accuracy
            metrics.top5_accuracy = result.top5_accuracy
            metrics.mean_class_accuracy = result.mca or 0.0
            metrics.loss = getattr(result, 'loss', 0.0)

            # ==================== 逐类别准确率 ====================
            if result.per_class_accuracy is not None:
                per_class_list = list(result.per_class_accuracy.values())
                metrics.per_class_accuracy = np.array(per_class_list)

                # 计算理想准确率（均匀分布下每类应有 0.5%）
                ideal_acc = 100.0 / self.num_classes

                # 分析易分/难分类别
                valid_acc = [(i, acc) for i, acc in enumerate(per_class_list) if acc > 0]
                valid_acc.sort(key=lambda x: x[1], reverse=True)

                metrics.easy_classes = [(self.class_names[i], acc) for i, acc in valid_acc[:5]]
                metrics.hard_classes = [(self.class_names[i], acc) for i, acc in valid_acc[-5:]]

                # 缺失类别（准确率为0）
                metrics.missing_classes = [i for i, acc in enumerate(per_class_list) if acc == 0]

                # 不平衡类别（准确率与理想值偏差大）
                for i, acc in enumerate(per_class_list):
                    if acc > 0:
                        deviation = abs(acc - ideal_acc)
                        if deviation > 2 * ideal_acc:  # 偏差超过 2 倍
                            metrics.unbalanced_classes.append((self.class_names[i], acc, ideal_acc))

            # ==================== Center Loss 统计 ====================
            if result.feature_stats is not None:
                metrics.center_loss = result.feature_stats.get('center_loss')
                metrics.avg_center_distance = result.feature_stats.get('avg_center_distance')
                metrics.feature_stats = result.feature_stats

            # ==================== 类内/类间距离比 ====================
            metrics.intra_inter_ratio = result.intra_inter_ratio

            # ==================== 混淆分析 ====================
            if result.confused_pairs is not None:
                # 转换混淆对为可读格式
                metrics.confused_pairs = []
                for pred_id, true_id, count in result.confused_pairs:
                    pred_name = self.class_names[pred_id] if pred_id < len(self.class_names) else f"class_{pred_id}"
                    true_name = self.class_names[true_id] if true_id < len(self.class_names) else f"class_{true_id}"
                    metrics.confused_pairs.append((pred_name, true_name, count))

                    # 收集高度混淆的类别对
                    if count >= 3:  # 混淆次数 >= 3
                        metrics.most_confused_pairs.append((pred_name, true_name, count))

            # ==================== 混淆熵计算 ====================
            # 从 confused_pairs 计算混淆熵的近似值
            if result.confused_pairs is not None and len(result.confused_pairs) > 0:
                total_confusions = sum(count for _, _, count in result.confused_pairs)
                if total_confusions > 0:
                    # 计算混淆的均匀程度（越均匀熵越高）
                    probs = [count / total_confusions for _, _, count in result.confused_pairs]
                    probs = [p for p in probs if p > 0]
                    entropy = -sum(p * np.log(p + 1e-10) for p in probs)
                    # 归一化到 [0, 1]
                    max_entropy = np.log(min(len(probs), 10))
                    metrics.confusion_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

            # ==================== 困难样本分析 ====================
            # 收集预测置信度低的样本（困难样本）
            if return_features:
                metrics = self._analyze_hard_samples(metrics, trainer, test_loader, device)

            logger.info(f"CUB-200 细粒度评估完成: Top-1={metrics.top1_accuracy:.2f}%, MCA={metrics.mean_class_accuracy:.2f}%")

        except Exception as e:
            logger.error(f"CUB-200 评估失败: {e}")
            import traceback
            traceback.print_exc()

        return metrics

    def _analyze_hard_samples(
        self,
        metrics: L9FinegrainedMetrics,
        trainer: CUB200Trainer,
        test_loader: DataLoader,
        device: torch.device,
    ) -> L9FinegrainedMetrics:
        """分析困难样本（低置信度误分类）

        数学形式化:
            困难样本: confidence < threshold 且 prediction != label
            置信度: p(y_pred | x) < τ (τ 通常取 0.3-0.5)
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            trainer.model.eval()
            all_labels = []
            all_preds = []
            all_probs = []

            # I35: 使用 get_extra_info API 获取 logits 和辅助信息
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # I35: 优先使用 get_extra_info API（支持分词器诊断信息收集）
                    if hasattr(trainer.model, 'get_extra_info'):
                        outputs, aux_infos = trainer.model.get_extra_info(inputs)
                    else:
                        outputs = trainer.model(inputs)
                        aux_infos = None

                    probs = torch.softmax(outputs, dim=1)
                    preds = outputs.argmax(dim=1)

                    all_labels.extend(labels.cpu().tolist())
                    all_preds.extend(preds.cpu().tolist())
                    all_probs.append(probs.cpu())

            all_probs = torch.cat(all_probs, dim=0)
            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds)

            # 收集困难样本
            for i in range(len(all_labels)):
                true_label = all_labels[i]
                pred_label = all_preds[i]
                confidence = all_probs[i, pred_label].item()

                # 误分类且置信度低
                if pred_label != true_label and confidence < 0.4:
                    metrics.hard_samples.append((i, int(true_label), int(pred_label), confidence))

            # 按置信度排序，取最难的 10 个
            metrics.hard_samples.sort(key=lambda x: x[3])
            metrics.hard_samples = metrics.hard_samples[:10]

            logger.debug(f"发现 {len(metrics.hard_samples)} 个困难样本")

        except Exception as e:
            logger.warning(f"困难样本分析失败: {e}")

        return metrics


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
    # CUB-200-2011: 细粒度鸟类分类数据集
    # N_train=5994, N_test=5794, C=200
    # I35: 使用动态分辨率 (image_size=None) 支持可变尺寸输入
    'cub200': {
        'num_classes': 200,
        'image_size': None,  # 动态分辨率，从数据集中获取实际尺寸
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'dynamic_resolution': True,  # I35: 标记为动态分辨率数据集
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


def download_cub200(data_root: Path) -> bool:
    """下载并设置 CUB-200-2011 细粒度鸟类分类数据集
    
    数学形式化分析
    ================
    数据集规格:
        N_train = 5,994, N_test = 5,794, C = 200
        图像尺寸: 变长 → resize 到 224×224
    
    下载源:
        - Caltech Data: https://data.caltech.edu/records/65de6-vp158
        - 大小: ~1.2GB
    """
    target_dir = data_root / "CUB_200_2011"
    
    # 检查是否已存在并正确组织
    if (target_dir / "train").exists() and (target_dir / "test").exists():
        train_classes = len(list((target_dir / "train").iterdir()))
        test_classes = len(list((target_dir / "test").iterdir()))
        if train_classes >= 200 and test_classes >= 200:
            print(f"[OK] CUB-200-2011 already exists at {target_dir}")
            return True
    
    print("\n" + "="*60)
    print("Downloading CUB-200-2011 Dataset")
    print("="*60)
    print(f"  Target: {target_dir}")
    print(f"  Size: ~1.2GB")
    print(f"  Classes: 200 bird species")
    print("="*60 + "\n")
    
    tgz_path = data_root / "CUB_200_2011.tgz"
    
    # 检查已缓存的文件
    if tgz_path.exists():
        print(f"[OK] Using cached archive: {tgz_path}")
    else:
        # 下载
        url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
        print(f"Downloading from: {url}")
        if not download_with_progress(url, tgz_path, "CUB-200-2011"):
            print("\n[ERROR] Download failed.")
            print("Please download manually from:")
            print("  https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz")
            print(f"And place it at: {tgz_path}")
            return False
    
    # 解压
    print("\nExtracting...")
    try:
        import tarfile
        with tarfile.open(tgz_path, 'r:gz') as tar:
            members = tar.getmembers()
            with tqdm(total=len(members), desc="Extracting", unit="files") as pbar:
                for member in members:
                    # filter='data' 兼容 Python 3.14+ (PEP 706)
                    tar.extract(member, data_root, filter='data')
                    pbar.update(1)
        print("[OK] Extraction complete")
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        return False
    
    # 组织为 train/test 目录结构
    print("\nOrganizing dataset by train/test split...")
    
    images_dir = target_dir / "images"
    split_file = target_dir / "train_test_split.txt"
    images_file = target_dir / "images.txt"
    
    if not images_dir.exists() or not split_file.exists() or not images_file.exists():
        print(f"[ERROR] Required files not found")
        return False
    
    # 读取图像列表
    image_id_to_path = {}
    with open(images_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img_id, img_path = parts
                image_id_to_path[img_id] = img_path
    
    # 读取 train/test 划分
    train_ids = set()
    test_ids = set()
    with open(split_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img_id, is_train = parts
                if is_train == '1':
                    train_ids.add(img_id)
                else:
                    test_ids.add(img_id)
    
    # 创建目录并复制文件
    train_dir = target_dir / "train"
    test_dir = target_dir / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    for img_id, img_path in tqdm(image_id_to_path.items(), desc="Organizing"):
        class_name = img_path.split('/')[0]
        img_name = img_path.split('/')[-1]
        src = images_dir / img_path
        
        if img_id in train_ids:
            dst_dir = train_dir / class_name
        else:
            dst_dir = test_dir / class_name
        
        dst_dir.mkdir(exist_ok=True)
        dst = dst_dir / img_name
        
        if src.exists() and not dst.exists():
            shutil.copy2(str(src), str(dst))
    
    print(f"\n[OK] CUB-200-2011 organized")
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
    elif dataset_name == 'cub200':
        return download_cub200(data_root)
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
        exp_config: Optional[Dict[str, Any]] = None,  # 实验配置（从 config.json 加载）
    ):
        """初始化分层评估器

        Parameters
        ----------
        checkpoint_path : str
            模型 checkpoint 文件路径
        dataset_name : str
            数据集名称 (cifar10, cifar100, mnist, tiny-imagenet, cub200)
        batch_size : int
            评估批次大小
        num_workers : int
            数据加载线程数
        device : str, optional
            设备 (cuda/cpu)，默认自动检测
        data_root : str, optional
            数据根目录，默认为项目 data 目录
        exp_config : dict, optional
            从 config.json 加载的实验配置
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.exp_config = exp_config or {}  # 保存实验配置

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
        raw_config = None
        if 'config' in checkpoint:
            raw_config = checkpoint['config']
        elif 'model_config' in checkpoint:
            raw_config = checkpoint['model_config']
        else:
            # 尝试从同目录加载 config.json
            config_path = self.checkpoint_path.parent / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    raw_config = json.load(f)
            else:
                raise ValueError("Cannot find model config in checkpoint or config.json")

        # 将配置转为字典 (如果是 dataclass 或 namespace)
        if hasattr(raw_config, '__dict__'):
            raw_config = vars(raw_config)
        elif hasattr(raw_config, '_asdict'):
            raw_config = raw_config._asdict()

        # 处理嵌套配置结构 (ExperimentConfig 保存为 {"model": {...}, "training": {...}, ...})
        # 优先使用 model 子配置，fallback 到扁平结构
        if 'model' in raw_config and isinstance(raw_config['model'], dict):
            config = raw_config['model']
            print("[INFO] Using 'model' section from nested config")
        else:
            config = raw_config

        # 合并 experiment_config 中的训练配置 (如果存在)
        if 'training' in raw_config and isinstance(raw_config['training'], dict):
            # 训练配置中的参数可能与模型配置互补
            training_config = raw_config['training']
            for key, value in training_config.items():
                if key not in config or config[key] is None:
                    config[key] = value

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
        # I30-17: 优先使用 checkpoint 中保存的配置
        # 优先级：checkpoint config > state_dict 推断 > 默认值
        max_depth_limit = config.get('max_depth_limit', None)
        num_scales_from_config = config.get('num_scales', None)

        # 兼容旧检查点：max_depth_hard_limit -> max_depth_limit
        if max_depth_limit is None:
            max_depth_limit = config.get('max_depth_hard_limit', None)

        # 如果 config 中有 num_scales，优先使用
        if num_scales_from_config is not None and max_depth_limit is None:
            max_depth_limit = num_scales_from_config - 1
            print(f"Using num_scales={num_scales_from_config} from checkpoint config")

        min_patch_size = config.get('min_patch_size', 4)
        if isinstance(min_patch_size, int):
            min_patch_size = (min_patch_size, min_patch_size)

        # 从数据集配置获取 num_classes
        num_classes = config.get('num_classes', self.dataset_config['num_classes'])
        image_size = config.get('image_size', self.dataset_config['image_size'])

        # I35: 动态分辨率支持 - 对于支持动态分辨率的数据集，使用 None
        if self.dataset_config.get('dynamic_resolution', False):
            if image_size is not None and not isinstance(image_size, int):
                # 如果配置中是具体值但数据集标记为动态分辨率，使用 None
                image_size = None
                print("Using dynamic resolution (image_size=None) for variable-size dataset")
        elif image_size is None:
            # 传统数据集不能使用动态分辨率，使用默认值 224
            image_size = 224
            print(f"Warning: Fixed-size dataset but image_size is None, using default {image_size}")

        # 加载权重以检测缺失的架构参数
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # 直接是 state_dict
            state_dict = checkpoint

        # P11-2: 从检查点恢复所有架构参数（与训练器完全对齐）
        # 优先级：检查点 state_dict > config.json > 默认值

        # 1. 检测 tokenizer_type（从 state_dict 结构）
        tokenizer_type = config.get('tokenizer_type', None)
        has_fractal_tokenizer = any(k.startswith('fractal_tokenizer') for k in state_dict.keys())
        has_tokenizer = any(k.startswith('tokenizer') and not k.startswith('fractal_tokenizer') for k in state_dict.keys())
        has_new_tokenizer = any(k.startswith('_orig_mod.tokenizer') for k in state_dict.keys())

        if has_fractal_tokenizer and not has_tokenizer:
            if tokenizer_type is None:
                tokenizer_type = 'fractal'
                print("Detected tokenizer_type='fractal' from checkpoint (old format)")
        elif has_tokenizer or has_new_tokenizer:
            if tokenizer_type is None:
                tokenizer_type = 'streaming_v3'
                print("Detected tokenizer_type='streaming_v3' from checkpoint (new format)")

        # 2. 检测 num_scales（仅当 config 中没有配置时）
        # I78: 修复混合版本 checkpoint 问题
        num_scales = None  # 初始化，确保在 if 块外可用
        if max_depth_limit is None:
            from collections import Counter
            depth_related_keys = []

            for k in state_dict.keys():
                # 收集所有与深度相关的参数
                if any(pattern in k for pattern in [
                    'threshold_offsets', 'quota_logits', '_depth_ema_mean', '_depth_ema_var',
                    'depth_embedding', '_depth_scale_raw', 'depth_embed'
                ]):
                    param_shape = state_dict[k].shape
                    # 处理 torch.Size 对象（支持索引访问）
                    if hasattr(param_shape, '__len__') and len(param_shape) > 0:
                        depth_related_keys.append((k, int(param_shape[0])))

            if depth_related_keys:
                # 统计各维度的出现次数，取众数（出现最频繁的维度）
                dim_counts = Counter(dim for _, dim in depth_related_keys)
                detected_num_scales = dim_counts.most_common(1)[0][0]
                num_scales = detected_num_scales
                max_depth_limit = detected_num_scales - 1

                # 打印维度分布
                dims_summary = ', '.join([f"{dim}×{count}" for dim, count in dim_counts.most_common()])
                print(f"Detected num_scales={num_scales} (max_depth_limit={max_depth_limit}) from {len(depth_related_keys)} depth-related parameters")
                print(f"  Dimension distribution: {dims_summary}")
        elif num_scales_from_config is not None:
            # 如果 max_depth_limit 来自 config，使用 num_scales_from_config
            num_scales = num_scales_from_config
        elif max_depth_limit is not None:
            # 如果 max_depth_limit 有值但 num_scales 未设置，从 max_depth_limit 推断
            num_scales = max_depth_limit + 1

        # 3. 从 num_scales 反推 min_patch_size（与训练器逻辑一致）
        # 公式: min_patch_size = image_size / 2^max_depth
        detected_min_patch_size = config.get('min_patch_size', None)
        if detected_min_patch_size is None and num_scales is not None:
            # 从 num_scales 反推: max_depth = num_scales - 1
            max_depth = num_scales - 1
            # 假设正方形图像: min_patch_size = image_size / 2^max_depth
            if isinstance(image_size, int):
                effective_image_size = (image_size, image_size)
            elif isinstance(image_size, tuple):
                effective_image_size = image_size
            else:
                effective_image_size = (224, 224)  # 默认值
            min_size = min(effective_image_size)
            detected_min_patch_size = max(1, min_size // (2 ** max_depth))
            print(f"Inferred min_patch_size={detected_min_patch_size} from num_scales={num_scales}")
        if detected_min_patch_size is None:
            detected_min_patch_size = 4  # 默认值
        if isinstance(detected_min_patch_size, int):
            min_patch_size = (detected_min_patch_size, detected_min_patch_size)
        else:
            min_patch_size = detected_min_patch_size

        # 4. 检测模型架构参数（dim, depth, heads, dim_head）
        ckpt_dim = config.get('dim', None)
        ckpt_depth = config.get('depth', None)
        ckpt_heads = config.get('heads', None)
        ckpt_dim_head = config.get('dim_head', None)

        # 检测 dim（从 to_qkv.weight: [3*dim, dim]）
        if ckpt_dim is None:
            for key in state_dict.keys():
                if 'to_qkv.weight' in key:
                    qkv_shape = state_dict[key].shape
                    if len(qkv_shape) == 2 and qkv_shape[0] == 3 * qkv_shape[1]:
                        ckpt_dim = qkv_shape[1]
                        print(f"Detected dim={ckpt_dim} from checkpoint")
                    break
        if ckpt_dim is None:
            ckpt_dim = 384  # 默认值

        # 检测 depth（从 transformer.layers 数量）
        if ckpt_depth is None:
            depth_count = 0
            for key in state_dict.keys():
                if key.startswith('transformer.layers.'):
                    parts = key.split('.')
                    if len(parts) > 2:
                        try:
                            layer_idx = int(parts[2])
                            depth_count = max(depth_count, layer_idx + 1)
                        except ValueError:
                            pass
            if depth_count > 0:
                ckpt_depth = depth_count
                print(f"Detected depth={ckpt_depth} from checkpoint")
        if ckpt_depth is None:
            ckpt_depth = 10  # 默认值

        # 检测 heads 和 dim_head（从 to_qkv.weight）
        if ckpt_heads is None or ckpt_dim_head is None:
            for key in state_dict.keys():
                if 'to_qkv.weight' in key:
                    qkv_shape = state_dict[key].shape
                    if len(qkv_shape) == 2:
                        total_dim = qkv_shape[1]
                        total_heads_dim = qkv_shape[0] // 3
                        # 推断 heads（假设 dim_head=64）
                        ckpt_dim_head = config.get('dim_head', 64)
                        if ckpt_heads is None:
                            ckpt_heads = total_heads_dim // ckpt_dim_head
                            print(f"Detected heads={ckpt_heads} from checkpoint")
                        break
        if ckpt_heads is None:
            ckpt_heads = 6  # 默认值
        if ckpt_dim_head is None:
            ckpt_dim_head = 64  # 默认值

        # 5. 检测 num_classes（从 mlp_head 或 head）
        detected_num_classes = config.get('num_classes', None)
        if detected_num_classes is None:
            for key in reversed(list(state_dict.keys())):
                if key.startswith('mlp_head') and '.weight' in key:
                    detected_num_classes = state_dict[key].shape[0]
                    print(f"Detected num_classes={detected_num_classes} from checkpoint")
                    break
                elif key == 'head.4.weight':
                    detected_num_classes = state_dict[key].shape[0]
                    print(f"Detected num_classes={detected_num_classes} from checkpoint")
                    break
        if detected_num_classes is None:
            detected_num_classes = num_classes
        num_classes = detected_num_classes

        # 6. 检测其他关键参数
        # dropout
        dropout = config.get('dropout', 0.1)
        emb_dropout = config.get('emb_dropout', 0.1)
        drop_path_rate = config.get('drop_path_rate', 0.15)

        # Pool 类型
        pool = config.get('pool', 'cls')

        # 编码器选项
        use_hilbert_encoding = config.get('use_hilbert_encoding', True)
        use_spatial_encoding = config.get('use_spatial_encoding', True)
        use_checkpoint = config.get('use_checkpoint', False)

        # FFN 类型
        ffn_type = config.get('ffn_type', 'swiglu_level')

        # LCA 温度
        lca_temperature = config.get('lca_temperature', 1.5)
        learnable_temperature = config.get('learnable_temperature', True)

        # I24-2: 可学习配额
        quota_learnable = config.get('quota_learnable', None)

        # I31-3: 形状-尺度编码
        use_area_encoding = config.get('use_area_encoding', False)
        use_affine_modulation = config.get('use_affine_modulation', True)
        fourier_levels = config.get('fourier_levels', 4)

        # 子模块 Dropout
        splitter_dropout = config.get('splitter_dropout', None)
        pos_dropout = config.get('pos_dropout', None)

        # 计算 mlp_dim（与训练器一致）
        mlp_dim = config.get('mlp_dim', ckpt_dim * 4)

        # channels
        channels = config.get('channels', 3)

        # 创建模型（使用检测到的所有参数）
        model = FractalCurveViT(
            image_size=image_size,
            num_classes=num_classes,
            dim=ckpt_dim,
            depth=ckpt_depth,
            heads=ckpt_heads,
            mlp_dim=mlp_dim,
            pool=pool,
            channels=channels,
            dim_head=ckpt_dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            min_patch_size=min_patch_size,
            max_level=None,  # P11-2: None = 自动从 tokenizer.max_depth 获取
            # I30-17: 使用 num_scales 参数（兼容性）
            num_scales=num_scales,
            use_hilbert_encoding=use_hilbert_encoding,
            use_spatial_encoding=use_spatial_encoding,
            use_checkpoint=use_checkpoint,
            drop_path_rate=drop_path_rate,
            ffn_type=ffn_type,
            tokenizer_type=tokenizer_type if tokenizer_type else 'streaming_v3',
            lca_temperature=lca_temperature,
            learnable_temperature=learnable_temperature,
            # I23-2: Token 数量约束（使用检测或默认值）
            K_min=16,
            K_max=64,
            # I27: 子模块 Dropout 配置
            splitter_dropout=splitter_dropout,
            pos_dropout=pos_dropout,
            # I31-3: 形状-尺度编码配置
            use_area_encoding=use_area_encoding,
            use_affine_modulation=use_affine_modulation,
            fourier_levels=fourier_levels,
            # I24-2: 可学习配额控制
            quota_learnable=quota_learnable,
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

        # I78: 过滤掉形状不匹配的参数（处理混合版本 checkpoint）
        # 即使 strict=False，shape mismatch 仍会抛出错误
        filtered_state_dict = {}
        skipped_mismatches = []
        for k, v in state_dict.items():
            if k in model.state_dict():
                model_param = model.state_dict()[k]
                if v.shape == model_param.shape:
                    filtered_state_dict[k] = v
                else:
                    skipped_mismatches.append((k, v.shape, model_param.shape))
            # else: 参数不存在于模型中，忽略（strict=False 会处理）

        if skipped_mismatches:
            print(f"Warning: Skipped {len(skipped_mismatches)} parameters with shape mismatch:")
            for k, ckpt_shape, model_shape in skipped_mismatches[:5]:
                print(f"  - {k}: checkpoint {tuple(ckpt_shape)} vs model {tuple(model_shape)}")
            if len(skipped_mismatches) > 5:
                print(f"  ... and {len(skipped_mismatches) - 5} more")
            print(f"  (This indicates a mixed-version or corrupted checkpoint)")

        # 加载过滤后的权重
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        
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
        
        elif self.dataset_name == 'cub200':
            # CUB-200-2011 细粒度鸟类分类
            train_path = data_root / "CUB_200_2011" / "train"
            test_path = data_root / "CUB_200_2011" / "test"
            
            if not train_path.exists() or not test_path.exists():
                raise FileNotFoundError(
                    f"CUB-200-2011 not found at {data_root / 'CUB_200_2011'}. "
                    "Please download it first."
                )
            
            train_dataset = datasets.ImageFolder(str(train_path), transform=test_transform)
            test_dataset = datasets.ImageFolder(str(test_path), transform=test_transform)
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
        evaluate_train: bool = False,  # 是否评估训练集（CUB-200 专用）
    ) -> LayeredEvaluationReport:
        """执行完整分层评估

        Parameters
        ----------
        skip_layers : list of str, optional
            跳过的层，例如 ['L3', 'L5']
        evaluate_train : bool
            是否评估训练集（用于过拟合诊断，仅 CUB-200 支持）

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

        # ====================================================================
        # L9-Train: 训练集评估（CUB-200 专用，用于过拟合诊断）
        # ====================================================================
        train_finegrained_metrics = None
        if evaluate_train and self.dataset_name == 'cub200' and self.train_loader is not None:
            if CUB200_AVAILABLE:
                print("\n[L9-Train] CUB-200 Training Set Evaluation...")
                num_classes = self.dataset_config.get('num_classes', 200)
                train_finegrained_eval = FinegrainedClassificationEvaluator(
                    num_classes=num_classes
                )
                train_finegrained_metrics = train_finegrained_eval.evaluate(
                    self.model, self.train_loader, self.device
                )

                # 打印训练集结果
                print(f"\n  [训练集性能]")
                print(f"    Top-1 Accuracy:  {train_finegrained_metrics.top1_accuracy:.2f}%")
                print(f"    Top-5 Accuracy:  {train_finegrained_metrics.top5_accuracy:.2f}%")
                print(f"    Mean Class Acc:  {train_finegrained_metrics.mean_class_accuracy:.2f}%")

                # 保存到报告（如果支持）
                if hasattr(report, 'L9_train'):
                    report.L9_train = train_finegrained_metrics
            else:
                print(f"  - CUB200Trainer not available: {CUB200_IMPORT_ERROR}")
        
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
        
        # L7: 分割器专项评估
        if 'L7' not in skip_layers:
            print("\n[L7] Splitter Evaluation...")
            splitter_eval = SplitterEvaluator()
            report.L7_splitter = splitter_eval.evaluate(
                self.model, eval_loader, self.device
            )
            if report.L7_splitter.temperature > 0:
                print(f"  - Temperature: {report.L7_splitter.temperature:.3f}")
                print(f"  - Selection prob mean: {report.L7_splitter.selection_prob_mean:.3f}")
                print(f"  - Decision confidence: {report.L7_splitter.decision_confidence_mean:.3f}")
                if report.L7_splitter.quotas:
                    print(f"  - Quotas: {report.L7_splitter.quotas}")
            else:
                print("  - (Splitter metrics not available)")
        
        # L8: 梯度流评估 (需要一个 sample batch)
        if 'L8' not in skip_layers:
            print("\n[L8] Gradient Flow Evaluation...")
            gradient_eval = GradientFlowEvaluator()
            sample_batch = next(iter(eval_loader))
            sample_imgs, sample_labels = sample_batch[0][:4], sample_batch[1][:4]
            report.L8_gradient_flow = gradient_eval.evaluate(
                self.model, sample_imgs, sample_labels, self.device
            )
            print(f"  - Total grad norm: {report.L8_gradient_flow.total_grad_norm:.4f}")
            print(f"  - Max grad norm: {report.L8_gradient_flow.max_grad_norm:.4f}")
            print(f"  - Vanishing gradients: {len(report.L8_gradient_flow.vanishing_gradients)}")
            print(f"  - Exploding gradients: {len(report.L8_gradient_flow.exploding_gradients)}")

        # L9: CUB-200 细粒度分类专用评估（仅当数据集为 cub200 时触发）
        if self.dataset_name == 'cub200' and 'L9' not in skip_layers:
            print("\n[L9] CUB-200 Fine-grained Classification Evaluation...")
            if CUB200_AVAILABLE:
                # 确保 dataset_config 已设置
                if self.dataset_config is None:
                    self.dataset_config = SUPPORTED_DATASETS.get(self.dataset_name, {'num_classes': 200})

                num_classes = self.dataset_config.get('num_classes', 200)
                finegrained_eval = FinegrainedClassificationEvaluator(
                    num_classes=num_classes
                )
                finegrained_metrics = finegrained_eval.evaluate(
                    self.model, eval_loader, self.device
                )

                # ==================== 打印详细结果 ====================
                print(f"\n  [基础性能]")
                print(f"    Top-1 Accuracy:  {finegrained_metrics.top1_accuracy:.2f}%")
                print(f"    Top-5 Accuracy:  {finegrained_metrics.top5_accuracy:.2f}%")
                print(f"    Mean Class Acc:  {finegrained_metrics.mean_class_accuracy:.2f}%")

                if finegrained_metrics.center_loss is not None:
                    print(f"\n  [Center Loss 分析]")
                    print(f"    Center Loss:     {finegrained_metrics.center_loss:.4f}")
                    print(f"    Avg Center Dist: {finegrained_metrics.avg_center_distance:.2f}")

                if finegrained_metrics.intra_inter_ratio is not None:
                    print(f"\n  [特征空间分析]")
                    print(f"    Intra/Inter Ratio: {finegrained_metrics.intra_inter_ratio:.2f} (< 1.0 为佳)")

                if finegrained_metrics.confusion_entropy is not None:
                    print(f"\n  [混淆分析]")
                    print(f"    Confusion Entropy: {finegrained_metrics.confusion_entropy:.2f} (越高越均匀)")

                if finegrained_metrics.most_confused_pairs:
                    print(f"\n  [高度混淆的类别对 (≥3次)]")
                    for pred_name, true_name, count in finegrained_metrics.most_confused_pairs[:5]:
                        print(f"    {pred_name} <-> {true_name}: {count}次")

                if finegrained_metrics.unbalanced_classes:
                    print(f"\n  [不平衡类别 (与理想值偏差大)]")
                    for name, acc, ideal in finegrained_metrics.unbalanced_classes[:3]:
                        deviation = acc - ideal
                        sign = '+' if deviation > 0 else ''
                        print(f"    {name}: {acc:.1f}% (理想:{ideal:.1f}%, 偏差:{sign}{deviation:.1f}%)")

                if finegrained_metrics.missing_classes:
                    print(f"\n  [缺失类别 (准确率=0%)]")
                    missing_count = len(finegrained_metrics.missing_classes)
                    print(f"    共 {missing_count} 个类别无正确预测")

                if finegrained_metrics.hard_classes:
                    print(f"\n  [最难分类别 (Top-5)]")
                    for cls_name, acc in finegrained_metrics.hard_classes[:5]:
                        print(f"    {cls_name}: {acc:.1f}%")

                if finegrained_metrics.easy_classes:
                    print(f"\n  [最易分类别 (Top-5)]")
                    for cls_name, acc in finegrained_metrics.easy_classes[:5]:
                        print(f"    {cls_name}: {acc:.1f}%")

                if finegrained_metrics.hard_samples:
                    print(f"\n  [困难样本 (置信度<40%的误分类)]")
                    for idx, true_label, pred_label, conf in finegrained_metrics.hard_samples[:5]:
                        true_name = f"class_{true_label}"
                        pred_name = f"class_{pred_label}"
                        print(f"    Sample #{idx}: {true_name} -> {pred_name} (conf:{conf:.2f})")
            else:
                print(f"  - CUB200Trainer not available: {CUB200_IMPORT_ERROR}")

        # ====================================================================
        # L9-Compare: 训练集与测试集对比分析（CUB-200 过拟合诊断）
        # ====================================================================
        if train_finegrained_metrics is not None and finegrained_metrics is not None:
            print("\n[L9-Compare] Train vs Test Comparison (Overfitting Diagnosis)")

            # 计算对比指标
            train_top1 = train_finegrained_metrics.top1_accuracy
            test_top1 = finegrained_metrics.top1_accuracy
            train_mca = train_finegrained_metrics.mean_class_accuracy
            test_mca = finegrained_metrics.mean_class_accuracy
            train_top5 = train_finegrained_metrics.top5_accuracy
            test_top5 = finegrained_metrics.top5_accuracy

            overfit_ratio = train_top1 / max(test_top1, 0.01)
            mca_gap = train_mca - test_mca
            top5_gap = train_top5 - test_top5

            # 打印对比结果
            print(f"\n  [基础指标对比]")
            print(f"    {'指标':<15} {'训练集':>10} {'测试集':>10} {'差值':>10}")
            print(f"    {'-'*45}")
            print(f"    {'Top-1 Acc':<15} {train_top1:>9.2f}% {test_top1:>9.2f}% {test_top1 - train_top1:>+9.2f}%")
            print(f"    {'MCA':<15} {train_mca:>9.2f}% {test_mca:>9.2f}% {test_mca - train_mca:>+9.2f}%")
            print(f"    {'Top-5 Acc':<15} {train_top5:>9.2f}% {test_top5:>9.2f}% {test_top5 - train_top5:>+9.2f}%")

            print(f"\n  [过拟合诊断]")
            print(f"    Overfit Ratio: {overfit_ratio:.3f} (训练/测试, >1.15 表示过拟合)")
            print(f"    MCA Gap: {mca_gap:+.2f}%")

            # 过拟合警告
            if overfit_ratio > 1.15:
                print(f"\n    ⚠️  过拟合警告: 训练准确率比测试高 {(overfit_ratio - 1) * 100:.1f}%")
                summary['warnings'].append(
                    f"Overfitting detected: overfit_ratio={overfit_ratio:.3f}"
                )
                summary['recommendations'].append(
                    "Consider stronger regularization, data augmentation, or early stopping"
                )
                if summary['overall_health'] != 'critical':
                    summary['overall_health'] = 'warning'

            # 保存对比结果到报告
            if not hasattr(report, 'L9_comparison'):
                report.L9_comparison = {}

            report.L9_comparison = {
                'train_top1': train_top1,
                'test_top1': test_top1,
                'train_mca': train_mca,
                'test_mca': test_mca,
                'train_top5': train_top5,
                'test_top5': test_top5,
                'overfit_ratio': overfit_ratio,
                'mca_gap': mca_gap,
                'top5_gap': top5_gap,
            }

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
        
        # L2: 深度坍缩检测
        if report.L2_tokenizer.depth_collapse_detected:
            summary['warnings'].append(
                f"Depth collapse detected! KL={report.L2_tokenizer.depth_kl_from_uniform:.2f}, "
                f"entropy_ratio={report.L2_tokenizer.depth_entropy_ratio:.2f}"
            )
            summary['recommendations'].append(
                "Enable LOG_COMPENSATION or increase DEPTH_KL_WEIGHT in constants.py"
            )
            if summary['overall_health'] != 'critical':
                summary['overall_health'] = 'warning'
        
        # L3: LCA 利用率
        if report.L3_attention.lca_attention_correlation is not None:
            if abs(report.L3_attention.lca_attention_correlation) < 0.1:
                summary['warnings'].append(
                    "Low LCA-attention correlation - Hilbert locality not utilized"
                )
                summary['recommendations'].append(
                    "Check LCA bias learning, may need to adjust lca_temperature"
                )
        
        # L7: 分割器健康
        if report.L7_splitter is not None:
            if report.L7_splitter.decision_confidence_mean < 0.3:
                summary['warnings'].append(
                    f"Low splitter confidence: {report.L7_splitter.decision_confidence_mean:.2f}"
                )
                summary['recommendations'].append(
                    "Splitter is uncertain - consider lower temperature or more training"
                )
            
            if report.L7_splitter.quota_entropy > 0 and report.L7_splitter.quota_entropy < 0.5:
                summary['warnings'].append("Quotas collapsing to single depth")
                summary['recommendations'].append("Increase quota regularization")
        
        # L8: 梯度流
        if report.L8_gradient_flow is not None:
            if len(report.L8_gradient_flow.vanishing_gradients) > 5:
                summary['warnings'].append(
                    f"{len(report.L8_gradient_flow.vanishing_gradients)} parameters with vanishing gradients"
                )
                summary['recommendations'].append("Check residual connections and initialization")
            
            if len(report.L8_gradient_flow.exploding_gradients) > 0:
                summary['warnings'].append(
                    f"{len(report.L8_gradient_flow.exploding_gradients)} parameters with exploding gradients"
                )
                summary['recommendations'].append("Reduce learning rate or add gradient clipping")
                if summary['overall_health'] != 'critical':
                    summary['overall_health'] = 'warning'
        
        # 更新报告的 warnings/recommendations
        report.warnings = summary['warnings']
        report.recommendations = summary['recommendations']
        
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
# 工厂函数
# ============================================================================

def create_evaluator(
    checkpoint_path: str,
    dataset_name: str = "cifar10",
    config: Optional['EvaluationConfig'] = None,
    **kwargs,
) -> LayeredEvaluator:
    """创建分层评估器的工厂函数
    
    提供便捷的评估器创建接口，支持从 EvaluationConfig 配置类创建。
    
    参数
    ----
    checkpoint_path : str
        模型 checkpoint 路径
    dataset_name : str
        数据集名称
    config : EvaluationConfig, optional
        评估配置，如果提供则覆盖 kwargs 中的对应参数
    **kwargs : dict
        传递给 LayeredEvaluator 的其他参数
    
    返回
    ----
    LayeredEvaluator
        配置好的评估器实例

    示例
    ----
    >>> from training import EvaluationConfig, create_evaluator
    >>>
    >>> # 使用默认配置
    >>> evaluator = create_evaluator("checkpoint.pt", "cifar10")
    >>> 
    >>> # 使用自定义配置
    >>> config = EvaluationConfig(
    ...     max_samples=5000,
    ...     enabled_layers=["L1", "L2", "L5"],
    ... )
    >>> evaluator = create_evaluator("checkpoint.pt", "tiny-imagenet", config=config)
    >>> report = evaluator.run_full_evaluation(skip_layers=["L3", "L4", "L6"])
    """
    # 从 config 提取参数
    if config is not None:
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = config.batch_size
    
    return LayeredEvaluator(
        checkpoint_path=checkpoint_path,
        dataset_name=dataset_name,
        **kwargs,
    )


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
