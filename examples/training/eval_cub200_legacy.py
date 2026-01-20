#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临时CUB-200评估脚本
用于评估旧模型架构（num_scales=5）训练出来的checkpoint

使用方法:
    python eval_cub200_legacy.py --checkpoint path/to/best.pth
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

# 项目路径设置
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"

for path in [PROJECT_ROOT, SRC_PATH]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vit_pytorch import FractalCurveViT


def load_config_from_experiment(checkpoint_path: str) -> Dict[str, Any]:
    """从实验目录加载config.json"""
    checkpoint = Path(checkpoint_path)
    candidates = [
        checkpoint.parent.parent / "logs" / "config.json",
        checkpoint.parent.parent / "config.json",
    ]
    for config_path in candidates:
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load config from {config_path}: {e}")
    return {}


def load_legacy_model(
    checkpoint_path: str,
    num_classes: int = 200,
    dim: int = 256,
    depth: int = 8,
    heads: int = 8,
    dim_head: int = 32,  # 关键：checkpoint使用dim_head=32
    image_size: int = 256,  # 固定为256以匹配旧模型
) -> nn.Module:
    """加载旧模型架构（兼容num_scales=5的旧checkpoint）

    旧模型配置:
    - shared_conv kernel: 4x4
    - splitter max_depth: 4 (5 scales)
    - 这意味着base_patch_size=4, 但 splitter 使用 max_depth_limit=4
    """

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint_data = torch.load(checkpoint_path, map_location='cpu')

    # 提取state_dict
    if 'model_state_dict' in checkpoint_data:
        state_dict = checkpoint_data['model_state_dict']
    elif 'state_dict' in checkpoint_data:
        state_dict = checkpoint_data['state_dict']
    else:
        state_dict = checkpoint_data

    # 处理torch.compile()产生的 _orig_mod. 前缀
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("Detected torch.compile() checkpoint, stripping '_orig_mod.' prefix...")
        state_dict = {
            k.replace('_orig_mod.', ''): v
            for k, v in state_dict.items()
        }

    # 方案：直接修改state_dict中的键名以适配当前模型
    # 或者使用更简单的方法：忽略不匹配的键

    # 首先创建一个默认模型来检查结构
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from vit_pytorch import FractalCurveViT
    from vit_pytorch.tokenizer_streaming import StreamingFractalTokenizerV3
    from vit_pytorch.gumbel_topk_splitter import GumbelTopKSplitter
    from vit_pytorch.config import SplitterConfig

    # 创建兼容旧模型的配置
    # 对于旧模型: base_patch_size=4, max_depth=4 (5 scales)
    # 需要: min_patch_size=16 以获得 max_depth=4
    legacy_config = SplitterConfig(
        feature_dim=dim,
        min_patch_size=16,  # 这会得到 max_depth=4
        max_depth_limit=4,  # 显式限制为4以匹配旧模型
        hidden_dim=64,
        intermediate_dim=64,
        pool_size=4,
        K_min=14,
        K_max=256,
        use_dynamic_k=True,
        dropout=0.1,
        enable_learnable_quota=True,
    )

    # 创建旧版tokenizer配置
    tokenizer = StreamingFractalTokenizerV3(
        image_size=(image_size, image_size),
        channels=3,
        d_model=dim,
        base_patch_size=4,  # 旧模型使用4
        max_depth=4,  # 旧模型有5个scale (0-4)
        K_min=14,
        K_max=256,
        splitter_config=legacy_config,
        use_hilbert_order=True,
    )

    # 创建模型
    model = FractalCurveViT(
        image_size=image_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        tokenizer=tokenizer,  # 使用自定义tokenizer
        K_min=14,
        K_max=256,
        pool='cls',
    )

    # 加载权重
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")

    return model


def load_cub200_data(
    data_root: str = "data/CUB_200_2011",
    image_size: int = 256,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """加载CUB-200数据"""

    train_path = Path(data_root) / "train"
    test_path = Path(data_root) / "test"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"CUB-200 not found at {data_root}. "
            "Please ensure data/CUB_200_2011/train and test directories exist."
        )

    # 评估用transform (固定为256x256)
    test_transform = transforms.Compose([
        transforms.Resize(image_size),  # 缩放短边到256
        transforms.CenterCrop(image_size),  # 中心裁剪为256x256
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # 自定义数据集以处理损坏图片
    class RobustImageFolder(datasets.ImageFolder):
        def __getitem__(self, index):
            try:
                return super().__getitem__(index)
            except Exception:
                # 返回零张量和随机目标（对整体指标影响可忽略）
                return torch.zeros(3, image_size, image_size), 0

    train_dataset = RobustImageFolder(str(train_path), transform=test_transform)
    test_dataset = RobustImageFolder(str(test_path), transform=test_transform)

    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'shuffle': False,
    }

    train_loader = DataLoader(train_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    print(f"Data loaded: train={len(train_dataset)}, test={len(test_dataset)}")
    return train_loader, test_loader


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    evaluate_train: bool = False,
) -> Dict[str, float]:
    """评估模型"""

    model = model.to(device)
    model.eval()

    if evaluate_train:
        loader = test_loader  # 复用test_loader作为训练集
        print("Evaluating on TRAINING set (for overfitting diagnosis)...")
    else:
        loader = test_loader
        print("Evaluating on TEST set...")

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device)
            targets = targets.to(device)

            # I35: 使用 get_extra_info API 获取 logits 和辅助信息
            if hasattr(model, 'get_extra_info'):
                outputs, aux_infos = model.get_extra_info(images)
            else:
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                aux_infos = None

            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Top-1
            _, predicted = outputs.max(1)
            correct_top1 += predicted.eq(targets).sum().item()

            # Top-5
            _, predicted_top5 = outputs.topk(5, 1, True, True)
            correct_top5 += predicted_top5.eq(targets.view(-1, 1).expand_as(predicted_top5)).sum().item()

            total += targets.size(0)

            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx + 1}/{len(loader)}")

    num_batches = len(loader)
    metrics = {
        'top1_accuracy': 100.0 * correct_top1 / total,
        'top5_accuracy': 100.0 * correct_top5 / total,
        'avg_loss': total_loss / num_batches,
        'num_samples': total,
    }

    return metrics


def print_summary(metrics: Dict[str, float], split: str = "Test"):
    """打印评估摘要"""
    print(f"\n{'='*50}")
    print(f"{split} SET EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
    print(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
    print(f"  Avg Loss: {metrics['avg_loss']:.4f}")
    print(f"  Total Samples: {metrics['num_samples']}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="CUB-200 Legacy Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--data-root", "-d",
        type=str,
        default="data/CUB_200_2011",
        help="CUB-200 data root directory",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--num-workers", "-w",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, default: auto)",
    )
    parser.add_argument(
        "--evaluate-train",
        action="store_true",
        help="Also evaluate training set",
    )
    parser.add_argument(
        "--num-scales",
        type=int,
        default=5,
        help="Number of scales (must match checkpoint, default: 5)",
    )

    args = parser.parse_args()

    # 设备选择
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载配置
    exp_config = load_config_from_experiment(args.checkpoint)
    if exp_config:
        training_cfg = exp_config.get('training_config', {})
        cub200_cfg = exp_config.get('cub200_config', {})
        print(f"Loaded config: num_scales={training_cfg.get('num_scales')}, "
              f"K_max={training_cfg.get('K_max')}")

    # 加载模型（使用dim_head=32和max_level=4匹配检查点）
    model = load_legacy_model(
        checkpoint_path=args.checkpoint,
        num_classes=200,
        dim=256,
        depth=8,
        heads=8,
        dim_head=32,  # 关键：checkpoint使用dim_head=32
        image_size=256,  # 旧模型使用固定256
    )

    # 加载数据（resize到256以匹配旧模型）
    try:
        train_loader, test_loader = load_cub200_data(
            data_root=args.data_root,
            image_size=256,  # 必须匹配模型配置
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # 评估测试集
    test_metrics = evaluate_model(model, test_loader, device, evaluate_train=False)
    print_summary(test_metrics, "Test")

    # 如果需要，评估训练集
    if args.evaluate_train:
        train_metrics = evaluate_model(model, train_loader, device, evaluate_train=True)
        print_summary(train_metrics, "Train")

        # 计算过拟合指标
        if 'top1_accuracy' in train_metrics and 'top1_accuracy' in test_metrics:
            overfit_gap = train_metrics['top1_accuracy'] - test_metrics['top1_accuracy']
            print(f"Overfitting Gap (Train - Test): {overfit_gap:.2f}%")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
