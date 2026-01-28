#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立评估入口

使用 InferenceWrapper 保证与训练器推理逻辑一致。

Usage:
    python src/training/eval.py --checkpoint path/to/model.pth --dataset cifar10
    python src/training/eval.py --checkpoint path/to/model.pth --dataset tiny-imagenet --batch-size 128

Author: Claude
Date: 2026-01-27
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# 添加项目路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

from training.core.checkpoint import load_model, load_checkpoint
from training.core.inference_wrapper import evaluate as evaluate_fn, EvalResult


# ============================================================================
# 数据集配置
# ============================================================================

DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    'cifar10': {
        'num_classes': 10,
        'image_size': 32,
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2470, 0.2435, 0.2616],
        'dataset_class': datasets.CIFAR10,
    },
    'cifar100': {
        'num_classes': 100,
        'image_size': 32,
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761],
        'dataset_class': datasets.CIFAR100,
    },
    'mnist': {
        'num_classes': 10,
        'image_size': 28,
        'mean': [0.1307],
        'std': [0.3081],
        'dataset_class': datasets.MNIST,
    },
    'tiny-imagenet': {
        'num_classes': 200,
        'image_size': 64,
        'mean': [0.4802, 0.4481, 0.3975],
        'std': [0.2302, 0.2265, 0.2262],
        'dataset_class': None,  # 特殊处理
    },
    'cub200': {
        'num_classes': 200,
        'image_size': None,  # 动态分辨率
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'dataset_class': None,  # 特殊处理
    },
}


# ============================================================================
# 评估管道
# ============================================================================

class EvalPipeline:
    """独立评估管道 - 使用 InferenceWrapper"""

    def __init__(
        self,
        checkpoint_path: str,
        dataset_name: str = "cifar10",
        batch_size: int = 64,
        device: Optional[str] = None,
        data_root: Optional[str] = None,
        use_amp: bool = False,
        output_path: Optional[str] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.data_root = Path(data_root) if data_root else PROJECT_ROOT / "data"
        self.use_amp = use_amp
        self.output_path = Path(output_path) if output_path else None

        self.model: Optional[nn.Module] = None
        self.gene: Optional[ModelGene] = None
        self.loader: Optional[DataLoader] = None

    def run(self) -> EvalResult:
        """执行评估"""
        print("=" * 60)
        print("FractalCurveViT Evaluation")
        print("=" * 60)
        print(f"  Checkpoint: {self.checkpoint_path}")
        print(f"  Dataset: {self.dataset_name}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Device: {self.device}")
        print("=" * 60)

        # 1. 加载模型
        self._load_model()

        # 2. 加载数据
        self._load_data()

        # 3. 评估
        result = evaluate_fn(
            self.model, self.loader,
            device=self.device,
            use_amp=self.use_amp,
            num_classes=self.gene.num_classes if self.gene else None,
        )

        # 4. 打印结果
        self._print_result(result)

        # 5. 保存报告
        if self.output_path:
            self._save_report(result)

        return result

    def _load_model(self):
        """加载模型（使用共享的 checkpoint 加载模块）"""
        print("\n[1/3] Loading model...")

        # 使用共享的 load_checkpoint 函数
        checkpoint = load_checkpoint(
            self.checkpoint_path,
            device=str(self.device),
        )

        # 检查 model_gene
        if 'model_gene' not in checkpoint:
            raise ValueError(
                f"Checkpoint does not contain 'model_gene' key.\n"
                f"This checkpoint was saved with an older version.\n"
                f"Please retrain with the updated training script,\n"
                f"or use layered_evaluator.py which supports legacy checkpoints."
            )

        # 使用共享的 load_model 函数（与训练器完全一致的代码路径）
        model, self.gene = load_model(
            checkpoint['model_gene'],
            state_dict=checkpoint.get('model_state_dict', checkpoint),
            device=str(self.device),
            strict=False,
            verbose=True,
        )

        self.model = model

    def _load_data(self):
        """加载数据加载器"""
        print("\n[2/3] Loading data...")

        if self.dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        config = DATASET_CONFIGS[self.dataset_name]

        # MNIST 需要特殊处理（转为 3 通道）
        if self.dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(config['mean'], config['std']),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ])
        else:
            # 动态分辨率数据集使用自适应 resize
            if config['image_size'] is None:
                resize_size = 256  # 对于动态分辨率，resize 到较大
            else:
                resize_size = max(config['image_size'], 32)

            transform = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.ToTensor(),
                transforms.Normalize(config['mean'], config['std']),
            ])

        # 加载数据集
        if self.dataset_name == 'tiny-imagenet':
            test_path = self.data_root / "tiny-imagenet-200" / "val"
            if not test_path.exists():
                raise FileNotFoundError(
                    f"Tiny ImageNet not found at {test_path}\n"
                    f"Please download it first."
                )
            test_dataset = datasets.ImageFolder(str(test_path), transform=transform)

        elif self.dataset_name == 'cub200':
            test_path = self.data_root / "CUB_200_2011" / "test"
            if not test_path.exists():
                raise FileNotFoundError(
                    f"CUB-200 not found at {test_path}\n"
                    f"Please download it first."
                )
            test_dataset = datasets.ImageFolder(str(test_path), transform=transform)

        else:
            dataset_class = config['dataset_class']
            test_dataset = dataset_class(
                root=str(self.data_root),
                train=False,
                download=True,
                transform=transform,
            )

        self.loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )

        print(f"  Dataset: {self.dataset_name}")
        print(f"  Samples: {len(test_dataset)}")
        print(f"  Classes: {config['num_classes']}")
        print(f"  [OK] Data loaded successfully")

    def _print_result(self, result: EvalResult):
        """打印评估结果"""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"  Accuracy:       {result.accuracy:.2f}%")
        if result.top5_accuracy > 0:
            print(f"  Top-5 Accuracy: {result.top5_accuracy:.2f}%")
        print(f"  Avg Loss:       {result.avg_loss:.4f}")
        print(f"  Samples:        {result.num_samples}")
        print(f"  ECE:            {result.ece:.2f}%")
        print("=" * 60)

        # 逐类别准确率（如果可用）
        if result.per_class_accuracy and len(result.per_class_accuracy) > 0:
            acc_values = list(result.per_class_accuracy.values())
            print(f"\n  Per-Class Stats:")
            print(f"    Best:  {max(acc_values):.2f}%")
            print(f"    Worst: {min(acc_values):.2f}%")
            print(f"    Mean:  {sum(acc_values)/len(acc_values):.2f}%")

    def _save_report(self, result: EvalResult):
        """保存评估报告"""
        if self.output_path is None:
            return

        report = {
            'meta': {
                'checkpoint_path': str(self.checkpoint_path),
                'dataset_name': self.dataset_name,
                'device': str(self.device),
            },
            'model_gene': self.gene.to_dict() if self.gene else None,
            'metrics': {
                'accuracy': result.accuracy,
                'top5_accuracy': result.top5_accuracy,
                'avg_loss': result.avg_loss,
                'num_samples': result.num_samples,
                'ece': result.ece,
            },
            'per_class_accuracy': result.per_class_accuracy,
        }

        output_path = self.output_path  # Type checker narrowing
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n[OK] Report saved to: {output_path}")


# ============================================================================
# 命令行入口
# ============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="FractalCurveViT Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/training/eval.py -c checkpoints/best.pth -d cifar10
  python src/training/eval.py -c checkpoints/best.pth -d tiny-imagenet -b 128
  python src/training/eval.py -c checkpoints/best.pth -d cub200 --amp
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
        choices=list(DATASET_CONFIGS.keys()),
        help='Dataset name (default: cifar10)',
    )

    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=64,
        help='Batch size (default: 64)',
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda/cpu, default: auto)',
    )

    parser.add_argument(
        '--amp',
        action='store_true',
        help='Use AMP (automatic mixed precision)',
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output JSON report path (default: auto-generated)',
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

    # 自动生成输出路径
    if args.output is None:
        ckpt_dir = Path(args.checkpoint).parent
        args.output = str(ckpt_dir / f"eval_{args.dataset}.json")

    pipeline = EvalPipeline(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        device=args.device,
        data_root=args.data_root,
        use_amp=args.amp,
        output_path=args.output,
    )

    try:
        result = pipeline.run()
        return 0 if result.accuracy > 0 else 1
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
