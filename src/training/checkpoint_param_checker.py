#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数一致性检查工具

用于验证训练、保存、评估三个阶段的参数一致性。

三层参数定义：
- 参数（第一层）：每次调用模型都需要传递的固定参数
- 变参数（第二层）：由模型架构内部动态计算的参数
- 超参数（第三层）：应当固定不变、保存在模型架构中的参数

Usage:
    # 检查 checkpoint 参数
    uv run python src/training/checkpoint_param_checker.py --checkpoint path/to/checkpoint.pth

    # 对比两个 checkpoint
    uv run python src/training/checkpoint_param_checker.py --checkpoint1 a.pth --checkpoint2 b.pth
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

# 添加 src 目录到路径（与 train_fractal_vit.py 一致）
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# 参数层级定义
PARAMETER_TIERS = {
    # ========== 第一层：架构参数（固定） ==========
    "layer1_params": {
        "name": "架构参数",
        "description": "每次调用模型都需要传递的固定参数",
        "params": [
            "dim", "num_layers", "heads", "mlp_dim", "num_classes",
            "image_size", "channels", "pool", "min_patch_size",
            "tokenizer_dropout", "transformer_dropout", "emb_dropout", "drop_path_rate",
            "use_hilbert_encoding", "use_spatial_encoding", "use_checkpoint", "ffn_type",
            "dim_head",  # 派生参数
            # 兼容旧版
            "dropout",  # 实际存储 transformer_dropout
        ],
    },
    # ========== 第二层：变参数（动态计算） ==========
    "layer2_variable": {
        "name": "变参数",
        "description": "由模型架构内部根据其他参数动态计算，不保存到 checkpoint",
        "params": [
            "max_level",  # 由 image_size 和 min_patch_size 计算
        ],
    },
    # ========== 第三层：超参数（固定） ==========
    "layer3_hyperparams": {
        "name": "超参数",
        "description": "应当固定不变、直接保存在模型架构中的参数",
        "params": [
            # 覆盖率参数
            "token_coverage_min", "token_coverage_max",
            "coverage_min", "coverage_max_hard",
            # K 值限制
            "K_min_abs", "K_max_hard", "coverage_base",
            # 温度参数
            "splitter_temp_start", "splitter_temp_end",
            # 可学习配额
            "quota_learnable", "quota_entropy_weight",
            # 形状-尺度编码
            "use_area_encoding", "use_affine_modulation", "fourier_levels",
            # Splitter 架构
            "splitter_hidden_dim", "splitter_feature_dim", "splitter_pool_size",
            # 语义分裂器
            "use_semantic_splitter", "semantic_splitter_config", "semantic_loss_weight",
            # 深度缩放
            "depth_scale_range",
        ],
    },
    # ========== 训练损失超参数（不保存到 ModelGene） ==========
    "training_hyperparams": {
        "name": "训练损失超参数",
        "description": "仅在训练时使用，影响损失函数但不改变模型结构，不应保存到 checkpoint",
        "params": [
            # Elastic Budget 配置 - 训练损失参数
            "elastic_coverage_min", "elastic_coverage_max",
            "elastic_lambda_over", "elastic_lambda_under",
            # 其他训练时动态设置的值
        ],
    },
    # ========== 训练元信息（不参与一致性检查） ==========
    "metadata": {
        "name": "训练元信息",
        "description": "训练过程元数据，不参与模型重建",
        "params": [
            "dataset_name", "training_epochs", "checkpoint_epoch",
        ],
    },
    # ========== 不序列化的内部状态 ==========
    "internal": {
        "name": "内部状态",
        "description": "不序列化的内部状态",
        "params": [
            "_parameter_tier", "_model_config",
        ],
    },
}


@dataclass
class ParameterCheckResult:
    """参数检查结果"""
    total_params: int = 0
    layer1_count: int = 0
    layer2_count: int = 0
    layer3_count: int = 0
    training_hyperparams_count: int = 0
    missing_in_checkpoint: List[str] = field(default_factory=list)
    extra_in_checkpoint: List[str] = field(default_factory=list)
    layer1_values: Dict[str, Any] = field(default_factory=dict)
    layer2_values: Dict[str, Any] = field(default_factory=dict)
    layer3_values: Dict[str, Any] = field(default_factory=dict)
    training_hyperparams_values: Dict[str, Any] = field(default_factory=dict)


def load_checkpoint(path: str, device: str = "cpu") -> Dict[str, Any]:
    """加载 checkpoint 文件"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint


def extract_model_gene(checkpoint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从 checkpoint 提取 ModelGene"""
    if "model_gene" in checkpoint:
        return checkpoint["model_gene"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        # 兼容旧格式
        return checkpoint["model"]
    return None


def classify_param(param_name: str) -> str:
    """将参数分类到对应层级"""
    for tier_key, tier_info in PARAMETER_TIERS.items():
        if param_name in tier_info["params"]:
            return tier_key
    # 未知参数
    return "unknown"


def check_parameters(checkpoint: Dict[str, Any]) -> ParameterCheckResult:
    """检查 checkpoint 中的参数"""
    result = ParameterCheckResult()

    gene = extract_model_gene(checkpoint)
    if gene is None:
        print("[WARN] No model_gene found in checkpoint")
        return result

    # 统计各层级参数
    for key, value in gene.items():
        tier = classify_param(key)
        result.total_params += 1

        if tier == "layer1_params":
            result.layer1_count += 1
            result.layer1_values[key] = value
        elif tier == "layer2_variable":
            result.layer2_count += 1
            result.layer2_values[key] = value
        elif tier == "layer3_hyperparams":
            result.layer3_count += 1
            result.layer3_values[key] = value
        elif tier == "training_hyperparams":
            result.training_hyperparams_count += 1
            result.training_hyperparams_values[key] = value

    return result


def format_value(value: Any, max_len: int = 50) -> str:
    """格式化参数值用于显示"""
    if value is None:
        return "None"
    if isinstance(value, (list, tuple)):
        if len(value) > 2:
            return f"{value[:2]}... ({len(value)} items)"
        return str(value)
    if isinstance(value, dict):
        keys = list(value.keys())
        if len(keys) > 3:
            return f"{{{', '.join(map(str, keys[:3]))}...}} ({len(keys)} keys)"
        return str(value)
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + "..."
    return str(value)


def print_checkpoint_info(checkpoint_path: str, device: str = "cpu"):
    """打印 checkpoint 详细信息"""
    print(f"\n{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*70}")

    checkpoint = load_checkpoint(checkpoint_path, device)
    gene = extract_model_gene(checkpoint)

    if gene is None:
        print("[ERROR] No model_gene found in checkpoint!")
        print("\nAvailable keys:", list(checkpoint.keys()))
        return

    # 打印各层级参数
    print(f"\n[INFO] 参数统计:")
    print(f"   总参数数: {len(gene)}")

    result = check_parameters(checkpoint)
    print(f"   第一层（架构参数）: {result.layer1_count}")
    print(f"   第二层（变参数）: {result.layer2_count}")
    print(f"   第三层（超参数）: {result.layer3_count}")
    print(f"   训练损失超参数: {result.training_hyperparams_count}")

    # 打印第一层参数
    print(f"\n[INFO] 第一层 - 架构参数（固定）:")
    print(f"   {'参数名':<25} {'值':<20}")
    print(f"   {'-'*45}")
    for key, value in sorted(result.layer1_values.items()):
        print(f"   {key:<25} {format_value(value):<20}")

    # 打印第二层参数
    print(f"\n[INFO] 第二层 - 变参数（动态计算，不保存）:")
    if result.layer2_values:
        for key, value in sorted(result.layer2_values.items()):
            print(f"   {key}: {value}")
    else:
        print("   [无] max_level 由模型架构内部计算")

    # 打印第三层参数
    print(f"\n[INFO] 第三层 - 超参数（保存到 checkpoint）:")
    print(f"   {'参数名':<30} {'值':<20}")
    print(f"   {'-'*50}")
    for key, value in sorted(result.layer3_values.items()):
        print(f"   {key:<30} {format_value(value):<20}")

    # 打印训练损失超参数（这些不应该保存到 ModelGene）
    if result.training_hyperparams_values:
        print(f"\n[WARN] 训练损失超参数（不应保存到 checkpoint）:")
        for key, value in sorted(result.training_hyperparams_values.items()):
            print(f"   {key}: {format_value(value)}")

    # 打印元信息
    print(f"\n[INFO] 训练元信息:")
    metadata_keys = [k for k in gene.keys() if classify_param(k) == "metadata"]
    for key in sorted(metadata_keys):
        print(f"   {key}: {gene[key]}")

    # 打印其他未知参数
    unknown_keys = [k for k in gene.keys() if classify_param(k) == "unknown"]
    if unknown_keys:
        print(f"\n[WARN] 未知参数（未分类）:")
        for key in sorted(unknown_keys):
            print(f"   {key}: {format_value(gene[key])}")

    print(f"\n{'='*70}")


def compare_checkpoints(
    checkpoint1: str,
    checkpoint2: str,
    device: str = "cpu"
):
    """对比两个 checkpoint 的参数"""
    print(f"\n{'='*70}")
    print(f"对比两个 Checkpoint")
    print(f"{'='*70}")

    ckpt1 = load_checkpoint(checkpoint1, device)
    ckpt2 = load_checkpoint(checkpoint2, device)

    gene1 = extract_model_gene(ckpt1)
    gene2 = extract_model_gene(ckpt2)

    if gene1 is None or gene2 is None:
        print("[ERROR] One or both checkpoints don't have model_gene!")
        return

    # 找出差异
    all_keys = set(gene1.keys()) | set(gene2.keys())
    differences = []

    for key in sorted(all_keys):
        val1 = gene1.get(key, "<missing>")
        val2 = gene2.get(key, "<missing>")

        if val1 != val2:
            differences.append((key, val1, val2))

    if not differences:
        print("\n[OK] 两个 checkpoint 的参数完全一致!")
    else:
        print(f"\n[WARN] 发现 {len(differences)} 处参数差异:")
        print(f"\n   {'参数名':<25} {'Checkpoint 1':<25} {'Checkpoint 2':<25}")
        print(f"   {'-'*75}")

        for key, val1, val2 in differences:
            # 分类参数
            tier = classify_param(key)
            tier_prefix = {
                "layer1_params": "[L1]",
                "layer2_variable": "[L2]",
                "layer3_hyperparams": "[L3]",
                "metadata": "[META]",
                "unknown": "[??]",
            }.get(tier, "[??]")

            print(f"   {tier_prefix} {key:<20} {format_value(val1, 20):<25} {format_value(val2, 20):<25}")

    print(f"\n{'='*70}")


def verify_training_save_load(
    checkpoint_path: str,
    device: str = "cpu"
):
    """验证训练-保存-加载的一致性"""
    print(f"\n{'='*70}")
    print(f"验证训练-保存-加载一致性")
    print(f"{'='*70}")

    # 加载 checkpoint
    checkpoint = load_checkpoint(checkpoint_path, device)
    gene_dict = extract_model_gene(checkpoint)

    if gene_dict is None:
        print("[ERROR] No model_gene found in checkpoint!")
        return

    # 尝试重建模型
    try:
        from training.core.model_gene import ModelGene

        gene = ModelGene.from_dict(gene_dict)
        print(f"\n[OK] ModelGene 解析成功")

        # 尝试构建模型
        model = gene.build_model()
        print(f"[OK] 模型构建成功")

        # 验证关键属性
        print(f"\n[INFO] 模型属性验证:")
        print(f"   dim: {model.dim}")
        print(f"   num_layers: {model.num_layers}")
        print(f"   heads: {model.heads}")
        print(f"   mlp_dim: {model.mlp_dim}")
        print(f"   num_classes: {model.num_classes}")
        print(f"   image_size: {model.image_size}")

        # 检查变参数 max_level
        if hasattr(model, 'max_level'):
            print(f"   max_level (变参数): {model.max_level}")

        # 检查 tokenizer
        if hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
            if hasattr(tokenizer, 'max_level'):
                print(f"   tokenizer.max_level: {tokenizer.max_level}")
            if hasattr(tokenizer, 'min_patch_size'):
                print(f"   tokenizer.min_patch_size: {tokenizer.min_patch_size}")

        # 检查 splitter
        if hasattr(model, 'splitter'):
            splitter = model.splitter
            if hasattr(splitter, 'config') and splitter.config:
                print(f"\n[INFO] Splitter 配置:")
                for key in ['coverage_min', 'coverage_max_hard', 'K_min_abs', 'K_max_hard']:
                    if hasattr(splitter.config, key):
                        print(f"   {key}: {getattr(splitter.config, key)}")

        print(f"\n[OK] 训练-保存-加载一致性验证通过!")

    except Exception as e:
        print(f"\n[ERROR] 验证失败: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="参数一致性检查工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查单个 checkpoint
  uv run python src/training/checkpoint_param_checker.py --checkpoint checkpoints/best.pth

  # 对比两个 checkpoint
  uv run python src/training/checkpoint_param_checker.py --checkpoint1 a.pth --checkpoint2 b.pth

  # 验证训练-保存-加载一致性
  uv run python src/training/checkpoint_param_checker.py --checkpoint checkpoints/best.pth --verify
        """
    )

    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        help="要检查的 checkpoint 路径"
    )
    parser.add_argument(
        "--checkpoint1",
        type=str,
        help="第一个 checkpoint 路径（对比用）"
    )
    parser.add_argument(
        "--checkpoint2",
        type=str,
        help="第二个 checkpoint 路径（对比用）"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="验证训练-保存-加载一致性"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="设备 (default: cpu)"
    )

    args = parser.parse_args()

    # 场景1: 对比两个 checkpoint
    if args.checkpoint1 and args.checkpoint2:
        compare_checkpoints(args.checkpoint1, args.checkpoint2, args.device)
        return

    # 场景2: 单个 checkpoint
    if args.checkpoint:
        print_checkpoint_info(args.checkpoint, args.device)

        if args.verify:
            verify_training_save_load(args.checkpoint, args.device)
        return

    # 无参数
    parser.print_help()


if __name__ == "__main__":
    main()
