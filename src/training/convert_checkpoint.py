#!/usr/bin/env python3
"""
Checkpoint 权重映射脚本 (v2)

将旧版架构的 checkpoint 权重映射到新版 FractalCurveViT 架构。

主要差异处理:
1. max_depth 差异: 旧版使用 max_depth=9, 新版使用 max_depth=5
2. 键名相同但形状不同的层需要截断或适配

使用方式:
    uv run python src/training/convert_checkpoint.py \
        --checkpoint "experiments/.../best.pth" \
        --output "experiments/.../best_converted.pth"
"""

import sys
import torch
from pathlib import Path


def load_checkpoint(checkpoint_path: str, device: str = 'cpu') -> dict:
    """加载 checkpoint"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return ckpt


def remap_state_dict(
    old_state: dict,
    old_max_depth: int = 9,
    new_max_depth: int = 5,
    verbose: bool = True
) -> dict:
    """
    重映射 state_dict，处理键名和形状差异

    Args:
        old_state: 原始 state_dict
        old_max_depth: checkpoint 的 max_depth
        new_max_depth: 目标模型的 max_depth
        verbose: 是否打印详细信息

    Returns:
        重映射后的 state_dict
    """
    new_state = {}

    # 计算深度缩放因子
    depth_ratio = new_max_depth / old_max_depth

    for old_key, value in old_state.items():
        # 跳过非张量
        if not isinstance(value, torch.Tensor):
            continue

        new_key = old_key

        # 1. 处理 max_depth 相关的权重缩放
        # 格式: XXX.[level_]embedding.weight: [old_depth, dim] -> [new_depth, dim]
        if 'level_embedding.weight' in old_key:
            if old_max_depth != new_max_depth and value.dim() == 2:
                old_depth = value.shape[0]
                new_depth = int(old_depth * depth_ratio)
                # 使用插值来调整
                if new_depth > 0 and old_depth > 0:
                    # 对每列进行插值
                    new_value = torch.nn.functional.interpolate(
                        value.unsqueeze(0).unsqueeze(0),
                        size=(new_depth, value.shape[1]),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)
                    new_state[new_key] = new_value
                    if verbose:
                        print(f"  [INTERP] {old_key}: {value.shape} -> {new_value.shape}")
                    continue

        # 2. 处理 ffn_gamma/ffn_beta: [old_depth+1, dim] -> [new_depth+1, dim]
        if any(x in old_key for x in ['ffn_gamma.weight', 'ffn_beta.weight']):
            if old_max_depth != new_max_depth and value.dim() == 2:
                old_depth = value.shape[0]
                new_depth = int(old_depth * depth_ratio)
                if new_depth > 0 and old_depth > 0:
                    new_value = torch.nn.functional.interpolate(
                        value.unsqueeze(0).unsqueeze(0),
                        size=(new_depth, value.shape[1]),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)
                    new_state[new_key] = new_value
                    if verbose:
                        print(f"  [INTERP] {old_key}: {value.shape} -> {new_value.shape}")
                    continue

        # 3. 处理 level_scale_raw: [old_depth, heads] -> [new_depth, heads]
        if '_level_scale_raw.weight' in old_key:
            if old_max_depth != new_max_depth and value.dim() == 2:
                old_depth = value.shape[0]
                new_depth = int(old_depth * depth_ratio)
                if new_depth > 0 and old_depth > 0:
                    new_value = torch.nn.functional.interpolate(
                        value.unsqueeze(0).unsqueeze(0),
                        size=(new_depth, value.shape[1]),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)
                    new_state[new_key] = new_value
                    if verbose:
                        print(f"  [INTERP] {old_key}: {value.shape} -> {new_value.shape}")
                    continue

        # 4. 处理 lca_embedding: [old_depth, dim_head] -> [new_depth, dim_head]
        if 'lca_embedding.weight' in old_key:
            if old_max_depth != new_max_depth and value.dim() == 2:
                old_depth = value.shape[0]
                new_depth = int(old_depth * depth_ratio)
                if new_depth > 0 and old_depth > 0:
                    new_value = torch.nn.functional.interpolate(
                        value.unsqueeze(0).unsqueeze(0),
                        size=(new_depth, value.shape[1]),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)
                    new_state[new_key] = new_value
                    if verbose:
                        print(f"  [INTERP] {old_key}: {value.shape} -> {new_value.shape}")
                    continue

        # 5. 处理 relative_pos_embedding: [2*old_depth+1, heads] -> [2*new_depth+1, heads]
        if 'relative_pos_embedding.weight' in old_key:
            if old_max_depth != new_max_depth and value.dim() == 2:
                old_len = value.shape[0]
                new_len = 2 * int((old_len - 1) // 2 * depth_ratio) + 1
                if new_len > 0 and old_len > 0:
                    new_value = torch.nn.functional.interpolate(
                        value.unsqueeze(0).unsqueeze(0),
                        size=(new_len, value.shape[1]),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)
                    new_state[new_key] = new_value
                    if verbose:
                        print(f"  [INTERP] {old_key}: {value.shape} -> {new_value.shape}")
                    continue

        # 6. 处理 threshold_offsets: [old_depth] -> [new_depth]
        if 'splitter.threshold_offsets' == old_key:
            if old_max_depth != new_max_depth and value.dim() == 1:
                old_depth = value.shape[0]
                new_depth = int(old_depth * depth_ratio)
                if new_depth > 0 and old_depth > 0:
                    new_value = torch.nn.functional.interpolate(
                        value.unsqueeze(0).unsqueeze(0),
                        size=(new_depth,),
                        mode='linear',
                        align_corners=True
                    ).squeeze(0).squeeze(0)
                    new_state[new_key] = new_value
                    if verbose:
                        print(f"  [INTERP] {old_key}: {value.shape} -> {new_value.shape}")
                    continue

        # 7. 直接复制其他键
        new_state[new_key] = value

    return new_state


def convert_checkpoint(
    checkpoint_path: str,
    output_path: str,
    new_max_depth: int = 5,
    strict: bool = False,
    verbose: bool = True
):
    """
    转换 checkpoint

    Args:
        checkpoint_path: 输入 checkpoint 路径
        output_path: 输出 checkpoint 路径
        new_max_depth: 目标模型的 max_depth
        strict: 严格模式
        verbose: 详细输出
    """
    sys.path.insert(0, 'src')
    from vit_pytorch import FractalCurveViT

    if verbose:
        print(f"加载 checkpoint: {checkpoint_path}")
    ckpt = load_checkpoint(checkpoint_path)
    old_state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))

    # 获取配置
    config = ckpt.get('config', {})
    old_max_depth = config.get('max_depth', 9)

    if verbose:
        print(f"  旧版 max_depth: {old_max_depth}")
        print(f"  新版 max_depth: {new_max_depth}")
        print(f"  深度缩放比: {new_max_depth / old_max_depth:.2f}")

    # 重映射 state_dict
    if verbose:
        print("\n重映射权重...")
    new_state = remap_state_dict(
        old_state,
        old_max_depth=old_max_depth,
        new_max_depth=new_max_depth,
        verbose=verbose
    )

    # 创建新版模型
    if verbose:
        print(f"\n创建新版模型 (max_depth={new_max_depth})...")
    model = FractalCurveViT(
        image_size=config.get('image_size', 224),
        num_classes=config.get('num_classes', 200),
        dim=config.get('dim', 256),
        depth=config.get('depth', 8),
        heads=config.get('heads', 8),
        dim_head=config.get('dim_head', 32),
        mlp_dim=config.get('mlp_dim', 512),
        max_depth=new_max_depth,
    )

    # 尝试加载权重
    if verbose:
        print(f"\n加载权重到模型...")
    try:
        model.load_state_dict(new_state, strict=strict)
        if verbose:
            print(f"  [OK] 成功加载所有权重")
    except Exception as e:
        if verbose:
            print(f"  [WARN] 加载失败: {e}")
            print(f"  使用 non-strict 模式继续...")

    # 统计
    old_count = sum(1 for v in old_state.values() if isinstance(v, torch.Tensor))
    new_count = len(new_state)
    matched_count = sum(1 for k in new_state if k in old_state)

    if verbose:
        print(f"\n[SUMMARY]")
        print(f"  旧版键数: {old_count}")
        print(f"  新版键数: {new_count}")
        print(f"  直接匹配: {matched_count}")
        print(f"  插值适配: {new_count - matched_count}")

    # 保存转换后的 checkpoint
    if verbose:
        print(f"\n保存转换后的 checkpoint: {output_path}")
    new_ckpt = {
        'epoch': ckpt.get('epoch', 0),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': ckpt.get('optimizer_state_dict', None),
        'val_acc': ckpt.get('val_acc', 0.0),
        'val_loss': ckpt.get('val_loss', 0.0),
        'config': {
            **config,
            'max_depth': new_max_depth,  # 更新为新版 max_depth
            '_converted_from': checkpoint_path,
            '_conversion_notes': f'max_depth: {old_max_depth} -> {new_max_depth}',
        },
    }
    torch.save(new_ckpt, output_path)

    if verbose:
        print(f"  [OK] 完成!")

    return model


def main():
    import argparse

    parser = argparse.ArgumentParser(description='转换 checkpoint 的 max_depth')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='输入 checkpoint 路径')
    parser.add_argument('--output', type=str, required=True,
                        help='输出 checkpoint 路径')
    parser.add_argument('--max-depth', type=int, default=5,
                        help='目标模型的 max_depth (默认: 5)')
    parser.add_argument('--strict', action='store_true',
                        help='严格模式')
    parser.add_argument('--quiet', action='store_true',
                        help='安静模式')

    args = parser.parse_args()

    convert_checkpoint(
        args.checkpoint,
        args.output,
        new_max_depth=args.max_depth,
        strict=args.strict,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
