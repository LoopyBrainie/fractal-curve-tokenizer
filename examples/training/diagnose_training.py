#!/usr/bin/env python3
"""
训练诊断脚本 - 排查 "一个类别 100%，其他 0%" 问题

问题描述:
    模型在训练/评估时表现出极端不平衡:
    - 一个类别准确率达到 100%
    - 其他所有类别准确率为 0%
    
这通常意味着模型坍缩 (Model Collapse)，总是预测同一个类别。

诊断步骤:
    1. 检查模型输出分布
    2. 检查分类头权重和偏置
    3. 检查梯度流
    4. 检查数据加载
    5. 检查损失函数
"""

import os
import sys
from pathlib import Path

# 项目路径设置
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from tqdm import tqdm


@torch.no_grad()
def diagnose_model_collapse(
    model: nn.Module,
    dataloader,
    device: torch.device,
    num_classes: int,
    max_batches: int = 10,
) -> dict:
    """
    诊断模型坍缩问题 (I78: 添加 no_grad 支持 torch.compile)。

    检查:
    1. 预测分布是否集中在单一类别
    2. Logits 分布是否异常
    3. 分类头偏置是否偏移
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_logits = []
    
    print("\n" + "=" * 70)
    print("步骤 1: 收集模型预测")
    print("=" * 70)
    
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(dataloader, desc="Collecting predictions")):
            if i >= max_batches:
                break
            
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # 获取模型输出
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            predictions = outputs.argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(outputs.detach().cpu())  # I78: 使用 detach() 支持 torch.compile
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_logits = torch.cat(all_logits, dim=0)
    
    # 分析预测分布
    pred_counter = Counter(all_predictions)
    label_counter = Counter(all_labels)
    
    print(f"\n  真实标签分布 (前10个类别): {dict(sorted(label_counter.items())[:10])}")
    print(f"  预测分布 (前10个类别):     {dict(sorted(pred_counter.items())[:10])}")
    
    # 检测坍缩
    total_samples = len(all_predictions)
    max_pred_class = pred_counter.most_common(1)[0]
    max_pred_ratio = max_pred_class[1] / total_samples
    
    print(f"\n  总样本数: {total_samples}")
    print(f"  最频繁预测的类别: {max_pred_class[0]} ({max_pred_class[1]} 次, {max_pred_ratio*100:.1f}%)")
    
    collapse_detected = max_pred_ratio > 0.8  # 80% 预测同一类别视为坍缩
    
    if collapse_detected:
        print(f"\n  ⚠️  检测到模型坍缩! 模型几乎总是预测类别 {max_pred_class[0]}")
    else:
        print(f"\n  ✓ 预测分布看起来正常")
    
    # 分析 logits
    print("\n" + "=" * 70)
    print("步骤 2: 分析 Logits 分布")
    print("=" * 70)
    
    logits_mean = all_logits.mean(dim=0)
    logits_std = all_logits.std(dim=0)
    logits_max = all_logits.max(dim=0)[0]
    logits_min = all_logits.min(dim=0)[0]
    
    print(f"\n  Logits 形状: {all_logits.shape}")
    print(f"  Per-class 均值: min={logits_mean.min():.4f}, max={logits_mean.max():.4f}")
    print(f"  Per-class 标准差: min={logits_std.min():.4f}, max={logits_std.max():.4f}")
    
    # 找出偏置最大的类别
    top_biased_classes = torch.argsort(logits_mean, descending=True)[:5]
    print(f"\n  平均 logit 最高的 5 个类别: {top_biased_classes.tolist()}")
    print(f"  对应均值: {logits_mean[top_biased_classes].tolist()}")
    
    # 检查 softmax 后的概率
    probs = F.softmax(all_logits, dim=1)
    avg_probs = probs.mean(dim=0)
    
    print(f"\n  平均概率最高的 5 个类别:")
    top_prob_classes = torch.argsort(avg_probs, descending=True)[:5]
    for c in top_prob_classes:
        print(f"    类别 {c.item()}: {avg_probs[c].item()*100:.2f}%")

    # 检查是否有某个类别概率远超其他
    prob_max = avg_probs.max().item()
    prob_mean = avg_probs.mean().item()

    if prob_max > 0.5:  # 50% 以上概率集中在一个类别
        print(f"\n  ⚠️  概率分布异常! 类别 {top_prob_classes[0].item()} 平均概率达到 {prob_max*100:.1f}%")
    
    return {
        'collapse_detected': collapse_detected,
        'dominant_class': max_pred_class[0],
        'dominant_ratio': max_pred_ratio,
        'logits_mean': logits_mean.numpy(),
        'logits_std': logits_std.numpy(),
        'prediction_distribution': dict(pred_counter),
        'label_distribution': dict(label_counter),
    }


def diagnose_classifier_head(model: nn.Module, num_classes: int) -> dict:
    """
    诊断分类头的权重和偏置。
    
    检查:
    1. 权重是否有异常值
    2. 偏置是否有大的偏移
    3. 权重范数是否不均匀
    """
    print("\n" + "=" * 70)
    print("步骤 3: 检查分类头 (MLP Head)")
    print("=" * 70)
    
    # 找到分类头的最后一个 Linear 层
    last_linear = None
    last_linear_name = None
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features == num_classes:
            last_linear = module
            last_linear_name = name
    
    if last_linear is None:
        print("  ⚠️  未找到分类头的 Linear 层")
        return {'found': False}
    
    print(f"\n  分类层: {last_linear_name}")
    print(f"  权重形状: {last_linear.weight.shape}")
    
    weight = last_linear.weight.detach()
    bias = last_linear.bias.detach() if last_linear.bias is not None else None
    
    # 分析权重
    weight_norms = weight.norm(dim=1)
    print(f"\n  权重 L2 范数: min={weight_norms.min():.4f}, max={weight_norms.max():.4f}, mean={weight_norms.mean():.4f}")
    
    # 检查范数是否不均匀
    norm_std = weight_norms.std().item()
    norm_mean = weight_norms.mean().item()
    if norm_std / norm_mean > 0.5:  # 变异系数 > 50%
        print(f"  ⚠️  权重范数分布不均匀 (CV = {norm_std/norm_mean*100:.1f}%)")
        top_norm_classes = torch.argsort(weight_norms, descending=True)[:5]
        print(f"      范数最大的类别: {top_norm_classes.tolist()}")
        print(f"      对应范数: {weight_norms[top_norm_classes].tolist()}")
    
    # 分析偏置
    if bias is not None:
        print(f"\n  偏置: min={bias.min():.4f}, max={bias.max():.4f}, mean={bias.mean():.4f}, std={bias.std():.4f}")
        
        # 检查偏置是否有大的偏移
        bias_range = bias.max() - bias.min()
        if bias_range > 1.0:  # 偏置范围超过 1.0 可能导致问题
            print(f"  ⚠️  偏置范围过大 ({bias_range:.2f})，可能导致预测偏向某些类别")
            top_bias_classes = torch.argsort(bias, descending=True)[:5]
            print(f"      偏置最大的类别: {top_bias_classes.tolist()}")
            print(f"      对应偏置: {bias[top_bias_classes].tolist()}")
        else:
            print(f"  ✓ 偏置范围正常 ({bias_range:.4f})")
    else:
        print(f"\n  ✓ 分类层无偏置项")
    
    return {
        'found': True,
        'layer_name': last_linear_name,
        'weight_norms': weight_norms.numpy(),
        'bias': bias.numpy() if bias is not None else None,
    }


def diagnose_gradient_flow(model: nn.Module, sample_input: torch.Tensor, sample_label: torch.Tensor) -> dict:
    """
    诊断梯度流是否正常。
    
    检查:
    1. 各层梯度是否为零
    2. 梯度是否有 NaN/Inf
    3. 梯度是否过小或过大
    """
    print("\n" + "=" * 70)
    print("步骤 4: 检查梯度流")
    print("=" * 70)
    
    model.train()
    model.zero_grad()
    
    # 前向传播
    outputs = model(sample_input)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    
    # 计算损失
    loss = F.cross_entropy(outputs, sample_label)
    print(f"\n  损失值: {loss.item():.4f}")
    
    # 反向传播
    loss.backward()

    # 收集梯度统计 (I78: 使用 no_grad 支持 torch.compile)
    grad_stats = []
    zero_grad_layers = []
    nan_grad_layers = []
    small_grad_layers = []
    large_grad_layers = []

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                grad_norm = grad.norm().item()
                grad_max = grad.abs().max().item()
                grad_min = grad.abs().min().item()
                grad_mean = grad.abs().mean().item()

                has_nan = torch.isnan(grad).any().item()
                has_inf = torch.isinf(grad).any().item()
                is_zero = grad_norm == 0

                if has_nan or has_inf:
                    nan_grad_layers.append(name)
                elif is_zero:
                    zero_grad_layers.append(name)
                elif grad_norm < 1e-8:
                    small_grad_layers.append((name, grad_norm))
                elif grad_norm > 1000:
                    large_grad_layers.append((name, grad_norm))

                grad_stats.append({
                    'name': name,
                    'norm': grad_norm,
                    'max': grad_max,
                    'min': grad_min,
                    'mean': grad_mean,
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                })
    
    # 报告问题
    total_params = len(grad_stats)
    print(f"\n  参数总数: {total_params}")
    
    if zero_grad_layers:
        print(f"\n  ⚠️  发现 {len(zero_grad_layers)} 个层梯度为零:")
        for name in zero_grad_layers[:5]:
            print(f"      - {name}")
        if len(zero_grad_layers) > 5:
            print(f"      ... 还有 {len(zero_grad_layers) - 5} 个")
    else:
        print(f"\n  ✓ 没有层梯度为零")
    
    if nan_grad_layers:
        print(f"\n  ⚠️  发现 {len(nan_grad_layers)} 个层梯度包含 NaN/Inf:")
        for name in nan_grad_layers[:5]:
            print(f"      - {name}")
    else:
        print(f"  ✓ 没有层梯度包含 NaN/Inf")
    
    if small_grad_layers:
        print(f"\n  ⚠️  发现 {len(small_grad_layers)} 个层梯度过小 (<1e-8):")
        for name, norm in small_grad_layers[:5]:
            print(f"      - {name}: norm={norm:.2e}")
    
    if large_grad_layers:
        print(f"\n  ⚠️  发现 {len(large_grad_layers)} 个层梯度过大 (>1000):")
        for name, norm in large_grad_layers[:5]:
            print(f"      - {name}: norm={norm:.2e}")
    
    # 检查关键层
    print("\n  关键层梯度统计:")
    key_patterns = ['mlp_head', 'classifier', 'to_latent', 'cls_token']
    for stat in grad_stats:
        for pattern in key_patterns:
            if pattern in stat['name'].lower():
                print(f"    {stat['name']}: norm={stat['norm']:.4e}, max={stat['max']:.4e}")
    
    model.zero_grad()
    model.eval()
    
    return {
        'zero_grad_count': len(zero_grad_layers),
        'nan_grad_count': len(nan_grad_layers),
        'small_grad_count': len(small_grad_layers),
        'large_grad_count': len(large_grad_layers),
        'grad_stats': grad_stats,
    }


@torch.no_grad()
def diagnose_data_loading(dataloader, num_classes: int, max_batches: int = 10) -> dict:
    """
    诊断数据加载是否正常 (I78: 添加 no_grad 支持 torch.compile)。

    检查:
    1. 数据是否正确归一化
    2. 标签分布是否均匀
    3. 数据是否有异常值
    """
    print("\n" + "=" * 70)
    print("步骤 5: 检查数据加载")
    print("=" * 70)
    
    all_labels = []
    img_stats = []
    
    for i, (imgs, labels) in enumerate(tqdm(dataloader, desc="Checking data")):
        if i >= max_batches:
            break
        
        all_labels.extend(labels.numpy())
        
        # 收集图像统计
        img_stats.append({
            'mean': imgs.mean().item(),
            'std': imgs.std().item(),
            'min': imgs.min().item(),
            'max': imgs.max().item(),
        })
    
    # 分析标签分布
    label_counter = Counter(all_labels)
    print(f"\n  收集的样本数: {len(all_labels)}")
    print(f"  类别数: {len(label_counter)}")
    
    # 检查类别分布
    label_counts = list(label_counter.values())
    min_count = min(label_counts)
    max_count = max(label_counts)
    
    print(f"  每类样本数: min={min_count}, max={max_count}")
    
    if max_count / (min_count + 1) > 10:
        print(f"\n  ⚠️  类别分布极度不平衡!")
        most_common = label_counter.most_common(5)
        least_common = label_counter.most_common()[-5:]
        print(f"      最多的类别: {most_common}")
        print(f"      最少的类别: {least_common}")
    else:
        print(f"  ✓ 类别分布相对均衡")
    
    # 检查图像数据
    avg_mean = np.mean([s['mean'] for s in img_stats])
    avg_std = np.mean([s['std'] for s in img_stats])
    global_min = min([s['min'] for s in img_stats])
    global_max = max([s['max'] for s in img_stats])
    
    print(f"\n  图像数据统计:")
    print(f"    平均值: {avg_mean:.4f}")
    print(f"    标准差: {avg_std:.4f}")
    print(f"    范围: [{global_min:.4f}, {global_max:.4f}]")
    
    # 检查是否正确归一化
    if abs(avg_mean) > 1.0:
        print(f"  ⚠️  数据均值过大，可能未正确归一化")
    if avg_std < 0.1 or avg_std > 2.0:
        print(f"  ⚠️  数据标准差异常")
    
    return {
        'label_distribution': dict(label_counter),
        'num_classes_in_data': len(label_counter),
        'avg_mean': avg_mean,
        'avg_std': avg_std,
        'global_min': global_min,
        'global_max': global_max,
    }


def diagnose_tokenizer(model: nn.Module, sample_input: torch.Tensor) -> dict:
    """
    诊断 Tokenizer 是否正常工作。
    
    检查:
    1. Token 数量是否合理
    2. Token 分布是否正常
    """
    print("\n" + "=" * 70)
    print("步骤 6: 检查 Tokenizer")
    print("=" * 70)
    
    if not hasattr(model, 'tokenizer'):
        print("  ⚠️  模型没有 tokenizer 属性")
        return {'has_tokenizer': False}
    
    tokenizer = model.tokenizer
    
    model.eval()
    with torch.no_grad():
        token_output = tokenizer.tokenize(sample_input)
    
    # 获取 token 统计
    padded_tokens, lengths = token_output.get_padded_tokens()
    
    print(f"\n  Token 形状: {padded_tokens.shape}")
    print(f"  每个样本的 token 数量: {lengths.tolist()}")
    print(f"  平均 token 数: {lengths.float().mean().item():.1f}")
    
    # 检查 token 值
    token_mean = padded_tokens.mean().item()
    token_std = padded_tokens.std().item()
    token_max = padded_tokens.max().item()
    token_min = padded_tokens.min().item()
    
    print(f"\n  Token 统计:")
    print(f"    均值: {token_mean:.4f}")
    print(f"    标准差: {token_std:.4f}")
    print(f"    范围: [{token_min:.4f}, {token_max:.4f}]")
    
    if token_std < 0.01:
        print(f"  ⚠️  Token 标准差过小，可能所有 token 都相似!")
    
    if torch.isnan(padded_tokens).any():
        print(f"  ⚠️  Token 包含 NaN!")
    
    if torch.isinf(padded_tokens).any():
        print(f"  ⚠️  Token 包含 Inf!")
    
    return {
        'has_tokenizer': True,
        'token_shape': list(padded_tokens.shape),
        'avg_tokens': lengths.float().mean().item(),
        'token_mean': token_mean,
        'token_std': token_std,
    }


def suggest_fixes(diagnostics: dict) -> None:
    """
    根据诊断结果给出修复建议。
    """
    print("\n" + "=" * 70)
    print("修复建议")
    print("=" * 70)
    
    suggestions = []
    
    if diagnostics.get('collapse_detected'):
        suggestions.append(
            "1. 模型坍缩检测到 - 请尝试:\n"
            "   a. 减小学习率 (--lr 1e-4 或更低)\n"
            "   b. 使用 Focal Loss (--use-focal-loss)\n"
            "   c. 检查分类头初始化\n"
            "   d. 增加 warmup epochs (--warmup-epochs 10)"
        )
    
    if diagnostics.get('zero_grad_count', 0) > 0:
        suggestions.append(
            "2. 发现梯度为零的层:\n"
            "   a. 检查模型是否正确连接\n"
            "   b. 检查是否有不需要梯度的参数\n"
            "   c. 检查是否有梯度被意外阻断"
        )
    
    if diagnostics.get('nan_grad_count', 0) > 0:
        suggestions.append(
            "3. 发现 NaN 梯度:\n"
            "   a. 减小学习率\n"
            "   b. 增加梯度裁剪 (--gradient-clip 0.5)\n"
            "   c. 检查数据是否包含 NaN"
        )
    
    classifier_result = diagnostics.get('classifier_head', {})
    if classifier_result.get('bias') is not None:
        bias = classifier_result['bias']
        if bias.max() - bias.min() > 1.0:
            suggestions.append(
                "4. 分类头偏置不均匀:\n"
                "   a. 重新初始化分类头偏置为零\n"
                "   b. 检查是否加载了不匹配的预训练权重"
            )
    
    if diagnostics.get('token_std', 1.0) < 0.01:
        suggestions.append(
            "5. Token 标准差过小:\n"
            "   a. 检查 Tokenizer 是否正确工作\n"
            "   b. 检查共享卷积层是否有问题\n"
            "   c. 检查深度编码是否正确"
        )
    
    if not suggestions:
        print("\n  ✓ 未发现明显问题。建议:\n"
              "   - 检查训练曲线是否正常收敛\n"
              "   - 尝试更长的训练时间\n"
              "   - 检查数据增强是否过强")
    else:
        for suggestion in suggestions:
            print(f"\n{suggestion}")


def main():
    """运行完整诊断。"""
    import argparse
    
    parser = argparse.ArgumentParser(description="训练诊断工具")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "tiny-imagenet", "cub200"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", type=str, default=None, help="要诊断的模型检查点路径")
    parser.add_argument("--max-batches", type=int, default=10, help="诊断时使用的最大批次数")
    
    args = parser.parse_args()
    
    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\n使用设备: {device}")
    
    # 数据集配置
    DATASETS = {
        "cifar10": {"num_classes": 10, "image_size": 32, "channels": 3, 
                    "mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
        "cifar100": {"num_classes": 100, "image_size": 32, "channels": 3,
                     "mean": (0.5071, 0.4865, 0.4409), "std": (0.2673, 0.2564, 0.2762)},
        "tiny-imagenet": {"num_classes": 200, "image_size": 64, "channels": 3,
                          "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
        "cub200": {"num_classes": 200, "image_size": 224, "channels": 3,
                   "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    }
    
    spec = DATASETS[args.dataset]
    
    # 创建数据加载器
    from torchvision import datasets, transforms
    
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(spec['mean'], spec['std']),
    ])
    
    data_root = PROJECT_ROOT / "data"
    
    if args.dataset == "cifar10":
        test_ds = datasets.CIFAR10(data_root, train=False, download=True, transform=test_tf)
    elif args.dataset == "cifar100":
        test_ds = datasets.CIFAR100(data_root, train=False, download=True, transform=test_tf)
    elif args.dataset == "tiny-imagenet":
        test_dir = data_root / "tiny-imagenet-200" / "val"
        if not test_dir.exists():
            print(f"⚠️  Tiny ImageNet 数据集不存在: {test_dir}")
            return
        test_ds = datasets.ImageFolder(str(test_dir), transform=test_tf)
    elif args.dataset == "cub200":
        test_dir = data_root / "CUB_200_2011" / "test"
        if not test_dir.exists():
            print(f"⚠️  CUB-200-2011 数据集不存在: {test_dir}")
            return
        test_ds = datasets.ImageFolder(str(test_dir), transform=test_tf)
    
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    
    # 创建或加载模型
    from vit_pytorch import FractalCurveViT
    from vit_pytorch.tokenizer_streaming import StreamingFractalTokenizerV3
    
    tokenizer = StreamingFractalTokenizerV3(
        image_size=max(spec['image_size'], 32),
        channels=spec['channels'],
        d_model=192,
        base_patch_size=4,
        max_depth=4,
    )
    
    model = FractalCurveViT(
        image_size=max(spec['image_size'], 32),
        num_classes=spec['num_classes'],
        dim=192,
        depth=8,
        heads=8,
        mlp_dim=192 * 4,
        channels=spec['channels'],
        tokenizer=tokenizer,
    ).to(device)
    
    if args.checkpoint:
        print(f"\n加载检查点: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
    
    # 收集诊断结果
    diagnostics = {}
    
    # 1. 模型坍缩诊断
    collapse_result = diagnose_model_collapse(
        model, test_loader, device, spec['num_classes'], args.max_batches
    )
    diagnostics.update(collapse_result)
    
    # 2. 分类头诊断
    classifier_result = diagnose_classifier_head(model, spec['num_classes'])
    diagnostics['classifier_head'] = classifier_result
    
    # 3. 数据加载诊断
    data_result = diagnose_data_loading(test_loader, spec['num_classes'], args.max_batches)
    diagnostics.update(data_result)
    
    # 4. Tokenizer 诊断
    sample_batch = next(iter(test_loader))
    sample_input = sample_batch[0][:4].to(device)
    sample_label = sample_batch[1][:4].to(device)
    
    tokenizer_result = diagnose_tokenizer(model, sample_input)
    diagnostics.update(tokenizer_result)
    
    # 5. 梯度流诊断
    grad_result = diagnose_gradient_flow(model, sample_input, sample_label)
    diagnostics.update(grad_result)
    
    # 6. 给出建议
    suggest_fixes(diagnostics)
    
    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
