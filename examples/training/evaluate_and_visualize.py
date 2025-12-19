#!/usr/bin/env python3
"""Fractal ViT 模型评估与可视化脚本

特性：
1. 模型评估：准确率、混淆矩阵、分类报告
2. Hilbert 曲线可视化：直观展示空间填充曲线
3. Tokenization 可视化：展示多尺度 patch 分割
4. 注意力热力图：可视化模型注意力分布
5. 特征图可视化：不同层的特征激活

使用示例：
    # 评估最佳模型
    python evaluate_and_visualize.py --checkpoint experiments/xxx/checkpoints/best.pth
    
    # 仅可视化 Hilbert 曲线
    python evaluate_and_visualize.py --visualize-hilbert --max-order 5
    
    # 可视化 tokenization 过程
    python evaluate_and_visualize.py --checkpoint xxx.pth --visualize-tokenization
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vit_pytorch import FractalCurveViT
from vit_pytorch.hilbert import HilbertCurve
from vit_pytorch.streaming_tokenizer import HilbertIndexer, StreamingFractalTokenizerV2

# 设置中文字体 (可选)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 数据类
# ============================================================================

@dataclass
class DatasetSpec:
    """数据集规格 (与 train_fractal_vit.py 对齐)"""
    name: str
    num_classes: int
    image_size: int
    channels: int
    mean: tuple
    std: tuple
    classes: Optional[List[str]] = None


# 数据集配置 (与 train_fractal_vit.py 对齐)
DATASETS: Dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec(
        "CIFAR10", 10, 32, 3, 
        (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616),
        ['airplane', 'automobile', 'bird', 'cat', 'deer', 
         'dog', 'frog', 'horse', 'ship', 'truck']
    ),
    "cifar100": DatasetSpec(
        "CIFAR100", 100, 32, 3, 
        (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    ),
    "mnist": DatasetSpec(
        "MNIST", 10, 28, 1, 
        (0.1307,), (0.3081,)
    ),
    "tiny-imagenet": DatasetSpec(
        "TinyImageNet", 200, 64, 3, 
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    ),
}


# ============================================================================
# 数据集下载功能
# ============================================================================

def download_with_progress(url: str, dest: Path, desc: str = "Downloading") -> bool:
    """带进度条的下载函数"""
    import urllib.request
    
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
    import shutil
    import zipfile
    
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


def prepare_dataset(dataset_name: str, data_root: Path) -> bool:
    """准备数据集，如果需要则自动下载
    
    Args:
        dataset_name: 数据集名称 (cifar10, cifar100, mnist, tiny-imagenet)
        data_root: 数据根目录
        
    Returns:
        bool: 是否准备成功
    """
    data_root.mkdir(parents=True, exist_ok=True)
    
    if dataset_name in ['cifar10', 'cifar100', 'mnist']:
        # CIFAR/MNIST 数据集会在加载时自动下载
        print(f"[*] {dataset_name.upper()} will be downloaded automatically if needed.")
        return True
    elif dataset_name == 'tiny-imagenet':
        return download_tiny_imagenet(data_root)
    else:
        print(f"[ERROR] Unknown dataset: {dataset_name}")
        return False


# ============================================================================
# Hilbert 曲线可视化
# ============================================================================

def generate_hilbert_curve_points(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """生成 Hilbert 曲线的所有点坐标
    
    Args:
        order: 曲线阶数 (n = 2^order)
        
    Returns:
        xs, ys: 坐标数组
    """
    n = 2 ** order
    xs, ys = [], []
    
    for d in range(n * n):
        x, y = HilbertCurve.d_to_xy(n, d)
        xs.append(x)
        ys.append(y)
    
    return np.array(xs), np.array(ys)


def visualize_hilbert_curve(
    max_order: int = 5,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """可视化不同阶数的 Hilbert 曲线
    
    Args:
        max_order: 最大阶数
        save_path: 保存路径
        show: 是否显示
        
    Returns:
        matplotlib Figure
    """
    orders = list(range(1, max_order + 1))
    n_cols = min(3, len(orders))
    n_rows = (len(orders) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_2d(axes)
    
    # 创建渐变颜色
    cmap = plt.cm.viridis
    
    for idx, order in enumerate(orders):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        xs, ys = generate_hilbert_curve_points(order)
        n = 2 ** order
        
        # 创建线段集合用于颜色渐变
        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # 根据遍历顺序着色
        colors = np.linspace(0, 1, len(segments))
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1))
        lc.set_array(colors)
        lc.set_linewidth(2 if order <= 3 else 1)
        
        ax.add_collection(lc)
        
        # 标记起点和终点
        ax.scatter([xs[0]], [ys[0]], color='green', s=100, zorder=5, 
                  label='Start', marker='o')
        ax.scatter([xs[-1]], [ys[-1]], color='red', s=100, zorder=5, 
                  label='End', marker='s')
        
        # 设置网格
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title(f'Order {order}: {n}×{n} = {n*n} cells', fontsize=12)
        
        if order <= 2:
            # 显示网格坐标
            for i in range(n):
                for j in range(n):
                    d = HilbertCurve.xy_to_d(n, i, j)
                    ax.text(i, j, str(d), ha='center', va='center', 
                           fontsize=8, alpha=0.7)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    # 隐藏多余的子图
    for idx in range(len(orders), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)
    
    fig.suptitle('Hilbert Space-Filling Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Hilbert curve visualization saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def visualize_hilbert_locality(
    order: int = 4,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """可视化 Hilbert 曲线的局部性保持特性
    
    展示 2D 距离 vs 1D 距离的关系
    """
    n = 2 ** order
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Hilbert 曲线本身
    ax1 = axes[0]
    xs, ys = generate_hilbert_curve_points(order)
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = np.linspace(0, 1, len(segments))
    lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(0, 1))
    lc.set_array(colors)
    lc.set_linewidth(1.5)
    ax1.add_collection(lc)
    ax1.set_xlim(-0.5, n - 0.5)
    ax1.set_ylim(-0.5, n - 0.5)
    ax1.set_aspect('equal')
    ax1.set_title(f'Hilbert Curve (Order {order})', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. Hilbert 距离热图
    ax2 = axes[1]
    distance_map = np.zeros((n, n))
    center_x, center_y = n // 2, n // 2
    center_d = HilbertCurve.xy_to_d(n, center_x, center_y)
    
    for y in range(n):
        for x in range(n):
            d = HilbertCurve.xy_to_d(n, x, y)
            distance_map[y, x] = abs(d - center_d)
    
    im2 = ax2.imshow(distance_map, cmap='hot', origin='lower')
    ax2.scatter([center_x], [center_y], color='cyan', s=100, marker='*', 
               label='Center', zorder=5)
    ax2.set_title('1D Distance from Center (Hilbert)', fontsize=12)
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    ax2.legend()
    
    # 3. 光栅扫描距离热图 (对比)
    ax3 = axes[2]
    raster_map = np.zeros((n, n))
    center_raster = center_y * n + center_x
    
    for y in range(n):
        for x in range(n):
            raster_d = y * n + x
            raster_map[y, x] = abs(raster_d - center_raster)
    
    im3 = ax3.imshow(raster_map, cmap='hot', origin='lower')
    ax3.scatter([center_x], [center_y], color='cyan', s=100, marker='*', 
               label='Center', zorder=5)
    ax3.set_title('1D Distance from Center (Raster Scan)', fontsize=12)
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    ax3.legend()
    
    fig.suptitle('Hilbert Curve Locality Preservation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Hilbert locality visualization saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def visualize_hilbert_on_image(
    image: torch.Tensor,
    patch_size: int = 4,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """在图像上可视化 Hilbert 曲线遍历顺序
    
    Args:
        image: [C, H, W] 或 [H, W, C] 的图像
        patch_size: patch 大小
        save_path: 保存路径
        show: 是否显示
    """
    # 转换为 numpy
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] in [1, 3]:
            image = image.permute(1, 2, 0).cpu().numpy()
        else:
            image = image.cpu().numpy()
    
    # 归一化到 [0, 1]
    if image.max() > 1:
        image = image / 255.0
    image = np.clip(image, 0, 1)
    
    H, W = image.shape[:2]
    grid_h, grid_w = H // patch_size, W // patch_size
    grid_size = max(grid_h, grid_w)
    
    # 找到最接近的 2 的幂
    n = 1
    while n < grid_size:
        n *= 2
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 1. 原图 + patch 网格
    ax1 = axes[0]
    ax1.imshow(image)
    
    # 绘制 patch 网格
    for i in range(grid_h + 1):
        ax1.axhline(i * patch_size, color='white', linewidth=0.5, alpha=0.5)
    for j in range(grid_w + 1):
        ax1.axvline(j * patch_size, color='white', linewidth=0.5, alpha=0.5)
    
    ax1.set_title(f'Original Image ({H}×{W}) with {grid_h}×{grid_w} patches', fontsize=11)
    ax1.axis('off')
    
    # 2. Hilbert 遍历顺序
    ax2 = axes[1]
    ax2.imshow(image)
    
    # 生成 Hilbert 曲线路径
    cmap = plt.cm.plasma
    path_points = []
    
    for d in range(n * n):
        x, y = HilbertCurve.d_to_xy(n, d)
        if x < grid_w and y < grid_h:
            # 计算 patch 中心点
            cx = x * patch_size + patch_size // 2
            cy = y * patch_size + patch_size // 2
            path_points.append((cx, cy, d))
    
    # 绘制路径
    if len(path_points) > 1:
        xs = [p[0] for p in path_points]
        ys = [p[1] for p in path_points]
        
        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors = np.linspace(0, 1, len(segments))
        
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1),
                           linewidths=2, alpha=0.8)
        lc.set_array(colors)
        ax2.add_collection(lc)
        
        # 标记序号
        for i, (cx, cy, d) in enumerate(path_points):
            if len(path_points) <= 64:  # 数量少时显示序号
                ax2.text(cx, cy, str(i), ha='center', va='center', 
                        fontsize=6, color='white', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5))
    
    ax2.set_title('Hilbert Curve Traversal Order', fontsize=11)
    ax2.axis('off')
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, len(path_points) - 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, shrink=0.8, pad=0.02)
    cbar.set_label('Token Index', fontsize=10)
    
    fig.suptitle('Hilbert Curve on Image Patches', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Hilbert on image saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# Tokenization 可视化
# ============================================================================

def visualize_multi_scale_tokenization(
    image: torch.Tensor,
    tokenizer: StreamingFractalTokenizerV2,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """可视化多尺度 tokenization 过程
    
    Args:
        image: [1, C, H, W] 输入图像
        tokenizer: Tokenizer 实例
        save_path: 保存路径
        show: 是否显示
    """
    device = image.device
    
    # 获取多尺度特征
    with torch.no_grad():
        output = tokenizer.tokenize(image)
    
    # 转换图像
    img_np = image[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    
    patch_sizes = tokenizer.patch_sizes
    n_scales = len(patch_sizes)
    
    fig, axes = plt.subplots(2, n_scales + 1, figsize=(4 * (n_scales + 1), 8))
    
    # 第一行：原图 + 各尺度网格
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Original Image', fontsize=10)
    axes[0, 0].axis('off')
    
    H, W = image.shape[2], image.shape[3]
    colors = plt.cm.Set1(np.linspace(0, 1, n_scales))
    
    for s, ps in enumerate(patch_sizes):
        ax = axes[0, s + 1]
        ax.imshow(img_np)
        
        grid_h, grid_w = H // ps, W // ps
        
        # 绘制网格
        for i in range(grid_h + 1):
            ax.axhline(i * ps, color=colors[s], linewidth=1, alpha=0.7)
        for j in range(grid_w + 1):
            ax.axvline(j * ps, color=colors[s], linewidth=1, alpha=0.7)
        
        ax.set_title(f'Scale {s+1}: {ps}×{ps} patches\n({grid_h}×{grid_w} = {grid_h*grid_w} tokens)', 
                    fontsize=10)
        ax.axis('off')
    
    # 第二行：Hilbert 遍历顺序
    axes[1, 0].text(0.5, 0.5, 'Hilbert\nTraversal\nOrder', 
                   ha='center', va='center', fontsize=14,
                   transform=axes[1, 0].transAxes)
    axes[1, 0].axis('off')
    
    for s, ps in enumerate(patch_sizes):
        ax = axes[1, s + 1]
        ax.imshow(img_np, alpha=0.3)
        
        grid_h, grid_w = H // ps, W // ps
        grid_size = max(grid_h, grid_w)
        n = 1
        while n < grid_size:
            n *= 2
        
        # 生成 Hilbert 路径
        path_points = []
        for d in range(n * n):
            x, y = HilbertCurve.d_to_xy(n, d)
            if x < grid_w and y < grid_h:
                cx = x * ps + ps // 2
                cy = y * ps + ps // 2
                path_points.append((cx, cy))
        
        if len(path_points) > 1:
            xs = [p[0] for p in path_points]
            ys = [p[1] for p in path_points]
            
            points = np.array([xs, ys]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            colors_line = np.linspace(0, 1, len(segments))
            
            lc = LineCollection(segments, cmap='viridis', 
                               norm=plt.Normalize(0, 1), linewidths=2)
            lc.set_array(colors_line)
            ax.add_collection(lc)
        
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)  # 翻转 y 轴
        ax.set_title(f'Hilbert Order (Scale {s+1})', fontsize=10)
        ax.axis('off')
    
    fig.suptitle('Multi-Scale Fractal Tokenization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Multi-scale tokenization saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# 混合 Level 分割可视化 (训练后模型)
# ============================================================================

def visualize_adaptive_scale_selection(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    labels: Optional[torch.Tensor] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """可视化模型的自适应多尺度分割
    
    展示每个区域选择的尺度/level，用不同颜色区分。
    
    Args:
        model: 训练好的 FractalCurveViT 模型
        images: [B, C, H, W] 输入图像
        device: 计算设备
        class_names: 类别名称
        labels: 真实标签
        save_path: 保存路径
        show: 是否显示
        
    Returns:
        matplotlib Figure
    """
    model.eval()
    
    # 获取 tokenizer
    tokenizer = model.tokenizer
    if not hasattr(tokenizer, 'complexity_head'):
        print("[WARN] Tokenizer does not support adaptive scale selection")
        return None
    
    B = images.shape[0]
    n_display = min(B, 8)  # 最多显示 8 张图
    
    H, W = images.shape[2], images.shape[3]
    patch_sizes = tokenizer.patch_sizes
    n_scales = len(patch_sizes)
    
    # 定义尺度颜色
    scale_colors = plt.cm.Set1(np.linspace(0, 0.8, n_scales))
    scale_cmap = LinearSegmentedColormap.from_list(
        'scale_cmap', scale_colors, N=n_scales
    )
    
    with torch.no_grad():
        images_device = images.to(device)
        
        # 获取尺度权重 (新 API: 通过 encoder 和 _compute_scale_weights)
        features_dict = tokenizer.encoder(images_device)
        min_ps = min(features_dict.keys())
        _, target_size = features_dict[min_ps]
        scale_weights = tokenizer._compute_scale_weights(features_dict, target_size)
        scale_indices = scale_weights.argmax(dim=1)  # [B, H', W']
        
        # 获取预测
        outputs, _ = model(images_device, return_aux_info=True)
        preds = outputs.argmax(dim=1)
    
    # 计算布局
    n_cols = min(4, n_display)
    n_rows = (n_display + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(n_display):
        row = idx // n_cols
        col = (idx % n_cols) * 2
        
        # 原图
        ax_img = axes[row, col]
        img_np = images[idx].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        ax_img.imshow(img_np)
        
        # 标题：预测/真实
        pred_class = preds[idx].item()
        title = f"Pred: {class_names[pred_class] if class_names else pred_class}"
        if labels is not None:
            true_class = labels[idx].item()
            title += f"\nTrue: {class_names[true_class] if class_names else true_class}"
            color = 'green' if pred_class == true_class else 'red'
        else:
            color = 'black'
        ax_img.set_title(title, fontsize=9, color=color)
        ax_img.axis('off')
        
        # 尺度选择可视化
        ax_scale = axes[row, col + 1]
        scale_map = scale_indices[idx].cpu().numpy()  # [H', W']
        
        # 上采样到原图大小
        scale_h, scale_w = scale_map.shape
        
        # 创建混合可视化
        ax_scale.imshow(img_np, alpha=0.4)
        
        # 覆盖尺度颜色
        scale_overlay = np.zeros((H, W, 4))
        
        for sy in range(scale_h):
            for sx in range(scale_w):
                scale_idx = scale_map[sy, sx]
                color_rgba = scale_colors[scale_idx]
                
                # 计算像素范围 (近似)
                y_start = int(sy * H / scale_h)
                y_end = int((sy + 1) * H / scale_h)
                x_start = int(sx * W / scale_w)
                x_end = int((sx + 1) * W / scale_w)
                
                scale_overlay[y_start:y_end, x_start:x_end, :3] = color_rgba[:3]
                scale_overlay[y_start:y_end, x_start:x_end, 3] = 0.5
        
        ax_scale.imshow(scale_overlay)
        
        # 绘制网格边界
        cell_h = H / scale_h
        cell_w = W / scale_w
        for i in range(scale_h + 1):
            ax_scale.axhline(i * cell_h, color='white', linewidth=0.5, alpha=0.3)
        for j in range(scale_w + 1):
            ax_scale.axvline(j * cell_w, color='white', linewidth=0.5, alpha=0.3)
        
        ax_scale.set_title('Adaptive Scale Map', fontsize=9)
        ax_scale.axis('off')
    
    # 隐藏多余子图
    for idx in range(n_display, n_rows * n_cols):
        row = idx // n_cols
        col = (idx % n_cols) * 2
        if row < axes.shape[0] and col < axes.shape[1]:
            axes[row, col].set_visible(False)
            axes[row, col + 1].set_visible(False)
    
    # 添加图例
    legend_patches = [
        mpatches.Patch(color=scale_colors[i], alpha=0.7, 
                      label=f'Scale {i+1}: {patch_sizes[i]}×{patch_sizes[i]}')
        for i in range(n_scales)
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=n_scales, 
              fontsize=10, bbox_to_anchor=(0.5, -0.02))
    
    fig.suptitle('Adaptive Multi-Scale Tokenization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Adaptive scale selection saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def visualize_scale_distribution(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    n_batches: int = 10,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """可视化数据集上的尺度分布统计
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 计算设备
        n_batches: 采样批次数
        save_path: 保存路径
        show: 是否显示
    """
    model.eval()
    tokenizer = model.tokenizer
    
    if not hasattr(tokenizer, 'complexity_head'):
        print("[WARN] Tokenizer does not support adaptive scale selection")
        return None
    
    patch_sizes = tokenizer.patch_sizes
    n_scales = len(patch_sizes)
    
    scale_counts = np.zeros(n_scales)
    total_regions = 0
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(data_loader):
            if batch_idx >= n_batches:
                break
            
            images = images.to(device)
            # 新 API: 通过 encoder 和 _compute_scale_weights
            features_dict = tokenizer.encoder(images)
            min_ps = min(features_dict.keys())
            _, target_size = features_dict[min_ps]
            scale_weights = tokenizer._compute_scale_weights(features_dict, target_size)
            scale_indices = scale_weights.argmax(dim=1)  # [B, H', W']
            
            for s in range(n_scales):
                scale_counts[s] += (scale_indices == s).sum().item()
            total_regions += scale_indices.numel()
    
    scale_percentages = scale_counts / total_regions * 100
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. 饼图
    ax1 = axes[0]
    colors = plt.cm.Set1(np.linspace(0, 0.8, n_scales))
    labels = [f'{ps}×{ps}' for ps in patch_sizes]
    explode = [0.02] * n_scales
    
    wedges, texts, autotexts = ax1.pie(
        scale_percentages, labels=labels, autopct='%1.1f%%',
        colors=colors, explode=explode, startangle=90,
        textprops={'fontsize': 10}
    )
    ax1.set_title('Scale Distribution (Pie)', fontsize=12)
    
    # 2. 柱状图
    ax2 = axes[1]
    x = np.arange(n_scales)
    bars = ax2.bar(x, scale_percentages, color=colors, alpha=0.8, edgecolor='black')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Scale {i+1}\n({ps}×{ps})' for i, ps in enumerate(patch_sizes)])
    ax2.set_ylabel('Percentage (%)', fontsize=11)
    ax2.set_title('Scale Distribution (Bar)', fontsize=12)
    ax2.set_ylim(0, max(scale_percentages) * 1.2)
    
    # 添加数值标注
    for bar, pct in zip(bars, scale_percentages):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
    
    fig.suptitle(f'Adaptive Scale Selection Distribution\n(Sampled {n_batches} batches, {total_regions:,} regions)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Scale distribution saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def visualize_scale_by_complexity(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """可视化尺度选择与图像复杂度的关系
    
    展示模型如何根据区域复杂度选择不同尺度：
    - 简单区域（纯色/平滑）→ 大 patch
    - 复杂区域（边缘/纹理）→ 小 patch
    """
    model.eval()
    tokenizer = model.tokenizer
    
    if not hasattr(tokenizer, 'complexity_head'):
        print("[WARN] Tokenizer does not support adaptive scale selection")
        return None
    
    B = min(images.shape[0], 4)
    images = images[:B]
    H, W = images.shape[2], images.shape[3]
    patch_sizes = tokenizer.patch_sizes
    n_scales = len(patch_sizes)
    
    with torch.no_grad():
        images_device = images.to(device)
        
        # 获取尺度 logits（新 API: 通过 encoder 和 complexity_head）
        features_dict = tokenizer.encoder(images_device)
        min_ps = min(features_dict.keys())
        base_feat, target_size = features_dict[min_ps]
        
        # 手动计算 logits（用于可视化）
        aligned_features = []
        for ps in tokenizer.patch_sizes:
            if ps in features_dict:
                feat, _ = features_dict[ps]
                if feat.shape[-2:] != target_size:
                    feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                aligned_features.append(feat)
        concat_features = torch.cat(aligned_features, dim=1)
        logits = tokenizer.complexity_head(concat_features)  # [B, n_scales, H', W']
        
        scale_weights = tokenizer._compute_scale_weights(features_dict, target_size)
        scale_indices = scale_weights.argmax(dim=1)  # [B, H', W']
        
        # 计算图像梯度（边缘检测）作为复杂度参考
        gray = images_device.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2).squeeze(1)  # [B, H, W]
    
    fig, axes = plt.subplots(B, 4, figsize=(16, 4 * B))
    if B == 1:
        axes = axes.reshape(1, -1)
    
    scale_colors = plt.cm.Set1(np.linspace(0, 0.8, n_scales))
    
    for idx in range(B):
        # 1. 原图
        ax1 = axes[idx, 0]
        img_np = images[idx].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        ax1.imshow(img_np)
        ax1.set_title('Original Image', fontsize=10)
        ax1.axis('off')
        
        # 2. 边缘/复杂度热图
        ax2 = axes[idx, 1]
        edge_np = edge_magnitude[idx].cpu().numpy()
        im2 = ax2.imshow(edge_np, cmap='hot')
        ax2.set_title('Edge Magnitude\n(Image Complexity)', fontsize=10)
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # 3. 尺度选择热图 (logits softmax)
        ax3 = axes[idx, 2]
        scale_probs = F.softmax(logits[idx], dim=0).cpu().numpy()  # [n_scales, H', W']
        
        # 显示细尺度的概率（小 patch = 高复杂度区域）
        fine_scale_prob = scale_probs[0]  # 最细尺度的概率
        im3 = ax3.imshow(fine_scale_prob, cmap='coolwarm', vmin=0, vmax=1)
        ax3.set_title(f'Fine Scale ({patch_sizes[0]}×{patch_sizes[0]}) Probability', fontsize=10)
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        
        # 4. 最终尺度选择
        ax4 = axes[idx, 3]
        scale_map = scale_indices[idx].cpu().numpy()
        scale_h, scale_w = scale_map.shape
        
        ax4.imshow(img_np, alpha=0.4)
        
        # 覆盖尺度颜色
        for sy in range(scale_h):
            for sx in range(scale_w):
                scale_idx = scale_map[sy, sx]
                color_rgba = scale_colors[scale_idx]
                
                y_start = int(sy * H / scale_h)
                y_end = int((sy + 1) * H / scale_h)
                x_start = int(sx * W / scale_w)
                x_end = int((sx + 1) * W / scale_w)
                
                rect = plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                                     facecolor=color_rgba, alpha=0.5, edgecolor='white', linewidth=0.5)
                ax4.add_patch(rect)
        
        ax4.set_xlim(0, W)
        ax4.set_ylim(H, 0)
        ax4.set_title('Selected Scale Map', fontsize=10)
        ax4.axis('off')
    
    # 图例
    legend_patches = [
        mpatches.Patch(color=scale_colors[i], alpha=0.7, 
                      label=f'Scale {i+1}: {patch_sizes[i]}×{patch_sizes[i]}')
        for i in range(n_scales)
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=n_scales, 
              fontsize=10, bbox_to_anchor=(0.5, -0.02))
    
    fig.suptitle('Scale Selection vs Image Complexity', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Scale by complexity saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# 模型评估
# ============================================================================

def load_model_and_config(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """加载模型和配置
    
    支持加载旧版检查点(使用 complexity_estimator)和新版检查点(使用 complexity_head)。
    对于旧版检查点，会尝试转换键名以保持兼容性。
    
    与 train_fractal_vit.py 保持完全一致的模型创建方式。
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    
    # 从配置重建模型
    dataset_name = config.get('dataset', 'cifar10')
    spec = DATASETS.get(dataset_name, DATASETS['cifar10'])
    
    # 与 train_fractal_vit.py 完全对齐的模型创建
    model = FractalCurveViT(
        image_size=max(spec.image_size, 32),
        num_classes=spec.num_classes,
        dim=config.get('dim', 192),
        depth=config.get('depth', 8),
        heads=config.get('heads', 8),
        mlp_dim=config.get('mlp_dim', 384),
        pool=config.get('pool', 'cls'),
        channels=spec.channels,
        dim_head=config.get('dim_head', 32),
        dropout=config.get('dropout', 0.1),
        emb_dropout=config.get('emb_dropout', 0.1),
        drop_path_rate=config.get('drop_path', 0.0),
        min_patch_size=(4, 4),
        max_level=config.get('max_level', 4),
        use_checkpoint=config.get('gradient_checkpoint', False),
        ffn_type=config.get('ffn_type', 'swiglu_level'),
        tokenizer_type="streaming_v2",
        num_scales=config.get('num_scales', 3),
        streaming_tau=config.get('gumbel_tau_init', 2.0),
        # V2 Tokenizer 配置 (与训练脚本对齐)
        variable_tokens=config.get('variable_tokens', True),
        use_soft_weights=config.get('use_soft_weights', False),
    ).to(device)
    
    # 配置 Tokenizer 的深度偏置预热参数 (与训练脚本一致)
    if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'set_depth_bias'):
        model.tokenizer.set_depth_bias(
            bias_strength=config.get('depth_bias_max', 2.0),
            max_bias=config.get('depth_bias_max', 2.0),
            decay_power=config.get('depth_bias_decay', 2.0),
            warmup_ratio=config.get('depth_bias_warmup', 0.2),
        )
        # 同步 Gumbel 温度范围
        if hasattr(model.tokenizer, 'tau_min'):
            model.tokenizer.tau_min = config.get('gumbel_tau_min', 0.5)
            model.tokenizer.tau_max = config.get('gumbel_tau_max', 5.0)
    
    # 处理检查点的键名映射
    state_dict = ckpt['model_state_dict']
    new_state_dict = {}
    renamed_keys = []
    compiled_prefix_stripped = False
    
    for key, value in state_dict.items():
        new_key = key
        
        # 处理 torch.compile 生成的 _orig_mod. 前缀
        if new_key.startswith('_orig_mod.'):
            new_key = new_key[len('_orig_mod.'):]
            compiled_prefix_stripped = True
        
        # 旧版: complexity_estimator -> 新版: complexity_head (结构不同，无法直接映射)
        # 旧版: depth_selector -> 新版: 已移除
        if 'complexity_estimator' in new_key or 'depth_selector' in new_key:
            # 这些键在新架构中不存在，跳过
            renamed_keys.append(key)
            continue
        new_state_dict[new_key] = value
    
    if compiled_prefix_stripped:
        print(f"[INFO] Stripped '_orig_mod.' prefix from torch.compile checkpoint ({len(state_dict)} keys)")
    
    if renamed_keys:
        print(f"[WARN] Checkpoint uses old architecture. Skipping {len(renamed_keys)} incompatible keys:")
        for k in renamed_keys[:5]:  # 只显示前5个
            print(f"       - {k}")
        if len(renamed_keys) > 5:
            print(f"       ... and {len(renamed_keys) - 5} more")
        print("[WARN] Scale selection visualization will not work for this checkpoint.")
    
    # 使用 strict=False 加载，允许新键未匹配
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
    if missing:
        # 过滤掉预期缺失的键（新架构的 complexity_head）
        truly_missing = [k for k in missing if 'complexity_head' not in k]
        if truly_missing:
            print(f"[WARN] Missing keys: {truly_missing[:5]}")
    
    model.eval()
    
    return model, config


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """详细评估模型"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            outputs, _ = model(imgs, return_aux_info=True)
            loss = F.cross_entropy(outputs, labels)
            
            probs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            total_loss += loss.item()
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算指标
    accuracy = (all_preds == all_labels).mean() * 100
    
    # 每类准确率
    per_class_acc = {}
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == all_labels[mask]).mean() * 100
            class_name = class_names[c] if class_names else str(c)
            per_class_acc[class_name] = class_acc
    
    # 混淆矩阵
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for pred, label in zip(all_preds, all_labels):
        confusion[label, pred] += 1
    
    return {
        'accuracy': accuracy,
        'loss': total_loss / len(test_loader),
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': confusion,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
    }


@torch.no_grad()
def check_train_eval_consistency(
    model: nn.Module,
    sample_images: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """检查模型在 train/eval 模式下的一致性
    
    与 train_fractal_vit.py 中的 verify_train_eval_consistency 保持一致。
    
    Args:
        model: 模型
        sample_images: 样本图像 [B, C, H, W]
        device: 计算设备
        
    Returns:
        一致性报告字典
    """
    report = {
        'passed': True,
        'checks': {},
        'warnings': [],
    }
    
    imgs = sample_images.to(device)
    
    # 检查 1: 深度偏置衰减
    if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'get_depth_bias'):
        depth_bias = model.tokenizer.get_depth_bias()
        bias_check = {
            'current_value': depth_bias,
            'threshold': 0.01,
            'passed': depth_bias <= 0.01,
        }
        report['checks']['depth_bias_decayed'] = bias_check
        
        if not bias_check['passed']:
            report['warnings'].append(
                f"⚠️ 深度偏置未完全衰减 ({depth_bias:.4f} > 0.01)"
            )
            report['passed'] = False
        else:
            print(f"  ✓ 深度偏置已衰减: {depth_bias:.6f} (< 0.01)")
    
    # 检查 2: train/eval 输出差异
    model.eval()
    out_eval, _ = model(imgs, return_aux_info=True)
    
    model.train()
    out_train, _ = model(imgs, return_aux_info=True)
    model.eval()  # 恢复 eval 模式
    
    # 计算输出差异
    output_diff = (out_eval - out_train).abs()
    max_diff = output_diff.max().item()
    mean_diff = output_diff.mean().item()
    
    output_check = {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'threshold': 0.1,
        'passed': max_diff < 0.1,
    }
    report['checks']['output_consistency'] = output_check
    
    if output_check['passed']:
        print(f"  ✓ 输出一致性: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    else:
        report['warnings'].append(
            f"⚠️ train/eval 输出差异较大 (max={max_diff:.4f})"
        )
        print(f"  ⚠ 输出差异: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    
    # 检查 3: 尺度选择稳定性
    if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'compute_scale_distribution'):
        dist1 = model.tokenizer.compute_scale_distribution(imgs)
        dist2 = model.tokenizer.compute_scale_distribution(imgs)
        
        scale_stable = all(
            abs(dist1['scale_ratios'][ps] - dist2['scale_ratios'][ps]) < 0.001
            for ps in dist1['scale_ratios']
        )
        
        stability_check = {
            'run1': dist1['scale_ratios'],
            'run2': dist2['scale_ratios'],
            'passed': scale_stable,
        }
        report['checks']['scale_stability'] = stability_check
        
        if scale_stable:
            print(f"  ✓ 尺度选择稳定 (eval 模式下确定性)")
        else:
            report['warnings'].append("⚠️ 尺度选择不稳定")
            report['passed'] = False
    
    # 检查 4: 训练状态信息
    if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'get_training_stats'):
        stats = model.tokenizer.get_training_stats()
        report['training_stats'] = stats
        print(f"  ✓ Gumbel τ={stats['gumbel_tau']:.4f}, use_soft_weights={stats['use_soft_weights']}")
    
    # 总结
    print()
    if report['passed']:
        print("  ✅ 一致性检查通过: 模型可正确用于推理")
    else:
        print("  ❌ 一致性检查警告:")
        for w in report['warnings']:
            print(f"     {w}")
    
    return report


def visualize_confusion_matrix(
    confusion: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """可视化混淆矩阵"""
    num_classes = confusion.shape[0]
    
    # 归一化
    confusion_norm = confusion.astype(float) / (confusion.sum(axis=1, keepdims=True) + 1e-8)
    
    fig, ax = plt.subplots(figsize=(min(12, num_classes), min(10, num_classes)))
    
    im = ax.imshow(confusion_norm, cmap='Blues')
    
    # 设置坐标轴
    if class_names and num_classes <= 20:
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14)
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # 添加数值标注 (仅小矩阵)
    if num_classes <= 15:
        for i in range(num_classes):
            for j in range(num_classes):
                val = confusion_norm[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       color=color, fontsize=6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Confusion matrix saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def visualize_per_class_accuracy(
    per_class_acc: Dict[str, float],
    save_path: Optional[Path] = None,
    show: bool = True,
    top_k: int = 20,
) -> plt.Figure:
    """可视化每类准确率"""
    # 排序
    sorted_items = sorted(per_class_acc.items(), key=lambda x: x[1])
    
    # 限制显示数量
    if len(sorted_items) > top_k * 2:
        # 显示最差和最好的
        display_items = sorted_items[:top_k] + sorted_items[-top_k:]
    else:
        display_items = sorted_items
    
    classes = [item[0] for item in display_items]
    accs = [item[1] for item in display_items]
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(classes) * 0.3)))
    
    colors = ['red' if acc < 50 else 'orange' if acc < 70 else 'green' for acc in accs]
    
    bars = ax.barh(classes, accs, color=colors, alpha=0.7)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Accuracy', fontsize=14)
    ax.set_xlim(0, 100)
    
    # 添加数值标注
    for bar, acc in zip(bars, accs):
        ax.text(acc + 1, bar.get_y() + bar.get_height()/2, 
               f'{acc:.1f}%', va='center', fontsize=8)
    
    # 添加平均线
    mean_acc = np.mean(list(per_class_acc.values()))
    ax.axvline(mean_acc, color='blue', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_acc:.1f}%')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Per-class accuracy saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def visualize_sample_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    n_samples: int = 16,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """可视化样本预测结果"""
    model.eval()
    
    samples = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            for i in range(min(len(imgs), n_samples - len(samples))):
                img = imgs[i:i+1].to(device)
                label = labels[i].item()
                
                output, _ = model(img, return_aux_info=True)
                probs = F.softmax(output, dim=1)
                pred = output.argmax(1).item()
                conf = probs[0, pred].item()
                
                samples.append({
                    'image': imgs[i].cpu(),
                    'label': label,
                    'pred': pred,
                    'confidence': conf,
                    'correct': pred == label,
                })
            
            if len(samples) >= n_samples:
                break
    
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()
    
    for idx, sample in enumerate(samples):
        ax = axes[idx]
        
        # 反归一化显示
        img = sample['image'].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        ax.imshow(img)
        
        label_name = class_names[sample['label']] if class_names else str(sample['label'])
        pred_name = class_names[sample['pred']] if class_names else str(sample['pred'])
        
        color = 'green' if sample['correct'] else 'red'
        title = f"True: {label_name}\nPred: {pred_name} ({sample['confidence']*100:.1f}%)"
        ax.set_title(title, fontsize=9, color=color)
        ax.axis('off')
    
    # 隐藏多余子图
    for idx in range(len(samples), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Sample Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Sample predictions saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# 注意力可视化
# ============================================================================

def visualize_attention_maps(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """可视化注意力图 (需要模型支持)"""
    model.eval()
    
    # 注册 hook 捕获注意力权重
    attention_maps = []
    
    def hook_fn(module, input, output):
        if hasattr(module, 'attn_weights'):
            attention_maps.append(module.attn_weights.detach().cpu())
    
    # 尝试注册 hook
    hooks = []
    for name, module in model.named_modules():
        if 'attention' in name.lower() or 'attn' in name.lower():
            hooks.append(module.register_forward_hook(hook_fn))
    
    # 前向传播
    with torch.no_grad():
        _ = model(image.to(device))
    
    # 移除 hooks
    for hook in hooks:
        hook.remove()
    
    if not attention_maps:
        print("[WARN] No attention maps captured. Model may not expose attention weights.")
        return None
    
    # 可视化
    n_layers = len(attention_maps)
    fig, axes = plt.subplots(1, min(4, n_layers), figsize=(16, 4))
    if n_layers == 1:
        axes = [axes]
    
    for i, attn in enumerate(attention_maps[:4]):
        ax = axes[i]
        # 取第一个头的平均
        attn_avg = attn[0].mean(0).numpy()  # [seq_len, seq_len]
        im = ax.imshow(attn_avg, cmap='viridis')
        ax.set_title(f'Layer {i+1} Attention', fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    fig.suptitle('Attention Maps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Attention maps saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# 综合可视化报告
# ============================================================================

def generate_full_report(
    checkpoint_path: Path,
    output_dir: Path,
    dataset_name: str = 'cifar10',
    device: Optional[torch.device] = None,
    show: bool = False,
) -> Dict[str, Any]:
    """生成完整的评估报告
    
    Args:
        checkpoint_path: 模型检查点路径
        output_dir: 输出目录
        dataset_name: 数据集名称
        device: 计算设备
        show: 是否显示图表
        
    Returns:
        评估结果字典
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("FRACTAL VIT EVALUATION REPORT")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # 1. 加载模型
    print("[1/6] Loading model...")
    model, config = load_model_and_config(checkpoint_path, device)
    print(f"      Config: dim={config.get('dim')}, depth={config.get('depth')}, "
          f"heads={config.get('heads')}")
    print(f"      FFN: {config.get('ffn_type', 'swiglu_level')}, "
          f"Scales: {config.get('num_scales', 3)}")
    print(f"      Gumbel τ: init={config.get('gumbel_tau_init', 2.0)}, "
          f"min={config.get('gumbel_tau_min', 0.5)}, max={config.get('gumbel_tau_max', 5.0)}")
    print(f"      Depth Bias: max={config.get('depth_bias_max', 2.0)}, "
          f"decay={config.get('depth_bias_decay', 2.0)}, warmup={config.get('depth_bias_warmup', 0.2)}")
    
    # 2. 准备数据
    print("[2/6] Loading test data...")
    spec = DATASETS.get(dataset_name, DATASETS['cifar10'])
    
    test_tf = transforms.Compose([
        transforms.Resize(max(spec.image_size, 32)),
        transforms.ToTensor(),
        transforms.Normalize(spec.mean, spec.std),
    ])
    
    data_root = PROJECT_ROOT / "data"
    
    # 自动下载数据集（如果需要）
    if not prepare_dataset(dataset_name, data_root):
        raise RuntimeError(f"Failed to prepare dataset: {dataset_name}")
    
    if dataset_name == 'cifar10':
        test_ds = datasets.CIFAR10(data_root, train=False, download=True, transform=test_tf)
    elif dataset_name == 'cifar100':
        test_ds = datasets.CIFAR100(data_root, train=False, download=True, transform=test_tf)
    elif dataset_name == 'mnist':
        test_ds = datasets.MNIST(data_root, train=False, download=True, transform=test_tf)
    elif dataset_name == 'tiny-imagenet':
        train_dir = data_root / "tiny-imagenet-200" / "train"
        test_dir = data_root / "tiny-imagenet-200" / "val"
        
        # 先加载训练集获取标准类别映射
        train_ds = datasets.ImageFolder(str(train_dir))
        test_ds = datasets.ImageFolder(str(test_dir), transform=test_tf)
        
        # 关键修复：重新映射测试集标签以匹配训练集
        train_class_to_idx = train_ds.class_to_idx
        test_class_to_idx = test_ds.class_to_idx
        
        label_mapping = {}
        for class_name, test_label in test_class_to_idx.items():
            if class_name in train_class_to_idx:
                label_mapping[test_label] = train_class_to_idx[class_name]
            else:
                print(f"[WARN] Class {class_name} not found in training set")
                label_mapping[test_label] = test_label
        
        # 重新映射 samples 和 targets
        test_ds.samples = [(path, label_mapping.get(label, label)) for path, label in test_ds.samples]
        test_ds.targets = [label_mapping.get(label, label) for label in test_ds.targets]
        test_ds.class_to_idx = train_class_to_idx
        
        print(f"[OK] Test set label mapping applied: {len(label_mapping)} classes")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)
    
    # 3. 评估模型
    print("[3/6] Evaluating model...")
    results = evaluate_model(
        model, test_loader, device,
        num_classes=spec.num_classes,
        class_names=spec.classes,
    )
    
    print(f"      Accuracy: {results['accuracy']:.2f}%")
    print(f"      Loss: {results['loss']:.4f}")
    
    # 4. 可视化 Hilbert 曲线
    print("[4/6] Generating Hilbert curve visualizations...")
    visualize_hilbert_curve(
        max_order=5,
        save_path=output_dir / "hilbert_curves.png",
        show=show,
    )
    
    visualize_hilbert_locality(
        order=4,
        save_path=output_dir / "hilbert_locality.png",
        show=show,
    )
    
    # 5. 在样本图像上可视化
    print("[5/8] Visualizing tokenization on sample images...")
    sample_imgs, sample_labels = next(iter(test_loader))
    
    visualize_hilbert_on_image(
        sample_imgs[0],
        patch_size=4,
        save_path=output_dir / "hilbert_on_image.png",
        show=show,
    )
    
    # 多尺度可视化
    if hasattr(model, 'tokenizer'):
        visualize_multi_scale_tokenization(
            sample_imgs[0:1].to(device),
            model.tokenizer,
            save_path=output_dir / "multi_scale_tokenization.png",
            show=show,
        )
    
    # 6. 混合 Level 分割可视化
    print("[6/8] Visualizing adaptive scale selection...")
    if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'complexity_head'):
        visualize_adaptive_scale_selection(
            model, sample_imgs[:8], device,
            class_names=spec.classes,
            labels=sample_labels[:8],
            save_path=output_dir / "adaptive_scale_selection.png",
            show=show,
        )
        
        visualize_scale_distribution(
            model, test_loader, device,
            n_batches=10,
            save_path=output_dir / "scale_distribution.png",
            show=show,
        )
        
        visualize_scale_by_complexity(
            model, sample_imgs[:4], device,
            save_path=output_dir / "scale_by_complexity.png",
            show=show,
        )
    else:
        print("      [SKIP] Tokenizer does not support adaptive scale selection")
    
    # 6.5. Train/Eval 一致性检查
    print("[6.5/8] Checking train/eval consistency...")
    consistency_report = check_train_eval_consistency(model, sample_imgs[:4], device)
    
    # 保存一致性报告
    with open(output_dir / "consistency_report.json", 'w') as f:
        # 转换不可序列化的类型
        serializable_report = {}
        for k, v in consistency_report.items():
            if isinstance(v, dict):
                serializable_report[k] = {
                    str(kk): (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                    for kk, vv in v.items()
                }
            else:
                serializable_report[k] = v
        json.dump(serializable_report, f, indent=2, default=str)
    
    # 7. 评估可视化
    print("[7/8] Generating evaluation visualizations...")
    
    visualize_confusion_matrix(
        results['confusion_matrix'],
        class_names=spec.classes,
        save_path=output_dir / "confusion_matrix.png",
        show=show,
    )
    
    visualize_per_class_accuracy(
        results['per_class_accuracy'],
        save_path=output_dir / "per_class_accuracy.png",
        show=show,
    )
    
    visualize_sample_predictions(
        model, test_loader, device,
        class_names=spec.classes,
        n_samples=16,
        save_path=output_dir / "sample_predictions.png",
        show=show,
    )
    
    # 8. 保存结果
    print("[8/8] Saving report...")
    
    # 获取 tokenizer 状态
    tokenizer_stats = {}
    if hasattr(model, 'tokenizer'):
        if hasattr(model.tokenizer, 'get_training_stats'):
            tokenizer_stats = model.tokenizer.get_training_stats()
        if hasattr(model.tokenizer, 'get_depth_bias'):
            tokenizer_stats['depth_bias'] = model.tokenizer.get_depth_bias()
        if hasattr(model.tokenizer, 'get_temperature'):
            tokenizer_stats['gumbel_tau'] = model.tokenizer.get_temperature()
    
    # 保存结果
    report = {
        'checkpoint': str(checkpoint_path),
        'dataset': dataset_name,
        'accuracy': results['accuracy'],
        'loss': results['loss'],
        'config': config,
        'per_class_accuracy': results['per_class_accuracy'],
        'consistency_passed': consistency_report.get('passed', None),
        'tokenizer_stats': tokenizer_stats,
    }
    
    with open(output_dir / "evaluation_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print("REPORT COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"  - hilbert_curves.png")
    print(f"  - hilbert_locality.png")
    print(f"  - hilbert_on_image.png")
    print(f"  - multi_scale_tokenization.png")
    print(f"  - adaptive_scale_selection.png")
    print(f"  - scale_distribution.png")
    print(f"  - scale_by_complexity.png")
    print(f"  - confusion_matrix.png")
    print(f"  - per_class_accuracy.png")
    print(f"  - sample_predictions.png")
    print(f"  - consistency_report.json       [NEW]")
    print(f"  - evaluation_report.json")
    print(f"{'='*70}\n")
    
    return report


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fractal ViT Evaluation & Visualization")
    
    # 模式
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Model checkpoint path")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for visualizations")
    parser.add_argument("--dataset", type=str, default="cifar10",
                       choices=["cifar10", "cifar100", "mnist", "tiny-imagenet"],
                       help="Dataset to use for evaluation")
    
    # 可视化选项
    parser.add_argument("--visualize-hilbert", action="store_true",
                       help="Only visualize Hilbert curves (no model needed)")
    parser.add_argument("--max-order", type=int, default=5,
                       help="Max order for Hilbert curve visualization")
    parser.add_argument("--visualize-tokenization", action="store_true",
                       help="Visualize tokenization on sample images")
    parser.add_argument("--visualize-scale", action="store_true",
                       help="Visualize adaptive scale selection (mixed level segmentation)")
    parser.add_argument("--n-samples", type=int, default=8,
                       help="Number of samples for scale visualization")
    
    # 系统
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--show", action="store_true",
                       help="Show plots interactively")
    
    args = parser.parse_args()
    
    # 确定设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "workspace" / "visualizations" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 仅可视化 Hilbert 曲线
    if args.visualize_hilbert:
        print("\n[*] Generating Hilbert curve visualizations...")
        visualize_hilbert_curve(
            max_order=args.max_order,
            save_path=output_dir / "hilbert_curves.png",
            show=args.show,
        )
        visualize_hilbert_locality(
            order=min(4, args.max_order),
            save_path=output_dir / "hilbert_locality.png",
            show=args.show,
        )
        print(f"[OK] Visualizations saved to: {output_dir}")
        return
    
    # 需要模型的评估
    if args.checkpoint is None:
        # 尝试找到最新的检查点
        exp_dir = PROJECT_ROOT / "experiments"
        if exp_dir.exists():
            exp_folders = sorted(exp_dir.glob("fractal_vit_*"))
            if exp_folders:
                latest = exp_folders[-1]
                ckpt_path = latest / "checkpoints" / "best.pth"
                if ckpt_path.exists():
                    args.checkpoint = str(ckpt_path)
                    print(f"[*] Using latest checkpoint: {args.checkpoint}")
    
    if args.checkpoint is None:
        print("[ERROR] No checkpoint specified. Use --checkpoint or --visualize-hilbert")
        parser.print_help()
        return
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return
    
    # 仅可视化尺度选择
    if args.visualize_scale:
        print("\n[*] Generating adaptive scale selection visualizations...")
        
        model, config = load_model_and_config(checkpoint_path, device)
        spec = DATASETS.get(args.dataset, DATASETS['cifar10'])
        
        test_tf = transforms.Compose([
            transforms.Resize(max(spec.image_size, 32)),
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
        ])
        
        data_root = PROJECT_ROOT / "data"
        
        # 自动下载数据集（如果需要）
        if not prepare_dataset(args.dataset, data_root):
            raise RuntimeError(f"Failed to prepare dataset: {args.dataset}")
        
        if args.dataset == 'cifar10':
            test_ds = datasets.CIFAR10(data_root, train=False, download=True, transform=test_tf)
        elif args.dataset == 'cifar100':
            test_ds = datasets.CIFAR100(data_root, train=False, download=True, transform=test_tf)
        elif args.dataset == 'mnist':
            test_ds = datasets.MNIST(data_root, train=False, download=True, transform=test_tf)
        elif args.dataset == 'tiny-imagenet':
            test_dir = data_root / "tiny-imagenet-200" / "val"
            test_ds = datasets.ImageFolder(str(test_dir), transform=test_tf)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        
        test_loader = DataLoader(test_ds, batch_size=args.n_samples, shuffle=True, num_workers=2)
        sample_imgs, sample_labels = next(iter(test_loader))
        
        visualize_adaptive_scale_selection(
            model, sample_imgs, device,
            class_names=spec.classes,
            labels=sample_labels,
            save_path=output_dir / "adaptive_scale_selection.png",
            show=args.show,
        )
        
        visualize_scale_distribution(
            model, test_loader, device,
            n_batches=10,
            save_path=output_dir / "scale_distribution.png",
            show=args.show,
        )
        
        visualize_scale_by_complexity(
            model, sample_imgs[:4], device,
            save_path=output_dir / "scale_by_complexity.png",
            show=args.show,
        )
        
        print(f"[OK] Scale visualizations saved to: {output_dir}")
        return
    
    # 生成完整报告
    generate_full_report(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        dataset_name=args.dataset,
        device=device,
        show=args.show,
    )


if __name__ == "__main__":
    main()
