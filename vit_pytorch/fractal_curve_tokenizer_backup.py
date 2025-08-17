import torch
import torch.nn as nn
import numpy as np

class FractalHilbertTokenizer(nn.Module):
    """
    分形Hilbert曲线tokenizer - 原始版本，保持向后兼容性
    """
    def __init__(self, min_patch_size=(4, 4), max_level=3, learnable_split=False):
        super().__init__()
        self.min_patch_size = min_patch_size
        self.max_level = max_level
        self.learnable_split = learnable_split
        
        if learnable_split:
            # 如果需要可学习分割，这里可以添加决策网络
            pass
            
    def forward(self, images):
        """
        对图像进行分形Hilbert tokenization
        """
        return self.tokenize(images)
    
    def tokenize(self, images):
        """
        Tokenize images using fractal Hilbert curve
        """
        B, C, H, W = images.shape
        tokens_list = []
        levels_list = []
        
        for b in range(B):
            tokens, levels = self.fractal_partition(images[b])
            tokens_list.append(torch.stack(tokens) if tokens else torch.empty(0, C * self.min_patch_size[0] * self.min_patch_size[1]))
            levels_list.append(torch.tensor(levels) if levels else torch.empty(0, dtype=torch.long))
        
        return tokens_list, levels_list
    
    def fractal_partition(self, image, level=0, coord=[]):
        """
        递归分形分割
        """
        C, H, W = image.shape
        tokens = []
        levels = []
        
        # 检查是否应该停止分割
        min_h, min_w = self.min_patch_size
        if H <= min_h or W <= min_w or level >= self.max_level:
            # 创建token并记录
            if H >= min_h and W >= min_w:
                # 裁剪到最小尺寸
                patch_cropped = image[:, :min_h, :min_w]
            else:
                # 补零到最小尺寸
                import torch.nn.functional as F
                pad_h = max(0, min_h - H)
                pad_w = max(0, min_w - W)
                patch_cropped = F.pad(image, (0, pad_w, 0, pad_h))
            
            token = patch_cropped.flatten()
            tokens.append(token)
            levels.append(level)
            return tokens, levels
        
        # 继续分割
        mid_h = H // 2
        mid_w = W // 2
        
        # 四个子区域
        quadrants = [
            image[:, :mid_h, :mid_w],      # 左上
            image[:, :mid_h, mid_w:],      # 右上
            image[:, mid_h:, :mid_w],      # 左下
            image[:, mid_h:, mid_w:]       # 右下
        ]
        
        # 按Hilbert顺序处理
        hilbert_order = self.get_hilbert_order(level, H, W)
        
        for i in hilbert_order:
            if i < len(quadrants):
                quad = quadrants[i]
                if quad.numel() > 0:
                    sub_tokens, sub_levels = self.fractal_partition(quad, level + 1, coord + [i])
                    tokens.extend(sub_tokens)
                    levels.extend(sub_levels)
        
        return tokens, levels
    
    def get_hilbert_order(self, level, h, w):
        """
        获取Hilbert曲线的遍历顺序
        """
        # 简化的Hilbert顺序，根据层级变化
        orders = [
            [0, 1, 3, 2],  # 标准顺序
            [2, 0, 1, 3],  # 左下开始
            [1, 3, 2, 0],  # 右上开始  
            [3, 2, 0, 1],  # 右下开始
        ]
        return orders[level % len(orders)]

# 兼容性别名
AdaptiveFractalHilbertTokenizer = FractalHilbertTokenizer
