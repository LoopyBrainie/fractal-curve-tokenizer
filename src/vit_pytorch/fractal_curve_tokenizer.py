import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hilbert import HilbertCurve, get_quadrant_order
from .tokenization import BaseTokenizer, TokenSequence, TokenizerOutput


@dataclass
class PatchInfo:
    """追踪 BFS 中每个 patch 的元信息"""
    patch: torch.Tensor      # [C, H, W]
    level: int               # 当前递归深度
    coord: List[int]         # Hilbert 路径
    dfs_order: float         # DFS 顺序索引，用于最终排序
    

class MiniCNN(nn.Module):
    """轻量级CNN特征提取器，用于分割决策"""

    def __init__(self, in_channels=3, hidden_dim=16, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.InstanceNorm2d(in_channels), # 归一化输入，防止数值过大
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, stride=2),  # 下采样
            nn.ReLU(),
            nn.InstanceNorm2d(hidden_dim), # 中间层归一化
            nn.Conv2d(hidden_dim, out_dim, kernel_size=3, padding=1, stride=2),  # 再次下采样
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局池化
            nn.Flatten(),
        )

    def forward(self, x):
        # x: [B, C, H, W] or [C, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.net(x)


class LearnableSplitDecision(nn.Module):
    """增强的可学习分割决策网络，支持CNN特征和Gumbel-Softmax"""

    def __init__(self, patch_features=6, cnn_features=32, hidden_dim=128):
        super().__init__()
        # patch_features: [level, height, width, variance, mean, edge_density]
        # cnn_features: 来自MiniCNN的特征维度
        input_dim = patch_features + cnn_features
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), # 关键：对混合特征进行归一化，防止方差等大数值主导
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2),  # 输出2个logits: [not_split, split]
        )

        # 初始化偏置，让初始分割倾向于继续分割 (index 1)
        with torch.no_grad():
            self.net[-1].bias[1] += 2.0

    def forward(self, combined_features):
        """
        combined_features: [batch_size, total_features]
        returns: logits [batch_size, 2]
        """
        return self.net(combined_features)


class FractalHilbertTokenizer(BaseTokenizer):
    def __init__(self, min_patch_size=(1, 1), max_level=None, learnable_split=True, adaptive_threshold=0.5, channels=3):
        """
        min_patch_size: 最小整数patch尺寸 (min_h, min_w) - 默认到像素级别
        max_level: 最大递归层数 (None表示无限制，只受min_patch_size限制)
        learnable_split: 是否使用可学习的分割决策
        adaptive_threshold: 自适应分割阈值
        channels: 输入图像的通道数 (RGB=3, 灰度=1)
        """
        super().__init__()
        self.min_patch_size = min_patch_size
        self.max_level = max_level  # 可以为None，表示无限制
        self.learnable_split = learnable_split
        self.adaptive_threshold = adaptive_threshold
        self.channels = channels

        if learnable_split:
            # 增强的分割决策网络，支持更多特征
            self.cnn_encoder = MiniCNN(in_channels=channels, hidden_dim=16, out_dim=32)
            self.split_decision = LearnableSplitDecision(patch_features=6, cnn_features=32, hidden_dim=128)
        else:
            self.cnn_encoder = None
            self.split_decision = None

        # 用于存储REINFORCE所需的log_probs
        self.saved_log_probs = []
        self.saved_entropies = []
        
        # 预注册 Sobel 核为 buffer，避免每次调用时重复创建
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def clear_saved_actions(self):
        self.saved_log_probs = []
        self.saved_entropies = []

    def tokenize(self, images):
        if images.dim() != 4:
            raise ValueError(f"Expected input with shape [B, C, H, W], got {tuple(images.shape)}")

        batch_size, channels, height, width = images.shape
        estimated_max_level = self._estimate_max_possible_level(height, width)
        dynamic_depth_cap = self.max_level if self.max_level is not None else max(estimated_max_level + 5, 12)

        if self.learnable_split and self.split_decision is not None:
            # 确保可学习分割网络与输入位于同一设备
            self.split_decision = self.split_decision.to(images.device)
            self.cnn_encoder = self.cnn_encoder.to(images.device)

        # 清空之前的动作记录
        self.clear_saved_actions()

        min_patch_dim = max(1, min(self.min_patch_size))
        longest_edge = max(height, width)
        additional_levels = int(math.ceil(math.log2(longest_edge / min_patch_dim))) if min_patch_dim > 0 else 0
        max_info_len = max(dynamic_depth_cap + additional_levels + 4, 16)

        sequences = []

        flattened_patch_dim = channels * self.min_patch_size[0] * self.min_patch_size[1]

        # 根据是否启用可学习分割决定使用批处理还是递归
        use_batch = self.learnable_split and self.split_decision is not None
        
        for b in range(batch_size):
            image = images[b]
            sample_device = image.device

            if use_batch:
                # 使用BFS批处理版本 - 减少GPU kernel调用
                tokens_raw, levels_raw = self.fractal_partition_batched(
                    image,
                    max_info_len=max_info_len,
                    depth_limit=dynamic_depth_cap,
                )
            else:
                # 使用原始递归版本 - 用于无可学习分割的场景
                tokens_raw, levels_raw = self.fractal_partition(
                    image,
                    level=0,
                    coord=[],
                    max_info_len=max_info_len,
                    depth_limit=dynamic_depth_cap,
                )

            if len(tokens_raw) == 0:
                empty_tokens = torch.empty(0, flattened_patch_dim, device=sample_device)
                empty_levels = torch.empty(0, max_info_len, dtype=torch.long, device=sample_device)
                sequences.append(TokenSequence(tokens=empty_tokens, metadata={"levels": empty_levels}))
                continue

            tokens = torch.stack(tokens_raw)
            if tokens.device != sample_device:
                tokens = tokens.to(sample_device)

            levels = torch.tensor(levels_raw, dtype=torch.long, device=sample_device)
            sequences.append(TokenSequence(tokens=tokens, metadata={"levels": levels}))

        return TokenizerOutput(sequences)

    def forward(self, images):
        output = self.tokenize(images)
        legacy = output.to_legacy()
        return legacy.tokens, legacy.levels

    def tokenize_legacy(self, images):
        output = self.tokenize(images)
        legacy = output.to_legacy()
        return legacy.tokens, legacy.levels

    def _estimate_max_possible_level(self, h, w):
        """估算给定尺寸下可能达到的最大层级"""
        min_h, min_w = self.min_patch_size
        max_level_h = 0
        max_level_w = 0

        # 计算高度方向最大可能层级
        temp_h = h
        while temp_h > min_h:
            temp_h = temp_h // 2
            max_level_h += 1
            if temp_h <= 0:
                break

        # 计算宽度方向最大可能层级
        temp_w = w
        while temp_w > min_w:
            temp_w = temp_w // 2
            max_level_w += 1
            if temp_w <= 0:
                break

        return max(max_level_h, max_level_w)

    def fractal_partition(self, patch, level, coord, max_info_len, depth_limit):
        # patch: [C, H, W], coord: 当前分形路径向量, max_info_len: 最大信息长度
        C, H, W = patch.shape
        min_h, min_w = self.min_patch_size
        tokens = []
        levels = []

        # 检查是否可以继续分割（支持无限细分）
        # 只要当前patch大于最小尺寸就可以分割，不要求分割后满足最小尺寸
        can_split_h = H > min_h
        can_split_w = W > min_w
        can_split = can_split_h or can_split_w  # 只要任一维度可分割就继续

        # 检查层级限制
        level_limit_reached = level >= depth_limit

        # 基本停止条件 - 增强版，防止无限递归
        basic_stop = not can_split or level_limit_reached

        # 增加额外的安全检查：如果patch太小或层级太深，强制停止
        extra_depth_cap = depth_limit + 5
        safety_check = (H <= 1 and W <= 1) or (level >= extra_depth_cap)

        if basic_stop or safety_check:
            should_stop = True
        else:
            # 使用增强的可学习分割决策
            if self.learnable_split and self.split_decision is not None and self.cnn_encoder is not None:
                # 计算增强的patch特征
                features = self._extract_enhanced_patch_features(patch, level, H, W)
                # 计算CNN特征
                cnn_feat = self.cnn_encoder(patch) # [1, 32]
                
                # 组合特征
                combined = torch.cat([features, cnn_feat], dim=1) # [1, 38]
                
                # 关键修复：检查输入特征是否包含 NaN/Inf，防止污染网络
                if torch.isnan(combined).any() or torch.isinf(combined).any():
                    combined = torch.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=-1.0)

                # 获取logits [1, 2]
                logits = self.split_decision(combined)
                
                # 关键修复：截断 logits 防止 softmax 数值不稳定
                logits = torch.clamp(logits, min=-10.0, max=10.0)
                
                # REINFORCE / Gumbel-Softmax 逻辑
                if self.training:
                    # 再次检查 logits NaN/Inf (虽然前面截断过，但为了双重保险)
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)

                    # 改用 logits 初始化 Categorical，数值更稳定
                    # 避免了手动 softmax 可能出现的极小值下溢问题
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    
                    # 保存 log_prob 用于 REINFORCE
                    self.saved_log_probs.append(dist.log_prob(action))
                    self.saved_entropies.append(dist.entropy())
                    
                    # action 0: stop (not split), action 1: split
                    should_stop = (action.item() == 0)
                else:
                    # 推理模式：直接取最大概率
                    action = torch.argmax(logits, dim=-1)
                    should_stop = (action.item() == 0)
                    
            else:
                # 默认策略：如果设置了 adaptive_threshold，则基于方差进行自适应分割
                if self.adaptive_threshold is not None and self.adaptive_threshold > 0:
                    patch_var = torch.var(patch)
                    # 如果方差小于阈值，说明区域平坦，可以停止分割
                    should_stop = patch_var < self.adaptive_threshold
                else:
                    # 否则尽可能深度分割，但有层级限制
                    should_stop = level >= 10  # 默认最大10层

        if should_stop:
            # 停止分形，输出当前patch作为token
            # 自适应patch尺寸处理
            processed_patch = self._process_patch_to_fixed_size(patch, H, W)
            patch_flat = processed_patch.reshape(-1)
            tokens.append(patch_flat)

            # 记录详细的层级信息（动态长度）
            depth = level
            path = coord.copy()
            # 动态填充到指定长度
            padded = [depth] + path + [0] * (max_info_len - 1 - len(path))
            levels.append(padded[:max_info_len])
            return tokens, levels

        sub_patches = self._adaptive_split(patch, H, W, can_split_h, can_split_w)

        if not sub_patches:
            processed_patch = self._process_patch_to_fixed_size(patch, H, W)
            patch_flat = processed_patch.reshape(-1)
            tokens.append(patch_flat)
            depth = level
            path = coord.copy()
            padded = [depth] + path + [0] * (max_info_len - 1 - len(path))
            levels.append(padded[:max_info_len])
            return tokens, levels

        traversal_order = self._determine_traversal_order(level, H, W, len(sub_patches))

        for patch_idx in traversal_order:
            if patch_idx < len(sub_patches):
                sub = sub_patches[patch_idx].contiguous()
                if sub.numel() > 0:
                    tks, lvls = self.fractal_partition(
                        sub,
                        level + 1,
                        coord + [patch_idx],
                        max_info_len,
                        depth_limit,
                    )
                    tokens.extend(tks)
                    levels.extend(lvls)
        return tokens, levels

    def _determine_traversal_order(self, level, h, w, num_patches):
        if num_patches == 4:
            # 使用统一的 Hilbert 模块获取遍历顺序
            return get_quadrant_order(level, h, w)

        if num_patches == 2:
            # 对于高度分割，按从上到下；宽度分割，从左到右
            return [0, 1]

        return list(range(num_patches))

    def _adaptive_split(self, patch, h, w, can_split_h, can_split_w):
        if can_split_h and can_split_w:
            return self._intelligent_quadrant_split(patch, h, w)

        if can_split_h:
            split_index = self._calculate_split_index(h, self.min_patch_size[0], w)
            split_index = max(1, min(h - 1, split_index))
            top = patch[:, :split_index, :]
            bottom = patch[:, split_index:, :]
            return [top, bottom]

        if can_split_w:
            split_index = self._calculate_split_index(w, self.min_patch_size[1], h)
            split_index = max(1, min(w - 1, split_index))
            left = patch[:, :, :split_index]
            right = patch[:, :, split_index:]
            return [left, right]

        return []

    def _extract_enhanced_patch_features(self, patch, level, h, w):
        """提取增强的patch特征用于分割决策"""
        import torch.nn.functional as F
        device = patch.device

        # 基础统计特征
        patch_var = torch.var(patch)
        patch_mean = torch.mean(patch)

        # 边缘密度特征（使用预注册的Sobel算子）
        gray_patch = torch.mean(patch, dim=0, keepdim=True).unsqueeze(0)  # [1, 1, H, W]

        if h >= 3 and w >= 3:  # 确保patch足够大来应用卷积
            # 确保 Sobel 核与输入在同一设备
            sobel_x = self.sobel_x.to(device)
            sobel_y = self.sobel_y.to(device)
            edge_x = F.conv2d(gray_patch, sobel_x, padding=1)
            edge_y = F.conv2d(gray_patch, sobel_y, padding=1)
            edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
            edge_density = torch.mean(edge_magnitude)
        else:
            edge_density = torch.tensor(0.0, device=device)

        # 纹理复杂度（基于局部方差）
        if h >= 2 and w >= 2:
            local_patches = F.unfold(gray_patch, kernel_size=2, stride=1)  # 2x2局部patch
            local_vars = torch.var(local_patches, dim=1)
            texture_complexity = torch.mean(local_vars)
        else:
            texture_complexity = patch_var

        # 关键修复：特征归一化和对数变换，防止数值过大
        # 1. 归一化尺寸和层级
        norm_level = float(level) / 10.0  # 假设最大层级约10
        norm_h = float(h) / 256.0         # 假设最大尺寸约256
        norm_w = float(w) / 256.0

        # 2. 对数变换处理方差和边缘密度（这些值可能跨度很大）
        log_var = torch.log1p(patch_var).clamp(max=10.0)
        log_edge = torch.log1p(edge_density).clamp(max=10.0)
        
        # 3. 均值归一化 (假设输入已经大致在0-1或-1-1之间，但为了保险起见)
        norm_mean = torch.clamp(patch_mean, -3.0, 3.0)

        # 构建特征向量: [level, height, width, variance, mean, edge_density]
        feature_values = torch.stack(
            [
                torch.tensor(norm_level, device=device),
                torch.tensor(norm_h, device=device),
                torch.tensor(norm_w, device=device),
                log_var.float(),
                norm_mean.float(),
                log_edge.float(),
            ]
        )

        features = feature_values.view(1, -1)

        return features

    def _process_patch_to_fixed_size(self, patch, h, w):
        """将patch处理为固定尺寸，支持任意输入尺寸"""
        import torch.nn.functional as F

        min_h, min_w = self.min_patch_size

        if h == min_h and w == min_w:
            return patch

        # 先padding到至少目标大小，避免整除导致的尺寸丢失
        if h < min_h or w < min_w:
            pad_h_total = max(0, min_h - h)
            pad_w_total = max(0, min_w - w)

            pad_top = pad_h_total // 2
            pad_bottom = pad_h_total - pad_top
            pad_left = pad_w_total // 2
            pad_right = pad_w_total - pad_left

            patch = F.pad(patch, (pad_left, pad_right, pad_top, pad_bottom))
            h = patch.shape[-2]
            w = patch.shape[-1]

        if h == min_h and w == min_w:
            return patch

        if h > min_h or w > min_w:
            start_h = max(0, (h - min_h) // 2)
            end_h = start_h + min_h
            start_w = max(0, (w - min_w) // 2)
            end_w = start_w + min_w
            patch = patch[:, start_h:end_h, start_w:end_w]

        # 再次确保尺寸一致（整数除法可能导致偏差）
        if patch.shape[-2:] != (min_h, min_w):
            patch = F.interpolate(
                patch.unsqueeze(0),
                size=(min_h, min_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return patch

    def _intelligent_quadrant_split(self, patch, h, w):
        """智能的四分法分割，确保每个子patch都是整数尺寸"""
        mid_h = self._calculate_split_index(h, self.min_patch_size[0], w)
        mid_w = self._calculate_split_index(w, self.min_patch_size[1], h)

        mid_h = max(1, min(h - 1, mid_h))
        mid_w = max(1, min(w - 1, mid_w))

        sub_patches = [
            patch[:, :mid_h, :mid_w],
            patch[:, :mid_h, mid_w:],
            patch[:, mid_h:, :mid_w],
            patch[:, mid_h:, mid_w:],
        ]

        return sub_patches

    def _calculate_split_index(self, length, min_size, secondary_length):
        if length <= 1:
            return 1

        if length <= min_size:
            return length - 1 if length > 1 else 1

        if length <= min_size * 2:
            candidate = length // 2
        else:
            aspect_ratio = length / max(secondary_length, 1)
            if aspect_ratio > 1.5:
                ratio = 0.5 + min(0.2, 0.1 * (aspect_ratio - 1.5))
            elif aspect_ratio < 0.67:
                ratio = 0.5 - min(0.2, 0.1 * (0.67 - aspect_ratio) / max(aspect_ratio, 1e-3))
            else:
                ratio = 0.5

            candidate = int(round(length * ratio))

        lower_bound = max(1, min_size)
        upper_bound = max(lower_bound, length - lower_bound)
        candidate = max(lower_bound, min(candidate, upper_bound))
        return candidate

    # 注意: Hilbert 曲线相关方法已迁移至 hilbert.py 模块
    # 使用 from .hilbert import HilbertCurve, get_quadrant_order

    def default_should_split(self, patch, level):
        """默认分割策略：只看层数和patch大小"""
        _channels, height, width = patch.shape
        min_h, min_w = self.min_patch_size
        return level < self.max_level and (height > min_h or width > min_w)

    # ==================== 批处理优化方法 ====================
    
    def fractal_partition_batched(
        self,
        patch: torch.Tensor,
        max_info_len: int,
        depth_limit: int,
    ) -> Tuple[List[torch.Tensor], List[List[int]]]:
        """
        批处理版本的分形分割，使用 BFS 实现层级批处理。
        
        核心优化：对同一层级的所有 patches 批量计算 CNN 特征和分割决策，
        而非逐个递归计算。保持 Hilbert 曲线遍历顺序不变。
        
        Args:
            patch: 输入图像 [C, H, W]
            max_info_len: levels_info 的最大长度
            depth_limit: 最大递归深度
            
        Returns:
            (tokens, levels): 与原 fractal_partition 相同的输出格式
        """
        device = patch.device
        min_h, min_w = self.min_patch_size
        
        # 初始化 BFS 队列
        # dfs_order 用于最终排序，初始为 0
        queue: List[PatchInfo] = [PatchInfo(patch=patch, level=0, coord=[], dfs_order=0.0)]
        
        # 最终输出的 token 列表（需要按 dfs_order 排序）
        final_tokens: List[Tuple[float, torch.Tensor, List[int]]] = []
        
        while queue:
            # 按层级分组，批量处理同一层的 patches
            current_level = queue[0].level
            
            # 收集当前层的所有 patches
            current_batch: List[PatchInfo] = []
            remaining: List[PatchInfo] = []
            
            for info in queue:
                if info.level == current_level:
                    current_batch.append(info)
                else:
                    remaining.append(info)
            
            queue = remaining
            
            if not current_batch:
                continue
            
            # 批量决策：哪些 patches 应该分割
            split_decisions = self._batch_decide_splits(current_batch, depth_limit)
            
            # 处理每个 patch
            for i, info in enumerate(current_batch):
                should_split = split_decisions[i]
                C, H, W = info.patch.shape
                
                can_split_h = H > min_h
                can_split_w = W > min_w
                can_split = can_split_h or can_split_w
                
                # 如果不能分割或决策为停止，输出为 token
                if not can_split or not should_split:
                    processed = self._process_patch_to_fixed_size(info.patch, H, W)
                    token = processed.reshape(-1)
                    
                    # 构建 levels_info
                    depth = info.level
                    path = info.coord.copy()
                    padded = [depth] + path + [0] * (max_info_len - 1 - len(path))
                    levels = padded[:max_info_len]
                    
                    final_tokens.append((info.dfs_order, token, levels))
                else:
                    # 分割并加入队列
                    sub_patches = self._adaptive_split(info.patch, H, W, can_split_h, can_split_w)
                    
                    if not sub_patches:
                        # 分割失败，作为 token 输出
                        processed = self._process_patch_to_fixed_size(info.patch, H, W)
                        token = processed.reshape(-1)
                        depth = info.level
                        path = info.coord.copy()
                        padded = [depth] + path + [0] * (max_info_len - 1 - len(path))
                        levels = padded[:max_info_len]
                        final_tokens.append((info.dfs_order, token, levels))
                    else:
                        # 获取 Hilbert 遍历顺序
                        traversal_order = self._determine_traversal_order(
                            info.level, H, W, len(sub_patches)
                        )
                        
                        # 计算子 patch 的 DFS 顺序
                        # 使用分数索引保持正确的 DFS 顺序
                        num_children = len(sub_patches)
                        for child_idx, patch_idx in enumerate(traversal_order):
                            if patch_idx < len(sub_patches):
                                sub = sub_patches[patch_idx].contiguous()
                                if sub.numel() > 0:
                                    # DFS 顺序：父节点顺序 + 子节点在遍历中的位置
                                    child_dfs_order = info.dfs_order + child_idx / (num_children * (10 ** (info.level + 1)))
                                    new_info = PatchInfo(
                                        patch=sub,
                                        level=info.level + 1,
                                        coord=info.coord + [patch_idx],
                                        dfs_order=child_dfs_order,
                                    )
                                    queue.append(new_info)
        
        # 按 DFS 顺序排序
        final_tokens.sort(key=lambda x: x[0])
        
        # 提取排序后的 tokens 和 levels
        tokens = [t[1] for t in final_tokens]
        levels = [t[2] for t in final_tokens]
        
        return tokens, levels
    
    def _batch_decide_splits(
        self,
        batch: List[PatchInfo],
        depth_limit: int,
    ) -> List[bool]:
        """
        批量决定一组 patches 是否应该分割。
        
        对于 learnable_split=True，批量计算 CNN 特征和分割网络输出。
        对于非学习模式，使用方差阈值。
        
        Args:
            batch: 同一层级的 PatchInfo 列表
            depth_limit: 最大深度限制
            
        Returns:
            布尔列表，True 表示应该分割
        """
        decisions: List[bool] = []
        device = batch[0].patch.device if batch else None
        min_h, min_w = self.min_patch_size
        
        # 收集可分割的 patches 用于批量处理
        splittable_indices: List[int] = []
        splittable_patches: List[PatchInfo] = []
        
        for i, info in enumerate(batch):
            C, H, W = info.patch.shape
            can_split_h = H > min_h
            can_split_w = W > min_w
            can_split = can_split_h or can_split_w
            level_limit_reached = info.level >= depth_limit
            
            # 安全检查
            extra_depth_cap = depth_limit + 5
            safety_check = (H <= 1 and W <= 1) or (info.level >= extra_depth_cap)
            
            if not can_split or level_limit_reached or safety_check:
                decisions.append(False)
            else:
                splittable_indices.append(i)
                splittable_patches.append(info)
                decisions.append(True)  # 暂时标记为 True，后面更新
        
        if not splittable_patches:
            return decisions
        
        # 批量决策
        if self.learnable_split and self.split_decision is not None and self.cnn_encoder is not None:
            # 批量计算 CNN 特征
            batch_decisions = self._batch_learnable_decision(splittable_patches)
            
            # 更新决策
            for idx, decision in zip(splittable_indices, batch_decisions):
                decisions[idx] = decision
        else:
            # 非学习模式：使用方差阈值
            for idx, info in zip(splittable_indices, splittable_patches):
                if self.adaptive_threshold is not None and self.adaptive_threshold > 0:
                    patch_var = torch.var(info.patch)
                    should_stop = patch_var < self.adaptive_threshold
                    decisions[idx] = not should_stop
                else:
                    # 默认继续分割，但有层级限制
                    decisions[idx] = info.level < 10
        
        return decisions
    
    def _batch_learnable_decision(
        self,
        patches: List[PatchInfo],
    ) -> List[bool]:
        """
        使用学习网络批量决定分割。
        
        批处理优化：将多个 patches 的特征提取合并为批量操作。
        
        Args:
            patches: 待决策的 PatchInfo 列表
            
        Returns:
            布尔列表，True 表示应该分割
        """
        if not patches:
            return []
        
        device = patches[0].patch.device
        batch_size = len(patches)
        
        # 1. 批量提取手工特征 (这部分仍需逐个计算，因为尺寸可能不同)
        features_list = []
        for info in patches:
            C, H, W = info.patch.shape
            features = self._extract_enhanced_patch_features(info.patch, info.level, H, W)
            features_list.append(features)
        
        # Stack features: [batch_size, 6]
        batch_features = torch.cat(features_list, dim=0)  # [batch_size, 6]
        
        # 2. 批量计算 CNN 特征
        # 由于 patches 尺寸可能不同，需要逐个处理或 padding
        # 这里采用逐个处理但保持批量决策
        cnn_features_list = []
        for info in patches:
            cnn_feat = self.cnn_encoder(info.patch)  # [1, 32]
            cnn_features_list.append(cnn_feat)
        
        batch_cnn_features = torch.cat(cnn_features_list, dim=0)  # [batch_size, 32]
        
        # 3. 组合特征
        combined = torch.cat([batch_features, batch_cnn_features], dim=1)  # [batch_size, 38]
        
        # 检查 NaN/Inf
        if torch.isnan(combined).any() or torch.isinf(combined).any():
            combined = torch.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 4. 批量获取 logits
        logits = self.split_decision(combined)  # [batch_size, 2]
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        
        # 检查 NaN/Inf
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # 5. 决策
        decisions: List[bool] = []
        
        if self.training:
            # REINFORCE: 采样动作
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()  # [batch_size]
            
            # 保存 log_probs 和 entropies (批量)
            log_probs = dist.log_prob(actions)  # [batch_size]
            entropies = dist.entropy()  # [batch_size]
            
            for i in range(batch_size):
                self.saved_log_probs.append(log_probs[i])
                self.saved_entropies.append(entropies[i])
                # action 0: stop (not split), action 1: split
                decisions.append(actions[i].item() == 1)
        else:
            # 推理模式：取最大概率
            actions = torch.argmax(logits, dim=-1)  # [batch_size]
            for i in range(batch_size):
                decisions.append(actions[i].item() == 1)
        
        return decisions
