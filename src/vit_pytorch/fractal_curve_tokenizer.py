import math

import torch
import torch.nn as nn

from .tokenization import BaseTokenizer, TokenSequence, TokenizerOutput


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
    def __init__(self, min_patch_size=(1, 1), max_level=None, learnable_split=True, adaptive_threshold=0.5):
        """
        min_patch_size: 最小整数patch尺寸 (min_h, min_w) - 默认到像素级别
        max_level: 最大递归层数 (None表示无限制，只受min_patch_size限制)
        learnable_split: 是否使用可学习的分割决策
        adaptive_threshold: 自适应分割阈值
        """
        super().__init__()
        self.min_patch_size = min_patch_size
        self.max_level = max_level  # 可以为None，表示无限制
        self.learnable_split = learnable_split
        self.adaptive_threshold = adaptive_threshold

        if learnable_split:
            # 增强的分割决策网络，支持更多特征
            self.cnn_encoder = MiniCNN(in_channels=3, hidden_dim=16, out_dim=32)
            self.split_decision = LearnableSplitDecision(patch_features=6, cnn_features=32, hidden_dim=128)
        else:
            self.cnn_encoder = None
            self.split_decision = None

        # 用于存储REINFORCE所需的log_probs
        self.saved_log_probs = []
        self.saved_entropies = []

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

        for b in range(batch_size):
            image = images[b]
            sample_device = image.device

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
            return self.get_enhanced_hilbert_order(level, h, w)

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

        # 边缘密度特征（使用Sobel算子）
        gray_patch = torch.mean(patch, dim=0, keepdim=True).unsqueeze(0)  # [1, 1, H, W]

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

        if h >= 3 and w >= 3:  # 确保patch足够大来应用卷积
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

    def recursive_hilbert_order(self, level, h, w, orientation="up"):
        """
        递归生成真正的Hilbert曲线遍历顺序

        Args:
            level: 当前递归层级
            h, w: 当前patch的高度和宽度
            orientation: 曲线方向 ('up', 'right', 'down', 'left')

        Returns:
            List[int]: 四个象限的遍历顺序 [0,1,2,3] -> [左上,右上,左下,右下]
        """
        # 基础情况：最小分割单元或达到层级限制
        max_level_check = (self.max_level is not None) and (level >= self.max_level)
        min_size_check = (h <= 2 and w <= 2)

        if max_level_check or min_size_check:
            return self._get_base_hilbert_order(orientation)

        # 计算宽高比，用于调整Hilbert曲线的形状
        aspect_ratio = w / h if h > 0 else 1.0

        # 根据递归层级和方向生成Hilbert顺序
        return self._generate_hilbert_sequence(level, aspect_ratio, orientation)

    def _get_base_hilbert_order(self, orientation):
        """
        根据方向返回基础的Hilbert曲线顺序
        标准Hilbert曲线在2x2网格中的四种基本形态
        """
        base_orders = {
            "up": [2, 0, 1, 3],  # 左下→左上→右上→右下 (向上开口)
            "right": [0, 2, 3, 1],  # 左上→左下→右下→右上 (向右开口)
            "down": [1, 3, 2, 0],  # 右上→右下→左下→左上 (向下开口)
            "left": [3, 1, 0, 2],  # 右下→右上→左上→左下 (向左开口)
        }
        return base_orders.get(orientation, base_orders["up"])

    def _generate_hilbert_sequence(self, level, aspect_ratio, orientation):
        """
        基于递归层级和宽高比生成Hilbert序列
        实现真正的分形递归性质
        """
        # 根据Hilbert曲线的递归性质，每个象限内部的方向会发生变化
        orientation_map = {
            "up": ["left", "up", "up", "right"],  # 四个象限的子方向
            "right": ["up", "right", "right", "down"],
            "down": ["right", "down", "down", "left"],
            "left": ["down", "left", "left", "up"],
        }

        # 获取当前方向对应的基础顺序
        base_order = self._get_base_hilbert_order(orientation)

        # 根据宽高比调整：非正方形区域的特殊处理
        if aspect_ratio > 1.6:  # 宽矩形
            # 对于宽矩形，优化水平遍历
            return self._adjust_for_wide_rectangle(base_order, level)
        if aspect_ratio < 0.625:  # 高矩形
            # 对于高矩形，优化垂直遍历
            return self._adjust_for_tall_rectangle(base_order, level)
        # 接近正方形，使用标准Hilbert顺序
        return base_order

    def _adjust_for_wide_rectangle(self, base_order, level):
        """为宽矩形调整Hilbert顺序，优化水平连续性"""
        # 宽矩形时，优先保持水平方向的连续性
        if level % 2 == 0:
            # 偶数层：左→右优先
            return [2, 0, 1, 3]  # 左下→左上→右上→右下
        # 奇数层：保持Hilbert性质但调整顺序
        return [0, 2, 3, 1]  # 左上→左下→右下→右上

    def _adjust_for_tall_rectangle(self, base_order, level):
        """为高矩形调整Hilbert顺序，优化垂直连续性"""
        # 高矩形时，优先保持垂直方向的连续性
        if level % 2 == 0:
            # 偶数层：上→下优先
            return [0, 1, 3, 2]  # 左上→右上→右下→左下
        # 奇数层：保持Hilbert性质但调整顺序
        return [2, 3, 1, 0]  # 左下→右下→右上→左上

    def generate_true_hilbert_curve(self, n=2):
        """
        生成真正的n阶Hilbert曲线坐标序列
        返回坐标点列表，可用于验证算法正确性

        Args:
            n: Hilbert曲线的阶数 (2^n x 2^n 网格)

        Returns:
            List[Tuple[int, int]]: 按Hilbert曲线顺序排列的坐标点
        """
        if n == 0:
            return [(0, 0)]

        # 递归生成前一阶的曲线
        prev_points = self.generate_true_hilbert_curve(n - 1)

        # 四个象限的变换
        # 左下象限：旋转90度逆时针，然后翻转x坐标
        q1 = [(y, x) for x, y in prev_points]

        # 左上象限：直接复制并上移
        q2 = [(x, y + 2 ** (n - 1)) for x, y in prev_points]

        # 右上象限：直接复制并右上移
        q3 = [(x + 2 ** (n - 1), y + 2 ** (n - 1)) for x, y in prev_points]

        # 右下象限：旋转90度顺时针，翻转y坐标，然后右移
        q4 = [(2 ** (n - 1) - 1 - y + 2 ** (n - 1), 2 ** (n - 1) - 1 - x) for x, y in prev_points]

        # 按Hilbert顺序连接四个象限
        return q1 + q2 + q3 + q4

    def hilbert_distance(self, x, y, n):
        """
        计算点(x,y)在n阶Hilbert曲线上的距离
        用于将2D坐标映射为1D距离

        Args:
            x, y: 2D坐标
            n: Hilbert曲线阶数

        Returns:
            int: 在Hilbert曲线上的距离
        """
        d = 0
        s = n // 2

        while s > 0:
            rx = 1 if x & s else 0
            ry = 1 if y & s else 0
            d += s * s * ((3 * rx) ^ ry)

            # 旋转坐标
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x

            s //= 2

        return d

    def adaptive_hilbert_mapping(self, patch_indices, h, w):
        """
        为任意尺寸的patch网格生成自适应Hilbert映射

        Args:
            patch_indices: 要排序的patch索引列表 [0,1,2,3]
            h, w: patch网格的高度和宽度

        Returns:
            List[int]: 按Hilbert顺序重排的索引
        """
        if len(patch_indices) != 4:
            return patch_indices

        # 将4个象限映射到坐标
        coords = [
            (0, 1),  # 左上 (0)
            (1, 1),  # 右上 (1)
            (0, 0),  # 左下 (2)
            (1, 0),  # 右下 (3)
        ]

        # 根据实际尺寸调整坐标缩放
        scaled_coords = []
        for i, (x, y) in enumerate(coords):
            # 将单位坐标映射到实际patch尺寸
            actual_x = int(x * (w - 1)) if w > 1 else x
            actual_y = int(y * (h - 1)) if h > 1 else y
            scaled_coords.append((actual_x, actual_y, i))

        # 计算每个坐标在Hilbert曲线上的距离
        max_dim = max(h, w)
        # 找到最小的2的幂次方大于等于max_dim
        n = 1
        while (1 << n) < max_dim:
            n += 1
        curve_size = 1 << n

        # 计算Hilbert距离并排序
        hilbert_distances = []
        for x, y, idx in scaled_coords:
            # 将坐标归一化到curve_size范围内
            norm_x = int(x * curve_size / max_dim) if max_dim > 0 else 0
            norm_y = int(y * curve_size / max_dim) if max_dim > 0 else 0
            dist = self.hilbert_distance(norm_x, norm_y, curve_size)
            hilbert_distances.append((dist, idx))

        # 按Hilbert距离排序
        hilbert_distances.sort()

        return [idx for _, idx in hilbert_distances]

    def get_enhanced_hilbert_order(self, level, h, w):
        """
        增强版Hilbert顺序生成器
        结合递归算法和自适应映射
        """
        # 基础patch索引
        patch_indices = [0, 1, 2, 3]

        aspect_ratio = w / h if h > 0 else 1.0

        # 使用自适应映射计算基础顺序
        adaptive_order = self.adaptive_hilbert_mapping(patch_indices, h, w)

        # 浅层保持自适应顺序，快速对齐边界情况
        if level <= 1:
            return adaptive_order

        orientation = self._get_orientation_for_level(level)
        base_order = self._get_base_hilbert_order(orientation)

        if aspect_ratio > 1.6:
            return self._adjust_for_wide_rectangle(base_order, level)
        if aspect_ratio < 0.625:
            return self._adjust_for_tall_rectangle(base_order, level)
        if abs(aspect_ratio - 1.0) < 0.3:
            return base_order

        return adaptive_order

    def _get_orientation_for_level(self, level):
        """根据递归层级确定Hilbert曲线方向"""
        orientations = ["up", "right", "down", "left"]
        return orientations[level % 4]

    def default_should_split(self, patch, level):
        """默认分割策略：只看层数和patch大小"""
        _channels, height, width = patch.shape
        min_h, min_w = self.min_patch_size
        return level < self.max_level and (height > min_h or width > min_w)
