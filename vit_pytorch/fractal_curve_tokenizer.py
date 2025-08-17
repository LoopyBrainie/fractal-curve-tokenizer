import torch
import torch.nn as nn

class LearnableSplitDecision(nn.Module):
    """增强的可学习分割决策网络"""
    def __init__(self, patch_features=6, hidden_dim=128):
        super().__init__()
        # patch_features: [level, height, width, variance, mean, edge_density] 等特征
        self.net = nn.Sequential(
            nn.Linear(patch_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1),
            nn.Sigmoid()  # 输出 0-1 的分割概率
        )
        
        # 初始化偏置，让初始分割倾向于继续分割
        with torch.no_grad():
            self.net[-2].bias.fill_(1.0)  # 偏向继续分割
        
    def forward(self, patch_features):
        """
        patch_features: [batch_size, patch_features]
        returns: [batch_size] 分割概率
        """
        return self.net(patch_features).squeeze(-1)

class FractalHilbertTokenizer(nn.Module):
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
            self.split_decision = LearnableSplitDecision(patch_features=6, hidden_dim=128)
        else:
            self.split_decision = None
        
        # 自适应的level权重，用于平衡不同层级的重要性
        self.level_weights = nn.Parameter(torch.ones(50))  # 支持更多层级

    def forward(self, image):
        # image: [B, C, H, W]
        B, C, H, W = image.shape
        # 返回按图片分开的 lists：tokens_per_image: List[Tensor(N_i, patch_dim)], levels_per_image: List[Tensor(N_i, max_info_len)]
        tokens_per_image = []
        levels_per_image = []
        
        # 动态确定最大信息长度（基于图像尺寸估算可能的最大层级）
        estimated_max_level = self._estimate_max_possible_level(H, W)
        max_info_len = estimated_max_level + 10  # 留有余量
        
        for b in range(B):
            tks, lvls = self.fractal_partition(image[b], level=0, coord=[], max_info_len=max_info_len)
            if len(tks) == 0:
                # 创建默认的最小patch
                min_patch_dim = C * self.min_patch_size[0] * self.min_patch_size[1]
                tokens_per_image.append(torch.empty(0, min_patch_dim))
                levels_per_image.append(torch.empty(0, max_info_len, dtype=torch.long))
                continue
            tokens = torch.stack(tks)  # [N_i, patch_dim]
            levels = torch.tensor(lvls, dtype=torch.long)
            tokens_per_image.append(tokens)
            levels_per_image.append(levels)
        return tokens_per_image, levels_per_image
    
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

    def fractal_partition(self, patch, level, coord, max_info_len):
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
        level_limit_reached = (self.max_level is not None) and (level >= self.max_level)
        
        # 基本停止条件
        basic_stop = not can_split or level_limit_reached
        
        if basic_stop:
            should_stop = True
        else:
            # 使用增强的可学习分割决策
            if self.learnable_split and self.split_decision is not None:
                # 计算增强的patch特征
                features = self._extract_enhanced_patch_features(patch, level, H, W)
                split_prob = self.split_decision(features).item()
                should_stop = split_prob < self.adaptive_threshold
            else:
                # 默认策略：尽可能深度分割
                should_stop = False
        
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

        # 智能四分法分割
        sub_patches = self._intelligent_quadrant_split(patch, H, W)
        
        # 获取当前层级对应的Hilbert遍历顺序
        hilbert_order = self.get_enhanced_hilbert_order(level, H, W)
        
        for i, patch_idx in enumerate(hilbert_order):
            if patch_idx < len(sub_patches):
                sub = sub_patches[patch_idx].contiguous()
                if sub.numel() > 0:
                    tks, lvls = self.fractal_partition(sub, level + 1, coord + [patch_idx], max_info_len)
                    tokens.extend(tks)
                    levels.extend(lvls)
        return tokens, levels

    def _extract_enhanced_patch_features(self, patch, level, h, w):
        """提取增强的patch特征用于分割决策"""
        import torch.nn.functional as F
        
        # 基础统计特征
        patch_var = torch.var(patch).item()
        patch_mean = torch.mean(patch).item()
        
        # 边缘密度特征（使用Sobel算子）
        gray_patch = torch.mean(patch, dim=0, keepdim=True).unsqueeze(0)  # [1, 1, H, W]
        
        # Sobel边缘检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        if h >= 3 and w >= 3:  # 确保patch足够大来应用卷积
            edge_x = F.conv2d(gray_patch, sobel_x, padding=1)
            edge_y = F.conv2d(gray_patch, sobel_y, padding=1)
            edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
            edge_density = torch.mean(edge_magnitude).item()
        else:
            edge_density = 0.0
        
        # 纹理复杂度（基于局部方差）
        if h >= 2 and w >= 2:
            local_patches = F.unfold(gray_patch, kernel_size=2, stride=1)  # 2x2局部patch
            local_vars = torch.var(local_patches, dim=1)
            texture_complexity = torch.mean(local_vars).item()
        else:
            texture_complexity = patch_var
        
        # 构建特征向量: [level, height, width, variance, mean, edge_density]
        features = torch.tensor([
            [level, h, w, patch_var, patch_mean, edge_density]
        ], dtype=torch.float32)
        
        return features
    
    def _process_patch_to_fixed_size(self, patch, h, w):
        """将patch处理为固定尺寸，支持任意输入尺寸"""
        import torch.nn.functional as F
        
        min_h, min_w = self.min_patch_size
        
        if h == min_h and w == min_w:
            # 已经是目标尺寸
            return patch
        elif h > min_h or w > min_w:
            # 需要裁剪：使用中心裁剪
            start_h = max(0, (h - min_h) // 2)
            start_w = max(0, (w - min_w) // 2)
            end_h = min(h, start_h + min_h)
            end_w = min(w, start_w + min_w)
            return patch[:, start_h:end_h, start_w:end_w]
        else:
            # 需要padding：使用零填充
            pad_h = max(0, min_h - h)
            pad_w = max(0, min_w - w)
            # 计算padding参数：(left, right, top, bottom)
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            return F.pad(patch, (pad_left, pad_right, pad_top, pad_bottom))
    
    def _intelligent_quadrant_split(self, patch, h, w):
        """智能的四分法分割，确保每个子patch都是整数尺寸"""
        # 计算更精确的分割点
        mid_h = h // 2
        mid_w = w // 2
        
        # 确保至少为1像素
        mid_h = max(1, mid_h)
        mid_w = max(1, mid_w)
        
        # 生成四个子patches
        sub_patches = []
        
        # 左上 (0)
        sub_patches.append(patch[:, :mid_h, :mid_w])
        
        # 右上 (1)
        sub_patches.append(patch[:, :mid_h, mid_w:])
        
        # 左下 (2)  
        sub_patches.append(patch[:, mid_h:, :mid_w])
        
        # 右下 (3)
        sub_patches.append(patch[:, mid_h:, mid_w:])
        
        return sub_patches
        """
        改进的Hilbert曲线递归算法
        基于真正的Hilbert曲线数学性质生成遍历顺序
        """
        return self.recursive_hilbert_order(level, h, w, orientation='up')
    
    def recursive_hilbert_order(self, level, h, w, orientation='up'):
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
            'up': [2, 0, 1, 3],      # 左下→左上→右上→右下 (向上开口)
            'right': [0, 2, 3, 1],   # 左上→左下→右下→右上 (向右开口)
            'down': [1, 3, 2, 0],    # 右上→右下→左下→左上 (向下开口)  
            'left': [3, 1, 0, 2]     # 右下→右上→左上→左下 (向左开口)
        }
        return base_orders.get(orientation, base_orders['up'])
    
    def _generate_hilbert_sequence(self, level, aspect_ratio, orientation):
        """
        基于递归层级和宽高比生成Hilbert序列
        实现真正的分形递归性质
        """
        # 根据Hilbert曲线的递归性质，每个象限内部的方向会发生变化
        orientation_map = {
            'up': ['left', 'up', 'up', 'right'],      # 四个象限的子方向
            'right': ['up', 'right', 'right', 'down'],
            'down': ['right', 'down', 'down', 'left'],
            'left': ['down', 'left', 'left', 'up']
        }
        
        # 获取当前方向对应的基础顺序
        base_order = self._get_base_hilbert_order(orientation)
        
        # 根据宽高比调整：非正方形区域的特殊处理
        if aspect_ratio > 1.6:  # 宽矩形
            # 对于宽矩形，优化水平遍历
            return self._adjust_for_wide_rectangle(base_order, level)
        elif aspect_ratio < 0.625:  # 高矩形  
            # 对于高矩形，优化垂直遍历
            return self._adjust_for_tall_rectangle(base_order, level)
        else:
            # 接近正方形，使用标准Hilbert顺序
            return base_order
    
    def _adjust_for_wide_rectangle(self, base_order, level):
        """为宽矩形调整Hilbert顺序，优化水平连续性"""
        # 宽矩形时，优先保持水平方向的连续性
        if level % 2 == 0:
            # 偶数层：左→右优先
            return [2, 0, 1, 3]  # 左下→左上→右上→右下
        else:
            # 奇数层：保持Hilbert性质但调整顺序
            return [0, 2, 3, 1]  # 左上→左下→右下→右上
    
    def _adjust_for_tall_rectangle(self, base_order, level):
        """为高矩形调整Hilbert顺序，优化垂直连续性"""
        # 高矩形时，优先保持垂直方向的连续性  
        if level % 2 == 0:
            # 偶数层：上→下优先
            return [0, 1, 3, 2]  # 左上→右上→右下→左下
        else:
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
        quadrants = []
        
        # 左下象限：旋转90度逆时针，然后翻转x坐标  
        q1 = [(y, x) for x, y in prev_points]
        
        # 左上象限：直接复制并上移
        q2 = [(x, y + 2**(n-1)) for x, y in prev_points]
        
        # 右上象限：直接复制并右上移
        q3 = [(x + 2**(n-1), y + 2**(n-1)) for x, y in prev_points]
        
        # 右下象限：旋转90度顺时针，翻转y坐标，然后右移
        q4 = [(2**(n-1) - 1 - y + 2**(n-1), 2**(n-1) - 1 - x) for x, y in prev_points]
        
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
            (1, 0)   # 右下 (3)
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
        
        # 使用自适应映射计算基础顺序
        adaptive_order = self.adaptive_hilbert_mapping(patch_indices, h, w)
        
        # 结合递归层级信息调整顺序
        if level <= 1:
            # 浅层使用自适应映射结果
            return adaptive_order
        else:
            # 深层结合递归方向信息
            orientation = self._get_orientation_for_level(level)
            base_order = self._get_base_hilbert_order(orientation)
            
            # 根据宽高比权衡两种方法
            aspect_ratio = w / h if h > 0 else 1.0
            if abs(aspect_ratio - 1.0) < 0.3:  # 接近正方形
                return base_order
            else:
                return adaptive_order
    
    def _get_orientation_for_level(self, level):
        """根据递归层级确定Hilbert曲线方向"""
        orientations = ['up', 'right', 'down', 'left']
        return orientations[level % 4]
    def default_should_split(self, patch, level):
        """默认分割策略：只看层数和patch大小"""
        C, H, W = patch.shape
        min_h, min_w = self.min_patch_size
        return level < self.max_level and (H > min_h or W > min_w)