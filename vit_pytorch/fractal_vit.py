import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

from .fractal_curve_tokenizer import FractalHilbertTokenizer

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes

class FractalPositionEmbedding(nn.Module):
    """分形位置编码，处理可变长度序列和层级信息"""
    
    def __init__(self, dim, max_level=5, max_seq_len=10000):
        super().__init__()
        self.dim = dim
        self.max_level = max_level
        self.max_seq_len = max_seq_len
        
        # 深度嵌入 - 为每个分形层级学习不同的表示
        self.depth_embedding = nn.Embedding(max_level + 1, dim)
        
        # 路径嵌入 - 为分形路径中的每个位置学习表示
        self.path_embedding = nn.Embedding(max_seq_len, dim)
        
        # 空间位置嵌入 - 基于原始图像中的空间位置
        self.spatial_embedding = nn.Parameter(torch.randn(1, max_seq_len, dim))
        
        # 层级注意力权重 - 不同层级的token可能需要不同的注意力模式
        self.level_attention_weights = nn.Parameter(torch.ones(max_level + 1))
        
        # 自适应缩放因子
        self.adaptive_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, level_info, seq_len):
        """
        Args:
            level_info: [N, max_level+1] 层级信息矩阵，第一列是depth，其余是path
            seq_len: 序列长度
            
        Returns:
            position_emb: [N, dim] 位置编码
        """
        if level_info.numel() == 0 or seq_len == 0:
            return torch.zeros(0, self.dim, device=level_info.device)
            
        device = level_info.device
        N = level_info.shape[0]
        
        # 提取深度和路径信息
        depths = level_info[:, 0]  # [N]
        paths = level_info[:, 1:]  # [N, max_level]
        
        # 深度嵌入
        depth_emb = self.depth_embedding(depths.clamp(0, self.max_level))  # [N, dim]
        
        # 路径嵌入：使用第一个有效路径索引
        path_indices = torch.clamp(paths[:, 0], 0, self.max_seq_len - 1)
        path_emb = self.path_embedding(path_indices)  # [N, dim]
        
        # 空间位置嵌入
        if seq_len <= self.max_seq_len:
            spatial_emb = self.spatial_embedding[:, :N, :]  # [1, N, dim]
        else:
            # 如果序列太长，使用插值
            spatial_emb = F.interpolate(
                self.spatial_embedding.transpose(1, 2), 
                size=N, mode='linear', align_corners=False
            ).transpose(1, 2)  # [1, N, dim]
        
        spatial_emb = spatial_emb.squeeze(0)  # [N, dim]
        
        # 层级自适应权重
        level_weights = self.level_attention_weights[depths.clamp(0, self.max_level)]  # [N]
        level_weights = level_weights.unsqueeze(-1)  # [N, 1]
        
        # 组合所有嵌入
        position_emb = (depth_emb + path_emb + spatial_emb) * level_weights * self.adaptive_scale
        
        return position_emb


class FractalTokenProcessor(nn.Module):
    """处理分形token的预处理器"""
    
    def __init__(self, input_dim, output_dim, min_patch_size=(16, 16)):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.min_patch_size = min_patch_size
        
        # token标准化和投影
        self.token_norm = nn.LayerNorm(input_dim)
        self.token_projection = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 可学习的token类型嵌入（用于区分不同层级的token）
        self.token_type_embedding = nn.Embedding(10, output_dim)  # 支持最多10个层级
        
    def forward(self, tokens_list, levels_list):
        """
        Args:
            tokens_list: List[Tensor] 每个图像的token列表
            levels_list: List[Tensor] 每个图像的层级信息列表
            
        Returns:
            processed_tokens: List[Tensor] 处理后的token
            enhanced_levels: List[Tensor] 增强的层级信息
        """
        processed_tokens = []
        enhanced_levels = []
        
        for tokens, levels in zip(tokens_list, levels_list):
            if tokens.numel() == 0:
                processed_tokens.append(torch.empty(0, self.output_dim, device=tokens.device))
                enhanced_levels.append(torch.empty(0, levels.shape[-1], device=levels.device, dtype=levels.dtype))
                continue
                
            # 标准化和投影token
            norm_tokens = self.token_norm(tokens)  # [N, input_dim]
            proj_tokens = self.token_projection(norm_tokens)  # [N, output_dim]
            
            # 添加token类型嵌入
            if levels.numel() > 0:
                depths = levels[:, 0].clamp(0, 9)  # 限制在0-9范围内
                type_emb = self.token_type_embedding(depths)  # [N, output_dim]
                proj_tokens = proj_tokens + type_emb
            
            processed_tokens.append(proj_tokens)
            enhanced_levels.append(levels)
            
        return processed_tokens, enhanced_levels


class MultiScaleAttention(nn.Module):
    """多尺度注意力机制，适应分形结构"""
    
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., scale_factor=1.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = (dim_head * scale_factor) ** -0.5
        inner_dim = dim_head * heads
        
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        # 多尺度注意力权重
        self.scale_weights = nn.Parameter(torch.ones(heads))
        
        # 层级相关的注意力偏置
        self.level_bias = nn.Parameter(torch.zeros(heads, 1, 1))
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, level_mask=None):
        """
        Args:
            x: [B, N, dim] token序列
            level_mask: [B, N] 层级掩码，用于区分不同层级的token
        """
        B, N, _ = x.shape
        
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # 计算注意力分数
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # 应用尺度权重
        scale_weights = self.scale_weights.view(1, -1, 1, 1)
        dots = dots * scale_weights
        
        # 应用层级偏置
        if exists(level_mask):
            level_bias = self.level_bias.expand(-1, N, N)
            dots = dots + level_bias
        
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
class AdaptiveFeedForward(nn.Module):
    """自适应前馈网络，根据token的层级信息调整处理"""
    
    def __init__(self, dim, hidden_dim, dropout=0., max_level=5):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.max_level = max_level
        
        self.norm = nn.LayerNorm(dim)
        
        # 主要的前馈网络
        self.main_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        # 层级自适应门控机制
        self.level_gate = nn.Sequential(
            nn.Linear(dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 层级特定的调制器
        self.level_modulators = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(max_level + 1)
        ])
        
    def forward(self, x, level_info=None):
        """
        Args:
            x: [B, N, dim] 输入token
            level_info: [B, N] 每个token的层级信息
        """
        x = self.norm(x)
        
        # 主要前馈网络
        main_out = self.main_net(x)
        
        # 如果有层级信息，应用自适应调制
        if exists(level_info):
            # 计算门控权重
            gate_weights = self.level_gate(x)  # [B, N, 1]
            
            # 应用层级特定的调制
            modulated_out = torch.zeros_like(main_out)
            for level in range(self.max_level + 1):
                level_mask = (level_info == level)
                if level_mask.any():
                    level_tokens = x[level_mask]
                    if level_tokens.numel() > 0:
                        modulated_tokens = self.level_modulators[level](level_tokens)
                        modulated_out[level_mask] = modulated_tokens
            
            # 组合主输出和调制输出
            output = main_out * gate_weights + modulated_out * (1 - gate_weights)
        else:
            output = main_out
            
        return output


class FractalTransformerBlock(nn.Module):
    """增强的Transformer块，专为分形token设计"""
    
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0., max_level=5):
        super().__init__()
        self.attention = MultiScaleAttention(dim, heads, dim_head, dropout)
        self.ff = AdaptiveFeedForward(dim, mlp_dim, dropout, max_level)
        
        # 残差连接的自适应权重
        self.residual_weight = nn.Parameter(torch.ones(2))
        
    def forward(self, x, level_info=None, level_mask=None):
        # 多尺度注意力
        attn_out = self.attention(x, level_mask)
        x = x + attn_out * self.residual_weight[0]
        
        # 自适应前馈
        ff_out = self.ff(x, level_info)
        x = x + ff_out * self.residual_weight[1]
        
        return x


class FractalTransformer(nn.Module):
    """分形Transformer，处理可变长度和多层级的token序列"""
    
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., max_level=5):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.max_level = max_level
        
        self.norm = nn.LayerNorm(dim)
        
        # 构建transformer层
        self.layers = nn.ModuleList([
            FractalTransformerBlock(dim, heads, dim_head, mlp_dim, dropout, max_level)
            for _ in range(depth)
        ])
        
        # 全局上下文聚合器
        self.global_context = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        
    def forward(self, x, level_info=None):
        """
        Args:
            x: [B, N, dim] token序列
            level_info: [B, N] 层级信息
        """
        B, N, _ = x.shape
        
        # 创建层级掩码
        level_mask = None
        if exists(level_info) and level_info is not None:
            level_mask = level_info.unsqueeze(1) == level_info.unsqueeze(2)  # [B, N, N]
        
        # 通过transformer层
        for layer in self.layers:
            x = layer(x, level_info, level_mask)
        
        # 全局上下文聚合
        if N > 1:
            global_out, _ = self.global_context(x, x, x)
            x = x + global_out * 0.1  # 轻微的全局调整
        
        return self.norm(x)

# 增强版本 - 更好地对齐tokenizer特性
class EnhancedFractalViT(nn.Module):
    """
    增强的分形ViT，完全对齐改进后的tokenizer特性:
    - 可学习的分割决策
    - 真正的Hilbert曲线递归
    - 自适应多尺度处理
    - 层级感知的注意力机制
    """
    
    def __init__(
        self, 
        *, 
        image_size, 
        num_classes, 
        dim=512, 
        depth=6, 
        heads=8, 
        mlp_dim=1024, 
        pool='cls', 
        channels=3, 
        dim_head=64, 
        dropout=0., 
        emb_dropout=0.,
        min_patch_size=(16, 16), 
        max_level=4,
        learnable_split=True
    ):
        super().__init__()
        
        self.image_size = pair(image_size)
        self.num_classes = num_classes
        self.dim = dim
        self.pool = pool
        
        # 增强的分形tokenizer
        self.fractal_tokenizer = FractalHilbertTokenizer(
            min_patch_size=min_patch_size,
            max_level=max_level,
            learnable_split=learnable_split
        )
        
        # 计算token维度
        patch_dim = channels * min_patch_size[0] * min_patch_size[1]
        
        # 分形token处理器
        self.token_processor = FractalTokenProcessor(
            input_dim=patch_dim,
            output_dim=dim,
            min_patch_size=min_patch_size
        )
        
        # 分形位置编码
        self.pos_embedding = FractalPositionEmbedding(
            dim=dim,
            max_level=max_level,
            max_seq_len=10000
        )
        
        # CLS token和dropout
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # 增强的Transformer
        self.transformer = FractalTransformer(
            dim=dim, 
            depth=depth, 
            heads=heads, 
            dim_head=dim_head, 
            mlp_dim=mlp_dim, 
            dropout=dropout,
            max_level=max_level
        )
        
        # 分类头
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, num_classes)
        )
        
        # 自适应权重，用于平衡不同层级token的贡献
        self.level_weights = nn.Parameter(torch.ones(max_level + 1))
        
        # 可选的辅助损失权重
        self.aux_loss_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, img, return_attention=False, return_aux_info=False):
        """
        Args:
            img: [B, C, H, W] 输入图像
            return_attention: 是否返回注意力权重
            return_aux_info: 是否返回辅助信息（用于分析）
            
        Returns:
            logits: [B, num_classes] 分类logits
            aux_info: dict 辅助信息（可选）
        """
        batch_size = img.shape[0]
        device = img.device
        
        # 分形tokenization
        tokens_list, levels_list = self.fractal_tokenizer(img)
        
        # 处理每个batch中的图像
        batch_outputs = []
        aux_infos = []
        
        for b in range(batch_size):
            tokens = tokens_list[b]
            levels = levels_list[b]
            
            # 处理空token的情况
            if tokens.numel() == 0 or tokens.shape[0] == 0:
                batch_outputs.append(torch.zeros(1, self.num_classes, device=device))
                if return_aux_info:
                    aux_infos.append({'num_tokens': 0, 'levels_used': []})
                continue
            
            # token处理和投影
            processed_tokens, enhanced_levels = self.token_processor([tokens], [levels])
            x = processed_tokens[0].unsqueeze(0)  # [1, N, dim]
            level_info = enhanced_levels[0]  # [N, max_level+1]
            
            # 提取深度信息用于后续处理
            depths = None
            if level_info.numel() > 0:
                depths = level_info[:, 0]  # [N]
                level_info_1d = depths.unsqueeze(0)  # [1, N]
            else:
                level_info_1d = None
            
            # 分形位置编码
            if level_info.numel() > 0:
                pos_emb = self.pos_embedding(level_info, x.shape[1])  # [N, dim]
                x = x + pos_emb.unsqueeze(0)  # [1, N, dim] + [1, N, dim]
            
            # 添加CLS token
            cls_tokens = self.cls_token.expand(1, -1, -1)  # [1, 1, dim]
            x = torch.cat((cls_tokens, x), dim=1)  # [1, 1+N, dim]
            
            # 更新level_info以包含CLS token
            if level_info_1d is not None:
                cls_level = torch.zeros(1, 1, device=device, dtype=level_info_1d.dtype)
                level_info_1d = torch.cat([cls_level, level_info_1d], dim=1)  # [1, 1+N]
            
            x = self.dropout(x)
            
            # 分形Transformer处理
            x = self.transformer(x, level_info_1d)  # [1, 1+N, dim]
            
            # 池化和分类
            if self.pool == 'mean':
                # 加权平均，根据层级权重
                if level_info_1d is not None and level_info_1d.shape[1] > 1:
                    token_weights = self.level_weights[level_info_1d[0, 1:].clamp(0, len(self.level_weights)-1)]
                    token_weights = F.softmax(token_weights, dim=0)
                    pooled = torch.sum(x[0, 1:] * token_weights.unsqueeze(-1), dim=0)  # [dim]
                else:
                    pooled = x[:, 1:].mean(dim=1).squeeze(0)  # [dim]
            else:
                pooled = x[:, 0].squeeze(0)  # [dim] - CLS token
            
            pooled = self.to_latent(pooled)
            output = self.mlp_head(pooled.unsqueeze(0))  # [1, num_classes]
            
            batch_outputs.append(output)
            
            # 收集辅助信息
            if return_aux_info:
                if depths is not None:
                    aux_info = {
                        'num_tokens': tokens.shape[0],
                        'levels_used': depths.unique().tolist(),
                        'token_distribution': torch.bincount(depths, minlength=self.fractal_tokenizer.max_level + 1).float()
                    }
                else:
                    aux_info = {
                        'num_tokens': tokens.shape[0],
                        'levels_used': [],
                        'token_distribution': torch.zeros(self.fractal_tokenizer.max_level + 1)
                    }
                aux_infos.append(aux_info)
        
        # 合并batch结果
        final_output = torch.cat(batch_outputs, dim=0)  # [B, num_classes]
        
        if return_aux_info:
            return final_output, aux_infos
        else:
            return final_output
    
    def get_tokenizer_loss(self):
        """获取tokenizer的辅助损失（如果启用了可学习分割）"""
        if hasattr(self.fractal_tokenizer, 'split_decision') and self.fractal_tokenizer.split_decision is not None:
            # 这里可以添加分割决策的正则化损失
            # 例如，鼓励合理的分割概率分布
            return torch.tensor(0.0, requires_grad=True)
        return torch.tensor(0.0)


# 保持向后兼容的简化版本
class SimpleFractalViT(nn.Module):
    """简化版本，保持向后兼容性"""
    def __init__(self, *, image_size, num_classes, dim=512, depth=6, heads=8, mlp_dim=1024, 
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.,
                 min_patch_size=(16, 9), max_level=3):
        super().__init__()
        
        # 使用增强版本作为后端
        self.enhanced_model = EnhancedFractalViT(
            image_size=image_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            min_patch_size=min_patch_size,
            max_level=max_level,
            learnable_split=False  # 简化版不启用可学习分割
        )
        
        # 保持原有属性以兼容旧代码
        self.fractal_tokenizer = self.enhanced_model.fractal_tokenizer
        self.patch_to_embedding = self.enhanced_model.token_processor
        self.pos_embedding = self.enhanced_model.pos_embedding
        self.cls_token = self.enhanced_model.cls_token
        self.dropout = self.enhanced_model.dropout
        self.transformer = self.enhanced_model.transformer
        self.pool = pool
        self.to_latent = self.enhanced_model.to_latent
        self.mlp_head = self.enhanced_model.mlp_head

    def forward(self, img):
        return self.enhanced_model(img)
