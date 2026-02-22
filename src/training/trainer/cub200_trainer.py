# ============================================================================
# CUB-200-2011 细粒度分类专用训练器 (独立实现)
# Fine-grained Classification Trainer for CUB-200-2011
# ============================================================================
"""
CUB-200-2011 数据集的专用训练器 - 完全独立实现。

设计原则:
1. 完全解耦 - 不依赖 ModularTrainer 或其他通用训练组件
2. 完整训练循环 - 独立管理训练、验证、检查点、早停
3. 数学形式化 - 所有参数有严格数学推导

数据集特性:
    - 200 类鸟类物种（细粒度）
    - 5,994 训练样本 / 5,794 测试样本
    - 每类约 30 样本（样本稀缺需强正则化）
    - 类间差异微小（同属不同种）
    - 关键判别特征：喙、眼、羽毛纹理

数学形式化:
    Center Loss: L_center = (1/2m) Σ_i ||f_i - c_{y_i}||²
    中心梯度: ∂L/∂c_j = (1/m_j) Σ_{i:y_i=j}(c_j - f_i)
    学习率关系: lr_center = lr_main × ratio (论文推荐 ratio=50)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

from ..losses.finegrained import FinegrainedLoss, FinegrainedLossConfig
from ..config import ModelArchitectureConfig  # I36: 统一架构配置
from ..core.checkpoint import save_checkpoint_with_gene  # ModelGene 自包含 checkpoint
from vit_pytorch import FractalCurveViT  # I36: 模型创建
from vit_pytorch.core.constants import (
    SPLITTER_TEMP_END, TEMPERATURE_MIN,  # I113-10: 温度常量
    GRAD_CLIP_BASE_LR, GRAD_CLIP_BASE_NORM,  # I121-6: 动态梯度裁剪
    GRAD_CLIP_MIN_NORM, GRAD_CLIP_MAX_NORM,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CUB-200-2011 鸟类类别名称 (200 类)
# ============================================================================
# 官方类别列表: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
# 数据集包含 200 种北美鸟类，涵盖 5550 张训练图像和 5784 张测试图像

CUB200_BIRD_CLASSES = [
    "Black-footed Albatross",  # 1. 黑脚信天翁
    "Laysan Albatross",        # 2. 莱桑信天翁
    "Sooty Albatross",         # 3. 烟色信天翁
    "Groove-billed Ani",       # 4. 沟嘴犀鹃
    "Crested Auklet",          # 5. 冠海鸦
    "Least Auklet",            # 6. 小海鸦
    "Parakeet Auklet",         # 7. 鹦鹉海鸦
    "Rhinoceros Auklet",       # 8. 角海鸦
    "Brewer Blackbird",        # 9. 布鲁尔黑鹂
    "Red-winged Blackbird",    # 10. 红翅黑鹂
    "Rusty Blackbird",         # 11. 锈色黑鹂
    "Yellow-headed Blackbird", # 12. 黄头黑鹂
    "Bobolink",                # 13. 稻田雀
    "Indigo Bunting",          # 14. 靛蓝雀
    "Lazuli Bunting",          # 15. 天蓝雀
    "Painted Bunting",         # 16. 彩雀
    "Cardinal",                # 17. 红雀
    "Spotted Catbird",         # 18. 斑猫鸟
    "Gray Catbird",            # 19. 灰猫鸟
    "Black Catbird",           # 20. 黑猫鸟
    "Yellow-breasted Chat",    # 21. 黄胸鹟
    "Eastern Towhee",          # 22. 东方斑雀
    "Spotted Towhee",          # 23. 斑翅斑雀
    "Chuck-will's Widow",      # 24. 夜鹰
    "Whip-poor-will",          # 25. 夜鹰
    "Barn Owl",                # 26. 谷仓猫头鹰
    "Screech Owl",             # 27. 鸣角鸮
    "Great Horned Owl",        # 28. 大角鸮
    "Snowy Owl",               # 29. 雪鸮
    "Northern Hawk Owl",       # 30. 北方鹰鸮
    "Northern Pygmy Owl",      # 31. 北方侏儒猫头鹰
    "Burrowing Owl",           # 32. 穴居猫头鹰
    "Barred Owl",              # 33. 横纹林鸮
    "Great Gray Owl",          # 34. 大灰林鸮
    "Long-eared Owl",          # 35. 长耳林鸮
    "Shorteared Owl",          # 36. 短耳鸮
    "Northern Saw-whet Owl",   # 37. 北方锯嘴猫头鹰
    "Belted Kingfisher",       # 38. 带状翠鸟
    "Ringed Kingfisher",       # 39. 环状翠鸟
    "Pied Kingfisher",         # 40. 斑鱼狗
    "Green Kingfisher",        # 41. 绿色翠鸟
    "Common Nighthawk",        # 42. 常见夜鹰
    "Antillean Nighthawk",     # 43. 安地列斯夜鹰
    "Common Poorwill",         # 44. 夜鹰
    "Chuck-will's-widow",      # 45. 夜鹰
    "Whip-poor-will",          # 46. 夜鹰
    "Swainson's Warbler",      # 47. 斯温森莺
    "Worm-eating Warbler",     # 48. 蠕虫食虫莺
    "Louisiana Waterthrush",   # 49. 路易斯安那水鸫
    "Northern Waterthrush",    # 50. 北方水鸫
    "Golden-winged Warbler",   # 51. 金翅莺
    "Tennessee Warbler",       # 52. 田纳西莺
    "Orange-crowned Warbler",  # 53. 橙冠莺
    "Nashville Warbler",       # 54. 纳什维尔莺
    "Connecticut Warbler",     # 55. 康涅狄格莺
    "MacGillivray's Warbler",  # 56. 麦氏林莺
    "Mourning Warbler",        # 57. 丧服莺
    "Kentucky Warbler",        # 58. 肯塔基莺
    "Common Yellowthroat",     # 59. 常见黄喉莺
    "Hooded Warbler",          # 60. 兜帽莺
    "Wilson's Warbler",        # 61. 威尔逊莺
    "Canada Warbler",          # 62. 加拿大莺
    "Red-faced Warbler",       # 63. 红脸莺
    "Painted Redstart",        # 64. 彩红尾鸲
    "Golden-fronted Woodpecker",  # 65. 金额啄木鸟
    "Red-bellied Woodpecker",  # 66. 红腹啄木鸟
    "Red-headed Woodpecker",   # 67. 红头啄木鸟
    "Downy Woodpecker",        # 68. 绒毛啄木鸟
    "Hairy Woodpecker",        # 69. 毛发啄木鸟
    "White-headed Woodpecker", # 70. 白头啄木鸟
    "American Three-toed Woodpecker",  # 71. 美洲三趾啄木鸟
    "Pileated Woodpecker",     # 72. 冠啄木鸟
    "Northern Flicker",        # 73. 北方扑翅鴷
    "Phaeochrome Woodpecker",  # 74. 褐色啄木鸟
    "Gila Woodpecker",         # 75. 吉拉啄木鸟
    "Golden-fronted Woodpecker",  # 76. 金额啄木鸟
    "Acorn Woodpecker",        # 77. 橡子啄木鸟
    "Black-backed Woodpecker", # 78. 黑背啄木鸟
    "American Kestrel",        # 79. 美洲红隼
    "Merlin",                  # 80. 灰背隼
    "Peregrine Falcon",        # 81. 游隼
    "Prairie Falcon",          # 82. 草原隼
    "Yellow-billed Cuckoo",    # 83. 黄嘴杜鹃
    "Black-billed Cuckoo",     # 84. 黑嘴杜鹃
    "Mourning Dove",           # 85. 哀鸽
    "Rock Dove",               # 86. 岩鸽
    "Band-tailed Pigeon",      # 87. 带尾鸽
    "Spotted Dove",            # 88. 斑鸠
    "Inca Dove",               # 89. 印加鸠
    "Common Ground Dove",      # 90. 常见地鸠
    "White-tipped Dove",       # 91. 白梢鸠
    "White-winged Dove",       # 92. 白翅鸠
    "Budgerigar",              # 93. 虎皮鹦鹉
    "Northern Cardinal",       # 94. 北方红雀
    "Pyrrhuloxia",             # 95. 冠红雀
    "Caspian Tern",            # 96. 里海燕鸥
    "Royal Tern",              # 97. 皇家燕鸥
    "Roseate Tern",            # 98. 粉红燕鸥
    "Least Tern",              # 99. 最小燕鸥
    "Sooty Tern",              # 100. 烟色燕鸥
    "Black Tern",              # 101. 黑燕鸥
    "Black Skimmer",           # 102. 黑剪嘴鸥
    "Laughing Gull",           # 103. 笑鸥
    "Franklin's Gull",         # 104. 富兰克林鸥
    "Bonaparte's Gull",        # 105. 博纳帕特鸥
    "Ring-billed Gull",        # 106. 环嘴鸥
    "Herring Gull",            # 107. 银鸥
    "Thayer's Gull",           # 108. 泰尔鸥
    "Western Gull",            # 109. 西方鸥
    "California Gull",         # 110. 加州鸥
    "Glaucous-winged Gull",    # 111. 淡翅鸥
    "Great Black-backed Gull", # 112. 大黑背鸥
    "Black-legged Kittiwake",  # 113. 黑腿三趾鸥
    "Ivory Gull",              # 114. 象牙鸥
    "Fulvous Whistling-Duck",  # 115. 棕褐树鸭
    "Snow Goose",              # 116. 雪雁
    "Canada Goose",            # 117. 加拿大雁
    "Cackling Goose",          # 118. 小加拿大雁
    "Barnacle Goose",          # 119. 苔原雁
    "Brant",                   # 120. 黑雁
    "Egyptian Goose",          # 121. 埃及雁
    "Muscovy Duck",            # 122. 麝香鸭
    "Wood Duck",               # 123. 林鸳鸯
    "American Wigeon",         # 124. 美洲赤颈鸭
    "Mallard",                 # 125. 绿头鸭
    "Northern Shoveler",       # 126. 北方铲嘴鸭
    "Northern Pintail",        # 127. 北方针尾鸭
    "Green-winged Teal",       # 128. 绿翅鸭
    "Canvasback",              # 129. 红头潜鸭
    "Redhead",                 # 130. 红头鸭
    "Ring-necked Duck",        # 131. 环颈潜鸭
    "Greater Scaup",           # 132. 较大潜鸭
    "Lesser Scaup",            # 133. 较小潜鸭
    "Harlequin Duck",          # 134. 海番鸭
    "Oldsquaw",                # 35. 长尾鸭
    "Black Scoter",            # 136. 黑海番鸭
    "Surf Scoter",             # 137. 冲浪海番鸭
    "White-winged Scoter",     # 138. 白翅海番鸭
    "Common Goldeneye",        # 139. 普通鹊鸭
    "Barrow's Goldeneye",      # 140. 巴罗鹊鸭
    "Bufflehead",              # 141. 巨头潜鸭
    "Hooded Merganser",        # 142. 冠秋沙鸭
    "Common Merganser",        # 143. 普通秋沙鸭
    "Red-breasted Merganser",  # 144. 红胸秋沙鸭
    "Ruddy Duck",              # 145. 棕硬尾鸭
    "Turkey Vulture",          # 146. 土耳其秃鹫
    "Black Vulture",           # 147. 黑秃鹫
    "California Condor",       # 148. 加州神鹫
    "Osprey",                  # 149. 鱼鹰
    "Bald Eagle",              # 150. 白头海雕
    "Northern Harrier",        # 151. 北方灰泽鹞
    "Sharp-shinned Hawk",      # 152. Sharp-shinned Hawk
    "Cooper's Hawk",           # 153. Cooper's Hawk
    "Northern Goshawk",        # 154. Northern Goshawk
    "Red-shouldered Hawk",     # 155. Red-shouldered Hawk
    "Broad-winged Hawk",       # 156. Broad-winged Hawk
    "Swainson's Hawk",         # 157. Swainson's Hawk
    "Red-tailed Hawk",         # 158. Red-tailed Hawk
    "Rough-legged Hawk",       # 159. Rough-legged Hawk
    "Golden Eagle",            # 160. Golden Eagle
    "American Dipper",         # 161. American Dipper
    "Horned Lark",             # 162. Horned Lark
    "Purple Martin",           # 163. Purple Martin
    "Tree Swallow",            # 164. Tree Swallow
    "Violet-green Swallow",    # 165. Violet-green Swallow
    "Northern Rough-winged Swallow",  # 166. Northern Rough-winged Swallow
    "Bank Swallow",            # 167. Bank Swallow
    "Cliff Swallow",           # 168. Cliff Swallow
    "Barn Swallow",            # 169. Barn Swallow
    "Blue Jay",                # 170. Blue Jay
    "American Crow",           # 171. American Crow
    "Fish Crow",               # 172. Fish Crow
    "Common Raven",            # 173. Common Raven
    "Black-capped Chickadee",  # 174. Black-capped Chickadee
    "Boreal Chickadee",        # 175. Boreal Chickadee
    "Carolina Chickadee",      # 176. Carolina Chickadee
    "Mountain Chickadee",      # 177. Mountain Chickadee
    "Chestnut-backed Chickadee",  # 178. Chestnut-backed Chickadee
    "Plain Chickadee",         # 179. Plain Chickadee
    "Tufted Titmouse",         # 180. Tufted Titmouse
    "Black-crested Titmouse",  # 181. Black-crested Titmouse
    "Verdin",                  # 182. Verdin
    "Bushtit",                 # 183. Bushtit
    "Red-breasted Nuthatch",   # 184. Red-breasted Nuthatch
    "White-breasted Nuthatch", # 185. White-breasted Nuthatch
    "Pygmy Nuthatch",          # 186. Pygmy Nuthatch
    "Brown Creeper",           # 187. Brown Creeper
    "Cactus Wren",             # 188. Cactus Wren
    "Rock Wren",               # 189. Rock Wren
    "Canyon Wren",             # 190. Canyon Wren
    "Carolina Wren",           # 191. Carolina Wren
    "Bewick's Wren",           # 192. Bewick's Wren
    "House Wren",              # 193. House Wren
    "Winter Wren",             # 194. Winter Wren
    "Sedge Wren",              # 195. Sedge Wren
    "Marsh Wren",              # 196. Marsh Wren
    "American Robin",          # 197. American Robin
    "Wood Thrush",             # 198. Wood Thrush
    "Hermit Thrush",           # 199. Hermit Thrush
    "Swainson's Thrush",       # 200. Swainson's Thrush
]


@dataclass
class CUB200TrainingConfig:
    """CUB-200 细粒度分类训练配置

    数学参数推导:
        1. 主学习率: lr = lr_base × (batch_eff / 256)
        2. Center Loss 学习率: lr_center = lr_main × center_lr_ratio
        3. 正则化: scale = √(N_ref / N) ≈ 1.5-2.0

    I36: 架构参数移至 ModelArchitectureConfig，此处仅保留训练策略参数。
         模型创建请使用 arch_config = ModelArchitectureConfig(...)

    Attributes:
        batch_size: 批次大小（有效批次 = batch_size × accum_steps）
        num_epochs: 训练轮数
        learning_rate: 主模型学习率
        warmup_epochs: 学习率预热轮数
        accum_steps: 梯度累积步数
        validate_interval: 验证间隔（每 N 个 epoch）
        center_lr_ratio: Center Loss 学习率与主学习率的比率（论文推荐 50）
        center_lr: Center Loss 绝对学习率（若设置则优先使用）
        arch_config: 模型架构配置（I36 统一配置）
    """
    # 基础训练配置
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 3e-4  # I121-6: 2.7e-4 → 3e-4 (+11% 提升收敛速度)
    warmup_epochs: int = 5  # I121-6: 7 → 5 (-29% 缩短warmup)
    accum_steps: int = 3
    validate_interval: int = 3  # 每 N 个 epoch 验证一次

    # Center Loss 学习率参数
    center_lr_ratio: float = 50.0  # lr_center / lr_main，论文推荐 50
    center_lr: Optional[float] = None  # 绝对学习率，优先于 ratio

    # 设备和混合精度
    device: str = "cuda"
    use_amp: bool = True
    gradient_clip_norm: Optional[float] = 1.0

    # 细粒度专用配置
    use_center_loss: bool = True
    center_loss_weight: float = 0.01

    # 正则化（比通用分类更强）
    label_smoothing: float = 0.13  # 0.1 × (1 + log₁₀(200/100))
    dropout: float = 0.15  # I121-6: 0.22 → 0.15 (-32% 减弱正则化)
    drop_path_rate: float = 0.10  # I121-6: 0.16 → 0.10 (-38% 减弱正则化)
    weight_decay: float = 0.15

    # Mixup/CutMix 数据增强
    mixup_alpha: float = 0.3
    cutmix_alpha: float = 0.8
    mixup_prob: float = 0.4

    # Focal Loss（处理困难样本）
    use_focal_loss: bool = True
    focal_gamma: float = 2.5  # P1-2: 统一为推荐值 (I28-1 优化，难/易比 243x)

    # 早停
    patience: int = 15
    min_delta: float = 0.001

    # 评估
    compute_per_class: bool = True
    compute_confusion: bool = True
    top_confused_pairs: int = 10

    # 日志
    log_level: str = "INFO"

    # I31: 面积编码配置 (从 arch_config 获取，无需重复定义)
    # use_area_encoding: bool = False  # 移至 arch_config
    # use_affine_modulation: bool = False  # 移至 arch_config
    # fourier_levels: int = 4  # 移至 arch_config

    # I111-1: 温度退火参数已移至 SplitterConfig
    # 训练器从 model.splitter.config 读取温度配置

    # M1: 辅助损失权重 (I36)
    # I121-6: 权重提升以解决配置链路覆盖问题
    # 有效权重 = config值 × splitter内部因子(0.1)
    # 目标有效权重: sparsity=0.03, elastic=0.005
    splitter_sparsity_weight: float = 0.3  # 0.01 → 0.30 (30x提升)
    elastic_budget_weight: float = 0.05     # 0.01 → 0.05 (5x提升)
    depth_kl_weight: float = 0.5            # 深度KL散度损失权重

    # =========================================================================
    # I36: 模型架构配置 (使用 ModelArchitectureConfig)
    # =========================================================================
    # 架构参数统一通过 arch_config 指定，确保与 ModelArchitectureConfig 一致
    # I121-7: 模型容量扩展 (16GB 显存约束下的最优配置)
    # - dim: 384 → 448 (+17%) 增强单层表达能力
    # - num_layers: 8 → 10 (+25%) 增强层级特征提取
    # - heads: 6 → 7 (+17%) 增强多头注意力覆盖
    # - 预期参数量: 24M → ~35M
    # - 预期显存: ~12GB → ~14GB
    arch_config: ModelArchitectureConfig = field(default_factory=lambda: ModelArchitectureConfig(
        num_classes=200,
        dim=448,
        num_layers=10,
        heads=7,
        image_size=None,  # I78: 动态分辨率
        min_patch_size=4,
    ))

    # =========================================================================
    # I99: 训练策略参数（补充 arch_config 中未包含的字段）
    # =========================================================================
    # 注意: learning_rate 是训练超参数，不在 arch_config 中定义
    # I121-6: 2.7e-4 → 3e-4 (+11% 提升收敛速度)
    learning_rate: float = 3e-4  # 主模型学习率
    use_checkpoint: bool = False
    use_channels_last: bool = False  # I78: channels-last 内存格式 (节省 ~20% VRAM)
    use_compile: bool = False        # I78: torch.compile 优化 (提升 ~30% 训练速度)
    compile_mode: str = "default"    # torch.compile 模式: default, reduce-overhead, max-autotune

    # I99: tokenizer 类型 (不在 ModelArchitectureConfig 中定义)
    tokenizer_type: str = "streaming_v3"  # Tokenizer 类型

    # =========================================================================
    # P1 Fix: 移除冗余字段（应从 arch_config 获取）
    # - dim_head: 从 arch_config.dim // arch_config.heads 计算
    # - channels: 从 arch_config.channels 获取
    # - K_min: 从 arch_config.K_min_abs 获取 (I33 相对预算)
    # - K_max: 从 token_coverage_* 计算，不应硬编码
    # - ffn_type: 从 arch_config.ffn_type 获取
    # =========================================================================

    def __post_init__(self):
        """参数验证 - 数学约束"""
        assert 0 < self.batch_size <= 512, f"batch_size={self.batch_size} 必须在 (0, 512]"
        assert 1 <= self.num_epochs <= 1000, f"num_epochs={self.num_epochs} 必须在 [1, 1000]"
        assert self.learning_rate > 0, f"learning_rate={self.learning_rate} 必须 > 0"
        assert 0 <= self.warmup_epochs <= self.num_epochs // 2, \
            f"warmup_epochs={self.warmup_epochs} 必须 <= num_epochs/2"
        assert 1 <= self.accum_steps <= 32, f"accum_steps={self.accum_steps} 必须在 [1, 32]"
        assert 1 <= self.validate_interval <= self.num_epochs, \
            f"validate_interval={self.validate_interval} 无效"
        assert 0 <= self.label_smoothing < 1.0, f"label_smoothing={self.label_smoothing} 必须在 [0, 1)"
        assert 0 <= self.dropout < 1.0, f"dropout={self.dropout} 必须在 [0, 1)"
        assert 0 <= self.drop_path_rate < 1.0, f"drop_path_rate={self.drop_path_rate} 必须在 [0, 1)"
        assert self.center_loss_weight >= 0, f"center_loss_weight={self.center_loss_weight} 必须 >= 0"
        assert 0 < self.center_lr_ratio <= 1000, f"center_lr_ratio={self.center_lr_ratio} 必须在 (0, 1000]"

        # 日志级别
        self.log_level = getattr(logging, self.log_level.upper(), logging.INFO)

    def validate_model_config_consistency(self) -> None:
        """P1 Fix: 验证 CUB200TrainingConfig 与 ModelArchitectureConfig 的一致性

        检查训练配置参数与模型架构配置是否一致，避免运行时错误。
        注意: K_min/K_max 由模型根据 token_coverage_* 动态计算，不再硬编码检查。

        Raises:
            ValueError: 当配置不一致时
        """
        issues = []

        # 通过便捷属性从 arch_config 获取参数
        arch = self.arch_config

        # 检查 dim_head 兼容性 (通过 arch 配置访问)
        if arch.dim % arch.heads != 0:
            issues.append(f"dim={arch.dim} 不能被 heads={arch.heads} 整除")

        # 检查 min_patch_size 合理性 (通过便捷属性访问)
        if self.min_patch_size < 1:
            issues.append(f"min_patch_size={self.min_patch_size} 必须 >= 1")

        # 检查覆盖率约束 (从 arch_config 验证)
        if not (0 < arch.token_coverage_min < arch.token_coverage_max < 1):
            issues.append(
                f"覆盖率约束违反: 0 < {arch.token_coverage_min} < {arch.token_coverage_max} < 1"
            )

        if issues:
            raise ValueError(
                f"CUB200TrainingConfig 与 ModelArchitectureConfig 不一致:\n  - " +
                "\n  - ".join(issues)
            )

    def _validate_gene_consistency(self, loaded_gene: "ModelGene") -> None:
        """验证加载的 ModelGene 与当前训练配置的一致性

        Args:
            loaded_gene: 从 checkpoint 加载的 ModelGene

        Raises:
            ValueError: 当配置不一致时
        """
        from ..core.model_gene import ModelGene

        # 当前配置构造的 ModelGene
        current_gene = ModelGene.from_config(
            self.config.arch_config,
            dataset_name='cub200',
        )

        # 关键架构参数一致性检查
        key_params = ['dim', 'num_layers', 'heads', 'mlp_dim', 'image_size',
                      'min_patch_size', 'token_coverage_min', 'token_coverage_max']

        mismatches = []
        for param in key_params:
            current_val = getattr(current_gene, param, None)
            loaded_val = getattr(loaded_gene, param, None)
            if current_val != loaded_val:
                mismatches.append(f"{param}: checkpoint={loaded_val}, current={current_val}")

        if mismatches:
            raise ValueError(
                f"Checkpoint 配置与当前训练配置不一致:\n  - " +
                "\n  - ".join(mismatches) +
                "\n\n请使用与训练时相同配置加载 checkpoint。"
            )

        self.logger.info(f"ModelGene 验证通过: {loaded_gene}")

    # =========================================================================
    # I36: 便捷属性（从 arch_config 获取，保持向后兼容）
    # =========================================================================
    @property
    def num_classes(self) -> int:
        """从 arch_config 获取类别数"""
        return self.arch_config.num_classes

    @property
    def dim(self) -> int:
        """从 arch_config 获取嵌入维度"""
        return self.arch_config.dim

    @property
    def num_layers(self) -> int:
        """从 arch_config 获取 Transformer 层数 (原 depth)"""
        return self.arch_config.num_layers

    @property
    def heads(self) -> int:
        """从 arch_config 获取注意力头数"""
        return self.arch_config.heads

    @property
    def mlp_dim(self) -> int:
        """从 arch_config 获取 FFN 维度"""
        return self.arch_config.mlp_dim

    @property
    def image_size(self) -> Optional[int]:
        """从 arch_config 获取图像尺寸"""
        return self.arch_config.image_size

    @property
    def min_patch_size(self) -> int:
        """从 arch_config 获取最小 patch 大小"""
        return self.arch_config.min_patch_size

    @property
    def use_area_encoding(self) -> bool:
        """从 arch_config 获取面积编码配置"""
        return self.arch_config.use_area_encoding

    @property
    def use_affine_modulation(self) -> bool:
        """从 arch_config 获取仿射调制配置"""
        return self.arch_config.use_affine_modulation

    @property
    def fourier_levels(self) -> int:
        """从 arch_config 获取傅里叶级别数"""
        return self.arch_config.fourier_levels


@dataclass
class CUB200EvalResult:
    """CUB-200 评估结果"""
    loss: float
    accuracy: float  # Top-1 准确率
    top5_accuracy: float
    per_class_accuracy: Optional[Dict[int, float]] = None
    confused_pairs: Optional[List[Tuple[int, int, int]]] = None  # (class_a, class_b, count)
    feature_stats: Optional[Dict[str, float]] = None
    mca: Optional[float] = None  # Mean Class Accuracy
    intra_inter_ratio: Optional[float] = None  # 类内/类间距离比


@dataclass
class CUB200TrainerState:
    """训练状态 - 完整记录训练过程"""
    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0
    best_epoch: int = 0
    patience_counter: int = 0

    # 当前 epoch 统计
    train_loss: float = 0.0
    train_accuracy: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0

    # 历史记录
    history: Dict[str, List[float]] = field(default_factory=lambda: {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_top5_accuracy": [],
        "learning_rate": [],
        "center_loss": [],
    })


class CUB200Trainer:
    """CUB-200-2011 细粒度分类训练器 - 独立实现

    数学形式化:
        总损失: L_total = L_ce + λ_center × L_center

        梯度累积:
            effective_loss = L_total / accum_steps
            grad = Σ_{i=1}^{accum_steps} ∂L_i/∂θ

        学习率预热:
            lr(epoch) = lr_base × min(epoch / warmup_epochs, 1.0)

    特征:
        1. 完整训练循环 - 独立管理所有训练逻辑
        2. 正确梯度累积 - Center Loss 与主模型同步更新
        3. 验证禁用 AMP - 确保指标精度
        4. 完整检查点 - 包含 scaler、center optimizer 状态
        5. 学习率预热 - 线性预热策略
    """

    def __init__(
        self,
        model: nn.Module,
        config: CUB200TrainingConfig,
        num_classes: int = 200,
        feat_dim: int = 256,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: Fractal ViT 模型
            config: 训练配置
            num_classes: 类别数（CUB-200 = 200）
            feat_dim: 特征维度
            device: 计算设备
        """
        self.model = model
        self.config = config
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        # 设备处理
        if device is None:
            self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # I78: 性能优化 - channels-last 内存格式
        # 优势: 卷积运算优化，节省 ~20% VRAM
        if self.config.use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
            self.logger.info("启用 channels-last 内存格式")

        # I78: torch.compile 优化
        # 优势: 提升 ~30% 训练速度，首次 forward 较慢
        if self.config.use_compile:
            self.logger.info(f"启用 torch.compile (mode={self.config.compile_mode})...")
            self.model = torch.compile(
                self.model,
                mode=self.config.compile_mode,
                dynamic=True,  # 支持动态序列长度
            )
            self.logger.info("torch.compile 完成")

        # 日志
        logging.basicConfig(level=config.log_level)
        self.logger = logging.getLogger(__name__)

        # 创建细粒度损失函数
        loss_config = FinegrainedLossConfig(
            num_classes=num_classes,
            feat_dim=feat_dim,
            label_smoothing=config.label_smoothing,
            use_center_loss=config.use_center_loss,
            center_loss_weight=config.center_loss_weight,
            use_focal=config.use_focal_loss,
            focal_gamma=config.focal_gamma,
        )
        self.loss_fn = FinegrainedLoss(loss_config).to(self.device)

        # Center Loss 优化器（延迟初始化，需要主优化器学习率）
        self.center_optimizer: Optional[torch.optim.Optimizer] = None
        self._center_lr: float = 0.0

        # 训练状态
        self.state = CUB200TrainerState()

        # I35: 混合精度 scaler - device 是关键字参数
        self.scaler: Optional[GradScaler] = GradScaler(enabled=config.use_amp, device='cuda')

        # I111-6: 初始化深度分布监控器
        self._init_depth_monitor()

        self.logger.info(f"CUB200Trainer 初始化完成: device={self.device}, num_classes={num_classes}")

    def _init_depth_monitor(self) -> None:
        """I111-6: 初始化深度分布监控器"""
        from vit_pytorch.gumbel_topk_splitter import DepthMonitor

        splitter = getattr(self.model, 'splitter', None)
        if splitter is not None and hasattr(splitter, 'get_depth_distribution_tensor'):
            self._depth_monitor: Optional[DepthMonitor] = DepthMonitor(splitter, history_max_size=0)
            self.logger.info("[I111-6] 深度分布监控器已初始化")
        else:
            self._depth_monitor = None
            self.logger.debug("[I111-6] 分割器不支持深度分布监控")

    def _log_depth_monitor_to_tensorboard(
        self,
        writer: 'SummaryWriter',
        epoch: int,
    ) -> None:
        """I111-6: 记录深度分布到 TensorBoard"""
        if self._depth_monitor is None or writer is None:
            return

        try:
            stats = self._depth_monitor.update(epoch)

            # 深度分布 histogram
            writer.add_histogram('Splitter/depth_distribution/pi',
                               stats['pi'].detach().cpu().numpy(), epoch)

            # 深度熵
            writer.add_scalar('Splitter/depth_entropy', stats['entropy'], epoch)
            writer.add_scalar('Splitter/kl_from_uniform', stats['kl_from_uniform'], epoch)

            # 配额概率（如果可用）
            quota_probs = stats.get('quota_probs')
            if quota_probs is not None:
                writer.add_histogram('Splitter/quota_probs',
                                   quota_probs.detach().cpu().numpy(), epoch)

            self.logger.debug(f"[I111-6] Depth monitor logged: H={stats['entropy']:.3f}, KL={stats['kl_from_uniform']:.3f}")
        except Exception as e:
            self.logger.warning(f"[I111-6] Depth monitor failed: {e}")

    def initialize_center_optimizer(self, main_optimizer: torch.optim.Optimizer) -> None:
        """初始化 Center Loss 优化器

        数学推导:
            论文推荐 lr_center = 0.5，当 lr_main = 0.01
            比率: ratio = lr_center / lr_main = 50

            对于 Fractal ViT: lr_main = config.learning_rate
            lr_center = lr_main × ratio (或使用绝对 center_lr)

        Args:
            main_optimizer: 主模型优化器
        """
        if not self.config.use_center_loss or self.loss_fn.center_loss is None:
            self.logger.info("Center Loss 未启用，跳过初始化")
            return

        # 计算 Center Loss 学习率
        base_lr = main_optimizer.param_groups[0]['lr']

        if self.config.center_lr is not None:
            # 绝对学习率
            self._center_lr = self.config.center_lr
            self.logger.info(f"使用绝对 Center Loss 学习率: {self._center_lr}")
        else:
            # 相对学习率 = lr_main × ratio
            self._center_lr = base_lr * self.config.center_lr_ratio
            self.logger.info(f"计算 Center Loss 学习率: {base_lr} × {self.config.center_lr_ratio} = {self._center_lr}")

        # 创建优化器
        self.center_optimizer = torch.optim.SGD(
            self.loss_fn.center_loss.parameters(),
            lr=self._center_lr,
            momentum=0.9,  # Center Loss 论文推荐动量
        )

        self.logger.info(f"Center Loss 优化器初始化完成: lr={self._center_lr}")

    # A21: 温度退火设置方法
    def _setup_temperature_annealing(self, train_loader: DataLoader) -> None:
        """设置温度退火 (A21 修复)

        使用 GumbelTopKSplitter 内置退火 API 替代手动计算。
        温度将以 batch 粒度自动更新，而非 epoch 粒度。

        Args:
            train_loader: 训练数据加载器
        """
        # P0 Fix: 使用 getattr 安全访问 splitter
        splitter = getattr(self.model, 'splitter', None)
        if splitter is None:
            self.logger.info("[A21] 模型无 splitter，跳过温度退火设置")
            return

        # 检查是否支持退火 API
        if not hasattr(splitter, 'enable_temperature_annealing'):
            self.logger.info("[A21] Splitter 不支持温度退火 API")
            return

        # I111-1: 从 SplitterConfig 读取温度参数
        # I113-10: 使用常量确保一致性
        config = getattr(splitter, 'config', None)
        if config is None:
            self.logger.warning("[A21] Splitter 无 config，使用默认值")
            T_start, T_end, warmup_epochs = 1.0, SPLITTER_TEMP_END, 8
        else:
            T_start = getattr(config, 'temperature_init', 1.0)
            T_end = getattr(config, 'temperature_min', TEMPERATURE_MIN)
            warmup_epochs = getattr(config, 'temperature_warmup_epochs', 8)

        # 计算每 epoch 的步数
        batches_per_epoch = len(train_loader) // self.config.accum_steps

        # warmup 后的步数用于退火
        post_warmup_steps = max(1, (self.config.num_epochs - warmup_epochs) * batches_per_epoch)

        # 启用退火 (warmup 期间使用 T_start)
        splitter.enable_temperature_annealing(
            total_steps=post_warmup_steps,
            T_start=T_start,
            T_end=T_end,
            schedule="exponential",
        )

        self.logger.info(f"[A21 OK] 温度退火已配置: {T_start} → {T_end}")
        self.logger.info(f"     步数: {post_warmup_steps} (warmup: {warmup_epochs} epochs)")

    # I36-2: 使用模型协议接口配置训练
    def _configure_training_via_protocol(self, train_loader: DataLoader) -> None:
        """通过模型协议接口配置训练参数

        使用 FractalModelProtocol 的 configure_training 方法，
        替代直接访问 splitter 的旧方式。

        Args:
            train_loader: 训练数据加载器，用于计算总步数
        """
        # 检查模型是否支持协议接口
        if not hasattr(self.model, 'configure_training'):
            self.logger.info("[I36-2] 模型不支持 configure_training，使用旧方法")
            # 回退到旧方法
            self._setup_temperature_annealing(train_loader)
            return

        # 计算每 epoch 的步数
        batches_per_epoch = len(train_loader) // self.config.accum_steps

        # warmup 后的步数
        warmup_epochs = 8  # 默认 warmup epochs
        post_warmup_steps = max(1, (self.config.num_epochs - warmup_epochs) * batches_per_epoch)

        # 构建配置字典
        training_config = {
            'temperature_annealing': True,
            'total_steps': post_warmup_steps,
            'temp_start': 1.0,
            'temp_end': SPLITTER_TEMP_END,
            'schedule': 'exponential',
            'aux_loss_weights': {
                'sparsity': getattr(self.config, 'splitter_sparsity_weight', 0.0),
                'elastic': getattr(self.config, 'elastic_budget_weight', 0.0),
                'depth_kl': getattr(self.config, 'depth_kl_weight', 0.0),
            }
        }

        # 调用模型的协议接口
        self.model.configure_training(training_config)

        self.logger.info(f"[I36-2 OK] 通过协议配置训练: temp 1.0 → {SPLITTER_TEMP_END}")
        self.logger.info(f"     步数: {post_warmup_steps} (warmup: {warmup_epochs} epochs)")

    # I36-3: 收集 Splitter 诊断信息
    def _collect_splitter_diagnostics(self) -> Dict[str, Any]:
        """收集分割器诊断信息

        使用 FractalModelProtocol 的 get_splitter_diagnostics 方法。
        在训练结束后调用，收集最终的诊断信息。

        Returns:
            诊断信息字典
        """
        if hasattr(self.model, 'get_splitter_diagnostics'):
            return self.model.get_splitter_diagnostics()
        return {}

    # A21: Warmup 处理方法 (I111-1: 从 SplitterConfig 读取温度)
    def _handle_warmup(self, epoch: int) -> None:
        """处理 warmup 期间的温度/偏置 (A21)

        在 warmup 期间禁用退火，固定 T_start。
        warmup 结束后重新启用退火。

        Args:
            epoch: 当前 epoch
        """
        # P0 Fix: 使用 getattr 安全访问 splitter
        splitter = getattr(self.model, 'splitter', None)
        if splitter is None:
            return

        # I111-1: 从 SplitterConfig 读取温度参数
        config = getattr(splitter, 'config', None)
        if config is None:
            T_start, T_end, warmup_epochs = 1.0, 0.3, 8
        else:
            T_start = getattr(config, 'temperature_init', 1.0)
            T_end = getattr(config, 'temperature_min', 0.3)
            warmup_epochs = getattr(config, 'temperature_warmup_epochs', 8)

        if epoch <= warmup_epochs:
            # Warmup: 禁用退火，固定 T_start
            if hasattr(splitter, 'disable_temperature_annealing'):
                splitter.disable_temperature_annealing()
            splitter.set_temperature(T_start)
        elif epoch == warmup_epochs + 1:
            # Warmup 结束: 重新启用退火
            if hasattr(self, 'train_loader'):
                batches_per_epoch = len(self.train_loader) // self.config.accum_steps
                post_warmup_steps = max(1, (self.config.num_epochs - warmup_epochs) * batches_per_epoch)

                if hasattr(splitter, 'enable_temperature_annealing'):
                    splitter.enable_temperature_annealing(
                        total_steps=post_warmup_steps,
                        T_start=T_start,
                        T_end=T_end,
                        schedule="exponential",
                    )
                    self.logger.info(f"[A21] Warmup 结束，重新启用温度退火: {T_start} → {T_end}")

    def _get_warmup_lr(self, epoch: int) -> float:
        """计算学习率（包含预热）

        数学公式:
            lr(epoch) = lr_base × min(epoch / warmup_epochs, 1.0)

        Args:
            epoch: 当前 epoch（从 1 开始）

        Returns:
            当前学习率
        """
        if epoch <= self.config.warmup_epochs:
            warmup_factor = epoch / max(1, self.config.warmup_epochs)
            return self.config.learning_rate * warmup_factor
        return self.config.learning_rate

    def _get_dynamic_clip_norm(self) -> float:
        """I121-6: 动态计算梯度裁剪范数

        数学公式:
            clip_norm = GRAD_CLIP_BASE_NORM × (current_lr / GRAD_CLIP_BASE_LR)

        原理:
            - 梯度范数与学习率成正比: ||∇L|| ∝ lr
            - 学习率增加时，梯度范数按比例增加
            - 动态调整确保训练稳定性

        Returns:
            动态裁剪范数 (已限制在 [GRAD_CLIP_MIN_NORM, GRAD_CLIP_MAX_NORM] 范围内)
        """
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            return self.config.gradient_clip_norm or GRAD_CLIP_BASE_NORM

        current_lr = self.optimizer.param_groups[0]['lr']

        # 计算动态裁剪范数
        clip_norm = GRAD_CLIP_BASE_NORM * (current_lr / GRAD_CLIP_BASE_LR)

        # 限制在合理范围内
        clip_norm = max(GRAD_CLIP_MIN_NORM, min(GRAD_CLIP_MAX_NORM, clip_norm))

        return clip_norm

    def _compute_splitter_loss(
        self,
        images: torch.Tensor,
        stats: Any,
    ) -> Optional[torch.Tensor]:
        """I142-1: 计算 splitter 辅助损失（与 train_fractal_vit.py 对齐）

        数学公式:
            L_total = L_ce + λ_sparsity × L_entropy + λ_elastic × L_elastic + λ_depth × L_depth

        其中:
            - L_entropy: 稀疏性正则化（鼓励使用更少 tokens）
            - L_elastic: 弹性预算损失（约束总 token 数在预算范围内）
            - L_depth: 深度分布 KL 散度损失

        Args:
            images: 输入图像 [B, C, H, W]
            stats: 模型返回的 TrainingStats

        Returns:
            splitter_loss: 辅助损失张量（如果 splitter 存在），否则返回 None
        """
        # 统一 Splitter 访问路径 (I99-对齐修复)
        splitter = None
        if hasattr(self.model, 'splitter'):
            splitter = self.model.splitter
        elif hasattr(self.model, 'tokenizer') and hasattr(self.model.tokenizer, 'splitter'):
            splitter = self.model.tokenizer.splitter

        if splitter is None:
            return None

        # 检查是否支持辅助损失计算
        if not hasattr(splitter, 'get_auxiliary_losses'):
            return None

        # I107-7: 从 stats.shared_features 获取已计算的 features
        # 避免重复调用 model.tokenizer.shared_conv(imgs)
        if stats is not None and hasattr(stats, 'shared_features') and stats.shared_features is not None:
            splitter_features = stats.shared_features
        else:
            # Fallback: 仍需计算时的回退方案
            splitter_features = self.model.tokenizer.shared_conv(images)

        # 计算辅助损失
        # 传递 TrainingStats 中的 num_tokens 和 depth_distribution 到 Splitter
        try:
            aux_losses = splitter.get_auxiliary_losses(
                features=splitter_features,
                image_size=(images.shape[2], images.shape[3]),
                include_elastic_budget=True,
                include_soft_entropy=True,
                batch_size=images.shape[0],
                # 传递 TrainingStats 数据以增强辅助损失计算
                actual_token_count=getattr(stats, 'num_tokens', None),
                depth_distribution=getattr(stats, 'depth_distribution', None),
                entropy_target=0.5,  # 默认目标熵
                entropy_weight=self.config.splitter_sparsity_weight,
                entropy_mode='minimize',
            )

            if aux_losses:
                # 累加所有辅助损失
                splitter_loss = sum(aux_losses.values())

                # 确保是 float32 类型（AMP 兼容性）
                if splitter_loss.dtype != torch.float32:
                    splitter_loss = splitter_loss.float()

                return splitter_loss
        except Exception as e:
            # 忽略辅助损失计算错误
            self.logger.debug(f"Splitter loss computation failed: {e}")

        return None

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算细粒度分类损失

        Args:
            logits: 模型输出 [B, C]
            labels: 标签 [B]
            features: 特征向量 [B, D]（用于 Center Loss）
            attention_weights: 注意力权重（用于熵正则化）

        Returns:
            loss: 总损失
            stats: 损失分量统计
        """
        return self.loss_fn(logits, labels, features, attention_weights)

    def train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int = 1,
    ) -> Tuple[float, float, Dict[str, float]]:
        """训练一个 epoch

        数学形式化:
            梯度累积:
                for batch i in accum_steps:
                    loss_i = L(x_i, y_i) / accum_steps
                    loss_i.backward()

                optimizer.step()  # 每 accum_steps 步更新一次

        Args:
            loader: 数据加载器
            optimizer: 主模型优化器

        Returns:
            avg_loss: 平均损失
            accuracy: 训练准确率
            stats: 统计信息
        """
        self.model.train()

        # A21: 温度退火由 _handle_warmup 在每个 epoch 开始时自动处理
        # 不再需要手动温度计算

        total_loss = 0.0
        total_center_loss = 0.0
        correct = 0
        total = 0

        accum_steps = self.config.accum_steps
        grad_norm = 0.0

        for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc="Training", leave=False)):
            imgs = imgs.to(self.device, non_blocking=True)
            # I78: 保持 channels-last 格式
            if self.config.use_channels_last and imgs.dim() == 4:
                imgs = imgs.to(memory_format=torch.channels_last)
            labels = labels.to(self.device, non_blocking=True)

            # 累积归一化
            scale_factor = 1.0 / accum_steps

            with autocast(device_type=self.device.type, enabled=self.config.use_amp):
                # 单一接口: forward() 返回 TrainingStats 或 Tensor
                stats = self.model(imgs)
                logits = stats.logits if hasattr(stats, 'logits') else stats

                if self.config.use_center_loss:
                    features = stats.features
                    loss, loss_stats = self.compute_loss(logits, labels, features=features)
                else:
                    loss, loss_stats = self.compute_loss(logits, labels)

                # I142-1: 添加 splitter 辅助损失（与 train_fractal_vit.py 对齐）
                # P-OPT: 预先计算 aux_weight，如果为 0 则跳过 splitter_loss 计算
                # I-CURRICULUM: 三阶段课程学习权重控制
                #   Stage 1-2: aux_weight = 0 (纯分类学习)
                #   Stage 3: aux_weight = warmup * max_weight (资源共适应)
                # 获取课程阶段
                curriculum_stage = 1
                if hasattr(self.model, 'get_curriculum_stage'):
                    curriculum_stage = self.model.get_curriculum_stage()
                elif hasattr(self.model, 'tokenizer') and hasattr(self.model.tokenizer, 'get_curriculum_stage'):
                    curriculum_stage = self.model.tokenizer.get_curriculum_stage()

                # 三阶段独立权重配置
                STAGE_TOKEN_WEIGHTS = {1: 0.0, 2: 0.0, 3: 1.0}
                STAGE_ENTROPY_WEIGHTS = {1: 0.0, 2: 0.0, 3: 0.1}
                stage_max_weight = max(
                    STAGE_TOKEN_WEIGHTS.get(curriculum_stage, 0.0),
                    STAGE_ENTROPY_WEIGHTS.get(curriculum_stage, 0.0)
                )

                # I-CURRICULUM 权重控制 - 预先计算
                if epoch < 20:
                    aux_weight = 0.0
                else:
                    # 保护锁：val_acc < 5% 时不施加资源惩罚
                    val_acc = getattr(self, '_current_val_acc', 0.0)
                    if val_acc < 5.0:
                        aux_weight = 0.0
                    else:
                        warmup_progress = min(1.0, (epoch - 20) / 10.0)
                        aux_weight = warmup_progress * stage_max_weight

                # P-OPT: 仅在 aux_weight > 0 时计算 splitter_loss
                if aux_weight > 0:
                    splitter_loss = self._compute_splitter_loss(imgs, stats)
                    if splitter_loss is not None:
                        loss = loss + splitter_loss * aux_weight

                # 累积归一化
                loss = loss * scale_factor

            # 反向传播
            self.scaler.scale(loss).backward()

            # 梯度累积步数检查
            if (batch_idx + 1) % accum_steps == 0:
                # I121-6: 动态梯度裁剪
                if self.config.gradient_clip_norm is not None:
                    # torch.compile + AMP 可能导致梯度为 FP16，GradScaler 需要 FP32 梯度
                    for param in self.model.parameters():
                        if param.grad is not None and param.grad.dtype == torch.float16:
                            param.grad.data = param.grad.data.float()
                    self.scaler.unscale_(optimizer)
                    dynamic_clip_norm = self._get_dynamic_clip_norm()
                    grad_norm_curr = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        dynamic_clip_norm
                    )
                    grad_norm = grad_norm_curr.item()

                # 更新主模型
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # 更新 Center Loss 优化器
                if self.center_optimizer is not None:
                    self.center_optimizer.step()
                    self.center_optimizer.zero_grad()

                self.state.global_step += 1

            # 统计（恢复原始损失值）
            total_loss += loss.item() / scale_factor
            total_center_loss += stats.get('center_loss', 0.0) * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = 100.0 * correct / max(total, 1)
        avg_center_loss = total_center_loss / max(total, 1)

        stats = {
            'center_loss': avg_center_loss,
            'grad_norm': grad_norm,
        }

        return avg_loss, accuracy, stats

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        return_features: bool = False,
    ) -> CUB200EvalResult:
        """评估模型

        注意: 验证阶段禁用 AMP 以确保指标精度

        计算指标:
            - Top-1 / Top-5 准确率
            - 逐类别准确率 (MCA)
            - 混淆矩阵分析
            - 类内/类间距离比

        Args:
            loader: 数据加载器
            return_features: 是否返回特征用于分析

        Returns:
            CUB200EvalResult 包含详细评估结果
        """
        self.model.eval()

        # 修复: 评估时禁用 persistent_workers 以避免多进程兼容性问题
        # 使用安全的数据加载方式
        try:
            # 尝试获取数据集
            eval_dataset = loader.dataset
            # 创建评估用的 DataLoader (禁用多进程)
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.config.batch_size,
                num_workers=0,  # 评估时禁用多进程
                pin_memory=True,
                shuffle=False,
                drop_last=False,
            )
            loader = eval_loader
        except Exception as e:
            self.logger.warning(f"无法创建安全的评估 DataLoader: {e}，使用原始 loader")
            # 如果失败，保持原样

        total_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        # 逐类别统计
        class_correct = torch.zeros(self.num_classes, device=self.device)
        class_total = torch.zeros(self.num_classes, device=self.device)

        # 混淆矩阵
        confusion = torch.zeros(
            self.num_classes, self.num_classes,
            device=self.device, dtype=torch.long
        )

        # 特征收集
        all_features = []
        all_labels = []

        for batch in tqdm(loader, desc="Evaluating", leave=False):
            imgs, labels = batch
            imgs = imgs.to(self.device, non_blocking=True)
            # I78: 保持 channels-last 格式
            if self.config.use_channels_last and imgs.dim() == 4:
                imgs = imgs.to(memory_format=torch.channels_last)
            labels = labels.to(self.device, non_blocking=True)

            # 验证禁用 AMP
            with autocast(device_type=self.device.type, enabled=False):
                # 单一接口: forward() 返回 TrainingStats 或 Tensor
                stats = self.model(imgs)
                outs = stats.logits if hasattr(stats, 'logits') else stats

                if return_features:
                    all_features.append(stats.features.cpu())
                    all_labels.append(labels.cpu())

                # 检查 NaN/Inf
                if torch.isnan(outs).any() or torch.isinf(outs).any():
                    self.logger.warning("检测到 NaN/Inf，跳过该批次")
                    continue

                loss = F.cross_entropy(outs, labels)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item() * labels.size(0)

            # Top-1 准确率
            preds = outs.argmax(dim=1)
            correct_top1 += (preds == labels).sum().item()

            # Top-5 准确率
            _, top5_preds = outs.topk(5, dim=1)
            correct_top5 += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)

            # 逐类别统计 (FIX: 向量化使用 bincount，避免 Python 循环)
            labels_long = labels.long()
            preds_long = preds.long()
            # _class_total[c] = count of samples with target class c
            class_total += torch.bincount(labels_long, minlength=self.num_classes)
            # 修复: 只统计正确预测的类别索引，避免错误预测被计入类别 0
            correct = preds_long == labels_long
            correct_class_indices = labels_long[correct]  # 只取正确预测对应的类别
            class_correct += torch.bincount(correct_class_indices, minlength=self.num_classes)

            # 混淆矩阵
            for pred, label in zip(preds, labels):
                confusion[label, pred] += 1

        # 计算指标
        avg_loss = total_loss / max(total, 1)
        top1_acc = 100.0 * correct_top1 / max(total, 1)
        top5_acc = 100.0 * correct_top5 / max(total, 1)

        # 逐类别准确率 (I145: 向量化计算，避免 Python 循环)
        # 原始实现:
        # for c in range(self.num_classes):
        #     if class_total[c] > 0:
        #         acc = (class_correct[c] / class_total[c]).item() * 100
        #         per_class_acc[c] = acc
        #     else:
        #         per_class_acc[c] = 0.0
        # 向量化实现:
        per_class_acc = None
        mca = None
        if self.config.compute_per_class:
            # 计算所有类别的准确率 [C]
            # clamp class_total to avoid division by zero
            class_total_safe = class_total.clamp(min=1)
            class_acc_tensor = (class_correct / class_total_safe) * 100  # [C]

            # 转换为字典
            per_class_acc = {c: class_acc_tensor[c].item() for c in range(self.num_classes)}

            # 只对有效类别计算 MCA (class_total > 0)
            valid_mask = class_total > 0
            mca = class_acc_tensor[valid_mask].mean().item()

        # 混淆对
        confused_pairs = None
        if self.config.compute_confusion:
            confused_pairs = self._find_confused_pairs(confusion)

        # 类内/类间距离
        intra_inter_ratio = None
        if return_features and len(all_features) > 0:
            features = torch.cat(all_features, dim=0)
            labels = torch.cat(all_labels, dim=0)
            intra_inter_ratio = self._compute_intra_inter_ratio(features, labels)

        return CUB200EvalResult(
            loss=avg_loss,
            accuracy=top1_acc,
            top5_accuracy=top5_acc,
            per_class_accuracy=per_class_acc,
            confused_pairs=confused_pairs,
            mca=mca,
            intra_inter_ratio=intra_inter_ratio,
        )

    def _find_confused_pairs(
        self,
        confusion: torch.Tensor,
        top_k: int = 10,
    ) -> List[Tuple[int, int, int]]:
        """找到最易混淆的类别对

        Args:
            confusion: 混淆矩阵 [C, C]
            top_k: 返回前 k 个

        Returns:
            List of (true_class, pred_class, count)
        """
        conf = confusion.clone()
        conf.fill_diagonal_(0)  # 移除对角线

        pairs = []
        flat_conf = conf.view(-1)
        top_values, top_indices = flat_conf.topk(top_k)

        for idx, val in zip(top_indices, top_values):
            if val.item() == 0:
                break
            true_class = (idx // self.num_classes).item()
            pred_class = (idx % self.num_classes).item()
            pairs.append((true_class, pred_class, val.item()))

        return pairs

    def _compute_intra_inter_ratio(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """计算类内/类间距离比

        数学公式:
            intra = (1/N) Σ_i ||f_i - c_{y_i}||²  (类内距离)
            inter = (1/C(C-1)/2) Σ_{j<k} ||c_j - c_k||²  (类间距离)

        Args:
            features: 特征 [N, D]
            labels: 标签 [N]

        Returns:
            intra/inter 比率（越小越好）
        """
        unique_labels = labels.unique()
        num_classes = len(unique_labels)
        feat_dim = features.shape[1]

        # 计算类中心
        centers = torch.zeros(num_classes, feat_dim, device=features.device)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            centers[i] = features[mask].mean(dim=0)

        # 类内距离
        intra_dist = 0.0
        for i, label in enumerate(unique_labels):
            mask = labels == label
            class_features = features[mask]
            center = centers[i]
            intra_dist += ((class_features - center) ** 2).sum(dim=1).mean()

        intra_dist = intra_dist / num_classes

        # 类间距离
        inter_dist = 0.0
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                inter_dist += ((centers[i] - centers[j]) ** 2).sum()

        inter_dist = inter_dist / (num_classes * (num_classes - 1) / 2)

        # 返回比率
        return intra_dist / max(inter_dist, 1e-8)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        exp_dir: Optional[Path] = None,
        writer: Optional[Any] = None,  # I111-6: TensorBoard writer
    ) -> Dict[str, List[float]]:
        """完整训练流程

        数学形式化:
            for epoch in 1..num_epochs:
                lr = get_warmup_lr(epoch)
                set_optimizer_lr(optimizer, lr)

                train_loss, train_acc = train_epoch()

                if epoch % validate_interval == 0:
                    val_loss, val_acc = evaluate()

                    if acc > best_acc + min_delta:
                        save_checkpoint()
                        best_acc = acc
                        patience_counter = 0
                    else:
                        patience_counter += 1

                if patience_counter >= patience:
                    break

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 主模型优化器
            scheduler: 学习率调度器（可选）
            exp_dir: 实验目录（可选）

        Returns:
            训练历史记录
        """
        # 初始化 Center Loss 优化器
        self.initialize_center_optimizer(optimizer)

        # A21: 配置训练参数 (使用模型协议接口)
        # I36-2: 使用 configure_training 替代旧的 _setup_temperature_annealing
        self._configure_training_via_protocol(train_loader)

        # 创建检查点目录
        checkpoint_dir = None
        if exp_dir is not None:
            checkpoint_dir = exp_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"检查点目录: {checkpoint_dir}")

        # 记录初始学习率
        initial_lr = optimizer.param_groups[0]['lr']

        for epoch in range(1, self.config.num_epochs + 1):
            self.state.epoch = epoch

            # A21: 处理 warmup 期间的温度/偏置
            self._handle_warmup(epoch)

            epoch_start = time.time()

            # 学习率预热
            warmup_lr = self._get_warmup_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

            # 训练
            train_loss, train_acc, train_stats = self.train_epoch(train_loader, optimizer, epoch)

            # 验证（每 N 个 epoch）
            val_result = None
            if epoch % self.config.validate_interval == 0:
                val_result = self.evaluate(val_loader)

                # 更新最佳指标
                if val_result.accuracy > self.state.best_metric + self.config.min_delta:
                    self.state.best_metric = val_result.accuracy
                    self.state.best_epoch = epoch
                    self.state.patience_counter = 0

                    # 保存最佳检查点
                    if checkpoint_dir is not None:
                        self._save_checkpoint(
                            checkpoint_dir / "best.pth",
                            optimizer,
                            epoch,
                            val_result.accuracy,
                        )
                        self.logger.info(f"保存最佳模型: val_acc={val_result.accuracy:.2f}%")
                else:
                    self.state.patience_counter += 1

                self.state.val_loss = val_result.loss
                self.state.val_accuracy = val_result.accuracy
                # I-CURRICULUM: 跟踪验证准确率用于课程权重控制
                self._current_val_acc = val_result.accuracy
            else:
                self.state.patience_counter += 1

            # 学习率调度
            if scheduler is not None:
                scheduler.step()

            epoch_time = time.time() - epoch_start

            # 更新历史
            self.state.history["train_loss"].append(train_loss)
            self.state.history["train_accuracy"].append(train_acc)
            self.state.history["learning_rate"].append(warmup_lr)
            self.state.history["center_loss"].append(train_stats.get('center_loss', 0.0))

            if val_result is not None:
                self.state.history["val_loss"].append(val_result.loss)
                self.state.history["val_accuracy"].append(val_result.accuracy)
                self.state.history["val_top5_accuracy"].append(val_result.top5_accuracy)

            # 日志输出
            current_lr = optimizer.param_groups[0]['lr']
            msg = (f"Epoch {epoch}/{self.config.num_epochs} | "
                   f"Time: {epoch_time:.1f}s | "
                   f"LR: {current_lr:.2e} | "
                   f"Train: loss={train_loss:.4f}, acc={train_acc:.2f}%")

            if val_result is not None:
                msg += f" | Val: loss={val_result.loss:.4f}, acc={val_result.accuracy:.2f}%, top5={val_result.top5_accuracy:.2f}%"

            self.logger.info(msg)

            # I111-6: 记录深度分布到 TensorBoard
            if writer is not None:
                self._log_depth_monitor_to_tensorboard(writer, epoch)

            # 早停检查
            if self.state.patience_counter >= self.config.patience:
                self.logger.info(f"早停触发: 连续 {self.config.patience} 个 epoch 无改善")
                break

            # 保存周期检查点
            if checkpoint_dir is not None and epoch % 10 == 0:
                self._save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch}.pth",
                    optimizer,
                    epoch,
                    val_result.accuracy if val_result else 0.0,
                )

        # 保存最终检查点
        if checkpoint_dir is not None:
            self._save_checkpoint(
                checkpoint_dir / "last.pth",
                optimizer,
                self.state.epoch,
                self.state.best_metric,
            )

        # I36-3: 收集最终诊断信息
        final_diagnostics = self._collect_splitter_diagnostics()
        if final_diagnostics:
            self.logger.info(f"[I36-3] 最终 Splitter 诊断:")
            for key, value in final_diagnostics.items():
                if isinstance(value, float):
                    self.logger.info(f"  {key}: {value:.4f}")
                else:
                    self.logger.info(f"  {key}: {value}")

        self.logger.info(f"训练完成! 最佳准确率: {self.state.best_metric:.2f}% (epoch {self.state.best_epoch})")

        return self.state.history

    def _save_checkpoint(
        self,
        path: Path,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_acc: float,
    ) -> None:
        """保存检查点 (使用 ModelGene 自包含格式)

        使用 ModelGene.from_config() 直接从配置构造，确保：
        1. 单一数据源 - 训练器持有配置，直接从配置构造
        2. 避免从模型提取 - 训练器已知所有参数，无需重复查询模型
        3. 变参数 (max_level) 不传入，由模型架构内部计算
        """
        # 直接从配置构造 ModelGene（单一数据源）
        gene = ModelGene.from_config(
            self.config.arch_config,
            dataset_name='cub200',
            epoch=epoch,
        )

        # 使用共享的 checkpoint 保存函数
        save_checkpoint_with_gene(
            path=path,
            model=self.model,
            gene=gene,
            optimizer_state=optimizer.state_dict(),
            epoch=epoch,
            val_acc=val_acc,
            extra={
                # Center Loss 状态
                'center_lr': self._center_lr if hasattr(self, '_center_lr') else None,
            }
        )

        self.logger.debug(f"检查点已保存: {path}")

    def load_checkpoint(
        self,
        path: Path,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[int, float]:
        """加载检查点

        Args:
            path: 检查点路径
            optimizer: 优化器（用于恢复状态）

        Returns:
            epoch: 恢复的 epoch
            best_metric: 最佳准确率
        """
        # 使用共享的 checkpoint 加载函数（与 eval.py 一致）
        from ..core.checkpoint import load_checkpoint as load_checkpoint_func
        from ..core.model_gene import ModelGene

        checkpoint = load_checkpoint_func(path, device=str(self.device))

        # 验证 model_gene 存在（与 eval.py 一致）
        if 'model_gene' not in checkpoint:
            raise ValueError(
                f"Checkpoint does not contain 'model_gene' key.\n"
                f"This checkpoint was saved with an older version.\n"
                f"Please retrain with the updated training script."
            )

        # 验证配置一致性
        loaded_gene = ModelGene.from_dict(checkpoint['model_gene'])
        self._validate_gene_consistency(loaded_gene)

        # 恢复模型
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 恢复优化器
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 恢复 scaler
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # 恢复 Center Loss 优化器
        if self.center_optimizer is not None and 'center_optimizer_state_dict' in checkpoint:
            self.center_optimizer.load_state_dict(checkpoint['center_optimizer_state_dict'])

        # 恢复状态
        epoch = checkpoint.get('epoch', 0)
        best_metric = checkpoint.get('best_metric', 0.0)
        best_epoch = checkpoint.get('best_epoch', 0)

        self.state.epoch = epoch
        self.state.best_metric = best_metric
        self.state.best_epoch = best_epoch

        self.logger.info(f"恢复检查点: epoch={epoch}, best_acc={best_metric:.2f}%")

        return epoch, best_metric

    def print_eval_summary(
        self,
        result: CUB200EvalResult,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """打印评估摘要"""
        print("\n" + "=" * 60)
        print("CUB-200-2011 细粒度分类评估结果")
        print("=" * 60)
        print(f"  Loss:          {result.loss:.4f}")
        print(f"  Top-1 Acc:     {result.accuracy:.2f}%")
        print(f"  Top-5 Acc:     {result.top5_accuracy:.2f}%")
        if result.mca is not None:
            print(f"  MCA:           {result.mca:.2f}%")
        if result.intra_inter_ratio is not None:
            print(f"  Intra/Inter:   {result.intra_inter_ratio:.4f}")

        if result.per_class_accuracy:
            sorted_classes = sorted(
                result.per_class_accuracy.items(),
                key=lambda x: x[1]
            )
            print("\n  最差的 5 个类别:")
            for cls_id, acc in sorted_classes[:5]:
                name = class_names[cls_id] if class_names else f"Class {cls_id}"
                print(f"    {name}: {acc:.1f}%")

        if result.confused_pairs:
            print("\n  最易混淆的类别对:")
            for true_cls, pred_cls, count in result.confused_pairs[:5]:
                true_name = class_names[true_cls] if class_names else f"Class {true_cls}"
                pred_name = class_names[pred_cls] if class_names else f"Class {pred_cls}"
                print(f"    {true_name} → {pred_name}: {count} 次")

        print("=" * 60 + "\n")


def get_cub200_augmentation(
    image_size: int = 224,
    is_training: bool = True,
    config: Optional[CUB200TrainingConfig] = None,
) -> "transforms.Compose":
    """获取 CUB-200 细粒度分类专用数据增强

    细粒度分类增强策略:
        1. 较大的 RandomResizedCrop scale 范围 → 学习不同尺度特征
        2. 适度的颜色增强 → 保留判别性颜色特征
        3. 较小的旋转角度 → 鸟类姿态敏感
        4. 不使用垂直翻转 → 鸟类通常不倒立

    Args:
        image_size: 目标图像尺寸
        is_training: 是否训练模式
        config: 训练配置

    Returns:
        torchvision transforms 组合
    """
    from torchvision import transforms

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if is_training:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.5, 1.0),
                ratio=(0.75, 1.33),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.RandAugment(num_ops=2, magnitude=6),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def create_cub200_trainer(
    config: Optional[CUB200TrainingConfig] = None,
    arch_config: Optional[ModelArchitectureConfig] = None,
    feat_dim: int = 256,
    data_dir: str = "./data",
    output_dir: str = "./checkpoints",
    **kwargs,
) -> Tuple[CUB200Trainer, 'FractalCurveViT']:
    """创建 CUB-200 训练器的便捷工厂函数 (I36 统一配置)

    设计原则:
        1. 通过 arch_config 统一指定模型架构参数
        2. 通过 config 指定训练策略参数
        3. 自动创建模型和训练器

    Args:
        config: 训练配置 (可选，默认使用 CUB200TrainingConfig)
        arch_config: 模型架构配置 (可选，默认使用 ModelArchitectureConfig)
        feat_dim: Center Loss 特征维度
        data_dir: 数据集目录
        output_dir: 输出目录
        **kwargs: 传递给 config 的参数（会覆盖默认值）

    Returns:
        (trainer, model) 元组

    用法:
        ```python
        from training.trainer import create_cub200_trainer
        from training.config import ModelArchitectureConfig

        # 推荐方式：使用统一的架构配置
        arch_config = ModelArchitectureConfig(
            num_classes=200,
            dim=384,
            num_layers=8,
            heads=6,
        )
        trainer, model = create_cub200_trainer(
            arch_config=arch_config,
            batch_size=16,
            learning_rate=0.0003,
        )
        ```
    """
    # 合并配置
    if arch_config is not None:
        # 合并 arch_config 到 kwargs
        arch_dict = {
            'num_classes': arch_config.num_classes,
            'dim': arch_config.dim,
            'num_layers': arch_config.num_layers,  # I145: depth -> num_layers
            'heads': arch_config.heads,
            'mlp_dim': arch_config.mlp_dim,
            'image_size': arch_config.image_size,
            'min_patch_size': arch_config.min_patch_size,
            'use_area_encoding': arch_config.use_area_encoding,
            'use_affine_modulation': arch_config.use_affine_modulation,
            'fourier_levels': arch_config.fourier_levels,
            'ffn_type': arch_config.ffn_type,  # P1: 添加缺失的 ffn_type
        }
        kwargs.update(arch_dict)

    # 创建训练配置
    if config is None:
        config = CUB200TrainingConfig(**kwargs)

    # P1 Fix: 验证配置一致性
    config.validate_model_config_consistency()

    # I36: 使用 config.arch_config 创建模型
    # P5-FIX: 补充所有缺失参数，与 train_fractal_vit.py 保持一致
    model = FractalCurveViT(
        num_classes=config.arch_config.num_classes,
        dim=config.arch_config.dim,
        num_layers=config.arch_config.num_layers,  # I145: depth -> num_layers
        heads=config.arch_config.heads,
        mlp_dim=config.arch_config.mlp_dim,
        image_size=config.arch_config.image_size,
        min_patch_size=config.arch_config.min_patch_size,
        # 编码配置
        use_area_encoding=config.arch_config.use_area_encoding,
        use_affine_modulation=config.arch_config.use_affine_modulation,
        fourier_levels=config.arch_config.fourier_levels,
        use_hilbert_encoding=config.arch_config.use_hilbert_encoding,
        use_spatial_encoding=config.arch_config.use_spatial_encoding,
        ffn_type=config.arch_config.ffn_type,
        # Dropout 配置 (P5-FIX: 补充所有 dropout 参数)
        dropout=config.dropout,
        drop_path_rate=config.drop_path_rate,
        tokenizer_dropout=config.arch_config.tokenizer_dropout,
        transformer_dropout=config.arch_config.transformer_dropout,
        emb_dropout=config.arch_config.emb_dropout,
        # Splitter 配置 (P5-FIX: 补充 splitter 参数)
        use_checkpoint=config.use_checkpoint,
        splitter_type=config.arch_config.splitter_type,
        splitter_temp_start=config.arch_config.splitter_temp_start,
        splitter_temp_end=config.arch_config.splitter_temp_end,
        quota_learnable=config.arch_config.quota_learnable,
        quota_entropy_weight=config.arch_config.quota_entropy_weight,
    )

    # 创庺训练器
    trainer = CUB200Trainer(
        model=model,
        config=config,
        num_classes=config.arch_config.num_classes,
        feat_dim=feat_dim,
    )

    return trainer, model


# ============================================================================
# I36: CUB200Trainer 继承版本 (基于 ModularTrainer)
# ============================================================================

def _import_modular_trainer():
    """延迟导入 ModularTrainer，避免循环导入"""
    from . import ModularTrainer, TrainerConfig
    return ModularTrainer, TrainerConfig


class CUB200ModularTrainer:
    """CUB-200 细粒度分类训练器 - 基于 ModularTrainer (I36)

    设计原则:
        1. 复用 ModularTrainer 的通用组件（优化器、调度器、Callback）
        2. 通过 FractalModelProtocol 配置模型，不直接访问内部实现
        3. 保持 CUB-200 特有功能（Center Loss、细粒度评估）

    数学形式:
        总损失: L_total = L_ce + λ_center × L_center + λ_aux × L_aux

    用法:
        ```python
        from training.trainer import CUB200ModularTrainer, CUB200TrainingConfig
        from vit_pytorch import FractalCurveViT

        model = FractalCurveViT(num_classes=200, dim=384, num_layers=8, heads=6)
        config = CUB200TrainingConfig(
            batch_size=16,
            learning_rate=0.0003,
            depth_kl_weight=0.1,
        )

        trainer = CUB200ModularTrainer(model, config)
        trainer.fit()
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        config: 'CUB200TrainingConfig',
        train_loader: DataLoader,
        val_loader: DataLoader,
        feat_dim: int = 256,
    ):
        """
        Args:
            model: FractalCurveViT 模型（需实现 FractalModelProtocol）
            config: CUB200TrainingConfig 配置
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            feat_dim: Center Loss 特征维度
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.feat_dim = feat_dim
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # 延迟导入 ModularTrainer
        ModularTrainer, TrainerConfig = _import_modular_trainer()

        # 计算温度退火总步数 (P0-3 修复: 必须显式传递 total_steps)
        batches_per_epoch = len(train_loader) // config.accum_steps
        post_warmup_steps = max(1, (config.num_epochs - config.splitter_temp_warmup) * batches_per_epoch)

        # 1. 通过 FractalModelProtocol 配置模型 (I36-2 解耦)
        self.model.configure_training({
            'temperature_annealing': True,
            'total_steps': post_warmup_steps,  # P0-3: 显式传递正确值
            'temp_start': config.splitter_temp_start,
            'temp_end': config.splitter_temp_end,
            'aux_loss_weights': {
                'sparsity': config.splitter_sparsity_weight,
                'elastic': config.elastic_budget_weight,
                'depth_kl': config.depth_kl_weight,
            }
        })

        # 2. 初始化 ModularTrainer 通用组件
        trainer_config = TrainerConfig(
            device=self.device.type,
            num_epochs=config.num_epochs,
            gradient_clip_norm=config.gradient_clip_norm,
            accumulation_steps=config.accum_steps,
            validate_interval=config.validate_interval,
            checkpoint_dir=config.checkpoint_dir,
            save_best_only=True,
            monitor_metric="val_accuracy",
            monitor_mode="max",
            use_amp=config.use_amp,
        )

        # 优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # 损失函数
        loss_fn = FinegrainedLoss(config=FinegrainedLossConfig(
            num_classes=config.num_classes,
            label_smoothing=config.label_smoothing,
        ))

        # 初始化 ModularTrainer
        self.trainer = ModularTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=trainer_config,
        )

        # 3. CUB-200 特有组件 (Center Loss)
        self._setup_center_loss(feat_dim)

    def _setup_center_loss(self, feat_dim: int):
        """设置 Center Loss"""
        self.center_loss_fn = nn.CrossEntropyLoss()
        self.center_features: List[torch.Tensor] = []
        self.center_labels: List[torch.Tensor] = []

        # Center Loss 参数
        self._center_lr = self.config.center_lr or (self.config.learning_rate / self.config.center_lr_ratio)

        # 可学习的类别中心
        self.register_buffer('center_bank', torch.zeros(self.config.num_classes, feat_dim))

        # Center 优化器
        self.center_optimizer = torch.optim.SGD(
            [self.center_bank],
            lr=self._center_lr,
        )

    def fit(self) -> Dict[str, List[float]]:
        """完整训练流程

        Returns:
            训练历史记录
        """
        return self.trainer.fit()

    def evaluate(self) -> CUB200EvalResult:
        """评估模型

        Returns:
            CUB200EvalResult 评估结果
        """
        return self.trainer.validate()
