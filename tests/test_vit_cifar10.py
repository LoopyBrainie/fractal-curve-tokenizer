import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from vit_pytorch import ViT
import os
import glob

# 数据预处理：resize到ViT输入尺寸，转为tensor
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
])

# 加载CIFAR-10测试集（已解压）
dataset = CIFAR10(
    root="workspace",  # 指向包含cifar-10-batches-py的目录
    train=False,
    download=False,
    transform=transform
)

loader = DataLoader(dataset, batch_size=8, shuffle=False)

# 初始化ViT模型（参数可根据实际需求调整）
model = ViT(
    image_size=256,
    patch_size=32,
    num_classes=10,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=1024,
    dropout=0.1,
    emb_dropout=0.1
)

# 自动加载训练后模型权重

# 自动查找最新的模型权重文件
ckpt_dir = "workspace"
ckpt_list = sorted(glob.glob(os.path.join(ckpt_dir, "vit_cifar10_*.pth")), reverse=True)
if ckpt_list:
    ckpt_path = ckpt_list[0]
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    print(f"已加载最新模型权重: {ckpt_path}")
else:
    print(f"未找到模型权重 {os.path.join(ckpt_dir, 'vit_cifar10_*.pth')}，请先训练并保存模型。")

model.eval()

# 取一批数据做推理
images, labels = next(iter(loader))
with torch.no_grad():
    logits = model(images)
    preds = torch.argmax(logits, dim=1)

print("预测类别:", preds.tolist())
print("真实类别:", labels.tolist())
