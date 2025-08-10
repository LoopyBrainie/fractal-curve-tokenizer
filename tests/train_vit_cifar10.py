import os
import tarfile

# 自动解压CIFAR-10数据集
data_dir = "workspace"
tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
extract_dir = os.path.join(data_dir, "cifar-10-batches-py")
if not os.path.exists(extract_dir):
    if os.path.exists(tar_path):
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        print("数据集已自动解压")
    else:
        raise FileNotFoundError(f"未找到 {tar_path}，请检查数据集文件")
import torch
import datetime
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from vit_pytorch import ViT
import torch.nn as nn

from torch.optim.adam import Adam

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
])

# 加载CIFAR-10训练集
train_dataset = CIFAR10(
    root="workspace",
    train=True,
    download=False,
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 初始化ViT模型
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

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 训练多个epoch，并在每一轮结束后输出状态
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (batch_idx + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} 完成，平均Loss: {avg_loss:.4f}")

# 保存模型，命名包含训练开始日期和序号
date_str = datetime.datetime.now().strftime("%Y%m%d")
base_name = f"vit_cifar10_{date_str}"
idx = 1
while True:
    ckpt_name = f"{base_name}_{idx}.pth"
    if not os.path.exists(ckpt_name):
        break
    idx += 1
torch.save(model.state_dict(), ckpt_name)
print(f"训练完成，模型已保存为: {ckpt_name}")
