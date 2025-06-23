# calculate_mean_std.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np

# 1) train 전용 transform (리사이즈/센터크롭만)
stats_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# 2) ImageFolder + split
base_ds = datasets.ImageFolder(root="./data", transform=stats_transform)

# Stratified split 로직 재사용
indices_per_class = [[] for _ in base_ds.classes]
for idx, (_, lbl) in enumerate(base_ds.samples):
    indices_per_class[lbl].append(idx)

train_indices, val_indices = [], []
for idx_list in indices_per_class:
    np.random.shuffle(idx_list)
    split = int(len(idx_list) * 0.8)
    train_indices += idx_list[:split]
    val_indices   += idx_list[split:]

train_ds = Subset(base_ds, train_indices)

# 3) train 전용 DataLoader
loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# 4) 통계 누적 계산
sum_   = torch.zeros(3)
sum_sq = torch.zeros(3)
n_pix  = 0

for imgs, _ in loader:
    B,C,H,W = imgs.shape
    n_pix  += B * H * W
    sum_   += imgs.sum(dim=[0,2,3])
    sum_sq += (imgs**2).sum(dim=[0,2,3])

mean = sum_ / n_pix
std  = (sum_sq / n_pix - mean**2).sqrt()

print("Computed mean:", mean.tolist())
print("Computed std: ", std.tolist())