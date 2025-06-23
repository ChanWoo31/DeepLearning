# train_full.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt  # ← 추가

#####################################
# 1. 설정 부분 (필요에 따라 수정)
#####################################

DATA_ROOT     = os.path.join(os.getcwd(), "data")
BATCH_SIZE    = 32
NUM_EPOCHS    = 15
TRAIN_RATIO   = 0.8
LEARNING_RATE = 0.0005
MOMENTUM      = 0.9
WEIGHT_DECAY  = 1e-3
SEED          = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 디바이스:", DEVICE)

#####################################
# 2. 데이터 전처리 및 증강 정의
#####################################
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3,0.3,0.3,0.1),
    transforms.RandomPerspective(0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4959222674369812, 0.5355183482170105, 0.5268104076385498],
        std=[0.2755642533302307, 0.22989842295646667, 0.22611673176288605]
    )
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4959222674369812, 0.5355183482170105, 0.5268104076385498],
        std=[0.2755642533302307, 0.22989842295646667, 0.22611673176288605]
    )
])

#####################################
# 3. 데이터셋 로드 & Stratified Split
#####################################
# (이전 코드와 동일)
temp_ds = datasets.ImageFolder(DATA_ROOT, transform=val_transform)
class_names = temp_ds.classes
num_classes = len(class_names)

indices_per_class = [[] for _ in range(num_classes)]
for idx, (_, lbl) in enumerate(temp_ds.samples):
    indices_per_class[lbl].append(idx)

train_indices, val_indices = [], []
for lst in indices_per_class:
    np.random.shuffle(lst)
    split = int(len(lst) * TRAIN_RATIO)
    train_indices += lst[:split]
    val_indices   += lst[split:]

full_train_ds = datasets.ImageFolder(DATA_ROOT, transform=train_transform)
full_val_ds   = datasets.ImageFolder(DATA_ROOT, transform=val_transform)
train_ds = Subset(full_train_ds, train_indices)
val_ds   = Subset(full_val_ds,   val_indices)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=True)

#####################################
# 4. 클래스 가중치 계산 (선택)
#####################################
train_counts = [0]*num_classes
for idx in train_indices:
    _, lbl = temp_ds.samples[idx]
    train_counts[lbl] += 1

total_train = float(len(train_indices))
class_weights = [ total_train/c for c in train_counts ]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

#####################################
# 5. 모델 정의 & 학습 설정
#####################################
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
in_f = model.fc.in_features
model.fc = nn.Linear(in_f, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.SGD(model.parameters(),
                      lr=LEARNING_RATE,
                      momentum=MOMENTUM,
                      weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

#####################################
# 6. 로그 저장용 리스트 초기화
#####################################
train_losses, train_accs = [], []
val_losses,   val_accs   = [], []

#####################################
# 7. 학습 및 검증 함수
#####################################
def train_one_epoch(model, loader):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*imgs.size(0)
        _, preds = out.max(1)
        correct    += (preds==lbls).sum().item()
        total      += lbls.size(0)
    return running_loss/total, correct/total

def validate(model, loader):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, lbls)
            running_loss += loss.item()*imgs.size(0)
            _, preds = out.max(1)
            correct    += (preds==lbls).sum().item()
            total      += lbls.size(0)
    return running_loss/total, correct/total

#####################################
# 8. 학습 루프 (Early Stopping)
#####################################
best_val_acc = 0.0
patience, counter = 3, 0

for epoch in range(1, NUM_EPOCHS+1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader)
    vl_loss, vl_acc = validate(model,   val_loader)
    train_losses.append(tr_loss)
    train_accs.append(tr_acc)
    val_losses.append(vl_loss)
    val_accs.append(vl_acc)

    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model.state_dict(), "best_model.pth")
        counter = 0
        print(f"→ Epoch {epoch}: New Best Val Acc = {best_val_acc:.4f}, 모델 저장")
    else:
        counter += 1
        if counter >= patience:
            print(f"EarlyStopping @ Epoch {epoch} (no improvement for {patience} epochs)")
            break

    print(f"[{epoch}/{NUM_EPOCHS}] "
          f"Train Loss={tr_loss:.4f}, Train Acc={tr_acc:.4f} | "
          f"Val Loss={vl_loss:.4f}, Val Acc={vl_acc:.4f}")
    scheduler.step()

print("학습 완료!")

#####################################
# 9. 학습 곡선 시각화 및 저장
#####################################
epochs = list(range(1, len(train_losses)+1))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

# Loss
ax1.plot(epochs, train_losses, label='Train Loss')
ax1.plot(epochs, val_losses,   label='Val Loss')
ax1.set_title('Loss Curve')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Accuracy
ax2.plot(epochs, train_accs, label='Train Acc')
ax2.plot(epochs, val_accs,   label='Val Acc')
ax2.set_title('Accuracy Curve')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
print("✅ training_curves.png 저장됨")
plt.show()
