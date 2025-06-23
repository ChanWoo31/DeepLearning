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

# 1. 설정 부분 

# 1) 학습/검증에 사용할 데이터 경로
DATA_ROOT = os.path.join(os.getcwd(), "data")

# 2) 배치 크기, epoch 수
BATCH_SIZE = 32
NUM_EPOCHS = 15

# 3) train/val 비율 (클래스당 80% train, 20% val)
TRAIN_RATIO = 0.8

# 4) 학습률, 모멘텀, 가중치 감쇠
LEARNING_RATE = 0.0005
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-3

# 5) 랜덤 시드 (재현성 확보)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 6) GPU 사용 여부
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 디바이스:", DEVICE)

# 2. 데이터 전처리 및 증강(Transforms) 정의

# Train 데이터에는 강력한 증강(Augmentation) 적용
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4959222674369812, 0.5355183482170105, 0.5268104076385498],
        std=[0.2755642533302307, 0.22989842295646667, 0.22611673176288605])
])

# Validation 데이터는 중앙 크롭만 적용
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4959222674369812, 0.5355183482170105, 0.5268104076385498],
        std=[0.2755642533302307, 0.22989842295646667, 0.22611673176288605])
])

# 3. 전체 데이터셋(ImageFolder) 로드 (train/val용 분리 전)

# 임시 transform으로 전체 샘플 수와 클래스별 개수만 확인
temp_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
base_dataset = datasets.ImageFolder(root=DATA_ROOT, transform=temp_transform)
class_names = base_dataset.classes
num_classes = len(class_names)
print("클래스 목록:", class_names)

# 전체 샘플 수 및 클래스별 개수 집계
total_samples = len(base_dataset)
counts_per_class = [0] * num_classes
for _, lbl in base_dataset.samples:
    counts_per_class[lbl] += 1
print("전체 샘플 수:", total_samples)
print("클래스별 샘플 개수:", dict(zip(class_names, counts_per_class)))

# 4. 클래스별 인덱스 수집 → Stratified Split

# 4-1) 클래스별 인덱스를 모은다
indices_per_class = [[] for _ in range(num_classes)]
for idx, (_, lbl) in enumerate(base_dataset.samples):
    indices_per_class[lbl].append(idx)

train_indices = []
val_indices = []

# 4-2) 클래스별로 shuffle 후 80:20 비율로 나눈다
for lbl, idx_list in enumerate(indices_per_class):
    np.random.shuffle(idx_list)
    split = int(len(idx_list) * TRAIN_RATIO)
    train_indices += idx_list[:split]
    val_indices   += idx_list[split:]

print("훈련용 샘플 수:", len(train_indices))
print("검증용 샘플 수:", len(val_indices))

# 5. 실제 학습/검증용 Dataset & DataLoader 생성

# 5-1) ImageFolder를 둘로 만들어, train에는 train_transform, val에는 val_transform을 적용
full_dataset_for_train = datasets.ImageFolder(root=DATA_ROOT, transform=train_transform)
full_dataset_for_val   = datasets.ImageFolder(root=DATA_ROOT, transform=val_transform)

# 5-2) Subset 클래스를 이용해 인덱스 기반으로 추출
train_dataset = Subset(full_dataset_for_train, train_indices)
val_dataset   = Subset(full_dataset_for_val,   val_indices)

# 5-3) DataLoader 생성
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# 6. 클래스 가중치 계산 (불균형 보정용)

# train_indices를 사용해 클래스별 개수 재집계
train_counts = [0] * num_classes
for idx in train_indices:
    _, lbl = base_dataset.samples[idx]
    train_counts[lbl] += 1

print("훈련용 클래스별 샘플 개수:", dict(zip(class_names, train_counts)))

total_train = float(len(train_indices))
class_weights = [total_train / cnt for cnt in train_counts]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print("클래스 가중치:", dict(zip(class_names, class_weights)))

# 7. 모델 정의 (ResNet-18) 및 학습 설정

# 7-1) weights 인자를 사용하여 경고 없이 Pretrained 모델 로드
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(DEVICE)

# 7-2) 손실 함수 및 Optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY
)

# 7-3) 학습률 스케줄러 (5 에폭마다 학습률을 0.1배 감소)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 8. 학습(Train) 및 검증(Validation) 함수 정의

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# 9. 실제 학습 루프 실행 (Early Stopping 추가)

best_val_acc = 0.0
patience = 3
counter = 0

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc     = validate(model, val_loader,   criterion, DEVICE)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        counter = 0
        print(f"  → 새로운 최고 검증 정확도: {best_val_acc:.4f}, 모델 저장됨")
    else:
        counter += 1
        if counter >= patience:
            print(f"검증 성능 개선 없음 {patience}회 연속, 학습 조기 종료 (Epoch {epoch+1})")
            break

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    scheduler.step()

print("학습 완료!")