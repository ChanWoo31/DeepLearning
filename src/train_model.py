# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# ─────────────────────────────────────────────────────────────────────────────
# 1) 데이터 준비 (이미 data_preparation.py에서 train_loader, val_loader, class_weights_tensor, full_dataset, class_names 등을 정의해 두었다고 가정)
#    아래 import 구문은 data_preparation.py 파일이 전역에 다음 변수들을 정의하고 있을 때 동작합니다:
#      - full_dataset
#      - train_loader
#      - val_loader
#      - class_weights_tensor
#      - class_names (혹은 full_dataset.classes)
# ─────────────────────────────────────────────────────────────────────────────
from data_preparation import full_dataset, train_loader, val_loader, class_weights_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 디바이스:", device)

# ─────────────────────────────────────────────────────────────────────────────
# 2) 사전학습된 ResNet-18 불러오기 + 마지막 레이어 교체 (클래스 수 = 6)
# ─────────────────────────────────────────────────────────────────────────────
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(full_dataset.classes))  # 6개 출력
model = model.to(device)

# ─────────────────────────────────────────────────────────────────────────────
# 3) 손실 함수 및 Optimizer 설정
# ─────────────────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
optimizer = optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.9,
    weight_decay=1e-4
)

# ─────────────────────────────────────────────────────────────────────────────
# 4) 학습용 함수 및 검증용 함수 정의
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# 5) 실제 학습 루프 실행
# ─────────────────────────────────────────────────────────────────────────────
num_epochs = 10
best_val_acc = 0.0

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc     = validate(model, val_loader, criterion, device)

    # ↓ 이 print 구문의 큰따옴표와 중괄호가 빠지거나 잘못 끊기면 SyntaxError가 발생합니다.
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  → 새로운 최고 검증 정확도: {best_val_acc:.4f}, 모델 저장됨")

print("학습 완료!")
