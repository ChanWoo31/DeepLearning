# evaluate_test.py

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader

#####################################
# 1. 설정 부분 (필요에 따라 수정)
#####################################

# 1) 학습할 때 사용한 클래스 순서와 동일한 label 순서여야 합니다.
#    train_full.py에서 ImageFolder로 load된 순서가 class_names입니다.
#    예: ['due_down','due_up','opencr_down','opencr_up','uno_down','uno_up']
CLASS_NAMES = ['due_down', 'due_up', 'opencr_down', 'opencr_up', 'uno_down', 'uno_up']

# 2) 테스트 전용 이미지가 들어 있는 최상위 폴더
TEST_DATA_ROOT = os.path.join(os.getcwd(), "test_data")

# 3) 배치 크기
BATCH_SIZE = 32

# 4) GPU 사용 여부
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 디바이스:", DEVICE)

#####################################
# 2. 테스트 데이터 전처리 정의 (Validation과 동일)
#####################################

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

#####################################
# 3. 테스트 데이터 로드
#####################################

# ImageFolder는 하위 폴더명을 클래스 레이블로 사용
test_dataset = datasets.ImageFolder(root=TEST_DATA_ROOT, transform=test_transform)
test_loader  = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print("테스트 전체 샘플 수:", len(test_dataset))
print("테스트 클래스별 샘플 수:", {cls: sum(1 for _, lbl in test_dataset.samples if test_dataset.classes[lbl] == cls) for cls in CLASS_NAMES})

#####################################
# 4. 모델 로드 (ResNet-18, 학습된 best_model.pth)
#####################################

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

#####################################
# 5. 테스트 루프 (정확도 계산)
#####################################

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total if total > 0 else 0
print(f"테스트 정확도: {test_acc * 100:.2f}% ({correct}/{total})")
