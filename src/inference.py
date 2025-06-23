# inference.py

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image

#####################################
# 1. 설정 부분 (여기에 이미지 경로를 직접 지정)
#####################################

# 1) 분류할 이미지 파일 경로를 여기에 직접 작성하세요.
#    예시: due_up 사진 하나
IMAGE_PATH = "/home/han/machine_learning/inference_data/t1.jpg"

# 2) 모델 가중치 파일 경로 (기본: best_model.pth)
MODEL_PATH = "/home/han/machine_learning/best_model.pth"

# 3) 클래스 이름 목록 (ResNet-18 학습할 때 사용한 순서와 동일해야 합니다)
CLASS_NAMES = ["due_down", "due_up", "opencr_down", "opencr_up", "uno_down", "uno_up"]

#####################################
# 2. 모델 로드 함수
#####################################

def load_model(model_path: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    """
    ResNet-18 모델을 불러와서, 마지막 fc 레이어를 num_classes로 교체한 뒤,
    저장된 가중치를 로드하고 evaluation 모드로 변환하여 반환합니다.
    """
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

#####################################
# 3. 전처리(Transform) 정의
#####################################

def get_transform() -> transforms.Compose:
    """
    단일 이미지를 모델 입력에 맞게 전처리하는 Transform을 반환합니다.
    (train/val 때 사용한 것과 동일한 Resize, CenterCrop, ToTensor, Normalize)
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])

#####################################
# 4. 추론 함수
#####################################

def infer_image(model: torch.nn.Module,
                image_path: str,
                class_names: list,
                device: torch.device) -> None:
    """
    단일 이미지에 대해 모델 예측을 수행하고,
    각 클래스별 확률을 퍼센트(%) 형태로 출력합니다.
    """
    # 1) 이미지 로드 및 전처리
    img = Image.open(image_path).convert("RGB")
    transform = get_transform()
    input_tensor = transform(img).unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]

    # 2) 모델 추론
    with torch.no_grad():
        outputs = model(input_tensor)            # shape: [1, num_classes]
        probs   = torch.softmax(outputs, dim=1)  # shape: [1, num_classes]

    probs = probs.cpu().squeeze(0)  # shape: [num_classes]

    # 3) 클래스별 확률 출력
    print(f"\n=== Inference 결과: `{os.path.basename(image_path)}` ===")
    for idx, cls_name in enumerate(class_names):
        p = probs[idx].item() * 100.0
        print(f"{cls_name:12s}: {p:6.2f}%")
    print("=" * 40)

#####################################
# 5. 메인 실행
#####################################

if __name__ == "__main__":
    # DEVICE 설정 (CUDA 사용 가능하면 GPU, 그렇지 않으면 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("사용 디바이스:", device)

    # 모델 로드
    num_classes = len(CLASS_NAMES)
    model = load_model(MODEL_PATH, num_classes, device)

    # 지정한 이미지 경로로 추론 수행
    infer_image(model, IMAGE_PATH, CLASS_NAMES, device)

