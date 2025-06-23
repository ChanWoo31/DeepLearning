import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image

#####################################
# 1. 설정 부분 (경로/클래스)
#####################################
MODEL_PATH  = "/home/han/machine_learning/best_model.pth"
CLASS_NAMES = [
    "due_down", "due_up",
    "motor_shield_down", "motor_shield_up",
    "opencr_down", "opencr_up",
    "rasp_pi_down", "rasp_pi_up",
    "uno_down", "uno_up"
]
IMG_SIZE = 224  # 입력 크기 (중앙 ROI 크기)

#####################################
# 2. 모델 로드 함수 (ResNet-18)
#####################################
def load_model(model_path: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    # PyTorch 2.0+: weights_only=True 권장
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

#####################################
# 3. Transform 함수
#####################################
def get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5035175681114197, 0.5433481931686401, 0.5339823961257935],
            std=[0.2726588547229767, 0.22477710247039795, 0.22121909260749817]
        )
    ])

#####################################
# 4. 실시간 웹캠 추론 함수
#####################################
def webcam_inference(model: torch.nn.Module,
                     class_names: list,
                     device: torch.device,
                     cam_id: int = 4) -> None:
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] Cannot open camera ID {cam_id}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    transform = get_transform()
    roi_size = IMG_SIZE

    print("\n[INFO] 웹캠 실시간 추론 시작 (q 또는 ESC로 종료)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame")
            break

        h, w = frame.shape[:2]
        x1 = w // 2 - roi_size // 2
        y1 = h // 2 - roi_size // 2
        x2, y2 = x1 + roi_size, y1 + roi_size
        roi = frame[y1:y2, x1:x2]

        pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        inp = transform(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(inp)
            probs = torch.softmax(out, dim=1).cpu().squeeze(0)
            idx = probs.argmax().item()
            label = class_names[idx]
            prob  = probs[idx].item() * 100.0

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"{label}: {prob:.1f}%",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
        cv2.imshow("Webcam Inference", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 웹캠 추론 종료")

#####################################
# 5. 메인 실행
#####################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("사용 디바이스:", device)
    num_classes = len(CLASS_NAMES)  # = 10
    model = load_model(MODEL_PATH, num_classes, device)
    webcam_inference(model, CLASS_NAMES, device, cam_id=4)
