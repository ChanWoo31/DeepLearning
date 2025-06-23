import cv2
import os
import time
from datetime import datetime

# =========================
# 사용자가 수정할 부분
# =========================

# 1) BRIO가 연결된 인덱스 (v4l2-ctl로 확인했을 때 /dev/video4 이면 4)
CAMERA_INDEX = 4

# 2) 저장을 원하는 최상위 폴더 경로 (예: "/home/han/machine_learning/data")
BASE_DIR = os.path.join(os.getcwd(), "data")

# 3) 보드 이름 (uno_up, due_up, opencr_up, uno_down, due_down, opencr_down)
# BOARD_NAME = "uno_up"
# BOARD_NAME = "due_up"
# BOARD_NAME = "opencr_up"
# BOARD_NAME = "uno_down"
# BOARD_NAME = "due_down" 
BOARD_NAME = "opencr_down"
# BOARD_NAME = "test"
# BOARD_NAME = "rasp_pi_up"
# BOARD_NAME = "rasp_pi_down"
# BOARD_NAME = "motor_shield_up"
# BOARD_NAME = "motor_shield_down"

# 4) 캡처 지속 시간(초)과 초당 캡처 횟수(fps)
DURATION_SEC = 20.0   # 예: 30초 동안, 360
FPS = 3               # 초당 3장

# 5) 캡처 해상도 (카메라가 확실히 지원하는 표준 해상도로 설정)
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

# 6) 학습용으로 리사이즈할 최종 이미지 크기
TARGET_WIDTH = 224
TARGET_HEIGHT = 224
OUTPUT_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)

# =========================
# 여기까지 수정
# =========================


def capture_and_preview():
    # 1) 저장 폴더 생성: BASE_DIR/BOARD_NAME
    save_dir = os.path.join(BASE_DIR, BOARD_NAME.lower())
    os.makedirs(save_dir, exist_ok=True)

    # 2) 카메라 열기 (V4L2 백엔드 명시)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[Error] 카메라 인덱스 {CAMERA_INDEX} 로 열 수 없습니다.")
        return

    # 3) 캡처 해상도를 표준 해상도(1280×720)로 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    time.sleep(0.5)  # 설정 반영 대기

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Info] 카메라 캡처 해상도 → 요청: {CAPTURE_WIDTH}×{CAPTURE_HEIGHT}, 실제: {actual_w}×{actual_h}")

    # 4) 캡처 루프
    interval = 1.0 / FPS
    start_time = time.time()
    count = 0

    # OpenCV 창 띄워서 실시간 프리뷰 보기 (프레임 사이즈는 캡처 해상도에 맞춰 둬도 상관없음)
    window_name = "Live Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 360)  # 미리보기 크기만 640×360

    while True:
        elapsed = time.time() - start_time
        if elapsed >= DURATION_SEC:
            break

        ret, frame = cap.read()
        if not ret or frame is None:
            # 프레임을 못 읽어 왔을 때 잠시 대기
            time.sleep(interval)
            continue

        # 4-1) 화면에 미리보기 띄우기
        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            f"{BOARD_NAME.upper()} ({count+1})",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        cv2.imshow(window_name, display_frame)

        # 4-2) 저장용 리사이즈 (1280×720 → 224×224)
        resized = cv2.resize(frame, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)

        # 4-3) 파일명 생성 (예: opencr_up_20250602_123045_000.jpg)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{BOARD_NAME.lower()}_{ts}_{count:03d}.jpg"
        full_path = os.path.join(save_dir, filename)

        # 4-4) 이미지 저장
        success = cv2.imwrite(full_path, resized)
        if success:
            print(f"[Saved] {full_path}")
        else:
            print(f"[Error] 저장 실패: {full_path}")

        count += 1

        # 4-5) 다음 캡처 타이밍 맞추기
        next_capture = start_time + (count / FPS)
        sleep_time = next_capture - time.time()
        if sleep_time > 0:
            if cv2.waitKey(int(sleep_time * 1000)) & 0xFF == 27:  # ESC키로 중단 가능
                break
        else:
            if cv2.waitKey(1) & 0xFF == 27:
                break

    # 5) 마무리: 카메라 해제, 창 닫기
    cap.release()
    cv2.destroyAllWindows()
    print(f"--- 완료: {DURATION_SEC:.1f}초 동안 총 {count}장 저장 ---")


if __name__ == "__main__":
    print("==============================================")
    print("   시작: BRIO 카메라로부터 학습용 224×224 캡처  ")
    print("==============================================\n")
    capture_and_preview()
