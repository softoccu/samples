import cv2
import time
from datetime import datetime, timedelta
from ultralytics import YOLO
import os
import numpy as np

# ---------------------------
# 通用配置
# ---------------------------
FRAME_IMAGE_DIR = 'person_frames'
VIDEO_DIR = 'videos'
CONFIDENCE_THRESHOLD = 0.3
AREA_THRESHOLD = 0.05
CAMERA_INDEX = 0
SKIP_FRAMES = 15  # 15FPS 每秒检测 1 帧
SEGMENT_DURATION_HOURS = 3
use_manual_exposure = True  # 部署用 True，开发用 False

# ---------------------------
# 初始化模型与路径
# ---------------------------
model = YOLO('yolov8n.pt')
model.fuse()
os.makedirs(FRAME_IMAGE_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
cap.set(cv2.CAP_PROP_FPS, 15)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 15

# ---------------------------
# 使用 XVID 编码
# ---------------------------
fourcc = cv2.VideoWriter_fourcc(*'XVID')

def get_new_video_writer():
    timestamp = datetime.now().strftime("%Y%m%d_%I%M%S_%p")
    filename = f"record_{timestamp}.avi"
    path = os.path.join(VIDEO_DIR, filename)
    return cv2.VideoWriter(path, fourcc, fps, (width, height)), datetime.now(), path

out, segment_start_time, current_video_path = get_new_video_writer()
print("🟢 开始录像：", current_video_path)

frame_count = 0
last_brightness = 100
last_exposure = -4

# 曝光设置
if use_manual_exposure:
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, last_exposure)
    print("🔧 手动曝光")
else:
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    print("🧪 自动曝光")

def adjust_exposure_by_brightness(avg_brightness):
    if avg_brightness < 50:
        return -2
    elif avg_brightness > 120:
        return -6
    else:
        return -4

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = datetime.now()
        timestamp_str = now.strftime('%Y-%m-%d %I:%M:%S %p')

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)

        if use_manual_exposure:
            if abs(avg_brightness - last_brightness) > 10:
                new_exp = adjust_exposure_by_brightness(avg_brightness)
                if new_exp != last_exposure:
                    cap.set(cv2.CAP_PROP_EXPOSURE, new_exp)
                    last_exposure = new_exp
                last_brightness = avg_brightness

        cv2.putText(frame, timestamp_str, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(frame)

        if now - segment_start_time >= timedelta(hours=SEGMENT_DURATION_HOURS):
            out.release()
            out, segment_start_time, current_video_path = get_new_video_writer()
            print("🔁 新视频文件：", current_video_path)

        frame_display = frame.copy()
        saved = False

        if frame_count % SKIP_FRAMES == 0:
            results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            detections = results[0].boxes

            for box in detections:
                cls_id = int(box.cls.item())
                if cls_id != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                area_ratio = area / (width * height)

                cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if area_ratio >= AREA_THRESHOLD and not saved:
                    img_name = f"person_{now.strftime('%Y%m%d_%I%M%S_%p')}.jpg"
                    img_path = os.path.join(FRAME_IMAGE_DIR, img_name)
                    cv2.imwrite(img_path, frame)
                    print(f"💾 保存截图：{img_path}")
                    saved = True

        cv2.putText(frame_display, timestamp_str, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow('Live Detection', frame_display)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🟥 用户退出。")
            break

except KeyboardInterrupt:
    print("🛑 中断。")

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ 结束，保存于：", current_video_path)

