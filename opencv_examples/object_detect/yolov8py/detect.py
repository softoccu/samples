import cv2
import time
import os
import numpy as np
from datetime import datetime, timedelta
from multiprocessing import Process, Queue
from ultralytics import YOLO
import signal

# ---------------------------
# 参数配置
# ---------------------------
FRAME_WIDTH = 1024
FRAME_HEIGHT = 576
FPS = 15
SEGMENT_HOURS = 3
AREA_THRESHOLD = 0.02
SKIP_FRAMES = 15
CONFIDENCE_THRESHOLD = 0.3
EXPOSURE_RANGE = (-11, -2)
TARGET_BRIGHTNESS_RANGE = (60, 110)
FRAME_IMAGE_DIR = 'person_frames'
VIDEO_DIR = 'videos'

os.makedirs(FRAME_IMAGE_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# ---------------------------
# 子进程：图像检测逻辑
# ---------------------------
def detector_process(frame_queue: Queue):
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # 忽略 Ctrl+C 中断

    print("[子进程] YOLO 正在加载中...")
    model = YOLO('yolov8s.pt')  # 切换为 yolov8s 模型
    model.fuse()
    print("[子进程] ✅ YOLO 加载完成，等待图像检测")

    while True:
        item = frame_queue.get()
        if item == "QUIT":
            print("[子进程] 🔚 收到退出信号，优雅退出")
            break

        frame, timestamp_str = item
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        detections = results[0].boxes

        for box in detections:
            cls_id = int(box.cls.item())
            conf_score = float(box.conf.item())
            if cls_id != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            area_ratio = area / (width * height)

            if area_ratio >= AREA_THRESHOLD:
                # 画绿色框和类别文字
                label = results[0].names[cls_id] if hasattr(results[0], 'names') else 'person'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame, f"{label} {conf_score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                img_name = f"person_{datetime.now().strftime('%Y%m%d_%I%M%S_%p')}.jpg"
                img_path = os.path.join(FRAME_IMAGE_DIR, img_name)
                cv2.imwrite(img_path, frame)
                print(f"[子进程] 💾 保存图像：{img_path}（面积比例={area_ratio:.3f}, 置信度={conf_score:.2f}）")
                break

# ---------------------------
# 主进程：录像与曝光调节
# ---------------------------
def recorder_main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    exposure_mode = "auto"
    current_exposure = -6
    last_exposure_update = datetime.min
    last_brightness = 0.0
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    print("🧪 启动自动曝光")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    def new_writer():
        tstr = datetime.now().strftime("%Y%m%d_%I%M%S_%p")
        path = os.path.join(VIDEO_DIR, f"record_{tstr}.avi")
        return cv2.VideoWriter(path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT)), datetime.now(), path

    out, segment_start_time, current_video_path = new_writer()
    print("🟢 开始录像：", current_video_path)

    frame_queue = Queue()
    detector = Process(target=detector_process, args=(frame_queue,))
    detector.start()

    frame_count = 0

    def adjust_manual_exposure():
        nonlocal current_exposure, last_brightness
        for attempt in range(10):
            cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
            time.sleep(0.2)
            ret, temp_frame = cap.read()
            if not ret:
                continue
            avg_brightness = np.mean(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2GRAY))
            last_brightness = avg_brightness

            print(f"🔧 曝光尝试 {attempt+1}: EXP={current_exposure}, 亮度={avg_brightness:.1f}, 时间={datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}")

            if TARGET_BRIGHTNESS_RANGE[0] <= avg_brightness <= TARGET_BRIGHTNESS_RANGE[1]:
                print(f"✅ 曝光调整成功：EXP={current_exposure}, 亮度={avg_brightness:.1f}")
                return

            if avg_brightness < TARGET_BRIGHTNESS_RANGE[0] and current_exposure < EXPOSURE_RANGE[1]:
                current_exposure += 1
            elif avg_brightness > TARGET_BRIGHTNESS_RANGE[1] and current_exposure > EXPOSURE_RANGE[0]:
                current_exposure -= 1
            else:
                break

        print(f"⚠️ 曝光调整失败，使用 EXP={current_exposure}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = datetime.now()
        timestamp_str = now.strftime('%Y-%m-%d %I:%M:%S %p')

        if (now - last_exposure_update) > timedelta(minutes=1):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            last_brightness = avg_brightness

            if avg_brightness > 150 and exposure_mode != "auto":
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                exposure_mode = "auto"
                print(f"🌞 自动曝光（亮度={avg_brightness:.1f}）")

            elif avg_brightness < 50 and exposure_mode != "manual":
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                exposure_mode = "manual"
                print(f"🌙 手动曝光（亮度={avg_brightness:.1f}） → 开始调整")
                adjust_manual_exposure()

            elif exposure_mode == "manual" and current_exposure > EXPOSURE_RANGE[0] and current_exposure < EXPOSURE_RANGE[1]:
                print(f"🔁 手动曝光保持模式，重新尝试设置 EXP={current_exposure}")
                adjust_manual_exposure()

            last_exposure_update = now

        if now - segment_start_time >= timedelta(hours=SEGMENT_HOURS):
            out.release()
            out, segment_start_time, current_video_path = new_writer()
            print("🔁 新录像：", current_video_path)

        overlay = f"{timestamp_str}  L:{last_brightness:.1f} EXP:{current_exposure}"
        cv2.putText(frame, overlay, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)

        if frame_count % SKIP_FRAMES == 0 and frame_queue.qsize() < 10:
            frame_queue.put((frame.copy(), timestamp_str))

        cv2.imshow("Live", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            print("🟥 用户主动按下 q 或 ESC，准备退出")
            break

        frame_count += 1

    print("[主进程] 正在通知子进程退出...")
    frame_queue.put("QUIT")
    detector.join(timeout=10)
    print("[主进程] 子进程已退出")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ 程序已优雅结束，视频保存至：", current_video_path)

# ---------------------------
# 启动主程序
# ---------------------------
if __name__ == '__main__':
    recorder_main()

