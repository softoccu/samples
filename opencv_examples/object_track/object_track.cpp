#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // 打开双镜头摄像头
    VideoCapture cap(0); // 单个设备

    if (!cap.isOpened()) {
        cerr << "无法打开摄像头" << endl;
        return -1;
    }

    // 设置摄像头分辨率
    cap.set(CAP_PROP_FRAME_WIDTH, 1280); // 假设分辨率宽度为 1280（两个镜头每个640）
    cap.set(CAP_PROP_FRAME_HEIGHT, 480); // 假设分辨率高度为 480

    Mat frame, frame1, frame2;
    Rect_<int> bbox1, bbox2;
    Ptr<Tracker> tracker1 = TrackerCSRT::create();
    Ptr<Tracker> tracker2 = TrackerCSRT::create();

    // 捕获初始帧并选择跟踪目标
    cap >> frame;
    if (frame.empty()) {
        cerr << "捕获图像失败" << endl;
        return -1;
    }

    // 将图像分割成左右两个部分
    frame1 = frame(Rect(0, 0, frame.cols / 2, frame.rows)); // 左半部分
    frame2 = frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows)); // 右半部分

    // 选择跟踪目标
    bbox1 = selectROI("Camera 1", frame1, false);
    bbox2 = selectROI("Camera 2", frame2, false);

    // 初始化跟踪器
    tracker1->init(frame1, bbox1);
    tracker2->init(frame2, bbox2);

    while (true) {
        // 捕获图像
        cap >> frame;

        if (frame.empty()) {
            cerr << "捕获图像失败" << endl;
            break;
        }

        // 将图像分割成左右两个部分
        frame1 = frame(Rect(0, 0, frame.cols / 2, frame.rows)); // 左半部分
        frame2 = frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows)); // 右半部分

        // 更新跟踪器
        bool ok1 = tracker1->update(frame1, bbox1);
        bool ok2 = tracker2->update(frame2, bbox2);

        // 绘制跟踪结果
        if (ok1) {
            rectangle(frame1, bbox1, Scalar(255, 0, 0), 2, 1);
        } else {
            putText(frame1, "Tracking failure detected", Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        }

        if (ok2) {
            rectangle(frame2, bbox2, Scalar(255, 0, 0), 2, 1);
        } else {
            putText(frame2, "Tracking failure detected", Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        }

        // 显示图像
        imshow("Camera 1", frame1);
        imshow("Camera 2", frame2);

        // 按下ESC键退出
        if (waitKey(30) == 27) {
            break;
        }
    }

    cap.release();
    destroyAllWindows();

    return 0;
}