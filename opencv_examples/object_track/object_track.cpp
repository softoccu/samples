#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Open dual-lens camera
    VideoCapture cap(0); // Single device

    if (!cap.isOpened()) {
        cerr << "Unable to open the camera" << endl;
        return -1;
    }

    // Set camera resolution
    cap.set(CAP_PROP_FRAME_WIDTH, 1280); // Assume resolution width is 1280 (two lenses, each 640)
    cap.set(CAP_PROP_FRAME_HEIGHT, 480); // Assume resolution height is 480

    Mat frame, frame1, frame2;
    Rect_<int> bbox1, bbox2;
    Ptr<Tracker> tracker1 = TrackerCSRT::create();
    Ptr<Tracker> tracker2 = TrackerCSRT::create();

    // Capture initial frame and select tracking targets
    cap >> frame;
    if (frame.empty()) {
        cerr << "Failed to capture image" << endl;
        return -1;
    }

    // Split the image into left and right parts
    frame1 = frame(Rect(0, 0, frame.cols / 2, frame.rows)); // Left half
    frame2 = frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows)); // Right half

    // Select tracking targets
    bbox1 = selectROI("Camera 1", frame1, false);
    bbox2 = selectROI("Camera 2", frame2, false);

    // Initialize trackers
    tracker1->init(frame1, bbox1);
    tracker2->init(frame2, bbox2);

    while (true) {
        // Capture image
        cap >> frame;

        if (frame.empty()) {
            cerr << "Failed to capture image" << endl;
            break;
        }

        // Split the image into left and right parts
        frame1 = frame(Rect(0, 0, frame.cols / 2, frame.rows)); // Left half
        frame2 = frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows)); // Right half

        // Update trackers
        bool ok1 = tracker1->update(frame1, bbox1);
        bool ok2 = tracker2->update(frame2, bbox2);

        // Draw tracking results
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

        // Display images
        imshow("Camera 1", frame1);
        imshow("Camera 2", frame2);

        // Exit on ESC key
        if (waitKey(30) == 27) {
            break;
        }
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
