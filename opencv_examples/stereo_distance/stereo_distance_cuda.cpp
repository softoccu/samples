#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Set camera parameters, adjust according to the actual camera
const double focal_length = 0.8;  // Focal length (in meters)
const double baseline = 0.1;      // Baseline (in meters)

// Compute disparity map
void computeDisparity(const Mat& left, const Mat& right, Mat& disparity) {
    if (left.empty() || right.empty()) {
        cerr << "Input images are empty!" << endl;
        return;
    }

    Ptr<cuda::StereoBM> stereo = cuda::createStereoBM(128, 15); // Set parameters
    cuda::GpuMat d_left, d_right, d_disparity;
    d_left.upload(left);
    d_right.upload(right);
    stereo->compute(d_left, d_right, d_disparity);
    d_disparity.download(disparity);

    // Convert disparity map to 8-bit unsigned integer type for display and pseudo-color mapping
    disparity.convertTo(disparity, CV_8U, 255.0 / (128 * 16));
}

// Compute distance map
void calculateDistance(const Mat& disparity, Mat& distance) {
    distance = Mat(disparity.size(), CV_32F);
    for (int y = 0; y < disparity.rows; y++) {
        for (int x = 0; x < disparity.cols; x++) {
            float disp = disparity.at<uchar>(y, x);
            if (disp > 0) {
                distance.at<float>(y, x) = (focal_length * baseline) / disp;
            } else {
                distance.at<float>(y, x) = 0;
            }
        }
    }

    // Convert distance map to 8-bit unsigned integer type for pseudo-color mapping
    distance.convertTo(distance, CV_8U, 255.0 / (focal_length * baseline));
}

int main() {
    // Open stereo camera
    VideoCapture cap(0); // Assume stereo camera as device 0

    if (!cap.isOpened()) {
        cerr << "Unable to open camera" << endl;
        return -1;
    }

    // Set camera resolution
    cap.set(CAP_PROP_FRAME_WIDTH, 1280); // Each left and right image is 640 wide, total 1280
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    Mat frame, left, right, disparity, distance, colorDisparity, colorDistance;

    while (true) {
        // Read image frame
        cap >> frame;

        if (frame.empty()) {
            cerr << "Unable to read image" << endl;
            break;
        }

        // Split the image frame into left image and right image
        left = frame(Rect(0, 0, frame.cols / 2, frame.rows)).clone();
        right = frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows)).clone();

        // Check if left or right images are empty
        if (left.empty() || right.empty()) {
            cerr << "Split images are empty!" << endl;
            continue;
        }

        // Convert to grayscale images
        cvtColor(left, left, COLOR_BGR2GRAY);
        cvtColor(right, right, COLOR_BGR2GRAY);

        // Compute disparity map
        computeDisparity(left, right, disparity);

        // Check if disparity map is empty
        if (disparity.empty()) {
            cerr << "Failed to compute disparity map!" << endl;
            continue;
        }

        // Compute distance map
        calculateDistance(disparity, distance);

        // Apply pseudo-color mapping
        applyColorMap(disparity, colorDisparity, COLORMAP_JET);
        applyColorMap(distance, colorDistance, COLORMAP_JET);

        // Display results
        imshow("Left Image", left);
        imshow("Right Image", right);
        imshow("Disparity", colorDisparity);
        imshow("Distance", colorDistance);

        // Press ESC to exit
        if (waitKey(30) == 27) {
            break;
        }
    }

    return 0;
}