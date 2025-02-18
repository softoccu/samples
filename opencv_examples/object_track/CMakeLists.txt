cmake_minimum_required(VERSION 3.10)

# 项目名称和版本
project(DualCameraTracking VERSION 1.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 指定OpenCV路径
set(OpenCV_DIR "~/opencv/build")

# 查找OpenCV包
find_package(OpenCV REQUIRED)

# 添加可执行文件
add_executable(object_track object_track.cpp)

# 链接OpenCV库
target_link_libraries(object_track ${OpenCV_LIBS} 
    opencv_cudaimgproc opencv_cudaarithm opencv_cudaobjdetect)
