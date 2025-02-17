cmake_minimum_required(VERSION 3.10)
project(StereoDistanceCUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 设置OpenCV路径
set(OpenCV_DIR "~/opencv/build")

# 查找OpenCV库
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# 包含OpenCV头文件和CUDA头文件
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

# 添加可执行文件
add_executable(stereo_distance_cuda stereo_distance_cuda.cpp)

# 链接OpenCV库和CUDA库
target_link_libraries(stereo_distance_cuda ${OpenCV_LIBS})
target_link_libraries(stereo_distance_cuda ${CUDA_LIBRARIES})
