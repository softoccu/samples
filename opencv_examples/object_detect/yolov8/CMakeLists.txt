cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(yolov8_libtorch_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

# -------------- OpenCV --------------
set(OpenCV_DIR "~/opencv/build")
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})

# -------------- libtorch --------------
set(Torch_DIR "~/libtorch/share/cmake/Torch")
set(TORCH_LIBRARY "~/libtorch/lib/libtorch.so")


find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message("${TORCH_LIBRARIES}")
message("${TORCH_INCLUDE_DIRS}")


include_directories(${TORCH_INCLUDE_DIRS})

add_executable(yolov8libtorch "${CMAKE_CURRENT_SOURCE_DIR}/yolov8libtorch.cpp")
target_link_libraries(yolov8libtorch ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET yolov8libtorch PROPERTY CXX_STANDARD 17)

