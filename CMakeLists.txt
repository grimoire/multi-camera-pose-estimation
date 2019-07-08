cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(CenterNet_Cpp)

# opencv
Set(OpenCV_DIR ~/SDK/opencv340/installation/OpenCV-3.4.0/share/OpenCV)
find_package(OpenCV REQUIRED)

#pytorch
set(Torch_DIR ~/SDK/pytorch/torch/share/cmake/Torch)
set(Caffe2_DIR ~/SDK/pytorch/torch/share/cmake/Caffe2)
find_package(Torch REQUIRED)

#hikvision
set(HIKVISION_INCLUDE ~/SDK/CH-HCNetSDKV6.0.2.2_build20181213_Linux64/incCn)

add_definitions(-std=c++1y)

include_directories(${PROJECT_SOURCE_DIR}/include)
link_directories(${PROJECT_SOURCE_DIR}/lib)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)