cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(DCNv2_ext)

add_definitions(-std=c++11)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/cuda)
add_subdirectory(cuda)

set(EXTRA_LIBS ${EXTRA_LIBS} DCNv2_cuda)

add_library(DCNv2_ext SHARED dcn_v2.h vision.cpp)
target_compile_features(DCNv2_ext PRIVATE cxx_range_for)
target_link_libraries(DCNv2_ext ${TORCH_LIBRARIES})
target_link_libraries(DCNv2_ext ${EXTRA_LIBS})