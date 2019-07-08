cmake_minimum_required(VERSION 2.8.11)
project(multi_pose)

add_definitions(-std=c++11)
add_compile_options(-fPIC)

add_subdirectory(dcn_v2)

file(GLOB MULTI_POSE_SRC *.h *.cpp)

set(MULTI_POSE_LIB	DCNv2_ext
                    opencv_world)

add_library(multi_pose SHARED ${MULTI_POSE_SRC})
target_link_libraries(multi_pose ${MULTI_POSE_LIB})
target_link_libraries(multi_pose ${TORCH_LIBRARIES})
set_property(TARGET multi_pose PROPERTY CXX_STANDARD 11)