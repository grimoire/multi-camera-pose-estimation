cmake_minimum_required(VERSION 2.8.11)
set(Boost_USE_STATIC_LIBS	ON)
set(Boost_USE_MULTITHREADED	ON)
find_package(Boost COMPONENTS date_time filesystem system thread chrono)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_definitions(-std=c++1y)
add_compile_options(-fPIC)

file(GLOB_RECURSE PIPELINE_SRC *.h *.cpp)

set(PIPELINE_LIB	HKIPcamera
					opencv_world
					multi_pose)

add_library(pipeline STATIC ${PIPELINE_SRC})
target_link_libraries(pipeline ${PIPELINE_LIB})
target_link_libraries(pipeline ${Boost_LIBRARIES})
target_link_libraries(pipeline ${TORCH_LIBRARIES})