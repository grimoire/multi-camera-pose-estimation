cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
set(Boost_USE_STATIC_LIBS	ON)
set(Boost_USE_MULTITHREADED	ON)
find_package(Boost COMPONENTS date_time filesystem system thread chrono)

set(MAIN_SRC    main.cpp)

set(MAIN_LIBS   opencv_world
                pipeline
                multi_pose
                )

add_executable(surveillance_demo    ${MAIN_SRC})
target_link_libraries(surveillance_demo ${MAIN_LIBS})
target_link_libraries(surveillance_demo ${TORCH_LIBRARIES})
target_link_libraries(surveillance_demo -Wl,--no-as-needed DCNv2_ext)
# target_link_libraries(surveillance_demo ${Boost_LIBRARIES})