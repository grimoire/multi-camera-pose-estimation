cmake_minimum_required(VERSION 2.8.11)

#find_package(CUDA REQUIRED)
enable_language(CUDA)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --compiler-options -fPIC)

file(GLOB_RECURSE NPP_CUDA_HEADS *.h *.hpp *.cuh)
file(GLOB NPP_CUDA_SOURCES *.cpp *.cu)

source_group("Include" FILES ${NPP_CUDA_HEADS})
source_group("Source" FILES ${NPP_CUDA_SOURCES})

cuda_add_library(npp_cuda STATIC ${NPP_CUDA_HEADS} ${NPP_CUDA_SOURCES})
target_link_libraries(npp_cuda ${CUDA_npp_LIBRARY})