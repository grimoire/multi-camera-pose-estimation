cmake_minimum_required(VERSION 3.1)

#find_package(CUDA REQUIRED)
enable_language(CUDA)

# set(CUDA_NVCC_FLAGS -O3;-G;-g;-std=c++11)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --compiler-options -fPIC)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -DCUDA_HAS_FP16=1;-D__CUDA_NO_HALF_OPERATORS__;-D__CUDA_NO_HALF_CONVERSIONS__;-D__CUDA_NO_HALF2_OPERATORS__)

file(GLOB_RECURSE DCNV2_CUDA_HEADS *.h *.hpp *.cuh)
file(GLOB DCNV2_CUDA_SOURCES *.cpp *.cu)

source_group("Include" FILES ${NPP_CUDA_HEADS})
source_group("Source" FILES ${NPP_CUDA_SOURCES})

cuda_add_library(DCNv2_cuda STATIC ${DCNV2_CUDA_HEADS} ${DCNV2_CUDA_SOURCES})
target_link_libraries(DCNv2_cuda ${TORCH_LIBRARIES})
# target_link_libraries(DCNv2_cuda ${CUDA_npp_LIBRARY})