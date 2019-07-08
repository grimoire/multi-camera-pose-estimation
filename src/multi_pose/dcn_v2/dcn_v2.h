#pragma once
#define WITH_CUDA
// #include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"


#endif

at::Tensor dcn_v2_forward(const at::Tensor &input, const at::Tensor &weight,
                          const at::Tensor &bias, const at::Tensor &offset,
                          const at::Tensor &mask, const int64_t kernel_h,
                          const int64_t kernel_w, const int64_t stride_h,
                          const int64_t stride_w, const int64_t pad_h,
                          const int64_t pad_w, const int64_t dilation_h,
                          const int64_t dilation_w,
                          const int64_t deformable_group) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return dcn_v2_cuda_forward(input, weight, bias, offset, mask, kernel_h,
                               kernel_w, stride_h, stride_w, pad_h, pad_w,
                               dilation_h, dilation_w, deformable_group);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor> dcn_v2_backward(
    const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias,
    const at::Tensor &offset, const at::Tensor &mask,
    const at::Tensor &grad_output, int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w, int64_t deformable_group) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return dcn_v2_cuda_backward(input, weight, bias, offset, mask, grad_output,
                                kernel_h, kernel_w, stride_h, stride_w, pad_h,
                                pad_w, dilation_h, dilation_w,
                                deformable_group);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor> dcn_v2_psroi_pooling_forward(
    const at::Tensor &input, const at::Tensor &bbox, const at::Tensor &trans,
    const int64_t no_trans, const double spatial_scale, const int64_t output_dim,
    const int64_t group_size, const int64_t pooled_size,
    const int64_t part_size, const int64_t sample_per_part,
    const double trans_std) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    auto tuple_result = dcn_v2_psroi_pooling_cuda_forward(
        input, bbox, trans, no_trans, spatial_scale, output_dim, group_size,
        pooled_size, part_size, sample_per_part, trans_std);
    std::vector<at::Tensor> result(2);
    result.push_back(std::get<0>(tuple_result));
    result.push_back(std::get<1>(tuple_result));
    return result;
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor> dcn_v2_psroi_pooling_backward(
    const at::Tensor &out_grad, const at::Tensor &input, const at::Tensor &bbox,
    const at::Tensor &trans, const at::Tensor &top_count,
    const int64_t no_trans, const double spatial_scale, const int64_t output_dim,
    const int64_t group_size, const int64_t pooled_size,
    const int64_t part_size, const int64_t sample_per_part,
    const double trans_std) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    auto tuple_result = dcn_v2_psroi_pooling_cuda_backward(
        out_grad, input, bbox, trans, top_count, no_trans, spatial_scale,
        output_dim, group_size, pooled_size, part_size, sample_per_part,
        trans_std);
    std::vector<at::Tensor> result(2);
    result.push_back(std::get<0>(tuple_result));
    result.push_back(std::get<1>(tuple_result));
    return result;
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}