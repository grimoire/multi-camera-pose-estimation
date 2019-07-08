#include "dcn_v2.h"
#include <torch/script.h>
#define WITH_CUDA


static auto registry =
    torch::jit::RegisterOperators("dcnv2::dcn_v2_forward", &dcn_v2_forward)
        .op("dcnv2::dcn_v2_backward", &dcn_v2_backward)
        .op("dcnv2::dcn_v2_psroi_pooling_forward",
            &dcn_v2_psroi_pooling_forward)
        .op("dcnv2::dcn_v2_psroi_pooling_backward",
            &dcn_v2_psroi_pooling_backward);