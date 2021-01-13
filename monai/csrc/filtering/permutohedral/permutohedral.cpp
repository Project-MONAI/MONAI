#include "utils/common_utils.h"
#include "utils/meta_macros.h"

#include "permutohedral.h"
#include <iostream>

torch::Tensor PermutohedralFilter(torch::Tensor input, torch::Tensor features) {

    input = input.contiguous();

    int batchCount = input.size(0);
    int batchStride = input.stride(0);
    int elementCount = input.stride(1);
    int channelCount = input.size(1);
    int featureCount = features.size(1);

    torch::Tensor data = input.clone().movedim(1, -1).contiguous();
    features = features.movedim(1, -1).contiguous();

    #ifdef WITH_CUDA
    if (torch::cuda::is_available() && data.is_cuda()) {
        CHECK_CONTIGUOUS_CUDA(data);
    
        #define CASE(dc, fc) AT_DISPATCH_FLOATING_TYPES(data.scalar_type(), "PermutohedralCuda", ([&] {     \
            for (int batchIndex = 0; batchIndex < batchCount; batchIndex++) {                               \
                scalar_t* offsetData = data.data_ptr<scalar_t>() + batchIndex * batchStride;                \
                scalar_t* offsetFeatures = features.data_ptr<scalar_t>() + batchIndex * fc * elementCount;  \
                PermutohedralCuda<scalar_t, dc, fc>(offsetData, offsetFeatures, elementCount, true);        \
        }}));
        SWITCH_AB(CASE, 16, 19, channelCount, featureCount);

    } 
    else {
    #endif
        AT_DISPATCH_FLOATING_TYPES(data.scalar_type(), "PermutohedralCPU", ([&] {                                   \
            for (int batchIndex = 0; batchIndex < batchCount; batchIndex++) {                                       \
                scalar_t* offsetData = data.data_ptr<scalar_t>() + batchIndex * batchStride;                        \
                scalar_t* offsetFeatures = features.data_ptr<scalar_t>() + batchIndex * featureCount * elementCount;\
                PermutohedralCPU<scalar_t>(offsetData, offsetFeatures, channelCount, featureCount, elementCount);   \
        }}));
    #ifdef WITH_CUDA
    }
    #endif

    data = data.movedim(-1, 1);

    return data;
}
