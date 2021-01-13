#include "utils/common_utils.h"
#include "utils/meta_macros.h"

#include "permutohedral.h"
#include <iostream>

torch::Tensor PermutohedralFilter(torch::Tensor input, torch::Tensor features) {

    input = input.contiguous();

    int elementCount = input.stride(1);
    int channelCount = input.size(1);
    int featureCount = features.size(1);

    torch::Tensor data = input.clone().movedim(1, -1).contiguous();
    features = features.movedim(1, -1).contiguous();

    #ifdef WITH_CUDA
    if (torch::cuda::is_available() && data.is_cuda()) {
        CHECK_CONTIGUOUS_CUDA(data);
    
        #define CASE(dc, fc) AT_DISPATCH_FLOATING_TYPES(data.scalar_type(), "PermutohedralCuda", ([&] {   \
            PermutohedralCuda<scalar_t, dc, fc>(data.data_ptr<scalar_t>(), features.data_ptr<scalar_t>(), elementCount, true);                                   \
        }));
        SWITCH_AB(CASE, 16, 19, channelCount, featureCount);

  } else {
        AT_DISPATCH_FLOATING_TYPES(data.scalar_type(), "PermutohedralCPU", ([&] {    \
            PermutohedralCPU<scalar_t>(data.data_ptr<scalar_t>(), features.data_ptr<scalar_t>(), channelCount, featureCount, elementCount);                \
        }));
  }
#else
    AT_DISPATCH_FLOATING_TYPES(data.scalar_type(), "PermutohedralCPU", ([&] {    \
        PermutohedralCPU<scalar_t>(data.data_ptr<scalar_t>(), features.data_ptr<scalar_t>(), channelCount, featureCount, elementCount);                \
    }));
#endif

    data = data.movedim(-1, 1);

    return data;
}
