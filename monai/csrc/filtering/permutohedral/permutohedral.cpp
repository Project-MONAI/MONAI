/*
Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <stdexcept>
#include <string>

#include "utils/common_utils.h"
#include "utils/meta_macros.h"

#include "permutohedral.h"

torch::Tensor PermutohedralFilter(torch::Tensor input, torch::Tensor features) {
  input = input.contiguous();

  int batchCount = input.size(0);
  int batchStride = input.stride(0);
  int elementCount = input.stride(1);
  int channelCount = input.size(1);
  int featureCount = features.size(1);

// movedim not support in torch < 1.7.1
#if MONAI_TORCH_VERSION >= 10701
  torch::Tensor data = input.clone().movedim(1, -1).contiguous();
  features = features.movedim(1, -1).contiguous();
#else
  torch::Tensor data = input.clone();
  features = features;

  for (int i = 1; i < input.dim() - 1; i++) {
    data = data.transpose(i, i + 1);
    features = features.transpose(i, i + 1);
  }

  data = data.contiguous();
  features = features.contiguous();
#endif

#ifdef WITH_CUDA
  if (torch::cuda::is_available() && data.is_cuda()) {
    CHECK_CONTIGUOUS_CUDA(data);

    if (channelCount > PHL_CUDA_MAX_CHANNELS) {
      throw std::runtime_error(
          "PHL filtering not implemented for channel count > " + std::to_string(PHL_CUDA_MAX_CHANNELS));
    }

    if (featureCount > PHL_CUDA_MAX_FEATURES) {
      throw std::runtime_error(
          "PHL filtering not implemented for feature count > " + std::to_string(PHL_CUDA_MAX_FEATURES));
    }

#define CASE(dc, fc)                                                                                                  \
  AT_DISPATCH_FLOATING_TYPES(data.scalar_type(), "PermutohedralCuda", ([&] {                                          \
                               for (int batchIndex = 0; batchIndex < batchCount; batchIndex++) {                      \
                                 scalar_t* offsetData = data.data_ptr<scalar_t>() + batchIndex * batchStride;         \
                                 scalar_t* offsetFeatures =                                                           \
                                     features.data_ptr<scalar_t>() + batchIndex * fc * elementCount;                  \
                                 PermutohedralCuda<scalar_t, dc, fc>(offsetData, offsetFeatures, elementCount, true); \
                               }                                                                                      \
                             }));
    SWITCH_AB(CASE, PHL_CUDA_MAX_CHANNELS, PHL_CUDA_MAX_FEATURES, channelCount, featureCount);

  } else {
#endif
    AT_DISPATCH_FLOATING_TYPES(
        data.scalar_type(), "PermutohedralCPU", ([&] {
          for (int batchIndex = 0; batchIndex < batchCount; batchIndex++) {
            scalar_t* offsetData = data.data_ptr<scalar_t>() + batchIndex * batchStride;
            scalar_t* offsetFeatures = features.data_ptr<scalar_t>() + batchIndex * featureCount * elementCount;
            PermutohedralCPU<scalar_t>(offsetData, offsetFeatures, channelCount, featureCount, elementCount);
          }
        }));
#ifdef WITH_CUDA
  }
#endif

// movedim not support in torch < 1.7.1
#if MONAI_TORCH_VERSION >= 10701
  data = data.movedim(-1, 1);
#else
  for (int i = input.dim() - 1; i > 1; i--) {
    data = data.transpose(i - 1, i);
  }
#endif

  return data;
}
