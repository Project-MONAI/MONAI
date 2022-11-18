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

#include <torch/extension.h>
#include <stdexcept>
#include <string>

#include "bilateral.h"
#include "utils/common_utils.h"

torch::Tensor BilateralFilter(torch::Tensor input, float spatial_sigma, float color_sigma, bool usePHL) {
  torch::Tensor (*filterFunction)(torch::Tensor, float, float);

#ifdef WITH_CUDA

  if (torch::cuda::is_available() && input.is_cuda()) {
    CHECK_CONTIGUOUS_CUDA(input);

    if (input.size(1) > BF_CUDA_MAX_CHANNELS) {
      throw std::runtime_error(
          "Bilateral filtering not implemented for channel count > " + std::to_string(BF_CUDA_MAX_CHANNELS));
    }

    if (input.dim() - 2 > BF_CUDA_MAX_SPATIAL_DIMENSION) {
      throw std::runtime_error(
          "Bilateral filtering not implemented for spatial dimension > " +
          std::to_string(BF_CUDA_MAX_SPATIAL_DIMENSION));
    }

    filterFunction = usePHL ? &BilateralFilterPHLCuda : &BilateralFilterCuda;
  } else {
    filterFunction = usePHL ? &BilateralFilterPHLCpu : &BilateralFilterCpu;
  }
#else
  filterFunction = usePHL ? &BilateralFilterPHLCpu : &BilateralFilterCpu;
#endif

  return filterFunction(input, spatial_sigma, color_sigma);
}
