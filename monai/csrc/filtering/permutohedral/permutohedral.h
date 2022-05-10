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

#pragma once

#include <torch/extension.h>

#define PHL_CUDA_MAX_CHANNELS 16
#define PHL_CUDA_MAX_FEATURES 19

template <typename scalar_t>
void PermutohedralCPU(scalar_t* data, scalar_t* features, int dataChannels, int featureChannels, int elementCount);
#ifdef WITH_CUDA
template <typename scalar_t, int dc, int fc>
void PermutohedralCuda(scalar_t* data, scalar_t* features, int elementCount, bool accurate);
#endif

torch::Tensor PermutohedralFilter(torch::Tensor input, torch::Tensor features);
