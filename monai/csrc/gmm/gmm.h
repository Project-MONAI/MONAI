/*
Copyright 2020 - 2021 MONAI Consortium
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
#include "gmm_cuda.h"

torch::Tensor GMM(torch::Tensor input_tensor, torch::Tensor label_tensor, int mixture_count, int gaussians_per_mixture)
{
    return GMM_Cuda(input_tensor, label_tensor, mixture_count, gaussians_per_mixture);
}