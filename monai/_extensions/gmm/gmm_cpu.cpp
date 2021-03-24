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

#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>

#include "gmm.h"

void GMM_Cpu(const float* input, const int* labels, float* output, int batch_count, int element_count)
{
    throw std::invalid_argument("GMM recieved a cpu tensor but is not yet implemented for the cpu");
}
