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

#if !defined(CHANNEL_COUNT) || !defined(MIXTURE_COUNT) || !defined(MIXTURE_SIZE)
#error definition of CHANNEL_COUNT, MIXTURE_COUNT, and MIXTURE_SIZE required
#endif

#define COMPONENT_COUNT ((CHANNEL_COUNT + 1) * (CHANNEL_COUNT + 2) / 2)
#define GMM_SIZE (MIXTURE_COUNT * MIXTURE_SIZE)

void GMM_Cuda(const float* input, const int* labels, float* output, int batch_count, int channel_count, int element_count, int mixture_count, int gaussians_per_mixture);
void GMM_Cpu(const float* input, const int* labels, float* output, int batch_count, int channel_count, int element_count, int mixture_count, int gaussians_per_mixture);
