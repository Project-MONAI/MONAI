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

#if !defined(CHANNEL_COUNT) || !defined(MIXTURE_COUNT) || !defined(MIXTURE_SIZE)
#error Definition of CHANNEL_COUNT, MIXTURE_COUNT, and MIXTURE_SIZE required
#endif

#if CHANNEL_COUNT < 1 || MIXTURE_COUNT < 1 || MIXTURE_SIZE < 1
#error CHANNEL_COUNT, MIXTURE_COUNT, and MIXTURE_SIZE must be positive
#endif

#define MATRIX_COMPONENT_COUNT ((CHANNEL_COUNT + 1) * (CHANNEL_COUNT + 2) / 2)
#define SUB_MATRIX_COMPONENT_COUNT (CHANNEL_COUNT * (CHANNEL_COUNT + 1) / 2)
#define GMM_COMPONENT_COUNT (MATRIX_COMPONENT_COUNT + 1)
#define GMM_COUNT (MIXTURE_COUNT * MIXTURE_SIZE)

void learn_cpu(
    const float* input,
    const int* labels,
    float* gmm,
    float* scratch_memory,
    unsigned int batch_count,
    unsigned int element_count);
void apply_cpu(
    const float* gmm,
    const float* input,
    float* output,
    unsigned int batch_count,
    unsigned int element_count);

void learn_cuda(
    const float* input,
    const int* labels,
    float* gmm,
    float* scratch_memory,
    unsigned int batch_count,
    unsigned int element_count);
void apply_cuda(
    const float* gmm,
    const float* input,
    float* output,
    unsigned int batch_count,
    unsigned int element_count);
