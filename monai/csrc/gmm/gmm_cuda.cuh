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

#include <cuda.h>
#include <cuda_runtime.h>

#include "utils/tensor_description.h"

#define BLOCK_SIZE 32
#define TILE(SIZE, STRIDE) (((SIZE - 1)/STRIDE) + 1)

void GMMInitialize(int gmm_N, float *gmm, float *scratch_mem, int gmm_pitch, const float *image, int *alpha, int width, int height);
void GMMUpdate(int gmm_N, float *gmm, float *scratch_mem, int gmm_pitch, const float *image, int *alpha, int width, int height);
void GMMDataTerm(const float *image, int gmmN, const float *gmm, int gmm_pitch, float* output, int width, int height);

torch::Tensor GMM_Cuda(torch::Tensor input_tensor, torch::Tensor label_tensor, int mixture_count, int gaussians_per_mixture)
{
    TensorDescription desc = TensorDescription(input_tensor);

    int width = desc.sizes[0];
    int height = desc.sizes[1];
    int element_count = width * height;

    size_t gmm_pitch = 11 * sizeof(float);
    int gmms = mixture_count * gaussians_per_mixture;
    int blocks = TILE(width, BLOCK_SIZE) * TILE(height, BLOCK_SIZE);
    int scratch_gmm_size = blocks * gmm_pitch * gmms + blocks * 4;

    torch::Tensor output_tensor = torch::empty({desc.batchCount, mixture_count, width, height}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    float* input = input_tensor.data_ptr<float>();
    float* output = output_tensor.data_ptr<float>();
    float* scratch_mem = output;

    float* gmm; cudaMalloc(&d_gmm, gmm_pitch * gmms);
    int* alpha; cudaMalloc(&d_alpha, width * height * sizeof(int));

    cudaMemcpyAsync(d_alpha, label_tensor.data_ptr<int>(), width * height * sizeof(int), cudaMemcpyDeviceToDevice);

    GMMInitialize(gmms, d_gmm, scratch_mem, gmm_pitch, input, d_alpha, width, height);
    GMMUpdate(gmms, d_gmm, scratch_mem, gmm_pitch, input, d_alpha, width, height);
    GMMDataTerm(input, gmms, d_gmm, gmm_pitch, output, width, height);

    cudaFree(d_alpha);
    cudaFree(d_gmm);

    return output;
}
