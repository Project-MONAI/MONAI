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

cudaError_t GMMAssign(int gmmN, const float *gmm, int gmm_pitch, const uchar4 *image, int image_pitch, unsigned char *alpha, int alpha_pitch, int width, int height);
cudaError_t GMMInitialize(int gmm_N, float *gmm, float *scratch_mem, int gmm_pitch, const uchar4 *image, int image_pitch, unsigned char *alpha, int alpha_pitch, int width, int height);
cudaError_t GMMUpdate(int gmm_N, float *gmm, float *scratch_mem, int gmm_pitch, const uchar4 *image, int image_pitch, unsigned char *alpha, int alpha_pitch, int width, int height);
cudaError_t GMMDataTerm(int *terminals, int terminal_pitch, int gmmN, const float *gmm, int gmm_pitch, const uchar4 *image, int image_pitch, const unsigned char *trimap, int trimap_pitch, int width, int height);

torch::Tensor GMM_Cuda(torch::Tensor image, torch::Tensor labels, int mixture_count, int gaussians_per_mixture)
{
    return image;
}