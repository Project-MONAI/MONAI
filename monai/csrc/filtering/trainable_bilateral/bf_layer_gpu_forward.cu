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

=========================================================================
Adapted from https://github.com/faebstn96/trainable-bilateral-filter-source
which has the following license...
https://github.com/faebstn96/trainable-bilateral-filter-source/blob/main/LICENSE.md

Copyright 2022 Fabian Wagner, Pattern Recognition Lab, FAU Erlangen-Nuernberg, Erlangen, Germany
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

#include "trainable_bilateral.h"
//#include "../utils/cuda_error_check.h"
#include "utils/meta_macros.h"
#include "utils/tensor_description.h"

__constant__ int cBatchStride;
__constant__ int cColorStride;

__constant__ int cSizes[3];
__constant__ int cStrides[3];

__constant__ int cKernelSizes[3];
__constant__ int cHalfWindowSize_arr[3];
__constant__ float cGaussianKernel_x[256];
__constant__ float cGaussianKernel_y[256];
__constant__ float cGaussianKernel_z[256];
__constant__ float cXDistanceSquared[256];
__constant__ float cYDistanceSquared[256];
__constant__ float cZDistanceSquared[256];
__constant__ float cColorExponentConstant;
__constant__ float cSigma_x;
__constant__ float cSigma_y;
__constant__ float cSigma_z;
__constant__ float cColorSigma;

template <typename scalar_t, int C>
__global__ void BilateralFilterCudaKernel3DForward(
    scalar_t* input,
    scalar_t* output,
    scalar_t* outputWeightsTensor,
    scalar_t* dO_dx_ki,
    scalar_t* dO_dsig_r,
    scalar_t* dO_dsig_x,
    scalar_t* dO_dsig_y,
    scalar_t* dO_dsig_z) {
  int homeOffset = blockIdx.x * blockDim.x + threadIdx.x;
  int batchOffset = blockIdx.y * cBatchStride;

  if (homeOffset >= cColorStride)
    return;

  int homeX = homeOffset / cStrides[0];
  int homeY = (homeOffset - homeX * cStrides[0]) / cStrides[1];
  int homeZ = (homeOffset - homeX * cStrides[0] - homeY * cStrides[1]) / cStrides[2];
  int homeIndex[] = {homeX, homeY, homeZ};

  // Zero kernel aggregates.
  scalar_t valueSum = 0;
  scalar_t dw_dx_ki = 0;
  scalar_t dfilter_dx_ki = 0;
  scalar_t colorSum_w = 0;
  scalar_t colorSum_alpha = 0;
  scalar_t xSum_w = 0;
  scalar_t xSum_alpha = 0;
  scalar_t ySum_w = 0;
  scalar_t ySum_alpha = 0;
  scalar_t zSum_w = 0;
  scalar_t zSum_alpha = 0;
  scalar_t weightSum = 0;

  for (int kernelX = 0; kernelX < cKernelSizes[0]; kernelX++) {
    int neighbourX = max(0, min(homeX + (kernelX - cHalfWindowSize_arr[0]), cSizes[0] - 1));
    scalar_t gaussianX = cGaussianKernel_x[kernelX];

    for (int kernelY = 0; kernelY < cKernelSizes[1]; kernelY++) {
      int neighbourY = max(0, min(homeY + (kernelY - cHalfWindowSize_arr[1]), cSizes[1] - 1));
      scalar_t gaussianY = cGaussianKernel_y[kernelY];

      for (int kernelZ = 0; kernelZ < cKernelSizes[2]; kernelZ++) {
        int neighbourZ = max(0, min(homeZ + (kernelZ - cHalfWindowSize_arr[2]), cSizes[2] - 1));
        scalar_t gaussianZ = cGaussianKernel_z[kernelZ];

        int neighbourOffset = neighbourX * cStrides[0] + neighbourY * cStrides[1] + neighbourZ;

        bool flagNotClamped = true;
        int kernelIndex[] = {kernelX, kernelY, kernelZ};
        int dimensions = 3; // Must equal the number of spatial dimensions.

        for (int i = 0; i < dimensions; i++) {
          int HalfWindowSizeBack = cHalfWindowSize_arr[i]; // Define constant memory as new variable here (!!),
                                                           // otherwise: cudaErrorMisalignedAddress
          int neighbourIndex = homeIndex[i] + kernelIndex[i] - HalfWindowSizeBack;
          int neighbourIndexClamped = min(cSizes[i] - 1, max(0, neighbourIndex));
          if (neighbourIndex != neighbourIndexClamped) {
            flagNotClamped = false;
          }
        }

        scalar_t colorDistance = 0;
        scalar_t colorDistanceSquared = 0;

#pragma unroll
        for (int c = 0; c < C; c++) {
          scalar_t a = input[batchOffset + homeOffset + c * cColorStride];
          scalar_t b = input[batchOffset + neighbourOffset + c * cColorStride]; // Home - neighbor (!!) in backward the
                                                                                // other way around !!
          scalar_t diff = a - b;
          colorDistance += diff; // Do not take the absolute value here. Be careful with the signs.
          colorDistanceSquared += diff * diff;
        }

        scalar_t spatialWeight = gaussianX * gaussianY * gaussianZ;
        scalar_t colorWeight = exp(cColorExponentConstant * colorDistanceSquared);
        scalar_t totalWeight = spatialWeight * colorWeight;

        // Aggregating values. Only do this if flagNotClamped: Pixels outside the image are disregarded.
        if (flagNotClamped) {
#pragma unroll
          for (int c = 0; c < C; c++) {
            valueSum += input[batchOffset + neighbourOffset + c * cColorStride] * totalWeight;

            // Derivative of weights with respect to X_i while i=k.
            dw_dx_ki += (-1) * totalWeight * colorDistance / (cColorSigma * cColorSigma);
            // Derivative of convolved image with respect to X_i while i=k.
            dfilter_dx_ki += (-1) * totalWeight * input[batchOffset + neighbourOffset + c * cColorStride] *
                colorDistance /
                (cColorSigma *
                 cColorSigma); // Be careful, the +1 is missing here -> Added before filling dfilter_dx_kiData

            colorSum_w += totalWeight * colorDistanceSquared / std::abs(cColorSigma * cColorSigma * cColorSigma);
            colorSum_alpha += totalWeight * input[batchOffset + neighbourOffset + c * cColorStride] *
                colorDistanceSquared / std::abs(cColorSigma * cColorSigma * cColorSigma);

            xSum_w += totalWeight * cXDistanceSquared[kernelX] / std::abs(cSigma_x * cSigma_x * cSigma_x);
            xSum_alpha += totalWeight * input[batchOffset + neighbourOffset + c * cColorStride] *
                cXDistanceSquared[kernelX] / std::abs(cSigma_x * cSigma_x * cSigma_x);

            ySum_w += totalWeight * cYDistanceSquared[kernelY] / std::abs(cSigma_y * cSigma_y * cSigma_y);
            ySum_alpha += totalWeight * input[batchOffset + neighbourOffset + c * cColorStride] *
                cYDistanceSquared[kernelY] / std::abs(cSigma_y * cSigma_y * cSigma_y);

            zSum_w += totalWeight * cZDistanceSquared[kernelZ] / std::abs(cSigma_z * cSigma_z * cSigma_z);
            zSum_alpha += totalWeight * input[batchOffset + neighbourOffset + c * cColorStride] *
                cZDistanceSquared[kernelZ] / std::abs(cSigma_z * cSigma_z * cSigma_z);
          }

          weightSum += totalWeight;
        }
      }
    }
  }

#pragma unroll
  for (int c = 0; c < C; c++) {
    //    output[batchOffset + homeOffset + c * cColorStride] /= weightSum;
    output[batchOffset + homeOffset + c * cColorStride] = valueSum / weightSum;

    // Pre-computations for the backward pass:
    outputWeightsTensor[batchOffset + homeOffset + c * cColorStride] = weightSum;
    dO_dx_ki[batchOffset + homeOffset + c * cColorStride] = -(1 / weightSum) * (valueSum / weightSum) * dw_dx_ki +
        (1 / weightSum) * (dfilter_dx_ki + 1); // +1 for dfilter_dx_ki is added here
    dO_dsig_r[batchOffset + homeOffset + c * cColorStride] =
        -(1 / weightSum) * (valueSum / weightSum) * colorSum_w + (1 / weightSum) * colorSum_alpha;
    dO_dsig_x[batchOffset + homeOffset + c * cColorStride] =
        -(1 / weightSum) * (valueSum / weightSum) * xSum_w + (1 / weightSum) * xSum_alpha;
    dO_dsig_y[batchOffset + homeOffset + c * cColorStride] =
        -(1 / weightSum) * (valueSum / weightSum) * ySum_w + (1 / weightSum) * ySum_alpha;
    dO_dsig_z[batchOffset + homeOffset + c * cColorStride] =
        -(1 / weightSum) * (valueSum / weightSum) * zSum_w + (1 / weightSum) * zSum_alpha;
  }
}

template <int C, int D>
void BilateralFilterCudaForwardFunction(
    torch::Tensor inputTensor,
    torch::Tensor outputTensor,
    torch::Tensor outputWeightsTensor,
    torch::Tensor dO_dx_ki,
    torch::Tensor dO_dsig_r,
    torch::Tensor dO_dsig_x,
    torch::Tensor dO_dsig_y,
    torch::Tensor dO_dsig_z,
    float sigma_x,
    float sigma_y,
    float sigma_z,
    float colorSigma) {
  // Getting tensor description.
  TensorDescription desc = TensorDescription(inputTensor);

  // Pre-calculating gaussian kernel.
  int windowSize_x = std::max(((int)ceil(5.0f * sigma_x) | 1), 5); // ORing last bit to ensure odd window size
  int windowSize_y = std::max(((int)ceil(5.0f * sigma_y) | 1), 5); // ORing last bit to ensure odd window size
  int windowSize_z = std::max(((int)ceil(5.0f * sigma_z) | 1), 5); // ORing last bit to ensure odd window size
  int halfWindowSize_x = floor(0.5f * windowSize_x);
  int halfWindowSize_y = floor(0.5f * windowSize_y);
  int halfWindowSize_z = floor(0.5f * windowSize_z);
  int halfWindowSize_arr[] = {halfWindowSize_x, halfWindowSize_y, halfWindowSize_z};
  float spatialExpConstant_x = -1.0f / (2 * sigma_x * sigma_x);
  float spatialExpConstant_y = -1.0f / (2 * sigma_y * sigma_y);
  float spatialExpConstant_z = -1.0f / (2 * sigma_z * sigma_z);
  float colorExpConstant = -1.0f / (2 * colorSigma * colorSigma);

  int* kernelSizes = new int[desc.dimensions];
  kernelSizes[0] = windowSize_x;
  kernelSizes[1] = windowSize_y;
  kernelSizes[2] = windowSize_z;

  auto* gaussianKernel_x = new float[windowSize_x];
  auto* gaussianKernel_y = new float[windowSize_y];
  auto* gaussianKernel_z = new float[windowSize_z];
  auto* xDistanceSquared = new float[windowSize_x];
  auto* yDistanceSquared = new float[windowSize_y];
  auto* zDistanceSquared = new float[windowSize_z];

  for (int i = 0; i < windowSize_x; i++) {
    int distance = i - halfWindowSize_x;
    gaussianKernel_x[i] = exp(distance * distance * spatialExpConstant_x);
    xDistanceSquared[i] = distance * distance;
  }
  for (int i = 0; i < windowSize_y; i++) {
    int distance = i - halfWindowSize_y;
    gaussianKernel_y[i] = exp(distance * distance * spatialExpConstant_y);
    yDistanceSquared[i] = distance * distance;
  }
  for (int i = 0; i < windowSize_z; i++) {
    int distance = i - halfWindowSize_z;
    gaussianKernel_z[i] = exp(distance * distance * spatialExpConstant_z);
    zDistanceSquared[i] = distance * distance;
  }

  // Writing constant memory.
  cudaMemcpyToSymbol(cBatchStride, &desc.batchStride, sizeof(int));
  cudaMemcpyToSymbol(cColorStride, &desc.channelStride, sizeof(int));
  cudaMemcpyToSymbol(cSizes, desc.sizes, sizeof(int) * 3);
  cudaMemcpyToSymbol(cStrides, desc.strides, sizeof(int) * 3);
  cudaMemcpyToSymbol(cKernelSizes, kernelSizes, sizeof(int) * desc.dimensions);
  cudaMemcpyToSymbol(cHalfWindowSize_arr, halfWindowSize_arr, sizeof(int) * desc.dimensions);
  cudaMemcpyToSymbol(cGaussianKernel_x, gaussianKernel_x, sizeof(float) * windowSize_x);
  cudaMemcpyToSymbol(cGaussianKernel_y, gaussianKernel_y, sizeof(float) * windowSize_y);
  cudaMemcpyToSymbol(cGaussianKernel_z, gaussianKernel_z, sizeof(float) * windowSize_z);
  cudaMemcpyToSymbol(cXDistanceSquared, xDistanceSquared, sizeof(float) * windowSize_x);
  cudaMemcpyToSymbol(cYDistanceSquared, yDistanceSquared, sizeof(float) * windowSize_y);
  cudaMemcpyToSymbol(cZDistanceSquared, zDistanceSquared, sizeof(float) * windowSize_z);
  cudaMemcpyToSymbol(cColorExponentConstant, &colorExpConstant, sizeof(float));
  cudaMemcpyToSymbol(cSigma_x, &sigma_x, sizeof(float));
  cudaMemcpyToSymbol(cSigma_y, &sigma_y, sizeof(float));
  cudaMemcpyToSymbol(cSigma_z, &sigma_z, sizeof(float));
  cudaMemcpyToSymbol(cColorSigma, &colorSigma, sizeof(float));

  //  cuda_error_check("Cuda check before kernel call.");

#define BLOCK_SIZE 32

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      inputTensor.scalar_type(), "BilateralFilterCudaKernel3DForward", ([&] {
        BilateralFilterCudaKernel3DForward<scalar_t, C>
            <<<dim3(int(desc.channelStride / BLOCK_SIZE) + 1, desc.batchCount), dim3(BLOCK_SIZE, 1)>>>(
                inputTensor.data_ptr<scalar_t>(),
                outputTensor.data_ptr<scalar_t>(),
                outputWeightsTensor.data_ptr<scalar_t>(),
                dO_dx_ki.data_ptr<scalar_t>(),
                dO_dsig_r.data_ptr<scalar_t>(),
                dO_dsig_x.data_ptr<scalar_t>(),
                dO_dsig_y.data_ptr<scalar_t>(),
                dO_dsig_z.data_ptr<scalar_t>());
      }));

  //  cuda_error_check("Cuda check after kernel call.");
  //  delete[] kernel;
  delete[] kernelSizes;
  delete[] gaussianKernel_x;
  delete[] gaussianKernel_y;
  delete[] gaussianKernel_z;
  delete[] xDistanceSquared;
  delete[] yDistanceSquared;
  delete[] zDistanceSquared;
}

// Function to choose template implementation based on dynamic, channels and dimensions
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BilateralFilterCudaForward(torch::Tensor inputTensor, float sigma_x, float sigma_y, float sigma_z, float colorSigma) {
  torch::Tensor outputTensor = torch::zeros_like(inputTensor);
  torch::Tensor outputWeightsTensor = torch::zeros_like(inputTensor);
  torch::Tensor dO_dx_ki = torch::zeros_like(inputTensor);
  torch::Tensor dO_dsig_r = torch::zeros_like(inputTensor);
  torch::Tensor dO_dsig_x = torch::zeros_like(inputTensor);
  torch::Tensor dO_dsig_y = torch::zeros_like(inputTensor);
  torch::Tensor dO_dsig_z = torch::zeros_like(inputTensor);
  //  cuda_error_check("beginning");

#define CASE(c, d)                          \
  BilateralFilterCudaForwardFunction<c, d>( \
      inputTensor,                          \
      outputTensor,                         \
      outputWeightsTensor,                  \
      dO_dx_ki,                             \
      dO_dsig_r,                            \
      dO_dsig_x,                            \
      dO_dsig_y,                            \
      dO_dsig_z,                            \
      sigma_x,                              \
      sigma_y,                              \
      sigma_z,                              \
      colorSigma);
  SWITCH_AB(CASE, BF_CUDA_MAX_CHANNELS, BF_CUDA_MAX_SPATIAL_DIMENSION, inputTensor.size(1), inputTensor.dim() - 2);

  return {outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_r, dO_dsig_x, dO_dsig_y, dO_dsig_z};
}
