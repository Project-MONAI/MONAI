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

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "bilateral.h"
#include "utils/meta_macros.h"
#include "utils/tensor_description.h"

__constant__ int cBatchStride;
__constant__ int cColorStride;

__constant__ int cSizes[3];
__constant__ int cStrides[3];

__constant__ int cKernelSize;
__constant__ float cKernel[256];

__constant__ float cColorExponentFactor;

template <typename scalar_t, int C>
__global__ void BilateralFilterCudaKernel1D(scalar_t* input, scalar_t* output) {
  int kernelHalfSize = cKernelSize / 2;

  int homeOffset = blockIdx.x * blockDim.x + threadIdx.x;
  int batchOffset = blockIdx.y * cBatchStride;

  if (homeOffset >= cColorStride)
    return;

  scalar_t weightSum = 0;

  for (int kernelOffset = 0; kernelOffset < cKernelSize; kernelOffset++) {
    int neighbourOffset = max(0, min(homeOffset + (kernelOffset - kernelHalfSize), cSizes[0] - 1));
    scalar_t gaussian = cKernel[kernelOffset];

    scalar_t distanceSquared = 0;

#pragma unroll
    for (int c = 0; c < C; c++) {
      scalar_t a = input[batchOffset + homeOffset + c * cColorStride];
      scalar_t b = input[batchOffset + neighbourOffset + c * cColorStride];
      scalar_t diff = a - b;
      distanceSquared += diff * diff;
    }

    scalar_t spatialWeight = gaussian;
    scalar_t colorWeight = exp(cColorExponentFactor * distanceSquared);
    scalar_t totalWeight = spatialWeight * colorWeight;

#pragma unroll
    for (int c = 0; c < C; c++) {
      scalar_t a = input[batchOffset + neighbourOffset + c * cColorStride];

      output[batchOffset + homeOffset + c * cColorStride] += a * totalWeight;
    }

    weightSum += totalWeight;
  }

#pragma unroll
  for (int c = 0; c < C; c++) {
    output[batchOffset + homeOffset + c * cColorStride] /= weightSum;
  }
}

template <typename scalar_t, int C>
__global__ void BilateralFilterCudaKernel2D(scalar_t* input, scalar_t* output) {
  int kernelHalfSize = cKernelSize / 2;

  int homeOffset = blockIdx.x * blockDim.x + threadIdx.x;
  int batchOffset = blockIdx.y * cBatchStride;

  if (homeOffset >= cColorStride)
    return;

  int homeX = homeOffset / cStrides[0];
  int homeY = (homeOffset - homeX * cStrides[0]) / cStrides[1];

  scalar_t weightSum = 0;

  for (int kernelX = 0; kernelX < cKernelSize; kernelX++) {
    int neighbourX = max(0, min(homeX + (kernelX - kernelHalfSize), cSizes[0] - 1));
    scalar_t gaussianX = cKernel[kernelX];

    for (int kernelY = 0; kernelY < cKernelSize; kernelY++) {
      int neighbourY = max(0, min(homeY + (kernelY - kernelHalfSize), cSizes[1] - 1));
      scalar_t gaussianY = cKernel[kernelY];

      int neighbourOffset = neighbourX * cStrides[0] + neighbourY;

      scalar_t distanceSquared = 0;

#pragma unroll
      for (int c = 0; c < C; c++) {
        scalar_t a = input[batchOffset + homeOffset + c * cColorStride];
        scalar_t b = input[batchOffset + neighbourOffset + c * cColorStride];
        scalar_t diff = a - b;
        distanceSquared += diff * diff;
      }

      scalar_t spatialWeight = gaussianX * gaussianY;
      scalar_t colorWeight = exp(cColorExponentFactor * distanceSquared);
      scalar_t totalWeight = spatialWeight * colorWeight;

#pragma unroll
      for (int c = 0; c < C; c++) {
        scalar_t a = input[batchOffset + neighbourOffset + c * cColorStride];

        output[batchOffset + homeOffset + c * cColorStride] += a * totalWeight;
      }

      weightSum += totalWeight;
    }
  }

#pragma unroll
  for (int c = 0; c < C; c++) {
    output[batchOffset + homeOffset + c * cColorStride] /= weightSum;
  }
}

template <typename scalar_t, int C>
__global__ void BilateralFilterCudaKernel3D(scalar_t* input, scalar_t* output) {
  int kernelHalfSize = cKernelSize / 2;

  int homeOffset = blockIdx.x * blockDim.x + threadIdx.x;
  int batchOffset = blockIdx.y * cBatchStride;

  if (homeOffset >= cColorStride)
    return;

  int homeX = homeOffset / cStrides[0];
  int homeY = (homeOffset - homeX * cStrides[0]) / cStrides[1];
  int homeZ = (homeOffset - homeX * cStrides[0] - homeY * cStrides[1]) / cStrides[2];

  scalar_t weightSum = 0;

  for (int kernelX = 0; kernelX < cKernelSize; kernelX++) {
    int neighbourX = max(0, min(homeX + (kernelX - kernelHalfSize), cSizes[0] - 1));
    scalar_t gaussianX = cKernel[kernelX];

    for (int kernelY = 0; kernelY < cKernelSize; kernelY++) {
      int neighbourY = max(0, min(homeY + (kernelY - kernelHalfSize), cSizes[1] - 1));
      scalar_t gaussianY = cKernel[kernelY];

      for (int kernelZ = 0; kernelZ < cKernelSize; kernelZ++) {
        int neighbourZ = max(0, min(homeZ + (kernelZ - kernelHalfSize), cSizes[2] - 1));
        scalar_t gaussianZ = cKernel[kernelZ];

        int neighbourOffset = neighbourX * cStrides[0] + neighbourY * cStrides[1] + neighbourZ;

        scalar_t distanceSquared = 0;

#pragma unroll
        for (int c = 0; c < C; c++) {
          scalar_t a = input[batchOffset + homeOffset + c * cColorStride];
          scalar_t b = input[batchOffset + neighbourOffset + c * cColorStride];
          scalar_t diff = a - b;
          distanceSquared += diff * diff;
        }

        scalar_t spatialWeight = gaussianX * gaussianY * gaussianZ;
        scalar_t colorWeight = exp(cColorExponentFactor * distanceSquared);
        scalar_t totalWeight = spatialWeight * colorWeight;

#pragma unroll
        for (int c = 0; c < C; c++) {
          scalar_t a = input[batchOffset + neighbourOffset + c * cColorStride];
          output[batchOffset + homeOffset + c * cColorStride] += a * totalWeight;
        }

        weightSum += totalWeight;
      }
    }
  }

#pragma unroll
  for (int c = 0; c < C; c++) {
    output[batchOffset + homeOffset + c * cColorStride] /= weightSum;
  }
}

template <int C, int D>
void BilateralFilterCuda(torch::Tensor inputTensor, torch::Tensor outputTensor, float spatialSigma, float colorSigma) {
  // Getting tensor description.
  TensorDescription desc = TensorDescription(inputTensor);

  // Pre-calculating exponent factors.
  float spatialExponentFactor = -1.0f / (2 * spatialSigma * spatialSigma);
  float colorExponentFactor = -1.0f / (2 * colorSigma * colorSigma);

  // Pre-calculating gaussian kernel.
  int kernelSize = (int)ceil(5.0f * spatialSigma) | 1; // ORing last bit to ensure odd window size
  int kernelHalfSize = floor(0.5f * kernelSize);
  float* kernel = new float[kernelSize];

  for (int i = 0; i < kernelSize; i++) {
    int distance = i - kernelHalfSize;
    kernel[i] = exp(distance * distance * spatialExponentFactor);
  }

  // Writing constant memory.
  cudaMemcpyToSymbol(cBatchStride, &desc.batchStride, sizeof(int));
  cudaMemcpyToSymbol(cColorStride, &desc.channelStride, sizeof(int));
  cudaMemcpyToSymbol(cSizes, desc.sizes, sizeof(int) * D);
  cudaMemcpyToSymbol(cStrides, desc.strides, sizeof(int) * D);
  cudaMemcpyToSymbol(cKernelSize, &kernelSize, sizeof(int));
  cudaMemcpyToSymbol(cKernel, kernel, sizeof(float) * kernelSize);
  cudaMemcpyToSymbol(cColorExponentFactor, &colorExponentFactor, sizeof(float));

#define BLOCK_SIZE 32

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      inputTensor.scalar_type(), "BilateralFilterCudaKernel", ([&] {
        // Dispatch kernel. (Partial template function specialisation not supported at present so using this switch
        // instead)
        switch (D) {
          case (1):
            BilateralFilterCudaKernel1D<scalar_t, C>
                <<<dim3(int(desc.channelStride / BLOCK_SIZE) + 1, desc.batchCount), dim3(BLOCK_SIZE, 1)>>>(
                    inputTensor.data_ptr<scalar_t>(), outputTensor.data_ptr<scalar_t>());
            break;
          case (2):
            BilateralFilterCudaKernel2D<scalar_t, C>
                <<<dim3(int(desc.channelStride / BLOCK_SIZE) + 1, desc.batchCount), dim3(BLOCK_SIZE, 1)>>>(
                    inputTensor.data_ptr<scalar_t>(), outputTensor.data_ptr<scalar_t>());
            break;
          case (3):
            BilateralFilterCudaKernel3D<scalar_t, C>
                <<<dim3(int(desc.channelStride / BLOCK_SIZE) + 1, desc.batchCount), dim3(BLOCK_SIZE, 1)>>>(
                    inputTensor.data_ptr<scalar_t>(), outputTensor.data_ptr<scalar_t>());
            break;
        }
      }));

  delete[] kernel;
}

// Function to choose template implementation based on dynamic, channels and dimensions
torch::Tensor BilateralFilterCuda(torch::Tensor inputTensor, float spatialSigma, float colorSigma) {
  torch::Tensor outputTensor = torch::zeros_like(inputTensor);

#define CASE(c, d) BilateralFilterCuda<c, d>(inputTensor, outputTensor, spatialSigma, colorSigma);
  SWITCH_AB(CASE, BF_CUDA_MAX_CHANNELS, BF_CUDA_MAX_SPATIAL_DIMENSION, inputTensor.size(1), inputTensor.dim() - 2);

  return outputTensor;
}
