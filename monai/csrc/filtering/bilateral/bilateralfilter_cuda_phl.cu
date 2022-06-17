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
#include "filtering/permutohedral/permutohedral.h"
#include "utils/meta_macros.h"
#include "utils/tensor_description.h"

__constant__ int cBatchStride;
__constant__ int cChannelStride;
__constant__ int cSpatialStrides[3];
__constant__ float cInvSpatialSigma;
__constant__ float cInvColorSigma;

template <typename scalar_t, int C, int D>
__global__ void FeatureCreation(const scalar_t* inputTensor, scalar_t* outputData, scalar_t* outputFeatures) {
  int elementIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int batchIndex = blockIdx.y;

  if (elementIndex >= cChannelStride)
    return;

  int dataBatchOffset = batchIndex * cBatchStride;
  int featureBatchOffset = batchIndex * (D + C) * cChannelStride;

#pragma unroll
  for (int i = 0; i < C; i++) {
    outputData[dataBatchOffset + elementIndex * C + i] =
        inputTensor[dataBatchOffset + elementIndex + i * cChannelStride];
    outputFeatures[featureBatchOffset + elementIndex * (C + D) + i] =
        inputTensor[dataBatchOffset + elementIndex + i * cChannelStride] * cInvColorSigma;
  }

  int remainder = elementIndex;

#pragma unroll
  for (int i = 0; i < D; i++) {
    int coord = remainder / cSpatialStrides[i];
    remainder -= coord * cSpatialStrides[i];

    outputFeatures[featureBatchOffset + elementIndex * (C + D) + C + i] = coord * cInvSpatialSigma;
  }
}

template <typename scalar_t, int C>
__global__ void WriteOutput(const scalar_t* data, scalar_t* outputTensor) {
  int elementIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int batchIndex = blockIdx.y;

  if (elementIndex >= cChannelStride)
    return;

  int batchOffset = batchIndex * cBatchStride;

#pragma unroll
  for (int i = 0; i < C; i++) {
    outputTensor[batchOffset + elementIndex + i * cChannelStride] = data[batchOffset + elementIndex * C + i];
  }
}

template <typename scalar_t, int C, int D>
void BilateralFilterPHLCuda(
    torch::Tensor inputTensor,
    torch::Tensor outputTensor,
    float spatialSigma,
    float colorSigma) {
  // Getting tensor description.
  TensorDescription desc = TensorDescription(inputTensor);

  int featureChannelCount = desc.channelCount + desc.dimensions;

  // Pre calculating inverse sigmas.
  float invSpatialSigma = 1.0f / spatialSigma;
  float invColorSigma = 1.0f / colorSigma;

  // Preparing global memory
  scalar_t* inputTensorData = inputTensor.data_ptr<scalar_t>();
  scalar_t* outputTensorData = outputTensor.data_ptr<scalar_t>();

  scalar_t* data;
  scalar_t* features;
  cudaMalloc(&data, desc.batchCount * desc.channelStride * desc.channelCount * sizeof(scalar_t));
  cudaMalloc(&features, desc.batchCount * desc.channelStride * featureChannelCount * sizeof(scalar_t));

  // Preparing constant memory
  cudaMemcpyToSymbol(cBatchStride, &desc.batchStride, sizeof(int));
  cudaMemcpyToSymbol(cChannelStride, &desc.channelStride, sizeof(int));
  cudaMemcpyToSymbol(cSpatialStrides, desc.strides, sizeof(int) * desc.dimensions);
  cudaMemcpyToSymbol(cInvSpatialSigma, &invSpatialSigma, sizeof(float));
  cudaMemcpyToSymbol(cInvColorSigma, &invColorSigma, sizeof(float));

#define BLOCK_SIZE 32

  // Creating features
  FeatureCreation<scalar_t, C, D>
      <<<dim3(int(desc.channelStride / BLOCK_SIZE) + 1, desc.batchCount), dim3(BLOCK_SIZE, 1)>>>(
          inputTensorData, data, features);

  // Filtering data with respect to the features for each sample in batch
  for (int batchIndex = 0; batchIndex < desc.batchCount; batchIndex++) {
    scalar_t* offsetData = data + batchIndex * desc.batchStride;
    scalar_t* offsetFeatures = features + batchIndex * featureChannelCount * desc.channelStride;

    PermutohedralCuda<scalar_t, C, C + D>(offsetData, offsetFeatures, desc.channelStride, true);
  }

  // Writing output
  WriteOutput<scalar_t, C><<<dim3(int(desc.channelStride / BLOCK_SIZE) + 1, desc.batchCount), dim3(BLOCK_SIZE, 1)>>>(
      data, outputTensorData);

  cudaFree(data);
  cudaFree(features);
}

// Function to choose template implementation based on dynamic, channels and dimensions
torch::Tensor BilateralFilterPHLCuda(torch::Tensor inputTensor, float spatialSigma, float colorSigma) {
  torch::Tensor outputTensor = torch::zeros_like(inputTensor);

#define CASE(c, d)                                                                       \
  AT_DISPATCH_FLOATING_TYPES(inputTensor.scalar_type(), "BilateralFilterCudaPHL", ([&] { \
                               BilateralFilterPHLCuda<scalar_t, c, d>(                   \
                                   inputTensor, outputTensor, spatialSigma, colorSigma); \
                             }));

  SWITCH_AB(CASE, BF_CUDA_MAX_CHANNELS, BF_CUDA_MAX_SPATIAL_DIMENSION, inputTensor.size(1), inputTensor.dim() - 2);

  return outputTensor;
}
