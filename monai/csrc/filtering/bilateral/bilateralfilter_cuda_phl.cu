/*
Copyright 2020 MONAI Consortium
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
#include <cuda.h>
#include <cuda_runtime.h>

#include "utils/common_utils.h"
#include "../permutohedral/permutohedral.h"

__constant__ int cBatchStride;
__constant__ int cChannelStride;
__constant__ int cSpatialStrides[3];
__constant__ float cInvSpatialSigma;
__constant__ float cInvColorSigma;

template <int C, int D>
__global__ void FeatureCreation(const float* inputTensor, float* outputData, float* outputFeatures)
{
    int elementIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int batchOffset = blockIdx.y * cBatchStride;

    #pragma unroll
    for (int i = 0; i < C; i++)
    {
        outputData[batchOffset + elementIndex * C + i] = inputTensor[batchOffset + elementIndex + i * cChannelStride];
        outputFeatures[batchOffset + elementIndex * (C + D) + i] = inputTensor[batchOffset + elementIndex + i * cChannelStride] * cInvColorSigma;
    }

    int remainder = elementIndex;

    #pragma unroll
    for (int i = 0; i < D; i++)
    {
        int coord = remainder / cSpatialStrides[i];
        remainder -= coord * cSpatialStrides[i];

        outputFeatures[batchOffset + elementIndex * (C + D) + C + i] = coord * cInvSpatialSigma;
    }
}

template <int C>
__global__ void WriteOutput(const float* data, float* outputTensor)
{
    int elementIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int batchIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int batchOffset = batchIndex * cBatchStride;

    #pragma unroll
    for (int i = 0; i < C; i++)
    {
        outputTensor[batchOffset + elementIndex + i * cChannelStride] = data[batchOffset + elementIndex * C + i];
    }
}

template<int C, int D>
void RunFilter(torch::Tensor inputTensor, torch::Tensor outputTensor, float spatialSigma, float colorSigma)
{
    // Getting tensor descriptors
    int dimensions = inputTensor.dim() - 2;
    int* strides = new int[dimensions];
    int elementCount = 1;

    for (int i = 0; i < dimensions; i++)
    {
        strides[i] = inputTensor.stride(i+2);
        elementCount *= inputTensor.size(i+2);
    }

    int batchStride = inputTensor.stride(0);
    int batchCount = inputTensor.size(0);

    int channelStride = inputTensor.stride(1);
    int dataChannels = inputTensor.size(1);
    int featureChannels = dataChannels + dimensions;

    float* data;
    float* features;

    cudaMalloc(&data, elementCount * dataChannels * sizeof(float));
    cudaMalloc(&features, elementCount * featureChannels * sizeof(float));

    float invSpatialSigma = 1.0f/spatialSigma;
    float invColorSigma = 1.0f/colorSigma;

    cudaMemcpyToSymbol(cBatchStride, &batchStride, sizeof(int));
    cudaMemcpyToSymbol(cChannelStride, &channelStride, sizeof(int));
    cudaMemcpyToSymbol(cSpatialStrides, strides, sizeof(int) * dimensions);
    cudaMemcpyToSymbol(cInvSpatialSigma, &invSpatialSigma, sizeof(float));
    cudaMemcpyToSymbol(cInvColorSigma, &invColorSigma, sizeof(float));

    // Creating features
    FeatureCreation<C, D><<<dim3(elementCount, batchCount), dim3(1, 1)>>>(inputTensor.data_ptr<float>(), data, features);

    // Filtering data with respect to the features for each sample in batch
    for (int batchIndex = 0; batchIndex < batchCount; batchIndex++)
    {
        PermutohedralCuda(data + batchIndex * batchStride, features + batchIndex * batchStride, dataChannels, featureChannels, elementCount);
    }

    // Writing output
    WriteOutput<C><<<dim3(elementCount, batchCount), dim3(1, 1)>>>(data, outputTensor.data_ptr<float>());

    cudaFree(data);
    cudaFree(features);
}

torch::Tensor BilateralFilterPHLCuda(torch::Tensor inputTensor, float spatialSigma, float colorSigma)
{
    torch::Tensor outputTensor = torch::zeros_like(inputTensor);

    SPECIALISE_C_AND_D(inputTensor.size(1), inputTensor.dim()-2, RunFilter, inputTensor, outputTensor, spatialSigma, colorSigma);

    return outputTensor;
}
