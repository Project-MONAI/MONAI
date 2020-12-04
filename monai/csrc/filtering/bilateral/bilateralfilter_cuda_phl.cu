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
#include "utils/tensor_description.h"
#include "filtering/permutohedral/permutohedral.h"

__constant__ int cBatchStride;
__constant__ int cChannelStride;
__constant__ int cSpatialStrides[3];
__constant__ float cInvSpatialSigma;
__constant__ float cInvColorSigma;

template <int C, int D>
__global__ void FeatureCreation(const float* inputTensor, float* outputData, float* outputFeatures)
{
    int elementIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int batchIndex= blockIdx.y;

    int dataBatchOffset = batchIndex * cBatchStride;
    int featureBatchOffset = batchIndex * (D + C) * cChannelStride;

    #pragma unroll
    for (int i = 0; i < C; i++)
    {
        outputData[dataBatchOffset + elementIndex * C + i] = inputTensor[dataBatchOffset + elementIndex + i * cChannelStride];
        outputFeatures[featureBatchOffset + elementIndex * (C + D) + i] = inputTensor[dataBatchOffset + elementIndex + i * cChannelStride] * cInvColorSigma;
    }

    int remainder = elementIndex;

    #pragma unroll
    for (int i = 0; i < D; i++)
    {
        int coord = remainder / cSpatialStrides[i];
        remainder -= coord * cSpatialStrides[i];

        outputFeatures[featureBatchOffset + elementIndex * (C + D) + C + i] = coord * cInvSpatialSigma;
    }
}

template <int C>
__global__ void WriteOutput(const float* data, float* outputTensor)
{
    int elementIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int batchIndex= blockIdx.y;
    int batchOffset = batchIndex * cBatchStride;

    #pragma unroll
    for (int i = 0; i < C; i++)
    {
        outputTensor[batchOffset + elementIndex + i * cChannelStride] = data[batchOffset + elementIndex * C + i];
    }
}

template<int C, int D>
void BilateralFilterPHLCuda(torch::Tensor inputTensor, torch::Tensor outputTensor, float spatialSigma, float colorSigma)
{
    // Getting tensor description.
    TensorDescription desc = TensorDescription(inputTensor);

    int featureChannelCount = desc.channelCount + desc.dimensions;

    // Pre calculating inverse sigmas.
    float invSpatialSigma = 1.0f/spatialSigma;
    float invColorSigma = 1.0f/colorSigma;

    // Preparing global memory
    float* inputTensorData = inputTensor.data_ptr<float>();
    float* outputTensorData = outputTensor.data_ptr<float>();

    float* data;
    float* features;
    cudaMalloc(&data, desc.batchCount * desc.channelStride * desc.channelCount * sizeof(float));
    cudaMalloc(&features, desc.batchCount * desc.channelStride * featureChannelCount * sizeof(float));

    // Prparing constant memory
    cudaMemcpyToSymbol(cBatchStride, &desc.batchStride, sizeof(int));
    cudaMemcpyToSymbol(cChannelStride, &desc.channelStride, sizeof(int));
    cudaMemcpyToSymbol(cSpatialStrides, desc.strides, sizeof(int) * desc.dimensions);
    cudaMemcpyToSymbol(cInvSpatialSigma, &invSpatialSigma, sizeof(float));
    cudaMemcpyToSymbol(cInvColorSigma, &invColorSigma, sizeof(float));

    // Creating features
    FeatureCreation<C, D><<<dim3(desc.channelStride, desc.batchCount), dim3(1, 1)>>>(inputTensorData, data, features);

    // Filtering data with respect to the features for each sample in batch
    for (int batchIndex = 0; batchIndex < desc.batchCount; batchIndex++)
    {
        float* offsetData = data + batchIndex * desc.batchStride;
        float* offsetFeatures = features + batchIndex * featureChannelCount * desc.channelStride;

        PermutohedralCuda(offsetData, offsetFeatures, desc.channelCount, featureChannelCount, desc.channelStride);
    }

    // Writing output
    WriteOutput<C><<<dim3(desc.channelStride, desc.batchCount), dim3(1, 1)>>>(data, outputTensorData);

    cudaFree(data);
    cudaFree(features);
}

torch::Tensor BilateralFilterPHLCuda(torch::Tensor inputTensor, float spatialSigma, float colorSigma)
{
    torch::Tensor outputTensor = torch::zeros_like(inputTensor);

    SPECIALISE_C_AND_D(inputTensor.size(1), inputTensor.dim()-2, BilateralFilterPHLCuda, inputTensor, outputTensor, spatialSigma, colorSigma);

    return outputTensor;
}
