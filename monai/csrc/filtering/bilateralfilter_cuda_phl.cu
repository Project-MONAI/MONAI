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

//#include "permutohedral.cu"

void filter(float *im, float *ref, int pd, int vd, int w, int h, bool accurate);

__constant__ int cChannelCount;
__constant__ int cWidthStride;
__constant__ int cChannelStride;
__constant__ float cInvSpatialSigma;
__constant__ float cInvColorSigma;

__global__ void FeatureCreation(const float* inputTensor, float* outputData, float* outputFeatures)
{
    int dispatchIndex = blockIdx.x * blockDim.x + threadIdx.x;

    int x = dispatchIndex / cWidthStride;
    int y = (dispatchIndex - x * cWidthStride);
    
    for (int i = 0; i < cChannelCount; i++)
    {
        outputData[dispatchIndex * cChannelCount + i] = inputTensor[dispatchIndex + i * cChannelStride];
        outputFeatures[dispatchIndex * (cChannelCount + 2) + i] = inputTensor[dispatchIndex + i * cChannelStride] * cInvColorSigma;
    }

    outputFeatures[dispatchIndex * (cChannelCount + 2) + cChannelCount + 0] = x * cInvSpatialSigma;
    outputFeatures[dispatchIndex * (cChannelCount + 2) + cChannelCount + 1] = y * cInvSpatialSigma;
}

__global__ void WriteOutput(const float* data, float* outputTensor)
{
    int dispatchIndex = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < cChannelCount; i++)
    {
        outputTensor[dispatchIndex + i * cChannelStride] = data[dispatchIndex * cChannelCount + i];
    }
}

torch::Tensor BilateralFilterPHLCuda(torch::Tensor inputTensor, float spatialSigma, float colorSigma)
{
    torch::Tensor outputTensor = torch::zeros_like(inputTensor);
    
    int channelCount = inputTensor.size(1);
    int width = inputTensor.size(2);
    int height = inputTensor.size(3);

    int batchStride = inputTensor.stride(0);
    int channelStride = inputTensor.stride(1);
    int widthStride = inputTensor.stride(2);

    int elementCount = width * height;
    int featureCount = channelCount + 2;

    float* data;
    float* features;

    cudaMalloc(&data, elementCount * channelCount * sizeof(float));
    cudaMalloc(&features, elementCount * featureCount * sizeof(float));

    float invSpatialSigma = 1.0f/spatialSigma;
    float invColorSigma = 1.0f/colorSigma;

    cudaMemcpyToSymbol(cChannelCount, &channelCount, sizeof(int));
    cudaMemcpyToSymbol(cChannelStride, &channelStride, sizeof(int));
    cudaMemcpyToSymbol(cWidthStride, &widthStride, sizeof(int));
    cudaMemcpyToSymbol(cInvSpatialSigma, &invSpatialSigma, sizeof(float));
    cudaMemcpyToSymbol(cInvColorSigma, &invColorSigma, sizeof(float));

    //No batch handling at present. Need to either loop through or make changes to filter()

    FeatureCreation<<<dim3(elementCount, 1), dim3(1, 1)>>>(inputTensor.data_ptr<float>(), data, features);

    filter(data, features, featureCount, channelCount, width, height, true);

    WriteOutput<<<dim3(elementCount, 1), dim3(1, 1)>>>(data, outputTensor.data_ptr<float>());

    cudaFree(data);
    cudaFree(features);

    return outputTensor;
}
