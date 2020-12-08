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

__constant__ int cBatchStride;
__constant__ int cColorStride;

__constant__ int cSizes[3];
__constant__ int cStrides[3];

__constant__ int cKernelSize;
__constant__ float cKernel[256];

__constant__ float cColorExponentFactor;

template<int C>
__global__ void BilateralFilterCudaKernel1D(float* input, float* output)
{
    int kernelHalfSize = cKernelSize / 2;

    int homeOffset = blockIdx.x * blockDim.x + threadIdx.x;
    int batchOffset = blockIdx.y * cBatchStride;

    float weightSum = 0;

    for(int kernelOffset = 0; kernelOffset < cKernelSize; kernelOffset++)
    {
        int neighbourOffset = max(0, min(homeOffset + (kernelOffset - kernelHalfSize), cSizes[0] - 1));
        float gaussian = cKernel[kernelOffset];
        
        float distanceSquared = 0;

        #pragma unroll
        for(int c = 0; c < C; c++)
        {
            float a = input[batchOffset + homeOffset + c * cColorStride];
            float b = input[batchOffset + neighbourOffset + c * cColorStride];
            float diff = a - b;
            distanceSquared += diff * diff;
        }

        float spatialWeight = gaussian;
        float colorWeight = exp(cColorExponentFactor * distanceSquared);
        float totalWeight = spatialWeight * colorWeight;
        
        #pragma unroll
        for(int c = 0; c < C; c++)
        {
            float a = input[batchOffset + neighbourOffset + c * cColorStride];

            output[batchOffset + homeOffset + c * cColorStride] += a * totalWeight;
        }

        weightSum += totalWeight;
    }

    #pragma unroll
    for(int c = 0; c < C; c++)
    {
        output[batchOffset + homeOffset + c * cColorStride] /= weightSum;
    }
}

template<int C>
__global__ void BilateralFilterCudaKernel2D(float* input, float* output)
{
    int kernelHalfSize = cKernelSize / 2;

    int homeOffset = blockIdx.x * blockDim.x + threadIdx.x;
    int batchOffset = blockIdx.y * cBatchStride;

    int homeX = homeOffset / cStrides[0];
    int homeY = (homeOffset - homeX * cStrides[0]) / cStrides[1];

    float weightSum = 0;

    for(int kernelX = 0; kernelX < cKernelSize; kernelX++)
    {
        int neighbourX = max(0, min(homeX + (kernelX - kernelHalfSize), cSizes[0] - 1));
        float gaussianX = cKernel[kernelX];

        for(int kernelY = 0; kernelY < cKernelSize; kernelY++)
        {
            int neighbourY = max(0, min(homeY + (kernelY - kernelHalfSize), cSizes[1] - 1));
            float gaussianY = cKernel[kernelY];
          
            int neighbourOffset = neighbourX * cStrides[0] + neighbourY;
            
            float distanceSquared = 0;

            #pragma unroll
            for(int c = 0; c < C; c++)
            {
                float a = input[batchOffset + homeOffset + c * cColorStride];
                float b = input[batchOffset + neighbourOffset + c * cColorStride];
                float diff = a - b;
                distanceSquared += diff * diff;
            }

            float spatialWeight = gaussianX * gaussianY;
            float colorWeight = exp(cColorExponentFactor * distanceSquared);
            float totalWeight = spatialWeight * colorWeight;
            
            #pragma unroll
            for(int c = 0; c < C; c++)
            {
                float a = input[batchOffset + neighbourOffset + c * cColorStride];

                output[batchOffset + homeOffset + c * cColorStride] += a * totalWeight;
            }

            weightSum += totalWeight;
        }
    }

    #pragma unroll
    for(int c = 0; c < C; c++)
    {
        output[batchOffset + homeOffset + c * cColorStride] /= weightSum;
    }
}

template<int C>
__global__ void BilateralFilterCudaKernel3D(float* input, float* output)
{
    int kernelHalfSize = cKernelSize / 2;

    int homeOffset = blockIdx.x * blockDim.x + threadIdx.x;
    int batchOffset = blockIdx.y * cBatchStride;

    int homeX = homeOffset / cStrides[0];
    int homeY = (homeOffset - homeX * cStrides[0]) / cStrides[1];
    int homeZ = (homeOffset - homeX * cStrides[0] - homeY * cStrides[1]) / cStrides[2];

    float weightSum = 0;

    for(int kernelX = 0; kernelX < cKernelSize; kernelX++)
    {
        int neighbourX = max(0, min(homeX + (kernelX - kernelHalfSize), cSizes[0] - 1));
        float gaussianX = cKernel[kernelX];

        for(int kernelY = 0; kernelY < cKernelSize; kernelY++)
        {
            int neighbourY = max(0, min(homeY + (kernelY - kernelHalfSize), cSizes[1] - 1));
            float gaussianY = cKernel[kernelY];

            for(int kernelZ = 0; kernelZ < cKernelSize; kernelZ++)
            {
                int neighbourZ = max(0, min(homeZ + (kernelZ - kernelHalfSize), cSizes[2] - 1));
                float gaussianZ = cKernel[kernelZ];
            
                int neighbourOffset = neighbourX * cStrides[0] + neighbourY * cStrides[1] + neighbourZ;
                
                float distanceSquared = 0;

                #pragma unroll
                for(int c = 0; c < C; c++)
                {
                    float a = input[batchOffset + homeOffset + c * cColorStride];
                    float b = input[batchOffset + neighbourOffset + c * cColorStride];
                    float diff = a - b;
                    distanceSquared += diff * diff;
                }

                float spatialWeight = gaussianX * gaussianY * gaussianZ;
                float colorWeight = exp(cColorExponentFactor * distanceSquared);
                float totalWeight = spatialWeight * colorWeight;
                
                #pragma unroll
                for(int c = 0; c < C; c++)
                {
                    float a = input[batchOffset + neighbourOffset + c * cColorStride];
                    output[batchOffset + homeOffset + c * cColorStride] += a * totalWeight;
                }

                weightSum += totalWeight;
            }
        }
    }

    #pragma unroll
    for(int c = 0; c < C; c++)
    {
        output[batchOffset + homeOffset + c * cColorStride] /= weightSum;
    }
}

template<int C, int D>
void BilateralFilterCuda(torch::Tensor inputTensor, torch::Tensor outputTensor, float spatialSigma, float colorSigma)
{
    // Getting tensor description.
    TensorDescription desc = TensorDescription(inputTensor);

    // Pre-calculating exponent factors.
    float spatialExponentFactor = -1.0f / (2 * spatialSigma * spatialSigma);
    float colorExponentFactor = -1.0f / (2 * colorSigma * colorSigma);
    
    // Pre-calculating gaussian kernel.
    int kernelSize = ceil(3.0f * spatialSigma);
    int kernelHalfSize = floor(0.5f * kernelSize);
    float* kernel = new float[kernelSize];

    for (int i = 0; i < kernelSize; i++)
    {
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

    // Dispatch kernel. (Partial template function specialisation not supported at present so using this switch instead)
    switch(D)
    {
        case(1): BilateralFilterCudaKernel1D<C><<<dim3(desc.channelStride, desc.batchCount), dim3(1, 1)>>>(inputTensor.data_ptr<float>(), outputTensor.data_ptr<float>()); break;
        case(2): BilateralFilterCudaKernel2D<C><<<dim3(desc.channelStride, desc.batchCount), dim3(1, 1)>>>(inputTensor.data_ptr<float>(), outputTensor.data_ptr<float>()); break;
        case(3): BilateralFilterCudaKernel3D<C><<<dim3(desc.channelStride, desc.batchCount), dim3(1, 1)>>>(inputTensor.data_ptr<float>(), outputTensor.data_ptr<float>()); break;
    }

    delete[] kernel;
}

torch::Tensor BilateralFilterCuda(torch::Tensor inputTensor, float spatialSigma, float colorSigma)
{
    torch::Tensor outputTensor = torch::zeros_like(inputTensor);

    SPECIALISE_C_AND_D(inputTensor.size(1), inputTensor.dim()-2, BilateralFilterCuda, inputTensor, outputTensor, spatialSigma, colorSigma);

    return outputTensor;
}
