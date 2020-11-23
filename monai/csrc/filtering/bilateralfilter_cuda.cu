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

__constant__ int cInputDimensions[5];
__constant__ int cInputStrides[5];

__constant__ int cKernelSize;
__constant__ float cKernel[256];

__constant__ float cColorExponentFactor;

__global__ void BilateralFilterCudaKernel(float* input, float* output)
{
    int batchCount = cInputDimensions[0];
    int channelCount = cInputDimensions[1];
    int width = cInputDimensions[2];
    int height = cInputDimensions[3];

    int batchStride = cInputStrides[0];
    int channelStride = cInputStrides[1];
    int widthStride = cInputStrides[2];
    int heightStride = cInputStrides[3];

    int kernelSize = cKernelSize;
    int kernelHalfSize = (int)(kernelSize * 0.5f);

    int batchOffset = blockIdx.y * batchStride;

    int homeOffset = blockIdx.x * blockDim.x + threadIdx.x;
    int homeX = homeOffset / widthStride;
    int homeY = (homeOffset - homeX * widthStride) / heightStride;

    float weightSum = 0;

    for(int kernelX = 0; kernelX < kernelSize; kernelX++)
    {
        int neighbourX = max(0, min(homeX + (kernelX - kernelHalfSize), width));
        float gaussianX = cKernel[kernelX];

        for(int kernelY = 0; kernelY < kernelSize; kernelY++)
        {
            int neighbourY = max(0, min(homeY + (kernelY - kernelHalfSize), height));
            float gaussianY = cKernel[kernelX];
          
            int neighbourOffset = neighbourX * widthStride + neighbourY;
            

            float distanceSquared = 0;

            for(int i = 0; i < channelCount; i++)
            {
                float a = input[batchOffset + homeOffset + i * channelStride];
                float b = input[batchOffset + neighbourOffset + i * channelStride];

                distanceSquared += a*a + b*b;
            }

            float spatialWeight = gaussianX * gaussianY;
            float colorWeight = exp(cColorExponentFactor * distanceSquared);
            float totalWeight = spatialWeight * colorWeight;
            
            for(int i = 0; i < channelCount; i++)
            {
                float a = input[batchOffset + neighbourOffset + i * channelStride];

                output[batchOffset + homeOffset + i * channelStride] += a * totalWeight;
            }

            weightSum += totalWeight;
        }
    }

    for(int i = 0; i < channelCount; i++)
    {
        output[batchOffset + homeOffset + i * channelStride] /= weightSum;
    }
}

torch::Tensor BilateralFilterCuda(torch::Tensor input, float spatialSigma, float colorSigma)
{
    // Preparing output tensor.
    torch::Tensor output = torch::zeros_like(input);

    // Gathering and input description.
    int* inputDimensions = new int[5];
    int* inputStrides = new int[5];

    int dimensionCount = input.dim();

    for (int i = 0; i < dimensionCount; i++)
    {
        inputDimensions[i] = input.size(i);
        inputStrides[i] = input.stride(i);
    }

    // Pre-calculating exponent factors.
    float spatialExponentFactor = -1.0f / (2 * spatialSigma * spatialSigma);
    float colorExponentFactor = -1.0f / (2 * colorSigma * colorSigma);
    
    // Pre-calculating gaussian kernel.
    int kernelSize = ceil(3 * spatialSigma);
    int kernelHalfSize = 0.5f * kernelSize;
    float* kernel = new float[kernelSize];

    for (int i = 0; i < kernelSize; i++)
    {
        int distance = i - kernelHalfSize;
        kernel[i] = exp(distance * spatialExponentFactor);
    }
    
    // Writing constant memory.
    cudaMemcpyToSymbol(cInputDimensions, inputDimensions, sizeof(int) * 5);
    cudaMemcpyToSymbol(cInputStrides, inputStrides, sizeof(int) * 5);
    cudaMemcpyToSymbol(cKernelSize, &kernelSize, sizeof(int));
    cudaMemcpyToSymbol(cKernel, kernel, sizeof(float) * kernelSize);
    cudaMemcpyToSymbol(cColorExponentFactor, &colorExponentFactor, sizeof(float));

    // Calculate dispatch parameters.
    int batchCount = inputDimensions[0];
    int elementCount = inputDimensions[2] * inputDimensions[3];
    int blockCount = elementCount;
    int blockWidth = 1;

    // Dispatch kernel.
    BilateralFilterCudaKernel<<<dim3(blockCount, batchCount), dim3(blockWidth, 1)>>>(input.data<float>(), output.data<float>());

    return output;
}
