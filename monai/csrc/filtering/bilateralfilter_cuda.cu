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

__constant__ int[5] cInputDimensions;
__constant__ int[5] cInputStrides;

__constant__ int cKernelSize;
__constant__ float[1024] cKernel;

__constant__ float cColorExponentFactor;

__device__ void ReadValue(float* dst, float* src, int offset, int stride, int count)
{
    for(int i = 0; i < count; i++)
    {
        dst[i] = src[offset + i * stride];
    }
}

__device__ void WriteValue(float* dst, float* src, int offset, int stride, int count)
{
    for(int i = 0; i < count; i++)
    {
        dst[offset + i * stride] = src[i];
    }
}

__device__ float DistanceSquared(float* a, float* b, int count)
{
    float result = 0;

    for(int i = 0; i < count; i++)
    {
        float diff = a[i] - b[i];
        result += diff * diff;
    }

    return result;
}

__device__ void SetValue(float* a, float scalar, int count)
{
    for(int i = 0; i < count; i++)
    {
        a[i] = scalar;
    }
}

__device__ void MulValue(float* a, float scalar, int count)
{
    for(int i = 0; i < count; i++)
    {
        a[i] *= scalar;
    }
}

__device__ void AddValue(float* a, float* b, int count)
{
    for(int i = 0; i < count; i++)
    {
        a[i] += b[i];
    }
}

__global__ void BilateralFilterCudaKernel(float* input, float* output)
{
    int batchCount = cInputDimensions[0];
    int channelCount = cInputDimensions[1];
    int width = cInputDimensions[2];
    int height = cInputDimensions[3];
    int depth = cInputDimensions[4];

    int batchStride = cInputStrides[0];
    int channelStride = cInputStrides[1];
    int widthStride = cInputStrides[2];
    int heightStride = cInputStrides[3];
    int depthStride = cInputStrides[4];

    bool hasHeight = height > 1;
    bool hasDepth = depth > 1;

    int kernelSizeX = cKernelSize;
    int kernelSizeY = hasHeight ? kernelSizeX : 1;
    int kernelSizeZ = hasDepth ? kernelSizeX : 1;

    int kernelHalfSize = (int)(kernelSize * 0.5f);

    int batchOffset = blockIdx.y * batchStride;

    int homeOffset = blockIdx.x * blockDim.x + threadIdx.x;
    int homeX = homeOffset / widthStride;
    int homeY = (homeOffset - homeX * widthStride) / heightStride;
    int homeZ = (homeOffset - homeX * widthStride - homeY * heightStride) / depthStride;

    float* homeValue = new float[channelCount];
    float* neighbourValue = new float[channelCount];

    float* valueSum = new float[channelCount];
    SetValue(valueSum, 0, channelCount);
    float weightSum = 0;

    ReadValue(homeValue, input, batchOffset + homeOffset, channelStride, channelCount);

    for(int kernelX = 0; kernelX < kernelSizeX; kernelX++)
    {
        int kernelOffsetX = kernelX - kernelHalfSize;
        float xGaussian = cKernel[kernelX];

        for(int kernelY = 0; kernelY < kernelSizeY; kernelY++)
        {
            float yGaussian = hasHeight ? cKernel[kernelY] : 1;

            for(int kernelZ = 0; kernelZ < kernelSizeZ; kernelZ++)
            {
                float zGaussian = hasDepth ? cKernel[kernelZ] : 1;

                ReadValue(neighbourValue, input, batchOffset + homeOffset + kernelOffset, channelStride, channelCount);

                float distanceSquared = DistanceSquared(homeValue, neighbourValue, channelCount);

                float spatialWeight = xGaussian * yGaussian * zGaussian;
                float colorWeight = exp(cColorExponentFactor * distanceSquared);
                float totalWeight = spatialWeight * colorWeight;
                
                MulValue(neighbourValue, totalWeight, channelCount);
                AddValue(valueSum, neighbourValue, channelCount);
                weightSum += totalWeight;
            }
        }
    }

    MulValue(valueSum, 1.0f/weightSum, channelCount);
    WriteValue(output, valueSum, batchOffset + homeOffset, channelStride, channelCount);
}

torch::Tensor BilateralFilterCuda(torch::Tensor input, float spatial_sigma, float color_sigma)
{
    // Preparing output tensor.
    torch::Tensor output = torch::zeros_like(input);

    // Gathering and input description.
    int* inputDimensions = new int[5];
    int* inputStrides = new int[5];

    int dimensionCount = input.dim();

    for (int i = 0; i < 5; i++)
    {
        inputDimensions[i] = i < dimensionCount ? input.size(i) : 1;
        inputStrides[i] = i < dimensionCount ? input.stride(i) : 0;
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
        kernel[i] = exp(distance * spatialExpConstant);
    }
    
    // Writing constant memory.
    cudaMemcpyToSymbol(cInputDimensions, inputDimensions, sizeof(int) * 5);
    cudaMemcpyToSymbol(cInputStrides, inputStrides, sizeof(int) * 5);
    cudaMemcpyToSymbol(cKernelSize, kernelSize, sizeof(int));
    cudaMemcpyToSymbol(cKernel, kernel, sizeof(float) * kernelSize);
    cudaMemcpyToSymbol(cColorExponentFactor, colorExponentFactor, sizeof(float));

    // Calculate dispatch parameters.
    int elementCount = inputDimensions[2] * inputDimensions[3] * inputDimensions[4];
    int blockCount = ceil(elementCount / kernelSize);

    const dim3 blockCounts(blockCount, batchCount);
    const dim3 blockSizes(kernelSize, 1);

    // Dispatch kernel.
    BilateralFilterCudaKernel<<<blockCounts, blockSize>>>(input.data<float>(), output.data<float>());
}
