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
#include <math.h>

torch::Tensor BilateralFilterCpu(torch::Tensor input, float spatialSigma, float colorSigma)
{
    // Prepare output tensor
    torch::Tensor output = torch::zeros_like(input);

    // Tensor descriptors.
    int batchCount = input.size(0);
    int channelCount = input.size(1);
    int width = input.size(2);
    int height = input.size(3);

    int batchStride = input.stride(0);
    int channelStride = input.stride(1);
    int widthStride = input.stride(2);
    int heightStride = input.stride(3);

    // Raw tensor data pointers. 
    float* inputData = input.data_ptr<float>();
    float* outputData = output.data_ptr<float>();

    // Pre-calculate common values
    int windowSize = ceil(3 * spatialSigma);
    int halfWindowSize = 0.5f * windowSize;
    float spatialExpConstant = -1.0f / (2 * spatialSigma * spatialSigma);
    float colorExpConstant = -1.0f / (2 * colorSigma * colorSigma);

    // Pre-calculate gaussian kernel in 1D.
    float* gaussianKernel = new float[windowSize];

    for (int i = 0; i < windowSize; i++)
    {
        int distance = i - halfWindowSize;
        gaussianKernel[i] = exp(distance * spatialExpConstant);
    }

    // Allocating arrays for color reads.
    float* homeColor = new float[channelCount];
    float* neighbourColor = new float[channelCount];

    // Kernel aggregates used to calculate
    // the output value.
    float* valueSum = new float[channelCount];
    float weightSum = 0;

    // Looping over the input data, calculating 
    // offsets into our tensor data.
    for (int b = 0; b < batchCount; b++)
    {
        int batchOffset = b * batchStride;

        for (int x = 0; x < width; x++)
        {
            int xOffset = x * widthStride;

            for (int y = 0; y < height; y++)
            {
                int yOffset = y * heightStride;

                // Reading the home "color" value.
                homeColor[0] = inputData[batchOffset + 0 * channelStride + xOffset + yOffset]; 
                homeColor[1] = inputData[batchOffset + 1 * channelStride + xOffset + yOffset]; 
                homeColor[2] = inputData[batchOffset + 2 * channelStride + xOffset + yOffset];

                // Zero kernel aggregates.
                valueSum[0] = 0;
                valueSum[1] = 0; 
                valueSum[2] = 0;

                weightSum = 0;

                // Looping over the kernel clamping pixel 
                // reads to the edge of the input.
                for (int i = 0; i < windowSize; i++)
                {
                    int xKernelOffset = i - halfWindowSize;
                    int neighbourX = std::min(width - 1, std::max(0, x + xKernelOffset));
                    int neighbourXStride = neighbourX * widthStride;

                    for (int j = 0; j < windowSize; j++)
                    {
                        int yKernelOffset = j - halfWindowSize;
                        int neighbourY = std::min(height - 1, std::max(0, y + yKernelOffset));
                        int neighbourYStride = neighbourY * heightStride;

                        // Read the neighbour "color" value.
                        neighbourColor[0] = inputData[batchStride + 0 * channelStride + neighbourXStride + neighbourYStride]; 
                        neighbourColor[1] = inputData[batchStride + 1 * channelStride + neighbourXStride + neighbourYStride]; 
                        neighbourColor[2] = inputData[batchStride + 2 * channelStride + neighbourXStride + neighbourYStride];

                        // Euclidean color distance.
                        float colorDistanceSquared = 0;
                        colorDistanceSquared += (homeColor[0] - neighbourColor[0]) * (homeColor[0] - neighbourColor[0]);
                        colorDistanceSquared += (homeColor[1] - neighbourColor[1]) * (homeColor[1] - neighbourColor[1]);
                        colorDistanceSquared += (homeColor[2] - neighbourColor[2]) * (homeColor[2] - neighbourColor[2]);

                        // Calculating and combining the spatial 
                        // and color weights.
                        float spatialWeight = gaussianKernel[i] * gaussianKernel[j];
                        float colorWeight = exp(colorDistanceSquared * colorExpConstant);
                        float totalWeight = spatialWeight * colorWeight;

                        // Aggregating values.
                        valueSum[0] += neighbourColor[0] * totalWeight;
                        valueSum[1] += neighbourColor[1] * totalWeight;
                        valueSum[2] += neighbourColor[2] * totalWeight;

                        weightSum += totalWeight;
                    }
                }
                
                outputData[batchOffset + 0 * channelStride + xOffset + yOffset] = valueSum[0] / weightSum; 
                outputData[batchOffset + 1 * channelStride + xOffset + yOffset] = valueSum[1] / weightSum; 
                outputData[batchOffset + 2 * channelStride + xOffset + yOffset] = valueSum[2] / weightSum;
            }
        }
    }

    return output;
}
