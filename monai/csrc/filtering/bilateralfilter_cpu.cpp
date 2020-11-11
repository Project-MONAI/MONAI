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

torch::Tensor CalculateGaussianKernel(int windowSize, float sigma)
{
    torch::Tensor gaussianKernel = torch::zeros(windowSize);

    float halfWindowSize = 0.5f * windowSize;
    float expConstant = -1.0f / (2 * sigma * sigma);

    for (int i = 0; i < windowSize; i++)
    {
        int distance = i - halfWindowSize;
        gaussianKernel[i] = exp(distance * expConstant);
    }

    return gaussianKernel;
}

torch::Tensor BilateralFilterCpu(torch::Tensor input, float spatialSigma, float colorSigma)
{
    int batchCount = input.size(0);
    int channelCount = input.size(1);
    int width = input.size(2);
    int height = input.size(3);

    int windowSize = ceil(3 * spatialSigma);
    int halfWindowSize = 0.5f * windowSize;
    float colorExpConstant = -1.0f / (2 * colorSigma * colorSigma);

    torch::Tensor gaussianKernel = CalculateGaussianKernel(windowSize, spatialSigma);

    torch::Tensor output = torch::zeros_like(input);

    // Looping over the input data 
    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            // Reading the current pixel data. Slice
            // is equivlent to a : in python
            torch::Tensor currentPixel = input.index({0, torch::indexing::Slice(), x, y});

            // Aggregates for this kernel used to 
            // calculate the output value at x, y
            torch::Tensor valueSum = torch::zeros({channelCount});
            torch::Tensor weightSum = torch::zeros({1});

            // Looping over the kernel clamping pixel 
            // reads to the edge of the input
            for (int i = 0; i < windowSize; i++)
            {
                int xOffset = i - halfWindowSize;
                int neighbourX = std::min(width - 1, std::max(0, x + xOffset));

                for (int j = 0; j < windowSize; j++)
                {
                    int yOffset = j - halfWindowSize;
                    int neighbourY = std::min(height - 1, std::max(0, y + yOffset));

                    // Read the neighbour value and calculating 
                    // the squared euclidian "color" distance
                    torch::Tensor neighbourPixel = input.index({0, torch::indexing::Slice(), neighbourX, neighbourY});
                    torch::Tensor colorDistanceSquared = (currentPixel - neighbourPixel).square().sum();

                    // Calculating and combining the spatial and color weights
                    torch::Tensor spatialWeight = gaussianKernel[i] * gaussianKernel[j];
                    torch::Tensor colorWeight = (colorDistanceSquared * colorExpConstant).exp();
                    torch::Tensor totalWeight = spatialWeight * colorWeight;

                    // Aggregating values
                    weightSum += totalWeight;
                    valueSum += neighbourPixel * totalWeight;
                }
            }

            // Writing output
            output.index_put_({0, torch::indexing::Slice(), x, y}, valueSum / weightSum);
        }
    }

    return output;
}
