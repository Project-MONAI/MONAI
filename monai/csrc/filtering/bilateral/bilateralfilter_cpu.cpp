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

#include <math.h>
#include <torch/extension.h>

#include "utils/tensor_description.h"

struct Indexer {
 public:
  Indexer(int dimensions, int* sizes) {
    m_dimensions = dimensions;
    m_sizes = sizes;
    m_index = new int[dimensions]{0};
  }

  bool operator++(int) {
    for (int i = 0; i < m_dimensions; i++) {
      m_index[i] += 1;

      if (m_index[i] < m_sizes[i]) {
        return true;
      } else {
        m_index[i] = 0;
      }
    }

    return false;
  }

  int& operator[](int dimensionIndex) {
    return m_index[dimensionIndex];
  }

 private:
  int m_dimensions;
  int* m_sizes;
  int* m_index;
};

template <typename scalar_t>
void BilateralFilterCpu(torch::Tensor inputTensor, torch::Tensor outputTensor, float spatialSigma, float colorSigma) {
  // Getting tensor description.
  TensorDescription desc = TensorDescription(inputTensor);

  // Raw tensor data pointers.
  scalar_t* inputTensorData = inputTensor.data_ptr<scalar_t>();
  scalar_t* outputTensorData = outputTensor.data_ptr<scalar_t>();

  // Pre-calculate common values
  int windowSize = (int)ceil(5.0f * spatialSigma) | 1; // ORing last bit to ensure odd window size
  int halfWindowSize = floor(0.5f * windowSize);
  scalar_t spatialExpConstant = -1.0f / (2 * spatialSigma * spatialSigma);
  scalar_t colorExpConstant = -1.0f / (2 * colorSigma * colorSigma);

  // Kernel sizes.
  int* kernelSizes = new int[desc.dimensions];

  for (int i = 0; i < desc.dimensions; i++) {
    kernelSizes[i] = windowSize;
  }

  // Pre-calculate gaussian kernel in 1D.
  scalar_t* gaussianKernel = new scalar_t[windowSize];

  for (int i = 0; i < windowSize; i++) {
    int distance = i - halfWindowSize;
    gaussianKernel[i] = exp(distance * distance * spatialExpConstant);
  }

  // Kernel aggregates used to calculate
  // the output value.
  scalar_t* valueSum = new scalar_t[desc.channelCount];
  scalar_t weightSum = 0;

  // Looping over the batches
  for (int b = 0; b < desc.batchCount; b++) {
    int batchOffset = b * desc.batchStride;

    // Looping over all dimensions for the home element
    Indexer homeIndex = Indexer(desc.dimensions, desc.sizes);
    do // while(homeIndex++)
    {
      // Calculating indexing offset for the home element
      int homeOffset = batchOffset;

      for (int i = 0; i < desc.dimensions; i++) {
        homeOffset += homeIndex[i] * desc.strides[i];
      }

      // Zero kernel aggregates.
      for (int i = 0; i < desc.channelCount; i++) {
        valueSum[i] = 0;
      }

      weightSum = 0.0f;

      // Looping over all dimensions for the neighbour element
      Indexer kernelIndex = Indexer(desc.dimensions, kernelSizes);
      do // while(kernelIndex++)
      {
        // Calculating buffer offset for the neighbour element
        // Index is clamped to the border in each dimension.
        int neighbourOffset = batchOffset;

        for (int i = 0; i < desc.dimensions; i++) {
          int neighbourIndex = homeIndex[i] + kernelIndex[i] - halfWindowSize;
          int neighbourIndexClamped = std::min(desc.sizes[i] - 1, std::max(0, neighbourIndex));
          neighbourOffset += neighbourIndexClamped * desc.strides[i];
        }

        // Euclidean color distance.
        scalar_t colorDistanceSquared = 0;

        for (int i = 0; i < desc.channelCount; i++) {
          scalar_t diff = inputTensorData[homeOffset + i * desc.channelStride] -
              inputTensorData[neighbourOffset + i * desc.channelStride];
          colorDistanceSquared += diff * diff;
        }

        // Calculating and combining the spatial
        // and color weights.
        scalar_t spatialWeight = 1;

        for (int i = 0; i < desc.dimensions; i++) {
          spatialWeight *= gaussianKernel[kernelIndex[i]];
        }

        scalar_t colorWeight = exp(colorDistanceSquared * colorExpConstant);
        scalar_t totalWeight = spatialWeight * colorWeight;

        // Aggregating values.
        for (int i = 0; i < desc.channelCount; i++) {
          valueSum[i] += inputTensorData[neighbourOffset + i * desc.channelStride] * totalWeight;
        }

        weightSum += totalWeight;
      } while (kernelIndex++);

      for (int i = 0; i < desc.channelCount; i++) {
        outputTensorData[homeOffset + i * desc.channelStride] = valueSum[i] / weightSum;
      }
    } while (homeIndex++);
  }
}

torch::Tensor BilateralFilterCpu(torch::Tensor inputTensor, float spatialSigma, float colorSigma) {
  // Preparing output tensor.
  torch::Tensor outputTensor = torch::zeros_like(inputTensor);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(inputTensor.scalar_type(), "BilateralFilterCpu", ([&] {
                                        BilateralFilterCpu<scalar_t>(
                                            inputTensor, outputTensor, spatialSigma, colorSigma);
                                      }));

  return outputTensor;
}
