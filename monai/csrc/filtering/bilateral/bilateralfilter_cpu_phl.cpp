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

#include <torch/extension.h>

#include "filtering/permutohedral/permutohedral.h"
#include "utils/tensor_description.h"

template <typename scalar_t>
void BilateralFilterPHLCpu(
    torch::Tensor inputTensor,
    torch::Tensor outputTensor,
    float spatialSigma,
    float colorSigma) {
  // Getting tensor description.
  TensorDescription desc = TensorDescription(inputTensor);

  int featureChannels = desc.channelCount + desc.dimensions;

  // Preparing memory
  scalar_t* inputTensorData = inputTensor.data_ptr<scalar_t>();
  scalar_t* outputTensorData = outputTensor.data_ptr<scalar_t>();
  scalar_t* data = new scalar_t[desc.channelStride * desc.channelCount];
  scalar_t* features = new scalar_t[desc.channelStride * featureChannels];

  // Precalculating inverse sigmas
  float invSpatialSigma = 1.0f / spatialSigma;
  float invColorSigma = 1.0f / colorSigma;

  // Looping over batches
  for (int b = 0; b < desc.batchCount; b++) {
    int batchOffset = b * desc.batchStride;

    // Creating features (also permuting input data to be channel last. Permutohedral
    // implementation should be changed to channel first to avoid this)
    for (int i = 0; i < desc.channelStride; i++) {
      // Color features (and permutation)
      for (int c = 0; c < desc.channelCount; c++) {
        features[i * featureChannels + c] = invColorSigma * inputTensorData[batchOffset + i + c * desc.channelStride];
        data[i * desc.channelCount + c] = inputTensorData[batchOffset + i + c * desc.channelStride];
      }

      // Spatial features
      int offsetRemainder = i;

      for (int d = 0; d < desc.dimensions; d++) {
        int coord = offsetRemainder / desc.strides[d];
        offsetRemainder -= coord * desc.strides[d];

        features[i * featureChannels + desc.channelCount + d] = (scalar_t)invSpatialSigma * coord;
      }
    }

    // Filtering data with respect to the features.
    PermutohedralCPU<scalar_t>(data, features, desc.channelCount, featureChannels, desc.channelStride);

    // Writing output tensor.
    for (int i = 0; i < desc.channelStride; i++) {
      for (int c = 0; c < desc.channelCount; c++) {
        outputTensorData[batchOffset + i + c * desc.channelStride] = data[i * desc.channelCount + c];
      }
    }
  }

  delete[] data;
  delete[] features;
}

// Function to choose template implementation based on dynamic, channels and dimensions
torch::Tensor BilateralFilterPHLCpu(torch::Tensor inputTensor, float spatialSigma, float colorSigma) {
  torch::Tensor outputTensor = torch::zeros_like(inputTensor);

  AT_DISPATCH_FLOATING_TYPES(inputTensor.scalar_type(), "BilateralFilterPhlCpu", ([&] {
                               BilateralFilterPHLCpu<scalar_t>(inputTensor, outputTensor, spatialSigma, colorSigma);
                             }));

  return outputTensor;
}
