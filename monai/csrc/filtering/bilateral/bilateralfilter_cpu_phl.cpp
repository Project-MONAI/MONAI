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

#include "utils/tensor_description.h"
#include "filtering/permutohedral/permutohedral.h"


torch::Tensor BilateralFilterPHLCpu(torch::Tensor inputTensor, float spatialSigma, float colorSigma)
{
    // Preparing output tensor.
    torch::Tensor outputTensor = torch::zeros_like(inputTensor);

    // Getting tensor description.
    TensorDescription desc = TensorDescription(inputTensor);
    
    int featureChannels = desc.channelCount + desc.dimensions;

    // Preparing memory
    float* inputTensorData = inputTensor.data_ptr<float>();
    float* outputTensorData = outputTensor.data_ptr<float>();
    float* data = new float[desc.channelStride * desc.channelCount];
    float* features = new float[desc.channelStride * featureChannels];

    // Precalculating inverse sigmas    
    float invSpatialSigma= 1.0f/spatialSigma;
    float invColorSigma = 1.0f/colorSigma;

    // Looping over batches
    for (int b = 0; b < desc.batchCount; b++)
    {
        int batchOffset = b * desc.batchStride;

        // Creating features (also permuting input data to be channel last. Permutohedral 
        // implementation should be changed to channel first to avoid this)
        for (int i = 0; i < desc.channelStride; i++) 
        {
            // Color features (and permutation)
            for (int c = 0; c < desc.channelCount; c++)
            {
                features[i * featureChannels + c] = invColorSigma * inputTensorData[batchOffset + i + c * desc.channelStride];
                data[i * desc.channelCount + c] = inputTensorData[batchOffset + i + c * desc.channelStride];
            }

            // Spatial features
            int offsetRemanider = i;

            for (int d = 0; d < desc.dimensions; d++)
            {
                int coord = offsetRemanider / desc.strides[d];
                offsetRemanider -= coord * desc.strides[d];

                features[i * featureChannels + desc.channelCount + d] = invSpatialSigma * coord;
            }
        }

        // Filtering data with respect to the features.
        float* output = PermutohedralCPU(data, features, desc.channelCount, featureChannels, desc.channelStride);

        // Writing output tensor.
        for (int i = 0; i < desc.channelStride; i++)
        {
            for (int c = 0; c < desc.channelCount; c++)
            {
                outputTensorData[batchOffset + i + c * desc.channelStride] = output[i * desc.channelCount + c];
            }
        }
    }

    delete[] data;
    delete[] features;

    return outputTensor;
}
