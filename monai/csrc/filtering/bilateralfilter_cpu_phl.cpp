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
#include "permutohedral.h"

torch::Tensor BilateralFilterPHLCpu(torch::Tensor input_tensor, float spatial_sigma, float color_sigma)
{
    // Getting tensor descriptors
    int width = input_tensor.size(2);
    int height = input_tensor.size(3);
    int elementCount = width * height;
    int dataChannels = input_tensor.size(1);
    int featureChannels = dataChannels + 2;
    int channelStride = input_tensor.stride(1);

    // Preparing memory
    float* input_tensor_ptr = input_tensor.data_ptr<float>();
    float* data = new float[elementCount * dataChannels];
    float* features = new float[elementCount * featureChannels];

    // Creating features
    float invSpatialStdev = 1.0f/spatial_sigma;
    float invColorStdev = 1.0f/color_sigma;

    for (int i = 0, x = 0; x < width; x++) 
    {
        for (int y = 0; y < height; y++, i++)
        {
            features[i*featureChannels + 0] = invSpatialStdev * x;
            features[i*featureChannels + 1] = invSpatialStdev * y;
            features[i*featureChannels + 2] = invColorStdev * input_tensor_ptr[i + 0 * channelStride];
            features[i*featureChannels + 3] = invColorStdev * input_tensor_ptr[i + 1 * channelStride];
            features[i*featureChannels + 4] = invColorStdev * input_tensor_ptr[i + 2 * channelStride];

            data[i*dataChannels + 0] = input_tensor_ptr[i + 0 * channelStride];
            data[i*dataChannels + 1] = input_tensor_ptr[i + 1 * channelStride];
            data[i*dataChannels + 2] = input_tensor_ptr[i + 2 * channelStride];
        }
    }

    // Filtering data with respect to the features
    float* output = PermutohedralLattice::filter(data, features, elementCount, dataChannels, featureChannels);

    // Writing output tensor
    torch::Tensor output_tensor = torch::zeros_like(input_tensor);
    float* output_tensor_ptr = output_tensor.data_ptr<float>();

    for (int i = 0; i < elementCount; i++)
    {
        output_tensor_ptr[i + 0 * channelStride] = output[i * dataChannels + 0];
        output_tensor_ptr[i + 1 * channelStride] = output[i * dataChannels + 1];
        output_tensor_ptr[i + 2 * channelStride] = output[i * dataChannels + 2];
    }

    return output_tensor;
}
