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
    int dimensions = input_tensor.dim() - 2;
    int* sizes = new int[dimensions];
    int* strides = new int[dimensions];
    int elementCount = 1;

    for (int i = 0; i < dimensions; i++)
    {
        sizes[i] = input_tensor.size(i+2);
        strides[i] = input_tensor.stride(i+2);
        elementCount *= sizes[i];
    }

    int dataChannels = input_tensor.size(1);
    int featureChannels = dataChannels + dimensions;
    int channelStride = input_tensor.stride(1);

    // Preparing memory
    float* input_tensor_ptr = input_tensor.data_ptr<float>();
    float* data = new float[elementCount * dataChannels];
    float* features = new float[elementCount * featureChannels];

    // Creating features
    float invSpatialStdev = 1.0f/spatial_sigma;
    float invColorStdev = 1.0f/color_sigma;

    for (int i = 0; i < elementCount; i++) 
    {
        for (int c = 0; c < dataChannels; c++)
        {
            features[i*featureChannels + c] = invColorStdev * input_tensor_ptr[i + c * channelStride];
            data[i*dataChannels + c] = input_tensor_ptr[i + c * channelStride];
        }

        int remainder = i;

        for (int d = 0; d < dimensions; d++)
        {
            int coord = remainder / strides[d];
            remainder -= coord * strides[d];

            features[i*featureChannels + dataChannels + d] = invSpatialStdev * coord;
        }
    }

    // Filtering data with respect to the features
    float* output = PermutohedralLattice::filter(data, features, elementCount, dataChannels, featureChannels);

    // Writing output tensor
    torch::Tensor output_tensor = torch::zeros_like(input_tensor);
    float* output_tensor_ptr = output_tensor.data_ptr<float>();

    for (int i = 0; i < elementCount; i++)
    {
        for (int c = 0; c < dataChannels; c++)
        {
            output_tensor_ptr[i + c * channelStride] = output[i * dataChannels + c];
        }
    }

    return output_tensor;
}
