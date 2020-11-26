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

#include "Image.h"
#include "permutohedral.h"

torch::Tensor BilateralFilterPHLCpu(torch::Tensor input_tensor, float spatial_sigma, float color_sigma)
{
    torch::Tensor temp_tensor = input_tensor.permute(c10::IntArrayRef(new int64_t[4]{0, 2, 3, 1}, 4));
    Image input = Image(1, temp_tensor.size(1), temp_tensor.size(2), 3, temp_tensor.data_ptr<float>());

    float invSpatialStdev = 1.0f/spatial_sigma;
    float invColorStdev = 1.0f/color_sigma;

    // Construct the position vectors out of x, y, r, g, and b.
    Image positions(1, input.width, input.height, 5);

    for (int y = 0; y < input.height; y++) {
	for (int x = 0; x < input.width; x++) {
	    positions(x, y)[0] = invSpatialStdev * x;
	    positions(x, y)[1] = invSpatialStdev * y;
	    positions(x, y)[2] = invColorStdev * input(x, y)[0];
	    positions(x, y)[3] = invColorStdev * input(x, y)[1];
	    positions(x, y)[4] = invColorStdev * input(x, y)[2];
	}
    }

    // Filter the input with respect to the position vectors. (see permutohedral.h)
    Image output = PermutohedralLattice::filter(input, positions);

    torch::Tensor output_tensor = torch::zeros_like(temp_tensor);
    memcpy(output_tensor.data_ptr<float>(), output.data, output_tensor.size(1) * output_tensor.size(2) * 3 * sizeof(float));
    
    return output_tensor.permute(c10::IntArrayRef(new int64_t[4]{0, 3, 1, 2}, 4));
}
