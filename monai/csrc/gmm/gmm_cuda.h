/*
Copyright 2020 - 2021 MONAI Consortium
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

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

void GMM_Cuda(const float* input, const int* labels, float* output, int batch_count, int channel_count, int width, int height, int mixture_count, int gaussians_per_mixture);

torch::Tensor GMM_Cuda(torch::Tensor input_tensor, torch::Tensor label_tensor, int mixture_count, int gaussians_per_mixture)
{
    int dim = input_tensor.dim();

    int* input_size = new int[dim];
    long int* output_size = new long int[dim];

    for (int i = 0; i < dim; i++)
    {
        input_size[i] = input_tensor.size(i);
        output_size[i] = input_tensor.size(i);
    }

    output_size[1] = mixture_count;

    torch::Tensor output_tensor = torch::empty(c10::IntArrayRef(output_size, dim), torch::dtype(torch::kFloat32).device(torch::kCUDA));

    float* input = input_tensor.data_ptr<float>();
    int* labels = label_tensor.data_ptr<int>();
    float* output = output_tensor.data_ptr<float>();

    GMM_Cuda(input, labels, output, input_size[0], input_size[1], input_size[2], input_size[3], mixture_count, gaussians_per_mixture);

    delete input_size;
    delete output_size;

    return output_tensor;
}