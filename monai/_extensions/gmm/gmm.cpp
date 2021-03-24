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

#include <torch/extension.h>

#include "gmm.h"

torch::Tensor GMM(torch::Tensor input_tensor, torch::Tensor label_tensor)
{
    c10::DeviceType device_type = input_tensor.device().type();

    int dim = input_tensor.dim();
    int batch_count = input_tensor.size(0);
    int element_count = input_tensor.stride(1);

    long int* output_size = new long int[dim];
    memcpy(output_size, input_tensor.sizes().data(), dim * sizeof(long int));

    output_size[1] = MIXTURE_COUNT;

    torch::Tensor output_tensor = torch::empty(c10::IntArrayRef(output_size, dim), torch::dtype(torch::kFloat32).device(device_type));
    
    delete output_size;

    float* input = input_tensor.data_ptr<float>();
    int* labels = label_tensor.data_ptr<int>();
    float* output = output_tensor.data_ptr<float>();

    if(device_type == torch::kCUDA)
    {
        GMM_Cuda(input, labels, output, batch_count, element_count);
    }
    else
    {
        GMM_Cpu(input, labels, output, batch_count, element_count);
    }

    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("gmm", torch::wrap_pybind_function(GMM), "gmm");
}