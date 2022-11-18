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

#pragma once

#include <torch/extension.h>
#include <vector>
#include "utils/common_utils.h"

#ifdef WITH_CUDA
std::vector<torch::Tensor> lltm_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell);

std::vector<torch::Tensor> lltm_cuda_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights);
#endif

std::vector<torch::Tensor> lltm_cpu_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell);

std::vector<torch::Tensor> lltm_cpu_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights);

std::vector<torch::Tensor> lltm_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  if (input.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CONTIGUOUS_CUDA(input);
    CHECK_CONTIGUOUS_CUDA(weights);
    CHECK_CONTIGUOUS_CUDA(bias);
    CHECK_CONTIGUOUS_CUDA(old_h);
    CHECK_CONTIGUOUS_CUDA(old_cell);

    return lltm_cuda_forward(input, weights, bias, old_h, old_cell);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return lltm_cpu_forward(input, weights, bias, old_h, old_cell);
}

std::vector<torch::Tensor> lltm_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights) {
  if (X.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CONTIGUOUS_CUDA(grad_h);
    CHECK_CONTIGUOUS_CUDA(grad_cell);
    CHECK_CONTIGUOUS_CUDA(new_cell);
    CHECK_CONTIGUOUS_CUDA(input_gate);
    CHECK_CONTIGUOUS_CUDA(output_gate);
    CHECK_CONTIGUOUS_CUDA(candidate_cell);
    CHECK_CONTIGUOUS_CUDA(X);
    CHECK_CONTIGUOUS_CUDA(gate_weights);
    CHECK_CONTIGUOUS_CUDA(weights);

    return lltm_cuda_backward(
        grad_h, grad_cell, new_cell, input_gate, output_gate, candidate_cell, X, gate_weights, weights);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return lltm_cpu_backward(
      grad_h, grad_cell, new_cell, input_gate, output_gate, candidate_cell, X, gate_weights, weights);
}
