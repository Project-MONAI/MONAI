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

#pragma once

#include <torch/extension.h>
#include <vector>

#include "utils/common_utils.h"

// Cuda kernels
std::vector<torch::Tensor> insert_cuda(torch::Tensor table, torch::Tensor n_entries, torch::Tensor keys, torch::Tensor hash);
torch::Tensor get_rank_cuda(torch::Tensor table, torch::Tensor keys, torch::Tensor hash);
torch::Tensor get_values_cuda(torch::Tensor table, int n_values);

std::vector<torch::Tensor> hashtable_insert(torch::Tensor table, torch::Tensor n_entries, torch::Tensor keys, torch::Tensor hash){
  CHECK_CONTIGUOUS_CUDA(table);
  CHECK_CONTIGUOUS_CUDA(n_entries);
  CHECK_CONTIGUOUS_CUDA(keys);
  CHECK_CONTIGUOUS_CUDA(hash);
  return insert_cuda(table, n_entries, keys, hash);        
}

torch::Tensor hashtable_get_rank(torch::Tensor table, torch::Tensor keys, torch::Tensor hash){
  CHECK_CONTIGUOUS_CUDA(table);
  CHECK_CONTIGUOUS_CUDA(keys);
  CHECK_CONTIGUOUS_CUDA(hash);
  return get_rank_cuda(table, keys, hash);
}

torch::Tensor hashtable_get_values(torch::Tensor table, int n_values){
  CHECK_CONTIGUOUS_CUDA(table);
  return get_values_cuda(table, n_values);
}