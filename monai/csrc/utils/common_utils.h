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

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor.")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous.")
#define CHECK_CONTIGUOUS_CUDA(x) \
  CHECK_CUDA(x);                 \
  CHECK_CONTIGUOUS(x)
#define CHECK_DEFINED(value) \
  TORCH_CHECK(value.defined(), "(): expected " #value " not be undefined, but it is ", value);
#define CHECK_STRIDED(value)                                              \
  TORCH_CHECK(                                                            \
      value.layout() == at::kStrided,                                     \
      "(): expected " #value "to have torch.strided layout, but it has ", \
      value.layout());
#define CHECK_SPATIAL_1D_2D_OR_3D(value)                                \
  TORCH_CHECK(                                                          \
      (value.dim() == 3 || value.dim() == 4 || value.dim() == 5),       \
      "(): expected 3D, 4D or 5D " #value " but got input with sizes ", \
      value.sizes());
#define CHECK_GRID_COMPONENT(value, dim)           \
  TORCH_CHECK(                                     \
      value.size(-1) == dim - 2,                   \
      "(): expected " #value " to have size ",     \
      dim - 2,                                     \
      " in last "                                  \
      "dimension, but got " #value " with sizes ", \
      value.sizes());
#define CHECK_SAME_DEVICE(value1, value2)     \
  TORCH_CHECK(                                \
      value1.device() == value2.device(),     \
      "(): expected " #value1 " and " #value2 \
      " to be on same device, "               \
      "but " #value1 " is on ",               \
      value1.device(),                        \
      " and " #value2 " is on ",              \
      value2.device());
#define CHECK_SAME_DTYPE(value1, value2)      \
  TORCH_CHECK(                                \
      value1.dtype() == value2.dtype(),       \
      "(): expected " #value1 " and " #value2 \
      " to have the same dtype, "             \
      "but " #value1 " has ",                 \
      value1.dtype(),                         \
      " and " #value2 " has ",                \
      value2.dtype());
#define CHECK_SPATIAL_NOT_EMPTY(value)                                                        \
  for (int64_t i = 2; i < value.dim(); i++) {                                                 \
    TORCH_CHECK(                                                                              \
        value.size(i) > 0,                                                                    \
        "(): expected " #value " to have non-empty spatial dimensions, but input has sizes ", \
        value.sizes(),                                                                        \
        " with dimension ",                                                                   \
        i,                                                                                    \
        " being empty");                                                                      \
  }
#define CHECK_GRID_TARGET_COMPAT(value1, value2)                                                          \
  TORCH_CHECK(                                                                                            \
      value2.size(0) == value1.size(0) && (value2.dim() <= 2 || value2.size(2) == value1.size(1)) &&      \
          (value2.dim() <= 3 || value2.size(3) == value1.size(2)) &&                                      \
          (value2.dim() <= 4 || value2.size(4) == value1.size(3)),                                        \
      "(): expected " #value2 " and " #value1                                                             \
      " to have same batch, width, height and (optionally) depth sizes, but got " #value2 " with sizes ", \
      value2.sizes(),                                                                                     \
      " and " #value1 " with sizes ",                                                                     \
      value1.sizes());
#define CHECK_SPATIAL_LENGTH(value, dim) \
  TORCH_CHECK(((int64_t)(value.size()) == dim - 2), "(): expected ", dim, #value " elements but got ", value.size());
#define CHECK_VEC_NOT_EMPTY(value) TORCH_CHECK(!value.empty(), "(): expected nonempty value " #value);
