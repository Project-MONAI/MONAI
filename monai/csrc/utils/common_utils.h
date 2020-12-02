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
#define CHECK_SPATIAL_2D_OR_3D(value)                               \
  TORCH_CHECK(                                                      \
      (value.dim() == 4 || value.dim() == 5),                       \
      "(): expected 4D or 5D " #value " but got input with sizes ", \
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
      "(): expected " #value2 " and " #value2 \
      " to be on same device, "               \
      "but " #value2 " is on ",               \
      value1.device(),                        \
      " and " #value2 " is on ",              \
      value2.device());
#define CHECK_SAME_DTYPE(value1, value2)      \
  TORCH_CHECK(                                \
      value1.dtype() == value2.dtype(),       \
      "(): expected " #value2 " and " #value2 \
      " to have the same dtype, "             \
      "but " #value2 " has ",                 \
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
#define CHECK_GRID_TARGET_COMPAT(value1, value2)                                                                  \
  TORCH_CHECK(                                                                                                    \
      value2.size(0) == value1.size(0) && value2.size(2) == value1.size(1) && value2.size(3) == value1.size(2) && \
          (value2.dim() == 4 || value2.size(4) == value1.size(3)),                                                \
      "(): expected " #value2 " and " #value1                                                                     \
      " to have same batch, width, height and (optionally) depth sizes, but got " #value2 " with sizes ",         \
      value2.sizes(),                                                                                             \
      " and " #value1 " with sizes ",                                                                             \
      value1.sizes());
#define CHECK_SPATIAL_LENGTH(value, dim) \
  TORCH_CHECK(((int64_t)(value.size()) == dim - 2), "(): expected ", dim, #value " elements but got ", value.size());
#define CHECK_VEC_NOT_EMPTY(value) TORCH_CHECK(!value.empty(), "(): expected nonempty value " #value);


// Dynamic template specialisation to ease template use based on variable parameters.

#define __DIMENSION_SWITCH__(C, D, FUNC, ...)	                \
{													                    	              \
	switch (D)										              	              \
	{												                    	              \
		case(1): FUNC<C, 1>(__VA_ARGS__); break;		              \
		case(2): FUNC<C, 2>(__VA_ARGS__); break;		              \
		case(3): FUNC<C, 3>(__VA_ARGS__); break;		              \
		default: throw std::invalid_argument(                     \
      "Tensor has invalid spatial dimensions, should be 1-3"  \
      );		                                                  \
	}													                                  \
}													                     	

#define __CHANNEL_SWITCH__(C, D, FUNC, ...)								          \
{																	                                  \
	switch (C)													                      	      \
	{																                                  \
		case( 1): __DIMENSION_SWITCH__( 1, D, FUNC, __VA_ARGS__) break;	\
		case( 2): __DIMENSION_SWITCH__( 2, D, FUNC, __VA_ARGS__) break;	\
		case( 3): __DIMENSION_SWITCH__( 3, D, FUNC, __VA_ARGS__) break;	\
		case( 4): __DIMENSION_SWITCH__( 4, D, FUNC, __VA_ARGS__) break;	\
		case( 5): __DIMENSION_SWITCH__( 5, D, FUNC, __VA_ARGS__) break;	\
		case( 6): __DIMENSION_SWITCH__( 6, D, FUNC, __VA_ARGS__) break;	\
		case( 7): __DIMENSION_SWITCH__( 7, D, FUNC, __VA_ARGS__) break;	\
		case( 8): __DIMENSION_SWITCH__( 8, D, FUNC, __VA_ARGS__) break;	\
		case( 9): __DIMENSION_SWITCH__( 9, D, FUNC, __VA_ARGS__) break;	\
		case(10): __DIMENSION_SWITCH__(10, D, FUNC, __VA_ARGS__) break;	\
		case(11): __DIMENSION_SWITCH__(11, D, FUNC, __VA_ARGS__) break;	\
		case(12): __DIMENSION_SWITCH__(12, D, FUNC, __VA_ARGS__) break;	\
		case(13): __DIMENSION_SWITCH__(13, D, FUNC, __VA_ARGS__) break;	\
		case(14): __DIMENSION_SWITCH__(14, D, FUNC, __VA_ARGS__) break;	\
		case(15): __DIMENSION_SWITCH__(15, D, FUNC, __VA_ARGS__) break;	\
		case(16): __DIMENSION_SWITCH__(16, D, FUNC, __VA_ARGS__) break;	\
		case(17): __DIMENSION_SWITCH__(17, D, FUNC, __VA_ARGS__) break;	\
		case(18): __DIMENSION_SWITCH__(18, D, FUNC, __VA_ARGS__) break;	\
		case(19): __DIMENSION_SWITCH__(19, D, FUNC, __VA_ARGS__) break;	\
		case(20): __DIMENSION_SWITCH__(20, D, FUNC, __VA_ARGS__) break;	\
		case(21): __DIMENSION_SWITCH__(21, D, FUNC, __VA_ARGS__) break;	\
		case(22): __DIMENSION_SWITCH__(22, D, FUNC, __VA_ARGS__) break;	\
		case(23): __DIMENSION_SWITCH__(23, D, FUNC, __VA_ARGS__) break;	\
		case(24): __DIMENSION_SWITCH__(24, D, FUNC, __VA_ARGS__) break;	\
		case(25): __DIMENSION_SWITCH__(25, D, FUNC, __VA_ARGS__) break;	\
		case(26): __DIMENSION_SWITCH__(26, D, FUNC, __VA_ARGS__) break;	\
		case(27): __DIMENSION_SWITCH__(27, D, FUNC, __VA_ARGS__) break;	\
		case(28): __DIMENSION_SWITCH__(28, D, FUNC, __VA_ARGS__) break;	\
		case(29): __DIMENSION_SWITCH__(29, D, FUNC, __VA_ARGS__) break;	\
		case(30): __DIMENSION_SWITCH__(30, D, FUNC, __VA_ARGS__) break;	\
		case(31): __DIMENSION_SWITCH__(31, D, FUNC, __VA_ARGS__) break;	\
		case(32): __DIMENSION_SWITCH__(32, D, FUNC, __VA_ARGS__) break;	\
		default: throw std::invalid_argument(                           \
      "Tensor has invalid channel count, should be 1-32"            \
      );		                                                        \
	}																                                  \
}																	                            

#define SPECIALISE_C_AND_D(Channels, Dimensions, TemplateFunction, ...)	    \
{													                                                  \
	__CHANNEL_SWITCH__(Channels, Dimensions, TemplateFunction, __VA_ARGS__);	\
}													                                                            	