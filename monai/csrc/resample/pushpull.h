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

// adapted from https://github.com/balbasty/nitorch

#include <ATen/ATen.h>
#include <deque>
#include <tuple>
#include <vector>
#include "utils/common_utils.h"
#include "utils/resample_utils.h"

#define MONAI_PUSHPULL_DECLARE(space)                                            \
  namespace space {                                                              \
  template <typename BoundType, typename InterpolationType, typename SourceType> \
  std::deque<at::Tensor> pushpull(                                               \
      const SourceType& source,                                                  \
      const at::Tensor& grid,                                                    \
      BoundType bound,                                                           \
      InterpolationType interpolation,                                           \
      bool extrapolate,                                                          \
      bool do_pull,                                                              \
      bool do_push,                                                              \
      bool do_count,                                                             \
      bool do_grad,                                                              \
      bool do_sgrad);                                                            \
  template <typename BoundType, typename InterpolationType, typename SourceType> \
  std::deque<at::Tensor> pushpull(                                               \
      const SourceType& source,                                                  \
      const at::Tensor& grid,                                                    \
      const at::Tensor& target,                                                  \
      BoundType bound,                                                           \
      InterpolationType interpolation,                                           \
      bool extrapolate,                                                          \
      bool do_pull,                                                              \
      bool do_push,                                                              \
      bool do_count,                                                             \
      bool do_grad,                                                              \
      bool do_sgrad);                                                            \
  }

namespace monai {

MONAI_PUSHPULL_DECLARE(cpu)
MONAI_PUSHPULL_DECLARE(cuda)

// PULL
at::Tensor grid_pull(
    const at::Tensor& input,
    const at::Tensor& grid,
    const std::vector<BoundType>& bound_mode,
    const std::vector<InterpolationType>& interpolation_mode,
    bool extrapolate) {
  CHECK_DEFINED(input)
  CHECK_DEFINED(grid)
  auto input_opt = input.options();
  auto grid_opt = grid.options();
  CHECK_STRIDED(input_opt)
  CHECK_STRIDED(grid_opt)
  CHECK_SAME_DEVICE(input_opt, grid_opt)
  CHECK_SAME_DTYPE(input_opt, grid_opt)
  CHECK_SPATIAL_1D_2D_OR_3D(input)
  CHECK_SPATIAL_1D_2D_OR_3D(grid)
  CHECK_GRID_COMPONENT(grid, grid.dim())
  CHECK_SPATIAL_NOT_EMPTY(input)
  CHECK_SPATIAL_NOT_EMPTY(grid)
  CHECK_VEC_NOT_EMPTY(bound_mode);
  CHECK_VEC_NOT_EMPTY(interpolation_mode);

  if (input.is_cuda())
#ifdef WITH_CUDA
    return cuda::pushpull(
               input,
               grid,
               BoundVectorRef(bound_mode),
               InterpolationVectorRef(interpolation_mode),
               extrapolate,
               true,
               false,
               false,
               false,
               false)
        .front();
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  else
    return cpu::pushpull(
               input,
               grid,
               BoundVectorRef(bound_mode),
               InterpolationVectorRef(interpolation_mode),
               extrapolate,
               true,
               false,
               false,
               false,
               false)
        .front();
}

std::deque<at::Tensor> grid_pull_backward(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& grid,
    const std::vector<BoundType>& bound_mode,
    const std::vector<InterpolationType>& interpolation_mode,
    bool extrapolate) {
  if (input.is_cuda()) {
#ifdef WITH_CUDA
    return cuda::pushpull(
        input,
        grid,
        grad,
        BoundVectorRef(bound_mode),
        InterpolationVectorRef(interpolation_mode),
        extrapolate,
        false,
        input.requires_grad(),
        false,
        grid.requires_grad(),
        false);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  } else {
    return cpu::pushpull(
        input,
        grid,
        grad,
        BoundVectorRef(bound_mode),
        InterpolationVectorRef(interpolation_mode),
        extrapolate,
        false,
        input.requires_grad(),
        false,
        grid.requires_grad(),
        false);
  }
}

// PUSH
at::Tensor grid_push(
    const at::Tensor& input,
    const at::Tensor& grid,
    c10::IntArrayRef source_size,
    const std::vector<BoundType>& bound_mode,
    const std::vector<InterpolationType>& interpolation_mode,
    bool extrapolate) {
  CHECK_DEFINED(input)
  CHECK_DEFINED(grid)
  auto input_opt = input.options();
  auto grid_opt = grid.options();
  CHECK_STRIDED(input_opt)
  CHECK_STRIDED(grid_opt)
  CHECK_SAME_DEVICE(input_opt, grid_opt)
  CHECK_SAME_DTYPE(input_opt, grid_opt)
  CHECK_SPATIAL_1D_2D_OR_3D(input)
  CHECK_SPATIAL_1D_2D_OR_3D(grid)
  CHECK_GRID_COMPONENT(grid, grid.dim())
  CHECK_SPATIAL_NOT_EMPTY(input)
  CHECK_SPATIAL_NOT_EMPTY(grid)
  CHECK_GRID_TARGET_COMPAT(grid, input)
  CHECK_VEC_NOT_EMPTY(bound_mode);
  CHECK_VEC_NOT_EMPTY(interpolation_mode);

  if (source_size.empty()) {
    auto size = c10::IntArrayRef(
        {input.dim() >= 3 ? input.size(2) : 1,
         input.dim() >= 4 ? input.size(3) : 1,
         input.dim() >= 5 ? input.size(4) : 1});
    if (input.is_cuda())
#ifdef WITH_CUDA
      return cuda::pushpull(
                 size,
                 grid,
                 input,
                 BoundVectorRef(bound_mode),
                 InterpolationVectorRef(interpolation_mode),
                 extrapolate,
                 false,
                 true,
                 false,
                 false,
                 false)
          .front();
#else
      AT_ERROR("Not compiled with GPU support.");
#endif
    else
      return cpu::pushpull(
                 size,
                 grid,
                 input,
                 BoundVectorRef(bound_mode),
                 InterpolationVectorRef(interpolation_mode),
                 extrapolate,
                 false,
                 true,
                 false,
                 false,
                 false)
          .front();
  } else {
    CHECK_SPATIAL_LENGTH(source_size, grid.dim())
    if (input.is_cuda())
#ifdef WITH_CUDA
      return cuda::pushpull(
                 source_size,
                 grid,
                 input,
                 BoundVectorRef(bound_mode),
                 InterpolationVectorRef(interpolation_mode),
                 extrapolate,
                 false,
                 true,
                 false,
                 false,
                 false)
          .front();
#else
      AT_ERROR("Not compiled with GPU support.");
#endif
    else
      return cpu::pushpull(
                 source_size,
                 grid,
                 input,
                 BoundVectorRef(bound_mode),
                 InterpolationVectorRef(interpolation_mode),
                 extrapolate,
                 false,
                 true,
                 false,
                 false,
                 false)
          .front();
  }
}

std::deque<at::Tensor> grid_push_backward(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& grid,
    const std::vector<BoundType>& bound_mode,
    const std::vector<InterpolationType>& interpolation_mode,
    bool extrapolate) {
  if (input.is_cuda()) {
#ifdef WITH_CUDA
    return cuda::pushpull(
        grad,
        grid,
        input,
        BoundVectorRef(bound_mode),
        InterpolationVectorRef(interpolation_mode),
        extrapolate,
        input.requires_grad(),
        false,
        false,
        grid.requires_grad(),
        false);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  } else {
    return cpu::pushpull(
        grad,
        grid,
        input,
        BoundVectorRef(bound_mode),
        InterpolationVectorRef(interpolation_mode),
        extrapolate,
        input.requires_grad(),
        false,
        false,
        grid.requires_grad(),
        false);
  }
}

// COUNT
at::Tensor grid_count(
    const at::Tensor& grid,
    c10::IntArrayRef source_size,
    const std::vector<BoundType>& bound_mode,
    const std::vector<InterpolationType>& interpolation_mode,
    bool extrapolate) {
  CHECK_DEFINED(grid)
  auto grid_opt = grid.options();
  CHECK_STRIDED(grid_opt)
  CHECK_SPATIAL_1D_2D_OR_3D(grid)
  CHECK_GRID_COMPONENT(grid, grid.dim())
  CHECK_SPATIAL_NOT_EMPTY(grid)
  CHECK_VEC_NOT_EMPTY(bound_mode);
  CHECK_VEC_NOT_EMPTY(interpolation_mode);

  if (source_size.empty()) {
    auto size = c10::IntArrayRef(
        {grid.dim() >= 3 ? grid.size(2) : 1, grid.dim() >= 4 ? grid.size(3) : 1, grid.dim() >= 5 ? grid.size(4) : 1});
    if (grid.is_cuda())
#ifdef WITH_CUDA
      return cuda::pushpull(
                 size,
                 grid,
                 BoundVectorRef(bound_mode),
                 InterpolationVectorRef(interpolation_mode),
                 extrapolate,
                 false,
                 false,
                 true,
                 false,
                 false)
          .front();
#else
      AT_ERROR("Not compiled with GPU support.");
#endif
    else
      return cpu::pushpull(
                 size,
                 grid,
                 BoundVectorRef(bound_mode),
                 InterpolationVectorRef(interpolation_mode),
                 extrapolate,
                 false,
                 false,
                 true,
                 false,
                 false)
          .front();
  } else {
    CHECK_SPATIAL_LENGTH(source_size, grid.dim())
    if (grid.is_cuda())
#ifdef WITH_CUDA
      return cuda::pushpull(
                 source_size,
                 grid,
                 BoundVectorRef(bound_mode),
                 InterpolationVectorRef(interpolation_mode),
                 extrapolate,
                 false,
                 false,
                 true,
                 false,
                 false)
          .front();
#else
      AT_ERROR("Not compiled with GPU support.");
#endif
    else
      return cpu::pushpull(
                 source_size,
                 grid,
                 BoundVectorRef(bound_mode),
                 InterpolationVectorRef(interpolation_mode),
                 extrapolate,
                 false,
                 false,
                 true,
                 false,
                 false)
          .front();
  }
}

at::Tensor grid_count_backward(
    const at::Tensor& grad,
    const at::Tensor& grid,
    const std::vector<BoundType>& bound_mode,
    const std::vector<InterpolationType>& interpolation_mode,
    bool extrapolate) {
  if (grid.is_cuda()) {
#ifdef WITH_CUDA
    return cuda::pushpull(
               grad,
               grid,
               BoundVectorRef(bound_mode),
               InterpolationVectorRef(interpolation_mode),
               extrapolate,
               false,
               false,
               false,
               grid.requires_grad(),
               false)
        .front();
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  } else {
    return cpu::pushpull(
               grad,
               grid,
               BoundVectorRef(bound_mode),
               InterpolationVectorRef(interpolation_mode),
               extrapolate,
               false,
               false,
               false,
               grid.requires_grad(),
               false)
        .front();
  }
}

// PULL GRADIENTS
at::Tensor grid_grad(
    const at::Tensor& input,
    const at::Tensor& grid,
    const std::vector<BoundType>& bound_mode,
    const std::vector<InterpolationType>& interpolation_mode,
    bool extrapolate) {
  CHECK_DEFINED(input)
  CHECK_DEFINED(grid)
  auto input_opt = input.options();
  auto grid_opt = grid.options();
  CHECK_STRIDED(input_opt)
  CHECK_STRIDED(grid_opt)
  CHECK_SAME_DEVICE(input_opt, grid_opt)
  CHECK_SAME_DTYPE(input_opt, grid_opt)
  CHECK_SPATIAL_1D_2D_OR_3D(input)
  CHECK_SPATIAL_1D_2D_OR_3D(grid)
  CHECK_GRID_COMPONENT(grid, grid.dim())
  CHECK_SPATIAL_NOT_EMPTY(input)
  CHECK_SPATIAL_NOT_EMPTY(grid)
  CHECK_VEC_NOT_EMPTY(bound_mode);
  CHECK_VEC_NOT_EMPTY(interpolation_mode);

  if (input.is_cuda())
#ifdef WITH_CUDA
    return cuda::pushpull(
               input,
               grid,
               BoundVectorRef(bound_mode),
               InterpolationVectorRef(interpolation_mode),
               extrapolate,
               false,
               false,
               false,
               false,
               true)
        .front();
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  else
    return cpu::pushpull(
               input,
               grid,
               BoundVectorRef(bound_mode),
               InterpolationVectorRef(interpolation_mode),
               extrapolate,
               false,
               false,
               false,
               false,
               true)
        .front();
}

std::deque<at::Tensor> grid_grad_backward(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& grid,
    const std::vector<BoundType>& bound_mode,
    const std::vector<InterpolationType>& interpolation_mode,
    bool extrapolate) {
  if (input.is_cuda()) {
#ifdef WITH_CUDA
    return cuda::pushpull(
        input,
        grid,
        grad,
        BoundVectorRef(bound_mode),
        InterpolationVectorRef(interpolation_mode),
        extrapolate,
        false,
        input.requires_grad(),
        false,
        grid.requires_grad(),
        false);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  } else {
    return cpu::pushpull(
        input,
        grid,
        grad,
        BoundVectorRef(bound_mode),
        InterpolationVectorRef(interpolation_mode),
        extrapolate,
        false,
        input.requires_grad(),
        false,
        grid.requires_grad(),
        false);
  }
}

} // namespace monai
