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

=========================================================================
Adapted from https://github.com/faebstn96/trainable-joint-bilateral-filter-source
which has the following license...
https://github.com/faebstn96/trainable-joint-bilateral-filter-source/blob/main/LICENSE

Copyright 2022 Fabian Wagner, Pattern Recognition Lab, FAU Erlangen-Nuernberg, Erlangen, Germany
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
#include <stdexcept>
#include <string>

#include "trainable_joint_bilateral.h"
#include "utils/common_utils.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
TrainableJointBilateralFilterForward(
    torch::Tensor inputTensor,
    torch::Tensor guidanceTensor,
    float sigma_x,
    float sigma_y,
    float sigma_z,
    float colorSigma) {
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> (
      *filterFunction)(torch::Tensor, torch::Tensor, float, float, float, float);

#ifdef WITH_CUDA

  if (torch::cuda::is_available() && inputTensor.is_cuda()) {
    CHECK_CONTIGUOUS_CUDA(inputTensor);

    if (inputTensor.size(1) > BF_CUDA_MAX_CHANNELS) {
      throw std::runtime_error(
          "Bilateral filtering not implemented for channel count > " + std::to_string(BF_CUDA_MAX_CHANNELS));
    }

    if (inputTensor.dim() - 2 > BF_CUDA_MAX_SPATIAL_DIMENSION) {
      throw std::runtime_error(
          "Bilateral filtering not implemented for spatial dimension > " +
          std::to_string(BF_CUDA_MAX_SPATIAL_DIMENSION));
    }

    filterFunction = &JointBilateralFilterCudaForward;
  } else {
    filterFunction = &JointBilateralFilterCpuForward;
  }
#else
  filterFunction = &JointBilateralFilterCpuForward;
#endif

  return filterFunction(inputTensor, guidanceTensor, sigma_x, sigma_y, sigma_z, colorSigma);
}

std::tuple<torch::Tensor, torch::Tensor> TrainableJointBilateralFilterBackward(
    torch::Tensor gradientInputTensor,
    torch::Tensor inputTensor,
    torch::Tensor guidanceTensor,
    torch::Tensor outputTensor,
    torch::Tensor outputWeightsTensor,
    torch::Tensor dO_dx_ki,
    float sigma_x,
    float sigma_y,
    float sigma_z,
    float colorSigma) {
  std::tuple<torch::Tensor, torch::Tensor> (*filterFunction)(
      torch::Tensor,
      torch::Tensor,
      torch::Tensor,
      torch::Tensor,
      torch::Tensor,
      torch::Tensor,
      float,
      float,
      float,
      float);

#ifdef WITH_CUDA

  if (torch::cuda::is_available() && gradientInputTensor.is_cuda()) {
    CHECK_CONTIGUOUS_CUDA(gradientInputTensor);

    if (gradientInputTensor.size(1) > BF_CUDA_MAX_CHANNELS) {
      throw std::runtime_error(
          "Bilateral filtering not implemented for channel count > " + std::to_string(BF_CUDA_MAX_CHANNELS));
    }

    if (gradientInputTensor.dim() - 2 > BF_CUDA_MAX_SPATIAL_DIMENSION) {
      throw std::runtime_error(
          "Bilateral filtering not implemented for spatial dimension > " +
          std::to_string(BF_CUDA_MAX_SPATIAL_DIMENSION));
    }

    filterFunction = &JointBilateralFilterCudaBackward;
  } else {
    filterFunction = &JointBilateralFilterCpuBackward;
  }
#else
  filterFunction = &JointBilateralFilterCpuBackward;
#endif

  return filterFunction(
      gradientInputTensor,
      inputTensor,
      guidanceTensor,
      outputTensor,
      outputWeightsTensor,
      dO_dx_ki,
      sigma_x,
      sigma_y,
      sigma_z,
      colorSigma);
}
