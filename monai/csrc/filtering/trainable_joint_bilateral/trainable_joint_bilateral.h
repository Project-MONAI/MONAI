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

#pragma once

#include <torch/extension.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include "utils/common_utils.h"
//#include "utils/tensor_description.h"

#define BF_CUDA_MAX_CHANNELS 16
#define BF_CUDA_MAX_SPATIAL_DIMENSION 3

#ifdef WITH_CUDA
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
JointBilateralFilterCudaForward(
    torch::Tensor inputTensor,
    torch::Tensor guidanceTensor,
    float sigma_x,
    float sigma_y,
    float sigma_z,
    float colorSigma);
std::tuple<torch::Tensor, torch::Tensor> JointBilateralFilterCudaBackward(
    torch::Tensor gradientInputTensor,
    torch::Tensor inputTensor,
    torch::Tensor guidanceTensor,
    torch::Tensor outputTensor,
    torch::Tensor outputWeightsTensor,
    torch::Tensor dO_dz_ki,
    float sigma_x,
    float sigma_y,
    float sigma_z,
    float colorSigma);
#endif

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
JointBilateralFilterCpuForward(
    torch::Tensor inputTensor,
    torch::Tensor guidanceTensor,
    float sigma_x,
    float sigma_y,
    float sigma_z,
    float colorSigma);

std::tuple<torch::Tensor, torch::Tensor> JointBilateralFilterCpuBackward(
    torch::Tensor gradientInputTensor,
    torch::Tensor inputTensor,
    torch::Tensor guidanceTensor,
    torch::Tensor outputTensor,
    torch::Tensor outputWeightsTensor,
    torch::Tensor dO_dz_ki,
    float sigma_x,
    float sigma_y,
    float sigma_z,
    float colorSigma);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
TrainableJointBilateralFilterForward(
    torch::Tensor inputTensor,
    torch::Tensor guidanceTensor,
    float sigma_x,
    float sigma_y,
    float sigma_z,
    float colorSigma);

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
    float colorSigma);
